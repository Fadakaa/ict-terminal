"""Forex calendar integration — single source (ForexFactory).

Live operations pull the FF weekly XML feed (https://nfs.faireconomy.media/
ff_calendar_thisweek.xml). Historical day-one backfill uses
``market-calendar-tool`` (see ``ml/calendar_backfill.py``).

Layers:
- Layer 1: prompt context (``ml/prompts.py``)
- Layer 2: scanner setup metadata + warnings (``ml/scanner.py``,
  ``ml/notifications.py``)
- Layer 3: deferred — data-gated grade adjustments
- Layer 4: ML features (``ml/features.py``, ``ml/feature_schema.py``)
"""
from __future__ import annotations

import re
import sqlite3
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Protocol


FF_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"

# faireconomy.media's CDN aggressively rate-limits Python's default urllib
# user agent (``Python-urllib/3.x``). Send a real browser-like UA so the
# scheduler's hourly refresh stays under the limit.
FF_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

_IMPACT_RANK = {"holiday": -1, "low": 0, "medium": 1, "high": 2}


# Order matters — first match wins. More specific patterns precede generic ones.
CATEGORY_RULES: list[tuple[str, str]] = [
    ("nfp", r"non[- ]?farm.*(payroll|employment)"),
    # fed_speak must precede fomc — "FOMC Member ... Speaks" matches both, and
    # speaker events are more specific than the policy-event bucket.
    ("fed_speak", r"(fed chair|fomc member|powell|fed governor|"
                  r"treasury sec|fed vice chair).*(speak|testif|speech)"),
    ("fomc", r"\b(fomc|federal funds rate|fomc statement|"
             r"fomc economic projections|fomc press conference)\b"),
    ("cpi", r"\bcpi\b"),
    ("ppi", r"\bppi\b"),
    ("gdp", r"\bgdp\b"),
    ("ism", r"\bism\b"),
    ("retail_sales", r"retail sales"),
    ("unemployment", r"unemployment (rate|claims)"),
    ("jolts", r"\bjolts\b"),
]


def categorise(title: str) -> str:
    """Map an event title to one of the canonical category labels.

    Falls back to ``"other_high"`` for unknown titles — the caller is expected
    to filter to high-impact USD events upstream, so unknown rows still count.
    """
    t = title.lower()
    for category, pattern in CATEGORY_RULES:
        if re.search(pattern, t):
            return category
    return "other_high"


@dataclass(frozen=True)
class CalendarEvent:
    event_id: str
    timestamp_utc: datetime
    currency: str
    impact: str  # "high" | "medium" | "low" | "holiday"
    title: str
    category: str
    forecast: str | None
    previous: str | None
    actual: str | None


class CalendarSource(Protocol):
    def fetch_window(
        self,
        start: datetime,
        end: datetime,
        currencies: tuple[str, ...] = ("USD",),
        min_impact: str = "high",
    ) -> list[CalendarEvent]:
        ...


def _stable_event_id(title: str, currency: str, ts: datetime) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return f"ff-{currency.lower()}-{ts.strftime('%Y%m%d-%H%M')}-{slug}"


def _parse_ff_timestamp(date_s: str, time_s: str):
    """FF feed: date is ``MM-DD-YYYY`` (US), time is ``h:MMam`` / ``h:MMpm``.

    Returns a UTC-aware datetime, or ``None`` if the row carries a non-time
    placeholder ("All Day", "Tentative", etc.) we cannot pin to a moment.
    """
    try:
        month, day, year = (int(p) for p in date_s.split("-"))
    except ValueError:
        return None
    t = time_s.strip().lower()
    m = re.match(r"^(\d{1,2}):(\d{2})(am|pm)$", t)
    if not m:
        return None
    hour = int(m.group(1)) % 12
    if m.group(3) == "pm":
        hour += 12
    return datetime(year, month, day, hour, int(m.group(2)), tzinfo=timezone.utc)


class ForexFactorySource:
    """ForexFactory weekly XML — single live source.

    Notes (validated 2026-04-29):
    - Times are already UTC. Do **not** apply ET→UTC conversion.
    - Encoding is ``windows-1252``, not UTF-8. Decode explicitly.
    - Date format is ``MM-DD-YYYY`` (US). Locked to that — no auto-detection.
    - Repeated listings within the same hour (typical FF amendment pattern)
      are deduped, keeping the later timestamp.
    """

    def __init__(self, url: str = FF_URL, _offline_path: str | None = None):
        self.url = url
        self._offline_path = _offline_path

    def _fetch_raw(self) -> bytes:
        if self._offline_path:
            with open(self._offline_path, "rb") as f:
                return f.read()
        # Send a browser-like User-Agent and Accept header — the FF CDN
        # rate-limits Python's default ``Python-urllib`` UA aggressively,
        # which makes the hourly scheduler tick prone to 429s.
        req = urllib.request.Request(self.url, headers={
            "User-Agent": FF_USER_AGENT,
            "Accept": "application/xml, text/xml, */*",
            "Accept-Language": "en-US,en;q=0.9",
        })
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read()

    def fetch_window(
        self,
        start: datetime,
        end: datetime,
        currencies: tuple[str, ...] = ("USD",),
        min_impact: str = "high",
    ) -> list[CalendarEvent]:
        raw = self._fetch_raw()
        text = raw.decode("windows-1252")
        root = ET.fromstring(text)
        min_rank = _IMPACT_RANK.get(min_impact, 2)
        events: dict[str, CalendarEvent] = {}

        for ev in root.findall("event"):
            currency = (ev.findtext("country") or "").upper()
            if currencies and currency not in currencies:
                continue
            impact_raw = (ev.findtext("impact") or "").lower()
            if _IMPACT_RANK.get(impact_raw, -2) < min_rank:
                continue
            ts = _parse_ff_timestamp(
                ev.findtext("date") or "", ev.findtext("time") or ""
            )
            if ts is None or not (start <= ts <= end):
                continue
            title = (ev.findtext("title") or "").strip()
            dedup_key = self._dedup_key(title, currency, ts)
            existing = events.get(dedup_key)
            if existing is not None and existing.timestamp_utc >= ts:
                continue
            events[dedup_key] = CalendarEvent(
                event_id=_stable_event_id(title, currency, ts),
                timestamp_utc=ts,
                currency=currency,
                impact=impact_raw,
                title=title,
                category=categorise(title),
                forecast=(ev.findtext("forecast") or None),
                previous=(ev.findtext("previous") or None),
                actual=None,
            )
        return sorted(events.values(), key=lambda e: e.timestamp_utc)

    @staticmethod
    def _dedup_key(title: str, currency: str, ts: datetime) -> str:
        # Hour-level bucket — catches FF's pattern of re-posting the same
        # event with a corrected time within the same hour.
        return f"{currency}|{title}|{ts.strftime('%Y-%m-%d')}|{ts.hour:02d}"


# ---- CalendarStore -------------------------------------------------------

PROXIMITY_IMMINENT_MINS = 30
PROXIMITY_CAUTION_MINS = 90
PROXIMITY_POST_EVENT_MINS = 90


@dataclass(frozen=True)
class ProximityStatus:
    state: str  # "clear" | "caution" | "imminent" | "post_event"
    next_event: CalendarEvent | None
    minutes_to_next: float | None
    last_event: CalendarEvent | None
    minutes_since_last: float | None
    warning: str | None


def _row_to_event(row: sqlite3.Row | tuple) -> CalendarEvent:
    if isinstance(row, sqlite3.Row):
        d = dict(row)
    else:
        # Generic tuple — caller must pass keys in the canonical order used
        # by select queries below.
        keys = ("event_id", "timestamp_utc", "currency", "impact", "title",
                "category", "forecast", "previous", "actual")
        d = dict(zip(keys, row))
    ts = datetime.fromisoformat(d["timestamp_utc"])
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return CalendarEvent(
        event_id=d["event_id"],
        timestamp_utc=ts,
        currency=d["currency"],
        impact=d["impact"],
        title=d["title"],
        category=d["category"],
        forecast=d.get("forecast"),
        previous=d.get("previous"),
        actual=d.get("actual"),
    )


class CalendarStore:
    """SQLite-backed cache and query layer for forex calendar events.

    Live operations: ``refresh()`` pulls the FF XML, upserts current-window
    rows into ``forex_calendar``, and appends a snapshot to
    ``forex_calendar_history`` for forensics + ML feature backfill.
    """

    def __init__(
        self,
        source: CalendarSource,
        db_path: str,
        cache_max_age_minutes: int = 60,
    ):
        self.source = source
        self.db_path = db_path
        self.cache_max_age_minutes = cache_max_age_minutes

    # ---- mutation -------------------------------------------------------

    def refresh(self, force: bool = False, now: datetime | None = None) -> int:
        """Refresh the calendar cache. Returns the count of rows that changed.

        ``force=False`` skips the fetch if the most recent ``fetched_at`` is
        within ``cache_max_age_minutes``.
        """
        now = now or datetime.now(timezone.utc)
        if not force and self._cache_fresh(now):
            return 0

        events = self.source.fetch_window(
            start=now - timedelta(days=2),
            end=now + timedelta(days=10),
            currencies=("USD",),
            min_impact="high",
        )

        fetched_at = now.isoformat()
        archived_at = fetched_at
        updated = 0
        with sqlite3.connect(self.db_path) as conn:
            for e in events:
                cur = conn.execute(
                    """INSERT INTO forex_calendar
                       (event_id, timestamp_utc, currency, impact, title,
                        category, forecast, previous, actual, fetched_at,
                        source)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'ff_xml')
                       ON CONFLICT(event_id) DO UPDATE SET
                         timestamp_utc=excluded.timestamp_utc,
                         currency=excluded.currency,
                         impact=excluded.impact,
                         title=excluded.title,
                         category=excluded.category,
                         forecast=excluded.forecast,
                         previous=excluded.previous,
                         actual=excluded.actual,
                         fetched_at=excluded.fetched_at,
                         source=excluded.source
                       WHERE forex_calendar.timestamp_utc != excluded.timestamp_utc
                          OR IFNULL(forex_calendar.impact, '') != IFNULL(excluded.impact, '')
                          OR IFNULL(forex_calendar.title, '') != IFNULL(excluded.title, '')
                          OR IFNULL(forex_calendar.category, '') != IFNULL(excluded.category, '')
                          OR IFNULL(forex_calendar.forecast, '') != IFNULL(excluded.forecast, '')
                          OR IFNULL(forex_calendar.previous, '') != IFNULL(excluded.previous, '')
                          OR IFNULL(forex_calendar.actual, '') != IFNULL(excluded.actual, '')
                    """,
                    (e.event_id, e.timestamp_utc.isoformat(), e.currency,
                     e.impact, e.title, e.category, e.forecast, e.previous,
                     e.actual, fetched_at),
                )
                if cur.rowcount > 0:
                    updated += 1

                conn.execute(
                    """INSERT OR IGNORE INTO forex_calendar_history
                       (event_id, archived_at, timestamp_utc, currency, impact,
                        title, category, forecast, previous, actual, source)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'ff_xml')""",
                    (e.event_id, archived_at, e.timestamp_utc.isoformat(),
                     e.currency, e.impact, e.title, e.category, e.forecast,
                     e.previous, e.actual),
                )
            conn.commit()
        return updated

    def _cache_fresh(self, now: datetime) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT fetched_at FROM forex_calendar "
                "ORDER BY fetched_at DESC LIMIT 1"
            ).fetchone()
        if not row:
            return False
        try:
            last = datetime.fromisoformat(row[0])
            if last.tzinfo is None:
                last = last.replace(tzinfo=timezone.utc)
        except ValueError:
            return False
        return (now - last) < timedelta(minutes=self.cache_max_age_minutes)

    # ---- queries --------------------------------------------------------

    def upcoming(
        self,
        hours: int = 24,
        min_impact: str = "high",
        currencies: tuple[str, ...] = ("USD",),
        now: datetime | None = None,
    ) -> list[CalendarEvent]:
        now = now or datetime.now(timezone.utc)
        end = now + timedelta(hours=hours)
        return self._query_window(now, end, min_impact, currencies, ascending=True)

    def recent(
        self,
        hours: int = 24,
        min_impact: str = "high",
        currencies: tuple[str, ...] = ("USD",),
        now: datetime | None = None,
    ) -> list[CalendarEvent]:
        now = now or datetime.now(timezone.utc)
        start = now - timedelta(hours=hours)
        return self._query_window(start, now, min_impact, currencies, ascending=False)

    def _query_window(
        self,
        start: datetime,
        end: datetime,
        min_impact: str,
        currencies: tuple[str, ...],
        ascending: bool,
    ) -> list[CalendarEvent]:
        min_rank = _IMPACT_RANK.get(min_impact, 2)
        impacts = [k for k, v in _IMPACT_RANK.items() if v >= min_rank]
        if not impacts or not currencies:
            return []
        order = "ASC" if ascending else "DESC"
        placeholders_curr = ",".join("?" * len(currencies))
        placeholders_imp = ",".join("?" * len(impacts))
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"SELECT event_id, timestamp_utc, currency, impact, title, "
                f"       category, forecast, previous, actual "
                f"FROM forex_calendar "
                f"WHERE timestamp_utc >= ? AND timestamp_utc <= ? "
                f"  AND currency IN ({placeholders_curr}) "
                f"  AND impact IN ({placeholders_imp}) "
                f"ORDER BY timestamp_utc {order}",
                (start.isoformat(), end.isoformat(), *currencies, *impacts),
            ).fetchall()
        return [_row_to_event(r) for r in rows]

    @classmethod
    def historical_view(cls, db_path: str) -> "HistoricalCalendarView":
        """Construct a read-only view that queries ``forex_calendar_history``
        instead of the live ``forex_calendar`` cache. Used by the day-one
        feature backfill to reconstruct proximity at past timestamps.
        """
        return HistoricalCalendarView(db_path)

    def proximity(
        self,
        ts: datetime,
        min_impact: str = "high",
    ) -> ProximityStatus:
        """Classify ``ts`` against the cached calendar.

        State precedence: imminent > caution > post_event > clear. An imminent
        next event eclipses any post-event noise window, since upcoming
        manipulation risk is more actionable than residual settlement noise.
        """
        next_events = self._query_window(
            ts, ts + timedelta(hours=24), min_impact, ("USD",), ascending=True,
        )
        last_events = self._query_window(
            ts - timedelta(hours=24), ts, min_impact, ("USD",), ascending=False,
        )
        nxt = next_events[0] if next_events else None
        last = last_events[0] if last_events else None
        # Clamp to non-negative so an event exactly at ``ts`` reports 0 minutes
        # to next rather than a tiny negative produced by float math.
        mins_to_next = (
            max(0.0, (nxt.timestamp_utc - ts).total_seconds() / 60.0)
            if nxt else None
        )
        mins_since_last = (
            max(0.0, (ts - last.timestamp_utc).total_seconds() / 60.0)
            if last else None
        )

        if mins_to_next is not None and mins_to_next <= PROXIMITY_IMMINENT_MINS:
            state = "imminent"
            warning = (
                f"{nxt.title} ({nxt.impact}) releases in "
                f"{int(round(mins_to_next))} minutes"
            )
        elif mins_to_next is not None and mins_to_next <= PROXIMITY_CAUTION_MINS:
            state = "caution"
            warning = (
                f"{nxt.title} ({nxt.impact}) releases in "
                f"{int(round(mins_to_next))} minutes"
            )
        elif (mins_since_last is not None
              and mins_since_last <= PROXIMITY_POST_EVENT_MINS):
            state = "post_event"
            warning = (
                f"{last.title} printed {int(round(mins_since_last))} "
                f"minutes ago — settlement noise still possible"
            )
        else:
            state = "clear"
            warning = None

        return ProximityStatus(
            state=state,
            next_event=nxt,
            minutes_to_next=mins_to_next,
            last_event=last,
            minutes_since_last=mins_since_last,
            warning=warning,
        )


class HistoricalCalendarView:
    """Read-only adapter that reconstructs proximity from ``forex_calendar_history``.

    Used by the day-one feature backfill. Mirrors the subset of
    ``CalendarStore`` that ``ml.features._extract_calendar_features``
    actually calls (``proximity()`` and ``upcoming()``).

    The history table is append-only and contains both live snapshots and
    one-shot historical imports — for any past timestamp ``ts``, the most
    recent archived snapshot is chosen via ``MAX(archived_at)`` to dedupe
    if the same event appears under both ``ff_xml`` and ``ff_historical``.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path

    def _query_window(
        self,
        start: datetime,
        end: datetime,
        min_impact: str,
        currencies: tuple[str, ...],
        ascending: bool,
    ) -> list[CalendarEvent]:
        min_rank = _IMPACT_RANK.get(min_impact, 2)
        impacts = [k for k, v in _IMPACT_RANK.items() if v >= min_rank]
        if not impacts or not currencies:
            return []
        order = "ASC" if ascending else "DESC"
        ph_curr = ",".join("?" * len(currencies))
        ph_imp = ",".join("?" * len(impacts))
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"SELECT event_id, "
                f"       MAX(timestamp_utc) AS timestamp_utc, "
                f"       MAX(currency)      AS currency, "
                f"       MAX(impact)        AS impact, "
                f"       MAX(title)         AS title, "
                f"       MAX(category)      AS category, "
                f"       MAX(forecast)      AS forecast, "
                f"       MAX(previous)      AS previous, "
                f"       MAX(actual)        AS actual "
                f"FROM forex_calendar_history "
                f"WHERE timestamp_utc >= ? AND timestamp_utc <= ? "
                f"  AND currency IN ({ph_curr}) "
                f"  AND impact IN ({ph_imp}) "
                f"GROUP BY event_id "
                f"ORDER BY timestamp_utc {order}",
                (start.isoformat(), end.isoformat(), *currencies, *impacts),
            ).fetchall()
        return [_row_to_event(r) for r in rows]

    def upcoming(
        self,
        hours: int = 24,
        min_impact: str = "high",
        currencies: tuple[str, ...] = ("USD",),
        now: datetime | None = None,
    ) -> list[CalendarEvent]:
        if now is None:
            now = datetime.now(timezone.utc)
        return self._query_window(
            now, now + timedelta(hours=hours),
            min_impact, currencies, ascending=True,
        )

    def proximity(
        self,
        ts: datetime,
        min_impact: str = "high",
    ) -> ProximityStatus:
        next_events = self._query_window(
            ts, ts + timedelta(hours=24), min_impact,
            ("USD",), ascending=True,
        )
        last_events = self._query_window(
            ts - timedelta(hours=24), ts, min_impact,
            ("USD",), ascending=False,
        )
        nxt = next_events[0] if next_events else None
        last = last_events[0] if last_events else None
        mins_to_next = (
            max(0.0, (nxt.timestamp_utc - ts).total_seconds() / 60.0)
            if nxt else None
        )
        mins_since_last = (
            max(0.0, (ts - last.timestamp_utc).total_seconds() / 60.0)
            if last else None
        )

        if mins_to_next is not None and mins_to_next <= PROXIMITY_IMMINENT_MINS:
            state = "imminent"
            warning = (
                f"{nxt.title} ({nxt.impact}) releases in "
                f"{int(round(mins_to_next))} minutes"
            )
        elif mins_to_next is not None and mins_to_next <= PROXIMITY_CAUTION_MINS:
            state = "caution"
            warning = (
                f"{nxt.title} ({nxt.impact}) releases in "
                f"{int(round(mins_to_next))} minutes"
            )
        elif (mins_since_last is not None
              and mins_since_last <= PROXIMITY_POST_EVENT_MINS):
            state = "post_event"
            warning = (
                f"{last.title} printed {int(round(mins_since_last))} "
                f"minutes ago — settlement noise still possible"
            )
        else:
            state = "clear"
            warning = None
        return ProximityStatus(
            state=state, next_event=nxt, minutes_to_next=mins_to_next,
            last_event=last, minutes_since_last=mins_since_last,
            warning=warning,
        )
