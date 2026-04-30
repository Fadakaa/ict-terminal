"""Historical FF calendar backfill via ``market-calendar-tool``.

One-shot import: pull N months of historical FF events, normalise through
the same pipeline as the live XML source (``ml.calendar.categorise``,
``CalendarEvent`` dataclass), upsert into ``forex_calendar_history`` with
``source='ff_historical'`` so it is distinguishable from live XML snapshots.

Run on day one (after ``ml/calendar.py`` is wired and tables exist):

    python -m ml.calendar_backfill --months 6 --db <db_path>

The retrain step uses ``live_only=True`` (per CLAUDE.md) and is gated by the
existing 55% accuracy floor in ``_maybe_auto_retrain()``. If the new model
with calendar features doesn't clear the gate, it's rejected and the
previous model stays.

Note: ``market-calendar-tool`` 0.2.x pins ``pyarrow<18,>=17`` which has no
Python 3.13 wheel. Install in a Python 3.12 venv when running this on day
one (see CLAUDE.md). Live operations do NOT depend on this package.
"""
from __future__ import annotations

import argparse
import logging
import re
import sqlite3
from datetime import datetime, timedelta, timezone

import pandas as pd

from ml.calendar import CalendarEvent, categorise

logger = logging.getLogger(__name__)


# market-calendar-tool 0.2.x scrapes the FF apply-settings/1 endpoint, whose
# times are returned in UTC+1, not UTC. The live XML feed used by the scanner
# is in plain UTC (validated 2026-04-29 against FOMC at 18:00 UTC, vs scraper
# 7:00pm). Subtract this offset on parse so historical and live rows agree on
# every fixed-time release.
SCRAPER_UTC_OFFSET = timedelta(hours=1)


def _scrape(site: str, date_from: str, date_to: str) -> pd.DataFrame:
    """Lazy thin wrapper around ``market_calendar_tool.scrape_calendar``.

    Tests monkeypatch this function directly; production calls go through to
    the upstream package. Importing inside the function keeps the rest of
    the module loadable without the dependency installed.

    Two adjustments relative to the older 0.1.x API the plan was written for:

    1. ``site`` must now be the ``Site`` enum, not a string.
    2. ``scrape_calendar`` returns a ``ScrapeResult`` whose ``.base`` is the
       DataFrame, with renamed columns (``name``, ``impactName``, ``timeLabel``)
       and abbreviated-month dates (``"Apr 29, 2026"``). We remap to the
       schema ``_to_calendar_events`` expects.
    3. The scraper's ``apply-settings/1`` endpoint returns times in UTC+1,
       while the live XML feed is plain UTC (verified empirically against
       FOMC 2026-04-29 — XML says ``6:00pm`` = 18:00 UTC, scraper says
       ``7:00pm``). We surface the raw scraped time here and let
       ``_to_calendar_events`` apply the −1h offset on parse so the unit
       tests (which feed already-normalised dataframes) are unaffected.
    """
    from market_calendar_tool import scrape_calendar, Site  # type: ignore
    site_enum = Site[site.upper()] if isinstance(site, str) else site
    result = scrape_calendar(site=site_enum,
                             date_from=date_from, date_to=date_to)
    df = result.base
    return df.rename(columns={
        "name": "title",
        "impactName": "impact",
        "timeLabel": "time",
    })


def _normalise_impact(s: str) -> str:
    """FF historical exports vary on impact casing/labelling — collapse to
    the same vocabulary used by the live XML parser."""
    s = (s or "").strip().lower()
    if s in {"high", "3"}:
        return "high"
    if s in {"medium", "2"}:
        return "medium"
    if s in {"low", "1"}:
        return "low"
    if s in {"holiday", "0"}:
        return "holiday"
    return s


def _clean(v):
    if v is None:
        return None
    if isinstance(v, float) and pd.isna(v):
        return None
    s = str(v)
    return s if s != "" else None


def _parse_row_timestamp(row, apply_scraper_offset: bool | None = None) -> datetime | None:
    """Parse the date/time fields from a ``market_calendar_tool`` row.

    Handles ISO date (``YYYY-MM-DD``), US format (``MM/DD/YYYY``,
    ``MM-DD-YYYY``), and the abbreviated-month form FF's web scraper returns
    (``"Apr 29, 2026"``). Time may be 24h (``HH:MM``) or 12h (``h:MMam``/
    ``h:MMpm``).

    ``apply_scraper_offset``:
        - ``None`` (default): auto-detect — apply the offset only when the
          date string matches the scraper's "Apr 29, 2026" format. The unit
          tests pass ISO/numeric dates, so they aren't affected.
        - ``True``/``False``: explicit override.

    Returns a UTC-aware datetime, or ``None`` for un-pinned rows
    ("All Day", "Tentative", etc.).
    """
    try:
        date_str = str(row["date"]).strip()
        time_str = str(row["time"]).strip().lower()
    except KeyError:
        return None

    is_scraper_date = False
    d = None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m-%d-%Y"):
        try:
            d = datetime.strptime(date_str, fmt)
            break
        except ValueError:
            continue
    if d is None:
        for fmt in ("%b %d, %Y", "%B %d, %Y"):
            try:
                d = datetime.strptime(date_str, fmt)
                is_scraper_date = True
                break
            except ValueError:
                continue
    if d is None:
        return None

    m = re.match(r"^(\d{1,2}):(\d{2})\s*(am|pm)?$", time_str)
    if not m:
        return None
    hour = int(m.group(1))
    suffix = m.group(3)
    if suffix:
        hour = hour % 12 + (12 if suffix == "pm" else 0)
    ts = datetime(d.year, d.month, d.day, hour, int(m.group(2)),
                  tzinfo=timezone.utc)

    use_offset = is_scraper_date if apply_scraper_offset is None else apply_scraper_offset
    if use_offset:
        ts = ts - SCRAPER_UTC_OFFSET
    return ts


def _stable_id(title: str, currency: str, ts: datetime) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return f"ff-{currency.lower()}-{ts.strftime('%Y%m%d-%H%M')}-{slug}"


def _to_calendar_events(df: pd.DataFrame) -> list[CalendarEvent]:
    """Normalise a scraped dataframe into ``CalendarEvent`` instances.

    Dedup key matches the live parser's hour-level bucket, so historical and
    live rows for the same event collapse to the same ``event_id``.
    """
    events: dict[str, CalendarEvent] = {}
    for _, row in df.iterrows():
        ts = _parse_row_timestamp(row)
        if ts is None:
            continue
        title = str(row["title"]).strip()
        currency = str(row["currency"]).upper()
        impact = _normalise_impact(str(row["impact"]))
        # Hour-level dedup — same key shape as ml.calendar.ForexFactorySource
        dedup_key = (
            f"{currency}|{title}|{ts.strftime('%Y-%m-%d')}|{ts.hour:02d}"
        )
        if dedup_key in events:
            continue
        events[dedup_key] = CalendarEvent(
            event_id=_stable_id(title, currency, ts),
            timestamp_utc=ts,
            currency=currency,
            impact=impact,
            title=title,
            category=categorise(title),
            forecast=_clean(row.get("forecast")),
            previous=_clean(row.get("previous")),
            actual=_clean(row.get("actual")),
        )
    return list(events.values())


def backfill_history(
    months: int,
    db_path: str,
    currencies: tuple[str, ...] = ("USD",),
    min_impact: str = "high",
    end: datetime | None = None,
) -> int:
    """Backfill ``months`` of FF history into ``forex_calendar_history``.

    Returns the count of newly inserted rows. Idempotent — running twice
    over the same window inserts zero rows the second time (PRIMARY KEY
    on ``(event_id, archived_at)`` plus the historical ``archived_at``
    being identical for matching events).

    Ensures the calendar tables exist before writing — production DBs that
    pre-date the calendar integration won't have ``forex_calendar_history``
    until ``ScannerDB`` is constructed against them.
    """
    from ml.scanner_db import init_db
    init_db(db_path)

    end = end or datetime.now(timezone.utc)
    start = end - timedelta(days=30 * months)
    logger.info("[BACKFILL] %s → %s", start.date(), end.date())

    df = _scrape(
        site="forexfactory",
        date_from=start.strftime("%Y-%m-%d"),
        date_to=end.strftime("%Y-%m-%d"),
    )
    events = _to_calendar_events(df)
    events = [
        e for e in events
        if e.currency in currencies and e.impact == min_impact
    ]

    # Use a stable archived_at per (event_id) so re-running on the same data
    # is a true no-op rather than appending duplicate snapshots. The original
    # event timestamp is the natural choice — historical events don't get
    # re-archived from this path.
    new_rows = 0
    with sqlite3.connect(db_path) as conn:
        for e in events:
            archived_at = e.timestamp_utc.isoformat()
            cur = conn.execute(
                "INSERT OR IGNORE INTO forex_calendar_history "
                "(event_id, archived_at, timestamp_utc, currency, impact, "
                " title, category, forecast, previous, actual, source) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'ff_historical')",
                (e.event_id, archived_at, e.timestamp_utc.isoformat(),
                 e.currency, e.impact, e.title, e.category, e.forecast,
                 e.previous, e.actual),
            )
            if cur.rowcount:
                new_rows += 1
        conn.commit()

    logger.info("[BACKFILL] inserted %d new rows", new_rows)
    return new_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-shot FF historical backfill + feature re-extract + retrain."
    )
    parser.add_argument("--months", type=int, default=6,
                        help="Months of history to pull (default: 6).")
    parser.add_argument("--db", type=str, required=True,
                        help="Path to the scanner SQLite DB.")
    parser.add_argument("--skip-retrain", action="store_true",
                        help="Skip the AutoGluon retrain after backfill.")
    args = parser.parse_args()

    n_events = backfill_history(months=args.months, db_path=args.db)
    print(f"[1/3] Backfilled {n_events} historical events.")

    from ml.backfill_features import backfill_calendar_features
    n_setups = backfill_calendar_features(db_path=args.db)
    print(f"[2/3] Re-extracted features for {n_setups} stored setups.")

    if args.skip_retrain:
        print("[3/3] Retrain skipped (--skip-retrain).")
        return

    try:
        from ml.training import train_classifier
        from ml.dataset import TrainingDatasetManager
        from ml.config import get_config
        from ml.database import TradeLogger

        cfg = get_config()
        dm = TrainingDatasetManager(config=cfg)
        # CLAUDE.md: live-only retrains exclude WFO seed (synthetic).
        result = train_classifier(
            TradeLogger(config=cfg), config=cfg, dataset_manager=dm,
        )
        acc = result.get("oos_accuracy") or result.get("accuracy") or "?"
        if isinstance(acc, (int, float)):
            print(f"[3/3] Retrain complete. OOS accuracy: {acc:.1%}")
        else:
            print(f"[3/3] Retrain complete. OOS accuracy: {acc}")
    except Exception as e:
        # Never fail the backfill itself when the retrain hook errors — the
        # historical data is already in. The auto-retrain scheduler will
        # pick the new dataset up on its next 6h tick anyway.
        print(f"[3/3] Retrain step failed: {e}")
        logger.exception("[BACKFILL] retrain failed")


if __name__ == "__main__":
    main()
