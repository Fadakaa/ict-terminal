"""Tests for ml/calendar_backfill.py — historical FF backfill.

Tests monkeypatch the ``_scrape`` function so they don't depend on the live
``market-calendar-tool`` install (which is currently blocked on a pyarrow
pin under Python 3.13). The lazy ``_scrape`` wrapper exists precisely to
keep these tests hermetic and the live one-shot backfill pluggable.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone, timedelta

import pandas as pd
import pytest

from ml.scanner_db import init_db


# ---------------------------------------------------------------------------
# Sample dataframe shaped like ``market_calendar_tool.scrape_calendar`` output
# ---------------------------------------------------------------------------

def _build_sample_df() -> pd.DataFrame:
    """Three USD high-impact rows in MM/DD/YYYY date format."""
    return pd.DataFrame([
        {"date": "2026-03-19", "time": "18:00",
         "currency": "USD", "impact": "High",
         "title": "Federal Funds Rate",
         "forecast": "5.25%", "previous": "5.25%", "actual": "5.25%"},
        {"date": "2026-03-07", "time": "13:30",
         "currency": "USD", "impact": "High",
         "title": "Non-Farm Employment Change",
         "forecast": "200K", "previous": "180K", "actual": "210K"},
        {"date": "2026-03-12", "time": "12:30",
         "currency": "USD", "impact": "High",
         "title": "Core CPI m/m",
         "forecast": "0.3%", "previous": "0.4%", "actual": "0.3%"},
    ])


def _build_sample_df_with_eur_and_low_impact() -> pd.DataFrame:
    return pd.DataFrame([
        {"date": "2026-03-19", "time": "18:00",
         "currency": "USD", "impact": "High",
         "title": "Federal Funds Rate",
         "forecast": "5.25%", "previous": "5.25%", "actual": None},
        {"date": "2026-03-07", "time": "12:45",
         "currency": "EUR", "impact": "High",
         "title": "Main Refinancing Rate",
         "forecast": "4.5%", "previous": "4.5%", "actual": None},
        {"date": "2026-03-12", "time": "13:00",
         "currency": "USD", "impact": "Medium",
         "title": "Random Indicator",
         "forecast": None, "previous": "0.1", "actual": None},
    ])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_backfill_history_writes_to_history_table(tmp_path, monkeypatch):
    from ml.calendar_backfill import backfill_history
    db = tmp_path / "test.db"
    init_db(str(db))
    monkeypatch.setattr("ml.calendar_backfill._scrape",
                        lambda **kw: _build_sample_df())
    rows = backfill_history(months=1, db_path=str(db),
                            end=datetime(2026, 3, 31, tzinfo=timezone.utc))
    assert rows > 0
    n = sqlite3.connect(str(db)).execute(
        "SELECT COUNT(*) FROM forex_calendar_history "
        "WHERE source='ff_historical'"
    ).fetchone()[0]
    assert n == rows


def test_backfill_idempotent(tmp_path, monkeypatch):
    from ml.calendar_backfill import backfill_history
    db = tmp_path / "test.db"
    init_db(str(db))
    monkeypatch.setattr("ml.calendar_backfill._scrape",
                        lambda **kw: _build_sample_df())
    end = datetime(2026, 3, 31, tzinfo=timezone.utc)
    n1 = backfill_history(months=1, db_path=str(db), end=end)
    n2 = backfill_history(months=1, db_path=str(db), end=end)
    assert n1 > 0 and n2 == 0


def test_backfill_filters_to_usd_high_impact(tmp_path, monkeypatch):
    from ml.calendar_backfill import backfill_history
    db = tmp_path / "test.db"
    init_db(str(db))
    monkeypatch.setattr(
        "ml.calendar_backfill._scrape",
        lambda **kw: _build_sample_df_with_eur_and_low_impact(),
    )
    backfill_history(months=1, db_path=str(db),
                     currencies=("USD",), min_impact="high",
                     end=datetime(2026, 3, 31, tzinfo=timezone.utc))
    rows = sqlite3.connect(str(db)).execute(
        "SELECT currency, impact FROM forex_calendar_history "
        "WHERE source='ff_historical'"
    ).fetchall()
    assert all(r[0] == "USD" and r[1] == "high" for r in rows)


@pytest.mark.integration
def test_backfill_matches_live_xml_for_overlap_week(tmp_path):
    """Cross-source sanity check: backfill the current FF week and compare to
    the live XML feed. Both sources are FF — fixed-time releases (NFP, FOMC,
    CPI, GDP, ISM, Retail Sales, PPI) MUST match exactly. Up to 4 events of
    drift in non-fixed slots (speakers, etc.) is acceptable.

    Network-bound + requires ``market-calendar-tool`` install. Run manually:

        pytest ml/tests/test_calendar_backfill.py::\
            test_backfill_matches_live_xml_for_overlap_week -v -m integration
    """
    from ml.calendar import ForexFactorySource
    from ml.calendar_backfill import backfill_history

    db = tmp_path / "test.db"
    init_db(str(db))
    now = datetime.now(timezone.utc)
    week_start = now - timedelta(days=now.weekday())
    week_end = week_start + timedelta(days=7)

    live = ForexFactorySource().fetch_window(
        start=week_start, end=week_end,
        currencies=("USD",), min_impact="high",
    )

    backfill_history(months=1, db_path=str(db), end=week_end)

    rows = sqlite3.connect(str(db)).execute(
        "SELECT timestamp_utc, currency, title, impact, category "
        "FROM forex_calendar_history "
        "WHERE timestamp_utc >= ? AND timestamp_utc < ? "
        "  AND source='ff_historical'",
        (week_start.isoformat(), week_end.isoformat()),
    ).fetchall()

    live_set = {(e.timestamp_utc.isoformat(), e.currency, e.title,
                 e.impact, e.category) for e in live}
    historical_set = {tuple(r) for r in rows}
    only_in_live = live_set - historical_set
    only_in_historical = historical_set - live_set

    fixed_release_categories = {"nfp", "fomc", "cpi", "ppi", "gdp",
                                "ism", "retail_sales"}
    fixed_diff = {e for e in only_in_live | only_in_historical
                  if e[4] in fixed_release_categories}
    assert not fixed_diff, f"Fixed-release events differ: {fixed_diff}"
    assert len(only_in_live) + len(only_in_historical) <= 4, (
        f"Too many diffs: live-only={only_in_live}, "
        f"hist-only={only_in_historical}"
    )


def test_backfill_uses_same_categorisation_as_live():
    """Historical events are normalised through the same ``categorise()`` and
    ``CalendarEvent`` shape as live XML rows. Only the ``source`` field is
    different downstream."""
    from ml.calendar_backfill import _to_calendar_events
    df = pd.DataFrame([
        {"date": "2026-03-19", "time": "18:00",
         "currency": "USD", "impact": "High",
         "title": "Federal Funds Rate",
         "forecast": "5.25%", "previous": "5.25%", "actual": "5.25%"},
    ])
    events = _to_calendar_events(df)
    assert len(events) == 1
    e = events[0]
    assert e.category == "fomc"
    assert e.impact == "high"
    assert e.timestamp_utc == datetime(2026, 3, 19, 18, 0, tzinfo=timezone.utc)
    assert e.currency == "USD"
    assert e.title == "Federal Funds Rate"
    assert e.forecast == "5.25%"


def test_backfill_scraper_offset_aligns_with_live_xml():
    """``market-calendar-tool`` 0.2.x returns scraped times in UTC+1
    (FOMC 2026-04-29 reads as 7:00pm scraped vs 6:00pm live XML = 18:00
    UTC). The parser detects the abbreviated-month date format the scraper
    uses (``"Apr 29, 2026"``) and applies a -1h correction so historical
    rows align with live UTC timestamps. ISO/numeric dates are NOT shifted
    — those are produced by the unit-test fixtures."""
    from ml.calendar_backfill import _to_calendar_events

    scraper_df = pd.DataFrame([
        {"date": "Apr 29, 2026", "time": "7:00pm",
         "currency": "USD", "impact": "high",
         "title": "Federal Funds Rate",
         "forecast": "3.75%", "previous": "3.75%", "actual": "3.75%"},
    ])
    iso_df = pd.DataFrame([
        {"date": "2026-04-29", "time": "18:00",
         "currency": "USD", "impact": "high",
         "title": "Federal Funds Rate",
         "forecast": "3.75%", "previous": "3.75%", "actual": "3.75%"},
    ])

    scraped_events = _to_calendar_events(scraper_df)
    iso_events = _to_calendar_events(iso_df)
    assert len(scraped_events) == 1 and len(iso_events) == 1
    # Both must land at 18:00 UTC — that's the actual FOMC release time.
    assert scraped_events[0].timestamp_utc == datetime(
        2026, 4, 29, 18, 0, tzinfo=timezone.utc
    )
    assert iso_events[0].timestamp_utc == datetime(
        2026, 4, 29, 18, 0, tzinfo=timezone.utc
    )
    # Same event_id either way → backfilled rows merge cleanly with the
    # live cache via INSERT OR IGNORE on (event_id, archived_at).
    assert scraped_events[0].event_id == iso_events[0].event_id
