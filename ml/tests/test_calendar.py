"""Tests for ml/calendar.py — forex calendar integration."""
from __future__ import annotations

import dataclasses
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

from ml.scanner_db import init_db


FIXTURE = Path(__file__).parent / "fixtures" / "ff_calendar_sample.xml"


def test_forex_calendar_tables_exist(tmp_path):
    db_path = tmp_path / "test.db"
    init_db(str(db_path))
    conn = sqlite3.connect(str(db_path))
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )}
    assert "forex_calendar" in tables
    assert "forex_calendar_history" in tables
    cols = {r[1] for r in conn.execute("PRAGMA table_info(forex_calendar)")}
    assert {"event_id", "timestamp_utc", "currency", "impact", "title",
            "category", "forecast", "previous", "actual", "fetched_at",
            "source"}.issubset(cols)
    history_cols = {r[1] for r in conn.execute(
        "PRAGMA table_info(forex_calendar_history)"
    )}
    assert "source" in history_cols


# ---- Task 2 — CalendarEvent + categorise() -------------------------------

def test_categorise_known_titles():
    from ml.calendar import categorise
    assert categorise("Non-Farm Employment Change") == "nfp"
    assert categorise("Core CPI m/m") == "cpi"
    assert categorise("Federal Funds Rate") == "fomc"
    assert categorise("FOMC Statement") == "fomc"
    assert categorise("FOMC Press Conference") == "fomc"
    assert categorise("Fed Chair Powell Testifies") == "fed_speak"
    assert categorise("FOMC Member Williams Speaks") == "fed_speak"
    assert categorise("ISM Manufacturing PMI") == "ism"
    assert categorise("Core Retail Sales m/m") == "retail_sales"
    assert categorise("Advance GDP q/q") == "gdp"
    assert categorise("Unemployment Rate") == "unemployment"
    assert categorise("JOLTS Job Openings") == "jolts"
    assert categorise("Random Other Event") == "other_high"


def test_calendar_event_dataclass_immutable():
    from ml.calendar import CalendarEvent
    e = CalendarEvent(
        event_id="ff-1",
        timestamp_utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
        currency="USD",
        impact="high",
        title="NFP",
        category="nfp",
        forecast=None,
        previous=None,
        actual=None,
    )
    with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
        e.title = "changed"


# ---- Task 3 — ForexFactorySource parser ---------------------------------

def test_ff_source_parses_known_event():
    """FOMC Statement on the 2026-04-29 fixture sits at 18:00 UTC (no ET→UTC
    conversion — FF feed is already UTC)."""
    from ml.calendar import ForexFactorySource
    src = ForexFactorySource(_offline_path=str(FIXTURE))
    events = src.fetch_window(
        start=datetime(2026, 4, 29, tzinfo=timezone.utc),
        end=datetime(2026, 4, 30, tzinfo=timezone.utc),
    )
    fomc = next(e for e in events
                if e.title == "FOMC Statement" and e.currency == "USD")
    assert fomc.timestamp_utc == datetime(2026, 4, 29, 18, 0, tzinfo=timezone.utc)
    assert fomc.impact == "high"
    assert fomc.category == "fomc"


def test_ff_source_filters_currency():
    from ml.calendar import ForexFactorySource
    src = ForexFactorySource(_offline_path=str(FIXTURE))
    usd_events = src.fetch_window(
        start=datetime(2026, 4, 27, tzinfo=timezone.utc),
        end=datetime(2026, 5, 2, tzinfo=timezone.utc),
        currencies=("USD",),
    )
    assert all(e.currency == "USD" for e in usd_events)
    assert len(usd_events) > 5


def test_ff_source_dedups_repeated_listings():
    """Building Permits appears twice on 2026-04-29 (12:28pm and 12:30pm). The
    parser dedups within the same hour and keeps the later listing."""
    from ml.calendar import ForexFactorySource
    src = ForexFactorySource(_offline_path=str(FIXTURE))
    # Building Permits is Low impact — pass min_impact="low" to surface it.
    events = src.fetch_window(
        start=datetime(2026, 4, 29, tzinfo=timezone.utc),
        end=datetime(2026, 4, 30, tzinfo=timezone.utc),
        min_impact="low",
    )
    bp = [e for e in events
          if e.title.startswith("Building Permits") and e.currency == "USD"]
    assert len(bp) == 1
    assert bp[0].timestamp_utc.minute == 30


# ---- Task 4 — CalendarStore (cache + query API + archive) ----------------

def _make_store(tmp_path):
    from ml.calendar import CalendarStore, ForexFactorySource
    db = tmp_path / "test.db"
    init_db(str(db))
    store = CalendarStore(
        source=ForexFactorySource(_offline_path=str(FIXTURE)),
        db_path=str(db),
    )
    return store, db


def test_store_refresh_upserts_events(tmp_path):
    store, db = _make_store(tmp_path)
    n = store.refresh(force=True)
    assert n > 0
    # A second refresh re-fetches the same fixture; no rows have changed in
    # forex_calendar so updated_count is 0.
    assert store.refresh(force=True) == 0


def test_store_archive_writes_history(tmp_path):
    store, db = _make_store(tmp_path)
    store.refresh(force=True)
    n = sqlite3.connect(str(db)).execute(
        "SELECT COUNT(*) FROM forex_calendar_history"
    ).fetchone()[0]
    assert n > 0


def test_store_proximity_imminent(tmp_path):
    store, db = _make_store(tmp_path)
    store.refresh(force=True)
    # Federal Funds Rate / FOMC Statement print at 2026-04-29 18:00 UTC.
    # 25 minutes before → imminent. Both events are category="fomc"; assert
    # on category since two events share the same minute.
    p = store.proximity(datetime(2026, 4, 29, 17, 35, tzinfo=timezone.utc))
    assert p.state == "imminent"
    assert p.next_event.category == "fomc"
    assert p.next_event.title in ("Federal Funds Rate", "FOMC Statement")
    assert 24 <= p.minutes_to_next <= 26


def test_store_proximity_caution(tmp_path):
    store, db = _make_store(tmp_path)
    store.refresh(force=True)
    # 75 minutes before the 18:00 cluster → caution band (30 < mins ≤ 90).
    p = store.proximity(datetime(2026, 4, 29, 16, 45, tzinfo=timezone.utc))
    assert p.state == "caution"
    assert p.next_event.category == "fomc"
    assert 74 <= p.minutes_to_next <= 76


def test_store_proximity_post_event(tmp_path):
    store, db = _make_store(tmp_path)
    store.refresh(force=True)
    # FOMC Press Conference at 18:30 is the latest USD high-impact event.
    # 19:30 → 60 min after → post_event band (≤90 min since last event), and
    # the next USD high event is the next-day GDP print, far outside any
    # warning window.
    p = store.proximity(datetime(2026, 4, 29, 19, 30, tzinfo=timezone.utc))
    assert p.state == "post_event"
    assert p.last_event.title == "FOMC Press Conference"
    assert 59 <= p.minutes_since_last <= 61


def test_store_proximity_clear(tmp_path):
    store, db = _make_store(tmp_path)
    store.refresh(force=True)
    # 4 hours before FOMC and well after any earlier USD high-impact print.
    p = store.proximity(datetime(2026, 4, 29, 14, 0, tzinfo=timezone.utc))
    assert p.state == "clear"
    assert p.warning is None


# ---- Task 5 — Archive accumulates across refreshes ----------------------

def test_archive_accumulates_across_refreshes(tmp_path):
    store, db = _make_store(tmp_path)
    store.refresh(force=True,
                  now=datetime(2026, 4, 29, 12, 0, tzinfo=timezone.utc))
    store.refresh(force=True,
                  now=datetime(2026, 4, 29, 13, 0, tzinfo=timezone.utc))
    n = sqlite3.connect(str(db)).execute(
        "SELECT COUNT(DISTINCT archived_at) FROM forex_calendar_history"
    ).fetchone()[0]
    assert n == 2


# ---- Task 6 — Scheduled refresh wiring ----------------------------------

def test_scheduler_refresh_job_inserts_events(tmp_path):
    """Driving the scheduler's ``_refresh_calendar_job`` against a fixture-backed
    store inserts events into ``forex_calendar`` and history.
    """
    import asyncio
    from ml.calendar import CalendarStore, ForexFactorySource
    from ml.scheduler import _refresh_calendar_job

    db = tmp_path / "test.db"
    init_db(str(db))
    store = CalendarStore(
        source=ForexFactorySource(_offline_path=str(FIXTURE)),
        db_path=str(db),
    )
    asyncio.run(_refresh_calendar_job(store=store))
    n = sqlite3.connect(str(db)).execute(
        "SELECT COUNT(*) FROM forex_calendar"
    ).fetchone()[0]
    assert n > 0
