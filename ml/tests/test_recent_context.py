"""Tests for ml/recent_context.py — recent context builder."""
import os
import sqlite3
import tempfile
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from ml.recent_context import (
    build_recent_context,
    format_recent_context,
    _time_ago,
    _fmt_price,
    _query_recent_resolutions,
)


# ── Helpers ──────────────────────────────────────────────────────────

def _make_db(tmp_path):
    """Create a minimal ScannerDB-like object with a real SQLite DB."""
    db_path = str(tmp_path / "test_scanner.db")
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE scanner_setups (
            id              TEXT PRIMARY KEY,
            created_at      TEXT NOT NULL,
            resolved_at     TEXT,
            status          TEXT NOT NULL DEFAULT 'pending',
            timeframe       TEXT NOT NULL DEFAULT '1h',
            direction       TEXT NOT NULL,
            bias            TEXT,
            entry_price     REAL,
            sl_price        REAL,
            calibrated_sl   REAL,
            tp1             REAL,
            tp2             REAL,
            tp3             REAL,
            setup_quality   TEXT,
            killzone        TEXT,
            rr_ratios       TEXT,
            analysis_json   TEXT,
            calibration_json TEXT,
            outcome         TEXT,
            resolved_price  REAL,
            pnl_rr          REAL,
            auto_resolved   INTEGER DEFAULT 0,
            candle_hash     TEXT,
            entry_zone_type TEXT,
            entry_zone_high REAL,
            entry_zone_low  REAL,
            entry_zone_position REAL,
            notified        INTEGER DEFAULT 0,
            detection_notified INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()

    db = MagicMock()
    db.db_path = db_path
    db.get_pending.return_value = []
    return db


def _insert_setup(db_path, **overrides):
    """Insert a setup row with sensible defaults."""
    defaults = {
        "id": "test001",
        "created_at": (datetime.utcnow() - timedelta(hours=2)).isoformat(),
        "resolved_at": (datetime.utcnow() - timedelta(minutes=20)).isoformat(),
        "status": "resolved",
        "timeframe": "1h",
        "direction": "long",
        "bias": "bullish",
        "entry_price": 2341.50,
        "sl_price": 2338.00,
        "calibrated_sl": 2337.50,
        "tp1": 2348.00,
        "tp2": 2355.00,
        "tp3": 2362.00,
        "setup_quality": "A",
        "killzone": "London",
        "outcome": "stopped_out",
        "resolved_price": 2338.00,
        "pnl_rr": -1.0,
        "entry_zone_type": "ob",
        "entry_zone_high": 2343.00,
        "entry_zone_low": 2340.00,
    }
    defaults.update(overrides)
    conn = sqlite3.connect(db_path)
    cols = ", ".join(defaults.keys())
    placeholders = ", ".join(["?"] * len(defaults))
    conn.execute(f"INSERT INTO scanner_setups ({cols}) VALUES ({placeholders})",
                 list(defaults.values()))
    conn.commit()
    conn.close()


# ── Tests: build_recent_context() ───────────────────────────────────

class TestBuildRecentContext:

    def test_empty_db_returns_empty_lists(self, tmp_path):
        db = _make_db(tmp_path)
        ctx = build_recent_context("1h", db)
        assert ctx["recent_resolutions"] == []
        assert ctx["consumed_zones"] == []
        assert ctx["swept_liquidity"] == []
        assert ctx["active_setups"] == []

    def test_recent_resolution_populated(self, tmp_path):
        db = _make_db(tmp_path)
        _insert_setup(db.db_path, id="s001", outcome="stopped_out")
        ctx = build_recent_context("1h", db)
        assert len(ctx["recent_resolutions"]) == 1
        assert ctx["recent_resolutions"][0]["id"] == "s001"
        assert ctx["recent_resolutions"][0]["outcome"] == "stopped_out"

    def test_consumed_zone_from_resolved_setup(self, tmp_path):
        db = _make_db(tmp_path)
        _insert_setup(db.db_path, id="s002",
                      entry_zone_type="ob",
                      entry_zone_high=2343.0,
                      entry_zone_low=2340.0,
                      outcome="tp1")
        ctx = build_recent_context("1h", db)
        assert len(ctx["consumed_zones"]) == 1
        assert ctx["consumed_zones"][0]["zone_type"] == "ob"
        assert ctx["consumed_zones"][0]["high"] == 2343.0
        assert ctx["consumed_zones"][0]["low"] == 2340.0

    def test_swept_liquidity_from_stopped_out(self, tmp_path):
        db = _make_db(tmp_path)
        _insert_setup(db.db_path, id="s003",
                      direction="long", sl_price=2338.00,
                      outcome="stopped_out")
        ctx = build_recent_context("1h", db)
        assert len(ctx["swept_liquidity"]) == 1
        assert ctx["swept_liquidity"][0]["level"] == 2338.00
        assert ctx["swept_liquidity"][0]["type"] == "SSL"

    def test_swept_liquidity_bsl_for_short(self, tmp_path):
        db = _make_db(tmp_path)
        _insert_setup(db.db_path, id="s004",
                      direction="short", sl_price=2360.00,
                      outcome="stopped_out")
        ctx = build_recent_context("1h", db)
        assert len(ctx["swept_liquidity"]) == 1
        assert ctx["swept_liquidity"][0]["type"] == "BSL"

    def test_no_swept_liquidity_for_tp(self, tmp_path):
        db = _make_db(tmp_path)
        _insert_setup(db.db_path, id="s005", outcome="tp1")
        ctx = build_recent_context("1h", db)
        assert len(ctx["swept_liquidity"]) == 0

    def test_active_setups_filtered_by_timeframe(self, tmp_path):
        db = _make_db(tmp_path)
        db.get_pending.return_value = [
            {"id": "p1", "timeframe": "1h", "direction": "long",
             "entry_price": 2340.0, "sl_price": 2336.0, "tp1": 2350.0,
             "setup_quality": "A", "killzone": "London"},
            {"id": "p2", "timeframe": "4h", "direction": "short",
             "entry_price": 2360.0, "sl_price": 2370.0, "tp1": 2340.0,
             "setup_quality": "B", "killzone": "NY_AM"},
        ]
        ctx = build_recent_context("1h", db)
        assert len(ctx["active_setups"]) == 1
        assert ctx["active_setups"][0]["id"] == "p1"

    def test_old_resolutions_excluded(self, tmp_path):
        """Resolutions > 24h old should not appear."""
        db = _make_db(tmp_path)
        old_time = (datetime.utcnow() - timedelta(hours=30)).isoformat()
        _insert_setup(db.db_path, id="sold", resolved_at=old_time)
        ctx = build_recent_context("1h", db)
        assert len(ctx["recent_resolutions"]) == 0

    def test_wrong_timeframe_excluded(self, tmp_path):
        db = _make_db(tmp_path)
        _insert_setup(db.db_path, id="s4h", timeframe="4h")
        ctx = build_recent_context("1h", db)
        assert len(ctx["recent_resolutions"]) == 0

    def test_max_three_resolutions(self, tmp_path):
        db = _make_db(tmp_path)
        for i in range(5):
            resolved = (datetime.utcnow() - timedelta(minutes=10 * (i + 1))).isoformat()
            _insert_setup(db.db_path, id=f"sm{i}",
                          resolved_at=resolved, outcome="tp1")
        ctx = build_recent_context("1h", db)
        assert len(ctx["recent_resolutions"]) == 3

    def test_no_consumed_zone_when_missing_entry_zone(self, tmp_path):
        db = _make_db(tmp_path)
        _insert_setup(db.db_path, id="snozone",
                      entry_zone_type=None, entry_zone_high=None,
                      entry_zone_low=None, outcome="tp1")
        ctx = build_recent_context("1h", db)
        assert len(ctx["consumed_zones"]) == 0


# ── Tests: format_recent_context() ──────────────────────────────────

class TestFormatRecentContext:

    def test_empty_context_returns_empty_string(self):
        assert format_recent_context({}) == ""

    def test_no_activity_returns_empty_string(self):
        ctx = {
            "recent_resolutions": [],
            "consumed_zones": [],
            "swept_liquidity": [],
            "active_setups": [],
        }
        assert format_recent_context(ctx) == ""

    def test_resolution_appears_in_output(self):
        ctx = {
            "recent_resolutions": [{
                "direction": "long",
                "outcome": "stopped_out",
                "entry_price": 2341.50,
                "sl_price": 2338.00,
                "pnl_rr": -1.0,
                "resolved_at": (datetime.utcnow() - timedelta(minutes=20)).isoformat(),
            }],
            "consumed_zones": [],
            "swept_liquidity": [],
            "active_setups": [],
        }
        result = format_recent_context(ctx)
        assert "RECENT CONTEXT" in result
        assert "LONG" in result
        assert "STOPPED OUT" in result
        assert "2,341.50" in result

    def test_consumed_zone_in_output(self):
        ctx = {
            "recent_resolutions": [{"direction": "long", "outcome": "tp1",
                                     "entry_price": 2340, "sl_price": 2336,
                                     "pnl_rr": 2.0,
                                     "resolved_at": datetime.utcnow().isoformat()}],
            "consumed_zones": [{"zone_type": "ob", "high": 2343.0,
                                "low": 2340.0, "setup_id": "s1", "outcome": "tp1"}],
            "swept_liquidity": [],
            "active_setups": [],
        }
        result = format_recent_context(ctx)
        assert "CONSUMED OB" in result
        assert "2,343.00" in result

    def test_swept_liquidity_in_output(self):
        ctx = {
            "recent_resolutions": [{"direction": "long", "outcome": "stopped_out",
                                     "entry_price": 2340, "sl_price": 2338,
                                     "pnl_rr": -1.0,
                                     "resolved_at": datetime.utcnow().isoformat()}],
            "consumed_zones": [],
            "swept_liquidity": [{"level": 2338.00, "type": "SSL",
                                 "swept_at": datetime.utcnow().isoformat(),
                                 "setup_id": "s3"}],
            "active_setups": [],
        }
        result = format_recent_context(ctx)
        assert "SSL SWEPT" in result
        assert "2,338.00" in result

    def test_active_setups_in_output(self):
        ctx = {
            "recent_resolutions": [],
            "consumed_zones": [],
            "swept_liquidity": [],
            "active_setups": [{"direction": "short", "entry_price": 2360.0}],
        }
        result = format_recent_context(ctx)
        assert "SHORT" in result
        assert "2,360.00" in result

    def test_no_active_setups_message(self):
        ctx = {
            "recent_resolutions": [{"direction": "long", "outcome": "tp1",
                                     "entry_price": 2340, "sl_price": 2336,
                                     "resolved_at": datetime.utcnow().isoformat()}],
            "consumed_zones": [],
            "swept_liquidity": [],
            "active_setups": [],
        }
        result = format_recent_context(ctx)
        assert "None on this timeframe" in result

    def test_implication_for_stopped_out(self):
        ctx = {
            "recent_resolutions": [{"direction": "long", "outcome": "stopped_out",
                                     "entry_price": 2340, "sl_price": 2336,
                                     "resolved_at": datetime.utcnow().isoformat()}],
            "consumed_zones": [],
            "swept_liquidity": [],
            "active_setups": [],
        }
        result = format_recent_context(ctx)
        assert "IMPLICATION" in result
        assert "Bearish pressure" in result

    def test_implication_for_tp_hit(self):
        ctx = {
            "recent_resolutions": [{"direction": "long", "outcome": "tp2",
                                     "entry_price": 2340, "sl_price": 2336,
                                     "pnl_rr": 3.5,
                                     "resolved_at": datetime.utcnow().isoformat()}],
            "consumed_zones": [],
            "swept_liquidity": [],
            "active_setups": [],
        }
        result = format_recent_context(ctx)
        assert "IMPLICATION" in result
        assert "consumed" in result

    def test_end_marker_present(self):
        ctx = {
            "recent_resolutions": [{"direction": "long", "outcome": "tp1",
                                     "entry_price": 2340, "sl_price": 2336,
                                     "resolved_at": datetime.utcnow().isoformat()}],
            "consumed_zones": [],
            "swept_liquidity": [],
            "active_setups": [],
        }
        result = format_recent_context(ctx)
        assert "=== END RECENT CONTEXT ===" in result


# ── Tests: helpers ──────────────────────────────────────────────────

class TestHelpers:

    def test_time_ago_minutes(self):
        ts = (datetime.utcnow() - timedelta(minutes=15)).isoformat()
        assert "15 min ago" == _time_ago(ts)

    def test_time_ago_hours(self):
        ts = (datetime.utcnow() - timedelta(hours=3)).isoformat()
        assert "3h ago" == _time_ago(ts)

    def test_time_ago_just_now(self):
        ts = datetime.utcnow().isoformat()
        assert "just now" == _time_ago(ts)

    def test_time_ago_empty(self):
        assert "unknown" == _time_ago("")

    def test_fmt_price_normal(self):
        assert "2,341.50" == _fmt_price(2341.5)

    def test_fmt_price_none(self):
        assert "?" == _fmt_price(None)

    def test_fmt_price_zero(self):
        assert "0.00" == _fmt_price(0)
