"""Tests for Phase 7: Missed Setup Recycling.

Tests that expired/missed setups feed back into the recent_context
system so Sonnet can reassess thesis continuity.
"""
import json
import pytest
import sqlite3
from datetime import datetime, timedelta

from ml.recent_context import build_recent_context, format_recent_context
from ml.scanner_db import ScannerDB


def _make_db(tmp_path):
    """Create a ScannerDB with some test setups."""
    db = ScannerDB(db_path=str(tmp_path / "test.db"))
    return db


class TestExpiredSetupInRecentContext:
    """Expired/missed setups appear in recent context."""

    def test_expired_setup_in_context(self, tmp_path):
        """Expired setups appear in the missed_setups section."""
        db = _make_db(tmp_path)

        # Create a setup then expire it
        sid = db.store_setup(
            direction="long", bias="bullish", entry_price=2350.0,
            sl_price=2340.0, calibrated_sl=2338.0,
            tps=[2370.0], setup_quality="B",
            killzone="London", rr_ratios=[2.0],
            analysis_json={"entry": {"price": 2350.0}},
            calibration_json={}, timeframe="1h",
            status="pending",
        )
        # Manually resolve as expired
        db.resolve_setup(sid, "expired")

        ctx = build_recent_context("1h", db)
        missed = ctx.get("missed_setups", [])
        assert len(missed) == 1
        assert missed[0]["direction"] == "long"
        assert missed[0]["entry_price"] == 2350.0
        assert missed[0]["outcome"] == "expired"

    def test_entry_missed_setup_in_context(self, tmp_path):
        """Entry-missed setups also appear in missed_setups."""
        db = _make_db(tmp_path)
        sid = db.store_setup(
            direction="short", bias="bearish", entry_price=2400.0,
            sl_price=2410.0, calibrated_sl=2412.0,
            tps=[2380.0], setup_quality="A",
            killzone="NY_AM", rr_ratios=[2.0],
            analysis_json={}, calibration_json={},
            timeframe="1h", status="pending",
        )
        db.resolve_setup(sid, "entry_missed")

        ctx = build_recent_context("1h", db)
        missed = ctx.get("missed_setups", [])
        assert len(missed) == 1
        assert missed[0]["direction"] == "short"
        assert missed[0]["outcome"] == "entry_missed"

    def test_no_missed_when_all_resolved_normally(self, tmp_path):
        """Normal resolutions (TP/SL) don't appear in missed_setups."""
        db = _make_db(tmp_path)
        sid = db.store_setup(
            direction="long", bias="bullish", entry_price=2350.0,
            sl_price=2340.0, calibrated_sl=2338.0,
            tps=[2370.0], setup_quality="A",
            killzone="London", rr_ratios=[2.0],
            analysis_json={}, calibration_json={},
            timeframe="1h", status="pending",
        )
        db.resolve_setup(sid, "tp1", resolved_price=2370.0, pnl_rr=2.0)

        ctx = build_recent_context("1h", db)
        missed = ctx.get("missed_setups", [])
        assert len(missed) == 0

    def test_only_recent_missed_included(self, tmp_path):
        """Missed setups older than 24h are excluded."""
        db = _make_db(tmp_path)
        sid = db.store_setup(
            direction="long", bias="bullish", entry_price=2350.0,
            sl_price=2340.0, calibrated_sl=2338.0,
            tps=[2370.0], setup_quality="B",
            killzone="London", rr_ratios=[2.0],
            analysis_json={}, calibration_json={},
            timeframe="1h", status="pending",
        )
        db.resolve_setup(sid, "expired")
        # Backdate the resolution to 30 hours ago
        old_time = (datetime.utcnow() - timedelta(hours=30)).isoformat()
        with db._conn() as conn:
            conn.execute(
                "UPDATE scanner_setups SET resolved_at = ? WHERE id = ?",
                (old_time, sid))

        ctx = build_recent_context("1h", db)
        missed = ctx.get("missed_setups", [])
        assert len(missed) == 0


class TestMissedSetupInSonnetPrompt:
    """Verify the prompt formatter includes missed setups."""

    def test_missed_setup_in_format_output(self):
        """Missed setups produce a MISSED SETUP section in the prompt."""
        ctx = {
            "recent_resolutions": [],
            "consumed_zones": [],
            "swept_liquidity": [],
            "active_setups": [],
            "missed_setups": [{
                "direction": "long",
                "entry_price": 2342.0,
                "outcome": "expired",
                "zone_description": "OB at 2342.00",
                "setup_quality": "B",
            }],
        }
        output = format_recent_context(ctx)
        assert "MISSED" in output
        assert "2342" in output

    def test_no_missed_no_section(self):
        """No missed setups = no MISSED section in output."""
        ctx = {
            "recent_resolutions": [],
            "consumed_zones": [],
            "swept_liquidity": [],
            "active_setups": [],
            "missed_setups": [],
        }
        output = format_recent_context(ctx)
        assert "MISSED" not in output

    def test_reassessment_guidance_in_prompt(self):
        """When missed setups exist, prompt asks for reassessment."""
        ctx = {
            "recent_resolutions": [],
            "consumed_zones": [],
            "swept_liquidity": [],
            "active_setups": [],
            "missed_setups": [{
                "direction": "long",
                "entry_price": 2342.0,
                "outcome": "expired",
                "zone_description": "OB at 2342.00",
                "setup_quality": "B",
            }],
        }
        output = format_recent_context(ctx)
        assert "reassess" in output.lower() or "Reassess" in output


class TestMissedSetupZoneDescription:
    """Test that zone descriptions are generated for missed setups."""

    def test_zone_from_entry_zone_type(self):
        """Zone description uses entry_zone_type when available."""
        setup = {
            "entry_price": 2342.0,
            "entry_zone_type": "ob",
            "entry_zone_high": 2345.0,
            "entry_zone_low": 2340.0,
        }
        if setup.get("entry_zone_type") and setup.get("entry_zone_high"):
            desc = (f"{setup['entry_zone_type'].upper()} at "
                    f"{setup['entry_zone_high']:.2f}-{setup['entry_zone_low']:.2f}")
        else:
            desc = f"Entry zone at {setup['entry_price']:.2f}"
        assert "OB at 2345.00-2340.00" in desc

    def test_zone_fallback_to_entry_price(self):
        """When no entry_zone_type, uses entry price only."""
        setup = {
            "entry_price": 2342.0,
        }
        if setup.get("entry_zone_type") and setup.get("entry_zone_high"):
            desc = (f"{setup['entry_zone_type'].upper()} at "
                    f"{setup['entry_zone_high']:.2f}-{setup['entry_zone_low']:.2f}")
        else:
            desc = f"Entry zone at {setup['entry_price']:.2f}"
        assert "Entry zone at 2342.00" in desc
