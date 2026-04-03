"""Tests for Phase C — displacement zone persistence in NarrativeStore + recent_context.

Verifies:
  1. update_displacement_zones() creates a zone entry on the active thesis
  2. update_displacement_zones() is a no-op when no active thesis exists
  3. update_displacement_zones() appends to existing zones
  4. build_recent_context() includes displacement_confirmed_zones when set
  5. displacement_confirmed_zones absent from context when none set
"""
import json
import pytest
from datetime import datetime

from ml.scanner_db import ScannerDB
from ml.narrative_state import NarrativeStore


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def db(tmp_path):
    return ScannerDB(db_path=str(tmp_path / "test_scanner.db"))


@pytest.fixture
def ns(db):
    return NarrativeStore(db_path=db.db_path)


def _save_minimal_thesis(ns: NarrativeStore, timeframe: str = "1h"):
    """Save a minimal active thesis for testing.

    Uses the fields that pass NarrativeStore.save() guard:
    invalidation must have price_level and direction.
    """
    ns.save(timeframe, {
        "thesis": "Bullish bias — accumulation complete",
        "p3_phase": "distribution",
        "directional_bias": "bullish",
        "bias_confidence": 0.75,
        "expected_next_move": "Break above 3060",
        "invalidation": {"price_level": 3020.0, "direction": "bearish"},
        "watching_for": ["FVG fill at 3040", "OB retest at 3030"],
        "key_levels": [{"price": 3060.0, "type": "resistance"}],
    })


# ── 1. update_displacement_zones() creates entry ─────────────────────────────

class TestUpdateDisplacementZonesCreates:

    def test_update_creates_zone_entry(self, ns):
        _save_minimal_thesis(ns, "1h")
        zone = {
            "zone_high": 3060.0, "zone_low": 3050.0,
            "zone_type": "ob", "direction": "long",
            "displacement_confirmed": True, "prospect_id": "p1",
        }
        ns.update_displacement_zones("1h", zone)

        current = ns.get_current("1h")
        assert current is not None
        raw = current.get("displacement_confirmed_zones")
        assert raw is not None
        zones = json.loads(raw) if isinstance(raw, str) else raw
        assert len(zones) == 1
        assert zones[0]["zone_high"] == 3060.0
        assert zones[0]["direction"] == "long"
        assert "recorded_at" in zones[0]

    def test_recorded_at_is_iso_timestamp(self, ns):
        _save_minimal_thesis(ns, "1h")
        zone = {"zone_high": 3060.0, "zone_low": 3050.0, "direction": "long",
                "displacement_confirmed": True}
        ns.update_displacement_zones("1h", zone)

        current = ns.get_current("1h")
        raw = current.get("displacement_confirmed_zones")
        zones = json.loads(raw) if isinstance(raw, str) else raw
        # Should parse without error as ISO datetime
        datetime.fromisoformat(zones[0]["recorded_at"])


# ── 2. No active thesis → no-op ──────────────────────────────────────────────

class TestUpdateNoActivethesis:

    def test_update_no_active_thesis_is_noop(self, ns):
        """With no active thesis, update_displacement_zones should not raise."""
        zone = {"zone_high": 3060.0, "zone_low": 3050.0, "direction": "long",
                "displacement_confirmed": True}
        # Must not raise
        ns.update_displacement_zones("1h", zone)
        # No thesis to check, just confirm no crash
        assert ns.get_current("1h") is None


# ── 3. Appends to existing zones ─────────────────────────────────────────────

class TestUpdateAppendsToExisting:

    def test_update_appends_to_existing(self, ns):
        _save_minimal_thesis(ns, "1h")
        zone1 = {"zone_high": 3060.0, "zone_low": 3050.0, "direction": "long",
                  "displacement_confirmed": True}
        zone2 = {"zone_high": 3100.0, "zone_low": 3090.0, "direction": "short",
                  "displacement_confirmed": True}
        ns.update_displacement_zones("1h", zone1)
        ns.update_displacement_zones("1h", zone2)

        current = ns.get_current("1h")
        raw = current.get("displacement_confirmed_zones")
        zones = json.loads(raw) if isinstance(raw, str) else raw
        assert len(zones) == 2
        assert zones[0]["zone_high"] == 3060.0
        assert zones[1]["zone_high"] == 3100.0


# ── 4. build_recent_context() includes zones ─────────────────────────────────

class TestBuildRecentContextIncludesZones:

    def test_build_recent_context_includes_zones(self, db):
        ns = NarrativeStore(db_path=db.db_path)
        _save_minimal_thesis(ns, "1h")
        zone = {"zone_high": 3060.0, "zone_low": 3050.0, "direction": "long",
                "displacement_confirmed": True}
        ns.update_displacement_zones("1h", zone)

        from ml.recent_context import build_recent_context
        ctx = build_recent_context("1h", db)

        assert "displacement_confirmed_zones" in ctx
        zones = ctx["displacement_confirmed_zones"]
        assert len(zones) == 1
        assert zones[0]["zone_high"] == 3060.0

    def test_displacement_zones_in_formatted_prompt_text(self, db):
        """When zones are present, format_recent_context() should include them."""
        ns = NarrativeStore(db_path=db.db_path)
        _save_minimal_thesis(ns, "1h")
        zone = {"zone_high": 3060.0, "zone_low": 3050.0, "direction": "long",
                "displacement_confirmed": True}
        ns.update_displacement_zones("1h", zone)

        from ml.recent_context import build_recent_context, format_recent_context
        # Build with an active setup so the context section renders
        db.store_setup(
            direction="long", bias="bullish", entry_price=3040.0,
            sl_price=3025.0, calibrated_sl=3023.0,
            tps=[3080.0], setup_quality="A",
            killzone="london", rr_ratios=[2.0],
            analysis_json={}, calibration_json={},
            timeframe="1h", status="pending",
        )

        ctx = build_recent_context("1h", db)
        text = format_recent_context(ctx)
        assert "DISPLACEMENT-CONFIRMED ZONES" in text
        assert "3,050" in text or "3050" in text


# ── 5. Zones absent when none set ────────────────────────────────────────────

class TestZonesAbsentWhenNoneSet:

    def test_zones_absent_when_none_set(self, db):
        """Without any displacement zone update, key should not appear in ctx."""
        from ml.recent_context import build_recent_context
        ctx = build_recent_context("1h", db)
        assert "displacement_confirmed_zones" not in ctx

    def test_zones_absent_when_no_active_thesis(self, db):
        """No active thesis → no displacement zones in context."""
        from ml.recent_context import build_recent_context
        ctx = build_recent_context("4h", db)
        assert ctx.get("displacement_confirmed_zones") is None
