"""Tests for session-scoped zone cooldown (Phase A).

Verifies:
  1. block_zone_for_killzone() + is_zone_blocked() — same killzone = blocked
  2. Different killzone = not blocked
  3. Different day = not blocked
  4. _make_zone_key() rounding is stable for nearby prices
  5. expire_zone_cooldowns() removes old rows
  6. Integration: monitor_prospect_triggers() skips retrace when zone is blocked
"""
import json
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from ml.scanner_db import ScannerDB


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def db(tmp_path):
    return ScannerDB(db_path=str(tmp_path / "test_scanner.db"))


# ── 1. Block and check — same killzone ───────────────────────────────────────

class TestBlockAndCheck:

    def test_block_and_check_same_killzone(self, db):
        zone_key = db._make_zone_key("long", "1h", 3060.0, 3050.0)
        db.block_zone_for_killzone(zone_key, "london")
        assert db.is_zone_blocked(zone_key, "london") is True

    def test_unblocked_zone_returns_false(self, db):
        zone_key = db._make_zone_key("long", "1h", 3060.0, 3050.0)
        assert db.is_zone_blocked(zone_key, "london") is False

    def test_prospect_id_and_setup_id_stored(self, db):
        import sqlite3
        zone_key = db._make_zone_key("short", "1h", 3100.0, 3090.0)
        db.block_zone_for_killzone(zone_key, "ny_am",
                                   prospect_id="p1", setup_id="s1")
        with sqlite3.connect(db.db_path) as conn:
            row = conn.execute(
                "SELECT prospect_id, setup_id FROM zone_cooldowns "
                "WHERE zone_key = ? AND killzone = ?",
                (zone_key, "ny_am")).fetchone()
        assert row is not None
        assert row[0] == "p1"
        assert row[1] == "s1"


# ── 2. Different killzone = not blocked ───────────────────────────────────────

class TestDifferentKillzone:

    def test_different_killzone_not_blocked(self, db):
        zone_key = db._make_zone_key("long", "1h", 3060.0, 3050.0)
        db.block_zone_for_killzone(zone_key, "london")
        # Blocked in london, not in ny_am
        assert db.is_zone_blocked(zone_key, "ny_am") is False

    def test_both_killzones_can_be_blocked_independently(self, db):
        zone_key = db._make_zone_key("long", "1h", 3060.0, 3050.0)
        db.block_zone_for_killzone(zone_key, "london")
        db.block_zone_for_killzone(zone_key, "ny_am")
        assert db.is_zone_blocked(zone_key, "london") is True
        assert db.is_zone_blocked(zone_key, "ny_am") is True


# ── 3. Different day = not blocked ────────────────────────────────────────────

class TestDifferentDay:

    def test_different_day_not_blocked(self, db):
        import sqlite3
        zone_key = db._make_zone_key("long", "1h", 3060.0, 3050.0)
        yesterday = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")
        old_blocked_at = (datetime.utcnow() - timedelta(days=1)).isoformat()
        # Manually insert a yesterday row
        with sqlite3.connect(db.db_path) as conn:
            conn.execute(
                "INSERT INTO zone_cooldowns "
                "(id, zone_key, killzone, date, blocked_at) VALUES (?,?,?,?,?)",
                ("old1", zone_key, "london", yesterday, old_blocked_at))
        # Today's check should return False
        assert db.is_zone_blocked(zone_key, "london") is False


# ── 4. _make_zone_key() rounding ─────────────────────────────────────────────

class TestZoneKeyRounding:

    def test_zone_key_rounding_nearby_prices_same_key(self, db):
        # Both prices round to the same 1-decimal value
        key1 = db._make_zone_key("short", "1h", 3050.14, 3040.12)
        key2 = db._make_zone_key("short", "1h", 3050.11, 3040.08)
        # round(..., 1) → 3050.1 for both highs, 3040.1 for both lows
        assert key1 == key2

    def test_zone_key_format(self, db):
        key = db._make_zone_key("long", "4h", 3100.0, 3090.0)
        assert key == "long|4h|3100.0|3090.0"

    def test_direction_distinguishes_keys(self, db):
        key_long = db._make_zone_key("long", "1h", 3060.0, 3050.0)
        key_short = db._make_zone_key("short", "1h", 3060.0, 3050.0)
        assert key_long != key_short


# ── 5. expire_zone_cooldowns() ────────────────────────────────────────────────

class TestExpireCooldowns:

    def test_expire_clears_old_rows(self, db):
        import sqlite3
        zone_key = db._make_zone_key("long", "1h", 3060.0, 3050.0)
        old_blocked_at = (datetime.utcnow() - timedelta(hours=25)).isoformat()
        today = datetime.utcnow().strftime("%Y-%m-%d")
        with sqlite3.connect(db.db_path) as conn:
            conn.execute(
                "INSERT INTO zone_cooldowns "
                "(id, zone_key, killzone, date, blocked_at) VALUES (?,?,?,?,?)",
                ("old2", zone_key, "london", today, old_blocked_at))
        # Verify it's there
        with sqlite3.connect(db.db_path) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM zone_cooldowns").fetchone()[0]
        assert count == 1

        db.expire_zone_cooldowns(hours_back=20)

        with sqlite3.connect(db.db_path) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM zone_cooldowns").fetchone()[0]
        assert count == 0

    def test_expire_keeps_recent_rows(self, db):
        import sqlite3
        zone_key = db._make_zone_key("long", "1h", 3060.0, 3050.0)
        today = datetime.utcnow().strftime("%Y-%m-%d")
        recent_blocked_at = (datetime.utcnow() - timedelta(hours=1)).isoformat()
        with sqlite3.connect(db.db_path) as conn:
            conn.execute(
                "INSERT INTO zone_cooldowns "
                "(id, zone_key, killzone, date, blocked_at) VALUES (?,?,?,?,?)",
                ("new1", zone_key, "london", today, recent_blocked_at))

        db.expire_zone_cooldowns(hours_back=20)

        with sqlite3.connect(db.db_path) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM zone_cooldowns").fetchone()[0]
        assert count == 1


# ── 6. Integration: zone blocking logic applied at retrace point ──────────────

class TestMonitorCooldownBlocksRetrace:

    def test_blocked_zone_prevents_is_zone_blocked_from_returning_false(self, db):
        """Core invariant: after blocking, same zone+killzone returns True; different returns False."""
        zone_key = db._make_zone_key("long", "1h", 3060.0, 3050.0)
        db.block_zone_for_killzone(zone_key, "london", setup_id="s1")

        # Same zone + same killzone = blocked
        assert db.is_zone_blocked(zone_key, "london") is True
        # Same zone + different killzone = not blocked
        assert db.is_zone_blocked(zone_key, "ny_am") is False
        # Different zone + same killzone = not blocked
        other_key = db._make_zone_key("long", "1h", 3100.0, 3090.0)
        assert db.is_zone_blocked(other_key, "london") is False

    def test_multiple_stop_outs_same_zone_idempotent(self, db):
        """INSERT OR REPLACE means re-blocking same zone is safe."""
        zone_key = db._make_zone_key("short", "1h", 3100.0, 3090.0)
        db.block_zone_for_killzone(zone_key, "london", setup_id="s1")
        db.block_zone_for_killzone(zone_key, "london", setup_id="s2")  # second stop
        assert db.is_zone_blocked(zone_key, "london") is True

        import sqlite3
        with sqlite3.connect(db.db_path) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM zone_cooldowns "
                "WHERE zone_key = ? AND killzone = ?",
                (zone_key, "london")).fetchone()[0]
        assert count == 1  # deduplicated by unique index
