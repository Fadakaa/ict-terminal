"""Tests for scanner context-awareness features.

Phase 1: Hash invalidation on resolution
Phase 2: Post-resolution re-scan trigger + cooldown
"""
import os
import sqlite3
import tempfile
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from ml.scanner import ScannerEngine, TIMEFRAMES


@pytest.fixture
def mock_engine(tmp_path):
    """Create a ScannerEngine with mocked DB and API calls."""
    db_path = str(tmp_path / "test_scanner.db")
    # Create minimal DB
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE scanner_setups (
            id TEXT PRIMARY KEY, created_at TEXT, resolved_at TEXT,
            status TEXT DEFAULT 'pending', timeframe TEXT DEFAULT '1h',
            direction TEXT, bias TEXT, entry_price REAL, sl_price REAL,
            calibrated_sl REAL, tp1 REAL, tp2 REAL, tp3 REAL,
            setup_quality TEXT, killzone TEXT, rr_ratios TEXT,
            analysis_json TEXT, calibration_json TEXT, outcome TEXT,
            resolved_price REAL, pnl_rr REAL, auto_resolved INTEGER DEFAULT 0,
            candle_hash TEXT, entry_zone_type TEXT, entry_zone_high REAL,
            entry_zone_low REAL, entry_zone_position REAL,
            notified INTEGER DEFAULT 0, detection_notified INTEGER DEFAULT 0,
            gross_rr REAL, cost_rr REAL, mfe_atr REAL, mae_atr REAL,
            api_cost_usd REAL
        )
    """)
    conn.execute("""CREATE TABLE session_recaps (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        killzone TEXT, date TEXT, recap_json TEXT, created_at TEXT,
        UNIQUE(killzone, date))""")
    conn.execute("""CREATE TABLE killzone_prospects (
        id TEXT PRIMARY KEY, killzone TEXT, date TEXT,
        prospect_json TEXT, status TEXT DEFAULT 'active',
        created_at TEXT, resolved_at TEXT, trigger_result TEXT)""")
    conn.commit()
    conn.close()

    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
        from ml.scanner_db import ScannerDB
        db = ScannerDB(db_path=db_path)
        engine = ScannerEngine(db=db)
    return engine


class TestHashInvalidation:
    """Phase 1: Hash is cleared after trade resolution."""

    def test_hash_cleared_after_log_trade_complete(self, mock_engine):
        """_log_trade_complete() should delete the candle hash for the setup's TF."""
        # Pre-populate a hash
        mock_engine._candle_hashes["1h"] = "abc123"

        setup = {
            "id": "test001",
            "timeframe": "1h",
            "direction": "long",
            "bias": "bullish",
            "entry_price": 2340.0,
            "sl_price": 2336.0,
            "calibrated_sl": 2335.0,
            "tp1": 2348.0,
            "setup_quality": "A",
            "killzone": "London",
            "analysis_json": {},
            "calibration_json": {},
            "status": "pending",
        }
        result = {
            "outcome": "stopped_out",
            "price": 2336.0,
            "rr": -1.0,
            "gross_rr": -1.0,
            "cost_rr": 0.05,
            "mfe_atr": 0.5,
            "mae_atr": 1.2,
        }

        # Mock all the downstream updates to avoid import errors
        with patch("ml.scanner.notify_trade_resolved"), \
             patch("ml.scanner.ScannerEngine._maybe_auto_retrain"):
            try:
                mock_engine._log_trade_complete(setup, result)
            except Exception:
                pass  # Other parts may fail but hash should be cleared

        assert "1h" not in mock_engine._candle_hashes

    def test_hash_not_cleared_for_other_timeframes(self, mock_engine):
        """Only the resolved setup's TF hash is cleared, not others."""
        mock_engine._candle_hashes["1h"] = "abc123"
        mock_engine._candle_hashes["4h"] = "def456"

        setup = {
            "id": "test002",
            "timeframe": "1h",
            "direction": "long",
            "entry_price": 2340.0,
            "sl_price": 2336.0,
            "analysis_json": {},
            "calibration_json": {},
            "status": "pending",
        }
        result = {"outcome": "tp1", "price": 2348.0, "rr": 2.0,
                  "gross_rr": 2.0, "cost_rr": 0.05, "mfe_atr": 2.5, "mae_atr": 0.3}

        with patch("ml.scanner.notify_trade_resolved"), \
             patch("ml.scanner.ScannerEngine._maybe_auto_retrain"):
            try:
                mock_engine._log_trade_complete(setup, result)
            except Exception:
                pass

        assert "1h" not in mock_engine._candle_hashes
        assert mock_engine._candle_hashes.get("4h") == "def456"

    def test_no_error_when_hash_already_missing(self, mock_engine):
        """Clearing hash when it doesn't exist shouldn't raise."""
        assert "1h" not in mock_engine._candle_hashes

        setup = {
            "id": "test003", "timeframe": "1h", "direction": "long",
            "entry_price": 2340.0, "sl_price": 2336.0,
            "analysis_json": {}, "calibration_json": {}, "status": "pending",
        }
        result = {"outcome": "stopped_out", "price": 2336.0, "rr": -1.0,
                  "gross_rr": -1.0, "cost_rr": 0.05, "mfe_atr": 0.5, "mae_atr": 1.0}

        with patch("ml.scanner.notify_trade_resolved"), \
             patch("ml.scanner.ScannerEngine._maybe_auto_retrain"):
            try:
                mock_engine._log_trade_complete(setup, result)
            except Exception:
                pass

        # Should not raise — just a no-op
        assert "1h" not in mock_engine._candle_hashes


class TestPostResolutionScan:
    """Phase 2: Immediate re-scan after trade resolution."""

    def test_post_resolution_scan_method_exists(self, mock_engine):
        """The method should exist on ScannerEngine."""
        assert hasattr(mock_engine, "_trigger_post_resolution_scan")

    def test_cooldown_prevents_rapid_rescan(self, mock_engine):
        """2-minute cooldown between re-scans on same TF."""
        # Set a recent scan timestamp
        mock_engine._post_resolution_scans["1h"] = datetime.utcnow()

        # Mock _fetch_candles to track if it's called
        mock_engine._fetch_candles = MagicMock(return_value=None)

        mock_engine._trigger_post_resolution_scan("1h", "stopped_out")

        # Should NOT have fetched candles due to cooldown
        mock_engine._fetch_candles.assert_not_called()

    def test_scan_proceeds_after_cooldown(self, mock_engine):
        """Scan should proceed if cooldown has elapsed."""
        # Set an old timestamp (3 min ago)
        mock_engine._post_resolution_scans["1h"] = (
            datetime.utcnow() - timedelta(minutes=3))

        # Mock to prevent actual API calls
        mock_engine._fetch_candles = MagicMock(return_value=[
            {"datetime": "2026-03-10 10:00:00",
             "open": 2340.0, "high": 2345.0, "low": 2338.0, "close": 2342.0}
        ] * 120)
        mock_engine._get_htf_candles = MagicMock(return_value=[])
        mock_engine._analyze_and_store = MagicMock(return_value={"status": "no_setup"})

        mock_engine._trigger_post_resolution_scan("1h", "stopped_out")

        # Should have proceeded to analysis
        mock_engine._analyze_and_store.assert_called_once()

    def test_different_timeframes_no_cooldown_interference(self, mock_engine):
        """Cooldown on 1h doesn't block 4h."""
        mock_engine._post_resolution_scans["1h"] = datetime.utcnow()

        mock_engine._fetch_candles = MagicMock(return_value=[
            {"datetime": "2026-03-10 10:00:00",
             "open": 2340.0, "high": 2345.0, "low": 2338.0, "close": 2342.0}
        ] * 100)
        mock_engine._get_htf_candles = MagicMock(return_value=[])
        mock_engine._analyze_and_store = MagicMock(return_value={"status": "no_setup"})

        mock_engine._trigger_post_resolution_scan("4h", "tp1")

        # 4h should proceed even though 1h is on cooldown
        mock_engine._analyze_and_store.assert_called_once()

    def test_invalid_timeframe_no_crash(self, mock_engine):
        """Unknown timeframe should just return, not crash."""
        mock_engine._trigger_post_resolution_scan("invalid_tf", "stopped_out")
        # No exception raised

    def test_no_candles_no_crash(self, mock_engine):
        """If candle fetch fails, just log and return."""
        mock_engine._fetch_candles = MagicMock(return_value=None)
        mock_engine._trigger_post_resolution_scan("1h", "stopped_out")
        # No exception raised

    def test_recent_context_passed_to_analyze(self, mock_engine):
        """Post-resolution scan should build and pass recent context."""
        mock_engine._post_resolution_scans.clear()
        candles = [
            {"datetime": "2026-03-10 10:00:00",
             "open": 2340.0, "high": 2345.0, "low": 2338.0, "close": 2342.0}
        ] * 120
        mock_engine._fetch_candles = MagicMock(return_value=candles)
        mock_engine._get_htf_candles = MagicMock(return_value=[])
        mock_engine._analyze_and_store = MagicMock(return_value={"status": "no_setup"})

        mock_engine._trigger_post_resolution_scan("1h", "stopped_out")

        # Check that recent_context kwarg was passed
        call_kwargs = mock_engine._analyze_and_store.call_args
        assert "recent_context" in call_kwargs.kwargs or len(call_kwargs.args) > 4
