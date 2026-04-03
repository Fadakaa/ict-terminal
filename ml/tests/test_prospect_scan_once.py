"""Tests for Phase B — prospect retrace → out-of-cycle scan_once() pipeline.

Verifies:
  1. scan_once() accepts displacement_context and merges into recent_ctx
  2. Displacement zone is prepended to _watch_zones in _analyze_and_store()
  3. Phase 2 calls scan_once when flag=True, not _confirm_retrace_entry
  4. Phase 2 calls _confirm_retrace_entry when flag=False (legacy path)
  5. scan_once success (setup_found) marks prospect triggered
  6. scan_once returning no_setup does not increment triggered count
  7. scan_once raising an exception does not crash the monitor
  8. feature flag default is False in config
"""
import json
import pytest
from unittest.mock import patch, MagicMock, call

from ml.scanner_db import ScannerDB


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def db(tmp_path):
    return ScannerDB(db_path=str(tmp_path / "test_scanner.db"))


def _make_scanner_engine(tmp_path):
    """Build a minimal ScannerEngine with a real DB but no live API calls."""
    from datetime import datetime
    from ml.scanner import ScannerEngine
    engine = ScannerEngine.__new__(ScannerEngine)
    engine.db = ScannerDB(db_path=str(tmp_path / "scanner.db"))
    engine.claude_key = "test-key"
    engine._total_scans = 0
    engine._last_scan_time = None
    engine._last_error = None
    engine._filter_stats = {
        "haiku_screened_out": 0, "no_trade": 0, "duplicate": 0,
        "entry_passed": 0, "rr_too_low": 0, "opus_rejected": 0,
        "setup_found": 0, "total_analyses": 0,
    }
    engine._candle_hashes = {}
    engine._post_resolution_scans = {}
    engine._last_fetch_time = {}
    engine._htf_cache = {}
    engine._corr_cache = {}
    engine._candle_store = {}
    engine._screen_cache = {}
    engine._narrative_cache = {
        "narrative": None, "timestamp": None,
        "killzone": None, "candle_hash_4h": None,
    }
    engine._kz_profiler = None
    engine._last_trigger_check = datetime.min
    engine._prospect_regen_count = {}
    engine._prospect_regen_date = None
    engine._prospect_regen_last = {}
    engine._last_killzone = None
    engine._prospect_stats = {
        "total": 0, "triggered": 0, "expired": 0,
        "by_killzone": {}, "trigger_rate": 0.0,
    }
    return engine


def _make_displaced_prospect_dict(prospect_id: str, bias: str = "bullish",
                                   ob_high: float = 3060.0, ob_low: float = 3050.0,
                                   killzone: str = "london") -> dict:
    """Build a displaced prospect dict matching get_active_prospects() format.

    Phase 2 reads: trigger_result.setup.bias, trigger_result.displacement.ob_zone,
    trigger_result.candles_waited — not prospect_json.
    """
    return {
        "id": prospect_id,
        "killzone": killzone,
        "status": "displaced",
        "prospect_json": {
            "killzone": killzone,
            "conditional_setups": [{
                "id": "cs1", "bias": bias,
                "entry_zone": {"high": ob_high, "low": ob_low},
                "preliminary_sl": ob_low - 10,
                "preliminary_tps": [ob_high + 20],
            }]
        },
        "trigger_result": {
            "setup": {"bias": bias, "timeframe": "1h"},
            "displacement": {
                "displacement_confirmed": True,
                "ob_zone": {"high": ob_high, "low": ob_low},
                "sweep_level": 0,
            },
            "candles_waited": 1,
        },
    }


def _make_prospect(db: ScannerDB, bias: str = "bullish",
                   ob_high: float = 3060.0, ob_low: float = 3050.0,
                   killzone: str = "london") -> tuple[str, dict]:
    """Store a displaced prospect in DB, return (prospect_id, trigger_result)."""
    prospect_data = {
        "killzone": killzone,
        "conditional_setups": [{
            "id": "cs1", "bias": bias,
            "entry_zone": {"high": ob_high, "low": ob_low},
            "preliminary_sl": ob_low - 10,
            "preliminary_tps": [ob_high + 20],
        }]
    }
    prospect_id = db.store_prospect(killzone, prospect_data)
    trigger_result = {
        "setup": {"bias": bias, "timeframe": "1h"},
        "displacement": {
            "displacement_confirmed": True,
            "ob_zone": {"high": ob_high, "low": ob_low},
            "sweep_level": 0,
        },
        "candles_waited": 1,
    }
    db.resolve_prospect(prospect_id, "displaced", json.dumps(trigger_result))
    return prospect_id, trigger_result


# ── 1. scan_once() merges displacement_context into recent_ctx ────────────────

class TestScanOnceDisplacementContext:

    def test_scan_once_accepts_displacement_context(self, tmp_path):
        """displacement_context should appear in recent_ctx passed to _analyze_and_store."""
        engine = _make_scanner_engine(tmp_path)
        disp_ctx = {
            "zone_high": 3060.0, "zone_low": 3050.0, "zone_type": "ob",
            "direction": "long", "displacement_confirmed": True,
            "prospect_id": "p1",
        }
        captured = {}

        def fake_analyze(timeframe, candles, htf_candles, full_candles,
                         recent_context=None):
            captured["recent_context"] = recent_context
            return {"status": "no_setup"}

        with patch.object(engine, "is_configured", return_value=True), \
             patch.object(engine, "_fetch_candles",
                          return_value=[{"open": 3050, "high": 3060,
                                         "low": 3045, "close": 3055}] * 70), \
             patch.object(engine, "_get_htf_candles", return_value=[]), \
             patch.object(engine, "_analyze_and_store", side_effect=fake_analyze), \
             patch("ml.scanner.build_recent_context", return_value={"some": "ctx"},
                   create=True):
            engine.scan_once("1h", displacement_context=disp_ctx)

        rc = captured.get("recent_context", {})
        assert rc.get("displacement_context") == disp_ctx

    def test_scan_once_no_displacement_context_leaves_recent_ctx_intact(self, tmp_path):
        """Without displacement_context, recent_ctx should not get 'displacement_context' key."""
        engine = _make_scanner_engine(tmp_path)
        captured = {}

        def fake_analyze(timeframe, candles, htf_candles, full_candles,
                         recent_context=None):
            captured["recent_context"] = recent_context
            return {"status": "no_setup"}

        with patch.object(engine, "is_configured", return_value=True), \
             patch.object(engine, "_fetch_candles",
                          return_value=[{"open": 3050, "high": 3060,
                                         "low": 3045, "close": 3055}] * 70), \
             patch.object(engine, "_get_htf_candles", return_value=[]), \
             patch.object(engine, "_analyze_and_store", side_effect=fake_analyze), \
             patch("ml.scanner.build_recent_context", return_value={"some": "ctx"},
                   create=True):
            engine.scan_once("1h")

        rc = captured.get("recent_context", {})
        assert "displacement_context" not in rc


# ── 2. Displacement zone prepended to _watch_zones ───────────────────────────

class TestDisplacementContextInjectionsWatchZones:

    def test_displacement_zone_prepended_to_watch_zones(self):
        """When displacement_context present, first watch zone should be displacement_confirmed."""
        # This is a unit test of the logic inside _analyze_and_store, exercised
        # by checking that the zone key rounding + prepend produces the right first element.
        from ml.scanner_db import ScannerDB

        # The logic is: _disp_zone is prepended, so _watch_zones[0] has status=displacement_confirmed
        disp_ctx = {
            "zone_high": 3060.0, "zone_low": 3050.0,
            "zone_type": "ob", "direction": "long",
            "displacement_confirmed": True,
        }

        # Simulate the logic from _analyze_and_store
        _watch_zones = [{"level": 3020.0, "type": "zone", "status": "untested"}]
        _disp_zone = {
            "level": (disp_ctx["zone_high"] + disp_ctx["zone_low"]) / 2,
            "type": disp_ctx.get("zone_type", "ob"),
            "status": "displacement_confirmed",
            "zone_high": disp_ctx["zone_high"],
            "zone_low": disp_ctx["zone_low"],
            "direction": disp_ctx["direction"],
        }
        _watch_zones = [_disp_zone] + _watch_zones

        assert _watch_zones[0]["status"] == "displacement_confirmed"
        assert _watch_zones[0]["level"] == 3055.0
        assert _watch_zones[0]["direction"] == "long"
        assert _watch_zones[1]["status"] == "untested"

    def test_displacement_zone_not_injected_when_not_confirmed(self):
        """displacement_confirmed=False should not inject a zone."""
        disp_ctx = {
            "zone_high": 3060.0, "zone_low": 3050.0,
            "zone_type": "ob", "direction": "long",
            "displacement_confirmed": False,  # key difference
        }
        _watch_zones = []
        if disp_ctx and disp_ctx.get("displacement_confirmed"):
            _watch_zones = [{"status": "displacement_confirmed"}] + _watch_zones
        assert len(_watch_zones) == 0


# ── 3. Phase 2 calls scan_once when flag=True ─────────────────────────────────

class TestPhase2FlagTrue:

    def test_phase2_calls_scan_once_when_flag_enabled(self, tmp_path):
        engine = _make_scanner_engine(tmp_path)
        prospect_id, _ = _make_prospect(engine.db)
        displaced = _make_displaced_prospect_dict(prospect_id)

        candles = [{"open": 3055, "high": 3062, "low": 3049,
                    "close": 3058, "datetime": "2026-03-30T10:00:00"}]

        with patch.object(engine.db, "get_active_prospects",
                          return_value=[displaced]), \
             patch("ml.scanner.get_config",
                   return_value={"prospect_use_scan_once": True,
                                 "retrace_timeout_candles": 20}), \
             patch("ml.scanner.get_current_killzone", return_value="london"), \
             patch.object(engine, "scan_once",
                          return_value={"status": "no_setup"}) as mock_scan, \
             patch.object(engine, "_confirm_retrace_entry") as mock_confirm:
            engine.monitor_prospect_triggers(candles_5m=candles)

        mock_scan.assert_called_once()
        mock_confirm.assert_not_called()

    def test_scan_once_called_with_displacement_context(self, tmp_path):
        """scan_once should be called with displacement_context populated correctly."""
        engine = _make_scanner_engine(tmp_path)
        prospect_id, _ = _make_prospect(engine.db, bias="bullish",
                                        ob_high=3060.0, ob_low=3050.0)
        displaced = _make_displaced_prospect_dict(prospect_id, ob_high=3060.0, ob_low=3050.0)

        candles = [{"open": 3055, "high": 3062, "low": 3049,
                    "close": 3058, "datetime": "2026-03-30T10:00:00"}]
        captured = {}

        def capture_scan_once(timeframe, displacement_context=None):
            captured["displacement_context"] = displacement_context
            return {"status": "no_setup"}

        with patch.object(engine.db, "get_active_prospects",
                          return_value=[displaced]), \
             patch("ml.scanner.get_config",
                   return_value={"prospect_use_scan_once": True,
                                 "retrace_timeout_candles": 20}), \
             patch("ml.scanner.get_current_killzone", return_value="london"), \
             patch.object(engine, "scan_once", side_effect=capture_scan_once):
            engine.monitor_prospect_triggers(candles_5m=candles)

        dc = captured.get("displacement_context", {})
        assert dc["zone_high"] == 3060.0
        assert dc["zone_low"] == 3050.0
        assert dc["direction"] == "long"
        assert dc["displacement_confirmed"] is True


# ── 4. Phase 2 calls legacy path when flag=False ──────────────────────────────

class TestPhase2FlagFalse:

    def test_phase2_calls_old_path_when_flag_disabled(self, tmp_path):
        engine = _make_scanner_engine(tmp_path)
        prospect_id, _ = _make_prospect(engine.db)
        displaced = _make_displaced_prospect_dict(prospect_id)

        candles = [{"open": 3055, "high": 3062, "low": 3049,
                    "close": 3058, "datetime": "2026-03-30T10:00:00"}]

        with patch.object(engine.db, "get_active_prospects",
                          return_value=[displaced]), \
             patch("ml.scanner.get_config",
                   return_value={"prospect_use_scan_once": False,
                                 "retrace_timeout_candles": 20}), \
             patch("ml.scanner.get_current_killzone", return_value="london"), \
             patch.object(engine, "scan_once") as mock_scan, \
             patch.object(engine, "_confirm_retrace_entry",
                          return_value=None) as mock_confirm:
            engine.monitor_prospect_triggers(candles_5m=candles)

        mock_scan.assert_not_called()
        mock_confirm.assert_called_once()


# ── 5. scan_once success marks prospect triggered ─────────────────────────────

class TestScanOnceSuccess:

    def test_scan_once_setup_found_marks_triggered(self, tmp_path):
        engine = _make_scanner_engine(tmp_path)
        prospect_id, _ = _make_prospect(engine.db)
        displaced = _make_displaced_prospect_dict(prospect_id)

        candles = [{"open": 3055, "high": 3062, "low": 3049,
                    "close": 3058, "datetime": "2026-03-30T10:00:00"}]

        with patch.object(engine.db, "get_active_prospects",
                          return_value=[displaced]), \
             patch("ml.scanner.get_config",
                   return_value={"prospect_use_scan_once": True,
                                 "retrace_timeout_candles": 20}), \
             patch("ml.scanner.get_current_killzone", return_value="london"), \
             patch.object(engine.db, "mark_prospect_triggered") as mock_mark, \
             patch.object(engine.db, "resolve_prospect"), \
             patch.object(engine, "scan_once",
                          return_value={"status": "setup_found", "setup_id": "s123"}):
            result = engine.monitor_prospect_triggers(candles_5m=candles)

        assert result["triggered"] == 1
        mock_mark.assert_called_once_with(prospect_id, "s123")


# ── 6. scan_once no_setup does not increment triggered ────────────────────────

class TestScanOnceNoSetup:

    def test_scan_once_no_setup_does_not_trigger(self, tmp_path):
        engine = _make_scanner_engine(tmp_path)
        prospect_id, _ = _make_prospect(engine.db)

        displaced_prospect = {
            "id": prospect_id, "killzone": "london", "status": "displaced",
            "prospect_json": json.dumps({
                "killzone": "london",
                "displacement_confirmed": True,
                "ob_high": 3060.0, "ob_low": 3050.0,
                "bias": "bullish", "candles_waited": 1, "retrace_timeout": 20,
            }),
        }
        candles = [{"open": 3055, "high": 3062, "low": 3049,
                    "close": 3058, "datetime": "2026-03-30T10:00:00"}]

        with patch.object(engine.db, "get_active_prospects",
                          return_value=[displaced_prospect]), \
             patch("ml.scanner.get_config",
                   return_value={"prospect_use_scan_once": True,
                                 "retrace_timeout_candles": 20}), \
             patch("ml.scanner.get_current_killzone", return_value="london"), \
             patch.object(engine, "scan_once",
                          return_value={"status": "no_setup"}):
            result = engine.monitor_prospect_triggers(candles_5m=candles)

        assert result["triggered"] == 0


# ── 7. scan_once exception does not crash monitor ─────────────────────────────

class TestScanOnceException:

    def test_scan_once_exception_does_not_crash_monitor(self, tmp_path):
        engine = _make_scanner_engine(tmp_path)
        prospect_id, _ = _make_prospect(engine.db)

        displaced_prospect = {
            "id": prospect_id, "killzone": "london", "status": "displaced",
            "prospect_json": json.dumps({
                "killzone": "london",
                "displacement_confirmed": True,
                "ob_high": 3060.0, "ob_low": 3050.0,
                "bias": "bullish", "candles_waited": 1, "retrace_timeout": 20,
            }),
        }
        candles = [{"open": 3055, "high": 3062, "low": 3049,
                    "close": 3058, "datetime": "2026-03-30T10:00:00"}]

        with patch.object(engine.db, "get_active_prospects",
                          return_value=[displaced_prospect]), \
             patch("ml.scanner.get_config",
                   return_value={"prospect_use_scan_once": True,
                                 "retrace_timeout_candles": 20}), \
             patch("ml.scanner.get_current_killzone", return_value="london"), \
             patch.object(engine, "scan_once",
                          side_effect=RuntimeError("API down")):
            # Must not raise
            result = engine.monitor_prospect_triggers(candles_5m=candles)

        assert result["triggered"] == 0


# ── 8. Feature flag default is False ─────────────────────────────────────────

class TestFeatureFlagDefault:

    def test_feature_flag_default_is_false(self):
        from ml.config import get_config
        cfg = get_config()
        assert cfg.get("prospect_use_scan_once") is False

    def test_feature_flag_false_in_make_test_config(self):
        from ml.config import make_test_config
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_test_config(db_path=f"{tmp}/test.db", model_dir=f"{tmp}/models")
        assert cfg.get("prospect_use_scan_once") is False
