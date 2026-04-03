"""Tests for Opus three-tier validation system."""
import json
import os
import pytest
from unittest.mock import patch, MagicMock

from ml.scanner import ScannerEngine
from ml.scanner_db import ScannerDB
from ml.claude_bridge import ClaudeAnalysisBridge
from ml.prompts import build_validation_prompt, OPUS_VALIDATION_SYSTEM


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def tmp_scanner_db(tmp_path):
    db_path = str(tmp_path / "test_scanner.db")
    return ScannerDB(db_path)


def _make_scanner(db):
    """Create a ScannerEngine with test defaults (no real API keys)."""
    scanner = ScannerEngine.__new__(ScannerEngine)
    scanner.db = db
    scanner.claude_key = "test"
    scanner.td_key = "test"
    scanner._last_error = None
    scanner._candle_hashes = {}
    scanner._corr_cache = {}
    scanner._htf_cache = {}
    scanner._last_fetch_time = {}
    scanner._total_scans = 0
    scanner._scans_by_tf = {}
    scanner._last_scan_time = None
    scanner._screen_cache = {}
    scanner._candle_store = {}
    scanner._narrative_cache = {"narrative": None, "timestamp": None,
                                 "killzone": None, "candle_hash_4h": None}
    scanner._CANDLE_TTL = {"5min": 240, "15min": 840, "1h": 3540,
                            "4h": 14340, "1day": 43200}
    scanner._filter_stats = {
        "haiku_screened_out": 0, "no_trade": 0, "duplicate": 0,
        "entry_passed": 0, "rr_too_low": 0, "opus_rejected": 0,
        "setup_found": 0, "total_analyses": 0,
    }
    # Stub out Haiku FN tracker (P5)
    from unittest.mock import MagicMock as _MM
    scanner._fn_tracker = _MM()
    scanner._fn_tracker.should_bypass_haiku.return_value = False
    scanner._fn_tracker.should_loosen_haiku.return_value = False
    return scanner


@pytest.fixture
def sample_analysis():
    return {
        "bias": "bearish",
        "summary": "4H dealing range shows price in premium",
        "htf_context": {"dealing_range_high": 3060, "dealing_range_low": 3020},
        "orderBlocks": [{"type": "bearish", "high": 3055, "low": 3050, "candleIndex": 45,
                         "strength": "strong", "times_tested": 0}],
        "fvgs": [{"type": "bearish", "high": 3048, "low": 3044, "startIndex": 42}],
        "liquidity": [{"type": "buyside", "price": 3058, "swept": True}],
        "structure": {"type": "choch", "direction": "bearish"},
        "entry": {"price": 3050, "direction": "short"},
        "stopLoss": {"price": 3060},
        "takeProfits": [{"price": 3035, "rr": 1.5}, {"price": 3025, "rr": 2.5}],
        "killzone": "London",
        "confluences": ["bearish OB", "bearish FVG", "BSL sweep"],
        "setup_quality": "A",
        "warnings": [],
    }


@pytest.fixture
def sample_candles():
    candles = []
    for i in range(60):
        candles.append({
            "datetime": f"2026-03-20 {i % 24:02d}:00:00",
            "open": 3050 + i * 0.1, "high": 3055 + i * 0.1,
            "low": 3045 + i * 0.1, "close": 3051 + i * 0.1,
        })
    return candles


# ── Prompt Tests ──────────────────────────────────────────────────────

class TestValidationPrompt:
    def test_builds_prompt_with_analysis(self, sample_analysis, sample_candles):
        prompt = build_validation_prompt(sample_analysis, sample_candles, [], None, "5min")
        assert "SENIOR REVIEW" in prompt
        assert "bearish" in prompt
        assert "5min" in prompt
        assert "short" in prompt.lower()

    def test_includes_intermarket_context(self, sample_analysis, sample_candles):
        im = {"narrative": "DXY falling, supports gold", "gold_pct_20": 0.5,
              "dxy_pct_20": -0.3, "gold_dxy_corr_20": -0.4, "gold_dxy_diverging": 0,
              "dxy_range_position": 0.3, "yield_direction": -1, "us10y_pct_20": -0.1,
              "session_strength": "strong"}
        prompt = build_validation_prompt(sample_analysis, sample_candles, [],
                                         im, "1h")
        assert "INTERMARKET" in prompt
        assert "DXY" in prompt

    def test_shorter_timeframe_strict_note(self, sample_analysis, sample_candles):
        prompt = build_validation_prompt(sample_analysis, sample_candles, [], None, "5min")
        assert "extra strict" in prompt.lower()

    def test_higher_timeframe_note(self, sample_analysis, sample_candles):
        prompt = build_validation_prompt(sample_analysis, sample_candles, [], None, "4h")
        assert "entry timing" in prompt.lower()

    def test_system_message_exists(self):
        assert "senior" in OPUS_VALIDATION_SYSTEM.lower()
        assert "skeptical" in OPUS_VALIDATION_SYSTEM.lower()


# ── Scanner Integration Tests ─────────────────────────────────────────

class TestOpusValidation:

    def test_c_quality_skips_opus(self, tmp_scanner_db):
        """C/D quality setups should never trigger Opus validation."""
        scanner = _make_scanner(tmp_scanner_db)

        analysis = {"entry": {"price": 3050, "direction": "short"},
                    "bias": "bearish", "setup_quality": "C",
                    "stopLoss": {"price": 3060},
                    "takeProfits": [{"price": 3040, "rr": 1.0}],
                    "killzone": "London", "warnings": []}

        # Mock Sonnet to return C quality
        with patch.object(scanner, "_call_claude_screen", return_value={"setup_possible": True}), \
             patch.object(scanner, "_call_claude", return_value=analysis), \
             patch.object(scanner, "_calibrate", return_value=None), \
             patch.object(scanner, "_fetch_correlated_candles", return_value={}), \
             patch.object(scanner, "_call_claude_validate") as mock_opus, \
             patch.object(scanner, "_hash_candles", return_value="abc"):
            scanner._analyze_and_store("5min", [{"datetime": "2026-03-20 10:00:00",
                                                  "open": 3050, "high": 3055,
                                                  "low": 3045, "close": 3048}],
                                        [], [])
            mock_opus.assert_not_called()

    def test_a_quality_triggers_opus(self, tmp_scanner_db):
        """A/B quality setups should trigger Opus validation."""
        scanner = _make_scanner(tmp_scanner_db)

        analysis = {"entry": {"price": 3050, "direction": "short"},
                    "bias": "bearish", "setup_quality": "A",
                    "stopLoss": {"price": 3060},
                    "takeProfits": [{"price": 3040, "rr": 1.0}],
                    "killzone": "London", "warnings": []}

        validation = {"verdict": "validated", "adjusted_quality": "A",
                      "validation_note": "Solid setup", "narrative_coherence": "strong",
                      "confidence_adjustment": 0.05}

        with patch.object(scanner, "_call_claude_screen", return_value={"setup_possible": True}), \
             patch.object(scanner, "_call_claude", return_value=analysis), \
             patch.object(scanner, "_calibrate", return_value=None), \
             patch.object(scanner, "_fetch_correlated_candles", return_value={}), \
             patch.object(scanner, "_call_claude_validate", return_value=validation) as mock_opus, \
             patch.object(scanner, "_hash_candles", return_value="abc"), \
             patch("ml.scanner.notify_new_setup"):
            result = scanner._analyze_and_store("5min",
                [{"datetime": "2026-03-20 10:00:00", "open": 3050,
                  "high": 3055, "low": 3045, "close": 3048}], [], [])
            mock_opus.assert_called_once()
            assert result["status"] == "setup_found"
            assert result.get("opus_validated") is True

    def test_opus_rejected_creates_shadow(self, tmp_scanner_db):
        """Opus rejection should store setup as shadow."""
        scanner = _make_scanner(tmp_scanner_db)

        analysis = {"entry": {"price": 3050, "direction": "short"},
                    "bias": "bearish", "setup_quality": "A",
                    "stopLoss": {"price": 3060},
                    "takeProfits": [{"price": 3040, "rr": 1.0}],
                    "killzone": "London", "warnings": []}

        validation = {"verdict": "rejected", "adjusted_quality": "no_trade",
                      "validation_note": "OB not from genuine displacement",
                      "narrative_coherence": "weak", "confidence_adjustment": -0.3}

        # Mock rejection policy to return "reject" (default conservative behavior)
        mock_bridge = MagicMock()
        mock_bridge.get_opus_rejection_policy.return_value = {
            "action": "reject", "false_negative_rate": 0}

        with patch.object(scanner, "_call_claude_screen", return_value={"setup_possible": True}), \
             patch.object(scanner, "_call_claude", return_value=analysis), \
             patch.object(scanner, "_calibrate", return_value=None), \
             patch.object(scanner, "_fetch_correlated_candles", return_value={}), \
             patch.object(scanner, "_call_claude_validate", return_value=validation), \
             patch.object(scanner, "_hash_candles", return_value="abc"), \
             patch("ml.claude_bridge.ClaudeAnalysisBridge", return_value=mock_bridge):
            result = scanner._analyze_and_store("5min",
                [{"datetime": "2026-03-20 10:00:00", "open": 3050,
                  "high": 3055, "low": 3045, "close": 3048}], [], [])

        assert result["status"] == "opus_rejected"

        # Verify shadow setup in DB
        all_setups = tmp_scanner_db.get_pending(include_shadow=True)
        assert len(all_setups) == 1
        assert all_setups[0]["status"] == "shadow"

        # Should NOT appear in normal pending
        normal = tmp_scanner_db.get_pending(include_shadow=False)
        assert len(normal) == 0

    def test_opus_downgrade(self, tmp_scanner_db):
        """Opus downgrade should change quality and add warning."""
        scanner = _make_scanner(tmp_scanner_db)

        analysis = {"entry": {"price": 3050, "direction": "short"},
                    "bias": "bearish", "setup_quality": "A",
                    "stopLoss": {"price": 3060},
                    "takeProfits": [{"price": 3040, "rr": 1.0}],
                    "killzone": "London", "warnings": []}

        validation = {"verdict": "downgraded", "adjusted_quality": "B",
                      "validation_note": "OB tested twice, weaker",
                      "narrative_coherence": "moderate", "confidence_adjustment": -0.1}

        with patch.object(scanner, "_call_claude_screen", return_value={"setup_possible": True}), \
             patch.object(scanner, "_call_claude", return_value=analysis), \
             patch.object(scanner, "_calibrate", return_value=None), \
             patch.object(scanner, "_fetch_correlated_candles", return_value={}), \
             patch.object(scanner, "_call_claude_validate", return_value=validation), \
             patch.object(scanner, "_hash_candles", return_value="abc"), \
             patch("ml.scanner.notify_new_setup"):
            result = scanner._analyze_and_store("5min",
                [{"datetime": "2026-03-20 10:00:00", "open": 3050,
                  "high": 3055, "low": 3045, "close": 3048}], [], [])

        assert result["status"] == "setup_found"
        assert result["quality"] == "B"

    def test_opus_failure_graceful_degradation(self, tmp_scanner_db):
        """Opus API failure should not block the setup — treat as validated."""
        scanner = _make_scanner(tmp_scanner_db)

        analysis = {"entry": {"price": 3050, "direction": "short"},
                    "bias": "bearish", "setup_quality": "A",
                    "stopLoss": {"price": 3060},
                    "takeProfits": [{"price": 3040, "rr": 1.0}],
                    "killzone": "London", "warnings": []}

        with patch.object(scanner, "_call_claude_screen", return_value={"setup_possible": True}), \
             patch.object(scanner, "_call_claude", return_value=analysis), \
             patch.object(scanner, "_calibrate", return_value=None), \
             patch.object(scanner, "_fetch_correlated_candles", return_value={}), \
             patch.object(scanner, "_call_claude_validate", return_value=None), \
             patch.object(scanner, "_hash_candles", return_value="abc"), \
             patch("ml.scanner.notify_new_setup"):
            result = scanner._analyze_and_store("5min",
                [{"datetime": "2026-03-20 10:00:00", "open": 3050,
                  "high": 3055, "low": 3045, "close": 3048}], [], [])

        # Should still create the setup (graceful degradation)
        assert result["status"] == "setup_found"


# ── Opus Tracker Tests ────────────────────────────────────────────────

class TestOpusTracker:

    def _make_bridge(self, tmp_path):
        """Create a bridge with accuracy file in tmp_path."""
        bridge = ClaudeAnalysisBridge()
        path = str(tmp_path / "accuracy.json")
        bridge._accuracy_path = path
        bridge._accuracy = bridge._load_accuracy()
        return bridge

    def test_tracker_init(self, tmp_path):
        bridge = self._make_bridge(tmp_path)
        assert "opus_tracker" not in bridge._accuracy

        bridge.update_opus_tracker("validated", True)
        assert bridge._accuracy["opus_tracker"]["total_validations"] == 1
        assert bridge._accuracy["opus_tracker"]["validated"] == 1
        assert bridge._accuracy["opus_tracker"]["validated_wins"] == 1

    def test_tracker_rejected_win(self, tmp_path):
        bridge = self._make_bridge(tmp_path)
        bridge.update_opus_tracker("rejected", True)
        assert bridge._accuracy["opus_tracker"]["rejected"] == 1
        assert bridge._accuracy["opus_tracker"]["rejected_would_have_won"] == 1

    def test_tracker_rejected_loss(self, tmp_path):
        bridge = self._make_bridge(tmp_path)
        bridge.update_opus_tracker("rejected", False)
        assert bridge._accuracy["opus_tracker"]["rejected_would_have_lost"] == 1

    def test_tracker_win_rate(self, tmp_path):
        bridge = self._make_bridge(tmp_path)
        bridge.update_opus_tracker("validated", True)
        bridge.update_opus_tracker("validated", True)
        bridge.update_opus_tracker("validated", False)
        assert bridge._accuracy["opus_tracker"]["validated_win_rate"] == round(2/3, 3)

    def test_tracker_persists(self, tmp_path):
        bridge = self._make_bridge(tmp_path)
        bridge.update_opus_tracker("validated", True)

        # Reload from same path
        bridge2 = ClaudeAnalysisBridge()
        bridge2._accuracy_path = bridge._accuracy_path
        bridge2._accuracy = bridge2._load_accuracy()
        assert bridge2._accuracy.get("opus_tracker", {}).get("validated", 0) == 1


# ── Shadow Mode DB Tests ─────────────────────────────────────────────

class TestShadowMode:

    def test_store_shadow_setup(self, tmp_scanner_db):
        setup_id = tmp_scanner_db.store_setup(
            direction="short", bias="bearish", entry_price=3050,
            sl_price=3060, calibrated_sl=3062,
            tps=[3040, 3030], setup_quality="no_trade", killzone="London",
            rr_ratios=[1.0, 2.0], analysis_json={"test": True},
            calibration_json={}, timeframe="5min", status="shadow")

        # Should appear in shadow-inclusive query
        all_pending = tmp_scanner_db.get_pending(include_shadow=True)
        assert len(all_pending) == 1
        assert all_pending[0]["status"] == "shadow"

        # Should NOT appear in normal pending
        normal = tmp_scanner_db.get_pending(include_shadow=False)
        assert len(normal) == 0

    def test_shadow_setup_resolves(self, tmp_scanner_db):
        setup_id = tmp_scanner_db.store_setup(
            direction="short", bias="bearish", entry_price=3050,
            sl_price=3060, calibrated_sl=None,
            tps=[3040], setup_quality="no_trade", killzone="London",
            rr_ratios=[1.0], analysis_json={}, calibration_json={},
            timeframe="5min", status="shadow")

        # Resolve it
        tmp_scanner_db.resolve_setup(setup_id, "tp1", resolved_price=3040,
                                     pnl_rr=1.0, auto=True)

        # Should now be in history
        history = tmp_scanner_db.get_history()
        assert len(history) == 1
        assert history[0]["outcome"] == "tp1"
