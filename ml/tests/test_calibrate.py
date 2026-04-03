"""Tests for ML Calibrator (calibrate.py)."""
import json
import os
import random

import pytest

from ml.calibrate import MLCalibrator
from ml.config import make_test_config


def _make_candles(n=100, base_price=2900.0):
    """Generate synthetic candle data."""
    rng = random.Random(42)
    candles = []
    price = base_price
    for i in range(n):
        move = rng.gauss(0, 3)
        o = price
        c = price + move
        h = max(o, c) + abs(rng.gauss(0, 1.5))
        l = min(o, c) - abs(rng.gauss(0, 1.5))
        hour = i % 24
        candles.append({
            "datetime": f"2026-03-{10 + i // 24:02d} {hour:02d}:00:00",
            "open": round(o, 2),
            "high": round(h, 2),
            "low": round(l, 2),
            "close": round(c, 2),
        })
        price = c
    return candles


def _mock_parsed(candles=None):
    """Mock parsed analysis from ClaudeAnalysisBridge."""
    return {
        "claude_entry_price": 2920.50,
        "claude_sl_price": 2916.00,
        "claude_tp_prices": [2928.00, 2935.00],
        "claude_direction": "long",
        "claude_bias": "bullish",
        "claude_killzone": "London AM Session",
        "claude_confluence_count": 5,
        "claude_sl_distance_atr": 1.1,
        "claude_tp_distances_atr": [1.8, 3.5],
        "claude_rr_ratios": [1.6, 3.2],
        "has_ob": True,
        "ob_count": 1,
        "has_fvg": True,
        "fvg_count": 1,
        "has_bsl": False,
        "has_ssl": True,
        "liq_swept": True,
        "entry_candle_idx": 55,
        "features": {},
    }


@pytest.fixture
def calibrator():
    cfg = make_test_config()
    return MLCalibrator(config=cfg)


@pytest.fixture
def candles():
    return _make_candles(100)


class TestCalibrateTradeStructure:

    def test_returns_required_keys(self, calibrator, candles):
        result = calibrator.calibrate_trade(_mock_parsed(), candles)
        assert "claude_original" in result
        assert "calibrated" in result
        assert "adjustments" in result
        assert "confidence" in result
        assert "session_context" in result
        assert "volatility_context" in result
        assert "warnings" in result
        assert "recommendation" in result

    def test_entry_never_overridden(self, calibrator, candles):
        result = calibrator.calibrate_trade(_mock_parsed(), candles)
        assert result["calibrated"]["entry"] == 2920.50
        assert result["claude_original"]["entry"] == 2920.50

    def test_sl_source_is_string(self, calibrator, candles):
        result = calibrator.calibrate_trade(_mock_parsed(), candles)
        assert result["calibrated"]["sl_source"] in {
            "claude", "volatility", "v1_session", "bayesian",
            "autogluon", "historical", "floor"
        }

    def test_confidence_bounded(self, calibrator, candles):
        result = calibrator.calibrate_trade(_mock_parsed(), candles)
        score = result["confidence"]["score"]
        assert 0.05 <= score <= 0.95

    def test_grade_is_valid(self, calibrator, candles):
        result = calibrator.calibrate_trade(_mock_parsed(), candles)
        assert result["confidence"]["grade"] in {"A", "B", "C", "D", "F"}

    def test_recommendation_is_string(self, calibrator, candles):
        result = calibrator.calibrate_trade(_mock_parsed(), candles)
        assert isinstance(result["recommendation"], str)
        assert len(result["recommendation"]) > 10

    def test_warnings_is_list(self, calibrator, candles):
        result = calibrator.calibrate_trade(_mock_parsed(), candles)
        assert isinstance(result["warnings"], list)


class TestSLCalibration:

    def test_sl_is_numeric(self, calibrator, candles):
        result = calibrator.calibrate_trade(_mock_parsed(), candles)
        assert isinstance(result["calibrated"]["sl"], (int, float))

    def test_sl_widened_flag_set(self, calibrator, candles):
        result = calibrator.calibrate_trade(_mock_parsed(), candles)
        adj = result["adjustments"]
        assert isinstance(adj["sl_widened"], bool)
        if adj["sl_widened"]:
            assert adj["sl_widened_by"] > 0
            assert adj["sl_widened_by_atr"] > 0

    def test_tight_sl_gets_widened(self, calibrator, candles):
        """A very tight SL should trigger widening."""
        parsed = _mock_parsed()
        parsed["claude_sl_price"] = 2919.50  # very tight: only 1.0 away
        parsed["claude_sl_distance_atr"] = 0.3
        result = calibrator.calibrate_trade(parsed, candles)
        # Calibrated SL should be wider than Claude's
        cal_dist = abs(result["calibrated"]["entry"] - result["calibrated"]["sl"])
        claude_dist = abs(2920.50 - 2919.50)
        assert cal_dist >= claude_dist


class TestTPCalibration:

    def test_tps_are_list(self, calibrator, candles):
        result = calibrator.calibrate_trade(_mock_parsed(), candles)
        assert isinstance(result["calibrated"]["tps"], list)

    def test_tp_adjustment_direction_valid(self, calibrator, candles):
        result = calibrator.calibrate_trade(_mock_parsed(), candles)
        assert result["adjustments"]["tp_adjustment_direction"] in {
            "widened", "narrowed", "unchanged"
        }


class TestVolatilityContext:

    def test_has_atr(self, calibrator, candles):
        result = calibrator.calibrate_trade(_mock_parsed(), candles)
        assert result["volatility_context"]["atr_14"] > 0

    def test_has_regime(self, calibrator, candles):
        result = calibrator.calibrate_trade(_mock_parsed(), candles)
        assert result["volatility_context"]["regime"] in {
            "low", "normal", "high"
        }


class TestEmptyResult:

    def test_empty_entry_returns_f_grade(self, calibrator, candles):
        parsed = _mock_parsed()
        parsed["claude_entry_price"] = 0
        result = calibrator.calibrate_trade(parsed, candles)
        assert result["confidence"]["grade"] == "F"

    def test_no_candles_returns_f_grade(self, calibrator):
        result = calibrator.calibrate_trade(_mock_parsed(), [])
        assert result["confidence"]["grade"] == "F"


class TestSessionMapping:

    def test_london_maps(self, calibrator):
        assert calibrator._map_session("london") == "london"

    def test_overlap_maps_to_ny_am(self, calibrator):
        assert calibrator._map_session("overlap_london_ny") == "ny_am"

    def test_new_york_maps_to_ny_pm(self, calibrator):
        assert calibrator._map_session("new_york") == "ny_pm"

    def test_asian_maps_to_asia(self, calibrator):
        assert calibrator._map_session("asian") == "asia"

    def test_off_hours_maps_to_off(self, calibrator):
        assert calibrator._map_session("off_hours") == "off"


class TestSLFloorEnforcement:

    def test_sl_floor_enforced_when_tight(self, candles):
        """Claude's SL at 1.1 ATR should be floored to 3.0 ATR."""
        cfg = make_test_config(sl_floor_atr=3.0)
        cal = MLCalibrator(config=cfg)
        parsed = _mock_parsed()
        parsed["claude_sl_distance_atr"] = 1.1
        result = cal.calibrate_trade(parsed, candles)
        assert result["calibrated"]["sl_distance_atr"] >= 2.5  # At least near floor

    def test_sl_floor_not_applied_when_wider(self, candles):
        """Claude's SL at 5.0 ATR should not be floored."""
        cfg = make_test_config(sl_floor_atr=3.0)
        cal = MLCalibrator(config=cfg)
        parsed = _mock_parsed()
        # Set a wide SL: 5.0 ATR away
        parsed["claude_sl_price"] = 2920.50 - 5.0 * 4.0  # 5 * ATR
        parsed["claude_sl_distance_atr"] = 5.0
        result = cal.calibrate_trade(parsed, candles)
        # Floor should not appear in warnings
        floor_warnings = [w for w in result["warnings"] if "noise band" in w]
        assert len(floor_warnings) == 0

    def test_layer_candidates_in_output(self, calibrator, candles):
        """Output should include layer_candidates dict."""
        result = calibrator.calibrate_trade(_mock_parsed(), candles)
        assert "layer_candidates" in result
        lc = result["layer_candidates"]
        assert isinstance(lc, dict)
        # Should have at least claude and floor
        assert "claude" in lc
        assert "sl_price" in lc["claude"]
        assert "sl_distance_atr" in lc["claude"]
