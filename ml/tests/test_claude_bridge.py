"""Tests for Claude Analysis Bridge (claude_bridge.py)."""
import json
import os
import tempfile

import pytest

from ml.claude_bridge import ClaudeAnalysisBridge
from ml.config import make_test_config


def _mock_analysis():
    """Standard mock Claude analysis JSON."""
    return {
        "bias": "bullish",
        "entry": {"price": 2920.50, "direction": "long", "rationale": "test"},
        "stopLoss": {"price": 2916.00, "rationale": "below OB"},
        "takeProfits": [
            {"price": 2928.00, "rr": 1.6, "rationale": "tp1"},
            {"price": 2935.00, "rr": 3.2, "rationale": "tp2"},
        ],
        "orderBlocks": [
            {"type": "bullish", "high": 2921, "low": 2917,
             "candleIndex": 45, "strength": "strong"}
        ],
        "fvgs": [
            {"type": "bullish", "high": 2919, "low": 2917.5,
             "startIndex": 43, "filled": False}
        ],
        "liquidity": [
            {"type": "sellside", "price": 2914, "candleIndex": 40,
             "swept": True}
        ],
        "confluences": [
            "Bullish OB", "Unfilled FVG", "SSL swept",
            "London killzone", "BOS confirmed"
        ],
        "killzone": "London AM Session",
        "setup_quality": "B",
    }


@pytest.fixture
def bridge():
    return ClaudeAnalysisBridge(config=make_test_config())


class TestParseAnalysis:

    def test_parses_entry_price(self, bridge):
        result = bridge.parse_analysis(_mock_analysis(), None)
        assert result["claude_entry_price"] == 2920.50

    def test_parses_direction(self, bridge):
        result = bridge.parse_analysis(_mock_analysis(), None)
        assert result["claude_direction"] == "long"

    def test_parses_sl_distance(self, bridge):
        result = bridge.parse_analysis(_mock_analysis(), None)
        # SL distance is 2920.50 - 2916.00 = 4.50
        # Without candles, ATR defaults to 1.0
        assert result["claude_sl_distance_atr"] == 4.5

    def test_parses_tp_prices(self, bridge):
        result = bridge.parse_analysis(_mock_analysis(), None)
        assert result["claude_tp_prices"] == [2928.00, 2935.00]

    def test_counts_confluences(self, bridge):
        result = bridge.parse_analysis(_mock_analysis(), None)
        assert result["claude_confluence_count"] == 5

    def test_detects_ob(self, bridge):
        result = bridge.parse_analysis(_mock_analysis(), None)
        assert result["has_ob"] is True
        assert result["ob_count"] == 1

    def test_detects_fvg(self, bridge):
        result = bridge.parse_analysis(_mock_analysis(), None)
        assert result["has_fvg"] is True
        assert result["fvg_count"] == 1

    def test_detects_liquidity_sweep(self, bridge):
        result = bridge.parse_analysis(_mock_analysis(), None)
        assert result["has_ssl"] is True
        assert result["liq_swept"] is True

    def test_no_entry_returns_zero(self, bridge):
        analysis = {"bias": "neutral"}
        result = bridge.parse_analysis(analysis, None)
        assert result["claude_entry_price"] == 0

    def test_setup_grade(self, bridge):
        result = bridge.parse_analysis(_mock_analysis(), None)
        assert result["claude_setup_grade"] == "B"

    def test_rr_ratios(self, bridge):
        result = bridge.parse_analysis(_mock_analysis(), None)
        assert len(result["claude_rr_ratios"]) == 2
        # TP1 distance = 7.5, risk = 4.5 → RR ≈ 1.667
        assert result["claude_rr_ratios"][0] == pytest.approx(1.6667, abs=0.01)


class TestClassifySetupType:

    def test_bull_ob_fvg_sweep_london(self, bridge):
        parsed = {
            "claude_direction": "long",
            "has_ob": True,
            "has_fvg": True,
            "liq_swept": True,
            "claude_killzone": "London AM Session",
        }
        assert bridge.classify_setup_type(parsed) == "bull_ob_fvg_sweep_london"

    def test_bear_ob_ny_am(self, bridge):
        parsed = {
            "claude_direction": "short",
            "has_ob": True,
            "has_fvg": False,
            "liq_swept": False,
            "claude_killzone": "New York AM Session",
        }
        assert bridge.classify_setup_type(parsed) == "bear_ob_ny_am"

    def test_no_ict_elements(self, bridge):
        parsed = {
            "claude_direction": "long",
            "has_ob": False,
            "has_fvg": False,
            "liq_swept": False,
            "claude_killzone": "Asian Session",
        }
        assert bridge.classify_setup_type(parsed) == "bull_asia"


class TestFindEntryCandle:

    def test_finds_nearest_low_for_long(self, bridge):
        candles = [
            {"low": 2920.0, "high": 2925.0},
            {"low": 2918.0, "high": 2923.0},
            {"low": 2920.5, "high": 2926.0},  # closest to 2920.50
        ]
        idx = bridge.find_entry_candle(candles, 2920.50, "long")
        assert idx == 2  # low of 2920.5 is nearest

    def test_finds_nearest_high_for_short(self, bridge):
        candles = [
            {"low": 2920.0, "high": 2925.0},
            {"low": 2918.0, "high": 2920.5},  # high closest to 2920.50
            {"low": 2915.0, "high": 2919.0},
        ]
        idx = bridge.find_entry_candle(candles, 2920.50, "short")
        assert idx == 1

    def test_returns_last_idx_if_no_candles(self, bridge):
        idx = bridge.find_entry_candle([], 2920.0, "long")
        assert idx == -1


class TestKillzoneMapping:

    def test_london_variants(self, bridge):
        assert bridge._map_killzone_to_session("London AM Session") == "london"
        assert bridge._map_killzone_to_session("London Open") == "london"
        assert bridge._map_killzone_to_session("london") == "london"

    def test_ny_am_variants(self, bridge):
        assert bridge._map_killzone_to_session("New York AM Session") == "ny_am"
        assert bridge._map_killzone_to_session("NY AM") == "ny_am"
        assert bridge._map_killzone_to_session("NY Open") == "ny_am"

    def test_asia_variants(self, bridge):
        assert bridge._map_killzone_to_session("Asian Session") == "asia"
        assert bridge._map_killzone_to_session("Tokyo Session") == "asia"

    def test_unknown_maps_to_off(self, bridge):
        assert bridge._map_killzone_to_session("Unknown") == "off"
        assert bridge._map_killzone_to_session("") == "off"


class TestCalibrationValue:

    def test_no_trades_returns_zero(self, bridge):
        bridge._accuracy = bridge._load_accuracy()  # fresh
        bridge._accuracy["total_trades"] = 0
        result = bridge.get_calibration_value()
        assert result["total_trades"] == 0
        assert "No trades logged" in result["recommendation"]
