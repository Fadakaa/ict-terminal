"""Tests for LayerPerformanceTracker (layer_performance.py)."""
import json
import os

import pytest

from ml.config import make_test_config
from ml.layer_performance import LayerPerformanceTracker


@pytest.fixture
def tmp_config(tmp_path):
    return make_test_config(model_dir=str(tmp_path))


@pytest.fixture
def tracker(tmp_config):
    return LayerPerformanceTracker(config=tmp_config)


def _cal_json(claude_sl_atr=1.3, vol_sl_atr=2.5, floor=3.0):
    """Build a minimal calibration_json with layer_candidates."""
    entry = 2900.0
    atr = 10.0  # $10 ATR for easy math
    return {
        "claude_original": {"entry": entry, "sl": entry - claude_sl_atr * atr},
        "calibrated": {
            "entry": entry, "sl": entry - floor * atr,
            "sl_source": "floor", "sl_distance_atr": floor,
        },
        "layer_candidates": {
            "claude": {"sl_price": entry - claude_sl_atr * atr, "sl_distance_atr": claude_sl_atr},
            "volatility": {"sl_price": entry - vol_sl_atr * atr, "sl_distance_atr": vol_sl_atr},
            "floor": {"sl_price": entry - floor * atr, "sl_distance_atr": floor},
        },
        "volatility_context": {"atr_14": atr, "effective_atr": atr},
        "session_context": {"v1_p95_drawdown": 2.0},
    }


class TestIngestTrade:

    def test_surviving_layer_counted(self, tracker):
        """Layer with SL > MAE should be counted as survived."""
        cal = _cal_json(claude_sl_atr=1.3, floor=3.0)
        tracker.ingest_trade(
            cal, outcome="tp1", mae_atr=2.0, mfe_atr=5.0,
            entry_price=2900.0, atr=10.0,
        )
        ls = tracker._stats["layers"]["floor"]
        assert ls["total"] == 1
        assert ls["survived"] == 1
        assert ls["survived_wins"] == 1

    def test_stopped_layer_counted(self, tracker):
        """Layer with SL < MAE should NOT be counted as survived."""
        cal = _cal_json(claude_sl_atr=1.3, floor=3.0)
        tracker.ingest_trade(
            cal, outcome="stopped_out", mae_atr=4.0, mfe_atr=1.0,
            entry_price=2900.0, atr=10.0,
        )
        ls = tracker._stats["layers"]["claude"]
        assert ls["total"] == 1
        assert ls["survived"] == 0  # MAE 4.0 > claude SL 1.3
        assert ls["wins"] == 0

    def test_tightest_survivor_identified(self, tracker):
        """The tightest SL that still survives should be marked."""
        cal = _cal_json(claude_sl_atr=1.3, vol_sl_atr=2.5, floor=3.0)
        # MAE=2.0: claude (1.3) stops, vol (2.5) survives, floor (3.0) survives
        # Tightest survivor = volatility at 2.5
        tracker.ingest_trade(
            cal, outcome="tp1", mae_atr=2.0, mfe_atr=5.0,
            entry_price=2900.0, atr=10.0,
        )
        assert tracker._stats["layers"]["volatility"]["tightest_survivor"] == 1
        assert tracker._stats["layers"]["floor"]["tightest_survivor"] == 0

    def test_segment_tracking(self, tracker):
        """Grade×killzone segments should accumulate separately."""
        cal = _cal_json()
        tracker.ingest_trade(
            cal, outcome="tp1", mae_atr=2.0, mfe_atr=5.0,
            entry_price=2900.0, atr=10.0,
            setup_grade="B", killzone="London",
        )
        tracker.ingest_trade(
            cal, outcome="stopped_out", mae_atr=4.0, mfe_atr=1.0,
            entry_price=2900.0, atr=10.0,
            setup_grade="A", killzone="NY_AM",
        )
        segs = tracker._stats["segments"]
        assert "B_London" in segs
        assert "A_NY_AM" in segs
        assert segs["B_London"]["floor"]["total"] == 1
        assert segs["A_NY_AM"]["floor"]["total"] == 1

    def test_skips_invalid_input(self, tracker):
        """Should silently skip when atr <= 0 or entry_price missing."""
        tracker.ingest_trade({}, outcome="tp1", mae_atr=2.0, mfe_atr=5.0,
                             entry_price=0, atr=10.0)
        tracker.ingest_trade({"a": 1}, outcome="tp1", mae_atr=2.0, mfe_atr=5.0,
                             entry_price=2900.0, atr=0)
        assert tracker._stats["total_trades"] == 0


class TestLayerReport:

    def test_report_requires_minimum_trades(self, tracker):
        """Layers with < 5 trades should be excluded from report."""
        cal = _cal_json()
        for _ in range(3):
            tracker.ingest_trade(
                cal, outcome="tp1", mae_atr=2.0, mfe_atr=5.0,
                entry_price=2900.0, atr=10.0,
            )
        report = tracker.get_layer_report()
        assert len(report) == 0  # 3 < 5

    def test_report_includes_metrics(self, tracker):
        """Report should include survival_rate, efficiency_rate, avg_sl_atr."""
        cal = _cal_json()
        for i in range(10):
            mae = 2.0 if i < 7 else 4.0  # 7 survive at floor, 3 don't
            tracker.ingest_trade(
                cal, outcome="tp1" if i < 6 else "stopped_out",
                mae_atr=mae, mfe_atr=5.0,
                entry_price=2900.0, atr=10.0,
            )
        report = tracker.get_layer_report()
        assert "floor" in report
        assert report["floor"]["trades"] == 10
        assert report["floor"]["survival_rate"] == 0.7  # 7/10
        assert report["floor"]["avg_sl_atr"] == 3.0


class TestAdaptiveFloor:

    def test_insufficient_data_returns_default(self, tracker):
        """With < 20 trades in segment, return default floor."""
        result = tracker.get_adaptive_floor("A", "London", default=3.0)
        assert result == 3.0

    def test_learned_floor_with_enough_data(self, tracker):
        """With >= 20 trades in segment, return learned value."""
        cal = _cal_json(vol_sl_atr=2.8, floor=3.0)
        for i in range(25):
            # Most trades survive at volatility (2.8 ATR), making it tightest
            tracker.ingest_trade(
                cal, outcome="tp1", mae_atr=2.0, mfe_atr=5.0,
                entry_price=2900.0, atr=10.0,
                setup_grade="B", killzone="London",
            )
        result = tracker.get_adaptive_floor("B", "London", default=3.0)
        # Should return avg SL of the most efficient layer
        assert 2.0 <= result <= 6.0
        assert result != 3.0  # Should differ from default


class TestPersistence:

    def test_save_and_reload(self, tmp_config):
        """Stats should persist across instances."""
        t1 = LayerPerformanceTracker(config=tmp_config)
        cal = _cal_json()
        for _ in range(5):
            t1.ingest_trade(
                cal, outcome="tp1", mae_atr=2.0, mfe_atr=5.0,
                entry_price=2900.0, atr=10.0,
            )
        t1.flush()

        t2 = LayerPerformanceTracker(config=tmp_config)
        assert t2._stats["total_trades"] == 5


class TestV1Decay:
    """Test V1 decay calculation (implemented in calibrate.py, tested here for logic)."""

    def test_decay_at_zero_live(self):
        """Full V1 influence at 0 live trades."""
        live_count = 0
        decay = max(0.2, 1.0 - live_count / 400)
        assert decay == 1.0

    def test_decay_at_200_live(self):
        """50% V1 influence at 200 live trades."""
        live_count = 200
        decay = max(0.2, 1.0 - live_count / 400)
        assert decay == 0.5

    def test_decay_at_400_live(self):
        """20% V1 influence at 400 live trades (minimum)."""
        live_count = 400
        decay = max(0.2, 1.0 - live_count / 400)
        assert decay == 0.2

    def test_decay_above_400_floored(self):
        """Should not go below 20% even at 1000 trades."""
        live_count = 1000
        decay = max(0.2, 1.0 - live_count / 400)
        assert decay == 0.2
