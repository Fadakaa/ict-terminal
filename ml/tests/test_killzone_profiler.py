"""Tests for P6+P8 — Killzone Performance Profiling + Adaptive Scan Config."""
import json
import pytest

from ml.killzone_profiler import KillzoneProfiler


def _make_trade(outcome="tp1", killzone="London", quality="B",
                confluences=3, timeframe="1h"):
    return {
        "outcome": outcome,
        "killzone": killzone,
        "setup_quality": quality,
        "timeframe": timeframe,
        "analysis_json": json.dumps({
            "confluences": ["OB", "FVG", "sweep"][:confluences],
        }),
    }


@pytest.fixture
def profiler(tmp_path):
    return KillzoneProfiler(model_dir=str(tmp_path))


class TestKillzoneStats:

    def test_compute_stats_per_killzone(self, profiler):
        trades = [
            _make_trade("tp1", "London"), _make_trade("tp1", "London"),
            _make_trade("stopped_out", "London"),
            _make_trade("tp1", "Asian"), _make_trade("stopped_out", "Asian"),
            _make_trade("stopped_out", "Asian"),
        ]
        stats = profiler.compute_stats(trades)
        assert stats["London"]["total"] == 3
        assert stats["London"]["win_rate"] == pytest.approx(0.667, abs=0.01)
        assert stats["Asian"]["total"] == 3
        assert stats["Asian"]["win_rate"] == pytest.approx(0.333, abs=0.01)

    def test_stats_by_quality(self, profiler):
        """Track WR per killzone×quality grade."""
        trades = [
            _make_trade("tp1", "London", "A"),
            _make_trade("tp1", "London", "A"),
            _make_trade("stopped_out", "London", "B"),
            _make_trade("stopped_out", "London", "B"),
        ]
        stats = profiler.compute_stats(trades)
        by_q = stats["London"]["by_quality"]
        assert by_q["A"]["win_rate"] == pytest.approx(1.0, abs=0.01)
        assert by_q["B"]["win_rate"] == pytest.approx(0.0, abs=0.01)

    def test_stats_by_timeframe(self, profiler):
        trades = [
            _make_trade("tp1", "London", timeframe="1h"),
            _make_trade("stopped_out", "London", timeframe="15min"),
            _make_trade("stopped_out", "London", timeframe="15min"),
        ]
        stats = profiler.compute_stats(trades)
        by_tf = stats["London"]["by_timeframe"]
        assert by_tf["1h"]["win_rate"] == pytest.approx(1.0, abs=0.01)
        assert by_tf["15min"]["win_rate"] == pytest.approx(0.0, abs=0.01)

    def test_empty_trades(self, profiler):
        stats = profiler.compute_stats([])
        assert stats == {}


class TestQualityGates:

    def test_low_wr_killzone_gets_higher_bar(self, profiler):
        """Asian at 40% WR should require A-grade minimum."""
        trades = []
        for _ in range(4):
            trades.append(_make_trade("tp1", "Asian"))
        for _ in range(6):
            trades.append(_make_trade("stopped_out", "Asian"))
        for _ in range(8):
            trades.append(_make_trade("tp1", "London"))
        for _ in range(2):
            trades.append(_make_trade("stopped_out", "London"))

        gates = profiler.compute_quality_gates(trades, min_trades=5)
        assert gates["Asian"]["min_quality"] == "A"
        # London at 80% WR → accept B
        assert gates["London"]["min_quality"] in ("B", "C")

    def test_high_wr_killzone_accepts_lower_quality(self, profiler):
        """NY_AM at 85% WR could accept C-grade."""
        trades = []
        for _ in range(17):
            trades.append(_make_trade("tp1", "NY_AM"))
        for _ in range(3):
            trades.append(_make_trade("stopped_out", "NY_AM"))

        gates = profiler.compute_quality_gates(trades, min_trades=10)
        assert gates["NY_AM"]["min_quality"] in ("B", "C")

    def test_insufficient_data_returns_default(self, profiler):
        trades = [_make_trade("tp1", "Asian")] * 3
        gates = profiler.compute_quality_gates(trades, min_trades=10)
        # Asian has only 3 trades → default gate
        assert "Asian" not in gates or gates["Asian"]["min_quality"] == "B"

    def test_should_skip_setup(self, profiler):
        """Profiler should reject C-grade Asian when gate requires A."""
        trades = []
        for _ in range(4):
            trades.append(_make_trade("tp1", "Asian"))
        for _ in range(6):
            trades.append(_make_trade("stopped_out", "Asian"))

        profiler.compute_quality_gates(trades, min_trades=5)
        assert profiler.should_skip("Asian", "C") is True
        assert profiler.should_skip("Asian", "A") is False

    def test_unknown_killzone_not_skipped(self, profiler):
        """Unknown killzones should never be skipped."""
        assert profiler.should_skip("Unknown", "C") is False


class TestAdaptiveScanConfig:

    def test_low_wr_killzone_reduces_15min(self, profiler):
        """Asian at low WR should skip 15min scans or increase interval."""
        trades = []
        for _ in range(4):
            trades.append(_make_trade("tp1", "Asian"))
        for _ in range(6):
            trades.append(_make_trade("stopped_out", "Asian"))

        config = profiler.get_scan_config(trades, min_trades=5)
        asian = config.get("Asian", {})
        # 15min should be skipped or interval increased
        if "15min" in asian.get("skip_timeframes", []):
            assert True
        elif "15min" in asian.get("interval_overrides", {}):
            assert asian["interval_overrides"]["15min"] > 15
        else:
            # At minimum, Asian should have some restriction
            assert asian.get("restricted", False) or "skip_timeframes" in asian

    def test_high_wr_killzone_keeps_all_timeframes(self, profiler):
        """London at 80% should scan all timeframes at normal interval."""
        trades = []
        for _ in range(16):
            trades.append(_make_trade("tp1", "London"))
        for _ in range(4):
            trades.append(_make_trade("stopped_out", "London"))

        config = profiler.get_scan_config(trades, min_trades=5)
        london = config.get("London", {})
        assert london.get("skip_timeframes", []) == []

    def test_persistence(self, profiler, tmp_path):
        trades = [_make_trade("tp1", "London")] * 10
        profiler.compute_stats(trades)
        import os
        assert os.path.exists(os.path.join(str(tmp_path), "killzone_profile.json"))
