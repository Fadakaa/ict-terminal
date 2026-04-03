"""Tests for execution cost simulator."""
import pytest
from ml.execution import ExecutionSimulator, _hour_to_session, SESSION_SPREADS
from ml.config import make_test_config


class TestHourToSession:
    def test_london(self):
        assert _hour_to_session(8) == "london"
        assert _hour_to_session(11) == "london"

    def test_new_york(self):
        assert _hour_to_session(17) == "new_york"

    def test_overlap(self):
        assert _hour_to_session(13) == "overlap_london_ny"

    def test_asian(self):
        assert _hour_to_session(3) == "asian"

    def test_off_hours(self):
        assert _hour_to_session(22) == "off_hours"


class TestExecutionSimulator:
    @pytest.fixture
    def sim(self):
        cfg = make_test_config(default_slippage_pips=0.5)
        return ExecutionSimulator(config=cfg)

    @pytest.fixture
    def sample_trades(self):
        return [
            {
                "candle_index": 55,
                "direction": "long",
                "outcome": "tp1_hit",
                "won": True,
                "max_favorable_atr": 2.0,
                "max_drawdown_atr": 0.5,
                "price_action_range_atr": 1.0,
                "ob_count": 2,
                "fvg_count": 1,
                "confluence_score": 3,
            },
            {
                "candle_index": 60,
                "direction": "short",
                "outcome": "stopped_out",
                "won": False,
                "max_favorable_atr": 0.3,
                "max_drawdown_atr": 1.5,
                "price_action_range_atr": 1.0,
                "ob_count": 1,
                "fvg_count": 0,
                "confluence_score": 2,
            },
        ]

    @pytest.fixture
    def sample_candles(self):
        candles = []
        for i in range(100):
            candles.append({
                "datetime": f"2026-03-10 {i % 24:02d}:00:00",
                "open": 2600 + i * 0.5,
                "high": 2603 + i * 0.5,
                "low": 2598 + i * 0.5,
                "close": 2601 + i * 0.5,
            })
        return candles

    def test_simulate_returns_list(self, sim, sample_trades, sample_candles):
        result = sim.simulate(sample_trades, sample_candles)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_execution_cost_added(self, sim, sample_trades, sample_candles):
        result = sim.simulate(sample_trades, sample_candles)
        for trade in result:
            assert "execution_cost_atr" in trade
            assert trade["execution_cost_atr"] >= 0

    def test_session_spread_recorded(self, sim, sample_trades, sample_candles):
        result = sim.simulate(sample_trades, sample_candles)
        for trade in result:
            assert "session_spread" in trade
            assert trade["session_spread"] > 0

    def test_mfe_reduced_by_cost(self, sim, sample_trades, sample_candles):
        original_mfe = sample_trades[0]["max_favorable_atr"]
        result = sim.simulate(sample_trades, sample_candles)
        winning_trade = [t for t in result if t["candle_index"] == 55]
        if winning_trade:
            assert winning_trade[0]["max_favorable_atr"] <= original_mfe

    def test_relabel_when_cost_kills_edge(self, sim, sample_candles):
        """A trade with tiny MFE should get relabeled to stopped_out."""
        trades = [{
            "candle_index": 55,
            "direction": "long",
            "outcome": "tp1_hit",
            "won": True,
            "max_favorable_atr": 0.001,  # Tiny edge
            "max_drawdown_atr": 0.5,
            "price_action_range_atr": 1.0,
        }]
        result = sim.simulate(trades, sample_candles)
        assert result[0]["outcome"] == "stopped_out"
        assert result[0]["won"] is False

    def test_empty_trades(self, sim, sample_candles):
        assert sim.simulate([], sample_candles) == []
