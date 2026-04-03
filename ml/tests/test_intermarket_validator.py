"""Tests for P7 — Intermarket Signal Validation."""
import json
import os
import pytest

from ml.intermarket_validator import IntermarketValidator


def _make_trade(outcome="tp1", killzone="London", diverging=0,
                corr=-0.4, yield_dir=-1, dxy_range=0.5):
    """Build a mock resolved trade dict with intermarket data."""
    return {
        "outcome": outcome,
        "killzone": killzone,
        "direction": "long",
        "calibration_json": json.dumps({
            "intermarket": {
                "gold_dxy_corr_20": corr,
                "gold_dxy_diverging": diverging,
                "dxy_range_position": dxy_range,
                "yield_direction": yield_dir,
            }
        }),
    }


@pytest.fixture
def validator(tmp_path):
    return IntermarketValidator(model_dir=str(tmp_path))


class TestIntermarketScoring:

    def test_divergence_correct_scores_high(self):
        """Divergence warning + loss = intermarket was correct."""
        score = IntermarketValidator.score_intermarket_signal(
            diverging=1, is_win=False, corr=-0.4, yield_dir=-1, direction="long")
        assert score >= 0.8

    def test_divergence_wrong_scores_low(self):
        """Divergence warning + win = intermarket was wrong (false alarm)."""
        score = IntermarketValidator.score_intermarket_signal(
            diverging=1, is_win=True, corr=-0.4, yield_dir=-1, direction="long")
        assert score <= 0.3

    def test_no_divergence_win_scores_neutral(self):
        """No divergence + win = no signal, neutral."""
        score = IntermarketValidator.score_intermarket_signal(
            diverging=0, is_win=True, corr=-0.4, yield_dir=-1, direction="long")
        assert 0.4 <= score <= 0.6

    def test_yield_alignment_long_falling_win(self):
        """Falling yields + long + win = yield signal was correct."""
        score = IntermarketValidator.score_intermarket_signal(
            diverging=0, is_win=True, corr=-0.4, yield_dir=-1, direction="long")
        assert score >= 0.4  # at least neutral, yield aligned

    def test_yield_misalignment_penalised(self):
        """Rising yields + long + loss = yield signal missed."""
        score = IntermarketValidator.score_intermarket_signal(
            diverging=0, is_win=False, corr=-0.4, yield_dir=1, direction="long")
        assert score <= 0.5


class TestStratifiedAnalysis:

    def test_stratify_by_divergence(self, validator):
        """Should split trades by divergence flag and compute WR."""
        trades = []
        # 10 diverging trades: 3 wins, 7 losses
        for _ in range(3):
            trades.append(_make_trade("tp1", diverging=1))
        for _ in range(7):
            trades.append(_make_trade("stopped_out", diverging=1))
        # 10 non-diverging: 7 wins, 3 losses
        for _ in range(7):
            trades.append(_make_trade("tp1", diverging=0))
        for _ in range(3):
            trades.append(_make_trade("stopped_out", diverging=0))

        result = validator.analyze(trades)
        div = result["by_divergence"]
        assert div["diverging"]["total"] == 10
        assert div["diverging"]["win_rate"] == pytest.approx(0.3, abs=0.01)
        assert div["not_diverging"]["total"] == 10
        assert div["not_diverging"]["win_rate"] == pytest.approx(0.7, abs=0.01)

    def test_stratify_by_killzone(self, validator):
        """Should compute intermarket WR per killzone."""
        trades = [
            _make_trade("tp1", killzone="London", diverging=1),
            _make_trade("stopped_out", killzone="London", diverging=1),
            _make_trade("tp1", killzone="NY_AM", diverging=1),
            _make_trade("tp1", killzone="NY_AM", diverging=1),
        ]
        result = validator.analyze(trades)
        by_kz = result["by_killzone"]
        assert by_kz["London"]["total"] == 2
        assert by_kz["NY_AM"]["win_rate"] == pytest.approx(1.0, abs=0.01)

    def test_stratify_by_yield_direction(self, validator):
        """Should split by yield direction."""
        trades = [
            _make_trade("tp1", yield_dir=-1),  # falling yields
            _make_trade("tp1", yield_dir=-1),
            _make_trade("stopped_out", yield_dir=1),  # rising yields
        ]
        result = validator.analyze(trades)
        by_yield = result["by_yield_direction"]
        assert by_yield["falling"]["win_rate"] == pytest.approx(1.0, abs=0.01)
        assert by_yield["rising"]["win_rate"] == pytest.approx(0.0, abs=0.01)

    def test_empty_trades_returns_empty(self, validator):
        result = validator.analyze([])
        assert result["total_trades"] == 0

    def test_recommendation_skip_when_noise(self, validator):
        """If divergence doesn't predict losses, recommend skipping intermarket."""
        trades = []
        # Divergence has SAME win rate as non-divergence → noise
        for _ in range(10):
            trades.append(_make_trade("tp1", diverging=1))
        for _ in range(10):
            trades.append(_make_trade("stopped_out", diverging=1))
        for _ in range(10):
            trades.append(_make_trade("tp1", diverging=0))
        for _ in range(10):
            trades.append(_make_trade("stopped_out", diverging=0))

        result = validator.analyze(trades)
        assert result["recommendation"] in ("noise", "useful", "insufficient_data")
        # 50% WR for both → noise
        assert result["recommendation"] == "noise"

    def test_recommendation_useful_when_predictive(self, validator):
        """If divergence predicts losses (lower WR), intermarket is useful."""
        trades = []
        # Divergence: 20% WR
        for _ in range(2):
            trades.append(_make_trade("tp1", diverging=1))
        for _ in range(8):
            trades.append(_make_trade("stopped_out", diverging=1))
        # Non-diverging: 70% WR
        for _ in range(7):
            trades.append(_make_trade("tp1", diverging=0))
        for _ in range(3):
            trades.append(_make_trade("stopped_out", diverging=0))

        result = validator.analyze(trades)
        assert result["recommendation"] == "useful"

    def test_persistence(self, validator, tmp_path):
        """Analysis results should be saved to disk."""
        trades = [_make_trade("tp1")] * 5 + [_make_trade("stopped_out")] * 5
        validator.analyze(trades)
        assert os.path.exists(os.path.join(str(tmp_path), "intermarket_validation.json"))
