"""Tests for the Bayesian belief updater."""
import pytest
from copy import deepcopy

from ml.bayesian import get_default_prior, update_beliefs, get_beliefs, adjust_confidence
from ml.config import make_test_config


@pytest.fixture
def prior():
    return get_default_prior()


@pytest.fixture
def test_config():
    return make_test_config()


# ── get_default_prior ────────────────────────────────────────

class TestGetDefaultPrior:
    def test_returns_correct_keys(self):
        p = get_default_prior()
        expected = {
            "alpha", "beta_param", "consecutive_losses", "max_consecutive_losses",
            "current_drawdown", "max_drawdown", "total_trades", "total_wins",
            "total_losses", "cumulative_pnl", "peak_pnl",
        }
        assert set(p.keys()) == expected

    def test_uniform_prior(self):
        p = get_default_prior()
        assert p["alpha"] == 1
        assert p["beta_param"] == 1

    def test_all_counters_zero(self):
        p = get_default_prior()
        assert p["total_trades"] == 0
        assert p["total_wins"] == 0
        assert p["total_losses"] == 0
        assert p["consecutive_losses"] == 0
        assert p["max_consecutive_losses"] == 0
        assert p["cumulative_pnl"] == 0.0
        assert p["peak_pnl"] == 0.0
        assert p["current_drawdown"] == 0.0
        assert p["max_drawdown"] == 0.0


# ── update_beliefs ───────────────────────────────────────────

class TestUpdateBeliefs:
    def test_win_increments_alpha(self, prior):
        post = update_beliefs(prior, "tp1_hit", 50.0)
        assert post["alpha"] == 2
        assert post["beta_param"] == 1

    def test_loss_increments_beta(self, prior):
        post = update_beliefs(prior, "stopped_out", -30.0)
        assert post["beta_param"] == 2
        assert post["alpha"] == 1

    def test_tp2_and_tp3_are_wins(self, prior):
        post = update_beliefs(prior, "tp2_hit", 100.0)
        assert post["total_wins"] == 1
        post2 = update_beliefs(post, "tp3_hit", 150.0)
        assert post2["total_wins"] == 2

    def test_consecutive_losses_tracked(self, prior):
        s = update_beliefs(prior, "stopped_out", -20.0)
        assert s["consecutive_losses"] == 1
        s = update_beliefs(s, "stopped_out", -20.0)
        assert s["consecutive_losses"] == 2

    def test_consecutive_losses_reset_on_win(self, prior):
        s = update_beliefs(prior, "stopped_out", -20.0)
        s = update_beliefs(s, "stopped_out", -20.0)
        assert s["consecutive_losses"] == 2
        s = update_beliefs(s, "tp1_hit", 50.0)
        assert s["consecutive_losses"] == 0

    def test_max_consecutive_losses_preserved(self, prior):
        s = update_beliefs(prior, "stopped_out", -20.0)
        s = update_beliefs(s, "stopped_out", -20.0)
        s = update_beliefs(s, "stopped_out", -20.0)
        assert s["max_consecutive_losses"] == 3
        s = update_beliefs(s, "tp1_hit", 50.0)
        assert s["max_consecutive_losses"] == 3  # preserved

    def test_drawdown_increases_on_loss(self, prior):
        s = update_beliefs(prior, "stopped_out", -30.0)
        assert s["current_drawdown"] == 30.0
        assert s["max_drawdown"] == 30.0

    def test_drawdown_resets_when_pnl_recovers(self, prior):
        s = update_beliefs(prior, "stopped_out", -30.0)
        s = update_beliefs(s, "tp1_hit", 50.0)
        assert s["current_drawdown"] == 0.0  # pnl recovered past peak

    def test_max_drawdown_preserved(self, prior):
        s = update_beliefs(prior, "stopped_out", -50.0)
        s = update_beliefs(s, "tp1_hit", 100.0)
        assert s["max_drawdown"] == 50.0

    def test_does_not_mutate_prior(self, prior):
        original = deepcopy(prior)
        update_beliefs(prior, "tp1_hit", 50.0)
        assert prior == original

    def test_total_trades_incremented(self, prior):
        s = update_beliefs(prior, "tp1_hit", 50.0)
        s = update_beliefs(s, "stopped_out", -20.0)
        assert s["total_trades"] == 2

    def test_cumulative_pnl_tracked(self, prior):
        s = update_beliefs(prior, "tp1_hit", 50.0)
        s = update_beliefs(s, "stopped_out", -20.0)
        assert s["cumulative_pnl"] == 30.0


# ── get_beliefs ──────────────────────────────────────────────

class TestGetBeliefs:
    def test_win_rate_mean_uniform(self, prior):
        b = get_beliefs(prior)
        assert b["win_rate_mean"] == pytest.approx(0.5, abs=0.01)

    def test_win_rate_mean_after_trades(self, prior):
        s = prior
        for _ in range(3):
            s = update_beliefs(s, "tp1_hit", 50.0)
        for _ in range(2):
            s = update_beliefs(s, "stopped_out", -20.0)
        # alpha=4, beta=3 → mean = 4/7 ≈ 0.571
        b = get_beliefs(s)
        assert b["win_rate_mean"] == pytest.approx(4 / 7, abs=0.01)

    def test_credible_interval_contains_mean(self, prior):
        s = prior
        for _ in range(5):
            s = update_beliefs(s, "tp1_hit", 50.0)
        b = get_beliefs(s)
        assert b["win_rate_lower_95"] < b["win_rate_mean"] < b["win_rate_upper_95"]

    def test_returns_all_keys(self, prior):
        b = get_beliefs(prior)
        expected = {
            "win_rate_mean", "win_rate_lower_95", "win_rate_upper_95",
            "consecutive_losses", "max_consecutive_losses", "max_drawdown",
            "current_drawdown", "total_trades",
        }
        assert set(b.keys()) == expected


# ── adjust_confidence ────────────────────────────────────────

class TestAdjustConfidence:
    def test_blends_with_default_weights(self, prior):
        s = prior
        for _ in range(5):
            s = update_beliefs(s, "tp1_hit", 50.0)
        beliefs = get_beliefs(s)
        # ag=0.8, bayesian ≈ 6/7 ≈ 0.857
        result = adjust_confidence(0.8, beliefs)
        expected = 0.7 * 0.8 + 0.3 * beliefs["win_rate_mean"]
        assert result == pytest.approx(expected, abs=0.01)

    def test_capped_at_one(self, prior):
        beliefs = get_beliefs(prior)
        beliefs["win_rate_mean"] = 1.0
        result = adjust_confidence(1.0, beliefs)
        assert result <= 1.0

    def test_capped_at_zero(self, prior):
        beliefs = get_beliefs(prior)
        beliefs["win_rate_mean"] = 0.0
        result = adjust_confidence(0.0, beliefs)
        assert result >= 0.0

    def test_none_beliefs_returns_ag_only(self):
        result = adjust_confidence(0.75, None)
        assert result == 0.75
