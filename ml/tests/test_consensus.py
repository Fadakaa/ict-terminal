"""Tests for the consensus engine."""
import pytest
from copy import deepcopy

from ml.consensus import build_consensus
from ml.config import make_test_config


@pytest.fixture
def ag_prediction():
    """Sample AutoGluon prediction result."""
    return {
        "confidence": 0.80,
        "classification": {"tp1_hit": 0.5, "tp2_hit": 0.2, "stopped_out": 0.2, "tp3_hit": 0.1},
        "suggested_sl": 1985.0,
        "suggested_tp1": 2020.0,
        "suggested_tp2": 2040.0,
        "model_status": "trained",
        "training_samples": 50,
        "feature_importances": {},
    }


@pytest.fixture
def bayesian_beliefs():
    return {
        "win_rate_mean": 0.65,
        "win_rate_lower_95": 0.45,
        "win_rate_upper_95": 0.82,
        "consecutive_losses": 0,
        "max_consecutive_losses": 2,
        "max_drawdown": 50.0,
        "current_drawdown": 0.0,
        "total_trades": 30,
    }


@pytest.fixture
def vol_calibration():
    return {
        "atr": 5.0,
        "ewma_vol": 4.5,
        "session": "london",
        "session_factor": 1.1,
        "regime": "normal",
        "regime_multiplier": 1.0,
        "calibrated_vol": 5.5,
    }


class TestBuildConsensus:
    def test_returns_all_keys(self, ag_prediction, bayesian_beliefs, vol_calibration):
        r = build_consensus(ag_prediction, bayesian_beliefs, vol_calibration)
        expected = {
            "grade", "blended_confidence", "conservative_sl", "tp1", "tp2", "tp3",
            "volatility_regime", "bayesian_win_rate", "session", "reasoning",
        }
        assert set(r.keys()) == expected

    def test_grade_A_high_confidence_normal_regime(self, ag_prediction, bayesian_beliefs, vol_calibration):
        # AG=0.80, Bayesian=0.65 → blended = 0.7*0.8 + 0.3*0.65 = 0.755 → A
        r = build_consensus(ag_prediction, bayesian_beliefs, vol_calibration)
        assert r["grade"] == "A"

    def test_grade_B_moderate_confidence(self, ag_prediction, bayesian_beliefs, vol_calibration):
        ag_prediction["confidence"] = 0.65
        bayesian_beliefs["win_rate_mean"] = 0.55
        # blended = 0.7*0.65 + 0.3*0.55 = 0.62 → B
        r = build_consensus(ag_prediction, bayesian_beliefs, vol_calibration)
        assert r["grade"] == "B"

    def test_grade_C_low_confidence(self, ag_prediction, bayesian_beliefs, vol_calibration):
        ag_prediction["confidence"] = 0.50
        bayesian_beliefs["win_rate_mean"] = 0.35
        # blended = 0.7*0.50 + 0.3*0.35 = 0.455 → C
        r = build_consensus(ag_prediction, bayesian_beliefs, vol_calibration)
        assert r["grade"] == "C"

    def test_grade_D_very_low(self, ag_prediction, bayesian_beliefs, vol_calibration):
        ag_prediction["confidence"] = 0.30
        bayesian_beliefs["win_rate_mean"] = 0.20
        # blended = 0.7*0.30 + 0.3*0.20 = 0.27 → D
        r = build_consensus(ag_prediction, bayesian_beliefs, vol_calibration)
        assert r["grade"] == "D"

    def test_high_regime_downgrades_from_A(self, ag_prediction, bayesian_beliefs, vol_calibration):
        vol_calibration["regime"] = "high"
        # blended is 0.755 which would be A, but high regime caps to B
        r = build_consensus(ag_prediction, bayesian_beliefs, vol_calibration)
        assert r["grade"] == "B"

    def test_graceful_without_bayesian(self, ag_prediction, vol_calibration):
        r = build_consensus(ag_prediction, None, vol_calibration)
        assert r["grade"] is not None
        assert r["bayesian_win_rate"] is None
        assert r["blended_confidence"] == ag_prediction["confidence"]

    def test_graceful_without_volatility(self, ag_prediction, bayesian_beliefs):
        r = build_consensus(ag_prediction, bayesian_beliefs, None)
        assert r["grade"] is not None
        assert r["volatility_regime"] is None
        assert r["conservative_sl"] == ag_prediction["suggested_sl"]

    def test_graceful_without_both(self, ag_prediction):
        r = build_consensus(ag_prediction, None, None)
        assert r["grade"] is not None
        assert r["blended_confidence"] == ag_prediction["confidence"]
        assert r["bayesian_win_rate"] is None
        assert r["volatility_regime"] is None

    def test_conservative_sl_picks_wider(self, ag_prediction, bayesian_beliefs, vol_calibration):
        # vol-scaled SL will be wider than AG SL (calibrated_vol > atr → scaling > 1)
        r = build_consensus(ag_prediction, bayesian_beliefs, vol_calibration)
        # conservative_sl should be the wider (lower for long, since SL below entry)
        assert r["conservative_sl"] is not None
        assert r["conservative_sl"] <= ag_prediction["suggested_sl"]

    def test_reasoning_list_populated(self, ag_prediction, bayesian_beliefs, vol_calibration):
        r = build_consensus(ag_prediction, bayesian_beliefs, vol_calibration)
        assert isinstance(r["reasoning"], list)
        assert len(r["reasoning"]) >= 2

    def test_does_not_mutate_inputs(self, ag_prediction, bayesian_beliefs, vol_calibration):
        orig_ag = deepcopy(ag_prediction)
        orig_b = deepcopy(bayesian_beliefs)
        orig_v = deepcopy(vol_calibration)
        build_consensus(ag_prediction, bayesian_beliefs, vol_calibration)
        assert ag_prediction == orig_ag
        assert bayesian_beliefs == orig_b
        assert vol_calibration == orig_v


class TestDefensiveMode:
    """Step 9: Defensive mode caps grade and dampens confidence."""

    def test_defensive_mode_caps_grade_at_C(self, ag_prediction, bayesian_beliefs, vol_calibration):
        """Grade A/B should be capped to C when defensive_mode=True."""
        cal = {"defensive_mode": True, "regime_adjustment": 0.7}
        r = build_consensus(ag_prediction, bayesian_beliefs, vol_calibration, calibration=cal)
        assert r["grade"] == "C"

    def test_defensive_mode_dampens_confidence(self, ag_prediction, bayesian_beliefs, vol_calibration):
        """Blended confidence should be reduced by regime_adjustment + defensive mode."""
        cal = {"defensive_mode": True, "regime_adjustment": 0.7}
        r_normal = build_consensus(ag_prediction, bayesian_beliefs, vol_calibration)
        r_defensive = build_consensus(ag_prediction, bayesian_beliefs, vol_calibration, calibration=cal)
        assert r_defensive["blended_confidence"] < r_normal["blended_confidence"]

    def test_no_defensive_mode_preserves_grade(self, ag_prediction, bayesian_beliefs, vol_calibration):
        """Without defensive_mode, grade should be computed normally."""
        cal = {"defensive_mode": False, "regime_adjustment": 1.0}
        r = build_consensus(ag_prediction, bayesian_beliefs, vol_calibration, calibration=cal)
        assert r["grade"] == "A"

    def test_defensive_mode_adds_reasoning(self, ag_prediction, bayesian_beliefs, vol_calibration):
        cal = {"defensive_mode": True, "regime_adjustment": 0.7}
        r = build_consensus(ag_prediction, bayesian_beliefs, vol_calibration, calibration=cal)
        reasoning_text = " ".join(r["reasoning"])
        assert "Defensive mode" in reasoning_text or "defensive" in reasoning_text.lower()

    def test_calibration_none_works(self, ag_prediction, bayesian_beliefs, vol_calibration):
        """calibration=None should work exactly like before."""
        r = build_consensus(ag_prediction, bayesian_beliefs, vol_calibration, calibration=None)
        assert r["grade"] is not None
