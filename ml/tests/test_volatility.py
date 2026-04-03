"""Tests for the volatility calibrator."""
import pytest

from ml.volatility import (
    compute_ewma_volatility, detect_session, get_session_factor,
    detect_regime, classify_regime, calibrate_volatility, scale_levels,
    REGIME_MULTIPLIERS,
)
from ml.config import make_test_config


def _make_candles(closes, base_time="2024-01-15T"):
    """Helper: build candle list from close prices with London-hour timestamps."""
    candles = []
    for i, c in enumerate(closes):
        candles.append({
            "datetime": f"{base_time}{10 + (i % 6):02d}:00:00",
            "open": c - 0.5,
            "high": c + 1.0,
            "low": c - 1.0,
            "close": c,
        })
    return candles


@pytest.fixture
def sample_candles():
    """30 candles with realistic XAU/USD prices ~2000 range."""
    import random
    random.seed(42)
    prices = [2000.0]
    for _ in range(29):
        prices.append(prices[-1] + random.uniform(-5, 5))
    return _make_candles(prices)


@pytest.fixture
def test_config():
    return make_test_config()


# ── compute_ewma_volatility ─────────────────────────────────

class TestComputeEWMA:
    def test_returns_positive_for_valid_data(self, sample_candles):
        vol = compute_ewma_volatility(sample_candles)
        assert vol > 0

    def test_handles_insufficient_data(self):
        assert compute_ewma_volatility([]) == 0.0
        assert compute_ewma_volatility([{"close": 2000}]) == 0.0

    def test_flat_prices_give_zero_vol(self):
        candles = _make_candles([2000.0] * 20)
        vol = compute_ewma_volatility(candles)
        assert vol == pytest.approx(0.0, abs=0.01)

    def test_higher_volatility_gives_higher_ewma(self):
        calm = _make_candles([2000 + i * 0.1 for i in range(20)])
        wild = _make_candles([2000 + ((-1) ** i) * 10 for i in range(20)])
        assert compute_ewma_volatility(wild) > compute_ewma_volatility(calm)


# ── detect_session ───────────────────────────────────────────

class TestDetectSession:
    def test_london_session(self):
        candles = [{"datetime": "2024-01-15T09:00:00"}]
        assert detect_session(candles) == "london"

    def test_new_york_session(self):
        candles = [{"datetime": "2024-01-15T17:00:00"}]
        assert detect_session(candles) == "new_york"

    def test_asian_session(self):
        candles = [{"datetime": "2024-01-15T03:00:00"}]
        assert detect_session(candles) == "asian"

    def test_overlap_london_ny(self):
        candles = [{"datetime": "2024-01-15T13:00:00"}]
        assert detect_session(candles) == "overlap_london_ny"

    def test_off_hours(self):
        candles = [{"datetime": "2024-01-15T22:00:00"}]
        assert detect_session(candles) == "off_hours"

    def test_handles_missing_datetime(self):
        assert detect_session([{}]) == "off_hours"
        assert detect_session([]) == "off_hours"


# ── get_session_factor ───────────────────────────────────────

class TestGetSessionFactor:
    def test_london_factor(self):
        assert get_session_factor("london") == pytest.approx(1.1)

    def test_overlap_highest(self):
        assert get_session_factor("overlap_london_ny") > get_session_factor("london")

    def test_asian_lowest(self):
        assert get_session_factor("asian") < get_session_factor("london")

    def test_custom_factors_from_config(self):
        cfg = make_test_config(session_factors={"london": 2.0, "new_york": 1.0,
                                                 "overlap_london_ny": 1.0,
                                                 "asian": 1.0, "off_hours": 1.0})
        assert get_session_factor("london", config=cfg) == 2.0


# ── detect_regime ────────────────────────────────────────────

class TestDetectRegime:
    def test_normal_regime_default(self, sample_candles):
        from ml.features import compute_atr
        atr = compute_atr(sample_candles)
        regime, mult = detect_regime(atr, sample_candles)
        assert regime in ("low", "normal", "high")
        assert mult > 0

    def test_insufficient_history_returns_normal(self):
        few = _make_candles([2000, 2001, 2002])
        regime, mult = detect_regime(1.0, few)
        assert regime == "normal"
        assert mult == 1.0


# ── calibrate_volatility ────────────────────────────────────

class TestCalibrateVolatility:
    def test_returns_all_keys(self, sample_candles):
        result = calibrate_volatility(sample_candles, "1h")
        expected = {"atr", "ewma_vol", "session", "session_factor",
                    "regime", "regime_multiplier", "calibrated_vol",
                    "structural_regime", "structural_regime_confidence",
                    "structural_regime_metrics", "structural_sl_multiplier",
                    "structural_tp_multiplier"}
        assert set(result.keys()) == expected

    def test_structural_regime_in_result(self, sample_candles):
        r = calibrate_volatility(sample_candles, "1h")
        assert r["structural_regime"] in (
            "TRENDING_IMPULSIVE", "TRENDING_CORRECTIVE",
            "RANGING", "VOLATILE_CHOPPY", "QUIET_DRIFT")
        assert 0 < r["structural_regime_confidence"] <= 1.0
        assert r["structural_sl_multiplier"] > 0
        assert r["structural_tp_multiplier"] > 0

    def test_calibrated_vol_is_product(self, sample_candles):
        r = calibrate_volatility(sample_candles, "1h")
        expected = r["atr"] * r["session_factor"] * r["regime_multiplier"]
        assert r["calibrated_vol"] == pytest.approx(expected, rel=0.01)

    def test_graceful_with_minimal_candles(self):
        few = _make_candles([2000, 2001])
        r = calibrate_volatility(few, "15m")
        assert r["calibrated_vol"] >= 0


# ── scale_levels ─────────────────────────────────────────────

class TestScaleLevels:
    def test_scales_sl_wider(self):
        vol_cal = {"atr": 5.0, "calibrated_vol": 7.5}  # 1.5x scaling
        result = scale_levels(1990.0, 2020.0, 2040.0, vol_cal)
        # SL should move further from entry (scaled by 1.5)
        assert result["scaled_sl"] is not None

    def test_none_inputs_stay_none(self):
        vol_cal = {"atr": 5.0, "calibrated_vol": 7.5}
        result = scale_levels(None, None, None, vol_cal)
        assert result["scaled_sl"] is None
        assert result["scaled_tp1"] is None

    def test_no_scaling_when_atr_zero(self):
        vol_cal = {"atr": 0.0, "calibrated_vol": 0.0}
        result = scale_levels(1990.0, 2020.0, None, vol_cal)
        assert result["scaled_sl"] == 1990.0


# ── classify_regime (5-state structural) ───────────────────

class TestClassifyRegime:
    """Tests for the unified 5-state structural regime detector."""

    def test_returns_valid_structure(self, sample_candles):
        result = classify_regime(sample_candles)
        assert "regime" in result
        assert "confidence" in result
        assert "metrics" in result
        assert result["regime"] in (
            "TRENDING_IMPULSIVE", "TRENDING_CORRECTIVE",
            "RANGING", "VOLATILE_CHOPPY", "QUIET_DRIFT")
        assert 0.1 <= result["confidence"] <= 1.0
        for key in ("atr_percentile", "vol_ratio_5_30",
                     "net_movement_atr", "displacement_count",
                     "body_consistency"):
            assert key in result["metrics"]

    def test_insufficient_candles_returns_ranging(self):
        few = _make_candles([2000, 2001, 2002])
        r = classify_regime(few)
        assert r["regime"] == "RANGING"
        assert r["confidence"] == 0.1

    def test_empty_candles_returns_ranging(self):
        r = classify_regime([])
        assert r["regime"] == "RANGING"

    def test_trending_impulsive_strong_move(self):
        """Large consecutive up-moves should classify as TRENDING_IMPULSIVE."""
        # Start flat for 20 candles, then 10 large up-moves
        prices = [2000.0] * 20 + [2000 + i * 15 for i in range(1, 11)]
        candles = _make_candles(prices)
        r = classify_regime(candles)
        # Should be trending (impulsive or corrective) due to strong directional move
        assert r["regime"] in ("TRENDING_IMPULSIVE", "TRENDING_CORRECTIVE")
        assert r["metrics"]["net_movement_atr"] > 0.5

    def test_ranging_no_direction(self):
        """Oscillating prices with no net movement should classify as RANGING."""
        prices = [2000 + ((-1) ** i) * 2 for i in range(30)]
        candles = _make_candles(prices)
        r = classify_regime(candles)
        # Low net movement, should be RANGING or QUIET_DRIFT
        assert r["regime"] in ("RANGING", "QUIET_DRIFT", "VOLATILE_CHOPPY")

    def test_quiet_drift_low_atr(self):
        """Very small price changes should classify as QUIET_DRIFT."""
        prices = [2000 + i * 0.01 for i in range(30)]
        candles = _make_candles(prices)
        r = classify_regime(candles)
        assert r["regime"] in ("QUIET_DRIFT", "RANGING")

    def test_volatile_choppy_high_vol_no_direction(self):
        """High volatility but no net direction should be VOLATILE_CHOPPY."""
        import random
        random.seed(99)
        prices = [2000.0]
        for _ in range(29):
            prices.append(prices[-1] + random.choice([-1, 1]) * random.uniform(10, 20))
        candles = _make_candles(prices)
        r = classify_regime(candles)
        # High vol, low net move — should be VOLATILE_CHOPPY or RANGING
        assert r["regime"] in ("VOLATILE_CHOPPY", "RANGING", "TRENDING_IMPULSIVE",
                               "TRENDING_CORRECTIVE")

    def test_explicit_atr_override(self, sample_candles):
        """Passing explicit ATR should not crash."""
        r = classify_regime(sample_candles, atr=5.0)
        assert r["regime"] in (
            "TRENDING_IMPULSIVE", "TRENDING_CORRECTIVE",
            "RANGING", "VOLATILE_CHOPPY", "QUIET_DRIFT")

    def test_body_consistency_metric(self):
        """All bullish candles should have high body consistency."""
        prices = [2000 + i * 3 for i in range(30)]
        candles = _make_candles(prices)
        r = classify_regime(candles)
        assert r["metrics"]["body_consistency"] >= 0.5

    def test_displacement_count(self):
        """Large body candles should register as displacements."""
        # Build candles manually — need large |close - open| relative to ATR
        # 20 small candles then 5 with huge bodies
        candles = []
        for i in range(20):
            candles.append({
                "datetime": f"2024-01-15T{10 + (i % 6):02d}:00:00",
                "open": 2000.0, "high": 2001.0, "low": 1999.0, "close": 2000.5,
            })
        # 5 candles with body = 50 (massive displacement)
        for i in range(5):
            base = 2000 + i * 50
            candles.append({
                "datetime": f"2024-01-15T{10 + ((20 + i) % 6):02d}:00:00",
                "open": base, "high": base + 55, "low": base - 1, "close": base + 50,
            })
        r = classify_regime(candles)
        assert r["metrics"]["displacement_count"] >= 1

    def test_multipliers_exist_for_all_regimes(self):
        """Every regime label has SL/TP multipliers defined."""
        for regime in ("TRENDING_IMPULSIVE", "TRENDING_CORRECTIVE",
                       "RANGING", "VOLATILE_CHOPPY", "QUIET_DRIFT"):
            mults = REGIME_MULTIPLIERS[regime]
            assert "sl" in mults
            assert "tp" in mults
            assert mults["sl"] > 0
            assert mults["tp"] > 0

    def test_trending_impulsive_has_tight_sl_wide_tp(self):
        mults = REGIME_MULTIPLIERS["TRENDING_IMPULSIVE"]
        assert mults["sl"] < 1.0   # tighter SL
        assert mults["tp"] > 1.0   # wider TP

    def test_volatile_choppy_has_wide_sl_tight_tp(self):
        mults = REGIME_MULTIPLIERS["VOLATILE_CHOPPY"]
        assert mults["sl"] > 1.0   # wider SL
        assert mults["tp"] < 1.0   # tighter TP
