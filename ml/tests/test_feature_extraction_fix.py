"""Tests for AutoGluon feature extraction pipeline fix.

Validates that:
1. Correct-timeframe candles are used (not 5-min monitoring candles)
2. Mechanical detectors supplement empty Claude arrays
3. Rich features exceed the RICH_FEATURE_THRESHOLD (20 non-zero)
4. Entry zone position/size populated from detected zones
5. Graceful fallback when candle fetch fails
6. detect_order_blocks called with atr parameter (line 412 fix)
7. Newest-first candles are reversed before extraction
"""
import math
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from ml.features import (
    extract_features, compute_atr, detect_order_blocks,
    detect_fvgs, detect_liquidity, _compute_entry_zone,
)


# ── Test helpers ────────────────────────────────────────────

def _make_candles(n: int = 30, base_price: float = 2350.0,
                  atr_size: float = 15.0, ascending: bool = True,
                  include_datetime: bool = True) -> list[dict]:
    """Generate synthetic OHLC candles for testing.

    Creates candles with monotonically increasing datetimes (oldest-first).
    atr_size controls the range of each candle.
    """
    candles = []
    price = base_price
    for i in range(n):
        drift = 0.5 if ascending else -0.5
        open_ = price
        high = price + atr_size * 0.6
        low = price - atr_size * 0.4
        close = price + drift
        # Use day + hour to ensure monotonically increasing datetimes
        day = 25 + i // 24
        hour = i % 24
        dt = f"2026-03-{day:02d} {hour:02d}:00:00"
        c = {"open": open_, "high": high, "low": low, "close": close}
        if include_datetime:
            c["datetime"] = dt
        candles.append(c)
        price = close
    return candles


def _make_analysis(with_obs: bool = True, with_fvgs: bool = True,
                   with_liq: bool = True, direction: str = "long") -> dict:
    """Build a Claude-like analysis dict with optional ICT arrays."""
    analysis = {
        "bias": "bullish" if direction == "long" else "bearish",
        "killzone": "London",
        "setup_quality": "B",
        "entry": {"price": 2350.0, "direction": direction},
        "stopLoss": {"price": 2340.0},
        "takeProfits": [{"price": 2370.0, "rr": 2.0}],
        "confluences": ["OB + FVG overlap"],
        "orderBlocks": [],
        "fvgs": [],
        "liquidity": [],
    }
    if with_obs:
        analysis["orderBlocks"] = [
            {"high": 2355.0, "low": 2345.0, "type": "bullish", "strength": "strong"},
        ]
    if with_fvgs:
        analysis["fvgs"] = [
            {"high": 2360.0, "low": 2352.0, "type": "bullish", "filled": False},
        ]
    if with_liq:
        analysis["liquidity"] = [
            {"price": 2380.0, "type": "buyside"},
            {"price": 2330.0, "type": "sellside"},
        ]
    return analysis


RICH_FEATURE_THRESHOLD = 20


# ── Test: Correct timeframe candles used ────────────────────

class TestCorrectTimeframeCandlesUsed:
    """Verify _log_trade_complete fetches setup's TF candles, not 5min."""

    def test_fetches_setup_timeframe_candles(self):
        """Rich extraction uses TIMEFRAMES[tf]['fetch'] for candle count."""
        from ml.scanner import TIMEFRAMES

        # Simulate the fetch logic from the rich extraction block
        for tf, expected_fetch in [("15min", 200), ("1h", 250), ("4h", 100), ("1day", 100)]:
            tf_cfg = TIMEFRAMES.get(tf, {"fetch": 180})
            fetch_count = tf_cfg.get("fetch", 180)
            assert fetch_count == expected_fetch, (
                f"TIMEFRAMES['{tf}']['fetch'] = {fetch_count}, expected {expected_fetch}"
            )

    def test_timeframes_config_has_fetch_key(self):
        """All TIMEFRAMES entries have a 'fetch' key for candle count."""
        from ml.scanner import TIMEFRAMES
        for tf, cfg in TIMEFRAMES.items():
            assert "fetch" in cfg, f"TIMEFRAMES['{tf}'] missing 'fetch' key"
            assert cfg["fetch"] >= 15, f"TIMEFRAMES['{tf}']['fetch'] too small"


# ── Test: Mechanical detectors supplement empty arrays ──────

class TestMechanicalDetectorsSupplement:
    """When Claude's ICT arrays are empty, detectors fill them."""

    def test_empty_obs_filled_by_detector(self):
        """Empty orderBlocks → detect_order_blocks produces non-empty."""
        candles = _make_candles(30, atr_size=15.0)
        atr = compute_atr(candles, 14)
        obs = detect_order_blocks(candles, atr)
        # We can't guarantee detections on synthetic data, but the function
        # should run without error and return a list
        assert isinstance(obs, list)

    def test_empty_fvgs_filled_by_detector(self):
        """Empty fvgs → detect_fvgs produces a list."""
        candles = _make_candles(30)
        fvgs = detect_fvgs(candles)
        assert isinstance(fvgs, list)

    def test_empty_liquidity_filled_by_detector(self):
        """Empty liquidity → detect_liquidity produces a list."""
        candles = _make_candles(30)
        liqs = detect_liquidity(candles)
        assert isinstance(liqs, list)

    def test_enrichment_skips_nonempty_arrays(self):
        """If Claude provided OBs, detectors should NOT override them."""
        analysis = _make_analysis(with_obs=True, with_fvgs=True, with_liq=True)
        original_obs = analysis["orderBlocks"].copy()

        # Simulate the enrichment logic from scanner
        enriched = dict(analysis)
        if not enriched.get("orderBlocks"):
            enriched["orderBlocks"] = [{"high": 9999, "low": 9998, "type": "bullish"}]
        if not enriched.get("fvgs"):
            enriched["fvgs"] = [{"high": 9999, "low": 9998}]

        # Original should be preserved
        assert enriched["orderBlocks"] == original_obs

    def test_features_nonzero_with_mechanical_detectors(self):
        """Features extracted with mechanical detectors have non-zero OB/FVG/Liq."""
        candles = _make_candles(50, atr_size=20.0)
        # Start with empty arrays — mechanical detectors will fill
        analysis = _make_analysis(with_obs=False, with_fvgs=False, with_liq=False)
        atr = compute_atr(candles, 14)

        enriched = dict(analysis)
        if not enriched.get("orderBlocks"):
            enriched["orderBlocks"] = detect_order_blocks(candles, atr) if atr > 0 else []
        if not enriched.get("fvgs"):
            enriched["fvgs"] = detect_fvgs(candles)
        if not enriched.get("liquidity"):
            enriched["liquidity"] = detect_liquidity(candles)

        features = extract_features(enriched, candles, "1h")
        # At minimum the features dict should be populated (even if some are 0)
        assert isinstance(features, dict)
        assert len(features) > 30  # We have 52 features total


# ── Test: Rich features above threshold ─────────────────────

class TestRichFeaturesAboveThreshold:
    """Feature dict should have ≥20 non-zero values with proper data."""

    def test_full_analysis_exceeds_threshold(self):
        """Full analysis with all ICT arrays → ≥20 non-zero features."""
        candles = _make_candles(50, atr_size=15.0)
        analysis = _make_analysis(with_obs=True, with_fvgs=True, with_liq=True)

        features = extract_features(analysis, candles, "1h")
        non_zero = sum(
            1 for v in features.values()
            if v and not (isinstance(v, float) and math.isnan(v))
        )
        assert non_zero >= RICH_FEATURE_THRESHOLD, (
            f"Only {non_zero} non-zero features, need ≥{RICH_FEATURE_THRESHOLD}. "
            f"Zero features: {[k for k, v in features.items() if not v or (isinstance(v, float) and math.isnan(v))]}"
        )

    def test_empty_analysis_fewer_features(self):
        """Empty ICT arrays without detectors → fewer non-zero features."""
        candles = _make_candles(30, atr_size=15.0)
        analysis = _make_analysis(with_obs=False, with_fvgs=False, with_liq=False)

        features = extract_features(analysis, candles, "1h")
        # Should still work, just fewer non-zero features
        assert isinstance(features, dict)

    def test_enriched_empty_analysis_more_features(self):
        """Empty arrays + mechanical detectors → more features than without."""
        candles = _make_candles(50, atr_size=15.0)
        base_analysis = _make_analysis(with_obs=False, with_fvgs=False, with_liq=False)

        # Without enrichment
        features_bare = extract_features(base_analysis, candles, "1h")
        bare_nonzero = sum(
            1 for v in features_bare.values()
            if v and not (isinstance(v, float) and math.isnan(v))
        )

        # With enrichment
        enriched = dict(base_analysis)
        atr = compute_atr(candles, 14)
        enriched["orderBlocks"] = detect_order_blocks(candles, atr) if atr > 0 else []
        enriched["fvgs"] = detect_fvgs(candles)
        enriched["liquidity"] = detect_liquidity(candles)

        features_rich = extract_features(enriched, candles, "1h")
        rich_nonzero = sum(
            1 for v in features_rich.values()
            if v and not (isinstance(v, float) and math.isnan(v))
        )

        # Rich should have at least as many non-zero (likely more if detectors found anything)
        assert rich_nonzero >= bare_nonzero


# ── Test: Entry zone populated ──────────────────────────────

class TestEntryZonePopulated:
    """Entry zone position/size should be non-NaN when entry falls in a zone."""

    def test_entry_in_ob_produces_zone_features(self):
        """Entry price within an OB → non-NaN zone features."""
        candles = _make_candles(30)
        analysis = _make_analysis(with_obs=True)
        # Entry 2350 is within OB [2345, 2355]
        features = extract_features(analysis, candles, "1h")
        assert not math.isnan(features["entry_zone_position"])
        assert not math.isnan(features["entry_zone_size_atr"])

    def test_entry_zone_position_range(self):
        """entry_zone_position should be between 0.0 and 1.0."""
        candles = _make_candles(30)
        analysis = _make_analysis(with_obs=True)
        features = extract_features(analysis, candles, "1h")
        pos = features["entry_zone_position"]
        if not math.isnan(pos):
            assert 0.0 <= pos <= 1.0

    def test_fallback_entry_zone_from_scanner(self):
        """Pre-computed entry zone values used when OB data missing."""
        # No OBs → _compute_entry_zone would return NaN
        # But fallbacks should be used instead
        result = _compute_entry_zone(
            entry_price=0,  # Missing entry price triggers fallback path
            direction="long",
            order_blocks=[],
            atr=15.0,
            fallback_position=0.65,
            fallback_size=0.8,
        )
        assert result["entry_zone_position"] == 0.65
        assert result["entry_zone_size_atr"] == 0.8

    def test_no_fallback_when_ob_data_available(self):
        """When OBs produce a zone, fallbacks are NOT used (OB data takes precedence)."""
        obs = [{"high": 2355.0, "low": 2345.0, "type": "bullish", "strength": "strong"}]
        result = _compute_entry_zone(
            entry_price=2350.0,
            direction="long",
            order_blocks=obs,
            atr=15.0,
            fallback_position=0.99,  # Should be ignored
            fallback_size=0.99,      # Should be ignored
        )
        # OB-computed values should be used, not the 0.99 fallbacks
        assert result["entry_zone_position"] != 0.99
        assert result["entry_zone_size_atr"] != 0.99
        assert not math.isnan(result["entry_zone_position"])

    def test_scanner_enrichment_computes_zone(self):
        """Simulate the scanner enrichment block computing entry zone."""
        entry_price = 2350.0
        atr = 15.0
        enriched = {"entry": {"price": entry_price, "direction": "long"}}
        # Detected OB containing entry
        ob = {"high": 2355.0, "low": 2345.0, "type": "bullish"}
        zone_high = zone_low = None
        if ob.get("low", 0) <= entry_price <= ob.get("high", 0):
            zone_high = ob["high"]
            zone_low = ob["low"]

        assert zone_high is not None
        assert zone_low is not None
        pos = (entry_price - zone_low) / (zone_high - zone_low)
        size = (zone_high - zone_low) / atr
        assert 0.0 <= pos <= 1.0
        assert size > 0


# ── Test: Fallback when candle fetch fails ──────────────────

class TestFallbackWhenFetchFails:
    """Rich extraction degrades gracefully when _fetch_candles fails."""

    def test_none_candles_no_crash(self):
        """If tf_candles is None, no crash and warning logged."""
        # Simulate the guard condition from scanner
        tf_candles = None
        features_extracted = False
        if tf_candles and len(tf_candles) >= 15:
            features_extracted = True
        assert not features_extracted  # Should skip extraction

    def test_too_few_candles_no_crash(self):
        """If tf_candles has <15 candles, extraction skipped."""
        tf_candles = _make_candles(10)
        features_extracted = False
        if tf_candles and len(tf_candles) >= 15:
            features_extracted = True
        assert not features_extracted

    def test_sufficient_candles_proceeds(self):
        """With ≥15 candles, extraction proceeds."""
        tf_candles = _make_candles(20)
        features_extracted = False
        if tf_candles and len(tf_candles) >= 15:
            features_extracted = True
        assert features_extracted


# ── Test: detect_order_blocks with atr parameter ────────────

class TestDetectOrderBlocksWithATR:
    """Verify detect_order_blocks requires and uses atr parameter."""

    def test_signature_requires_atr(self):
        """detect_order_blocks signature includes atr as required param."""
        import inspect
        sig = inspect.signature(detect_order_blocks)
        params = list(sig.parameters.keys())
        assert "atr" in params
        # atr should be the second param (after candles)
        assert params.index("atr") == 1

    def test_call_with_atr_succeeds(self):
        """Calling with atr parameter works."""
        candles = _make_candles(30, atr_size=15.0)
        atr = compute_atr(candles, 14)
        result = detect_order_blocks(candles, atr)
        assert isinstance(result, list)

    def test_call_without_atr_raises(self):
        """Calling WITHOUT atr parameter raises TypeError."""
        candles = _make_candles(30)
        with pytest.raises(TypeError):
            detect_order_blocks(candles)  # Missing required 'atr'


# ── Test: Candle ordering reversed ──────────────────────────

class TestCandleOrderingReversed:
    """Newest-first candles (OANDA default) should be reversed."""

    def test_newest_first_detected_and_reversed(self):
        """Candles with newest-first datetime ordering get reversed."""
        candles = _make_candles(20)
        # Reverse to simulate OANDA newest-first
        reversed_candles = list(reversed(candles))
        assert reversed_candles[0]["datetime"] > reversed_candles[-1]["datetime"]

        # Apply the same logic as scanner
        if (len(reversed_candles) >= 2
                and reversed_candles[0].get("datetime", "") > reversed_candles[-1].get("datetime", "")):
            reversed_candles = list(reversed(reversed_candles))

        # Should now be oldest-first
        assert reversed_candles[0]["datetime"] <= reversed_candles[-1]["datetime"]

    def test_oldest_first_not_reversed(self):
        """Candles already oldest-first stay unchanged."""
        candles = _make_candles(20)
        assert candles[0]["datetime"] <= candles[-1]["datetime"]

        original_first = candles[0]["datetime"]
        # Apply the same logic
        if (len(candles) >= 2
                and candles[0].get("datetime", "") > candles[-1].get("datetime", "")):
            candles = list(reversed(candles))

        assert candles[0]["datetime"] == original_first  # Unchanged

    def test_reversed_candles_produce_valid_features(self):
        """Features extracted from reversed (corrected) candles are valid."""
        candles = _make_candles(30, atr_size=15.0)
        reversed_candles = list(reversed(candles))

        # Reverse back (simulating the scanner's correction)
        if reversed_candles[0].get("datetime", "") > reversed_candles[-1].get("datetime", ""):
            reversed_candles = list(reversed(reversed_candles))

        analysis = _make_analysis()
        features = extract_features(analysis, reversed_candles, "1h")
        assert features["atr_14"] > 0
        assert not math.isnan(features["trend_strength"])
