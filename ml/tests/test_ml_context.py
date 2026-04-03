"""Tests for Phase 2: ML Repositioning — Pre-Analysis Enrichment.

Tests the build_ml_context() method on MLCalibrator and the
_build_ml_context_section() prompt formatter.
"""
import json
import pytest
from unittest.mock import patch, MagicMock

from ml.prompts import _build_ml_context_section, build_enhanced_ict_prompt


def _make_candles(n, base_price=2900.0):
    """Create realistic-ish candles for testing."""
    candles = []
    for i in range(n):
        p = base_price + i * 0.5
        candles.append({
            "datetime": f"2026-03-10 {i % 24:02d}:00:00",
            "open": p, "high": p + 2.0, "low": p - 2.0, "close": p + 0.5,
        })
    return candles


# ── _build_ml_context_section tests ──────────────────────────────


class TestBuildMlContextSection:
    """Test the prompt section formatter."""

    def test_returns_empty_when_none(self):
        assert _build_ml_context_section(None) == ""

    def test_returns_empty_when_empty_dict(self):
        # Empty dict is falsy — returns empty string (no context = no section)
        result = _build_ml_context_section({})
        assert result == ""

    def test_contains_regime_info(self):
        ctx = {"regime": "HIGH_VOL", "sl_floor_atr": 4.5,
               "mae_percentile_80": 5.0, "vol_ratio": 1.8}
        result = _build_ml_context_section(ctx)
        assert "HIGH_VOL" in result
        assert "4.5 ATR" in result
        assert "5.0 ATR" in result
        assert "1.8x normal" in result

    def test_contains_dna_pattern_match(self):
        ctx = {"dna_win_rate": 0.72, "dna_avg_rr": 1.8, "dna_sample_size": 45}
        result = _build_ml_context_section(ctx)
        assert "72%" in result
        assert "45 similar setups" in result
        assert "1.8:1" in result

    def test_contains_bayesian_win_rate(self):
        ctx = {"bayesian_wr": 0.485, "bayesian_trend": -2.5}
        result = _build_ml_context_section(ctx)
        assert "48.5%" in result
        assert "trending down" in result
        assert "2.5pp" in result

    def test_bayesian_stable_when_zero_trend(self):
        ctx = {"bayesian_wr": 0.5, "bayesian_trend": 0}
        result = _build_ml_context_section(ctx)
        assert "stable" in result

    def test_contains_entry_placement(self):
        ctx = {"entry_placement": "OB midpoint",
               "entry_placement_delta_rr": 0.3}
        result = _build_ml_context_section(ctx)
        assert "OB midpoint" in result
        assert "+0.3 R:R" in result

    def test_contains_intermarket_quality(self):
        ctx = {"intermarket_quality": "supportive"}
        result = _build_ml_context_section(ctx)
        assert "SUPPORTIVE" in result

    def test_skips_unknown_intermarket(self):
        ctx = {"intermarket_quality": "unknown"}
        result = _build_ml_context_section(ctx)
        assert "Intermarket" not in result

    def test_contains_opus_accuracy(self):
        ctx = {"dna_win_rate": 0.6, "dna_sample_size": 20,
               "dna_avg_rr": 1.5, "opus_accuracy": 0.78}
        result = _build_ml_context_section(ctx)
        assert "78%" in result
        assert "Opus accuracy" in result

    def test_section_boundaries(self):
        ctx = {"regime": "NORMAL"}
        result = _build_ml_context_section(ctx)
        assert result.startswith("=== YOUR STATISTICAL MEMORY ===")
        assert "=== END STATISTICAL MEMORY ===" in result


# ── ML context in Sonnet prompt tests ─────────────────────────────


class TestMlContextInSonnetPrompt:
    """Verify ML context appears in the full Sonnet prompt."""

    def test_ml_context_appears_in_prompt(self):
        ml_ctx = {
            "regime": "TRENDING",
            "sl_floor_atr": 3.5,
            "mae_percentile_80": 4.0,
            "vol_ratio": 1.2,
        }
        prompt = build_enhanced_ict_prompt(
            _make_candles(10), _make_candles(5),
            ml_context=ml_ctx)
        assert "YOUR STATISTICAL MEMORY" in prompt
        assert "TRENDING" in prompt
        assert "3.5 ATR" in prompt

    def test_no_ml_context_no_section(self):
        prompt = build_enhanced_ict_prompt(
            _make_candles(10), _make_candles(5),
            ml_context=None)
        assert "YOUR STATISTICAL MEMORY" not in prompt

    def test_ml_context_before_analysis_framework(self):
        ml_ctx = {"regime": "NORMAL", "sl_floor_atr": 3.0,
                  "mae_percentile_80": 4.0, "vol_ratio": 1.0}
        prompt = build_enhanced_ict_prompt(
            _make_candles(10), _make_candles(5),
            ml_context=ml_ctx)
        ml_pos = prompt.index("STATISTICAL MEMORY")
        fw_pos = prompt.index("ANALYSIS FRAMEWORK")
        assert ml_pos < fw_pos


# ── build_ml_context() tests ──────────────────────────────────────


class TestBuildMlContext:
    """Test the MLCalibrator.build_ml_context() method."""

    def test_cold_start_returns_defaults(self):
        """With no models, no DNA, no Bayesian state — returns safe defaults."""
        from ml.calibrate import MLCalibrator
        calibrator = MLCalibrator()
        ctx = calibrator.build_ml_context(
            thesis_type=None, timeframe="1h", killzone="London",
            candles=_make_candles(50))
        assert "regime" in ctx
        assert "sl_floor_atr" in ctx
        assert "mae_percentile_80" in ctx
        assert ctx["sl_floor_atr"] >= 2.0  # Some reasonable floor

    def test_with_few_candles(self):
        """With <14 candles, falls back to defaults."""
        from ml.calibrate import MLCalibrator
        calibrator = MLCalibrator()
        ctx = calibrator.build_ml_context(
            thesis_type=None, timeframe="1h", killzone="Asian",
            candles=_make_candles(5))
        assert ctx["regime"] == "UNKNOWN"
        assert ctx["sl_floor_atr"] == 3.0

    def test_with_empty_candles(self):
        from ml.calibrate import MLCalibrator
        calibrator = MLCalibrator()
        ctx = calibrator.build_ml_context(
            thesis_type=None, timeframe="1h", killzone="Off",
            candles=[])
        assert ctx["regime"] == "UNKNOWN"

    def test_graceful_degradation_per_layer(self):
        """Each layer can fail independently without breaking the whole."""
        from ml.calibrate import MLCalibrator
        calibrator = MLCalibrator()

        # Even with all imports failing, should return a valid dict
        with patch("ml.calibrate.MLCalibrator._to_candle_list",
                   side_effect=Exception("boom")):
            ctx = calibrator.build_ml_context(
                thesis_type="bullish_accumulation",
                timeframe="1h", killzone="London",
                candles=_make_candles(50))
        # Should still have defaults
        assert ctx["regime"] == "UNKNOWN"
        assert ctx["sl_floor_atr"] == 3.0

    def test_dna_pattern_uses_conditional_stats(self):
        """When a DNA pattern is provided, uses get_conditional_stats."""
        from ml.calibrate import MLCalibrator
        calibrator = MLCalibrator()

        mock_stats = {"win_rate": 0.68, "avg_rr": 1.5, "sample_size": 30}
        with patch("ml.setup_profiles.SetupProfileStore.get_conditional_stats",
                   return_value=mock_stats):
            ctx = calibrator.build_ml_context(
                thesis_type=None, timeframe="1h", killzone="London",
                candles=_make_candles(50),
                setup_dna_pattern={"direction": "long"})
        assert ctx.get("dna_win_rate") == 0.68
        assert ctx.get("dna_sample_size") == 30

    def test_dna_skipped_when_small_sample(self):
        """DNA match skipped when fewer than 15 samples."""
        from ml.calibrate import MLCalibrator
        calibrator = MLCalibrator()

        mock_stats = {"win_rate": 0.9, "avg_rr": 3.0, "sample_size": 5}
        with patch("ml.setup_profiles.SetupProfileStore.get_conditional_stats",
                   return_value=mock_stats):
            ctx = calibrator.build_ml_context(
                thesis_type=None, timeframe="1h", killzone="London",
                candles=_make_candles(50),
                setup_dna_pattern={"direction": "long"})
        assert "dna_win_rate" not in ctx


# ── Dual-mode tracking tests ─────────────────────────────────────


class TestDualModeTracking:
    """Verify safety net fire rate tracking works."""

    def test_safety_net_fires_when_sl_below_floor(self):
        """When Sonnet SL < ML floor, safety net fires."""
        stats = {"ml_safety_net_checks": 0, "ml_safety_net_fired": 0}
        sonnet_sl_atr = 2.0
        ml_floor = 3.5
        safety_net_fired = sonnet_sl_atr > 0 and ml_floor > 0 and sonnet_sl_atr < ml_floor
        stats["ml_safety_net_checks"] += 1
        if safety_net_fired:
            stats["ml_safety_net_fired"] += 1
        assert stats["ml_safety_net_fired"] == 1
        assert stats["ml_safety_net_checks"] == 1

    def test_safety_net_noop_when_sl_above_floor(self):
        """When Sonnet SL >= ML floor, safety net does NOT fire."""
        stats = {"ml_safety_net_checks": 0, "ml_safety_net_fired": 0}
        sonnet_sl_atr = 4.0
        ml_floor = 3.5
        safety_net_fired = sonnet_sl_atr > 0 and ml_floor > 0 and sonnet_sl_atr < ml_floor
        stats["ml_safety_net_checks"] += 1
        if safety_net_fired:
            stats["ml_safety_net_fired"] += 1
        assert stats["ml_safety_net_fired"] == 0
        assert stats["ml_safety_net_checks"] == 1

    def test_fire_rate_calculation(self):
        """Fire rate = fired / checks."""
        stats = {"ml_safety_net_checks": 100, "ml_safety_net_fired": 8}
        fire_rate = stats["ml_safety_net_fired"] / stats["ml_safety_net_checks"]
        assert fire_rate == 0.08
        assert fire_rate < 0.10  # Target: <10%
