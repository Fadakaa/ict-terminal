"""Tests for enhanced ICT prompt builder (prompts.py)."""
import json
import pytest

from ml.prompts import build_enhanced_ict_prompt, _slim_candles


def _make_candles(n):
    return [
        {"datetime": f"2026-03-10 {i:02d}:00:00",
         "open": 2900.0 + i, "high": 2901.0 + i,
         "low": 2899.0 + i, "close": 2900.5 + i}
        for i in range(n)
    ]


class TestBuildEnhancedPrompt:

    def test_returns_string(self):
        prompt = build_enhanced_ict_prompt(_make_candles(10), _make_candles(5))
        assert isinstance(prompt, str)

    def test_contains_4h_section(self):
        prompt = build_enhanced_ict_prompt(_make_candles(10), _make_candles(5))
        assert "4H CANDLES" in prompt

    def test_contains_execution_section(self):
        prompt = build_enhanced_ict_prompt(_make_candles(10), _make_candles(5))
        assert "EXECUTION CANDLES" in prompt

    def test_contains_analysis_framework(self):
        prompt = build_enhanced_ict_prompt(_make_candles(10), _make_candles(5))
        assert "ANALYSIS FRAMEWORK" in prompt
        assert "premium" in prompt.lower()
        assert "discount" in prompt.lower()

    def test_contains_json_schema(self):
        prompt = build_enhanced_ict_prompt(_make_candles(10), _make_candles(5))
        assert "orderBlocks" in prompt
        assert "fvgs" in prompt
        assert "htf_context" in prompt
        assert "setup_quality" in prompt

    def test_passes_full_candles_without_truncation(self):
        prompt = build_enhanced_ict_prompt(_make_candles(100), _make_candles(5))
        # Full candle set is passed through — no truncation
        assert "EXECUTION CANDLES (100 candles)" in prompt

    def test_passes_full_4h_candles(self):
        prompt = build_enhanced_ict_prompt(_make_candles(10), _make_candles(50))
        assert "4H CANDLES (50 candles" in prompt

    def test_contains_power_of_3(self):
        prompt = build_enhanced_ict_prompt(_make_candles(10), _make_candles(5))
        assert "Power of 3" in prompt

    def test_contains_rejection_warning(self):
        prompt = build_enhanced_ict_prompt(_make_candles(10), _make_candles(5))
        assert "Do NOT enter on displacement" in prompt


class TestSlimCandles:

    def test_slim_format(self):
        candles = [{"datetime": "2026-03-10 08:00:00",
                     "open": 2900.123, "high": 2901.456,
                     "low": 2899.789, "close": 2900.555}]
        slim = _slim_candles(candles)
        assert len(slim) == 1
        assert "dt" in slim[0]
        assert "o" in slim[0]
        assert "h" in slim[0]
        assert "l" in slim[0]
        assert "c" in slim[0]
        # Rounded to 2 decimal places
        assert slim[0]["o"] == 2900.12

    def test_empty_input(self):
        assert _slim_candles([]) == []


class TestRecentContextInjection:
    """Test that recent_context param is correctly injected into prompt."""

    def test_recent_context_included_when_provided(self):
        from datetime import datetime
        ctx = {
            "recent_resolutions": [{
                "direction": "long", "outcome": "stopped_out",
                "entry_price": 2341.50, "sl_price": 2338.00,
                "pnl_rr": -1.0,
                "resolved_at": datetime.utcnow().isoformat(),
            }],
            "consumed_zones": [{"zone_type": "ob", "high": 2343.0,
                                "low": 2340.0, "setup_id": "s1",
                                "outcome": "stopped_out"}],
            "swept_liquidity": [],
            "active_setups": [],
        }
        prompt = build_enhanced_ict_prompt(
            _make_candles(10), _make_candles(5), recent_context=ctx)
        assert "RECENT CONTEXT" in prompt
        assert "STOPPED OUT" in prompt
        assert "CONSUMED OB" in prompt

    def test_recent_context_excluded_when_none(self):
        prompt = build_enhanced_ict_prompt(
            _make_candles(10), _make_candles(5), recent_context=None)
        assert "RECENT CONTEXT" not in prompt

    def test_recent_context_excluded_when_empty(self):
        ctx = {
            "recent_resolutions": [],
            "consumed_zones": [],
            "swept_liquidity": [],
            "active_setups": [],
        }
        prompt = build_enhanced_ict_prompt(
            _make_candles(10), _make_candles(5), recent_context=ctx)
        assert "RECENT CONTEXT" not in prompt

    def test_recent_context_before_analysis_framework(self):
        from datetime import datetime
        ctx = {
            "recent_resolutions": [{
                "direction": "long", "outcome": "tp1",
                "entry_price": 2340, "sl_price": 2336,
                "resolved_at": datetime.utcnow().isoformat(),
            }],
            "consumed_zones": [],
            "swept_liquidity": [],
            "active_setups": [],
        }
        prompt = build_enhanced_ict_prompt(
            _make_candles(10), _make_candles(5), recent_context=ctx)
        ctx_pos = prompt.index("RECENT CONTEXT")
        fw_pos = prompt.index("ANALYSIS FRAMEWORK")
        assert ctx_pos < fw_pos


class TestRegimeSectionInjection:
    """Tests for the MARKET REGIME prompt section."""

    def test_regime_section_included_when_provided(self):
        regime_ctx = {
            "regime": "TRENDING_CORRECTIVE",
            "confidence": 0.75,
            "metrics": {
                "atr_percentile": 0.62,
                "vol_ratio_5_30": 1.1,
                "net_movement_atr": 0.8,
                "displacement_count": 2,
                "body_consistency": 0.7,
            },
        }
        prompt = build_enhanced_ict_prompt(
            _make_candles(10), _make_candles(5), regime_context=regime_ctx)
        assert "MARKET REGIME" in prompt
        assert "TRENDING_CORRECTIVE" in prompt
        assert "OB/FVG retest entries are ideal" in prompt

    def test_regime_section_excluded_when_none(self):
        prompt = build_enhanced_ict_prompt(
            _make_candles(10), _make_candles(5), regime_context=None)
        assert "MARKET REGIME" not in prompt

    def test_regime_section_before_analysis_framework(self):
        regime_ctx = {
            "regime": "RANGING",
            "confidence": 0.6,
            "metrics": {
                "atr_percentile": 0.5,
                "vol_ratio_5_30": 1.0,
                "net_movement_atr": 0.3,
                "displacement_count": 0,
                "body_consistency": 0.5,
            },
        }
        prompt = build_enhanced_ict_prompt(
            _make_candles(10), _make_candles(5), regime_context=regime_ctx)
        regime_pos = prompt.index("MARKET REGIME")
        fw_pos = prompt.index("ANALYSIS FRAMEWORK")
        assert regime_pos < fw_pos

    def test_all_regimes_have_implications(self):
        from ml.prompts import _build_regime_section
        for regime in ("TRENDING_IMPULSIVE", "TRENDING_CORRECTIVE",
                       "RANGING", "VOLATILE_CHOPPY", "QUIET_DRIFT"):
            ctx = {"regime": regime, "confidence": 0.5,
                   "metrics": {"atr_percentile": 0.5, "vol_ratio_5_30": 1.0,
                               "net_movement_atr": 0, "displacement_count": 0,
                               "body_consistency": 0.5}}
            section = _build_regime_section(ctx)
            assert regime in section
            assert "Implication:" in section


class TestKeyLevelsSectionInjection:
    """Tests for the === KEY LEVELS === prompt section."""

    def _sample_levels(self):
        return {
            "pdh": 3045.20, "pdl": 3028.50, "pd_eq": 3036.85,
            "pwh": 3058.00, "pwl": 3012.30, "pw_eq": 3035.15,
            "pmh": 3078.50, "pml": 2985.20, "pm_eq": 3031.85,
            "asia_high": 3042.10, "asia_low": 3035.80, "asia_eq": 3038.95,
            "prev_session_high": 3045.20, "prev_session_low": 3030.50,
            "prev_session_eq": 3037.85, "prev_session_name": "London",
            "levels_computed": 15,
        }

    def test_key_levels_included_when_provided(self):
        prompt = build_enhanced_ict_prompt(
            _make_candles(10), _make_candles(5), key_levels=self._sample_levels())
        assert "KEY LEVELS" in prompt
        assert "Previous Day" in prompt
        assert "Previous Week" in prompt
        assert "Previous Month" in prompt
        assert "Asia Session" in prompt
        assert "Prev Session (London)" in prompt

    def test_key_levels_excluded_when_none(self):
        prompt = build_enhanced_ict_prompt(
            _make_candles(10), _make_candles(5), key_levels=None)
        assert "=== KEY LEVELS (computed from price data) ===" not in prompt

    def test_key_levels_excluded_when_empty(self):
        prompt = build_enhanced_ict_prompt(
            _make_candles(10), _make_candles(5),
            key_levels={"levels_computed": 0, "pdh": None, "pdl": None})
        assert "=== KEY LEVELS (computed from price data) ===" not in prompt

    def test_key_levels_before_analysis_framework(self):
        prompt = build_enhanced_ict_prompt(
            _make_candles(10), _make_candles(5), key_levels=self._sample_levels())
        kl_pos = prompt.index("KEY LEVELS")
        fw_pos = prompt.index("ANALYSIS FRAMEWORK")
        assert kl_pos < fw_pos

    def test_key_levels_contains_prices(self):
        levels = self._sample_levels()
        prompt = build_enhanced_ict_prompt(
            _make_candles(10), _make_candles(5), key_levels=levels)
        assert "3045.20" in prompt  # PDH
        assert "3028.50" in prompt  # PDL
        assert "3058.00" in prompt  # PWH

    def test_partial_levels_only_shows_computed(self):
        levels = {
            "pdh": 3045.20, "pdl": 3028.50, "pd_eq": 3036.85,
            "pwh": None, "pwl": None, "pw_eq": None,
            "pmh": None, "pml": None, "pm_eq": None,
            "asia_high": None, "asia_low": None, "asia_eq": None,
            "prev_session_high": None, "prev_session_low": None,
            "prev_session_eq": None, "prev_session_name": None,
            "levels_computed": 3,
        }
        prompt = build_enhanced_ict_prompt(
            _make_candles(10), _make_candles(5), key_levels=levels)
        assert "Previous Day" in prompt
        assert "Previous Week" not in prompt
        assert "Previous Month" not in prompt
        assert "Asia Session" not in prompt
