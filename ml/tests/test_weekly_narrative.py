"""Tests for weekly HTF context layer."""
import json
import pytest
from ml.prompts import build_enhanced_ict_prompt


def _make_candles(n):
    return [
        {"datetime": f"2026-04-{i+1:02d} 00:00:00",
         "open": 3300.0 + i, "high": 3305.0 + i,
         "low": 3295.0 + i, "close": 3302.0 + i}
        for i in range(n)
    ]


_WEEKLY_NARRATIVE = {
    "macro_thesis": "Gold is in weekly distribution after sweeping the 2025 highs.",
    "directional_bias": "bearish",
    "bias_confidence": 0.75,
    "p3_phase": "distribution",
    "dealing_range": {"high": 3500.0, "low": 3100.0, "equilibrium": 3300.0},
    "premium_array": [{"price": 3450.0, "label": "Weekly OB", "role": "resistance"}],
    "discount_array": [{"price": 3150.0, "label": "Weekly Discount OB", "role": "support"}],
    "key_levels": [{"price": 3380.0, "label": "Previous Weekly High", "role": "resistance"}],
    "bias_invalidation": {"condition": "Weekly close above 3500", "price_level": 3500.0, "direction": "above"},
}

_MATCHED_LEVEL = {"price": 3380.0, "label": "Previous Weekly High", "role": "resistance"}


class TestWeeklyPromptBlocks:

    def test_no_weekly_narrative_no_block(self):
        prompt = build_enhanced_ict_prompt(_make_candles(10), _make_candles(5))
        assert "WEEKLY MACRO CONTEXT" not in prompt
        assert "WEEKLY KEY LEVEL PROXIMITY" not in prompt

    def test_full_weekly_block_when_no_matched_level(self):
        prompt = build_enhanced_ict_prompt(
            _make_candles(10), _make_candles(5),
            weekly_narrative=_WEEKLY_NARRATIVE,
            weekly_matched_level=None,
        )
        assert "WEEKLY MACRO CONTEXT" in prompt
        assert "WEEKLY KEY LEVEL PROXIMITY" not in prompt
        assert "Gold is in weekly distribution" in prompt
        assert "bearish" in prompt
        assert "3500" in prompt
        assert "3100" in prompt

    def test_proximity_block_when_matched_level_provided(self):
        prompt = build_enhanced_ict_prompt(
            _make_candles(10), _make_candles(5),
            weekly_narrative=_WEEKLY_NARRATIVE,
            weekly_matched_level=_MATCHED_LEVEL,
        )
        assert "WEEKLY KEY LEVEL PROXIMITY" in prompt
        assert "WEEKLY MACRO CONTEXT" not in prompt
        assert "Previous Weekly High" in prompt
        assert "3380" in prompt
        assert "bearish" in prompt

    def test_weekly_block_appears_before_execution_candles(self):
        prompt = build_enhanced_ict_prompt(
            _make_candles(10), _make_candles(5),
            weekly_narrative=_WEEKLY_NARRATIVE,
        )
        weekly_pos = prompt.index("WEEKLY MACRO CONTEXT")
        exec_pos = prompt.index("EXECUTION CANDLES")
        assert weekly_pos < exec_pos

    def test_weekly_block_includes_invalidation(self):
        prompt = build_enhanced_ict_prompt(
            _make_candles(10), _make_candles(5),
            weekly_narrative=_WEEKLY_NARRATIVE,
        )
        assert "Weekly close above 3500" in prompt


class TestBuildOpusWeeklyNarrativePrompt:

    def test_function_exists_and_returns_string(self):
        from ml.prompts import build_opus_weekly_narrative_prompt
        candles = _make_candles(24)
        result = build_opus_weekly_narrative_prompt(candles, _make_candles(20))
        assert isinstance(result, str)

    def test_prompt_contains_weekly_candles(self):
        from ml.prompts import build_opus_weekly_narrative_prompt
        result = build_opus_weekly_narrative_prompt(_make_candles(24), _make_candles(20))
        assert "WEEKLY CANDLES" in result

    def test_prompt_contains_daily_candles(self):
        from ml.prompts import build_opus_weekly_narrative_prompt
        result = build_opus_weekly_narrative_prompt(_make_candles(24), _make_candles(20))
        assert "DAILY CANDLES" in result

    def test_prompt_contains_json_schema(self):
        from ml.prompts import build_opus_weekly_narrative_prompt
        result = build_opus_weekly_narrative_prompt(_make_candles(24), _make_candles(20))
        assert "macro_thesis" in result
        assert "dealing_range" in result
        assert "premium_array" in result
        assert "discount_array" in result
        assert "bias_invalidation" in result

    def test_prompt_works_without_daily_candles(self):
        from ml.prompts import build_opus_weekly_narrative_prompt
        result = build_opus_weekly_narrative_prompt(_make_candles(24), None)
        assert isinstance(result, str)
        assert "WEEKLY CANDLES" in result

    def test_system_constant_exists(self):
        from ml.prompts import OPUS_WEEKLY_SYSTEM
        assert isinstance(OPUS_WEEKLY_SYSTEM, str)
        assert len(OPUS_WEEKLY_SYSTEM) > 50


class TestWeeklyCacheStale:

    def _make_scanner(self):
        """Return a scanner instance with mocked dependencies."""
        from ml.scanner import ScannerEngine
        engine = ScannerEngine.__new__(ScannerEngine)
        engine._weekly_narrative_cache = None
        engine._weekly_narrative_fetched_at = None
        return engine

    def test_stale_when_cache_none(self):
        from ml.scanner import ScannerEngine
        engine = self._make_scanner()
        assert engine._is_weekly_cache_stale() is True

    def test_stale_when_fetched_at_none(self):
        engine = self._make_scanner()
        engine._weekly_narrative_cache = {"directional_bias": "bullish"}
        engine._weekly_narrative_fetched_at = None
        assert engine._is_weekly_cache_stale() is True

    def test_stale_when_older_than_7_days(self):
        from datetime import datetime, timedelta
        engine = self._make_scanner()
        engine._weekly_narrative_cache = {"directional_bias": "bullish"}
        engine._weekly_narrative_fetched_at = datetime.utcnow() - timedelta(days=8)
        assert engine._is_weekly_cache_stale() is True

    def test_fresh_when_fetched_this_week_and_recent(self):
        from datetime import datetime
        engine = self._make_scanner()
        engine._weekly_narrative_cache = {"directional_bias": "bullish"}
        engine._weekly_narrative_fetched_at = datetime.utcnow()
        assert engine._is_weekly_cache_stale() is False


class TestIsNearWeeklyLevel:

    def _make_scanner(self):
        from ml.scanner import ScannerEngine
        engine = ScannerEngine.__new__(ScannerEngine)
        return engine

    def test_near_level_returns_true_and_level(self):
        engine = self._make_scanner()
        levels = [{"price": 3380.0, "label": "PWH", "role": "resistance"}]
        near, matched = engine._is_near_weekly_level(3370.0, 5.0, levels)  # 10 pts away = 2 ATR
        assert near is True
        assert matched["price"] == 3380.0

    def test_far_from_level_returns_false(self):
        engine = self._make_scanner()
        levels = [{"price": 3380.0, "label": "PWH", "role": "resistance"}]
        near, matched = engine._is_near_weekly_level(3200.0, 5.0, levels)  # 180 pts = 36 ATR
        assert near is False
        assert matched is None

    def test_exactly_at_threshold_returns_true(self):
        engine = self._make_scanner()
        levels = [{"price": 3380.0, "label": "PWH", "role": "resistance"}]
        # 3 ATR = 15 pts at ATR=5; price at 3380-15=3365 is exactly at boundary
        near, matched = engine._is_near_weekly_level(3365.0, 5.0, levels)
        assert near is True

    def test_empty_levels_returns_false(self):
        engine = self._make_scanner()
        near, matched = engine._is_near_weekly_level(3380.0, 5.0, [])
        assert near is False
        assert matched is None

    def test_zero_atr_returns_false(self):
        engine = self._make_scanner()
        levels = [{"price": 3380.0, "label": "PWH", "role": "resistance"}]
        near, matched = engine._is_near_weekly_level(3380.0, 0.0, levels)
        assert near is False
        assert matched is None

    def test_level_missing_price_key_skipped(self):
        engine = self._make_scanner()
        levels = [{"label": "PWH", "role": "resistance"}]  # no price key
        near, matched = engine._is_near_weekly_level(3380.0, 5.0, levels)
        assert near is False
