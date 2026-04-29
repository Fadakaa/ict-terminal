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


class TestGetWeeklyNarrative:

    def _make_scanner_with_cache(self, cache=None, fetched_at=None):
        from unittest.mock import patch
        from datetime import datetime
        from ml.scanner import ScannerEngine
        engine = ScannerEngine.__new__(ScannerEngine)
        engine._weekly_narrative_cache = cache
        engine._weekly_narrative_fetched_at = fetched_at or (datetime.utcnow() if cache else None)
        return engine

    def test_returns_none_when_call_fails_and_no_cache(self):
        from unittest.mock import patch
        engine = self._make_scanner_with_cache()
        with patch.object(engine, "_call_opus_weekly_narrative", return_value=None):
            result = engine._get_weekly_narrative()
        assert result is None

    def test_returns_cached_when_fresh(self):
        from datetime import datetime
        cached = {"directional_bias": "bullish", "macro_thesis": "Gold is going up."}
        engine = self._make_scanner_with_cache(cache=cached, fetched_at=datetime.utcnow())
        result = engine._get_weekly_narrative()
        assert result is cached

    def test_calls_opus_when_stale(self):
        from unittest.mock import patch
        from datetime import datetime
        fresh_narrative = {"directional_bias": "bearish", "macro_thesis": "Selling."}
        engine = self._make_scanner_with_cache()  # cache=None → stale

        def _set_cache():
            engine._weekly_narrative_cache = fresh_narrative
            engine._weekly_narrative_fetched_at = datetime.utcnow()

        with patch.object(engine, "_call_opus_weekly_narrative", side_effect=_set_cache) as mock_call:
            result = engine._get_weekly_narrative()
        mock_call.assert_called_once()
        assert result == fresh_narrative


class TestWeeklyInjectionInScanTimeframe:
    """Verify weekly narrative is passed through _call_claude correctly."""

    _WEEKLY_NR = {
        "macro_thesis": "Gold bearish weekly.",
        "directional_bias": "bearish",
        "bias_confidence": 0.8,
        "p3_phase": "distribution",
        "dealing_range": {"high": 3500.0, "low": 3100.0, "equilibrium": 3300.0},
        "premium_array": [{"price": 3450.0, "label": "Weekly OB", "role": "resistance"}],
        "discount_array": [],
        "key_levels": [{"price": 3380.0, "label": "PWH", "role": "resistance"}],
        "bias_invalidation": {"condition": "Close above 3500", "price_level": 3500.0, "direction": "above"},
    }

    def test_call_claude_accepts_weekly_narrative_param(self):
        """_call_claude signature must accept weekly_narrative and weekly_matched_level."""
        import inspect
        from ml.scanner import ScannerEngine
        sig = inspect.signature(ScannerEngine._call_claude)
        assert "weekly_narrative" in sig.parameters
        assert "weekly_matched_level" in sig.parameters

    def test_build_enhanced_ict_prompt_called_with_weekly_narrative(self):
        from unittest.mock import patch, MagicMock
        from ml.scanner import ScannerEngine

        engine = MagicMock(spec=ScannerEngine)
        engine.claude_key = "test-key"
        engine._pending_api_cost = 0.0

        candles = [{"datetime": f"2026-04-{i+1:02d}", "open": 3300.0, "high": 3305.0,
                    "low": 3295.0, "close": 3302.0} for i in range(10)]

        with patch("ml.scanner.build_enhanced_ict_prompt", return_value="test prompt") as mock_prompt, \
             patch("ml.scanner.httpx.post") as mock_post:
            mock_post.return_value = MagicMock(
                status_code=200,
                json=lambda: {"content": [{"type": "text", "text": '{"bias": "bearish"}'}],
                              "usage": {"input_tokens": 100, "output_tokens": 50}}
            )
            ScannerEngine._call_claude(
                engine, candles, [], "1day",
                weekly_narrative=self._WEEKLY_NR,
                weekly_matched_level=None,
            )

        call_kwargs = mock_prompt.call_args[1]
        assert call_kwargs.get("weekly_narrative") == self._WEEKLY_NR
        assert call_kwargs.get("weekly_matched_level") is None


class TestWeeklySchedulerJob:

    def test_weekly_job_registered_in_scheduler(self):
        """Verify the weekly cache clear job is registered on Sunday 21:00 UTC."""
        from unittest.mock import patch, MagicMock
        import importlib
        import ml.scheduler

        added_jobs = []

        def capture_add_job(fn, trigger, **kwargs):
            added_jobs.append({"fn": fn, "trigger": trigger, "kwargs": kwargs})

        mock_scheduler = MagicMock()
        mock_scheduler.add_job.side_effect = capture_add_job

        # Reload first to get a clean module state, then apply patches and call
        importlib.reload(ml.scheduler)

        with patch("ml.scheduler.AsyncIOScheduler", return_value=mock_scheduler), \
             patch("ml.scheduler.os.environ.get", return_value="fake-key"), \
             patch("ml.scheduler.get_config", return_value={
                 "oanda_account_id": "x", "oanda_access_token": "y"}):
            from ml.scheduler import start_scheduler as start_scheduler_fresh
            start_scheduler_fresh()

        job_ids = [j["kwargs"].get("id") for j in added_jobs]
        assert "weekly_cache_clear" in job_ids

        weekly_job = next(j for j in added_jobs if j["kwargs"].get("id") == "weekly_cache_clear")
        assert weekly_job["kwargs"].get("day_of_week") == "sun"
        assert weekly_job["kwargs"].get("hour") == 21
        assert weekly_job["kwargs"].get("minute") == 0


class TestWeeklyNarrativeEndpoints:

    def test_get_weekly_narrative_404_when_no_cache(self):
        from unittest.mock import patch, MagicMock
        from fastapi.testclient import TestClient
        from ml.server import app

        mock_engine = MagicMock()
        mock_engine._weekly_narrative_cache = None

        with patch("ml.server._get_scanner", return_value=mock_engine):
            client = TestClient(app)
            resp = client.get("/weekly/narrative")

        assert resp.status_code == 404
        assert "POST /weekly/refresh" in resp.json()["detail"]

    def test_get_weekly_narrative_200_with_cache(self):
        from unittest.mock import patch, MagicMock
        from datetime import datetime
        from fastapi.testclient import TestClient
        from ml.server import app

        cache = {
            "macro_thesis": "Gold bearish.", "directional_bias": "bearish",
            "bias_confidence": 0.75, "p3_phase": "distribution",
            "dealing_range": {"high": 3500.0, "low": 3100.0, "equilibrium": 3300.0},
            "premium_array": [], "discount_array": [], "key_levels": [],
            "bias_invalidation": {"condition": "Close above 3500", "price_level": 3500.0, "direction": "above"},
        }
        mock_engine = MagicMock()
        mock_engine._weekly_narrative_cache = cache
        mock_engine._weekly_narrative_fetched_at = datetime.utcnow()

        with patch("ml.server._get_scanner", return_value=mock_engine):
            client = TestClient(app)
            resp = client.get("/weekly/narrative")

        assert resp.status_code == 200
        body = resp.json()
        assert body["directional_bias"] == "bearish"
        assert "last_updated" in body
        assert "next_refresh" in body
        assert "cache_age_hours" in body

    def test_post_weekly_refresh_triggers_call(self):
        from unittest.mock import patch, MagicMock
        from datetime import datetime
        from fastapi.testclient import TestClient
        from ml.server import app

        fresh = {
            "macro_thesis": "Gold now bullish.", "directional_bias": "bullish",
            "bias_confidence": 0.6, "p3_phase": "accumulation",
            "dealing_range": {"high": 3500.0, "low": 3100.0, "equilibrium": 3300.0},
            "premium_array": [], "discount_array": [], "key_levels": [],
            "bias_invalidation": {"condition": "Close below 3100", "price_level": 3100.0, "direction": "below"},
        }
        mock_engine = MagicMock()
        mock_engine._weekly_narrative_cache = None
        mock_engine._weekly_narrative_fetched_at = None

        def side_effect():
            mock_engine._weekly_narrative_cache = fresh
            mock_engine._weekly_narrative_fetched_at = datetime.utcnow()
            return fresh

        mock_engine._call_opus_weekly_narrative.side_effect = side_effect

        with patch("ml.server._get_scanner", return_value=mock_engine):
            client = TestClient(app)
            resp = client.post("/weekly/refresh")

        assert resp.status_code == 200
        assert resp.json()["directional_bias"] == "bullish"
        mock_engine._call_opus_weekly_narrative.assert_called_once()

    def test_post_weekly_refresh_500_when_opus_fails(self):
        from unittest.mock import patch, MagicMock
        from fastapi.testclient import TestClient
        from ml.server import app

        mock_engine = MagicMock()
        mock_engine._weekly_narrative_cache = None
        mock_engine._weekly_narrative_fetched_at = None
        mock_engine._call_opus_weekly_narrative.return_value = None

        with patch("ml.server._get_scanner", return_value=mock_engine):
            client = TestClient(app)
            resp = client.post("/weekly/refresh")

        assert resp.status_code == 500
