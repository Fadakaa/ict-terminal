# Weekly HTF Context Layer — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a weekly timeframe as a pure HTF context layer — Opus generates a macro weekly narrative (directional bias, key levels, dealing ranges, P3 phase) cached 7 days, injected into all scanning timeframes conditionally.

**Architecture:** Extend the existing Opus narrative pattern: a new `_call_opus_weekly_narrative()` method on the scanner instance, a dedicated 7-day cache (`_weekly_narrative_cache`), and a proximity helper that decides whether to inject full context (1day, always) or condensed proximity context (4H/1H/15min, when price is within 3×ATR of any weekly level). A Sunday 21:00 UTC scheduler job clears the cache on weekly close. Two new server endpoints surface the narrative.

**Tech Stack:** Python, FastAPI, APScheduler, httpx, Anthropic API (claude-opus-4-6), existing `ml/scanner.py` / `ml/prompts.py` / `ml/scheduler.py` / `ml/server.py` patterns.

---

## File Map

| File | Change |
|---|---|
| `ml/prompts.py` | Add `build_opus_weekly_narrative_prompt()`, `OPUS_WEEKLY_SYSTEM`, `weekly_narrative`/`weekly_matched_level` params to `build_enhanced_ict_prompt()` |
| `ml/scanner.py` | Add `_weekly_narrative_cache`, `_weekly_narrative_fetched_at`, `_is_weekly_cache_stale()`, `_is_near_weekly_level()`, `_call_opus_weekly_narrative()`, `_get_weekly_narrative()` |
| `ml/scanner.py` (cont.) | Add weekly context retrieval + injection into `_scan_timeframe()`, add `weekly_narrative`/`weekly_matched_level` params to `_call_claude()` |
| `ml/scheduler.py` | Add `_weekly_cache_clear_job()`, register Sunday 21:00 UTC cron |
| `ml/server.py` | Add `GET /weekly/narrative`, `POST /weekly/refresh` |
| `ml/tests/test_weekly_narrative.py` | New test file — cache staleness, proximity check, prompt blocks, endpoints |

---

## Task 1: Weekly prompt blocks in `prompts.py`

**Files:**
- Modify: `ml/prompts.py` (function `build_enhanced_ict_prompt` at line 30, main return at line 123)
- Test: `ml/tests/test_weekly_narrative.py` (create)

- [ ] **Step 1: Write failing tests**

Create `ml/tests/test_weekly_narrative.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd ~/dealfinder/ict-terminal && source ~/dealfinder/bin/activate
python -m pytest ml/tests/test_weekly_narrative.py::TestWeeklyPromptBlocks -v
```
Expected: `FAILED — TypeError: build_enhanced_ict_prompt() got unexpected keyword argument 'weekly_narrative'`

- [ ] **Step 3: Add `weekly_narrative` and `weekly_matched_level` params to `build_enhanced_ict_prompt()`**

In `ml/prompts.py`, change the function signature at line 30 from:
```python
def build_enhanced_ict_prompt(candles_1h: list, candles_4h: list,
                              ...
                              key_levels: dict | None = None) -> str:
```
to:
```python
def build_enhanced_ict_prompt(candles_1h: list, candles_4h: list,
                              intermarket: dict | None = None,
                              htf_narrative: dict | None = None,
                              setup_context: dict | None = None,
                              narrative_weights: dict | None = None,
                              prev_narrative: dict | None = None,
                              invalidation_status: str | None = None,
                              recent_context: dict | None = None,
                              regime_context: dict | None = None,
                              ml_context: dict | None = None,
                              htf_label: str | None = None,
                              key_levels: dict | None = None,
                              weekly_narrative: dict | None = None,
                              weekly_matched_level: dict | None = None) -> str:
```

- [ ] **Step 4: Build the `weekly_block` string and insert it into the prompt**

In `ml/prompts.py`, after the `narrative_block = ""` block (around line 78) and before the `return f"""...` at line 123, add:

```python
    # Build weekly macro context block
    weekly_block = ""
    if weekly_narrative:
        wn = weekly_narrative
        dr = wn.get("dealing_range", {})
        if weekly_matched_level:
            # Condensed proximity mode — 4H/1H/15min when near a weekly level
            lvl = weekly_matched_level
            weekly_block = (
                f"WEEKLY KEY LEVEL PROXIMITY — price is approaching a significant weekly structural level:\n"
                f"Level: {lvl.get('label', '?')} at {lvl.get('price', '?')} (role: {lvl.get('role', '?')})\n"
                f"Weekly Macro Bias: {wn.get('directional_bias', '?')} "
                f"(confidence: {wn.get('bias_confidence', 0):.0%})\n"
                f"Weight your entry criteria accordingly. This weekly level may act as institutional support/resistance.\n\n"
            )
        else:
            # Full mode — 1day always gets the complete weekly context
            weekly_block = (
                f"WEEKLY MACRO CONTEXT (authoritative macro framework — your analysis must be consistent with "
                f"this or explicitly explain any divergence):\n"
                f"Macro Thesis: {wn.get('macro_thesis', '?')}\n"
                f"Directional Bias: {wn.get('directional_bias', '?')} "
                f"(confidence: {wn.get('bias_confidence', 0):.0%})\n"
                f"Power of 3 Phase: {wn.get('p3_phase', '?')}\n"
                f"Dealing Range: {dr.get('high', '?')} – {dr.get('low', '?')} "
                f"(equilibrium: {dr.get('equilibrium', '?')})\n"
                f"Premium Array: {json.dumps(wn.get('premium_array', []))}\n"
                f"Discount Array: {json.dumps(wn.get('discount_array', []))}\n"
                f"Key Levels: {json.dumps(wn.get('key_levels', []))}\n"
                f"Invalidation: {wn.get('bias_invalidation', {}).get('condition', '?')} — "
                f"level: {wn.get('bias_invalidation', {}).get('price_level', '?')} "
                f"({wn.get('bias_invalidation', {}).get('direction', '?')})\n\n"
            )
```

Then in the `return f"""..."""` at line 123, insert `{weekly_block}` before `{narrative_block}`:

```python
    return f"""You are an expert ICT (Inner Circle Trader) analyst for Gold XAU/USD.

CURRENT TIME: {time_str} ({day_str})
CURRENT KILLZONE: {current_kz}

{weekly_block}{narrative_block}{"Analyse these candles to identify the highest-probability trade setup. The senior analyst has provided the HTF narrative above — use it as your directional framework but VERIFY it against the 4H candles below. If your 4H reading contradicts the narrative, flag it." if htf_narrative else "Analyse these candles on TWO timeframes to identify the highest-probability trade setup."}
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
python -m pytest ml/tests/test_weekly_narrative.py::TestWeeklyPromptBlocks -v
```
Expected: all 5 tests PASS

- [ ] **Step 6: Commit**

```bash
git add ml/prompts.py ml/tests/test_weekly_narrative.py
git commit -m "feat: add weekly_narrative/weekly_matched_level params to build_enhanced_ict_prompt"
```

---

## Task 2: Weekly Opus prompt builder in `prompts.py`

**Files:**
- Modify: `ml/prompts.py` (add `OPUS_WEEKLY_SYSTEM` constant and `build_opus_weekly_narrative_prompt()` function)
- Test: `ml/tests/test_weekly_narrative.py`

- [ ] **Step 1: Write failing tests**

Append to `ml/tests/test_weekly_narrative.py`:

```python
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
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest ml/tests/test_weekly_narrative.py::TestBuildOpusWeeklyNarrativePrompt -v
```
Expected: `FAILED — ImportError: cannot import name 'build_opus_weekly_narrative_prompt'`

- [ ] **Step 3: Add `OPUS_WEEKLY_SYSTEM` and `build_opus_weekly_narrative_prompt()` to `ml/prompts.py`**

At the end of `ml/prompts.py`, after `build_opus_narrative_prompt()`, add:

```python
OPUS_WEEKLY_SYSTEM = """You are a senior ICT (Inner Circle Trader) macro analyst specialising in XAU/USD.
Your role is to identify the weekly-level institutional dealing range, Power of 3 phase, and key structural levels that define the macro environment for the coming week(s).

Focus on positional, not execution. Return ONLY valid JSON. No commentary, no markdown."""


def build_opus_weekly_narrative_prompt(weekly_candles: list,
                                        daily_candles: list | None = None,
                                        intermarket: dict | None = None) -> str:
    """Build prompt for Opus weekly macro narrative.

    Args:
        weekly_candles: Last 24 weekly candles (~6 months of macro context)
        daily_candles: Last 20 daily candles for macro anchoring (optional)
        intermarket: Intermarket context dict (optional)

    Returns:
        Complete weekly narrative prompt string
    """
    weekly_slim = _slim_candles(weekly_candles[-24:]) if weekly_candles else []
    daily_slim = _slim_candles(daily_candles[-20:]) if daily_candles else []

    now_utc = datetime.now(timezone.utc)
    time_str = now_utc.strftime("%Y-%m-%d %H:%M UTC")
    day_str = now_utc.strftime("%A")

    daily_block = ""
    if daily_slim:
        daily_block = f"""DAILY CANDLES (last {len(daily_slim)} — macro anchoring):
{json.dumps(daily_slim)}

"""

    intermarket_block = _build_intermarket_section(intermarket) if intermarket else ""

    return f"""Provide the WEEKLY MACRO NARRATIVE for XAU/USD.
Time: {time_str} ({day_str})

WEEKLY CANDLES (last {len(weekly_slim)} — ~{len(weekly_slim) * 7 // 30} months):
{json.dumps(weekly_slim)}

{daily_block}{intermarket_block}ANALYSIS (macro/positional — not execution):
1. QUARTERLY BIAS: Identify the dominant swing high and low over the past 3-6 months. Is price in institutional premium (upper half) or discount (lower half) of this weekly dealing range?
2. POWER OF 3 PHASE (weekly): Has weekly-level accumulation completed (stop runs below multi-week lows)? Is manipulation occurring (false breaks of range extremes)? Has distribution begun (sustained sell pressure after sweep)?
3. KEY WEEKLY LEVELS: Identify Previous Weekly Highs/Lows, significant weekly Order Blocks, weekly Fair Value Gaps, and quarterly highs/lows. Label each with its role (resistance/support/magnet/void).
4. PREMIUM ARRAY: List significant supply levels above current price where institutions are likely selling (weekly OBs, swept BSL zones, FVG tops).
5. DISCOUNT ARRAY: List significant demand levels below current price where institutions are likely buying (weekly OBs, swept SSL zones, FVG bottoms).
6. BIAS INVALIDATION: What single price event would flip the weekly directional bias?

Return ONLY valid JSON:
{{
  "macro_thesis": "string — 1-2 sentences on the dominant weekly narrative",
  "directional_bias": "bullish|bearish|neutral",
  "bias_confidence": 0.0,
  "p3_phase": "accumulation|manipulation|distribution",
  "dealing_range": {{"high": 0.0, "low": 0.0, "equilibrium": 0.0}},
  "premium_array": [{{"price": 0.0, "label": "string", "role": "resistance|magnet|void"}}],
  "discount_array": [{{"price": 0.0, "label": "string", "role": "support|magnet|void"}}],
  "key_levels": [{{"price": 0.0, "label": "string", "role": "resistance|support|magnet"}}],
  "bias_invalidation": {{"condition": "string", "price_level": 0.0, "direction": "above|below"}}
}}"""
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest ml/tests/test_weekly_narrative.py::TestBuildOpusWeeklyNarrativePrompt -v
```
Expected: all 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add ml/prompts.py ml/tests/test_weekly_narrative.py
git commit -m "feat: add build_opus_weekly_narrative_prompt and OPUS_WEEKLY_SYSTEM to prompts"
```

---

## Task 3: Scanner weekly cache fields, staleness check, proximity helper

**Files:**
- Modify: `ml/scanner.py` (`__init__` at line ~140, new methods after `_call_opus_narrative`)
- Test: `ml/tests/test_weekly_narrative.py`

- [ ] **Step 1: Write failing tests**

Append to `ml/tests/test_weekly_narrative.py`:

```python
class TestWeeklyCacheStale:

    def _make_scanner(self):
        """Return a scanner instance with mocked dependencies."""
        from unittest.mock import MagicMock, patch
        with patch("ml.scanner.TradeLogger"), \
             patch("ml.scanner.ScannerDB"), \
             patch("ml.scanner.get_config", return_value={"model_dir": "/tmp", "oanda_account_id": "x", "oanda_access_token": "y"}):
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
        from unittest.mock import patch
        with patch("ml.scanner.TradeLogger"), \
             patch("ml.scanner.ScannerDB"), \
             patch("ml.scanner.get_config", return_value={"model_dir": "/tmp", "oanda_account_id": "x", "oanda_access_token": "y"}):
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
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest ml/tests/test_weekly_narrative.py::TestWeeklyCacheStale ml/tests/test_weekly_narrative.py::TestIsNearWeeklyLevel -v
```
Expected: `FAILED — AttributeError: '_weekly_narrative_cache'`

- [ ] **Step 3: Add cache fields to `ScannerEngine.__init__`**

In `ml/scanner.py`, find the block at line ~192 (`# Prospect regeneration tracking...`) and add before it:

```python
        # Weekly macro narrative cache — 7-day TTL, cleared on weekly close (Sunday 21:00 UTC)
        self._weekly_narrative_cache: dict | None = None
        self._weekly_narrative_fetched_at: datetime | None = None
```

- [ ] **Step 4: Add `_is_weekly_cache_stale()` method**

In `ml/scanner.py`, add after `_call_opus_narrative()` (around line 2300):

```python
    def _is_weekly_cache_stale(self) -> bool:
        """Return True if the weekly narrative cache needs regeneration."""
        if self._weekly_narrative_cache is None or self._weekly_narrative_fetched_at is None:
            return True
        age_seconds = (datetime.utcnow() - self._weekly_narrative_fetched_at).total_seconds()
        if age_seconds > 7 * 24 * 3600:
            return True
        # Stale if fetched in a different ISO week (handles server restarts mid-week)
        now = datetime.utcnow()
        fetched = self._weekly_narrative_fetched_at
        if now.year != fetched.year or now.isocalendar()[1] != fetched.isocalendar()[1]:
            return True
        return False
```

- [ ] **Step 5: Add `_is_near_weekly_level()` method**

In `ml/scanner.py`, directly after `_is_weekly_cache_stale()`:

```python
    def _is_near_weekly_level(self, price: float, atr: float,
                               weekly_levels: list) -> tuple[bool, dict | None]:
        """Return (True, matched_level) if price is within 3×ATR of any weekly level."""
        if not weekly_levels or atr <= 0:
            return False, None
        for level in weekly_levels:
            level_price = level.get("price")
            if level_price is None:
                continue
            if abs(price - level_price) <= 3.0 * atr:
                return True, level
        return False, None
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
python -m pytest ml/tests/test_weekly_narrative.py::TestWeeklyCacheStale ml/tests/test_weekly_narrative.py::TestIsNearWeeklyLevel -v
```
Expected: all 11 tests PASS

- [ ] **Step 7: Commit**

```bash
git add ml/scanner.py ml/tests/test_weekly_narrative.py
git commit -m "feat: add weekly narrative cache fields and staleness/proximity helpers to scanner"
```

---

## Task 4: `_call_opus_weekly_narrative()` and `_get_weekly_narrative()`

**Files:**
- Modify: `ml/scanner.py` (add two methods after `_is_near_weekly_level`)
- Test: `ml/tests/test_weekly_narrative.py`

- [ ] **Step 1: Write failing tests**

Append to `ml/tests/test_weekly_narrative.py`:

```python
class TestGetWeeklyNarrative:

    def _make_scanner_with_cache(self, cache=None, fetched_at=None):
        from unittest.mock import patch
        from datetime import datetime
        with patch("ml.scanner.TradeLogger"), \
             patch("ml.scanner.ScannerDB"), \
             patch("ml.scanner.get_config", return_value={"model_dir": "/tmp", "oanda_account_id": "x", "oanda_access_token": "y"}):
            from ml.scanner import ScannerEngine
            engine = ScannerEngine.__new__(ScannerEngine)
            engine._weekly_narrative_cache = cache
            engine._weekly_narrative_fetched_at = fetched_at or (datetime.utcnow() if cache else None)
            return engine

    def test_returns_none_when_call_fails_and_no_cache(self):
        from unittest.mock import patch, MagicMock
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
        from unittest.mock import patch, MagicMock
        fresh_narrative = {"directional_bias": "bearish", "macro_thesis": "Selling."}
        engine = self._make_scanner_with_cache()  # cache=None → stale
        with patch.object(engine, "_call_opus_weekly_narrative", return_value=fresh_narrative) as mock_call:
            result = engine._get_weekly_narrative()
        mock_call.assert_called_once()
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest ml/tests/test_weekly_narrative.py::TestGetWeeklyNarrative -v
```
Expected: `FAILED — AttributeError: '_get_weekly_narrative'`

- [ ] **Step 3: Add `_call_opus_weekly_narrative()` to `ml/scanner.py`**

Add after `_is_near_weekly_level()`:

```python
    def _call_opus_weekly_narrative(self) -> dict | None:
        """Call Opus to generate the weekly macro narrative. Caches result for 7 days.

        Returns structured weekly narrative dict or None on failure.
        Cost: ~$0.05/call, called at most once per week (Sunday close clears cache).
        """
        from ml.prompts import build_opus_weekly_narrative_prompt, OPUS_WEEKLY_SYSTEM

        weekly_candles = self._get_htf_candles("1week", 24)
        daily_candles = self._get_htf_candles("1day", 20)

        if not weekly_candles:
            logger.warning("Weekly narrative: no weekly candles available from OANDA")
            return None

        prompt = build_opus_weekly_narrative_prompt(weekly_candles, daily_candles)

        for attempt in range(3):
            try:
                resp = httpx.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": self.claude_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": "claude-opus-4-6",
                        "max_tokens": 600,
                        "temperature": 0,
                        "system": OPUS_WEEKLY_SYSTEM,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                    timeout=90,
                )

                if resp.status_code in (429, 529):
                    wait = (2 ** attempt) * 2
                    logger.warning("Opus weekly narrative rate limited, waiting %ds", wait)
                    time.sleep(wait)
                    continue

                if resp.status_code != 200:
                    logger.error("Opus weekly narrative API error %d: %s",
                                 resp.status_code, resp.text[:200])
                    if attempt < 2:
                        time.sleep(2 ** attempt * 2)
                        continue
                    return None

                data = resp.json()
                text = ""
                for block in data.get("content", []):
                    if block.get("type") == "text":
                        text = block["text"]
                        break

                if not text:
                    return None

                clean = text.replace("```json", "").replace("```", "").strip()
                json_start = clean.find("{")
                json_end = clean.rfind("}")
                if json_start >= 0 and json_end > json_start:
                    clean = clean[json_start:json_end + 1]

                narrative = _safe_load_claude_json(clean, "opus_weekly_narrative")

                if not narrative.get("directional_bias"):
                    logger.warning("Weekly narrative missing directional_bias — discarding")
                    return None

                self._weekly_narrative_cache = narrative
                self._weekly_narrative_fetched_at = datetime.utcnow()

                try:
                    from ml.cost_tracker import get_cost_tracker
                    usage = data.get("usage", {})
                    get_cost_tracker().log_call(
                        "claude-opus-4-6",
                        usage.get("input_tokens", 1500),
                        usage.get("output_tokens", 400),
                        "weekly_narrative")
                except Exception:
                    pass

                logger.info("Opus weekly narrative: %s bias, %s P3 phase",
                            narrative.get("directional_bias"),
                            narrative.get("p3_phase", "?"))
                return narrative

            except Exception as e:
                logger.warning("Opus weekly narrative attempt %d failed: %s", attempt + 1, e)
                if attempt < 2:
                    time.sleep(2 ** attempt * 2)

        return None
```

- [ ] **Step 4: Add `_get_weekly_narrative()` to `ml/scanner.py`**

Directly after `_call_opus_weekly_narrative()`:

```python
    def _get_weekly_narrative(self) -> dict | None:
        """Return weekly narrative from cache, regenerating if stale."""
        if self._is_weekly_cache_stale():
            try:
                self._call_opus_weekly_narrative()
            except Exception as e:
                logger.warning("Weekly narrative call failed: %s", e)
        return self._weekly_narrative_cache
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
python -m pytest ml/tests/test_weekly_narrative.py::TestGetWeeklyNarrative -v
```
Expected: all 3 tests PASS

- [ ] **Step 6: Commit**

```bash
git add ml/scanner.py ml/tests/test_weekly_narrative.py
git commit -m "feat: add _call_opus_weekly_narrative and _get_weekly_narrative to scanner"
```

---

## Task 5: Inject weekly context in `_scan_timeframe()` and `_call_claude()`

**Files:**
- Modify: `ml/scanner.py` (`_scan_timeframe` around line 493, `_call_claude` signature at line 3616 and call at line 3676)
- Test: `ml/tests/test_weekly_narrative.py`

- [ ] **Step 1: Write failing tests**

Append to `ml/tests/test_weekly_narrative.py`:

```python
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
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest ml/tests/test_weekly_narrative.py::TestWeeklyInjectionInScanTimeframe -v
```
Expected: `FAILED — TypeError: _call_claude() got unexpected keyword argument 'weekly_narrative'`

- [ ] **Step 3: Add `weekly_narrative` and `weekly_matched_level` to `_call_claude()` signature**

In `ml/scanner.py` at line 3616, change:

```python
    def _call_claude(self, candles: list, htf_candles: list,
                     timeframe: str = "1h",
                     intermarket: dict | None = None,
                     htf_narrative: dict | None = None,
                     setup_context: dict | None = None,
                     prev_narrative: dict | None = None,
                     invalidation_status: str | None = None,
                     recent_context: dict | None = None,
                     haiku_zone_hint: str | None = None,
                     ml_context: dict | None = None) -> dict | None:
```
to:
```python
    def _call_claude(self, candles: list, htf_candles: list,
                     timeframe: str = "1h",
                     intermarket: dict | None = None,
                     htf_narrative: dict | None = None,
                     setup_context: dict | None = None,
                     prev_narrative: dict | None = None,
                     invalidation_status: str | None = None,
                     recent_context: dict | None = None,
                     haiku_zone_hint: str | None = None,
                     ml_context: dict | None = None,
                     weekly_narrative: dict | None = None,
                     weekly_matched_level: dict | None = None) -> dict | None:
```

- [ ] **Step 4: Pass weekly params through to `build_enhanced_ict_prompt()`**

In `ml/scanner.py`, find the `build_enhanced_ict_prompt(...)` call around line 3676 and add the two new kwargs:

```python
        prompt = build_enhanced_ict_prompt(
            candles,
            htf_candles,
            intermarket=intermarket,
            htf_narrative=htf_narrative,
            setup_context=setup_context,
            narrative_weights=_nw,
            prev_narrative=prev_narrative,
            invalidation_status=invalidation_status,
            recent_context=recent_context,
            regime_context=_regime_ctx,
            ml_context=ml_context,
            htf_label=htf_tf,
            key_levels=_key_levels,
            weekly_narrative=weekly_narrative,
            weekly_matched_level=weekly_matched_level)
```

- [ ] **Step 5: Add weekly context retrieval in `_scan_timeframe()` after the Opus narrative block**

In `ml/scanner.py`, find the Opus narrative block ending around line 493:
```python
            except Exception as e:
                logger.warning("Scanner [%s]: Opus narrative failed (proceeding without): %s",
                               timeframe, e)
```

After that block, add:

```python
            # Weekly macro context — 7-day cache, injected based on timeframe + proximity
            weekly_narrative = None
            weekly_matched_level = None
            try:
                _wn = self._get_weekly_narrative()
                if _wn:
                    if timeframe == "1day":
                        weekly_narrative = _wn
                    else:
                        _all_weekly_levels = (
                            _wn.get("key_levels", []) +
                            _wn.get("premium_array", []) +
                            _wn.get("discount_array", [])
                        )
                        _current_price = candles[-1]["close"] if candles else 0.0
                        _atr = (sum(c["high"] - c["low"] for c in candles[-14:]) / 14
                                if len(candles) >= 14 else 0.0)
                        _near, _matched = self._is_near_weekly_level(
                            _current_price, _atr, _all_weekly_levels)
                        if _near:
                            weekly_narrative = _wn
                            weekly_matched_level = _matched
                            logger.info(
                                "Scanner [%s]: weekly level proximity — %s at %.2f",
                                timeframe,
                                _matched.get("label", "?"),
                                _matched.get("price", 0.0),
                            )
            except Exception as e:
                logger.warning("Scanner [%s]: weekly narrative retrieval failed: %s",
                               timeframe, e)
```

- [ ] **Step 6: Pass `weekly_narrative` and `weekly_matched_level` to the `_call_claude()` call**

Find the `_call_claude(...)` call around line 664 and add the two new kwargs:

```python
            analysis = self._call_claude(candles, htf_candles or [], timeframe,
                                         intermarket=intermarket_ctx,
                                         htf_narrative=htf_narrative,
                                         setup_context=setup_context,
                                         prev_narrative=prev_narrative,
                                         invalidation_status=invalidation_status,
                                         recent_context=recent_context,
                                         haiku_zone_hint=haiku_zone_hint,
                                         ml_context=ml_context,
                                         weekly_narrative=weekly_narrative,
                                         weekly_matched_level=weekly_matched_level)
```

- [ ] **Step 7: Run tests to verify they pass**

```bash
python -m pytest ml/tests/test_weekly_narrative.py::TestWeeklyInjectionInScanTimeframe -v
```
Expected: all 2 tests PASS

- [ ] **Step 8: Run full test suite to check for regressions**

```bash
python -m pytest ml/tests/ -v --tb=short 2>&1 | tail -30
```
Expected: existing tests all PASS; new tests PASS

- [ ] **Step 9: Commit**

```bash
git add ml/scanner.py
git commit -m "feat: inject weekly narrative context into _scan_timeframe and _call_claude"
```

---

## Task 6: Weekly close scheduler job

**Files:**
- Modify: `ml/scheduler.py` (new job function, register in `start_scheduler`)
- Test: `ml/tests/test_weekly_narrative.py`

- [ ] **Step 1: Write failing test**

Append to `ml/tests/test_weekly_narrative.py`:

```python
class TestWeeklySchedulerJob:

    def test_weekly_job_registered_in_scheduler(self):
        """Verify the weekly cache clear job is registered on Sunday 21:00 UTC."""
        from unittest.mock import patch, MagicMock
        mock_scheduler = MagicMock()
        added_jobs = []

        def capture_add_job(fn, trigger, **kwargs):
            added_jobs.append({"fn": fn, "trigger": trigger, "kwargs": kwargs})

        mock_scheduler.add_job.side_effect = capture_add_job

        with patch("ml.scheduler.AsyncIOScheduler", return_value=mock_scheduler), \
             patch("ml.scheduler.os.environ.get", return_value="fake-key"), \
             patch("ml.scheduler.get_config", return_value={
                 "oanda_account_id": "x", "oanda_access_token": "y"}):
            from ml.scheduler import start_scheduler
            start_scheduler()

        job_ids = [j["kwargs"].get("id") for j in added_jobs]
        assert "weekly_cache_clear" in job_ids

        weekly_job = next(j for j in added_jobs if j["kwargs"].get("id") == "weekly_cache_clear")
        assert weekly_job["kwargs"].get("day_of_week") == "sun"
        assert weekly_job["kwargs"].get("hour") == 21
        assert weekly_job["kwargs"].get("minute") == 0
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest ml/tests/test_weekly_narrative.py::TestWeeklySchedulerJob -v
```
Expected: `FAILED — AssertionError: 'weekly_cache_clear' not in job_ids`

- [ ] **Step 3: Add `_weekly_cache_clear_job()` to `ml/scheduler.py`**

In `ml/scheduler.py`, add after `_recompute_intermarket_job()` and before `start_scheduler()`:

```python
async def _weekly_cache_clear_job():
    """Clear weekly narrative cache on Sunday 21:00 UTC (weekly candle close).

    The scanner will regenerate the narrative on the next scan after close.
    """
    try:
        engine = get_shared_engine()
        engine._weekly_narrative_cache = None
        engine._weekly_narrative_fetched_at = None
        logger.info("Weekly narrative cache cleared — will regenerate on next scan")
        print("[WEEKLY] Cache cleared at Sunday 21:00 UTC — will regenerate on next 1day scan")
    except Exception as e:
        print(f"[WEEKLY] Cache clear failed: {e}")
        logger.error("Weekly cache clear failed: %s", e, exc_info=True)
```

- [ ] **Step 4: Register the job in `start_scheduler()`**

In `ml/scheduler.py`, inside `start_scheduler()`, add after the intermarket recompute job registration (around line 446):

```python
        # Clear weekly narrative cache on Sunday 21:00 UTC (weekly candle close)
        _scheduler.add_job(
            _weekly_cache_clear_job, "cron",
            day_of_week="sun", hour=21, minute=0,
            id="weekly_cache_clear",
            replace_existing=True,
        )
```

- [ ] **Step 5: Run test to verify it passes**

```bash
python -m pytest ml/tests/test_weekly_narrative.py::TestWeeklySchedulerJob -v
```
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add ml/scheduler.py ml/tests/test_weekly_narrative.py
git commit -m "feat: add weekly narrative cache clear job to scheduler (Sunday 21:00 UTC)"
```

---

## Task 7: Server endpoints

**Files:**
- Modify: `ml/server.py` (add two endpoints)
- Test: `ml/tests/test_weekly_narrative.py`

- [ ] **Step 1: Write failing tests**

Append to `ml/tests/test_weekly_narrative.py`:

```python
class TestWeeklyNarrativeEndpoints:

    def _make_client(self, weekly_cache=None, fetched_at=None):
        from unittest.mock import patch, MagicMock
        from datetime import datetime
        from fastapi.testclient import TestClient
        from ml.server import app

        mock_engine = MagicMock()
        mock_engine._weekly_narrative_cache = weekly_cache
        mock_engine._weekly_narrative_fetched_at = fetched_at or (datetime.utcnow() if weekly_cache else None)

        with patch("ml.server.get_shared_engine", return_value=mock_engine):
            client = TestClient(app)
            return client, mock_engine

    def test_get_weekly_narrative_404_when_no_cache(self):
        client, _ = self._make_client(weekly_cache=None)
        resp = client.get("/weekly/narrative")
        assert resp.status_code == 404
        assert "POST /weekly/refresh" in resp.json()["detail"]

    def test_get_weekly_narrative_200_with_cache(self):
        from datetime import datetime
        cache = {
            "macro_thesis": "Gold bearish.", "directional_bias": "bearish",
            "bias_confidence": 0.75, "p3_phase": "distribution",
            "dealing_range": {"high": 3500.0, "low": 3100.0, "equilibrium": 3300.0},
            "premium_array": [], "discount_array": [], "key_levels": [],
            "bias_invalidation": {"condition": "Close above 3500", "price_level": 3500.0, "direction": "above"},
        }
        client, _ = self._make_client(weekly_cache=cache, fetched_at=datetime.utcnow())
        resp = client.get("/weekly/narrative")
        assert resp.status_code == 200
        body = resp.json()
        assert body["directional_bias"] == "bearish"
        assert "last_updated" in body
        assert "next_refresh" in body
        assert "cache_age_hours" in body

    def test_post_weekly_refresh_triggers_call(self):
        from datetime import datetime
        fresh = {
            "macro_thesis": "Gold now bullish.", "directional_bias": "bullish",
            "bias_confidence": 0.6, "p3_phase": "accumulation",
            "dealing_range": {"high": 3500.0, "low": 3100.0, "equilibrium": 3300.0},
            "premium_array": [], "discount_array": [], "key_levels": [],
            "bias_invalidation": {"condition": "Close below 3100", "price_level": 3100.0, "direction": "below"},
        }
        client, mock_engine = self._make_client(weekly_cache=None)
        mock_engine._call_opus_weekly_narrative.return_value = fresh
        mock_engine._weekly_narrative_cache = None

        def side_effect():
            mock_engine._weekly_narrative_cache = fresh
            mock_engine._weekly_narrative_fetched_at = datetime.utcnow()
            return fresh

        mock_engine._call_opus_weekly_narrative.side_effect = side_effect

        from unittest.mock import patch
        with patch("ml.server.get_shared_engine", return_value=mock_engine):
            from fastapi.testclient import TestClient
            from ml.server import app
            c = TestClient(app)
            resp = c.post("/weekly/refresh")

        assert resp.status_code == 200
        mock_engine._call_opus_weekly_narrative.assert_called_once()

    def test_post_weekly_refresh_500_when_opus_fails(self):
        client, mock_engine = self._make_client(weekly_cache=None)
        mock_engine._call_opus_weekly_narrative.return_value = None

        from unittest.mock import patch
        with patch("ml.server.get_shared_engine", return_value=mock_engine):
            from fastapi.testclient import TestClient
            from ml.server import app
            c = TestClient(app)
            resp = c.post("/weekly/refresh")

        assert resp.status_code == 500
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest ml/tests/test_weekly_narrative.py::TestWeeklyNarrativeEndpoints -v
```
Expected: `FAILED — 404 Not Found` (routes don't exist yet)

- [ ] **Step 3: Add endpoints to `ml/server.py`**

Find a logical grouping point in `ml/server.py` — add after the `/narrative/evolution` or `/narrative/weights` endpoints. Add:

```python
@app.get("/weekly/narrative")
def get_weekly_narrative():
    """Return current weekly macro narrative from Opus cache.

    Returns 404 if not yet generated — POST /weekly/refresh to generate.
    """
    from ml.scanner import get_shared_engine
    from datetime import datetime, timedelta
    from fastapi import HTTPException

    engine = get_shared_engine()
    cache = getattr(engine, "_weekly_narrative_cache", None)
    fetched_at = getattr(engine, "_weekly_narrative_fetched_at", None)

    if not cache:
        raise HTTPException(
            status_code=404,
            detail="Weekly narrative not yet generated. POST /weekly/refresh to generate."
        )

    now = datetime.utcnow()
    days_until_sunday = (6 - now.weekday()) % 7
    if days_until_sunday == 0 and now.hour >= 21:
        days_until_sunday = 7
    next_sunday = (now + timedelta(days=days_until_sunday)).replace(
        hour=21, minute=0, second=0, microsecond=0)

    cache_age_hours = (
        round((now - fetched_at).total_seconds() / 3600, 1) if fetched_at else None
    )

    return {
        **cache,
        "last_updated": fetched_at.isoformat() if fetched_at else None,
        "next_refresh": next_sunday.isoformat(),
        "cache_age_hours": cache_age_hours,
    }


@app.post("/weekly/refresh")
def refresh_weekly_narrative():
    """Clear the weekly narrative cache and regenerate immediately via Opus.

    Returns the fresh narrative or 500 if Opus call failed.
    """
    from ml.scanner import get_shared_engine
    from fastapi import HTTPException
    from datetime import datetime

    engine = get_shared_engine()
    engine._weekly_narrative_cache = None
    engine._weekly_narrative_fetched_at = None

    result = engine._call_opus_weekly_narrative()
    if not result:
        raise HTTPException(
            status_code=500,
            detail="Opus weekly narrative call failed — check logs for details."
        )

    return get_weekly_narrative()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest ml/tests/test_weekly_narrative.py::TestWeeklyNarrativeEndpoints -v
```
Expected: all 4 tests PASS

- [ ] **Step 5: Run the full new test file**

```bash
python -m pytest ml/tests/test_weekly_narrative.py -v
```
Expected: all tests PASS

- [ ] **Step 6: Run full test suite**

```bash
python -m pytest ml/tests/ -v --tb=short 2>&1 | tail -40
```
Expected: all existing tests PASS, new tests PASS

- [ ] **Step 7: Commit**

```bash
git add ml/server.py ml/tests/test_weekly_narrative.py
git commit -m "feat: add GET /weekly/narrative and POST /weekly/refresh endpoints"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Covered by |
|---|---|
| Weekly Opus prompt (macro_thesis, bias, p3_phase, dealing_range, premium/discount arrays, key_levels, bias_invalidation) | Task 2 |
| `_call_opus_weekly_narrative()` with 24 weekly + 20 daily candles | Task 4 |
| 7-day cache (`_weekly_narrative_cache`, `_weekly_narrative_fetched_at`) | Task 3 |
| `_is_weekly_cache_stale()` — None/age/ISO week checks | Task 3 |
| `_is_near_weekly_level()` — 3×ATR threshold, key_levels+premium+discount | Task 3 |
| 1day always gets full WEEKLY MACRO CONTEXT block | Task 1 + Task 5 |
| 4H/1H/15min get WEEKLY KEY LEVEL PROXIMITY block on proximity hit | Task 1 + Task 5 |
| No injection for 4H/1H/15min when far from levels | Task 5 |
| Weekly context appears before HTF narrative block in prompt | Task 1 |
| Sunday 21:00 UTC cache clear scheduler job | Task 6 |
| `GET /weekly/narrative` — 200 with cache, 404 without | Task 7 |
| `POST /weekly/refresh` — clears cache, regenerates, returns narrative | Task 7 |
| Error handling — Opus fail → log warning, scan proceeds | Task 4 + Task 5 |
| `OPUS_WEEKLY_SYSTEM` constant | Task 2 |

All spec requirements have a corresponding task. No gaps found.
