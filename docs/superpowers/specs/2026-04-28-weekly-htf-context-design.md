# Weekly HTF Context Layer — Design Spec
**Date:** 2026-04-28

## Summary

Add a weekly timeframe as a pure HTF context layer. Opus generates a macro weekly narrative (directional bias, key levels, dealing ranges, P3 phase) from 24 weekly candles. This narrative is cached for 7 days, refreshed when the weekly candle closes (Sunday 21:00 UTC), and injected conditionally into all scanning timeframes. No tradeable setups are generated from the weekly chart.

---

## Goals

- Provide authoritative weekly-level macro context to all timeframe analysis
- Give the daily scan a genuine HTF anchor (currently it has `htf: None`)
- Cascade weekly key levels down to 4H/1H/15min when price is in proximity
- Surface the weekly narrative via a dedicated endpoint (visible, not buried)

## Non-Goals

- Weekly is not a scanning timeframe — no Haiku screen, no Sonnet analysis, no setup notifications
- No thesis lifecycle tracking (no prediction scoring, invalidation engine, confidence decay)
- No changes to NarrativeStore

---

## Architecture

```
Weekly candles (24) + Daily candles (20, macro anchor)
    ↓
_call_opus_weekly_narrative()   [new, 7-day cache on scanner instance]
    ↓
_weekly_narrative_cache: {
    macro_thesis, directional_bias, bias_confidence,
    p3_phase, dealing_range, premium_array, discount_array,
    key_levels, bias_invalidation, last_updated
}
    ↓
Per-scan injection logic:
  1day        → always inject as "WEEKLY MACRO CONTEXT"
  4H/1H/15min → inject as "WEEKLY KEY LEVEL PROXIMITY" only when
                price is within 3×ATR of any weekly key_level

GET /weekly/narrative   → returns cache + metadata (last_updated, next_refresh, cache_age_hours)
POST /weekly/refresh    → clears cache, regenerates immediately
Scheduled job           → clears cache Sunday 21:00 UTC (weekly candle close)
```

---

## Components & Changes

### `scanner.py`

**New instance fields:**
```python
_weekly_narrative_cache: dict | None = None
_weekly_narrative_fetched_at: datetime | None = None
```

**New methods:**

`_is_weekly_cache_stale() -> bool`
- Returns True if cache is None
- Returns True if cache age > 7 days
- Returns True if `_weekly_narrative_fetched_at` is from a previous ISO week (Sunday boundary)
- Returns False otherwise

`_call_opus_weekly_narrative() -> dict | None`
- Fetches 24 weekly candles + 20 daily candles via `_get_htf_candles()`
- Calls Opus with the weekly macro prompt (see Prompt section)
- On success: sets `_weekly_narrative_cache` and `_weekly_narrative_fetched_at`
- On failure: logs warning, returns None (analysis proceeds without weekly context)

`_get_weekly_narrative() -> dict | None`
- Checks `_is_weekly_cache_stale()`; if stale calls `_call_opus_weekly_narrative()`
- Returns `_weekly_narrative_cache` (may be None if Opus failed)

`_is_near_weekly_level(price: float, atr: float, weekly_levels: list) -> tuple[bool, dict | None]`
- Iterates `weekly_levels` (key_levels + premium_array + discount_array from cache)
- Returns `(True, level)` if `abs(price - level["price"]) <= 3.0 * atr`
- Returns `(False, None)` if no match

**Changes to `_scan_timeframe()`:**
- Call `_get_weekly_narrative()` once at the start of each scan cycle (cached, cheap)
- For 1day: pass full `weekly_narrative` to `build_enhanced_ict_prompt()`
- For 4H/1H/15min: run `_is_near_weekly_level()` using current close price + ATR; pass `weekly_narrative` only if proximity check fires
- Log when weekly context is injected (timeframe + matched level label if proximity)

**New scheduled job:**
- Fires Sunday 21:00 UTC weekly
- Clears `_weekly_narrative_cache = None` and `_weekly_narrative_fetched_at = None`
- Logs: "Weekly narrative cache cleared — will regenerate on next scan"

### `prompts.py`

**`build_enhanced_ict_prompt()` changes:**
- Add `weekly_narrative: dict | None = None` parameter
- When provided, insert a `WEEKLY MACRO CONTEXT` block immediately before the existing HTF narrative section
- Block content varies by injection mode:
  - **Full (1day):** macro thesis, directional bias + confidence, P3 phase, dealing range (premium/discount equilibrium), all key levels with labels and roles
  - **Proximity (4H/1H/15min):** header `WEEKLY KEY LEVEL PROXIMITY`, matched level price + label + role, overall weekly directional bias only — not the full context dump

**Wording guidance in prompt block:**
- Frame weekly context as authoritative constraint: "This is the macro framework. Your analysis must be consistent with this weekly structure or explicitly explain why local conditions diverge."
- For proximity block: "Price is approaching a significant weekly level. Weight your entry criteria accordingly."

### `server.py`

**`GET /weekly/narrative`**
```
Response 200:
{
  "macro_thesis": str,
  "directional_bias": "bullish|bearish|neutral",
  "bias_confidence": float,
  "p3_phase": "accumulation|manipulation|distribution",
  "dealing_range": {"high": float, "low": float, "equilibrium": float},
  "premium_array": [{"price": float, "label": str, "role": str}],
  "discount_array": [{"price": float, "label": str, "role": str}],
  "key_levels": [{"price": float, "label": str, "role": str}],
  "bias_invalidation": {"condition": str, "price_level": float, "direction": "above|below"},
  "last_updated": "ISO timestamp",
  "next_refresh": "ISO timestamp (next Sunday 21:00 UTC)",
  "cache_age_hours": float
}
Response 404: { "detail": "Weekly narrative not yet generated. POST /weekly/refresh to generate." }
```

**`POST /weekly/refresh`**
```
Clears _weekly_narrative_cache
Calls scanner._call_opus_weekly_narrative() immediately
Response 200: same shape as GET /weekly/narrative
Response 500: { "detail": "Opus call failed: <reason>" }
```

---

## Weekly Opus Prompt

**Input:** 24 weekly candles (XAU/USD) + 20 daily candles (for macro anchoring) + intermarket context (DXY proxy, US10Y)

**Prompt focus:** positional/macro, not execution. Opus identifies:
- Quarterly bias: is gold in long-term premium or discount relative to the weekly institutional dealing range?
- Power of 3 phase at the weekly level
- Significant OBs, FVGs, previous weekly/quarterly highs and lows — with role labels
- 1–2 sentence macro thesis
- Single invalidation condition that would flip the weekly bias

**Structured JSON output:**
```json
{
  "macro_thesis": "string (1-2 sentences)",
  "directional_bias": "bullish|bearish|neutral",
  "bias_confidence": 0.0–1.0,
  "p3_phase": "accumulation|manipulation|distribution",
  "dealing_range": {
    "high": float,
    "low": float,
    "equilibrium": float
  },
  "premium_array": [{"price": float, "label": str, "role": str}],
  "discount_array": [{"price": float, "label": str, "role": str}],
  "key_levels": [{"price": float, "label": str, "role": str}],
  "bias_invalidation": {
    "condition": str,
    "price_level": float,
    "direction": "above|below"
  }
}
```

---

## Weekly Close Detection

The weekly candle closes Sunday at 21:00–22:00 UTC (Forex market close). The scheduled job fires at **Sunday 21:00 UTC** using the existing asyncio scheduler in `server.py`. This clears the cache; the next scan on any timeframe after Sunday 21:00 UTC will trigger a fresh Opus call.

Stale cache detection in `_is_weekly_cache_stale()` also uses ISO week number as a fallback — if the cache was fetched in a previous ISO week, it's considered stale regardless of age in hours. This handles edge cases like server restarts mid-week.

---

## Error Handling

- Opus weekly call fails → log warning, `_weekly_narrative_cache` stays None, scan proceeds without weekly context (no crash)
- Weekly candles unavailable from OANDA → same as above
- `/weekly/narrative` GET before first generation → 404 with helpful message
- Proximity check on missing ATR → skip injection silently (ATR is required for the check; if unavailable, treat as no proximity)

---

## Testing

New tests in `ml/tests/`:
- `test_weekly_narrative.py`: cache staleness logic, proximity check (near/far/edge cases), Opus output parsing, endpoint responses (200/404/500)
- Extend `test_scanner.py`: verify weekly context injected for 1day, verify proximity injection for 4H when near level, verify no injection for 4H when far from levels
- Extend `test_prompts.py`: verify WEEKLY MACRO CONTEXT block present in 1day prompt, verify WEEKLY KEY LEVEL PROXIMITY block present in proximity case, verify neither block present when weekly_narrative=None

---

## Affected Files

| File | Change type |
|---|---|
| `ml/scanner.py` | New methods + scheduled job + injection logic |
| `ml/prompts.py` | New `weekly_narrative` param + two prompt blocks |
| `ml/server.py` | Two new endpoints + scheduled job wiring |
| `ml/tests/test_weekly_narrative.py` | New test file |
