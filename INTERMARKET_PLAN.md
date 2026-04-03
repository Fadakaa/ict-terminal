# Intermarket Correlation Plan — DXY + US10Y for Gold Analysis

## Why This Matters

Gold moves inversely to the US dollar and real yields. When DXY goes up, gold tends to go down (gold priced in dollars — stronger dollar = more expensive for foreign buyers). Higher US 10Y yields make bonds more attractive vs gold (which pays no yield), pulling money out of gold. This is the strongest macro driver of gold's medium-term direction.

The ICT angle: when gold sweeps buy-side liquidity but DXY is simultaneously breaking structure bullish, that sweep is likely manipulation, not distribution — price will reverse down. If gold drops into an OB in discount AND DXY is hitting resistance AND yields are rolling over, that's a triple-confluence long. The system currently has zero visibility into any of this.

## Implementation

### 1. Data Fetching — `ml/scanner.py`

Add a method to fetch correlated instruments alongside XAU/USD:

```python
def _fetch_candles_symbol(self, symbol: str, interval: str, count: int) -> list | None:
    """Fetch candles for any symbol from Twelve Data."""
    # Same as _fetch_candles but with configurable symbol param
```

Then a wrapper:
```python
def _fetch_correlated(self, interval: str, count: int) -> dict:
    dxy = self._fetch_candles_symbol("DXY", interval, count)
    us10y = self._fetch_candles_symbol("US10Y", interval, count)
    return {"dxy": dxy, "us10y": us10y}
```

**CRITICAL — Timezone and timeframe alignment:**
- ALL three instruments (XAU/USD, DXY, US10Y) MUST use `timezone: "UTC"` in the Twelve Data API call
- ALL three MUST use the SAME interval parameter (if gold is 1h, DXY and US10Y must also be 1h)
- When computing correlation, only use candles where timestamps MATCH EXACTLY across all instruments. Drop any candle where one instrument has data but another doesn't (e.g. different trading hours). Log a warning if >20% of candles can't be aligned.
- DXY and US10Y have different trading hours than XAU/USD. Gold trades ~23h/day (Sunday 5pm - Friday 5pm ET). DXY is forex hours. US10Y is US market hours. During Asian session, DXY may have thin data and US10Y may have no data at all. The correlation context must account for this — if US10Y candles aren't available for the current session, note "Yield data unavailable (outside US market hours)" rather than computing on stale data.

**Rate limit budget:**
- Free tier: 8 credits/min, limiter uses 7
- Each correlated fetch = 2 extra credits (DXY + US10Y)
- Cache DXY and US10Y for 30 minutes on sub-1H timeframes (macro relationships don't shift on 5-min intervals)
- On 1H+ timeframes, fetch fresh each cycle (still within budget since those TFs check infrequently)

### 2. Intermarket Context Module — `ml/intermarket.py` (NEW FILE)

Create a new module with:

```python
def compute_intermarket_context(gold: list, dxy: list, us10y: list, session: str) -> dict:
```

**Step 1: Timestamp alignment**
- Align gold, DXY, and US10Y candles by matching on datetime string (truncated to minute)
- Only keep candles present in ALL instruments
- If alignment drops >20% of candles, log warning
- Store aligned candle count in output

**Step 2: Compute metrics**
- `gold_20bar_change_pct` — gold close change over last 20 aligned candles (%)
- `dxy_20bar_change_pct` — DXY close change over last 20 aligned candles (%)
- `us10y_20bar_change_pct` — US10Y close change over last 20 aligned candles (%)
- `gold_dxy_correlation_20` — Pearson correlation of gold vs DXY close-to-close returns over last 20 candles. Normally this is negative (-0.3 to -0.8). If positive, that's a divergence.
- `gold_dxy_diverging` — boolean: are gold and DXY both moving in the same direction? (Both up or both down over 20 bars)
- `dxy_range_position` — where DXY sits in its own 20-bar range (0.0 = at low, 1.0 = at high)
- `us10y_range_position` — same for yields
- `dxy_current` — latest DXY close
- `us10y_current` — latest US10Y close

**Step 3: Session-aware narrative builder**

The narrative must account for which session we're in because the gold-DXY relationship strength varies:

- **London/NY overlap (12:00-16:00 UTC):** Tightest inverse correlation. Both gold and FX are heavily traded by the same institutional desks. A divergence here is a STRONG warning signal.
- **London solo (07:00-12:00 UTC):** Moderate correlation. FX is active, gold is active. Divergence is meaningful.
- **NY solo (16:00-21:00 UTC):** Moderate correlation. Both still active.
- **Asian (00:00-07:00 UTC):** WEAKEST correlation. Gold often moves independently (Chinese central bank buying, physical demand from Asia). Don't over-weight a divergence during Asian session — gold may legitimately move against DXY here.
- **Off hours (21:00-00:00 UTC):** Very thin liquidity for both. Correlation data is unreliable. Note this.

Narrative examples:
- "CONFIRMING: Gold rising as DXY weakens — normal inverse relationship intact. Gold longs have macro support. [London/NY overlap — highest confidence]"
- "WARNING: Gold rising WITH DXY — unusual positive correlation. Gold rally may be unsustainable if DXY strength continues. [London session — moderate significance]"
- "Gold dropping while yields are falling — suggests a liquidity grab, not a real move driven by rate expectations. Potential manipulation."
- "Gold-DXY correlation has broken down (r=0.15) — gold trading on idiosyncratic flows (geopolitical, central bank buying). Rely on gold structure alone."
- "Yield data unavailable (Asian session — outside US market hours). DXY correlation only."

### 3. Prompt Injection — `ml/prompts.py`

Add an `INTERMARKET CONTEXT` section to `build_enhanced_ict_prompt()`. The function signature changes to:

```python
def build_enhanced_ict_prompt(candles_1h: list, candles_4h: list,
                               intermarket: dict = None) -> str:
```

Insert after the candle data, before the analysis framework:

```
INTERMARKET CONTEXT:
DXY (Dollar Index): {dxy_current} | 20-bar change: {dxy_change}% | Range position: {dxy_range_position}
US 10Y Yield: {us10y_current} | 20-bar change: {yield_change}% | Range position: {us10y_range_position}
Gold-DXY Correlation (20-bar): {correlation} (normal: -0.3 to -0.8)
Divergence: {divergence_status}
Session context: {session_narrative}

INTERMARKET RULES — use these to validate or invalidate your gold setup:
- Gold LONG setup + DXY breaking out bullish → LOWER confidence, flag in warnings
- Gold SHORT setup + DXY at support / yields falling → LOWER confidence, flag in warnings
- Gold-DXY divergence (both same direction) during London/NY → HIGH risk, consider no_trade
- Gold-DXY divergence during Asian session → LOWER significance (gold trades independently in Asia)
- Correlation breakdown (|r| < 0.3) → gold on its own flows, rely on structure alone
- Yields rising sharply → headwind for gold longs, tailwind for gold shorts
- Yields falling sharply → tailwind for gold longs, headwind for gold shorts
- Gold dropping while yields ALSO falling → likely a liquidity grab / manipulation, not a real bearish move
```

If `intermarket` is None (e.g. API fetch failed, or US10Y unavailable during Asian session), omit the section entirely rather than sending stale/empty data.

### 4. Scanner Integration — `ml/scanner.py`

In `_scan_timeframe()`, after fetching gold candles and before calling Claude:

```python
# Fetch correlated instruments (cached for 30min on sub-1H)
correlated = self._fetch_correlated(interval, count=20)

# Compute intermarket context
from ml.intermarket import compute_intermarket_context
from ml.volatility import detect_session
session = detect_session(candles)
intermarket_ctx = compute_intermarket_context(
    candles, correlated.get("dxy"), correlated.get("us10y"), session
) if correlated.get("dxy") else None
```

Then pass `intermarket_ctx` to the prompt builder.

### 5. ML Features — `ml/training.py` + `ml/claude_bridge.py`

Add 4 new features to `INFERENCE_FEATURES` in training.py:
```python
"gold_dxy_corr_20",       # Pearson correlation (-1 to 1)
"gold_dxy_diverging",     # 0 or 1
"dxy_range_position",     # 0.0 to 1.0
"yield_direction",        # 1 (rising) or -1 (falling)
```

Update `_build_minimal_features()` in claude_bridge.py to include these 4 fields with default values (0.0, 0, 0.5, 0) when intermarket data isn't available.

Store intermarket context in the scanner DB `calibration_json` so it's available for backfill and analysis later.

### 6. Haiku Screen Enhancement

Also pass a condensed intermarket summary to `build_screen_prompt()` so Haiku can factor in macro context when deciding whether to escalate to Sonnet:

```
DXY: {direction} {change}% | Yields: {direction} {change}% | Divergence: {yes/no}
```

This is 1 line of extra tokens — negligible cost increase.

## Testing

- Unit test `compute_intermarket_context()` with mock candle data for all 5 sessions
- Test timestamp alignment with mismatched candle arrays (different lengths, gaps)
- Test narrative generation for all divergence/convergence combinations
- Test graceful degradation when DXY or US10Y fetch returns None
- Test that prompt builder omits intermarket section when data unavailable
- Integration test: mock Twelve Data responses for all 3 symbols, verify end-to-end flow

## Priority

This should be implemented AFTER the three-tier Claude escalation (Haiku → Sonnet → Opus) since the intermarket data will be most valuable in the Opus validation step where the senior analyst can synthesise gold structure + dollar context + yield picture into one coherent narrative assessment.
