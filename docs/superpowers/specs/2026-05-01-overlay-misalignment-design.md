# Overlay Misalignment — Design

**Status:** approved (pre-implementation)
**Author:** brainstorming session 2026-05-01
**Target:** [src/App.jsx](../../../src/App.jsx) live chart, [ml/server.py](../../../ml/server.py) `/calibrate`, [ml/prompts.py](../../../ml/prompts.py), [src/App.jsx](../../../src/App.jsx) inline `buildEnhancedICTPrompt`

## Context

The price chart was just shipped with manual axis scaling via [PR #5](https://github.com/Fadakaa/ict-terminal/pull/5). During verification the user noticed that on live (non-demo) data the analysis overlays — Order Blocks, Fair Value Gaps, liquidity levels — do not line up with the candles they claim to anchor to.

Concrete example from a real run (XAU/USD 1H, 95 candles, last close 4630.27): Claude's narrative summary said "the most relevant bullish OB on the 1H is the displacement candle origin at the 07:00–08:00 zone (4584-4598), with a bullish FVG between 4605 and 4614," but the rendered "BULL OB 1H ★" rectangle sat at ~4540. The `analysis.orderBlocks[i].{high, low}` values Claude returned in structured JSON simply do not agree with the candle wicks at `analysis.orderBlocks[i].candleIndex`, nor with the prose Claude wrote in the summary.

The demo path does not show this bug because [src/App.jsx:91-113](../../../src/App.jsx) uses an `inject(idx, o, h, l, c)` helper to surgically pin candles to the OHLC values that match the hardcoded analysis. Live data has no such option — candles come from OANDA and cannot be rewritten. The mismatch must be resolved by adjusting the analysis values to fit the candles instead.

The current Pydantic schema in [ml/models.py:14-31](../../../ml/models.py) accepts any numeric `high`/`low` on an `OrderBlock` or `FVG`. Both prompts ([src/App.jsx:299](../../../src/App.jsx) inline and [ml/prompts.py:202](../../../ml/prompts.py)) ask Claude to return high/low/candleIndex without any internal-consistency constraint, and nothing downstream validates that constraint either.

## Goal

Make every overlay on the live chart sit exactly on the candle wick at the index it claims, without changing how Claude is invoked or how the chart renders. The fix has two layers: a prompt instruction asking Claude to keep numeric and narrative values consistent (cheap, defense in depth), and a deterministic post-validation step that snaps any divergent OB/FVG/liquidity to the actual candle wick before it ever reaches the chart.

A small adjacent rendering bug — duplicate liquidity labels stacking on top of each other when two `analysis.liquidity` entries land at the same price — is fixed in the same pass.

## Non-goals

- Manual axis scaling, drag handlers, wheel zoom, `useChartScale`, `chartScaling.js` — already shipped in PR #5, untouched.
- Future-space rendering and `FUTURE_BARS` (main commit 594b706) — untouched.
- Demo path `inject()` candle pinning — correct as-is, untouched.
- Cross-collisions between `analysis.liquidity` labels and the `4H DR HIGH/LOW` markers (different render loops) — out of scope.
- Any chart-level UI badge or callout indicating that a snap happened. Snapping is silent in the chart; diagnostics live in console + backend response only.
- Backfilling historical `scanner_setups.analysis_json` rows that were stored before this fix shipped. The fix is forward-looking; old rows can stay as-is until a future explicit backfill task.

## User-facing behavior

After the fix lands:

- Every OB rectangle on the live chart spans exactly the wick range of the candle at `candleIndex`. No more rectangles floating $40–$80 above or below the candle they're anchored to.
- Every FVG rectangle spans exactly the gap range of its 3-candle pattern (anchor candle and the candle two indices later).
- Every liquidity line sits exactly on the wick (high for buyside, low for sellside) of the candle at `candleIndex`.
- When two or more liquidity entries cluster at the same price-level (within $0.50, same type and tf), the chart shows a single label tagged `×N` instead of N labels stacked on top of each other.
- If Claude returns an OB/FVG/liquidity entry with a `candleIndex` outside the visible candle window, the entry is silently dropped (rather than clamped onto the wrong candle).
- Console shows a `console.warn` whenever a snap or drop happens, with the kind, the claimed value, the snapped value, and the delta. Quiet otherwise.
- The `/calibrate` response gains a `snap_diagnostics` field. No UI hookup in this iteration; the data path is wired so a future ML-tab surface can consume it.

## Architecture

### Snap algorithm (canonical contract)

Both languages implement the same algorithm. Tolerance is $0.50 — matches the existing "close enough" convention in [src/App.jsx:1151](../../../src/App.jsx) and [src/App.jsx:1184](../../../src/App.jsx).

For each `analysis.orderBlocks[i]`:

- Resolve `c = candles[ob.candleIndex]`.
- Drop the item (do not include in output) if `candleIndex` is missing, negative, or `>= len(candles)`.
- If `|ob.high - c.high| > 0.50` or `|ob.low - c.low| > 0.50`, replace both with `c.high` and `c.low`, set `ob.snapped = true`, record a delta in diagnostics.
- Otherwise pass the OB through unchanged.

For each `analysis.fvgs[i]`:

- Resolve `c0 = candles[startIndex]`, `c2 = candles[startIndex + 2]`.
- Drop if `startIndex` is missing/negative or `startIndex + 2 >= len(candles)`.
- Compute the expected gap range: bullish FVG → `(low, high) = (c0.high, c2.low)`; bearish FVG → `(high, low) = (c0.low, c2.high)`.
- If `expected_low >= expected_high`, drop the FVG (degenerate gap — Claude misidentified the anchor).
- If `|fvg.high - expected_high| > 0.50` or `|fvg.low - expected_low| > 0.50`, replace both, set `fvg.snapped = true`, record delta.
- Otherwise pass through.

For each `analysis.liquidity[i]`:

- Resolve `c = candles[liq.candleIndex]`.
- Drop if `candleIndex` is missing/negative or `>= len(candles)`.
- Compute the expected price: `c.high` for buyside, `c.low` for sellside.
- If `|liq.price - expected| > 0.50`, replace, set `liq.snapped = true`, record delta.
- Otherwise pass through.

### Diagnostics

Per-item: a `snapped: true` flag is added to any item whose values were replaced. Available downstream if a future visual indicator wants to consume it; the chart render layer ignores it for now.

Aggregate: a single `diagnostics` object is returned alongside the snapped analysis with shape:

```json
{
  "snapped_obs": 0, "dropped_obs": 0,
  "snapped_fvgs": 0, "dropped_fvgs": 0,
  "snapped_liquidity": 0, "dropped_liquidity": 0,
  "deltas": [
    { "kind": "ob", "candleIndex": 42, "claimed": { "high": 4540.0, "low": 4520.0 },
      "snapped": { "high": 4598.5, "low": 4584.2 } }
  ]
}
```

The frontend logs the diagnostics to `console.warn` only when at least one snap or drop occurred (no spam in the happy path). The backend logs to `logger.warning` under the same condition. The backend additionally returns the diagnostics in the `/calibrate` response under `snap_diagnostics`.

### Where the snap fires

**Frontend (live UI flow):** [src/App.jsx](../../../src/App.jsx) `runAnalysis` already does:

```js
const parsed = JSON.parse(clean);
analysisCacheRef.current = { hash, result: parsed };
setAnalysis(parsed);
```

That becomes:

```js
const raw = JSON.parse(clean);
const { analysis: parsed, diagnostics } = snapAnalysisToCandles(raw, cds);
if (diagnostics.snapped_obs || diagnostics.snapped_fvgs || diagnostics.snapped_liquidity ||
    diagnostics.dropped_obs || diagnostics.dropped_fvgs || diagnostics.dropped_liquidity) {
  console.warn("[analysis] overlay snap diagnostics:", diagnostics);
}
analysisCacheRef.current = { hash, result: parsed };
setAnalysis(parsed);
```

The snapped analysis is what gets cached and stored in state, so `runCalibration` later sends an already-snapped analysis to `/calibrate`. The chart render loop is unchanged — it still does `y(ob.high)` etc., only now `ob.high` is correct.

**Backend (defensive):** [ml/server.py:798-802](../../../ml/server.py) `/calibrate` does:

```py
bridge = ClaudeAnalysisBridge()
parsed = bridge.parse_analysis(analysis, candles)
calibrator = MLCalibrator()
result = calibrator.calibrate_trade(parsed, candles)
return result
```

That becomes:

```py
analysis, snap_diagnostics = snap_analysis_to_candles(analysis, candles)
if snap_diagnostics["snapped_obs"] or snap_diagnostics["snapped_fvgs"] or snap_diagnostics["snapped_liquidity"] \
        or snap_diagnostics["dropped_obs"] or snap_diagnostics["dropped_fvgs"] or snap_diagnostics["dropped_liquidity"]:
    logger.warning("overlay snap diagnostics: %s", snap_diagnostics)

bridge = ClaudeAnalysisBridge()
parsed = bridge.parse_analysis(analysis, candles)
calibrator = MLCalibrator()
result = calibrator.calibrate_trade(parsed, candles)
result["snap_diagnostics"] = snap_diagnostics
return result
```

In the live UI flow the analysis is already snapped client-side, so the backend snap is a no-op for that path. **Note:** the scanner pipeline (`Scanner._calibrate` in `ml/scanner.py`) calls `MLCalibrator.calibrate_trade` directly in-process — it does NOT go through the HTTP `/calibrate` endpoint. As a result, the backend snap added here only protects the live UI flow's calibration call. Snapping the scanner path is a separate task; tracked in the open issues section below.

### Liquidity dedup

Inside [src/App.jsx:1089](../../../src/App.jsx) the existing render loop is:

```js
analysis.liquidity?.forEach((liq) => { /* draw line + label */ });
```

It becomes:

```js
const groups = groupLiquidityByLevel(analysis.liquidity || [], 0.50);
groups.forEach((group) => {
  const liq = group.items[0];           // representative
  const count = group.items.length;
  // draw line at y(liq.price) — once per group, not once per item
  // label = `${prefix}${tfTag}${count > 1 ? ` ×${count}` : ""}`
});
```

`groupLiquidityByLevel` is a small pure helper exported from [src/analysisSnap.js](../../../src/analysisSnap.js). Items are grouped by `(type, round(price / 0.50) * 0.50, tf)`. Within a group, `candleIndex` is taken from the leftmost item so the line still anchors at the earliest candle.

The line is drawn once per group (not once per item) so we don't paint the same line twice; only the label gets the `×N` suffix.

### Prompt tightening (Approach A)

Both prompts gain a new framework item, identical text in both:

> 12. CRITICAL CONSISTENCY: Numeric fields you return MUST match the actual candle data, not paraphrased values from your prose. For each `orderBlocks[i]`: `high` and `low` MUST equal the actual high and low of the candle at `candleIndex`. For each `fvgs[i]`: bullish → `low = candle[startIndex].high`, `high = candle[startIndex+2].low`; bearish → `high = candle[startIndex].low`, `low = candle[startIndex+2].high`. For each `liquidity[i]`: `price` MUST equal `candle[candleIndex].high` (buyside) or `candle[candleIndex].low` (sellside). Do not round, paraphrase, or use values from a non-anchor candle. Mismatches will be silently corrected, but they signal you got the anchor candle wrong.

This is purely belt-and-suspenders. The snap is the load-bearing fix; the prompt addition is so Claude has the chance to get it right on the first pass and so the diagnostics surface real divergences when they happen.

## Module structure

### New files

- **[src/analysisSnap.js](../../../src/analysisSnap.js)** — pure-function module. Exports:
  - `snapAnalysisToCandles(analysis, candles, options?)` returning `{ analysis: snappedAnalysis, diagnostics }`. Default `options.tolerance = 0.50`.
  - `groupLiquidityByLevel(liquidity, tolerance)` returning `[{ items: [...] }, ...]`.
  - No JSX, no imports of React or D3 — keeps it trivially unit-testable in vitest.
- **[src/test/analysisSnap.test.js](../../../src/test/analysisSnap.test.js)** — vitest tests (cases listed below).
- **[ml/analysis_snap.py](../../../ml/analysis_snap.py)** — pure Python module. Exports:
  - `snap_analysis_to_candles(analysis: dict, candles: list[dict], tolerance: float = 0.50) -> tuple[dict, dict]` returning `(snapped_analysis, diagnostics)`.
  - No I/O, no DB, no logging — pure compute. Calling code logs.
- **[ml/tests/test_analysis_snap.py](../../../ml/tests/test_analysis_snap.py)** — pytest tests.

### Modified files

- **[src/App.jsx](../../../src/App.jsx)**:
  - Top of file: add `import { snapAnalysisToCandles, groupLiquidityByLevel } from "./analysisSnap.js";`.
  - Inline prompt builder `buildEnhancedICTPrompt` (~line 250): add the Approach-A framework item before the `Return ONLY valid JSON:` line.
  - `runAnalysis` (~line 638-640): wrap the parsed JSON through `snapAnalysisToCandles` before caching and `setAnalysis`. Conditional `console.warn` on diagnostics.
  - Liquidity render loop (~line 1089): replace `forEach` with the grouped iteration.
- **[ml/prompts.py](../../../ml/prompts.py)** (~line 188): add the Approach-A framework item as numbered point 12.
- **[ml/server.py](../../../ml/server.py)** (~line 777-804): import and call `snap_analysis_to_candles` at the top of the `/calibrate` handler. Attach `snap_diagnostics` to response. Conditional `logger.warning`.

## Testing

### Frontend (vitest)

`src/test/analysisSnap.test.js`:

- `snapAnalysisToCandles` — OB cases:
  - both high/low diverge by >$0.50 → snap to candle wick, `ob.snapped === true`, `diagnostics.snapped_obs === 1`
  - both within tolerance → unchanged, `diagnostics.snapped_obs === 0`
  - partial: high in tolerance, low out → both replaced (canonical wick is always atomic)
  - missing `candleIndex` → dropped, `diagnostics.dropped_obs === 1`
  - negative `candleIndex` → dropped
  - `candleIndex >= candles.length` → dropped
- FVG cases:
  - bullish, `startIndex+2` valid, diverged → snaps to `(c0.high, c2.low)`
  - bearish, diverged → snaps to `(c0.low, c2.high)`
  - degenerate gap (expected_low >= expected_high) → dropped, `diagnostics.dropped_fvgs === 1`
  - `startIndex+2 >= candles.length` → dropped
  - within tolerance → unchanged
- Liquidity cases:
  - buyside, price diverges from `c.high` by >$0.50 → snaps to `c.high`
  - sellside, price diverges from `c.low` → snaps to `c.low`
  - within tolerance → unchanged
  - OOB candleIndex → dropped
- Round-trip: feed the existing demo `analysis` and demo `candles` through the snapper → zero snaps, zero drops, output deeply equal to input. Idempotency check.
- `groupLiquidityByLevel`:
  - two BSLs at $4630.30 and $4630.50 with same tf → one group of 2
  - two BSLs same price but different tf → two groups
  - one BSL one SSL same price → two groups (different type)
  - empty input → empty output

### Backend (pytest)

`ml/tests/test_analysis_snap.py`: mirror every frontend case, plus:

- Integration: `TestClient` POST to `/calibrate` with an analysis whose first OB has `high`/`low` diverged by $50 from the candle wick. Response includes `snap_diagnostics.snapped_obs == 1`. Spying on `ClaudeAnalysisBridge.parse_analysis` confirms it was called with the snapped (corrected) high/low — not the original — so calibration math is computed against the corrected anchor.
- Demo round-trip: construct an analysis matching demo `candles` inline (or vendor a snapshot under `ml/tests/fixtures/` if it grows large) and snap → zero changes.

### What's not tested

The App.jsx render loop change for dedup is not directly tested — it's a forEach swap and the existing chart-render code has no test harness. The dedup *grouping* logic is in `groupLiquidityByLevel` and is fully unit-tested in vitest. The render-loop change is a 5-line wiring edit that we'll verify by manual inspection on the live chart.

## Verification (manual)

After implementation:

1. `npm run test` — all vitest cases pass including the new analysis-snap suite.
2. `python -m pytest ml/tests/test_analysis_snap.py -v` — all backend cases pass.
3. `python -m pytest ml/tests/ -v` — full suite still passes (~1280 tests + new ones, no regressions).
4. `npm run build && npm run dev` (or whichever serves the SPA against the FastAPI backend), open the live XAU/USD 1H chart, run an analysis on real OANDA candles. Confirm:
   - Every OB rectangle visibly sits on the wick of the candle at its `candleIndex`.
   - Every FVG rectangle visibly spans the gap of its 3-candle pattern.
   - Every liquidity line sits on the wick (top for BSL, bottom for SSL).
   - If two BSLs cluster, a single `BSL 1H ×2` label is rendered, no stacking.
   - If snaps occurred, `console.warn` shows the diagnostics.

## Open issues / future work

- **Backend snap does NOT cover the scanner path.** `Scanner._calibrate` invokes `MLCalibrator.calibrate_trade` directly via in-process import, bypassing the HTTP `/calibrate` endpoint. The backend snap added in this PR only protects the live UI's calibration call. Adding `snap_analysis_to_candles` directly inside `Scanner._calibrate` (or at the analysis-storage boundary) is a follow-up task; until then, scanner-stored `scanner_setups.analysis_json` rows continue to contain unsnapped values.
- **Cross-collision label cleanup.** The user reported `BSL 1H` colliding visually with `4H DR HIGH` labels at the right edge. That's a separate render loop (dealing range markers at [src/App.jsx:1106-1114](../../../src/App.jsx)) and a separate fix — punted for a future task.
- **Visual snap indicator.** We deliberately do not add a chart-side badge today. If users start asking "did Claude get this anchor right?" we can revisit and use the per-item `snapped: true` flag we're already setting.
