# Manual Axis Scaling — Design

**Status:** approved (pre-implementation)
**Author:** brainstorming session 2026-04-30
**Target:** [src/App.jsx](../../../src/App.jsx) chart component

## Context

The price chart in `src/App.jsx` renders candles via raw D3 SVG. The X axis uses `d3.scaleBand()` over candle indices; the Y axis uses `d3.scaleLinear()` over price. Both axes auto-fit to the data extent (plus overlays from analysis/calibration: order blocks, liquidity sweeps, TP/SL levels) with 9% padding.

Today the only chart interaction is a crosshair driven by `onMouseMove` (~line 3480). The auto-fit behavior is fine for clean candle data but compresses candles vertically when far-away TP/SL overlays force the price domain to stretch. Users have no way to override.

## Goal

Add TradingView-style manual scaling so users can drag axes, pan, wheel-zoom, and reset back to auto-fit. Behavior matches TradingView gestures so traders don't have to relearn anything.

## Non-goals

- Replacing or wrapping D3 with a different chart library
- Persisting manual scale across sessions or page reloads
- Persisting manual scale across timeframe changes (each TF starts fresh)
- Mobile/touch gestures (desktop-only for v1)
- Interactive drawing tools (lines, fibs) — out of scope
- Multi-pane charts — out of scope

## User-facing behavior

### Y-axis scaling

- **Mousedown on the Y-axis label strip** → cursor changes to `ns-resize`
- **Drag up** → compresses price range (zoom in vertically); **drag down** → expands range (zoom out)
- **Anchor**: price under cursor at drag start stays under cursor throughout drag
- **Mouseup** → manual Y domain locks in; "⊕ Auto" button appears

### X-axis scaling

- **Mousedown on the X-axis label strip** → cursor changes to `ew-resize`
- **Drag left** → expands time range (more candles visible, narrower); **drag right** → compresses (fewer candles visible, wider)
- **Anchor**: candle under cursor stays under cursor throughout drag
- **Bounds**: visible span clamped to `[5, totalCandles]`

### Chart-body pan

- **Mousedown on chart body** → `grabbing` cursor, crosshair hidden during drag
- **Drag horizontally** → shifts visible candle range left/right by `(deltaX / bandWidth)` candles
- **Bounds**: clamped to data extent (can't pan past first/last candle)
- **No-op when X is in auto mode** (full data already visible — nothing to pan to)
- **Mouseup** → resumes crosshair

### Wheel zoom

- **Wheel over chart body** → zooms both axes by ~10% per tick
- **Shift+wheel** → zooms X only
- **Ctrl+wheel** → zooms Y only
- **Anchor**: cursor position
- `e.preventDefault()` so the page doesn't scroll

### Reset gestures

- **Double-click Y axis** → resets Y to auto-fit (X stays put)
- **Double-click X axis** → resets X to auto-fit (Y stays put)
- **"⊕ Auto" button** in chart toolbar → resets both axes; visible only when at least one axis is in manual mode

### Auto-reset triggers

- Timeframe change (15M/1H/4H/1D button click) → both axes reset
- New candle data arriving while in manual mode → manual scale stays put (does NOT auto-follow)

## Architecture

### State

Two new pieces of React state in the chart component:

```js
const [yManualDomain, setYManualDomain] = useState(null);
// null = auto-fit, [min, max] = manual price domain

const [xManualRange, setXManualRange] = useState(null);
// null = auto (all candles), [startCandleIndex, endCandleIndex] = manual visible range
// Stored as candleIndex values (stable IDs), NOT array positions
```

Plus a transient drag tracker (ref, not state — to avoid re-renders during drag):

```js
const dragStateRef = useRef(null);
// During a drag:
// {
//   kind: 'y-axis' | 'x-axis' | 'pan',
//   startMouseX, startMouseY,
//   startDomain,         // y-axis or pan: snapshot of yManualDomain at drag start
//   startRange,          // x-axis or pan: snapshot of xManualRange at drag start
//   anchorPrice,         // y-axis: price under cursor at mousedown
//   anchorIndex,         // x-axis: candleIndex under cursor at mousedown
// }
```

### Domain resolution

In the chart render block (~line 944), replace the current auto-domain computation with:

```js
const yDomain = yManualDomain ?? computeAutoYDomain(candles, overlays); // existing logic
const visibleCandles = xManualRange
  ? candles.filter(c => c.candleIndex >= xManualRange[0] && c.candleIndex <= xManualRange[1])
  : candles;
const xDomain = visibleCandles.map(c => c.candleIndex);
```

The existing `chartScalesRef.current` write (~line 1158) keeps the same shape; the crosshair handler at line 3480 doesn't need changes since it reads from this ref.

### Reset on TF change

```js
useEffect(() => {
  setYManualDomain(null);
  setXManualRange(null);
}, [timeframe]);
```

### Hit areas

Three transparent SVG `<rect>` elements with explicit event handlers:

| Hit area | Position | Cursor on hover | Events |
|---|---|---|---|
| Y-axis strip | x: chartWidth, width: yLabelWidth (~50px), full height | `ns-resize` | mousedown, dblclick |
| X-axis strip | y: chartHeight, height: xLabelHeight (~24px), full width | `ew-resize` | mousedown, dblclick |
| Chart body | existing area | `crosshair` (default) or `grabbing` during pan | mousedown, wheel, mousemove (existing crosshair) |

### Drag lifecycle

On `mousedown` over a hit area:
1. Capture `dragStateRef.current` with the start state
2. Attach window-level `mousemove` and `mouseup` handlers (so drag continues if cursor leaves SVG)
3. Set crosshair-visible flag to false (for body pan)

On window `mousemove`:
1. Read `dragStateRef.current`
2. Compute new domain/range using the math below
3. Call `setYManualDomain` or `setXManualRange`

On window `mouseup`:
1. Detach window listeners
2. Clear `dragStateRef.current`
3. Restore crosshair-visible flag

### Pure helpers

Extract a new module `src/chartScaling.js` with these pure functions (testable without DOM):

```js
export function scaleYDomain({ startDomain, anchorPrice, deltaY, chartHeight }) → [min, max]
export function scaleXRange({ startRange, anchorIndex, deltaX, chartWidth, allCandleIndices }) → [start, end]
export function panXRange({ startRange, deltaX, bandWidth, allCandleIndices }) → [start, end]
export function wheelZoom({ startYDomain, startXRange, anchorPrice, anchorIndex, deltaY, modifiers, chartHeight, chartWidth, allCandleIndices }) → { yDomain, xRange }
```

The handler in `App.jsx` is just a thin wrapper that reads `dragStateRef.current`, calls the helper, and dispatches state.

## Gesture math

### Y-axis drag (anchored at cursor price)

```
deltaY      = currentMouseY - dragStartMouseY
scaleFactor = exp(deltaY / chartHeight * 2)   // exponential = symmetric zoom feel
                                              // up = shrink (zoom in), down = expand (zoom out)

newMin = anchorPrice - (anchorPrice - startMin) * scaleFactor
newMax = anchorPrice + (startMax - anchorPrice) * scaleFactor

newSpan = newMax - newMin
clamp newSpan to [0.1, 10 * originalAutoSpan]    // sanity bounds
```

### X-axis drag (anchored at cursor candle)

```
deltaX      = currentMouseX - dragStartMouseX
scaleFactor = exp(-deltaX / chartWidth * 2)   // right = compress (fewer candles)

startSpan = startEnd - startStart + 1
newSpan   = round(startSpan * scaleFactor)
newSpan   = clamp(newSpan, 5, totalCandles)

// Keep anchorIndex at the same fractional position within the visible range
fractionFromStart = (anchorIndex - startStart) / (startSpan - 1)
newStart = round(anchorIndex - fractionFromStart * (newSpan - 1))
newStart = clamp(newStart, firstCandleIndex, lastCandleIndex - newSpan + 1)
newEnd   = newStart + newSpan - 1
```

### Pan drag (chart body)

`bandWidth` is `chartScalesRef.current.x.bandwidth()` from the existing D3 `scaleBand`. `anchorPrice` is captured at mousedown via `chartScalesRef.current.y.invert(mouseY)`; `anchorIndex` is captured by reading `chartScalesRef.current.x.domain()[Math.floor(mouseX / bandWidth)]`.

```
deltaX        = currentMouseX - dragStartMouseX
candlesShift  = round(deltaX / bandWidth)

newStart = clamp(startStart - candlesShift, firstCandleIndex, lastCandleIndex - currentSpan + 1)
newEnd   = newStart + currentSpan - 1
```

If `xManualRange` is null when pan starts (auto mode, all candles visible), pan is a no-op — there's nowhere to pan to.

### Wheel zoom

```
zoomFactor = e.deltaY > 0 ? 1.1 : 1/1.1   // ~10% per tick
// Apply Y-drag math (with implied deltaY = log(zoomFactor) * chartHeight / 2) for Y axis
// Apply X-drag math for X axis
// Modifiers:
//   Shift held → only update X
//   Ctrl held  → only update Y
//   Neither    → update both
e.preventDefault()                         // prevent page scroll
```

### Edge cases

- **Drag past sanity limits**: clamp silently (no rubber-band animation, no error)
- **Crosshair during drag**: hide tooltip and crosshair lines while `dragStateRef.current` is set
- **New candles arrive while X is manual**: storing `xManualRange` as `[startCandleIndex, endCandleIndex]` (stable IDs) means the slice naturally tracks the data — new candles past `endCandleIndex` are simply not rendered until the user pans/resets
- **No data**: when `candles.length === 0`, hit areas render but ignore mousedown
- **Single candle**: `xManualRange` lower clamp of 5 means we never end up with span < 5; if `totalCandles < 5`, X-drag is a no-op
- **Wheel without modifier**: zooms both axes — make sure both helpers are called from the same handler invocation (one render, not two)

## UI changes

### Reset button

Add a button in the chart toolbar area (where the timeframe buttons live, ~line 3319). Button:
- Label: `⊕ Auto`
- Style: matches existing `btn()` helper for consistency
- Visible only when `yManualDomain !== null || xManualRange !== null`
- onClick: `setYManualDomain(null); setXManualRange(null)`
- Tooltip: "Reset chart scale to auto-fit"

### Cursor states

Updated via inline `style={{ cursor }}` on each hit-area `<rect>`:

| Element | Default | While dragging |
|---|---|---|
| Y-axis strip | `ns-resize` | `ns-resize` |
| X-axis strip | `ew-resize` | `ew-resize` |
| Chart body | `crosshair` | `grabbing` |

## Testing

### Unit tests — `src/test/chartScaling.test.js`

Tests for pure helpers in `src/chartScaling.js`:

- `scaleYDomain`:
  - anchor price stays fixed across scale factors
  - drag up shrinks domain, drag down expands
  - clamps to min span (0.1) and max span (10× original)
  - no NaN/Infinity for zero deltaY
- `scaleXRange`:
  - anchor candle stays at same fractional position within visible range
  - drag right shrinks range (fewer candles), drag left expands
  - clamps to `[5, totalCandles]`
  - clamps `newStart` so `newEnd` doesn't exceed last candle
- `panXRange`:
  - shift left/right adjusts both ends equally
  - clamps at array bounds (can't go past first or last candle)
  - no-op when range covers full data
- `wheelZoom`:
  - plain wheel updates both axes
  - Shift modifier updates X only, leaves Y untouched
  - Ctrl modifier updates Y only, leaves X untouched
  - direction matches TradingView (wheel up = zoom in)

### Component tests — `src/test/chart-scaling.test.jsx`

Tests with React Testing Library on the chart component:
- TF change clears manual scale (mount with manual domain, change TF, assert state cleared)
- Reset button only renders when at least one axis is manual
- Reset button click clears both states
- Double-click Y axis hit area resets Y, leaves X untouched
- Double-click X axis hit area resets X, leaves Y untouched
- Body pan is no-op when X is in auto mode

### Manual verification (cannot be unit-tested)

Run `npm run dev`, open chart at 1H, exercise:
1. Drag Y axis up → candles stretch vertically; price under cursor stays put
2. Drag X axis right → fewer candles visible; candle under cursor stays put
3. Drag chart body (after zooming X) → candles shift horizontally; no jitter
4. Wheel over chart → both axes zoom toward cursor
5. Shift+wheel → only X zooms; Ctrl+wheel → only Y zooms
6. Double-click Y axis → snaps back to auto-fit
7. Switch from 1H to 4H → both axes auto-fit; "⊕ Auto" button hidden
8. Run analysis with TP/SL overlays while in manual mode → manual domain persists
9. Drag at sustained 60fps without dropped frames

If step 9 stutters, throttle drag updates with `requestAnimationFrame`.

## Files touched

| File | Change |
|---|---|
| [src/App.jsx](../../../src/App.jsx) | Add state + refs; add hit-area `<rect>`s; add event handlers; add `useEffect` for TF reset; add reset button in toolbar |
| `src/chartScaling.js` | NEW — pure helpers for domain/range math |
| `src/test/chartScaling.test.js` | NEW — unit tests for helpers |
| `src/test/chart-scaling.test.jsx` | NEW — component tests |

No backend changes. No CSS file changes (inline styles on the new SVG elements).

## Open questions

None — all behavior decisions made during brainstorming.
