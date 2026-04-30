# Manual Axis Scaling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add TradingView-style manual scaling to the price chart in `src/App.jsx`: drag axes to scale, drag body to pan, wheel to zoom, double-click axis to reset, "⊕ Auto" toolbar button to reset both.

**Architecture:** Two new modules — `src/chartScaling.js` (pure math) and `src/useChartScale.js` (React state + TF reset). `src/App.jsx` consumes the hook, derives domains from manual state with auto-fit fallback, and attaches drag/wheel/dblclick handlers via a mix of React (chart body, wheel) and D3 (axis hit-area `<rect>`s painted in `drawChart`).

**Tech Stack:** React 18, D3 7 (existing), Vitest + jsdom + React Testing Library (existing).

**Spec:** [docs/superpowers/specs/2026-04-30-manual-axis-scaling-design.md](../specs/2026-04-30-manual-axis-scaling-design.md)

---

## File Structure

```
src/
├── chartScaling.js              [NEW] 4 pure helpers
├── useChartScale.js             [NEW] React hook
├── App.jsx                      [MODIFY] integrate hook + handlers
└── test/
    ├── chartScaling.test.js     [NEW] helper unit tests
    └── useChartScale.test.js    [NEW] hook tests
```

**Boundary justification:** Pure math (no React) lives in `chartScaling.js` so it tests cleanly without DOM. State + lifecycle live in `useChartScale.js` as a hook so it tests with `renderHook`. The 3592-line `App.jsx` only gets minimal additions (no extraction needed) — wiring the hook, using its values in the existing `drawChart`, and adding event handlers.

**Note on spec deviation:** The spec mentioned `chart-scaling.test.jsx` for component tests. Mounting the full `App.jsx` (3592 lines, fetch effects, scheduler effects) for a unit test is impractical. We test state behavior via the hook instead — same coverage of the spec's testable invariants (TF reset, manual flag, reset action) without the mounting overhead.

---

## Task 1: Skeleton — `chartScaling.js` with `scaleYDomain` (TDD)

**Files:**
- Create: `src/chartScaling.js`
- Create: `src/test/chartScaling.test.js`

- [ ] **Step 1: Write the failing test for `scaleYDomain`**

Create `src/test/chartScaling.test.js`:

```js
import { describe, it, expect } from "vitest";
import { scaleYDomain } from "../chartScaling.js";

describe("scaleYDomain", () => {
  const base = { startDomain: [100, 200], anchorPrice: 150, chartHeight: 400 };

  it("returns startDomain unchanged when deltaY is 0", () => {
    const [min, max] = scaleYDomain({ ...base, deltaY: 0 });
    expect(min).toBeCloseTo(100);
    expect(max).toBeCloseTo(200);
  });

  it("dragging up shrinks the domain (zoom in)", () => {
    const [min, max] = scaleYDomain({ ...base, deltaY: -100 });
    expect(max - min).toBeLessThan(100);
  });

  it("dragging down expands the domain (zoom out)", () => {
    const [min, max] = scaleYDomain({ ...base, deltaY: 100 });
    expect(max - min).toBeGreaterThan(100);
  });

  it("anchor price stays fixed across scale factors", () => {
    // Anchor at top of domain
    const top = scaleYDomain({ ...base, anchorPrice: 200, deltaY: -100 });
    expect(top[1]).toBeCloseTo(200);
    // Anchor at bottom of domain
    const bottom = scaleYDomain({ ...base, anchorPrice: 100, deltaY: -100 });
    expect(bottom[0]).toBeCloseTo(100);
    // Anchor in middle
    const mid = scaleYDomain({ ...base, anchorPrice: 150, deltaY: -100 });
    const midpoint = (mid[0] + mid[1]) / 2;
    expect(midpoint).toBeCloseTo(150);
  });

  it("clamps span to [0.1, 10 * original]", () => {
    const tiny = scaleYDomain({ ...base, deltaY: -10000 });
    expect(tiny[1] - tiny[0]).toBeGreaterThanOrEqual(0.1);
    const huge = scaleYDomain({ ...base, deltaY: 10000 });
    expect(huge[1] - huge[0]).toBeLessThanOrEqual(1000); // 10 * 100
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run src/test/chartScaling.test.js`

Expected: FAIL — `scaleYDomain is not a function` or "Cannot find module".

- [ ] **Step 3: Write minimal implementation**

Create `src/chartScaling.js`:

```js
export function scaleYDomain({ startDomain, anchorPrice, deltaY, chartHeight }) {
  const [startMin, startMax] = startDomain;
  const startSpan = startMax - startMin;
  const scaleFactor = Math.exp((deltaY / chartHeight) * 2);

  let newMin = anchorPrice - (anchorPrice - startMin) * scaleFactor;
  let newMax = anchorPrice + (startMax - anchorPrice) * scaleFactor;

  const newSpan = newMax - newMin;
  const minSpan = 0.1;
  const maxSpan = startSpan * 10;

  if (newSpan < minSpan) {
    const center = (newMin + newMax) / 2;
    newMin = center - minSpan / 2;
    newMax = center + minSpan / 2;
  } else if (newSpan > maxSpan) {
    const center = (newMin + newMax) / 2;
    newMin = center - maxSpan / 2;
    newMax = center + maxSpan / 2;
  }

  return [newMin, newMax];
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run src/test/chartScaling.test.js`

Expected: PASS — all 5 tests for `scaleYDomain` pass.

- [ ] **Step 5: Commit**

```bash
git add src/chartScaling.js src/test/chartScaling.test.js
git commit -m "feat: add scaleYDomain helper for manual Y-axis scaling"
```

---

## Task 2: `scaleXRange` helper (TDD)

**Files:**
- Modify: `src/chartScaling.js` — add `scaleXRange`
- Modify: `src/test/chartScaling.test.js` — add `scaleXRange` tests

- [ ] **Step 1: Write the failing tests**

Append to `src/test/chartScaling.test.js`:

```js
import { scaleXRange } from "../chartScaling.js";

describe("scaleXRange", () => {
  const allIndices = Array.from({ length: 60 }, (_, i) => i); // candleIndex 0..59
  const base = { startRange: [0, 59], anchorIndex: 30, chartWidth: 800, allCandleIndices: allIndices };

  it("returns startRange unchanged when deltaX is 0", () => {
    const [start, end] = scaleXRange({ ...base, deltaX: 0 });
    expect(start).toBe(0);
    expect(end).toBe(59);
  });

  it("dragging right shrinks the visible range (zoom in)", () => {
    const [start, end] = scaleXRange({ ...base, deltaX: 200 });
    expect(end - start).toBeLessThan(59);
  });

  it("dragging left expands toward full range (zoom out, capped)", () => {
    const sub = { ...base, startRange: [20, 39] };
    const [start, end] = scaleXRange({ ...sub, deltaX: -200, anchorIndex: 30 });
    expect(end - start).toBeGreaterThan(19);
  });

  it("clamps span to minimum of 5 candles", () => {
    const [start, end] = scaleXRange({ ...base, deltaX: 10000 });
    expect(end - start + 1).toBeGreaterThanOrEqual(5);
  });

  it("clamps span to total candle count", () => {
    const [start, end] = scaleXRange({ ...base, deltaX: -10000 });
    expect(end - start + 1).toBeLessThanOrEqual(60);
    expect(start).toBe(0);
    expect(end).toBe(59);
  });

  it("anchor candle stays at same fractional position", () => {
    // Anchor near the right edge: position should remain near right edge after zoom
    const [start, end] = scaleXRange({ ...base, anchorIndex: 50, deltaX: 200 });
    const span = end - start + 1;
    const fraction = (50 - start) / (span - 1);
    expect(fraction).toBeCloseTo((50 - 0) / (59 - 0), 1);
  });

  it("clamps newStart so newEnd does not exceed last candle", () => {
    // Anchor near right edge, zoom in (smaller span) — start should shift left, not push end past 59
    const [start, end] = scaleXRange({ ...base, anchorIndex: 58, deltaX: 200 });
    expect(end).toBeLessThanOrEqual(59);
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npx vitest run src/test/chartScaling.test.js`

Expected: FAIL — `scaleXRange is not a function`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/chartScaling.js`:

```js
export function scaleXRange({ startRange, anchorIndex, deltaX, chartWidth, allCandleIndices }) {
  const [startStart, startEnd] = startRange;
  const startSpan = startEnd - startStart + 1;
  const scaleFactor = Math.exp((-deltaX / chartWidth) * 2);

  const totalCandles = allCandleIndices.length;
  const minSpan = Math.min(5, totalCandles);
  const maxSpan = totalCandles;

  let newSpan = Math.round(startSpan * scaleFactor);
  newSpan = Math.max(minSpan, Math.min(maxSpan, newSpan));

  const fractionFromStart = startSpan === 1 ? 0 : (anchorIndex - startStart) / (startSpan - 1);
  let newStart = Math.round(anchorIndex - fractionFromStart * (newSpan - 1));

  const firstIdx = allCandleIndices[0];
  const lastIdx = allCandleIndices[allCandleIndices.length - 1];
  newStart = Math.max(firstIdx, Math.min(lastIdx - newSpan + 1, newStart));
  const newEnd = newStart + newSpan - 1;

  return [newStart, newEnd];
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `npx vitest run src/test/chartScaling.test.js`

Expected: PASS — all `scaleYDomain` and `scaleXRange` tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/chartScaling.js src/test/chartScaling.test.js
git commit -m "feat: add scaleXRange helper for manual X-axis scaling"
```

---

## Task 3: `panXRange` helper (TDD)

**Files:**
- Modify: `src/chartScaling.js` — add `panXRange`
- Modify: `src/test/chartScaling.test.js` — add `panXRange` tests

- [ ] **Step 1: Write the failing tests**

Append to `src/test/chartScaling.test.js`:

```js
import { panXRange } from "../chartScaling.js";

describe("panXRange", () => {
  const allIndices = Array.from({ length: 60 }, (_, i) => i);
  const base = { startRange: [10, 29], bandWidth: 10, allCandleIndices: allIndices };

  it("returns startRange unchanged when deltaX is 0", () => {
    expect(panXRange({ ...base, deltaX: 0 })).toEqual([10, 29]);
  });

  it("dragging right shifts range left (showing earlier candles)", () => {
    const [start, end] = panXRange({ ...base, deltaX: 50 }); // 50 / 10 = 5 candles right
    expect(start).toBe(5);
    expect(end).toBe(24);
  });

  it("dragging left shifts range right (showing later candles)", () => {
    const [start, end] = panXRange({ ...base, deltaX: -50 });
    expect(start).toBe(15);
    expect(end).toBe(34);
  });

  it("clamps to first candle (cannot pan past start)", () => {
    const [start, end] = panXRange({ ...base, deltaX: 1000 });
    expect(start).toBe(0);
    expect(end).toBe(19);
  });

  it("clamps to last candle (cannot pan past end)", () => {
    const [start, end] = panXRange({ ...base, deltaX: -1000 });
    expect(start).toBe(40);
    expect(end).toBe(59);
  });

  it("preserves span size", () => {
    const [start, end] = panXRange({ ...base, deltaX: 75 });
    expect(end - start).toBe(19);
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npx vitest run src/test/chartScaling.test.js`

Expected: FAIL — `panXRange is not a function`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/chartScaling.js`:

```js
export function panXRange({ startRange, deltaX, bandWidth, allCandleIndices }) {
  const [startStart, startEnd] = startRange;
  const span = startEnd - startStart + 1;
  const candlesShift = Math.round(deltaX / bandWidth);

  const firstIdx = allCandleIndices[0];
  const lastIdx = allCandleIndices[allCandleIndices.length - 1];
  let newStart = startStart - candlesShift;
  newStart = Math.max(firstIdx, Math.min(lastIdx - span + 1, newStart));

  return [newStart, newStart + span - 1];
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `npx vitest run src/test/chartScaling.test.js`

Expected: PASS — all helper tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/chartScaling.js src/test/chartScaling.test.js
git commit -m "feat: add panXRange helper for chart-body pan"
```

---

## Task 4: `wheelZoom` helper (TDD)

**Files:**
- Modify: `src/chartScaling.js` — add `wheelZoom`
- Modify: `src/test/chartScaling.test.js` — add `wheelZoom` tests

- [ ] **Step 1: Write the failing tests**

Append to `src/test/chartScaling.test.js`:

```js
import { wheelZoom } from "../chartScaling.js";

describe("wheelZoom", () => {
  const allIndices = Array.from({ length: 60 }, (_, i) => i);
  const base = {
    startYDomain: [100, 200],
    startXRange: [0, 59],
    anchorPrice: 150,
    anchorIndex: 30,
    chartHeight: 400,
    chartWidth: 800,
    allCandleIndices: allIndices,
  };

  it("plain wheel up zooms both axes in", () => {
    const result = wheelZoom({ ...base, deltaY: -100, modifiers: {} });
    expect(result.yDomain[1] - result.yDomain[0]).toBeLessThan(100);
    expect(result.xRange[1] - result.xRange[0]).toBeLessThan(59);
  });

  it("plain wheel down zooms both axes out", () => {
    const sub = { ...base, startXRange: [20, 39] };
    const result = wheelZoom({ ...sub, deltaY: 100, modifiers: {} });
    expect(result.yDomain[1] - result.yDomain[0]).toBeGreaterThan(100);
    expect(result.xRange[1] - result.xRange[0]).toBeGreaterThanOrEqual(19);
  });

  it("Shift modifier zooms only X", () => {
    const result = wheelZoom({ ...base, deltaY: -100, modifiers: { shift: true } });
    expect(result.yDomain).toEqual(base.startYDomain);
    expect(result.xRange[1] - result.xRange[0]).toBeLessThan(59);
  });

  it("Ctrl modifier zooms only Y", () => {
    const result = wheelZoom({ ...base, deltaY: -100, modifiers: { ctrl: true } });
    expect(result.xRange).toEqual(base.startXRange);
    expect(result.yDomain[1] - result.yDomain[0]).toBeLessThan(100);
  });

  it("zero deltaY returns inputs unchanged", () => {
    const result = wheelZoom({ ...base, deltaY: 0, modifiers: {} });
    expect(result.yDomain[0]).toBeCloseTo(100);
    expect(result.yDomain[1]).toBeCloseTo(200);
    expect(result.xRange).toEqual([0, 59]);
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npx vitest run src/test/chartScaling.test.js`

Expected: FAIL — `wheelZoom is not a function`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/chartScaling.js`:

```js
export function wheelZoom({
  startYDomain,
  startXRange,
  anchorPrice,
  anchorIndex,
  deltaY,
  modifiers,
  chartHeight,
  chartWidth,
  allCandleIndices,
}) {
  if (deltaY === 0) {
    return { yDomain: startYDomain, xRange: startXRange };
  }

  const zoomDirection = deltaY > 0 ? 1 : -1;
  const equivalentDeltaY = zoomDirection * chartHeight * 0.05;
  const equivalentDeltaX = -zoomDirection * chartWidth * 0.05;

  const updateY = !modifiers?.shift;
  const updateX = !modifiers?.ctrl;

  const yDomain = updateY
    ? scaleYDomain({ startDomain: startYDomain, anchorPrice, deltaY: equivalentDeltaY, chartHeight })
    : startYDomain;

  const xRange = updateX
    ? scaleXRange({ startRange: startXRange, anchorIndex, deltaX: equivalentDeltaX, chartWidth, allCandleIndices })
    : startXRange;

  return { yDomain, xRange };
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `npx vitest run src/test/chartScaling.test.js`

Expected: PASS — all helper tests pass (~22 tests total across the four helpers).

- [ ] **Step 5: Commit**

```bash
git add src/chartScaling.js src/test/chartScaling.test.js
git commit -m "feat: add wheelZoom helper combining axis scaling with modifiers"
```

---

## Task 5: `useChartScale` hook (TDD)

**Files:**
- Create: `src/useChartScale.js`
- Create: `src/test/useChartScale.test.js`

- [ ] **Step 1: Write the failing tests**

Create `src/test/useChartScale.test.js`:

```js
import { describe, it, expect } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useChartScale } from "../useChartScale.js";

describe("useChartScale", () => {
  it("starts with null manual values (auto mode)", () => {
    const { result } = renderHook(() => useChartScale("1h"));
    expect(result.current.yManualDomain).toBeNull();
    expect(result.current.xManualRange).toBeNull();
    expect(result.current.isManual).toBe(false);
  });

  it("isManual is true when yManualDomain is set", () => {
    const { result } = renderHook(() => useChartScale("1h"));
    act(() => result.current.setYManualDomain([100, 200]));
    expect(result.current.isManual).toBe(true);
  });

  it("isManual is true when xManualRange is set", () => {
    const { result } = renderHook(() => useChartScale("1h"));
    act(() => result.current.setXManualRange([10, 30]));
    expect(result.current.isManual).toBe(true);
  });

  it("reset() clears both manual values", () => {
    const { result } = renderHook(() => useChartScale("1h"));
    act(() => {
      result.current.setYManualDomain([100, 200]);
      result.current.setXManualRange([10, 30]);
    });
    act(() => result.current.reset());
    expect(result.current.yManualDomain).toBeNull();
    expect(result.current.xManualRange).toBeNull();
    expect(result.current.isManual).toBe(false);
  });

  it("changing timeframe clears both manual values", () => {
    const { result, rerender } = renderHook(({ tf }) => useChartScale(tf), {
      initialProps: { tf: "1h" },
    });
    act(() => {
      result.current.setYManualDomain([100, 200]);
      result.current.setXManualRange([10, 30]);
    });
    rerender({ tf: "4h" });
    expect(result.current.yManualDomain).toBeNull();
    expect(result.current.xManualRange).toBeNull();
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npx vitest run src/test/useChartScale.test.js`

Expected: FAIL — `Cannot find module '../useChartScale.js'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/useChartScale.js`:

```js
import { useState, useEffect, useCallback } from "react";

export function useChartScale(timeframe) {
  const [yManualDomain, setYManualDomain] = useState(null);
  const [xManualRange, setXManualRange] = useState(null);

  useEffect(() => {
    setYManualDomain(null);
    setXManualRange(null);
  }, [timeframe]);

  const reset = useCallback(() => {
    setYManualDomain(null);
    setXManualRange(null);
  }, []);

  const isManual = yManualDomain !== null || xManualRange !== null;

  return { yManualDomain, setYManualDomain, xManualRange, setXManualRange, reset, isManual };
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `npx vitest run src/test/useChartScale.test.js`

Expected: PASS — all 5 hook tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/useChartScale.js src/test/useChartScale.test.js
git commit -m "feat: add useChartScale hook for manual scale state with TF reset"
```

---

## Task 6: Wire `useChartScale` into App.jsx + add drag ref

**Files:**
- Modify: `src/App.jsx` — import hook, call it, add `dragStateRef`

- [ ] **Step 1: Locate the chart component imports**

Run: `grep -n "^import" src/App.jsx | head -20`

Expected output includes `import { useState, useEffect, useRef, useCallback, useMemo } from "react";` (or similar). Note the React import line number.

- [ ] **Step 2: Add hook import**

Locate the imports block at the top of `src/App.jsx`. After the existing local imports (e.g., `import { ... } from "./market.js";`), add:

```js
import { useChartScale } from "./useChartScale.js";
```

- [ ] **Step 3: Find the chart component's existing state declarations**

Run: `grep -n "useState" src/App.jsx | head -30`

Locate the block where `timeframe`, `candles`, `analysis`, `calibration` are declared with `useState`. The hook needs to be called *after* `timeframe` exists.

- [ ] **Step 4: Add the hook call and drag ref**

In `src/App.jsx`, immediately after the line that declares `timeframe` state (e.g., `const [timeframe, setTimeframe] = useState(...)`), add:

```js
const {
  yManualDomain,
  setYManualDomain,
  xManualRange,
  setXManualRange,
  reset: resetChartScale,
  isManual: isChartManual,
} = useChartScale(timeframe);
const dragStateRef = useRef(null);
```

- [ ] **Step 5: Run tests to make sure nothing broke**

Run: `npm run test`

Expected: PASS — no existing tests should be affected, helper + hook tests still pass.

- [ ] **Step 6: Commit**

```bash
git add src/App.jsx
git commit -m "feat: wire useChartScale hook into App.jsx"
```

---

## Task 7: Use manual domain in `drawChart`

**Files:**
- Modify: `src/App.jsx` — `drawChart` callback (~line 928–1182)

- [ ] **Step 1: Locate the auto-domain code**

Read `src/App.jsx` lines 940–965 to confirm the current shape of:

```js
const x = d3.scaleBand().domain(candles.map((_, i) => i)).range([0, w]).padding(0.22);
// ...
const allP = candles.flatMap((c) => [c.high, c.low]);
// ...
const [mn, mx] = d3.extent(allP);
const pad = (mx - mn) * 0.09;
const y = d3.scaleLinear().domain([mn - pad, mx + pad]).range([h, 0]);
```

- [ ] **Step 2: Replace the X scale to honor `xManualRange`**

Replace the existing `const x = d3.scaleBand()...` line (around line 941) with:

```js
const visibleArrayIndices = xManualRange
  ? candles.reduce((acc, c, i) => {
      if (c.candleIndex >= xManualRange[0] && c.candleIndex <= xManualRange[1]) acc.push(i);
      return acc;
    }, [])
  : candles.map((_, i) => i);
const x = d3.scaleBand().domain(visibleArrayIndices).range([0, w]).padding(0.22);
```

This narrows the scale's domain to only the visible candles (by array index). Overlay rendering further down uses `x(ci)` which will return `undefined` for hidden candles — that's already handled by `?? 0` in the existing code.

- [ ] **Step 3: Replace the Y scale to honor `yManualDomain`**

Replace the existing `const y = d3.scaleLinear()...` line (around line 964) with:

```js
const yDomainAuto = [mn - pad, mx + pad];
const y = d3.scaleLinear().domain(yManualDomain ?? yDomainAuto).range([h, 0]);
```

- [ ] **Step 4: Guard candlestick rendering for hidden candles**

When `xManualRange` is set, candles outside the range have `x(i) === undefined`, leading to `NaN` SVG attributes. Locate the existing candlestick loop (around line 1136):

```js
candles.forEach((c, i) => {
  const cx = x(i);
  const bw = x.bandwidth();
  const bull = c.close >= c.open;
  ...
});
```

Add an early return after the `cx` line:

```js
candles.forEach((c, i) => {
  const cx = x(i);
  if (cx === undefined) return;
  const bw = x.bandwidth();
  const bull = c.close >= c.open;
  ...
});
```

- [ ] **Step 5: Compute axis ticks from visible candles**

Locate the X-axis tick computation (around line 1161):

```js
const ticks = [0, Math.floor(candles.length * 0.25), Math.floor(candles.length * 0.5), Math.floor(candles.length * 0.75), candles.length - 1];
```

Replace with a version that picks ticks from `visibleArrayIndices`:

```js
const tickPositions = [0, 0.25, 0.5, 0.75, 1];
const ticks = tickPositions.map((p) => visibleArrayIndices[Math.min(visibleArrayIndices.length - 1, Math.floor(p * (visibleArrayIndices.length - 1)))]);
```

- [ ] **Step 6: Update the `drawChart` callback's dependency array**

Find the closing of the `useCallback` for `drawChart`. The current line (around line 1182) reads:

```js
}, [candles, analysis, calibration]);
```

Change it to:

```js
}, [candles, analysis, calibration, yManualDomain, xManualRange]);
```

- [ ] **Step 7: Run tests + start dev server to verify chart still renders**

Run: `npm run test`

Expected: PASS — no test regressions.

Run: `npm run dev` in another terminal. Open the app, click "LAUNCH DEMO MODE" or load data. The chart should render exactly as before (auto-fit, all candles visible). No visual change yet.

Stop dev server.

- [ ] **Step 8: Commit**

```bash
git add src/App.jsx
git commit -m "feat: use manual domain/range in chart render with auto-fit fallback"
```

---

## Task 8: Y-axis hit area + drag + double-click reset

**Files:**
- Modify: `src/App.jsx` — `drawChart` callback (axes section near line 1175); add `startDrag` handler

- [ ] **Step 1: Add `startDrag` import for helpers**

Add to the imports block (near where `useChartScale` was added in Task 6):

```js
import { scaleYDomain, scaleXRange, panXRange, wheelZoom } from "./chartScaling.js";
```

- [ ] **Step 2: Define `startDrag` callback in the component**

In `src/App.jsx`, after the `drawChart` `useCallback` ends (after the `}, [...]);` around line 1182), add this new callback:

```js
const startDrag = useCallback((kind, evt) => {
  const s = chartScalesRef.current;
  if (!s || !candles.length) return;

  const rect = svgRef.current.getBoundingClientRect();
  const startMouseX = evt.clientX - rect.left - s.m.left;
  const startMouseY = evt.clientY - rect.top - s.m.top;

  const allCandleIndices = candles.map((c) => c.candleIndex);
  const firstIdx = allCandleIndices[0];
  const lastIdx = allCandleIndices[allCandleIndices.length - 1];

  let anchorPrice = null;
  let anchorIndex = null;
  if (kind === "y-axis") {
    anchorPrice = s.y.invert(startMouseY);
  } else if (kind === "x-axis" || kind === "pan") {
    const step = s.x.step();
    const arrIdx = Math.max(
      0,
      Math.min(Math.round((startMouseX - s.x.bandwidth() / 2) / step), candles.length - 1)
    );
    anchorIndex = candles[arrIdx]?.candleIndex ?? firstIdx;
  }

  if (kind === "pan" && xManualRange === null) return; // no-op in auto mode

  dragStateRef.current = {
    kind,
    startMouseX,
    startMouseY,
    startYDomain: yManualDomain ?? [s.y.domain()[0], s.y.domain()[1]],
    startXRange: xManualRange ?? [firstIdx, lastIdx],
    anchorPrice,
    anchorIndex,
    chartHeight: s.h,
    chartWidth: s.w,
    bandWidth: s.x.bandwidth(),
    allCandleIndices,
    rectLeft: rect.left + s.m.left,
    rectTop: rect.top + s.m.top,
  };

  const handleMove = (e) => {
    const ds = dragStateRef.current;
    if (!ds) return;
    const dx = e.clientX - ds.rectLeft - ds.startMouseX;
    const dy = e.clientY - ds.rectTop - ds.startMouseY;

    if (ds.kind === "y-axis") {
      setYManualDomain(
        scaleYDomain({
          startDomain: ds.startYDomain,
          anchorPrice: ds.anchorPrice,
          deltaY: dy,
          chartHeight: ds.chartHeight,
        })
      );
    } else if (ds.kind === "x-axis") {
      setXManualRange(
        scaleXRange({
          startRange: ds.startXRange,
          anchorIndex: ds.anchorIndex,
          deltaX: dx,
          chartWidth: ds.chartWidth,
          allCandleIndices: ds.allCandleIndices,
        })
      );
    } else if (ds.kind === "pan") {
      setXManualRange(
        panXRange({
          startRange: ds.startXRange,
          deltaX: dx,
          bandWidth: ds.bandWidth,
          allCandleIndices: ds.allCandleIndices,
        })
      );
    }
  };

  const handleUp = () => {
    window.removeEventListener("mousemove", handleMove);
    window.removeEventListener("mouseup", handleUp);
    dragStateRef.current = null;
    if (svgRef.current) {
      d3.select(svgRef.current).select(".crosshair").style("display", "none");
    }
  };

  window.addEventListener("mousemove", handleMove);
  window.addEventListener("mouseup", handleUp);
}, [candles, yManualDomain, xManualRange, setYManualDomain, setXManualRange]);
```

- [ ] **Step 3: Add a ref for `startDrag` so D3 closures stay current**

Because `drawChart` is a `useCallback` whose D3 event closures may capture a stale `startDrag`, route the call through a ref that always points to the latest function. Immediately after the `startDrag` definition added in Step 2, add:

```js
const startDragRef = useRef(startDrag);
useEffect(() => { startDragRef.current = startDrag; }, [startDrag]);
```

- [ ] **Step 4: Add Y-axis hit-area `<rect>` in `drawChart`**

Locate the Y-axis rendering block in the `drawChart` callback (around line 1175):

```js
g.append("g")
  .call(d3.axisLeft(y).ticks(7).tickFormat((d) => d.toFixed(0)))
  .call((ax) => {
    ax.select(".domain").attr("stroke", "#1a1a2e");
    ax.selectAll(".tick line").attr("stroke", "#1a1a2e");
    ax.selectAll("text").attr("fill", "#444466").attr("font-size", "9px").attr("font-family", "monospace");
  });
```

Immediately after this block, add:

```js
g.append("rect")
  .attr("class", "y-axis-hit")
  .attr("x", -m.left)
  .attr("y", 0)
  .attr("width", m.left)
  .attr("height", h)
  .attr("fill", "transparent")
  .style("cursor", "ns-resize")
  .on("mousedown", (evt) => startDragRef.current?.("y-axis", evt))
  .on("dblclick", () => setYManualDomain(null));
```

- [ ] **Step 5: Run dev server and verify Y-axis drag**

Run: `npm run dev`

Open the app, load data (demo mode is fine). Verify:
- Cursor changes to `ns-resize` over the price labels
- Drag up: candles stretch vertically (price range compresses)
- Drag down: candles compress vertically (price range expands)
- Price under cursor at drag start stays roughly under cursor while dragging
- Mouseup leaves the manual scale in place
- Double-click on price labels resets to auto-fit (candles snap back to their original size)

Stop dev server.

- [ ] **Step 6: Commit**

```bash
git add src/App.jsx
git commit -m "feat: Y-axis drag-to-scale + double-click reset"
```

---

## Task 9: X-axis hit area + drag + double-click reset

**Files:**
- Modify: `src/App.jsx` — `drawChart` callback (axes section near line 1162)

- [ ] **Step 1: Locate the X-axis rendering block**

Read `src/App.jsx` lines 1161–1174 to confirm the X-axis section:

```js
const ticks = [0, Math.floor(candles.length * 0.25), ...];
g.append("g").attr("transform", `translate(0,${h})`)
  .call(d3.axisBottom(x).tickValues(ticks).tickFormat(...))
  .call((ax) => { ... });
g.append("text").attr("x", w).attr("y", h + 22)...text("GMT");
```

- [ ] **Step 2: Add X-axis hit-area `<rect>`**

Immediately after the `g.append("text")...text("GMT");` line (around line 1174), add:

```js
g.append("rect")
  .attr("class", "x-axis-hit")
  .attr("x", 0)
  .attr("y", h)
  .attr("width", w + m.right)
  .attr("height", m.bottom)
  .attr("fill", "transparent")
  .style("cursor", "ew-resize")
  .on("mousedown", (evt) => startDragRef.current?.("x-axis", evt))
  .on("dblclick", () => setXManualRange(null));
```

- [ ] **Step 3: Run dev server and verify X-axis drag**

Run: `npm run dev`

Open the app, load data. Verify:
- Cursor changes to `ew-resize` over the time labels at the bottom
- Drag right: fewer candles visible, candles get wider
- Drag left: more candles visible (capped at full data), candles get narrower
- Candle under cursor at drag start stays roughly under cursor
- Double-click on time labels resets X to auto (all candles visible again)

Stop dev server.

- [ ] **Step 4: Commit**

```bash
git add src/App.jsx
git commit -m "feat: X-axis drag-to-scale + double-click reset"
```

---

## Task 10: Chart-body pan + cursor handling

**Files:**
- Modify: `src/App.jsx` — add `isDragging` state, update `startDrag`, update `<svg>` element (around line 3479)

- [ ] **Step 1: Add `isDragging` state**

In `src/App.jsx`, immediately after the existing `const dragStateRef = useRef(null);` line (added in Task 6), add:

```js
const [isDragging, setIsDragging] = useState(false);
```

- [ ] **Step 2: Update `startDrag` to track dragging state**

Locate the `startDrag` `useCallback` (added in Task 8). Add `setIsDragging(true)` immediately after the `dragStateRef.current = { ... };` assignment, and `setIsDragging(false)` inside the `handleUp` function before clearing the ref.

The updated relevant lines look like:

```js
  dragStateRef.current = {
    kind,
    startMouseX,
    startMouseY,
    startYDomain: yManualDomain ?? [s.y.domain()[0], s.y.domain()[1]],
    startXRange: xManualRange ?? [firstIdx, lastIdx],
    anchorPrice,
    anchorIndex,
    chartHeight: s.h,
    chartWidth: s.w,
    bandWidth: s.x.bandwidth(),
    allCandleIndices,
    rectLeft: rect.left + s.m.left,
    rectTop: rect.top + s.m.top,
  };
  setIsDragging(true);

  const handleMove = (e) => { /* unchanged */ };

  const handleUp = () => {
    window.removeEventListener("mousemove", handleMove);
    window.removeEventListener("mouseup", handleUp);
    dragStateRef.current = null;
    setIsDragging(false);
    if (svgRef.current) {
      d3.select(svgRef.current).select(".crosshair").style("display", "none");
    }
  };
```

Also extend the `useCallback` deps array to include `setIsDragging` (stable, but explicit):

```js
}, [candles, yManualDomain, xManualRange, setYManualDomain, setXManualRange, setIsDragging]);
```

- [ ] **Step 3: Locate the SVG element**

Read `src/App.jsx` lines 3478–3506 to confirm the existing SVG markup:

```jsx
<svg ref={svgRef} style={{ width: "100%", height: "100%", minHeight: 300, display: "block", cursor: "crosshair" }}
  onMouseMove={(e) => { ... existing crosshair logic ... }}
  onMouseLeave={() => { ... }}
/>
```

- [ ] **Step 4: Replace the SVG element with pan-aware version**

Replace the entire `<svg ... />` element (from `<svg ref={svgRef}` through `/>`) with:

```jsx
<svg ref={svgRef}
  style={{
    width: "100%", height: "100%", minHeight: 300, display: "block",
    cursor: isDragging && dragStateRef.current?.kind === "pan" ? "grabbing"
          : isDragging ? "default"
          : (xManualRange ? "grab" : "crosshair"),
  }}
  onMouseDown={(e) => {
    const s = chartScalesRef.current;
    if (!s) return;
    const rect = svgRef.current.getBoundingClientRect();
    const mx = e.clientX - rect.left - s.m.left;
    const my = e.clientY - rect.top - s.m.top;
    if (mx < 0 || mx > s.w || my < 0 || my > s.h) return; // outside chart body
    startDrag("pan", e); // no-op in auto X mode (handled inside startDrag)
  }}
  onMouseMove={(e) => {
    // Hide crosshair while dragging
    if (dragStateRef.current) {
      d3.select(svgRef.current).select(".crosshair").style("display", "none");
      return;
    }
    // ── existing crosshair logic ──
    const s = chartScalesRef.current;
    if (!s || !candles.length) return;
    const rect = svgRef.current.getBoundingClientRect();
    const mx = e.clientX - rect.left - s.m.left;
    const my = e.clientY - rect.top - s.m.top;
    const ch = d3.select(svgRef.current).select(".crosshair");
    if (mx < 0 || mx > s.w || my < 0 || my > s.h) { ch.style("display", "none"); return; }
    ch.style("display", null);
    const step = s.x.step();
    const idx = Math.max(0, Math.min(Math.round((mx - s.x.bandwidth() / 2) / step), candles.length - 1));
    const cx = s.x(idx) + s.x.bandwidth() / 2;
    const c = candles[idx];
    const bull = c.close >= c.open;
    ch.select(".ch-h").attr("y1", my).attr("y2", my);
    ch.select(".ch-v").attr("x1", cx).attr("x2", cx);
    ch.select(".ch-price-bg").attr("y", my - 8);
    ch.select(".ch-price").attr("y", my).text(s.y.invert(my).toFixed(2));
    const [dd, tt] = c.datetime.split(" ");
    const [, mo, dy] = dd.split("-");
    ch.select(".ch-time-bg").attr("x", cx - 36);
    ch.select(".ch-time").attr("x", cx).text(`${+mo}/${+dy} ${tt?.slice(0, 5) || "00:00"}`);
    ch.select(".ch-ohlc").attr("fill", bull ? "#26a69a" : "#ef5350")
      .text(`O ${c.open.toFixed(2)}  H ${c.high.toFixed(2)}  L ${c.low.toFixed(2)}  C ${c.close.toFixed(2)}`);
  }}
  onMouseLeave={() => { if (svgRef.current) d3.select(svgRef.current).select(".crosshair").style("display", "none"); }}
/>
```

The pan no-op when `xManualRange === null` is already handled inside `startDrag` (Task 8 Step 2 has `if (kind === "pan" && xManualRange === null) return;`).

- [ ] **Step 5: Run dev server and verify pan**

Run: `npm run dev`

Open the app, load data. First, drag the X-axis right to zoom in (so `xManualRange` is set). Then verify:
- Cursor over chart body shows `grab` when X is manual, `crosshair` when X is auto
- Mousedown + drag on chart body: cursor changes to `grabbing`, candles shift left/right
- Cannot pan past first or last candle
- Crosshair is hidden during drag
- Mouseup: cursor returns to `grab`/`crosshair`, crosshair resumes on next mouseover

Stop dev server.

- [ ] **Step 6: Commit**

```bash
git add src/App.jsx
git commit -m "feat: chart-body pan with grab cursor and crosshair-hide during drag"
```

---

## Task 11: Wheel zoom

**Files:**
- Modify: `src/App.jsx` — `<svg>` element

- [ ] **Step 1: Add `onWheel` handler to the SVG**

Add `onWheel` to the SVG element opening tag (alongside the existing `onMouseDown`, `onMouseMove`, `onMouseLeave` from Task 10 Step 4). Insert this before `onMouseLeave`:

```jsx
  onWheel={(e) => {
    const s = chartScalesRef.current;
    if (!s || !candles.length) return;
    const rect = svgRef.current.getBoundingClientRect();
    const mx = e.clientX - rect.left - s.m.left;
    const my = e.clientY - rect.top - s.m.top;
    if (mx < 0 || mx > s.w || my < 0 || my > s.h) return;
    // preventDefault for page scroll is handled by the native listener in Step 2

    const allCandleIndices = candles.map((c) => c.candleIndex);
    const firstIdx = allCandleIndices[0];
    const lastIdx = allCandleIndices[allCandleIndices.length - 1];
    const step = s.x.step();
    const arrIdx = Math.max(0, Math.min(Math.round((mx - s.x.bandwidth() / 2) / step), candles.length - 1));
    const anchorIndex = candles[arrIdx]?.candleIndex ?? firstIdx;
    const anchorPrice = s.y.invert(my);

    const result = wheelZoom({
      startYDomain: yManualDomain ?? [s.y.domain()[0], s.y.domain()[1]],
      startXRange: xManualRange ?? [firstIdx, lastIdx],
      anchorPrice,
      anchorIndex,
      deltaY: e.deltaY,
      modifiers: { shift: e.shiftKey, ctrl: e.ctrlKey || e.metaKey },
      chartHeight: s.h,
      chartWidth: s.w,
      allCandleIndices,
    });

    if (!e.ctrlKey && !e.metaKey) setXManualRange(result.xRange);
    if (!e.shiftKey) setYManualDomain(result.yDomain);
  }}
```

- [ ] **Step 2: Add native wheel listener for `preventDefault`**

React 17+ attaches `onWheel` as a passive listener by default — calling `preventDefault()` inside the React handler won't block page scroll. Add a non-passive native listener via `useEffect`. After the existing `useEffect` for `window.resize` (around line 1186–1189), add:

```js
useEffect(() => {
  const svg = svgRef.current;
  if (!svg) return;
  const handler = (e) => {
    const s = chartScalesRef.current;
    if (!s || !candles.length) return;
    const rect = svg.getBoundingClientRect();
    const mx = e.clientX - rect.left - s.m.left;
    const my = e.clientY - rect.top - s.m.top;
    if (mx >= 0 && mx <= s.w && my >= 0 && my <= s.h) {
      e.preventDefault();
    }
  };
  svg.addEventListener("wheel", handler, { passive: false });
  return () => svg.removeEventListener("wheel", handler);
}, [candles.length]);
```

(The React `onWheel` does the actual zoom; this native listener only blocks page scroll.)

- [ ] **Step 3: Run dev server and verify wheel zoom**

Run: `npm run dev`

Open the app, load data. Verify:
- Wheel up over chart: both axes zoom in toward cursor; price/candle under cursor stays put
- Wheel down: both axes zoom out
- Page does NOT scroll while wheeling over the chart
- Shift+wheel: only X axis zooms; Y stays put
- Ctrl+wheel: only Y axis zooms; X stays put

Stop dev server.

- [ ] **Step 4: Commit**

```bash
git add src/App.jsx
git commit -m "feat: wheel zoom with Shift/Ctrl modifiers"
```

---

## Task 12: Reset button in chart toolbar

**Files:**
- Modify: `src/App.jsx` — chart toolbar (around line 3450)

- [ ] **Step 1: Locate the chart toolbar**

Read `src/App.jsx` lines 3449–3475 to confirm the toolbar:

```jsx
<div style={{ display: "flex", gap: 5, alignItems: "center", flexWrap: "wrap", flexShrink: 0 }}>
  <span style={{ color: "#33334d", fontSize: 8 }}>TF:</span>
  {TF_OPTIONS.map((tf) => (
    <button key={tf.value} style={btn(timeframe === tf.value)} onClick={() => setTimeframe(tf.value)}>{tf.label}</button>
  ))}
  <div style={{ width: 1, height: 16, background: "#14142a", margin: "0 3px" }} />
  <button style={btn(false)} onClick={() => fetchCandles(false)} disabled={loadingData}>{loadingData ? "..." : "↻ REFRESH"}</button>
  ...
```

- [ ] **Step 2: Add the reset button**

Find the line `<button style={btn(false)} onClick={() => fetchCandles(false)} disabled={loadingData}>{loadingData ? "..." : "↻ REFRESH"}</button>` (around line 3456). Immediately after it, add:

```jsx
{isChartManual && (
  <button
    style={btn(false, "#9370db")}
    onClick={resetChartScale}
    title="Reset chart scale to auto-fit"
  >
    ⊕ AUTO
  </button>
)}
```

- [ ] **Step 3: Run dev server and verify reset button**

Run: `npm run dev`

Open the app, load data. Verify:
- "⊕ AUTO" button is NOT visible initially
- Drag Y axis up → "⊕ AUTO" button appears
- Click "⊕ AUTO" → both axes reset; button disappears
- Set both Y and X manually, click "⊕ AUTO" → both reset

Stop dev server.

- [ ] **Step 4: Commit**

```bash
git add src/App.jsx
git commit -m "feat: add Auto reset button to chart toolbar"
```

---

## Task 13: Manual end-to-end verification + final commit

**Files:** none modified — final integration check.

- [ ] **Step 1: Run the full test suite**

Run: `npm run test`

Expected: PASS — all existing tests + new tests (~22 helper + 5 hook tests) pass.

- [ ] **Step 2: Start dev server and verify all gestures**

Run: `npm run dev`

Open the app at http://localhost:5173, click "LAUNCH DEMO MODE" or load data with a real API key. Run through this checklist:

- [ ] Y-axis drag: drag up shrinks price range, drag down expands; price under cursor stays put
- [ ] Y-axis double-click: snaps back to auto-fit
- [ ] X-axis drag: drag right shows fewer candles, drag left shows more (capped at full data)
- [ ] X-axis double-click: snaps back to all candles visible
- [ ] Body drag: when X is zoomed, drag body horizontally pans; cannot pan past first/last candle; crosshair hidden during drag
- [ ] Body drag in auto mode: no-op (cursor stays `crosshair`)
- [ ] Wheel up over chart: both axes zoom in toward cursor
- [ ] Wheel down: both axes zoom out
- [ ] Shift+wheel: only X zooms
- [ ] Ctrl+wheel: only Y zooms
- [ ] Page does NOT scroll while wheeling over chart
- [ ] "⊕ AUTO" button appears when any axis is manual; disappears when both are auto
- [ ] Click "⊕ AUTO" → both axes reset
- [ ] Switch timeframe (1H → 4H): both axes reset; "⊕ AUTO" hidden
- [ ] Run analysis (or wait for live update) while in manual mode → manual scale persists; new TP/SL overlays render but don't trigger auto-rescale
- [ ] Crosshair still works in manual mode (over chart body when not dragging)

If any checkbox fails, debug and fix. Common issues:
- **Drag stutters at high frequency** — add `requestAnimationFrame` throttling inside `handleMove` in `startDrag`
- **Cursor doesn't change** — confirm `isDragging` state is wired (Task 10 Step 2)
- **Wheel scrolls page** — confirm the native `useEffect` wheel listener is attached (Task 11 Step 2)

- [ ] **Step 3: Stop dev server**

- [ ] **Step 4: Final integration commit (only if any fix was made in Step 2)**

If no fixes needed, skip this step. Otherwise:

```bash
git add src/App.jsx
git commit -m "fix: address manual verification findings"
```

---

## Summary

13 tasks. Each TDD-style for the 4 helpers + hook (5 tasks); each integration-style for the App.jsx wiring (7 tasks); one final manual verification task. Roughly 11 commits total (one per task that produces output).

Total new code: ~250 lines across `chartScaling.js` (~80) and `useChartScale.js` (~25); ~140 lines added to `App.jsx`. Test code: ~200 lines.
