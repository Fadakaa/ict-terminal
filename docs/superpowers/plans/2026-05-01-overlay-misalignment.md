# Overlay Misalignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Snap divergent OB/FVG/liquidity values onto the actual candle wicks at their claimed indices before they reach the chart, fix duplicate liquidity labels stacking on top of each other, and tighten prompt instructions so Claude is asked to keep numeric and narrative values consistent.

**Architecture:** Two parallel pure-function modules — `src/analysisSnap.js` (frontend) and `ml/analysis_snap.py` (backend) — implementing the same algorithm. Frontend snap fires in `runAnalysis()` between `JSON.parse` and `setAnalysis`. Backend snap fires in `/calibrate` before `bridge.parse_analysis`. Liquidity dedup is a small grouping helper used by the existing chart render loop in `App.jsx`.

**Tech Stack:** React 18, D3 7, Vitest + jsdom (frontend); FastAPI, Pydantic, pytest + starlette TestClient (backend).

**Spec:** [docs/superpowers/specs/2026-05-01-overlay-misalignment-design.md](../specs/2026-05-01-overlay-misalignment-design.md)

---

## File Structure

```
src/
├── analysisSnap.js              [NEW] pure helpers: snapAnalysisToCandles, groupLiquidityByLevel
├── App.jsx                      [MODIFY] import snap, call in runAnalysis, swap liquidity render
└── test/
    └── analysisSnap.test.js     [NEW] vitest unit tests

ml/
├── analysis_snap.py             [NEW] pure Python: snap_analysis_to_candles
├── prompts.py                   [MODIFY] add framework item 12 (consistency)
├── server.py                    [MODIFY] call snap in /calibrate, attach diagnostics
└── tests/
    └── test_analysis_snap.py    [NEW] pytest unit + integration tests
```

**Boundary justification:** Snap math is pure compute with no I/O — both modules export single-purpose functions with identical contracts. Tests live next to source in the project's existing convention (`src/test/*.test.js` and `ml/tests/test_*.py`). Render-loop dedup uses an exported helper (`groupLiquidityByLevel`) so the grouping logic is unit-tested even though the D3 render path is not.

---

## Task 1: Frontend snap — OBs

**Files:**
- Create: `src/analysisSnap.js`
- Create: `src/test/analysisSnap.test.js`

- [ ] **Step 1: Write the failing tests for OB snap behavior**

Create `src/test/analysisSnap.test.js`:

```js
import { describe, it, expect } from "vitest";
import { snapAnalysisToCandles } from "../analysisSnap.js";

const c = (h, l) => ({ datetime: "2026-04-30 12:00", open: l, high: h, low: l, close: h });

describe("snapAnalysisToCandles — orderBlocks", () => {
  it("snaps high/low when both diverge by more than $0.50", () => {
    const candles = [c(100, 90), c(110, 95), c(120, 105)];
    const analysis = {
      orderBlocks: [{ type: "bullish", high: 80, low: 70, candleIndex: 1 }],
    };
    const { analysis: out, diagnostics } = snapAnalysisToCandles(analysis, candles);
    expect(out.orderBlocks[0].high).toBe(110);
    expect(out.orderBlocks[0].low).toBe(95);
    expect(out.orderBlocks[0].snapped).toBe(true);
    expect(diagnostics.snapped_obs).toBe(1);
    expect(diagnostics.dropped_obs).toBe(0);
  });

  it("leaves OB unchanged when both within $0.50 tolerance", () => {
    const candles = [c(100, 90), c(110.3, 95.2), c(120, 105)];
    const analysis = {
      orderBlocks: [{ type: "bullish", high: 110, low: 95, candleIndex: 1 }],
    };
    const { analysis: out, diagnostics } = snapAnalysisToCandles(analysis, candles);
    expect(out.orderBlocks[0].high).toBe(110);
    expect(out.orderBlocks[0].low).toBe(95);
    expect(out.orderBlocks[0].snapped).toBeUndefined();
    expect(diagnostics.snapped_obs).toBe(0);
  });

  it("snaps both fields when only one is out of tolerance", () => {
    const candles = [c(110.2, 95)];
    const analysis = {
      orderBlocks: [{ type: "bullish", high: 110, low: 80, candleIndex: 0 }],
    };
    const { analysis: out, diagnostics } = snapAnalysisToCandles(analysis, candles);
    expect(out.orderBlocks[0].high).toBe(110.2);
    expect(out.orderBlocks[0].low).toBe(95);
    expect(diagnostics.snapped_obs).toBe(1);
  });

  it("drops OB when candleIndex is missing", () => {
    const candles = [c(100, 90)];
    const analysis = { orderBlocks: [{ type: "bullish", high: 99, low: 91 }] };
    const { analysis: out, diagnostics } = snapAnalysisToCandles(analysis, candles);
    expect(out.orderBlocks).toHaveLength(0);
    expect(diagnostics.dropped_obs).toBe(1);
  });

  it("drops OB when candleIndex is negative", () => {
    const candles = [c(100, 90)];
    const analysis = { orderBlocks: [{ type: "bullish", high: 99, low: 91, candleIndex: -1 }] };
    const { analysis: out, diagnostics } = snapAnalysisToCandles(analysis, candles);
    expect(out.orderBlocks).toHaveLength(0);
    expect(diagnostics.dropped_obs).toBe(1);
  });

  it("drops OB when candleIndex is out of bounds", () => {
    const candles = [c(100, 90), c(110, 95)];
    const analysis = { orderBlocks: [{ type: "bullish", high: 99, low: 91, candleIndex: 5 }] };
    const { analysis: out, diagnostics } = snapAnalysisToCandles(analysis, candles);
    expect(out.orderBlocks).toHaveLength(0);
    expect(diagnostics.dropped_obs).toBe(1);
  });

  it("preserves other OB fields (type, tf, strength, note) on snap", () => {
    const candles = [c(100, 90), c(110, 95)];
    const analysis = {
      orderBlocks: [{
        type: "bearish", high: 80, low: 70, candleIndex: 1,
        tf: "1H", strength: "strong", times_tested: 2, note: "key zone",
      }],
    };
    const { analysis: out } = snapAnalysisToCandles(analysis, candles);
    expect(out.orderBlocks[0]).toMatchObject({
      type: "bearish", tf: "1H", strength: "strong", times_tested: 2,
      note: "key zone", snapped: true, high: 110, low: 95,
    });
  });

  it("preserves analysis fields untouched when no orderBlocks present", () => {
    const candles = [c(100, 90)];
    const analysis = { bias: "bullish", summary: "stuff" };
    const { analysis: out, diagnostics } = snapAnalysisToCandles(analysis, candles);
    expect(out.bias).toBe("bullish");
    expect(out.summary).toBe("stuff");
    expect(diagnostics.snapped_obs).toBe(0);
    expect(diagnostics.dropped_obs).toBe(0);
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npx vitest run src/test/analysisSnap.test.js`

Expected: FAIL — "Cannot find module '../analysisSnap.js'".

- [ ] **Step 3: Write minimal implementation**

Create `src/analysisSnap.js`:

```js
const DEFAULT_TOLERANCE = 0.50;

function makeDiagnostics() {
  return {
    snapped_obs: 0, dropped_obs: 0,
    snapped_fvgs: 0, dropped_fvgs: 0,
    snapped_liquidity: 0, dropped_liquidity: 0,
    deltas: [],
  };
}

function snapOrderBlocks(obs, candles, tolerance, diag) {
  const out = [];
  const n = candles.length;
  for (const ob of obs) {
    const ci = ob.candleIndex;
    if (ci === undefined || ci === null || ci < 0 || ci >= n) {
      diag.dropped_obs += 1;
      continue;
    }
    const c = candles[ci];
    const highOff = Math.abs((ob.high ?? 0) - c.high);
    const lowOff = Math.abs((ob.low ?? 0) - c.low);
    if (highOff > tolerance || lowOff > tolerance) {
      diag.snapped_obs += 1;
      diag.deltas.push({
        kind: "ob", candleIndex: ci,
        claimed: { high: ob.high, low: ob.low },
        snapped: { high: c.high, low: c.low },
      });
      out.push({ ...ob, high: c.high, low: c.low, snapped: true });
    } else {
      out.push(ob);
    }
  }
  return out;
}

export function snapAnalysisToCandles(analysis, candles, options = {}) {
  const tolerance = options.tolerance ?? DEFAULT_TOLERANCE;
  const diag = makeDiagnostics();
  const obs = analysis.orderBlocks ?? [];
  return {
    analysis: { ...analysis, orderBlocks: snapOrderBlocks(obs, candles, tolerance, diag) },
    diagnostics: diag,
  };
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `npx vitest run src/test/analysisSnap.test.js`

Expected: PASS — 8 OB tests green.

- [ ] **Step 5: Commit**

```bash
git add src/analysisSnap.js src/test/analysisSnap.test.js
git commit -m "feat: snap OBs to candle wicks (frontend helper)"
```

---

## Task 2: Frontend snap — FVGs

**Files:**
- Modify: `src/analysisSnap.js`
- Modify: `src/test/analysisSnap.test.js`

- [ ] **Step 1: Append FVG tests to the existing file**

Append to `src/test/analysisSnap.test.js`:

```js
describe("snapAnalysisToCandles — fvgs", () => {
  it("snaps bullish FVG to (c0.high, c2.low) gap range", () => {
    // c0 high=110, c2 low=120 → expected gap is (low=110, high=120)
    const candles = [c(110, 100), c(118, 112), c(125, 120), c(130, 122)];
    const analysis = {
      fvgs: [{ type: "bullish", high: 130, low: 90, startIndex: 0 }],
    };
    const { analysis: out, diagnostics } = snapAnalysisToCandles(analysis, candles);
    expect(out.fvgs[0].low).toBe(110);
    expect(out.fvgs[0].high).toBe(120);
    expect(out.fvgs[0].snapped).toBe(true);
    expect(diagnostics.snapped_fvgs).toBe(1);
  });

  it("snaps bearish FVG to (c2.high, c0.low) — high=c0.low, low=c2.high", () => {
    // bearish: c0 low=90, c2 high=85 → expected (high=90, low=85)
    const candles = [c(110, 90), c(95, 85), c(90, 80), c(88, 78)];
    const analysis = {
      fvgs: [{ type: "bearish", high: 80, low: 70, startIndex: 0 }],
    };
    const { analysis: out, diagnostics } = snapAnalysisToCandles(analysis, candles);
    expect(out.fvgs[0].high).toBe(90);
    expect(out.fvgs[0].low).toBe(85);
    expect(diagnostics.snapped_fvgs).toBe(1);
  });

  it("leaves FVG unchanged when within tolerance", () => {
    const candles = [c(110, 100), c(118, 112), c(125, 120.3)];
    const analysis = {
      fvgs: [{ type: "bullish", high: 120, low: 110, startIndex: 0 }],
    };
    const { analysis: out, diagnostics } = snapAnalysisToCandles(analysis, candles);
    expect(out.fvgs[0].snapped).toBeUndefined();
    expect(diagnostics.snapped_fvgs).toBe(0);
  });

  it("drops bullish FVG when expected_low >= expected_high (no real gap)", () => {
    // c0.high=120, c2.low=110 → expected (low=120, high=110), degenerate
    const candles = [c(120, 100), c(115, 105), c(115, 110)];
    const analysis = {
      fvgs: [{ type: "bullish", high: 120, low: 110, startIndex: 0 }],
    };
    const { analysis: out, diagnostics } = snapAnalysisToCandles(analysis, candles);
    expect(out.fvgs).toHaveLength(0);
    expect(diagnostics.dropped_fvgs).toBe(1);
  });

  it("drops FVG when startIndex+2 is out of bounds", () => {
    const candles = [c(110, 100), c(118, 112)];
    const analysis = {
      fvgs: [{ type: "bullish", high: 120, low: 110, startIndex: 0 }],
    };
    const { analysis: out, diagnostics } = snapAnalysisToCandles(analysis, candles);
    expect(out.fvgs).toHaveLength(0);
    expect(diagnostics.dropped_fvgs).toBe(1);
  });

  it("drops FVG when startIndex is missing or negative", () => {
    const candles = [c(110, 100), c(118, 112), c(125, 120)];
    const analysis = {
      fvgs: [
        { type: "bullish", high: 120, low: 110 },
        { type: "bearish", high: 90, low: 85, startIndex: -1 },
      ],
    };
    const { analysis: out, diagnostics } = snapAnalysisToCandles(analysis, candles);
    expect(out.fvgs).toHaveLength(0);
    expect(diagnostics.dropped_fvgs).toBe(2);
  });

  it("preserves filled, fill_percentage, overlaps_ob, tf on snap", () => {
    const candles = [c(110, 100), c(118, 112), c(125, 120)];
    const analysis = {
      fvgs: [{
        type: "bullish", high: 130, low: 90, startIndex: 0,
        tf: "1H", filled: false, fill_percentage: 25, overlaps_ob: true, note: "n",
      }],
    };
    const { analysis: out } = snapAnalysisToCandles(analysis, candles);
    expect(out.fvgs[0]).toMatchObject({
      tf: "1H", filled: false, fill_percentage: 25, overlaps_ob: true, note: "n",
      snapped: true, high: 120, low: 110,
    });
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npx vitest run src/test/analysisSnap.test.js`

Expected: FAIL — FVG cases all error (`Cannot read property '...' of undefined` on the array, since the implementation doesn't process `fvgs` yet).

- [ ] **Step 3: Extend the implementation**

Edit `src/analysisSnap.js`. Add a `snapFvgs` helper after `snapOrderBlocks` and call it from `snapAnalysisToCandles`. The full updated file:

```js
const DEFAULT_TOLERANCE = 0.50;

function makeDiagnostics() {
  return {
    snapped_obs: 0, dropped_obs: 0,
    snapped_fvgs: 0, dropped_fvgs: 0,
    snapped_liquidity: 0, dropped_liquidity: 0,
    deltas: [],
  };
}

function snapOrderBlocks(obs, candles, tolerance, diag) {
  const out = [];
  const n = candles.length;
  for (const ob of obs) {
    const ci = ob.candleIndex;
    if (ci === undefined || ci === null || ci < 0 || ci >= n) {
      diag.dropped_obs += 1;
      continue;
    }
    const c = candles[ci];
    const highOff = Math.abs((ob.high ?? 0) - c.high);
    const lowOff = Math.abs((ob.low ?? 0) - c.low);
    if (highOff > tolerance || lowOff > tolerance) {
      diag.snapped_obs += 1;
      diag.deltas.push({
        kind: "ob", candleIndex: ci,
        claimed: { high: ob.high, low: ob.low },
        snapped: { high: c.high, low: c.low },
      });
      out.push({ ...ob, high: c.high, low: c.low, snapped: true });
    } else {
      out.push(ob);
    }
  }
  return out;
}

function snapFvgs(fvgs, candles, tolerance, diag) {
  const out = [];
  const n = candles.length;
  for (const fvg of fvgs) {
    const si = fvg.startIndex;
    if (si === undefined || si === null || si < 0 || si + 2 >= n) {
      diag.dropped_fvgs += 1;
      continue;
    }
    const c0 = candles[si];
    const c2 = candles[si + 2];
    let expectedHigh, expectedLow;
    if (fvg.type === "bullish") {
      expectedLow = c0.high;
      expectedHigh = c2.low;
    } else {
      expectedHigh = c0.low;
      expectedLow = c2.high;
    }
    if (expectedLow >= expectedHigh) {
      diag.dropped_fvgs += 1;
      continue;
    }
    const highOff = Math.abs((fvg.high ?? 0) - expectedHigh);
    const lowOff = Math.abs((fvg.low ?? 0) - expectedLow);
    if (highOff > tolerance || lowOff > tolerance) {
      diag.snapped_fvgs += 1;
      diag.deltas.push({
        kind: "fvg", startIndex: si,
        claimed: { high: fvg.high, low: fvg.low },
        snapped: { high: expectedHigh, low: expectedLow },
      });
      out.push({ ...fvg, high: expectedHigh, low: expectedLow, snapped: true });
    } else {
      out.push(fvg);
    }
  }
  return out;
}

export function snapAnalysisToCandles(analysis, candles, options = {}) {
  const tolerance = options.tolerance ?? DEFAULT_TOLERANCE;
  const diag = makeDiagnostics();
  const obs = analysis.orderBlocks ?? [];
  const fvgs = analysis.fvgs ?? [];
  return {
    analysis: {
      ...analysis,
      orderBlocks: snapOrderBlocks(obs, candles, tolerance, diag),
      fvgs: snapFvgs(fvgs, candles, tolerance, diag),
    },
    diagnostics: diag,
  };
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `npx vitest run src/test/analysisSnap.test.js`

Expected: PASS — all OB + FVG tests green (15 total).

- [ ] **Step 5: Commit**

```bash
git add src/analysisSnap.js src/test/analysisSnap.test.js
git commit -m "feat: snap FVGs to candle gap ranges"
```

---

## Task 3: Frontend snap — Liquidity

**Files:**
- Modify: `src/analysisSnap.js`
- Modify: `src/test/analysisSnap.test.js`

- [ ] **Step 1: Append liquidity tests**

Append to `src/test/analysisSnap.test.js`:

```js
describe("snapAnalysisToCandles — liquidity", () => {
  it("snaps buyside price to candle.high when out of tolerance", () => {
    const candles = [c(100, 90), c(112.5, 100)];
    const analysis = {
      liquidity: [{ type: "buyside", price: 105, candleIndex: 1 }],
    };
    const { analysis: out, diagnostics } = snapAnalysisToCandles(analysis, candles);
    expect(out.liquidity[0].price).toBe(112.5);
    expect(out.liquidity[0].snapped).toBe(true);
    expect(diagnostics.snapped_liquidity).toBe(1);
  });

  it("snaps sellside price to candle.low when out of tolerance", () => {
    const candles = [c(100, 90), c(110, 88.7)];
    const analysis = {
      liquidity: [{ type: "sellside", price: 95, candleIndex: 1 }],
    };
    const { analysis: out, diagnostics } = snapAnalysisToCandles(analysis, candles);
    expect(out.liquidity[0].price).toBe(88.7);
    expect(diagnostics.snapped_liquidity).toBe(1);
  });

  it("leaves liquidity unchanged when within tolerance", () => {
    const candles = [c(100.3, 90), c(110, 95)];
    const analysis = {
      liquidity: [{ type: "buyside", price: 100, candleIndex: 0 }],
    };
    const { analysis: out, diagnostics } = snapAnalysisToCandles(analysis, candles);
    expect(out.liquidity[0].snapped).toBeUndefined();
    expect(diagnostics.snapped_liquidity).toBe(0);
  });

  it("drops liquidity with missing/negative/OOB candleIndex", () => {
    const candles = [c(100, 90)];
    const analysis = {
      liquidity: [
        { type: "buyside", price: 100 },
        { type: "sellside", price: 90, candleIndex: -1 },
        { type: "buyside", price: 100, candleIndex: 99 },
      ],
    };
    const { analysis: out, diagnostics } = snapAnalysisToCandles(analysis, candles);
    expect(out.liquidity).toHaveLength(0);
    expect(diagnostics.dropped_liquidity).toBe(3);
  });

  it("preserves swept, tf, note on snap", () => {
    const candles = [c(112, 90)];
    const analysis = {
      liquidity: [{
        type: "buyside", price: 100, candleIndex: 0,
        tf: "4H", swept: false, note: "key high",
      }],
    };
    const { analysis: out } = snapAnalysisToCandles(analysis, candles);
    expect(out.liquidity[0]).toMatchObject({
      type: "buyside", tf: "4H", swept: false, note: "key high",
      snapped: true, price: 112,
    });
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npx vitest run src/test/analysisSnap.test.js`

Expected: FAIL — liquidity cases error (output has no `liquidity` field since impl doesn't process it).

- [ ] **Step 3: Extend the implementation**

Edit `src/analysisSnap.js`. Add `snapLiquidity` helper after `snapFvgs` and call it from `snapAnalysisToCandles`. Insert this function:

```js
function snapLiquidity(liqs, candles, tolerance, diag) {
  const out = [];
  const n = candles.length;
  for (const liq of liqs) {
    const ci = liq.candleIndex;
    if (ci === undefined || ci === null || ci < 0 || ci >= n) {
      diag.dropped_liquidity += 1;
      continue;
    }
    const c = candles[ci];
    const expected = liq.type === "buyside" ? c.high : c.low;
    const off = Math.abs((liq.price ?? 0) - expected);
    if (off > tolerance) {
      diag.snapped_liquidity += 1;
      diag.deltas.push({
        kind: "liquidity", candleIndex: ci,
        claimed: { price: liq.price },
        snapped: { price: expected },
      });
      out.push({ ...liq, price: expected, snapped: true });
    } else {
      out.push(liq);
    }
  }
  return out;
}
```

And update `snapAnalysisToCandles` to call it:

```js
export function snapAnalysisToCandles(analysis, candles, options = {}) {
  const tolerance = options.tolerance ?? DEFAULT_TOLERANCE;
  const diag = makeDiagnostics();
  const obs = analysis.orderBlocks ?? [];
  const fvgs = analysis.fvgs ?? [];
  const liqs = analysis.liquidity ?? [];
  return {
    analysis: {
      ...analysis,
      orderBlocks: snapOrderBlocks(obs, candles, tolerance, diag),
      fvgs: snapFvgs(fvgs, candles, tolerance, diag),
      liquidity: snapLiquidity(liqs, candles, tolerance, diag),
    },
    diagnostics: diag,
  };
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `npx vitest run src/test/analysisSnap.test.js`

Expected: PASS — 20 tests green (8 OB + 7 FVG + 5 liquidity).

- [ ] **Step 5: Commit**

```bash
git add src/analysisSnap.js src/test/analysisSnap.test.js
git commit -m "feat: snap liquidity prices to candle wicks"
```

---

## Task 4: Frontend `groupLiquidityByLevel` + idempotency check

**Files:**
- Modify: `src/analysisSnap.js`
- Modify: `src/test/analysisSnap.test.js`

- [ ] **Step 1: Append grouping + idempotency tests**

Append to `src/test/analysisSnap.test.js`:

```js
import { groupLiquidityByLevel } from "../analysisSnap.js";

describe("groupLiquidityByLevel", () => {
  it("groups two BSLs at nearby prices with same tf into one group", () => {
    const liq = [
      { type: "buyside", price: 4630.30, candleIndex: 5, tf: "1H" },
      { type: "buyside", price: 4630.50, candleIndex: 9, tf: "1H" },
    ];
    const groups = groupLiquidityByLevel(liq, 0.50);
    expect(groups).toHaveLength(1);
    expect(groups[0].items).toHaveLength(2);
  });

  it("keeps two BSLs at same price but different tf as separate groups", () => {
    const liq = [
      { type: "buyside", price: 4630.30, candleIndex: 5, tf: "1H" },
      { type: "buyside", price: 4630.30, candleIndex: 9, tf: "4H" },
    ];
    const groups = groupLiquidityByLevel(liq, 0.50);
    expect(groups).toHaveLength(2);
  });

  it("keeps BSL and SSL at same price as separate groups", () => {
    const liq = [
      { type: "buyside", price: 4630, candleIndex: 5, tf: "1H" },
      { type: "sellside", price: 4630, candleIndex: 9, tf: "1H" },
    ];
    const groups = groupLiquidityByLevel(liq, 0.50);
    expect(groups).toHaveLength(2);
  });

  it("uses leftmost candleIndex as the group representative", () => {
    const liq = [
      { type: "buyside", price: 4630.30, candleIndex: 9, tf: "1H" },
      { type: "buyside", price: 4630.50, candleIndex: 5, tf: "1H" },
    ];
    const groups = groupLiquidityByLevel(liq, 0.50);
    expect(groups[0].items[0].candleIndex).toBe(5);
  });

  it("returns empty array on empty input", () => {
    expect(groupLiquidityByLevel([], 0.50)).toEqual([]);
  });

  it("treats prices outside tolerance as separate groups", () => {
    const liq = [
      { type: "buyside", price: 4630.00, candleIndex: 5, tf: "1H" },
      { type: "buyside", price: 4631.00, candleIndex: 9, tf: "1H" },
    ];
    const groups = groupLiquidityByLevel(liq, 0.50);
    expect(groups).toHaveLength(2);
  });
});

describe("snapAnalysisToCandles — idempotency", () => {
  it("is a no-op when analysis already matches candles (round-trip)", () => {
    const candles = [c(100, 90), c(110, 95), c(120, 105), c(130, 115)];
    const analysis = {
      bias: "bullish",
      orderBlocks: [{ type: "bullish", high: 110, low: 95, candleIndex: 1, tf: "1H" }],
      fvgs: [{ type: "bullish", high: 105, low: 95, startIndex: 0, tf: "1H" }],
      liquidity: [{ type: "buyside", price: 130, candleIndex: 3, tf: "1H" }],
    };
    const { analysis: out, diagnostics } = snapAnalysisToCandles(analysis, candles);
    expect(diagnostics.snapped_obs).toBe(0);
    expect(diagnostics.snapped_fvgs).toBe(0);
    expect(diagnostics.snapped_liquidity).toBe(0);
    expect(diagnostics.dropped_obs).toBe(0);
    expect(diagnostics.dropped_fvgs).toBe(0);
    expect(diagnostics.dropped_liquidity).toBe(0);
    expect(out.orderBlocks[0].snapped).toBeUndefined();

    // Second pass against snapped output should also be a no-op
    const second = snapAnalysisToCandles(out, candles);
    expect(second.diagnostics.snapped_obs).toBe(0);
    expect(second.diagnostics.snapped_fvgs).toBe(0);
    expect(second.diagnostics.snapped_liquidity).toBe(0);
  });
});
```

- [ ] **Step 2: Run tests to verify the new ones fail**

Run: `npx vitest run src/test/analysisSnap.test.js`

Expected: FAIL on `groupLiquidityByLevel` cases — "groupLiquidityByLevel is not a function". Idempotency tests should already pass.

- [ ] **Step 3: Add `groupLiquidityByLevel` export**

Append to `src/analysisSnap.js`:

```js
export function groupLiquidityByLevel(liquidity, tolerance = DEFAULT_TOLERANCE) {
  const groups = [];
  for (const liq of liquidity) {
    const bucket = Math.round(liq.price / tolerance) * tolerance;
    const key = `${liq.type}|${bucket}|${liq.tf ?? ""}`;
    let group = groups.find((g) => g.key === key);
    if (!group) {
      group = { key, items: [] };
      groups.push(group);
    }
    group.items.push(liq);
  }
  // Sort each group's items so the leftmost candleIndex is first
  for (const g of groups) {
    g.items.sort((a, b) => (a.candleIndex ?? 0) - (b.candleIndex ?? 0));
  }
  return groups;
}
```

- [ ] **Step 4: Run tests to verify all pass**

Run: `npx vitest run src/test/analysisSnap.test.js`

Expected: PASS — 27 tests green.

- [ ] **Step 5: Commit**

```bash
git add src/analysisSnap.js src/test/analysisSnap.test.js
git commit -m "feat: add groupLiquidityByLevel and idempotency coverage"
```

---

## Task 5: Backend snap — full module (all three sections)

**Files:**
- Create: `ml/analysis_snap.py`
- Create: `ml/tests/test_analysis_snap.py`

- [ ] **Step 1: Write the failing pytest tests**

Create `ml/tests/test_analysis_snap.py`:

```python
"""Tests for ml/analysis_snap.py — overlay snap-to-candle helpers."""
from ml.analysis_snap import snap_analysis_to_candles


def _candle(h, l, dt="2026-04-30 12:00"):
    return {"datetime": dt, "open": l, "high": h, "low": l, "close": h}


class TestOrderBlockSnap:
    def test_snaps_high_low_when_diverged(self):
        candles = [_candle(100, 90), _candle(110, 95), _candle(120, 105)]
        analysis = {"orderBlocks": [{"type": "bullish", "high": 80, "low": 70, "candleIndex": 1}]}
        out, diag = snap_analysis_to_candles(analysis, candles)
        assert out["orderBlocks"][0]["high"] == 110
        assert out["orderBlocks"][0]["low"] == 95
        assert out["orderBlocks"][0]["snapped"] is True
        assert diag["snapped_obs"] == 1
        assert diag["dropped_obs"] == 0

    def test_within_tolerance_unchanged(self):
        candles = [_candle(100, 90), _candle(110.3, 95.2)]
        analysis = {"orderBlocks": [{"type": "bullish", "high": 110, "low": 95, "candleIndex": 1}]}
        out, diag = snap_analysis_to_candles(analysis, candles)
        assert "snapped" not in out["orderBlocks"][0]
        assert diag["snapped_obs"] == 0

    def test_drops_missing_index(self):
        candles = [_candle(100, 90)]
        analysis = {"orderBlocks": [{"type": "bullish", "high": 99, "low": 91}]}
        out, diag = snap_analysis_to_candles(analysis, candles)
        assert out["orderBlocks"] == []
        assert diag["dropped_obs"] == 1

    def test_drops_negative_index(self):
        candles = [_candle(100, 90)]
        analysis = {"orderBlocks": [{"type": "bullish", "high": 99, "low": 91, "candleIndex": -1}]}
        out, diag = snap_analysis_to_candles(analysis, candles)
        assert out["orderBlocks"] == []
        assert diag["dropped_obs"] == 1

    def test_drops_oob_index(self):
        candles = [_candle(100, 90), _candle(110, 95)]
        analysis = {"orderBlocks": [{"type": "bullish", "high": 99, "low": 91, "candleIndex": 5}]}
        out, diag = snap_analysis_to_candles(analysis, candles)
        assert out["orderBlocks"] == []
        assert diag["dropped_obs"] == 1

    def test_preserves_other_fields(self):
        candles = [_candle(100, 90), _candle(110, 95)]
        analysis = {"orderBlocks": [{
            "type": "bearish", "high": 80, "low": 70, "candleIndex": 1,
            "tf": "1H", "strength": "strong", "note": "key zone",
        }]}
        out, _ = snap_analysis_to_candles(analysis, candles)
        ob = out["orderBlocks"][0]
        assert ob["type"] == "bearish"
        assert ob["tf"] == "1H"
        assert ob["strength"] == "strong"
        assert ob["note"] == "key zone"
        assert ob["snapped"] is True


class TestFvgSnap:
    def test_snaps_bullish_to_gap_range(self):
        candles = [_candle(110, 100), _candle(118, 112), _candle(125, 120), _candle(130, 122)]
        analysis = {"fvgs": [{"type": "bullish", "high": 130, "low": 90, "startIndex": 0}]}
        out, diag = snap_analysis_to_candles(analysis, candles)
        assert out["fvgs"][0]["low"] == 110
        assert out["fvgs"][0]["high"] == 120
        assert out["fvgs"][0]["snapped"] is True
        assert diag["snapped_fvgs"] == 1

    def test_snaps_bearish_to_gap_range(self):
        candles = [_candle(110, 90), _candle(95, 85), _candle(90, 80)]
        analysis = {"fvgs": [{"type": "bearish", "high": 80, "low": 70, "startIndex": 0}]}
        out, _ = snap_analysis_to_candles(analysis, candles)
        assert out["fvgs"][0]["high"] == 90
        assert out["fvgs"][0]["low"] == 85

    def test_drops_degenerate_gap(self):
        # bullish: c0.high=120, c2.low=110 → expected_low (120) >= expected_high (110) → drop
        candles = [_candle(120, 100), _candle(115, 105), _candle(115, 110)]
        analysis = {"fvgs": [{"type": "bullish", "high": 120, "low": 110, "startIndex": 0}]}
        out, diag = snap_analysis_to_candles(analysis, candles)
        assert out["fvgs"] == []
        assert diag["dropped_fvgs"] == 1

    def test_drops_oob_startindex(self):
        candles = [_candle(110, 100), _candle(118, 112)]
        analysis = {"fvgs": [{"type": "bullish", "high": 120, "low": 110, "startIndex": 0}]}
        out, diag = snap_analysis_to_candles(analysis, candles)
        assert out["fvgs"] == []
        assert diag["dropped_fvgs"] == 1

    def test_within_tolerance_unchanged(self):
        candles = [_candle(110, 100), _candle(118, 112), _candle(125, 120.3)]
        analysis = {"fvgs": [{"type": "bullish", "high": 120, "low": 110, "startIndex": 0}]}
        out, diag = snap_analysis_to_candles(analysis, candles)
        assert "snapped" not in out["fvgs"][0]
        assert diag["snapped_fvgs"] == 0


class TestLiquiditySnap:
    def test_snaps_buyside_to_high(self):
        candles = [_candle(100, 90), _candle(112.5, 100)]
        analysis = {"liquidity": [{"type": "buyside", "price": 105, "candleIndex": 1}]}
        out, diag = snap_analysis_to_candles(analysis, candles)
        assert out["liquidity"][0]["price"] == 112.5
        assert out["liquidity"][0]["snapped"] is True
        assert diag["snapped_liquidity"] == 1

    def test_snaps_sellside_to_low(self):
        candles = [_candle(100, 90), _candle(110, 88.7)]
        analysis = {"liquidity": [{"type": "sellside", "price": 95, "candleIndex": 1}]}
        out, _ = snap_analysis_to_candles(analysis, candles)
        assert out["liquidity"][0]["price"] == 88.7

    def test_within_tolerance_unchanged(self):
        candles = [_candle(100.3, 90)]
        analysis = {"liquidity": [{"type": "buyside", "price": 100, "candleIndex": 0}]}
        out, diag = snap_analysis_to_candles(analysis, candles)
        assert "snapped" not in out["liquidity"][0]
        assert diag["snapped_liquidity"] == 0

    def test_drops_missing_or_oob_index(self):
        candles = [_candle(100, 90)]
        analysis = {"liquidity": [
            {"type": "buyside", "price": 100},
            {"type": "sellside", "price": 90, "candleIndex": -1},
            {"type": "buyside", "price": 100, "candleIndex": 99},
        ]}
        out, diag = snap_analysis_to_candles(analysis, candles)
        assert out["liquidity"] == []
        assert diag["dropped_liquidity"] == 3


class TestIdempotency:
    def test_round_trip_no_snaps(self):
        candles = [_candle(100, 90), _candle(110, 95), _candle(120, 105), _candle(130, 115)]
        analysis = {
            "bias": "bullish",
            "orderBlocks": [{"type": "bullish", "high": 110, "low": 95, "candleIndex": 1}],
            "fvgs": [{"type": "bullish", "high": 105, "low": 95, "startIndex": 0}],
            "liquidity": [{"type": "buyside", "price": 130, "candleIndex": 3}],
        }
        out, diag = snap_analysis_to_candles(analysis, candles)
        assert diag["snapped_obs"] == 0
        assert diag["snapped_fvgs"] == 0
        assert diag["snapped_liquidity"] == 0
        # Second pass on snapped output is also a no-op
        _, diag2 = snap_analysis_to_candles(out, candles)
        assert diag2["snapped_obs"] == 0
        assert diag2["snapped_fvgs"] == 0
        assert diag2["snapped_liquidity"] == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest ml/tests/test_analysis_snap.py -v`

Expected: FAIL — `ImportError: cannot import name 'snap_analysis_to_candles' from 'ml.analysis_snap'` (module does not exist).

- [ ] **Step 3: Implement the backend module**

Create `ml/analysis_snap.py`:

```python
"""Pure helpers to snap Claude's analysis JSON onto actual candle wicks.

Mirror of src/analysisSnap.js. Both modules implement the same algorithm:
- OBs: snap (high, low) to candles[candleIndex].(high, low) when off by > tolerance
- FVGs: snap (high, low) to gap range across (startIndex, startIndex+2)
- Liquidity: snap price to candle wick (high for buyside, low for sellside)

Items with missing / negative / out-of-bounds indices are dropped.
Degenerate FVGs (expected_low >= expected_high) are also dropped.
"""
from typing import Any

DEFAULT_TOLERANCE = 0.50


def _make_diagnostics() -> dict:
    return {
        "snapped_obs": 0, "dropped_obs": 0,
        "snapped_fvgs": 0, "dropped_fvgs": 0,
        "snapped_liquidity": 0, "dropped_liquidity": 0,
        "deltas": [],
    }


def _snap_obs(obs: list[dict], candles: list[dict], tolerance: float, diag: dict) -> list[dict]:
    out = []
    n = len(candles)
    for ob in obs:
        ci = ob.get("candleIndex")
        if ci is None or ci < 0 or ci >= n:
            diag["dropped_obs"] += 1
            continue
        c = candles[ci]
        c_high = float(c["high"])
        c_low = float(c["low"])
        high_off = abs(float(ob.get("high", 0)) - c_high)
        low_off = abs(float(ob.get("low", 0)) - c_low)
        if high_off > tolerance or low_off > tolerance:
            diag["snapped_obs"] += 1
            diag["deltas"].append({
                "kind": "ob", "candleIndex": ci,
                "claimed": {"high": ob.get("high"), "low": ob.get("low")},
                "snapped": {"high": c_high, "low": c_low},
            })
            out.append({**ob, "high": c_high, "low": c_low, "snapped": True})
        else:
            out.append(ob)
    return out


def _snap_fvgs(fvgs: list[dict], candles: list[dict], tolerance: float, diag: dict) -> list[dict]:
    out = []
    n = len(candles)
    for fvg in fvgs:
        si = fvg.get("startIndex")
        if si is None or si < 0 or si + 2 >= n:
            diag["dropped_fvgs"] += 1
            continue
        c0 = candles[si]
        c2 = candles[si + 2]
        if fvg.get("type") == "bullish":
            expected_low = float(c0["high"])
            expected_high = float(c2["low"])
        else:
            expected_high = float(c0["low"])
            expected_low = float(c2["high"])
        if expected_low >= expected_high:
            diag["dropped_fvgs"] += 1
            continue
        high_off = abs(float(fvg.get("high", 0)) - expected_high)
        low_off = abs(float(fvg.get("low", 0)) - expected_low)
        if high_off > tolerance or low_off > tolerance:
            diag["snapped_fvgs"] += 1
            diag["deltas"].append({
                "kind": "fvg", "startIndex": si,
                "claimed": {"high": fvg.get("high"), "low": fvg.get("low")},
                "snapped": {"high": expected_high, "low": expected_low},
            })
            out.append({**fvg, "high": expected_high, "low": expected_low, "snapped": True})
        else:
            out.append(fvg)
    return out


def _snap_liquidity(liqs: list[dict], candles: list[dict], tolerance: float, diag: dict) -> list[dict]:
    out = []
    n = len(candles)
    for liq in liqs:
        ci = liq.get("candleIndex")
        if ci is None or ci < 0 or ci >= n:
            diag["dropped_liquidity"] += 1
            continue
        c = candles[ci]
        expected = float(c["high"]) if liq.get("type") == "buyside" else float(c["low"])
        off = abs(float(liq.get("price", 0)) - expected)
        if off > tolerance:
            diag["snapped_liquidity"] += 1
            diag["deltas"].append({
                "kind": "liquidity", "candleIndex": ci,
                "claimed": {"price": liq.get("price")},
                "snapped": {"price": expected},
            })
            out.append({**liq, "price": expected, "snapped": True})
        else:
            out.append(liq)
    return out


def snap_analysis_to_candles(
    analysis: dict[str, Any],
    candles: list[dict],
    tolerance: float = DEFAULT_TOLERANCE,
) -> tuple[dict, dict]:
    """Return (snapped_analysis, diagnostics).

    Pure: no I/O, no logging. Caller decides whether/how to log.
    """
    diag = _make_diagnostics()
    obs = analysis.get("orderBlocks") or []
    fvgs = analysis.get("fvgs") or []
    liqs = analysis.get("liquidity") or []
    snapped = {
        **analysis,
        "orderBlocks": _snap_obs(obs, candles, tolerance, diag),
        "fvgs": _snap_fvgs(fvgs, candles, tolerance, diag),
        "liquidity": _snap_liquidity(liqs, candles, tolerance, diag),
    }
    return snapped, diag
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest ml/tests/test_analysis_snap.py -v`

Expected: PASS — all classes (TestOrderBlockSnap, TestFvgSnap, TestLiquiditySnap, TestIdempotency) green.

- [ ] **Step 5: Commit**

```bash
git add ml/analysis_snap.py ml/tests/test_analysis_snap.py
git commit -m "feat: backend snap helper for OB/FVG/liquidity overlays"
```

---

## Task 6: Wire frontend snap into `runAnalysis`

**Files:**
- Modify: `src/App.jsx` (~line 1 imports, ~line 638-640 in `runAnalysis`)

- [ ] **Step 1: Add the import**

In `src/App.jsx`, find the existing `import` block near the top of the file (where `useChartScale` and other local imports are). Add:

```js
import { snapAnalysisToCandles, groupLiquidityByLevel } from "./analysisSnap.js";
```

Place it grouped with other `from "./..."` imports.

- [ ] **Step 2: Wire the snap into `runAnalysis`**

In `src/App.jsx`, find the block in `runAnalysis` that currently looks like:

```js
const parsed = JSON.parse(clean);
analysisCacheRef.current = { hash, result: parsed };
setAnalysis(parsed);
```

(Around line 638-640.) Replace with:

```js
const raw = JSON.parse(clean);
const { analysis: parsed, diagnostics } = snapAnalysisToCandles(raw, cds);
if (
  diagnostics.snapped_obs || diagnostics.snapped_fvgs || diagnostics.snapped_liquidity ||
  diagnostics.dropped_obs || diagnostics.dropped_fvgs || diagnostics.dropped_liquidity
) {
  console.warn("[analysis] overlay snap diagnostics:", diagnostics);
}
analysisCacheRef.current = { hash, result: parsed };
setAnalysis(parsed);
```

Note: `cds` is the candles list already in scope from `runAnalysis`'s top — verify it's the right binding. If renamed, use whatever local variable holds the 1H candles passed into the function.

- [ ] **Step 3: Run the existing test suite to confirm no regressions**

Run: `npm run test`

Expected: PASS — all existing tests pass (the new tests from Tasks 1-4 also pass; no existing test should break, since the snap is a no-op when analysis already matches candles, and our existing tests don't exercise the live `runAnalysis` flow).

- [ ] **Step 4: Build to confirm no syntax errors**

Run: `npm run build`

Expected: PASS — Vite build succeeds.

- [ ] **Step 5: Commit**

```bash
git add src/App.jsx
git commit -m "feat: snap analysis JSON before chart render"
```

---

## Task 7: Wire dedup into liquidity render loop

**Files:**
- Modify: `src/App.jsx` (~line 1089 in the chart-draw effect)

- [ ] **Step 1: Replace the liquidity forEach with a grouped pass**

In `src/App.jsx`, find the block that currently renders liquidity (around line 1089-1099):

```js
// Liquidity levels — anchored at formation candle, project forward
analysis.liquidity?.forEach((liq) => {
  const lCol = liq.type === "buyside" ? "#f5c842" : "#ff6b6b";
  const lci = Math.max(0, Math.min(liq.candleIndex ?? 0, candles.length - 1));
  const lx = x(lci) ?? 0;
  g.append("line").attr("x1", lx).attr("x2", w).attr("y1", y(liq.price)).attr("y2", y(liq.price))
    .attr("stroke", lCol).attr("stroke-width", 1).attr("stroke-dasharray", "7,4").attr("opacity", 0.8);
  const liqTfTag = liq.tf ? ` ${liq.tf}` : "";
  g.append("text").attr("x", w + 3).attr("y", y(liq.price) + 4)
    .attr("fill", lCol).attr("font-size", "8px").attr("font-family", "monospace")
    .text(`${liq.type === "buyside" ? "BSL" : "SSL"}${liqTfTag}`);
});
```

Replace with:

```js
// Liquidity levels — grouped by (type, price-bucket, tf) so duplicate labels collapse to ×N
const liqGroups = groupLiquidityByLevel(analysis.liquidity || [], 0.50);
liqGroups.forEach((group) => {
  const liq = group.items[0]; // representative — leftmost candleIndex after group sort
  const count = group.items.length;
  const lCol = liq.type === "buyside" ? "#f5c842" : "#ff6b6b";
  const lci = Math.max(0, Math.min(liq.candleIndex ?? 0, candles.length - 1));
  const lx = x(lci) ?? 0;
  g.append("line").attr("x1", lx).attr("x2", w).attr("y1", y(liq.price)).attr("y2", y(liq.price))
    .attr("stroke", lCol).attr("stroke-width", 1).attr("stroke-dasharray", "7,4").attr("opacity", 0.8);
  const liqTfTag = liq.tf ? ` ${liq.tf}` : "";
  const countTag = count > 1 ? ` ×${count}` : "";
  g.append("text").attr("x", w + 3).attr("y", y(liq.price) + 4)
    .attr("fill", lCol).attr("font-size", "8px").attr("font-family", "monospace")
    .text(`${liq.type === "buyside" ? "BSL" : "SSL"}${liqTfTag}${countTag}`);
});
```

- [ ] **Step 2: Run tests + build**

Run: `npm run test && npm run build`

Expected: PASS — all tests still green, build succeeds.

- [ ] **Step 3: Commit**

```bash
git add src/App.jsx
git commit -m "feat: dedup stacked liquidity labels with ×N count"
```

---

## Task 8: Wire backend snap into `/calibrate` (with integration test)

**Files:**
- Modify: `ml/server.py` (~line 777-804)
- Modify: `ml/tests/test_analysis_snap.py` (append integration test)

- [ ] **Step 1: Append the failing integration test**

Append to `ml/tests/test_analysis_snap.py`:

```python
class TestCalibrateEndpointIntegration:
    def test_calibrate_snaps_analysis_and_returns_diagnostics(self):
        """Hitting /calibrate with diverged OBs returns snap_diagnostics in the response."""
        from starlette.testclient import TestClient
        from ml.server import app

        # Build candles where candle[1] has high=4598.5, low=4584.0
        candles = [
            {"datetime": f"2026-04-30 {i:02d}:00:00",
             "open": 4580 + i, "high": 4598.5 if i == 1 else 4595 + i,
             "low": 4584.0 if i == 1 else 4580 + i, "close": 4590 + i}
            for i in range(60)
        ]
        # Claude returns OB with high/low diverged by ~$60 from candle[1] wick
        analysis = {
            "bias": "bullish",
            "orderBlocks": [
                {"type": "bullish", "high": 4540.0, "low": 4520.0, "candleIndex": 1, "tf": "1H"},
            ],
            "fvgs": [],
            "liquidity": [],
            "entry": {"price": 4590.0, "direction": "long", "rationale": "test"},
            "stopLoss": {"price": 4580.0, "rationale": "test"},
            "takeProfits": [{"price": 4610.0, "rationale": "test", "rr": 2.0}],
            "killzone": "London",
            "confluences": ["test"],
        }

        with TestClient(app) as client:
            r = client.post("/calibrate", json={"analysis": analysis, "candles": candles})

        assert r.status_code == 200
        body = r.json()
        assert "snap_diagnostics" in body
        assert body["snap_diagnostics"]["snapped_obs"] == 1
        assert body["snap_diagnostics"]["dropped_obs"] == 0
        # The first delta should record the OB snap
        deltas = body["snap_diagnostics"]["deltas"]
        assert any(d["kind"] == "ob" and d["candleIndex"] == 1 for d in deltas)

    def test_calibrate_no_snaps_for_aligned_analysis(self):
        """When analysis already matches candles, /calibrate returns zero-count diagnostics."""
        from starlette.testclient import TestClient
        from ml.server import app

        candles = [
            {"datetime": f"2026-04-30 {i:02d}:00:00",
             "open": 4580 + i, "high": 4595 + i, "low": 4580 + i, "close": 4590 + i}
            for i in range(60)
        ]
        analysis = {
            "bias": "bullish",
            "orderBlocks": [
                {"type": "bullish", "high": 4596, "low": 4581, "candleIndex": 1, "tf": "1H"},
            ],
            "fvgs": [],
            "liquidity": [],
            "entry": {"price": 4590.0, "direction": "long", "rationale": "test"},
            "stopLoss": {"price": 4580.0, "rationale": "test"},
            "takeProfits": [{"price": 4610.0, "rationale": "test", "rr": 2.0}],
            "killzone": "London",
            "confluences": ["test"],
        }

        with TestClient(app) as client:
            r = client.post("/calibrate", json={"analysis": analysis, "candles": candles})

        assert r.status_code == 200
        body = r.json()
        assert body["snap_diagnostics"]["snapped_obs"] == 0
        assert body["snap_diagnostics"]["dropped_obs"] == 0
```

- [ ] **Step 2: Run integration tests to verify they fail**

Run: `python -m pytest ml/tests/test_analysis_snap.py::TestCalibrateEndpointIntegration -v`

Expected: FAIL — `KeyError: 'snap_diagnostics'` (the endpoint doesn't add it yet).

- [ ] **Step 3: Wire the snap into `/calibrate`**

In `ml/server.py`, find the `/calibrate` endpoint (around line 777-804):

```python
@app.post("/calibrate")
async def calibrate_endpoint(request: dict):
    """Calibrate Claude's ICT analysis with ML layers.

    Body: {
        "analysis": { ... },    // Claude's ICT analysis JSON
        "candles": [ ... ],     // 1H OHLC candles
        "candles_4h": [ ... ]   // optional 4H candles
    }
    """
    from ml.claude_bridge import ClaudeAnalysisBridge
    from ml.calibrate import MLCalibrator

    analysis = request.get("analysis")
    candles = request.get("candles", [])

    if not analysis:
        raise HTTPException(status_code=400, detail="Missing 'analysis' field")
    if not candles:
        raise HTTPException(status_code=400, detail="Missing 'candles' field")

    bridge = ClaudeAnalysisBridge()
    parsed = bridge.parse_analysis(analysis, candles)

    calibrator = MLCalibrator()
    result = calibrator.calibrate_trade(parsed, candles)

    return result
```

Replace with:

```python
@app.post("/calibrate")
async def calibrate_endpoint(request: dict):
    """Calibrate Claude's ICT analysis with ML layers.

    Body: {
        "analysis": { ... },    // Claude's ICT analysis JSON
        "candles": [ ... ],     // 1H OHLC candles
        "candles_4h": [ ... ]   // optional 4H candles
    }
    """
    from ml.analysis_snap import snap_analysis_to_candles
    from ml.claude_bridge import ClaudeAnalysisBridge
    from ml.calibrate import MLCalibrator

    analysis = request.get("analysis")
    candles = request.get("candles", [])

    if not analysis:
        raise HTTPException(status_code=400, detail="Missing 'analysis' field")
    if not candles:
        raise HTTPException(status_code=400, detail="Missing 'candles' field")

    analysis, snap_diagnostics = snap_analysis_to_candles(analysis, candles)
    if (snap_diagnostics["snapped_obs"] or snap_diagnostics["snapped_fvgs"]
            or snap_diagnostics["snapped_liquidity"] or snap_diagnostics["dropped_obs"]
            or snap_diagnostics["dropped_fvgs"] or snap_diagnostics["dropped_liquidity"]):
        logger.warning("overlay snap diagnostics: %s", snap_diagnostics)

    bridge = ClaudeAnalysisBridge()
    parsed = bridge.parse_analysis(analysis, candles)

    calibrator = MLCalibrator()
    result = calibrator.calibrate_trade(parsed, candles)
    result["snap_diagnostics"] = snap_diagnostics

    return result
```

Verify `logger` is already imported at the top of `ml/server.py`. If not, add `import logging` and `logger = logging.getLogger(__name__)` near the other module-level setup.

- [ ] **Step 4: Run integration tests to verify they pass**

Run: `python -m pytest ml/tests/test_analysis_snap.py -v`

Expected: PASS — all unit + integration tests green.

- [ ] **Step 5: Run the full ML test suite to confirm no regressions**

Run: `python -m pytest ml/tests/ -v`

Expected: PASS — full suite passes (the snap is a no-op when analysis already matches candles, and existing test fixtures don't construct misaligned analyses).

- [ ] **Step 6: Commit**

```bash
git add ml/server.py ml/tests/test_analysis_snap.py
git commit -m "feat: snap analysis in /calibrate before bridge.parse_analysis"
```

---

## Task 9: Prompt tightening (Approach A) in both prompts

**Files:**
- Modify: `src/App.jsx` (~line 285 in `buildEnhancedICTPrompt`)
- Modify: `ml/prompts.py` (~line 188 — analysis framework section)

- [ ] **Step 1: Add framework item 12 to inline frontend prompt**

In `src/App.jsx`, find `buildEnhancedICTPrompt`. The existing analysis framework ends with item 9 (around line 285). After the line:

```
9. For every OB, FVG, and liquidity level you return, set "tf" to the timeframe where you identified it ("1H" or "4H"). Use 1H-relative candleIndex/startIndex even for 4H zones — find the 1H candle that aligns with the 4H zone's anchor time. Include 4H zones if they are within or near the visible 1H window and relevant to the setup.
```

Add a new line immediately after, before the `Return ONLY valid JSON:` line:

```
10. CRITICAL CONSISTENCY: Numeric fields you return MUST match the actual candle data, not paraphrased values from your prose. For each "orderBlocks[i]": "high" and "low" MUST equal the actual high and low of the candle at "candleIndex". For each "fvgs[i]": bullish → "low" = candle[startIndex].high, "high" = candle[startIndex+2].low; bearish → "high" = candle[startIndex].low, "low" = candle[startIndex+2].high. For each "liquidity[i]": "price" MUST equal candle[candleIndex].high (buyside) or candle[candleIndex].low (sellside). Do not round, paraphrase, or use values from a non-anchor candle. Mismatches will be silently corrected, but they signal you got the anchor candle wrong.
```

Note this is item 10 in the frontend prompt (since the inline prompt has 9 framework items, not 11 like the backend prompt). Numbering matches the local list — what matters is the rule, not its number.

- [ ] **Step 2: Add framework item 12 to backend prompt**

In `ml/prompts.py`, find the analysis framework block (around line 188). After the existing item 11:

```
11. For every OB, FVG, and liquidity level you return, set "tf" to the timeframe where you identified it (e.g. "1H" or "4H" — match the labels of the candle blocks above). HTF zones generally carry more weight than LTF zones; flag them so the system can weight them accordingly. Use execution-timeframe-relative candleIndex/startIndex even for HTF zones — find the execution candle that aligns with the HTF zone's anchor time.
```

Add immediately after, before the `Return ONLY valid JSON:` line:

```
12. CRITICAL CONSISTENCY: Numeric fields you return MUST match the actual candle data, not paraphrased values from your prose. For each "orderBlocks[i]": "high" and "low" MUST equal the actual high and low of the candle at "candleIndex". For each "fvgs[i]": bullish → "low" = candle[startIndex].high, "high" = candle[startIndex+2].low; bearish → "high" = candle[startIndex].low, "low" = candle[startIndex+2].high. For each "liquidity[i]": "price" MUST equal candle[candleIndex].high (buyside) or candle[candleIndex].low (sellside). Do not round, paraphrase, or use values from a non-anchor candle. Mismatches will be silently corrected, but they signal you got the anchor candle wrong.
```

Note: in `ml/prompts.py` the prompt uses `f"""..."""`-style strings with `{{` / `}}` escaping for JSON braces. Plain text (like our framework item) does NOT need brace escaping, but if the file uses double braces around literal JSON in the immediately-surrounding context, keep the format consistent — only the `Return ONLY valid JSON:` block requires `{{`/`}}`. Item 12 is plain text.

- [ ] **Step 3: Run prompt tests to confirm no regression**

Run: `python -m pytest ml/tests/test_prompts.py -v`

Expected: PASS — existing prompt tests should still pass (we added a line, didn't change semantics of existing items).

If a prompt test asserts on the count of framework items or a specific snippet near the end of the framework, the test may need to be updated to expect the new item. Update the assertion to match.

- [ ] **Step 4: Run vitest to confirm frontend prompt changes don't break anything**

Run: `npm run test`

Expected: PASS — vitest tests don't assert on prompt content.

- [ ] **Step 5: Commit**

```bash
git add src/App.jsx ml/prompts.py
git commit -m "feat: tighten OB/FVG/liquidity consistency in both ICT prompts"
```

---

## Task 10: Final verification

**Files:** none — verification only.

- [ ] **Step 1: Run the full vitest suite**

Run: `npm run test`

Expected: PASS — all vitest tests green.

- [ ] **Step 2: Run the full pytest suite**

Run: `python -m pytest ml/tests/ -v`

Expected: PASS — all ~1280 existing tests + the new ones from Tasks 5 and 8 green.

- [ ] **Step 3: Build the frontend bundle**

Run: `npm run build`

Expected: Vite build succeeds with no errors or warnings about the new module.

- [ ] **Step 4: Manual chart verification**

Boot the dev stack and observe live behavior:

```bash
# Terminal 1 — backend
source ~/dealfinder/bin/activate
cd ~/ict-terminal
python -m uvicorn ml.server:app --reload --port 8000

# Terminal 2 — frontend
cd ~/ict-terminal
npm run dev
```

Open the SPA, paste your Claude API key, switch off demo mode, load XAU/USD 1H. Run an analysis. Confirm in the browser:

- Each OB rectangle visibly sits on the wick of the candle at its claimed `candleIndex` — top of rectangle aligns with that candle's high, bottom with its low.
- Each FVG rectangle visibly spans the gap of its 3-candle pattern.
- Each liquidity line sits on the wick (top for BSL, bottom for SSL).
- If two BSLs cluster at the same level (same tf), a single `BSL 1H ×2` label is rendered with no stacked overlap.
- DevTools console: if any snaps occurred, `console.warn("[analysis] overlay snap diagnostics:", {...})` is logged with non-zero counts. If alignment was already correct, no warning fires.
- DevTools network panel: `/api/ml/calibrate` response body includes a `snap_diagnostics` field.

If any rectangle still looks misaligned, capture the analysis JSON from the network tab and the corresponding candles, save to a fixture, and add a regression test in `src/test/analysisSnap.test.js` before patching.

- [ ] **Step 5: Hand off**

No commit needed — verification only. Report back: tests passing, build clean, manual chart inspection confirms overlays sit on candle wicks, and snap diagnostics surface in console + `/calibrate` response. Mark the implementation complete.

---

## Self-review

**Spec coverage**

| Spec section | Covered by |
|---|---|
| Snap algorithm — OB | Task 1 (frontend), Task 5 (backend) |
| Snap algorithm — FVG bullish/bearish/degenerate | Task 2 (frontend), Task 5 (backend) |
| Snap algorithm — liquidity | Task 3 (frontend), Task 5 (backend) |
| OOB candleIndex drop semantics | Tasks 1, 2, 3, 5 |
| Diagnostics shape (snapped_*, dropped_*, deltas) | Tasks 1, 5 |
| Per-item `snapped: true` flag | Tasks 1, 2, 3, 5 |
| Frontend wiring (`runAnalysis` snap before `setAnalysis`) | Task 6 |
| Backend wiring (`/calibrate` snap before `bridge.parse_analysis` + `snap_diagnostics` in response) | Task 8 |
| Conditional `console.warn` / `logger.warning` | Tasks 6, 8 |
| Liquidity dedup (`groupLiquidityByLevel`, `×N` label) | Tasks 4, 7 |
| Prompt tightening (Approach A) in both prompts | Task 9 |
| Idempotency (snap on already-snapped → no-op) | Tasks 4, 5 |
| Tolerance = $0.50 | All snap tasks |
| Manual verification on live chart | Task 10 |

**Placeholder scan:** No TBD/TODO/"add appropriate error handling" patterns. Every step contains the actual code or command.

**Type / signature consistency:**
- `snapAnalysisToCandles(analysis, candles, options?)` returns `{ analysis, diagnostics }` — used identically in Tasks 1-4 (definition) and Task 6 (call site).
- `groupLiquidityByLevel(liquidity, tolerance)` returns `[{ key, items }]` — defined Task 4, used Task 7 (`group.items[0]`, `group.items.length`).
- `snap_analysis_to_candles(analysis, candles, tolerance=0.50)` returns `(snapped, diag)` — defined Task 5, used Task 8.
- Diagnostics field names (`snapped_obs`, `dropped_obs`, etc.) match across both languages and are referenced consistently in conditional log checks.

No gaps found — plan is consistent with the spec and self-consistent across tasks.
