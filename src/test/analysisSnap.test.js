import { describe, it, expect } from "vitest";
import { snapAnalysisToCandles, groupLiquidityByLevel } from "../analysisSnap.js";

const c = (h, l, dt = "2026-04-30 12:00") => ({ datetime: dt, open: l, high: h, low: l, close: h });

describe("snapAnalysisToCandles — orderBlocks", () => {
  it("snaps high/low when both diverge by more than $0.50", () => {
    const candles = [c(100, 90), c(110, 95), c(120, 105)];
    const analysis = {
      orderBlocks: [{ type: "bearish", high: 80, low: 70, candleIndex: 1 }],
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
      orderBlocks: [{ type: "bearish", high: 110, low: 95, candleIndex: 1 }],
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
      orderBlocks: [{ type: "bearish", high: 110, low: 80, candleIndex: 0 }],
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
    const candles = [c(110, 90), c(95, 85), c(85, 80), c(88, 78)];
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

  it("drops bearish FVG when expected_low >= expected_high (no real gap)", () => {
    // c0.low=90, c2.high=110 → expected (high=90, low=110), degenerate
    const candles = [c(110, 90), c(105, 95), c(110, 95)];
    const analysis = {
      fvgs: [{ type: "bearish", high: 90, low: 80, startIndex: 0 }],
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

  it("groups bin-edge straddlers within tolerance (4630.49 and 4630.51)", () => {
    // bucket-based grouping would split these (4630.49→4630.50 bucket, 4630.51→4631.00 bucket).
    // Single-link clustering correctly groups them as 0.02 apart.
    const liq = [
      { type: "buyside", price: 4630.49, candleIndex: 5, tf: "1H" },
      { type: "buyside", price: 4630.51, candleIndex: 9, tf: "1H" },
    ];
    const groups = groupLiquidityByLevel(liq, 0.50);
    expect(groups).toHaveLength(1);
    expect(groups[0].items).toHaveLength(2);
  });
});

describe("snapAnalysisToCandles — idempotency", () => {
  it("is a no-op when analysis already matches candles (round-trip)", () => {
    const candles = [c(100, 90), c(110, 95), c(120, 105), c(130, 115)];
    const analysis = {
      bias: "bullish",
      orderBlocks: [{ type: "bearish", high: 110, low: 95, candleIndex: 1, tf: "1H" }],
      fvgs: [{ type: "bullish", high: 105, low: 100, startIndex: 0, tf: "1H" }],
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
    expect(second.analysis).toEqual(out);
  });
});

describe("snapAnalysisToCandles — anchor_dt resolution", () => {
  const candleAt = (h, l, dt) => ({ datetime: dt, open: l, high: h, low: l, close: h });
  const buildCandles = () => [
    candleAt(100, 90, "2026-04-30 08:00"),
    candleAt(110, 95, "2026-04-30 09:00"),
    candleAt(120, 105, "2026-04-30 10:00"),
    candleAt(130, 115, "2026-04-30 11:00"),
  ];

  it("resolves OB anchor_dt to correct numeric candleIndex", () => {
    const candles = buildCandles();
    const analysis = {
      orderBlocks: [{ type: "bearish", high: 110, low: 95, anchor_dt: "2026-04-30 09:00" }],
    };
    const { analysis: out, diagnostics } = snapAnalysisToCandles(analysis, candles);
    expect(out.orderBlocks).toHaveLength(1);
    expect(out.orderBlocks[0].candleIndex).toBe(1);
    expect(out.orderBlocks[0].anchor_dt).toBe("2026-04-30 09:00");
    expect(diagnostics.snapped_obs).toBe(0);
  });

  it("drops OB and increments unresolved_anchor when anchor_dt doesn't match any candle", () => {
    const candles = buildCandles();
    const analysis = {
      orderBlocks: [{ type: "bullish", high: 110, low: 95, anchor_dt: "2099-01-01 00:00" }],
    };
    const { analysis: out, diagnostics } = snapAnalysisToCandles(analysis, candles);
    expect(out.orderBlocks).toHaveLength(0);
    expect(diagnostics.dropped_obs).toBe(1);
    expect(diagnostics.unresolved_anchor).toBe(1);
  });

  it("anchor_dt wins over legacy candleIndex when both are present", () => {
    const candles = buildCandles();
    const analysis = {
      orderBlocks: [{
        type: "bearish",
        high: 130, low: 115,                       // matches index 3 (candleAt 11:00)
        candleIndex: 0,                            // wrong index
        anchor_dt: "2026-04-30 11:00",            // correct datetime
      }],
    };
    const { analysis: out, diagnostics } = snapAnalysisToCandles(analysis, candles);
    expect(out.orderBlocks[0].candleIndex).toBe(3);
    expect(diagnostics.snapped_obs).toBe(0);  // values already match candle[3]
  });

  it("falls back to legacy candleIndex when anchor_dt is missing", () => {
    const candles = buildCandles();
    const analysis = {
      orderBlocks: [{ type: "bearish", high: 110, low: 95, candleIndex: 1 }],
    };
    const { analysis: out, diagnostics } = snapAnalysisToCandles(analysis, candles);
    expect(out.orderBlocks).toHaveLength(1);
    expect(out.orderBlocks[0].candleIndex).toBe(1);
    expect(diagnostics.unresolved_anchor).toBe(0);
  });

  it("snaps high/low when anchor_dt resolves but values diverge", () => {
    const candles = buildCandles();
    const analysis = {
      orderBlocks: [{ type: "bearish", high: 200, low: 50, anchor_dt: "2026-04-30 09:00" }],
    };
    const { analysis: out, diagnostics } = snapAnalysisToCandles(analysis, candles);
    expect(out.orderBlocks[0].high).toBe(110);
    expect(out.orderBlocks[0].low).toBe(95);
    expect(out.orderBlocks[0].snapped).toBe(true);
    expect(diagnostics.snapped_obs).toBe(1);
  });

  it("resolves FVG anchor_dt to correct startIndex", () => {
    const candles = buildCandles();
    const analysis = {
      // bullish FVG: anchor=09:00 (idx 1), so c0=idx1, c2=idx3 → expectedLow=110, expectedHigh=115
      fvgs: [{
        type: "bullish",
        high: 115, low: 110,
        anchor_dt: "2026-04-30 09:00",
      }],
    };
    const { analysis: out, diagnostics } = snapAnalysisToCandles(analysis, candles);
    expect(out.fvgs[0].startIndex).toBe(1);
    expect(diagnostics.snapped_fvgs).toBe(0);
  });

  it("drops FVG when anchor_dt+2 is out of bounds", () => {
    const candles = buildCandles();
    // Only 4 candles. anchor_dt at idx 3 means startIndex+2 = 5 → OOB.
    const analysis = {
      fvgs: [{ type: "bullish", high: 130, low: 115, anchor_dt: "2026-04-30 11:00" }],
    };
    const { analysis: out, diagnostics } = snapAnalysisToCandles(analysis, candles);
    expect(out.fvgs).toHaveLength(0);
    expect(diagnostics.dropped_fvgs).toBe(1);
  });

  it("resolves liquidity anchor_dt", () => {
    const candles = buildCandles();
    const analysis = {
      liquidity: [{ type: "buyside", price: 130, anchor_dt: "2026-04-30 11:00" }],
    };
    const { analysis: out } = snapAnalysisToCandles(analysis, candles);
    expect(out.liquidity[0].candleIndex).toBe(3);
    expect(out.liquidity[0].anchor_dt).toBe("2026-04-30 11:00");
  });
});

describe("snapAnalysisToCandles — OB color validation", () => {
  const bullishCandle = (h, l, dt = "2026-04-30 10:00") => ({
    datetime: dt, open: l, high: h, low: l, close: h,  // close > open
  });
  const bearishCandle = (h, l, dt = "2026-04-30 10:00") => ({
    datetime: dt, open: h, high: h, low: l, close: l,  // close < open
  });
  const dojiCandle = (h, l, dt = "2026-04-30 10:00") => ({
    datetime: dt, open: (h + l) / 2, high: h, low: l, close: (h + l) / 2,
  });

  it("drops bullish OB anchored to a bullish (up-closed) candle", () => {
    const candles = [bullishCandle(110, 95)];
    const analysis = {
      orderBlocks: [{ type: "bullish", high: 110, low: 95, candleIndex: 0 }],
    };
    const { analysis: out, diagnostics } = snapAnalysisToCandles(analysis, candles);
    expect(out.orderBlocks).toHaveLength(0);
    expect(diagnostics.dropped_obs).toBe(1);
    expect(diagnostics.wrong_color_obs).toBe(1);
  });

  it("drops bearish OB anchored to a bearish (down-closed) candle", () => {
    const candles = [bearishCandle(110, 95)];
    const analysis = {
      orderBlocks: [{ type: "bearish", high: 110, low: 95, candleIndex: 0 }],
    };
    const { analysis: out, diagnostics } = snapAnalysisToCandles(analysis, candles);
    expect(out.orderBlocks).toHaveLength(0);
    expect(diagnostics.dropped_obs).toBe(1);
    expect(diagnostics.wrong_color_obs).toBe(1);
  });

  it("keeps bullish OB anchored to a bearish (down-closed) candle (correct ICT)", () => {
    const candles = [bearishCandle(110, 95)];
    const analysis = {
      orderBlocks: [{ type: "bullish", high: 110, low: 95, candleIndex: 0 }],
    };
    const { analysis: out, diagnostics } = snapAnalysisToCandles(analysis, candles);
    expect(out.orderBlocks).toHaveLength(1);
    expect(diagnostics.dropped_obs).toBe(0);
    expect(diagnostics.wrong_color_obs).toBe(0);
  });

  it("keeps bearish OB anchored to a bullish (up-closed) candle (correct ICT)", () => {
    const candles = [bullishCandle(110, 95)];
    const analysis = {
      orderBlocks: [{ type: "bearish", high: 110, low: 95, candleIndex: 0 }],
    };
    const { analysis: out, diagnostics } = snapAnalysisToCandles(analysis, candles);
    expect(out.orderBlocks).toHaveLength(1);
    expect(diagnostics.dropped_obs).toBe(0);
    expect(diagnostics.wrong_color_obs).toBe(0);
  });

  it("accepts doji candles (close === open) for either OB type", () => {
    const candles = [dojiCandle(110, 95)];
    const bullAnalysis = {
      orderBlocks: [{ type: "bullish", high: 110, low: 95, candleIndex: 0 }],
    };
    const { analysis: bullOut, diagnostics: bullDiag } = snapAnalysisToCandles(bullAnalysis, candles);
    expect(bullOut.orderBlocks).toHaveLength(1);
    expect(bullDiag.wrong_color_obs).toBe(0);

    const bearAnalysis = {
      orderBlocks: [{ type: "bearish", high: 110, low: 95, candleIndex: 0 }],
    };
    const { analysis: bearOut, diagnostics: bearDiag } = snapAnalysisToCandles(bearAnalysis, candles);
    expect(bearOut.orderBlocks).toHaveLength(1);
    expect(bearDiag.wrong_color_obs).toBe(0);
  });

  it("does not validate FVG color (FVGs ignore color rules)", () => {
    const candles = [
      bullishCandle(100, 90, "2026-04-30 10:00"),
      bullishCandle(115, 110, "2026-04-30 11:00"),
      bullishCandle(125, 120, "2026-04-30 12:00"),
    ];
    const analysis = {
      fvgs: [{ type: "bullish", high: 120, low: 100, startIndex: 0 }],
    };
    const { analysis: out, diagnostics } = snapAnalysisToCandles(analysis, candles);
    expect(out.fvgs).toHaveLength(1);
    expect(diagnostics.wrong_color_obs).toBe(0);
  });

  it("does not validate liquidity color (liquidity ignores color rules)", () => {
    const candles = [bullishCandle(110, 95)];
    const analysis = {
      liquidity: [{ type: "sellside", price: 95, candleIndex: 0 }],
    };
    const { analysis: out, diagnostics } = snapAnalysisToCandles(analysis, candles);
    expect(out.liquidity).toHaveLength(1);
    expect(diagnostics.wrong_color_obs).toBe(0);
  });
});
