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
