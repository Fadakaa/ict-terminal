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
