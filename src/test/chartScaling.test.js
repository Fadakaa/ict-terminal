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
