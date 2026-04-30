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
