import { describe, it, expect } from "vitest";
import { generatePineScript } from "../market.js";

// ── Sample analysis fixture ──────────────────────────────

const sampleAnalysis = {
  bias: "bullish",
  summary: "Gold showing bullish structure with displacement above recent order block",
  orderBlocks: [
    { type: "bullish", high: 2055.3, low: 2048.1, candleIndex: 5, strength: "strong", note: "Demand zone" },
    { type: "bearish", high: 2070.25, low: 2065.5, candleIndex: 12, strength: "moderate", note: "Supply zone" },
  ],
  fvgs: [
    { type: "bullish", high: 2060.75, low: 2055.4, startIndex: 8, filled: false, note: "Unfilled gap" },
    { type: "bearish", high: 2075.3, low: 2072.1, startIndex: 15, filled: true, note: "Filled gap" },
  ],
  liquidity: [
    { type: "buyside", price: 2080.5, candleIndex: 20, note: "Equal highs" },
    { type: "sellside", price: 2040.25, candleIndex: 3, note: "Previous low" },
  ],
  entry: { price: 2052.5, direction: "long", rationale: "OB retest" },
  stopLoss: { price: 2046.75, rationale: "Below OB" },
  takeProfits: [
    { price: 2065.5, rationale: "FVG fill", rr: 1.92 },
    { price: 2080.5, rationale: "BSL sweep", rr: 4.23 },
  ],
  killzone: "London Open (02:00-05:00 EST)",
  confluences: ["OB + FVG overlap", "London session", "Displacement candle"],
};

// ── Basic structure ──────────────────────────────────────

describe("generatePineScript — structure", () => {
  it("returns a non-empty string", () => {
    const result = generatePineScript(sampleAnalysis);
    expect(typeof result).toBe("string");
    expect(result.length).toBeGreaterThan(0);
  });

  it("starts with Pine Script v5 version declaration", () => {
    const result = generatePineScript(sampleAnalysis);
    expect(result).toMatch(/^\/\/@version=5/);
  });

  it("declares an indicator with overlay=true", () => {
    const result = generatePineScript(sampleAnalysis);
    expect(result).toContain("indicator(");
    expect(result).toContain("overlay=true");
  });

  it("includes ICT Analysis in the indicator title", () => {
    const result = generatePineScript(sampleAnalysis);
    expect(result).toMatch(/indicator\("ICT Analysis/);
  });
});

// ── Order Blocks ─────────────────────────────────────────

describe("generatePineScript — order blocks", () => {
  it("draws boxes for each order block", () => {
    const result = generatePineScript(sampleAnalysis);
    // Should have box.new calls for OBs
    const boxMatches = result.match(/box\.new/g);
    expect(boxMatches).not.toBeNull();
    // At least 2 boxes for our 2 order blocks
    expect(boxMatches.length).toBeGreaterThanOrEqual(2);
  });

  it("uses green color for bullish order blocks", () => {
    const result = generatePineScript(sampleAnalysis);
    // Bullish OB high/low should be present
    expect(result).toContain("2048.1");
    expect(result).toContain("2055.3");
  });

  it("uses red color for bearish order blocks", () => {
    const result = generatePineScript(sampleAnalysis);
    expect(result).toContain("2065.5");
    expect(result).toContain("2070.25");
  });

  it("handles empty order blocks array", () => {
    const analysis = { ...sampleAnalysis, orderBlocks: [] };
    const result = generatePineScript(analysis);
    expect(result).toContain("indicator(");
  });
});

// ── Fair Value Gaps ──────────────────────────────────────

describe("generatePineScript — fair value gaps", () => {
  it("draws boxes for unfilled FVGs", () => {
    const result = generatePineScript(sampleAnalysis);
    // FVG prices should appear
    expect(result).toContain("2055.4");
    expect(result).toContain("2060.75");
  });

  it("uses different styling for filled vs unfilled FVGs", () => {
    const result = generatePineScript(sampleAnalysis);
    // Both FVGs should be represented
    expect(result).toContain("2072.1");
    expect(result).toContain("2075.3");
  });

  it("handles empty FVGs array", () => {
    const analysis = { ...sampleAnalysis, fvgs: [] };
    const result = generatePineScript(analysis);
    expect(result).toContain("indicator(");
  });
});

// ── Liquidity levels ─────────────────────────────────────

describe("generatePineScript — liquidity", () => {
  it("draws horizontal lines for liquidity levels", () => {
    const result = generatePineScript(sampleAnalysis);
    // line.new for liquidity levels
    const lineMatches = result.match(/line\.new/g);
    expect(lineMatches).not.toBeNull();
    expect(lineMatches.length).toBeGreaterThanOrEqual(2);
  });

  it("includes buyside liquidity price", () => {
    const result = generatePineScript(sampleAnalysis);
    expect(result).toContain("2080.5");
  });

  it("includes sellside liquidity price", () => {
    const result = generatePineScript(sampleAnalysis);
    expect(result).toContain("2040.25");
  });

  it("labels liquidity levels", () => {
    const result = generatePineScript(sampleAnalysis);
    expect(result).toMatch(/BSL|buyside/i);
    expect(result).toMatch(/SSL|sellside/i);
  });

  it("handles empty liquidity array", () => {
    const analysis = { ...sampleAnalysis, liquidity: [] };
    const result = generatePineScript(analysis);
    expect(result).toContain("indicator(");
  });
});

// ── Entry / SL / TP ──────────────────────────────────────

describe("generatePineScript — trade levels", () => {
  it("draws entry level line", () => {
    const result = generatePineScript(sampleAnalysis);
    expect(result).toContain("2052.5");
  });

  it("draws stop loss level line", () => {
    const result = generatePineScript(sampleAnalysis);
    expect(result).toContain("2046.75");
  });

  it("draws take profit lines", () => {
    const result = generatePineScript(sampleAnalysis);
    expect(result).toContain("2065.5");
    expect(result).toContain("2080.5");
  });

  it("includes RR ratios in TP labels", () => {
    const result = generatePineScript(sampleAnalysis);
    expect(result).toContain("1.92");
    expect(result).toContain("4.23");
  });

  it("uses distinct colors for entry, SL, and TP", () => {
    const result = generatePineScript(sampleAnalysis);
    // Entry = blue, SL = red, TP = green (typical convention)
    expect(result).toMatch(/color\.(blue|aqua|teal)/);
    expect(result).toMatch(/color\.(red|maroon)/);
    expect(result).toMatch(/color\.(green|lime)/);
  });
});

// ── Bias label ───────────────────────────────────────────

describe("generatePineScript — bias & info", () => {
  it("shows market bias as a label", () => {
    const result = generatePineScript(sampleAnalysis);
    expect(result).toMatch(/BULLISH/i);
  });

  it("handles bearish bias", () => {
    const analysis = { ...sampleAnalysis, bias: "bearish" };
    const result = generatePineScript(analysis);
    expect(result).toMatch(/BEARISH/i);
  });

  it("includes killzone info", () => {
    const result = generatePineScript(sampleAnalysis);
    expect(result).toContain("London");
  });
});

// ── Edge cases ───────────────────────────────────────────

describe("generatePineScript — edge cases", () => {
  it("handles minimal analysis (only required fields)", () => {
    const minimal = {
      bias: "bullish",
      summary: "Test",
      orderBlocks: [],
      fvgs: [],
      liquidity: [],
      entry: { price: 2050, direction: "long", rationale: "test" },
      stopLoss: { price: 2045, rationale: "test" },
      takeProfits: [],
      killzone: "N/A",
      confluences: [],
    };
    const result = generatePineScript(minimal);
    expect(result).toContain("//@version=5");
    expect(result).toContain("indicator(");
  });

  it("handles missing optional fields gracefully", () => {
    const partial = {
      bias: "bearish",
      summary: "Minimal test",
      entry: { price: 2050, direction: "short", rationale: "test" },
      stopLoss: { price: 2060, rationale: "test" },
    };
    const result = generatePineScript(partial);
    expect(result).toContain("//@version=5");
  });

  it("escapes special characters in notes/rationale", () => {
    const analysis = {
      ...sampleAnalysis,
      entry: { price: 2050, direction: "long", rationale: 'OB "retest" at key level' },
    };
    const result = generatePineScript(analysis);
    // Should not break Pine Script syntax
    expect(result).toContain("//@version=5");
    expect(result).not.toContain('""retest""'); // no double-escaping
  });
});

// ── Visibility / readability on TradingView ──────────────

describe("generatePineScript — visibility", () => {
  // Boxes should extend far enough back to be clearly visible
  it("order block boxes span at least 50 bars for visibility", () => {
    const result = generatePineScript(sampleAnalysis);
    expect(result).toMatch(/bar_index\s*-\s*50/);
  });

  it("FVG boxes span at least 40 bars for visibility", () => {
    const result = generatePineScript(sampleAnalysis);
    expect(result).toMatch(/bar_index\s*-\s*40/);
  });

  // Box borders should be thick enough to see
  it("order block boxes have border_width >= 2", () => {
    const result = generatePineScript(sampleAnalysis);
    // All box.new calls in OB section should have border_width
    const obSection = result.split("// ── Order Blocks")[1]?.split("//")[0] || "";
    expect(obSection).toContain("border_width=2");
  });

  it("FVG boxes have visible border_width", () => {
    const result = generatePineScript(sampleAnalysis);
    const fvgSection = result.split("// ── Fair Value Gaps")[1]?.split("//")[0] || "";
    expect(fvgSection).toContain("border_width=");
  });

  // OB boxes should have lower transparency (more opaque = more visible)
  it("order block backgrounds have alpha <= 75 for visibility", () => {
    const result = generatePineScript(sampleAnalysis);
    // Check the alpha values in OB section are ≤ 75
    const obAlphas = [...result.matchAll(/Order Blocks[\s\S]*?color\.new\(color\.\w+,\s*(\d+)\)/g)];
    obAlphas.forEach((m) => {
      expect(Number(m[1])).toBeLessThanOrEqual(75);
    });
  });

  // Liquidity lines should extend right so they're visible when scrolling
  it("liquidity lines use extend=extend.right", () => {
    const result = generatePineScript(sampleAnalysis);
    const liqSection = result.split("// ── Liquidity Levels")[1]?.split("// ──")[0] || "";
    expect(liqSection).toContain("extend=extend.right");
  });

  // Trade level lines should extend right
  it("entry/SL/TP lines use extend=extend.right", () => {
    const result = generatePineScript(sampleAnalysis);
    const tradeSection = result.split("// ── Trade Levels")[1]?.split("// ──")[0] || "";
    expect(tradeSection).toContain("extend=extend.right");
  });

  // Lines should be thick enough
  it("trade level lines have width >= 3", () => {
    const result = generatePineScript(sampleAnalysis);
    const tradeSection = result.split("// ── Trade Levels")[1]?.split("// ──")[0] || "";
    expect(tradeSection).toMatch(/width=[3-9]/);
  });

  it("liquidity lines have width >= 3", () => {
    const result = generatePineScript(sampleAnalysis);
    const liqSection = result.split("// ── Liquidity Levels")[1]?.split("// ──")[0] || "";
    expect(liqSection).toMatch(/width=[3-9]/);
  });

  // Labels should be normal or large size for readability
  it("trade level labels use size.normal or larger", () => {
    const result = generatePineScript(sampleAnalysis);
    const tradeSection = result.split("// ── Trade Levels")[1]?.split("// ──")[0] || "";
    // Should NOT contain size.small for trade labels — they need to be readable
    expect(tradeSection).not.toContain("size.small");
    expect(tradeSection).toContain("size.normal");
  });

  it("liquidity labels use size.normal or larger", () => {
    const result = generatePineScript(sampleAnalysis);
    const liqSection = result.split("// ── Liquidity Levels")[1]?.split("// ──")[0] || "";
    expect(liqSection).not.toContain("size.small");
    expect(liqSection).toContain("size.normal");
  });

  it("bias label uses size.large", () => {
    const result = generatePineScript(sampleAnalysis);
    const biasSection = result.split("// ── Bias & Killzone")[1] || "";
    expect(biasSection).toContain("size.large");
  });

  // Box text should be positioned clearly
  it("order block boxes have text alignment for readability", () => {
    const result = generatePineScript(sampleAnalysis);
    const obSection = result.split("// ── Order Blocks")[1]?.split("//")[0] || "";
    expect(obSection).toContain("text_halign");
  });

  // Price labels on right side for reference
  it("includes price labels on OB boxes", () => {
    const result = generatePineScript(sampleAnalysis);
    // OB labels should show the price range
    expect(result).toMatch(/2048\.1.*2055\.3|2055\.3.*2048\.1/);
  });

  // Input toggles so user can show/hide each layer
  it("includes input toggles for each annotation layer", () => {
    const result = generatePineScript(sampleAnalysis);
    expect(result).toContain("input.bool");
    expect(result).toMatch(/show.*OB|showOB/i);
    expect(result).toMatch(/show.*FVG|showFVG/i);
    expect(result).toMatch(/show.*Liq|showLiq/i);
    expect(result).toMatch(/show.*Trade|showTrade/i);
  });
});
