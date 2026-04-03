import { describe, it, expect } from "vitest";
import { computeATR, extractMLFeatures, generateSetupId, formatMLPrediction } from "../market.js";

// ── Sample data ────────────────────────────────────────────

const sampleCandles = Array.from({ length: 100 }, (_, i) => {
  const o = 2600 + i * 0.5;
  return {
    datetime: `2026-03-${String(10 + Math.floor(i / 24)).padStart(2, "0")} ${String(i % 24).padStart(2, "0")}:00:00`,
    open: o,
    high: o + 3,
    low: o - 2,
    close: i % 2 === 0 ? o + 1 : o - 0.5,
  };
});

const sampleAnalysis = {
  bias: "bullish",
  summary: "Strong bullish structure",
  orderBlocks: [
    { type: "bullish", high: 2650, low: 2645, candleIndex: 80, strength: "strong", note: "" },
    { type: "bearish", high: 2680, low: 2675, candleIndex: 60, strength: "moderate", note: "" },
  ],
  fvgs: [
    { type: "bullish", high: 2660, low: 2655, startIndex: 85, filled: false, note: "" },
    { type: "bearish", high: 2670, low: 2668, startIndex: 70, filled: true, note: "" },
  ],
  liquidity: [
    { type: "buyside", price: 2690, candleIndex: 50, note: "" },
    { type: "sellside", price: 2630, candleIndex: 55, note: "" },
  ],
  entry: { price: 2650, direction: "long", rationale: "" },
  stopLoss: { price: 2643, rationale: "" },
  takeProfits: [
    { price: 2680, rationale: "", rr: 4.3 },
    { price: 2690, rationale: "", rr: 5.7 },
  ],
  killzone: "London Open",
  confluences: ["Bullish OB + FVG overlap", "Higher TF bullish", "Session timing"],
};


// ── computeATR tests ───────────────────────────────────────

describe("computeATR", () => {
  it("returns positive for valid candle data", () => {
    expect(computeATR(sampleCandles)).toBeGreaterThan(0);
  });

  it("returns 0 for insufficient data", () => {
    const short = Array(5).fill({ open: 100, high: 105, low: 95, close: 102 });
    expect(computeATR(short, 14)).toBe(0);
  });

  it("uses correct true range formula", () => {
    const candles = [
      { open: 100, high: 110, low: 90, close: 105 },
      { open: 105, high: 108, low: 102, close: 106 }, // TR=6
      { open: 106, high: 115, low: 100, close: 112 }, // TR=15
    ];
    expect(computeATR(candles, 2)).toBeCloseTo(10.5, 1);
  });

  it("returns 0 for single candle", () => {
    expect(computeATR([{ open: 100, high: 105, low: 95, close: 102 }])).toBe(0);
  });

  it("returns 0 for empty array", () => {
    expect(computeATR([])).toBe(0);
  });
});


// ── extractMLFeatures tests ────────────────────────────────

describe("extractMLFeatures", () => {
  it("returns object with 32 keys", () => {
    const f = extractMLFeatures(sampleAnalysis, sampleCandles, "1h");
    expect(Object.keys(f).length).toBe(32);
  });

  it("counts order blocks correctly", () => {
    const f = extractMLFeatures(sampleAnalysis, sampleCandles, "1h");
    expect(f.ob_count).toBe(2);
    expect(f.ob_bullish_count).toBe(1);
    expect(f.ob_bearish_count).toBe(1);
  });

  it("encodes OB strength (strong=3)", () => {
    const f = extractMLFeatures(sampleAnalysis, sampleCandles, "1h");
    expect(f.ob_strongest_strength).toBe(3);
  });

  it("computes positive OB nearest distance", () => {
    const f = extractMLFeatures(sampleAnalysis, sampleCandles, "1h");
    expect(f.ob_nearest_distance_atr).toBeGreaterThanOrEqual(0);
  });

  it("counts FVGs correctly", () => {
    const f = extractMLFeatures(sampleAnalysis, sampleCandles, "1h");
    expect(f.fvg_count).toBe(2);
    expect(f.fvg_unfilled_count).toBe(1);
  });

  it("counts liquidity levels", () => {
    const f = extractMLFeatures(sampleAnalysis, sampleCandles, "1h");
    expect(f.liq_buyside_count).toBe(1);
    expect(f.liq_sellside_count).toBe(1);
  });

  it("extracts risk-reward from take profits", () => {
    const f = extractMLFeatures(sampleAnalysis, sampleCandles, "1h");
    expect(f.risk_reward_tp1).toBeCloseTo(4.3);
    expect(f.risk_reward_tp2).toBeCloseTo(5.7);
  });

  it("encodes long direction as 1", () => {
    const f = extractMLFeatures(sampleAnalysis, sampleCandles, "1h");
    expect(f.entry_direction).toBe(1);
  });

  it("encodes short direction as 0", () => {
    const a = { ...sampleAnalysis, bias: "bearish", entry: { price: 2680, direction: "short", rationale: "" } };
    const f = extractMLFeatures(a, sampleCandles, "1h");
    expect(f.entry_direction).toBe(0);
  });

  it("detects bias-direction match", () => {
    const f = extractMLFeatures(sampleAnalysis, sampleCandles, "1h");
    expect(f.bias_direction_match).toBe(1);
  });

  it("counts confluences", () => {
    const f = extractMLFeatures(sampleAnalysis, sampleCandles, "1h");
    expect(f.num_confluences).toBe(3);
  });

  it("encodes London killzone as 1", () => {
    const f = extractMLFeatures(sampleAnalysis, sampleCandles, "1h");
    expect(f.killzone_encoded).toBe(1);
  });

  it("encodes timeframes correctly", () => {
    expect(extractMLFeatures(sampleAnalysis, sampleCandles, "15min").timeframe_encoded).toBe(1);
    expect(extractMLFeatures(sampleAnalysis, sampleCandles, "1h").timeframe_encoded).toBe(2);
    expect(extractMLFeatures(sampleAnalysis, sampleCandles, "4h").timeframe_encoded).toBe(3);
    expect(extractMLFeatures(sampleAnalysis, sampleCandles, "1day").timeframe_encoded).toBe(4);
  });

  it("handles empty analysis gracefully", () => {
    const empty = {
      bias: "neutral", orderBlocks: [], fvgs: [], liquidity: [],
      entry: null, stopLoss: null, takeProfits: [], killzone: "", confluences: [],
    };
    const f = extractMLFeatures(empty, sampleCandles, "1h");
    expect(f.ob_count).toBe(0);
    expect(f.entry_direction).toBe(0);
    expect(Object.keys(f).length).toBe(32);
  });

  it("does not mutate inputs", () => {
    const aBefore = JSON.stringify(sampleAnalysis);
    const cBefore = JSON.stringify(sampleCandles);
    extractMLFeatures(sampleAnalysis, sampleCandles, "1h");
    expect(JSON.stringify(sampleAnalysis)).toBe(aBefore);
    expect(JSON.stringify(sampleCandles)).toBe(cBefore);
  });

  it("computes ATR-based features", () => {
    const f = extractMLFeatures(sampleAnalysis, sampleCandles, "1h");
    expect(f.atr_14).toBeGreaterThan(0);
    expect(typeof f.price_vs_20sma).toBe("number");
    expect(typeof f.recent_volatility_ratio).toBe("number");
  });
});


// ── generateSetupId tests ──────────────────────────────────

describe("generateSetupId", () => {
  it("returns unique IDs", () => {
    const a = generateSetupId();
    const b = generateSetupId();
    expect(a).not.toBe(b);
  });

  it("returns string format", () => {
    expect(typeof generateSetupId()).toBe("string");
    expect(generateSetupId().length).toBeGreaterThan(5);
  });
});


// ── formatMLPrediction tests ───────────────────────────────

describe("formatMLPrediction", () => {
  it("formats confidence as percentage", () => {
    const text = formatMLPrediction({ confidence: 0.78, model_status: "trained" });
    expect(text).toContain("78");
  });

  it("returns N/A for cold start", () => {
    const text = formatMLPrediction({ confidence: 0, model_status: "cold_start" });
    expect(text).toContain("N/A");
  });

  it("returns learning message for insufficient data", () => {
    const text = formatMLPrediction({ confidence: 0, model_status: "insufficient_data", training_samples: 12 });
    expect(text).toMatch(/12/);
  });
});
