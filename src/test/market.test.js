import { describe, it, expect } from "vitest";
import {
  TF_OPTIONS,
  TF_CANDLES,
  TARGET_CANDLES,
  REFRESH_OPTIONS,
  filterWeekendCandles,
  trimToTarget,
  hashCandles,
  buildICTSystemMessage,
  buildICTPrompt,
  buildWFOCalibrationBlock,
} from "../market.js";

// ── Constants ─────────────────────────────────────────────

describe("TF_OPTIONS", () => {
  it("has all seven timeframes", () => {
    expect(TF_OPTIONS).toHaveLength(7);
    const values = TF_OPTIONS.map((t) => t.value);
    expect(values).toEqual(["5min", "15min", "30min", "1h", "2h", "4h", "1day"]);
  });
});

describe("TF_CANDLES (API request sizes)", () => {
  it("over-fetches for 15min to compensate for weekend filter", () => {
    expect(TF_CANDLES["15min"]).toBeGreaterThan(TARGET_CANDLES["15min"]);
  });

  it("over-fetches for 1h to compensate for weekend filter", () => {
    expect(TF_CANDLES["1h"]).toBeGreaterThan(TARGET_CANDLES["1h"]);
  });

  it("matches target for 4h and 1day (no weekend filtering)", () => {
    expect(TF_CANDLES["4h"]).toBe(TARGET_CANDLES["4h"]);
    expect(TF_CANDLES["1day"]).toBe(TARGET_CANDLES["1day"]);
  });
});

describe("TARGET_CANDLES (final display counts)", () => {
  it("15min targets 150 candles (~1.5 trading days)", () => {
    expect(TARGET_CANDLES["15min"]).toBe(150);
  });

  it("1h targets 100 candles (~4 trading days)", () => {
    expect(TARGET_CANDLES["1h"]).toBe(100);
  });
});

// ── Candle hashing ───────────────────────────────────────

describe("hashCandles", () => {
  const candles = [
    { datetime: "2026-03-02 10:00:00", open: 100, high: 105, low: 99, close: 103 },
    { datetime: "2026-03-02 10:15:00", open: 103, high: 108, low: 101, close: 106 },
    { datetime: "2026-03-02 10:30:00", open: 106, high: 110, low: 104, close: 109 },
  ];

  it("returns empty string for empty array", () => {
    expect(hashCandles([])).toBe("");
  });

  it("returns a deterministic hash for the same candles", () => {
    const h1 = hashCandles(candles);
    const h2 = hashCandles(candles);
    expect(h1).toBe(h2);
    expect(h1).not.toBe("");
  });

  it("changes when last candle close changes", () => {
    const modified = [...candles.slice(0, -1), { ...candles[2], close: 999 }];
    expect(hashCandles(modified)).not.toBe(hashCandles(candles));
  });

  it("changes when candle count changes", () => {
    const fewer = candles.slice(1);
    expect(hashCandles(fewer)).not.toBe(hashCandles(candles));
  });

  it("changes when first candle changes", () => {
    const modified = [{ ...candles[0], datetime: "2026-03-01 09:00:00" }, ...candles.slice(1)];
    expect(hashCandles(modified)).not.toBe(hashCandles(candles));
  });

  it("works with a single candle", () => {
    const single = [candles[0]];
    const h = hashCandles(single);
    expect(h).toBeTruthy();
    expect(h).toContain("1|"); // starts with count
  });

  it("does not mutate the input array", () => {
    const original = JSON.parse(JSON.stringify(candles));
    hashCandles(candles);
    expect(candles).toEqual(original);
  });
});

// ── Weekend filtering ─────────────────────────────────────

describe("filterWeekendCandles", () => {
  const weekdayCandles = [
    { datetime: "2026-03-02 10:00:00", open: 100, high: 105, low: 99, close: 103 }, // Monday
    { datetime: "2026-03-03 10:00:00", open: 103, high: 108, low: 101, close: 106 }, // Tuesday
    { datetime: "2026-03-04 10:00:00", open: 106, high: 110, low: 104, close: 109 }, // Wednesday
    { datetime: "2026-03-05 10:00:00", open: 109, high: 112, low: 107, close: 111 }, // Thursday
    { datetime: "2026-03-06 10:00:00", open: 111, high: 115, low: 110, close: 114 }, // Friday
  ];

  const weekendCandles = [
    { datetime: "2026-03-07 10:00:00", open: 114, high: 114, low: 114, close: 114 }, // Saturday
    { datetime: "2026-03-08 10:00:00", open: 114, high: 114, low: 114, close: 114 }, // Sunday
  ];

  const mixedCandles = [...weekdayCandles, ...weekendCandles];

  it("removes Saturday and Sunday candles for 15min timeframe", () => {
    const result = filterWeekendCandles(mixedCandles, "15min");
    expect(result).toHaveLength(5);
    result.forEach((c) => {
      const day = new Date(c.datetime).getDay();
      expect(day).not.toBe(0); // not Sunday
      expect(day).not.toBe(6); // not Saturday
    });
  });

  it("removes Saturday and Sunday candles for 1h timeframe", () => {
    const result = filterWeekendCandles(mixedCandles, "1h");
    expect(result).toHaveLength(5);
  });

  it("does NOT filter weekends for 4h timeframe", () => {
    const result = filterWeekendCandles(mixedCandles, "4h");
    expect(result).toHaveLength(7); // all candles kept
  });

  it("does NOT filter weekends for 1day timeframe", () => {
    const result = filterWeekendCandles(mixedCandles, "1day");
    expect(result).toHaveLength(7);
  });

  it("returns empty array for empty input", () => {
    expect(filterWeekendCandles([], "15min")).toEqual([]);
  });

  it("preserves candle data integrity (no mutation)", () => {
    const original = JSON.parse(JSON.stringify(mixedCandles));
    filterWeekendCandles(mixedCandles, "15min");
    expect(mixedCandles).toEqual(original);
  });
});

// ── Trim to target ────────────────────────────────────────

describe("trimToTarget", () => {
  const candles = Array.from({ length: 250 }, (_, i) => ({
    datetime: `2026-03-01 ${String(i).padStart(2, "0")}:00:00`,
    open: 100 + i,
    high: 105 + i,
    low: 99 + i,
    close: 103 + i,
  }));

  it("trims 15min to 150 most recent candles", () => {
    const result = trimToTarget(candles, "15min");
    expect(result).toHaveLength(150);
    // Should keep the LAST 150 candles (most recent)
    expect(result[result.length - 1]).toEqual(candles[candles.length - 1]);
    expect(result[0]).toEqual(candles[100]); // 250 - 150 = 100
  });

  it("trims 1h to 100 most recent candles", () => {
    const result = trimToTarget(candles, "1h");
    expect(result).toHaveLength(100);
    expect(result[result.length - 1]).toEqual(candles[candles.length - 1]);
  });

  it("does not trim if already under target", () => {
    const small = candles.slice(0, 50);
    const result = trimToTarget(small, "15min");
    expect(result).toHaveLength(50);
  });

  it("returns exactly target count when input matches", () => {
    const exact = candles.slice(0, 150);
    const result = trimToTarget(exact, "15min");
    expect(result).toHaveLength(150);
  });

  it("falls back to 100 for unknown timeframe", () => {
    const result = trimToTarget(candles, "unknown");
    expect(result).toHaveLength(100);
  });
});

// ── System message ───────────────────────────────────────

describe("buildICTSystemMessage", () => {
  const sys = buildICTSystemMessage();

  it("defines the ICT expert persona", () => {
    expect(sys).toContain("ICT (Inner Circle Trader)");
    expect(sys).toContain("XAU/USD");
  });

  it("includes core ICT methodology", () => {
    expect(sys).toContain("MARKET STRUCTURE");
    expect(sys).toContain("ORDER BLOCKS");
    expect(sys).toContain("FAIR VALUE GAPS");
    expect(sys).toContain("LIQUIDITY");
    expect(sys).toContain("PREMIUM/DISCOUNT");
    expect(sys).toContain("KILLZONES");
  });

  it("defines BOS and CHoCH concepts", () => {
    expect(sys).toContain("Break of Structure");
    expect(sys).toContain("Change of Character");
  });

  it("specifies OB validation criteria", () => {
    expect(sys).toContain("displacement");
    expect(sys).toContain("last bearish candle before a strong rally");
  });

  it("defines FVG as 3-candle imbalance", () => {
    expect(sys).toContain("3-candle imbalance");
    expect(sys).toContain("candle 3 low > candle 1 high");
  });

  it("includes killzone time windows", () => {
    expect(sys).toContain("London Open");
    expect(sys).toContain("New York AM");
  });

  it("sets minimum trade quality standards", () => {
    expect(sys).toContain("2:1 risk-reward");
    expect(sys).toContain("do not force a trade");
  });
});

// ── Prompt builder ────────────────────────────────────────

describe("buildICTPrompt", () => {
  const candles = [
    { datetime: "2026-03-07 10:00:00", open: 2050.5, high: 2055.3, low: 2048.1, close: 2053.7 },
    { datetime: "2026-03-07 10:15:00", open: 2053.7, high: 2058.0, low: 2052.0, close: 2056.5 },
  ];

  it("includes candle count and timeframe", () => {
    const prompt = buildICTPrompt(candles, "15min");
    expect(prompt).toContain("2 OHLC candles");
    expect(prompt).toContain("15min timeframe");
  });

  it("includes current price and equilibrium context", () => {
    const prompt = buildICTPrompt(candles, "1h");
    expect(prompt).toContain("Current price: 2056.5");
    expect(prompt).toContain("Equilibrium:");
    expect(prompt).toContain("Range:");
  });

  it("includes time period covered", () => {
    const prompt = buildICTPrompt(candles, "15min");
    expect(prompt).toContain("2026-03-07 10:00:00");
    expect(prompt).toContain("2026-03-07 10:15:00");
  });

  it("includes analysis checklist for systematic ICT review", () => {
    const prompt = buildICTPrompt(candles, "15min");
    expect(prompt).toContain("STRUCTURE:");
    expect(prompt).toContain("ORDER BLOCKS:");
    expect(prompt).toContain("FVGs:");
    expect(prompt).toContain("LIQUIDITY:");
    expect(prompt).toContain("PREMIUM/DISCOUNT:");
    expect(prompt).toContain("KILLZONE:");
    expect(prompt).toContain("TRADE SETUP:");
  });

  it("formats candle data as JSON with correct fields", () => {
    const prompt = buildICTPrompt(candles, "15min");
    expect(prompt).toContain('"i":0');
    expect(prompt).toContain('"dt":"2026-03-07 10:00:00"');
    expect(prompt).toContain('"o":2050.5');
    expect(prompt).toContain('"h":2055.3');
    expect(prompt).toContain('"l":2048.1');
    expect(prompt).toContain('"c":2053.7');
  });

  it("rounds prices to 2 decimal places", () => {
    const messyCandles = [
      { datetime: "2026-03-07 10:00:00", open: 2050.556, high: 2055.999, low: 2048.111, close: 2053.777 },
    ];
    const prompt = buildICTPrompt(messyCandles, "1h");
    expect(prompt).toContain('"o":2050.56');
    expect(prompt).toContain('"h":2056');
    expect(prompt).toContain('"l":2048.11');
    expect(prompt).toContain('"c":2053.78');
  });

  it("requests JSON response format with all required ICT fields", () => {
    const prompt = buildICTPrompt(candles, "4h");
    expect(prompt).toContain('"bias"');
    expect(prompt).toContain('"orderBlocks"');
    expect(prompt).toContain('"fvgs"');
    expect(prompt).toContain('"liquidity"');
    expect(prompt).toContain('"entry"');
    expect(prompt).toContain('"stopLoss"');
    expect(prompt).toContain('"takeProfits"');
    expect(prompt).toContain('"killzone"');
    expect(prompt).toContain('"confluences"');
  });

  it("includes mitigated field in orderBlocks schema", () => {
    const prompt = buildICTPrompt(candles, "15min");
    expect(prompt).toContain('"mitigated"');
  });
});

// ── buildWFOCalibrationBlock ─────────────────────────────
describe("buildWFOCalibrationBlock", () => {
  const wfo = {
    recommended_sl_atr: 1.8,
    recommended_tp_atr: [2.5, 4.0, 6.2],
    grade: "B",
    oos_win_rate: 0.62,
    oos_profit_factor: 1.45,
    timestamp: "2026-03-12T10:00:00Z",
  };
  const atr = 15.0;

  it("returns empty string when wfoReport is null", () => {
    expect(buildWFOCalibrationBlock(null, 15)).toBe("");
  });

  it("returns empty string when wfoReport is undefined", () => {
    expect(buildWFOCalibrationBlock(undefined, 15)).toBe("");
  });

  it("includes recommended SL ATR multiplier", () => {
    const block = buildWFOCalibrationBlock(wfo, atr);
    expect(block).toContain("1.8");
    expect(block).toContain("ATR");
  });

  it("includes all TP ATR multipliers", () => {
    const block = buildWFOCalibrationBlock(wfo, atr);
    expect(block).toContain("2.5");
    expect(block).toContain("4.0");
    expect(block).toContain("6.2");
  });

  it("includes system grade", () => {
    const block = buildWFOCalibrationBlock(wfo, atr);
    expect(block).toContain("B");
  });

  it("includes OOS win rate as percentage", () => {
    const block = buildWFOCalibrationBlock(wfo, atr);
    expect(block).toContain("62");
  });

  it("computes concrete SL distance from ATR", () => {
    const block = buildWFOCalibrationBlock(wfo, atr);
    // 1.8 * 15 = 27
    expect(block).toContain("27");
  });

  it("handles missing recommended_tp_atr gracefully", () => {
    const wfoNoTp = { ...wfo, recommended_tp_atr: undefined };
    const block = buildWFOCalibrationBlock(wfoNoTp, atr);
    expect(block).not.toContain("undefined");
    expect(block).toContain("1.8");
  });

  it("does not mutate the wfoReport input", () => {
    const copy = JSON.parse(JSON.stringify(wfo));
    buildWFOCalibrationBlock(wfo, atr);
    expect(wfo).toEqual(copy);
  });
});

// ── buildICTPrompt with WFO context ─────────────────────
describe("buildICTPrompt with WFO context", () => {
  const candles = [
    { datetime: "2026-03-07 10:00:00", open: 2050.5, high: 2055.3, low: 2048.1, close: 2053.7 },
    { datetime: "2026-03-07 10:15:00", open: 2053.7, high: 2058.0, low: 2052.0, close: 2056.5 },
  ];
  const wfo = {
    recommended_sl_atr: 1.8,
    recommended_tp_atr: [2.5, 4.0, 6.2],
    grade: "A",
    oos_win_rate: 0.55,
    oos_profit_factor: 1.6,
    timestamp: "2026-03-12T10:00:00Z",
  };

  it("includes WFO calibration block when wfoReport is provided", () => {
    const prompt = buildICTPrompt(candles, "1h", wfo);
    expect(prompt).toContain("WFO CALIBRATION");
    expect(prompt).toContain("1.8");
  });

  it("does NOT include WFO block when wfoReport is null", () => {
    const prompt = buildICTPrompt(candles, "1h", null);
    expect(prompt).not.toContain("WFO CALIBRATION");
  });

  it("does NOT include WFO block when wfoReport is omitted", () => {
    const prompt = buildICTPrompt(candles, "1h");
    expect(prompt).not.toContain("WFO CALIBRATION");
  });

  it("still includes all standard ICT checklist items with WFO", () => {
    const prompt = buildICTPrompt(candles, "1h", wfo);
    expect(prompt).toContain("STRUCTURE:");
    expect(prompt).toContain("ORDER BLOCKS:");
    expect(prompt).toContain("FVGs:");
    expect(prompt).toContain("LIQUIDITY:");
    expect(prompt).toContain("PREMIUM/DISCOUNT:");
    expect(prompt).toContain("KILLZONE:");
    expect(prompt).toContain("TRADE SETUP:");
  });

  it("still includes JSON schema with WFO", () => {
    const prompt = buildICTPrompt(candles, "1h", wfo);
    expect(prompt).toContain('"stopLoss"');
    expect(prompt).toContain('"takeProfits"');
    expect(prompt).toContain('"entry"');
  });

  it("references WFO-calibrated in checklist when WFO present", () => {
    const prompt = buildICTPrompt(candles, "1h", wfo);
    expect(prompt).toContain("WFO-calibrated");
  });

  it("does NOT reference WFO-calibrated in checklist when no WFO", () => {
    const prompt = buildICTPrompt(candles, "1h");
    expect(prompt).not.toContain("WFO-calibrated");
  });

  it("backward compatible: 2-arg call identical to 3-arg with null", () => {
    const promptTwo = buildICTPrompt(candles, "1h");
    const promptThree = buildICTPrompt(candles, "1h", null);
    expect(promptTwo).toBe(promptThree);
  });
});
