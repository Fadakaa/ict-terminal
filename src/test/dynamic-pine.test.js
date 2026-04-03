import { describe, it, expect } from "vitest";
import { generateDynamicPineScript } from "../market.js";

// ── Basic structure ──────────────────────────────────────

describe("generateDynamicPineScript — structure", () => {
  it("returns a non-empty string", () => {
    const result = generateDynamicPineScript();
    expect(typeof result).toBe("string");
    expect(result.length).toBeGreaterThan(0);
  });

  it("starts with Pine Script v5 version declaration", () => {
    const result = generateDynamicPineScript();
    expect(result).toMatch(/^\/\/@version=5/);
  });

  it("declares an indicator with overlay=true", () => {
    const result = generateDynamicPineScript();
    expect(result).toContain("indicator(");
    expect(result).toContain("overlay=true");
  });

  it("includes ICT Dynamic in the indicator title", () => {
    const result = generateDynamicPineScript();
    expect(result).toMatch(/indicator\("ICT Dynamic/);
  });

  it("sets max_boxes_count, max_lines_count, max_labels_count", () => {
    const result = generateDynamicPineScript();
    expect(result).toContain("max_boxes_count=");
    expect(result).toContain("max_lines_count=");
    expect(result).toContain("max_labels_count=");
  });
});

// ── Input toggles ────────────────────────────────────────

describe("generateDynamicPineScript — input toggles", () => {
  it("has input toggle for Order Blocks", () => {
    const result = generateDynamicPineScript();
    expect(result).toMatch(/input\.bool\(true,\s*"Show Order Blocks"\)/);
  });

  it("has input toggle for Fair Value Gaps", () => {
    const result = generateDynamicPineScript();
    expect(result).toMatch(/input\.bool\(true,\s*"Show Fair Value Gaps"\)/);
  });

  it("has input toggle for Liquidity Levels", () => {
    const result = generateDynamicPineScript();
    expect(result).toMatch(/input\.bool\(true,\s*"Show Liquidity Levels"\)/);
  });

  it("has input toggle for Bias", () => {
    const result = generateDynamicPineScript();
    expect(result).toMatch(/input\.bool\(true,\s*"Show Bias/);
  });

  it("has configurable swing lookback parameter", () => {
    const result = generateDynamicPineScript();
    expect(result).toMatch(/input\.int\(/);
    expect(result).toMatch(/Swing/i);
  });

  it("has configurable OB displacement strength parameter", () => {
    const result = generateDynamicPineScript();
    expect(result).toMatch(/input\.float\(/);
    expect(result).toMatch(/Displacement|OB Strength/i);
  });
});

// ── ATR for dynamic thresholds ───────────────────────────

describe("generateDynamicPineScript — ATR", () => {
  it("uses ATR for dynamic threshold calculation", () => {
    const result = generateDynamicPineScript();
    expect(result).toMatch(/ta\.atr\(/);
  });
});

// ── Order Block detection ────────────────────────────────

describe("generateDynamicPineScript — order blocks", () => {
  it("detects bullish order blocks (bearish candle before bullish displacement)", () => {
    const result = generateDynamicPineScript();
    // Should reference close < open (bearish candle) pattern
    expect(result).toMatch(/close\[1\]\s*<\s*open\[1\]|close\s*<\s*open/);
    // And bullish displacement
    expect(result).toMatch(/close\s*>\s*open|close\[1\]\s*>\s*open\[1\]/);
  });

  it("detects bearish order blocks (bullish candle before bearish displacement)", () => {
    const result = generateDynamicPineScript();
    expect(result).toMatch(/bearOB|bear_ob|bearish.*OB/i);
  });

  it("draws boxes for detected order blocks using box.new", () => {
    const result = generateDynamicPineScript();
    // OB section should use box.new
    const obSection = result.split("Order Block Detection")[1]?.split("Fair Value Gap")[0] || "";
    expect(obSection).toContain("box.new");
  });

  it("uses green for bullish OBs and red for bearish OBs", () => {
    const result = generateDynamicPineScript();
    expect(result).toMatch(/color\.green.*Bull.*OB|Bull.*OB.*color\.green/s);
    expect(result).toMatch(/color\.red.*Bear.*OB|Bear.*OB.*color\.red/s);
  });

  it("conditions OB drawing on showOB toggle", () => {
    const result = generateDynamicPineScript();
    expect(result).toMatch(/showOB\s+and\s+bullOB|showOB and bullOB/);
    expect(result).toMatch(/showOB\s+and\s+bearOB|showOB and bearOB/);
  });
});

// ── Fair Value Gap detection ─────────────────────────────

describe("generateDynamicPineScript — fair value gaps", () => {
  it("detects bullish FVGs (low > high[2] gap)", () => {
    const result = generateDynamicPineScript();
    expect(result).toMatch(/low\s*>\s*high\[2\]/);
  });

  it("detects bearish FVGs (high < low[2] gap)", () => {
    const result = generateDynamicPineScript();
    expect(result).toMatch(/high\s*<\s*low\[2\]/);
  });

  it("draws boxes for detected FVGs using box.new", () => {
    const result = generateDynamicPineScript();
    const fvgSection = result.split("Fair Value Gap Detection")[1]?.split("Liquidity Levels —")[0] || "";
    expect(fvgSection).toContain("box.new");
  });

  it("uses aqua for bullish FVGs and orange for bearish FVGs", () => {
    const result = generateDynamicPineScript();
    expect(result).toMatch(/color\.aqua.*Bull.*FVG|Bull.*FVG.*color\.aqua/s);
    expect(result).toMatch(/color\.orange.*Bear.*FVG|Bear.*FVG.*color\.orange/s);
  });

  it("conditions FVG drawing on showFVG toggle", () => {
    const result = generateDynamicPineScript();
    expect(result).toMatch(/showFVG\s+and\s+bullFVG|showFVG and bullFVG/);
    expect(result).toMatch(/showFVG\s+and\s+bearFVG|showFVG and bearFVG/);
  });

  it("filters FVGs by minimum size relative to ATR", () => {
    const result = generateDynamicPineScript();
    // FVG detection should reference atr
    const fvgSection = result.split("Fair Value Gap Detection")[1]?.split("Liquidity Levels —")[0] || "";
    expect(fvgSection).toMatch(/atr/i);
  });
});

// ── Liquidity detection ──────────────────────────────────

describe("generateDynamicPineScript — liquidity", () => {
  it("uses pivothigh for buyside liquidity detection", () => {
    const result = generateDynamicPineScript();
    expect(result).toMatch(/ta\.pivothigh\(/);
  });

  it("uses pivotlow for sellside liquidity detection", () => {
    const result = generateDynamicPineScript();
    expect(result).toMatch(/ta\.pivotlow\(/);
  });

  it("draws dashed lines for liquidity levels", () => {
    const result = generateDynamicPineScript();
    const liqSection = result.split("Liquidity Levels —")[1]?.split("Bias —")[0] || "";
    expect(liqSection).toContain("line.new");
    expect(liqSection).toContain("line.style_dashed");
  });

  it("labels BSL for buyside and SSL for sellside", () => {
    const result = generateDynamicPineScript();
    expect(result).toContain("BSL");
    expect(result).toContain("SSL");
  });

  it("extends liquidity lines to the right", () => {
    const result = generateDynamicPineScript();
    const liqSection = result.split("Liquidity Levels —")[1]?.split("Bias —")[0] || "";
    expect(liqSection).toContain("extend=extend.right");
  });

  it("conditions liquidity on showLiq toggle", () => {
    const result = generateDynamicPineScript();
    expect(result).toMatch(/showLiq\s+and/);
  });
});

// ── Bias / Market Structure ──────────────────────────────

describe("generateDynamicPineScript — bias", () => {
  it("tracks swing highs and lows for structure", () => {
    const result = generateDynamicPineScript();
    const biasSection = result.split("Bias —")[1] || "";
    expect(biasSection).toMatch(/lastSwingHigh|lastHH|swingHigh/);
    expect(biasSection).toMatch(/lastSwingLow|lastLL|swingLow/);
  });

  it("determines bullish/bearish bias from structure breaks", () => {
    const result = generateDynamicPineScript();
    expect(result).toContain('"bullish"');
    expect(result).toContain('"bearish"');
  });

  it("shows bias label with small size for chart visibility", () => {
    const result = generateDynamicPineScript();
    const biasSection = result.split("Bias —")[1] || "";
    expect(biasSection).toContain("size=size.small");
    expect(biasSection).not.toContain("size.large");
  });

  it("shows bias only on last bar", () => {
    const result = generateDynamicPineScript();
    const biasSection = result.split("Bias —")[1] || "";
    expect(biasSection).toContain("barstate.islast");
  });

  it("conditions bias label on showBias toggle", () => {
    const result = generateDynamicPineScript();
    expect(result).toMatch(/showBias\s+and\s+barstate\.islast/);
  });
});

// ── Trade Levels (Entry / SL / TP) ───────────────────────

describe("generateDynamicPineScript — trade levels", () => {
  it("has input toggle for trade levels", () => {
    const result = generateDynamicPineScript();
    expect(result).toMatch(/input\.bool\(true,\s*"Show Trade Levels/);
  });

  it("tracks last bullish OB zone for long entries", () => {
    const result = generateDynamicPineScript();
    const tradeSection = result.split("Trade Levels —")[1] || "";
    expect(tradeSection).toMatch(/lastBullOB|lastBullOBHigh|lastBullOB_hi/i);
  });

  it("tracks last bearish OB zone for short entries", () => {
    const result = generateDynamicPineScript();
    const tradeSection = result.split("Trade Levels —")[1] || "";
    expect(tradeSection).toMatch(/lastBearOB|lastBearOBLow|lastBearOB_lo/i);
  });

  it("sets entry price based on bias and nearest OB", () => {
    const result = generateDynamicPineScript();
    const tradeSection = result.split("Trade Levels —")[1] || "";
    expect(tradeSection).toContain("entryPrice");
  });

  it("sets stop loss below/above OB zone", () => {
    const result = generateDynamicPineScript();
    const tradeSection = result.split("Trade Levels —")[1] || "";
    expect(tradeSection).toContain("slPrice");
  });

  it("sets take profit at liquidity target", () => {
    const result = generateDynamicPineScript();
    const tradeSection = result.split("Trade Levels —")[1] || "";
    expect(tradeSection).toContain("tpPrice");
  });

  it("draws entry line with gold/yellow color", () => {
    const result = generateDynamicPineScript();
    const tradeSection = result.split("Trade Levels —")[1] || "";
    expect(tradeSection).toMatch(/color\.yellow|#f5c842|color\.new\(color\.yellow/);
    expect(tradeSection).toContain("ENTRY");
  });

  it("draws stop loss line with red color", () => {
    const result = generateDynamicPineScript();
    const tradeSection = result.split("Trade Levels —")[1] || "";
    expect(tradeSection).toMatch(/color\.red/);
    expect(tradeSection).toContain("SL");
  });

  it("draws take profit line with green color", () => {
    const result = generateDynamicPineScript();
    const tradeSection = result.split("Trade Levels —")[1] || "";
    expect(tradeSection).toMatch(/color\.green|#00e676/);
    expect(tradeSection).toContain("TP");
  });

  it("uses line.new for trade level lines", () => {
    const result = generateDynamicPineScript();
    const tradeSection = result.split("Trade Levels —")[1] || "";
    expect(tradeSection).toContain("line.new");
  });

  it("uses label.new for trade level labels", () => {
    const result = generateDynamicPineScript();
    const tradeSection = result.split("Trade Levels —")[1] || "";
    expect(tradeSection).toContain("label.new");
  });

  it("extends trade lines to the right", () => {
    const result = generateDynamicPineScript();
    const tradeSection = result.split("Trade Levels —")[1] || "";
    expect(tradeSection).toContain("extend=extend.right");
  });

  it("trade lines have width=1 (thin solid lines)", () => {
    const result = generateDynamicPineScript();
    const tradeSection = result.split("Trade Levels —")[1] || "";
    expect(tradeSection).toMatch(/width=1/);
    // Should NOT have width=3 in trade section
    expect(tradeSection).not.toMatch(/width=3/);
  });

  it("trade lines are solid (no dashed style)", () => {
    const result = generateDynamicPineScript();
    const tradeSection = result.split("Trade Levels —")[1] || "";
    expect(tradeSection).not.toContain("line.style_dashed");
  });

  it("trade labels use size.small for readability", () => {
    const result = generateDynamicPineScript();
    const tradeSection = result.split("Trade Levels —")[1] || "";
    expect(tradeSection).toContain("size=size.small");
    expect(tradeSection).not.toContain("size=size.normal");
  });

  it("conditions trade levels on showTrade toggle", () => {
    const result = generateDynamicPineScript();
    expect(result).toMatch(/showTrade\s+and/);
  });

  it("only draws trade levels on the last bar", () => {
    const result = generateDynamicPineScript();
    const tradeSection = result.split("Trade Levels —")[1] || "";
    expect(tradeSection).toContain("barstate.islast");
  });

  it("includes price values in trade labels", () => {
    const result = generateDynamicPineScript();
    const tradeSection = result.split("Trade Levels —")[1] || "";
    expect(tradeSection).toMatch(/str\.tostring\(entryPrice/);
    expect(tradeSection).toMatch(/str\.tostring\(slPrice/);
    expect(tradeSection).toMatch(/str\.tostring\(tpPrice/);
  });

  it("calculates risk-reward ratio", () => {
    const result = generateDynamicPineScript();
    const tradeSection = result.split("Trade Levels —")[1] || "";
    expect(tradeSection).toMatch(/rr|risk.*reward|R:R/i);
  });
});

// ── Visibility / readability ─────────────────────────────

describe("generateDynamicPineScript — visibility", () => {
  it("OB boxes have border_width >= 2", () => {
    const result = generateDynamicPineScript();
    const obSection = result.split("Order Block Detection")[1]?.split("Fair Value Gap")[0] || "";
    expect(obSection).toContain("border_width=2");
  });

  it("FVG boxes have border_width >= 2", () => {
    const result = generateDynamicPineScript();
    const fvgSection = result.split("Fair Value Gap Detection")[1]?.split("Liquidity Levels —")[0] || "";
    expect(fvgSection).toContain("border_width=2");
  });

  it("liquidity lines have width=1 for chart visibility", () => {
    const result = generateDynamicPineScript();
    const liqSection = result.split("Liquidity Levels —")[1]?.split("Bias —")[0] || "";
    expect(liqSection).toMatch(/width=1/);
    expect(liqSection).not.toMatch(/width=3/);
  });

  it("liquidity labels use size.normal", () => {
    const result = generateDynamicPineScript();
    const liqSection = result.split("Liquidity Levels —")[1]?.split("Bias —")[0] || "";
    expect(liqSection).toContain("size.normal");
  });

  it("OB boxes have text alignment", () => {
    const result = generateDynamicPineScript();
    const obSection = result.split("Order Block Detection")[1]?.split("Fair Value Gap")[0] || "";
    expect(obSection).toContain("text_halign");
  });

  it("manages box/line cleanup to avoid exceeding limits", () => {
    const result = generateDynamicPineScript();
    // Should have array management for cleanup
    expect(result).toMatch(/array\.new_box|var\s+box\[\]/);
    expect(result).toMatch(/array\.new_line|var\s+line\[\]/);
  });

  it("includes price in liquidity labels via str.tostring", () => {
    const result = generateDynamicPineScript();
    expect(result).toMatch(/str\.tostring\(/);
  });
});
