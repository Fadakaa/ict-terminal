// ── Pure business logic for ICT Backtest Terminal ─────────
// All functions are side-effect free and independently testable.

export const TF_OPTIONS = [
  { label: "15M", value: "15min" },
  { label: "1H", value: "1h" },
  { label: "4H", value: "4h" },
  { label: "1D", value: "1day" },
];

export const REFRESH_OPTIONS = [
  { label: "30S", seconds: 30 },
  { label: "1M", seconds: 60 },
  { label: "5M", seconds: 300 },
  { label: "15M", seconds: 900 },
];

// API request sizes — over-fetched for sub-daily to compensate for weekend removal
// 5M  → ~8 hours (96)       → request 500, keep 200 after weekend filter
// 15M → 1–3 days (96–288)   → request 400, keep 150 after weekend filter
// 30M → 2–4 days (96–192)   → request 300, keep 150 after weekend filter
// 1H  → 2–5 days (48–120)   → request 180, keep 100 after weekend filter
// 2H  → 4–10 days (48–120)  → request 150, keep 100 after weekend filter
// 4H  → 2–3 weeks (84–126)  → 120 candles (weekends minimal at this scale)
// 1D  → 3–6 months (60–120) → 120 candles (no weekend bars on daily)
export const TF_CANDLES = {
  "5min": 500,
  "15min": 400,
  "30min": 300,
  "1h": 180,
  "2h": 150,
  "4h": 120,
  "1day": 120,
};

// Final display counts after weekend filtering + trimming
export const TARGET_CANDLES = {
  "5min": 200,
  "15min": 150,
  "30min": 150,
  "1h": 100,
  "2h": 100,
  "4h": 120,
  "1day": 120,
};

/**
 * Remove Saturday (6) and Sunday (0) candles for intraday timeframes.
 * Gold doesn't trade on weekends — these are stale/flat data.
 * Returns a new array (no mutation).
 */
export function filterWeekendCandles(candles, timeframe) {
  if (!["15min", "1h"].includes(timeframe)) return [...candles];
  return candles.filter((c) => {
    const day = new Date(c.datetime).getDay();
    return day !== 0 && day !== 6;
  });
}

/**
 * Keep only the most recent N candles for the given timeframe,
 * so the lookback window matches ICT analysis requirements.
 */
export function trimToTarget(candles, timeframe) {
  const target = TARGET_CANDLES[timeframe] || 100;
  if (candles.length <= target) return candles;
  return candles.slice(candles.length - target);
}

/**
 * Fingerprint a candle array for cache comparison.
 * Uses count + first/last candle datetime + close — cheap O(1) check.
 * Returns "" for empty arrays so cache always misses on no data.
 * Pure function — does not mutate input.
 */
export function hashCandles(candles) {
  if (!candles || candles.length === 0) return "";
  const first = candles[0];
  const last = candles[candles.length - 1];
  return `${candles.length}|${first.datetime}|${first.close}|${last.datetime}|${last.close}`;
}

/**
 * ICT system message — defines the expert persona and methodology.
 * Separated from user prompt so Claude treats this as its core identity.
 * Pure function.
 */
export function buildICTSystemMessage() {
  return `You are an elite ICT (Inner Circle Trader) analyst specialising in XAU/USD (Gold). You follow Michael J. Huddleston's ICT methodology precisely.

Core ICT Principles You Apply:
1. MARKET STRUCTURE — Identify Break of Structure (BOS) and Change of Character (CHoCH/MSS). Bullish structure = higher highs + higher lows. Bearish = lower highs + lower lows. A CHoCH signals potential reversal.
2. ORDER BLOCKS — The last opposing candle before displacement (a strong impulsive move). Bullish OB = last bearish candle before a strong rally with displacement. Bearish OB = last bullish candle before a strong sell-off. Validate with: displacement must exceed the OB range, price must break structure after the OB forms. Rate strength based on displacement magnitude and whether the OB created a BOS.
3. FAIR VALUE GAPS — 3-candle imbalance where candle 1 wick and candle 3 wick don't overlap, leaving an inefficiency. Bullish FVG: candle 3 low > candle 1 high. Bearish FVG: candle 3 high < candle 1 low. Track whether price has returned to fill the gap.
4. LIQUIDITY — Resting orders at swing highs (buy-side liquidity/BSL) and swing lows (sell-side liquidity/SSL). Also at equal highs, equal lows, and previous session highs/lows. Smart money hunts these pools before reversing.
5. PREMIUM/DISCOUNT — Calculate equilibrium (50% of the current dealing range). Above equilibrium = premium zone (look for shorts). Below = discount zone (look for longs). Optimal Trade Entry (OTE) sits at the 62-79% retracement level.
6. KILLZONES (all times UTC/GMT) — Asian (00:00-04:00), London Open (07:00-10:00), New York AM (12:00-15:00), New York PM (15:00-17:00). Highest-probability setups occur during London and NY AM killzones.

Trade Setup Quality Standards:
- Only recommend entries where bias aligns with the direction (bullish bias → longs, bearish bias → shorts)
- Entry should be at or near a validated order block or FVG in the discount/premium zone appropriate to the direction
- Stop loss below/above the order block with a small buffer (not arbitrary)
- Minimum 2:1 risk-reward for TP1; aim for 3:1+ for TP2/TP3
- If no high-quality setup exists, say so in the summary — do not force a trade

You respond with precise, actionable JSON only. No explanation text outside the JSON.`;
}

/**
 * Produce a prompt block with WFO-calibrated SL/TP levels for Claude.
 * Returns "" when wfoReport is null/undefined (no WFO data available).
 */
export function buildWFOCalibrationBlock(wfoReport, atr) {
  if (!wfoReport) return "";
  const slDist = +(wfoReport.recommended_sl_atr * atr).toFixed(1);
  const tps = wfoReport.recommended_tp_atr || [];
  const tpLines = tps
    .map((m, i) => `  TP${i + 1}: ${Number(m).toFixed(1)} ATR = ${(m * atr).toFixed(1)} points from entry`)
    .join("\n");
  const winPct = wfoReport.oos_win_rate != null
    ? (wfoReport.oos_win_rate * 100).toFixed(1)
    : "—";
  const pf = wfoReport.oos_profit_factor != null
    ? wfoReport.oos_profit_factor.toFixed(2)
    : "—";
  return `
WFO CALIBRATION (Walk-Forward Optimized — use these for SL/TP placement):
  System Grade: ${wfoReport.grade} | OOS Win Rate: ${winPct}% | Profit Factor: ${pf}
  Stop Loss: ${wfoReport.recommended_sl_atr} ATR = ${slDist} points from entry
${tpLines}
  IMPORTANT: Place your stop loss at ${wfoReport.recommended_sl_atr} ATR from entry, validated against the nearest order block.
  Place take profits at the ATR multiples above. These distances are statistically validated through walk-forward optimization on out-of-sample data.
`;
}

/**
 * Build the Claude user prompt for ICT analysis.
 * Pure function — formats candle data + analysis instructions, returns a string.
 * Optional wfoReport injects WFO-calibrated SL/TP guidance.
 */
export function buildICTPrompt(candles, tf, wfoReport = null) {
  const lastCandle = candles[candles.length - 1];
  const firstCandle = candles[0];
  const currentPrice = lastCandle ? lastCandle.close : 0;
  const rangeHigh = Math.max(...candles.map((c) => c.high));
  const rangeLow = Math.min(...candles.map((c) => c.low));
  const equilibrium = +((rangeHigh + rangeLow) / 2).toFixed(2);

  const atr = computeATR(candles, 14);
  const wfoBlock = buildWFOCalibrationBlock(wfoReport, atr);

  const data = candles.map((c, i) => ({
    i,
    dt: c.datetime,
    o: +c.open.toFixed(2),
    h: +c.high.toFixed(2),
    l: +c.low.toFixed(2),
    c: +c.close.toFixed(2),
  }));

  const tradeSetupStep = wfoReport
    ? "7. TRADE SETUP: Combine the above for an entry. Use WFO-calibrated ATR multipliers for SL/TP placement, validated against ICT structure. If no quality setup exists, state that clearly."
    : "7. TRADE SETUP: Combine the above for an entry. If no quality setup exists, state that clearly.";

  return `Analyse these ${candles.length} OHLC candles on the ${tf} timeframe for XAU/USD.

Current price: ${currentPrice} | Range: ${rangeLow} — ${rangeHigh} | Equilibrium: ${equilibrium}
Period: ${firstCandle?.datetime || "?"} to ${lastCandle?.datetime || "?"}

Candle data (i=index, dt=datetime, o=open, h=high, l=low, c=close):
${JSON.stringify(data)}
${wfoBlock}
Analysis checklist — work through each systematically:
1. STRUCTURE: Identify the most recent BOS or CHoCH. Is current structure bullish or bearish?
2. ORDER BLOCKS: Find validated OBs (must have displacement + BOS after formation). Note which are mitigated vs unmitigated.
3. FVGs: Identify all fair value gaps. Note which are filled vs unfilled.
4. LIQUIDITY: Mark swing high BSL and swing low SSL pools. Where are equal highs/lows?
5. PREMIUM/DISCOUNT: Is current price in premium or discount relative to the dealing range?
6. KILLZONE: Based on the most recent candle timestamp, which session/killzone applies?
${tradeSetupStep}

Return ONLY valid JSON, no markdown:
{
"bias":"bullish|bearish",
"summary":"Detailed analysis explaining the market structure, key levels, and trade rationale",
"orderBlocks":[{"type":"bullish|bearish","high":0,"low":0,"candleIndex":0,"strength":"strong|moderate|weak","mitigated":false,"note":"string"}],
"fvgs":[{"type":"bullish|bearish","high":0,"low":0,"startIndex":0,"filled":false,"note":"string"}],
"liquidity":[{"type":"buyside|sellside","price":0,"candleIndex":0,"note":"string"}],
"entry":{"price":0,"direction":"long|short","rationale":"string"},
"stopLoss":{"price":0,"rationale":"string"},
"takeProfits":[{"price":0,"rationale":"string","rr":0}],
"killzone":"string",
"confluences":["string"]
}`;
}

// ── Dynamic Pine Script generator ────────────────────────
// Self-calculating ICT indicator that works on ANY timeframe
// in TradingView — no hardcoded prices, recalculates live.

/**
 * Generate a fully dynamic Pine Script v5 indicator that detects
 * ICT structures (Order Blocks, FVGs, Liquidity, Bias) natively.
 * No parameters needed — the script adapts to whatever timeframe
 * and symbol is loaded on the TradingView chart.
 *
 * Pure function — returns a copyable Pine Script string.
 */
export function generateDynamicPineScript() {
  const lines = [];

  // ── Header ─────────────────────────────────────────────
  lines.push(`//@version=5`);
  lines.push(`indicator("ICT Dynamic Analysis — XAU/USD", overlay=true, max_boxes_count=500, max_lines_count=500, max_labels_count=500)`);
  lines.push(``);

  // ── Layer Toggles ──────────────────────────────────────
  lines.push(`// ── Layer Toggles ──`);
  lines.push(`showOB    = input.bool(true, "Show Order Blocks")`);
  lines.push(`showFVG   = input.bool(true, "Show Fair Value Gaps")`);
  lines.push(`showLiq   = input.bool(true, "Show Liquidity Levels")`);
  lines.push(`showBias  = input.bool(true, "Show Bias & Structure")`);
  lines.push(`showTrade = input.bool(true, "Show Trade Levels (Entry/SL/TP)")`);
  lines.push(``);

  // ── Configurable Parameters ────────────────────────────
  lines.push(`// ── Parameters ──`);
  lines.push(`obStrength  = input.float(1.5, "OB Displacement Multiplier", minval=1.0, step=0.1, tooltip="Strength of move after OB candle relative to ATR")`);
  lines.push(`fvgMinSize  = input.float(0.5, "FVG Minimum Size (ATR mult)", minval=0.1, step=0.1)`);
  lines.push(`swingLen    = input.int(5, "Swing Lookback Length", minval=2, maxval=20)`);
  lines.push(`maxBoxes    = input.int(5, "Max OB/FVG Boxes Shown", minval=1, maxval=20)`);
  lines.push(``);

  // ── ATR for dynamic thresholds ─────────────────────────
  lines.push(`// ── ATR for dynamic thresholds ──`);
  lines.push(`atr = ta.atr(14)`);
  lines.push(``);

  // ── Order Block Detection ──────────────────────────────
  lines.push(`// ── Order Block Detection ──`);
  lines.push(`// Bullish OB: bearish candle[1] followed by bullish displacement`);
  lines.push(`// Bearish OB: bullish candle[1] followed by bearish displacement`);
  lines.push(`bullCandle = close > open`);
  lines.push(`bearCandle = close < open`);
  lines.push(`displacement_up   = bullCandle and (close - open) > atr * obStrength`);
  lines.push(`displacement_down = bearCandle and (open - close) > atr * obStrength`);
  lines.push(``);
  lines.push(`bullOB = close[1] < open[1] and displacement_up`);
  lines.push(`bearOB = close[1] > open[1] and displacement_down`);
  lines.push(``);

  lines.push(`var box[] obBoxes = array.new_box()`);
  lines.push(``);
  lines.push(`if showOB and bullOB`);
  lines.push(`    if array.size(obBoxes) >= maxBoxes * 2`);
  lines.push(`        box.delete(array.shift(obBoxes))`);
  lines.push(`    array.push(obBoxes, box.new(bar_index - 1, high[1], bar_index + 20, low[1], bgcolor=color.new(color.green, 70), border_color=color.green, border_width=2, text="Bull OB", text_color=color.green, text_halign=text.align_left, text_valign=text.align_top, text_size=size.small))`);
  lines.push(``);
  lines.push(`if showOB and bearOB`);
  lines.push(`    if array.size(obBoxes) >= maxBoxes * 2`);
  lines.push(`        box.delete(array.shift(obBoxes))`);
  lines.push(`    array.push(obBoxes, box.new(bar_index - 1, high[1], bar_index + 20, low[1], bgcolor=color.new(color.red, 70), border_color=color.red, border_width=2, text="Bear OB", text_color=color.red, text_halign=text.align_left, text_valign=text.align_top, text_size=size.small))`);
  lines.push(``);

  // Track last OB zones for trade level calculation
  lines.push(`// Track last OB zones for trade entries`);
  lines.push(`var float lastBullOBHigh = na`);
  lines.push(`var float lastBullOBLow  = na`);
  lines.push(`var float lastBearOBHigh = na`);
  lines.push(`var float lastBearOBLow  = na`);
  lines.push(``);
  lines.push(`if bullOB`);
  lines.push(`    lastBullOBHigh := high[1]`);
  lines.push(`    lastBullOBLow  := low[1]`);
  lines.push(`if bearOB`);
  lines.push(`    lastBearOBHigh := high[1]`);
  lines.push(`    lastBearOBLow  := low[1]`);
  lines.push(``);

  // ── Fair Value Gap Detection ───────────────────────────
  lines.push(`// ── Fair Value Gap Detection ──`);
  lines.push(`// Bullish FVG: gap up — current low above 2-bars-ago high`);
  lines.push(`// Bearish FVG: gap down — current high below 2-bars-ago low`);
  lines.push(`bullFVG = low > high[2] and (low - high[2]) > atr * fvgMinSize`);
  lines.push(`bearFVG = high < low[2] and (low[2] - high) > atr * fvgMinSize`);
  lines.push(``);

  lines.push(`var box[] fvgBoxes = array.new_box()`);
  lines.push(``);
  lines.push(`if showFVG and bullFVG`);
  lines.push(`    if array.size(fvgBoxes) >= maxBoxes * 2`);
  lines.push(`        box.delete(array.shift(fvgBoxes))`);
  lines.push(`    array.push(fvgBoxes, box.new(bar_index - 1, low, bar_index + 15, high[2], bgcolor=color.new(color.aqua, 65), border_color=color.aqua, border_width=2, text="Bull FVG", text_color=color.aqua, text_halign=text.align_left, text_valign=text.align_top, text_size=size.small))`);
  lines.push(``);
  lines.push(`if showFVG and bearFVG`);
  lines.push(`    if array.size(fvgBoxes) >= maxBoxes * 2`);
  lines.push(`        box.delete(array.shift(fvgBoxes))`);
  lines.push(`    array.push(fvgBoxes, box.new(bar_index - 1, low[2], bar_index + 15, high, bgcolor=color.new(color.orange, 65), border_color=color.orange, border_width=2, text="Bear FVG", text_color=color.orange, text_halign=text.align_left, text_valign=text.align_top, text_size=size.small))`);
  lines.push(``);

  // ── Liquidity Levels (Swing Highs / Lows) ──────────────
  lines.push(`// ── Liquidity Levels — Swing Highs (BSL) & Lows (SSL) ──`);
  lines.push(`swingHigh = ta.pivothigh(high, swingLen, swingLen)`);
  lines.push(`swingLow  = ta.pivotlow(low, swingLen, swingLen)`);
  lines.push(``);

  lines.push(`var line[] liqLines  = array.new_line()`);
  lines.push(`var label[] liqLabels = array.new_label()`);
  lines.push(``);

  lines.push(`if showLiq and not na(swingHigh)`);
  lines.push(`    if array.size(liqLines) >= maxBoxes * 2`);
  lines.push(`        line.delete(array.shift(liqLines))`);
  lines.push(`        label.delete(array.shift(liqLabels))`);
  lines.push(`    array.push(liqLines, line.new(bar_index - swingLen, swingHigh, bar_index, swingHigh, color=color.lime, style=line.style_dashed, width=1, extend=extend.right))`);
  lines.push(`    array.push(liqLabels, label.new(bar_index - swingLen, swingHigh, "BSL " + str.tostring(swingHigh, "#.##"), style=label.style_label_down, color=color.new(color.lime, 40), textcolor=color.white, size=size.normal))`);
  lines.push(``);

  lines.push(`if showLiq and not na(swingLow)`);
  lines.push(`    if array.size(liqLines) >= maxBoxes * 2`);
  lines.push(`        line.delete(array.shift(liqLines))`);
  lines.push(`        label.delete(array.shift(liqLabels))`);
  lines.push(`    array.push(liqLines, line.new(bar_index - swingLen, swingLow, bar_index, swingLow, color=color.maroon, style=line.style_dashed, width=1, extend=extend.right))`);
  lines.push(`    array.push(liqLabels, label.new(bar_index - swingLen, swingLow, "SSL " + str.tostring(swingLow, "#.##"), style=label.style_label_up, color=color.new(color.maroon, 40), textcolor=color.white, size=size.normal))`);
  lines.push(``);

  // ── Bias — Break of Structure ──────────────────────────
  lines.push(`// ── Bias — Break of Structure ──`);
  lines.push(`var float lastSwingHigh = na`);
  lines.push(`var float lastSwingLow  = na`);
  lines.push(`var string bias = "neutral"`);
  lines.push(``);
  lines.push(`if not na(swingHigh)`);
  lines.push(`    if not na(lastSwingHigh) and swingHigh > lastSwingHigh`);
  lines.push(`        bias := "bullish"`);
  lines.push(`    else if not na(lastSwingHigh) and swingHigh < lastSwingHigh`);
  lines.push(`        bias := "bearish"`);
  lines.push(`    lastSwingHigh := swingHigh`);
  lines.push(``);
  lines.push(`if not na(swingLow)`);
  lines.push(`    if not na(lastSwingLow) and swingLow > lastSwingLow`);
  lines.push(`        bias := "bullish"`);
  lines.push(`    else if not na(lastSwingLow) and swingLow < lastSwingLow`);
  lines.push(`        bias := "bearish"`);
  lines.push(`    lastSwingLow := swingLow`);
  lines.push(``);
  lines.push(`biasColor = bias == "bullish" ? color.green : bias == "bearish" ? color.red : color.gray`);
  lines.push(`biasIcon  = bias == "bullish" ? "▲" : bias == "bearish" ? "▼" : "◆"`);
  lines.push(``);
  lines.push(`if showBias and barstate.islast`);
  lines.push(`    label.new(bar_index, high, biasIcon + " BIAS: " + str.upper(bias), style=label.style_label_down, color=biasColor, textcolor=color.white, size=size.small)`);
  lines.push(``);

  // ── Trade Levels — Entry / SL / TP ─────────────────────
  lines.push(`// ── Trade Levels — Entry / SL / TP ──`);
  lines.push(`// Bullish bias → long entry at last bull OB, SL below OB, TP at BSL`);
  lines.push(`// Bearish bias → short entry at last bear OB, SL above OB, TP at SSL`);
  lines.push(``);
  lines.push(`var float entryPrice = na`);
  lines.push(`var float slPrice    = na`);
  lines.push(`var float tpPrice    = na`);
  lines.push(`var float rr         = na`);
  lines.push(``);
  lines.push(`if bias == "bullish" and not na(lastBullOBHigh) and not na(lastSwingHigh)`);
  lines.push(`    entryPrice := lastBullOBHigh`);
  lines.push(`    slPrice    := lastBullOBLow - atr * 0.25`);
  lines.push(`    tpPrice    := lastSwingHigh`);
  lines.push(`    risk       = math.abs(entryPrice - slPrice)`);
  lines.push(`    rr         := risk > 0 ? math.abs(tpPrice - entryPrice) / risk : na`);
  lines.push(``);
  lines.push(`if bias == "bearish" and not na(lastBearOBLow) and not na(lastSwingLow)`);
  lines.push(`    entryPrice := lastBearOBLow`);
  lines.push(`    slPrice    := lastBearOBHigh + atr * 0.25`);
  lines.push(`    tpPrice    := lastSwingLow`);
  lines.push(`    risk       = math.abs(entryPrice - slPrice)`);
  lines.push(`    rr         := risk > 0 ? math.abs(entryPrice - tpPrice) / risk : na`);
  lines.push(``);
  lines.push(`if showTrade and barstate.islast and not na(entryPrice)`);
  lines.push(`    entryDir = bias == "bullish" ? "LONG" : "SHORT"`);
  lines.push(`    rrText   = not na(rr) ? " (" + str.tostring(rr, "#.#") + "R)" : ""`);
  lines.push(``);
  lines.push(`    line.new(bar_index - 20, entryPrice, bar_index, entryPrice, color=color.yellow, width=1, extend=extend.right)`);
  lines.push(`    label.new(bar_index, entryPrice, "▶ ENTRY " + entryDir + " " + str.tostring(entryPrice, "#.##"), style=label.style_label_left, color=color.new(color.yellow, 20), textcolor=color.black, size=size.small)`);
  lines.push(``);
  lines.push(`    line.new(bar_index - 20, slPrice, bar_index, slPrice, color=color.red, width=1, extend=extend.right)`);
  lines.push(`    label.new(bar_index, slPrice, "✖ SL " + str.tostring(slPrice, "#.##"), style=label.style_label_left, color=color.new(color.red, 20), textcolor=color.white, size=size.small)`);
  lines.push(``);
  lines.push(`    line.new(bar_index - 20, tpPrice, bar_index, tpPrice, color=color.green, width=1, extend=extend.right)`);
  lines.push(`    label.new(bar_index, tpPrice, "◎ TP " + str.tostring(tpPrice, "#.##") + rrText, style=label.style_label_left, color=color.new(color.green, 20), textcolor=color.white, size=size.small)`);

  return lines.join("\n");
}

// ── Static Pine Script generator (analysis-specific) ─────

/**
 * Escape double-quotes for Pine Script string literals.
 */
function pineEscape(str) {
  if (!str) return "";
  return String(str).replace(/"/g, "'");
}

/**
 * Convert ICT analysis JSON into a TradingView Pine Script v5 indicator.
 * Pure function — returns a copyable Pine Script string.
 *
 * Plots: Order Blocks (boxes), FVGs (boxes), Liquidity (dashed lines),
 *        Entry/SL/TP (horizontal lines with labels), Bias label.
 */
export function generatePineScript(analysis) {
  const a = analysis || {};
  const obs = a.orderBlocks || [];
  const fvgs = a.fvgs || [];
  const liqs = a.liquidity || [];
  const tps = a.takeProfits || [];
  const bias = (a.bias || "neutral").toUpperCase();
  const killzone = pineEscape(a.killzone || "N/A");
  const entryPrice = a.entry?.price ?? 0;
  const entryDir = a.entry?.direction || "long";
  const entryNote = pineEscape(a.entry?.rationale || "");
  const slPrice = a.stopLoss?.price ?? 0;
  const slNote = pineEscape(a.stopLoss?.rationale || "");

  const lines = [];

  // ── Header ─────────────────────────────────────────────
  lines.push(`//@version=5`);
  lines.push(`indicator("ICT Analysis — XAU/USD", overlay=true, max_boxes_count=500, max_lines_count=500, max_labels_count=500)`);
  lines.push(``);

  // ── User inputs — toggle each layer on/off ─────────────
  lines.push(`// ── Layer Toggles ──`);
  lines.push(`showOB    = input.bool(true, "Show Order Blocks")`);
  lines.push(`showFVG   = input.bool(true, "Show Fair Value Gaps")`);
  lines.push(`showLiq   = input.bool(true, "Show Liquidity Levels")`);
  lines.push(`showTrade = input.bool(true, "Show Trade Levels (Entry/SL/TP)")`);
  lines.push(`showBias  = input.bool(true, "Show Bias & Killzone")`);
  lines.push(``);

  // ── Order Blocks ───────────────────────────────────────
  if (obs.length > 0) {
    lines.push(`// ── Order Blocks ──`);
    lines.push(`if barstate.islast and showOB`);
    obs.forEach((ob) => {
      const isBull = ob.type === "bullish";
      const bgColor = isBull ? "color.new(color.green, 70)" : "color.new(color.red, 70)";
      const borderColor = isBull ? "color.green" : "color.red";
      const label = `${isBull ? "Bull" : "Bear"} OB${ob.strength ? " (" + ob.strength + ")" : ""} | ${ob.high} — ${ob.low}`;
      lines.push(`    box.new(bar_index - 50, ${ob.high}, bar_index, ${ob.low}, bgcolor=${bgColor}, border_color=${borderColor}, border_width=2, text="${pineEscape(label)}", text_color=${borderColor}, text_halign=text.align_left, text_valign=text.align_top, text_size=size.small)`);
    });
    lines.push(``);
  }

  // ── Fair Value Gaps ────────────────────────────────────
  if (fvgs.length > 0) {
    lines.push(`// ── Fair Value Gaps ──`);
    lines.push(`if barstate.islast and showFVG`);
    fvgs.forEach((fvg) => {
      const isBull = fvg.type === "bullish";
      const alpha = fvg.filled ? 85 : 65;
      const bgColor = isBull ? `color.new(color.aqua, ${alpha})` : `color.new(color.orange, ${alpha})`;
      const borderColor = isBull ? "color.aqua" : "color.orange";
      const style = fvg.filled ? "line.style_dotted" : "line.style_solid";
      const label = `${isBull ? "Bull" : "Bear"} FVG${fvg.filled ? " (filled)" : ""} | ${fvg.high} — ${fvg.low}`;
      lines.push(`    box.new(bar_index - 40, ${fvg.high}, bar_index, ${fvg.low}, bgcolor=${bgColor}, border_color=${borderColor}, border_width=2, border_style=${style}, text="${pineEscape(label)}", text_color=${borderColor}, text_halign=text.align_left, text_valign=text.align_top, text_size=size.small)`);
    });
    lines.push(``);
  }

  // ── Liquidity Levels ───────────────────────────────────
  if (liqs.length > 0) {
    lines.push(`// ── Liquidity Levels ──`);
    lines.push(`if barstate.islast and showLiq`);
    liqs.forEach((liq) => {
      const isBSL = liq.type === "buyside";
      const col = isBSL ? "color.lime" : "color.maroon";
      const tag = isBSL ? "BSL" : "SSL";
      const note = liq.note ? ` — ${pineEscape(liq.note)}` : "";
      lines.push(`    line.new(bar_index - 50, ${liq.price}, bar_index, ${liq.price}, color=${col}, style=line.style_dashed, width=3, extend=extend.right)`);
      lines.push(`    label.new(bar_index + 3, ${liq.price}, "${tag} ${liq.price}${note}", style=label.style_label_left, color=color.new(${col}, 40), textcolor=color.white, size=size.normal)`);
    });
    lines.push(``);
  }

  // ── Entry / SL / TP ────────────────────────────────────
  lines.push(`// ── Trade Levels ──`);
  lines.push(`if barstate.islast and showTrade`);

  // Entry
  if (entryPrice) {
    lines.push(`    line.new(bar_index - 50, ${entryPrice}, bar_index, ${entryPrice}, color=color.blue, style=line.style_solid, width=3, extend=extend.right)`);
    lines.push(`    label.new(bar_index + 3, ${entryPrice}, "▶ ENTRY ${entryDir.toUpperCase()} @ ${entryPrice}\\n${entryNote}", style=label.style_label_left, color=color.blue, textcolor=color.white, size=size.normal)`);
  }

  // Stop Loss
  if (slPrice) {
    lines.push(`    line.new(bar_index - 50, ${slPrice}, bar_index, ${slPrice}, color=color.red, style=line.style_solid, width=3, extend=extend.right)`);
    lines.push(`    label.new(bar_index + 3, ${slPrice}, "✖ SL @ ${slPrice}\\n${slNote}", style=label.style_label_left, color=color.red, textcolor=color.white, size=size.normal)`);
  }

  // Take Profits
  tps.forEach((tp, i) => {
    const rrText = tp.rr ? ` (${tp.rr}R)` : "";
    const note = pineEscape(tp.rationale || "");
    lines.push(`    line.new(bar_index - 50, ${tp.price}, bar_index, ${tp.price}, color=color.green, style=line.style_dashed, width=3, extend=extend.right)`);
    lines.push(`    label.new(bar_index + 3, ${tp.price}, "◎ TP${i + 1} @ ${tp.price}${rrText}\\n${note}", style=label.style_label_left, color=color.green, textcolor=color.white, size=size.normal)`);
  });
  lines.push(``);

  // ── Bias + Killzone info label ─────────────────────────
  lines.push(`// ── Bias & Killzone ──`);
  lines.push(`if barstate.islast and showBias`);
  const biasColor = bias === "BULLISH" ? "color.green" : bias === "BEARISH" ? "color.red" : "color.gray";
  const biasIcon = bias === "BULLISH" ? "▲" : bias === "BEARISH" ? "▼" : "◆";
  lines.push(`    label.new(bar_index, high, "${biasIcon} BIAS: ${bias}\\nKillzone: ${killzone}", style=label.style_label_down, color=${biasColor}, textcolor=color.white, size=size.large)`);

  return lines.join("\n");
}

// ── ML Feature Extraction ─────────────────────────────────
// Mirrors ml/features.py — lightweight JS version for UI display.

/**
 * Compute Average True Range from OHLC candles.
 * TR = max(H-L, |H-prevC|, |L-prevC|); ATR = SMA of TR over `period`.
 * Pure function.
 */
export function computeATR(candles, period = 14) {
  if (candles.length < 2) return 0;
  const trs = [];
  for (let i = 1; i < candles.length; i++) {
    const c = candles[i];
    const pc = candles[i - 1].close;
    trs.push(Math.max(c.high - c.low, Math.abs(c.high - pc), Math.abs(c.low - pc)));
  }
  if (trs.length < period) return 0;
  const slice = trs.slice(-period);
  return slice.reduce((a, b) => a + b, 0) / period;
}

/**
 * Extract 32 ATR-normalised ML features from ICT analysis + candles.
 * Pure function — does not mutate inputs.
 */
export function extractMLFeatures(analysis, candles, timeframe) {
  const atrRaw = computeATR(candles, 14);
  const atr = atrRaw || 1;

  const obs = analysis.orderBlocks || [];
  const fvgs = analysis.fvgs || [];
  const liqs = analysis.liquidity || [];
  const tps = analysis.takeProfits || [];
  const entry = analysis.entry || {};
  const sl = analysis.stopLoss || {};
  const bias = analysis.bias || "neutral";
  const killzone = analysis.killzone || "";
  const confluences = analysis.confluences || [];

  const entryPrice = entry.price || 0;
  const direction = entry.direction || "short";
  const slPrice = sl.price || 0;

  // Order Blocks (7)
  const obBull = obs.filter((o) => o.type === "bullish");
  const obBear = obs.filter((o) => o.type === "bearish");
  const strengthMap = { strong: 3, moderate: 2, weak: 1 };
  const strengths = obs.map((o) => strengthMap[o.strength] || 0);
  const obSizes = obs.map((o) => o.high - o.low);
  const obDists = entryPrice ? obs.map((o) => Math.abs(entryPrice - (o.high + o.low) / 2)) : [];
  const obAlign = (bias === "bullish" && obBull.length > 0) || (bias === "bearish" && obBear.length > 0) ? 1 : 0;

  // FVGs (5)
  const unfilled = fvgs.filter((f) => !f.filled);
  const fvgSizes = fvgs.map((f) => f.high - f.low);
  const fvgDists = entryPrice ? unfilled.map((f) => Math.abs(entryPrice - (f.high + f.low) / 2)) : [];
  let fvgAlign = 0;
  for (const f of unfilled) { if (f.type === bias) { fvgAlign = 1; break; } }

  // Liquidity (4)
  const bsl = liqs.filter((l) => l.type === "buyside");
  const ssl = liqs.filter((l) => l.type === "sellside");
  const targets = direction === "long" ? bsl : ssl;
  const threats = direction === "long" ? ssl : bsl;
  const tgtDist = entryPrice && targets.length ? Math.min(...targets.map((l) => Math.abs(entryPrice - l.price))) : 0;
  const thrDist = entryPrice && threats.length ? Math.min(...threats.map((l) => Math.abs(entryPrice - l.price))) : 0;

  // Trade Setup (6)
  const rr1 = tps.length > 0 ? (tps[0].rr || 0) : 0;
  const rr2 = tps.length > 1 ? (tps[1].rr || 0) : 0;
  const slDist = entryPrice && slPrice ? Math.abs(entryPrice - slPrice) : 0;
  const tp1Dist = tps.length > 0 && entryPrice ? Math.abs(entryPrice - tps[0].price) : 0;
  const dirEncoded = direction === "long" ? 1 : 0;
  const biasMatch = (bias === "bullish" && direction === "long") || (bias === "bearish" && direction === "short") ? 1 : 0;

  // Confluence (4)
  let obFvgOverlap = 0;
  for (const c of confluences) {
    const cl = c.toLowerCase();
    if ((cl.includes("ob") || cl.includes("order block")) && (cl.includes("fvg") || cl.includes("fair value") || cl.includes("overlap"))) {
      obFvgOverlap = 1; break;
    }
  }
  const kzMap = (kz) => { const k = kz.toLowerCase(); if (k.includes("london")) return 1; if (k.includes("new york") || k.includes("ny")) return 2; if (k.includes("asian") || k.includes("asia") || k.includes("tokyo")) return 3; return 0; };
  const tfMap = { "15min": 1, "1h": 2, "4h": 3, "1day": 4 };

  // Price Action (6)
  const closes = candles.map((c) => c.close);
  const sma20 = closes.length >= 20 ? closes.slice(-20).reduce((a, b) => a + b, 0) / 20 : (closes.length ? closes.reduce((a, b) => a + b, 0) / closes.length : 0);
  const priceVsSma = closes.length ? (closes[closes.length - 1] - sma20) / atr : 0;
  const atr5 = candles.length > 6 ? computeATR(candles, 5) : atrRaw;
  const volRatio = atr ? (atr5 || 0) / atr : 0;
  const lastBody = candles.length ? Math.abs(candles[candles.length - 1].close - candles[candles.length - 1].open) : 0;
  const trend = closes.length >= 20 ? (closes[closes.length - 1] - closes[closes.length - 20]) / atr : 0;
  let sessionHour = 0;
  if (candles.length) {
    try { sessionHour = parseInt(candles[candles.length - 1].datetime.split(" ")[1].split(":")[0], 10) || 0; } catch { sessionHour = 0; }
  }

  const r = (v) => Math.round(v * 10000) / 10000;
  return {
    ob_count: obs.length,
    ob_bullish_count: obBull.length,
    ob_bearish_count: obBear.length,
    ob_strongest_strength: strengths.length ? Math.max(...strengths) : 0,
    ob_nearest_distance_atr: r(obDists.length ? Math.min(...obDists) / atr : 0),
    ob_avg_size_atr: r(obSizes.length ? obSizes.reduce((a, b) => a + b, 0) / obSizes.length / atr : 0),
    ob_alignment: obAlign,
    fvg_count: fvgs.length,
    fvg_unfilled_count: unfilled.length,
    fvg_nearest_distance_atr: r(fvgDists.length ? Math.min(...fvgDists) / atr : 0),
    fvg_avg_size_atr: r(fvgSizes.length ? fvgSizes.reduce((a, b) => a + b, 0) / fvgSizes.length / atr : 0),
    fvg_alignment: fvgAlign,
    liq_buyside_count: bsl.length,
    liq_sellside_count: ssl.length,
    liq_nearest_target_distance_atr: r(tgtDist / atr),
    liq_nearest_threat_distance_atr: r(thrDist / atr),
    risk_reward_tp1: rr1,
    risk_reward_tp2: rr2,
    sl_distance_atr: r(slDist / atr),
    tp1_distance_atr: r(tp1Dist / atr),
    entry_direction: dirEncoded,
    bias_direction_match: biasMatch,
    num_confluences: confluences.length,
    has_ob_fvg_overlap: obFvgOverlap,
    killzone_encoded: kzMap(killzone),
    timeframe_encoded: tfMap[timeframe] || 0,
    atr_14: r(atrRaw),
    price_vs_20sma: r(priceVsSma),
    recent_volatility_ratio: r(volRatio),
    last_candle_body_atr: r(lastBody / atr),
    trend_strength: r(trend),
    session_hour: sessionHour,
  };
}

let _idCounter = 0;
/**
 * Generate a unique setup ID for trade logging.
 */
export function generateSetupId() {
  _idCounter++;
  return `setup-${Date.now()}-${_idCounter}-${Math.random().toString(36).slice(2, 7)}`;
}

/**
 * Format an ML prediction response for display.
 */
export function formatMLPrediction(prediction) {
  if (!prediction) return "N/A";
  if (prediction.model_status === "cold_start") return "N/A — ML learning mode";
  if (prediction.model_status === "insufficient_data") {
    return `Learning: ${prediction.training_samples || 0}/30 trades logged`;
  }
  return `${Math.round(prediction.confidence * 100)}% confidence`;
}
