import { useState, useEffect, useRef, useCallback } from "react";
import * as d3 from "d3";
import {
  TF_OPTIONS,
  TF_CANDLES,
  REFRESH_OPTIONS,
  filterWeekendCandles,
  trimToTarget,
  hashCandles,
  buildICTSystemMessage,
  computeATR,
  generateSetupId,
} from "./market.js";
import { useChartScale } from "./useChartScale.js";
import { scaleYDomain, scaleXRange, panXRange, wheelZoom } from "./chartScaling.js";
import { snapAnalysisToCandles, groupLiquidityByLevel } from "./analysisSnap.js";

// ═══════════════════════════════════════════════════════════════
//  HELPERS
// ═══════════════════════════════════════════════════════════════

function loadSaved(key, fallback) {
  try { const v = localStorage.getItem(key); return v !== null ? JSON.parse(v) : fallback; }
  catch { return fallback; }
}

function mapKillzone(kz) {
  if (!kz) return "off";
  const k = kz.toLowerCase();
  if (k.includes("london")) return "london";
  if (k.includes("ny") && k.includes("am") || k.includes("new york") && k.includes("am") || k.includes("ny open")) return "ny_am";
  if (k.includes("ny") && k.includes("pm") || k.includes("new york") && k.includes("pm")) return "ny_pm";
  if (k.includes("asia") || k.includes("tokyo") || k.includes("asian")) return "asia";
  if (k.includes("ny") || k.includes("new york")) return "ny_am";
  return "off";
}

function classifySetupType(parsed) {
  const dir = parsed?.claude_direction === "long" ? "bull" : "bear";
  const parts = [];
  if (parsed?.has_ob) parts.push("ob");
  if (parsed?.has_fvg) parts.push("fvg");
  if (parsed?.liq_swept) parts.push("sweep");
  parts.push(mapKillzone(parsed?.claude_killzone));
  return `${dir}_${parts.join("_")}`;
}

const SESSION_COLORS = {
  london: "#64b5f6",
  ny_am: "#f5c842",
  ny_pm: "#ffa726",
  asia: "#666",
  off: "#444",
};

const GRADE_COLORS = {
  A: "#26a69a", B: "#26a69a", C: "#f5c842", D: "#ffa726", F: "#ef5350",
};

// ═══════════════════════════════════════════════════════════════
//  DEMO DATA
// ═══════════════════════════════════════════════════════════════

function generateDemoCandles(count, base, spread, startDate) {
  const candles = [];
  let price = base;
  for (let i = 0; i < count; i++) {
    const d = new Date(startDate);
    d.setHours(d.getHours() + i);
    const o = price;
    const move = (Math.random() - 0.48) * spread;
    const c = o + move;
    const h = Math.max(o, c) + Math.random() * spread * 0.4;
    const l = Math.min(o, c) - Math.random() * spread * 0.4;
    candles.push({
      datetime: d.toISOString().replace("T", " ").slice(0, 16),
      open: +o.toFixed(2), high: +h.toFixed(2), low: +l.toFixed(2), close: +c.toFixed(2),
    });
    price = c;
  }
  return candles;
}

function getDemoData() {
  const candles1h = generateDemoCandles(60, 2920, 4, new Date("2026-03-13T00:00:00Z"));
  const candles4h = generateDemoCandles(20, 2905, 8, new Date("2026-03-10T00:00:00Z"));

  // Pin the candles at each formation index to the OHLC implied by the
  // hardcoded analysis below — without this, the random walk produces price
  // ranges that don't match the OB/FVG/liquidity bounds and the zones look
  // like they're floating instead of anchored to a candle.
  const inject = (idx, o, h, l, c) => {
    if (!candles1h[idx]) return;
    candles1h[idx] = {
      ...candles1h[idx],
      open: +o.toFixed(2), high: +h.toFixed(2), low: +l.toFixed(2), close: +c.toFixed(2),
    };
  };
  // BSL @ 25 — equal-highs liquidity at 2936.50 (unswept)
  inject(25, 2935.50, 2936.50, 2935.20, 2935.80);
  // Bearish OB @ 28 — last bullish candle before bearish displacement, 2933.50–2936.00
  inject(28, 2933.80, 2936.00, 2933.50, 2935.80);
  // Bearish FVG @ 30–32 — gap from 2929.50 (low) to 2932.00 (high)
  inject(30, 2933.20, 2933.40, 2932.00, 2932.20);
  inject(31, 2932.00, 2932.20, 2929.60, 2929.80); // displacement
  inject(32, 2929.80, 2929.50, 2927.50, 2928.00);
  // SSL @ 38 — sweep wick below 2912.00, close back above (manipulation)
  inject(38, 2913.50, 2913.80, 2911.80, 2913.20);
  // Bullish OB @ 42 — last bearish candle before bullish displacement, 2915.20–2918.50
  inject(42, 2918.30, 2918.50, 2915.20, 2915.50);
  // Bullish FVG @ 43–45 — gap from 2917.50 (low) to 2919.80 (high), overlaps the OB
  inject(43, 2916.00, 2917.50, 2915.80, 2917.30);
  inject(44, 2917.30, 2922.00, 2917.10, 2921.50); // BOS break candle (close > 2920)
  inject(45, 2921.00, 2924.00, 2919.80, 2923.50);

  const lastClose = candles1h[candles1h.length - 1].close;
  return {
    candles: candles1h,
    candles4h: candles4h,
    analysis: {
      bias: "bullish",
      summary: "4H dealing range established between 2905-2935. Price currently in discount zone below equilibrium at 2920. Recent SSL sweep at 2912 confirms manipulation phase complete. 1H structure shows bullish BOS with strong displacement from bullish OB at 2915-2918. Unfilled FVG overlapping the OB zone. London session providing optimal timing. Narrative: accumulation complete, manipulation sweep done, now expecting distribution to the upside targeting buy-side liquidity above 2935.",
      htf_context: {
        dealing_range_high: 2935.00,
        dealing_range_low: 2905.00,
        premium_discount: "discount",
        power_of_3_phase: "distribution",
        recent_sweep: "ssl",
        htf_bias: "bullish",
      },
      orderBlocks: [
        { type: "bullish", high: 2918.50, low: 2915.20, candleIndex: 42, tf: "1H", strength: "strong", times_tested: 0, note: "Untested OB from 4.5 ATR displacement" },
        { type: "bearish", high: 2936.00, low: 2933.50, candleIndex: 28, tf: "4H", strength: "moderate", times_tested: 1, note: "4H OB overhead — tested once, still holding" },
      ],
      fvgs: [
        { type: "bullish", high: 2919.80, low: 2917.50, startIndex: 43, tf: "1H", filled: false, fill_percentage: 0.15, overlaps_ob: true, note: "Overlaps bullish OB" },
        { type: "bearish", high: 2932.00, low: 2929.50, startIndex: 30, tf: "4H", filled: true, fill_percentage: 0.92, note: "4H FVG nearly fully filled" },
      ],
      liquidity: [
        { type: "sellside", price: 2912.00, candleIndex: 38, tf: "1H", swept: true, note: "SSL swept during London open" },
        { type: "buyside", price: 2936.50, candleIndex: 25, tf: "4H", swept: false, note: "4H equal highs — unswept BSL target" },
      ],
      structure: { type: "bos", direction: "bullish", break_candle_index: 44, note: "Bullish BOS confirmed above 2920" },
      entry: { price: 2917.80, direction: "long", entry_type: "rejection", rationale: "Rejection from bullish OB + FVG overlap in discount zone after SSL sweep" },
      stopLoss: { price: 2914.50, rationale: "Below OB low with 0.3 ATR buffer" },
      takeProfits: [
        { price: 2925.00, rationale: "Previous resistance / equilibrium", rr: 2.2 },
        { price: 2933.00, rationale: "Bearish OB zone", rr: 4.6 },
        { price: 2937.00, rationale: "BSL target", rr: 5.8 },
      ],
      killzone: "London AM Session",
      confluences: ["Bullish OB in discount", "Unfilled FVG overlapping OB", "SSL swept (manipulation)", "Bullish BOS confirmed", "London killzone"],
      setup_quality: "B",
      warnings: ["4H bearish OB overhead at 2933-2936 may cause resistance"],
    },
    calibration: {
      claude_original: { entry: 2917.80, sl: 2914.50, tps: [2925.00, 2933.00, 2937.00], direction: "long", rr_ratios: [2.2, 4.6, 5.8] },
      calibrated: {
        entry: 2917.80, sl: 2911.80, sl_source: "v1_session", sl_distance_atr: 2.1,
        tps: [2924.50, 2932.50, 2936.80], tp_distances_atr: [2.3, 5.1, 6.6], rr_ratios: [2.1, 4.4, 5.6], risk_amount: 6.00,
      },
      adjustments: {
        sl_widened: true, sl_widened_by: 2.70, sl_widened_by_atr: 0.9,
        sl_widened_reason: "V1 session data shows 95th percentile drawdown of 0.92 ATR during London — Claude's SL at 1.1 ATR would have been hit by normal noise on 42% of winning trades",
        tp_adjusted: true, tp_adjustment_direction: "narrowed",
      },
      confidence: {
        score: 0.62, grade: "B", claude_signal_strength: 0.83,
        bayesian_win_rate: 0.446, bayesian_trades_for_type: 8,
        autogluon_win_prob: null, historical_match_count: 18,
        historical_match_win_rate: 0.44, session_win_rate: 0.393,
      },
      session_context: {
        session: "london", v1_median_drawdown: 0.48, v1_p95_drawdown: 0.92,
        v1_median_favorable: 2.95, v1_session_win_rate: 0.393, v1_session_trades: 28,
      },
      volatility_context: { atr_14: 2.87, effective_atr: 3.16, regime: "normal", session_scale: 1.1, regime_scale: 1.0 },
      warnings: [
        "Claude's SL at 1.1 ATR is tighter than V1 session p95 drawdown (0.92 ATR + buffer)",
        "4H bearish OB overhead may limit TP3 potential",
      ],
      recommendation: "SL widened from 2914.50 to 2911.80 (+$2.70, +0.9 ATR). During London session, V1 data shows 95th percentile drawdown of 0.92 ATR on winning trades — Claude's SL at 1.1 ATR would have been stopped out 42% of the time. Bayesian WR for bull_ob_fvg_sweep_london: 44.6% (8 trades). Confidence B.",
    },
    bayesian: {
      win_rate_mean: 0.446, win_rate_lower_95: 0.31, win_rate_upper_95: 0.59,
      consecutive_losses: 1, max_consecutive_losses: 4, max_drawdown: 12.5,
      current_drawdown: 2.1, total_trades: 65,
    },
    sessionStats: {
      seeded: true,
      session_stats: {
        london: { trades: 28, wins: 11, win_rate: 0.393, median_drawdown: 0.48, p95_drawdown: 0.92, median_favorable: 2.95, median_bars_held: 8 },
        ny_am: { trades: 23, wins: 13, win_rate: 0.565, median_drawdown: 0.30, p95_drawdown: 0.96, median_favorable: 2.83, median_bars_held: 7 },
        ny_pm: { trades: 10, wins: 5, win_rate: 0.500, median_drawdown: 1.08, p95_drawdown: 1.12, median_favorable: 3.28, median_bars_held: 11 },
        asia: { trades: 3, wins: 0, win_rate: 0.0, median_drawdown: 0, p95_drawdown: 0, median_favorable: 0, median_bars_held: 0 },
        off: { trades: 1, wins: 0, win_rate: 0.0, median_drawdown: 0, p95_drawdown: 0, median_favorable: 0, median_bars_held: 0 },
      },
      bayesian_priors: { drawdown_mu: 0.42, favorable_mu: 2.95, win_alpha: 8.92, win_beta: 11.08, overall_win_rate: 0.446, total_trades: 65 },
      dataset_stats: { total: 92, wfo_count: 60, live_count: 32 },
    },
    accuracy: {
      total_trades: 45, claude_direction_correct: 31,
      claude_sl_would_survive: 22, calibrated_sl_survived: 29,
      trades_saved_by_calibration: 7, claude_tp1_reached: 18, calibrated_tp1_reached: 20,
      avg_claude_sl_distance_atr: 1.2, avg_calibrated_sl_distance_atr: 2.0,
      avg_sl_widening_atr: 0.8,
      by_session: {
        london: { trades: 18, claude_survived: 9, calibrated_survived: 13 },
        ny_am: { trades: 15, claude_survived: 8, calibrated_survived: 10 },
        ny_pm: { trades: 8, claude_survived: 3, calibrated_survived: 4 },
        asia: { trades: 3, claude_survived: 1, calibrated_survived: 1 },
        off: { trades: 1, claude_survived: 1, calibrated_survived: 1 },
      },
      by_setup_type: {
        bull_ob_fvg_sweep_london: { trades: 8, wins: 4 },
        bear_ob_fvg_ny_am: { trades: 5, wins: 3 },
        bull_ob_sweep_ny_am: { trades: 4, wins: 2 },
      },
    },
    calibrationValue: {
      total_trades: 45, claude_alone_survival_rate: 0.489, calibrated_survival_rate: 0.644,
      trades_saved: 7, survival_improvement: "+15.5%", avg_sl_widening: "0.8 ATR",
      best_session: "london", worst_session: "asia",
      recommendation: "Calibration is adding significant value — continue using calibrated SL levels",
    },
    pipelineHealth: { status: "ok" },
    datasetStats: { total: 92, wfo_count: 60, live_count: 32, v1_seed_count: 0, outcome_distribution: { stopped_out: 48, tp1_hit: 24, tp3_hit: 20 } },
    journal: [
      { id: Date.now(), ts: new Date().toLocaleString(), session: "london", direction: "long", setup_type: "bull_ob_fvg_sweep_london", entry: 2917.80, sl_used: 2911.80, sl_was_calibrated: true, outcome: "pending", tp_hit: null, rr: null, saved_by_calibration: false, note: "Auto: B setup — rejection", claude_sl: 2914.50, calibrated_sl: 2911.80, tps: [2924.50, 2932.50, 2936.80], rr_ratios: [2.1, 4.4, 5.6], setup_quality: "B", bias: "bullish" },
      { id: 2, ts: "2026-03-14 10:32", session: "london", direction: "long", setup_type: "bull_ob_fvg_sweep_london", entry: 2918.50, sl_used: 2912.00, sl_was_calibrated: true, outcome: "tp2", tp_hit: 2932.00, rr: 3.8, saved_by_calibration: true, note: "Perfect London OB entry", auto_resolved: true, resolved_at: "2026-03-14 14:21" },
      { id: 3, ts: "2026-03-14 15:15", session: "ny_am", direction: "short", setup_type: "bear_ob_fvg_ny_am", entry: 2941.20, sl_used: 2945.00, sl_was_calibrated: true, outcome: "stopped_out", tp_hit: null, rr: -1, saved_by_calibration: false, note: "Went against trend" },
      { id: 4, ts: "2026-03-13 09:45", session: "london", direction: "long", setup_type: "bull_ob_sweep_london", entry: 2910.80, sl_used: 2905.50, sl_was_calibrated: true, outcome: "tp1", tp_hit: 2918.00, rr: 1.4, saved_by_calibration: false, note: "" },
    ],
  };
}

// ═══════════════════════════════════════════════════════════════
//  ENHANCED PROMPT BUILDER (JS fallback)
// ═══════════════════════════════════════════════════════════════

function getCurrentKillzone() {
  const now = new Date();
  const h = now.getUTCHours();
  if (h >= 0 && h < 4) return "Asian Session";
  if (h >= 7 && h < 10) return "London Open";
  if (h >= 12 && h < 15) return "New York AM";
  if (h >= 15 && h < 17) return "New York PM";
  return "Off-Session";
}

function buildEnhancedICTPrompt(candles1h, candles4h) {
  const slim = (arr) => (arr || []).map((c) => ({
    dt: c.datetime, o: +Number(c.open).toFixed(2), h: +Number(c.high).toFixed(2),
    l: +Number(c.low).toFixed(2), c: +Number(c.close).toFixed(2),
  }));
  const h1 = slim(candles1h.slice(-60));
  const h4 = slim(candles4h.slice(-20));

  const now = new Date();
  const timeStr = now.toISOString().replace("T", " ").slice(0, 16) + " UTC";
  const dayStr = now.toLocaleDateString("en-US", { weekday: "long", timeZone: "UTC" });
  const kz = getCurrentKillzone();

  return `You are an expert ICT (Inner Circle Trader) analyst for Gold XAU/USD.

CURRENT TIME: ${timeStr} (${dayStr})
CURRENT KILLZONE: ${kz}

Analyse these candles on TWO timeframes to identify the highest-probability trade setup.

4H CANDLES (higher timeframe context — use the 20 most recent):
${JSON.stringify(h4)}

1H CANDLES (primary execution timeframe — use the 60 most recent):
${JSON.stringify(h1)}

ANALYSIS FRAMEWORK:
1. Determine the 4H dealing range (recent swing high to swing low). Is price in premium (upper half) or discount (lower half)?
2. Has 4H sell-side or buy-side liquidity been swept recently? This determines the Power of 3 phase.
3. On the 1H: identify the strongest Order Block born from genuine displacement. Has it been tested before?
4. Is there an unfilled Fair Value Gap overlapping or near the OB zone?
5. Was there a recent liquidity sweep on the 1H confirming manipulation?
6. Is there a break of structure or change of character confirming direction?
7. CRITICAL: Only suggest entry if there is a pullback or rejection into the zone. Do NOT enter on displacement candles.
8. If there is no high-probability setup right now, say so honestly. Set entry to null.
9. For every OB, FVG, and liquidity level you return, set "tf" to the timeframe where you identified it ("1H" or "4H"). Use 1H-relative candleIndex/startIndex even for 4H zones — find the 1H candle that aligns with the 4H zone's anchor time. Include 4H zones if they are within or near the visible 1H window and relevant to the setup.

Return ONLY valid JSON:
{
  "bias": "bullish|bearish",
  "summary": "string — include 4H dealing range context, premium/discount position, and the narrative",
  "htf_context": {
    "dealing_range_high": number,
    "dealing_range_low": number,
    "premium_discount": "premium|discount|equilibrium",
    "power_of_3_phase": "accumulation|manipulation|distribution",
    "recent_sweep": "bsl|ssl|none",
    "htf_bias": "bullish|bearish|neutral"
  },
  "orderBlocks": [{"type": "bullish|bearish", "high": number, "low": number, "candleIndex": number, "tf": "1H|4H", "strength": "strong|moderate|weak", "times_tested": number, "note": "string"}],
  "fvgs": [{"type": "bullish|bearish", "high": number, "low": number, "startIndex": number, "tf": "1H|4H", "filled": boolean, "fill_percentage": number, "overlaps_ob": boolean, "note": "string"}],
  "liquidity": [{"type": "buyside|sellside", "price": number, "candleIndex": number, "tf": "1H|4H", "swept": boolean, "note": "string"}],
  "structure": {"type": "bos|choch|none", "direction": "bullish|bearish", "break_candle_index": number, "note": "string"},
  "entry": {"price": number, "direction": "long|short", "entry_type": "rejection|retracement", "rationale": "string"} | null,
  "stopLoss": {"price": number, "rationale": "string"} | null,
  "takeProfits": [{"price": number, "rationale": "string", "rr": number}] | null,
  "killzone": "string",
  "confluences": ["string"],
  "setup_quality": "A|B|C|D|no_trade",
  "warnings": ["string"]
}`;
}


// ═══════════════════════════════════════════════════════════════
//  APP COMPONENT
// ═══════════════════════════════════════════════════════════════

export default function App() {
  // ── Market data ──
  const [candles, setCandles] = useState([]);
  const [candles4h, setCandles4h] = useState([]);
  const [timeframe, setTimeframe] = useState("1h");
  const {
    yManualDomain,
    setYManualDomain,
    xManualRange,
    setXManualRange,
    reset: resetChartScale,
    isManual: isChartManual,
  } = useChartScale(timeframe);
  const dragStateRef = useRef(null);
  const [isDragging, setIsDragging] = useState(false);

  // ── API keys ──
  const [claudeKey, setClaudeKey] = useState(() => loadSaved("ict_claude_key", ""));

  // ── Analysis + Calibration ──
  const [analysis, setAnalysis] = useState(null);
  const [calibration, setCalibration] = useState(null);

  // ── Backend state ──
  const [bayesian, setBayesian] = useState(null);
  const [sessionStats, setSessionStats] = useState(null);
  const [accuracy, setAccuracy] = useState(null);
  const [calibrationValue, setCalibrationValue] = useState(null);
  const [datasetStats, setDatasetStats] = useState(null);
  const [pipelineHealth, setPipelineHealth] = useState(null);

  // ── UI state ──
  const [screen, setScreen] = useState("setup");
  const [isDemo, setIsDemo] = useState(false);
  const [activeIntelTab, setActiveIntelTab] = useState("bayes");
  const [analysisExpanded, setAnalysisExpanded] = useState(false);
  const [loadingStep, setLoadingStep] = useState(null);
  const [error, setError] = useState("");
  const [loadingData, setLoadingData] = useState(false);
  const [toast, setToast] = useState("");

  // ── Journal ──
  const [journal, setJournal] = useState(() => loadSaved("ict_journal_v2", []));
  const [journalOutcome, setJournalOutcome] = useState(null);
  const [journalUsedCalSL, setJournalUsedCalSL] = useState(true);
  const [journalNote, setJournalNote] = useState("");
  const [journalDirection, setJournalDirection] = useState("long");
  const [journalEntryPrice, setJournalEntryPrice] = useState("");
  const [journalSLPrice, setJournalSLPrice] = useState("");

  // ── Scanner ──
  const [scannerStatus, setScannerStatus] = useState(null);
  const [scannerSetups, setScannerSetups] = useState([]);
  const [scannerHistory, setScannerHistory] = useState([]);
  const [modelInfo, setModelInfo] = useState(null);

  // ── ML Dashboard ──
  const [evalDual, setEvalDual] = useState(null);
  const [costBudget, setCostBudget] = useState(null);
  const [costHistory, setCostHistory] = useState(null);
  const [backtestMeta, setBacktestMeta] = useState(null);
  const [backtestStatus, setBacktestStatus] = useState(null);
  const [prospects, setProspects] = useState(null);
  const [bayesianDrift, setBayesianDrift] = useState(null);
  const [killzoneGates, setKillzoneGates] = useState(null);
  const [calendarUpcoming, setCalendarUpcoming] = useState(null);
  const [calendarProximity, setCalendarProximity] = useState(null);

  // ── Narrative / Thesis / Lifecycle ──
  const [currentThesis, setCurrentThesis] = useState(null);
  const [thesisHistory, setThesisHistory] = useState(null);
  const [thesisAccuracy, setThesisAccuracy] = useState(null);
  const [lifecycleRecent, setLifecycleRecent] = useState(null);
  const [recentContext, setRecentContext] = useState(null);

  // ── P&L tracker ──
  const [pnlHistory, setPnlHistory] = useState([]);
  const [pnlBalance, setPnlBalance] = useState(() => loadSaved("ict_pnl_balance", 10000));
  const [pnlRiskPct, setPnlRiskPct] = useState(() => loadSaved("ict_pnl_risk_pct", 1));
  const [pnlSpread, setPnlSpread] = useState(() => loadSaved("ict_pnl_spread", 0.50));
  const [propAccount, setPropAccount] = useState(() => loadSaved("ict_prop_account", "one_phase_micro"));
  const [propRiskPct, setPropRiskPct] = useState(() => loadSaved("ict_prop_risk_pct", 0.5));

  // ── Live mode ──
  const [liveMode, setLiveMode] = useState(false);
  const [refreshInterval, setRefreshInterval] = useState(60);
  const [autoAnalyze, setAutoAnalyze] = useState(true);
  const [lastUpdate, setLastUpdate] = useState(null);
  const [liveError, setLiveError] = useState("");

  // ── Responsive ──
  const [isNarrow, setIsNarrow] = useState(false);
  const [mobileTab, setMobileTab] = useState("chart");

  // ── Refs ──
  const svgRef = useRef(null);
  const chartScalesRef = useRef(null);
  const candlesRef = useRef(candles);
  const claudeKeyRef = useRef(claudeKey);
  const analysisAbortRef = useRef(null);
  const analysisCacheRef = useRef({ hash: "", result: null });

  useEffect(() => { candlesRef.current = candles; }, [candles]);
  useEffect(() => { claudeKeyRef.current = claudeKey; }, [claudeKey]);

  // Persist
  useEffect(() => { localStorage.setItem("ict_claude_key", JSON.stringify(claudeKey)); }, [claudeKey]);
  useEffect(() => { localStorage.setItem("ict_journal_v2", JSON.stringify(journal)); }, [journal]);
  useEffect(() => { localStorage.setItem("ict_pnl_balance", JSON.stringify(pnlBalance)); }, [pnlBalance]);
  useEffect(() => { localStorage.setItem("ict_pnl_risk_pct", JSON.stringify(pnlRiskPct)); }, [pnlRiskPct]);
  useEffect(() => { localStorage.setItem("ict_pnl_spread", JSON.stringify(pnlSpread)); }, [pnlSpread]);
  useEffect(() => { localStorage.setItem("ict_prop_account", JSON.stringify(propAccount)); }, [propAccount]);
  useEffect(() => { localStorage.setItem("ict_prop_risk_pct", JSON.stringify(propRiskPct)); }, [propRiskPct]);

  // Responsive breakpoint
  useEffect(() => {
    const mql = window.matchMedia("(max-width: 1200px)");
    const handler = (e) => setIsNarrow(e.matches);
    handler(mql);
    mql.addEventListener("change", handler);
    return () => mql.removeEventListener("change", handler);
  }, []);

  // Toast auto-dismiss
  useEffect(() => {
    if (!toast) return;
    const t = setTimeout(() => setToast(""), 3000);
    return () => clearTimeout(t);
  }, [toast]);

  // Probe backend on setup screen mount
  useEffect(() => {
    if (screen !== "setup") return;
    fetch("/api/ml/health").then((r) => r.ok ? r.json() : null).then(setPipelineHealth).catch(() => setPipelineHealth(null));
    // Fetch real Bayesian stats from scanner DB (primary), fall back to ML seed stats
    fetch("/api/bayesian").then((r) => r.ok ? r.json() : null).then((data) => {
      if (data?.bayesian) setBayesian(data.bayesian);
      if (data?.sessionStats) setSessionStats(data.sessionStats);
      if (data?.drift) setBayesianDrift(data.drift);
    }).catch(() => {
      // Fallback to ML seed stats if /api/bayesian unavailable
      fetch("/api/ml/seed/stats").then((r) => r.ok ? r.json() : null).then(setSessionStats).catch(() => {});
    });
  }, [screen]);

  // ═══════════════════════════════════════════════════════════
  //  BACKEND STATE REFRESH
  // ═══════════════════════════════════════════════════════════

  const refreshBackendState = useCallback(async () => {
    // Fetch real Bayesian stats from scanner_setups DB (takes priority over ML beliefs)
    let bayesianFromDb = false;
    try {
      const bRes = await fetch("/api/bayesian");
      if (bRes.ok) {
        const bData = await bRes.json();
        if (bData.bayesian) { setBayesian(bData.bayesian); bayesianFromDb = true; }
        if (bData.sessionStats) { setSessionStats(bData.sessionStats); }
        if (bData.drift) { setBayesianDrift(bData.drift); }
      }
    } catch {}

    const endpoints = [
      ["/api/ml/health", setPipelineHealth],
      // Only fall back to ML beliefs/seed/drift if /api/bayesian didn't provide data
      ...(!bayesianFromDb ? [["/api/ml/beliefs", setBayesian], ["/api/ml/seed/stats", setSessionStats], ["/api/ml/bayesian/drift", setBayesianDrift]] : []),
      ["/api/ml/claude/accuracy", setAccuracy],
      ["/api/ml/calibration/value", setCalibrationValue],
      ["/api/ml/dataset/stats", setDatasetStats],
      ["/api/ml/scanner/status", setScannerStatus],
      ["/api/ml/scanner/pending", setScannerSetups],
      ["/api/ml/scanner/history", setScannerHistory],
      ["/api/ml/scanner/pnl", setPnlHistory],
      ["/api/ml/model/info", setModelInfo],
      ["/api/ml/evaluation/dual", setEvalDual],
      ["/api/ml/cost/budget", setCostBudget],
      ["/api/ml/cost/history", setCostHistory],
      ["/api/ml/backtest/meta", setBacktestMeta],
      ["/api/ml/backtest/status", setBacktestStatus],
      ["/api/ml/scanner/prospects", setProspects],
      ["/api/ml/killzone/gates", setKillzoneGates],
      ["/api/ml/calendar/upcoming?hours=24", setCalendarUpcoming],
      ["/api/ml/calendar/proximity", setCalendarProximity],
      [`/api/ml/narrative/thesis/current?timeframe=${timeframe}`, setCurrentThesis],
      [`/api/ml/narrative/thesis/history?timeframe=${timeframe}&limit=10`, setThesisHistory],
      ["/api/ml/narrative/thesis/accuracy", setThesisAccuracy],
      ["/api/ml/lifecycle/recent?limit=30", setLifecycleRecent],
      [`/api/ml/context/recent?timeframe=${timeframe}`, setRecentContext],
    ];
    const results = await Promise.allSettled(
      endpoints.map(([url]) => fetch(url).then((r) => r.ok ? r.json() : null))
    );
    results.forEach((r, i) => {
      endpoints[i][1](r.status === "fulfilled" ? r.value : null);
    });
  }, [timeframe]);

  // ═══════════════════════════════════════════════════════════
  //  DATA FETCHING
  // ═══════════════════════════════════════════════════════════

  const fetchCandles = useCallback(async (silent = false) => {
    if (!silent) { setLoadingData(true); setError(""); }
    try {
      const res = await fetch(
        `/api/ml/candles?symbol=XAU/USD&interval=${timeframe}&count=${TF_CANDLES[timeframe] || 100}`
      );
      const json = await res.json();
      if (!json.values?.length) throw new Error("No data returned from OANDA");
      let parsed = json.values.map((v) => ({
        datetime: v.datetime, open: parseFloat(v.open), high: parseFloat(v.high),
        low: parseFloat(v.low), close: parseFloat(v.close),
      }));
      parsed = filterWeekendCandles(parsed, timeframe);
      parsed = trimToTarget(parsed, timeframe);
      setCandles(parsed);
      setLastUpdate(new Date());
      setLiveError("");
      return parsed;
    } catch (e) {
      if (silent) setLiveError(e.message); else setError(e.message);
      return null;
    } finally { if (!silent) setLoadingData(false); }
  }, [timeframe]);

  const fetch4hCandles = useCallback(async () => {
    try {
      const res = await fetch(
        `/api/ml/candles?symbol=XAU/USD&interval=4h&count=20`
      );
      const json = await res.json();
      if (!json.values?.length) return null;
      const parsed = json.values.map((v) => ({
        datetime: v.datetime, open: parseFloat(v.open), high: parseFloat(v.high),
        low: parseFloat(v.low), close: parseFloat(v.close),
      }));
      setCandles4h(parsed);
      return parsed;
    } catch { return null; }
  }, []);

  // ═══════════════════════════════════════════════════════════
  //  ANALYSIS + CALIBRATION FLOW
  // ═══════════════════════════════════════════════════════════

  const runCalibration = useCallback(async (analysisData, candleData) => {
    if (!analysisData?.entry) return null;
    try {
      const recent = (candleData || candlesRef.current).slice(-60).map((c) => ({
        datetime: c.datetime, open: c.open, high: c.high, low: c.low, close: c.close,
      }));
      const res = await fetch("/api/ml/calibrate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ analysis: analysisData, candles: recent }),
      });
      if (res.ok) {
        const cal = await res.json();
        setCalibration(cal);
        return cal;
      }
    } catch { /* backend offline */ }
    return null;
  }, []);

  const runAnalysis = useCallback(async (candleData, fourHData) => {
    const cds = candleData || candlesRef.current;
    const key = (claudeKeyRef.current || "").trim();
    if (!cds.length || !key) return;
    // HTTP header values must be printable ASCII; smart quotes, NBSP, or zero-width
    // characters from copy-paste cause fetch() to throw "string did not match the
    // expected pattern". Catch it early with a useful message instead.
    if (!/^[\x21-\x7e]+$/.test(key)) {
      setError("API key contains invalid characters — re-copy it from console.anthropic.com (avoid pasting through rich-text apps).");
      return;
    }

    const hash = hashCandles(cds);
    if (hash && hash === analysisCacheRef.current.hash && analysisCacheRef.current.result) {
      setAnalysis(analysisCacheRef.current.result);
      return;
    }

    if (analysisAbortRef.current) analysisAbortRef.current.abort();
    const abort = new AbortController();
    analysisAbortRef.current = abort;

    setError("");
    setCalibration(null);

    try {
      // Step 1: Fetch 4H candles
      setLoadingStep("4h");
      let h4 = fourHData || candles4h;
      if (!h4.length) {
        h4 = await fetch4hCandles() || [];
      }

      // Step 2: Claude analysis with enhanced prompt
      setLoadingStep("claude");
      const prompt = buildEnhancedICTPrompt(cds, h4);
      const res = await fetch("/api/anthropic/v1/messages", {
        method: "POST", signal: abort.signal,
        headers: { "Content-Type": "application/json", "x-api-key": key, "anthropic-version": "2023-06-01" },
        body: JSON.stringify({
          model: "claude-sonnet-4-6", max_tokens: 5000, temperature: 0,
          system: buildICTSystemMessage(),
          messages: [{ role: "user", content: prompt }],
        }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.error?.message || `API error ${res.status}`);
      }
      const data = await res.json();
      const text = data.content?.find((b) => b.type === "text")?.text || "";
      let clean = text.replace(/```json\n?|```/g, "").trim();
      const jsonStart = clean.indexOf("{");
      const jsonEnd = clean.lastIndexOf("}");
      if (jsonStart >= 0 && jsonEnd > jsonStart) clean = clean.slice(jsonStart, jsonEnd + 1);
      const raw = JSON.parse(clean);
      const { analysis: parsed, diagnostics } = snapAnalysisToCandles(raw, cds);
      if (
        diagnostics.snapped_obs || diagnostics.snapped_fvgs || diagnostics.snapped_liquidity ||
        diagnostics.dropped_obs || diagnostics.dropped_fvgs || diagnostics.dropped_liquidity
      ) {
        console.warn("[analysis] overlay snap diagnostics:", diagnostics);
      }
      analysisCacheRef.current = { hash, result: parsed };
      setAnalysis(parsed);

      // Step 3: Auto-calibrate + auto-journal
      if (parsed.entry) {
        setLoadingStep("calibrate");
        const cal = await runCalibration(parsed, cds);

        // Auto-journal the setup as "pending"
        const isDuplicate = (() => {
          const stored = JSON.parse(localStorage.getItem("ict_journal_v2") || "[]");
          if (!stored.length) return false;
          const last = stored[0];
          const timeDiff = Date.now() - (last.id || 0);
          return last.direction === parsed.entry.direction
            && Math.abs((last.entry || 0) - parsed.entry.price) < 1
            && timeDiff < 30 * 60 * 1000;
        })();

        if (!isDuplicate) {
          const autoEntry = {
            id: Date.now(), ts: new Date().toLocaleString(),
            session: mapKillzone(parsed.killzone),
            direction: parsed.entry.direction,
            setup_type: classifySetupType({
              claude_direction: parsed.entry.direction,
              has_ob: (parsed.orderBlocks?.length || 0) > 0,
              has_fvg: (parsed.fvgs?.length || 0) > 0,
              liq_swept: (parsed.liquidity || []).some((l) => l.swept),
              claude_killzone: parsed.killzone,
            }),
            entry: parsed.entry.price,
            sl_used: cal ? cal.calibrated.sl : parsed.stopLoss?.price,
            sl_was_calibrated: !!cal,
            outcome: "pending",
            tp_hit: null, rr: null,
            saved_by_calibration: false,
            note: `Auto: ${parsed.setup_quality || "?"} setup — ${parsed.entry.entry_type || "entry"}`,
            claude_sl: parsed.stopLoss?.price || null,
            calibrated_sl: cal?.calibrated?.sl || null,
            tps: cal?.calibrated?.tps || parsed.takeProfits?.map((t) => t.price) || [],
            rr_ratios: cal?.calibrated?.rr_ratios || parsed.takeProfits?.map((t) => t.rr) || [],
            setup_quality: parsed.setup_quality,
            bias: parsed.bias,
          };
          setJournal((prev) => [autoEntry, ...prev]);
          setToast("Setup auto-logged — update outcome when trade resolves");
        }
      }
    } catch (e) {
      if (e.name === "AbortError") return;
      setError("Analysis failed: " + e.message);
    } finally {
      setLoadingStep(null);
    }
  }, [timeframe, candles4h, fetch4hCandles, runCalibration]);

  // ═══════════════════════════════════════════════════════════
  //  TRADE LOGGING
  // ═══════════════════════════════════════════════════════════

  const logTrade = async () => {
    if (!journalOutcome) return;
    const hasAnalysis = analysis?.entry && calibration;
    const manualPrice = parseFloat(journalEntryPrice) || null;
    const manualSL = parseFloat(journalSLPrice) || null;

    // Must have either an active analysis or a manual entry price
    if (!hasAnalysis && !manualPrice) return;

    const dir = hasAnalysis ? (analysis.entry?.direction || "long") : journalDirection;
    const entryPrice = hasAnalysis ? analysis.entry?.price : manualPrice;
    const slPrice = hasAnalysis
      ? (journalUsedCalSL ? calibration.calibrated.sl : calibration.claude_original.sl)
      : manualSL;

    const entry = {
      id: Date.now(), ts: new Date().toLocaleString(),
      session: hasAnalysis ? mapKillzone(analysis.killzone) : mapKillzone(""),
      direction: dir,
      setup_type: hasAnalysis ? classifySetupType({
        claude_direction: analysis.entry?.direction,
        has_ob: (analysis.orderBlocks?.length || 0) > 0,
        has_fvg: (analysis.fvgs?.length || 0) > 0,
        liq_swept: (analysis.liquidity || []).some((l) => l.swept),
        claude_killzone: analysis.killzone,
      }) : "manual_entry",
      entry: entryPrice,
      sl_used: slPrice,
      sl_was_calibrated: hasAnalysis ? journalUsedCalSL : false,
      outcome: journalOutcome,
      tp_hit: hasAnalysis && journalOutcome.startsWith("tp") ? (calibration.calibrated.tps?.[parseInt(journalOutcome[2]) - 1] || null) : null,
      rr: journalOutcome.startsWith("tp") && hasAnalysis ? (calibration.calibrated.rr_ratios?.[parseInt(journalOutcome[2]) - 1] || 0) : journalOutcome === "stopped_out" ? -1 : 0,
      saved_by_calibration: false,
      note: journalNote || (hasAnalysis ? "" : "Ad-hoc manual entry"),
    };
    setJournal((prev) => [entry, ...prev]);

    // Post to backend
    try {
      await fetch("/api/ml/trade/complete", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          original_analysis: analysis || {},
          calibrated_result: calibration || {},
          actual_outcome: journalOutcome,
          actual_pnl_atr: entry.rr,
          used_calibrated_sl: hasAnalysis ? journalUsedCalSL : false,
          notes: journalNote || "Ad-hoc manual entry",
        }),
      });
      await refreshBackendState();
    } catch { /* backend offline */ }

    const wr = bayesian?.win_rate_mean ? `${(bayesian.win_rate_mean * 100).toFixed(0)}%` : "—";
    setToast(`Trade logged. Bayesian WR: ${wr}`);
    setJournalOutcome(null);
    setJournalNote("");
    setJournalUsedCalSL(true);
    setJournalEntryPrice("");
    setJournalSLPrice("");
  };

  // Resolve a pending journal entry (manual inline button click)
  const resolveJournalEntry = async (entryId, outcome) => {
    let resolvedEntry = null;
    setJournal((prev) => prev.map((e) => {
      if (e.id !== entryId) return e;
      const isWin = outcome.startsWith("tp");
      const tpIdx = isWin ? parseInt(outcome[2]) - 1 : -1;
      resolvedEntry = {
        ...e, outcome,
        tp_hit: isWin ? (e.tps?.[tpIdx] || null) : null,
        rr: isWin ? (e.rr_ratios?.[tpIdx] || 0) : outcome === "stopped_out" ? -1 : 0,
        resolved_at: new Date().toLocaleString(),
      };
      return resolvedEntry;
    }));

    // Post to backend
    try {
      const entry = journal.find((e) => e.id === entryId);
      if (entry) {
        await fetch("/api/ml/trade/complete", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            original_analysis: analysis || {},
            calibrated_result: calibration || {},
            actual_outcome: outcome,
            actual_pnl_atr: outcome.startsWith("tp") ? (entry.rr_ratios?.[parseInt(outcome[2]) - 1] || 0) : outcome === "stopped_out" ? -1 : 0,
            used_calibrated_sl: entry.sl_was_calibrated,
            notes: entry.note || "",
          }),
        });
        await refreshBackendState();
      }
    } catch { /* backend offline */ }

    setToast(`Trade resolved: ${outcome.toUpperCase()}`);
  };

  // Auto-resolve pending journal entries by monitoring price vs SL/TP levels
  useEffect(() => {
    if (!candles.length || isDemo) return;
    const latest = candles[candles.length - 1];
    if (!latest) return;
    const currentHigh = Number(latest.high);
    const currentLow = Number(latest.low);

    setJournal((prev) => {
      let changed = false;
      const updated = prev.map((entry) => {
        if (entry.outcome !== "pending" || !entry.entry) return entry;

        const isLong = entry.direction === "long";
        const sl = entry.sl_used || entry.calibrated_sl || entry.claude_sl;
        const tps = entry.tps || [];

        // Check SL hit first
        if (sl != null) {
          if ((isLong && currentLow <= sl) || (!isLong && currentHigh >= sl)) {
            changed = true;
            const claudeSlHit = entry.claude_sl && ((isLong && currentLow <= entry.claude_sl) || (!isLong && currentHigh >= entry.claude_sl));
            const calSlHit = entry.calibrated_sl && ((isLong && currentLow <= entry.calibrated_sl) || (!isLong && currentHigh >= entry.calibrated_sl));
            const savedByCal = claudeSlHit && !calSlHit && entry.sl_was_calibrated;
            return { ...entry, outcome: "stopped_out", rr: -1, resolved_at: new Date().toLocaleString(), auto_resolved: true, saved_by_calibration: savedByCal };
          }
        }

        // Check TPs hit (highest first for best outcome)
        for (let i = tps.length - 1; i >= 0; i--) {
          const tp = tps[i];
          if (tp != null && ((isLong && currentHigh >= tp) || (!isLong && currentLow <= tp))) {
            changed = true;
            return { ...entry, outcome: `tp${i + 1}`, tp_hit: tp, rr: entry.rr_ratios?.[i] || 0, resolved_at: new Date().toLocaleString(), auto_resolved: true };
          }
        }

        return entry;
      });

      if (changed) {
        // Fire backend updates for newly resolved entries
        updated.forEach((entry, idx) => {
          if (entry.auto_resolved && entry.resolved_at && prev[idx].outcome === "pending") {
            fetch("/api/ml/trade/complete", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                original_analysis: analysis || {},
                calibrated_result: calibration || {},
                actual_outcome: entry.outcome,
                actual_pnl_atr: entry.rr || 0,
                used_calibrated_sl: entry.sl_was_calibrated,
                notes: `Auto-resolved: ${entry.outcome} at ${entry.resolved_at}`,
              }),
            }).then(() => refreshBackendState()).catch(() => {});
            setToast(`Trade auto-resolved: ${entry.outcome.toUpperCase()} ${entry.rr > 0 ? "+" + entry.rr.toFixed(1) + "R" : entry.rr != null ? entry.rr.toFixed(1) + "R" : ""}`);
          }
        });
        return updated;
      }
      return prev;
    });
  }, [candles, isDemo]); // eslint-disable-line react-hooks/exhaustive-deps

  // ═══════════════════════════════════════════════════════════
  //  INITIALISATION
  // ═══════════════════════════════════════════════════════════

  const fetchData = async () => {
    const key = claudeKey.trim();
    if (!key) { setError("Enter your Anthropic API key"); return; }
    if (!/^[\x21-\x7e]+$/.test(key)) {
      setError("API key contains invalid characters — re-copy it from console.anthropic.com (avoid pasting through rich-text apps).");
      return;
    }
    setLoadingData(true); setError(""); setAnalysis(null); setCalibration(null);
    const data = await fetchCandles(false);
    if (data) {
      setScreen("live");
      refreshBackendState();
    }
    setLoadingData(false);
  };

  const launchDemo = () => {
    const d = getDemoData();
    setCandles(d.candles);
    setCandles4h(d.candles4h);
    setAnalysis(d.analysis);
    setCalibration(d.calibration);
    setBayesian(d.bayesian);
    setSessionStats(d.sessionStats);
    setAccuracy(d.accuracy);
    setCalibrationValue(d.calibrationValue);
    setDatasetStats(d.datasetStats);
    setPipelineHealth(d.pipelineHealth);
    setJournal(d.journal);
    setIsDemo(true);
    setScreen("live");
  };

  // Live mode auto-refresh
  useEffect(() => {
    if (!liveMode || screen !== "live" || isDemo) return;
    const tick = async () => {
      const data = await fetchCandles(true);
      if (data && autoAnalyze) await runAnalysis(data);
    };
    const id = setInterval(tick, refreshInterval * 1000);
    return () => clearInterval(id);
  }, [liveMode, refreshInterval, screen, isDemo, fetchCandles, runAnalysis, autoAnalyze]);

  // Poll ALL tab data every 60s so ML/Bayes/Session/Accuracy tabs stay current.
  // Runs on both "live" and "setup" screens (not demo mode).
  useEffect(() => {
    if (isDemo) return;
    const id = setInterval(() => {
      // Fetch real Bayesian stats from scanner DB (replaces /api/ml/beliefs + drift)
      fetch("/api/bayesian").then(r => r.ok ? r.json() : null).then(data => {
        if (data?.bayesian) setBayesian(data.bayesian);
        if (data?.sessionStats) setSessionStats(data.sessionStats);
        if (data?.drift) setBayesianDrift(data.drift);
      }).catch(() => {});

      Promise.allSettled([
        fetch("/api/ml/scanner/pending").then(r => r.ok ? r.json() : null),
        fetch("/api/ml/scanner/history").then(r => r.ok ? r.json() : null),
        fetch("/api/ml/scanner/status").then(r => r.ok ? r.json() : null),
        fetch("/api/ml/scanner/pnl").then(r => r.ok ? r.json() : null),
        fetch("/api/ml/claude/accuracy").then(r => r.ok ? r.json() : null),
        fetch("/api/ml/calibration/value").then(r => r.ok ? r.json() : null),
        fetch("/api/ml/model/info").then(r => r.ok ? r.json() : null),
        fetch("/api/ml/evaluation/dual").then(r => r.ok ? r.json() : null),
        fetch("/api/ml/dataset/stats").then(r => r.ok ? r.json() : null),
        fetch("/api/ml/cost/budget").then(r => r.ok ? r.json() : null),
        fetch("/api/ml/scanner/prospects").then(r => r.ok ? r.json() : null),
        fetch("/api/ml/killzone/gates").then(r => r.ok ? r.json() : null),
        fetch(`/api/ml/narrative/thesis/current?timeframe=${timeframe}`).then(r => r.ok ? r.json() : null),
        fetch("/api/ml/narrative/thesis/accuracy").then(r => r.ok ? r.json() : null),
        fetch("/api/ml/lifecycle/recent?limit=30").then(r => r.ok ? r.json() : null),
        fetch(`/api/ml/context/recent?timeframe=${timeframe}`).then(r => r.ok ? r.json() : null),
      ]).then(([p, h, s, pnl, acc, calVal, mi, evalD, ds, cb, prosp, kzg, thesis, thAcc, lc, rctx]) => {
        if (p.status === "fulfilled" && p.value) setScannerSetups(p.value);
        if (h.status === "fulfilled" && h.value) setScannerHistory(h.value);
        if (s.status === "fulfilled" && s.value) setScannerStatus(s.value);
        if (pnl.status === "fulfilled" && pnl.value) setPnlHistory(pnl.value);
        if (acc.status === "fulfilled" && acc.value) setAccuracy(acc.value);
        if (calVal.status === "fulfilled" && calVal.value) setCalibrationValue(calVal.value);
        if (mi.status === "fulfilled" && mi.value) setModelInfo(mi.value);
        if (evalD.status === "fulfilled" && evalD.value) setEvalDual(evalD.value);
        if (ds.status === "fulfilled" && ds.value) setDatasetStats(ds.value);
        if (cb.status === "fulfilled" && cb.value) setCostBudget(cb.value);
        if (prosp.status === "fulfilled" && prosp.value) setProspects(prosp.value);
        if (kzg.status === "fulfilled" && kzg.value) setKillzoneGates(kzg.value);
        if (thesis.status === "fulfilled" && thesis.value) setCurrentThesis(thesis.value);
        if (thAcc.status === "fulfilled" && thAcc.value) setThesisAccuracy(thAcc.value);
        if (lc.status === "fulfilled" && lc.value) setLifecycleRecent(lc.value);
        if (rctx.status === "fulfilled" && rctx.value) setRecentContext(rctx.value);
      });
    }, 60000);
    return () => clearInterval(id);
  }, [isDemo, timeframe]);

  // Auto re-fetch on timeframe switch
  useEffect(() => {
    if (screen !== "live" || isDemo) return;
    setAnalysis(null); setCalibration(null);
    analysisCacheRef.current = { hash: "", result: null };
    const timer = setTimeout(async () => {
      const data = await fetchCandles(false);
      if (data) await runAnalysis(data);
    }, 400);
    return () => clearTimeout(timer);
  }, [timeframe]); // eslint-disable-line react-hooks/exhaustive-deps

  // ═══════════════════════════════════════════════════════════
  //  D3 CHART
  // ═══════════════════════════════════════════════════════════

  const drawChart = useCallback(() => {
    if (!svgRef.current || !candles.length) return;
    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();
    const W = svgRef.current.clientWidth;
    const H = svgRef.current.clientHeight;
    if (!W || !H) return;
    const m = { top: 16, right: 80, bottom: 30, left: 65 };
    const w = W - m.left - m.right;
    const h = H - m.top - m.bottom;
    const g = svg.append("g").attr("transform", `translate(${m.left},${m.top})`);
    g.append("rect").attr("width", w).attr("height", h).attr("fill", "#06060e");

    // Future-space padding — virtual slots to the right of the last candle so
    // unmitigated OB/FVG zones (which extend from formation to chart's right edge)
    // visually project into empty future space. Standard ICT/TradingView convention.
    const FUTURE_BARS = Math.max(10, Math.ceil(candles.length * 0.2));
    const totalSlots = candles.length + FUTURE_BARS;
    const allSlotIndices = Array.from({ length: totalSlots }, (_, i) => i);
    const visibleArrayIndices = xManualRange
      ? allSlotIndices.filter((i) => i >= xManualRange[0] && i <= xManualRange[1])
      : allSlotIndices;
    const x = d3.scaleBand().domain(visibleArrayIndices).range([0, w]).padding(0.22);

    // Collect all prices for y-domain
    const allP = candles.flatMap((c) => [c.high, c.low]);
    if (analysis) {
      if (analysis.entry) allP.push(analysis.entry.price);
      if (analysis.stopLoss) allP.push(analysis.stopLoss.price);
      analysis.takeProfits?.forEach((t) => allP.push(t.price));
      analysis.liquidity?.forEach((l) => allP.push(l.price));
      analysis.orderBlocks?.forEach((ob) => { allP.push(ob.high); allP.push(ob.low); });
      // 4H dealing range
      if (analysis.htf_context) {
        if (analysis.htf_context.dealing_range_high) allP.push(analysis.htf_context.dealing_range_high);
        if (analysis.htf_context.dealing_range_low) allP.push(analysis.htf_context.dealing_range_low);
      }
    }
    if (calibration?.calibrated) {
      allP.push(calibration.calibrated.sl);
      calibration.calibrated.tps?.forEach((t) => allP.push(t));
    }

    const [mn, mx] = d3.extent(allP);
    const pad = (mx - mn) * 0.09;
    const yDomainAuto = [mn - pad, mx + pad];
    const y = d3.scaleLinear().domain(yManualDomain ?? yDomainAuto).range([h, 0]);

    // Grid
    g.selectAll(".hg").data(y.ticks(7)).join("line")
      .attr("x1", 0).attr("x2", w).attr("y1", (d) => y(d)).attr("y2", (d) => y(d))
      .attr("stroke", "#10101e").attr("stroke-width", 1);

    // ── ICT overlays ──
    if (analysis) {
      // Order Blocks — mechanical (solid border) vs Claude (dashed, dimmer)
      analysis.orderBlocks?.forEach((ob) => {
        const ci = Math.max(0, Math.min(ob.candleIndex ?? ob.index ?? 0, candles.length - 1));
        const ox = x(ci) ?? 0;
        const ow = Math.max(0, w - ox);
        const bCol = ob.type === "bullish" ? "#26a69a" : "#ef5350";
        const isMech = ob.source === "mechanical";
        const strength = ob.strength || "moderate";
        const opacity = isMech
          ? (strength === "strong" ? 0.18 : 0.12)
          : (strength === "strong" ? 0.1 : 0.06);
        g.append("rect").attr("x", ox).attr("y", y(ob.high)).attr("width", ow)
          .attr("height", Math.max(1, y(ob.low) - y(ob.high)))
          .attr("fill", ob.type === "bullish" ? `rgba(38,166,154,${opacity})` : `rgba(239,83,80,${opacity})`)
          .attr("stroke", bCol).attr("stroke-width", isMech ? 1.2 : 0.8)
          .attr("stroke-dasharray", isMech ? "none" : "4,3");
        const tag = ob.type === "bullish" ? "BULL" : "BEAR";
        const tfTag = ob.tf ? ` ${ob.tf}` : "";
        const suffix = strength === "strong" ? " ★" : "";
        const srcTag = isMech ? "" : " ᶜ";
        g.append("text").attr("x", ox + 3).attr("y", y(ob.high) - 2)
          .attr("fill", bCol).attr("font-size", "8px").attr("font-family", "monospace")
          .attr("opacity", isMech ? 1 : 0.7)
          .text(`${tag} OB${tfTag}${suffix}${srcTag}`);
      });

      // FVGs — mechanical (solid border) vs Claude (dashed, dimmer)
      // Filled FVGs rendered dimmer; open FVGs rendered brighter
      analysis.fvgs?.forEach((fvg) => {
        const si = Math.max(0, Math.min(fvg.startIndex ?? fvg.index ?? 0, candles.length - 1));
        const fx = x(si) ?? 0;
        const fw = Math.max(0, w - fx);
        const isFilled = fvg.filled === true || fvg.fill_percentage >= 100;
        const isMech = fvg.source === "mechanical";
        const fCol = fvg.type === "bullish" ? "#64b5f6" : "#ffa726";
        const fillOpacity = isFilled ? 0.03 : (isMech ? 0.14 : 0.08);
        const strokeOpacity = isFilled ? 0.3 : (isMech ? 0.9 : 0.6);
        g.append("rect").attr("x", fx).attr("y", y(fvg.high)).attr("width", fw)
          .attr("height", Math.max(1, y(fvg.low) - y(fvg.high)))
          .attr("fill", fvg.type === "bullish" ? `rgba(100,181,246,${fillOpacity})` : `rgba(255,167,38,${fillOpacity})`)
          .attr("stroke", fCol).attr("stroke-width", isFilled ? 0.5 : (isMech ? 1 : 0.6))
          .attr("stroke-dasharray", isMech ? "none" : "3,3").attr("opacity", strokeOpacity);
        const fvgTfTag = fvg.tf ? ` ${fvg.tf}` : "";
        const label = isFilled ? `FVG${fvgTfTag} ✗`
          : fvg.fill_percentage > 0 ? `FVG${fvgTfTag} ${Math.round(fvg.fill_percentage)}%`
          : `FVG${fvgTfTag}${isMech ? "" : " ᶜ"}`;
        g.append("text").attr("x", fx + 3).attr("y", y(fvg.high) - 2)
          .attr("fill", fCol).attr("font-size", "7.5px").attr("font-family", "monospace")
          .attr("opacity", isFilled ? 0.4 : (isMech ? 1 : 0.7)).text(label);
      });

      // Liquidity levels — grouped by (type, price-bucket, tf) so duplicate labels collapse to ×N
      const liqGroups = groupLiquidityByLevel(analysis.liquidity || [], 0.50);
      liqGroups.forEach((group) => {
        const liq = group.items[0]; // representative — leftmost candleIndex after group sort
        const count = group.items.length;
        const lCol = liq.type === "buyside" ? "#f5c842" : "#ff6b6b";
        const lci = Math.max(0, Math.min(liq.candleIndex ?? 0, candles.length - 1));
        const lx = x(lci) ?? 0;
        g.append("line").attr("x1", lx).attr("x2", w).attr("y1", y(liq.price)).attr("y2", y(liq.price))
          .attr("stroke", lCol).attr("stroke-width", 1).attr("stroke-dasharray", "7,4").attr("opacity", 0.8);
        const liqTfTag = liq.tf ? ` ${liq.tf}` : "";
        const countTag = count > 1 ? ` ×${count}` : "";
        g.append("text").attr("x", w + 3).attr("y", y(liq.price) + 4)
          .attr("fill", lCol).attr("font-size", "8px").attr("font-family", "monospace")
          .text(`${liq.type === "buyside" ? "BSL" : "SSL"}${liqTfTag}${countTag}`);
      });

      // ── 4H Dealing Range markers ──
      if (analysis.htf_context?.dealing_range_high && analysis.htf_context?.dealing_range_low) {
        const drh = analysis.htf_context.dealing_range_high;
        const drl = analysis.htf_context.dealing_range_low;
        const eq = (drh + drl) / 2;
        [
          [drh, "4H DR HIGH", 0.6], [drl, "4H DR LOW", 0.6], [eq, "EQ", 0.35],
        ].forEach(([price, label, opacity]) => {
          g.append("line").attr("x1", 0).attr("x2", w).attr("y1", y(price)).attr("y2", y(price))
            .attr("stroke", "#9370db").attr("stroke-width", 0.75).attr("stroke-dasharray", "3,6").attr("opacity", opacity);
          g.append("text").attr("x", w + 3).attr("y", y(price) + 4)
            .attr("fill", "#9370db").attr("font-size", "7px").attr("font-family", "monospace").attr("opacity", opacity)
            .text(label);
        });
      }

      // ── Dual SL/TP lines ──
      const hasCal = calibration?.calibrated;
      const claudeSL = analysis.stopLoss?.price;
      const calSL = hasCal ? calibration.calibrated.sl : null;
      const claudeTPs = analysis.takeProfits || [];
      const calTPs = hasCal ? (calibration.calibrated.tps || []) : [];

      // Claude lines (dashed, 50% opacity when calibration exists)
      const claudeOpacity = hasCal ? 0.4 : 1;
      const claudeDash = hasCal ? "6,3" : "0";

      // Trade levels anchor at the entry candle (last actual candle) and project
      // forward into future space \u2014 they describe the upcoming trade, not the past.
      const tradeAnchor = x(candles.length - 1) ?? 0;
      const labelX = tradeAnchor + 3;

      if (analysis.entry) {
        g.append("line").attr("x1", tradeAnchor).attr("x2", w).attr("y1", y(analysis.entry.price)).attr("y2", y(analysis.entry.price))
          .attr("stroke", "#f5c842").attr("stroke-width", 1.5);
        g.append("text").attr("x", labelX).attr("y", y(analysis.entry.price) - 3)
          .attr("fill", "#f5c842").attr("font-size", "9px").attr("font-family", "monospace").attr("font-weight", "bold")
          .text(`ENTRY ${analysis.entry.price.toFixed(2)}`);
        // Entry arrow
        const arrowX = w - 10;
        const arrowY = y(analysis.entry.price);
        const isLong = analysis.entry.direction === "long";
        g.append("path")
          .attr("d", isLong
            ? `M${arrowX},${arrowY + 6}L${arrowX + 7},${arrowY}L${arrowX},${arrowY - 6}Z`
            : `M${arrowX + 7},${arrowY + 6}L${arrowX},${arrowY}L${arrowX + 7},${arrowY - 6}Z`)
          .attr("fill", isLong ? "#26a69a" : "#ef5350");
      }

      if (claudeSL) {
        const slClose = calSL && Math.abs(claudeSL - calSL) < 0.50;
        if (!slClose || !hasCal) {
          g.append("line").attr("x1", tradeAnchor).attr("x2", w).attr("y1", y(claudeSL)).attr("y2", y(claudeSL))
            .attr("stroke", "#ef5350").attr("stroke-width", 1.5).attr("stroke-dasharray", claudeDash).attr("opacity", claudeOpacity);
          g.append("text").attr("x", labelX).attr("y", y(claudeSL) - 3)
            .attr("fill", "#ef5350").attr("font-size", "8px").attr("font-family", "monospace").attr("opacity", claudeOpacity)
            .text(hasCal ? `CLAUDE SL ${claudeSL.toFixed(2)}` : `SL ${claudeSL.toFixed(2)}`);
        }
      }
      claudeTPs.forEach((tp, i) => {
        const calTP = calTPs[i];
        const tpClose = calTP && Math.abs(tp.price - calTP) < 0.50;
        if (!tpClose || !hasCal) {
          g.append("line").attr("x1", tradeAnchor).attr("x2", w).attr("y1", y(tp.price)).attr("y2", y(tp.price))
            .attr("stroke", "#00e676").attr("stroke-width", 1).attr("stroke-dasharray", claudeDash).attr("opacity", claudeOpacity);
          g.append("text").attr("x", labelX).attr("y", y(tp.price) - 3)
            .attr("fill", "#00e676").attr("font-size", "8px").attr("font-family", "monospace").attr("opacity", claudeOpacity)
            .text(hasCal ? `CLAUDE TP${i + 1}` : `TP${i + 1} ${tp.price.toFixed(2)}${tp.rr ? ` (${tp.rr.toFixed(1)}R)` : ""}`);
        }
      });

      // Calibrated lines (solid, full opacity)
      if (hasCal) {
        if (calSL) {
          const slClose = claudeSL && Math.abs(claudeSL - calSL) < 0.50;
          g.append("line").attr("x1", tradeAnchor).attr("x2", w).attr("y1", y(calSL)).attr("y2", y(calSL))
            .attr("stroke", "#ef5350").attr("stroke-width", 1.5);
          g.append("text").attr("x", labelX).attr("y", y(calSL) - 3)
            .attr("fill", "#ef5350").attr("font-size", "9px").attr("font-family", "monospace").attr("font-weight", "bold")
            .text(slClose ? `SL ${calSL.toFixed(2)} \u2713` : `CAL SL ${calSL.toFixed(2)}`);
        }
        calTPs.forEach((tp, i) => {
          const claudeTP = claudeTPs[i]?.price;
          const tpClose = claudeTP && Math.abs(claudeTP - tp) < 0.50;
          const rr = calibration.calibrated.rr_ratios?.[i];
          g.append("line").attr("x1", tradeAnchor).attr("x2", w).attr("y1", y(tp)).attr("y2", y(tp))
            .attr("stroke", "#00e676").attr("stroke-width", 1.5);
          g.append("text").attr("x", labelX).attr("y", y(tp) - 3)
            .attr("fill", "#00e676").attr("font-size", "9px").attr("font-family", "monospace").attr("font-weight", "bold")
            .text(tpClose ? `TP${i + 1} ${tp.toFixed(2)} \u2713${rr ? ` (${rr.toFixed(1)}R)` : ""}` : `CAL TP${i + 1} ${tp.toFixed(2)}${rr ? ` (${rr.toFixed(1)}R)` : ""}`);
        });

        // SL buffer zone
        if (calibration.adjustments?.sl_widened && claudeSL && calSL) {
          const y1 = y(Math.max(claudeSL, calSL));
          const y2 = y(Math.min(claudeSL, calSL));
          g.append("rect").attr("x", tradeAnchor).attr("y", y1).attr("width", Math.max(0, w - tradeAnchor))
            .attr("height", Math.abs(y2 - y1))
            .attr("fill", "rgba(239,83,80,0.06)")
            .attr("stroke", "#ef5350").attr("stroke-width", 0.5).attr("stroke-dasharray", "2,4");
          g.append("text").attr("x", w - 3).attr("y", (y1 + y2) / 2 + 3)
            .attr("fill", "#ef5350").attr("font-size", "7px").attr("font-family", "monospace")
            .attr("text-anchor", "end").attr("opacity", 0.7)
            .text(`BUFFER +$${calibration.adjustments.sl_widened_by?.toFixed(2) || "?"}`);
        }
      }
    }

    // Candlesticks
    candles.forEach((c, i) => {
      const cx = x(i);
      if (cx === undefined) return;
      const bw = x.bandwidth();
      const bull = c.close >= c.open;
      const col = bull ? "#26a69a" : "#ef5350";
      g.append("line").attr("x1", cx + bw / 2).attr("x2", cx + bw / 2)
        .attr("y1", y(c.high)).attr("y2", y(c.low)).attr("stroke", col).attr("stroke-width", 1);
      g.append("rect").attr("x", cx).attr("y", y(Math.max(c.open, c.close)))
        .attr("width", bw).attr("height", Math.max(1, Math.abs(y(c.open) - y(c.close))))
        .attr("fill", col).attr("opacity", 0.88);
    });

    // Crosshair
    const ch = g.append("g").attr("class", "crosshair").style("display", "none").attr("pointer-events", "none");
    ch.append("line").attr("class", "ch-h").attr("x1", 0).attr("x2", w).attr("stroke", "#666").attr("stroke-width", 0.5).attr("stroke-dasharray", "4,3");
    ch.append("line").attr("class", "ch-v").attr("y1", 0).attr("y2", h).attr("stroke", "#666").attr("stroke-width", 0.5).attr("stroke-dasharray", "4,3");
    ch.append("rect").attr("class", "ch-price-bg").attr("x", w + 4).attr("width", 58).attr("height", 16).attr("rx", 2).attr("fill", "#1a1a2e");
    ch.append("text").attr("class", "ch-price").attr("x", w + 8).attr("fill", "#e0e0e0").attr("font-size", "9px").attr("font-family", "monospace").attr("dominant-baseline", "central");
    ch.append("rect").attr("class", "ch-time-bg").attr("y", h + 4).attr("width", 72).attr("height", 16).attr("rx", 2).attr("fill", "#1a1a2e");
    ch.append("text").attr("class", "ch-time").attr("y", h + 12).attr("fill", "#e0e0e0").attr("font-size", "9px").attr("font-family", "monospace").attr("text-anchor", "middle").attr("dominant-baseline", "central");
    ch.append("text").attr("class", "ch-ohlc").attr("x", 6).attr("y", 12).attr("font-size", "9px").attr("font-family", "monospace").attr("dominant-baseline", "hanging");

    chartScalesRef.current = { x, y, w, h, m };

    // Axes
    const tickPositions = [0, 0.25, 0.5, 0.75, 1];
    const ticks = tickPositions.map((p) => visibleArrayIndices[Math.min(visibleArrayIndices.length - 1, Math.floor(p * (visibleArrayIndices.length - 1)))]);
    g.append("g").attr("transform", `translate(0,${h})`)
      .call(d3.axisBottom(x).tickValues(ticks).tickFormat((i) => {
        const c = candles[i]; if (!c) return "";
        const [dd, tt] = c.datetime.split(" ");
        const [, mo, dy] = dd.split("-");
        return `${+mo}/${+dy} ${tt ? tt.slice(0, 5) : "00:00"}`;
      }))
      .call((ax) => {
        ax.select(".domain").attr("stroke", "#1a1a2e");
        ax.selectAll(".tick line").attr("stroke", "#1a1a2e");
        ax.selectAll("text").attr("fill", "#444466").attr("font-size", "8px").attr("font-family", "monospace");
      });
    g.append("text").attr("x", w).attr("y", h + 22).attr("fill", "#444466").attr("font-size", "8px").attr("font-family", "monospace").attr("text-anchor", "end").text("GMT");
    g.append("rect")
      .attr("class", "x-axis-hit")
      .attr("x", 0)
      .attr("y", h)
      .attr("width", w + m.right)
      .attr("height", m.bottom)
      .attr("fill", "transparent")
      .style("cursor", "ew-resize")
      .on("mousedown", (evt) => startDragRef.current?.("x-axis", evt))
      .on("dblclick", () => setXManualRange(null));
    g.append("g")
      .call(d3.axisLeft(y).ticks(7).tickFormat((d) => d.toFixed(0)))
      .call((ax) => {
        ax.select(".domain").attr("stroke", "#1a1a2e");
        ax.selectAll(".tick line").attr("stroke", "#1a1a2e");
        ax.selectAll("text").attr("fill", "#444466").attr("font-size", "9px").attr("font-family", "monospace");
      });
    g.append("rect")
      .attr("class", "y-axis-hit")
      .attr("x", -m.left)
      .attr("y", 0)
      .attr("width", m.left)
      .attr("height", h)
      .attr("fill", "transparent")
      .style("cursor", "ns-resize")
      .on("mousedown", (evt) => startDragRef.current?.("y-axis", evt))
      .on("dblclick", () => setYManualDomain(null));
  }, [candles, analysis, calibration, yManualDomain, xManualRange]);

  const startDrag = useCallback((kind, evt) => {
    const s = chartScalesRef.current;
    if (!s || !candles.length) return;

    const rect = svgRef.current.getBoundingClientRect();
    const startMouseX = evt.clientX - rect.left - s.m.left;
    const startMouseY = evt.clientY - rect.top - s.m.top;

    // Match the slot count drawChart uses so manual range can include future space.
    const FUTURE_BARS = Math.max(10, Math.ceil(candles.length * 0.2));
    const totalSlots = candles.length + FUTURE_BARS;
    const allCandleIndices = Array.from({ length: totalSlots }, (_, i) => i);
    const firstIdx = allCandleIndices[0];
    const lastIdx = allCandleIndices[allCandleIndices.length - 1];

    let anchorPrice = null;
    let anchorIndex = null;
    if (kind === "y-axis") {
      anchorPrice = s.y.invert(startMouseY);
    } else if (kind === "x-axis" || kind === "pan") {
      const step = s.x.step();
      anchorIndex = Math.max(
        0,
        Math.min(Math.round((startMouseX - s.x.bandwidth() / 2) / step), totalSlots - 1)
      );
    }

    if (kind === "pan" && xManualRange === null) return; // no-op in auto mode

    dragStateRef.current = {
      kind,
      startMouseX,
      startMouseY,
      startYDomain: yManualDomain ?? [s.y.domain()[0], s.y.domain()[1]],
      startXRange: xManualRange ?? [firstIdx, lastIdx],
      anchorPrice,
      anchorIndex,
      chartHeight: s.h,
      chartWidth: s.w,
      bandWidth: s.x.bandwidth(),
      allCandleIndices,
      rectLeft: rect.left + s.m.left,
      rectTop: rect.top + s.m.top,
    };
    setIsDragging(true);

    const handleMove = (e) => {
      const ds = dragStateRef.current;
      if (!ds) return;
      const dx = e.clientX - ds.rectLeft - ds.startMouseX;
      const dy = e.clientY - ds.rectTop - ds.startMouseY;

      if (ds.kind === "y-axis") {
        setYManualDomain(
          scaleYDomain({
            startDomain: ds.startYDomain,
            anchorPrice: ds.anchorPrice,
            deltaY: dy,
            chartHeight: ds.chartHeight,
          })
        );
      } else if (ds.kind === "x-axis") {
        setXManualRange(
          scaleXRange({
            startRange: ds.startXRange,
            anchorIndex: ds.anchorIndex,
            deltaX: dx,
            chartWidth: ds.chartWidth,
            allCandleIndices: ds.allCandleIndices,
          })
        );
      } else if (ds.kind === "pan") {
        setXManualRange(
          panXRange({
            startRange: ds.startXRange,
            deltaX: dx,
            bandWidth: ds.bandWidth,
            allCandleIndices: ds.allCandleIndices,
          })
        );
      }
    };

    const handleUp = () => {
      window.removeEventListener("mousemove", handleMove);
      window.removeEventListener("mouseup", handleUp);
      dragStateRef.current = null;
      setIsDragging(false);
      if (svgRef.current) {
        d3.select(svgRef.current).select(".crosshair").style("display", "none");
      }
    };

    window.addEventListener("mousemove", handleMove);
    window.addEventListener("mouseup", handleUp);
  }, [candles, yManualDomain, xManualRange, setYManualDomain, setXManualRange, setIsDragging]);

  const startDragRef = useRef(startDrag);
  useEffect(() => { startDragRef.current = startDrag; }, [startDrag]);

  useEffect(() => { drawChart(); }, [drawChart]);
  useEffect(() => {
    const handler = () => drawChart();
    window.addEventListener("resize", handler);
    return () => window.removeEventListener("resize", handler);
  }, [drawChart]);

  useEffect(() => {
    const svg = svgRef.current;
    if (!svg) return;
    const handler = (e) => {
      const s = chartScalesRef.current;
      if (!s || !candles.length) return;
      const rect = svg.getBoundingClientRect();
      const mx = e.clientX - rect.left - s.m.left;
      const my = e.clientY - rect.top - s.m.top;
      if (mx >= 0 && mx <= s.w && my >= 0 && my <= s.h) {
        e.preventDefault();
      }
    };
    svg.addEventListener("wheel", handler, { passive: false });
    return () => svg.removeEventListener("wheel", handler);
  }, [candles.length]);

  // ═══════════════════════════════════════════════════════════
  //  STYLES
  // ═══════════════════════════════════════════════════════════

  const btn = (active, color = "#f5c842") => ({
    background: active ? color : "transparent", border: `1px solid ${color}`,
    color: active ? "#08080f" : color, padding: "5px 12px", cursor: "pointer",
    fontSize: "9px", letterSpacing: "1.5px", fontFamily: "monospace", transition: "all 0.15s",
  });
  const inp = {
    background: "#0c0c18", border: "1px solid #1e1e30", color: "#cdd6f4",
    padding: "6px 10px", fontSize: "10px", fontFamily: "monospace", outline: "none", borderRadius: 0,
  };
  const sec = { background: "#0a0a16", border: "1px solid #14142a", padding: "8px 10px", marginBottom: "6px" };
  const secT = { color: "#f5c842", fontSize: "7px", letterSpacing: "3px", borderBottom: "1px solid #14142a", paddingBottom: "4px", marginBottom: "6px" };
  const miniBar = (pct, color) => ({
    height: 4, borderRadius: 2, background: "#14142a", position: "relative", overflow: "hidden",
    marginTop: 2, marginBottom: 6,
    ...(pct != null ? {} : {}),
  });

  // ═══════════════════════════════════════════════════════════
  //  RENDER HELPERS
  // ═══════════════════════════════════════════════════════════

  // ── Trade Panel ──
  const renderTradePanel = () => {
    if (!analysis) {
      return (
        <div style={{ padding: 16, textAlign: "center", marginTop: 40 }}>
          <div style={{ fontSize: 28, color: "#1e1e33", marginBottom: 10 }}>⬡</div>
          <div style={{ fontSize: 9, letterSpacing: 2, color: "#33334d" }}>AWAITING ANALYSIS</div>
          <div style={{ fontSize: 8, color: "#1e1e33", marginTop: 6 }}>Click ⬡ RUN ICT ANALYSIS</div>
          <button style={{ ...btn(true, "#26a69a"), marginTop: 16, width: "100%", padding: "10px" }}
            onClick={() => runAnalysis()} disabled={!!loadingStep || !candles.length}>
            ⬡ RUN ICT ANALYSIS
          </button>
        </div>
      );
    }
    if (!analysis.entry) {
      // Find best scanner setup to surface (prefer highest quality, most recent)
      const qualityOrder = { A: 0, B: 1, C: 2, D: 3 };
      const bestScannerSetup = scannerSetups?.length > 0
        ? [...scannerSetups].sort((a, b) => (qualityOrder[a.setup_quality] ?? 9) - (qualityOrder[b.setup_quality] ?? 9))[0]
        : null;

      return (
        <div style={{ padding: 12 }}>
          <div style={{ textAlign: "center", marginTop: 12 }}>
            <div style={{ fontSize: 16, color: "#33334d", marginBottom: 6 }}>○</div>
            <div style={{ fontSize: 9, letterSpacing: 2, color: "#33334d" }}>NO SETUP ON {(timeframe || "1h").toUpperCase()}</div>
          </div>
          <div style={{ ...sec, marginTop: 12 }}>
            <div style={{ color: "#6e7a9a", fontSize: 8, lineHeight: 1.7 }}>{analysis.summary}</div>
          </div>
          <div style={{ display: "flex", gap: 8, fontSize: 8, color: "#33334d", marginTop: 8 }}>
            <span>Bias: <span style={{ color: analysis.bias === "bullish" ? "#26a69a" : "#ef5350" }}>{analysis.bias?.toUpperCase()}</span></span>
            {analysis.killzone && <span>KZ: <span style={{ color: "#f5c842" }}>{analysis.killzone}</span></span>}
          </div>

          {/* Active thesis context */}
          {currentThesis?.thesis && (
            <div style={{ ...sec, marginTop: 8, borderLeft: `3px solid ${
              currentThesis.thesis.directional_bias === "bullish" ? "#26a69a" :
              currentThesis.thesis.directional_bias === "bearish" ? "#ef5350" : "#f5c842"
            }`, paddingLeft: 8, background: "rgba(245,200,66,0.03)" }}>
              <div style={{ fontSize: 7, color: "#f5c842", letterSpacing: 1, marginBottom: 3 }}>ACTIVE THESIS</div>
              <div style={{ display: "flex", gap: 4, alignItems: "center", marginBottom: 3 }}>
                <span style={{ color: currentThesis.thesis.directional_bias === "bullish" ? "#26a69a" : "#ef5350", fontSize: 9, fontWeight: 900 }}>
                  {currentThesis.thesis.directional_bias === "bullish" ? "\u25B2" : "\u25BC"}
                </span>
                <span style={{ fontSize: 8, color: "#cdd6f4" }}>
                  {(currentThesis.thesis.directional_bias || "").toUpperCase()}
                </span>
                <span style={{ fontSize: 7, color: "#ffa726" }}>
                  {((currentThesis.thesis.bias_confidence || 0) * 100).toFixed(0)}%
                </span>
                {currentThesis.thesis.p3_phase && (
                  <span style={{ fontSize: 6, padding: "1px 4px", background: "rgba(124,77,255,0.15)", color: "#7c4dff" }}>
                    {currentThesis.thesis.p3_phase.toUpperCase()} {currentThesis.thesis.p3_progress || ""}
                  </span>
                )}
              </div>
              {currentThesis.thesis.expected_next_move && (
                <div style={{ fontSize: 7, color: "#6e7a9a", lineHeight: 1.5 }}>
                  Next: {currentThesis.thesis.expected_next_move}
                </div>
              )}
              {currentThesis.thesis.invalidation?.price_level && (
                <div style={{ fontSize: 6, color: "#ef5350", marginTop: 2 }}>
                  Invalidation: {Number(currentThesis.thesis.invalidation.price_level).toFixed(2)} ({(currentThesis.thesis.invalidation.direction || "").toUpperCase()})
                </div>
              )}
              <div style={{ fontSize: 6, color: "#444466", marginTop: 2 }}>
                {currentThesis.thesis.scan_count || 1} scans \u00B7 {currentThesis.thesis.thesis_age_minutes || 0}min old
              </div>
            </div>
          )}

          {/* Surface best scanner setup */}
          {bestScannerSetup && (
            <div style={{ ...sec, marginTop: 12, borderLeft: "3px solid #7c4dff", paddingLeft: 8, background: "rgba(124,77,255,0.05)" }}>
              <div style={{ fontSize: 7, color: "#7c4dff", letterSpacing: 1, marginBottom: 4 }}>SCANNER FOUND SETUP</div>
              <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 4 }}>
                <span style={{ color: bestScannerSetup.direction === "long" ? "#26a69a" : "#ef5350", fontSize: 14, fontWeight: 900 }}>
                  {bestScannerSetup.direction === "long" ? "▲" : "▼"}
                </span>
                <span style={{ fontSize: 6, padding: "1px 4px", background: "rgba(245,200,66,0.15)", color: "#f5c842" }}>{bestScannerSetup.timeframe?.toUpperCase()}</span>
                {bestScannerSetup.setup_quality && (
                  <span style={{
                    background: bestScannerSetup.setup_quality === "A" ? "#26a69a" : bestScannerSetup.setup_quality === "B" ? "#f5c842" : "#666",
                    color: "#08080f", padding: "2px 6px", fontSize: 9, fontWeight: 900,
                  }}>{bestScannerSetup.setup_quality}</span>
                )}
                <span style={{ color: "#cdd6f4", fontSize: 10 }}>{bestScannerSetup.entry_price?.toFixed(2)}</span>
              </div>
              <div style={{ fontSize: 7, color: "#ffa726" }}>
                SL: {(bestScannerSetup.calibrated_sl || bestScannerSetup.sl_price)?.toFixed(2) || "—"} · TPs: {[bestScannerSetup.tp1, bestScannerSetup.tp2, bestScannerSetup.tp3].filter(Boolean).map(t => t.toFixed(2)).join(" / ") || "—"}
              </div>
              {bestScannerSetup.bias && <div style={{ fontSize: 7, color: "#6e7a9a", marginTop: 2 }}>Bias: {bestScannerSetup.bias}</div>}
              <div style={{ fontSize: 6, color: "#444466", marginTop: 4 }}>{bestScannerSetup.created_at?.replace("T", " ").slice(0, 16)}</div>
              {scannerSetups.length > 1 && (
                <div style={{ fontSize: 6, color: "#7c4dff", marginTop: 4 }}>+ {scannerSetups.length - 1} more setups in LOG tab</div>
              )}
            </div>
          )}

          <div style={{ display: "flex", gap: 6, marginTop: 12 }}>
            <button style={{ ...btn(false, "#26a69a"), flex: 1, fontSize: 8 }} onClick={() => { analysisCacheRef.current = { hash: "", result: null }; runAnalysis(); }}>⬡ RE-ANALYSE</button>
            <button style={{ ...btn(false), flex: 1, fontSize: 8 }} onClick={() => fetchCandles(false)}>↻ REFRESH</button>
          </div>
        </div>
      );
    }

    const cal = calibration;
    const co = cal?.claude_original;
    const cc = cal?.calibrated;
    const adj = cal?.adjustments;
    const conf = cal?.confidence;
    const dir = analysis.entry.direction;
    const dirColor = dir === "long" ? "#26a69a" : "#ef5350";

    return (
      <div style={{ padding: "8px 10px", fontSize: 8.5 }}>
        {/* Setup header */}
        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
          <span style={{ color: dirColor, fontSize: 16, fontWeight: 900 }}>{dir === "long" ? "▲" : "▼"}</span>
          {conf && (
            <span style={{
              background: GRADE_COLORS[conf.grade] || "#444", color: "#08080f",
              padding: "2px 8px", fontSize: 11, fontWeight: 900, borderRadius: 2,
            }}>{conf.grade}</span>
          )}
          {conf && <span style={{ color: "#cdd6f4", fontSize: 11 }}>{(conf.score * 100).toFixed(0)}%</span>}
        </div>
        {cal && <div style={{ color: "#33334d", fontSize: 7, letterSpacing: 1, marginBottom: 4 }}>
          {classifySetupType({ claude_direction: dir, has_ob: (analysis.orderBlocks?.length || 0) > 0, has_fvg: (analysis.fvgs?.length || 0) > 0, liq_swept: (analysis.liquidity || []).some(l => l.swept), claude_killzone: analysis.killzone }).toUpperCase()}
        </div>}
        {currentThesis?.thesis && (
          <div style={{ fontSize: 6, color: "#444466", marginBottom: 8 }}>
            Thesis: <span style={{ color: currentThesis.thesis.directional_bias === "bullish" ? "#26a69a" : "#ef5350" }}>
              {(currentThesis.thesis.directional_bias || "").toUpperCase()}
            </span> {"\u00B7"} {currentThesis.thesis.p3_phase || ""} {"\u00B7"} {((currentThesis.thesis.bias_confidence || 0) * 100).toFixed(0)}% conf {"\u00B7"} {currentThesis.thesis.scan_count || 1} scans
          </div>
        )}

        {/* Side-by-side levels */}
        <div style={sec}>
          <div style={{ display: "flex", justifyContent: "space-between", ...secT }}>
            <span>LEVELS</span>
            <span style={{ display: "flex", gap: 20 }}>
              <span style={{ color: "#6e7a9a" }}>CLAUDE</span>
              {cc && <span style={{ color: "#cdd6f4" }}>CALIBRATED</span>}
            </span>
          </div>
          {[
            { label: "ENTRY", color: "#f5c842", claude: co?.entry || analysis.entry.price, cal: cc?.entry },
            { label: "SL", color: "#ef5350", claude: co?.sl || analysis.stopLoss?.price, cal: cc?.sl, highlight: adj?.sl_widened },
            ...((co?.tps || analysis.takeProfits || []).map((tp, i) => ({
              label: `TP${i + 1}`, color: "#00e676",
              claude: typeof tp === "number" ? tp : tp?.price,
              cal: cc?.tps?.[i],
              rr: cc?.rr_ratios?.[i],
            }))),
          ].map((row) => (
            <div key={row.label} style={{
              display: "flex", justifyContent: "space-between", padding: "3px 4px",
              borderLeft: row.highlight ? "2px solid #ffa726" : "2px solid transparent",
              background: row.highlight ? "rgba(239,83,80,0.06)" : "transparent",
              marginBottom: 2,
            }}>
              <span style={{ color: row.color, minWidth: 28 }}>{row.label}</span>
              <span style={{ display: "flex", gap: 16 }}>
                <span style={{ color: "#6e7a9a", minWidth: 55, textAlign: "right" }}>{row.claude?.toFixed(2) || "—"}</span>
                {cc && <span style={{ color: "#cdd6f4", minWidth: 55, textAlign: "right", fontWeight: row.highlight ? 700 : 400 }}>
                  {row.cal?.toFixed(2) || "—"}{row.rr ? ` (${row.rr.toFixed(1)}R)` : ""}
                </span>}
              </span>
            </div>
          ))}
        </div>

        {/* SL widening callout */}
        {adj?.sl_widened && (
          <div style={{ ...sec, borderLeft: "2px solid #ffa726", background: "rgba(255,167,38,0.05)" }}>
            <div style={{ color: "#ffa726", fontSize: 8, fontWeight: 700, marginBottom: 4 }}>
              ⚠ SL WIDENED +${adj.sl_widened_by?.toFixed(2)} (+{adj.sl_widened_by_atr?.toFixed(1)} ATR)
            </div>
            <div style={{ color: "#6e7a9a", fontSize: 7.5 }}>Source: {cc?.sl_source}</div>
            <div style={{ color: "#444466", fontSize: 7, lineHeight: 1.6, marginTop: 3 }}>{adj.sl_widened_reason}</div>
          </div>
        )}

        {/* Confidence breakdown */}
        {conf && (
          <div style={sec}>
            <div style={secT}>CONFIDENCE BREAKDOWN</div>
            {[
              { label: "Claude signals", value: conf.claude_signal_strength, max: 1, display: `${(analysis.confluences?.length || 0)}/6` },
              { label: "Bayesian WR", value: conf.bayesian_win_rate, max: 1, display: `${(conf.bayesian_win_rate * 100).toFixed(0)}%` },
              { label: "Session WR", value: conf.session_win_rate, max: 1, display: `${(conf.session_win_rate * 100).toFixed(0)}%` },
              { label: "Historical", value: Math.min(conf.historical_match_count / 30, 1), max: 1, display: `${conf.historical_match_count} trades` },
              { label: "AutoGluon", value: conf.autogluon_win_prob, max: 1, display: conf.autogluon_win_prob != null ? `${(conf.autogluon_win_prob * 100).toFixed(0)}%` : "—" },
            ].map((item) => (
              <div key={item.label} style={{ marginBottom: 4 }}>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: 7, color: "#6e7a9a" }}>
                  <span>{item.label}</span>
                  <span style={{ color: item.value != null ? "#cdd6f4" : "#33334d" }}>{item.display}</span>
                </div>
                <div style={{ height: 4, borderRadius: 2, background: "#14142a", marginTop: 2 }}>
                  {item.value != null && (
                    <div style={{ height: "100%", borderRadius: 2, width: `${Math.min(item.value * 100, 100)}%`,
                      background: item.value > 0.5 ? "#26a69a" : item.value > 0.3 ? "#f5c842" : "#ef5350" }} />
                  )}
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Warnings */}
        {cal?.warnings?.length > 0 && (
          <div style={{ ...sec, borderLeft: "2px solid #ffa726", background: "rgba(255,167,38,0.04)" }}>
            <div style={secT}>WARNINGS</div>
            {cal.warnings.map((w, i) => (
              <div key={i} style={{ color: "#ffa726", fontSize: 7, lineHeight: 1.6, marginBottom: 3 }}>· {w}</div>
            ))}
          </div>
        )}

        {/* Recommendation */}
        {cal?.recommendation && (
          <div style={{ ...sec, background: "#06060e" }}>
            <div style={{ color: "#6e7a9a", fontSize: 7, lineHeight: 1.7, fontStyle: "italic" }}>"{cal.recommendation}"</div>
          </div>
        )}

        {/* Collapsible Claude narrative */}
        <div style={{ marginTop: 4 }}>
          <div style={{ cursor: "pointer", color: "#33334d", fontSize: 8, letterSpacing: 1.5, padding: "4px 0" }}
            onClick={() => setAnalysisExpanded(!analysisExpanded)}>
            {analysisExpanded ? "▾" : "▸"} CLAUDE'S ANALYSIS
          </div>
          {analysisExpanded && (
            <div style={{ ...sec, marginTop: 4 }}>
              <div style={{ color: "#6e7a9a", fontSize: 7.5, lineHeight: 1.7, marginBottom: 8 }}>{analysis.summary}</div>
              {analysis.htf_context && (
                <div style={{ marginBottom: 8 }}>
                  <div style={{ ...secT, fontSize: 6.5 }}>HTF CONTEXT</div>
                  <div style={{ fontSize: 7, color: "#444466", lineHeight: 2 }}>
                    <div>Dealing Range: {analysis.htf_context.dealing_range_low?.toFixed(2)} — {analysis.htf_context.dealing_range_high?.toFixed(2)}</div>
                    <div>Position: <span style={{ color: analysis.htf_context.premium_discount === "discount" ? "#26a69a" : "#ef5350" }}>{analysis.htf_context.premium_discount?.toUpperCase()}</span></div>
                    <div>Power of 3: {analysis.htf_context.power_of_3_phase}</div>
                    <div>Recent Sweep: {analysis.htf_context.recent_sweep}</div>
                    <div>4H Bias: <span style={{ color: analysis.htf_context.htf_bias === "bullish" ? "#26a69a" : "#ef5350" }}>{analysis.htf_context.htf_bias?.toUpperCase()}</span></div>
                  </div>
                </div>
              )}
              {analysis.confluences?.length > 0 && (
                <div>
                  <div style={{ ...secT, fontSize: 6.5 }}>CONFLUENCES ({analysis.confluences.length})</div>
                  {analysis.confluences.map((c, i) => (
                    <div key={i} style={{ color: "#6e7a9a", fontSize: 7, lineHeight: 1.8 }}>◦ {c}</div>
                  ))}
                </div>
              )}
              {analysis.orderBlocks?.length > 0 && (
                <div style={{ marginTop: 6 }}>
                  <div style={{ ...secT, fontSize: 6.5 }}>ORDER BLOCKS ({analysis.orderBlocks.length})</div>
                  {analysis.orderBlocks.map((ob, i) => (
                    <div key={i} style={{ borderLeft: `2px solid ${ob.type === "bullish" ? "#26a69a" : "#ef5350"}`, paddingLeft: 6, marginBottom: 4 }}>
                      <div style={{ color: ob.type === "bullish" ? "#26a69a" : "#ef5350", fontSize: 7 }}>{ob.type.toUpperCase()} OB · {ob.strength?.toUpperCase()} <span style={{ color: "#555", fontSize: 6 }}>{ob.source === "mechanical" ? "⚙" : "🤖"}</span></div>
                      <div style={{ color: "#444466", fontSize: 7 }}>{ob.high?.toFixed(2)} — {ob.low?.toFixed(2)}</div>
                    </div>
                  ))}
                </div>
              )}
              {analysis.fvgs?.length > 0 && (
                <div style={{ marginTop: 6 }}>
                  <div style={{ ...secT, fontSize: 6.5 }}>FAIR VALUE GAPS ({analysis.fvgs.length})</div>
                  {analysis.fvgs.map((fvg, i) => (
                    <div key={i} style={{ borderLeft: `2px solid ${fvg.type === "bullish" ? "#64b5f6" : "#ffa726"}`, paddingLeft: 6, marginBottom: 4 }}>
                      <div style={{ color: fvg.type === "bullish" ? "#64b5f6" : "#ffa726", fontSize: 7 }}>{fvg.type.toUpperCase()} FVG · {fvg.filled ? "FILLED" : "OPEN"} <span style={{ color: "#555", fontSize: 6 }}>{fvg.source === "mechanical" ? "⚙" : "🤖"}</span></div>
                      <div style={{ color: "#444466", fontSize: 7 }}>{fvg.high?.toFixed(2)} — {fvg.low?.toFixed(2)}</div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Action buttons */}
        <div style={{ display: "flex", gap: 6, marginTop: 8 }}>
          <button style={{ ...btn(false, "#26a69a"), flex: 1, fontSize: 8 }}
            onClick={() => { analysisCacheRef.current = { hash: "", result: null }; runAnalysis(); }}
            disabled={!!loadingStep}>⬡ RE-ANALYSE</button>
          <button style={{ ...btn(false), flex: 1, fontSize: 8 }} onClick={() => fetchCandles(false)} disabled={loadingData}>↻ REFRESH</button>
        </div>
      </div>
    );
  };

  // ── Intelligence Panel ──
  const renderIntelPanel = () => {
    const tabs = [
      { key: "ml", label: "ML" },
      { key: "bayes", label: "BAYES" },
      { key: "sessions", label: "SESSIONS" },
      { key: "accuracy", label: "ACCURACY" },
      { key: "pnl", label: "P&L" },
      { key: "prop", label: "PROP" },
      { key: "thesis", label: "THESIS" },
      { key: "log", label: `LOG (${journal.length + (scannerSetups?.length || 0)})` },
    ];

    return (
      <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
        {/* Tab bar */}
        <div style={{ display: "flex", borderBottom: "1px solid #14142a", flexShrink: 0 }}>
          {tabs.map((tab) => (
            <div key={tab.key} onClick={() => setActiveIntelTab(tab.key)} style={{
              flex: 1, padding: "7px 2px", textAlign: "center", cursor: "pointer",
              fontSize: 7, letterSpacing: 1.5,
              color: activeIntelTab === tab.key ? "#f5c842" : "#33334d",
              borderBottom: activeIntelTab === tab.key ? "1px solid #f5c842" : "none",
              background: activeIntelTab === tab.key ? "#0c0c18" : "transparent",
            }}>{tab.label}</div>
          ))}
        </div>

        {/* Tab content */}
        <div style={{ flex: 1, overflow: "auto", padding: "8px 8px" }}>
          {activeIntelTab === "ml" && renderMLTab()}
          {activeIntelTab === "bayes" && renderBayesTab()}
          {activeIntelTab === "sessions" && renderSessionsTab()}
          {activeIntelTab === "accuracy" && renderAccuracyTab()}
          {activeIntelTab === "pnl" && renderPnlTab()}
          {activeIntelTab === "prop" && renderPropTab()}
          {activeIntelTab === "thesis" && renderThesisTab()}
          {activeIntelTab === "log" && renderLogTab()}
        </div>
      </div>
    );
  };

  // ── ML Performance Dashboard Tab ──
  const renderMLTab = () => {
    const noData = !evalDual && !costBudget && !backtestMeta && !backtestStatus && !bayesianDrift && !modelInfo && !datasetStats && !accuracy && !calibrationValue && !scannerStatus && !calendarUpcoming && !calendarProximity;
    if (noData) return <div style={{ color: "#33334d", fontSize: 8, textAlign: "center", marginTop: 20 }}>No ML data. Start backend to populate.</div>;

    const pct = (v) => v != null ? `${(v * 100).toFixed(1)}%` : "—";
    const usd = (v) => v != null ? `$${Number(v).toFixed(2)}` : "—";
    const pill = (label, color) => (
      <span style={{ display: "inline-block", padding: "1px 5px", fontSize: 6, letterSpacing: 1, border: `1px solid ${color}`, color, marginLeft: 4 }}>{label}</span>
    );
    const row = (label, value, valueColor = "#cdd6f4") => (
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 7, padding: "2px 0" }}>
        <span style={{ color: "#444466" }}>{label}</span>
        <span style={{ color: valueColor, fontWeight: 600 }}>{value}</span>
      </div>
    );
    const bar = (value, max, color) => (
      <div style={{ height: 4, borderRadius: 2, background: "#14142a", marginTop: 3 }}>
        <div style={{ height: "100%", borderRadius: 2, width: `${Math.min((value / (max || 1)) * 100, 100)}%`, background: color }} />
      </div>
    );

    return (
      <>
        {/* ─── CLASSIFIER STATUS ─── */}
        {evalDual && (
          <div style={{ ...sec, borderLeft: "3px solid #7c4dff", paddingLeft: 8 }}>
            <div style={secT}>CLASSIFIER STATUS</div>
            {row("Model Type", (evalDual.active_gate_source === "live" ? "MULTI3 (live-validated)" : "MULTI3 (all-source)"), "#7c4dff")}
            {row("OOS Accuracy (all)", pct(evalDual.multi3_oos_all), evalDual.multi3_oos_all >= 0.45 ? "#26a69a" : "#ef5350")}
            {row("OOS Accuracy (live)", evalDual.multi3_oos_live != null ? pct(evalDual.multi3_oos_live) : "n/a", evalDual.multi3_oos_live != null && evalDual.multi3_oos_live >= 0.45 ? "#26a69a" : "#444466")}
            {row("Gate Threshold", pct(evalDual.gate_threshold), "#f5c842")}
            {row("Live Test Rows", evalDual.live_test_rows || 0)}
            {bar(evalDual.multi3_oos_all || 0, 1, evalDual.multi3_oos_all >= 0.45 ? "#26a69a" : "#ef5350")}
            <div style={{ fontSize: 6, color: "#444466", marginTop: 3 }}>
              Gate: {evalDual.multi3_oos_all >= 0.45 ? "✓ PASSED" : "✗ BELOW 45%"} · Source: {evalDual.active_gate_source || "all"}
            </div>
          </div>
        )}

        {/* ─── ML MODEL INFO ─── */}
        {modelInfo && modelInfo.status === "trained" && (
          <div style={{ ...sec, borderLeft: "3px solid #64b5f6", paddingLeft: 8 }}>
            <div style={secT}>AUTOGLUON MODEL</div>
            {row("Status", "TRAINED", "#26a69a")}
            {row("Best Model", modelInfo.best_model || "—")}
            {row("Ensemble Size", `${modelInfo.models_used?.length || 0} models`)}
            {modelInfo.leaderboard?.length > 0 && row("Top Score", pct(modelInfo.leaderboard[0].score_val), "#f5c842")}
            {modelInfo.leaderboard?.slice(0, 3).map((m, i) => (
              <div key={m.model} style={{ display: "flex", justifyContent: "space-between", fontSize: 6.5, padding: "1px 0", color: i === 0 ? "#f5c842" : "#444466" }}>
                <span>{i + 1}. {m.model}</span>
                <span>{pct(m.score_val)}</span>
              </div>
            ))}
            {/* Feature importance */}
            {modelInfo.feature_importances && (() => {
              const entries = Object.entries(modelInfo.feature_importances);
              const maxImp = Math.max(...entries.map(([, v]) => Math.abs(v)), 0.01);
              return entries.length > 0 && (
                <div style={{ marginTop: 4 }}>
                  <div style={{ fontSize: 6, color: "#444466", letterSpacing: 1, marginBottom: 2 }}>TOP FEATURES</div>
                  {entries.slice(0, 5).map(([feat, imp]) => (
                    <div key={feat} style={{ display: "flex", alignItems: "center", gap: 4, marginBottom: 2 }}>
                      <span style={{ fontSize: 6, color: "#888", width: 80, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{feat}</span>
                      <div style={{ flex: 1, height: 3, borderRadius: 1, background: "#14142a" }}>
                        <div style={{ height: "100%", borderRadius: 1, width: `${Math.abs(imp) / maxImp * 100}%`, background: "#7c4dff" }} />
                      </div>
                      <span style={{ fontSize: 6, color: "#7c4dff", width: 25, textAlign: "right" }}>{(Math.abs(imp) * 100).toFixed(0)}</span>
                    </div>
                  ))}
                </div>
              );
            })()}
          </div>
        )}

        {/* ─── DATASET COMPOSITION ─── */}
        {datasetStats && (
          <div style={sec}>
            <div style={secT}>DATASET COMPOSITION</div>
            {row("Total Rows", datasetStats.total || 0)}
            {row("WFO/Backtest", datasetStats.wfo_count || 0, "#64b5f6")}
            {row("Live Trades", datasetStats.live_count || 0, "#26a69a")}
            {datasetStats.backtest_count != null && row("Backtest (Claude)", datasetStats.backtest_count, "#ffa726")}
            {/* Outcome distribution */}
            {datasetStats.outcome_distribution && (() => {
              const od = datasetStats.outcome_distribution;
              const total = Object.values(od).reduce((a, b) => a + b, 0) || 1;
              return (
                <div style={{ marginTop: 4 }}>
                  <div style={{ fontSize: 6, color: "#444466", letterSpacing: 1, marginBottom: 2 }}>OUTCOME DISTRIBUTION</div>
                  {Object.entries(od).map(([outcome, count]) => {
                    const oColor = outcome === "stopped_out" ? "#ef5350" : outcome === "tp1" ? "#ffa726" : "#26a69a";
                    return (
                      <div key={outcome} style={{ display: "flex", alignItems: "center", gap: 4, marginBottom: 2 }}>
                        <span style={{ fontSize: 6, color: "#888", width: 60 }}>{outcome}</span>
                        <div style={{ flex: 1, height: 3, borderRadius: 1, background: "#14142a" }}>
                          <div style={{ height: "100%", borderRadius: 1, width: `${(count / total) * 100}%`, background: oColor }} />
                        </div>
                        <span style={{ fontSize: 6, color: oColor, width: 20, textAlign: "right" }}>{count}</span>
                      </div>
                    );
                  })}
                </div>
              );
            })()}
            {/* Regime distribution */}
            {datasetStats.regime_distribution && (() => {
              const rd = datasetStats.regime_distribution;
              const total = Object.values(rd).reduce((a, b) => a + b, 0) || 1;
              return (
                <div style={{ marginTop: 4 }}>
                  <div style={{ fontSize: 6, color: "#444466", letterSpacing: 1, marginBottom: 2 }}>REGIME DISTRIBUTION</div>
                  {Object.entries(rd).map(([regime, count]) => {
                    const rColor = regime === "trending" ? "#26a69a" : regime === "volatile" ? "#ef5350" : "#f5c842";
                    return (
                      <div key={regime} style={{ display: "flex", alignItems: "center", gap: 4, marginBottom: 2 }}>
                        <span style={{ fontSize: 6, color: "#888", width: 50 }}>{regime}</span>
                        <div style={{ flex: 1, height: 3, borderRadius: 1, background: "#14142a" }}>
                          <div style={{ height: "100%", borderRadius: 1, width: `${(count / total) * 100}%`, background: rColor }} />
                        </div>
                        <span style={{ fontSize: 6, color: rColor, width: 20, textAlign: "right" }}>{count}</span>
                      </div>
                    );
                  })}
                </div>
              );
            })()}
          </div>
        )}

        {/* ─── BACKTEST FIDELITY ─── */}
        {(backtestMeta || backtestStatus) && (
          <div style={{ ...sec, borderLeft: `3px solid ${backtestMeta?.checks_failed > 2 ? "#ef5350" : backtestMeta?.checks_failed > 0 ? "#ffa726" : "#26a69a"}`, paddingLeft: 8 }}>
            <div style={secT}>BACKTEST DATA</div>
            {backtestStatus && (
              <>
                {row("Setups Generated", backtestStatus.setups_generated || 0)}
                {backtestStatus.regime_counts && Object.entries(backtestStatus.regime_counts).map(([r, c]) => (
                  row(`  ${r}`, c, r === "trending" ? "#26a69a" : r === "volatile" ? "#ef5350" : "#f5c842")
                ))}
                {backtestStatus.cost_usd != null && row("Generation Cost", usd(backtestStatus.cost_usd))}
              </>
            )}
            {backtestMeta && (
              <>
                <div style={{ fontSize: 6, color: "#444466", letterSpacing: 1, marginTop: 4, marginBottom: 2 }}>FIDELITY CHECKS</div>
                {row("Checks Passed", `${(backtestMeta.checks_total || 5) - (backtestMeta.checks_failed || 0)} / ${backtestMeta.checks_total || 5}`,
                  backtestMeta.checks_failed > 2 ? "#ef5350" : backtestMeta.checks_failed > 0 ? "#ffa726" : "#26a69a")}
                {row("Adjusted Weight", backtestMeta.adjusted_weight != null ? backtestMeta.adjusted_weight.toFixed(2) : "—",
                  backtestMeta.adjusted_weight >= 0.6 ? "#26a69a" : "#ffa726")}
                {backtestMeta.checks && Object.entries(backtestMeta.checks).map(([name, result]) => (
                  <div key={name} style={{ display: "flex", justifyContent: "space-between", fontSize: 6.5, padding: "1px 0" }}>
                    <span style={{ color: "#888" }}>{name}</span>
                    <span style={{ color: result.passed ? "#26a69a" : "#ef5350" }}>{result.passed ? "✓" : "✗"}</span>
                  </div>
                ))}
              </>
            )}
          </div>
        )}

        {/* ─── CALIBRATION LAYERS ─── */}
        {(calibrationValue || accuracy) && (
          <div style={sec}>
            <div style={secT}>CALIBRATION PERFORMANCE</div>
            {accuracy && (
              <>
                {row("Claude Direction Acc", accuracy.total_trades > 0 ? pct(accuracy.claude_direction_correct / accuracy.total_trades) : "—", "#64b5f6")}
                {row("Claude SL Survival", accuracy.total_trades > 0 ? pct(accuracy.claude_sl_would_survive / accuracy.total_trades) : "—", "#ef5350")}
                {row("Calibrated SL Survival", accuracy.total_trades > 0 ? pct(accuracy.calibrated_sl_survived / accuracy.total_trades) : "—", "#26a69a")}
                {accuracy.trades_saved_by_calibration != null && row("Trades Saved", `+${accuracy.trades_saved_by_calibration}`, "#26a69a")}
                {accuracy.avg_sl_widening_atr != null && row("Avg SL Widening", `${accuracy.avg_sl_widening_atr.toFixed(1)} ATR`, "#ffa726")}
              </>
            )}
            {calibrationValue && (
              <>
                {row("Survival Improvement", calibrationValue.survival_improvement || "—", "#26a69a")}
                {row("Best Session", calibrationValue.best_session || "—", SESSION_COLORS[calibrationValue.best_session] || "#cdd6f4")}
                {row("Total Tracked", calibrationValue.total_trades || 0)}
              </>
            )}
            {/* Layer status indicators */}
            <div style={{ marginTop: 4 }}>
              <div style={{ fontSize: 6, color: "#444466", letterSpacing: 1, marginBottom: 2 }}>ACTIVE LAYERS</div>
              {[
                { name: "Volatility (ATR+Regime)", active: true, color: "#64b5f6" },
                { name: "V1 Session Stats", active: !!sessionStats?.seeded, color: "#ffa726" },
                { name: "Bayesian Updater", active: !!bayesian, color: "#f5c842" },
                { name: "AutoGluon Quantile", active: modelInfo?.status === "trained", color: "#7c4dff" },
                { name: "Historical Matching", active: true, color: "#26a69a" },
                { name: "Consensus (widest SL, median TP)", active: true, color: "#cdd6f4" },
              ].map((layer) => (
                <div key={layer.name} style={{ display: "flex", alignItems: "center", gap: 4, fontSize: 6.5, padding: "1px 0" }}>
                  <span style={{ width: 6, height: 6, borderRadius: "50%", background: layer.active ? layer.color : "#333", flexShrink: 0 }} />
                  <span style={{ color: layer.active ? "#888" : "#333" }}>{layer.name}</span>
                  {pill(layer.active ? "ON" : "OFF", layer.active ? layer.color : "#333")}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ─── BAYESIAN DRIFT ─── */}
        {bayesianDrift && (
          <div style={{ ...sec, borderLeft: `3px solid ${bayesianDrift.level === "critical" ? "#ef5350" : bayesianDrift.level === "significant" ? "#ffa726" : "#26a69a"}`, paddingLeft: 8 }}>
            <div style={secT}>BAYESIAN DRIFT</div>
            {bayesianDrift.drift_pp != null
              ? row("Drift", `${bayesianDrift.drift_pp}pp`, bayesianDrift.level === "critical" ? "#ef5350" : bayesianDrift.level === "significant" ? "#ffa726" : "#26a69a")
              : row("Drift SD", bayesianDrift.drift_sd != null ? bayesianDrift.drift_sd.toFixed(3) : "—", "#cdd6f4")}
            {bayesianDrift.posterior_wr != null && row("Current WR", pct(bayesianDrift.posterior_wr), "#f5c842")}
            {bayesianDrift.reference_wr != null && row("Reference WR", pct(bayesianDrift.reference_wr), "#444466")}
            {row("Level", (bayesianDrift.level || "none").toUpperCase(),
              bayesianDrift.level === "critical" ? "#ef5350" : bayesianDrift.level === "significant" ? "#ffa726" : "#26a69a")}
            <div style={{ fontSize: 6.5, color: "#6e7a9a", marginTop: 3, lineHeight: 1.4 }}>
              {bayesianDrift.recommendation || ""}
            </div>
          </div>
        )}

        {/* ─── COST TRACKING ─── */}
        {costBudget && (
          <div style={sec}>
            <div style={secT}>API COST (TODAY)</div>
            {row("Spent", usd(costBudget.spent_today_usd), costBudget.warning ? "#ef5350" : "#cdd6f4")}
            {row("Remaining", usd(costBudget.remaining_usd), costBudget.warning ? "#ef5350" : "#26a69a")}
            {row("API Calls", costBudget.call_count_today || 0)}
            {bar(costBudget.spent_today_usd || 0, costBudget.daily_limit_usd || 10, costBudget.warning ? "#ef5350" : "#f5c842")}
            {/* By purpose breakdown */}
            {costBudget.by_purpose && Object.keys(costBudget.by_purpose).length > 0 && (
              <div style={{ marginTop: 4 }}>
                <div style={{ fontSize: 6, color: "#444466", letterSpacing: 1, marginBottom: 2 }}>BY PURPOSE</div>
                {Object.entries(costBudget.by_purpose).map(([purpose, amount]) => (
                  <div key={purpose} style={{ display: "flex", justifyContent: "space-between", fontSize: 6.5, padding: "1px 0" }}>
                    <span style={{ color: "#888" }}>{purpose}</span>
                    <span style={{ color: "#f5c842" }}>{usd(amount)}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* ─── PROSPECT TRACKER ─── */}
        {prospects && (
          <div style={sec}>
            <div style={secT}>ACTIVE PROSPECTS</div>
            {row("Count", prospects.count || 0, "#f5c842")}
            {prospects.prospects?.length > 0 ? (
              prospects.prospects.slice(0, 5).map((p, i) => (
                <div key={i} style={{ display: "flex", justifyContent: "space-between", fontSize: 6.5, padding: "2px 0", borderBottom: "1px solid #0c0c18" }}>
                  <span style={{ color: SESSION_COLORS[mapKillzone(p.killzone)] || "#888" }}>
                    {p.symbol || "XAU"} · {p.timeframe || "?"} · {p.killzone || "?"}
                  </span>
                  <span style={{ color: p.phase === "setup" ? "#f5c842" : p.phase === "entry" ? "#26a69a" : "#888" }}>
                    {(p.phase || "?").toUpperCase()}
                  </span>
                </div>
              ))
            ) : (
              <div style={{ fontSize: 6.5, color: "#333", marginTop: 2 }}>No active prospects</div>
            )}
          </div>
        )}

        {/* ─── FOREX CALENDAR ─── */}
        {(calendarUpcoming || calendarProximity) && (
          <div style={sec}>
            <div style={secT}>FOREX CALENDAR (USD)</div>
            {calendarProximity && (() => {
              const stateColor = {
                imminent: "#ef5350",
                caution: "#f5c842",
                post_event: "#7c4dff",
                clear: "#26a69a",
                unavailable: "#444466",
              }[calendarProximity.state] || "#888";
              return (
                <>
                  {row("State", (calendarProximity.state || "—").toUpperCase(), stateColor)}
                  {calendarProximity.warning && (
                    <div style={{ fontSize: 6.5, color: stateColor, padding: "2px 0" }}>
                      ⚠ {calendarProximity.warning}
                    </div>
                  )}
                </>
              );
            })()}
            {calendarUpcoming?.events?.length > 0 ? (
              <div style={{ marginTop: 4 }}>
                <div style={{ fontSize: 6, color: "#444466", letterSpacing: 1, marginBottom: 2 }}>NEXT 24H</div>
                {calendarUpcoming.events.slice(0, 3).map((e) => (
                  <div key={e.event_id} style={{ display: "flex", justifyContent: "space-between", fontSize: 6.5, padding: "1px 0" }}>
                    <span style={{ color: "#cdd6f4" }}>{e.title}</span>
                    <span style={{ color: "#888" }}>{e.timestamp_utc?.slice(11, 16)}Z</span>
                  </div>
                ))}
                {calendarUpcoming.count > 3 && (
                  <div style={{ fontSize: 6, color: "#444466", marginTop: 2 }}>
                    + {calendarUpcoming.count - 3} more
                  </div>
                )}
              </div>
            ) : (
              <div style={{ fontSize: 6.5, color: "#333", marginTop: 4 }}>
                No high-impact USD events in the next 24h
              </div>
            )}
            <div style={{ marginTop: 6 }}>
              <button
                onClick={async () => {
                  try {
                    await fetch("/api/ml/calendar/refresh", { method: "POST" });
                    const [u, p] = await Promise.all([
                      fetch("/api/ml/calendar/upcoming?hours=24").then(r => r.ok ? r.json() : null),
                      fetch("/api/ml/calendar/proximity").then(r => r.ok ? r.json() : null),
                    ]);
                    if (u) setCalendarUpcoming(u);
                    if (p) setCalendarProximity(p);
                  } catch { /* network errors swallowed — UI keeps last state */ }
                }}
                style={{
                  fontSize: 6, padding: "2px 6px", letterSpacing: 1,
                  background: "transparent", color: "#7c4dff",
                  border: "1px solid #33334d", cursor: "pointer",
                }}
              >REFRESH NOW</button>
            </div>
          </div>
        )}

        {/* ─── SCANNER STATUS ─── */}
        {scannerStatus && (
          <div style={sec}>
            <div style={secT}>SCANNER</div>
            {row("Status", scannerStatus.running ? "RUNNING" : "STOPPED", scannerStatus.running ? "#26a69a" : "#ef5350")}
            {scannerStatus.timeframes && row("Timeframes", scannerStatus.timeframes.join(", "))}
            {scannerStatus.last_scan && row("Last Scan", new Date(scannerStatus.last_scan).toLocaleTimeString())}
            {scannerStatus.total_scans != null && row("Total Scans", scannerStatus.total_scans)}
            {scannerStatus.last_error && row("Last Error", scannerStatus.last_error, "#ef5350")}
          </div>
        )}
      </>
    );
  };

  const renderBayesTab = () => {
    const b = bayesian;
    const ss = sessionStats;
    if (!b && !ss) return <div style={{ color: "#33334d", fontSize: 8, textAlign: "center", marginTop: 20 }}>No Bayesian data. Start backend or use demo mode.</div>;

    return (
      <>
        {b && (
          <>
            <div style={sec}>
              <div style={secT}>WIN RATE BELIEF</div>
              <div style={{ fontSize: 16, color: "#f5c842", fontWeight: 700 }}>{(b.win_rate_mean * 100).toFixed(1)}%</div>
              <div style={{ height: 4, borderRadius: 2, background: "#14142a", marginTop: 4 }}>
                <div style={{ height: "100%", borderRadius: 2, width: `${b.win_rate_mean * 100}%`, background: "#f5c842" }} />
              </div>
              <div style={{ fontSize: 7, color: "#444466", marginTop: 3 }}>
                CI: {(b.win_rate_lower_95 * 100).toFixed(0)}% — {(b.win_rate_upper_95 * 100).toFixed(0)}% · {b.total_trades} trades
              </div>
            </div>
            <div style={sec}>
              <div style={secT}>DRAWDOWN (WINNERS)</div>
              <div style={{ fontSize: 7.5, color: "#ef5350", lineHeight: 2 }}>
                <div>Expected: {ss?.bayesian_priors?.drawdown_mu?.toFixed(2) || "?"} ATR</div>
                <div>→ Rec SL: {(ss?.bayesian_priors?.drawdown_mu * 2 || 0).toFixed(1)} ATR</div>
              </div>
            </div>
            <div style={sec}>
              <div style={secT}>FAVORABLE EXCURSION</div>
              <div style={{ fontSize: 7.5, color: "#00e676", lineHeight: 2 }}>
                <div>Expected: {ss?.bayesian_priors?.favorable_mu?.toFixed(2) || "?"} ATR</div>
              </div>
            </div>
          </>
        )}
        {/* Per setup type */}
        {accuracy?.by_setup_type && Object.keys(accuracy.by_setup_type).length > 0 && (
          <div style={sec}>
            <div style={secT}>SETUP TYPES</div>
            {Object.entries(accuracy.by_setup_type).map(([type, data]) => {
              const wr = data.trades > 0 ? data.wins / data.trades : 0;
              return (
                <div key={type} style={{ display: "flex", justifyContent: "space-between", fontSize: 7, color: "#6e7a9a", marginBottom: 3 }}>
                  <span style={{ maxWidth: 120, overflow: "hidden", textOverflow: "ellipsis" }}>{type}</span>
                  <span style={{ color: wr > 0.4 ? "#26a69a" : "#ef5350" }}>{data.trades}t {(wr * 100).toFixed(0)}%</span>
                </div>
              );
            })}
          </div>
        )}
      </>
    );
  };

  const renderSessionsTab = () => {
    // Build live killzone stats from scanner history
    const kzMap = { "Asian": "asia", "London": "london", "NY_AM": "ny_am", "NY_PM": "ny_pm", "Off": "off" };
    const kzMapRev = { "asia": "Asian", "london": "London", "ny_am": "NY_AM", "ny_pm": "NY_PM", "off": "Off" };
    const liveKZ = {};
    (scannerHistory || []).filter(t => t.outcome && t.outcome !== "expired").forEach(t => {
      const kzKey = kzMap[t.killzone] || t.killzone?.toLowerCase() || "off";
      if (!liveKZ[kzKey]) liveKZ[kzKey] = { wins: 0, losses: 0, total: 0, pnl: 0, trades: [], grades: {} };
      liveKZ[kzKey].total++;
      const isWin = t.outcome?.startsWith("tp");
      if (isWin) liveKZ[kzKey].wins++;
      else liveKZ[kzKey].losses++;
      liveKZ[kzKey].pnl += (t.pnl_rr || 0);
      const g = t.setup_quality || "?";
      liveKZ[kzKey].grades[g] = (liveKZ[kzKey].grades[g] || 0) + 1;
      liveKZ[kzKey].trades.push(t);
    });

    const ss = sessionStats?.session_stats;
    const hasLiveData = Object.keys(liveKZ).length > 0;
    const gates = killzoneGates?.gates || {};

    if (!ss && !hasLiveData) return <div style={{ color: "#33334d", fontSize: 8, textAlign: "center", marginTop: 20 }}>No session data</div>;

    const sessionOrder = ["london", "ny_am", "ny_pm", "asia", "off"];
    const sessionLabels = { london: "LONDON", ny_am: "NY AM", ny_pm: "NY PM", asia: "ASIAN", off: "OFF-SESSION" };
    const sessionHours = { london: "07:00-12:00", ny_am: "12:00-16:00", ny_pm: "16:00-20:00", asia: "00:00-07:00", off: "20:00-00:00" };

    return (
      <>
        {/* Live killzone performance header */}
        {hasLiveData && (
          <div style={{ ...sec, padding: "6px 10px" }}>
            <div style={secT}>KILLZONE PERFORMANCE</div>
            <div style={{ fontSize: 7, color: "#6e7a9a", marginBottom: 6 }}>
              {(scannerHistory || []).filter(t => t.outcome && t.outcome !== "expired").length} resolved setups from scanner
            </div>
          </div>
        )}

        {sessionOrder.map((session) => {
          const live = liveKZ[session];
          const seed = ss?.[session];
          const col = SESSION_COLORS[session] || "#444";
          const gateKey = kzMapRev[session];
          const gate = gates[gateKey];
          const wr = live ? live.wins / live.total : (seed?.win_rate || 0);
          const total = live?.total || seed?.trades || 0;
          if (total === 0) return null;

          // Recent trades (last 5)
          const recent = (live?.trades || [])
            .filter(t => t.resolved_at)
            .sort((a, b) => b.resolved_at.localeCompare(a.resolved_at))
            .slice(0, 5);

          return (
            <div key={session} style={{ ...sec, borderLeft: `3px solid ${col}`, paddingLeft: 8 }}>
              {/* Header row */}
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
                <div>
                  <div style={{ color: col, fontSize: 8, fontWeight: 700, letterSpacing: 1.5 }}>{sessionLabels[session]}</div>
                  <div style={{ color: "#33334d", fontSize: 6, marginTop: 1 }}>{sessionHours[session]} UTC</div>
                </div>
                {gate && (
                  <div style={{
                    padding: "1px 5px", fontSize: 6, letterSpacing: 1,
                    border: `1px solid ${col}33`, color: col, background: `${col}11`,
                  }}>
                    MIN: {gate.min_quality}
                  </div>
                )}
              </div>

              {/* Win rate bar */}
              <div style={{ position: "relative", height: 6, background: "#14142a", marginBottom: 6 }}>
                <div style={{
                  position: "absolute", height: "100%", width: `${Math.min(wr * 100, 100)}%`,
                  background: wr >= 0.6 ? "#26a69a" : wr >= 0.45 ? "#f5c842" : "#ef5350",
                  transition: "width 0.5s",
                }} />
                <div style={{
                  position: "absolute", right: 2, top: -1, fontSize: 6, color: "#cdd6f4", fontWeight: 700,
                }}>{total >= 5 ? `${(wr * 100).toFixed(0)}%` : "..."}</div>
              </div>

              {/* Stats grid */}
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "4px 6px", fontSize: 7, marginBottom: 4 }}>
                <div style={{ textAlign: "center" }}>
                  <div style={{ color: "#33334d", fontSize: 6 }}>TRADES</div>
                  <div style={{ color: "#cdd6f4", fontWeight: 600 }}>{total}</div>
                </div>
                <div style={{ textAlign: "center" }}>
                  <div style={{ color: "#33334d", fontSize: 6 }}>W / L</div>
                  <div>
                    <span style={{ color: "#26a69a" }}>{live?.wins || seed?.wins || 0}</span>
                    <span style={{ color: "#33334d" }}> / </span>
                    <span style={{ color: "#ef5350" }}>{live?.losses || (seed ? seed.trades - seed.wins : 0)}</span>
                  </div>
                </div>
                <div style={{ textAlign: "center" }}>
                  <div style={{ color: "#33334d", fontSize: 6 }}>P&L</div>
                  <div style={{ color: (live?.pnl || 0) >= 0 ? "#26a69a" : "#ef5350", fontWeight: 600 }}>
                    {live ? `${live.pnl >= 0 ? "+" : ""}${live.pnl.toFixed(1)}R` : "—"}
                  </div>
                </div>
              </div>

              {/* Seed baseline row (if available) */}
              {seed && seed.trades >= 5 && (
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "2px 6px", fontSize: 6, color: "#33334d", borderTop: "1px solid #14142a", paddingTop: 3, marginBottom: 3 }}>
                  <div>Med DD: <span style={{ color: "#ef5350" }}>{seed.median_drawdown?.toFixed(2) || "—"}</span></div>
                  <div>P95 DD: <span style={{ color: "#ef5350" }}>{seed.p95_drawdown?.toFixed(2) || "—"}</span></div>
                  <div>Med Fav: <span style={{ color: "#00e676" }}>{seed.median_favorable?.toFixed(2) || "—"}</span></div>
                </div>
              )}

              {/* Grade breakdown */}
              {live && Object.keys(live.grades).length > 0 && (
                <div style={{ display: "flex", gap: 4, marginTop: 2, marginBottom: 2 }}>
                  {["A", "B", "C"].filter(g => live.grades[g]).map(g => {
                    const cnt = live.grades[g];
                    const gCol = g === "A" ? "#26a69a" : g === "B" ? "#f5c842" : "#ef5350";
                    return (
                      <div key={g} style={{ fontSize: 6, color: gCol, background: `${gCol}11`, border: `1px solid ${gCol}33`, padding: "0 4px" }}>
                        {g}: {cnt}
                      </div>
                    );
                  })}
                </div>
              )}

              {/* Recent trades */}
              {recent.length > 0 && (
                <div style={{ marginTop: 3, borderTop: "1px solid #14142a", paddingTop: 3 }}>
                  <div style={{ fontSize: 6, color: "#33334d", marginBottom: 2, letterSpacing: 1 }}>RECENT</div>
                  {recent.map((t, i) => {
                    const isWin = t.outcome?.startsWith("tp");
                    const ago = t.resolved_at ? (() => {
                      const h = Math.floor((Date.now() - new Date(t.resolved_at).getTime()) / 3600000);
                      return h < 24 ? `${h}h` : `${Math.floor(h/24)}d`;
                    })() : "";
                    return (
                      <div key={i} style={{ display: "flex", justifyContent: "space-between", fontSize: 6, color: "#6e7a9a", marginBottom: 1 }}>
                        <span>
                          <span style={{ color: isWin ? "#26a69a" : "#ef5350", fontWeight: 600 }}>{t.outcome}</span>
                          {" "}{t.timeframe} {t.direction === "long" ? "\u2191" : "\u2193"}
                        </span>
                        <span style={{ color: "#33334d" }}>
                          {t.pnl_rr != null && <span style={{ color: t.pnl_rr >= 0 ? "#26a69a" : "#ef5350", marginRight: 4 }}>{t.pnl_rr >= 0 ? "+" : ""}{t.pnl_rr.toFixed(1)}R</span>}
                          {ago}
                        </span>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          );
        })}

        {/* Dataset composition */}
        <div style={{ ...sec, marginTop: 4 }}>
          <div style={secT}>DATASET</div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "2px 8px", fontSize: 7, color: "#444466" }}>
            <span>V1 Seed</span>
            <span style={{ color: "#6e7a9a" }}>{sessionStats?.dataset_stats?.wfo_count || 0} <span style={{ color: "#33334d" }}>(wt 0.5)</span></span>
            <span>Backtest</span>
            <span style={{ color: "#6e7a9a" }}>{sessionStats?.dataset_stats?.backtest_count || datasetStats?.backtest_count || 0} <span style={{ color: "#33334d" }}>(wt 0.7)</span></span>
            <span>Live</span>
            <span style={{ color: "#cdd6f4" }}>{sessionStats?.dataset_stats?.live_count || 0} <span style={{ color: "#33334d" }}>(wt 5.0)</span></span>
            <span style={{ borderTop: "1px solid #14142a", paddingTop: 2 }}>Total</span>
            <span style={{ color: "#cdd6f4", borderTop: "1px solid #14142a", paddingTop: 2, fontWeight: 600 }}>{sessionStats?.dataset_stats?.total || 0}</span>
          </div>
          <div style={{
            display: "inline-block", marginTop: 6, padding: "2px 6px", fontSize: 7, letterSpacing: 1,
            border: `1px solid ${(sessionStats?.dataset_stats?.live_count || 0) > 20 ? "#26a69a" : "#f5c842"}`,
            color: (sessionStats?.dataset_stats?.live_count || 0) > 20 ? "#26a69a" : "#f5c842",
          }}>
            {(sessionStats?.dataset_stats?.live_count || 0) > 20 ? "LIVE_VALIDATED" :
             (sessionStats?.dataset_stats?.live_count || 0) > 0 ? "TRANSITIONING" : "V1_SEED_ONLY"}
          </div>
        </div>
      </>
    );
  };

  const renderAccuracyTab = () => {
    const a = accuracy;
    const cv = calibrationValue;
    if (!a && !cv && !modelInfo) return <div style={{ color: "#33334d", fontSize: 8, textAlign: "center", marginTop: 20 }}>No accuracy data yet. Log trades to populate.</div>;

    const dirAcc = a?.total_trades > 0 ? a.claude_direction_correct / a.total_trades : 0;
    const claudeSurvival = a?.total_trades > 0 ? a.claude_sl_would_survive / a.total_trades : 0;
    const calSurvival = a?.total_trades > 0 ? a.calibrated_sl_survived / a.total_trades : 0;

    // Per-timeframe accuracy from resolved scanner history
    const tfStats = {};
    (scannerHistory || []).filter(t => t.outcome && t.outcome !== "expired").forEach(t => {
      const tf = t.timeframe || "?";
      if (!tfStats[tf]) tfStats[tf] = { wins: 0, losses: 0, total: 0 };
      tfStats[tf].total++;
      if (t.outcome?.startsWith("tp")) tfStats[tf].wins++;
      else tfStats[tf].losses++;
    });

    // Feature importance bar max
    const featEntries = modelInfo?.feature_importances ? Object.entries(modelInfo.feature_importances) : [];
    const maxImp = featEntries.length > 0 ? Math.max(...featEntries.map(([, v]) => Math.abs(v)), 0.01) : 1;

    return (
      <>
        {/* ML MODEL STATUS */}
        {modelInfo && modelInfo.status === "trained" && (
          <div style={{ ...sec, marginBottom: 6, borderLeft: "3px solid #7c4dff", paddingLeft: 8 }}>
            <div style={secT}>ML MODEL</div>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 7, padding: "2px 0" }}>
              <span style={{ color: "#444466" }}>Status</span>
              <span style={{ color: "#26a69a", fontWeight: 700 }}>TRAINED</span>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 7, padding: "2px 0" }}>
              <span style={{ color: "#444466" }}>Best Model</span>
              <span style={{ color: "#cdd6f4" }}>{modelInfo.best_model || "—"}</span>
            </div>
            {modelInfo.leaderboard?.length > 0 && (
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: 7, padding: "2px 0" }}>
                <span style={{ color: "#444466" }}>Top Accuracy</span>
                <span style={{ color: "#f5c842", fontWeight: 700 }}>{(modelInfo.leaderboard[0].score_val * 100).toFixed(1)}%</span>
              </div>
            )}
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 7, padding: "2px 0" }}>
              <span style={{ color: "#444466" }}>Models</span>
              <span style={{ color: "#cdd6f4" }}>{modelInfo.models_used?.length || 0} ensemble</span>
            </div>

            {/* Top 3 models */}
            {modelInfo.leaderboard?.slice(0, 3).map((m, i) => (
              <div key={m.model} style={{ display: "flex", justifyContent: "space-between", fontSize: 6.5, padding: "1px 0", color: i === 0 ? "#f5c842" : "#444466" }}>
                <span>{i + 1}. {m.model}</span>
                <span>{(m.score_val * 100).toFixed(1)}%</span>
              </div>
            ))}

            {/* Feature Importance bars */}
            {featEntries.length > 0 && (
              <div style={{ marginTop: 4 }}>
                <div style={{ fontSize: 6, color: "#444466", letterSpacing: 1, marginBottom: 2 }}>TOP FEATURES</div>
                {featEntries.slice(0, 5).map(([feat, imp]) => (
                  <div key={feat} style={{ display: "flex", alignItems: "center", gap: 4, marginBottom: 2 }}>
                    <span style={{ fontSize: 6, color: "#888", width: 80, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{feat}</span>
                    <div style={{ flex: 1, height: 3, borderRadius: 1, background: "#14142a" }}>
                      <div style={{ height: "100%", borderRadius: 1, width: `${Math.abs(imp) / maxImp * 100}%`, background: "#7c4dff" }} />
                    </div>
                    <span style={{ fontSize: 6, color: "#7c4dff", width: 25, textAlign: "right" }}>{(Math.abs(imp) * 100).toFixed(0)}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* PER-TIMEFRAME ACCURACY */}
        {Object.keys(tfStats).length > 0 && (
          <div style={{ ...sec, marginBottom: 6 }}>
            <div style={secT}>WIN RATE BY TIMEFRAME</div>
            {Object.entries(tfStats).sort((a, b) => (b[1].wins / b[1].total) - (a[1].wins / a[1].total)).map(([tf, s]) => {
              const wr = s.total > 0 ? s.wins / s.total : 0;
              return (
                <div key={tf} style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 3 }}>
                  <span style={{ fontSize: 7, color: "#f5c842", width: 30, fontWeight: 700 }}>{tf.toUpperCase()}</span>
                  <div style={{ flex: 1, height: 4, borderRadius: 2, background: "#14142a" }}>
                    <div style={{ height: "100%", borderRadius: 2, width: `${wr * 100}%`, background: wr >= 0.6 ? "#26a69a" : wr >= 0.4 ? "#ffa726" : "#ef5350" }} />
                  </div>
                  <span style={{ fontSize: 7, color: wr >= 0.6 ? "#26a69a" : wr >= 0.4 ? "#ffa726" : "#ef5350", fontWeight: 700, width: 30, textAlign: "right" }}>{(wr * 100).toFixed(0)}%</span>
                  <span style={{ fontSize: 6, color: "#444466", width: 30 }}>{s.wins}W/{s.losses}L</span>
                </div>
              );
            })}
          </div>
        )}

        {a && (
          <>
            <div style={sec}>
              <div style={secT}>DIRECTION ACCURACY</div>
              <div style={{ fontSize: 14, color: dirAcc > 0.5 ? "#26a69a" : "#ef5350", fontWeight: 700 }}>
                {(dirAcc * 100).toFixed(0)}%
              </div>
              <div style={{ height: 4, borderRadius: 2, background: "#14142a", marginTop: 3 }}>
                <div style={{ height: "100%", borderRadius: 2, width: `${dirAcc * 100}%`, background: dirAcc > 0.5 ? "#26a69a" : "#ef5350" }} />
              </div>
              <div style={{ fontSize: 7, color: "#444466", marginTop: 2 }}>{a.claude_direction_correct}/{a.total_trades} trades</div>
            </div>

            <div style={sec}>
              <div style={secT}>SL SURVIVAL</div>
              <div style={{ fontSize: 7.5, color: "#ffa726", marginBottom: 3 }}>Claude alone: {(claudeSurvival * 100).toFixed(0)}%</div>
              <div style={{ height: 4, borderRadius: 2, background: "#14142a", marginBottom: 6 }}>
                <div style={{ height: "100%", borderRadius: 2, width: `${claudeSurvival * 100}%`, background: "#ffa726" }} />
              </div>
              <div style={{ fontSize: 7.5, color: "#26a69a", marginBottom: 3 }}>Calibrated: {(calSurvival * 100).toFixed(0)}%</div>
              <div style={{ height: 4, borderRadius: 2, background: "#14142a" }}>
                <div style={{ height: "100%", borderRadius: 2, width: `${calSurvival * 100}%`, background: "#26a69a" }} />
              </div>
            </div>

            <div style={{ textAlign: "center", padding: "12px 0" }}>
              <div style={{ fontSize: 24, color: "#f5c842", fontWeight: 900 }}>{a.trades_saved_by_calibration}</div>
              <div style={{ fontSize: 7, color: "#33334d", letterSpacing: 2 }}>TRADES SAVED BY CALIBRATION</div>
              {cv?.survival_improvement && <div style={{ fontSize: 10, color: "#26a69a", fontWeight: 700, marginTop: 3 }}>{cv.survival_improvement}</div>}
            </div>

            <div style={sec}>
              <div style={secT}>AVG SL DISTANCE</div>
              <div style={{ fontSize: 7.5, color: "#ffa726" }}>Claude: {a.avg_claude_sl_distance_atr?.toFixed(1)} ATR</div>
              <div style={{ height: 4, borderRadius: 2, background: "#14142a", marginTop: 2, marginBottom: 6 }}>
                <div style={{ height: "100%", borderRadius: 2, width: `${Math.min(a.avg_claude_sl_distance_atr / 3 * 100, 100)}%`, background: "#ffa726" }} />
              </div>
              <div style={{ fontSize: 7.5, color: "#26a69a" }}>Calibrated: {a.avg_calibrated_sl_distance_atr?.toFixed(1)} ATR</div>
              <div style={{ height: 4, borderRadius: 2, background: "#14142a", marginTop: 2 }}>
                <div style={{ height: "100%", borderRadius: 2, width: `${Math.min(a.avg_calibrated_sl_distance_atr / 3 * 100, 100)}%`, background: "#26a69a" }} />
              </div>
              <div style={{ fontSize: 7, color: "#444466", marginTop: 3 }}>Avg widening: +{a.avg_sl_widening_atr?.toFixed(1)} ATR</div>
            </div>
          </>
        )}
      </>
    );
  };

  // ── Prop Firm Account Tracker ──
  const PROP_PRESETS = {
    one_phase_micro: { name: "One-Phase Micro $100K", size: 100000, dailyDD: 0.04, maxDD: 0.07, profitTarget: 0.07, splitPct: 0.90, phases: 1, price: 221 },
    if_micro:        { name: "IF Micro $100K",        size: 100000, dailyDD: 0.04, maxDD: 0.06, profitTarget: null, splitPct: 0.90, phases: 0, price: 833 },
    if1:             { name: "IF1 $100K",              size: 100000, dailyDD: 0.02, maxDD: 0.04, profitTarget: null, splitPct: 0.90, phases: 0, price: 472 },
    if_go:           { name: "IF GO $100K",            size: 100000, dailyDD: 0.04, maxDD: 0.10, profitTarget: null, splitPct: 0.90, phases: 0, price: 2042 },
    instant_funding: { name: "Instant Funding $100K",  size: 100000, dailyDD: null,  maxDD: 0.10, profitTarget: null, splitPct: 0.90, phases: 0, price: 3173 },
    one_phase:       { name: "One-Phase $100K",        size: 100000, dailyDD: 0.03, maxDD: 0.08, profitTarget: 0.08, splitPct: 0.90, phases: 1, price: 787 },
    two_phase:       { name: "Two-Phase $100K",        size: 100000, dailyDD: 0.05, maxDD: 0.12, profitTarget: 0.08, splitPct: 0.90, phases: 2, price: 1334 },
    two_phase_max:   { name: "Two-Phase Max $100K",    size: 100000, dailyDD: 0.04, maxDD: 0.10, profitTarget: 0.08, splitPct: 0.95, phases: 2, price: 809 },
  };

  const renderPropTab = () => {
    const trades = (pnlHistory || []).filter(t => t.pnl_rr != null);
    const acct = PROP_PRESETS[propAccount] || PROP_PRESETS.one_phase_micro;
    const riskPct = parseFloat(propRiskPct) || 0.5;
    const bal = acct.size;
    const riskAmt = bal * (riskPct / 100);
    const spread = parseFloat(pnlSpread) || 0.50;

    // Simulate account equity
    let equity = bal;
    let peak = bal;
    let maxDD = 0;
    let maxDailyDD = 0;
    let dayPnl = 0;
    let prevDay = null;
    let blown = false;
    let blownTrade = -1;
    let passed = false;
    let passedTrade = -1;

    const tradesWithEquity = trades.map((t, i) => {
      const costRR = t.cost_rr != null ? t.cost_rr : spread / 15;
      const netRR = t.pnl_rr;
      const pnlDollar = netRR * riskAmt;

      // Track daily DD
      const day = t.resolved_at ? t.resolved_at.slice(0, 10) : "";
      if (day !== prevDay) { dayPnl = 0; prevDay = day; }
      dayPnl += pnlDollar;
      if (dayPnl < 0) {
        const dailyDDPct = Math.abs(dayPnl) / bal;
        if (dailyDDPct > maxDailyDD) maxDailyDD = dailyDDPct;
      }

      equity += pnlDollar;
      if (equity > peak) peak = equity;
      const dd = peak - equity;
      const ddPct = dd / bal;
      if (ddPct > maxDD) maxDD = ddPct;

      // Check blown
      if (!blown) {
        if (acct.maxDD && ddPct >= acct.maxDD) { blown = true; blownTrade = i; }
        if (acct.dailyDD && dayPnl < 0 && Math.abs(dayPnl) / bal >= acct.dailyDD) { blown = true; blownTrade = i; }
      }

      // Check passed
      if (!passed && !blown && acct.profitTarget) {
        if ((equity - bal) / bal >= acct.profitTarget) { passed = true; passedTrade = i; }
      }

      return { ...t, equity, ddPct, netRR, pnlDollar };
    });

    const finalEquity = equity;
    const totalPnl = finalEquity - bal;
    const totalPnlPct = (totalPnl / bal) * 100;
    const status = blown ? "BLOWN" : passed ? "PASSED" : "ACTIVE";
    const statusColor = blown ? "#ef5350" : passed ? "#26a69a" : "#f5c842";

    // Daily DD usage
    const dailyDDLimit = acct.dailyDD ? acct.dailyDD * bal : null;
    const dailyDDUsed = acct.dailyDD ? maxDailyDD / acct.dailyDD : 0;

    // Max DD usage
    const maxDDLimit = acct.maxDD * bal;
    const maxDDUsed = maxDD / acct.maxDD;

    // Profit target progress
    const targetAmt = acct.profitTarget ? acct.profitTarget * bal : null;
    const targetProg = targetAmt ? Math.max(0, totalPnl) / targetAmt : 0;

    // Safe risk calculation
    const safeRiskPerTrade = dailyDDLimit ? dailyDDLimit / 8 : bal * 0.03;

    const statBox = { display: "flex", justifyContent: "space-between", fontSize: 7, padding: "3px 0", borderBottom: "1px solid #14142a" };
    const statLabel = { color: "#444466" };
    const statVal = { color: "#cdd6f4", fontFamily: "monospace" };
    const barBg = { height: 6, borderRadius: 3, background: "#14142a", overflow: "hidden", marginTop: 3, marginBottom: 4 };

    // Simulate all accounts
    const simAll = Object.entries(PROP_PRESETS).map(([key, a]) => {
      let eq = a.size;
      let pk = a.size;
      let mxDD = 0;
      let dPnl = 0;
      let pDay = null;
      let bl = false;
      let blAt = -1;
      let ps = false;
      let psAt = -1;

      trades.forEach((t, i) => {
        const ra = a.size * (riskPct / 100);
        const pnl = t.pnl_rr * ra;
        const day = t.resolved_at ? t.resolved_at.slice(0, 10) : "";
        if (day !== pDay) { dPnl = 0; pDay = day; }
        dPnl += pnl;
        eq += pnl;
        if (eq > pk) pk = eq;
        const dd = (pk - eq) / a.size;
        if (dd > mxDD) mxDD = dd;

        if (!bl) {
          if (a.maxDD && dd >= a.maxDD) { bl = true; blAt = i; }
          if (a.dailyDD && dPnl < 0 && Math.abs(dPnl) / a.size >= a.dailyDD) { bl = true; blAt = i; }
        }
        if (!ps && !bl && a.profitTarget) {
          if ((eq - a.size) / a.size >= a.profitTarget) { ps = true; psAt = i; }
        }
      });

      return { key, name: a.name, size: a.size, blown: bl, blownAt: blAt, passed: ps, passedAt: psAt, finalEq: eq, maxDD: mxDD, maxDDLimit: a.maxDD, price: a.price };
    });

    return (
      <>
        {/* Account Selector */}
        <div style={{ ...sec, marginBottom: 6 }}>
          <div style={secT}>PROP FIRM ACCOUNT</div>
          <select value={propAccount} onChange={(e) => setPropAccount(e.target.value)}
            style={{ ...inp, width: "100%", boxSizing: "border-box", cursor: "pointer", marginBottom: 6 }}>
            {Object.entries(PROP_PRESETS).map(([k, v]) => (
              <option key={k} value={k}>{v.name} - {"\u00A3"}{v.price}</option>
            ))}
          </select>
          <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
            <div style={{ flex: 1 }}>
              <div style={{ fontSize: 6, color: "#444466", marginBottom: 2 }}>RISK %</div>
              <input type="text" inputMode="decimal" value={propRiskPct}
                onChange={(e) => { const v = e.target.value; if (v === "" || /^\d*\.?\d*$/.test(v)) setPropRiskPct(v === "" ? "" : v); }}
                onBlur={() => { const n = parseFloat(propRiskPct); setPropRiskPct(isNaN(n) || n <= 0 ? 0.5 : Math.min(5, n)); }}
                style={{ ...inp, width: "100%", boxSizing: "border-box" }} />
            </div>
            <div style={{ flex: 1 }}>
              <div style={{ fontSize: 6, color: "#444466", marginBottom: 2 }}>RISK/TRADE</div>
              <div style={{ ...inp, width: "100%", boxSizing: "border-box", background: "#0a0a16" }}>${riskAmt.toFixed(0)}</div>
            </div>
            <div style={{ flex: 1 }}>
              <div style={{ fontSize: 6, color: "#444466", marginBottom: 2 }}>SAFE RISK</div>
              <div style={{ ...inp, width: "100%", boxSizing: "border-box", background: "#0a0a16", color: "#26a69a" }}>${safeRiskPerTrade.toFixed(0)}</div>
            </div>
          </div>
        </div>

        {/* Account Rules */}
        <div style={{ ...sec, marginBottom: 6 }}>
          <div style={secT}>ACCOUNT RULES</div>
          <div style={statBox}><span style={statLabel}>Account Size</span><span style={statVal}>${acct.size.toLocaleString()}</span></div>
          <div style={statBox}><span style={statLabel}>Daily Drawdown</span><span style={statVal}>{acct.dailyDD ? `${(acct.dailyDD * 100).toFixed(0)}% ($${(acct.dailyDD * bal).toLocaleString()})` : "None"}</span></div>
          <div style={statBox}><span style={statLabel}>Max Drawdown</span><span style={statVal}>{(acct.maxDD * 100).toFixed(0)}% (${acct.maxDD === 0.10 ? "Smart" : "Static"}) = ${maxDDLimit.toLocaleString()}</span></div>
          <div style={statBox}><span style={statLabel}>Profit Target</span><span style={statVal}>{acct.profitTarget ? `${(acct.profitTarget * 100).toFixed(0)}% = $${targetAmt.toLocaleString()}` : "None (Instant)"}</span></div>
          <div style={statBox}><span style={statLabel}>Profit Split</span><span style={statVal}>{(acct.splitPct * 100).toFixed(0)}%</span></div>
          <div style={{ ...statBox, borderBottom: "none" }}><span style={statLabel}>Phases</span><span style={statVal}>{acct.phases === 0 ? "Instant Funding" : `${acct.phases}-Phase`}</span></div>
        </div>

        {/* Account Status */}
        <div style={{ ...sec, marginBottom: 6 }}>
          <div style={secT}>SIMULATION STATUS</div>
          <div style={{ textAlign: "center", padding: "6px 0 4px" }}>
            <span style={{ fontSize: 14, fontWeight: 900, color: statusColor, letterSpacing: 2 }}>{status}</span>
            {blown && <div style={{ fontSize: 7, color: "#ef5350", marginTop: 2 }}>Blown on trade #{blownTrade + 1}</div>}
            {passed && <div style={{ fontSize: 7, color: "#26a69a", marginTop: 2 }}>Passed on trade #{passedTrade + 1}</div>}
          </div>

          <div style={statBox}>
            <span style={statLabel}>Equity</span>
            <span style={{ ...statVal, color: totalPnl >= 0 ? "#26a69a" : "#ef5350" }}>
              ${finalEquity.toLocaleString(undefined, {maximumFractionDigits: 0})} ({totalPnl >= 0 ? "+" : ""}{totalPnlPct.toFixed(1)}%)
            </span>
          </div>

          {/* Max DD bar */}
          <div style={{ ...statBox, flexDirection: "column", gap: 2 }}>
            <div style={{ display: "flex", justifyContent: "space-between", width: "100%" }}>
              <span style={statLabel}>Max DD Used</span>
              <span style={{ ...statVal, color: maxDDUsed > 0.75 ? "#ef5350" : maxDDUsed > 0.5 ? "#ff9800" : "#26a69a" }}>
                {(maxDD * 100).toFixed(1)}% / {(acct.maxDD * 100).toFixed(0)}%
              </span>
            </div>
            <div style={barBg}>
              <div style={{ height: "100%", width: `${Math.min(maxDDUsed * 100, 100)}%`, borderRadius: 3,
                background: maxDDUsed > 0.75 ? "#ef5350" : maxDDUsed > 0.5 ? "#ff9800" : "#26a69a" }} />
            </div>
          </div>

          {/* Daily DD bar */}
          {acct.dailyDD && (
            <div style={{ ...statBox, flexDirection: "column", gap: 2 }}>
              <div style={{ display: "flex", justifyContent: "space-between", width: "100%" }}>
                <span style={statLabel}>Worst Daily DD</span>
                <span style={{ ...statVal, color: dailyDDUsed > 0.75 ? "#ef5350" : dailyDDUsed > 0.5 ? "#ff9800" : "#26a69a" }}>
                  {(maxDailyDD * 100).toFixed(1)}% / {(acct.dailyDD * 100).toFixed(0)}%
                </span>
              </div>
              <div style={barBg}>
                <div style={{ height: "100%", width: `${Math.min(dailyDDUsed * 100, 100)}%`, borderRadius: 3,
                  background: dailyDDUsed > 0.75 ? "#ef5350" : dailyDDUsed > 0.5 ? "#ff9800" : "#26a69a" }} />
              </div>
            </div>
          )}

          {/* Profit target bar */}
          {acct.profitTarget && (
            <div style={{ ...statBox, flexDirection: "column", gap: 2, borderBottom: "none" }}>
              <div style={{ display: "flex", justifyContent: "space-between", width: "100%" }}>
                <span style={statLabel}>Profit Target</span>
                <span style={{ ...statVal, color: passed ? "#26a69a" : "#f5c842" }}>
                  {totalPnl >= 0 ? "$" + totalPnl.toFixed(0) : "-$" + Math.abs(totalPnl).toFixed(0)} / ${targetAmt.toLocaleString()}
                </span>
              </div>
              <div style={barBg}>
                <div style={{ height: "100%", width: `${Math.min(targetProg * 100, 100)}%`, borderRadius: 3,
                  background: passed ? "#26a69a" : "#f5c842" }} />
              </div>
            </div>
          )}
        </div>

        {/* Simulation Comparison */}
        <div style={{ ...sec, marginBottom: 6 }}>
          <div style={secT}>ALL ACCOUNTS vs YOUR {trades.length} TRADES</div>
          <div style={{ fontSize: 6, color: "#444466", marginBottom: 4 }}>At {riskPct}% risk per trade</div>
          {simAll.sort((a, b) => {
            // Sort: passed first (by earliest pass), then active, then blown (by latest blow)
            if (a.passed && !b.passed) return -1;
            if (!a.passed && b.passed) return 1;
            if (a.passed && b.passed) return a.passedAt - b.passedAt;
            if (!a.blown && b.blown) return -1;
            if (a.blown && !b.blown) return 1;
            return (b.finalEq - b.size) - (a.finalEq - a.size);
          }).map((s) => {
            const st = s.blown ? "BLOWN" : s.passed ? "PASSED" : "ACTIVE";
            const col = s.blown ? "#ef5350" : s.passed ? "#26a69a" : "#f5c842";
            const pnl = s.finalEq - s.size;
            const ddPct = (s.maxDD * 100).toFixed(1);
            const ddLimitPct = (s.maxDDLimit * 100).toFixed(0);
            const isSelected = s.key === propAccount;
            return (
              <div key={s.key} onClick={() => setPropAccount(s.key)} style={{
                display: "flex", alignItems: "center", justifyContent: "space-between",
                padding: "4px 6px", marginBottom: 2, cursor: "pointer",
                background: isSelected ? "#14142a" : "transparent",
                border: isSelected ? "1px solid #f5c842" : "1px solid transparent",
                borderRadius: 2,
              }}>
                <div style={{ flex: 2 }}>
                  <div style={{ fontSize: 7, color: isSelected ? "#f5c842" : "#cdd6f4" }}>{s.name}</div>
                  <div style={{ fontSize: 6, color: "#444466" }}>{"\u00A3"}{s.price} | DD: {ddPct}%/{ddLimitPct}%</div>
                </div>
                <div style={{ flex: 1, textAlign: "right" }}>
                  <div style={{ fontSize: 8, fontWeight: 700, color: col, fontFamily: "monospace" }}>{st}</div>
                  <div style={{ fontSize: 6, color: pnl >= 0 ? "#26a69a" : "#ef5350", fontFamily: "monospace" }}>
                    {pnl >= 0 ? "+" : ""}{pnl.toFixed(0)}
                    {s.passed && ` T#${s.passedAt + 1}`}
                    {s.blown && ` T#${s.blownAt + 1}`}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </>
    );
  };

  const renderPnlTab = () => {
    const trades = (pnlHistory || []).filter(t => t.pnl_rr != null);
    const bal = parseFloat(pnlBalance) || 0;
    const riskPct = parseFloat(pnlRiskPct) || 0;
    const spread = parseFloat(pnlSpread) || 0;
    const riskAmt = bal * (riskPct / 100);

    // For each trade, compute net RR using cost_rr from backend if available,
    // otherwise estimate from spread / SL distance
    const tradesWithCosts = trades.map(t => {
      const costRR = t.cost_rr != null ? t.cost_rr
        : (t.entry_price && t.resolved_price && t.pnl_rr != null)
          ? spread / Math.max(Math.abs(t.entry_price - (t.resolved_price || t.entry_price)), 1)
          : spread / 15;  // fallback: typical 15-point SL
      const grossRR = t.gross_rr != null ? t.gross_rr : t.pnl_rr + costRR;
      const netRR = t.pnl_rr;  // Backend already deducts cost for new trades
      return { ...t, costRR, grossRR, netRR };
    });

    // Gross totals (before costs)
    const totalGrossRR = tradesWithCosts.reduce((s, t) => s + t.grossRR, 0);
    const totalGrossPnl = totalGrossRR * riskAmt;

    // Net totals (after costs) — this is the realistic P&L
    const totalRR = tradesWithCosts.reduce((s, t) => s + t.netRR, 0);
    const totalPnl = totalRR * riskAmt;
    const totalCosts = (totalGrossRR - totalRR) * riskAmt;

    const wins = tradesWithCosts.filter(t => t.netRR > 0);
    const losses = tradesWithCosts.filter(t => t.netRR <= 0);
    const winRate = trades.length > 0 ? wins.length / trades.length : 0;
    const grossWin = wins.reduce((s, t) => s + t.netRR * riskAmt, 0);
    const grossLoss = Math.abs(losses.reduce((s, t) => s + t.netRR * riskAmt, 0));
    const profitFactor = grossLoss > 0 ? grossWin / grossLoss : grossWin > 0 ? Infinity : 0;
    const bestTrade = trades.length > 0 ? Math.max(...trades.map(t => t.pnl_rr)) : 0;
    const worstTrade = trades.length > 0 ? Math.min(...trades.map(t => t.pnl_rr)) : 0;

    // Equity curve + max drawdown (uses net RR)
    let equity = bal;
    let peak = 0;
    let maxDD = 0;
    const curve = [{ x: 0, y: equity }];
    tradesWithCosts.forEach((t, i) => {
      equity += t.netRR * riskAmt;
      curve.push({ x: i + 1, y: equity });
      const cumPnl = equity - bal;
      if (cumPnl > peak) peak = cumPnl;
      const dd = peak - cumPnl;
      if (dd > maxDD) maxDD = dd;
    });

    const pnlCol = totalPnl >= 0 ? "#26a69a" : "#ef5350";
    const chartW = 260;
    const chartH = 120;
    const pad = { t: 10, r: 10, b: 20, l: 45 };
    const innerW = chartW - pad.l - pad.r;
    const innerH = chartH - pad.t - pad.b;

    const xMin = 0;
    const xMax = Math.max(curve.length - 1, 1);
    const yMin = Math.min(...curve.map(p => p.y), bal);
    const yMax = Math.max(...curve.map(p => p.y), bal);
    const yRange = yMax - yMin || 1;
    const sx = (v) => pad.l + ((v - xMin) / (xMax - xMin)) * innerW;
    const sy = (v) => pad.t + innerH - ((v - yMin) / yRange) * innerH;

    const linePath = curve.map((p, i) => `${i === 0 ? "M" : "L"}${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join(" ");
    const baseY = sy(bal);

    const statBox = { display: "flex", justifyContent: "space-between", fontSize: 7, padding: "3px 0", borderBottom: "1px solid #14142a" };
    const statLabel = { color: "#444466" };
    const statVal = { color: "#cdd6f4", fontFamily: "monospace" };

    return (
      <>
        {/* Settings */}
        <div style={{ ...sec, marginBottom: 6 }}>
          <div style={secT}>ACCOUNT SETTINGS</div>
          <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
            <div style={{ flex: 1 }}>
              <div style={{ fontSize: 6, color: "#444466", marginBottom: 2 }}>BALANCE ($)</div>
              <input type="text" inputMode="decimal" value={pnlBalance}
                onChange={(e) => { const v = e.target.value; if (v === "" || /^\d*\.?\d*$/.test(v)) setPnlBalance(v === "" ? "" : v); }}
                onBlur={() => { const n = parseFloat(pnlBalance); setPnlBalance(isNaN(n) || n <= 0 ? 10000 : n); }}
                style={{ ...inp, width: "100%", boxSizing: "border-box" }} />
            </div>
            <div style={{ flex: 1 }}>
              <div style={{ fontSize: 6, color: "#444466", marginBottom: 2 }}>RISK %</div>
              <input type="text" inputMode="decimal" value={pnlRiskPct}
                onChange={(e) => { const v = e.target.value; if (v === "" || /^\d*\.?\d*$/.test(v)) setPnlRiskPct(v === "" ? "" : v); }}
                onBlur={() => { const n = parseFloat(pnlRiskPct); setPnlRiskPct(isNaN(n) || n <= 0 ? 1 : Math.min(10, n)); }}
                style={{ ...inp, width: "100%", boxSizing: "border-box" }} />
            </div>
            <div style={{ flex: 0.8 }}>
              <div style={{ fontSize: 6, color: "#444466", marginBottom: 2 }}>SPREAD ($)</div>
              <input type="text" inputMode="decimal" value={pnlSpread}
                onChange={(e) => { const v = e.target.value; if (v === "" || /^\d*\.?\d*$/.test(v)) setPnlSpread(v === "" ? "" : v); }}
                onBlur={() => { const n = parseFloat(pnlSpread); setPnlSpread(isNaN(n) || n < 0 ? 0.50 : n); }}
                style={{ ...inp, width: "100%", boxSizing: "border-box" }} />
            </div>
            <div style={{ flex: 0.8 }}>
              <div style={{ fontSize: 6, color: "#444466", marginBottom: 2 }}>RISK/TRADE</div>
              <div style={{ ...inp, width: "100%", boxSizing: "border-box", background: "#0a0a16" }}>${riskAmt.toFixed(0)}</div>
            </div>
          </div>
        </div>

        {/* Summary */}
        <div style={{ ...sec, marginBottom: 6 }}>
          <div style={secT}>PERFORMANCE (NET OF COSTS)</div>
          <div style={{ textAlign: "center", padding: "8px 0 4px" }}>
            <div style={{ fontSize: 22, fontWeight: 900, color: pnlCol, fontFamily: "monospace" }}>
              {totalPnl >= 0 ? "+" : ""}{totalPnl.toFixed(2)}
            </div>
            <div style={{ fontSize: 7, color: "#444466", letterSpacing: 1 }}>NET P&L (AFTER SPREAD)</div>
          </div>
          <div style={statBox}><span style={statLabel}>Net R</span><span style={{ ...statVal, color: pnlCol }}>{totalRR >= 0 ? "+" : ""}{totalRR.toFixed(1)}R</span></div>
          <div style={statBox}><span style={statLabel}>Gross P&L</span><span style={{ ...statVal, color: totalGrossPnl >= 0 ? "#26a69a" : "#ef5350" }}>{totalGrossPnl >= 0 ? "+" : ""}{totalGrossPnl.toFixed(2)} ({totalGrossRR >= 0 ? "+" : ""}{totalGrossRR.toFixed(1)}R)</span></div>
          <div style={statBox}><span style={statLabel}>Spread Costs</span><span style={{ ...statVal, color: "#ff9800" }}>-${totalCosts.toFixed(2)}</span></div>
          <div style={statBox}><span style={statLabel}>Win Rate</span><span style={statVal}>{(winRate * 100).toFixed(1)}% ({wins.length}W / {losses.length}L)</span></div>
          <div style={statBox}><span style={statLabel}>Trades</span><span style={statVal}>{trades.length}</span></div>
          <div style={statBox}><span style={statLabel}>Profit Factor</span><span style={{ ...statVal, color: profitFactor >= 1 ? "#26a69a" : "#ef5350" }}>{profitFactor === Infinity ? "∞" : profitFactor.toFixed(2)}</span></div>
          <div style={statBox}><span style={statLabel}>Best Trade</span><span style={{ ...statVal, color: "#26a69a" }}>+{(bestTrade * riskAmt).toFixed(2)} ({bestTrade.toFixed(1)}R)</span></div>
          <div style={statBox}><span style={statLabel}>Worst Trade</span><span style={{ ...statVal, color: "#ef5350" }}>{(worstTrade * riskAmt).toFixed(2)} ({worstTrade.toFixed(1)}R)</span></div>
          <div style={{ ...statBox, borderBottom: "none" }}><span style={statLabel}>Max Drawdown</span><span style={{ ...statVal, color: "#ef5350" }}>-${maxDD.toFixed(2)}</span></div>
        </div>

        {/* Equity Curve */}
        {trades.length > 0 && (
          <div style={{ ...sec, marginBottom: 6 }}>
            <div style={secT}>EQUITY CURVE</div>
            <svg width={chartW} height={chartH} style={{ display: "block", margin: "0 auto" }}>
              {/* Grid lines */}
              {[0, 0.25, 0.5, 0.75, 1].map((pct) => {
                const yv = yMin + pct * yRange;
                return (
                  <g key={pct}>
                    <line x1={pad.l} x2={chartW - pad.r} y1={sy(yv)} y2={sy(yv)} stroke="#14142a" strokeWidth={0.5} />
                    <text x={pad.l - 3} y={sy(yv) + 3} fill="#33334d" fontSize={6} textAnchor="end" fontFamily="monospace">{yv >= 1000 ? `${(yv / 1000).toFixed(1)}k` : yv.toFixed(0)}</text>
                  </g>
                );
              })}
              {/* Starting balance reference */}
              <line x1={pad.l} x2={chartW - pad.r} y1={baseY} y2={baseY} stroke="#444466" strokeWidth={0.5} strokeDasharray="3,3" />
              {/* Equity line */}
              <path d={linePath} fill="none" stroke={pnlCol} strokeWidth={1.5} />
              {/* Endpoint dot */}
              <circle cx={sx(curve[curve.length - 1].x)} cy={sy(curve[curve.length - 1].y)} r={2.5} fill={pnlCol} />
              {/* X axis label */}
              <text x={chartW / 2} y={chartH - 2} fill="#33334d" fontSize={6} textAnchor="middle" fontFamily="monospace">{trades.length} TRADES</text>
            </svg>
          </div>
        )}

        {/* Trade History */}
        <div style={{ ...sec }}>
          <div style={secT}>TRADE HISTORY ({trades.length})</div>
          <div style={{ maxHeight: 200, overflowY: "auto" }}>
            {[...trades].reverse().map((t) => {
              const isWin = t.pnl_rr > 0;
              const dollarPnl = t.pnl_rr * riskAmt;
              return (
                <div key={t.id} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "3px 0", borderBottom: "1px solid #0e0e1a", fontSize: 7 }}>
                  <div style={{ display: "flex", gap: 4, alignItems: "center" }}>
                    <span style={{ color: t.direction === "long" ? "#26a69a" : "#ef5350", fontSize: 8 }}>
                      {t.direction === "long" ? "▲" : "▼"}
                    </span>
                    <span style={{ fontSize: 6, padding: "1px 3px", background: "rgba(245,200,66,0.15)", color: "#f5c842" }}>{(t.timeframe || "").toUpperCase()}</span>
                    <span style={{ color: "#444466" }}>{t.resolved_at?.replace("T", " ").slice(5, 16) || "—"}</span>
                  </div>
                  <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
                    <span style={{ color: isWin ? "#26a69a" : "#ef5350", fontFamily: "monospace", fontWeight: 700 }}>
                      {t.pnl_rr > 0 ? "+" : ""}{t.pnl_rr.toFixed(1)}R
                    </span>
                    <span style={{ color: isWin ? "#26a69a" : "#ef5350", fontFamily: "monospace", minWidth: 55, textAlign: "right" }}>
                      {dollarPnl >= 0 ? "+" : ""}${dollarPnl.toFixed(0)}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
          {trades.length === 0 && <div style={{ color: "#33334d", fontSize: 8, textAlign: "center", padding: 12 }}>No resolved trades yet</div>}
        </div>
      </>
    );
  };

  const renderLogTab = () => {
    const outcomes = [
      { key: "tp1", label: "TP1", color: "#26a69a" },
      { key: "tp2", label: "TP2", color: "#26a69a" },
      { key: "tp3", label: "TP3", color: "#26a69a" },
      { key: "stopped_out", label: "STOPPED", color: "#ef5350" },
      { key: "breakeven", label: "BE", color: "#ffa726" },
      { key: "manual_close", label: "MANUAL", color: "#666" },
    ];

    return (
      <>
        {/* Logging form */}
        {analysis?.entry && calibration && (
          <div style={{ ...sec, marginBottom: 10 }}>
            <div style={secT}>AD-HOC TRADE LOG</div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 3, marginBottom: 6 }}>
              {outcomes.map((o) => (
                <button key={o.key} onClick={() => setJournalOutcome(o.key)} style={{
                  padding: "3px 6px", fontSize: 7, fontFamily: "monospace", cursor: "pointer",
                  border: `1px solid ${o.color}`,
                  background: journalOutcome === o.key ? o.color : "transparent",
                  color: journalOutcome === o.key ? "#08080f" : o.color,
                }}>{o.label}</button>
              ))}
            </div>
            <div style={{ display: "flex", gap: 4, marginBottom: 6, fontSize: 7 }}>
              <span style={{ color: "#444466" }}>Used Cal SL?</span>
              <button onClick={() => setJournalUsedCalSL(true)} style={{
                padding: "2px 6px", fontSize: 7, fontFamily: "monospace", cursor: "pointer",
                border: "1px solid #26a69a", background: journalUsedCalSL ? "#26a69a" : "transparent",
                color: journalUsedCalSL ? "#08080f" : "#26a69a",
              }}>YES</button>
              <button onClick={() => setJournalUsedCalSL(false)} style={{
                padding: "2px 6px", fontSize: 7, fontFamily: "monospace", cursor: "pointer",
                border: "1px solid #33334d", background: !journalUsedCalSL ? "#33334d" : "transparent",
                color: !journalUsedCalSL ? "#cdd6f4" : "#33334d",
              }}>NO</button>
            </div>
            <textarea value={journalNote} onChange={(e) => setJournalNote(e.target.value)}
              placeholder="Notes..." rows={2} style={{ ...inp, width: "100%", boxSizing: "border-box", resize: "none", marginBottom: 6 }} />
            <button onClick={logTrade} disabled={!journalOutcome}
              style={{ ...btn(!!journalOutcome, "#f5c842"), width: "100%", padding: "6px", fontSize: 8 }}>+ LOG TRADE</button>
          </div>
        )}

        {/* Scanner setups from backend */}
        {scannerSetups.length > 0 && (
          <div style={{ ...sec, marginBottom: 6, borderLeft: "3px solid #7c4dff", paddingLeft: 8 }}>
            <div style={secT}>SCANNER ({scannerSetups.length} pending)</div>
            {scannerSetups.map((s) => (
              <div key={s.id} style={{ marginBottom: 6, padding: "4px 0", borderBottom: "1px solid #1a1a2e" }}>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: 7 }}>
                  <span style={{ color: "#444466" }}>{s.created_at?.replace("T", " ").slice(0, 16)}</span>
                  <div style={{ display: "flex", gap: 3 }}>
                    {s.timeframe && <span style={{ fontSize: 6, padding: "1px 4px", background: "rgba(245,200,66,0.15)", color: "#f5c842" }}>{s.timeframe.toUpperCase()}</span>}
                    <span style={{ fontSize: 6, padding: "1px 4px", background: "rgba(124,77,255,0.2)", color: "#7c4dff" }}>SCAN</span>
                  </div>
                </div>
                <div style={{ display: "flex", gap: 4, alignItems: "center", marginTop: 2, flexWrap: "wrap" }}>
                  <span style={{ color: s.direction === "long" ? "#26a69a" : "#ef5350", fontSize: 9 }}>
                    {s.direction === "long" ? "▲" : "▼"}
                  </span>
                  <span style={{ color: "#cdd6f4", fontSize: 8 }}>{Number(s.entry_price || 0).toFixed(2)}</span>
                  <span style={{ fontSize: 6, padding: "1px 4px", background: "rgba(255,167,38,0.2)", color: "#ffa726" }}>PENDING</span>
                  {s.setup_quality && (
                    <span style={{ fontSize: 6, color: s.setup_quality === "A" ? "#26a69a" : "#f5c842" }}>{s.setup_quality}</span>
                  )}
                  {(() => {
                    const cal = s.calibration_json || {};
                    const conf = cal.confidence || {};
                    const grade = conf.grade;
                    const winProb = conf.autogluon_win_prob;
                    const slSrc = cal.calibrated?.sl_source;
                    const gradeColors = { A: "#26a69a", B: "#f5c842", C: "#ffa726", D: "#ef5350", F: "#666" };
                    return (
                      <>
                        {grade && <span style={{ fontSize: 6, padding: "1px 4px", borderRadius: 2, background: `${gradeColors[grade] || "#444"}22`, color: gradeColors[grade] || "#444", fontWeight: 700 }}>{grade}</span>}
                        {winProb != null && <span style={{ fontSize: 6, color: winProb > 0.6 ? "#26a69a" : "#ffa726" }}>ML:{(winProb * 100).toFixed(0)}%</span>}
                        {slSrc && slSrc !== "claude" && <span style={{ fontSize: 5.5, color: "#7c4dff" }}>SL:{slSrc}</span>}
                      </>
                    );
                  })()}
                </div>
                <div style={{ fontSize: 7, color: "#ffa726", marginTop: 2 }}>
                  SL: {(s.calibrated_sl || s.sl_price) ? Number(s.calibrated_sl || s.sl_price).toFixed(2) : "—"} · TPs: {[s.tp1, s.tp2, s.tp3].filter(Boolean).map((t) => Number(t).toFixed(2)).join(" / ") || "—"}
                </div>
                {/* Lifecycle stage indicator */}
                {(() => {
                  const stgEmoji = { 1: "\u{1F4AD}", 2: "\u{1F52C}", 3: "\u{1F3AF}", 4: "\u2705", 5: "\u{1F4CA}", 6: "\u26A0\uFE0F" };
                  const stgShort = { 1: "FORMING", 2: "CONFIRMED", 3: "DETECTED", 4: "ENTRY", 5: "RESOLVED", 6: "REVISED" };
                  const lcEvts = (lifecycleRecent?.events || []).filter(e => e.setup_id === s.id || (s.analysis_json?.narrative_state?.id && e.thesis_id === s.analysis_json.narrative_state.id));
                  if (lcEvts.length === 0) return null;
                  const maxStg = Math.max(...lcEvts.map(e => e.stage));
                  const doneStages = new Set(lcEvts.map(e => e.stage));
                  return (
                    <div style={{ display: "flex", gap: 2, alignItems: "center", marginTop: 2 }}>
                      {[1, 2, 3, 4, 5].map(stg => (
                        <span key={stg} style={{ fontSize: 6, opacity: doneStages.has(stg) ? 1 : 0.2 }}>{stgEmoji[stg]}</span>
                      ))}
                      <span style={{ fontSize: 5.5, color: "#444466", marginLeft: 2 }}>{stgShort[maxStg]}</span>
                    </div>
                  );
                })()}
              </div>
            ))}
          </div>
        )}

        {/* Resolved scanner setups */}
        {scannerHistory?.length > 0 && (
          <div style={{ ...sec, marginBottom: 6, borderLeft: "3px solid #444466", paddingLeft: 8 }}>
            <div style={secT}>RESOLVED ({scannerHistory.length})</div>
            {scannerHistory.map((s) => {
              const isWin = s.outcome?.startsWith("tp");
              const isSL = s.outcome === "sl_hit";
              const isExpired = s.outcome === "expired";
              const outcomeCol = isWin ? "#26a69a" : isSL ? "#ef5350" : isExpired ? "#666" : "#ffa726";
              return (
                <div key={s.id} style={{ marginBottom: 4, padding: "3px 0", borderBottom: "1px solid #1a1a2e" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", fontSize: 7 }}>
                    <span style={{ color: "#444466" }}>{s.created_at?.replace("T", " ").slice(0, 16)}</span>
                    <div style={{ display: "flex", gap: 3 }}>
                      {s.timeframe && <span style={{ fontSize: 6, padding: "1px 4px", background: "rgba(245,200,66,0.15)", color: "#f5c842" }}>{s.timeframe.toUpperCase()}</span>}
                      <span style={{ fontSize: 6, padding: "1px 4px", background: "rgba(124,77,255,0.2)", color: "#7c4dff" }}>SCAN</span>
                    </div>
                  </div>
                  <div style={{ display: "flex", gap: 4, alignItems: "center", marginTop: 2, flexWrap: "wrap" }}>
                    <span style={{ color: s.direction === "long" ? "#26a69a" : "#ef5350", fontSize: 9 }}>
                      {s.direction === "long" ? "▲" : "▼"}
                    </span>
                    <span style={{ color: "#cdd6f4", fontSize: 8 }}>{Number(s.entry_price || 0).toFixed(2)}</span>
                    <span style={{ fontSize: 6, padding: "1px 4px", background: `${outcomeCol}22`, color: outcomeCol }}>{(s.outcome || "—").toUpperCase()}</span>
                    {s.pnl_rr != null && s.pnl_rr !== 0 && (
                      <span style={{ color: Number(s.pnl_rr) > 0 ? "#26a69a" : "#ef5350", fontSize: 8, fontWeight: 700 }}>{Number(s.pnl_rr) > 0 ? "+" : ""}{Number(s.pnl_rr).toFixed(1)}R</span>
                    )}
                    {(() => {
                      const cal = s.calibration_json || {};
                      const conf = cal.confidence || {};
                      const grade = conf.grade;
                      const winProb = conf.autogluon_win_prob;
                      const gradeColors = { A: "#26a69a", B: "#f5c842", C: "#ffa726", D: "#ef5350", F: "#666" };
                      // Check if ML prediction was correct
                      const predicted = winProb != null ? (winProb > 0.5 ? "win" : "loss") : null;
                      const actual = isWin ? "win" : "loss";
                      const correct = predicted ? (predicted === actual) : null;
                      return (
                        <>
                          {grade && <span style={{ fontSize: 6, padding: "1px 4px", borderRadius: 2, background: `${gradeColors[grade] || "#444"}22`, color: gradeColors[grade] || "#444", fontWeight: 700 }}>{grade}</span>}
                          {winProb != null && <span style={{ fontSize: 6, color: winProb > 0.6 ? "#26a69a" : "#ffa726" }}>ML:{(winProb * 100).toFixed(0)}%</span>}
                          {correct != null && <span style={{ fontSize: 6 }}>{correct ? "✅" : "❌"}</span>}
                        </>
                      );
                    })()}
                  </div>
                  {s.resolved_at && <div style={{ fontSize: 6, color: "#33334d", marginTop: 1 }}>Resolved: {s.resolved_at?.replace("T", " ").slice(0, 16)}</div>}
                  {(() => {
                    const ns = s.analysis_json?.narrative_state;
                    if (!ns) return null;
                    return (
                      <div style={{ fontSize: 6, color: "#444466", marginTop: 1 }}>
                        Thesis: <span style={{ color: ns.directional_bias === "bullish" ? "#26a69a" : "#ef5350" }}>{(ns.directional_bias || "").toUpperCase()}</span> {"\u00B7"} {ns.p3_phase || ""} {"\u00B7"} {ns.scan_count || 1} scans
                      </div>
                    );
                  })()}
                </div>
              );
            })}
          </div>
        )}

        {/* Scanner status indicator */}
        {scannerStatus?.scheduler_running && (
          <div style={{ fontSize: 7, color: "#444466", textAlign: "center", marginBottom: 6 }}>
            Scanner active · {scannerStatus.total_scans || 0} scans · Next: {scannerStatus.next_scan ? new Date(scannerStatus.next_scan).toLocaleTimeString() : "—"}
          </div>
        )}

        {/* Journal entries */}
        {journal.map((entry) => {
          const isPending = entry.outcome === "pending";
          const isWin = entry.outcome?.startsWith("tp");
          const isStopped = entry.outcome === "stopped_out";
          const borderCol = isPending ? "#ffa726" : isWin ? "#26a69a" : isStopped ? "#ef5350" : "#ffa726";
          const bgCol = isPending ? "rgba(255,167,38,0.05)" : "transparent";
          return (
            <div key={entry.id} style={{ ...sec, borderLeft: `3px solid ${borderCol}`, paddingLeft: 8, marginBottom: 4, background: bgCol }}>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: 7, marginBottom: 2 }}>
                <span style={{ color: "#444466" }}>{entry.ts}</span>
                <span style={{ color: SESSION_COLORS[entry.session] || "#444", fontSize: 7 }}>{entry.session?.toUpperCase()}</span>
              </div>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <div style={{ display: "flex", gap: 4, alignItems: "center" }}>
                  <span style={{ color: entry.direction === "long" ? "#26a69a" : "#ef5350", fontSize: 9 }}>
                    {entry.direction === "long" ? "▲" : "▼"}
                  </span>
                  <span style={{
                    padding: "1px 5px", fontSize: 7, fontFamily: "monospace",
                    background: isPending ? "rgba(255,167,38,0.2)" : isWin ? "rgba(38,166,154,0.2)" : isStopped ? "rgba(239,83,80,0.2)" : "rgba(255,167,38,0.2)",
                    color: borderCol,
                  }}>{entry.outcome?.toUpperCase() || "PENDING"}</span>
                  {entry.auto_resolved && (
                    <span style={{ fontSize: 6, color: "#f5c842", marginLeft: 2 }}>⚡ AUTO</span>
                  )}
                </div>
                {entry.rr != null && entry.rr !== 0 && (
                  <span style={{ color: entry.rr > 0 ? "#26a69a" : "#ef5350", fontSize: 8, fontWeight: 700 }}>{entry.rr > 0 ? "+" : ""}{entry.rr.toFixed(1)}R</span>
                )}
              </div>
              {isPending && entry.entry && (
                <div style={{ fontSize: 7, color: "#ffa726", marginTop: 2 }}>
                  Entry: {entry.entry} · SL: {entry.sl_used || "—"} · TPs: {(entry.tps || []).join(" / ") || "—"}
                </div>
              )}
              <div style={{ fontSize: 7, color: "#33334d", marginTop: 2 }}>{entry.setup_type}</div>
              {entry.saved_by_calibration && (
                <div style={{ fontSize: 7, color: "#f5c842", marginTop: 2 }}>✓ Calibration saved this trade</div>
              )}
              {entry.note && <div style={{ fontSize: 7, color: "#444466", fontStyle: "italic", marginTop: 2 }}>{entry.note}</div>}
              {entry.resolved_at && (
                <div style={{ fontSize: 6, color: "#33334d", marginTop: 2 }}>Resolved: {entry.resolved_at}</div>
              )}
              {/* Inline resolution buttons for pending entries */}
              {isPending && (
                <div style={{ display: "flex", flexWrap: "wrap", gap: 2, marginTop: 4 }}>
                  {outcomes.map((o) => (
                    <button key={o.key} onClick={() => resolveJournalEntry(entry.id, o.key)} style={{
                      padding: "2px 5px", fontSize: 6, fontFamily: "monospace", cursor: "pointer",
                      border: `1px solid ${o.color}`, background: "transparent", color: o.color,
                    }}>{o.label}</button>
                  ))}
                </div>
              )}
            </div>
          );
        })}
        {journal.length === 0 && scannerSetups.length === 0 && (!scannerHistory || scannerHistory.length === 0) && <div style={{ color: "#33334d", fontSize: 8, textAlign: "center", marginTop: 12 }}>No trades logged yet</div>}
      </>
    );
  };

  // ── Narrative Thesis Dashboard Tab ──
  const renderThesisTab = () => {
    const STAGES = { 1: "THESIS_FORMING", 2: "THESIS_CONFIRMED", 3: "SETUP_DETECTED", 4: "ENTRY_READY", 5: "TRADE_RESOLVED", 6: "THESIS_REVISED" };
    const EMOJIS = { 1: "\u{1F4AD}", 2: "\u{1F52C}", 3: "\u{1F3AF}", 4: "\u2705", 5: "\u{1F4CA}", 6: "\u26A0\uFE0F" };
    const pct = (v) => v != null ? `${(v * 100).toFixed(1)}%` : "\u2014";
    const row = (label, value, valColor = "#cdd6f4") => (
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 7, padding: "2px 0" }}>
        <span style={{ color: "#444466" }}>{label}</span>
        <span style={{ color: valColor }}>{value}</span>
      </div>
    );
    const bar = (value, max, color) => (
      <div style={{ height: 4, background: "#14142a", flex: 1, marginLeft: 6, borderRadius: 2 }}>
        <div style={{ height: "100%", width: `${Math.min((value / max) * 100, 100)}%`, background: color, borderRadius: 2 }} />
      </div>
    );
    const pill = (label, color) => (
      <span style={{ fontSize: 6, padding: "1px 4px", background: `${color}22`, color, fontWeight: 700 }}>{label}</span>
    );

    const thesis = currentThesis?.thesis;
    const noData = !thesis && !thesisAccuracy?.predictions_scored && (!recentContext?.recent_resolutions?.length);

    if (noData) return <div style={{ color: "#33334d", fontSize: 8, textAlign: "center", marginTop: 20 }}>No thesis data. Scanner must run to populate.</div>;

    const biasColor = thesis?.directional_bias === "bullish" ? "#26a69a" : thesis?.directional_bias === "bearish" ? "#ef5350" : "#f5c842";
    const confidence = thesis?.bias_confidence || 0;
    const p3Progress = thesis?.p3_progress ? parseFloat(thesis.p3_progress) / 100 : 0;

    // Find lifecycle events for current thesis
    const thesisId = thesis?.id;
    const lcEvents = thesisId ? (lifecycleRecent?.events || []).filter(e => e.thesis_id === thesisId) : [];
    const completedStages = new Set(lcEvents.map(e => e.stage));
    const maxStage = completedStages.size > 0 ? Math.max(...completedStages) : 0;
    const stageTimestamps = {};
    lcEvents.forEach(e => { stageTimestamps[e.stage] = e.sent_at; });

    // Recent context helpers
    const rc = recentContext || {};
    const resolutions = rc.recent_resolutions || [];
    const consumed = rc.consumed_zones || [];
    const swept = rc.swept_liquidity || [];
    const activeCtx = rc.active_setups || [];

    const timeAgo = (iso) => {
      if (!iso) return "";
      const diff = (Date.now() - new Date(iso).getTime()) / 60000;
      if (diff < 60) return `${Math.round(diff)}m ago`;
      if (diff < 1440) return `${Math.round(diff / 60)}h ago`;
      return `${Math.round(diff / 1440)}d ago`;
    };

    return (
      <>
        {/* Section 1: Current Thesis */}
        {thesis && (
          <div style={{ ...sec, borderLeft: `3px solid ${biasColor}`, paddingLeft: 8, marginBottom: 6 }}>
            <div style={secT}>CURRENT THESIS</div>
            <div style={{ display: "flex", gap: 4, alignItems: "center", marginBottom: 4 }}>
              <span style={{ color: biasColor, fontSize: 14, fontWeight: 900 }}>
                {thesis.directional_bias === "bullish" ? "\u25B2" : thesis.directional_bias === "bearish" ? "\u25BC" : "\u25C6"}
              </span>
              <span style={{ fontSize: 9, color: biasColor, fontWeight: 700 }}>
                {(thesis.directional_bias || "NEUTRAL").toUpperCase()}
              </span>
              <span style={{ fontSize: 7, color: "#ffa726" }}>{(confidence * 100).toFixed(0)}%</span>
              {bar(confidence, 1, biasColor)}
            </div>

            {thesis.p3_phase && (
              <div style={{ display: "flex", gap: 4, alignItems: "center", marginBottom: 4 }}>
                <span style={{ fontSize: 6, padding: "1px 4px", background: "rgba(124,77,255,0.15)", color: "#7c4dff" }}>
                  {thesis.p3_phase.toUpperCase()}
                </span>
                {thesis.p3_progress && (
                  <>
                    <span style={{ fontSize: 6, color: "#444466" }}>{thesis.p3_progress}</span>
                    {bar(p3Progress, 1, "#7c4dff")}
                  </>
                )}
              </div>
            )}

            <div style={{ display: "flex", gap: 6, fontSize: 6, color: "#444466", marginBottom: 4 }}>
              <span>{thesis.scan_count || 1} scans</span>
              <span>{thesis.thesis_age_minutes || 0}min old</span>
              {thesis.is_revision ? pill("REVISED", "#ffa726") : null}
            </div>

            {thesis.thesis && (
              <div style={{ fontSize: 7, color: "#6e7a9a", fontStyle: "italic", lineHeight: 1.5, marginBottom: 4 }}>
                {thesis.thesis.length > 200 ? thesis.thesis.slice(0, 200) + "\u2026" : thesis.thesis}
              </div>
            )}

            {thesis.expected_next_move && (
              <div style={{ fontSize: 7, color: "#cdd6f4", marginBottom: 3 }}>
                <span style={{ color: "#444466" }}>Next: </span>{thesis.expected_next_move}
              </div>
            )}

            {thesis.invalidation?.price_level && (
              <div style={{ fontSize: 7, color: "#ef5350", marginBottom: 3 }}>
                Invalidation: {Number(thesis.invalidation.price_level).toFixed(2)} ({(thesis.invalidation.direction || "").toUpperCase()})
              </div>
            )}

            {thesis.watching_for?.length > 0 && (
              <div style={{ marginBottom: 3 }}>
                <span style={{ fontSize: 6, color: "#444466" }}>Watching:</span>
                {thesis.watching_for.map((w, i) => (
                  <div key={i} style={{ fontSize: 6, color: "#6e7a9a", paddingLeft: 8 }}>{"\u2022"} {w}</div>
                ))}
              </div>
            )}

            {thesis.key_levels?.length > 0 && (
              <div style={{ fontSize: 7, color: "#444466" }}>
                Levels: <span style={{ color: "#f5c842" }}>{thesis.key_levels.map(l => Number(l).toFixed(2)).join(" \u00B7 ")}</span>
              </div>
            )}
          </div>
        )}

        {/* Section 2: Lifecycle Journey */}
        {(lcEvents.length > 0 || thesis) && (
          <div style={{ ...sec, marginBottom: 6 }}>
            <div style={secT}>LIFECYCLE JOURNEY</div>
            {[1, 2, 3, 4, 5, 6].map(stg => {
              const done = completedStages.has(stg);
              const isCurrent = stg === maxStage && maxStage > 0;
              const ts = stageTimestamps[stg];
              return (
                <div key={stg} style={{ display: "flex", alignItems: "center", gap: 6, padding: "2px 0", fontSize: 7 }}>
                  <span style={{ fontSize: 9, opacity: done ? 1 : 0.25 }}>{EMOJIS[stg]}</span>
                  <span style={{
                    color: isCurrent ? "#f5c842" : done ? "#cdd6f4" : "#33334d",
                    fontWeight: isCurrent ? 700 : 400, flex: 1,
                  }}>
                    {isCurrent ? "\u25B8 " : ""}{STAGES[stg]}
                  </span>
                  <span style={{ color: "#444466", fontSize: 6 }}>
                    {ts ? new Date(ts).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) : "\u2014"}
                  </span>
                </div>
              );
            })}
          </div>
        )}

        {/* Section 3: Recent Context */}
        {(resolutions.length > 0 || consumed.length > 0 || swept.length > 0 || activeCtx.length > 0) && (
          <div style={{ ...sec, marginBottom: 6 }}>
            <div style={secT}>RECENT CONTEXT</div>

            {resolutions.length > 0 && (
              <div style={{ marginBottom: 6 }}>
                <div style={{ fontSize: 6, color: "#444466", letterSpacing: 1, marginBottom: 3 }}>LAST RESOLUTIONS</div>
                {resolutions.map((r, i) => {
                  const isWin = r.outcome?.startsWith("tp");
                  const outcomeCol = isWin ? "#26a69a" : r.outcome === "stopped_out" ? "#ef5350" : "#666";
                  return (
                    <div key={i} style={{ marginBottom: 4, padding: "2px 0", borderBottom: "1px solid #1a1a2e" }}>
                      <div style={{ display: "flex", gap: 4, alignItems: "center", fontSize: 7 }}>
                        <span style={{ color: r.direction === "long" ? "#26a69a" : "#ef5350", fontSize: 9 }}>
                          {r.direction === "long" ? "\u25B2" : "\u25BC"}
                        </span>
                        {pill((r.outcome || "").toUpperCase(), outcomeCol)}
                        {r.pnl_rr != null && (
                          <span style={{ color: Number(r.pnl_rr) > 0 ? "#26a69a" : "#ef5350", fontWeight: 700 }}>
                            {Number(r.pnl_rr) > 0 ? "+" : ""}{Number(r.pnl_rr).toFixed(1)}R
                          </span>
                        )}
                        {r.setup_quality && pill(r.setup_quality, r.setup_quality === "A" ? "#26a69a" : "#f5c842")}
                        <span style={{ color: "#444466", fontSize: 6 }}>{timeAgo(r.resolved_at)}</span>
                      </div>
                      <div style={{ fontSize: 6, color: "#444466", marginTop: 1 }}>
                        Entry: {Number(r.entry_price || 0).toFixed(2)} \u00B7 SL: {Number(r.sl_price || 0).toFixed(2)}
                        {r.killzone && <span> \u00B7 {r.killzone}</span>}
                      </div>
                    </div>
                  );
                })}
              </div>
            )}

            {consumed.length > 0 && (
              <div style={{ marginBottom: 4 }}>
                <div style={{ fontSize: 6, color: "#444466", letterSpacing: 1, marginBottom: 2 }}>CONSUMED ZONES</div>
                {consumed.map((z, i) => (
                  <div key={i} style={{ fontSize: 6, color: "#6e7a9a", padding: "1px 0" }}>
                    {(z.zone_type || "zone").toUpperCase()} {Number(z.low || 0).toFixed(2)}\u2013{Number(z.high || 0).toFixed(2)}
                    <span style={{ color: "#444466" }}> ({z.outcome || "used"})</span>
                  </div>
                ))}
              </div>
            )}

            {swept.length > 0 && (
              <div style={{ marginBottom: 4 }}>
                <div style={{ fontSize: 6, color: "#444466", letterSpacing: 1, marginBottom: 2 }}>SWEPT LIQUIDITY</div>
                {swept.map((s, i) => (
                  <div key={i} style={{ fontSize: 7, color: s.type === "SSL" ? "#ef5350" : "#64b5f6" }}>
                    {s.type || "LIQ"} @ {Number(s.level || 0).toFixed(2)}
                    <span style={{ fontSize: 6, color: "#444466" }}> {timeAgo(s.swept_at)}</span>
                  </div>
                ))}
              </div>
            )}

            {activeCtx.length > 0 && (
              <div>
                <div style={{ fontSize: 6, color: "#444466", letterSpacing: 1, marginBottom: 2 }}>ACTIVE SETUPS</div>
                {activeCtx.map((a, i) => (
                  <div key={i} style={{ display: "flex", gap: 4, alignItems: "center", fontSize: 7, padding: "1px 0" }}>
                    <span style={{ color: a.direction === "long" ? "#26a69a" : "#ef5350", fontSize: 9 }}>
                      {a.direction === "long" ? "\u25B2" : "\u25BC"}
                    </span>
                    <span style={{ color: "#cdd6f4" }}>{Number(a.entry_price || 0).toFixed(2)}</span>
                    {a.setup_quality && pill(a.setup_quality, a.setup_quality === "A" ? "#26a69a" : "#f5c842")}
                    {a.killzone && <span style={{ fontSize: 6, color: "#444466" }}>{a.killzone}</span>}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Section 4: Thesis Accuracy */}
        {thesisAccuracy?.predictions_scored > 0 && (
          <div style={{ ...sec, marginBottom: 6 }}>
            <div style={secT}>THESIS ACCURACY</div>
            <div style={{ display: "flex", alignItems: "center", gap: 4, marginBottom: 3 }}>
              <span style={{ fontSize: 7, color: "#444466" }}>Prediction Accuracy</span>
              <span style={{
                fontSize: 8, fontWeight: 700,
                color: thesisAccuracy.prediction_accuracy >= 0.6 ? "#26a69a" : thesisAccuracy.prediction_accuracy >= 0.45 ? "#ffa726" : "#ef5350",
              }}>
                {pct(thesisAccuracy.prediction_accuracy)}
              </span>
              {bar(thesisAccuracy.prediction_accuracy || 0, 1, thesisAccuracy.prediction_accuracy >= 0.6 ? "#26a69a" : "#ffa726")}
            </div>
            {row("Predictions Scored", thesisAccuracy.predictions_scored || 0)}
            <div style={{ display: "flex", alignItems: "center", gap: 4, marginBottom: 3 }}>
              <span style={{ fontSize: 7, color: "#444466" }}>Thesis Stability</span>
              <span style={{
                fontSize: 8, fontWeight: 700,
                color: thesisAccuracy.thesis_stability >= 0.7 ? "#26a69a" : thesisAccuracy.thesis_stability >= 0.5 ? "#ffa726" : "#ef5350",
              }}>
                {pct(thesisAccuracy.thesis_stability)}
              </span>
              {bar(thesisAccuracy.thesis_stability || 0, 1, thesisAccuracy.thesis_stability >= 0.7 ? "#26a69a" : "#ffa726")}
            </div>
            {row("Revision Rate", pct(thesisAccuracy.revision_rate))}
            {row("Avg Thesis Age", `${Math.round(thesisAccuracy.avg_thesis_age_minutes || 0)}min`)}
            {row("Total States", thesisAccuracy.total_states || 0)}
            {row("Revisions", thesisAccuracy.revisions || 0, "#ffa726")}
            {row("Continuations", thesisAccuracy.continuations || 0, "#26a69a")}
          </div>
        )}

        {/* Section 5: Thesis History */}
        {thesisHistory?.history?.length > 0 && (
          <div style={{ ...sec, marginBottom: 6 }}>
            <div style={secT}>HISTORY ({thesisHistory.history.length})</div>
            {thesisHistory.history.slice(0, 10).map((h, i) => {
              const bc = h.directional_bias === "bullish" ? "#26a69a" : h.directional_bias === "bearish" ? "#ef5350" : "#f5c842";
              const isActive = h.status === "active";
              return (
                <div key={i} style={{
                  display: "flex", gap: 4, alignItems: "center", fontSize: 7, padding: "2px 0",
                  borderBottom: "1px solid #1a1a2e",
                  borderLeft: isActive ? "2px solid #f5c842" : "2px solid transparent",
                  paddingLeft: isActive ? 4 : 6,
                }}>
                  <span style={{ color: bc, fontSize: 9, fontWeight: 700 }}>
                    {h.directional_bias === "bullish" ? "\u25B2" : h.directional_bias === "bearish" ? "\u25BC" : "\u25C6"}
                  </span>
                  <span style={{ color: "#6e7a9a", fontSize: 6 }}>{(h.p3_phase || "").slice(0, 5).toUpperCase()}</span>
                  <span style={{ color: bc, fontSize: 7 }}>{((h.bias_confidence || 0) * 100).toFixed(0)}%</span>
                  <span style={{ flex: 1 }} />
                  {h.is_revision ? pill("REV", "#ffa726") : null}
                  {isActive ? pill("ACTIVE", "#f5c842") : pill(h.status || "done", "#33334d")}
                  <span style={{ color: "#444466", fontSize: 6 }}>
                    {h.created_at ? new Date(h.created_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) : ""}
                  </span>
                </div>
              );
            })}
          </div>
        )}
      </>
    );
  };

  // ═══════════════════════════════════════════════════════════
  //  SETUP SCREEN
  // ═══════════════════════════════════════════════════════════

  if (screen === "setup") {
    const backendOk = pipelineHealth?.status === "ok";
    const seeded = sessionStats?.seeded;

    return (
      <div style={{ background: "#08080f", minHeight: "100vh", color: "#cdd6f4", fontFamily: "monospace", display: "flex", flexDirection: "column" }}>
        <div style={{ borderBottom: "1px solid #14142a", padding: "12px 20px", background: "#0a0a14" }}>
          <div style={{ color: "#f5c842", fontSize: 16, letterSpacing: 4, fontWeight: 900 }}>◈ ICT BACKTEST TERMINAL</div>
          <div style={{ color: "#33334d", fontSize: 9, letterSpacing: 3, marginTop: 2 }}>XAU/USD · GOLD · CLAUDE-PRIMARY ANALYSIS ENGINE</div>
        </div>
        <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", padding: 32 }}>
          <div style={{ maxWidth: 440, width: "100%" }}>
            <div style={{ ...sec, padding: 28 }}>
              <div style={{ color: "#f5c842", fontSize: 12, letterSpacing: 3, marginBottom: 24 }}>SYSTEM INITIALIZATION</div>

              <div style={{ marginBottom: 18 }}>
                <div style={{ color: "#33334d", fontSize: 9, letterSpacing: 2, marginBottom: 6 }}>▸ ANTHROPIC API KEY</div>
                <input style={{ ...inp, width: "100%", boxSizing: "border-box" }} type="password"
                  placeholder="sk-ant-..." value={claudeKey} onChange={(e) => setClaudeKey(e.target.value)} />
              </div>

              <div style={{ marginBottom: 20 }}>
                <div style={{ color: "#33334d", fontSize: 9, letterSpacing: 2, marginBottom: 6 }}>▸ TIMEFRAME</div>
                <div style={{ display: "flex", gap: 6 }}>
                  {TF_OPTIONS.map((tf) => (
                    <button key={tf.value} style={btn(timeframe === tf.value)} onClick={() => setTimeframe(tf.value)}>{tf.label}</button>
                  ))}
                </div>
              </div>

              {error && <div style={{ color: "#ef5350", fontSize: 9, marginBottom: 12 }}>⚠ {error}</div>}

              <button style={{ ...btn(true), width: "100%", padding: 12, fontSize: 10, letterSpacing: 3 }}
                onClick={fetchData} disabled={loadingData}>
                {loadingData ? "FETCHING XAU/USD DATA..." : "▶  INITIALISE ENGINE"}
              </button>

              <button style={{ ...btn(false, "#9370db"), width: "100%", padding: 10, fontSize: 9, letterSpacing: 2, marginTop: 8 }}
                onClick={launchDemo}>
                ⬡ LAUNCH DEMO MODE
              </button>

              {/* System status */}
              <div style={{ marginTop: 18, padding: 12, background: "#06060c", border: "1px solid #14142a", fontSize: 8.5, lineHeight: 2.2, color: "#33334d" }}>
                <div style={{ color: "#f5c842", fontSize: 8, letterSpacing: 2, marginBottom: 4 }}>SYSTEM STATUS</div>
                <div style={{ color: backendOk ? "#26a69a" : "#ef5350" }}>{backendOk ? "✓" : "✗"} Backend: {backendOk ? "Connected" : "Not reachable"}</div>
                <div style={{ color: bayesian ? "#26a69a" : "#33334d" }}>{bayesian ? "✓" : "○"} Bayesian: {bayesian ? `${bayesian.total_trades} trades · ${(bayesian.win_rate_mean * 100).toFixed(1)}% WR` : "No data"}</div>
                <div style={{ color: sessionStats?.dataset_stats ? "#26a69a" : "#33334d" }}>{sessionStats?.dataset_stats ? "✓" : "○"} Dataset: {sessionStats?.dataset_stats?.total || 0} total · {sessionStats?.dataset_stats?.resolved || 0} resolved</div>
                {!backendOk && <div style={{ color: "#2a2a44", marginTop: 4 }}>Start: cd ml && python -m uvicorn server:app --port 8000</div>}
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // ═══════════════════════════════════════════════════════════
  //  LIVE SCREEN
  // ═══════════════════════════════════════════════════════════

  const last = candles[candles.length - 1];
  const prev = candles[candles.length - 2];
  const chg = last && prev ? ((last.close - prev.close) / prev.close) * 100 : 0;
  const grade = calibration?.confidence?.grade;

  return (
    <div style={{ background: "#08080f", height: "100vh", color: "#cdd6f4", fontFamily: "'JetBrains Mono', monospace", display: "flex", flexDirection: "column", overflow: "hidden" }}>
      <style>{`@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700;900&display=swap');
        @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }`}</style>

      {/* ── HEADER BAR ── */}
      <div style={{ borderBottom: "1px solid #14142a", padding: "6px 14px", display: "flex", alignItems: "center", background: "#0a0a14", flexShrink: 0 }}>
        {/* Left: Logo */}
        <div style={{ minWidth: 0 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span style={{ color: "#f5c842", fontSize: 12, letterSpacing: 4, fontWeight: 900 }}>◈ ICT BACKTEST TERMINAL</span>
            {isDemo && <span style={{ padding: "1px 6px", fontSize: 7, letterSpacing: 2, border: "1px solid #9370db", color: "#9370db" }}>DEMO</span>}
            {liveMode && (
              <span style={{ display: "inline-flex", alignItems: "center", gap: 4, padding: "1px 6px", border: "1px solid #26a69a", fontSize: 7, letterSpacing: 2, color: "#26a69a" }}>
                <span style={{ width: 5, height: 5, borderRadius: "50%", background: "#26a69a", animation: "pulse 1.5s infinite" }} />LIVE
              </span>
            )}
          </div>
          <div style={{ color: "#33334d", fontSize: 7, letterSpacing: 3 }}>
            XAU/USD · {timeframe.toUpperCase()} · {candles.length} CANDLES
            {lastUpdate && <span style={{ marginLeft: 8, color: "#1e1e33" }}>UPD {lastUpdate.toLocaleTimeString()}</span>}
          </div>
        </div>

        {/* Centre: Price + Bias */}
        {last && (
          <div style={{ marginLeft: "auto", display: "flex", gap: 16, alignItems: "center" }}>
            <div>
              <div style={{ color: "#33334d", fontSize: 7 }}>LAST CLOSE</div>
              <div style={{ color: "#f5c842", fontSize: 15, fontWeight: 700 }}>{last.close.toFixed(2)}</div>
            </div>
            <div>
              <div style={{ color: "#33334d", fontSize: 7 }}>CHANGE</div>
              <div style={{ color: chg >= 0 ? "#26a69a" : "#ef5350", fontSize: 11 }}>{chg >= 0 ? "+" : ""}{chg.toFixed(3)}%</div>
            </div>
            {analysis && (
              <span style={{
                padding: "3px 8px", fontSize: 9, fontWeight: 700, letterSpacing: 2, borderRadius: 2,
                background: analysis.bias === "bullish" ? "rgba(38,166,154,0.15)" : "rgba(239,83,80,0.15)",
                color: analysis.bias === "bullish" ? "#26a69a" : "#ef5350",
                border: `1px solid ${analysis.bias === "bullish" ? "#26a69a" : "#ef5350"}`,
              }}>
                {analysis.bias === "bullish" ? "▲ BULLISH" : "▼ BEARISH"}
              </span>
            )}
          </div>
        )}

        {/* Right: Pipeline dots + Grade */}
        <div style={{ marginLeft: 20, display: "flex", alignItems: "center", gap: 10 }}>
          {/* Pipeline dots */}
          <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
            {[
              { label: "Claude", ok: !!analysis },
              { label: "Cal", ok: !!calibration, warn: pipelineHealth?.status === "ok" && !calibration },
              { label: `Bayes ${bayesian?.total_trades || 0}`, ok: !!bayesian },
              { label: "AG", ok: false, warn: !!sessionStats?.seeded },
              { label: "Scan", ok: scannerStatus?.scheduler_running, warn: scannerStatus && !scannerStatus.scheduler_running },
            ].map((dot) => (
              <div key={dot.label} style={{ display: "flex", alignItems: "center", gap: 3 }}>
                <div style={{
                  width: 6, height: 6, borderRadius: "50%",
                  background: dot.ok ? "#26a69a" : dot.warn ? "#f5c842" : "#33334d",
                }} />
                <span style={{ fontSize: 6.5, color: "#444466" }}>{dot.label}</span>
              </div>
            ))}
          </div>

          {/* Grade badge */}
          {grade ? (
            <div style={{
              width: 28, height: 28, display: "flex", alignItems: "center", justifyContent: "center",
              fontSize: 14, fontWeight: 900, borderRadius: 3,
              background: GRADE_COLORS[grade] || "#444", color: "#08080f",
            }}>{grade}</div>
          ) : (
            <div style={{
              width: 28, height: 28, display: "flex", alignItems: "center", justifyContent: "center",
              fontSize: 14, fontWeight: 900, borderRadius: 3, background: "#1e1e33", color: "#33334d",
            }}>—</div>
          )}
        </div>
      </div>

      {/* ── MAIN LAYOUT ── */}
      <div style={{ display: "flex", flex: 1, overflow: "hidden", flexDirection: isNarrow ? "column" : "row" }}>
        {/* Chart column */}
        <div style={{ flex: "1 1 55%", display: "flex", flexDirection: "column", padding: "8px 10px", gap: 6, minWidth: 0 }}>
          {/* Toolbar */}
          <div style={{ display: "flex", gap: 5, alignItems: "center", flexWrap: "wrap", flexShrink: 0 }}>
            <span style={{ color: "#33334d", fontSize: 8 }}>TF:</span>
            {TF_OPTIONS.map((tf) => (
              <button key={tf.value} style={btn(timeframe === tf.value)} onClick={() => setTimeframe(tf.value)}>{tf.label}</button>
            ))}
            <div style={{ width: 1, height: 16, background: "#14142a", margin: "0 3px" }} />
            <button style={btn(false)} onClick={() => fetchCandles(false)} disabled={loadingData}>{loadingData ? "..." : "↻ REFRESH"}</button>
            {isChartManual && (
              <button
                style={btn(false, "#9370db")}
                onClick={resetChartScale}
                title="Reset chart scale to auto-fit"
              >
                ⊕ AUTO
              </button>
            )}
            <button style={btn(false, "#26a69a")} onClick={() => runAnalysis()} disabled={!!loadingStep || !candles.length}>
              {loadingStep ? `STEP ${loadingStep === "4h" ? "1/3" : loadingStep === "claude" ? "2/3" : "3/3"}...` : "⬡ RUN ICT ANALYSIS"}
            </button>
            <div style={{ width: 1, height: 16, background: "#14142a", margin: "0 3px" }} />
            <button style={btn(liveMode, "#26a69a")} onClick={() => setLiveMode((v) => !v)}>
              {liveMode ? "◉ LIVE" : "○ LIVE"}
            </button>
            {liveMode && (
              <>
                {REFRESH_OPTIONS.map((r) => (
                  <button key={r.seconds} style={btn(refreshInterval === r.seconds, "#33334d")} onClick={() => setRefreshInterval(r.seconds)}>{r.label}</button>
                ))}
                <button style={btn(autoAnalyze, "#9370db")} onClick={() => setAutoAnalyze((v) => !v)}>
                  {autoAnalyze ? "⬡ AUTO" : "⬡ AUTO"}
                </button>
              </>
            )}
            {(error || liveError) && <span style={{ color: "#ef5350", fontSize: 8 }}>⚠ {error || liveError}</span>}
          </div>

          {/* Chart */}
          <div style={{ flex: 1, position: "relative", minHeight: 0 }}>
            <svg ref={svgRef}
              style={{
                width: "100%", height: "100%", minHeight: 300, display: "block",
                cursor: isDragging && dragStateRef.current?.kind === "pan" ? "grabbing"
                      : isDragging ? "default"
                      : (xManualRange ? "grab" : "crosshair"),
              }}
              onMouseDown={(e) => {
                const s = chartScalesRef.current;
                if (!s) return;
                const rect = svgRef.current.getBoundingClientRect();
                const mx = e.clientX - rect.left - s.m.left;
                const my = e.clientY - rect.top - s.m.top;
                if (mx < 0 || mx > s.w || my < 0 || my > s.h) return;
                startDrag("pan", e);
              }}
              onMouseMove={(e) => {
                if (dragStateRef.current) {
                  d3.select(svgRef.current).select(".crosshair").style("display", "none");
                  return;
                }
                // ── existing crosshair logic — KEEP EVERYTHING BELOW UNCHANGED ──
                const s = chartScalesRef.current;
                if (!s || !candles.length) return;
                const rect = svgRef.current.getBoundingClientRect();
                const mx = e.clientX - rect.left - s.m.left;
                const my = e.clientY - rect.top - s.m.top;
                const ch = d3.select(svgRef.current).select(".crosshair");
                if (mx < 0 || mx > s.w || my < 0 || my > s.h) { ch.style("display", "none"); return; }
                ch.style("display", null);
                const step = s.x.step();
                const idx = Math.max(0, Math.min(Math.round((mx - s.x.bandwidth() / 2) / step), candles.length - 1));
                const cx = s.x(idx) + s.x.bandwidth() / 2;
                const c = candles[idx];
                const bull = c.close >= c.open;
                ch.select(".ch-h").attr("y1", my).attr("y2", my);
                ch.select(".ch-v").attr("x1", cx).attr("x2", cx);
                ch.select(".ch-price-bg").attr("y", my - 8);
                ch.select(".ch-price").attr("y", my).text(s.y.invert(my).toFixed(2));
                const [dd, tt] = c.datetime.split(" ");
                const [, mo, dy] = dd.split("-");
                ch.select(".ch-time-bg").attr("x", cx - 36);
                ch.select(".ch-time").attr("x", cx).text(`${+mo}/${+dy} ${tt?.slice(0, 5) || "00:00"}`);
                ch.select(".ch-ohlc").attr("fill", bull ? "#26a69a" : "#ef5350")
                  .text(`O ${c.open.toFixed(2)}  H ${c.high.toFixed(2)}  L ${c.low.toFixed(2)}  C ${c.close.toFixed(2)}`);
              }}
              onWheel={(e) => {
                const s = chartScalesRef.current;
                if (!s || !candles.length) return;
                const rect = svgRef.current.getBoundingClientRect();
                const mx = e.clientX - rect.left - s.m.left;
                const my = e.clientY - rect.top - s.m.top;
                if (mx < 0 || mx > s.w || my < 0 || my > s.h) return;
                // preventDefault for page scroll is handled by the native listener in Step 2

                const FUTURE_BARS = Math.max(10, Math.ceil(candles.length * 0.2));
                const totalSlots = candles.length + FUTURE_BARS;
                const allCandleIndices = Array.from({ length: totalSlots }, (_, i) => i);
                const firstIdx = allCandleIndices[0];
                const lastIdx = allCandleIndices[allCandleIndices.length - 1];
                const step = s.x.step();
                const anchorIndex = Math.max(0, Math.min(Math.round((mx - s.x.bandwidth() / 2) / step), totalSlots - 1));
                const anchorPrice = s.y.invert(my);

                const result = wheelZoom({
                  startYDomain: yManualDomain ?? [s.y.domain()[0], s.y.domain()[1]],
                  startXRange: xManualRange ?? [firstIdx, lastIdx],
                  anchorPrice,
                  anchorIndex,
                  deltaY: e.deltaY,
                  modifiers: { shift: e.shiftKey, ctrl: e.ctrlKey || e.metaKey },
                  chartHeight: s.h,
                  chartWidth: s.w,
                  allCandleIndices,
                });

                if (!e.ctrlKey && !e.metaKey) setXManualRange(result.xRange);
                if (!e.shiftKey) setYManualDomain(result.yDomain);
              }}
              onMouseLeave={() => { if (svgRef.current) d3.select(svgRef.current).select(".crosshair").style("display", "none"); }}
            />
            {loadingStep && (
              <div style={{ position: "absolute", inset: 0, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", background: "rgba(6,6,14,0.92)" }}>
                <div style={{ color: "#f5c842", fontSize: 10, letterSpacing: 3, marginBottom: 6 }}>
                  {loadingStep === "4h" ? "STEP 1/3: FETCHING 4H CONTEXT..." : loadingStep === "claude" ? "STEP 2/3: CLAUDE ANALYSING SETUP..." : "STEP 3/3: CALIBRATING LEVELS..."}
                </div>
                <div style={{ color: "#33334d", fontSize: 8 }}>XAU/USD · GOLD</div>
              </div>
            )}
            {loadingData && !loadingStep && (
              <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", background: "rgba(6,6,14,0.9)" }}>
                <div style={{ color: "#f5c842", fontSize: 10, letterSpacing: 3 }}>FETCHING MARKET DATA...</div>
              </div>
            )}
          </div>

          {/* Legend */}
          <div style={{ display: "flex", gap: 10, fontSize: 7, flexWrap: "wrap", flexShrink: 0 }}>
            {[
              ["#26a69a", "■ Bull OB"], ["#ef5350", "■ Bear OB"], ["#64b5f6", "■ FVG"],
              ["#f5c842", "— Entry"], ["#00e676", "━ Cal TP"], ["#ef5350", "━ Cal SL"],
              ["#9370db", "┄ 4H DR"], ["#6e7a9a", "- - Claude"],
            ].map(([c, l]) => <span key={l} style={{ color: c }}>{l}</span>)}
          </div>
        </div>

        {/* Responsive: narrow stacks panels */}
        {isNarrow && (
          <div style={{ display: "flex", borderTop: "1px solid #14142a", flexShrink: 0 }}>
            {["chart", "trade", "intel"].map((t) => (
              <div key={t} onClick={() => setMobileTab(t)} style={{
                flex: 1, padding: 6, textAlign: "center", cursor: "pointer", fontSize: 8, letterSpacing: 2,
                color: mobileTab === t ? "#f5c842" : "#33334d",
                borderBottom: mobileTab === t ? "1px solid #f5c842" : "none",
              }}>{t.toUpperCase()}</div>
            ))}
          </div>
        )}

        {/* Trade Panel */}
        {(!isNarrow || mobileTab === "trade") && (
          <div style={{
            width: isNarrow ? "100%" : 260, flexShrink: 0, borderLeft: isNarrow ? "none" : "1px solid #14142a",
            borderTop: isNarrow ? "1px solid #14142a" : "none",
            overflowY: "auto", maxHeight: isNarrow ? 400 : "none",
          }}>
            {renderTradePanel()}
          </div>
        )}

        {/* Intelligence Panel */}
        {(!isNarrow || mobileTab === "intel") && (
          <div style={{
            width: isNarrow ? "100%" : 220, flexShrink: 0, borderLeft: isNarrow ? "none" : "1px solid #14142a",
            borderTop: isNarrow ? "1px solid #14142a" : "none",
            overflowY: "auto", maxHeight: isNarrow ? 400 : "none",
          }}>
            {renderIntelPanel()}
          </div>
        )}
      </div>

      {/* ── BOTTOM BAR ── */}
      <div style={{ borderTop: "1px solid #14142a", padding: "4px 14px", display: "flex", justifyContent: "space-between", alignItems: "center", background: "#0a0a14", flexShrink: 0 }}>
        <div style={{ display: "flex", gap: 10, fontSize: 7 }}>
          {[["#26a69a", "■ Bull OB"], ["#ef5350", "■ Bear OB"], ["#64b5f6", "■ FVG"], ["#f5c842", "— Entry"], ["#00e676", "━ TP"], ["#ef5350", "━ SL"], ["#9370db", "┄ 4H DR"]].map(([c, l]) => (
            <span key={l} style={{ color: c }}>{l}</span>
          ))}
        </div>
        <div style={{ fontSize: 7, color: "#33334d" }}>
          Bayesian: {bayesian?.total_trades || 0} trades · {bayesian ? `${(bayesian.win_rate_mean * 100).toFixed(1)}% WR` : "—"}
          · AutoGluon: {datasetStats?.total >= 30 ? "TRAINED" : "LEARNING"}
          {lastUpdate && ` · Last: ${lastUpdate.toLocaleTimeString()}`}
        </div>
      </div>

      {/* Toast notification */}
      {toast && (
        <div style={{
          position: "fixed", bottom: 20, left: "50%", transform: "translateX(-50%)",
          background: "#26a69a", color: "#08080f", padding: "8px 20px", borderRadius: 4,
          fontSize: 10, fontWeight: 700, fontFamily: "monospace", zIndex: 9999,
        }}>{toast}</div>
      )}
    </div>
  );
}
