"""Enhanced ICT analysis prompts for Claude — multi-timeframe with structured output.

Sends both 1H and 4H candles so Claude can do proper ICT analysis:
  - 4H: dealing range, premium/discount, Power of 3
  - 1H: order blocks, FVGs, liquidity sweeps, entry signals
"""
import json
from datetime import datetime, timezone


# Killzone definitions in UTC (GMT) hours — standardized names
KILLZONES = [
    {"name": "Asian", "start": 0, "end": 7},
    {"name": "London", "start": 7, "end": 12},
    {"name": "NY_AM", "start": 12, "end": 16},
    {"name": "NY_PM", "start": 16, "end": 20},
]


def get_current_killzone() -> str:
    """Return the current killzone name based on UTC time, or 'Off'."""
    now = datetime.now(timezone.utc)
    hour = now.hour
    for kz in KILLZONES:
        if kz["start"] <= hour < kz["end"]:
            return kz["name"]
    return "Off"


def build_enhanced_ict_prompt(candles_1h: list, candles_4h: list,
                              intermarket: dict | None = None,
                              htf_narrative: dict | None = None,
                              setup_context: dict | None = None,
                              narrative_weights: dict | None = None,
                              prev_narrative: dict | None = None,
                              invalidation_status: str | None = None,
                              recent_context: dict | None = None,
                              regime_context: dict | None = None,
                              ml_context: dict | None = None,
                              htf_label: str | None = None,
                              key_levels: dict | None = None) -> str:
    """Build the enhanced multi-timeframe ICT analysis prompt.

    Args:
        candles_1h: List of execution OHLC candle dicts (up to 120 for 1H, 96 for 15min)
        candles_4h: List of 4H OHLC candle dicts (up to 20) — always sent, even with Opus narrative
        intermarket: Optional intermarket context dict from compute_intermarket_context()
        htf_narrative: Optional Opus HTF narrative dict (from _call_opus_narrative)
        setup_context: Optional dict with learned_rules and/or conditional_stats
                       from SetupProfileStore for historical pattern injection
        narrative_weights: Optional dict of per-field EMA weights (0.0-1.0) from
                          ClaudeAnalysisBridge.get_narrative_weights()
        prev_narrative: Optional previous narrative state dict from NarrativeStore.get_current()
        invalidation_status: 'TRIGGERED', 'APPROACHING', or 'CLEAR' from check_invalidation()
        recent_context: Optional dict from build_recent_context() with recent resolutions,
                        consumed zones, swept liquidity, and active setups
        regime_context: Optional dict from classify_regime() with structural regime label,
                        confidence, and metrics (ATR percentile, vol ratio, net movement,
                        displacements, body consistency)

    Returns:
        Complete prompt string for Claude
    """
    # Full windows — no truncation. Scanner TIMEFRAMES config controls the counts.
    h1_data = candles_1h
    h4_data = candles_4h

    # Slim candle format to reduce token usage
    h1_slim = _slim_candles(h1_data)
    h4_slim = _slim_candles(h4_data)

    now_utc = datetime.now(timezone.utc)
    current_kz = get_current_killzone()
    time_str = now_utc.strftime("%Y-%m-%d %H:%M UTC")
    day_str = now_utc.strftime("%A")

    # Build HTF narrative authority block
    narrative_block = ""
    if htf_narrative and htf_narrative.get("macro_narrative"):
        n = htf_narrative

        # Per-field trust levels from tracked accuracy
        trust_section = ""
        if narrative_weights:
            trust_section = _build_narrative_trust_section(narrative_weights)

        # System learning trends (from snapshots over time)
        learning_status = ""
        try:
            from ml.system_snapshot import SystemSnapshotRecorder
            learning_status = SystemSnapshotRecorder().build_prompt_context(days=14)
            if learning_status:
                learning_status = "\n" + learning_status
        except Exception:
            pass

        # Include weekly/daily dealing ranges if available from expanded Opus
        weekly_dr = n.get('weekly_dealing_range', {})
        daily_dr = n.get('daily_dealing_range', {})
        h4_dr = n.get('dealing_range', {})

        dealing_range_block = ""
        if weekly_dr.get('high'):
            dealing_range_block += f"Weekly Dealing Range: {weekly_dr.get('high', '?')} - {weekly_dr.get('low', '?')}\n"
        if daily_dr.get('high'):
            dealing_range_block += f"Daily Dealing Range: {daily_dr.get('high', '?')} - {daily_dr.get('low', '?')}\n"
        dealing_range_block += f"4H Dealing Range: {h4_dr.get('high', '?')} - {h4_dr.get('low', '?')}"

        narrative_block = f"""SENIOR ANALYST HTF NARRATIVE (use as directional framework, but verify against 4H candles below):
{dealing_range_block}
Premium/Discount: {n.get('premium_discount', '?')}
Power of 3 Phase: {n.get('power_of_3_phase', '?')} (confidence: {n.get('phase_confidence', '?')})
Directional Bias: {n.get('directional_bias', '?')}
Key Levels: {json.dumps(n.get('key_levels', []))}
Narrative: {n['macro_narrative']}
{('Intermarket: ' + n['intermarket_synthesis']) if n.get('intermarket_synthesis') else ''}
Session Outlook: {n.get('session_outlook', 'Not provided')}
{trust_section}{learning_status}
IMPORTANT: The senior analyst has provided the above higher-timeframe reading from weekly, daily, and 4H analysis. You MUST verify the 4H dealing range and structure against the 4H candles provided below. If your 4H reading contradicts the narrative, flag the discrepancy explicitly.{' Trust the HIGH-accuracy fields strongly; treat LOW-accuracy fields as suggestive only.' if trust_section else ''}{' Use the system learning trends to calibrate your confidence — lean into improving signals, be cautious with declining ones.' if learning_status else ''}

"""

    return f"""You are an expert ICT (Inner Circle Trader) analyst for Gold XAU/USD.

CURRENT TIME: {time_str} ({day_str})
CURRENT KILLZONE: {current_kz}

{narrative_block}{"Analyse these candles to identify the highest-probability trade setup. The senior analyst has provided the HTF narrative above — use it as your directional framework but VERIFY it against the 4H candles below. If your 4H reading contradicts the narrative, flag it." if htf_narrative else "Analyse these candles on TWO timeframes to identify the highest-probability trade setup."}

{((htf_label or "4H") + " CANDLES (" + str(len(h4_slim)) + " candles — verify the HTF narrative against these):" + chr(10) + json.dumps(h4_slim) + chr(10) if h4_slim else "")}
EXECUTION CANDLES ({len(h1_slim)} candles):
{json.dumps(h1_slim)}

{_build_intermarket_section(intermarket)}{_build_regime_section(regime_context)}{_build_setup_context_section(setup_context)}{_build_recent_context_section(recent_context)}{_build_narrative_state_section(prev_narrative, invalidation_status)}{_build_ml_context_section(ml_context)}{_build_key_levels_section(key_levels)}ANALYSIS FRAMEWORK:
1. Determine the 4H dealing range (recent swing high to swing low). Is price in premium (upper half) or discount (lower half)?
2. Has 4H sell-side or buy-side liquidity been swept recently? This determines the Power of 3 phase.
3. Check KEY LEVELS: Has PDH/PDL been swept? Has the Asia session high/low been raided? A sweep of these levels followed by reversal is a high-probability entry signal (Judas swing). Price position relative to PDH/PDL and PWH/PWL determines premium/discount context.
4. On the 1H: identify the strongest Order Block born from genuine displacement. Has it been tested before?
5. Is there an unfilled Fair Value Gap overlapping or near the OB zone?
6. Was there a recent liquidity sweep on the 1H confirming manipulation? Cross-reference with KEY LEVELS — a sweep of Asia H/L or PDH/PDL during a killzone is the highest-conviction manipulation signal.
7. Is there a break of structure or change of character confirming direction?
8. CRITICAL: Only suggest entry if there is a pullback or rejection into the zone. Do NOT enter on displacement candles.
9. STOP LOSS: Gold's noise floor is 3.0 ATR. Place SL below/above the structural level (OB boundary) but ensure minimum 3.0 ATR distance from entry. Tighter SLs get stopped by normal volatility ~65%+ of the time.
10. If there is no high-probability setup right now, say so honestly. Set entry to null.

Return ONLY valid JSON:
{{
  "bias": "bullish|bearish",
  "summary": "string — include 4H dealing range context, premium/discount position, and the narrative",
  "htf_context": {{
    "dealing_range_high": number,
    "dealing_range_low": number,
    "premium_discount": "premium|discount|equilibrium",
    "power_of_3_phase": "accumulation|manipulation|distribution",
    "recent_sweep": "bsl|ssl|none",
    "htf_bias": "bullish|bearish|neutral"
  }},
  "orderBlocks": [{{
    "type": "bullish|bearish", "high": number, "low": number, "candleIndex": number,
    "strength": "strong|moderate|weak",
    "times_tested": number,
    "note": "string"
  }}],
  "fvgs": [{{
    "type": "bullish|bearish", "high": number, "low": number, "startIndex": number,
    "filled": boolean, "fill_percentage": number,
    "overlaps_ob": boolean,
    "note": "string"
  }}],
  "liquidity": [{{
    "type": "buyside|sellside", "price": number, "candleIndex": number,
    "swept": boolean,
    "note": "string"
  }}],
  "structure": {{
    "type": "bos|choch|none",
    "direction": "bullish|bearish",
    "break_candle_index": number,
    "note": "string"
  }},
  "entry": {{
    "price": number,
    "direction": "long|short",
    "entry_type": "rejection|retracement",
    "rationale": "string"
  }} | null,
  "stopLoss": {{"price": number, "rationale": "string"}} | null,
  "takeProfits": [{{"price": number, "rationale": "string", "rr": number}}] | null,
  "killzone": "Asian|London|NY_AM|NY_PM|Off",
  "confluences": ["string"],
  "setup_quality": "A|B|C|D|no_trade",
  "warnings": ["string"],
  "narrative_state": {{
    "thesis": "string — 1-2 sentence narrative of what you think is happening",
    "p3_phase": "accumulation|manipulation|distribution|none",
    "p3_progress": "early|mid|late",
    "directional_bias": "bullish|bearish|neutral",
    "bias_confidence": 0.0-1.0,
    "key_levels": [{{"price": number, "label": "string", "role": "target|support|resistance|invalidation"}}],
    "expected_next_move": "string — what you expect to happen next",
    "invalidation": {{"condition": "string — what would kill this thesis", "price_level": number, "direction": "above|below"}},
    "watching_for": ["string — structural events you want to see before acting"],
    "last_revision": "string|null — if you changed your thesis this scan, explain why"
  }}
}}"""


def build_haiku_backtest_prompt(candles: list[dict],
                                structural_elements: dict) -> str:
    """Build compact Haiku validation prompt for backtest candidates.

    Includes pre-detected structural elements so Haiku validates rather than
    discovers. Uses slim candle format (last 40, no dates).

    Args:
        candles: OHLCV candle dicts (last 40 used, dates already stripped)
        structural_elements: Dict from structural_scan with OB/FVG/sweep info
    """
    # Format structural summary
    parts = []
    if structural_elements.get("ob_count", 0) > 0:
        types = structural_elements.get("ob_types", [])
        parts.append(f"Order Blocks: {structural_elements['ob_count']} ({', '.join(types)})")
    if structural_elements.get("fvg_count", 0) > 0:
        types = structural_elements.get("fvg_types", [])
        parts.append(f"FVGs: {structural_elements['fvg_count']} ({', '.join(types)})")
    if structural_elements.get("sweep_detected"):
        parts.append("Liquidity sweep confirmed")
    if structural_elements.get("displacement"):
        parts.append("Displacement candle present")
    struct = structural_elements.get("structure_score", 0)
    if abs(struct) > 0.3:
        bias = "bullish" if struct > 0 else "bearish"
        parts.append(f"Market structure: {bias} ({struct:+.2f})")
    if structural_elements.get("price_in_zone"):
        parts.append("Price is inside an OB/FVG zone")

    summary = "\n".join(f"- {p}" for p in parts) if parts else "- Minimal confluence"

    # Slim candle format (last 40 only, no dates)
    tail = candles[-40:]
    slim = []
    for j, c in enumerate(tail):
        slim.append(
            f"{j}: O={c['open']:.2f} H={c['high']:.2f} "
            f"L={c['low']:.2f} C={c['close']:.2f} V={c.get('volume', 0)}"
        )
    candle_str = "\n".join(slim)

    return f"""You are screening XAU/USD 1H candles for ICT trade setups.

PRE-DETECTED structural elements at the current candle:
{summary}

CANDLES (last 40, index: OHLCV, no dates):
{candle_str}

TASK: Given the structural elements already identified, determine if this is a valid ICT trade setup.

Respond with JSON only:
{{"valid": true, "direction": "long or short", "entry_price": float, "sl_price": float, "reasoning": "one sentence"}}
OR
{{"valid": false, "reasoning": "one sentence"}}

Say valid=true ONLY if: (1) market structure supports the direction, (2) entry is at a validated OB or FVG, (3) there is a clear liquidity target for TP. If marginal, say false."""


def build_screen_prompt(candles: list, htf_candles: list, timeframe: str,
                        prev_narrative: dict | None = None,
                        watch_zones: list | None = None,
                        pending_setups: list | None = None) -> str:
    """Build a short screening prompt for Haiku — checks if a setup is forming or developing.

    This is ~70% cheaper than sending the full analysis prompt to Sonnet.
    Only if Haiku says 'yes' do we escalate to the full Sonnet analysis.

    When prev_narrative and/or watch_zones are provided, Haiku gets a
    context-aware prompt that tells it WHAT to look for (thesis direction,
    specific price zones) instead of blind-screening raw candles.

    Candle windows are sized per-timeframe to ensure Haiku sees enough
    structure to make an informed screen decision:
      15min: 96 candles (24h) + 48 HTF (2 days)
      1h:    72 candles (3 days) + 20 HTF (3.3 days)
      4h+:   full candle set + HTF
    """
    # Per-timeframe screen windows — enough for a full swing/P3 cycle
    screen_counts = {
        "15min": {"exec": 96, "htf": 48},
        "1h":    {"exec": 72, "htf": 20},
        "4h":    {"exec": 40, "htf": 12},
        "1day":  {"exec": 30, "htf": 0},
    }
    counts = screen_counts.get(timeframe, {"exec": 60, "htf": 20})
    slim = _slim_candles(candles[-counts["exec"]:])
    htf_slim = _slim_candles(htf_candles[-counts["htf"]:]) if htf_candles and counts["htf"] > 0 else []

    now_utc = datetime.now(timezone.utc)
    current_kz = get_current_killzone()

    htf_block = ("HTF context (" + str(len(htf_slim)) + " candles):\n"
                 + json.dumps(htf_slim)) if htf_slim else ""

    # ── Context-aware prompt: when Opus thesis + watch zones available ──
    if prev_narrative or watch_zones:
        thesis_line = ""
        if prev_narrative:
            bias = prev_narrative.get("directional_bias", "neutral")
            phase = prev_narrative.get("p3_phase") or prev_narrative.get("power_of_3_phase", "?")
            conf = prev_narrative.get("bias_confidence", 0)
            scans = prev_narrative.get("scan_count", 1)
            thesis_line = f"Current thesis: {bias.title()} {phase}, {conf:.0%} confidence, scan {scans}."
            inv = prev_narrative.get("invalidation") or prev_narrative.get("invalidation_level")
            if inv:
                thesis_line += f"\nInvalidation level: {inv}"

        zones_line = ""
        if watch_zones:
            zone_strs = []
            for z in watch_zones[:5]:
                level = z.get("level") or z.get("price", "?")
                ztype = z.get("type", "zone")
                status = z.get("status", "untested")
                zone_strs.append(f"{level} {ztype} ({status})")
            zones_line = f"Opus watch zones: {', '.join(zone_strs)}"

        pending_line = ""
        if pending_setups:
            pending_line = f"Active pending: {len(pending_setups)} setup(s) on this TF."

        return f"""XAU/USD ICT screen — {timeframe} candles ({len(slim)} candles). Time: {now_utc.strftime("%H:%M UTC")} ({current_kz})

{thesis_line}
{zones_line}
{pending_line}

{timeframe} candles:
{json.dumps(slim)}
{htf_block}

Given the thesis and watch zones above, is price:
- Approaching, testing, or reacting to any watch zone?
- Showing displacement (3+ ATR candle body) in the thesis direction?
- Showing structure shift that would CHANGE the thesis?

Also flag any standard ICT setup elements (OB, FVG, liquidity sweep, BOS/CHoCH).

Err on the side of YES — it's better to pass a marginal setup to the senior analyst than miss a valid one.

Reply ONLY valid JSON:
{{"setup_possible": true|false, "zone_interaction": "zone description"|null, "direction": "long"|"short"|null, "reason": "one sentence"}}"""

    # ── Generic prompt: no context available (cold start) ──
    return f"""XAU/USD ICT screen — {timeframe} candles ({len(slim)} candles). Time: {now_utc.strftime("%H:%M UTC")} ({current_kz})

{timeframe} candles:
{json.dumps(slim)}
{htf_block}

Is there an ICT setup forming, developing, OR has one recently completed? Answer YES if ANY of these are present:
- Liquidity sweep (SSL or BSL taken) in recent candles
- Order block formed after displacement
- Fair value gap (unfilled or partially filled)
- Break of structure or change of character
- Price approaching a key level, OB, or FVG zone
- Clear market structure shift (higher highs/lows or lower highs/lows)

Err on the side of YES — it's better to pass a marginal setup to the senior analyst than miss a valid one. Only say NO if the market is clearly ranging with no structure.

Reply ONLY valid JSON:
{{"setup_possible": true|false, "direction": "long"|"short"|null, "reason": "one sentence"}}"""


def _build_narrative_state_section(narrative_state: dict | None,
                                    invalidation_status: str | None) -> str:
    """Build the YOUR PREVIOUS THESIS prompt section.

    Injected after RECENT CONTEXT and before ANALYSIS FRAMEWORK so Claude sees
    what it was thinking on the previous scan before getting analysis instructions.
    """
    if not narrative_state:
        return ""  # First scan on this TF — no prior thesis

    lines = ["=== YOUR PREVIOUS THESIS ==="]

    # Stale warning
    age = narrative_state.get("thesis_age_minutes", 0)
    status = narrative_state.get("status", "active")
    if status == "expired":
        lines.append(f"(STALE — {age} min old, may no longer apply)")

    # The thesis itself
    thesis = narrative_state.get("thesis", "")
    if thesis:
        lines.append(f"Thesis: {thesis}")

    p3 = narrative_state.get("p3_phase", "none")
    p3_prog = narrative_state.get("p3_progress", "?")
    lines.append(f"P3 Phase: {p3} ({p3_prog})")

    bias = narrative_state.get("directional_bias", "neutral")
    conf = narrative_state.get("bias_confidence", 0.5)
    lines.append(f"Bias: {bias} (confidence: {conf:.0%})")

    scan_count = narrative_state.get("scan_count", 1)
    lines.append(f"Scans on this thesis: {scan_count}")

    # What you were watching for
    watching = narrative_state.get("watching_for", [])
    if watching:
        lines.append("You were watching for: " + ", ".join(str(w) for w in watching))

    # What you expected
    expected = narrative_state.get("expected_next_move")
    if expected:
        lines.append(f"You expected: {expected}")

    # Key levels
    key_levels = narrative_state.get("key_levels", [])
    if key_levels:
        level_strs = [f"{kl.get('price', '?')} ({kl.get('label', '?')})"
                      for kl in key_levels[:5]]
        lines.append("Key levels: " + ", ".join(level_strs))

    # Invalidation check result
    if invalidation_status == "TRIGGERED":
        inv = narrative_state.get("invalidation", {})
        lines.append("")
        lines.append("*** INVALIDATION TRIGGERED ***")
        lines.append(f"Your stated condition was: {inv.get('condition', '?')}")
        lines.append("Price has breached your invalidation level.")
        lines.append("You MUST re-assess from current structure. Do not continue")
        lines.append("the previous thesis without explicit structural justification.")
    elif invalidation_status == "APPROACHING":
        inv = narrative_state.get("invalidation", {})
        lines.append("")
        lines.append(f"NOTE: Price is approaching your invalidation at "
                     f"{inv.get('price_level')}. Assess carefully.")

    # Safeguard 5: Structural contradiction (set by scanner before prompt build)
    contradiction = narrative_state.get("_structural_contradiction")
    if contradiction:
        lines.append("")
        lines.append(f"STRUCTURAL CONTRADICTION: {contradiction}")

    # Confidence decay warning
    if conf < 0.30:
        lines.append("")
        lines.append("This thesis is losing conviction. Consider whether the "
                     "market picture has fundamentally changed.")

    # Phase 6: Killzone handoff context
    kz_summary = narrative_state.get("killzone_summary")
    if kz_summary:
        lines.append("")
        lines.append("=== PRIOR SESSION HANDOFF ===")
        lines.append(kz_summary)
        lines.append("=== END HANDOFF ===")

    lines.append("")
    lines.append("Continue, refine, or revise this thesis based on current candles.")
    lines.append("If revising, explain why in last_revision.")
    lines.append("=== END PREVIOUS THESIS ===")
    lines.append("")
    return "\n".join(lines) + "\n"


def _build_ml_context_section(ml_context: dict | None) -> str:
    """Format ML context as a prompt section for Sonnet.

    This is Claude's statistical memory — historical patterns and outcome
    distributions it cannot remember between API calls. Placed AFTER the
    narrative state and BEFORE the analysis framework.
    """
    if not ml_context:
        return ""

    lines = ["=== YOUR STATISTICAL MEMORY ==="]

    # Pattern match
    dna_wr = ml_context.get("dna_win_rate")
    if dna_wr is not None:
        dna_rr = ml_context.get("dna_avg_rr", 0)
        dna_n = ml_context.get("dna_sample_size", 0)
        lines.append(
            f"Pattern match: {dna_wr:.0%} WR across {dna_n} similar setups, "
            f"avg R:R {dna_rr:.1f}:1")
        opus_acc = ml_context.get("opus_accuracy")
        if opus_acc is not None:
            lines.append(
                f"  Opus accuracy on this narrative type: {opus_acc:.0%}")

    # Regime
    regime = ml_context.get("regime", "UNKNOWN")
    sl_floor = ml_context.get("sl_floor_atr", 3.0)
    mae = ml_context.get("mae_percentile_80", 4.0)
    vol_ratio = ml_context.get("vol_ratio", 1.0)
    lines.append(f"\nRegime: {regime} (vol_ratio {vol_ratio:.1f}x normal)")
    lines.append(
        f"  Required SL floor: {sl_floor:.1f} ATR "
        f"(below this, high stop-out risk)")
    lines.append(
        f"  Expected drawdown before move: {mae:.1f} ATR "
        f"(80th percentile MAE)")

    # Entry placement
    ep = ml_context.get("entry_placement")
    if ep:
        delta = ml_context.get("entry_placement_delta_rr", 0)
        lines.append(
            f"  Best entry position: {ep} "
            f"(outperforms alternatives by +{delta:.1f} R:R)")

    # Overall win rate
    wr = ml_context.get("bayesian_wr")
    if wr is not None:
        trend = ml_context.get("bayesian_trend", 0)
        if trend:
            trend_str = (f"trending {'up' if trend > 0 else 'down'} "
                         f"{abs(trend):.1f}pp")
        else:
            trend_str = "stable"
        lines.append(f"\nOverall win rate: {wr:.1%} ({trend_str})")

    # Intermarket
    im = ml_context.get("intermarket_quality")
    if im and im != "unknown":
        lines.append(f"\nIntermarket signal quality: {im.upper()}")

    lines.append("=== END STATISTICAL MEMORY ===")
    lines.append("")
    return "\n".join(lines) + "\n"


def _build_key_levels_section(key_levels: dict | None) -> str:
    """Build the === KEY LEVELS === section for the prompt.

    Formats pre-computed ICT key levels (PDH/PDL, PWH/PWL, PMH/PML,
    Asia session H/L, previous session H/L) with equilibrium midpoints.
    These are objective price levels — not from LLM analysis.
    """
    if not key_levels or key_levels.get("levels_computed", 0) == 0:
        return ""

    try:
        from ml.key_levels import format_key_levels_for_prompt
        section = format_key_levels_for_prompt(key_levels)
        if section:
            return section + "\n\n"
    except Exception:
        pass
    return ""


def _build_regime_section(regime_context: dict | None) -> str:
    """Build the MARKET REGIME section for the prompt.

    Tells Claude what structural regime the market is in so it can calibrate
    its thesis confidence and setup expectations accordingly.
    """
    if not regime_context:
        return ""

    regime = regime_context.get("regime", "RANGING")
    confidence = regime_context.get("confidence", 0)
    metrics = regime_context.get("metrics", {})

    # Regime-specific ICT implications
    implications = {
        "TRENDING_IMPULSIVE": (
            "Breakout entries work well. OB retests may not pull back far. "
            "Tight SL, wide TP. Power of 3 distribution phase."
        ),
        "TRENDING_CORRECTIVE": (
            "OB/FVG retest entries are ideal. Standard SL width. "
            "Classic ICT pullback setups."
        ),
        "RANGING": (
            "Range extremes are key. BSL/SSL at range boundaries become targets. "
            "Accumulation/manipulation setups. Wider SL."
        ),
        "VOLATILE_CHOPPY": (
            "Dangerous for entries. Many false displacements. "
            "Requires higher confluence threshold. Consider sitting out."
        ),
        "QUIET_DRIFT": (
            "Asian session classic. Only highest-confluence setups. "
            "Very tight SL possible. FVGs tend to fill slowly."
        ),
    }

    atr_pct = metrics.get("atr_percentile", 0.5)
    net_move = metrics.get("net_movement_atr", 0)
    disp_count = metrics.get("displacement_count", 0)

    lines = [
        f"=== MARKET REGIME ({regime}) ===",
        f"Regime: {regime} (confidence: {confidence:.0%})",
        f"ATR: {atr_pct:.0%} percentile | Net move: {net_move:.1f} ATR | "
        f"Displacements: {disp_count}",
        f"Implication: {implications.get(regime, 'Standard analysis.')}",
        "=== END REGIME ===",
        "",
    ]
    return "\n".join(lines) + "\n"


def _build_intermarket_section(intermarket: dict | None) -> str:
    """Build the INTERMARKET CONTEXT section for the prompt."""
    if not intermarket or not intermarket.get("narrative"):
        return ""

    im = intermarket
    lines = ["INTERMARKET CONTEXT (use to validate or invalidate your gold setup):"]
    lines.append(f"Gold 20-bar change: {im.get('gold_pct_20', 0):+.2f}%")

    if im.get("dxy_pct_20") is not None:
        lines.append(f"DXY 20-bar change: {im['dxy_pct_20']:+.2f}% | Range position: {im.get('dxy_range_position', 0.5):.0%}")
        lines.append(f"Gold-DXY correlation (20-bar): {im.get('gold_dxy_corr_20', 0):.2f}")
        if im.get("gold_dxy_diverging"):
            lines.append("DIVERGENCE: Gold and DXY moving in same direction — unusual, validate carefully.")

    if im.get("us10y_pct_20") is not None:
        direction = "rising" if im.get("yield_direction", 0) == 1 else "falling" if im.get("yield_direction", 0) == -1 else "flat"
        lines.append(f"US10Y 20-bar change: {im['us10y_pct_20']:+.2f}% ({direction} yields)")

    lines.append(f"Session: {im.get('session_strength', 'unknown')} correlation period")
    lines.append(f"Narrative: {im['narrative']}")
    lines.append("")
    lines.append("INTERMARKET RULES:")
    lines.append("- Gold LONG while DXY breaking out bullish → lower confidence")
    lines.append("- Gold SHORT while DXY breaking down → lower confidence")
    lines.append("- Gold dropping while yields falling → likely liquidity grab, not real sell-off")
    lines.append("- Gold rising despite rising yields → strong demand, momentum may override correlation")
    lines.append("- Divergence during London/NY overlap is a STRONG warning; during Asian, less significant")
    lines.append("")

    return "\n".join(lines) + "\n"


_TRUST_FIELD_LABELS = {
    "directional_bias": "Directional Bias",
    "p3_phase": "Power of 3 Phase",
    "premium_discount": "Premium/Discount",
    "confidence_calibration": "Confidence Calibration",
    "intermarket_synthesis": "Intermarket Synthesis",
    "key_levels": "Key Levels",
}


def _build_narrative_trust_section(weights: dict) -> str:
    """Build per-field trust level annotation for narrative block.

    Categorizes each field as HIGH (>0.7), MEDIUM (0.3-0.7), or LOW (<0.3)
    based on EMA accuracy weights from resolved trade outcomes.
    """
    lines = ["FIELD ACCURACY (from tracked trade outcomes):"]
    for field, label in _TRUST_FIELD_LABELS.items():
        w = weights.get(field, 0.5)
        if w >= 0.7:
            trust = "HIGH"
        elif w >= 0.3:
            trust = "MEDIUM"
        else:
            trust = "LOW"
        lines.append(f"  {label}: {trust} ({w:.0%} accuracy)")
    return "\n".join(lines)


def _build_recent_context_section(recent_context: dict | None) -> str:
    """Build the RECENT CONTEXT section for the prompt.

    Tells Claude what just happened on this timeframe — consumed zones,
    swept liquidity, recent resolutions, active setups.
    Uses format_recent_context() from ml.recent_context.
    """
    if not recent_context:
        return ""
    try:
        from ml.recent_context import format_recent_context
        return format_recent_context(recent_context)
    except Exception:
        return ""


def _build_setup_context_section(setup_context: dict | None) -> str:
    """Build the HISTORICAL PATTERN INTELLIGENCE section for the prompt.

    Injects learned rules and/or conditional stats from the SetupProfileStore.
    Capped at 5 rules to stay within ~200 token budget.
    """
    if not setup_context:
        return ""

    lines = []

    stats = setup_context.get("conditional_stats")
    if stats and stats.get("match_count", 0) >= 5:
        lines.append(f"HISTORICAL PATTERN INTELLIGENCE (from {stats['match_count']} similar past setups):")
        lines.append(f"Win rate for this pattern: {stats['win_rate']:.0%}")
        lines.append(f"Average R:R achieved: {stats['avg_rr']:.1f}R")
        if stats.get("best_outcome"):
            lines.append(f"Most common outcome: {stats['best_outcome']}")
        if stats.get("avg_mfe") is not None:
            lines.append(f"Avg max favourable move: ${stats['avg_mfe']:.1f}")
        if stats.get("avg_mae") is not None:
            lines.append(f"Avg max adverse move: ${stats['avg_mae']:.1f}")
        lines.append("")

    rules = setup_context.get("learned_rules", [])
    if rules:
        if not lines:
            lines.append("HISTORICAL PATTERN INTELLIGENCE:")
        lines.append("LEARNED PATTERNS FROM TRADE HISTORY:")
        for rule in rules[:5]:
            lines.append(f"- {rule}")
        lines.append("")

    # Entry placement guidance (from resolved trade analysis)
    pg = setup_context.get("placement_guidance")
    if pg and pg.get("status") == "active" and pg.get("rules"):
        if not lines:
            lines.append("HISTORICAL PATTERN INTELLIGENCE:")
        lines.append(f"ENTRY PLACEMENT INTELLIGENCE ({pg.get('total_trades', 0)} historical trades analysed):")
        for rule in pg["rules"][:3]:  # Cap at 3 placement rules
            lines.append(f"- {rule}")
        lines.append("")

    if lines:
        # Token budget: trim to ~400 tokens max (~100 words)
        combined = "\n".join(lines)
        if len(combined) > 1600:  # ~400 tokens ≈ 1600 chars
            lines = lines[:15]  # Keep first 15 lines
        lines.append("NOTE: These statistics inform confidence calibration, not direction. "
                      "ICT structure analysis always takes priority.")
        lines.append("")

    return "\n".join(lines) + "\n" if lines else ""


OPUS_NARRATIVE_SYSTEM = """You are the SENIOR MACRO ANALYST for a gold trading desk. You read market structure on the 4H and daily timeframes. You think in terms of accumulation/manipulation/distribution (Power of 3), premium/discount arrays, and institutional order flow.

Your job is to provide the higher-timeframe narrative — the "story" the market is telling — so that junior analysts can find precise entries within your framework. You are NOT looking for entries yourself. You are setting the directional bias and identifying the key structural levels.

Be honest about uncertainty. If the structure is unclear, say so. "I don't know" is a valid answer. A bad HTF narrative leads to bad entries downstream.

You respond ONLY with valid JSON. No explanation text outside the JSON."""


OPUS_NARRATIVE_SYSTEM_RECAP = """You are a senior ICT analyst reviewing what happened during a trading session. Summarize the session objectively: what levels were tested, what structure formed, what the market was trying to do. Focus on facts from the trade data, not speculation about the next session."""


def _build_narrative_feedback_block(feedback: dict | None) -> str:
    """Build the track record + gold examples block for the narrative prompt."""
    if not feedback:
        return ""

    blocks = []

    # Track record from EMA weights
    weights = feedback.get("weights", {})
    if weights:
        field_labels = {
            "directional_bias": "Directional bias",
            "p3_phase": "Power of 3 phase calls",
            "premium_discount": "Premium/discount zone",
            "confidence_calibration": "Confidence calibration",
            "intermarket_synthesis": "Intermarket synthesis",
            "key_levels": "Key level identification",
        }
        # Only show fields with enough data
        lines = []
        for field, label in field_labels.items():
            w = weights.get(field)
            if isinstance(w, dict):
                pct = w.get("weight", 0.5)
                total = w.get("total", 0)
            else:
                pct = w if w is not None else 0.5
                total = 0
            if total >= 5:
                lines.append(f"- {label} accuracy: {pct * 100:.0f}% ({total} trades)")

        if lines:
            # Determine emphasis from bandit params if available
            arm_params = feedback.get("arm_params", {})
            emphasis = arm_params.get("emphasis", "weak_fields")
            if emphasis == "weak_fields":
                emphasis_note = "Focus more effort on improving your weakest areas (below 50%). For any field below 45%, express genuine uncertainty rather than committing."
            elif emphasis == "strong_fields":
                emphasis_note = "Double down on your strongest areas — they're driving profitable trades. Spend less effort on weak areas."
            else:
                emphasis_note = "Weight your analysis effort proportionally — more depth on what's working, less on what isn't."

            blocks.append(
                "YOUR RECENT TRACK RECORD (use to calibrate your confidence):\n"
                + "\n".join(lines) + "\n\n"
                + emphasis_note + "\n"
            )

    # Gold examples
    examples = feedback.get("examples", [])
    if examples:
        ex_lines = []
        for i, ex in enumerate(examples[:3], 1):
            n = ex.get("narrative_json", {})
            ex_lines.append(
                f"Example {i} ({ex.get('session', '?')}, {ex.get('direction', '?')}, hit {ex.get('outcome', '?')}):\n"
                f"  Bias: {n.get('directional_bias', '?')} | Phase: {n.get('power_of_3_phase', '?')} | "
                f"Zone: {n.get('premium_discount', '?')}\n"
                f"  Narrative: {(n.get('macro_narrative') or 'N/A')[:150]}"
            )
        blocks.append(
            "EXAMPLES OF HIGH-QUALITY NARRATIVES THAT LED TO WINNING TRADES:\n"
            + "\n\n".join(ex_lines) + "\n\n"
            + "Study these. They represent your best recent work. Aim for this level of specificity.\n"
        )

    # Opus rejection accuracy per session (segmented policy feedback)
    try:
        from ml.claude_bridge import ClaudeAnalysisBridge
        rejection_ctx = ClaudeAnalysisBridge().build_opus_rejection_context()
        if rejection_ctx:
            blocks.append(rejection_ctx + "\n")
    except Exception:
        pass

    # System learning trends
    try:
        from ml.system_snapshot import SystemSnapshotRecorder
        learning_ctx = SystemSnapshotRecorder().build_prompt_context(days=14)
        if learning_ctx:
            blocks.append(learning_ctx + "\n")
    except Exception:
        pass

    if not blocks:
        return ""
    return "\n".join(blocks) + "\n"


def build_opus_narrative_prompt(candles_4h: list, candles_daily: list = None,
                                 intermarket: dict | None = None,
                                 session_recap: dict | None = None,
                                 narrative_feedback: dict | None = None,
                                 weekly_candles: list | None = None) -> str:
    """Build prompt for Opus HTF narrative architect.

    Args:
        candles_4h: Last 40 4H candles (~7 days)
        candles_daily: Last 45 daily candles (~6.5 weeks)
        intermarket: Intermarket context dict (optional)
        session_recap: Most recent session recap from Opus (optional)
        narrative_feedback: Optional dict with 'weights' and 'examples' for self-improvement
        weekly_candles: Last 12 weekly candles (~3 months) for macro dealing range

    Returns:
        Complete narrative prompt string
    """
    h4_slim = _slim_candles(candles_4h[-40:])
    daily_slim = _slim_candles(candles_daily[-45:]) if candles_daily else []
    weekly_slim = _slim_candles(weekly_candles[-12:]) if weekly_candles else []

    now_utc = datetime.now(timezone.utc)
    current_kz = get_current_killzone()
    time_str = now_utc.strftime("%Y-%m-%d %H:%M UTC")
    day_str = now_utc.strftime("%A")

    intermarket_block = _build_intermarket_section(intermarket) if intermarket else ""

    recap_block = ""
    if session_recap:
        recap_block = f"""PREVIOUS SESSION RECAP (from the last killzone):
{session_recap.get('macro_narrative', 'No recap available.')}
Key levels tested: {json.dumps(session_recap.get('key_levels', []))}
Carry-forward notes: {session_recap.get('session_outlook', 'None')}

"""

    weekly_block = ""
    if weekly_slim:
        weekly_block = f"""WEEKLY CANDLES (last 12 — ~3 months, macro dealing range):
{json.dumps(weekly_slim)}

"""

    return f"""Provide the HIGHER TIMEFRAME NARRATIVE for XAU/USD.
Time: {time_str} ({day_str}), entering {current_kz} killzone.

{weekly_block}DAILY CANDLES (last {len(daily_slim)} — ~6 weeks):
{json.dumps(daily_slim)}

4H CANDLES (last {len(h4_slim)} — ~7 days):
{json.dumps(h4_slim)}

{intermarket_block}{recap_block}{_build_narrative_feedback_block(narrative_feedback)}ANALYSIS (read TOP-DOWN — weekly first, then daily, then 4H):
1. WEEKLY: Identify the macro dealing range (the dominant swing high to swing low over the past 2-3 months). Is price in the premium or discount half of this range? Has weekly liquidity (BSL/SSL) been swept recently?
2. DAILY: Within the weekly context, identify the daily dealing range. What Power of 3 phase is playing out on the daily? Has accumulation completed? Is manipulation occurring? Is distribution underway? Look for the PRIOR cycle to confirm where we are in the CURRENT cycle.
3. 4H: How does the 4H structure nest inside the daily narrative? Is 4H confirming or diverging from the daily bias? Identify the 4H dealing range, key OBs, FVGs, and recent liquidity sweeps.
4. Has significant buy-side or sell-side liquidity been swept on ANY timeframe?
5. What are the key structural levels — unfilled FVGs, tested/untested OBs, liquidity pools?
6. What is the directional bias for the current killzone given the full top-down picture?
7. If intermarket data is available, synthesize it into the narrative

Return ONLY valid JSON:
{{
  "weekly_dealing_range": {{"high": number, "low": number}},
  "daily_dealing_range": {{"high": number, "low": number}},
  "dealing_range": {{"high": number, "low": number}},
  "premium_discount": "premium|discount|equilibrium",
  "power_of_3_phase": "accumulation|manipulation|distribution",
  "phase_confidence": "high|medium|low",
  "p3_progress": "early|mid|late",
  "directional_bias": "bullish|bearish|neutral",
  "bias_confidence": 0.0-1.0,
  "key_levels": [{{"price": number, "type": "bsl|ssl|ob|fvg", "timeframe": "weekly|daily|4h", "note": "string"}}],
  "watch_zones": [{{"level": number, "type": "OB|FVG|BSL|SSL", "direction": "bullish|bearish|neutral", "status": "untested|tested|swept", "note": "string"}}],
  "macro_narrative": "string — 3-5 sentence top-down synthesis starting from weekly, through daily, to 4H. Explain how the timeframes nest.",
  "invalidation_level": number | null,
  "intermarket_synthesis": "string — how DXY/yields affect gold" | null,
  "session_outlook": "string — what to expect in the current killzone given the full HTF story"
}}"""


def build_session_recap_prompt(setups: list, killzone: str, date: str) -> str:
    """Build prompt for Opus session recap.

    Args:
        setups: List of resolved/pending setup dicts from the session
        killzone: The killzone that just ended
        date: Date string

    Returns:
        Recap prompt string
    """
    # Summarize each setup concisely
    setup_summaries = []
    for s in setups[:20]:  # Cap at 20 to control tokens
        summary = {
            "tf": s.get("timeframe", "?"),
            "dir": s.get("direction", "?"),
            "entry": s.get("entry_price", 0),
            "outcome": s.get("outcome", "pending"),
            "quality": s.get("setup_quality", "?"),
            "rr": s.get("pnl_rr", 0),
        }
        setup_summaries.append(summary)

    return f"""SESSION RECAP: {killzone} killzone on {date}

Setups detected during this session:
{json.dumps(setup_summaries, indent=2)}

Summarize this session objectively:
1. What key levels were tested or swept?
2. What was the dominant direction?
3. Did the Power of 3 narrative play out (accumulation → manipulation → distribution)?
4. What structure carries forward to the next session?

Return ONLY valid JSON:
{{
  "session": "{killzone}",
  "date": "{date}",
  "dominant_direction": "bullish|bearish|mixed",
  "levels_tested": [{{"price": number, "type": "bsl|ssl|ob|fvg", "result": "held|broken|swept"}}],
  "p3_played_out": true|false,
  "narrative_summary": "string — 2-3 sentences on what happened",
  "carry_forward": "string — what the next session should watch for"
}}"""


OPUS_PROSPECT_SYSTEM = """You are a senior ICT analyst marking your chart BEFORE a trading session opens. You are identifying zones and conditional setups ahead of time.

Do NOT predict what will happen. Instead, identify WHERE setups would form and WHAT conditions would confirm them. Think like a trader preparing their chart before the session — marking liquidity, OBs, FVGs, and writing IF/THEN scenarios.

Be specific with price levels. Every zone needs exact prices. Every conditional setup needs a clear trigger condition that can be verified by checking price against levels.

You respond ONLY with valid JSON. No explanation text outside the JSON."""


def build_prospect_prompt(candles_4h: list, candles_1h: list,
                           intermarket: dict | None = None,
                           session_recap: dict | None = None,
                           upcoming_killzone: str = "London",
                           current_price: float | None = None,
                           prior_session_stats: dict | None = None) -> str:
    """Build prompt for zone prospecting ahead of a killzone.

    Args:
        candles_4h: Last 20 4H candles
        candles_1h: Last 50 1H candles
        intermarket: Intermarket context dict
        session_recap: Previous session recap
        upcoming_killzone: Which killzone is about to open
        current_price: Live XAU/USD price (helps Opus draw relevant zones)
        prior_session_stats: Dict with prior killzone win/loss stats for trend context

    Returns:
        Prospect prompt string
    """
    h4_slim = _slim_candles(candles_4h[-20:])
    h1_slim = _slim_candles(candles_1h[-50:])

    now_utc = datetime.now(timezone.utc)
    time_str = now_utc.strftime("%Y-%m-%d %H:%M UTC")

    intermarket_block = _build_intermarket_section(intermarket) if intermarket else ""

    # Current price context — critical for avoiding stale zone mapping
    price_block = ""
    if current_price:
        price_block = f"📍 Current XAU/USD price: ${current_price:.2f}\n\n"

    recap_block = ""
    if session_recap:
        recap_block = f"""PREVIOUS SESSION RECAP:
{session_recap.get('narrative_summary', session_recap.get('macro_narrative', 'No recap'))}
Carry forward: {session_recap.get('carry_forward', session_recap.get('session_outlook', 'None'))}

"""

    # Prior session trend context — helps Opus gauge continuation vs reversal
    trend_block = ""
    if prior_session_stats:
        ps = prior_session_stats
        kz = ps.get("killzone", "Prior session")
        wins = ps.get("wins", 0)
        losses = ps.get("losses", 0)
        total = wins + losses
        dominant_dir = ps.get("dominant_direction", "")
        price_move = ps.get("price_move", 0)
        if total >= 2:
            trend_block = f"""PRIOR SESSION PERFORMANCE ({kz}):
{wins}W/{losses}L — dominant direction was {dominant_dir}. Price moved {'+' if price_move > 0 else ''}{price_move:.1f} points.
{'⚠️ Structure may be extended — consider continuation setups as well as reversals.' if losses >= 3 and losses > wins else ''}

"""

    return f"""Mark the chart for the upcoming {upcoming_killzone} killzone.
Time: {time_str}. Session opens soon.

{price_block}4H CANDLES (last 20):
{json.dumps(h4_slim)}

1H CANDLES (last 50):
{json.dumps(h1_slim)}

{intermarket_block}{recap_block}{trend_block}Identify ALL key zones and conditional setups for this session:

1. Mark all buy-side liquidity (recent swing highs that haven't been swept)
2. Mark all sell-side liquidity (recent swing lows that haven't been swept)
3. Identify any untested order blocks (from displacement candles)
4. Identify any unfilled fair value gaps
5. Define the current dealing range (4H swing high to swing low)
6. Calculate the premium/discount line (midpoint)
7. Create conditional setups: IF price does X at zone Y, THEN entry Z with SL/TPs
   - Maximum 4 conditional setups
   - Each must have a clear trigger, entry zone, invalidation, and targets
   - Include both directions if structure supports it

Return ONLY valid JSON:
{{
  "buyside_liquidity": [{{"price": number, "strength": "strong|weak"}}],
  "sellside_liquidity": [{{"price": number, "strength": "strong|weak"}}],
  "order_blocks": [{{"high": number, "low": number, "tf": "1H|4H", "type": "bull|bear"}}],
  "fvgs": [{{"high": number, "low": number, "tf": "1H|4H", "filled": boolean}}],
  "dealing_range": {{"high": number, "low": number}},
  "premium_discount_line": number,
  "conditional_setups": [
    {{
      "id": "setup_1",
      "bias": "bullish|bearish",
      "trigger_condition": "string — describe the IF condition",
      "entry_zone": {{"high": number, "low": number}},
      "invalidation": number,
      "preliminary_sl": number,
      "preliminary_tps": [number, number, number],
      "confidence": "high|medium|low"
    }}
  ]
}}"""


def build_displacement_check_prompt(prospect_setup: dict, recent_candles: list,
                                      current_price: float) -> str:
    """Build prompt to check if displacement has occurred at a prospect zone.

    This runs when price reaches a prospect's trigger level. It confirms
    the sweep + displacement before moving to retrace monitoring.
    """
    slim = _slim_candles(recent_candles[-8:])

    return f"""Check if a liquidity sweep + displacement has occurred at this level.

PRE-IDENTIFIED SETUP:
Bias: {prospect_setup.get('bias', '?')}
Trigger: {prospect_setup.get('trigger_condition', '?')}
Entry Zone (OB): {prospect_setup.get('entry_zone', {}).get('high', '?')} - {prospect_setup.get('entry_zone', {}).get('low', '?')}
Invalidation: {prospect_setup.get('invalidation', '?')}

CURRENT PRICE: ${current_price:.2f}

LAST 8 CANDLES:
{json.dumps(slim)}

Check:
1. Was liquidity actually SWEPT? (price went beyond the level then reversed — not just touched)
2. Is there DISPLACEMENT? (a strong impulsive candle body in the expected direction, not just a wick)
3. Identify the ORDER BLOCK created by the displacement candle (its OHLC range)
4. Identify any FVG created during the displacement move

Return ONLY valid JSON:
{{
  "displacement_confirmed": true|false,
  "sweep_confirmed": true|false,
  "displacement_candle": {{"open": number, "high": number, "low": number, "close": number}} | null,
  "ob_zone": {{"high": number, "low": number}} | null,
  "fvg_zone": {{"high": number, "low": number}} | null,
  "sweep_level": number | null,
  "reason": "string — max 100 chars"
}}"""


def build_retrace_confirmation_prompt(displacement_data: dict, prospect_setup: dict,
                                        recent_candles: list, ltf_candles: list | None,
                                        current_price: float) -> str:
    """Build prompt for entry confirmation when price retraces into the OB/FVG zone.

    This is the highest-conviction confirmation:
    sweep (✓) + displacement (✓) + retracement (✓) + now checking OB holds.
    """
    slim = _slim_candles(recent_candles[-5:])
    ltf_slim = _slim_candles(ltf_candles[-10:]) if ltf_candles else []

    ob = displacement_data.get("ob_zone", {})
    fvg = displacement_data.get("fvg_zone", {})

    return f"""Price has RETRACED into the OB zone after sweep + displacement. Confirm entry.

SETUP CONTEXT:
Bias: {prospect_setup.get('bias', '?')}
Displacement candle: {json.dumps(displacement_data.get('displacement_candle', {}))}
OB Zone: {ob.get('high', '?')} - {ob.get('low', '?')}
{"FVG Zone: " + str(fvg.get('high', '?')) + " - " + str(fvg.get('low', '?')) if fvg else "No FVG"}
Sweep level: {displacement_data.get('sweep_level', '?')}
Preliminary SL: {prospect_setup.get('preliminary_sl', '?')}
Preliminary TPs: {prospect_setup.get('preliminary_tps', [])}

CURRENT PRICE: ${current_price:.2f}

EXECUTION TIMEFRAME CANDLES:
{json.dumps(slim)}

{"LOWER TIMEFRAME (5M/15M) CANDLES:" + chr(10) + json.dumps(ltf_slim) + chr(10) if ltf_slim else ""}
Verify:
1. Is the OB HOLDING? (price entering zone but showing rejection — wicks, dojis, engulfing)
2. Is there a lower-timeframe entry signal? (LTF market structure shift, engulfing candle inside OB)
3. Is market structure still supporting the direction? (no new BOS against the trade)
4. EXACT ENTRY: where within the OB is the best entry?
5. TIGHT SL: if LTF data available, can SL be placed below the LTF swing low inside the OB (tighter)?

Return ONLY valid JSON:
{{
  "confirmed": true|false,
  "entry": number,
  "sl": number,
  "tps": [number, number, number],
  "sl_type": "below_sweep|ob_low|ltf_swing",
  "ltf_signal": "string — what LTF confirmation exists" | null,
  "reason": "string — max 100 chars"
}}"""


def build_trigger_confirmation_prompt(prospect_setup: dict, recent_candles: list,
                                       current_price: float) -> str:
    """Legacy trigger confirmation — used when retrace system is bypassed."""
    slim = _slim_candles(recent_candles[-5:])

    return f"""A pre-identified ICT setup has triggered. Confirm whether this is a valid entry.

PRE-IDENTIFIED SETUP:
Bias: {prospect_setup.get('bias', '?')}
Trigger: {prospect_setup.get('trigger_condition', '?')}
Entry Zone: {prospect_setup.get('entry_zone', {}).get('high', '?')} - {prospect_setup.get('entry_zone', {}).get('low', '?')}
Invalidation: {prospect_setup.get('invalidation', '?')}
Preliminary SL: {prospect_setup.get('preliminary_sl', '?')}
Preliminary TPs: {prospect_setup.get('preliminary_tps', [])}

CURRENT PRICE: ${current_price:.2f}

LAST 5 CANDLES:
{json.dumps(slim)}

Verify:
1. Has the trigger condition actually been met? (liquidity swept, price in zone)
2. Is there displacement confirming direction? (strong candle body, not just a wick)
3. Is market structure supporting the entry? (BOS/CHoCH in the right direction)
4. Are there any reasons NOT to enter? (news, conflicting structure, false sweep)

Return ONLY valid JSON:
{{
  "confirmed": true|false,
  "entry": number|null,
  "sl": number|null,
  "tps": [number]|null,
  "reason": "string — max 100 chars explaining confirmation or rejection",
  "adjustments": "string — any modifications to the pre-identified levels" | null
}}"""


OPUS_VALIDATION_SYSTEM = """You are a SENIOR ICT analyst reviewing a junior analyst's XAU/USD trade setup. You are skeptical by default and looking for reasons NOT to take this trade.

Your job is to protect capital. A missed trade costs nothing. A bad trade costs real money. Be critical.

Review criteria:
1. NARRATIVE COHERENCE — Is the accumulation/manipulation/distribution story actually supported by price action, or is it cherry-picked?
2. ORDER BLOCK VALIDITY — Was the OB born from genuine displacement (strong impulsive move)? Or is it just a random candle the analyst labeled as an OB?
3. LIQUIDITY SWEEP — Did a sweep actually occur, or is a wick just touching noise levels?
4. ENTRY QUALITY — Is the entry at a pullback/rejection into the zone? Or is it chasing a displacement candle?
5. INTERMARKET ALIGNMENT — If macro data is available, does DXY/yield context support the direction?
6. RISK:REWARD — Are the TP levels realistic given current market structure?

You respond ONLY with valid JSON. No explanation text outside the JSON."""


def build_validation_prompt(analysis_json: dict, candles: list,
                            htf_candles: list, intermarket: dict | None = None,
                            timeframe: str = "1h") -> str:
    """Build prompt for Opus validation of a Sonnet-detected setup.

    Args:
        analysis_json: Sonnet's full analysis JSON
        candles: Primary timeframe candles (same as Sonnet received)
        htf_candles: Higher timeframe candles
        intermarket: Intermarket context dict (or None)
        timeframe: Timeframe being analyzed

    Returns:
        Complete validation prompt string
    """
    h1_slim = _slim_candles(candles[-40:])  # less candles, Opus doesn't need all 60
    h4_slim = _slim_candles(htf_candles[-12:]) if htf_candles else []

    now_utc = datetime.now(timezone.utc)
    current_kz = get_current_killzone()

    intermarket_block = _build_intermarket_section(intermarket) if intermarket else ""

    # Strip verbose fields from the analysis to reduce tokens
    analysis_summary = {
        k: v for k, v in analysis_json.items()
        if k not in ("warnings",)  # keep everything relevant
    }

    return f"""SENIOR REVIEW: A junior ICT analyst has identified a {timeframe} XAU/USD setup at {now_utc.strftime("%H:%M UTC")} ({current_kz} killzone).

JUNIOR ANALYST'S COMPLETE ANALYSIS:
{json.dumps(analysis_summary, indent=2)}

CANDLE DATA THE ANALYST USED:
4H context: {json.dumps(h4_slim)}
{timeframe} candles: {json.dumps(h1_slim)}

{intermarket_block}TIMEFRAME CONTEXT: This is a {timeframe} setup. {"Short timeframe — displacement can be noise. Be extra strict on OB validity and entry quality." if timeframe in ("1min", "5min", "15min") else "Higher timeframe — structure signals are more reliable, but entry timing matters more."}

REVIEW THIS SETUP CRITICALLY. Consider:
- Is the market narrative (accumulation/manipulation/distribution) actually coherent with the candle data?
- Is the identified order block genuinely born from displacement, or is it a random candle?
- Did the claimed liquidity sweep actually happen, or is it just wick noise?
- Is the entry at a pullback/rejection or is it chasing the move?
- Does the intermarket context (if available) support this direction?
- Is the risk:reward realistic given the structure?

Return ONLY valid JSON:
{{
  "verdict": "validated|downgraded|rejected",
  "adjusted_quality": "A|B|C|D|no_trade",
  "validation_note": "string max 200 chars explaining your reasoning",
  "narrative_coherence": "strong|moderate|weak",
  "confidence_adjustment": number between -0.3 and +0.1
}}

Rules:
- "validated" = setup is solid, keep current quality grade
- "downgraded" = setup has merit but quality is overstated, lower by one letter
- "rejected" = setup is not tradeable, set to no_trade
- confidence_adjustment: negative = reduce ML confidence, positive = boost it
- Be honest. If the setup is genuinely strong, say so. Don't reject for the sake of it."""


def _slim_candles(candles: list) -> list:
    """Reduce candle data to essential fields for the prompt."""
    slim = []
    for c in candles:
        slim.append({
            "dt": c.get("datetime", ""),
            "o": round(float(c.get("open", 0)), 2),
            "h": round(float(c.get("high", 0)), 2),
            "l": round(float(c.get("low", 0)), 2),
            "c": round(float(c.get("close", 0)), 2),
        })
    return slim
