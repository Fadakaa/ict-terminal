# Unified Scanner Architecture — v3

**Status:** Proposed
**Date:** 2026-04-01
**Author:** Michael + Claude

---

## Glossary

**Haiku / Sonnet / Opus** — Three Claude model tiers used in the pipeline. Haiku is the cheapest and fastest (~$0.001/call), used as a pre-screen gate. Sonnet is the primary analyst (~$0.03/call), makes trade decisions. Opus is the most capable (~$0.05-0.10/call), handles macro strategy and final validation. All three are language models doing ICT chart analysis, not traditional ML classifiers.

**ICT Methodology** — Inner Circle Trader concepts that define how the system reads gold (XAU/USD) price action. Key terms: Order Blocks (OB) = supply/demand zones, Fair Value Gaps (FVG) = price imbalances, Liquidity sweeps (BSL/SSL) = stop hunts, Break of Structure (BOS) / Change of Character (CHoCH) = trend shifts, Power of 3 (P3) = accumulation → manipulation → distribution cycle, Premium/Discount = above/below equilibrium.

**Narrative Engine** (`narrative_state.py`) — A SQLite-backed state store that persists Sonnet's thesis per timeframe between scan cycles. It is NOT an AI model — it is a persistence layer with invalidation checking and confidence decay logic. Sonnet writes the thesis; the narrative engine stores, tracks, and validates it.

**ML Stack** — The collective statistical memory of the system. Claude is stateless (every API call starts fresh), so ML components persist what Claude cannot: outcome distributions, pattern win rates, regime survival rates, thesis conversion rates, and calibration layer performance. In the unified architecture, ML's role shifts from overriding Claude's decisions to giving Claude access to its own track record before it makes decisions. Components: volatility classifier, Bayesian beliefs, AutoGluon models, setup DNA profiles, intermarket validator, entry placement analyzer, narrative tracker, layer performance tracker.

**AutoGluon** — The ML framework storing trained models on disk. Classifier predicts win/loss (or multi-3: stopped/tp1/runner) from 52 features. Quantile regressor predicts optimal SL/TP bands. In the unified architecture, AutoGluon's predictions are surfaced in Sonnet's prompt as statistical context ("for this feature pattern, predicted SL band is 3.8-5.1 ATR") rather than applied as a post-hoc override. Active after 30+ resolved trades.

**Setup DNA** (`setup_dna.py` + `setup_profiles.py`) — Encodes each setup as a pattern string (direction × killzone × timeframe × ICT elements present). Tracks historical win rate per DNA pattern. Used to upgrade/downgrade Sonnet's grade: >70% WR with 15+ matches → upgrade, <35% WR → downgrade.

**Killzone** — Time-of-day trading windows (UTC): Asian (00:00-06:59), London (07:00-11:59), NY_AM (12:00-15:59), NY_PM (16:00-19:59). Different killzones have different quality profiles.

---

## Problem Statement

The scanner currently runs two separate paths that evolved independently:

**Standard Path** (scan_all_timeframes → Haiku screen → Sonnet analysis → proximity monitor → entry alert) operates on a fixed timer. Every 15 minutes to 1 day, it checks each timeframe, asks Haiku "is anything here?", and if Haiku says yes, escalates to Sonnet for a full analysis.

**Prospect Path** (pre-killzone zone ID → sweep monitoring → displacement confirmation → retrace entry) activates 15 minutes before each killzone. Opus identifies zones of interest, then a separate trigger monitor polls 5-minute candles every 90 seconds watching for displacement.

The two paths produce different notification types, use different monitoring loops, and don't share context. A setup detected by the standard path doesn't inform the prospect path, and vice versa.

### Core Issues

**1. Haiku is screening blind.** Haiku receives raw candles and has to answer "is there a setup forming?" with no context about what the narrative engine thinks, what Opus's macro bias is, or what happened in the previous scan. This is like asking someone to spot a pattern without telling them what pattern to look for. The result: high false-negative rates on segments where structure is subtle but the narrative context would make it obvious. The FN tracker (P5) is a band-aid — it learns where Haiku is blind and bypasses it, but the root cause is that Haiku lacks the context to screen intelligently.

**2. Too few setups are identified.** The system currently misses valid setups, especially on lower timeframes. This is partly the Haiku screening problem, partly because C/D grade setups are silently dropped instead of being monitored for confirmation, and partly because the two paths don't feed each other — a zone identified by the prospect path can't promote a borderline standard-path analysis.

**3. Two monitoring loops with no shared state.** `monitor_pending()` watches standard-path setups on 5-minute candles. `monitor_prospect_triggers()` watches prospect zones on a separate 90-second loop. Neither knows about the other. A prospect displacement that confirms a pending standard setup doesn't trigger anything.

**4. Notifications are fragmented.** A single thesis-to-trade journey can produce up to 6 notification types across both paths. The user receives 🔍 SETUP DETECTED, then NEW Setup, then possibly 🎯 ENTER from the prospect path for the same underlying move. Two notifications maximum is the goal: thesis forming → entry signal.

**5. Claude is stateless but the system has memory it can't access.** Every Sonnet call is a blank slate — it can read 120 candles and apply ICT theory, but it has zero memory of what happened in the last 500 trades. Meanwhile, the ML stack holds exactly that memory: Bayesian beliefs know the live win rate, setup DNA knows which patterns win, AutoGluon knows which features predict outcomes, the volatility classifier knows what regime we're in. But this memory is used to *overrule* Claude after the fact (ML produces competing SL/TP, consensus picks widest) instead of *informing* Claude before it decides. The result: Claude makes a decision blind, then ML second-guesses it, then a dumb arbiter picks between them. The memory should flow into the prompt so Claude makes better decisions in the first place.

---

## Design Principles

1. **Narrative engine is the brain.** Every scan decision starts with "what does the narrative engine say about this timeframe right now?" Not "are there candles that look like a pattern?"

2. **Context flows downhill.** Opus sets macro direction → narrative engine maintains per-timeframe thesis → Haiku screens with thesis context → Sonnet analyzes with full context → ML calibrates the numbers. No component operates in isolation.

3. **Nothing is silently dropped.** C/D grade setups enter a monitoring queue. Displacement or additional confluence can promote them. Missed setups feed back through the narrative engine for the next scan cycle.

4. **One monitoring loop.** A single unified monitor handles all pending setups and prospect zones, with different poll intervals based on urgency.

5. **Two notifications per setup lifecycle, maximum.** Thesis forming (awareness) → entry signal (action). Everything else is logged but not pushed to the user.

6. **ML is the system's memory — Claude is stateless.** Every Claude API call starts from zero. Sonnet cannot remember that the last 30 accumulation setups in London needed 4.5 ATR stops to survive. Opus cannot remember that its manipulation phase calls are 80% accurate but its accumulation calls are only 55%. AutoGluon, Bayesian beliefs, setup DNA profiles, and the outcome distributions ARE that memory — compressed into model weights, probability distributions, and pattern stores that persist on disk. ML's role isn't to produce competing opinions. It's to recall what Claude literally cannot: outcome distributions, historical drawdown profiles, regime-specific survival rates, and pattern match statistics. ML serves the narrative by giving Claude access to its own track record.

---

## Architecture

### The Unified Scan Cycle

```
SCHEDULER fires (per-timeframe timer)
    │
    ▼
┌─────────────────────────────────────────────────┐
│  NARRATIVE ENGINE — "What's the current thesis?" │
│                                                   │
│  • Fetch previous thesis for this timeframe       │
│  • Check invalidation against current price       │
│  • Detect structural contradictions               │
│  • Apply confidence decay if thesis stale         │
│  • Carry state from previous killzone             │
│                                                   │
│  Output: thesis context (or "no thesis — first    │
│          scan on this TF")                        │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  OPUS STRATEGY CALL — unified macro + zones      │
│                                                   │
│  Single call replaces the old HTF narrative +     │
│  prospect zone ID. Opus receives:                 │
│  • 4H/Daily/Weekly candles                        │
│  • Intermarket context (DXY, US10Y)               │
│  • Current narrative thesis (from above)           │
│                                                   │
│  Returns:                                         │
│  • directional_bias + confidence                  │
│  • P3 phase + progress                            │
│  • key_levels (premium/discount/equilibrium)      │
│  • watch_zones (specific OB/FVG/liquidity zones   │
│    to monitor — replaces separate prospect call)  │
│  • macro_narrative (free text)                    │
│                                                   │
│  Cached for 1 hour (shared across TF scans)       │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  CONTEXT-AWARE HAIKU SCREEN                      │
│                                                   │
│  OLD: "Here are candles. Is there a setup?"       │
│  NEW: "Here are candles. The thesis is bullish    │
│       accumulation. Opus says watch 2340-2345     │
│       OB zone. Is price approaching, testing,     │
│       or reacting to any of these zones?"         │
│                                                   │
│  Haiku now knows WHAT to look for, not just       │
│  whether something generic exists.                │
│                                                   │
│  Inputs:                                          │
│  • Execution-TF candles (same window as today)    │
│  • HTF candles                                    │
│  • Narrative thesis summary (1-2 lines)           │
│  • Opus watch_zones (specific levels)             │
│  • Active pending setups on this TF               │
│                                                   │
│  Returns:                                         │
│  • setup_possible: true/false                     │
│  • zone_interaction: which watch_zone, if any     │
│  • direction hint                                 │
│  • reason                                         │
│                                                   │
│  FN tracker still active as safety net.           │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  ML ENRICHMENT — "What does the data say?"       │
│                                                   │
│  Runs BEFORE Sonnet, not after. Builds a          │
│  statistical context block for the prompt:        │
│                                                   │
│  • Setup DNA match: WR + avg R:R for this thesis  │
│    type × killzone × timeframe pattern            │
│  • Regime classification: current volatility      │
│    state + recommended SL floor                   │
│  • Bayesian beliefs: live win rate + trend         │
│  • MAE percentiles: expected drawdown before move  │
│  • AutoGluon quantile: predicted SL/TP bands      │
│    (if model trained)                              │
│  • Intermarket signal quality: validated/noise     │
│                                                   │
│  Output: ML context block injected into Sonnet's  │
│  prompt so it makes informed SL/TP decisions       │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  SONNET ANALYSIS — "Full trade decision"         │
│                                                   │
│  Now has complete context from ALL sources:        │
│  • Previous thesis + invalidation status          │
│  • Opus strategy (bias, zones, narrative)         │
│  • Haiku's zone_interaction hint                  │
│  • ★ ML context (regime, SL floor, DNA WR, MAE)  │
│  • Intermarket context                            │
│  • Recent context (resolutions, consumed zones)   │
│  • Learned rules + placement guidance             │
│                                                   │
│  Sonnet sets SL/TP WITH statistical guidance —    │
│  not blind, not overridden after the fact.         │
│                                                   │
│  Returns: analysis JSON + updated narrative_state │
│  Grade: A / B / C / D                             │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  ML SAFETY NET — post-analysis guard rail        │
│                                                   │
│  Consensus layer still runs as a catch:           │
│  If Sonnet's SL is below the volatility floor,    │
│  widen it. But this should rarely fire because     │
│  Sonnet already saw the floor in its prompt.       │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  GRADE-BASED ROUTING                             │
│                                                   │
│  A/B grade:                                       │
│    → Setup DNA check (can upgrade/downgrade grade  │
│      based on historical WR of matching patterns)  │
│    → Opus validation (go/no-go for A/B only)       │
│    → Killzone quality gate (learned per-KZ bar)    │
│    → Store + notify (if passes all gates)          │
│                                                   │
│  C/D grade:                                       │
│    → Store as "monitoring" status                  │
│    → Added to unified monitor queue                │
│    → If displacement confirms within window:       │
│      promote to B, run ML enrichment + Sonnet      │
│      re-evaluation, Opus validation, notify        │
│    → If no confirmation within expiry: resolve     │
│      as expired, feed back to narrative engine      │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  UNIFIED MONITOR                                 │
│                                                   │
│  Single polling loop replaces both                │
│  monitor_pending() and monitor_prospect_triggers()│
│                                                   │
│  Watches:                                         │
│  • A/B setups: entry proximity + SL/TP hit        │
│    (poll every 60s when price within 0.3%)        │
│  • C/D setups: displacement confirmation           │
│    (poll every 90s, promote if confirmed)          │
│  • Opus watch zones: sweep + displacement          │
│    (poll every 90s during killzone)                │
│                                                   │
│  All share the same 5-min candle fetches.          │
│  Zone hits feed back to pending setup checks.      │
│                                                   │
│  On resolution → full feedback cascade             │
│  (12 feedback loops, same as today)                │
└─────────────────────────────────────────────────┘
```

### Narrative Engine Specifics

The narrative engine (`narrative_state.py`) is a state store, not an AI. Here's exactly what it does at each step:

**Per-timeframe thesis persistence:** Each thesis record stores: `thesis` (free text), `directional_bias` (bullish/bearish/neutral), `bias_confidence` (0.0-1.0), `p3_phase` (accumulation/manipulation/distribution), `p3_progress` (early/mid/late), `key_levels` (JSON array), `expected_next_move`, `invalidation` (price level that kills the thesis), `watching_for` (specific triggers), `scan_count` (scans survived), `is_revision` (did bias flip from previous).

**Invalidation check** (`check_invalidation()`): Before the prompt is built, compares current price against the thesis's invalidation level. If price has breached it, the invalidation_status is set to "INVALIDATED" and injected into the prompt so Sonnet is told explicitly.

**Structural contradiction detection:** Computes `compute_market_structure(candles, lookback=20)` which returns a score from -1 (strongly bearish) to +1 (strongly bullish). If thesis says bullish but score < -0.2, or thesis says bearish but score > +0.2, a contradiction warning is injected into the prompt.

**Confidence decay:** If the thesis's `expected_next_move` hasn't happened, confidence decays by 15% per scan cycle (`CONFIDENCE_DECAY_RATE = 0.15`). A thesis at 80% confidence drops to 68% → 58% → 49% over three scans if the expected move doesn't materialize. Below 50%, it stops triggering THESIS_ACTIVE notifications.

**Stale thesis expiry:** Per-timeframe TTLs: 15min → 2 hours, 1H → 8 hours, 4H → 24 hours, 1D → 72 hours. Expired theses are purged before each scan via `expire_stale()`.

**Killzone handoff (NEW):** A `killzone_summary` field is added to each thesis. When `_check_killzone_transition()` detects a killzone change, the outgoing killzone's summary is injected into the next scan prompt. Example: "London saw bullish displacement from 2340 OB, swept SSL at 2335. Thesis: distribution setup now possible in NY. Watch for BSL sweep above 2358." This gives Sonnet session-to-session continuity instead of cold-reading each killzone.

### ML as the System's Statistical Memory

This is the fundamental shift from the current architecture. Claude is stateless — every API call starts fresh. It can read 120 candles and apply ICT theory, but it has zero memory of what happened in the last 500 trades. It doesn't know that accumulation setups in London have a 72% win rate. It doesn't know that SLs under 4 ATR get stopped 65% of the time in elevated volatility. It doesn't know that its own Opus macro calls are 80% accurate on distribution but only 55% on accumulation.

The ML stack knows all of this. It's been learning from every resolved trade. But right now, that knowledge is used to *overrule* Claude after the fact — ML produces competing SL/TP levels and consensus picks the widest. Claude never sees what ML knows.

**The fix: give Claude its own memory back.**

```
CURRENT:  Sonnet decides blind  →  ML overrides  →  Consensus arbitrates
UNIFIED:  ML recalls history    →  Sonnet decides informed  →  Safety net catches rare misses
```

**What ML remembers that Claude cannot:**

| Memory Type | Stored In | What It Holds | Example |
|-------------|-----------|---------------|---------|
| **Outcome distributions** | Bayesian beliefs (`bayesian.py`) | Win rate as Beta distribution, updated after every trade. Drawdown distributions per session. | Beta(12.3, 13.2) → 48.2% WR, trending +3.6pp from V1 prior |
| **Regime-specific survival rates** | Volatility classifier (`volatility.py`) + V1 stats (`seed.py`) | How wide SL needs to be per regime to survive noise. 5-state regime classification with historical MAE percentiles. | "Elevated regime: 80th percentile MAE is 4.2 ATR. SLs under 4.0 ATR stopped 65% of the time." |
| **Pattern win rates** | Setup DNA profiles (`setup_profiles.py`) | Win rate, avg R:R, avg MFE/MAE for every combination of direction × killzone × TF × ICT elements. 15+ matches needed. | "bullish × London × 1H × OB+FVG+sweep: 72% WR across 18 matches, avg R:R 2.8:1" |
| **Thesis-type conversion rates** | AutoGluon classifier (`training.py`) | Learned from 52 features including thesis_confidence, p3_progress, scan_count. Which thesis shapes actually convert to winning trades. | "Accumulation theses surviving 3+ scans in London convert at 78%" |
| **SL/TP band predictions** | AutoGluon quantile regressor (`training.py`) | Predicted optimal SL/TP bands based on the full feature vector — regime, killzone, ICT elements, narrative state. | "For this feature pattern, predicted SL band: 3.8-5.1 ATR, predicted TP1: 2.2-3.4 R:R" |
| **Intermarket signal reliability** | Intermarket validator (`intermarket_validator.py`) | Which DXY/US10Y signals actually correlated with gold outcomes per killzone. | "DXY divergence during London: signal quality STRONG (72% predictive). During Asian: NOISE (51%)." |
| **Entry placement performance** | Entry placement analyzer (`entry_placement.py`) | Where in the OB/FVG zone entries perform best, measured by MFE/MAE. | "OB midpoint entries outperform edge entries by +0.4 R:R in this segment" |
| **Opus accuracy by narrative type** | Narrative tracker (`claude_bridge.py`) | Track record of Opus's macro calls broken down by call type and killzone. | "Opus distribution calls: 80% accurate. Accumulation calls: 55%. Weight accordingly." |
| **Prompt variant performance** | Narrative bandit (`narrative_bandit.py`) | Which prompt wordings produce better trade outcomes via Thompson sampling. | "Variant B (structured levels) outperforms variant A (narrative flow) by +8% WR" |
| **Haiku blind spots** | FN tracker (`haiku_fn_tracker.py`) | Segments where Haiku's screening misses real setups. | "15min × London: 62% FN rate — bypass Haiku, go straight to Sonnet" |
| **Calibration layer value** | Layer performance (`layer_performance.py`) | Which of the 6 calibration layers actually add predictive value vs noise. | "Volatility layer: +3.2% WR. Historical matching: +0.1% (essentially noise)." |

**None of this exists in Claude's context window.** Every Sonnet call is a blank slate that can read candles but has never seen a resolved trade. ML is the bridge — it compresses 500+ trade outcomes into actionable context that fits in a prompt section.

**What Sonnet sees in its prompt (new section):**

```
=== YOUR STATISTICAL MEMORY ===
Pattern match: bullish accumulation × London × 1H
  Your track record: 72% WR across 18 similar setups, avg R:R 2.8:1
  Opus accuracy on accumulation calls: 55% (weight HTF bias cautiously)

Regime: ELEVATED (vol_ratio 1.4x normal)
  Required SL floor: 4.0 ATR (below this, 65% get stopped out)
  Expected drawdown before move: 4.2 ATR (80th percentile MAE)
  Best entry position: OB midpoint (outperforms edge by +0.4 R:R)

Overall win rate: 48.2% (trending up +3.6pp over last 25 trades)
  AutoGluon SL band for this pattern: 3.8-5.1 ATR
  AutoGluon TP1 band: 2.2-3.4 R:R

Intermarket: DXY falling, supports gold long
  Signal quality for London: STRONG (72% predictive in this segment)
```

This isn't ML telling Sonnet what to do — it's Sonnet reading its own history. "You've been here before. Here's what happened." Sonnet still makes the decision, but now it has context that would otherwise require remembering hundreds of past trades.

**What remains as post-analysis safety net:** The consensus layer still runs after Sonnet as a guard rail. If Sonnet ignores the SL floor and sets SL at 2.0 ATR in an elevated regime, the floor catches it. But this should rarely fire (<10% of setups) because Sonnet already saw the floor in its prompt.

**How each ML component shifts role:**

| Component | Current Role | New Role |
|-----------|-------------|----------|
| Volatility classifier | Produces competing SL after Sonnet | Provides regime + SL floor + MAE percentiles BEFORE Sonnet |
| V1 Session Stats | Produces competing SL after Sonnet | Provides session-specific drawdown distributions BEFORE Sonnet |
| Bayesian beliefs | Adjusts confidence after Sonnet | Provides live win rate + drift trend BEFORE Sonnet |
| AutoGluon classifier | Predicts win/loss after Sonnet | Provides thesis-type conversion rates + SL/TP bands BEFORE Sonnet |
| AutoGluon quantile | Produces competing SL/TP bands | Provides predicted optimal bands BEFORE Sonnet |
| Historical matching | Produces competing SL/TP | Merged into Setup DNA (pattern match stats) |
| Setup DNA | Upgrades/downgrades grade after Sonnet | Provides pattern WR + avg R:R BEFORE Sonnet |
| Consensus | Blends competing opinions | Safety net — catches the rare case where Sonnet ignores ML floor |

The calibration engine doesn't disappear — it restructures. `calibrate_trade()` still runs post-Sonnet as a guard rail, but most of its value has already been delivered via prompt enrichment. Over time, as Sonnet consistently uses the ML context, the safety net becomes a no-op.

**How AutoGluon training evolves:**

AutoGluon currently trains on 52 features to predict win/loss. With ML serving the narrative, AutoGluon also learns thesis-type patterns: `thesis_type` (accumulation/manipulation/distribution × direction), `thesis_scan_count`, `thesis_confidence`, `killzone_at_thesis_start`. This lets it answer questions like "what's the conversion rate for accumulation theses that survive 3+ scans in London?" — exactly the kind of insight that feeds back into the narrative engine's confidence scoring and into Sonnet's prompt.

The 4 narrative features we built (thesis_confidence, p3_progress_encoded, thesis_scan_count, opus_sonnet_agreement) become the most important bridge between narrative state and statistical memory. They let AutoGluon learn that a high-confidence thesis in its 4th scan during London with Opus agreement is qualitatively different from a fresh thesis in Asian with Opus disagreement — even when the candle patterns look similar.

### Opus: Two Calls, Not Three

The old architecture had up to three Opus calls per scan cycle: HTF narrative, prospect zone ID, and setup validation. The unified architecture collapses to two:

**Call 1 — Strategy** (cached 1 hour): "What's the macro picture and where should we watch?" Replaces both the HTF narrative and prospect zone ID. Opus returns directional bias, P3 phase, key levels, AND specific watch zones. One call, one context window, better coherence.

**Call 2 — Validation** (A/B grades only): "Here's what Sonnet found. Should we trade it?" Unchanged — this must remain separate because it needs to see the completed analysis.

### Context-Aware Haiku Screen

The biggest change. Today's screen prompt:

```
XAU/USD ICT screen — 1h candles (72 candles).
[raw candle data]
Is there an ICT setup forming? Answer YES if ANY of these are present...
```

New screen prompt:

```
XAU/USD ICT screen — 1h candles (72 candles).
Current thesis: Bullish accumulation, 75% confidence, 3rd scan.
Opus watch zones: 2340-2345 bullish OB (untested), 2358 BSL (unswept).
Active pending: 1 long setup at 2342, monitoring.

[candle data]

Is price interacting with any of these zones? Specifically:
- Is price approaching, testing, or reacting to the 2340-2345 OB?
- Has BSL at 2358 been swept?
- Is there a displacement move (3+ ATR candle body)?
- Any new structure shift that would change the thesis?

```

Haiku's primary job is thesis-guided zone checking. The "also flag any setup" catch-all from the old prompt is removed — if there's no thesis or watch zones, Haiku falls back to the current generic prompt (same as today). This focused approach should dramatically reduce false negatives on lower timeframes where the thesis provides the context Haiku needs.

**Fallback when no thesis exists:** If the narrative engine has no active thesis for this timeframe (first scan, or thesis expired), and Opus returned no watch zones (cache miss or Opus failure), Haiku reverts to the current generic screen prompt. This is the safe degradation path — the system is never worse than today.

### C/D Grade Promotion Pipeline

Today, C/D setups are either silently ignored (standard path) or never generated (prospect path only produces when Opus identifies high-conviction zones).

In the unified architecture:

1. Sonnet identifies a setup but grades it C or D (e.g., OB present but no sweep confirmation, FVG but structure unclear)
2. Setup is stored with `status="monitoring"` instead of being discarded
3. Unified monitor watches for displacement confirmation near the setup's entry zone
4. **Promotion criteria** (ALL must be met):
   - Price moves within 1.0 ATR of the setup's entry zone ("near the zone")
   - A single candle body ≥ 2.0 ATR in the setup's direction (displacement)
   - Displacement candle sweeps a visible liquidity level (BSL/SSL within 0.5 ATR)
5. On promotion: upgrade to B grade → run ML enrichment → Sonnet re-evaluation with ML context → Opus validation → store as pending → notify user
6. **Expiry windows** (same as existing EXPIRY_HOURS): 15min → 8h, 1H → 48h, 4H → 168h, 1D → 336h. If no confirmation within the window: resolve as expired, write to recent_context for narrative continuity

This catches the setups that Sonnet correctly identifies as marginal but that price action later confirms. Based on the existing FN data, an estimated 15-25% of currently missed setups fall into this category.

### Missed Setups Recycle Through Narrative

When a setup expires without entry (price never reached it) or is missed (entry passed while unmonitored), it's not just logged — it's fed back to the narrative engine:

- The expired/missed setup's zone is marked as "consumed" or "tested" in recent_context
- On the next scan, Sonnet sees: "Previous scan identified OB at 2342 but entry was never reached. Price instead swept BSL at 2335. Reassess whether the thesis still holds or if this was a failed accumulation."

This creates continuity. Instead of each scan being independent, the narrative engine ensures Sonnet processes what happened to its previous calls.

---

## Notification Simplification

### Current State (up to 6 types)

| Type | Path | When |
|------|------|------|
| 🔍 SETUP DETECTED | Standard | Setup identified, before monitoring |
| NEW [Dir] Setup [TF] | Standard | Full trade signal with levels |
| 🔍 ZONE ALERT | Prospect | Opus identified zone pre-killzone |
| ⚡ DISPLACEMENT CONFIRMED | Prospect | Sweep + displacement at zone |
| 🎯 ENTER | Prospect | Retrace entry after displacement |
| ⚠️ ENTRY MISSED | Both | Price passed entry without fill |

### Unified (2 types maximum)

| Type | When | Content |
|------|------|---------|
| **📊 THESIS ACTIVE** | Narrative engine has a confirmed thesis (survived 2+ scans, confidence ≥0.7) | Bias, key levels, what to watch. Awareness only — no action needed. |
| **🎯 ENTRY SIGNAL** | A/B grade setup (or promoted C/D) passes all gates | Direction, entry, SL, TPs, grade, calibration summary. This is the action notification. |

Everything else (thesis forming, displacement at zone, zone alerts) is logged to the Intelligence Panel but not pushed to Telegram/macOS. The user checks the panel when they want depth; notifications are only for "pay attention now."

**Risk warning (not counted toward the 2-notification limit):** ⚠️ THESIS_REVISED fires if the user has a pending setup AND the thesis flips direction. This is a warning to reconsider an open position, not a new trade signal. It fires rarely (only when bias_confidence was ≥0.5 and direction changes with a pending setup on that TF).

---

## Implementation Phases

### Phase 1: Context-Aware Haiku Screen (Highest Impact)
**Effort:** 2-3 hours
**Impact:** Directly addresses "too few setups" on lower timeframes
**Prerequisite:** Narrative engine already implemented and stable (confirmed: `narrative_state.py` is live with 842+ passing tests)

**File changes:**

**1. `prompts.py` — modify `build_screen_prompt()`**

Add parameters: `prev_narrative: dict | None = None`, `watch_zones: list | None = None`, `pending_setups: list | None = None`

New prompt template when thesis + zones exist:
```
XAU/USD ICT screen — {timeframe} candles ({len(slim)} candles). Time: {HH:MM UTC} ({killzone})

Thesis: {prev_narrative['directional_bias']} {prev_narrative['p3_phase']}, {prev_narrative['bias_confidence']:.0%} confidence, scan {prev_narrative['scan_count']}.
{f"Invalidation: {prev_narrative['invalidation']}" if invalidation else ""}
Watch zones: {', '.join(f"{z['level']} {z['type']} ({z['status']})" for z in watch_zones)}
{f"Active pending: {len(pending)} setup(s) on this TF" if pending else ""}

{timeframe} candles:
{json.dumps(slim)}
{htf section if applicable}

Given the thesis and watch zones above, is price:
- Approaching, testing, or reacting to any watch zone?
- Showing displacement (3+ ATR candle body) in the thesis direction?
- Showing structure shift that would CHANGE the thesis?

Reply ONLY valid JSON:
{"setup_possible": true|false, "zone_interaction": "zone description"|null, "direction": "long"|"short"|null, "reason": "one sentence"}
```

When no thesis/zones exist: fall back to current generic prompt (unchanged).

**2. `scanner.py` — modify `_analyze_and_store()`**

Move the Haiku screen call AFTER the narrative engine fetch and Opus strategy call (currently it's before Opus). Pass the new context:

```python
screen = self._call_claude_screen(
    candles, htf_candles or [], timeframe,
    prev_narrative=prev_narrative,
    watch_zones=(htf_narrative or {}).get("watch_zones"),
    pending_setups=[s for s in self.db.get_pending() if s.get("timeframe") == timeframe],
)
```

**3. `scanner.py` — modify `_call_claude_screen()`**

Add the three new parameters, pass them to `build_screen_prompt()`. Cache key must include a hash of watch_zones (so new Opus zones invalidate the cache).

**4. Response handling**

If Haiku returns `zone_interaction`, pass it into Sonnet's prompt as a hint (new field in `_call_claude()` kwargs). If `zone_interaction` is null and `setup_possible` is false, same exit as today. If `zone_interaction` is null but `setup_possible` is true, proceed to Sonnet as today.

**Edge cases:**
- Opus cache miss (no watch_zones): fall back to generic prompt
- Narrative engine empty (no thesis): fall back to generic prompt
- Haiku timeout: fall through to Sonnet (same as today)
- Multiple zone interactions: Haiku picks the most relevant one; Sonnet sees all zones anyway

**No changes to:** Sonnet prompt, ML calibration, notifications, feedback loops. Fully backward-compatible.

### Phase 2: ML Repositioning — Pre-Analysis Enrichment
**Effort:** 4-5 hours
**Impact:** Sonnet makes better SL/TP decisions upfront instead of being overridden after

Changes:
- New function `build_ml_context()` in `calibrate.py` — runs the 6 layers but returns a prompt-ready context block instead of competing SL/TP values
- Inputs: thesis type (from narrative engine), timeframe, killzone, candles (for regime/ATR), setup DNA pattern (if available from Opus watch zones)
- Output: dict with `sl_floor_atr`, `mae_percentile_80`, `regime`, `dna_win_rate`, `dna_avg_rr`, `bayesian_wr`, `intermarket_quality`
- New prompt section builder `_build_ml_context_section()` in `prompts.py` — formats the ML context for Sonnet's prompt
- `_analyze_and_store()` calls `build_ml_context()` BEFORE `_call_claude()` and passes the result as a new kwarg
- `build_enhanced_ict_prompt()` gains an `ml_context` parameter → renders the "ML CONTEXT FOR YOUR ANALYSIS" block
- Existing `_calibrate()` call remains as post-analysis safety net but should rarely override
- Track "safety net fired" rate as a metric — target: <10% of setups

### Phase 3: Opus Strategy Consolidation
**Effort:** 3-4 hours
**Impact:** Reduces API cost, improves coherence

Changes:
- New `build_opus_strategy_prompt()` that combines HTF narrative + zone identification
- Response schema includes `watch_zones` array alongside existing fields
- `_prospect_killzone_zones()` removed — zones come from the strategy call
- Cache key unchanged (still 1-hour TTL)
- Prospect path's separate Opus call eliminated

### Phase 4: C/D Grade Monitoring Pipeline
**Effort:** 4-5 hours
**Impact:** Catches 15-25% more valid setups

Changes:
- New `status="monitoring"` in scanner_db for C/D setups
- Unified monitor checks monitoring setups for displacement
- Promotion logic: displacement confirmed → ML enrichment → Sonnet re-evaluation → Opus validation → store as pending
- Expiry logic for monitoring setups (same windows as current EXPIRY_HOURS)
- New test coverage for promotion/expiry flow

### Phase 5: Unified Monitor Loop
**Effort:** 5-6 hours
**Impact:** Architectural cleanup, enables shared state

Changes:
- Merge `monitor_pending()` and `monitor_prospect_triggers()` into `unified_monitor()`
- Single candle-fetch loop shared across all monitored items
- Priority queue: A/B setups near entry > C/D watching for displacement > zone watches
- Prospect zone watches integrated as first-class monitor items
- Zone hits cross-reference pending setups

### Phase 6: Killzone Handoff + Notification Simplification
**Effort:** 3-4 hours
**Impact:** Better continuity, cleaner UX

Changes:
- `killzone_summary` field added to narrative_state schema
- `_check_killzone_transition()` triggers summary generation and injection into next scan
- Notification functions consolidated: keep `notify_thesis_active()` and `notify_entry_signal()`
- Remove or demote: SETUP_DETECTED, ZONE_ALERT, DISPLACEMENT_CONFIRMED to log-only
- Keep THESIS_REVISED as risk warning

### Phase 7: Missed Setup Recycling
**Effort:** 2-3 hours
**Impact:** Narrative continuity

Changes:
- Expired/missed setups written to recent_context with outcome
- Narrative engine prompt section includes "what happened to your previous setups"
- Sonnet explicitly asked to reassess thesis if previous setup failed

### Phase 8: ML Legacy Revocation
**Effort:** 3-4 hours
**Impact:** Removes redundant compute, enforces clean architecture
**Prerequisite:** Phase 2 (ML Repositioning) live for 2+ weeks with safety net fire rate confirmed <10%

This is not optional cleanup — it's mandatory. If the old parallel path stays alive alongside the new prompt-enrichment path, the system pays double compute for the same information, and worse, there's ambiguity about which path is authoritative. Once Sonnet is reliably using ML context from the prompt, the old override path must be revoked.

**What gets removed:**

| Component | Current Behaviour | After Revocation |
|-----------|------------------|------------------|
| `calibrate_trade()` competing SL/TP | Runs 6 layers, produces its own SL/TP, consensus picks widest | **Removed.** Replaced by `build_ml_context()` which produces prompt context only |
| `consensus.build_consensus()` | Blends AutoGluon, Bayesian, volatility into composite grade + SL/TP | **Reduced to floor check only.** Single function: "is Sonnet's SL below the volatility floor? If yes, widen to floor. Done." No grade, no competing TPs |
| `_calibrate()` in scanner.py | Full calibration pipeline post-Sonnet | **Replaced with `_apply_safety_floor()`** — one check, one possible override (SL floor), nothing else |
| `prediction.predict()` standalone | Full inference pipeline that recomputes features, runs AG, builds consensus | **Removed as a standalone endpoint.** AutoGluon inference folded into `build_ml_context()` |
| `_build_calibration()` in prediction.py | Constructs calibration context for consensus | **Removed.** Context now built pre-prompt by `build_ml_context()` |
| `_enrich_with_consensus()` | Adds consensus grade/confidence post-inference | **Removed.** Grade comes from Sonnet (informed by ML context), not from consensus blending |
| Calibrated SL/TP fields in scanner_db | `calibrated_sl`, `calibrated_tps` stored alongside Claude's values | **Removed.** Only one set of levels exists: Sonnet's (which already incorporates ML context). The `ml_floor_applied` boolean replaces `calibrated_sl` to track safety net interventions |
| `/calibrate` API endpoint | POST — runs full calibration for external callers | **Deprecated.** Replaced by `/ml-context` GET that returns the prompt context block. External callers who need calibration use the new endpoint and feed it to their own prompt |

**What stays:**

| Component | Why It Survives |
|-----------|----------------|
| `build_ml_context()` (new) | This IS the new ML path — pre-prompt enrichment |
| Volatility floor check | Single safety net: "SL ≥ regime floor?" Always needed |
| `bayesian.update_beliefs()` | Still updates after every trade — this is the memory being written to |
| `dataset.ingest_live_trade()` | Still ingests features — this is training data accumulation |
| AutoGluon `train_classifier()` | Still retrains periodically — the model needs to keep learning |
| Setup DNA profiling | Still encodes + matches patterns — this is memory too |
| All 12 feedback loops | All still fire on resolution — writing to the memory stores |
| `layer_performance.ingest_trade()` | **Repurposed.** Instead of tracking which calibration layer was closest, tracks "did Sonnet respect the ML context?" (new metric: ml_context_adherence) |

**Migration safety:**

Before revoking, run Phase 2 in "dual mode" for 2 weeks: both paths active, both logged, neither overriding the other. Compare:
- How often Sonnet's ML-informed SL differs from the old consensus SL
- Whether the old consensus would have saved a trade that Sonnet's SL didn't
- The safety net fire rate (target: <10%)

If the safety net fires >10% after 2 weeks, Sonnet isn't incorporating the ML context properly — debug the prompt before revoking. If <10%, the old path is redundant and safe to remove.

**Code cleanup:**

After revocation, the following files can be significantly simplified:
- `calibrate.py` — shrinks from 18 methods to ~4 (build_ml_context, apply_safety_floor, compute_setup_ev, compute_kelly_size)
- `consensus.py` — shrinks from 2 functions to 1 (floor check only)
- `prediction.py` — standalone predict() removed, inference wrapped into build_ml_context()
- `server.py` — `/calibrate` endpoint deprecated, `/ml-context` added
- `scanner.py` — `_calibrate()` replaced with `_apply_safety_floor()`, much simpler

Estimated line count reduction: ~400-500 lines of calibration/consensus/prediction code that no longer serves a purpose.

---

## Migration Strategy

Each phase is independently deployable and backward-compatible. Phase 1 can ship today with no changes to Sonnet, ML, or notifications — it's purely a Haiku prompt improvement.

Phases 1-2 are the highest-value changes and can ship together: Haiku gets context (finds more setups) and Sonnet gets ML context (makes better level decisions). Phase 2 runs in "dual mode" (old + new paths both active) for 2 weeks of validation. Phase 3 consolidates Opus calls. Phase 4 catches C/D grade setups. Phases 5-7 are architectural cleanup. **Phase 8 (ML legacy revocation) ships only after Phase 2 is validated** — this is the point of no return where the old override architecture is permanently removed.

The existing FN tracker, cost-per-winner tracking, and all feedback loops remain unchanged — they write to the memory stores that ML reads from, regardless of how trades were identified or how the prompt was built.

---

## Metrics to Track

After each phase ships, measure:

- **Setups per day** (target: 2-3x current, especially on 15min/1H)
- **Haiku false-negative rate** (target: <20% across all segments)
- **ML safety net fire rate** (target: <10% — means Sonnet is using ML context properly)
- **Sonnet SL vs ML floor delta** (how often Sonnet sets SL above vs below the floor)
- **C/D promotion rate** (how many monitoring setups get confirmed)
- **Promoted C/D win rate** (are promoted setups actually good?)
- **API cost per setup** (should decrease as Opus calls consolidate)
- **Notification count per day** (target: fewer but higher signal)
- **Time from thesis to entry signal** (should decrease with killzone handoff)
- **Thesis-type feature importance in AutoGluon** (validates narrative features matter)
- **ML context adherence** (does Sonnet set SL above the floor? Does it use DNA WR info?)
- **Old consensus vs new Sonnet SL** (during dual-mode: how often would the old path have been better?)
