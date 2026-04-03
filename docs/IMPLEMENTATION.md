# Unified Scanner Architecture — Implementation Instructions

**Reference:** `docs/unified_scanner_architecture.md` (full design rationale)
**Codebase:** `ml/` directory
**Tests:** `ml/tests/` — run `python -m pytest ml/tests/ -v` after each phase

Each phase is independently deployable and backward-compatible. Ship in order. Do NOT start Phase 8 until Phase 2 has run in dual-mode for 2+ weeks with safety net fire rate confirmed <10%.

---

## Phase 1: Context-Aware Haiku Screen

**Goal:** Give Haiku the narrative thesis + Opus watch zones so it knows WHAT to look for instead of blind-screening raw candles. Directly increases setup detection on lower timeframes.

### 1A. Modify `build_screen_prompt()` in `ml/prompts.py`

**Current signature (line ~255):**
```python
def build_screen_prompt(candles: list, htf_candles: list, timeframe: str) -> str:
```

**New signature:**
```python
def build_screen_prompt(candles: list, htf_candles: list, timeframe: str,
                        prev_narrative: dict | None = None,
                        watch_zones: list | None = None,
                        pending_setups: list | None = None) -> str:
```

**Logic change:** When `prev_narrative` and/or `watch_zones` are provided, use the context-aware prompt template. When both are None, fall back to the existing generic prompt (unchanged).

**Context-aware prompt template** (replaces the current generic body when context exists):

```python
# After the existing slim/htf_slim candle prep (keep all that)...

if prev_narrative or watch_zones:
    # Context-aware screen
    thesis_line = ""
    if prev_narrative:
        bias = prev_narrative.get('directional_bias', 'neutral')
        phase = prev_narrative.get('p3_phase', '?')
        conf = prev_narrative.get('bias_confidence', 0)
        scans = prev_narrative.get('scan_count', 1)
        thesis_line = f"Current thesis: {bias.title()} {phase}, {conf:.0%} confidence, scan {scans}."
        inv = prev_narrative.get('invalidation')
        if inv:
            thesis_line += f"\nInvalidation level: {inv}"

    zones_line = ""
    if watch_zones:
        zone_strs = []
        for z in watch_zones[:5]:  # Cap at 5 zones to keep prompt short
            level = z.get('level', z.get('price', '?'))
            ztype = z.get('type', 'zone')
            status = z.get('status', 'untested')
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
{"HTF context (" + str(len(htf_slim)) + " candles):" + chr(10) + json.dumps(htf_slim) if htf_slim else ""}

Given the thesis and watch zones above, is price:
- Approaching, testing, or reacting to any watch zone?
- Showing displacement (3+ ATR candle body) in the thesis direction?
- Showing structure shift that would CHANGE the thesis?

Reply ONLY valid JSON:
{{"setup_possible": true|false, "zone_interaction": "zone description"|null, "direction": "long"|"short"|null, "reason": "one sentence"}}"""

else:
    # No context — fall back to existing generic prompt (keep current code unchanged)
    return f"""XAU/USD ICT screen — ...  # existing prompt body, verbatim
```

**Key detail:** The response schema now includes `zone_interaction` (string or null). Update the JSON example in the prompt. The generic fallback still returns the old schema `{setup_possible, direction, reason}` — that's fine, the caller handles missing keys.

### 1B. Modify `_call_claude_screen()` in `ml/scanner.py` (line ~2611)

**Current signature:**
```python
def _call_claude_screen(self, candles: list, htf_candles: list, timeframe: str) -> dict | None:
```

**New signature:**
```python
def _call_claude_screen(self, candles: list, htf_candles: list, timeframe: str,
                        prev_narrative: dict | None = None,
                        watch_zones: list | None = None,
                        pending_setups: list | None = None) -> dict | None:
```

**Changes:**
1. Pass the three new params to `build_screen_prompt()`:
   ```python
   prompt = build_screen_prompt(candles, htf_candles, timeframe,
                                prev_narrative=prev_narrative,
                                watch_zones=watch_zones,
                                pending_setups=pending_setups)
   ```

2. Update cache key to include watch_zones hash (so new Opus zones invalidate cache):
   ```python
   import hashlib
   zones_hash = hashlib.md5(json.dumps(watch_zones or [], sort_keys=True).encode()).hexdigest()[:8]
   cache_key = f"{timeframe}_{hashlib.md5(str(candles[-5:]).encode()).hexdigest()}_{zones_hash}"
   ```

### 1C. Modify `_analyze_and_store()` in `ml/scanner.py` (line ~339)

**Current flow order:**
1. Narrative engine fetch + invalidation check (lines ~348-379)
2. Haiku FN bypass check (lines ~382-392)
3. Haiku screen call (line ~396) — **no context passed**
4. Intermarket fetch (lines ~447-477)
5. Opus HTF narrative (lines ~479-492)
6. Sonnet analysis (line ~512)

**New flow order — move Haiku screen AFTER Opus:**
1. Narrative engine fetch + invalidation check (unchanged)
2. Intermarket fetch (move up — Opus needs it)
3. Opus HTF narrative (move up — Haiku needs its watch_zones)
4. Haiku FN bypass check (unchanged)
5. Haiku screen call — **NOW with context**
6. Sonnet analysis (unchanged for Phase 1)

**The Haiku screen call becomes:**
```python
if not haiku_bypassed:
    screen = self._call_claude_screen(
        candles, htf_candles or [], timeframe,
        prev_narrative=prev_narrative,
        watch_zones=(htf_narrative or {}).get("watch_zones"),
        pending_setups=[s for s in self.db.get_pending()
                        if s.get("timeframe") == timeframe],
    )
else:
    screen = None
```

**Haiku's `zone_interaction` hint passed to Sonnet** (add after screen handling, before `_call_claude` call):
```python
# If Haiku identified a specific zone interaction, pass as hint to Sonnet
haiku_zone_hint = None
if screen and screen.get("zone_interaction"):
    haiku_zone_hint = screen["zone_interaction"]
```
Then add `haiku_zone_hint=haiku_zone_hint` to the `_call_claude()` kwargs. For Phase 1, Sonnet can simply receive this as an additional context line: `"Haiku zone interaction: {haiku_zone_hint}"`. Full integration into Sonnet's prompt comes in Phase 2.

### 1D. Edge Cases

- **Opus cache miss (no watch_zones):** `watch_zones` is None → falls back to generic Haiku prompt. System is never worse than today.
- **No narrative thesis:** `prev_narrative` is None → falls back to generic Haiku prompt.
- **Haiku timeout / API error:** Return None → fall through to Sonnet (same as today, line ~2662).
- **watch_zones schema:** Opus already returns `key_levels` in its response. If `watch_zones` is not a separate field yet, extract zones from `key_levels` in the Opus response and format as `[{"level": price, "type": "OB"|"FVG"|"BSL"|"SSL", "status": "untested"|"tested"|"swept"}]`.

### 1E. Tests

Create `ml/tests/test_context_screen.py`:
- `test_screen_prompt_with_thesis_and_zones()` — verify prompt contains thesis line and zone lines
- `test_screen_prompt_no_context_fallback()` — verify generic prompt when prev_narrative=None and watch_zones=None
- `test_screen_prompt_partial_context()` — thesis only, zones only
- `test_cache_key_includes_zones()` — different zones produce different cache keys
- `test_zone_interaction_passthrough()` — Haiku's zone_interaction reaches Sonnet kwargs
- `test_flow_order()` — mock Opus, verify Haiku is called after Opus

**Run:** `python -m pytest ml/tests/test_context_screen.py -v`

---

## Phase 2: ML Repositioning — Pre-Analysis Enrichment

**Goal:** ML context flows INTO Sonnet's prompt BEFORE it makes decisions, not as a post-hoc override. This is the fundamental architectural shift.

### 2A. New function `build_ml_context()` in `ml/calibrate.py`

Add this method to `MLCalibrator`:

```python
def build_ml_context(self, thesis_type: str | None, timeframe: str,
                     killzone: str, candles: list,
                     setup_dna_pattern: str | None = None) -> dict:
    """Build prompt-ready ML context block for Sonnet enrichment.

    Runs the statistical memory layers and returns structured data
    for injection into Sonnet's prompt — NOT competing SL/TP values.

    Args:
        thesis_type: e.g. "bullish_accumulation" from narrative engine
        timeframe: "15min", "1h", "4h", "1day"
        killzone: "Asian", "London", "NY_AM", "NY_PM", "Off"
        candles: OHLC candle list for ATR/regime computation
        setup_dna_pattern: optional DNA string for pattern matching

    Returns:
        dict with keys: regime, sl_floor_atr, mae_percentile_80,
        dna_win_rate, dna_avg_rr, dna_sample_size, bayesian_wr,
        bayesian_trend, ag_sl_band, ag_tp1_band, intermarket_quality,
        opus_accuracy, entry_placement_guidance
    """
    ctx = {}

    # 1. Regime + volatility floor
    candle_list = self._to_candle_list(candles)
    if candle_list:
        from ml.volatility import calibrate_volatility, detect_session
        from ml.features import compute_atr
        atr = compute_atr(candle_list)
        vol = calibrate_volatility(candle_list)
        ctx["regime"] = vol.get("regime", "NORMAL")
        ctx["vol_ratio"] = vol.get("vol_ratio", 1.0)
        ctx["sl_floor_atr"] = vol.get("min_sl_atr", 3.0)
        # MAE from V1 session stats
        session = detect_session(candle_list)
        session_stats = self._v1_session_stats.get(session, {})
        ctx["mae_percentile_80"] = session_stats.get("mae_p80_atr", 4.0)
    else:
        ctx["regime"] = "UNKNOWN"
        ctx["sl_floor_atr"] = 3.0
        ctx["mae_percentile_80"] = 4.0

    # 2. Setup DNA pattern match
    if setup_dna_pattern:
        try:
            from ml.setup_profiles import SetupProfileStore
            store = SetupProfileStore()
            match = store.get_profile(setup_dna_pattern)
            if match and match.get("sample_size", 0) >= 15:
                ctx["dna_win_rate"] = match["win_rate"]
                ctx["dna_avg_rr"] = match.get("avg_rr", 0)
                ctx["dna_sample_size"] = match["sample_size"]
        except Exception:
            pass

    # 3. Bayesian beliefs
    try:
        from ml.bayesian import get_beliefs
        beliefs = get_beliefs()
        if beliefs:
            ctx["bayesian_wr"] = beliefs.get("win_rate_mean", 0.5)
            ctx["bayesian_trend"] = beliefs.get("trend_pp", 0)
    except Exception:
        pass

    # 4. AutoGluon quantile predictions (if model exists)
    if self._autogluon_available:
        try:
            from ml.training import load_model
            # Feature extraction needs candles + thesis context
            from ml.features import engineer_features_from_candles
            features = engineer_features_from_candles(candle_list, timeframe)
            model = load_model("quantile")
            if model and features is not None:
                pred = model.predict(features)
                ctx["ag_sl_band"] = [float(pred.get("sl_low", 0)),
                                     float(pred.get("sl_high", 0))]
                ctx["ag_tp1_band"] = [float(pred.get("tp1_low", 0)),
                                      float(pred.get("tp1_high", 0))]
        except Exception:
            pass

    # 5. Intermarket signal quality
    try:
        from ml.intermarket_validator import IntermarketValidator
        iv = IntermarketValidator()
        result = iv.get_last_result()
        if result:
            ctx["intermarket_quality"] = result.get("recommendation", "unknown")
    except Exception:
        pass

    # 6. Opus accuracy by narrative type
    if thesis_type:
        try:
            from ml.claude_bridge import ClaudeAnalysisBridge
            bridge = ClaudeAnalysisBridge()
            tracker = bridge.get_narrative_tracker()
            if tracker:
                # Extract phase from thesis_type (e.g. "bullish_accumulation" → "accumulation")
                phase = thesis_type.split("_")[-1] if "_" in thesis_type else thesis_type
                accuracy = tracker.get(f"accuracy_{phase}")
                if accuracy is not None:
                    ctx["opus_accuracy"] = accuracy
        except Exception:
            pass

    # 7. Entry placement guidance
    try:
        from ml.entry_placement import EntryPlacementAnalyzer
        guidance = EntryPlacementAnalyzer().get_placement_guidance()
        if guidance and guidance.get("status") == "active":
            ctx["entry_placement"] = guidance.get("best_zone", "OB midpoint")
            ctx["entry_placement_delta_rr"] = guidance.get("improvement_rr", 0)
    except Exception:
        pass

    return ctx
```

### 2B. New function `_build_ml_context_section()` in `ml/prompts.py`

```python
def _build_ml_context_section(ml_context: dict | None) -> str:
    """Format ML context as a prompt section for Sonnet.

    This is Claude's statistical memory — historical patterns and outcome
    distributions it cannot remember between API calls.
    """
    if not ml_context:
        return ""

    lines = ["=== YOUR STATISTICAL MEMORY ==="]

    # Pattern match
    dna_wr = ml_context.get("dna_win_rate")
    if dna_wr is not None:
        dna_rr = ml_context.get("dna_avg_rr", 0)
        dna_n = ml_context.get("dna_sample_size", 0)
        lines.append(f"Pattern match: {dna_wr:.0%} WR across {dna_n} similar setups, avg R:R {dna_rr:.1f}:1")
        opus_acc = ml_context.get("opus_accuracy")
        if opus_acc is not None:
            lines.append(f"  Opus accuracy on this narrative type: {opus_acc:.0%}")

    # Regime
    regime = ml_context.get("regime", "UNKNOWN")
    sl_floor = ml_context.get("sl_floor_atr", 3.0)
    mae = ml_context.get("mae_percentile_80", 4.0)
    vol_ratio = ml_context.get("vol_ratio", 1.0)
    lines.append(f"\nRegime: {regime} (vol_ratio {vol_ratio:.1f}x normal)")
    lines.append(f"  Required SL floor: {sl_floor:.1f} ATR (below this, high stop-out risk)")
    lines.append(f"  Expected drawdown before move: {mae:.1f} ATR (80th percentile MAE)")

    # Entry placement
    ep = ml_context.get("entry_placement")
    if ep:
        delta = ml_context.get("entry_placement_delta_rr", 0)
        lines.append(f"  Best entry position: {ep} (outperforms alternatives by +{delta:.1f} R:R)")

    # Overall win rate
    wr = ml_context.get("bayesian_wr")
    if wr is not None:
        trend = ml_context.get("bayesian_trend", 0)
        trend_str = f"trending {'up' if trend > 0 else 'down'} {abs(trend):.1f}pp" if trend else "stable"
        lines.append(f"\nOverall win rate: {wr:.1%} ({trend_str})")

    # AutoGluon bands
    ag_sl = ml_context.get("ag_sl_band")
    ag_tp = ml_context.get("ag_tp1_band")
    if ag_sl:
        lines.append(f"  AutoGluon SL band for this pattern: {ag_sl[0]:.1f}-{ag_sl[1]:.1f} ATR")
    if ag_tp:
        lines.append(f"  AutoGluon TP1 band: {ag_tp[0]:.1f}-{ag_tp[1]:.1f} R:R")

    # Intermarket
    im = ml_context.get("intermarket_quality")
    if im and im != "unknown":
        lines.append(f"\nIntermarket signal quality: {im.upper()}")

    lines.append("")  # Trailing newline
    return "\n".join(lines)
```

### 2C. Modify `build_enhanced_ict_prompt()` in `ml/prompts.py` (line ~30)

**Add parameter:**
```python
def build_enhanced_ict_prompt(candles_1h: list, candles_4h: list,
                              ...,  # existing params unchanged
                              ml_context: dict | None = None,  # NEW
                              htf_label: str | None = None) -> str:
```

**Insert the ML context section** into the prompt body. Place it AFTER the "YOUR PREVIOUS THESIS" section and BEFORE the "ANALYSIS FRAMEWORK" section:

```python
# After _build_narrative_state_section(...)
ml_section = _build_ml_context_section(ml_context)
# Insert ml_section into the prompt string
```

### 2D. Modify `_analyze_and_store()` in `ml/scanner.py`

**Add ML context computation BEFORE the Sonnet call** (after Opus, after Haiku screen, before `_call_claude`):

```python
# ── ML enrichment: build statistical memory for Sonnet ──
ml_context = None
try:
    from ml.calibrate import MLCalibrator
    calibrator = MLCalibrator()
    # Build thesis type from narrative + Opus
    thesis_type = None
    if prev_narrative:
        bias = prev_narrative.get("directional_bias", "")
        phase = prev_narrative.get("p3_phase", "")
        if bias and phase:
            thesis_type = f"{bias}_{phase}"

    # Build DNA pattern from Opus watch zones if available
    dna_pattern = None
    try:
        from ml.setup_dna import encode_dna
        direction_hint = (screen or {}).get("direction") or (prev_narrative or {}).get("directional_bias")
        if direction_hint:
            dna_pattern = encode_dna(direction_hint, current_kz, timeframe, [])
    except Exception:
        pass

    ml_context = calibrator.build_ml_context(
        thesis_type=thesis_type,
        timeframe=timeframe,
        killzone=current_kz,
        candles=candles,
        setup_dna_pattern=dna_pattern,
    )
except Exception as e:
    logger.debug("ML context build failed (proceeding without): %s", e)
```

**Pass `ml_context` to `_call_claude()`:**
```python
analysis = self._call_claude(candles, htf_candles or [], timeframe,
                             intermarket=intermarket_ctx,
                             htf_narrative=htf_narrative,
                             setup_context=setup_context,
                             prev_narrative=prev_narrative,
                             invalidation_status=invalidation_status,
                             recent_context=recent_context,
                             ml_context=ml_context)  # NEW
```

**Modify `_call_claude()` to accept and pass `ml_context`** to `build_enhanced_ict_prompt()`.

### 2E. Dual-Mode Tracking

**Keep the existing `_calibrate()` call** (line ~2900) running as before. Add a comparison metric:

```python
# After _calibrate() runs, compare with ML-informed Sonnet levels
if calibration_result and ml_context:
    sonnet_sl = analysis.get("entry", {}).get("stop_loss_atr", 0)
    old_consensus_sl = calibration_result.get("consensus", {}).get("conservative_sl_atr", 0)
    ml_floor = ml_context.get("sl_floor_atr", 0)
    safety_net_fired = sonnet_sl < ml_floor if sonnet_sl and ml_floor else False

    logger.info("Scanner [%s] DUAL-MODE: Sonnet SL=%.1f ATR, old consensus=%.1f ATR, "
                "ML floor=%.1f ATR, safety_net_fired=%s",
                timeframe, sonnet_sl, old_consensus_sl, ml_floor, safety_net_fired)

    # Track safety net fire rate
    self._filter_stats.setdefault("ml_safety_net_checks", 0)
    self._filter_stats.setdefault("ml_safety_net_fired", 0)
    self._filter_stats["ml_safety_net_checks"] += 1
    if safety_net_fired:
        self._filter_stats["ml_safety_net_fired"] += 1
```

### 2F. Tests

Create `ml/tests/test_ml_context.py`:
- `test_build_ml_context_full()` — all layers return data
- `test_build_ml_context_cold_start()` — no models, no DNA, graceful defaults
- `test_build_ml_context_section_formatting()` — verify prompt section renders correctly
- `test_ml_context_in_sonnet_prompt()` — verify "YOUR STATISTICAL MEMORY" appears in the full prompt
- `test_dual_mode_logging()` — verify safety net fire rate tracking
- `test_ml_context_graceful_degradation()` — each layer can fail independently

**Run:** `python -m pytest ml/tests/test_ml_context.py -v`

---

## Phase 3: Opus Strategy Consolidation

**Goal:** Collapse the separate HTF narrative + prospect zone ID Opus calls into one strategy call. Reduces API cost, improves coherence.

### 3A. New `build_opus_strategy_prompt()` in `ml/prompts.py`

Create a combined prompt that asks Opus for BOTH macro analysis AND specific watch zones in a single call. The response schema includes:

```json
{
  "directional_bias": "bullish|bearish|neutral",
  "bias_confidence": 0.0-1.0,
  "power_of_3_phase": "accumulation|manipulation|distribution",
  "p3_progress": "early|mid|late",
  "key_levels": [...],
  "watch_zones": [
    {"level": 2340.5, "type": "OB", "direction": "bullish", "status": "untested"},
    {"level": 2358.0, "type": "BSL", "direction": "neutral", "status": "unswept"}
  ],
  "macro_narrative": "free text",
  "invalidation_level": 2325.0
}
```

### 3B. Remove `_prospect_killzone_zones()`

The prospect zone identification logic (currently a separate Opus call before each killzone) is replaced by `watch_zones` in the strategy response. Remove:
- `_prospect_killzone_zones()` method in scanner.py
- The separate pre-killzone scheduler job that calls it
- `generate_prospect_zones()` or equivalent prospect-specific functions

### 3C. Update `_call_opus_narrative()` → rename to `_call_opus_strategy()`

Modify to use the new combined prompt. Response parsing now extracts `watch_zones` alongside existing fields. Cache TTL remains 1 hour.

### 3D. Tests

- `test_opus_strategy_returns_watch_zones()` — mock Opus response, verify watch_zones parsed
- `test_opus_strategy_replaces_prospect_zones()` — verify old prospect path removed
- `test_opus_cache_shared_across_timeframes()` — one call serves all TF scans within TTL

---

## Phase 4: C/D Grade Monitoring Pipeline

**Goal:** Store marginal setups and promote them if price action confirms. Catches 15-25% more valid setups.

### 4A. New `status="monitoring"` in scanner_db

Add to `ScannerDB`:
```python
def store_monitoring_setup(self, setup_data: dict) -> str:
    """Store a C/D grade setup for displacement monitoring."""
    setup_data["status"] = "monitoring"
    setup_data["monitoring_since"] = datetime.utcnow().isoformat()
    return self.store_setup(setup_data)

def get_monitoring_setups(self, timeframe: str = None) -> list:
    """Get all C/D setups currently being monitored."""
    # Query scanner_setups WHERE status = 'monitoring' and not expired
```

### 4B. Route C/D grades to monitoring in `_analyze_and_store()`

After Sonnet returns a grade C or D setup (currently these are dropped at the quality gate):

```python
grade = analysis.get("setup_quality", "D")
if grade in ("C", "D") and analysis.get("entry", {}).get("price"):
    # Store for monitoring instead of dropping
    self.db.store_monitoring_setup({
        "timeframe": timeframe,
        "direction": analysis["entry"]["direction"],
        "entry_price": analysis["entry"]["price"],
        "grade": grade,
        "analysis_json": analysis,
        "killzone": current_kz,
        "expiry_hours": EXPIRY_HOURS.get(timeframe, 48),
    })
    return {"status": "monitoring", "grade": grade}
```

### 4C. Promotion criteria (checked in unified monitor — Phase 5)

ALL three must be met:
1. Price within 1.0 ATR of setup's entry zone
2. Single candle body ≥ 2.0 ATR in setup's direction (displacement)
3. Displacement candle sweeps a liquidity level (BSL/SSL within 0.5 ATR of candle extremes)

On promotion: upgrade to B → run `build_ml_context()` → Sonnet re-evaluation → Opus validation → store as pending → notify user.

### 4D. Expiry windows

Same as existing `EXPIRY_HOURS`: 15min→8h, 1H→48h, 4H→168h, 1D→336h. Expired monitoring setups → resolve as expired → write to `recent_context` for narrative continuity.

### 4E. Tests

- `test_cd_grade_stored_as_monitoring()` — C/D setups not dropped
- `test_monitoring_expiry()` — expired after window
- `test_promotion_criteria()` — all 3 conditions checked
- `test_promoted_setup_goes_through_opus()` — Opus validation required after promotion
- `test_expired_monitoring_feeds_narrative()` — expired setup appears in recent_context

---

## Phase 5: Unified Monitor Loop

**Goal:** Merge `monitor_pending()` (line ~1091) and `monitor_prospect_triggers()` (line ~1970) into one loop with shared candle fetches.

### 5A. New `unified_monitor()` method

```python
def unified_monitor(self) -> dict:
    """Single monitoring loop for all active items.

    Replaces both monitor_pending() and monitor_prospect_triggers().
    Uses a priority queue:
      1. A/B setups near entry (poll every 60s when price within 0.3%)
      2. C/D setups watching for displacement (poll every 90s)
      3. Opus watch zones during killzone (poll every 90s)
    """
```

### 5B. Shared candle fetching

All monitored items for the same instrument share one 5-minute candle fetch per poll cycle. Currently `monitor_pending()` and `monitor_prospect_triggers()` each fetch independently.

### 5C. Cross-referencing

When a watch zone gets hit (sweep + displacement), check if any pending C/D setup is near that zone. If so, fast-track promotion.

### 5D. Scheduler update

Replace the two separate scheduler jobs with one `unified_monitor` job. Default poll interval: 60s (same as current fastest).

### 5E. Tests

- `test_unified_monitor_handles_all_types()` — A/B pending, C/D monitoring, watch zones
- `test_shared_candle_fetch()` — one fetch serves all items
- `test_zone_hit_promotes_cd()` — cross-reference works
- `test_priority_ordering()` — A/B checked before C/D before zones

---

## Phase 6: Killzone Handoff + Notification Simplification

**Goal:** Session continuity between killzones. Reduce notifications to 2 types max.

### 6A. Killzone handoff

Add `killzone_summary` column to `narrative_states` table in `ml/narrative_state.py`.

In scanner, detect killzone transition via `_check_killzone_transition()`:
```python
def _check_killzone_transition(self, current_kz: str, timeframe: str) -> str | None:
    """If killzone changed since last scan, generate a handoff summary."""
    last_kz = self._last_killzone.get(timeframe)
    if last_kz and last_kz != current_kz and last_kz != "Off":
        # Fetch the outgoing KZ's narrative and summarize
        ns_store = NarrativeStore(self.db.db_path)
        thesis = ns_store.get_current(timeframe)
        if thesis:
            summary = (f"{last_kz} session summary: {thesis.get('thesis', 'no thesis')}. "
                       f"Bias was {thesis.get('directional_bias', '?')} "
                       f"({thesis.get('bias_confidence', 0):.0%} confidence).")
            ns_store.update_killzone_summary(thesis['id'], summary)
            return summary
    self._last_killzone[timeframe] = current_kz
    return None
```

Inject the summary into Sonnet's prompt in the next scan.

### 6B. Notification consolidation

**Keep:**
- `notify_thesis_active()` — fires when thesis confirmed (2+ scans, ≥70% confidence)
- `notify_entry_signal()` — fires when A/B setup passes all gates (or promoted C/D)
- `notify_thesis_revised()` — risk warning (uncounted), fires when bias flips with pending setup

**Demote to log-only (no Telegram/macOS push):**
- `notify_setup_detected()` → log to Intelligence Panel only
- `notify_zone_alert()` → log only
- `notify_displacement_confirmed()` → log only

### 6C. Tests

- `test_killzone_handoff_summary_generated()` — transition detected, summary written
- `test_handoff_injected_in_prompt()` — next scan sees previous KZ summary
- `test_only_two_notification_types_pushed()` — verify only THESIS_ACTIVE + ENTRY_SIGNAL reach Telegram
- `test_thesis_revised_fires_on_flip()` — direction change + pending setup triggers warning

---

## Phase 7: Missed Setup Recycling

**Goal:** Expired/missed setups feed back into narrative engine for continuity.

### 7A. On setup expiry/miss

When a setup resolves as expired or entry-missed:
```python
# Write to recent_context
from ml.recent_context import log_missed_setup
log_missed_setup(
    timeframe=setup["timeframe"],
    direction=setup["direction"],
    entry_price=setup["entry_price"],
    outcome="expired" | "entry_missed",
    zone_description=f"OB at {setup['entry_price']:.2f}",
)
```

### 7B. Prompt enrichment

In `build_recent_context()`, include missed setups:
```
Previous scan identified OB at 2342 (long) but entry was never reached.
Price instead moved to {current_price}. Reassess whether thesis holds.
```

### 7C. Tests

- `test_expired_setup_in_recent_context()` — expired setup appears
- `test_missed_setup_in_sonnet_prompt()` — Sonnet sees the missed setup info
- `test_reassessment_guidance()` — prompt asks Sonnet to reassess

---

## Phase 8: ML Legacy Revocation

**PREREQUISITE:** Phase 2 live for 2+ weeks. Safety net fire rate confirmed <10% via `self._filter_stats["ml_safety_net_fired"] / self._filter_stats["ml_safety_net_checks"]`.

**Goal:** Remove the old parallel ML override path. Enforce clean architecture where ML serves the prompt, not competes with it.

### 8A. What Gets Removed

| Target | File | Action |
|--------|------|--------|
| `calibrate_trade()` competing SL/TP | `ml/calibrate.py` | Remove method. `build_ml_context()` replaces it. |
| `build_consensus()` full blending | `ml/consensus.py` | Reduce to `apply_safety_floor(sonnet_sl_atr, regime_floor_atr) -> float`. One function, one check. |
| `_calibrate()` | `ml/scanner.py` line ~2900 | Replace with `_apply_safety_floor()` — calls the reduced consensus. |
| `predict()` standalone | `ml/prediction.py` | Remove. AG inference folded into `build_ml_context()`. |
| `_build_calibration()` | `ml/prediction.py` | Remove. |
| `_enrich_with_consensus()` | `ml/prediction.py` | Remove. |
| `calibrated_sl`, `calibrated_tps` columns | `ml/scanner_db.py` | Replace with `ml_floor_applied` boolean column. |
| `/calibrate` endpoint | `ml/server.py` | Deprecate. Add `/ml-context` GET endpoint returning `build_ml_context()` output. |

### 8B. What Stays

| Component | Why |
|-----------|-----|
| `build_ml_context()` | This IS the new ML path |
| Volatility floor check | Single safety net |
| `bayesian.update_beliefs()` | Memory write — still needed |
| `dataset.ingest_live_trade()` | Training data accumulation |
| AutoGluon `train_classifier()` | Model retraining |
| Setup DNA profiling | Pattern memory |
| All 12 feedback loops in `_log_trade_complete()` | Write to memory stores |
| `layer_performance.ingest_trade()` | Repurpose: track `ml_context_adherence` (did Sonnet respect ML floor?) |

### 8C. New `_apply_safety_floor()` in `ml/scanner.py`

```python
def _apply_safety_floor(self, analysis: dict, candles: list, ml_context: dict | None) -> dict:
    """Minimal post-Sonnet guard rail. Fires <10% of the time.

    If Sonnet's SL is below the volatility floor, widen to floor.
    Returns the analysis dict with sl potentially widened and ml_floor_applied flag.
    """
    if not ml_context:
        return analysis

    floor = ml_context.get("sl_floor_atr", 3.0)
    entry = analysis.get("entry", {})
    sonnet_sl_atr = entry.get("stop_loss_atr", 0)

    if sonnet_sl_atr and sonnet_sl_atr < floor:
        logger.info("Safety floor: Sonnet SL %.1f ATR < floor %.1f ATR — widening",
                     sonnet_sl_atr, floor)
        entry["stop_loss_atr"] = floor
        # Recalculate price-based SL from ATR
        # ... (use ATR and entry price to compute new SL price)
        analysis["ml_floor_applied"] = True
    else:
        analysis["ml_floor_applied"] = False

    return analysis
```

### 8D. New `/ml-context` endpoint in `ml/server.py`

```python
@app.get("/ml-context")
def ml_context_endpoint(timeframe: str = "1h", killzone: str = "London",
                        thesis_type: str = None):
    """Return ML context block for external callers."""
    from ml.calibrate import MLCalibrator
    calibrator = MLCalibrator()
    # Fetch recent candles for regime computation
    candles = _fetch_recent_candles(timeframe, count=60)
    ctx = calibrator.build_ml_context(thesis_type, timeframe, killzone, candles)
    return ctx
```

### 8E. Migration Safety — Dual-Mode Validation

Before removing anything:
1. Run Phase 2 dual-mode for 2 weeks
2. Query: `ml_safety_net_fired / ml_safety_net_checks` — must be <10%
3. Query: how often old consensus SL would have saved a trade that Sonnet's SL didn't
4. If safety net >10%, debug Sonnet's prompt adherence before proceeding

### 8F. Tests

- `test_safety_floor_widens_sl()` — floor applied when Sonnet SL < floor
- `test_safety_floor_noop()` — no change when Sonnet SL ≥ floor
- `test_old_calibrate_removed()` — `calibrate_trade()` no longer exists
- `test_old_consensus_removed()` — `build_consensus()` replaced by `apply_safety_floor()`
- `test_ml_context_endpoint()` — `/ml-context` returns valid dict
- `test_feedback_loops_intact()` — all 12 feedback loops still fire on resolution
- `test_ml_floor_applied_column()` — `ml_floor_applied` boolean tracked in scanner_db

**Run full suite after Phase 8:** `python -m pytest ml/tests/ -v` — all 842+ tests must pass (with old calibration tests updated/removed).

---

## Execution Order Summary

| Phase | Effort | Depends On | Key Metric |
|-------|--------|------------|------------|
| 1. Context-Aware Haiku | 2-3h | Nothing (ship today) | Haiku FN rate <20% |
| 2. ML Repositioning | 4-5h | Phase 1 | Safety net fire rate <10% |
| 3. Opus Consolidation | 3-4h | Phase 1 | API cost per setup drops |
| 4. C/D Monitoring | 4-5h | Phase 2 | Setups/day increases 2-3x |
| 5. Unified Monitor | 5-6h | Phase 4 | Single loop, shared fetches |
| 6. KZ Handoff + Notifications | 3-4h | Phase 5 | ≤2 notifications per setup |
| 7. Missed Setup Recycling | 2-3h | Phase 6 | Narrative continuity |
| 8. ML Legacy Revocation | 3-4h | Phase 2 validated 2+ weeks | Clean architecture |

**Total estimated effort:** 27-34 hours across 8 phases.

**After all phases:** Run `python -m pytest ml/tests/ -v` — full green. The system should produce 2-3x more setups per day, with Sonnet making informed SL/TP decisions from its statistical memory, and the old ML override path fully retired.
