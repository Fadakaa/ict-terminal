"""Setup DNA encoder and similarity engine.

Encodes ICT setup characteristics into a categorical feature vector (DNA)
for pattern matching against historical trades. Used by SetupProfileStore
to find similar past setups and compute conditional win rates.

All features are extracted from existing analysis_json and calibration_json —
no candle re-processing or API calls.
"""

# Strength ordering for max() comparison
_STRENGTH_ORDER = {"strong": 3, "moderate": 2, "weak": 1}


def encode_setup_dna(analysis_json: dict, calibration_json: dict = None,
                     timeframe: str = "1h", killzone: str = "Off") -> dict:
    """Extract 21 categorical features from a setup's analysis + calibration.

    Every field defaults to a safe value when missing — the sample_analysis
    fixture in conftest.py lacks structure, htf_context, setup_quality,
    opus_validated, and warnings, so robustness to missing keys is critical.

    Args:
        analysis_json: Claude's ICT analysis dict (from scanner_db.analysis_json)
        calibration_json: ML calibration result dict (from scanner_db.calibration_json)
        timeframe: Candle timeframe (15min, 1h, 4h, 1day)
        killzone: Trading session (Asian, London, NY_AM, NY_PM, Off)

    Returns:
        Dict with 21 string/bool/int features suitable for similarity matching.
    """
    a = analysis_json or {}
    cal = calibration_json or {}

    obs = a.get("orderBlocks", [])
    fvgs = a.get("fvgs", [])
    liqs = a.get("liquidity", [])
    entry = a.get("entry") or {}
    tps = a.get("takeProfits") or []
    htf = a.get("htf_context") or {}
    structure = a.get("structure") or {}
    confluences = a.get("confluences", [])

    # ── Order Blocks ──
    has_ob = len(obs) > 0
    ob_strength = "none"
    ob_times_tested = 0
    if obs:
        strengths = [_STRENGTH_ORDER.get(ob.get("strength", ""), 0) for ob in obs]
        max_s = max(strengths) if strengths else 0
        ob_strength = {3: "strong", 2: "moderate", 1: "weak"}.get(max_s, "none")
        tested = [ob.get("times_tested", 0) for ob in obs]
        ob_times_tested = min(tested) if tested else 0

    # ── FVGs ──
    has_fvg = len(fvgs) > 0
    fvg_overlaps_ob = any(f.get("overlaps_ob", False) for f in fvgs)

    unfilled = [f for f in fvgs if not f.get("filled", True)]
    fill_pcts = [f.get("fill_percentage", 0) for f in unfilled if "fill_percentage" in f]
    if fill_pcts:
        avg_fill = sum(fill_pcts) / len(fill_pcts)
        fvg_fill_pct = "low" if avg_fill < 33 else ("mid" if avg_fill < 66 else "high")
    else:
        fvg_fill_pct = "none"

    # ── Liquidity / Sweep ──
    swept_entries = [l for l in liqs if l.get("swept", False)]
    has_sweep = len(swept_entries) > 0
    sweep_type = "none"
    if swept_entries:
        liq_type = swept_entries[0].get("type", "")
        if liq_type == "buyside":
            sweep_type = "bsl"
        elif liq_type == "sellside":
            sweep_type = "ssl"

    # ── Structure ──
    structure_type = structure.get("type", "none") or "none"
    structure_direction = structure.get("direction", "unknown") or "unknown"

    # ── HTF Context ──
    premium_discount = htf.get("premium_discount", "unknown") or "unknown"
    p3_phase = htf.get("power_of_3_phase", "unknown") or "unknown"

    # ── Confluence ──
    confluence_count = len(confluences)

    # ── Opus ──
    opus_validated = bool(a.get("opus_validated", False))

    # Opus-Sonnet agreement: does Opus narrative bias match entry direction?
    opus_sonnet_agree = False
    opus_narrative = (cal.get("opus_narrative") or {})
    opus_bias = opus_narrative.get("directional_bias", "")
    entry_dir = entry.get("direction", "")
    if opus_bias and entry_dir:
        opus_sonnet_agree = (
            (opus_bias == "bullish" and entry_dir == "long")
            or (opus_bias == "bearish" and entry_dir == "short")
        )

    # ── Entry ──
    direction = entry.get("direction", "unknown") or "unknown"
    entry_type = entry.get("entry_type", "unknown") or "unknown"

    # ── R:R bucket ──
    rr_tp1 = tps[0].get("rr", 0) if tps else 0
    if rr_tp1 >= 4:
        rr_ratio_tp1 = "high"
    elif rr_tp1 >= 2:
        rr_ratio_tp1 = "mid"
    else:
        rr_ratio_tp1 = "low"

    # ── Volatility regime ──
    vol_ctx = cal.get("volatility_context") or {}
    volatility_regime = vol_ctx.get("regime", "unknown") or "unknown"

    # ── Killzone (prefer analysis_json, fall back to arg) ──
    kz = a.get("killzone", killzone) or killzone

    return {
        "killzone": kz,
        "timeframe": timeframe,
        "has_ob": has_ob,
        "ob_strength": ob_strength,
        "ob_times_tested": ob_times_tested,
        "has_fvg": has_fvg,
        "fvg_overlaps_ob": fvg_overlaps_ob,
        "fvg_fill_pct": fvg_fill_pct,
        "has_sweep": has_sweep,
        "sweep_type": sweep_type,
        "structure_type": structure_type,
        "structure_direction": structure_direction,
        "premium_discount": premium_discount,
        "p3_phase": p3_phase,
        "confluence_count": confluence_count,
        "opus_validated": opus_validated,
        "opus_sonnet_agree": opus_sonnet_agree,
        "direction": direction,
        "entry_type": entry_type,
        "rr_ratio_tp1": rr_ratio_tp1,
        "volatility_regime": volatility_regime,
    }


def compute_similarity(dna_a: dict, dna_b: dict) -> float:
    """Compute weighted similarity between two setup DNAs.

    Uses categorical exact-match scoring. Weights emphasise ICT elements
    that matter most (sweeps, OBs, structure) over metadata (direction, opus).

    Returns:
        Float 0.0 to 1.0 (1.0 = identical DNA).
    """
    score = 0.0

    # Killzone: 0.15
    if dna_a.get("killzone") == dna_b.get("killzone"):
        score += 0.15

    # Timeframe: 0.10
    if dna_a.get("timeframe") == dna_b.get("timeframe"):
        score += 0.10

    # Sweep: 0.15 (has_sweep match = 0.075, sweep_type match = 0.075)
    if dna_a.get("has_sweep") == dna_b.get("has_sweep"):
        score += 0.075
        if dna_a.get("sweep_type") == dna_b.get("sweep_type"):
            score += 0.075

    # OB: 0.15 (has_ob = 0.05, ob_strength = 0.10)
    if dna_a.get("has_ob") == dna_b.get("has_ob"):
        score += 0.05
        if dna_a.get("ob_strength") == dna_b.get("ob_strength"):
            score += 0.10

    # FVG: 0.10 (has_fvg = 0.05, fvg_overlaps_ob = 0.05)
    if dna_a.get("has_fvg") == dna_b.get("has_fvg"):
        score += 0.05
        if dna_a.get("fvg_overlaps_ob") == dna_b.get("fvg_overlaps_ob"):
            score += 0.05

    # Structure type: 0.10
    if dna_a.get("structure_type") == dna_b.get("structure_type"):
        score += 0.10

    # Premium/discount: 0.10
    if dna_a.get("premium_discount") == dna_b.get("premium_discount"):
        score += 0.10

    # Direction: 0.05
    if dna_a.get("direction") == dna_b.get("direction"):
        score += 0.05

    # Confluence count: 0.05 (within ±1)
    cc_a = dna_a.get("confluence_count", 0)
    cc_b = dna_b.get("confluence_count", 0)
    if abs(cc_a - cc_b) <= 1:
        score += 0.05

    # Opus validated: 0.05
    if dna_a.get("opus_validated") == dna_b.get("opus_validated"):
        score += 0.05

    return round(score, 4)
