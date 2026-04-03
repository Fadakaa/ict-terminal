"""Consensus engine — blend AutoGluon + Bayesian + Volatility into a single recommendation.

Gracefully degrades when any component is None (cold start, insufficient data).
All functions are pure — no side effects, no mutation of inputs.
"""
from ml.config import get_config
from ml.bayesian import adjust_confidence
from ml.volatility import scale_levels


def build_consensus(ag_prediction: dict, bayesian_beliefs: dict | None = None,
                    vol_calibration: dict | None = None,
                    config: dict = None, calibration: dict = None) -> dict:
    """Blend all ML signals into a single consensus recommendation.

    Args:
        ag_prediction: AutoGluon prediction result dict
            (confidence, suggested_sl, suggested_tp1, suggested_tp2, classification)
        bayesian_beliefs: output from get_beliefs(), or None
        vol_calibration: output from calibrate_volatility(), or None
        calibration: optional dict with defensive_mode, coverage_score, regime_adjustment

    Returns: dict with grade, blended_confidence, conservative_sl,
             tp1, tp2, tp3, volatility_regime, bayesian_win_rate,
             session, reasoning
    """
    cfg = config or get_config()
    reasoning = []
    cal = calibration or {}

    # ── Blended confidence ──────────────────────────────────────
    blended = adjust_confidence(
        ag_prediction["confidence"], bayesian_beliefs, config=cfg
    )

    # Apply defensive mode: single regime adjustment dampening
    if cal.get("defensive_mode"):
        regime_adj = cal.get("regime_adjustment", 0.7)
        blended = blended * regime_adj
        reasoning.append(f"Defensive mode active: regime adjustment {regime_adj:.2f}")

    reasoning.append(
        f"AG confidence {ag_prediction['confidence']:.2f}"
        + (f", Bayesian win rate {bayesian_beliefs['win_rate_mean']:.2f}"
           if bayesian_beliefs else ", no Bayesian data")
        + f" → blended {blended:.2f}"
    )

    # ── Conservative SL ─────────────────────────────────────────
    ag_sl = ag_prediction.get("suggested_sl")
    conservative_sl = ag_sl

    if vol_calibration is not None:
        scaled = scale_levels(
            ag_sl, ag_prediction.get("suggested_tp1"),
            ag_prediction.get("suggested_tp2"), vol_calibration, config=cfg
        )
        vol_sl = scaled["scaled_sl"]
        if vol_sl is not None and ag_sl is not None:
            # Pick wider (lower for long = more conservative)
            conservative_sl = min(ag_sl, vol_sl)
            reasoning.append(
                f"SL: AG={ag_sl:.1f}, vol-scaled={vol_sl:.1f} → conservative={conservative_sl:.1f}"
            )
        else:
            reasoning.append("SL: using AG suggestion (no vol scaling)")
    else:
        reasoning.append("SL: using AG suggestion (no vol data)")

    # ── Take profits (passthrough from AG) ──────────────────────
    tp1 = ag_prediction.get("suggested_tp1")
    tp2 = ag_prediction.get("suggested_tp2")
    # tp3 derived from classification if available
    tp3 = None
    classification = ag_prediction.get("classification", {})
    if classification.get("tp3_hit", 0) > 0.1 and tp2 is not None:
        # If model thinks tp3 is plausible, extrapolate from tp1→tp2 distance
        if tp1 is not None:
            tp3 = round(tp2 + (tp2 - tp1), 2)

    # ── Grade ───────────────────────────────────────────────────
    regime = vol_calibration.get("regime") if vol_calibration else None
    grade = _compute_grade(blended, regime, cfg)

    # Defensive mode caps grade at C
    if cal.get("defensive_mode") and grade in ("A", "B"):
        grade = "C"
        reasoning.append("Defensive mode: grade capped at C")

    if regime == "high":
        reasoning.append(f"High volatility regime → grade capped to B or lower")
    reasoning.append(f"Grade: {grade} (blended confidence {blended:.2f})")

    # ── Metadata ────────────────────────────────────────────────
    bayesian_win_rate = (
        bayesian_beliefs["win_rate_mean"] if bayesian_beliefs else None
    )
    session = vol_calibration.get("session") if vol_calibration else None
    volatility_regime = regime

    return {
        "grade": grade,
        "blended_confidence": blended,
        "conservative_sl": conservative_sl,
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp3,
        "volatility_regime": volatility_regime,
        "bayesian_win_rate": bayesian_win_rate,
        "session": session,
        "reasoning": reasoning,
    }


def _compute_grade(blended: float, regime: str | None,
                   config: dict = None) -> str:
    """Map blended confidence + regime to a letter grade.

    A: ≥0.75 AND not high-vol regime
    B: ≥0.60 (or A downgraded by high-vol)
    C: ≥0.45
    D: <0.45
    """
    cfg = config or get_config()
    thresholds = cfg.get("grade_thresholds", {"A": 0.75, "B": 0.60, "C": 0.45})

    if blended >= thresholds["A"] and regime != "high":
        return "A"
    elif blended >= thresholds["B"] or (blended >= thresholds["A"] and regime == "high"):
        return "B"
    elif blended >= thresholds["C"]:
        return "C"
    else:
        return "D"
