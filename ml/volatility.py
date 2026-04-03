"""Volatility calibrator — EWMA + session scaling + regime detection.

Combines ATR with exponentially weighted volatility, session-aware scaling
(London/NY/Asian), and regime classification for more accurate SL/TP calibration.
All functions are pure — no side effects, no mutation of inputs.
"""
import math
from ml.config import get_config
from ml.features import compute_atr


def _percentileofscore(data: list, score: float) -> float:
    """Compute percentile rank of score in data (0-100), no scipy needed."""
    if not data:
        return 50.0
    n = len(data)
    left = sum(1 for x in data if x < score)
    right = sum(1 for x in data if x <= score)
    return 100.0 * (left + right) / (2 * n)


def compute_ewma_volatility(candles: list[dict], lambda_: float = None,
                            config: dict = None) -> float:
    """Exponentially weighted moving average of absolute returns (RiskMetrics).

    Uses |close[i] - close[i-1]| as return proxy.
    Returns: EWMA volatility estimate (same units as price).
    """
    if len(candles) < 2:
        return 0.0

    cfg = config or get_config()
    lam = lambda_ if lambda_ is not None else cfg.get("ewma_lambda", 0.94)

    closes = [c.get("close", 0) for c in candles if "close" in c]
    if len(closes) < 2:
        return 0.0

    # Compute absolute returns
    returns = [abs(closes[i] - closes[i - 1]) for i in range(1, len(closes))]
    if not returns:
        return 0.0

    # EWMA: variance_t = lambda * variance_{t-1} + (1-lambda) * r_t^2
    var = returns[0] ** 2
    for r in returns[1:]:
        var = lam * var + (1 - lam) * r ** 2

    return math.sqrt(var)


def detect_session(candles: list[dict]) -> str:
    """Classify the current trading session from the last candle timestamp.

    Session hours (UTC):
        Asian:    00:00-07:00
        London:   07:00-12:00
        Overlap:  12:00-16:00 (London+NY)
        New York: 16:00-21:00
        Off:      21:00-00:00
    """
    if not candles:
        return "off_hours"

    dt_str = candles[-1].get("datetime", "")
    if not dt_str:
        return "off_hours"

    try:
        # Extract hour from ISO format ("2026-03-16T01:30:00")
        # or Twelve Data format ("2026-03-16 01:30:00")
        if "T" in dt_str:
            time_part = dt_str.split("T")[1]
        elif " " in dt_str:
            time_part = dt_str.split(" ")[1]
        else:
            time_part = dt_str
        hour = int(time_part.split(":")[0])
    except (ValueError, IndexError):
        return "off_hours"

    if 12 <= hour < 16:
        return "overlap_london_ny"
    elif 7 <= hour < 12:
        return "london"
    elif 16 <= hour < 21:
        return "new_york"
    elif 0 <= hour < 7:
        return "asian"
    else:
        return "off_hours"


def get_session_factor(session: str, config: dict = None) -> float:
    """Return volatility scaling factor for the given session."""
    cfg = config or get_config()
    factors = cfg.get("session_factors", {
        "london": 1.1,
        "new_york": 1.15,
        "overlap_london_ny": 1.3,
        "asian": 0.7,
        "off_hours": 0.5,
    })
    return factors.get(session, 1.0)


def detect_regime(atr: float, candles: list[dict], lookback: int = None,
                  config: dict = None) -> tuple[str, float]:
    """Classify volatility regime based on ATR percentile vs rolling history.

    Returns: (regime_label, regime_multiplier)
        "low":    ATR < 25th percentile → 0.8
        "normal": 25th-75th percentile → 1.0
        "high":   > 75th percentile → 1.2
    """
    cfg = config or get_config()
    lb = lookback or cfg.get("regime_lookback", 100)
    thresholds = cfg.get("regime_thresholds", [25, 75])
    multipliers = cfg.get("regime_multipliers", {"low": 0.8, "normal": 1.0, "high": 1.2})

    # Need enough history for meaningful percentile
    if len(candles) < 15:
        return "normal", multipliers["normal"]

    # Compute ATR for rolling windows
    window = min(len(candles), lb)
    atr_history = []
    for i in range(14, window):
        sub = candles[max(0, i - 14):i + 1]
        a = compute_atr(sub, period=14)
        if a > 0:
            atr_history.append(a)

    if len(atr_history) < 3:
        return "normal", multipliers["normal"]

    atr_history.sort()
    n = len(atr_history)
    p_low = atr_history[int(n * thresholds[0] / 100)]
    p_high = atr_history[int(min(n - 1, n * thresholds[1] / 100))]

    if atr < p_low:
        return "low", multipliers["low"]
    elif atr > p_high:
        return "high", multipliers["high"]
    else:
        return "normal", multipliers["normal"]


def classify_regime(candles: list, atr: float = None,
                    config: dict = None) -> dict:
    """Unified 5-state regime classification from candle data.

    Combines ATR percentile ranking (from volatility.py's detect_regime) with
    directional movement and displacement analysis (inspired by wfo.py's
    detect_regime) into a single, richer taxonomy that feeds the prompt,
    calibration Layer 1, training dataset, and quality gates.

    States:
        TRENDING_IMPULSIVE  — Strong move, breakout entries work, tight SL wide TP
        TRENDING_CORRECTIVE — Pullback in trend, OB/FVG retests are ideal
        RANGING             — No direction, range extremes are key (BSL/SSL)
        VOLATILE_CHOPPY     — High ATR no direction, dangerous for entries
        QUIET_DRIFT         — Low ATR slow drift, only highest-confluence setups

    Returns:
        {
            'regime': str,          # One of 5 regime labels
            'confidence': float,    # 0-1 how clearly this regime fits
            'metrics': {
                'atr_percentile': float,  # Where current ATR sits vs history
                'vol_ratio_5_30': float,  # Short vs long volatility ratio
                'net_movement_atr': float,# 20-bar directional move in ATR units
                'displacement_count': int,# Displacement candles in last 10
                'body_consistency': float,# % candles with bodies same direction
            }
        }
    """
    if atr is None:
        atr = compute_atr(candles)
    cfg = config or get_config()

    # Need minimum candles for meaningful classification
    if not candles or len(candles) < 15 or atr <= 0:
        return {
            "regime": "RANGING",
            "confidence": 0.1,
            "metrics": {
                "atr_percentile": 50.0,
                "vol_ratio_5_30": 1.0,
                "net_movement_atr": 0.0,
                "displacement_count": 0,
                "body_consistency": 0.5,
            },
        }

    # Step 1: ATR percentile (from volatility.py approach)
    atr_history = [compute_atr(candles[max(0, i - 14):i + 1])
                   for i in range(14, len(candles))]
    atr_history = [a for a in atr_history if a > 0]
    pct = _percentileofscore(atr_history, atr) / 100 if atr_history else 0.5

    # Step 2: Vol ratio (from wfo.py approach)
    returns_5 = [abs(candles[i]["close"] - candles[i - 1]["close"])
                 for i in range(max(1, len(candles) - 5), len(candles))]
    vol_5 = (sum(r ** 2 for r in returns_5) / len(returns_5)) ** 0.5 if returns_5 else 0

    lookback_30 = min(30, len(candles) - 1)
    returns_30 = [abs(candles[i]["close"] - candles[i - 1]["close"])
                  for i in range(max(1, len(candles) - lookback_30), len(candles))]
    vol_30 = (sum(r ** 2 for r in returns_30) / len(returns_30)) ** 0.5 if returns_30 else 0
    vol_ratio = vol_5 / vol_30 if vol_30 > 0 else 1.0

    # Step 3: Net directional movement (20-bar net move in ATR units)
    lookback_dir = min(21, len(candles))
    net_move = abs(candles[-1]["close"] - candles[-lookback_dir]["close"]) / atr

    # Step 4: Displacement count (large body candles in last 10)
    lookback_disp = min(10, len(candles))
    displacements = sum(
        1 for c in candles[-lookback_disp:]
        if abs(c["close"] - c["open"]) > 1.5 * atr
    )

    # Step 5: Body consistency (% candles with bodies in same direction)
    lookback_body = min(10, len(candles))
    bodies = [c["close"] - c["open"] for c in candles[-lookback_body:]]
    same_dir = max(
        sum(1 for b in bodies if b > 0),
        sum(1 for b in bodies if b < 0),
    )
    body_consistency = same_dir / len(bodies) if bodies else 0.5

    # Classification tree
    if vol_ratio > 1.5 and net_move > 1.0:
        regime = "TRENDING_IMPULSIVE"
        conf = min(1.0, (vol_ratio - 1.5) / 1.0 + (net_move - 1.0) / 2.0)
    elif net_move > 0.5 and body_consistency > 0.6:
        regime = "TRENDING_CORRECTIVE"
        conf = min(1.0, net_move / 2.0 + body_consistency / 2.0)
    elif vol_ratio > 1.5 and net_move < 0.5:
        regime = "VOLATILE_CHOPPY"
        conf = min(1.0, vol_ratio / 3.0)
    elif pct < 0.25:
        regime = "QUIET_DRIFT"
        conf = min(1.0, (0.25 - pct) / 0.25)
    else:
        regime = "RANGING"
        conf = 1.0 - abs(net_move) / 2.0

    return {
        "regime": regime,
        "confidence": round(max(0.1, min(1.0, conf)), 2),
        "metrics": {
            "atr_percentile": round(pct, 3),
            "vol_ratio_5_30": round(vol_ratio, 3),
            "net_movement_atr": round(net_move, 3),
            "displacement_count": displacements,
            "body_consistency": round(body_consistency, 3),
        },
    }


# 5-state regime SL/TP multiplier config
REGIME_MULTIPLIERS = {
    "TRENDING_IMPULSIVE": {"sl": 0.9, "tp": 1.3},
    "TRENDING_CORRECTIVE": {"sl": 1.0, "tp": 1.1},
    "RANGING": {"sl": 1.15, "tp": 0.9},
    "VOLATILE_CHOPPY": {"sl": 1.25, "tp": 0.8},
    "QUIET_DRIFT": {"sl": 0.8, "tp": 0.85},
}


def calibrate_volatility(candles: list[dict], timeframe: str,
                         config: dict = None) -> dict:
    """Main entry point: combine all volatility signals.

    Returns dict with atr, ewma_vol, session, session_factor,
    regime, regime_multiplier, and calibrated_vol.
    """
    cfg = config or get_config()

    atr = compute_atr(candles, period=14)
    ewma = compute_ewma_volatility(candles, config=cfg)
    session = detect_session(candles)
    session_factor = get_session_factor(session, config=cfg)
    regime, regime_mult = detect_regime(atr, candles, config=cfg)

    # 5-state structural regime (new)
    structural = classify_regime(candles, atr=atr, config=cfg)
    struct_regime = structural["regime"]
    struct_mults = REGIME_MULTIPLIERS.get(struct_regime,
                                          {"sl": 1.0, "tp": 1.0})

    calibrated = atr * session_factor * regime_mult

    return {
        "atr": round(atr, 4),
        "ewma_vol": round(ewma, 4),
        "session": session,
        "session_factor": session_factor,
        "regime": regime,
        "regime_multiplier": regime_mult,
        "calibrated_vol": round(calibrated, 4),
        # 5-state structural regime
        "structural_regime": struct_regime,
        "structural_regime_confidence": structural["confidence"],
        "structural_regime_metrics": structural["metrics"],
        "structural_sl_multiplier": struct_mults["sl"],
        "structural_tp_multiplier": struct_mults["tp"],
    }


def scale_levels(suggested_sl: float | None, suggested_tp1: float | None,
                 suggested_tp2: float | None, vol_calibration: dict,
                 config: dict = None) -> dict:
    """Scale SL/TP suggestions by volatility calibration ratio.

    SL is scaled outward (wider = more conservative).
    TPs pass through unchanged (consensus engine handles TP blending).
    """
    atr = vol_calibration.get("atr", 0)
    cal_vol = vol_calibration.get("calibrated_vol", 0)

    # If ATR is zero, no scaling possible
    if atr == 0:
        return {
            "scaled_sl": suggested_sl,
            "scaled_tp1": suggested_tp1,
            "scaled_tp2": suggested_tp2,
        }

    ratio = cal_vol / atr  # how much wider/tighter than raw ATR

    scaled_sl = round(suggested_sl * ratio, 2) if suggested_sl is not None else None

    return {
        "scaled_sl": scaled_sl,
        "scaled_tp1": suggested_tp1,  # passthrough
        "scaled_tp2": suggested_tp2,  # passthrough
    }
