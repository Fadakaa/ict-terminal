"""Feature extraction for ICT ML prediction.

Part 1 (lines ~7-215): 32 ATR-normalised features from Claude analysis JSON.
Part 2 (lines ~217+): Raw candle ICT structure detection for WFO.

All functions are pure — no side effects, no mutation of inputs.
"""
import math


def compute_atr(candles: list[dict], period: int = 14) -> float:
    """Calculate Average True Range from OHLC candle data.

    TR = max(H - L, |H - prev_close|, |L - prev_close|)
    ATR = simple moving average of TR over `period` bars.
    """
    if len(candles) < 2:
        return 0.0

    trs = []
    for i in range(1, len(candles)):
        c = candles[i]
        prev_close = candles[i - 1]["close"]
        tr = max(
            c["high"] - c["low"],
            abs(c["high"] - prev_close),
            abs(c["low"] - prev_close),
        )
        trs.append(tr)

    if len(trs) < period:
        return 0.0

    return sum(trs[-period:]) / period


def _encode_strength(strength: str) -> int:
    return {"strong": 3, "moderate": 2, "weak": 1}.get(strength, 0)


def _encode_killzone(killzone: str) -> int:
    kz = killzone.lower()
    if "london" in kz:
        return 1
    if "new york" in kz or "ny" in kz:
        return 2
    if "asian" in kz or "asia" in kz or "tokyo" in kz:
        return 3
    return 0


def _encode_timeframe(tf: str) -> int:
    return {"15min": 1, "1h": 2, "4h": 3, "1day": 4}.get(tf, 0)


def _safe_divide(a: float, b: float) -> float:
    return a / b if b != 0 else 0.0


def _encode_premium_discount(analysis: dict) -> int:
    """Encode premium/discount zone from analysis JSON."""
    raw = (analysis.get("premiumDiscount")
           or analysis.get("premium_discount")
           or (analysis.get("htf_context") or {}).get("premium_discount", ""))
    return {"premium": 1, "discount": -1, "equilibrium": 0}.get(
        raw.lower() if isinstance(raw, str) else "", 0)


def _encode_p3_phase(analysis: dict) -> int:
    """Encode Power of 3 phase from analysis JSON."""
    raw = (analysis.get("powerOf3Phase")
           or analysis.get("power_of_3_phase")
           or (analysis.get("htf_context") or {}).get("power_of_3_phase", ""))
    return {"accumulation": 1, "manipulation": 2, "distribution": 3}.get(
        raw.lower() if isinstance(raw, str) else "", 0)


def _encode_setup_quality(analysis: dict) -> int:
    """Encode setup quality grade from analysis JSON."""
    raw = analysis.get("setup_quality") or analysis.get("setupQuality") or ""
    return {"A": 4, "B": 3, "C": 2, "D": 1, "F": 0}.get(
        raw.upper() if isinstance(raw, str) else "", 0)


def extract_features(analysis: dict, candles: list[dict], timeframe: str,
                     intermarket: dict = None,
                     calibration: dict = None,
                     key_levels: dict = None) -> dict:
    """Extract ML features from an ICT analysis JSON + candle data.

    All price-distance features are normalised by ATR(14) so the model
    generalises across timeframes. Column names match ml/feature_schema.py.

    Args:
        analysis: Claude's ICT analysis JSON (includes htf_context, narrative_state)
        candles: OHLC candles at the execution timeframe
        timeframe: e.g. "15min", "1h", "4h"
        intermarket: Optional intermarket context dict
        calibration: Optional calibration result dict (for Opus narrative)
    """
    atr = compute_atr(candles, 14)
    if atr == 0:
        atr = 1.0  # Prevent division by zero; features will be less meaningful

    obs = analysis.get("orderBlocks") or []
    fvgs = analysis.get("fvgs") or []
    liqs = analysis.get("liquidity") or []
    tps = analysis.get("takeProfits") or []
    entry = analysis.get("entry") or {}
    sl = analysis.get("stopLoss") or {}
    bias = analysis.get("bias", "neutral")
    killzone = analysis.get("killzone", "")
    confluences = analysis.get("confluences") or []

    entry_price = entry.get("price", 0)
    direction = entry.get("direction", "short")
    sl_price = sl.get("price", 0)

    # ── Order Block features (7) ────────────────────────────
    ob_bullish = [ob for ob in obs if ob.get("type") == "bullish"]
    ob_bearish = [ob for ob in obs if ob.get("type") == "bearish"]
    ob_strengths = [_encode_strength(ob.get("strength", "")) for ob in obs]

    ob_sizes = [(ob["high"] - ob["low"]) for ob in obs if "high" in ob and "low" in ob]

    ob_distances = []
    if entry_price:
        for ob in obs:
            mid = (ob.get("high", 0) + ob.get("low", 0)) / 2
            ob_distances.append(abs(entry_price - mid))

    # Alignment: does the nearest OB match the bias?
    ob_alignment = 0
    if bias == "bullish" and len(ob_bullish) > 0:
        ob_alignment = 1
    elif bias == "bearish" and len(ob_bearish) > 0:
        ob_alignment = 1

    # ── FVG features (5) ────────────────────────────────────
    unfilled_fvgs = [f for f in fvgs if not f.get("filled", True)]

    fvg_distances = []
    if entry_price:
        for f in unfilled_fvgs:
            mid = (f.get("high", 0) + f.get("low", 0)) / 2
            fvg_distances.append(abs(entry_price - mid))

    fvg_sizes = [(f["high"] - f["low"]) for f in fvgs if "high" in f and "low" in f]

    fvg_alignment = 0
    if unfilled_fvgs:
        for f in unfilled_fvgs:
            if f.get("type") == bias:
                fvg_alignment = 1
                break

    # ── Liquidity features (4) ──────────────────────────────
    bsl = [l for l in liqs if l.get("type") == "buyside"]
    ssl = [l for l in liqs if l.get("type") == "sellside"]

    # Target = liquidity in trade direction; threat = opposite
    if direction == "long":
        target_prices = [l["price"] for l in bsl if "price" in l]
        threat_prices = [l["price"] for l in ssl if "price" in l]
    else:
        target_prices = [l["price"] for l in ssl if "price" in l]
        threat_prices = [l["price"] for l in bsl if "price" in l]

    liq_target_dist = min(abs(entry_price - p) for p in target_prices) if entry_price and target_prices else 0
    liq_threat_dist = min(abs(entry_price - p) for p in threat_prices) if entry_price and threat_prices else 0

    # ── Trade setup features (6) ─────────────────────────────
    rr1 = tps[0].get("rr", 0) if len(tps) > 0 else 0
    rr2 = tps[1].get("rr", 0) if len(tps) > 1 else 0
    sl_dist = abs(entry_price - sl_price) if entry_price and sl_price else 0
    tp1_dist = abs(entry_price - tps[0]["price"]) if len(tps) > 0 and entry_price else 0
    dir_encoded = 1 if direction == "long" else 0
    bias_match = 1 if (bias == "bullish" and direction == "long") or (bias == "bearish" and direction == "short") else 0

    # ── Confluence features (4) ──────────────────────────────
    has_ob_fvg_overlap = 0
    for c in confluences:
        cl = c.lower()
        if ("ob" in cl or "order block" in cl) and ("fvg" in cl or "fair value" in cl or "overlap" in cl):
            has_ob_fvg_overlap = 1
            break

    # ── Price action context (6) ─────────────────────────────
    closes = [c["close"] for c in candles]

    # SMA 20
    sma20 = sum(closes[-20:]) / 20 if len(closes) >= 20 else (sum(closes) / len(closes) if closes else 0)
    price_vs_sma = _safe_divide(closes[-1] - sma20, atr) if closes else 0

    # Volatility ratio: ATR(5) / ATR(14)
    atr5 = compute_atr(candles, 5) if len(candles) > 6 else atr
    vol_ratio = _safe_divide(atr5, atr)

    # Last candle body
    last_body = abs(candles[-1]["close"] - candles[-1]["open"]) if candles else 0

    # Trend strength
    trend = _safe_divide(closes[-1] - closes[-20], atr) if len(closes) >= 20 else 0

    # Session hour
    session_hour = 0
    if candles:
        dt_str = candles[-1].get("datetime", "")
        try:
            session_hour = int(dt_str.split(" ")[1].split(":")[0])
        except (IndexError, ValueError):
            session_hour = 0

    return {
        # Order Blocks (7)
        "ob_count": len(obs),
        "ob_bullish_count": len(ob_bullish),
        "ob_bearish_count": len(ob_bearish),
        "ob_strongest_strength": max(ob_strengths) if ob_strengths else 0,
        "ob_nearest_distance_atr": _safe_divide(min(ob_distances) if ob_distances else 0, atr),
        "ob_avg_size_atr": _safe_divide(sum(ob_sizes) / len(ob_sizes) if ob_sizes else 0, atr),
        "ob_alignment": ob_alignment,
        # FVGs (5)
        "fvg_count": len(fvgs),
        "fvg_unfilled_count": len(unfilled_fvgs),
        "fvg_nearest_distance_atr": _safe_divide(min(fvg_distances) if fvg_distances else 0, atr),
        "fvg_avg_size_atr": _safe_divide(sum(fvg_sizes) / len(fvg_sizes) if fvg_sizes else 0, atr),
        "fvg_alignment": fvg_alignment,
        # Liquidity (4)
        "liq_buyside_count": len(bsl),
        "liq_sellside_count": len(ssl),
        "liq_nearest_target_distance_atr": _safe_divide(liq_target_dist, atr),
        "liq_nearest_threat_distance_atr": _safe_divide(liq_threat_dist, atr),
        # Trade Setup (6)
        "risk_reward_tp1": rr1,
        "risk_reward_tp2": rr2,
        "sl_distance_atr": _safe_divide(sl_dist, atr),
        "tp1_distance_atr": _safe_divide(tp1_dist, atr),
        "entry_direction": dir_encoded,
        "bias_direction_match": bias_match,
        # Confluence (4)
        "num_confluences": len(confluences),
        "has_ob_fvg_overlap": has_ob_fvg_overlap,
        "killzone_encoded": _encode_killzone(killzone),
        "timeframe_encoded": _encode_timeframe(timeframe),
        # Price Action (6)
        "atr_14": round(atr, 4),
        "price_vs_20sma": round(price_vs_sma, 4),
        "recent_volatility_ratio": round(vol_ratio, 4),
        "last_candle_body_atr": round(_safe_divide(last_body, atr), 4),
        "trend_strength": round(trend, 4),
        "session_hour": session_hour,
        # ICT Context from Claude analysis (4)
        "premium_discount_encoded": _encode_premium_discount(analysis),
        "p3_phase_encoded": _encode_p3_phase(analysis),
        "setup_quality_encoded": _encode_setup_quality(analysis),
        "claude_direction_encoded": dir_encoded,
        # Intermarket (4) — pass through from intermarket module, NaN if unavailable
        "gold_dxy_corr_20": _intermarket_or_nan(intermarket, "gold_dxy_corr_20"),
        "gold_dxy_diverging": 1 if (intermarket or {}).get("gold_dxy_diverging") else 0,
        "dxy_range_position": _intermarket_or_nan(intermarket, "dxy_range_position"),
        "yield_direction": _intermarket_or_nan(intermarket, "yield_direction"),
        # Regime — populated by classify_regime() in scanner, NaN if not yet set
        "volatility_regime": float('nan'),
        # Entry zone placement — computed from OB data + entry price
        **_compute_entry_zone(entry_price, direction, obs, atr,
                              fallback_position=analysis.get("entry_zone_position"),
                              fallback_size=analysis.get("entry_zone_size_atr")),
        # HTF context from Claude's analysis (5)
        **_extract_htf_features(analysis, calibration),
        # Narrative state (4)
        **_extract_narrative_features(analysis, calibration),
        # Key level proximity (6) — ATR-normalised distance to ICT levels
        **_extract_key_level_features(entry_price, atr, key_levels),
    }


def _intermarket_or_nan(intermarket: dict | None, key: str) -> float:
    """Return intermarket value or NaN if absent/default.

    The intermarket module returns 0.0 as default for corr and yield,
    and 0.5 for dxy_range_position. These defaults are indistinguishable
    from real values so we pass them through — but if intermarket is None
    (no data fetched at all), we use NaN to signal truly missing data.
    """
    if not intermarket:
        return float('nan')
    val = intermarket.get(key)
    if val is None:
        return float('nan')
    return float(val)


def _extract_htf_features(analysis: dict, calibration: dict | None) -> dict:
    """Extract 5 HTF context features from Claude's analysis JSON.

    Fields sourced from analysis["htf_context"]:
      - htf_bias_encoded: bullish=1, bearish=-1, neutral=0
      - htf_sweep_encoded: bsl=1 (bullish signal after SSL), ssl=-1, none=0
      - dealing_range_position: 0.0=range low, 1.0=range high
      - htf_structure_alignment: 1=HTF+LTF agree, 0=neutral, -1=conflict
      - htf_displacement_quality: 1=strong displacement formed OB, 0=weak/none
    """
    htf = analysis.get("htf_context") or {}
    structure = analysis.get("structure") or {}
    entry = analysis.get("entry") or {}
    direction = entry.get("direction", "short")

    # htf_bias_encoded
    htf_bias_raw = htf.get("htf_bias", "")
    htf_bias = {"bullish": 1, "bearish": -1, "neutral": 0}.get(
        htf_bias_raw.lower() if isinstance(htf_bias_raw, str) else "", 0)

    # htf_sweep_encoded — bsl sweep is bullish signal (liquidity taken above),
    # ssl sweep is bearish signal (liquidity taken below)
    sweep_raw = htf.get("recent_sweep", "")
    htf_sweep = {"bsl": 1, "ssl": -1, "none": 0}.get(
        sweep_raw.lower() if isinstance(sweep_raw, str) else "", 0)

    # dealing_range_position — price position within HTF dealing range
    dr_high = htf.get("dealing_range_high")
    dr_low = htf.get("dealing_range_low")
    entry_price = entry.get("price", 0)
    if dr_high and dr_low and dr_high > dr_low and entry_price:
        dr_pos = (entry_price - dr_low) / (dr_high - dr_low)
        dr_pos = max(0.0, min(1.0, dr_pos))
    else:
        dr_pos = float('nan')

    # htf_structure_alignment — does HTF bias agree with LTF trade direction?
    # 1 = agree, -1 = conflict, 0 = neutral/unknown
    if htf_bias_raw and htf_bias_raw.lower() != "neutral":
        htf_dir = "long" if htf_bias_raw.lower() == "bullish" else "short"
        htf_align = 1 if htf_dir == direction else -1
    else:
        htf_align = 0

    # htf_displacement_quality — did a strong displacement form an OB?
    # Check structure for BOS/CHoCH + any strong OB in the analysis
    has_structure_break = structure.get("type", "none").lower() in ("bos", "choch")
    obs = analysis.get("orderBlocks") or []
    has_strong_ob = any(ob.get("strength", "").lower() == "strong" for ob in obs)
    htf_disp = 1 if (has_structure_break and has_strong_ob) else 0

    return {
        "htf_bias_encoded": htf_bias,
        "htf_sweep_encoded": htf_sweep,
        "dealing_range_position": round(dr_pos, 4) if not math.isnan(dr_pos) else dr_pos,
        "htf_structure_alignment": htf_align,
        "htf_displacement_quality": htf_disp,
    }


def _extract_narrative_features(analysis: dict, calibration: dict | None) -> dict:
    """Extract 4 narrative state features.

    Fields sourced from:
      - analysis["narrative_state"] for thesis_confidence, p3_progress, scan_count
      - calibration["opus_narrative"] for opus_sonnet_agreement
    """
    ns = analysis.get("narrative_state") or {}
    cal = calibration or {}
    opus = cal.get("opus_narrative") or {}

    # thesis_confidence — 0.0 to 1.0 from bias_confidence
    conf = ns.get("bias_confidence")
    if conf is not None:
        try:
            thesis_conf = max(0.0, min(1.0, float(conf)))
        except (ValueError, TypeError):
            thesis_conf = float('nan')
    else:
        thesis_conf = float('nan')

    # p3_progress_encoded — early=1, mid=2, late=3, none=0
    p3_prog_raw = ns.get("p3_progress", "")
    p3_prog = {"early": 1, "mid": 2, "late": 3, "none": 0}.get(
        p3_prog_raw.lower() if isinstance(p3_prog_raw, str) else "", 0)

    # thesis_scan_count — how many scans this thesis has survived
    scan_count = ns.get("scan_count")
    if scan_count is not None:
        try:
            scan_count = int(scan_count)
        except (ValueError, TypeError):
            scan_count = 0
    else:
        scan_count = 0

    # opus_sonnet_agreement — does Opus HTF bias agree with Sonnet's trade direction?
    entry = analysis.get("entry") or {}
    sonnet_dir = entry.get("direction", "")
    opus_bias = opus.get("directional_bias", "")
    if opus_bias and sonnet_dir:
        opus_dir = "long" if opus_bias.lower() == "bullish" else (
            "short" if opus_bias.lower() == "bearish" else "")
        agreement = 1 if (opus_dir and opus_dir == sonnet_dir) else 0
    else:
        agreement = 0

    return {
        "thesis_confidence": thesis_conf,
        "p3_progress_encoded": p3_prog,
        "thesis_scan_count": scan_count,
        "opus_sonnet_agreement": agreement,
    }


def _extract_key_level_features(entry_price: float, atr: float,
                                 key_levels: dict | None) -> dict:
    """Extract 6 ATR-normalised key level proximity features.

    Each feature = (entry_price - level) / ATR.
    Positive = price above level, negative = below.
    NaN when level or entry price unavailable.
    """
    nan = float('nan')
    result = {
        "price_vs_pdh_atr": nan,
        "price_vs_pdl_atr": nan,
        "price_vs_pwh_atr": nan,
        "price_vs_pwl_atr": nan,
        "price_vs_asia_high_atr": nan,
        "price_vs_asia_low_atr": nan,
    }

    if not key_levels or not entry_price or atr <= 0:
        return result

    level_map = {
        "price_vs_pdh_atr": "pdh",
        "price_vs_pdl_atr": "pdl",
        "price_vs_pwh_atr": "pwh",
        "price_vs_pwl_atr": "pwl",
        "price_vs_asia_high_atr": "asia_high",
        "price_vs_asia_low_atr": "asia_low",
    }

    for feature_name, level_key in level_map.items():
        level = key_levels.get(level_key)
        if level is not None:
            result[feature_name] = round((entry_price - level) / atr, 4)

    return result


def _compute_entry_zone(entry_price: float, direction: str,
                        order_blocks: list, atr: float,
                        fallback_position: float = None,
                        fallback_size: float = None) -> dict:
    """Compute entry zone position and size from OB data.

    Matches entry price to the nearest OB in the trade direction:
    - For longs: nearest bullish OB (or any OB below entry)
    - For shorts: nearest bearish OB (or any OB above entry)

    Falls back to pre-computed values from scanner enrichment when OB data
    is missing or entry price is unavailable.

    Returns:
        entry_zone_position: 0.0 (shallow/edge) to 1.0 (deep/optimal)
        entry_zone_size_atr: zone height in ATR units
    """
    result = {"entry_zone_position": float('nan'), "entry_zone_size_atr": float('nan')}

    if not entry_price or not order_blocks or atr <= 0:
        # Allow pre-computed entry zone from scanner enrichment
        if fallback_position is not None:
            result["entry_zone_position"] = round(float(fallback_position), 4)
        if fallback_size is not None:
            result["entry_zone_size_atr"] = round(float(fallback_size), 4)
        return result

    # Find the OB that contains or is nearest to entry price
    best_ob = None
    best_dist = float('inf')

    for ob in order_blocks:
        ob_high = ob.get("high", 0)
        ob_low = ob.get("low", 0)
        if not ob_high or not ob_low or ob_high <= ob_low:
            continue

        ob_mid = (ob_high + ob_low) / 2
        dist = abs(entry_price - ob_mid)

        # Prefer OBs that match trade direction
        ob_type = ob.get("type", "")
        direction_match = (
            (direction == "long" and ob_type == "bullish") or
            (direction == "short" and ob_type == "bearish")
        )
        # Give direction-matching OBs a distance bonus
        effective_dist = dist * 0.5 if direction_match else dist

        if effective_dist < best_dist:
            best_dist = effective_dist
            best_ob = ob

    if best_ob:
        ob_high = best_ob["high"]
        ob_low = best_ob["low"]
        ob_size = ob_high - ob_low

        # Position within the OB: 0.0 = edge (shallow), 1.0 = optimal (deep)
        if direction == "long":
            # For longs, deeper into OB = closer to low = better
            pos = (ob_high - entry_price) / ob_size if ob_size > 0 else 0.5
        else:
            # For shorts, deeper into OB = closer to high = better
            pos = (entry_price - ob_low) / ob_size if ob_size > 0 else 0.5

        # Clamp to 0-1 (entry might be slightly outside OB)
        pos = max(0.0, min(1.0, pos))

        result["entry_zone_position"] = round(pos, 4)
        result["entry_zone_size_atr"] = round(ob_size / atr, 4)

    return result


def classify_setup_type(analysis: dict, candles: list[dict] = None,
                        timeframe: str = "1h") -> str:
    """Derive WFO-compatible setup type taxonomy from Claude analysis JSON.

    Produces the same ``{bull|bear}_{'_'.join(sorted(tags))}`` format used
    by the WFO engine's internal ``_classify_setup_type()`` so that
    prediction-time setup types can be matched against WFO backtested stats.

    Tags checked: ob, fvg, structure, displacement, sweep, london, ny_am.
    """
    entry = analysis.get("entry") or {}
    direction = (entry.get("direction") or "").lower()
    prefix = "bull" if direction == "long" else "bear"

    tags = []

    # OB present
    if analysis.get("orderBlocks"):
        tags.append("ob")

    # FVG present
    if analysis.get("fvgs"):
        tags.append("fvg")

    # Confluences — support both dict (new format) and list (legacy)
    confluences = analysis.get("confluences", {})
    if isinstance(confluences, dict):
        if confluences.get("structureAlignment") or confluences.get("marketStructure"):
            tags.append("structure")
        if confluences.get("displacement"):
            tags.append("displacement")
        if confluences.get("liquiditySweep"):
            tags.append("sweep")

    # Killzone
    kz = (analysis.get("killzone") or "").lower()
    if "london" in kz:
        tags.append("london")
    elif "ny" in kz or "new_york" in kz or "new york" in kz:
        tags.append("ny_am")

    if not tags:
        return f"{prefix}_unclassified"
    return f"{prefix}_{'_'.join(sorted(tags))}"


# ═══════════════════════════════════════════════════════════════════════
# Part 2 — Raw candle ICT structure detection (for WFO, no Claude JSON)
# ═══════════════════════════════════════════════════════════════════════


def detect_order_blocks(candles: list[dict], atr: float,
                        displacement_threshold: float = 1.5) -> list[dict]:
    """Detect order blocks from raw OHLC candle data.

    An OB forms when a displacement candle (body > threshold * ATR) occurs.
    The candle immediately before the displacement is marked as the OB.

    Returns list of dicts: {type, high, low, index, body_size}
    """
    if len(candles) < 2 or atr <= 0:
        return []

    obs = []
    threshold = displacement_threshold * atr

    for i in range(1, len(candles)):
        c = candles[i]
        body = abs(c["close"] - c["open"])
        if body <= threshold:
            continue

        prev = candles[i - 1]
        if c["close"] > c["open"]:
            # Bullish displacement → preceding candle is bullish OB
            obs.append({
                "type": "bullish",
                "high": prev["high"],
                "low": prev["low"],
                "index": i - 1,
                "body_size": prev["high"] - prev["low"],
            })
        else:
            # Bearish displacement → preceding candle is bearish OB
            obs.append({
                "type": "bearish",
                "high": prev["high"],
                "low": prev["low"],
                "index": i - 1,
                "body_size": prev["high"] - prev["low"],
            })

    return obs


def detect_fvgs(candles: list[dict]) -> list[dict]:
    """Detect Fair Value Gaps (3-candle imbalances) from raw OHLC data.

    Bullish FVG: candle[i].low > candle[i-2].high  (gap up)
    Bearish FVG: candle[i].high < candle[i-2].low  (gap down)

    Returns list of dicts: {type, high, low, index, size}
    """
    if len(candles) < 3:
        return []

    gaps = []
    for i in range(2, len(candles)):
        # Bullish FVG: current low above the high from 2 candles ago
        if candles[i]["low"] > candles[i - 2]["high"]:
            gaps.append({
                "type": "bullish",
                "high": candles[i]["low"],
                "low": candles[i - 2]["high"],
                "index": i - 1,
                "size": candles[i]["low"] - candles[i - 2]["high"],
            })
        # Bearish FVG: current high below the low from 2 candles ago
        if candles[i]["high"] < candles[i - 2]["low"]:
            gaps.append({
                "type": "bearish",
                "high": candles[i - 2]["low"],
                "low": candles[i]["high"],
                "index": i - 1,
                "size": candles[i - 2]["low"] - candles[i]["high"],
            })

    return gaps


def detect_liquidity(candles: list[dict], window: int = 20) -> list[dict]:
    """Detect liquidity levels (swing highs = BSL, swing lows = SSL).

    A swing high at index i requires high[i] > all highs within window
    on both sides. Same logic for swing lows.

    Returns list of dicts: {type: "buyside"|"sellside", price, index}
    """
    if len(candles) < 3:
        return []

    levels = []
    # Use smaller effective window if data is limited
    eff_window = min(window, len(candles) // 3)
    if eff_window < 1:
        eff_window = 1

    for i in range(eff_window, len(candles) - eff_window):
        left_start = max(0, i - eff_window)
        right_end = min(len(candles), i + eff_window + 1)

        left_highs = [candles[j]["high"] for j in range(left_start, i)]
        right_highs = [candles[j]["high"] for j in range(i + 1, right_end)]

        if left_highs and right_highs:
            if candles[i]["high"] > max(left_highs) and candles[i]["high"] > max(right_highs):
                levels.append({
                    "type": "buyside",
                    "price": candles[i]["high"],
                    "index": i,
                })

        left_lows = [candles[j]["low"] for j in range(left_start, i)]
        right_lows = [candles[j]["low"] for j in range(i + 1, right_end)]

        if left_lows and right_lows:
            if candles[i]["low"] < min(left_lows) and candles[i]["low"] < min(right_lows):
                levels.append({
                    "type": "sellside",
                    "price": candles[i]["low"],
                    "index": i,
                })

    return levels


def compute_market_structure(candles: list[dict], lookback: int = 20) -> float:
    """Score market structure from raw candles.

    Counts higher-highs/higher-lows (+1 each) vs lower-highs/lower-lows (-1)
    over the last `lookback` candles. Returns normalized score in [-1, 1].
    """
    if len(candles) < 2:
        return 0.0

    n = min(lookback, len(candles))
    subset = candles[-n:]

    score = 0
    comparisons = 0
    for i in range(1, len(subset)):
        # Higher high vs lower high
        if subset[i]["high"] > subset[i - 1]["high"]:
            score += 1
        elif subset[i]["high"] < subset[i - 1]["high"]:
            score -= 1
        # Higher low vs lower low
        if subset[i]["low"] > subset[i - 1]["low"]:
            score += 1
        elif subset[i]["low"] < subset[i - 1]["low"]:
            score -= 1
        comparisons += 2

    if comparisons == 0:
        return 0.0
    return max(-1.0, min(1.0, score / comparisons))


def create_trade_labels(candles: list[dict], entry_idx: int,
                        direction: str, atr: float,
                        sl_atr_mult: float = 1.5,
                        tp_atr_mults: list[float] = None,
                        max_bars: int = 50) -> dict:
    """Simulate a trade forward from entry_idx and return outcome labels.

    Args:
        candles: full candle array
        entry_idx: index of entry candle
        direction: "long" or "short"
        atr: current ATR for SL/TP sizing
        sl_atr_mult: SL distance in ATR multiples
        tp_atr_mults: TP distances in ATR multiples (default [1.0, 2.0, 3.5])
        max_bars: max bars to hold before expiry

    Returns dict: {outcome, max_drawdown_atr, max_favorable_atr, bars_held, won}
    """
    if tp_atr_mults is None:
        tp_atr_mults = [1.0, 2.0, 3.5]

    if atr <= 0 or entry_idx >= len(candles) - 1:
        return {
            "outcome": "expired",
            "max_drawdown_atr": 0.0,
            "max_favorable_atr": 0.0,
            "bars_held": 0,
            "won": False,
        }

    entry_price = candles[entry_idx]["close"]
    is_long = direction == "long"

    if is_long:
        sl_price = entry_price - sl_atr_mult * atr
        tp_prices = [entry_price + m * atr for m in tp_atr_mults]
    else:
        sl_price = entry_price + sl_atr_mult * atr
        tp_prices = [entry_price - m * atr for m in tp_atr_mults]

    max_favorable = 0.0
    max_adverse = 0.0
    highest_tp_hit = -1
    outcome = "expired"
    bars_held = 0

    end_idx = min(entry_idx + max_bars + 1, len(candles))
    for i in range(entry_idx + 1, end_idx):
        c = candles[i]
        bars_held = i - entry_idx

        if is_long:
            favorable = c["high"] - entry_price
            adverse = entry_price - c["low"]
        else:
            favorable = entry_price - c["low"]
            adverse = c["high"] - entry_price

        max_favorable = max(max_favorable, favorable)
        max_adverse = max(max_adverse, adverse)

        # Check SL
        if is_long and c["low"] <= sl_price:
            outcome = "stopped_out"
            break
        if not is_long and c["high"] >= sl_price:
            outcome = "stopped_out"
            break

        # Check TPs (highest hit wins)
        for tp_idx, tp_price in enumerate(tp_prices):
            if is_long and c["high"] >= tp_price:
                highest_tp_hit = max(highest_tp_hit, tp_idx)
            if not is_long and c["low"] <= tp_price:
                highest_tp_hit = max(highest_tp_hit, tp_idx)

    if outcome != "stopped_out" and highest_tp_hit >= 0:
        outcome = f"tp{highest_tp_hit + 1}_hit"

    won = outcome in {"tp1_hit", "tp2_hit", "tp3_hit"}

    return {
        "outcome": outcome,
        "max_drawdown_atr": round(_safe_divide(max_adverse, atr), 4),
        "max_favorable_atr": round(_safe_divide(max_favorable, atr), 4),
        "bars_held": bars_held,
        "won": won,
    }


def _extract_hour(dt_str: str) -> int:
    """Extract UTC hour from a datetime string (handles T-separated and space-separated)."""
    if not dt_str:
        return 0
    try:
        time_part = dt_str.split("T")[-1] if "T" in dt_str else dt_str.split(" ")[-1]
        return int(time_part.split(":")[0])
    except (ValueError, IndexError):
        return 0


def engineer_features_from_candles(candles: list[dict], idx: int,
                                   direction: str, atr: float,
                                   obs: list[dict], fvgs_list: list[dict],
                                   liqs: list[dict],
                                   ms_score: float) -> dict:
    """Extract ML features from raw candle data at a specific index.

    Produces a feature dict compatible with WFO training (38 features).
    Unlike extract_features() which works on Claude analysis JSON,
    this operates directly on detected ICT structures.
    """
    safe_atr = atr if atr > 0 else 1.0
    price = candles[idx]["close"]
    is_long = direction == "long"

    # ── OB features (7) ──────────────────────────────────────
    # Only consider OBs that exist before current index
    active_obs = [ob for ob in obs if ob["index"] < idx]
    bull_obs = [ob for ob in active_obs if ob["type"] == "bullish"]
    bear_obs = [ob for ob in active_obs if ob["type"] == "bearish"]

    ob_distances = [abs(price - (ob["high"] + ob["low"]) / 2) for ob in active_obs] if active_obs else []
    ob_sizes = [ob["body_size"] for ob in active_obs] if active_obs else []

    ob_alignment = 0
    if is_long and bull_obs:
        ob_alignment = 1
    elif not is_long and bear_obs:
        ob_alignment = 1

    max_ob_size = max(ob_sizes) if ob_sizes else 0

    # ── FVG features (5) ─────────────────────────────────────
    active_fvgs = [f for f in fvgs_list if f["index"] < idx]
    fvg_distances = [abs(price - (f["high"] + f["low"]) / 2) for f in active_fvgs] if active_fvgs else []
    fvg_sizes = [f["size"] for f in active_fvgs] if active_fvgs else []

    fvg_alignment = 0
    matching_type = "bullish" if is_long else "bearish"
    if any(f["type"] == matching_type for f in active_fvgs):
        fvg_alignment = 1

    # ── Liquidity features (4) ────────────────────────────────
    active_liqs = [lq for lq in liqs if lq["index"] < idx]
    bsl_levels = [lq for lq in active_liqs if lq["type"] == "buyside"]
    ssl_levels = [lq for lq in active_liqs if lq["type"] == "sellside"]

    if is_long:
        target_prices = [lq["price"] for lq in bsl_levels]
        threat_prices = [lq["price"] for lq in ssl_levels]
    else:
        target_prices = [lq["price"] for lq in ssl_levels]
        threat_prices = [lq["price"] for lq in bsl_levels]

    target_dist = min(abs(price - p) for p in target_prices) if target_prices else 0
    threat_dist = min(abs(price - p) for p in threat_prices) if threat_prices else 0

    # ── Market structure (2) ──────────────────────────────────
    trend_strength = abs(ms_score)

    # ── Trade setup (4) ──────────────────────────────────────
    entry_body = abs(candles[idx]["close"] - candles[idx]["open"])
    closes = [candles[j]["close"] for j in range(max(0, idx - 20), idx + 1)]
    sma20 = sum(closes) / len(closes) if closes else price
    price_vs_sma = _safe_divide(price - sma20, safe_atr)

    atr5 = compute_atr(candles[max(0, idx - 6):idx + 1], 5) if idx >= 6 else safe_atr
    vol_ratio = _safe_divide(atr5, safe_atr)

    # ── Session/timing (3) ────────────────────────────────────
    dt_str = candles[idx].get("datetime", "")
    session_hour = _extract_hour(dt_str)

    kz = 0
    if 7 <= session_hour < 10:
        kz = 1  # London
    elif 13 <= session_hour < 16:
        kz = 2  # NY AM
    elif 16 <= session_hour < 19:
        kz = 3  # NY PM
    elif 0 <= session_hour < 4:
        kz = 4  # Asian

    dow = 0
    try:
        from datetime import datetime as _dt
        parsed = _dt.fromisoformat(dt_str.replace("Z", "+00:00"))
        dow = parsed.weekday()
    except Exception:
        pass

    # ── Price action (6) ──────────────────────────────────────
    # RSI(14) approximation
    gains, losses_list = [], []
    rsi_start = max(0, idx - 14)
    for j in range(rsi_start + 1, idx + 1):
        delta = candles[j]["close"] - candles[j - 1]["close"]
        if delta > 0:
            gains.append(delta)
            losses_list.append(0)
        else:
            gains.append(0)
            losses_list.append(abs(delta))

    if gains:
        avg_gain = sum(gains) / len(gains)
        avg_loss = sum(losses_list) / len(losses_list)
        if avg_loss > 0:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        else:
            rsi = 100.0
    else:
        rsi = 50.0

    # Momentum (sum of last 5 close-to-close changes / ATR)
    momentum = 0.0
    mom_start = max(0, idx - 5)
    if idx > mom_start:
        momentum = _safe_divide(candles[idx]["close"] - candles[mom_start]["close"], safe_atr)

    # Range position: where close sits in the candle range
    c = candles[idx]
    candle_range = c["high"] - c["low"]
    range_pos = _safe_divide(c["close"] - c["low"], candle_range) if candle_range > 0 else 0.5
    body_range_ratio = _safe_divide(entry_body, candle_range) if candle_range > 0 else 0
    upper_wick = _safe_divide(c["high"] - max(c["open"], c["close"]), candle_range) if candle_range > 0 else 0

    # ── Volatility (7) ────────────────────────────────────────
    # 5-bar std of close-to-close returns / ATR
    recent_returns = []
    for j in range(max(1, idx - 5), idx + 1):
        recent_returns.append(abs(candles[j]["close"] - candles[j - 1]["close"]))
    std_5 = 0.0
    if len(recent_returns) >= 2:
        mean_r = sum(recent_returns) / len(recent_returns)
        std_5 = math.sqrt(sum((r - mean_r) ** 2 for r in recent_returns) / len(recent_returns))

    # Displacement count in last 10 bars
    disp_count = 0
    for j in range(max(0, idx - 10), idx + 1):
        if abs(candles[j]["close"] - candles[j]["open"]) > 1.5 * safe_atr:
            disp_count += 1

    # FVG count in last 10 bars
    recent_fvg_count = sum(1 for f in fvgs_list if idx - 10 <= f["index"] <= idx)

    # Vol regime proxy: 5-bar vol / 30-bar vol
    vol_30_returns = []
    for j in range(max(1, idx - 30), idx + 1):
        vol_30_returns.append(abs(candles[j]["close"] - candles[j - 1]["close"]))
    std_30 = 0.0
    if len(vol_30_returns) >= 2:
        mean_30 = sum(vol_30_returns) / len(vol_30_returns)
        std_30 = math.sqrt(sum((r - mean_30) ** 2 for r in vol_30_returns) / len(vol_30_returns))
    regime_proxy = _safe_divide(std_5, std_30)

    return {
        # OB (7)
        "ob_count": len(active_obs),
        "ob_bullish_count": len(bull_obs),
        "ob_bearish_count": len(bear_obs),
        "ob_nearest_distance_atr": round(_safe_divide(min(ob_distances) if ob_distances else 0, safe_atr), 4),
        "ob_avg_size_atr": round(_safe_divide(sum(ob_sizes) / len(ob_sizes) if ob_sizes else 0, safe_atr), 4),
        "ob_alignment": ob_alignment,
        "ob_max_size_atr": round(_safe_divide(max_ob_size, safe_atr), 4),
        # FVG (5)
        "fvg_count": len(active_fvgs),
        "fvg_nearest_distance_atr": round(_safe_divide(min(fvg_distances) if fvg_distances else 0, safe_atr), 4),
        "fvg_avg_size_atr": round(_safe_divide(sum(fvg_sizes) / len(fvg_sizes) if fvg_sizes else 0, safe_atr), 4),
        "fvg_alignment": fvg_alignment,
        "fvg_recent_count": recent_fvg_count,
        # Liquidity (4)
        "liq_buyside_count": len(bsl_levels),
        "liq_sellside_count": len(ssl_levels),
        "liq_target_distance_atr": round(_safe_divide(target_dist, safe_atr), 4),
        "liq_threat_distance_atr": round(_safe_divide(threat_dist, safe_atr), 4),
        # Market structure (2)
        "ms_score": round(ms_score, 4),
        "ms_trend_strength": round(trend_strength, 4),
        # Trade setup (4)
        "entry_direction": 1 if is_long else 0,
        "entry_body_atr": round(_safe_divide(entry_body, safe_atr), 4),
        "price_vs_20sma": round(price_vs_sma, 4),
        "recent_volatility_ratio": round(vol_ratio, 4),
        # Session/timing (3)
        "session_hour": session_hour,
        "killzone_encoded": kz,
        "day_of_week": dow,
        # Price action (6)
        "rsi_14": round(rsi, 2),
        "momentum_5": round(momentum, 4),
        "range_position": round(range_pos, 4),
        "body_range_ratio": round(body_range_ratio, 4),
        "upper_wick_ratio": round(upper_wick, 4),
        "last_candle_body_atr": round(_safe_divide(entry_body, safe_atr), 4),
        # Volatility (7)
        "atr_14": round(safe_atr, 4),
        "vol_std_5_atr": round(_safe_divide(std_5, safe_atr), 4),
        "vol_regime_proxy": round(regime_proxy, 4),
        "displacement_count_10": disp_count,
        "candle_range_atr": round(_safe_divide(candle_range, safe_atr), 4),
        "ewma_vol_5": round(sum(recent_returns) / len(recent_returns) if recent_returns else 0, 4),
        "vol_ratio_5_14": round(vol_ratio, 4),
    }


# ═══════════════════════════════════════════════════════════════════════
# Part 3 — Quality scoring helpers + HTF feature engineering
# ═══════════════════════════════════════════════════════════════════════


def detect_swing_points(candles: list[dict], lookback: int = 5) -> list[dict]:
    """Detect swing highs and swing lows for BOS/ChoCH and sweep detection.

    A swing high at index i: high[i] > all highs within lookback on both sides.
    A swing low at index i: low[i] < all lows within lookback on both sides.

    Returns list of dicts: {type: "high"|"low", price, index}
    """
    if len(candles) < 2 * lookback + 1:
        return []

    swings = []
    for i in range(lookback, len(candles) - lookback):
        left_start = max(0, i - lookback)

        # Check swing high
        is_high = True
        for j in range(left_start, i):
            if candles[j]["high"] >= candles[i]["high"]:
                is_high = False
                break
        if is_high:
            for j in range(i + 1, min(len(candles), i + lookback + 1)):
                if candles[j]["high"] >= candles[i]["high"]:
                    is_high = False
                    break
        if is_high:
            swings.append({"type": "high", "price": candles[i]["high"], "index": i})

        # Check swing low
        is_low = True
        for j in range(left_start, i):
            if candles[j]["low"] <= candles[i]["low"]:
                is_low = False
                break
        if is_low:
            for j in range(i + 1, min(len(candles), i + lookback + 1)):
                if candles[j]["low"] <= candles[i]["low"]:
                    is_low = False
                    break
        if is_low:
            swings.append({"type": "low", "price": candles[i]["low"], "index": i})

    return swings


def compute_ob_freshness(candles: list[dict], ob: dict, current_idx: int) -> int:
    """Count how many candles have wicked into the OB range since creation.

    Returns count of retests (0 = untested/first touch).
    """
    retests = 0
    for i in range(ob["index"] + 1, min(current_idx, len(candles))):
        if candles[i]["low"] <= ob["high"] and candles[i]["high"] >= ob["low"]:
            retests += 1
    return retests


def compute_fvg_fill_percentage(candles: list[dict], fvg: dict,
                                 current_idx: int) -> float:
    """Compute what percentage of an FVG has been filled by subsequent price action.

    Returns 0.0 (unfilled) to 1.0 (fully filled).
    """
    fvg_range = fvg["high"] - fvg["low"]
    if fvg_range <= 0:
        return 1.0

    max_fill = 0.0
    for i in range(fvg["index"] + 1, min(current_idx + 1, len(candles))):
        if fvg["type"] == "bullish":
            # Bullish FVG filled from above — price drops into gap
            if candles[i]["low"] <= fvg["high"]:
                fill = (fvg["high"] - max(candles[i]["low"], fvg["low"])) / fvg_range
                max_fill = max(max_fill, min(fill, 1.0))
        else:
            # Bearish FVG filled from below — price rises into gap
            if candles[i]["high"] >= fvg["low"]:
                fill = (min(candles[i]["high"], fvg["high"]) - fvg["low"]) / fvg_range
                max_fill = max(max_fill, min(fill, 1.0))

    return max_fill


def _default_htf_features() -> dict:
    """Return neutral HTF features when insufficient data."""
    return {
        "htf_premium_discount": 0.5,
        "htf_array_alignment": 0,
        "htf_ob_above_dist": 10.0,
        "htf_ob_below_dist": 10.0,
        "htf_fvg_above_dist": 10.0,
        "htf_fvg_below_dist": 10.0,
        "htf_bsl_swept": 0,
        "htf_ssl_swept": 0,
        "htf_liq_narrative": 0,
        "htf_last_candle_type": 0,
        "htf_last_candle_strength": 0.0,
    }


def engineer_htf_features(candles: list[dict], idx: int, direction: str,
                           atr: float, htf_candles_ratio: int = 4) -> dict:
    """Extract 11 HTF (4H-equivalent) features from 1H candles.

    Simulates 4H by aggregating candles in groups and computing
    ICT-relevant features: premium/discount, POI proximity,
    liquidity status, candle context.

    Args:
        candles: full 1H candle array
        idx: current candle index
        direction: "long" or "short"
        atr: 1H ATR for normalisation
        htf_candles_ratio: how many 1H candles per HTF candle (4 = 4H)
    """
    safe_atr = atr if atr > 0 else 1.0
    is_long = direction == "long"

    # Aggregate into pseudo-4H candles
    htf_lookback = min(idx, 80)  # ~20 HTF candles
    start = max(0, idx - htf_lookback)
    raw = candles[start:idx + 1]

    htf_candles = []
    for i in range(0, len(raw) - htf_candles_ratio + 1, htf_candles_ratio):
        group = raw[i:i + htf_candles_ratio]
        htf_candles.append({
            "open": group[0]["open"],
            "high": max(c["high"] for c in group),
            "low": min(c["low"] for c in group),
            "close": group[-1]["close"],
            "datetime": group[0].get("datetime", ""),
        })

    if len(htf_candles) < 3:
        return _default_htf_features()

    price = candles[idx]["close"]

    # ── 1. Premium/Discount Array ─────────────────────────
    swing_lb = min(3, max(1, len(htf_candles) // 4))
    htf_swings = detect_swing_points(htf_candles, lookback=swing_lb)
    swing_highs = [s for s in htf_swings if s["type"] == "high"]
    swing_lows = [s for s in htf_swings if s["type"] == "low"]

    if swing_highs and swing_lows:
        recent_high = max(s["price"] for s in swing_highs[-3:])
        recent_low = min(s["price"] for s in swing_lows[-3:])
        dealing_range = recent_high - recent_low
        premium_discount = (price - recent_low) / dealing_range if dealing_range > 0 else 0.5
    else:
        recent = htf_candles[-10:] if len(htf_candles) >= 10 else htf_candles
        range_high = max(c["high"] for c in recent)
        range_low = min(c["low"] for c in recent)
        dealing_range = range_high - range_low
        premium_discount = (price - range_low) / dealing_range if dealing_range > 0 else 0.5

    premium_discount = max(0.0, min(1.0, premium_discount))

    # Array alignment
    if abs(premium_discount - 0.5) < 0.1:
        htf_array_alignment = 0
    elif (is_long and premium_discount < 0.5) or (not is_long and premium_discount > 0.5):
        htf_array_alignment = 1
    else:
        htf_array_alignment = -1

    # ── 2. 4H POI Proximity ──────────────────────────────
    htf_atr = compute_atr(htf_candles, min(14, len(htf_candles) - 1))
    if htf_atr <= 0:
        htf_atr = safe_atr * htf_candles_ratio
    htf_obs = detect_order_blocks(htf_candles, htf_atr, 1.5)
    htf_fvgs = detect_fvgs(htf_candles)

    # Supply above / demand below
    supply_obs = [ob for ob in htf_obs if ob["type"] == "bearish" and ob["low"] > price]
    demand_obs = [ob for ob in htf_obs if ob["type"] == "bullish" and ob["high"] < price]
    bear_fvgs_above = [f for f in htf_fvgs if f["type"] == "bearish" and f["low"] > price]
    bull_fvgs_below = [f for f in htf_fvgs if f["type"] == "bullish" and f["high"] < price]

    htf_ob_above_dist = min((ob["low"] - price) / safe_atr for ob in supply_obs) if supply_obs else 10.0
    htf_ob_below_dist = min((price - ob["high"]) / safe_atr for ob in demand_obs) if demand_obs else 10.0
    htf_fvg_above_dist = min((f["low"] - price) / safe_atr for f in bear_fvgs_above) if bear_fvgs_above else 10.0
    htf_fvg_below_dist = min((price - f["high"]) / safe_atr for f in bull_fvgs_below) if bull_fvgs_below else 10.0

    # ── 3. 4H Liquidity Status ───────────────────────────
    liq_window = min(5, max(1, len(htf_candles) // 3))
    htf_liqs = detect_liquidity(htf_candles, window=liq_window)
    htf_bsl_levels = [lq for lq in htf_liqs if lq["type"] == "buyside"]
    htf_ssl_levels = [lq for lq in htf_liqs if lq["type"] == "sellside"]

    recent_htf = htf_candles[-5:] if len(htf_candles) >= 5 else htf_candles
    htf_bsl_swept = 0
    htf_ssl_swept = 0

    for bsl in htf_bsl_levels:
        for c in recent_htf:
            if c["high"] > bsl["price"]:
                htf_bsl_swept = 1
                break
        if htf_bsl_swept:
            break

    for ssl in htf_ssl_levels:
        for c in recent_htf:
            if c["low"] < ssl["price"]:
                htf_ssl_swept = 1
                break
        if htf_ssl_swept:
            break

    # Liquidity narrative (Power of 3 phase)
    htf_liq_narrative = 0
    if htf_ssl_swept and not htf_bsl_swept:
        if htf_candles[-1]["close"] > htf_candles[-1]["open"]:
            htf_liq_narrative = 1  # SSL swept + bullish reversal
    elif htf_bsl_swept and not htf_ssl_swept:
        if htf_candles[-1]["close"] < htf_candles[-1]["open"]:
            htf_liq_narrative = -1  # BSL swept + bearish reversal

    # ── 4. 4H Candle Context ─────────────────────────────
    last_htf = htf_candles[-1]
    htf_last_candle_type = 1 if last_htf["close"] > last_htf["open"] else -1
    htf_last_body = abs(last_htf["close"] - last_htf["open"])
    htf_last_candle_strength = round(htf_last_body / safe_atr, 4) if safe_atr > 0 else 0.0

    return {
        "htf_premium_discount": round(premium_discount, 4),
        "htf_array_alignment": htf_array_alignment,
        "htf_ob_above_dist": round(min(htf_ob_above_dist, 10.0), 4),
        "htf_ob_below_dist": round(min(htf_ob_below_dist, 10.0), 4),
        "htf_fvg_above_dist": round(min(htf_fvg_above_dist, 10.0), 4),
        "htf_fvg_below_dist": round(min(htf_fvg_below_dist, 10.0), 4),
        "htf_bsl_swept": htf_bsl_swept,
        "htf_ssl_swept": htf_ssl_swept,
        "htf_liq_narrative": htf_liq_narrative,
        "htf_last_candle_type": htf_last_candle_type,
        "htf_last_candle_strength": htf_last_candle_strength,
    }
