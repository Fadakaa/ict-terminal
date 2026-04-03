"""Backfill 52-column features from stored analysis_json + calibration_json.

Option A: Zero API cost. Re-extracts HTF context, narrative state, and Opus
agreement features from the JSON blobs already stored on every resolved
scanner_setups row.

Usage:
    python -m ml.backfill_features              # dry-run (shows stats only)
    python -m ml.backfill_features --apply      # write enriched rows to dataset
    python -m ml.backfill_features --apply --replace  # replace existing live rows
"""
import json
import math
import os
import sys
import logging
import sqlite3

import pandas as pd

from ml.feature_schema import FEATURE_COLUMNS, RICH_FEATURE_THRESHOLD
from ml.features import (
    _extract_htf_features,
    _extract_narrative_features,
    _compute_entry_zone,
    _encode_premium_discount,
    _encode_p3_phase,
    _encode_setup_quality,
    _encode_killzone,
    _encode_timeframe,
    _intermarket_or_nan,
)
from ml.dataset import TrainingDatasetManager

logger = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(__file__), "models", "scanner.db")


def _load_resolved_setups() -> list[dict]:
    """Load all resolved setups with stored JSON blobs."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM scanner_setups "
        "WHERE outcome IS NOT NULL "
        "AND analysis_json IS NOT NULL "
        "AND calibration_json IS NOT NULL "
        "ORDER BY created_at"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _extract_from_stored_json(setup: dict) -> dict | None:
    """Extract 52-column features from a single stored setup's JSONs.

    Since we don't have the original candles, candle-derived features
    (ATR, SMA, vol_ratio, etc.) come from the calibration_json's
    volatility_context when available, otherwise NaN.

    Everything else — HTF context, narrative state, Opus agreement,
    OB/FVG/liquidity counts, trade setup params — comes from analysis_json.
    """
    try:
        analysis = json.loads(setup["analysis_json"]) if isinstance(
            setup["analysis_json"], str) else setup["analysis_json"]
        calibration = json.loads(setup["calibration_json"]) if isinstance(
            setup["calibration_json"], str) else setup["calibration_json"]
    except (json.JSONDecodeError, TypeError):
        return None

    vol = (calibration or {}).get("volatility_context", {})
    atr = vol.get("atr_14") or 1.0
    if atr <= 0:
        atr = 1.0

    # ── OB/FVG/Liquidity from analysis ──
    obs = analysis.get("orderBlocks") or []
    fvgs = analysis.get("fvgs") or []
    liqs = analysis.get("liquidity") or []
    tps = analysis.get("takeProfits") or []
    entry = analysis.get("entry") or {}
    sl = analysis.get("stopLoss") or {}
    bias = analysis.get("bias", "neutral")
    killzone = analysis.get("killzone") or setup.get("killzone", "")
    confluences = analysis.get("confluences") or []
    timeframe = setup.get("timeframe", "1h")

    entry_price = entry.get("price", 0) or setup.get("entry_price", 0)
    direction = entry.get("direction", "short") or setup.get("direction", "short")
    sl_price = sl.get("price", 0) or setup.get("sl_price", 0)

    # OB features
    ob_bullish = [ob for ob in obs if ob.get("type") == "bullish"]
    ob_bearish = [ob for ob in obs if ob.get("type") == "bearish"]
    ob_strengths = [{"strong": 3, "moderate": 2, "weak": 1}.get(
        ob.get("strength", ""), 0) for ob in obs]
    ob_sizes = [(ob["high"] - ob["low"]) for ob in obs
                if "high" in ob and "low" in ob]
    ob_distances = []
    if entry_price:
        for ob in obs:
            mid = (ob.get("high", 0) + ob.get("low", 0)) / 2
            ob_distances.append(abs(entry_price - mid))

    ob_alignment = 0
    if bias == "bullish" and ob_bullish:
        ob_alignment = 1
    elif bias == "bearish" and ob_bearish:
        ob_alignment = 1

    # FVG features
    unfilled_fvgs = [f for f in fvgs if not f.get("filled", True)]
    fvg_distances = []
    if entry_price:
        for f in unfilled_fvgs:
            mid = (f.get("high", 0) + f.get("low", 0)) / 2
            fvg_distances.append(abs(entry_price - mid))
    fvg_sizes = [(f["high"] - f["low"]) for f in fvgs
                 if "high" in f and "low" in f]
    fvg_alignment = 0
    if unfilled_fvgs:
        for f in unfilled_fvgs:
            if f.get("type") == bias:
                fvg_alignment = 1
                break

    # Liquidity features
    bsl = [l for l in liqs if l.get("type") == "buyside"]
    ssl = [l for l in liqs if l.get("type") == "sellside"]
    if direction == "long":
        target_prices = [l["price"] for l in bsl if "price" in l]
        threat_prices = [l["price"] for l in ssl if "price" in l]
    else:
        target_prices = [l["price"] for l in ssl if "price" in l]
        threat_prices = [l["price"] for l in bsl if "price" in l]
    liq_target_dist = (min(abs(entry_price - p) for p in target_prices)
                       if entry_price and target_prices else 0)
    liq_threat_dist = (min(abs(entry_price - p) for p in threat_prices)
                       if entry_price and threat_prices else 0)

    # Trade setup
    rr1 = tps[0].get("rr", 0) if len(tps) > 0 else 0
    rr2 = tps[1].get("rr", 0) if len(tps) > 1 else 0
    sl_dist = abs(entry_price - sl_price) if entry_price and sl_price else 0
    tp1_dist = abs(entry_price - tps[0]["price"]) if len(tps) > 0 and entry_price else 0
    dir_encoded = 1 if direction == "long" else 0
    bias_match = 1 if ((bias == "bullish" and direction == "long") or
                       (bias == "bearish" and direction == "short")) else 0

    # Confluence
    has_ob_fvg_overlap = 0
    for c in confluences:
        cl = c.lower() if isinstance(c, str) else ""
        if ("ob" in cl or "order block" in cl) and (
                "fvg" in cl or "fair value" in cl or "overlap" in cl):
            has_ob_fvg_overlap = 1
            break

    # Candle-derived — use calibration volatility_context where available
    price_vs_sma = 0.0
    vol_ratio = vol.get("vol_ratio_5_14") or 1.0
    session_hour = 0
    try:
        created = setup.get("created_at", "")
        session_hour = int(created.split(" ")[1].split(":")[0])
    except (IndexError, ValueError):
        pass

    _safe_div = lambda a, b: a / b if b else 0.0

    # Intermarket from calibration
    im = calibration.get("intermarket") if calibration else None

    # Regime from calibration
    regime = vol.get("structural_regime", float('nan'))

    features = {
        "ob_count": len(obs),
        "ob_bullish_count": len(ob_bullish),
        "ob_bearish_count": len(ob_bearish),
        "ob_strongest_strength": max(ob_strengths) if ob_strengths else 0,
        "ob_nearest_distance_atr": round(_safe_div(
            min(ob_distances) if ob_distances else 0, atr), 4),
        "ob_avg_size_atr": round(_safe_div(
            sum(ob_sizes) / len(ob_sizes) if ob_sizes else 0, atr), 4),
        "ob_alignment": ob_alignment,
        "fvg_count": len(fvgs),
        "fvg_unfilled_count": len(unfilled_fvgs),
        "fvg_nearest_distance_atr": round(_safe_div(
            min(fvg_distances) if fvg_distances else 0, atr), 4),
        "fvg_avg_size_atr": round(_safe_div(
            sum(fvg_sizes) / len(fvg_sizes) if fvg_sizes else 0, atr), 4),
        "fvg_alignment": fvg_alignment,
        "liq_buyside_count": len(bsl),
        "liq_sellside_count": len(ssl),
        "liq_nearest_target_distance_atr": round(_safe_div(liq_target_dist, atr), 4),
        "liq_nearest_threat_distance_atr": round(_safe_div(liq_threat_dist, atr), 4),
        "risk_reward_tp1": rr1,
        "risk_reward_tp2": rr2,
        "sl_distance_atr": round(_safe_div(sl_dist, atr), 4),
        "tp1_distance_atr": round(_safe_div(tp1_dist, atr), 4),
        "entry_direction": dir_encoded,
        "bias_direction_match": bias_match,
        "num_confluences": len(confluences),
        "has_ob_fvg_overlap": has_ob_fvg_overlap,
        "killzone_encoded": _encode_killzone(killzone),
        "timeframe_encoded": _encode_timeframe(timeframe),
        "atr_14": round(atr, 4),
        "price_vs_20sma": price_vs_sma,
        "recent_volatility_ratio": round(vol_ratio, 4),
        "last_candle_body_atr": 0.0,  # Not available without candles
        "trend_strength": 0.0,        # Not available without candles
        "session_hour": session_hour,
        "premium_discount_encoded": _encode_premium_discount(analysis),
        "p3_phase_encoded": _encode_p3_phase(analysis),
        "setup_quality_encoded": _encode_setup_quality(analysis),
        "claude_direction_encoded": dir_encoded,
        # Intermarket
        "gold_dxy_corr_20": _intermarket_or_nan(im, "gold_dxy_corr_20"),
        "gold_dxy_diverging": 1 if (im or {}).get("gold_dxy_diverging") else 0,
        "dxy_range_position": _intermarket_or_nan(im, "dxy_range_position"),
        "yield_direction": _intermarket_or_nan(im, "yield_direction"),
        # Regime
        "volatility_regime": regime,
        # Entry zone
        **_compute_entry_zone(entry_price, direction, obs, atr),
        # HTF context (5 new columns)
        **_extract_htf_features(analysis, calibration),
        # Narrative state (4 new columns)
        **_extract_narrative_features(analysis, calibration),
    }

    return features


def run_backfill(apply: bool = False, replace: bool = False) -> dict:
    """Re-extract features from all resolved scanner setups.

    Args:
        apply: If True, write enriched rows to the training dataset CSV.
        replace: If True, remove existing live rows before inserting.

    Returns:
        Stats dict with counts and coverage.
    """
    setups = _load_resolved_setups()
    print(f"Loaded {len(setups)} resolved setups with stored JSONs")

    results = []
    errors = 0
    for setup in setups:
        features = _extract_from_stored_json(setup)
        if features is None:
            errors += 1
            continue
        results.append({
            "setup_id": setup["id"],
            "features": features,
            "outcome": setup["outcome"],
            "pnl_rr": setup.get("pnl_rr", 0) or 0,
            "mfe_atr": setup.get("mfe_atr", 0) or 0,
            "mae_atr": setup.get("mae_atr", 0) or 0,
            "timeframe": setup.get("timeframe", "1h"),
            "killzone": setup.get("killzone", ""),
        })

    print(f"Extracted features for {len(results)} setups ({errors} errors)")

    # ── Coverage analysis ──
    htf_filled = 0
    narrative_filled = 0
    opus_filled = 0
    dealing_range_filled = 0
    full_quality = 0

    for r in results:
        f = r["features"]
        if f.get("htf_bias_encoded", 0) != 0:
            htf_filled += 1
        if not (isinstance(f.get("thesis_confidence"), float)
                and math.isnan(f["thesis_confidence"])):
            narrative_filled += 1
        if f.get("opus_sonnet_agreement", 0) == 1:
            opus_filled += 1
        if not (isinstance(f.get("dealing_range_position"), float)
                and math.isnan(f["dealing_range_position"])):
            dealing_range_filled += 1

        filled_count = sum(
            1 for col in FEATURE_COLUMNS
            if col in f and f[col] is not None
            and not (isinstance(f[col], float) and math.isnan(f[col]))
        )
        if filled_count >= RICH_FEATURE_THRESHOLD:
            full_quality += 1

    total = len(results) or 1
    stats = {
        "total_setups": len(setups),
        "extracted": len(results),
        "errors": errors,
        "htf_bias_filled": htf_filled,
        "dealing_range_filled": dealing_range_filled,
        "narrative_filled": narrative_filled,
        "opus_agreement_filled": opus_filled,
        "full_quality_rows": full_quality,
        "partial_quality_rows": len(results) - full_quality,
    }

    print(f"\n{'='*50}")
    print(f"BACKFILL COVERAGE REPORT")
    print(f"{'='*50}")
    print(f"Total extracted:        {len(results)}")
    print(f"Full quality (≥{RICH_FEATURE_THRESHOLD} cols): {full_quality} ({100*full_quality/total:.1f}%)")
    print(f"HTF bias populated:     {htf_filled} ({100*htf_filled/total:.1f}%)")
    print(f"Dealing range filled:   {dealing_range_filled} ({100*dealing_range_filled/total:.1f}%)")
    print(f"Narrative state filled: {narrative_filled} ({100*narrative_filled/total:.1f}%)")
    print(f"Opus agreement (=1):    {opus_filled} ({100*opus_filled/total:.1f}%)")

    # Outcome breakdown
    outcomes = {}
    for r in results:
        o = r["outcome"]
        outcomes[o] = outcomes.get(o, 0) + 1
    print(f"\nOutcome distribution:")
    for o, count in sorted(outcomes.items()):
        print(f"  {o}: {count}")

    if apply:
        mgr = TrainingDatasetManager()
        if replace:
            # Remove existing live rows
            if not mgr._df.empty and "source" in mgr._df.columns:
                before = len(mgr._df)
                mgr._df = mgr._df[mgr._df["source"] != "live"]
                print(f"\nRemoved {before - len(mgr._df)} existing live rows")

        ingested = 0
        for r in results:
            try:
                mgr.ingest_live_trade(
                    features=r["features"],
                    outcome=r["outcome"],
                    mfe=r["mfe_atr"],
                    mae=r["mae_atr"],
                    pnl=r["pnl_rr"],
                    setup_id=r["setup_id"],
                )
                ingested += 1
            except Exception as e:
                logger.warning("Failed to ingest setup %s: %s", r["setup_id"], e)

        print(f"\nIngested {ingested} enriched rows into training dataset")
        print(f"Dataset now has {len(mgr._df)} total rows")
        stats["ingested"] = ingested
        stats["dataset_total"] = len(mgr._df)
    else:
        print(f"\n[DRY RUN] Pass --apply to write to dataset")

    return stats


def backfill_regimes(apply: bool = False) -> dict:
    """Backfill volatility_regime by fetching historical candles from OANDA.

    For each resolved setup, fetches 60 candles at the execution timeframe
    ending at the setup's created_at timestamp, then runs classify_regime().

    Uses a cache keyed on (timeframe, hourly_bucket) to avoid redundant fetches.
    Zero API cost — OANDA data is free.

    Args:
        apply: If True, update the training dataset CSV in-place.
    """
    import time
    from datetime import datetime, timedelta
    from ml.config import get_config
    from ml.volatility import classify_regime

    setups = _load_resolved_setups()
    print(f"Loaded {len(setups)} resolved setups for regime backfill")

    cfg = get_config()
    account_id = cfg.get("oanda_account_id", "")
    access_token = cfg.get("oanda_access_token", "")

    if not account_id or not access_token:
        print("ERROR: OANDA credentials not found in config. Set oanda_account_id and oanda_access_token.")
        return {"error": "no_credentials"}

    from ml.data_providers import OandaProvider
    provider = OandaProvider(account_id=account_id, access_token=access_token)

    # Map non-standard timeframes to nearest OANDA-supported granularity
    TF_REMAP = {"2h": "1h"}  # 2h not in OANDA GRAN_MAP; use 1h with 2x count
    interval_hours = {"5min": 0.083, "15min": 0.25, "30min": 0.5,
                      "1h": 1, "2h": 2, "4h": 4, "1day": 24}

    # Cache: (timeframe, hour_bucket) → candle list
    candle_cache = {}
    regime_map = {}  # setup_id → regime_result
    fetch_count = 0
    cache_hits = 0
    errors = 0

    for i, setup in enumerate(setups):
        setup_id = setup["id"]
        tf = setup.get("timeframe", "1h")
        created = setup.get("created_at", "")

        # Parse created_at to datetime
        try:
            if "T" in created:
                dt = datetime.strptime(created[:19], "%Y-%m-%dT%H:%M:%S")
            else:
                dt = datetime.strptime(created[:19], "%Y-%m-%d %H:%M:%S")
        except (ValueError, IndexError):
            errors += 1
            continue

        # Cache key: round to hourly bucket for the timeframe
        bucket_hours = max(interval_hours.get(tf, 1), 1)
        bucket_key = (tf, dt.strftime("%Y-%m-%d") + f"_{int(dt.hour / bucket_hours)}")

        if bucket_key in candle_cache:
            candles = candle_cache[bucket_key]
            cache_hits += 1
        else:
            # Remap unsupported timeframes
            fetch_tf = TF_REMAP.get(tf, tf)
            count = 120 if tf in TF_REMAP else 60  # double count for remapped TFs
            hours_back = count * interval_hours.get(fetch_tf, 1) * 1.5
            start = (dt - timedelta(hours=hours_back)).strftime("%Y-%m-%dT%H:%M:%S")
            end = dt.strftime("%Y-%m-%dT%H:%M:%S")

            try:
                candles = provider.fetch_candles("XAU/USD", fetch_tf, start, end)
                if candles and len(candles) > count:
                    candles = candles[-count:]
                candle_cache[bucket_key] = candles
                fetch_count += 1
                # Rate limit: OANDA is generous but don't hammer it
                if fetch_count % 20 == 0:
                    time.sleep(0.5)
            except Exception as e:
                logger.warning("OANDA fetch failed for %s at %s: %s", tf, created, e)
                candles = None
                errors += 1

        if candles and len(candles) >= 15:
            try:
                result = classify_regime(candles)
                regime_map[setup_id] = result
            except Exception as e:
                logger.debug("classify_regime failed for %s: %s", setup_id, e)
                errors += 1
        else:
            errors += 1

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(setups)} "
                  f"({fetch_count} fetches, {cache_hits} cache hits, {errors} errors)")

    print(f"\nRegime classification complete:")
    print(f"  Classified: {len(regime_map)}/{len(setups)}")
    print(f"  OANDA fetches: {fetch_count}")
    print(f"  Cache hits: {cache_hits}")
    print(f"  Errors: {errors}")

    # Regime distribution
    regime_counts = {}
    for r in regime_map.values():
        label = r["regime"]
        regime_counts[label] = regime_counts.get(label, 0) + 1
    print(f"\nRegime distribution:")
    for label, count in sorted(regime_counts.items(), key=lambda x: -x[1]):
        print(f"  {label}: {count}")

    if apply:
        mgr = TrainingDatasetManager()
        df = mgr._df

        if df.empty:
            print("No training dataset to update")
            return {"error": "empty_dataset"}

        # Match by setup_id and update volatility_regime
        updated = 0
        if "setup_id" in df.columns:
            for idx, row in df.iterrows():
                sid = row.get("setup_id", "")
                if sid in regime_map:
                    regime_result = regime_map[sid]
                    df.at[idx, "volatility_regime"] = regime_result["regime"]
                    updated += 1

        # Also try matching by index position for live rows (setup_id = live-NNNN)
        # The backfill ingested setups in order, so live-0000 = first setup, etc.
        if updated == 0:
            print("No setup_id matches — trying positional matching...")
            live_mask = df["source"] == "live" if "source" in df.columns else pd.Series([True]*len(df))
            live_indices = df.index[live_mask].tolist()
            for i, (setup, li) in enumerate(zip(setups, live_indices)):
                if setup["id"] in regime_map:
                    df.at[li, "volatility_regime"] = regime_map[setup["id"]]["regime"]
                    updated += 1

        mgr._df = df
        mgr._save()
        print(f"\nUpdated {updated} rows with volatility_regime in training dataset")

    else:
        print(f"\n[DRY RUN] Pass --apply to update dataset")

    return {
        "classified": len(regime_map),
        "fetches": fetch_count,
        "cache_hits": cache_hits,
        "errors": errors,
        "regime_distribution": regime_counts,
    }


def _fetch_instrument_candles(provider, symbol: str, interval: str,
                              end_dt, count: int = 30):
    """Fetch historical candles for an instrument ending at end_dt.

    Handles DXY inversion (EUR/USD proxy) same as scanner.py.
    """
    from datetime import timedelta

    interval_hours = {"5min": 0.083, "15min": 0.25, "30min": 0.5,
                      "1h": 1, "2h": 1, "4h": 4, "1day": 24}
    # Remap unsupported timeframes
    fetch_interval = {"2h": "1h"}.get(interval, interval)
    fetch_count = count * 2 if interval != fetch_interval else count

    hours_back = fetch_count * interval_hours.get(fetch_interval, 1) * 1.5
    start = (end_dt - timedelta(hours=hours_back)).strftime("%Y-%m-%dT%H:%M:%S")
    end = end_dt.strftime("%Y-%m-%dT%H:%M:%S")

    candles = provider.fetch_candles(symbol, fetch_interval, start, end)

    # DXY inversion (same logic as scanner.py _fetch_candles_oanda)
    if symbol == "DXY" and candles:
        for c in candles:
            o, h, l, cl = c["open"], c["high"], c["low"], c["close"]
            if o > 0 and cl > 0:
                c["open"] = round(1.0 / o * 104, 4)
                c["high"] = round(1.0 / l * 104, 4)
                c["low"] = round(1.0 / h * 104, 4)
                c["close"] = round(1.0 / cl * 104, 4)

    if candles and len(candles) > count:
        candles = candles[-count:]

    return candles


def backfill_intermarket(apply: bool = False) -> dict:
    """Backfill intermarket features (gold_dxy_corr_20, dxy_range_position, yield_direction).

    Fetches historical XAU/USD, DXY (EUR/USD proxy), and US10Y candles from OANDA
    for each resolved setup, then runs compute_intermarket_context().

    Zero API cost — OANDA data is free.

    Args:
        apply: If True, update the training dataset CSV in-place.
    """
    import time
    from datetime import datetime, timedelta
    from ml.config import get_config
    from ml.intermarket import compute_intermarket_context

    setups = _load_resolved_setups()
    print(f"Loaded {len(setups)} resolved setups for intermarket backfill")

    cfg = get_config()
    account_id = cfg.get("oanda_account_id", "")
    access_token = cfg.get("oanda_access_token", "")

    if not account_id or not access_token:
        print("ERROR: OANDA credentials not found in config.")
        return {"error": "no_credentials"}

    from ml.data_providers import OandaProvider
    provider = OandaProvider(account_id=account_id, access_token=access_token)

    # Cache: (instrument, timeframe, hour_bucket) → candle list
    candle_cache = {}
    intermarket_map = {}  # setup_id → intermarket dict
    fetch_count = 0
    cache_hits = 0
    errors = 0

    interval_hours = {"5min": 0.083, "15min": 0.25, "30min": 0.5,
                      "1h": 1, "2h": 2, "4h": 4, "1day": 24}

    for i, setup in enumerate(setups):
        setup_id = setup["id"]
        tf = setup.get("timeframe", "1h")
        created = setup.get("created_at", "")
        killzone = setup.get("killzone", "Off") or "Off"

        try:
            if "T" in created:
                dt = datetime.strptime(created[:19], "%Y-%m-%dT%H:%M:%S")
            else:
                dt = datetime.strptime(created[:19], "%Y-%m-%d %H:%M:%S")
        except (ValueError, IndexError):
            errors += 1
            continue

        bucket_hours = max(interval_hours.get(tf, 1), 1)
        bucket_key_base = (tf, dt.strftime("%Y-%m-%d") + f"_{int(dt.hour / bucket_hours)}")

        # Fetch 3 instruments: XAU/USD, DXY, US10Y
        instrument_candles = {}
        for instrument in ["XAU/USD", "DXY", "US10Y"]:
            cache_key = (instrument,) + bucket_key_base
            if cache_key in candle_cache:
                instrument_candles[instrument] = candle_cache[cache_key]
                cache_hits += 1
            else:
                try:
                    candles = _fetch_instrument_candles(
                        provider, instrument, tf, dt, count=30)
                    candle_cache[cache_key] = candles
                    instrument_candles[instrument] = candles
                    fetch_count += 1
                    if fetch_count % 30 == 0:
                        time.sleep(0.5)
                except Exception as e:
                    logger.debug("Fetch failed %s %s: %s", instrument, tf, e)
                    candle_cache[cache_key] = None
                    instrument_candles[instrument] = None

        gold = instrument_candles.get("XAU/USD")
        dxy = instrument_candles.get("DXY")
        us10y = instrument_candles.get("US10Y")

        if gold and len(gold) >= 5:
            try:
                ctx = compute_intermarket_context(
                    gold, dxy, us10y, session=killzone, lookback=20)
                intermarket_map[setup_id] = {
                    "gold_dxy_corr_20": ctx.get("gold_dxy_corr_20", 0.0),
                    "gold_dxy_diverging": 1 if ctx.get("gold_dxy_diverging") else 0,
                    "dxy_range_position": ctx.get("dxy_range_position", 0.5),
                    "yield_direction": ctx.get("yield_direction", 0),
                }
            except Exception as e:
                logger.debug("Intermarket compute failed for %s: %s", setup_id, e)
                errors += 1
        else:
            errors += 1

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(setups)} "
                  f"({fetch_count} fetches, {cache_hits} cache hits, {errors} errors)")

    print(f"\nIntermarket computation complete:")
    print(f"  Computed: {len(intermarket_map)}/{len(setups)}")
    print(f"  OANDA fetches: {fetch_count}")
    print(f"  Cache hits: {cache_hits}")
    print(f"  Errors: {errors}")

    # Quick stats
    if intermarket_map:
        corrs = [v["gold_dxy_corr_20"] for v in intermarket_map.values()]
        divs = sum(1 for v in intermarket_map.values() if v["gold_dxy_diverging"])
        yields_up = sum(1 for v in intermarket_map.values() if v["yield_direction"] == 1)
        yields_dn = sum(1 for v in intermarket_map.values() if v["yield_direction"] == -1)
        yields_flat = sum(1 for v in intermarket_map.values() if v["yield_direction"] == 0)
        print(f"\n  Avg Gold-DXY correlation: {sum(corrs)/len(corrs):.3f}")
        print(f"  Divergence count: {divs}")
        print(f"  Yield direction: up={yields_up}, down={yields_dn}, flat={yields_flat}")

    if apply:
        mgr = TrainingDatasetManager()
        df = mgr._df

        if df.empty:
            print("No training dataset to update")
            return {"error": "empty_dataset"}

        updated = 0
        # Try setup_id match first
        if "setup_id" in df.columns:
            for idx, row in df.iterrows():
                sid = row.get("setup_id", "")
                if sid in intermarket_map:
                    for col, val in intermarket_map[sid].items():
                        df.at[idx, col] = val
                    updated += 1

        # Fallback to positional matching
        if updated == 0:
            print("No setup_id matches — trying positional matching...")
            live_mask = df["source"] == "live" if "source" in df.columns else pd.Series([True]*len(df))
            live_indices = df.index[live_mask].tolist()
            for setup, li in zip(setups, live_indices):
                if setup["id"] in intermarket_map:
                    for col, val in intermarket_map[setup["id"]].items():
                        df.at[li, col] = val
                    updated += 1

        mgr._df = df
        mgr._save()
        print(f"\nUpdated {updated} rows with intermarket features in training dataset")

    else:
        print(f"\n[DRY RUN] Pass --apply to update dataset")

    return {
        "computed": len(intermarket_map),
        "fetches": fetch_count,
        "cache_hits": cache_hits,
        "errors": errors,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    apply = "--apply" in sys.argv
    replace = "--replace" in sys.argv

    if "--regimes" in sys.argv:
        backfill_regimes(apply=apply)
    elif "--intermarket" in sys.argv:
        backfill_intermarket(apply=apply)
    else:
        run_backfill(apply=apply, replace=replace)
