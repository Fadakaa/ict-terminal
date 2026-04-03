"""Backfill accuracy tracker + training dataset from scanner DB.

Rebuilds claude_accuracy.json from all resolved scanner trades and
ensures every resolved trade is ingested into the training dataset.

Usage:
    python -m ml.backfill
"""
import json
import os
import sqlite3
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.claude_bridge import ClaudeAnalysisBridge
from ml.scanner_db import ScannerDB
from ml.dataset import TrainingDatasetManager


def _session_from_timestamp(ts: str) -> str:
    """Derive session name from trade timestamp (GMT).

    Returns session in the format used by volatility.py:
    'asian', 'london', 'overlap_london_ny', 'new_york', 'off_hours'
    """
    try:
        if "T" in ts:
            hour = int(ts.split("T")[1].split(":")[0])
        elif " " in ts:
            hour = int(ts.split(" ")[1].split(":")[0])
        else:
            return "off_hours"
    except (ValueError, IndexError):
        return "off_hours"

    if 0 <= hour < 7:
        return "asian"
    elif 7 <= hour < 12:
        return "london"
    elif 12 <= hour < 16:
        return "overlap_london_ny"
    elif 16 <= hour < 21:
        return "new_york"
    else:
        return "off_hours"


def backfill_accuracy_tracker(trades: list[dict]) -> dict:
    """Reset and rebuild accuracy tracker from scanner trades."""
    bridge = ClaudeAnalysisBridge()

    # Reset accuracy to fresh state
    bridge._accuracy = {
        "total_trades": 0,
        "claude_direction_correct": 0,
        "claude_sl_would_survive": 0,
        "calibrated_sl_survived": 0,
        "trades_saved_by_calibration": 0,
        "claude_tp1_reached": 0,
        "calibrated_tp1_reached": 0,
        "avg_claude_sl_distance_atr": 0.0,
        "avg_calibrated_sl_distance_atr": 0.0,
        "avg_sl_widening_atr": 0.0,
        "by_session": {},
        "by_setup_type": {},
    }
    bridge._save_accuracy()

    success = 0
    errors = 0

    for t in trades:
        try:
            raw_analysis = json.loads(t["analysis_json"]) if isinstance(t["analysis_json"], str) else (t["analysis_json"] or {})
            calibration = json.loads(t["calibration_json"]) if isinstance(t["calibration_json"], str) else (t["calibration_json"] or {})

            parsed = bridge.parse_analysis(raw_analysis)

            # Fix stale session: all stored calibrations have session="off"
            # because detect_session() was broken when these were calibrated.
            # Re-derive from the trade's creation timestamp instead.
            correct_session = _session_from_timestamp(t.get("created_at", ""))
            if "session_context" not in calibration:
                calibration["session_context"] = {}
            calibration["session_context"]["session"] = correct_session

            bridge.log_completed_trade(
                original_analysis=parsed,
                calibrated_result=calibration,
                actual_outcome=t["outcome"],
                actual_pnl_atr=t.get("pnl_rr", 0) or 0,
                used_calibrated_sl=t.get("calibrated_sl") is not None,
                notes=f"backfill [{t.get('timeframe', '?')}]",
            )
            success += 1
        except Exception as e:
            errors += 1
            print(f"  ERROR on {t.get('id', '?')}: {e}")

    return {"success": success, "errors": errors, "accuracy": bridge._accuracy}


def backfill_training_dataset(trades: list[dict]) -> dict:
    """Ingest all scanner trades into training dataset (deduped)."""
    bridge = ClaudeAnalysisBridge()
    dm = TrainingDatasetManager()

    # Load existing dataset to check for dupes
    existing_ids = set()
    try:
        import pandas as pd
        dataset_path = dm.dataset_path
        if os.path.exists(dataset_path):
            df = pd.read_csv(dataset_path)
            if "setup_id" in df.columns:
                existing_ids = set(df["setup_id"].dropna().tolist())
    except Exception:
        pass

    ingested = 0
    skipped = 0
    errors = 0

    win_outcomes = {"tp1", "tp2", "tp3", "tp1_hit", "tp2_hit", "tp3_hit"}

    for t in trades:
        setup_id = t.get("id", "")
        if setup_id in existing_ids:
            skipped += 1
            continue

        try:
            raw_analysis = json.loads(t["analysis_json"]) if isinstance(t["analysis_json"], str) else (t["analysis_json"] or {})
            calibration = json.loads(t["calibration_json"]) if isinstance(t["calibration_json"], str) else (t["calibration_json"] or {})

            parsed = bridge.parse_analysis(raw_analysis)
            features = bridge._build_minimal_features(parsed, calibration)

            if not features:
                errors += 1
                continue

            # Add setup_id so we can dedup later
            features["setup_id"] = setup_id

            outcome = t["outcome"]
            is_win = outcome in win_outcomes
            pnl_rr = t.get("pnl_rr", 0) or 0

            dm.ingest_live_trade(
                features, outcome,
                mfe=max(0, pnl_rr) if is_win else 0,
                mae=abs(pnl_rr) if not is_win else 0,
                pnl=pnl_rr,
            )
            ingested += 1
            existing_ids.add(setup_id)
        except Exception as e:
            errors += 1
            print(f"  DATASET ERROR on {setup_id}: {e}")

    stats = dm.get_stats()
    return {"ingested": ingested, "skipped": skipped, "errors": errors, "dataset_stats": stats}


def main():
    print("=" * 60)
    print("ML PIPELINE BACKFILL")
    print("=" * 60)

    # Load all resolved trades from scanner DB, chronological order
    sdb = ScannerDB()
    conn = sqlite3.connect(sdb.db_path)
    conn.row_factory = sqlite3.Row
    trades = conn.execute(
        """SELECT * FROM scanner_setups
           WHERE status != 'pending' AND outcome IS NOT NULL AND outcome != 'expired'
           ORDER BY resolved_at ASC"""
    ).fetchall()
    trades = [dict(r) for r in trades]
    conn.close()

    print(f"\nFound {len(trades)} resolved trades in scanner DB")
    print(f"Date range: {trades[0]['created_at'][:10]} to {trades[-1]['created_at'][:10]}")

    # --- Task 5: Backfill accuracy tracker ---
    print("\n" + "-" * 40)
    print("TASK 5: Backfilling Accuracy Tracker...")
    print("-" * 40)

    acc_result = backfill_accuracy_tracker(trades)
    acc = acc_result["accuracy"]

    print(f"\n  Processed: {acc_result['success']} | Errors: {acc_result['errors']}")
    print(f"  Total trades: {acc['total_trades']}")
    print(f"  Direction correct: {acc['claude_direction_correct']}")
    print(f"\n  Sessions:")
    for session, data in sorted(acc["by_session"].items()):
        print(f"    {session:12s}: {data['trades']} trades | claude_survived: {data['claude_survived']} | calibrated: {data['calibrated_survived']}")
    print(f"\n  Setup Types:")
    for stype, data in sorted(acc["by_setup_type"].items(), key=lambda x: x[1]["trades"], reverse=True):
        wr = data["wins"] / data["trades"] * 100 if data["trades"] > 0 else 0
        print(f"    {stype:35s}: {data['trades']:3d} trades | {data['wins']:3d} wins | {wr:.0f}% WR")

    # --- Task 6: Backfill training dataset ---
    print("\n" + "-" * 40)
    print("TASK 6: Backfilling Training Dataset...")
    print("-" * 40)

    ds_result = backfill_training_dataset(trades)
    stats = ds_result["dataset_stats"]

    print(f"\n  Ingested: {ds_result['ingested']} | Skipped (existing): {ds_result['skipped']} | Errors: {ds_result['errors']}")
    print(f"  Dataset total: {stats['total']} rows")
    print(f"  WFO: {stats.get('wfo_count', 0)} | Live: {stats.get('live_count', 0)}")
    print(f"  Outcome distribution: {stats.get('outcome_distribution', {})}")

    print("\n" + "=" * 60)
    print("BACKFILL COMPLETE")
    print("=" * 60)


def backfill_features_from_stored_json() -> dict:
    """Re-extract features from stored analysis JSON for all resolved trades.

    Does NOT re-fetch candles. Computes everything possible from the stored
    Claude analysis + calibration JSON. The 6 candle-derived features
    will use defaults (0) but the ~28-32 other features will be populated.

    This gets most trades to 20+ non-zero features (the rich threshold).
    """
    from ml.scanner_db import ScannerDB
    from ml.claude_bridge import ClaudeAnalysisBridge
    from ml.dataset import TrainingDatasetManager
    from ml.features import (_encode_premium_discount, _encode_p3_phase,
                              _encode_setup_quality, _encode_killzone,
                              _encode_timeframe, _safe_divide, _encode_strength)

    db = ScannerDB()
    bridge = ClaudeAnalysisBridge()
    dm = TrainingDatasetManager()

    # Clear old live rows first
    dm.clear_live_trades()
    print(f"Cleared live trades from dataset. WFO rows retained.")

    resolved = db.get_resolved_setups()
    print(f"Found {len(resolved)} resolved setups to backfill.")

    backfilled = 0
    errors = 0

    for setup in resolved:
        try:
            raw = setup.get("analysis_json", {})
            cal = setup.get("calibration_json", {})
            vol = cal.get("volatility_context", {})
            atr = vol.get("atr_14", 1.0) or 1.0
            im = cal.get("intermarket", {})

            entry_data = raw.get("entry") or {}
            sl_data = raw.get("stopLoss") or {}
            entry_price = entry_data.get("price", 0)
            sl_price = sl_data.get("price", 0)
            direction = entry_data.get("direction", "short")
            bias = raw.get("bias", "neutral")
            killzone = raw.get("killzone", "")
            tps = raw.get("takeProfits") or []
            obs = raw.get("orderBlocks") or []
            fvgs_list = raw.get("fvgs") or []
            liqs = raw.get("liquidity") or []
            confluences = raw.get("confluences") or []

            # OB features
            ob_bullish = [o for o in obs if o.get("type") == "bullish"]
            ob_bearish = [o for o in obs if o.get("type") == "bearish"]
            ob_strengths = [_encode_strength(o.get("strength", "")) for o in obs]
            ob_distances = [abs(entry_price - (o.get("high", 0) + o.get("low", 0)) / 2) for o in obs if entry_price]
            ob_sizes = [abs(o.get("high", 0) - o.get("low", 0)) for o in obs]
            dir_enc = 1 if direction == "long" else -1
            ob_alignment = sum(1 for o in obs if (o.get("type") == "bullish") == (direction == "long"))

            # FVG features
            unfilled = [f for f in fvgs_list if not f.get("filled")]
            fvg_distances = [abs(entry_price - (f.get("high", 0) + f.get("low", 0)) / 2) for f in fvgs_list if entry_price]
            fvg_sizes = [abs(f.get("high", 0) - f.get("low", 0)) for f in fvgs_list]
            fvg_alignment = sum(1 for f in fvgs_list if (f.get("type") == "bullish") == (direction == "long"))

            # Liquidity features
            bsl = [l for l in liqs if l.get("type") == "buyside"]
            ssl = [l for l in liqs if l.get("type") == "sellside"]
            targets = bsl if direction == "long" else ssl
            threats = ssl if direction == "long" else bsl
            liq_target_dist = min(abs(l.get("price", 0) - entry_price) for l in targets) if targets and entry_price else 0
            liq_threat_dist = min(abs(l.get("price", 0) - entry_price) for l in threats) if threats and entry_price else 0

            # Trade setup
            sl_dist = abs(entry_price - sl_price) if entry_price and sl_price else 0
            tp_prices = [tp.get("price", 0) if isinstance(tp, dict) else tp for tp in tps]
            tp1_dist = abs(tp_prices[0] - entry_price) if tp_prices else 0
            rr1 = tp1_dist / sl_dist if sl_dist > 0 else 0
            rr2 = abs(tp_prices[1] - entry_price) / sl_dist if len(tp_prices) > 1 and sl_dist > 0 else 0
            bias_match = 1.0 if (bias == "bullish" and direction == "long") or (bias == "bearish" and direction == "short") else 0.0

            # OB/FVG overlap
            has_overlap = 0
            for ob in obs:
                for fvg in fvgs_list:
                    if ob.get("low", 0) <= fvg.get("high", 0) and ob.get("high", 0) >= fvg.get("low", 0):
                        has_overlap = 1
                        break

            features = {
                "ob_count": len(obs),
                "ob_bullish_count": len(ob_bullish),
                "ob_bearish_count": len(ob_bearish),
                "ob_strongest_strength": max(ob_strengths) if ob_strengths else 0,
                "ob_nearest_distance_atr": _safe_divide(min(ob_distances) if ob_distances else 0, atr),
                "ob_avg_size_atr": _safe_divide(sum(ob_sizes) / len(ob_sizes) if ob_sizes else 0, atr),
                "ob_alignment": ob_alignment,
                "fvg_count": len(fvgs_list),
                "fvg_unfilled_count": len(unfilled),
                "fvg_nearest_distance_atr": _safe_divide(min(fvg_distances) if fvg_distances else 0, atr),
                "fvg_avg_size_atr": _safe_divide(sum(fvg_sizes) / len(fvg_sizes) if fvg_sizes else 0, atr),
                "fvg_alignment": fvg_alignment,
                "liq_buyside_count": len(bsl),
                "liq_sellside_count": len(ssl),
                "liq_nearest_target_distance_atr": _safe_divide(liq_target_dist, atr),
                "liq_nearest_threat_distance_atr": _safe_divide(liq_threat_dist, atr),
                "risk_reward_tp1": round(rr1, 4),
                "risk_reward_tp2": round(rr2, 4),
                "sl_distance_atr": _safe_divide(sl_dist, atr),
                "tp1_distance_atr": _safe_divide(tp1_dist, atr),
                "entry_direction": dir_enc,
                "bias_direction_match": bias_match,
                "num_confluences": len(confluences),
                "has_ob_fvg_overlap": has_overlap,
                "killzone_encoded": _encode_killzone(killzone),
                "timeframe_encoded": _encode_timeframe(setup.get("timeframe", "1h")),
                "atr_14": round(atr, 4),
                "price_vs_20sma": 0.0,
                "recent_volatility_ratio": 1.0,
                "last_candle_body_atr": 0.0,
                "trend_strength": 0.0,
                "session_hour": 0,
                "premium_discount_encoded": _encode_premium_discount(raw),
                "p3_phase_encoded": _encode_p3_phase(raw),
                "setup_quality_encoded": _encode_setup_quality(raw),
                "claude_direction_encoded": dir_enc,
                "gold_dxy_corr_20": im.get("gold_dxy_corr_20", 0.0),
                "gold_dxy_diverging": 1 if im.get("gold_dxy_diverging") else 0,
                "dxy_range_position": im.get("dxy_range_position", 0.5),
                "yield_direction": im.get("yield_direction", 0),
            }

            outcome = setup.get("outcome", "stopped_out")
            pnl_rr = setup.get("pnl_rr", 0)
            is_win = outcome.startswith("tp")

            dm.ingest_live_trade(
                features, outcome,
                mfe=max(0, pnl_rr) if is_win else 0,
                mae=abs(pnl_rr) if not is_win else 0,
                pnl=pnl_rr,
            )
            backfilled += 1

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  Error on {setup.get('id', '?')}: {e}")

    nonzero_counts = []
    stats = dm.get_stats()
    print(f"\nBackfill complete: {backfilled} trades, {errors} errors")
    print(f"Dataset: {stats.get('total', 0)} total rows "
          f"({stats.get('wfo_count', 0)} WFO + {stats.get('live_count', 0)} live)")
    return {"backfilled": backfilled, "errors": errors, "dataset_stats": stats}


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--features":
        backfill_features_from_stored_json()
    else:
        main()
