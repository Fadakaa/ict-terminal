"""Walk-forward evaluation for AutoGluon classifier.

Produces honest out-of-sample accuracy by training on past data and
testing on future data across multiple time horizons.  Also detects
per-segment weaknesses for active learning.

Usage:
    from ml.evaluation import evaluate_classifier_walkforward
    result = evaluate_classifier_walkforward(dataset_manager)
"""
import json
import logging
import os
from datetime import datetime

import pandas as pd

from ml.config import get_config

logger = logging.getLogger(__name__)

# The 32 features that extract_features() produces at inference time.
INFERENCE_FEATURES = {
    "ob_count", "ob_bullish_count", "ob_bearish_count", "ob_strongest_strength",
    "ob_nearest_distance_atr", "ob_avg_size_atr", "ob_alignment",
    "fvg_count", "fvg_unfilled_count", "fvg_nearest_distance_atr",
    "fvg_avg_size_atr", "fvg_alignment",
    "liq_buyside_count", "liq_sellside_count",
    "liq_nearest_target_distance_atr", "liq_nearest_threat_distance_atr",
    "risk_reward_tp1", "risk_reward_tp2", "sl_distance_atr", "tp1_distance_atr",
    "entry_direction", "bias_direction_match", "num_confluences",
    "has_ob_fvg_overlap", "killzone_encoded", "timeframe_encoded",
    "atr_14", "price_vs_20sma", "recent_volatility_ratio",
    "last_candle_body_atr", "trend_strength", "session_hour",
}

WIN_OUTCOMES = {"tp1", "tp2", "tp3", "tp1_hit", "tp2_hit", "tp3_hit"}


def _is_minimal_feature_row(row: pd.Series) -> bool:
    """Detect rows produced by _build_minimal_features() (near-zero features)."""
    zero_cols = ["ob_nearest_distance_atr", "fvg_nearest_distance_atr",
                 "price_vs_20sma", "last_candle_body_atr"]
    return all(row.get(c, 0) == 0 for c in zero_cols if c in row.index)


def _prepare_df(dataset_manager, config: dict = None) -> pd.DataFrame:
    """Load and prepare dataset for evaluation."""
    cfg = config or get_config()
    df = dataset_manager.get_blended_dataset()
    if df.empty:
        return df

    label = "outcome"
    keep = [c for c in df.columns if c in INFERENCE_FEATURES or c == label]
    df = df[keep].copy()

    # Binary outcome for accuracy: win vs loss
    df["is_win"] = df[label].isin(WIN_OUTCOMES).astype(int)
    return df


def _train_and_eval(train_df: pd.DataFrame, test_df: pd.DataFrame,
                    cfg: dict) -> dict:
    """Train AutoGluon on train_df, predict on test_df, return metrics."""
    from autogluon.tabular import TabularPredictor
    import tempfile

    label = "outcome"
    if len(train_df) < 20 or len(test_df) < 5:
        return {"accuracy": None, "trades": 0}

    # Use binary label for evaluation (win vs loss)
    binary_label = "__binary_outcome"
    train_binary = train_df.copy()
    test_binary = test_df.copy()
    train_binary[binary_label] = train_binary[label].isin(WIN_OUTCOMES).map(
        {True: "win", False: "loss"})
    test_binary[binary_label] = test_binary[label].isin(WIN_OUTCOMES).map(
        {True: "win", False: "loss"})

    drop_cols = [label, "is_win"]

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            predictor = TabularPredictor(
                label=binary_label,
                path=tmpdir,
                problem_type="binary",
                verbosity=0,
            ).fit(
                train_binary.drop(columns=drop_cols, errors="ignore"),
                time_limit=min(cfg.get("autogluon_time_limit", 300), 90),
                presets="best_quality",
                verbosity=0,
            )

            eval_df = test_binary.drop(columns=drop_cols, errors="ignore")
            preds = predictor.predict(eval_df)
            actual_binary = test_binary[binary_label]

            # Binary accuracy
            binary_acc = (preds == actual_binary).mean()

            # Win probabilities
            try:
                proba = predictor.predict_proba(eval_df)
                win_probs = proba["win"].tolist() if "win" in proba.columns else []
            except Exception:
                win_probs = []

            actual_win = test_df["is_win"]

            return {
                "accuracy": round(float(binary_acc), 4),
                "binary_accuracy": round(float(binary_acc), 4),
                "trades": len(test_df),
                "win_probs": win_probs,
                "actual_wins": actual_win.tolist(),
            }
        except Exception as e:
            logger.warning("Fold training failed: %s", e)
            return {"accuracy": None, "trades": 0}


def _compute_weaknesses(results: list[dict], cfg: dict) -> list[dict]:
    """Compute per-segment weaknesses from fold results.

    Each result should contain 'predictions' list of dicts with
    timeframe, session_hour, direction, predicted_win, actual_win.
    """
    segments = {}
    for r in results:
        for p in r.get("predictions", []):
            tf = p.get("timeframe", "?")
            hour = p.get("session_hour", 0)
            if 0 <= hour < 7:
                session = "Asian"
            elif 7 <= hour < 12:
                session = "London"
            elif 12 <= hour < 16:
                session = "NY_AM"
            elif 16 <= hour < 20:
                session = "NY_PM"
            else:
                session = "Off"
            direction = "long" if p.get("direction", 0) == 1 else "short"
            seg = f"{tf}_{session}_{direction}"
            if seg not in segments:
                segments[seg] = {"correct": 0, "total": 0}
            segments[seg]["total"] += 1
            if p["predicted_win"] == p["actual_win"]:
                segments[seg]["correct"] += 1

    min_trades = cfg.get("weakness_min_trades", 10)
    weaknesses = []
    for seg, data in segments.items():
        if data["total"] < min_trades:
            continue
        acc = data["correct"] / data["total"]
        priority = "high" if acc < 0.5 else "medium" if acc < 0.6 else "low"
        weaknesses.append({
            "segment": seg,
            "accuracy": round(acc, 3),
            "trades": data["total"],
            "priority": priority,
        })

    weaknesses.sort(key=lambda x: x["accuracy"])
    return weaknesses


def _compute_calibration(all_win_probs: list, all_actual_wins: list) -> list:
    """Compute calibration curve: predicted probability vs actual win rate."""
    if not all_win_probs:
        return []

    buckets = {}
    for prob, actual in zip(all_win_probs, all_actual_wins):
        # Round to nearest 0.1 bucket
        b = round(prob * 10) / 10
        b = max(0.0, min(1.0, b))
        key = f"{b:.1f}"
        if key not in buckets:
            buckets[key] = {"predicted_sum": 0, "actual_sum": 0, "count": 0}
        buckets[key]["predicted_sum"] += prob
        buckets[key]["actual_sum"] += actual
        buckets[key]["count"] += 1

    curve = []
    for key in sorted(buckets.keys()):
        b = buckets[key]
        curve.append({
            "bucket": key,
            "predicted": round(b["predicted_sum"] / b["count"], 3),
            "actual": round(b["actual_sum"] / b["count"], 3),
            "count": b["count"],
        })
    return curve


def evaluate_classifier_walkforward(dataset_manager, config: dict = None,
                                    scanner_db=None) -> dict:
    """Run walk-forward evaluation with short/medium/expanding windows.

    Args:
        dataset_manager: TrainingDatasetManager instance
        config: Optional config override
        scanner_db: Optional ScannerDB for timestamp ordering

    Returns:
        Evaluation results dict (saved to classifier_evaluation.json)
    """
    cfg = config or get_config()
    df = _prepare_df(dataset_manager, cfg)

    if len(df) < 50:
        return {
            "oos_accuracy": 0,
            "test_trades": 0,
            "model_trustworthy": False,
            "periods": {},
            "weaknesses": [],
            "feature_quality": {"minimal_feature_pct": 1.0, "full_feature_count": 0},
            "evaluated_at": datetime.utcnow().isoformat(),
            "error": f"Insufficient data: {len(df)} rows (need 50+)",
        }

    # Feature quality assessment
    minimal_count = df.apply(_is_minimal_feature_row, axis=1).sum()
    full_count = len(df) - minimal_count
    minimal_pct = round(minimal_count / len(df), 3)

    # --- Expanding window evaluation ---
    # Start at 50%, expand by 10% increments
    expanding_results = []
    all_win_probs = []
    all_actual_wins = []
    all_predictions = []
    n = len(df)

    for pct in range(50, 95, 10):
        split_idx = int(n * pct / 100)
        train = df.iloc[:split_idx].copy()
        test = df.iloc[split_idx:].copy()
        if len(test) < 5:
            continue

        result = _train_and_eval(train, test, cfg)
        if result["accuracy"] is not None:
            expanding_results.append(result)
            all_win_probs.extend(result.get("win_probs", []))
            all_actual_wins.extend(result.get("actual_wins", []))

            # Collect per-trade predictions for weakness analysis
            for i, (wp, aw) in enumerate(zip(
                    result.get("win_probs", []),
                    result.get("actual_wins", []))):
                row = test.iloc[i] if i < len(test) else None
                if row is not None:
                    all_predictions.append({
                        "timeframe": str(row.get("timeframe_encoded", "?")),
                        "session_hour": int(row.get("session_hour", 0)),
                        "direction": int(row.get("entry_direction", 0)),
                        "predicted_win": 1 if wp > 0.5 else 0,
                        "actual_win": int(aw),
                    })

    # Aggregate expanding results
    if expanding_results:
        total_trades = sum(r["trades"] for r in expanding_results)
        weighted_acc = sum(r["accuracy"] * r["trades"]
                          for r in expanding_results) / total_trades
        expanding_summary = {
            "accuracy": round(weighted_acc, 4),
            "trades": total_trades,
            "folds": len(expanding_results),
        }
    else:
        expanding_summary = {"accuracy": 0, "trades": 0, "folds": 0}

    # --- Short evaluation (last 20% as single test) ---
    holdout_frac = cfg.get("eval_holdout_fraction", 0.2)
    split = int(n * (1 - holdout_frac))
    short_result = _train_and_eval(df.iloc[:split], df.iloc[split:], cfg)
    short_summary = {
        "accuracy": short_result.get("accuracy", 0) or 0,
        "trades": short_result.get("trades", 0),
    }

    # --- Medium evaluation (middle third) ---
    third = n // 3
    med_result = _train_and_eval(df.iloc[:third], df.iloc[third:2*third], cfg)
    medium_summary = {
        "accuracy": med_result.get("accuracy", 0) or 0,
        "trades": med_result.get("trades", 0),
    }

    # Overall OOS accuracy (from expanding — most comprehensive)
    oos_accuracy = expanding_summary["accuracy"]
    test_trades = expanding_summary["trades"]

    # Trustworthiness gate
    model_trustworthy = (
        oos_accuracy > cfg.get("oos_min_accuracy", 0.55)
        and test_trades >= cfg.get("oos_min_test_trades", 30)
    )

    # Weaknesses
    weakness_results = [{"predictions": all_predictions}]
    weaknesses = _compute_weaknesses(weakness_results, cfg)

    # Calibration curve
    calibration = _compute_calibration(all_win_probs, all_actual_wins)

    # Per-timeframe accuracy (from timeframe_encoded)
    by_timeframe = {}
    tf_map = {1: "15min", 2: "1h", 3: "4h", 4: "1day", 0: "5min"}
    for p in all_predictions:
        tf_code = p.get("timeframe", "?")
        tf_name = tf_map.get(int(float(tf_code)), tf_code) if tf_code != "?" else "?"
        if tf_name not in by_timeframe:
            by_timeframe[tf_name] = {"correct": 0, "total": 0}
        by_timeframe[tf_name]["total"] += 1
        if p["predicted_win"] == p["actual_win"]:
            by_timeframe[tf_name]["correct"] += 1

    by_tf_summary = {}
    for tf, data in by_timeframe.items():
        by_tf_summary[tf] = {
            "accuracy": round(data["correct"] / data["total"], 3) if data["total"] > 0 else 0,
            "trades": data["total"],
        }

    result = {
        "oos_accuracy": oos_accuracy,
        "test_trades": test_trades,
        "model_trustworthy": model_trustworthy,
        "periods": {
            "short": short_summary,
            "medium": medium_summary,
            "expanding": expanding_summary,
        },
        "by_timeframe": by_tf_summary,
        "weaknesses": weaknesses,
        "calibration_curve": calibration,
        "feature_quality": {
            "minimal_feature_pct": minimal_pct,
            "full_feature_count": int(full_count),
        },
        "evaluated_at": datetime.utcnow().isoformat(),
    }

    # Save to disk
    eval_path = os.path.join(cfg.get("model_dir", "models"),
                             "classifier_evaluation.json")
    os.makedirs(os.path.dirname(eval_path), exist_ok=True)
    with open(eval_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info("Walk-forward evaluation complete: OOS=%.1f%% on %d trades, trustworthy=%s",
                oos_accuracy * 100, test_trades, model_trustworthy)

    return result
