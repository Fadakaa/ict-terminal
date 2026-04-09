"""Training pipeline — AutoGluon binary classifier + quantile regression.

Binary-only mode: predicts win vs loss.  Multi-3 (stopped_out/tp1/runner)
was disabled after analysis showed repeated OOS degradation over 5 retrains
(43% OOS accuracy, model_trustworthy=false).  Binary is simpler, needs less
data, and avoids the overfitting issues that plagued multi-class.

Active model type is recorded in model_meta.json.  The prediction module
reads this file to load the correct model.

Follows DI pattern: all functions accept optional config dict.
"""
import json
import math
import os
import logging
import tempfile

from ml.config import get_config

logger = logging.getLogger(__name__)

from ml.feature_schema import (FEATURE_SET as INFERENCE_FEATURES,
                               RICH_FEATURE_THRESHOLD,
                               RICH_FEATURE_MIN_FOR_MULTI3)

WIN_OUTCOMES = {"tp1", "tp2", "tp3", "tp1_hit", "tp2_hit", "tp3_hit"}
MULTI3_CLASSES = {"stopped_out", "tp1", "runner"}
MULTI3_OOS_THRESHOLD = 0.45

try:
    from autogluon.tabular import TabularPredictor
except ImportError:
    TabularPredictor = None


def _count_rich_rows(df, feature_cols) -> int:
    """Count rows where at least RICH_FEATURE_THRESHOLD features are non-zero."""
    numeric = df[list(feature_cols)].fillna(0)
    nonzero_per_row = (numeric != 0).sum(axis=1)
    return int((nonzero_per_row >= RICH_FEATURE_THRESHOLD).sum())


def _save_model_meta(model_dir: str, active_type: str, rich_count: int,
                     binary_oos: float, multi3_oos: float, **extra):
    """Persist which model type is active."""
    meta = {
        "active_model_type": active_type,  # "binary" or "multi3"
        "rich_feature_count": rich_count,
        "binary_oos_accuracy": round(binary_oos, 4),
        "multi3_oos_accuracy": round(multi3_oos, 4),
        "upgraded_to_multi3": active_type == "multi3",
    }
    # Add dual evaluation fields if provided
    for key in ("multi3_oos_live_accuracy", "multi3_oos_live_sample_size",
                "backtest_setups_count", "active_evaluation_source"):
        if key in extra:
            meta[key] = extra[key]
    path = os.path.join(model_dir, "model_meta.json")
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def get_active_model_type(model_dir: str) -> str:
    """Read which model type is currently active (binary or multi3)."""
    meta_path = os.path.join(model_dir, "model_meta.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path) as f:
                return json.load(f).get("active_model_type", "binary")
        except Exception:
            pass
    return "binary"


def get_model_meta(model_dir: str) -> dict:
    """Read full model meta including upgrade status."""
    meta_path = os.path.join(model_dir, "model_meta.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path) as f:
                return json.load(f)
        except Exception:
            pass
    return {"active_model_type": "binary", "upgraded_to_multi3": False,
            "multi3_oos_accuracy": 0}


def is_multi3_active(config: dict = None) -> bool:
    """Multi-3 is disabled — always returns False."""
    return False


def train_classifier(db, config: dict = None, dataset_manager=None,
                     live_only=False) -> dict:
    """Train AutoGluon classifier on completed trades.

    Args:
        db: TradeLogger instance
        config: optional config dict
        dataset_manager: if provided, use blended dataset instead of raw DB

    Returns dict with training results (accuracy, samples, model_type).
    """
    cfg = config or get_config()

    # Get training data from dataset manager (blended WFO+live) or raw DB
    if dataset_manager is not None:
        df = dataset_manager.get_blended_dataset(live_only=live_only)
        label = "outcome"
    else:
        df = db.get_training_data()
        label = "actual_result"

    if len(df) < cfg["min_training_samples"]:
        return {
            "status": "insufficient_data",
            "samples": len(df),
            "min_required": cfg["min_training_samples"],
        }

    if TabularPredictor is None:
        return {"status": "autogluon_not_installed", "samples": len(df)}

    # Check for sample weights (from dataset manager blending)
    has_weights = "sample_weight" in df.columns

    # Whitelist: only keep features from the canonical schema.
    # INFERENCE_FEATURES imported from ml.feature_schema at module level.
    keep_cols = [c for c in df.columns if c in INFERENCE_FEATURES or c == label]
    if has_weights and "sample_weight" in df.columns:
        keep_cols.append("sample_weight")
    train_df = df[keep_cols]

    model_path = os.path.join(cfg["model_dir"], "classifier")
    os.makedirs(model_path, exist_ok=True)

    fit_kwargs = {
        "time_limit": cfg["autogluon_time_limit"],
        "presets": cfg["autogluon_presets"],
        "verbosity": 0,
    }

    # --- Apply weakness-boosted sample weights (active learning) ---
    eval_path = os.path.join(cfg["model_dir"], "classifier_evaluation.json")
    if has_weights and os.path.exists(eval_path):
        try:
            import json
            with open(eval_path) as f:
                eval_data = json.load(f)
            weakness_map = {}
            for w in eval_data.get("weaknesses", []):
                weakness_map[w["segment"]] = w["priority"]

            if weakness_map:
                boost_high = cfg.get("weakness_boost_high", 3.0)
                boost_med = cfg.get("weakness_boost_medium", 1.5)
                boosts = {"high": boost_high, "medium": boost_med, "low": 1.0}

                # Build segment key per row and apply boost
                import math
                for idx in train_df.index:
                    tf_raw = train_df.at[idx, "timeframe_encoded"] if "timeframe_encoded" in train_df.columns else 0
                    tf = 0 if (isinstance(tf_raw, float) and math.isnan(tf_raw)) else tf_raw
                    hour_raw = train_df.at[idx, "session_hour"] if "session_hour" in train_df.columns else 0
                    hour = 0 if (isinstance(hour_raw, float) and math.isnan(hour_raw)) else hour_raw
                    dir_raw = train_df.at[idx, "entry_direction"] if "entry_direction" in train_df.columns else 0
                    direction = 0 if (isinstance(dir_raw, float) and math.isnan(dir_raw)) else dir_raw
                    tf_map = {0: "5min", 1: "15min", 2: "1h", 3: "4h", 4: "1day"}
                    tf_name = tf_map.get(int(float(tf)), "?")
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
                    dir_name = "long" if direction == 1 else "short"
                    seg = f"{tf_name}_{session}_{dir_name}"
                    priority = weakness_map.get(seg, "low")
                    train_df.at[idx, "sample_weight"] *= boosts[priority]
                logger.info("Applied weakness boosts to %d segments", len(weakness_map))
        except Exception as e:
            logger.warning("Could not apply weakness boosts: %s", e)

    # --- Count rich-feature rows ---
    feature_cols = INFERENCE_FEATURES & set(train_df.columns)
    rich_count = _count_rich_rows(train_df, feature_cols)
    logger.info("Rich-feature rows: %d / %d (threshold: %d for multi3)",
                rich_count, len(train_df), RICH_FEATURE_MIN_FOR_MULTI3)

    # --- Temporal holdout split (time-ordered, no shuffle) ---
    # Sort by setup_id to ensure chronological order (wfo-NNNN < live-NNNN < bt-NNNN)
    if "setup_id" in train_df.columns:
        train_df = train_df.sort_values("setup_id").reset_index(drop=True)
    holdout_frac = cfg.get("eval_holdout_fraction", 0.2)
    split_idx = int(len(train_df) * (1 - holdout_frac))
    holdout_df = train_df.iloc[split_idx:].copy()
    train_portion = train_df.iloc[:split_idx].copy()
    fast_kwargs = {**fit_kwargs, "time_limit": min(fit_kwargs.get("time_limit", 120), 60)}

    # ──────────────────────────────────────────────
    # STEP 1: Always train a binary (win/loss) model
    # ──────────────────────────────────────────────
    binary_label = "__binary_outcome"
    train_df_binary = train_df.copy()
    train_df_binary[binary_label] = train_df_binary[label].isin(WIN_OUTCOMES).map(
        {True: "win", False: "loss"})

    binary_path = os.path.join(cfg["model_dir"], "classifier_binary")
    os.makedirs(binary_path, exist_ok=True)

    binary_cols = [c for c in train_df_binary.columns
                   if c in feature_cols or c == binary_label
                   or (c == "sample_weight" and has_weights)]
    train_binary = train_df_binary[binary_cols]

    # Holdout eval for binary
    binary_oos = 0
    if len(train_portion) >= 20 and len(holdout_df) >= 5:
        hp_binary = train_portion.copy()
        hp_binary[binary_label] = hp_binary[label].isin(WIN_OUTCOMES).map(
            {True: "win", False: "loss"})
        hh_binary = holdout_df.copy()
        hh_binary[binary_label] = hh_binary[label].isin(WIN_OUTCOMES).map(
            {True: "win", False: "loss"})
        hp_cols = [c for c in hp_binary.columns
                   if c in feature_cols or c == binary_label
                   or (c == "sample_weight" and has_weights)]
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                bp = TabularPredictor(
                    label=binary_label, path=tmpdir, problem_type="binary",
                    sample_weight="sample_weight" if has_weights else None,
                    verbosity=0,
                ).fit(hp_binary[hp_cols], **fast_kwargs)
                binary_oos = bp.evaluate(hh_binary[hp_cols]).get("accuracy", 0)
            except Exception as e:
                logger.warning("Binary holdout eval failed: %s", e)

    # Production binary model
    TabularPredictor(
        label=binary_label, path=binary_path, problem_type="binary",
        sample_weight="sample_weight" if has_weights else None,
    ).fit(train_binary, **fit_kwargs)

    # ──────────────────────────────────────────────
    # STEP 2: Multi-3 DISABLED — binary only
    # Multi-3 showed repeated OOS degradation (43% accuracy after 5 retrains).
    # Binary is more stable with the current feature set and data volume.
    # ──────────────────────────────────────────────
    multi3_oos = 0
    multi3_trained = False
    active_model_type = "binary"
    logger.info("Binary-only mode — multi-3 training disabled")

    # ──────────────────────────────────────────────
    # STEP 3: Save model_meta.json + backwards-compat classifier symlink
    # ──────────────────────────────────────────────
    meta = _save_model_meta(
        cfg["model_dir"], active_model_type, rich_count, binary_oos, multi3_oos,
    )

    # Copy binary model to the canonical "classifier" path for backwards compat
    import shutil
    canonical_path = os.path.join(cfg["model_dir"], "classifier")
    source_path = os.path.join(cfg["model_dir"], "classifier_binary")
    try:
        if os.path.exists(canonical_path):
            shutil.rmtree(canonical_path)
        shutil.copytree(source_path, canonical_path)
    except PermissionError:
        logger.warning(
            "Could not update canonical classifier symlink at %s "
            "(permission denied) — model still usable at %s",
            canonical_path, source_path,
        )

    # Best accuracy for logging
    best_oos = binary_oos if active_model_type == "binary" else multi3_oos

    db.log_training_run(
        model_type=f"classifier_{active_model_type}",
        samples=len(df),
        accuracy=best_oos,
        feature_version=cfg["feature_version"],
    )

    result = {
        "status": "trained",
        "model_type": f"classifier_{active_model_type}",
        "active_model": active_model_type,
        "samples": len(df),
        "rich_feature_count": rich_count,
        "binary_oos_accuracy": round(binary_oos, 4),
        "multi3_oos_accuracy": round(multi3_oos, 4) if multi3_trained else None,
        "multi3_trained": multi3_trained,
        "oos_accuracy": round(best_oos, 4),
        "accuracy": round(best_oos, 4),
        "model_path": canonical_path,
        "model_meta": meta,
    }

    # Extract narrative-relevant feature importance (pass holdout for permutation importance)
    try:
        _extract_narrative_feature_importance(cfg["model_dir"], canonical_path,
                                              test_data=holdout_df)
    except Exception as e:
        logger.warning("Narrative AG weight extraction failed: %s", e)

    # Train quantile model if enough data
    if len(df) >= cfg["min_training_samples_quantile"]:
        q_result = train_quantile(db, config=cfg)
        result["quantile"] = q_result

    return result


def train_quantile(db, config: dict = None) -> dict:
    """Train quantile regression on MFE for SL/TP optimisation.

    Predicts distribution of maximum favorable excursion.
    """
    cfg = config or get_config()
    df = db.get_training_data()

    if len(df) < cfg["min_training_samples_quantile"]:
        return {
            "status": "insufficient_data",
            "samples": len(df),
            "min_required": cfg["min_training_samples_quantile"],
        }

    if "mfe" not in df.columns:
        return {"status": "no_mfe_data", "samples": len(df)}

    if TabularPredictor is None:
        return {"status": "autogluon_not_installed", "samples": len(df)}

    # Prepare — predict MFE using features
    label = "mfe"
    drop_cols = [c for c in ["actual_result", "mae", "pnl"] if c in df.columns]
    train_df = df.drop(columns=drop_cols)

    model_path = os.path.join(cfg["model_dir"], "quantile_mfe")
    os.makedirs(model_path, exist_ok=True)

    predictor = TabularPredictor(
        label=label,
        path=model_path,
        problem_type="regression",
    ).fit(
        train_df,
        time_limit=cfg["autogluon_time_limit"],
        presets=cfg["autogluon_presets"],
        verbosity=0,
    )

    # Log
    db.log_training_run(
        model_type="quantile_mfe",
        samples=len(df),
        accuracy=0,
        feature_version=cfg["feature_version"],
    )

    return {
        "status": "trained",
        "model_type": "quantile_mfe",
        "samples": len(df),
        "model_path": model_path,
    }


def should_retrain(db, config: dict = None) -> bool:
    """Check if retraining is needed based on new trades since last training."""
    cfg = config or get_config()
    completed = db.get_completed_trade_count()

    if completed < cfg["min_training_samples"]:
        return False

    last = db.get_last_training("classifier")
    if not last:
        return True

    new_since_last = completed - last["samples_used"]
    return new_since_last >= cfg["retrain_on_n_new_trades"]


FEATURE_TO_NARRATIVE = {
    "entry_direction": "directional_bias",
    "bias_direction_match": "directional_bias",
    "trend_strength": "p3_phase",
    "price_vs_20sma": "premium_discount",
    "num_confluences": "confidence_calibration",
    "gold_dxy_corr_20": "intermarket_synthesis",
    "gold_dxy_diverging": "intermarket_synthesis",
    "dxy_range_position": "intermarket_synthesis",
    "yield_direction": "intermarket_synthesis",
    "ob_nearest_distance_atr": "key_levels",
    "fvg_nearest_distance_atr": "key_levels",
    "liq_nearest_target_distance_atr": "key_levels",
}

# Killzone decoding for per-killzone AG weight extraction
_KZ_ENCODED_TO_NAME = {1: "London", 2: "NY", 3: "Asian"}


def _importance_to_narrative(imp: dict) -> dict:
    """Map raw feature importances to narrative field weights, normalized 0-1.

    Uses min-max normalization with a floor of 0.05 so that features with
    real but small signal are not squished to exactly 0.
    """
    narrative_scores: dict[str, list[float]] = {}
    for feat, narrative_field in FEATURE_TO_NARRATIVE.items():
        if feat in imp:
            narrative_scores.setdefault(narrative_field, []).append(abs(imp[feat]))

    narrative_weights = {}
    for field, scores in narrative_scores.items():
        narrative_weights[field] = sum(scores) / len(scores)

    if narrative_weights:
        max_w = max(narrative_weights.values())
        if max_w > 0:
            narrative_weights = {
                k: round(max(0.05, v / max_w), 4) if v > 0 else 0.0
                for k, v in narrative_weights.items()
            }

    return narrative_weights


def _get_feature_importance(predictor, test_data=None) -> dict:
    """Extract feature importance from a predictor, preferring fast methods.

    Tries (in order):
    1. Average native importances across all tree models in the ensemble
    2. Permutation importance (slower, needs test data)
    3. Uniform fallback
    """
    import numpy as np
    feat_names = predictor.feature_metadata_in.get_features()

    # Method 1: Average native importances across ensemble's tree models
    try:
        trainer = predictor._trainer
        all_models = trainer.get_model_names()
        collected = []
        for name in all_models:
            try:
                model_obj = trainer.load_model(name)
                inner = getattr(model_obj, "model", None)
                fi = getattr(inner, "feature_importances_", None)
                if fi is not None and len(fi) == len(feat_names):
                    collected.append(np.array(fi, dtype=float))
            except Exception:
                continue
        if collected:
            avg_fi = np.mean(collected, axis=0)
            imp = dict(zip(feat_names, avg_fi.tolist()))
            logger.info("feature_importance: averaged native importances from "
                        "%d tree models", len(collected))
            return imp
    except Exception as e:
        logger.info("Native importance extraction failed: %s", e)

    # Method 2: Permutation importance (needs test data, use more shuffles for stability)
    if test_data is not None:
        try:
            sample_n = min(200, len(test_data))
            logger.info("feature_importance: running permutation on %d rows, label=%s",
                        sample_n, predictor.label)
            imp_df = predictor.feature_importance(
                data=test_data, subsample_size=sample_n,
                num_shuffle_sets=5, silent=True)
            imp = imp_df["importance"].to_dict() if "importance" in imp_df.columns else {}
            if imp:
                logger.info("feature_importance: permutation got %d features",
                            len(imp))
                return imp
        except Exception as e:
            logger.error("feature_importance permutation failed: %s", e)

    # Fallback: uniform importance
    logger.warning("feature_importance: using uniform fallback for %d features",
                   len(feat_names))
    return {f: 1.0 / len(feat_names) for f in feat_names}


def _extract_narrative_feature_importance(model_dir: str, classifier_path: str,
                                          test_data=None):
    """Extract feature importance for narrative-derived features, normalized to 0-1.

    Maps ML feature names to narrative field names and writes to
    narrative_weights_ag.json. These override EMA weights when the
    AutoGluon model is active.

    Writes per-killzone weights when test_data contains killzone_encoded
    and session_hour columns.
    """
    if TabularPredictor is None:
        return

    predictor = TabularPredictor.load(classifier_path, verbosity=0,
                                      require_version_match=False)

    imp = _get_feature_importance(predictor, test_data)
    if not imp:
        return

    # Global weights
    global_weights = _importance_to_narrative(imp)
    if not global_weights:
        return

    result = {"_global": global_weights}

    # Per-killzone weights (when test data with killzone info is available)
    min_kz = 10
    if test_data is not None and "killzone_encoded" in test_data.columns:
        try:
            for kz_code, kz_name in _KZ_ENCODED_TO_NAME.items():
                mask = test_data["killzone_encoded"] == kz_code
                if kz_name == "NY" and "session_hour" in test_data.columns:
                    # Split NY into AM/PM
                    for sub_name, hour_lo, hour_hi in [("NY_AM", 12, 16), ("NY_PM", 16, 20)]:
                        sub_mask = mask & test_data["session_hour"].between(hour_lo, hour_hi - 1)
                        subset = test_data[sub_mask]
                        if len(subset) >= min_kz:
                            kz_imp = _get_feature_importance(predictor, subset)
                            if kz_imp:
                                kz_w = _importance_to_narrative(kz_imp)
                                if kz_w:
                                    result[sub_name] = kz_w
                else:
                    subset = test_data[mask]
                    if len(subset) >= min_kz:
                        kz_imp = _get_feature_importance(predictor, subset)
                        if kz_imp:
                            kz_w = _importance_to_narrative(kz_imp)
                            if kz_w:
                                result[kz_name] = kz_w

            # Off = everything not in 1/2/3
            off_mask = ~test_data["killzone_encoded"].isin([1, 2, 3])
            off_subset = test_data[off_mask]
            if len(off_subset) >= min_kz:
                off_imp = _get_feature_importance(predictor, off_subset)
                if off_imp:
                    off_w = _importance_to_narrative(off_imp)
                    if off_w:
                        result["Off"] = off_w
        except Exception as e:
            logger.warning("Per-killzone AG weight extraction failed: %s", e)

    path = os.path.join(model_dir, "narrative_weights_ag.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Narrative AG weights extracted: %s", result)
    return result


def extract_ag_weights(model_dir: str = None, db=None) -> dict:
    """Standalone extraction of AG narrative weights from the trained classifier.

    Can be called independently of training to populate narrative_weights_ag.json.
    Uses TrainingDatasetManager (same data source as train_classifier in production).

    Returns:
        Extracted weights dict or empty dict on failure.
    """
    cfg = get_config()
    model_dir = model_dir or cfg["model_dir"]

    # Find classifier
    classifier_path = os.path.join(model_dir, "classifier")
    if not os.path.exists(classifier_path):
        classifier_path = os.path.join(model_dir, "classifier_binary")
    if not os.path.exists(classifier_path):
        logger.error("No classifier found in %s", model_dir)
        return {}

    # Load test data for permutation importance
    # Use TrainingDatasetManager (production path), fall back to TradeLogger
    test_data = None
    try:
        from ml.dataset import TrainingDatasetManager
        dm = TrainingDatasetManager(config=cfg)
        df = dm.get_blended_dataset(live_only=True)
        label = "outcome"
        if df.empty:
            df = dm.get_blended_dataset(live_only=False)
        if df.empty and db is not None:
            df = db.get_training_data()
            label = "actual_result"
        logger.info("AG extraction: loaded %d rows via %s",
                    len(df), "dataset_manager" if label == "outcome" else "trade_logger")
        if len(df) >= 10:
            binary_label = "__binary_outcome"
            df[binary_label] = df[label].isin(WIN_OUTCOMES).map(
                {True: "win", False: "loss"})
            label_dist = df[binary_label].value_counts().to_dict()
            logger.info("AG extraction: binary label distribution: %s", label_dist)
            keep_cols = [c for c in df.columns
                         if c in INFERENCE_FEATURES or c == binary_label]
            test_data = df[keep_cols]
            logger.info("AG extraction: test_data shape=%s", test_data.shape)
    except Exception as e:
        logger.warning("Could not load test data for AG extraction: %s", e, exc_info=True)

    result = _extract_narrative_feature_importance(model_dir, classifier_path,
                                                   test_data=test_data)
    return result or {}


# ── Outcome-based narrative weights ──────────────────────────────────


def compute_narrative_weights_from_outcomes(db=None) -> dict:
    """Compute narrative field weights from actual trade outcomes.

    For each resolved trade with an Opus narrative, score each field's
    correctness (reusing ClaudeAnalysisBridge's logic), then compute
    win-rate lift: P(win | field_aligned) - P(win | field_not_aligned).

    This measures how much *trusting* each narrative field actually
    predicts winning, segmented by killzone.

    Returns:
        Segmented weights dict {"_global": {...}, "Asian": {...}, ...}
        written to narrative_weights_ag.json.
    """
    from ml.scanner_db import ScannerDB
    from ml.claude_bridge import ClaudeAnalysisBridge

    cfg = get_config()
    if db is None:
        db = ScannerDB()

    setups = db.get_resolved_setups() if hasattr(db, "get_resolved_setups") else []
    if not setups:
        logger.warning("outcome weights: no resolved setups")
        return {}

    bridge = ClaudeAnalysisBridge()
    KILLZONE_KEYS = bridge.KILLZONE_KEYS

    # Collect per-field correctness scores + outcomes, grouped by killzone
    # Each entry: {field: score, "_win": bool, "_kz": str}
    records = []
    for setup in setups:
        cal_json = setup.get("calibration_json") or {}
        if isinstance(cal_json, str):
            try:
                cal_json = json.loads(cal_json)
            except Exception:
                continue
        narrative_json = cal_json.get("opus_narrative", {})
        if not narrative_json:
            continue

        outcome = setup.get("actual_result", "") or setup.get("outcome", "")
        is_win = outcome in WIN_OUTCOMES
        entry_dir = setup.get("direction", "")
        kz = setup.get("killzone", "")
        mfe_atr = setup.get("mfe_atr")

        # Compute per-field correctness using the same logic as EMA update
        scores = _score_narrative_fields(
            narrative_json, entry_dir, is_win, outcome, setup, mfe_atr)
        if scores:
            scores["_win"] = is_win
            scores["_kz"] = kz if kz in KILLZONE_KEYS else ""
            records.append(scores)

    if len(records) < 10:
        logger.warning("outcome weights: only %d trades with narratives", len(records))
        return {}

    logger.info("outcome weights: analyzing %d trades", len(records))

    # Compute win-rate lift per field, per group
    FIELDS = list(bridge.NARRATIVE_FIELDS)
    ALIGNED_THRESHOLD = 0.6  # score >= this = "field was aligned/correct"

    def _compute_lift(subset: list) -> dict:
        """Win-rate lift for each field in a subset of records."""
        weights = {}
        base_wr = sum(1 for r in subset if r["_win"]) / len(subset) if subset else 0.5
        for field in FIELDS:
            aligned = [r for r in subset if r.get(field, 0.5) >= ALIGNED_THRESHOLD]
            misaligned = [r for r in subset if r.get(field, 0.5) < ALIGNED_THRESHOLD]
            if len(aligned) >= 3 and len(misaligned) >= 3:
                wr_aligned = sum(1 for r in aligned if r["_win"]) / len(aligned)
                wr_misaligned = sum(1 for r in misaligned if r["_win"]) / len(misaligned)
                lift = wr_aligned - wr_misaligned  # range: -1.0 to 1.0
                weights[field] = max(0.0, lift)  # only positive lift = useful field
            else:
                weights[field] = None  # insufficient data
        return weights

    def _normalize(raw: dict) -> dict:
        """Normalize to 0-1 range, using 0.05 floor for non-zero values."""
        valid = {k: v for k, v in raw.items() if v is not None}
        if not valid:
            return {}
        max_v = max(valid.values()) if valid else 1
        if max_v <= 0:
            return {k: 0.5 for k in valid}  # no field has positive lift
        return {
            k: round(max(0.05, v / max_v), 4) if v > 0 else 0.0
            for k, v in valid.items()
        }

    # Global
    result = {"_global": _normalize(_compute_lift(records))}
    raw_global = _compute_lift(records)
    logger.info("outcome weights global (raw lift): %s",
                {k: round(v, 4) if v is not None else None for k, v in raw_global.items()})

    # Per-killzone
    min_kz = cfg.get("narrative_min_kz_trades", 10)
    for kz in KILLZONE_KEYS:
        subset = [r for r in records if r["_kz"] == kz]
        if len(subset) >= min_kz:
            result[kz] = _normalize(_compute_lift(subset))
            logger.info("outcome weights %s: %d trades", kz, len(subset))

    # Write
    model_dir = cfg["model_dir"]
    path = os.path.join(model_dir, "narrative_weights_ag.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Outcome-based narrative weights written: %s", list(result.keys()))
    return result


def _score_narrative_fields(narrative_json: dict, entry_direction: str,
                            is_win: bool, outcome: str, setup: dict,
                            mfe_atr: float | None = None) -> dict:
    """Score each narrative field's correctness for a single trade.

    Same logic as ClaudeAnalysisBridge.update_narrative_field_weights but
    returns the raw scores without applying EMA.
    """
    _mfe = mfe_atr or 0.0
    is_type2_loss = not is_win and _mfe >= 1.0
    is_type1_loss = not is_win and _mfe < 0.5
    aligned_loss_score = 0.6 if is_type2_loss else (0.15 if is_type1_loss else 0.3)

    bias = narrative_json.get("directional_bias", "")
    phase = narrative_json.get("power_of_3_phase", "")
    pd_zone = narrative_json.get("premium_discount", "")
    conf = narrative_json.get("phase_confidence", "")
    intermarket = narrative_json.get("intermarket_synthesis")
    key_levels = narrative_json.get("key_levels", [])
    entry_price = setup.get("entry_price", 0)

    scores = {}

    # directional_bias
    bias_aligned = (
        (bias == "bullish" and entry_direction == "long") or
        (bias == "bearish" and entry_direction == "short")
    )
    if bias_aligned and is_win:
        scores["directional_bias"] = 1.0
    elif bias_aligned and not is_win:
        scores["directional_bias"] = aligned_loss_score
    elif not bias_aligned and is_win:
        scores["directional_bias"] = 0.2
    else:
        scores["directional_bias"] = 0.0

    # p3_phase
    phase_aligned = (
        (phase == "distribution" and entry_direction == "short") or
        (phase == "accumulation" and entry_direction == "long") or
        (phase == "manipulation")
    )
    if phase_aligned and is_win:
        scores["p3_phase"] = 1.0
    elif phase_aligned and not is_win:
        scores["p3_phase"] = aligned_loss_score
    elif phase == "manipulation":
        scores["p3_phase"] = 0.5
    elif not phase_aligned and is_win:
        scores["p3_phase"] = 0.2
    else:
        scores["p3_phase"] = 0.0

    # premium_discount
    pd_aligned = (
        (pd_zone == "premium" and entry_direction == "short") or
        (pd_zone == "discount" and entry_direction == "long")
    )
    if pd_zone == "equilibrium":
        scores["premium_discount"] = 0.5
    elif pd_aligned and is_win:
        scores["premium_discount"] = 1.0
    elif pd_aligned and not is_win:
        scores["premium_discount"] = aligned_loss_score
    elif not pd_aligned and is_win:
        scores["premium_discount"] = 0.2
    else:
        scores["premium_discount"] = 0.0

    # confidence_calibration
    if (conf == "high" and is_win) or (conf == "low" and not is_win):
        scores["confidence_calibration"] = 1.0
    elif conf == "medium":
        scores["confidence_calibration"] = 0.5
    elif conf == "high" and not is_win:
        scores["confidence_calibration"] = 0.1
    elif conf == "low" and is_win:
        scores["confidence_calibration"] = 0.3
    else:
        scores["confidence_calibration"] = 0.5

    # intermarket_synthesis
    if intermarket:
        try:
            from ml.intermarket_validator import IntermarketValidator
            cal_json = setup.get("calibration_json", {})
            if isinstance(cal_json, str):
                cal_json = json.loads(cal_json)
            im_block = cal_json.get("intermarket", {})
            scores["intermarket_synthesis"] = IntermarketValidator.score_intermarket_signal(
                diverging=im_block.get("gold_dxy_diverging", 0),
                is_win=is_win,
                corr=im_block.get("gold_dxy_corr_20", 0),
                yield_dir=im_block.get("yield_direction", 0),
                direction=entry_direction,
            )
        except Exception:
            scores["intermarket_synthesis"] = 0.5
    else:
        scores["intermarket_synthesis"] = 0.5

    # key_levels
    best_proximity = float('inf')
    if entry_price and key_levels:
        for level in key_levels:
            price = level.get("price", 0) if isinstance(level, dict) else 0
            if price and entry_price:
                dist_pct = abs(price - entry_price) / entry_price
                best_proximity = min(best_proximity, dist_pct)

    if best_proximity <= 0.003 and is_win:
        scores["key_levels"] = 1.0
    elif best_proximity <= 0.003 and not is_win:
        scores["key_levels"] = 0.4
    elif best_proximity <= 0.01:
        scores["key_levels"] = 0.3
    else:
        scores["key_levels"] = 0.2

    return scores
