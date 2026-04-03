"""Training pipeline — AutoGluon classifier + quantile regression.

Supports two classifier modes:
  - Binary:  win vs loss (default when data is sparse/low-quality)
  - Multi-3: stopped_out / tp1 / tp2 / tp3 (upgrades automatically when
    rich-feature data exceeds 100 samples AND multi-3 OOS accuracy > 0.45)

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
    """Check if multi3 model is active and meets OOS threshold."""
    cfg = config or get_config()
    meta = get_model_meta(cfg["model_dir"])
    return (meta.get("upgraded_to_multi3", False)
            and meta.get("multi3_oos_accuracy", 0) >= MULTI3_OOS_THRESHOLD)


def train_classifier(db, config: dict = None, dataset_manager=None) -> dict:
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
        df = dataset_manager.get_blended_dataset()
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
    # STEP 2: Conditionally train multi-3 model
    # (stopped_out / tp1 / runner where runner = tp2+tp3)
    # ──────────────────────────────────────────────
    multi3_oos = 0
    multi3_oos_live = None
    multi3_oos_live_count = 0
    backtest_count = 0
    eval_source = "all"
    multi3_trained = False
    active_model_type = "binary"

    if rich_count >= RICH_FEATURE_MIN_FOR_MULTI3:
        logger.info("Rich features >= %d — training multi-3 classifier",
                     RICH_FEATURE_MIN_FOR_MULTI3)

        # Remap outcome to 3 classes: stopped_out / tp1 / runner
        multi3_label = "__multi3_outcome"
        train_df_multi3 = train_df.copy()
        outcome_map = {}
        for o in train_df_multi3[label].unique():
            if o in ("stopped_out",):
                outcome_map[o] = "stopped_out"
            elif o in ("tp1", "tp1_hit"):
                outcome_map[o] = "tp1"
            elif o in ("tp2", "tp2_hit", "tp3", "tp3_hit"):
                outcome_map[o] = "runner"
            else:
                outcome_map[o] = "stopped_out"  # default unknown → loss
        train_df_multi3[multi3_label] = train_df_multi3[label].map(outcome_map)

        multi3_path = os.path.join(cfg["model_dir"], "classifier_multi3")
        os.makedirs(multi3_path, exist_ok=True)

        m3_cols = [c for c in train_df_multi3.columns
                   if c in feature_cols or c == multi3_label
                   or (c == "sample_weight" and has_weights)]
        train_multi3 = train_df_multi3[m3_cols]

        # Holdout eval for multi3
        if len(train_portion) >= 20 and len(holdout_df) >= 5:
            hp_m3 = train_portion.copy()
            hp_m3[multi3_label] = hp_m3[label].map(outcome_map)
            hh_m3 = holdout_df.copy()
            hh_m3[multi3_label] = hh_m3[label].map(outcome_map)
            hp_m3_cols = [c for c in hp_m3.columns
                          if c in feature_cols or c == multi3_label
                          or (c == "sample_weight" and has_weights)]
            with tempfile.TemporaryDirectory() as tmpdir:
                try:
                    mp = TabularPredictor(
                        label=multi3_label, path=tmpdir, problem_type="multiclass",
                        sample_weight="sample_weight" if has_weights else None,
                        verbosity=0,
                    ).fit(hp_m3[hp_m3_cols], **fast_kwargs)
                    multi3_oos = mp.evaluate(hh_m3[hp_m3_cols]).get("accuracy", 0)

                    # Dual OOS: compute live-only accuracy
                    if "source" in holdout_df.columns:
                        live_mask = holdout_df["source"] != "wfo"
                        # Map source column through to hh_m3
                        hh_m3["__source"] = holdout_df["source"].values
                        live_test = hh_m3[hh_m3["__source"] == "live"]
                        multi3_oos_live_count = len(live_test)
                        if multi3_oos_live_count >= 30:
                            live_eval_cols = [c for c in hp_m3_cols if c != "__source"]
                            multi3_oos_live = mp.evaluate(
                                live_test[live_eval_cols]).get("accuracy", 0)
                        hh_m3 = hh_m3.drop(columns=["__source"])
                except Exception as e:
                    logger.warning("Multi3 holdout eval failed: %s", e)

        # Production multi3 model
        TabularPredictor(
            label=multi3_label, path=multi3_path, problem_type="multiclass",
            sample_weight="sample_weight" if has_weights else None,
        ).fit(train_multi3, **fit_kwargs)
        multi3_trained = True

        # Count sources for metadata
        if "source" in df.columns:
            backtest_count = int((df["source"] == "backtest").sum())
            live_total = int((df["source"] == "live").sum())
        else:
            backtest_count = 0
            live_total = len(df)

        # Gate logic: use live-only OOS once 200+ live rows exist
        if live_total >= 200 and multi3_oos_live is not None:
            gate_oos = multi3_oos_live
            eval_source = "live"
        else:
            gate_oos = multi3_oos
            eval_source = "all"

        # Auto-upgrade: use multi3 if it beats the threshold
        if gate_oos >= MULTI3_OOS_THRESHOLD:
            active_model_type = "multi3"
            logger.info("Multi-3 OOS %.1f%% (%s) >= %.0f%% threshold — UPGRADED to multi3",
                        gate_oos * 100, eval_source, MULTI3_OOS_THRESHOLD * 100)
        else:
            logger.info("Multi-3 OOS %.1f%% (%s) < %.0f%% threshold — keeping binary",
                        gate_oos * 100, eval_source, MULTI3_OOS_THRESHOLD * 100)

    # ──────────────────────────────────────────────
    # STEP 3: Save model_meta.json + backwards-compat classifier symlink
    # ──────────────────────────────────────────────
    meta = _save_model_meta(
        cfg["model_dir"], active_model_type, rich_count, binary_oos, multi3_oos,
        multi3_oos_live_accuracy=round(multi3_oos_live, 4) if multi3_oos_live is not None else None,
        multi3_oos_live_sample_size=multi3_oos_live_count,
        backtest_setups_count=backtest_count,
        active_evaluation_source=eval_source,
    )

    # Copy the active model to the canonical "classifier" path for backwards compat
    # (prediction.py loads from model_dir/classifier)
    import shutil
    canonical_path = os.path.join(cfg["model_dir"], "classifier")
    source_path = os.path.join(cfg["model_dir"],
                               "classifier_binary" if active_model_type == "binary"
                               else "classifier_multi3")
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

    # Extract narrative-relevant feature importance
    try:
        _extract_narrative_feature_importance(cfg["model_dir"], canonical_path)
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


def _extract_narrative_feature_importance(model_dir: str, classifier_path: str):
    """Extract feature importance for narrative-derived features, normalized to 0-1.

    Maps ML feature names to narrative field names and writes to
    narrative_weights_ag.json. These override EMA weights when the
    AutoGluon model is active.
    """
    if TabularPredictor is None:
        return

    predictor = TabularPredictor.load(classifier_path, verbosity=0)

    try:
        imp_df = predictor.feature_importance(silent=True)
        imp = imp_df["importance"].to_dict() if "importance" in imp_df.columns else {}
    except Exception:
        # Try native importance
        try:
            feat_names = predictor.feature_metadata_in.get_features()
            imp = {f: 1.0 / len(feat_names) for f in feat_names}
        except Exception:
            return

    if not imp:
        return

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

    narrative_scores = {}
    for feat, narrative_field in FEATURE_TO_NARRATIVE.items():
        if feat in imp:
            narrative_scores.setdefault(narrative_field, []).append(abs(imp[feat]))

    narrative_weights = {}
    for field, scores in narrative_scores.items():
        narrative_weights[field] = sum(scores) / len(scores)

    if narrative_weights:
        max_w = max(narrative_weights.values()) if narrative_weights.values() else 1
        if max_w > 0:
            narrative_weights = {k: round(v / max_w, 4)
                                 for k, v in narrative_weights.items()}

    path = os.path.join(model_dir, "narrative_weights_ag.json")
    with open(path, "w") as f:
        json.dump(narrative_weights, f, indent=2)
    logger.info("Narrative AG weights extracted: %s", narrative_weights)
