"""Prediction engine — load models, run inference, handle cold start.

The system gracefully degrades:
- 0 trades: cold_start (no predictions)
- 1-29 trades: insufficient_data (collecting, no predictions)
- 30+ trades: trained classifier available
- 50+ trades: quantile regression for SL/TP suggestions

After AutoGluon inference, the Bayesian updater, volatility calibrator,
and consensus engine are layered on top — all gracefully degrade to None.
"""
import os

from ml.config import get_config
from ml.features import extract_features
from ml.bayesian import get_beliefs
from ml.volatility import calibrate_volatility
from ml.consensus import build_consensus


def predict(analysis: dict, candles: list[dict], timeframe: str,
            db=None, config: dict = None, dataset_manager=None) -> dict:
    """Run ML prediction on an ICT analysis setup.

    Returns a dict matching PredictionResponse schema.
    """
    cfg = config or get_config()
    min_samples = cfg["min_training_samples"]

    # Extract features for potential use by trained model
    features = extract_features(analysis, candles, timeframe)

    # Check data readiness — use effective count from both DB and dataset
    db_completed = db.get_completed_trade_count() if db else 0
    dm_total = 0
    if dataset_manager is not None:
        try:
            stats = dataset_manager.get_stats()
            dm_total = stats.get("total", 0)
        except Exception:
            pass
    completed = max(db_completed, dm_total)

    if completed == 0:
        result = _cold_start_response(completed)
    elif completed < min_samples:
        result = _insufficient_data_response(completed)
    else:
        # Check for trained classifier model
        model_dir = cfg["model_dir"]
        classifier_path = os.path.join(model_dir, "classifier")

        predictor_file = os.path.join(classifier_path, "predictor.pkl")
        if not os.path.exists(predictor_file):
            result = _insufficient_data_response(completed)
        else:
            result = _run_inference(features, classifier_path, model_dir, completed, cfg)

    # Build calibration context for consensus engine
    calibration = _build_calibration(result, dataset_manager, db, cfg)
    result.update({k: v for k, v in calibration.items() if k not in result or result[k] is None})

    # Gate by WFO setup type filter (before consensus — may downgrade confidence)
    result = _apply_wfo_filter(result, analysis, candles, timeframe, cfg)

    # Layer on Bayesian + Volatility + Consensus (graceful degradation)
    result = _enrich_with_consensus(result, candles, timeframe, db, cfg, calibration=calibration)

    return result


def _cold_start_response(count: int) -> dict:
    return {
        "confidence": 0,
        "classification": {},
        "suggested_sl": None,
        "suggested_tp1": None,
        "suggested_tp2": None,
        "model_status": "cold_start",
        "training_samples": count,
        "feature_importances": {},
    }


def _insufficient_data_response(count: int) -> dict:
    return {
        "confidence": 0,
        "classification": {},
        "suggested_sl": None,
        "suggested_tp1": None,
        "suggested_tp2": None,
        "model_status": "insufficient_data",
        "training_samples": count,
        "feature_importances": {},
    }


def _run_inference(features: dict, classifier_path: str, model_dir: str,
                   count: int, cfg: dict) -> dict:
    """Run AutoGluon inference. Handles both binary and multi3 models."""
    try:
        from autogluon.tabular import TabularPredictor
        from ml.training import get_active_model_type
        import pandas as pd

        predictor = TabularPredictor.load(classifier_path, verbosity=0)
        feat_df = pd.DataFrame([features])

        # Remove non-feature columns if present
        for col in ["actual_result", "mfe", "mae", "pnl"]:
            if col in feat_df.columns:
                feat_df = feat_df.drop(columns=[col])

        probs = predictor.predict_proba(feat_df).iloc[0].to_dict()
        active_type = get_active_model_type(model_dir)

        # For binary models: map "win"/"loss" to a unified classification
        win_outcomes = {"tp1", "tp2", "tp3", "tp1_hit", "tp2_hit", "tp3_hit", "win"}
        if active_type == "binary":
            win_prob = probs.get("win", 0)
            loss_prob = probs.get("loss", 0)
            confidence = max(win_prob, loss_prob)
            # Present as unified classification
            classification = {
                "win": round(win_prob, 4),
                "loss": round(loss_prob, 4),
            }
        else:
            # Multi3: stopped_out / tp1 / runner
            confidence = max(probs.values()) if probs else 0
            classification = {k: round(v, 4) for k, v in probs.items()}
            win_prob = sum(v for k, v in probs.items() if k in win_outcomes)

        importances = {}
        try:
            imp = predictor.feature_importance(feat_df, silent=True)
            importances = imp.head(5).to_dict().get("importance", {})
        except Exception:
            pass

        result = {
            "confidence": round(confidence, 4),
            "win_probability": round(win_prob, 4),
            "classification": classification,
            "active_model_type": active_type,
            "suggested_sl": None,
            "suggested_tp1": None,
            "suggested_tp2": None,
            "model_status": "trained",
            "training_samples": count,
            "feature_importances": importances,
        }

        # Check for quantile model
        quantile_path = os.path.join(model_dir, "quantile_mfe")
        if os.path.exists(quantile_path) and count >= cfg["min_training_samples_quantile"]:
            try:
                q_predictor = TabularPredictor.load(quantile_path)
                q_pred = q_predictor.predict(feat_df).iloc[0]
                # Quantile predictions inform SL/TP suggestions
                entry_price = features.get("entry_price", 0)
                if entry_price and hasattr(q_pred, '__len__'):
                    result["suggested_tp1"] = round(float(q_pred), 2)
            except Exception:
                pass

        return result

    except (ImportError, KeyError, Exception) as e:
        import logging
        logging.getLogger(__name__).warning("AutoGluon inference failed: %s", e)
        return _insufficient_data_response(count)


def _build_calibration(result: dict, dataset_manager, db, cfg: dict) -> dict:
    """Build calibration metadata for consensus engine.

    Computes coverage score, take_trade flag, data maturity, dataset backing,
    and prior drift info. All fields degrade gracefully to None.
    """
    cal = {
        "confidence_raw": result.get("confidence"),
        "confidence_calibrated": None,
        "coverage_score": None,
        "novelty_flags": None,
        "regime_coverage": None,
        "regime_adjustment": None,
        "defensive_mode": None,
        "data_maturity": None,
        "dataset_backing": None,
        "take_trade": None,
        "prior_drift": None,
    }

    # take_trade: block if no_trade class probability is high
    classification = result.get("classification", {})
    no_trade_prob = classification.get("no_trade", 0)
    if classification:
        cal["take_trade"] = no_trade_prob <= 0.4
    else:
        cal["take_trade"] = None  # No model → can't decide

    # Dataset backing stats
    if dataset_manager is not None:
        try:
            stats = dataset_manager.get_stats()
            cal["dataset_backing"] = {
                "wfo_count": stats.get("wfo_count", 0),
                "live_count": stats.get("live_count", 0),
                "total": stats.get("total", 0),
            }
            total = stats.get("total", 0)
            live = stats.get("live_count", 0)
            if total == 0:
                cal["data_maturity"] = "cold_start"
            elif live < 10:
                cal["data_maturity"] = "early"
            elif live < 50:
                cal["data_maturity"] = "maturing"
            else:
                cal["data_maturity"] = "mature"

            # Regime coverage from dataset stats
            regime_dist = stats.get("regime_distribution", {})
            if regime_dist:
                from ml.dataset import RegimeBalancer
                balancer = RegimeBalancer(config=cfg)
                # Check if any regime is underrepresented
                total_regime = sum(regime_dist.values())
                if total_regime > 0:
                    min_pct = min(regime_dist.values()) / total_regime
                    if min_pct < 0.1:
                        cal["regime_coverage"] = "low"
                        cal["defensive_mode"] = True
                        cal["regime_adjustment"] = 0.7
                    elif min_pct < 0.2:
                        cal["regime_coverage"] = "moderate"
                        cal["defensive_mode"] = False
                        cal["regime_adjustment"] = 0.85
                    else:
                        cal["regime_coverage"] = "good"
                        cal["defensive_mode"] = False
                        cal["regime_adjustment"] = 1.0
        except Exception:
            pass

    # Prior drift check
    if db is not None:
        try:
            state = db.get_bayesian_state()
            if state is not None:
                from ml.dataset import DriftAlarm
                alarm = DriftAlarm(config=cfg)
                drift = alarm.check_drift(state)
                cal["prior_drift"] = {
                    "drift_sd": drift.get("drift_sd", 0),
                    "level": drift.get("level", "none"),
                }
        except Exception:
            pass

    return cal


def _enrich_with_consensus(result: dict, candles: list[dict], timeframe: str,
                           db, cfg: dict, calibration: dict = None) -> dict:
    """Layer Bayesian + Volatility + Consensus onto the AG result.

    All components gracefully degrade — if data is missing, fields stay None.
    """
    # Default consensus fields to None
    result.setdefault("grade", None)
    result.setdefault("blended_confidence", None)
    result.setdefault("conservative_sl", None)
    result.setdefault("volatility_regime", None)
    result.setdefault("bayesian_win_rate", None)
    result.setdefault("session", None)
    result.setdefault("reasoning", None)

    # Need a non-zero confidence to run consensus
    if result["confidence"] == 0:
        return result

    # Bayesian beliefs (may be None if no state yet)
    beliefs = None
    if db is not None:
        try:
            bayesian_state = db.get_bayesian_state()
            if bayesian_state is not None:
                beliefs = get_beliefs(bayesian_state, config=cfg)
        except Exception:
            pass

    # Volatility calibration (may fail with insufficient candles)
    vol_cal = None
    try:
        if candles and len(candles) >= 2:
            vol_cal = calibrate_volatility(candles, timeframe, config=cfg)
    except Exception:
        pass

    # Build consensus
    try:
        consensus = build_consensus(result, beliefs, vol_cal,
                                    config=cfg, calibration=calibration)
        result["grade"] = consensus["grade"]
        result["blended_confidence"] = consensus["blended_confidence"]
        result["conservative_sl"] = consensus["conservative_sl"]
        result["volatility_regime"] = consensus["volatility_regime"]
        result["bayesian_win_rate"] = consensus["bayesian_win_rate"]
        result["session"] = consensus["session"]
        result["reasoning"] = consensus["reasoning"]
    except Exception:
        pass

    return result


def _apply_wfo_filter(result: dict, analysis: dict, candles: list,
                      timeframe: str, cfg: dict) -> dict:
    """Apply WFO setup type filter to prediction result.

    Checks if the current setup type was validated as profitable by WFO
    backtesting. Downgrades confidence for unprofitable or unvalidated types.
    """
    from ml.wfo import load_report, build_setup_filter
    from ml.features import classify_setup_type

    report_path = cfg.get("wfo_report_path", "ml/models/wfo_report.json")
    report = load_report(report_path)
    if report is None or not report.setup_type_stats:
        result["wfo_filter"] = {"status": "no_report", "setup_type": None}
        return result

    setup_type = classify_setup_type(analysis, candles, timeframe)
    wfo_filter = build_setup_filter(
        report,
        min_win_rate=cfg.get("wfo_filter_min_wr", 0.40),
        min_trades=cfg.get("wfo_filter_min_trades", 3),
    )

    filter_result = {
        "setup_type": setup_type,
        "stats": wfo_filter["stats"].get(setup_type),
    }

    # Exact match
    if setup_type in wfo_filter["profitable"]:
        filter_result["status"] = "validated"
        filter_result["action"] = "pass"
    elif setup_type in wfo_filter["unprofitable"]:
        filter_result["status"] = "unprofitable"
        filter_result["action"] = "downgrade"
        result["confidence"] = round(result["confidence"] * 0.5, 4)
        result.setdefault("reasoning", [])
        if isinstance(result["reasoning"], list):
            wr = wfo_filter["stats"][setup_type]["win_rate"]
            result["reasoning"].append(
                f"WFO filter: {setup_type} has {wr:.0%} win rate — confidence halved"
            )
    else:
        # Fuzzy match: check tag subset/superset
        matched = _fuzzy_match_setup_type(setup_type, wfo_filter)
        if matched:
            filter_result["status"] = "partial_match"
            filter_result["matched_to"] = matched["type"]
            filter_result["action"] = matched["action"]
            if matched["action"] == "downgrade":
                result["confidence"] = round(result["confidence"] * 0.7, 4)
                result.setdefault("reasoning", [])
                if isinstance(result["reasoning"], list):
                    result["reasoning"].append(
                        f"WFO filter: similar type {matched['type']} unprofitable — confidence reduced"
                    )
        else:
            filter_result["status"] = "unvalidated"
            filter_result["action"] = "caution"
            result["confidence"] = round(result["confidence"] * 0.8, 4)
            result.setdefault("reasoning", [])
            if isinstance(result["reasoning"], list):
                result["reasoning"].append(
                    f"WFO filter: {setup_type} not seen in backtest — confidence reduced 20%"
                )

    result["wfo_filter"] = filter_result
    return result


def _fuzzy_match_setup_type(setup_type: str, wfo_filter: dict) -> dict | None:
    """Try tag subset/superset matching against WFO setup types.

    Matches if the direction is the same and the tag overlap is within 1
    of the query tag count (i.e., at most 1 tag missing).
    """
    parts = setup_type.split("_")
    direction = parts[0]  # bull or bear
    tags = set(parts[1:])

    best = None
    best_overlap = 0

    for category in ["profitable", "unprofitable"]:
        for st in wfo_filter[category]:
            st_parts = st.split("_")
            if st_parts[0] != direction:
                continue
            st_tags = set(st_parts[1:])
            overlap = len(tags & st_tags)
            if overlap > best_overlap and overlap >= len(tags) - 1:
                best_overlap = overlap
                action = "pass" if category == "profitable" else "downgrade"
                best = {"type": st, "action": action, "overlap": overlap}

    return best
