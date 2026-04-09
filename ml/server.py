# Load .env before anything reads os.environ
from dotenv import load_dotenv as _load_dotenv
import pathlib as _pathlib
_load_dotenv(_pathlib.Path(__file__).resolve().parent.parent / ".env", override=True)
del _load_dotenv, _pathlib

"""FastAPI ML prediction server for ICT terminal.

Endpoints:
  GET  /health            — deep health check (scheduler, DB, API keys)
  GET  /status            — model status + training sample count
  POST /predict           — run ML prediction on analysis + candles
  POST /log-setup         — log a trade setup for future training
  POST /log-outcome       — record outcome for a logged setup + Bayesian update
  GET  /pending-outcomes  — setups awaiting outcome resolution
  GET  /beliefs           — current Bayesian belief summary
  POST /retrain           — manually trigger retraining
  POST /wfo/run           — run walk-forward optimization on candle data
  GET  /wfo/report        — retrieve saved WFO report
  GET  /dataset/stats     — training dataset statistics
  GET  /bayesian/drift    — prior drift check

  # v2 Claude-Primary endpoints
  POST /seed              — run V1 data harvesting and seed ML models
  POST /calibrate         — calibrate Claude's analysis with ML layers
  POST /trade/complete    — log completed trade outcome
  GET  /claude/accuracy   — Claude accuracy tracker
  GET  /calibration/value — calibration value-add summary
  GET  /prompt/enhanced   — enhanced multi-timeframe prompt template
  GET  /seed/stats        — V1 session statistics and seeding status
"""
import json
import logging
import os
import signal
import subprocess
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from ml.config import get_config
from ml.database import TradeLogger
from ml.features import extract_features
from ml.prediction import predict
from ml.bayesian import get_default_prior, update_beliefs, get_beliefs
from ml.models import (
    PredictionRequest, PredictionResponse, SetupLogRequest,
    TradeOutcomeRequest, ModelStatus, WFORunRequest,
)

logger = logging.getLogger(__name__)


def _kill_stale_port(port: int = 8000):
    """Kill any process holding our port (stale from previous crash).

    Prevents [Errno 48] address already in use on restart.
    Only kills processes that are NOT us.
    Skipped in containerized deployments (no lsof, single-process-per-port).
    """
    if os.environ.get("RAILWAY_ENVIRONMENT") or os.environ.get("DATA_DIR"):
        return
    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"], capture_output=True, text=True
        )
        pids = result.stdout.strip().split()
        my_pid = str(os.getpid())
        for pid in pids:
            if pid and pid != my_pid:
                try:
                    os.kill(int(pid), signal.SIGKILL)
                    print(f"[STARTUP] Killed stale process {pid} on port {port}", flush=True)
                except ProcessLookupError:
                    pass  # Already dead
    except Exception as e:
        print(f"[STARTUP] Port cleanup skipped: {e}", flush=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start scanner on server startup, stop on shutdown."""
    import sys
    print("=== LIFESPAN STARTUP ===", flush=True)
    sys.stdout.flush()

    # Kill any stale process holding port 8000 from a previous crash
    _kill_stale_port(8000)

    stop_scheduler = None
    try:
        from ml.scheduler import start_scheduler, stop_scheduler
        start_scheduler()
        print("=== LIFESPAN: scheduler called ===", flush=True)
    except Exception as e:
        print(f"=== LIFESPAN ERROR: {e} ===", flush=True)
        import traceback
        traceback.print_exc()
    yield
    if stop_scheduler:
        stop_scheduler()


app = FastAPI(title="ICT ML Prediction Server", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_db_instance = None
_dataset_manager_instance = None


def get_db() -> TradeLogger:
    """Dependency — singleton TradeLogger."""
    global _db_instance
    if _db_instance is None:
        cfg = get_config()
        os.makedirs(os.path.dirname(cfg["db_path"]) or ".", exist_ok=True)
        _db_instance = TradeLogger(config=cfg)
    return _db_instance


def get_dataset_manager():
    """Dependency — singleton TrainingDatasetManager."""
    global _dataset_manager_instance
    if _dataset_manager_instance is None:
        from ml.dataset import TrainingDatasetManager
        _dataset_manager_instance = TrainingDatasetManager()
    return _dataset_manager_instance


@app.get("/health")
def health():
    """Deep health check — verifies scheduler, database, and API keys."""
    try:
        from ml.scheduler import is_running
        scheduler_ok = is_running()
    except ImportError:
        scheduler_ok = False

    checks = {
        "server": "ok",
        "scheduler": "ok" if scheduler_ok else "down",
        "api_key": "ok" if os.environ.get("ANTHROPIC_API_KEY") else "missing",
        "oanda": "ok" if os.environ.get("OANDA_ACCESS_TOKEN") else "missing",
    }

    try:
        db = get_db()
        db.get_completed_trade_count()
        checks["database"] = "ok"
    except Exception as e:
        checks["database"] = f"error: {e}"

    all_ok = all(v == "ok" for v in checks.values())
    return JSONResponse(
        content={"status": "ok" if all_ok else "degraded", "checks": checks},
        status_code=200 if all_ok else 503,
    )


@app.get("/status")
def status(db: TradeLogger = Depends(get_db), dm=Depends(get_dataset_manager)):
    cfg = get_config()
    completed = db.get_completed_trade_count()
    pending = len(db.get_setups_without_outcomes())
    last = db.get_last_training("classifier")

    # Dataset-backed trade count
    dm_total = 0
    try:
        stats = dm.get_stats()
        dm_total = stats.get("total", 0)
    except Exception:
        pass

    classifier_path = os.path.join(cfg["model_dir"], "classifier")
    quantile_path = os.path.join(cfg["model_dir"], "quantile_mfe")

    effective = max(completed, dm_total)

    return ModelStatus(
        classifier_trained=os.path.exists(classifier_path),
        quantile_trained=os.path.exists(quantile_path),
        total_trades_logged=completed + pending,
        completed_trades=effective,
        dataset_trades=dm_total,
        last_trained=last["timestamp"] if last else None,
        next_retrain_trigger=max(0, cfg["retrain_on_n_new_trades"] - (
            effective - (last["samples_used"] if last else 0))),
    ).model_dump()


@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(req: PredictionRequest, db: TradeLogger = Depends(get_db),
                     dm=Depends(get_dataset_manager)):
    analysis = req.analysis.model_dump()
    candles = [c.model_dump() for c in req.candles]
    result = predict(analysis, candles, req.timeframe, db=db, dataset_manager=dm)
    return result


@app.post("/log-setup", status_code=201)
def log_setup(req: SetupLogRequest, db: TradeLogger = Depends(get_db)):
    analysis = req.analysis.model_dump()
    candles = [c.model_dump() for c in req.candles]
    features = extract_features(analysis, candles, req.timeframe)

    entry = analysis.get("entry") or {}
    sl = analysis.get("stopLoss") or {}
    tps = analysis.get("takeProfits", [])

    db.log_setup(
        setup_id=req.setup_id,
        features=features,
        analysis_json=json.dumps(analysis),
        candles_json=json.dumps(candles),
        metadata={
            "timeframe": req.timeframe,
            "bias": analysis.get("bias", ""),
            "direction": entry.get("direction", ""),
            "entry_price": entry.get("price", 0),
            "sl_price": sl.get("price", 0),
            "tp1_price": tps[0]["price"] if tps else 0,
            "tp2_price": tps[1]["price"] if len(tps) > 1 else None,
            "tp3_price": tps[2]["price"] if len(tps) > 2 else None,
        },
    )
    return {"setup_id": req.setup_id, "status": "logged"}


@app.post("/log-outcome")
def log_outcome(req: TradeOutcomeRequest, db: TradeLogger = Depends(get_db),
                dm=Depends(get_dataset_manager)):
    success = db.log_outcome(
        req.setup_id, req.result,
        req.max_favorable_excursion, req.max_adverse_excursion, req.pnl_pips,
    )
    if not success:
        raise HTTPException(status_code=404, detail=f"Setup {req.setup_id} not found")

    # Update Bayesian beliefs after recording outcome
    prior = db.get_bayesian_state() or get_default_prior()
    posterior = update_beliefs(prior, req.result, req.pnl_pips)
    db.save_bayesian_state(posterior)

    # Ingest live trade into persistent dataset
    cfg = get_config()
    try:
        # Get features from the stored setup
        features = extract_features({}, [], "1h")  # placeholder features
        dm.ingest_live_trade(features, req.result,
                             req.max_favorable_excursion,
                             req.max_adverse_excursion, req.pnl_pips)

        # Periodic drift check
        interval = cfg.get("drift_check_interval", 20)
        if posterior["total_trades"] % interval == 0 and posterior["total_trades"] > 0:
            from ml.dataset import DriftAlarm
            drift = DriftAlarm(config=cfg).check_drift(posterior)
            if drift.get("level") in ("significant", "critical"):
                logger.warning("Prior drift detected: %s", drift)
    except Exception:
        pass  # graceful degradation

    return {"setup_id": req.setup_id, "status": "outcome_recorded"}


@app.delete("/delete-setup/{setup_id}")
def delete_setup(setup_id: str, db: TradeLogger = Depends(get_db)):
    success = db.delete_setup(setup_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Setup {setup_id} not found")
    return {"setup_id": setup_id, "status": "deleted"}


@app.get("/pending-outcomes")
def pending_outcomes(db: TradeLogger = Depends(get_db)):
    return db.get_setups_without_outcomes()


@app.get("/beliefs")
def beliefs(db: TradeLogger = Depends(get_db)):
    """Return current Bayesian belief summary, or null if no trades yet."""
    state = db.get_bayesian_state()
    if state is None:
        return None
    return get_beliefs(state)


@app.post("/model/reset")
def model_reset():
    """Wipe old AutoGluon model files so retrain starts fresh.
    Also clears stale SQLite journal/WAL files that cause disk I/O errors."""
    import shutil, glob as _glob
    cfg = get_config()
    removed = []
    for name in ("classifier", "classifier_binary", "classifier_multi3",
                 "quantile_mfe", "model_meta.json", "classifier_evaluation.json"):
        p = os.path.join(cfg["model_dir"], name)
        if os.path.exists(p):
            if os.path.isdir(p):
                shutil.rmtree(p)
            else:
                os.remove(p)
            removed.append(name)
    # Clear stale SQLite lock/journal files
    db_path = cfg.get("db_path", "")
    for suffix in ("-journal", "-wal", "-shm"):
        jf = db_path + suffix
        if os.path.exists(jf):
            os.remove(jf)
            removed.append(os.path.basename(jf))
    return {"status": "reset", "removed": removed}


@app.post("/retrain")
def retrain(db: TradeLogger = Depends(get_db), dm=Depends(get_dataset_manager)):
    cfg = get_config()

    # Try blended dataset first, fall back to DB-only
    try:
        stats = dm.get_stats()
        total = stats.get("total", 0)
    except Exception:
        total = 0

    completed = db.get_completed_trade_count()
    effective_count = max(total, completed)

    if effective_count < cfg["min_training_samples"]:
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient data: {effective_count}/{cfg['min_training_samples']} trades needed",
        )

    try:
        from ml.training import train_classifier
        result = train_classifier(db, config=cfg, dataset_manager=dm if total > 0 else None)

        # Auto-evaluate after retrain
        try:
            from ml.evaluation import evaluate_classifier_walkforward
            eval_result = evaluate_classifier_walkforward(dm if total > 0 else None, config=cfg)
            result["evaluation"] = eval_result
        except Exception as e:
            result["evaluation_error"] = str(e)

        return result
    except ImportError:
        raise HTTPException(status_code=500, detail="AutoGluon not installed")


@app.get("/model/info")
def model_info():
    """Return AutoGluon model details: leaderboard, feature importances, status."""
    cfg = get_config()
    model_path = os.path.join(cfg["model_dir"], "classifier")

    if not os.path.exists(os.path.join(model_path, "predictor.pkl")):
        return {
            "status": "not_trained",
            "leaderboard": [],
            "feature_importances": {},
            "models_used": [],
        }

    try:
        from autogluon.tabular import TabularPredictor
        predictor = TabularPredictor.load(model_path, verbosity=0,
                                          require_version_match=False)

        # Leaderboard: model names + scores
        lb = predictor.leaderboard(silent=True)
        leaderboard = []
        for _, row in lb.iterrows():
            leaderboard.append({
                "model": row.get("model", "?"),
                "score_val": round(row.get("score_val", 0), 4),
                "fit_time": round(row.get("fit_time", 0), 1),
            })

        # Feature importances — try model-native first, then permutation
        feat_imp = {}
        try:
            # Model-native feature importance (fast, no data needed)
            imp_dict = predictor.feature_importance(
                silent=True, importance_type="native"
            )
            feat_imp = {k: round(v, 4) for k, v in imp_dict.head(10).to_dict().get("importance", {}).items()}
        except Exception:
            pass
        if not feat_imp:
            # Fallback: use feature names with equal importance as placeholder
            try:
                feat_imp = {f: 1.0 for f in predictor.feature_metadata.get_features()[:10]}
            except Exception:
                pass

        models_used = [r["model"] for r in leaderboard]

        response = {
            "status": "trained",
            "leaderboard": leaderboard,
            "feature_importances": feat_imp,
            "models_used": models_used,
            "best_model": models_used[0] if models_used else None,
        }

        # Merge model meta (binary vs multi3)
        meta_path = os.path.join(cfg["model_dir"], "model_meta.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                response["active_model_type"] = meta.get("active_model_type", "binary")
                response["rich_feature_count"] = meta.get("rich_feature_count", 0)
                response["binary_oos_accuracy"] = meta.get("binary_oos_accuracy")
                response["multi3_oos_accuracy"] = meta.get("multi3_oos_accuracy")
                response["upgraded_to_multi3"] = meta.get("upgraded_to_multi3", False)
            except Exception:
                pass

        # Merge evaluation data if available
        eval_path = os.path.join(cfg["model_dir"], "classifier_evaluation.json")
        if os.path.exists(eval_path):
            try:
                with open(eval_path) as f:
                    eval_data = json.load(f)
                response["oos_accuracy"] = eval_data.get("oos_accuracy")
                response["training_accuracy"] = leaderboard[0]["score_val"] if leaderboard else None
                response["model_trustworthy"] = eval_data.get("model_trustworthy", False)
                response["oos_test_trades"] = eval_data.get("test_trades", 0)
                response["feature_quality"] = eval_data.get("feature_quality")
                response["weaknesses"] = eval_data.get("weaknesses", [])
                response["periods"] = eval_data.get("periods")
                response["by_timeframe"] = eval_data.get("by_timeframe")
                response["calibration_curve"] = eval_data.get("calibration_curve")
                response["evaluated_at"] = eval_data.get("evaluated_at")
            except Exception:
                pass

        # Merge retrain history if available
        history_path = os.path.join(cfg["model_dir"], "retrain_history.json")
        if os.path.exists(history_path):
            try:
                with open(history_path) as f:
                    history = json.load(f)
                response["retrain_history"] = history[-5:]  # last 5
                # Compute improvement trend
                if len(history) >= 2:
                    accs = [h["oos_accuracy_after"] for h in history if h.get("oos_accuracy_after")]
                    response["improvement_trend"] = round(accs[-1] - accs[0], 4) if len(accs) >= 2 else 0
            except Exception:
                pass

        # Next retrain info
        try:
            from ml.scanner_db import ScannerDB
            sdb = ScannerDB()
            stats = sdb.get_stats()
            resolved = stats.get("resolved", 0)
            retrain_n = cfg.get("retrain_every_n_trades", 50)
            last_retrain = 0
            if os.path.exists(history_path):
                with open(history_path) as f:
                    h = json.load(f)
                if h:
                    last_retrain = h[-1].get("trades_used", 0)
            response["next_retrain_in"] = max(0, retrain_n - (resolved - last_retrain))
        except Exception:
            pass

        return response
    except ImportError:
        return {"status": "autogluon_not_installed", "leaderboard": [], "feature_importances": {}, "models_used": []}
    except Exception as e:
        return {"status": f"error: {str(e)}", "leaderboard": [], "feature_importances": {}, "models_used": []}


@app.post("/model/evaluate")
def model_evaluate(dm=Depends(get_dataset_manager)):
    """Run full walk-forward evaluation of the classifier."""
    try:
        from ml.evaluation import evaluate_classifier_walkforward
        result = evaluate_classifier_walkforward(dm, config=get_config())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/wfo/run")
def wfo_run(req: WFORunRequest, db: TradeLogger = Depends(get_db)):
    """Run walk-forward optimization on provided candle data."""
    cfg = get_config()
    min_candles = req.train_window + req.test_window
    candles = [c.model_dump() for c in req.candles]

    if len(candles) < min_candles:
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient candles: {len(candles)}/{min_candles} needed "
                   f"(train_window={req.train_window} + test_window={req.test_window})",
        )

    from ml.wfo import WalkForwardEngine, WFOConfig, save_report, update_bayesian_from_wfo

    wfo_cfg = WFOConfig(
        train_window=req.train_window,
        test_window=req.test_window,
        step_size=req.step_size,
        sl_atr_mult=cfg.get("wfo_sl_atr_mult", 1.5),
        tp_atr_mults=cfg.get("wfo_tp_atr_mults", [1.0, 2.0, 3.5]),
        max_bars_in_trade=cfg.get("wfo_max_bars_in_trade", 20),
        max_folds=cfg.get("wfo_max_folds", 20),
        displacement_threshold=cfg.get("wfo_displacement_threshold", 1.5),
        min_confluence_score=cfg.get("wfo_min_confluence_score", 2),
        min_setups_per_fold=cfg.get("wfo_min_setups_per_fold", 5),
    )

    engine = WalkForwardEngine(wfo_cfg, use_autogluon=not req.no_autogluon)
    report = engine.run(candles, req.timeframe)

    report_path = cfg.get("wfo_report_path", "ml/models/wfo_report.json")
    save_report(report, report_path)
    update_bayesian_from_wfo(report, db)

    result = report.to_dict()

    # Auto-ingest OOS trades into persistent dataset (unless skip_ingest)
    if not req.skip_ingest:
        try:
            oos_trades = getattr(engine, "oos_trades", [])
            ingested = 0
            negative = 0
            auto_trained = False

            if oos_trades:
                from ml.execution import ExecutionSimulator
                from ml.quality_filter import SetupQualityFilter
                from ml.dataset import TrainingDatasetManager, generate_negative_examples

                # Apply execution costs
                sim = ExecutionSimulator(config=cfg)
                trades = sim.simulate(oos_trades, candles)

                # Quality filter
                qf = SetupQualityFilter(config=cfg)
                trades = qf.filter_basic(trades)

                # Generate negative examples
                neg_examples = generate_negative_examples(candles, trades, config=cfg)
                all_trades = trades + neg_examples
                negative = len(neg_examples)

                # Ingest into dataset
                dm = get_dataset_manager()
                ingested = dm.ingest_wfo_trades(all_trades)

                # Auto-train if enough data
                stats = dm.get_stats()
                if stats.get("total", 0) >= cfg["min_training_samples"]:
                    try:
                        from ml.training import train_classifier
                        train_classifier(db, config=cfg, dataset_manager=dm)
                        auto_trained = True
                    except Exception:
                        pass

            result["wfo_trades_ingested"] = ingested
            result["negative_examples"] = negative
            result["auto_trained"] = auto_trained
        except Exception as e:
            logger.warning("WFO ingest failed: %s", e)
            result["wfo_trades_ingested"] = 0
            result["negative_examples"] = 0
            result["auto_trained"] = False

    return result


@app.get("/wfo/report")
def wfo_report():
    """Return saved WFO report, or null if none exists."""
    cfg = get_config()
    report_path = cfg.get("wfo_report_path", "ml/models/wfo_report.json")

    from ml.wfo import load_report
    report = load_report(report_path)
    if report is None:
        return None
    return report.to_dict()


@app.get("/trades/history")
def trades_history(db: TradeLogger = Depends(get_db)):
    """Return completed trades with running cumulative PnL."""
    return db.get_trade_history()


@app.get("/dataset/stats")
def dataset_stats(dm=Depends(get_dataset_manager)):
    """Return training dataset statistics."""
    try:
        return dm.get_stats()
    except Exception:
        return {"total": 0, "wfo_count": 0, "live_count": 0,
                "regime_distribution": {}, "outcome_distribution": {}}


@app.get("/bayesian/drift")
def bayesian_drift(db: TradeLogger = Depends(get_db)):
    """Return prior drift check results — compared against V1 seed priors."""
    state = db.get_bayesian_state()
    if state is None:
        return {"drift_sd": 0, "level": "none", "recommendation": "No Bayesian state yet"}
    from ml.dataset import DriftAlarm
    cfg = get_config()
    alarm = DriftAlarm(config=cfg)

    # Use V1 seed priors as reference (not Beta(1,1) uniform)
    reference = None
    priors_path = os.path.join(cfg["model_dir"], "v1_bayesian_priors.json")
    if os.path.exists(priors_path):
        try:
            with open(priors_path) as f:
                v1 = json.load(f)
            reference = {
                "alpha": v1.get("win_alpha", 1.0),
                "beta_param": v1.get("win_beta", 1.0),
            }
        except Exception:
            pass
    return alarm.check_drift(state, reference_state=reference)


@app.post("/bayesian/reset")
def bayesian_reset(db: TradeLogger = Depends(get_db)):
    """Reset Bayesian state to V1 seed priors.

    Use when synthetic data (backtest/WFO) has contaminated beliefs.
    Only live trades should update Bayesian beliefs going forward.
    """
    cfg = get_config()
    priors_path = os.path.join(cfg["model_dir"], "v1_bayesian_priors.json")
    if not os.path.exists(priors_path):
        raise HTTPException(status_code=404, detail="No V1 priors found — run seed first")

    with open(priors_path) as f:
        v1 = json.load(f)

    reset_state = {
        "alpha": v1.get("win_alpha", 1.0),
        "beta_param": v1.get("win_beta", 1.0),
        "consecutive_losses": 0,
        "max_consecutive_losses": 0,
        "current_drawdown": 0.0,
        "max_drawdown": 0.0,
        "total_trades": 0,
        "total_wins": 0,
        "total_losses": 0,
        "cumulative_pnl": 0.0,
        "peak_pnl": 0.0,
    }
    db.save_bayesian_state(reset_state)
    return {
        "status": "reset",
        "alpha": reset_state["alpha"],
        "beta_param": reset_state["beta_param"],
        "win_rate": round(reset_state["alpha"] / (reset_state["alpha"] + reset_state["beta_param"]), 4),
    }


# ═══════════════════════════════════════════════════════════════════════
# v2 Claude-Primary Endpoints
# ═══════════════════════════════════════════════════════════════════════


@app.post("/seed")
def seed_endpoint():
    """Run V1 data harvesting and seed all ML models.

    Accepts JSON body with either:
      - {"candles": [...], "candles_count": 2000}
      - {"td_key": "...", "candles_count": 2000}
    """
    from fastapi import Request
    import asyncio

    # Use a simpler approach — accept raw JSON
    from starlette.requests import Request as StarletteRequest

    # This endpoint is called with JSON body
    return _seed_handler()


def _seed_handler():
    """Internal seed handler — called from endpoint or CLI."""
    raise HTTPException(status_code=501, detail="Use CLI: python -m ml.seed --td-key KEY --candles 2000")


@app.post("/seed/run")
async def seed_run(request: dict = None):
    """Run V1 seeding with provided candles or Twelve Data key."""
    import pandas as pd
    from ml.seed import V1DataHarvester, fetch_candles
    from ml.dataset import TrainingDatasetManager

    body = request or {}
    td_key = body.get("td_key")
    candles_data = body.get("candles")
    candles_count = body.get("candles_count", 2000)

    if candles_data:
        df = pd.DataFrame(candles_data)
        for col in ["open", "high", "low", "close"]:
            df[col] = df[col].astype(float)
    elif td_key:
        df = fetch_candles(td_key, candles_count)
    else:
        raise HTTPException(
            status_code=400,
            detail="Provide either 'candles' array or 'td_key' for Twelve Data"
        )

    harvester = V1DataHarvester()
    harvested = harvester.harvest_v1_data(df)

    if harvested.empty:
        return {"status": "no_trades", "trades_harvested": 0}

    result = harvester.seed_bayesian(harvested)
    dm = TrainingDatasetManager()
    count = harvester.seed_training_dataset(harvested, dm)

    return {
        "status": "seeded",
        "trades_harvested": len(harvested),
        "priors": result["priors"],
        "session_stats": result["session_stats"],
        "dataset_rows": count,
    }


@app.post("/calibrate")
async def calibrate_endpoint(request: dict):
    """Calibrate Claude's ICT analysis with ML layers.

    Body: {
        "analysis": { ... },    // Claude's ICT analysis JSON
        "candles": [ ... ],     // 1H OHLC candles
        "candles_4h": [ ... ]   // optional 4H candles
    }
    """
    from ml.claude_bridge import ClaudeAnalysisBridge
    from ml.calibrate import MLCalibrator

    analysis = request.get("analysis")
    candles = request.get("candles", [])

    if not analysis:
        raise HTTPException(status_code=400, detail="Missing 'analysis' field")
    if not candles:
        raise HTTPException(status_code=400, detail="Missing 'candles' field")

    bridge = ClaudeAnalysisBridge()
    parsed = bridge.parse_analysis(analysis, candles)

    calibrator = MLCalibrator()
    result = calibrator.calibrate_trade(parsed, candles)

    return result


@app.post("/trade/complete")
async def trade_complete(request: dict):
    """Log a completed trade and update all models.

    Body: {
        "original_analysis": { ... },
        "calibrated_result": { ... },
        "actual_outcome": "tp1|tp2|tp3|stopped_out|breakeven|manual_close",
        "actual_pnl_atr": 1.5,
        "used_calibrated_sl": true,
        "notes": "string"
    }
    """
    from ml.claude_bridge import ClaudeAnalysisBridge

    bridge = ClaudeAnalysisBridge()
    result = bridge.log_completed_trade(
        original_analysis=request.get("original_analysis", {}),
        calibrated_result=request.get("calibrated_result", {}),
        actual_outcome=request.get("actual_outcome", "stopped_out"),
        actual_pnl_atr=request.get("actual_pnl_atr", 0),
        used_calibrated_sl=request.get("used_calibrated_sl", False),
        notes=request.get("notes", ""),
    )

    return result


@app.get("/claude/accuracy")
def claude_accuracy():
    """Return Claude accuracy tracker."""
    from ml.claude_bridge import ClaudeAnalysisBridge
    bridge = ClaudeAnalysisBridge()
    return bridge._accuracy


@app.get("/calibration/value")
def calibration_value():
    """Return calibration value-add summary."""
    from ml.claude_bridge import ClaudeAnalysisBridge
    bridge = ClaudeAnalysisBridge()
    return bridge.get_calibration_value()


@app.get("/prompt/enhanced")
def prompt_enhanced():
    """Return the enhanced multi-timeframe ICT analysis prompt template."""
    from ml.prompts import build_enhanced_ict_prompt
    # Return with placeholder data so frontend knows the format
    prompt = build_enhanced_ict_prompt(
        [{"datetime": "example", "open": 0, "high": 0, "low": 0, "close": 0}],
        [{"datetime": "example", "open": 0, "high": 0, "low": 0, "close": 0}],
    )
    return {"prompt_template": prompt}


@app.get("/seed/stats")
def seed_stats():
    """Return V1 session statistics and seeding status."""
    import os
    models_dir = os.path.join(os.path.dirname(__file__), "models")

    stats_path = os.path.join(models_dir, "v1_session_stats.json")
    priors_path = os.path.join(models_dir, "v1_bayesian_priors.json")

    session_stats = {}
    priors = {}
    seeded = False

    if os.path.exists(stats_path):
        with open(stats_path) as f:
            session_stats = json.load(f)
        seeded = True

    if os.path.exists(priors_path):
        with open(priors_path) as f:
            priors = json.load(f)

    # Also get dataset stats
    dm = get_dataset_manager()
    ds_stats = dm.get_stats()

    return {
        "seeded": seeded,
        "session_stats": session_stats,
        "bayesian_priors": priors,
        "dataset_stats": ds_stats,
    }


# ═══════════════════════════════════════════════════════════════════════
# Scanner Endpoints — headless background scanning
# ═══════════════════════════════════════════════════════════════════════

_scanner_engine = None


def _get_scanner():
    global _scanner_engine
    if _scanner_engine is None:
        from ml.scanner import ScannerEngine
        _scanner_engine = ScannerEngine()
    return _scanner_engine


@app.get("/scanner/pending")
def scanner_pending():
    """Return all pending setups from the scanner."""
    engine = _get_scanner()
    return engine.db.get_pending()


@app.get("/scanner/history")
def scanner_history():
    """Return all resolved scanner setups."""
    engine = _get_scanner()
    return engine.db.get_history()


@app.get("/scanner/pnl")
def scanner_pnl():
    """Return all resolved setups with pnl_rr for PnL calculation (chronological)."""
    engine = _get_scanner()
    return engine.db.get_pnl_history()


@app.get("/scanner/prospects")
def scanner_prospects():
    """Return active + displaced prospects with current phase."""
    engine = _get_scanner()
    prospects = engine.db.get_active_prospects(include_displaced=True)
    return {"prospects": prospects, "count": len(prospects)}


# ═══════════════════════════════════════════════════════════════════════
# Entry Placement Endpoints
# ═══════════════════════════════════════════════════════════════════════

@app.get("/entry-placement/stats")
def entry_placement_stats():
    """Get current entry placement statistics."""
    from ml.entry_placement import EntryPlacementAnalyzer
    analyzer = EntryPlacementAnalyzer()
    return analyzer.compute_summary()


@app.get("/entry-placement/guidance")
def entry_placement_guidance():
    """Get current placement guidance for prompt injection."""
    from ml.entry_placement import EntryPlacementAnalyzer
    analyzer = EntryPlacementAnalyzer()
    return analyzer.get_placement_guidance()


@app.post("/entry-placement/backfill")
def entry_placement_backfill():
    """Backfill entry zone data for existing setups."""
    from ml.backfill_entry_zones import backfill_entry_zones
    from ml.scanner_db import ScannerDB
    db = ScannerDB()
    result = backfill_entry_zones(db)
    return result


@app.post("/entry-placement/recompute")
def entry_placement_recompute():
    """Recompute placement statistics from all resolved setups with zones."""
    import sqlite3 as _sqlite3
    from ml.entry_placement import EntryPlacementAnalyzer, extract_entry_zone_metrics
    from ml.scanner_db import ScannerDB

    db = ScannerDB()
    analyzer = EntryPlacementAnalyzer()

    with db._conn() as conn:
        conn.row_factory = _sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM scanner_setups WHERE outcome IS NOT NULL "
            "AND outcome != '' AND outcome != 'expired' "
            "AND entry_zone_position IS NOT NULL "
            "AND mfe_atr IS NOT NULL"
        ).fetchall()

    ingested = 0
    errors = 0
    for row in rows:
        try:
            setup = dict(row)
            metric = extract_entry_zone_metrics(setup)
            if metric:
                analyzer.ingest_metric(metric)
                ingested += 1
        except Exception:
            errors += 1

    summary = analyzer.compute_summary()
    return {"ingested": ingested, "errors": errors, "summary": summary}


# ═══════════════════════════════════════════════════════════════════════
# Annotation Analysis Endpoints
# ═══════════════════════════════════════════════════════════════════════

@app.get("/annotations/analysis")
def annotations_analysis():
    """Compare Claude's OB/FVG annotations vs trade outcomes."""
    from ml.analyse_annotations import analyse
    return analyse()


# ═══════════════════════════════════════════════════════════════════════
# Calibration Layer Performance Endpoints
# ═══════════════════════════════════════════════════════════════════════

@app.get("/calibration/layers")
def calibration_layers():
    """Per-layer performance report."""
    from ml.layer_performance import LayerPerformanceTracker
    tracker = LayerPerformanceTracker()
    return tracker.get_layer_report()


@app.post("/calibration/layers/recompute")
def recompute_layers():
    """Recompute all layer stats from resolved trades."""
    from ml.layer_performance import LayerPerformanceTracker
    from ml.scanner_db import ScannerDB
    tracker = LayerPerformanceTracker()
    return tracker.full_recompute(ScannerDB())


@app.get("/calibration/sl-floor")
def sl_floor_info():
    """Current SL floor and adaptive overrides per segment."""
    from ml.layer_performance import LayerPerformanceTracker
    tracker = LayerPerformanceTracker()
    segments = {}
    for seg_key in tracker._stats.get("segments", {}).keys():
        parts = seg_key.split("_", 1)
        grade = parts[0] if parts else ""
        kz = parts[1] if len(parts) > 1 else ""
        segments[seg_key] = tracker.get_adaptive_floor(grade, kz)
    return {
        "default_floor_atr": tracker.cfg.get("sl_floor_atr", 3.0),
        "segments": segments,
    }


# ═══════════════════════════════════════════════════════════════════════
# Cost Tracking Endpoints
# ═══════════════════════════════════════════════════════════════════════

@app.get("/cost/today")
def cost_today():
    """Current day's API spend by model and purpose."""
    from ml.cost_tracker import get_cost_tracker
    tracker = get_cost_tracker()
    summary = tracker.get_daily_summary()
    summary["remaining_usd"] = tracker.get_remaining_budget()
    summary["daily_limit_usd"] = cfg.get("daily_api_budget_usd", 5.0)
    summary["warning"] = tracker.is_warning()
    return summary


@app.get("/cost/history")
def cost_history():
    """Last 30 days of daily API costs."""
    from ml.cost_tracker import get_cost_tracker
    return {"history": get_cost_tracker().get_history(30)}


@app.get("/cost/budget")
def cost_budget():
    """Budget status — remaining, projected, limit."""
    from ml.cost_tracker import get_cost_tracker
    tracker = get_cost_tracker()
    summary = tracker.get_daily_summary()
    return {
        "daily_limit_usd": cfg.get("daily_api_budget_usd", 5.0),
        "spent_today_usd": summary["total_usd"],
        "remaining_usd": tracker.get_remaining_budget(),
        "within_budget": tracker.check_budget(),
        "warning": tracker.is_warning(),
        "call_count_today": summary["call_count"],
        "by_purpose": summary["by_purpose"],
    }


# ═══════════════════════════════════════════════════════════════════════
# P7: Intermarket Signal Validation
# ═══════════════════════════════════════════════════════════════════════

@app.get("/intermarket/validation")
def intermarket_validation():
    """Stratified analysis of intermarket signal accuracy."""
    from ml.intermarket_validator import IntermarketValidator
    v = IntermarketValidator()
    result = v.get_last_result()
    if result:
        return result
    return {"status": "no_data", "message": "Run POST /intermarket/recompute first"}


@app.post("/intermarket/recompute")
def intermarket_recompute():
    """Recompute intermarket validation from resolved trades."""
    from ml.intermarket_validator import IntermarketValidator
    db = _get_scanner().db
    import sqlite3
    with db._conn() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT outcome, killzone, direction, calibration_json "
            "FROM scanner_setups "
            "WHERE outcome IS NOT NULL AND calibration_json IS NOT NULL"
        ).fetchall()
    trades = [dict(r) for r in rows]
    v = IntermarketValidator()
    return v.analyze(trades)


# ═══════════════════════════════════════════════════════════════════════
# P6+P8: Killzone Performance Profiling + Adaptive Scan
# ═══════════════════════════════════════════════════════════════════════

@app.get("/killzone/profile")
def killzone_profile():
    """Per-killzone performance stats and quality gates."""
    from ml.killzone_profiler import KillzoneProfiler
    p = KillzoneProfiler()
    data = p.load()
    if data:
        return data
    return {"status": "no_data", "message": "Run POST /killzone/recompute first"}


@app.post("/killzone/recompute")
def killzone_recompute():
    """Recompute killzone stats, quality gates, and scan config."""
    from ml.killzone_profiler import KillzoneProfiler
    db = _get_scanner().db
    import sqlite3
    with db._conn() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT outcome, killzone, setup_quality, timeframe, analysis_json "
            "FROM scanner_setups "
            "WHERE outcome IS NOT NULL AND killzone IS NOT NULL"
        ).fetchall()
    trades = [dict(r) for r in rows]
    p = KillzoneProfiler()
    stats = p.compute_stats(trades)
    gates = p.compute_quality_gates(trades)
    scan_config = p.get_scan_config(trades)
    return {
        "trades_analyzed": len(trades),
        "stats": stats,
        "quality_gates": gates,
        "scan_config": scan_config,
    }


@app.get("/narrative/trust-by-segment")
def narrative_trust_by_segment():
    """Per-killzone×phase narrative accuracy data (P4)."""
    from ml.claude_bridge import ClaudeAnalysisBridge
    bridge = ClaudeAnalysisBridge()
    return bridge.get_narrative_trust_by_segment()


@app.get("/opus/rejection-policy")
def opus_rejection_policy():
    """Current Opus rejection override policy (P4)."""
    from ml.claude_bridge import ClaudeAnalysisBridge
    bridge = ClaudeAnalysisBridge()
    return bridge.get_opus_rejection_policy()


@app.get("/opus/rejection-policy/segmented")
def opus_rejection_policy_segmented(killzone: str = None, timeframe: str = None):
    """Segmented Opus rejection policy with per-session false-negative tracking.

    Query params:
      killzone: e.g. "london", "ny_am", "ny_pm", "asian"
      timeframe: e.g. "1h", "4h", "15min"

    Returns the per-segment policy and a human-readable prompt context block
    showing Opus its rejection accuracy per session.
    """
    from ml.claude_bridge import ClaudeAnalysisBridge
    bridge = ClaudeAnalysisBridge()
    policy = bridge.get_opus_rejection_policy(killzone=killzone, timeframe=timeframe)
    context = bridge.build_opus_rejection_context()
    return {"policy": policy, "prompt_context": context}


@app.get("/killzone/gates")
def killzone_gates():
    """Current per-killzone quality gates."""
    from ml.killzone_profiler import KillzoneProfiler
    p = KillzoneProfiler()
    p.load()
    return {"gates": p._gates}


@app.get("/td/usage")
def td_usage():
    """Twelve Data API call count for today."""
    engine = _get_scanner()
    return engine._td_call_count


@app.get("/candles")
def get_candles(symbol: str = "XAU/USD", interval: str = "1h", count: int = 100):
    """Fetch candles via OANDA. Used by the frontend terminal."""
    from ml.data_providers import OandaProvider
    cfg = get_config()
    provider = OandaProvider(
        account_id=cfg.get("oanda_account_id", ""),
        access_token=cfg.get("oanda_access_token", ""),
    )
    from datetime import datetime, timedelta
    interval_hours = {"5min": 0.083, "15min": 0.25, "30min": 0.5,
                      "1h": 1, "4h": 4, "1day": 24}
    hours_back = count * interval_hours.get(interval, 1) * 1.5
    start = (datetime.utcnow() - timedelta(hours=hours_back)).strftime("%Y-%m-%dT%H:%M:%S")
    end = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")

    candles = provider.fetch_candles(symbol, interval, start, end)

    # Trim to requested count
    if candles and len(candles) > count:
        candles = candles[-count:]

    return {"values": candles, "count": len(candles) if candles else 0}


@app.get("/data/verify")
def data_verify():
    """Compare last 5 1H XAU/USD candles from Twelve Data and OANDA."""
    from ml.data_providers import TwelveDataProvider, OandaProvider
    from datetime import datetime, timedelta
    cfg = get_config()
    end = datetime.utcnow()
    start = end - timedelta(hours=6)
    result = {"twelvedata": {}, "oanda": {}, "aligned": None}
    try:
        td = TwelveDataProvider(cfg.get("td_api_key", ""))
        td_candles = td.fetch_candles("XAU/USD", "1h", start, end, count=5)
        if td_candles:
            last = td_candles[-1]
            result["twelvedata"] = {"last_candle": last["datetime"],
                                     "close": last["close"], "status": "ok",
                                     "count": len(td_candles)}
        else:
            result["twelvedata"] = {"status": "no_data"}
    except Exception as e:
        result["twelvedata"] = {"status": "error", "error": str(e)[:100]}
    try:
        oa = OandaProvider(
            account_id=cfg.get("oanda_account_id", ""),
            access_token=cfg.get("oanda_access_token", ""),
        )
        oa_candles = oa.fetch_candles("XAU/USD", "1h", start, end, count=5)
        if oa_candles:
            last = oa_candles[-1]
            result["oanda"] = {"last_candle": last["datetime"],
                                "close": last["close"], "status": "ok",
                                "count": len(oa_candles)}
        else:
            result["oanda"] = {"status": "no_data"}
    except Exception as e:
        result["oanda"] = {"status": "error", "error": str(e)[:100]}
    # Compare
    td_close = result["twelvedata"].get("close")
    oa_close = result["oanda"].get("close")
    if td_close is not None and oa_close is not None:
        delta = abs(td_close - oa_close)
        result["price_delta"] = round(delta, 2)
        result["aligned"] = delta <= 0.50
    return result


@app.post("/backtest/generate")
async def backtest_generate(request: Request):
    """Start backtest generation in background."""
    from fastapi import BackgroundTasks
    body = await request.json() if request.headers.get("content-type") == "application/json" else {}

    def _run_backtest():
        from ml.backtest_generator import BacktestGenerator
        gen = BacktestGenerator()
        gen.generate(
            months_back=body.get("months_back", 6),
            max_setups=body.get("max_setups", 300),
            budget_limit_usd=body.get("budget_limit_usd", 30.0),
            dry_run=body.get("dry_run", False),
        )

    import threading
    t = threading.Thread(target=_run_backtest, daemon=True)
    t.start()
    return {"status": "started", "message": "Backtest generation running in background"}


@app.get("/backtest/status")
def backtest_status():
    """Return backtest generation progress from checkpoint."""
    checkpoint_path = os.path.join(cfg["model_dir"], "backtest_checkpoint.json")
    if not os.path.exists(checkpoint_path):
        return {"status": "not_started"}
    try:
        with open(checkpoint_path) as f:
            cp = json.load(f)
        return {
            "status": cp.get("status", "running"),
            "setups_generated": cp.get("setups_found", 0),
            "runners_found": cp.get("runners_found", 0),
            "regime_counts": cp.get("regime_counts", {}),
            "cost_usd": cp.get("cost_usd", 0),
            "fidelity_score": cp.get("fidelity_score"),
        }
    except Exception as e:
        return {"status": "error", "error": str(e)[:200]}


@app.get("/backtest/meta")
def backtest_meta():
    """Return backtest fidelity metadata."""
    meta_path = os.path.join(cfg["model_dir"], "backtest_meta.json")
    if not os.path.exists(meta_path):
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=404, content={"error": "No backtest meta found"})
    with open(meta_path) as f:
        return json.load(f)


@app.get("/evaluation/dual")
def evaluation_dual():
    """Return dual OOS evaluation (all sources vs live-only)."""
    cfg = get_config()
    meta_path = os.path.join(cfg["model_dir"], "model_meta.json")
    if not os.path.exists(meta_path):
        return {"multi3_oos_all": None, "multi3_oos_live": None,
                "live_test_rows": 0, "active_gate_source": "all",
                "gate_threshold": 0.45}
    with open(meta_path) as f:
        meta = json.load(f)
    return {
        "multi3_oos_all": meta.get("multi3_oos_accuracy"),
        "multi3_oos_live": meta.get("multi3_oos_live_accuracy"),
        "live_test_rows": meta.get("multi3_oos_live_sample_size", 0),
        "active_gate_source": meta.get("active_evaluation_source", "all"),
        "gate_threshold": 0.45,
    }


@app.post("/backfill/features")
async def backfill_features():
    """Re-extract features from stored analysis JSON for all resolved trades."""
    import asyncio
    from ml.backfill import backfill_features_from_stored_json
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, backfill_features_from_stored_json)
    return result


# ═══════════════════════════════════════════════════════════════════════
# Narrative Optimization Endpoints
# ═══════════════════════════════════════════════════════════════════════

@app.get("/narrative/weights")
def narrative_weights():
    """Return current narrative EMA weights + AG override if active."""
    cfg = get_config()
    from ml.claude_bridge import ClaudeAnalysisBridge
    from ml.prompts import get_current_killzone
    bridge = ClaudeAnalysisBridge()
    current_kz = get_current_killzone()
    ema_weights = bridge.get_narrative_weights(killzone=current_kz)

    ag_weights = None
    ag_path = os.path.join(cfg["model_dir"], "narrative_weights_ag.json")
    if os.path.exists(ag_path):
        try:
            with open(ag_path) as f:
                ag_weights = json.load(f)
        except Exception:
            pass

    # Flatten all killzone weights for the response
    all_kz = {}
    for key, bucket in bridge._narrative_weights.items():
        if isinstance(bucket, dict):
            all_kz[key] = {k: v.get("weight", 0.5) if isinstance(v, dict) else v
                           for k, v in bucket.items()}

    return {
        "ema_weights": ema_weights,
        "ema_all_killzones": all_kz,
        "ag_weights": ag_weights,
        "current_killzone": current_kz,
        "active_source": "autogluon" if ag_weights and ag_weights.get("_global") else "ema",
    }


@app.post("/narrative/weights/extract")
def narrative_weights_extract():
    """Trigger AG narrative weight extraction from the trained classifier (background)."""
    import threading

    def _run():
        try:
            _cfg = get_config()
            from ml.training import extract_ag_weights
            from ml.database import TradeLogger
            db = TradeLogger(config=_cfg)
            result = extract_ag_weights(model_dir=_cfg["model_dir"], db=db)
            logger.info("AG weight extraction complete: %s", result)
        except Exception as e:
            logger.error("AG weight extraction failed: %s", e)

    threading.Thread(target=_run, daemon=True).start()
    return {"status": "started", "message": "AG weight extraction running in background"}



@app.post("/narrative/weights/backfill-killzones")
def narrative_weights_backfill():
    """Replay resolved trades to warm up per-killzone EMA weight buckets (background)."""
    import threading

    def _run():
        try:
            from ml.claude_bridge import ClaudeAnalysisBridge
            from ml.scanner_db import ScannerDB
            bridge = ClaudeAnalysisBridge()
            db = ScannerDB()
            result = bridge.backfill_killzone_weights(db)
            logger.info("Killzone weight backfill complete: %s", result)
        except Exception as e:
            logger.error("Killzone weight backfill failed: %s", e)

    threading.Thread(target=_run, daemon=True).start()
    return {"status": "started", "message": "Killzone weight backfill running in background"}


@app.get("/narrative/examples")
def narrative_examples():
    """Return gold narrative examples."""
    cfg = get_config()
    from ml.narrative_examples import NarrativeExampleStore
    store = NarrativeExampleStore()
    return {
        "examples": store._examples,
        "count": len(store._examples),
        "max": cfg.get("gold_example_max_store", 8),
    }


@app.get("/narrative/bandit")
def narrative_bandit_state():
    """Return bandit arm performance."""
    from ml.narrative_bandit import NarrativeBandit
    bandit = NarrativeBandit()
    return bandit.get_state()


@app.get("/narrative/evolution")
def narrative_evolution():
    """Combined view of narrative optimization state."""
    from ml.claude_bridge import ClaudeAnalysisBridge
    from ml.narrative_examples import NarrativeExampleStore
    from ml.narrative_bandit import NarrativeBandit

    bridge = ClaudeAnalysisBridge()
    store = NarrativeExampleStore()
    bandit = NarrativeBandit()

    tracker = bridge._accuracy.get("narrative_tracker", {})

    return {
        "weights": bridge.get_narrative_weights(),
        "examples_count": len(store._examples),
        "bandit": bandit.get_state(),
        "narrative_tracker": tracker,
        "total_with_narrative": tracker.get("total_with_narrative", 0),
    }


@app.get("/narrative/thesis/current")
def narrative_thesis_current(timeframe: str = "1h"):
    """Return the current active narrative thesis for a timeframe."""
    from ml.narrative_state import NarrativeStore
    store = NarrativeStore()
    current = store.get_current(timeframe)
    return {"timeframe": timeframe, "thesis": current}


@app.get("/narrative/thesis/history")
def narrative_thesis_history(timeframe: str = "1h", limit: int = 10):
    """Return recent narrative thesis history for a timeframe."""
    from ml.narrative_state import NarrativeStore
    store = NarrativeStore()
    history = store.get_history(timeframe, limit=limit)
    return {"timeframe": timeframe, "count": len(history), "history": history}


@app.get("/narrative/thesis/accuracy")
def narrative_thesis_accuracy(timeframe: str = None):
    """Return prediction accuracy + thesis stability metrics."""
    from ml.narrative_state import NarrativeStore
    store = NarrativeStore()
    metrics = store.get_accuracy_metrics(timeframe)
    metrics["timeframe"] = timeframe or "all"
    return metrics


@app.get("/scanner/status")
def scanner_status():
    """Return scanner health and state."""
    try:
        from ml.scheduler import is_running, get_next_scan_time
        sched_running = is_running()
        next_scan = get_next_scan_time()
    except ImportError:
        sched_running = False
        next_scan = None
    engine = _get_scanner()
    status = engine.get_status()
    status["scheduler_running"] = sched_running
    status["next_scan"] = next_scan
    return status


@app.post("/scanner/trigger")
def scanner_trigger(request: dict = None):
    """Manually trigger a scan cycle.

    Body (optional): {"timeframe": "5min"} for single TF, or omit for all TFs.
    """
    engine = _get_scanner()
    body = request or {}
    tf = body.get("timeframe")
    if tf:
        return engine.scan_once(timeframe=tf)
    return engine.scan_all_timeframes()


@app.post("/scanner/monitor")
def scanner_monitor_trigger():
    """Manually trigger the price monitor to check pending setups."""
    engine = _get_scanner()
    return engine.monitor_pending()


@app.post("/scanner/cd-monitor")
def scanner_cd_monitor_trigger():
    """Manually trigger C/D grade monitoring for promotion checks."""
    engine = _get_scanner()
    return engine.monitor_cd_setups()


@app.post("/scanner/unified-monitor")
def scanner_unified_monitor():
    """Manually trigger the unified monitor loop (pending + CD + prospects)."""
    engine = _get_scanner()
    return engine.unified_monitor()


# ── Priority 5: Haiku False Negative Tracking ──

@app.get("/haiku/fn/report")
def haiku_fn_report():
    """Full Haiku false negative analysis — by timeframe, killzone, reason."""
    engine = _get_scanner()
    return engine._fn_tracker.get_fn_report()


@app.get("/haiku/fn/stats")
def haiku_fn_stats():
    """Quick FN stats for dashboard."""
    engine = _get_scanner()
    return engine._fn_tracker.get_stats()


@app.get("/haiku/fn/adjustments")
def haiku_fn_adjustments():
    """Current screening adjustments based on FN data."""
    engine = _get_scanner()
    return engine._fn_tracker.get_screening_adjustments()


@app.get("/haiku/fn/recent")
def haiku_fn_recent():
    """Recent resolved Haiku rejections."""
    engine = _get_scanner()
    return engine._fn_tracker.get_recent(limit=30)


# ── Priority 8: Cost-Per-Winner Endpoints ──


@app.get("/cost/per-winner")
def cost_per_winner_stats():
    """Full cost-per-winner stats by segment."""
    engine = _get_scanner()
    return engine._cpw_tracker.get_stats()


@app.get("/cost/per-winner/ranking")
def cost_per_winner_ranking():
    """Segments ranked by cost-effectiveness (best ROI first)."""
    engine = _get_scanner()
    return engine._cpw_tracker.get_segment_ranking()


@app.get("/cost/per-winner/recommendations")
def cost_per_winner_recommendations():
    """Scan frequency recommendations per segment."""
    engine = _get_scanner()
    return engine._cpw_tracker.get_recommendations()


@app.post("/cost/per-winner/recompute")
def cost_per_winner_recompute():
    """Recompute cost-per-winner from all resolved setups."""
    engine = _get_scanner()
    return engine._cpw_tracker.recompute_from_db(engine.db)


# ── Notification Lifecycle + Recent Context ──


@app.get("/lifecycle/recent")
def lifecycle_recent(timeframe: str = None, limit: int = 50):
    """Recent lifecycle notifications across all theses."""
    engine = _get_scanner()
    events = engine.db.get_recent_lifecycle(limit=limit, timeframe=timeframe)
    return {"events": events, "count": len(events)}


@app.get("/lifecycle/thesis/{thesis_id}")
def lifecycle_for_thesis(thesis_id: str):
    """Full lifecycle journey for a specific thesis."""
    engine = _get_scanner()
    history = engine.db.get_lifecycle_history(thesis_id)
    return {"thesis_id": thesis_id, "stages": history, "count": len(history)}


@app.get("/context/recent")
def context_recent(timeframe: str = "1h"):
    """Recent context — resolutions, consumed zones, swept liquidity, active setups."""
    from ml.recent_context import build_recent_context
    engine = _get_scanner()
    ctx = build_recent_context(timeframe, engine.db)
    ctx["timeframe"] = timeframe
    return ctx


# ── System Evolution Snapshot endpoints ──────────────────────────

@app.post("/snapshot/take")
def snapshot_take():
    """Manually trigger a system snapshot."""
    from ml.system_snapshot import SystemSnapshotRecorder
    recorder = SystemSnapshotRecorder()
    snap = recorder.take_snapshot(trigger="manual")
    return {"status": "ok", "snapshot": snap}


@app.get("/snapshot/trends")
def snapshot_trends(days: int = 14):
    """Compute trend directions for all tracked metrics."""
    from ml.system_snapshot import SystemSnapshotRecorder
    recorder = SystemSnapshotRecorder()
    return recorder.compute_trends(days=days)


@app.get("/snapshot/history")
def snapshot_history(days: int = 30, limit: int = 100):
    """Return recent snapshots."""
    from ml.system_snapshot import SystemSnapshotRecorder
    recorder = SystemSnapshotRecorder()
    snaps = recorder.get_snapshots(days=days, limit=limit)
    return {"count": len(snaps), "snapshots": snaps}


@app.get("/snapshot/weekly-report")
def snapshot_weekly_report():
    """Generate a weekly system evolution report."""
    from ml.system_snapshot import SystemSnapshotRecorder
    recorder = SystemSnapshotRecorder()
    return recorder.generate_weekly_report()


@app.post("/snapshot/backfill")
def snapshot_backfill():
    """Backfill historical snapshots from resolved scanner trades."""
    from ml.system_snapshot import SystemSnapshotRecorder
    recorder = SystemSnapshotRecorder()
    count = recorder.backfill_from_trades()
    return {
        "status": "ok",
        "snapshots_backfilled": count,
        "total_snapshots": recorder.get_snapshot_count(),
    }


@app.get("/snapshot/prompt-context")
def snapshot_prompt_context(days: int = 14):
    """Preview the system learning context that gets injected into prompts."""
    from ml.system_snapshot import SystemSnapshotRecorder
    recorder = SystemSnapshotRecorder()
    ctx = recorder.build_prompt_context(days=days)
    return {"context": ctx, "note": "This block is injected into both Opus and Sonnet prompts."}


# ─── Admin: one-time data migration endpoints ───────────────────────────────

@app.post("/restore/db")
async def admin_restore_db(request: Request):
    """Upload local scanner.db to the Railway Volume. Protected by ADMIN_SECRET."""
    secret = os.environ.get("ADMIN_SECRET", "")
    if not secret or request.headers.get("X-Admin-Secret") != secret:
        raise HTTPException(status_code=403, detail="Forbidden")
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="Empty body")
    import tempfile, shutil
    db_path = get_config()["db_path"]
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as f:
        f.write(body)
        tmp = f.name
    shutil.move(tmp, db_path)
    return {"status": "ok", "db_path": db_path, "size_bytes": len(body)}


@app.post("/restore/file/{filename}")
async def admin_restore_file(filename: str, request: Request):
    """Upload a JSON/CSV file to the models directory. Protected by ADMIN_SECRET."""
    secret = os.environ.get("ADMIN_SECRET", "")
    if not secret or request.headers.get("X-Admin-Secret") != secret:
        raise HTTPException(status_code=403, detail="Forbidden")
    import re
    if not re.match(r'^[\w\-]+\.(json|csv|parquet)$', filename):
        raise HTTPException(status_code=400, detail="Invalid filename")
    body = await request.body()
    model_dir = get_config()["model_dir"]
    os.makedirs(model_dir, exist_ok=True)
    dest = os.path.join(model_dir, filename)
    with open(dest, "wb") as f:
        f.write(body)
    return {"status": "ok", "path": dest, "size_bytes": len(body)}
