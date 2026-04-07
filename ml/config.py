"""Configuration for the ICT ML prediction server."""
import os
from copy import deepcopy
from pathlib import Path

from dotenv import load_dotenv

# Load .env before reading any env vars
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

# Resolve paths relative to the project root (parent of ml/)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ML_DIR = os.path.join(_PROJECT_ROOT, "ml")

# Railway Volume: override storage paths when DATA_DIR is set
_DATA_DIR = os.environ.get("DATA_DIR")
if _DATA_DIR:
    _MODELS_DIR = os.path.join(_DATA_DIR, "models")
    try:
        os.makedirs(_MODELS_DIR, exist_ok=True)
    except OSError:
        pass  # Volume not yet mounted; Dockerfile pre-creates /data/models
else:
    _MODELS_DIR = os.path.join(_ML_DIR, "models")

_DEFAULT_CONFIG = {
    "db_path": os.path.join(_MODELS_DIR, "scanner.db"),
    "model_dir": _MODELS_DIR,
    "min_training_samples": 30,
    "min_training_samples_quantile": 50,
    "retrain_on_n_new_trades": 10,
    "classification_labels": ["stopped_out", "tp1_hit", "tp2_hit", "tp3_hit", "no_trade", "expired"],
    "quantile_percentiles": [0.05, 0.25, 0.50, 0.75, 0.90],
    "autogluon_time_limit": 300,
    "autogluon_presets": "best_quality",
    "feature_version": 2,
    # Bayesian updater
    "bayesian_prior_alpha": 1,
    "bayesian_prior_beta": 1,
    "bayesian_confidence_weight": 0.3,
    # Volatility calibrator
    "ewma_lambda": 0.94,
    "session_factors": {
        "london": 1.1,
        "new_york": 1.15,
        "overlap_london_ny": 1.3,
        "asian": 0.7,
        "off_hours": 0.5,
    },
    "regime_thresholds": [25, 75],
    "regime_multipliers": {"low": 0.8, "normal": 1.0, "high": 1.2},
    "regime_lookback": 100,
    # SL floor — gold's noise floor from 337 resolved trades
    "sl_floor_atr": 3.0,
    # Consensus engine
    "grade_thresholds": {"A": 0.75, "B": 0.60, "C": 0.45},
    "ag_confidence_weight": 0.7,
    # Walk-Forward Optimization
    "wfo_train_window": 500,
    "wfo_test_window": 100,
    "wfo_step_size": 50,
    "wfo_sl_atr_mult": 1.5,
    "wfo_tp_atr_mults": [1.0, 2.0, 3.5],
    "wfo_max_bars_in_trade": 50,
    "wfo_max_folds": 80,
    "wfo_displacement_threshold": 1.5,
    "wfo_min_confluence_score": 2,
    "wfo_min_setups_per_fold": 20,
    "wfo_report_path": os.path.join(_MODELS_DIR, "wfo_report.json"),
    # Dataset manager
    "dataset_parquet_path": os.path.join(_MODELS_DIR, "training_dataset.parquet"),
    "live_weight_multiplier": 5.0,
    "wfo_weight_decay_rate": 200,
    # Execution sim
    "default_slippage_pips": 0.5,
    # Bayesian safeguards
    "kappa_cap": 30,
    "drift_significant_threshold": 2.0,
    "drift_critical_threshold": 3.0,
    "drift_check_interval": 20,
    # OOS evaluation + active learning
    "oos_min_accuracy": 0.58,
    "oos_min_test_trades": 30,
    "eval_holdout_fraction": 0.2,
    "retrain_every_n_trades": 50,
    "weakness_boost_high": 3.0,
    "weakness_boost_medium": 1.5,
    "weakness_min_trades": 10,
    # Narrative prompt optimization
    "narrative_ema_alpha": 0.15,
    "narrative_ema_initial": 0.5,
    "gold_example_max_store": 8,
    "gold_example_min_score": 0.3,
    "gold_example_retrieve_n": 3,
    "gold_example_recency_decay": 0.95,
    "narrative_bandit_min_trades": 100,
    "narrative_bandit_max_arms": 3,
    "narrative_bandit_retire_wr": 0.35,
    "narrative_bandit_retire_min_trials": 30,
    "narrative_bandit_new_variant_every": 100,
    # Anticipatory alert system
    "prospect_before_killzone_minutes": 15,
    "trigger_poll_interval_seconds": 90,
    "trigger_candle_timeframe": "5min",
    "prospect_max_setups": 4,
    "trigger_price_tolerance_pips": 3.0,
    # Retracement entry timing
    "retrace_timeout_candles": 15,
    "retrace_entry_type": "ob_midpoint",  # ob_midpoint | zone_touch | ltf_confirm
    "ltf_refinement_enabled": True,
    "ltf_refinement_timeframe": "5min",
    # Notification preferences
    "notify_zone_alerts": True,
    "notify_displacement_alerts": True,
    "notify_entry_alerts": True,
    "notify_detection_alerts": True,
    "prospect_max_regen": {"Asian": 12, "London": 30, "NY_AM": 30, "NY_PM": 30},
    # Phase B feature flag — route prospect retraces through scan_once() pipeline
    # Set True after Phase A has been validated for 1+ trading day
    "prospect_use_scan_once": False,
    "notify_macos": True,
    "notify_telegram": True,
    # Lifecycle notification gates
    "notify_thesis_forming": True,
    "notify_thesis_confirmed": True,
    "notify_thesis_revised": True,
    "macos_sound_zone": "Tink",
    "macos_sound_displacement": "Ping",
    "macos_sound_entry": "Glass",
    "macos_sound_resolved": "Glass",
    # Regime-aware quality gates (5-state structural regime)
    "regime_quality_gates": {
        "TRENDING_IMPULSIVE": {"min_grade": "C", "min_confluences": 2},
        "TRENDING_CORRECTIVE": {"min_grade": "C", "min_confluences": 2},
        "RANGING": {"min_grade": "B", "min_confluences": 2},
        "VOLATILE_CHOPPY": {"min_grade": "B", "min_confluences": 3},
        "QUIET_DRIFT": {"min_grade": "A", "min_confluences": 2},
    },
    # Cost optimization
    "narrative_cache_ttl_seconds": 3600,
    "narrative_cache_on_killzone_change": True,
    "screen_cache_ttl_seconds": 1800,
    "intermarket_cache_ttl_seconds": 900,
    "daily_api_budget_usd": 9999.00,
    "budget_warning_threshold": 0.80,
    "sonnet_compressed_prompt": True,
    # Multi-3 post-upgrade enhancements
    "max_position_multiplier": 2.0,
    "min_ev_ratio": 0.15,
    "spread_cost_rr": 0.05,
    "kelly_fraction": 0.5,
    "management_tracker_min_trades": 30,
    # Data providers
    "oanda_account_id": os.getenv("OANDA_ACCOUNT_ID", ""),
    "oanda_access_token": os.getenv("OANDA_ACCESS_TOKEN", ""),
    "td_api_key": os.getenv("TWELVE_DATA_API_KEY", ""),
    "backtest_data_source": "oanda",
}

# Disable macOS-only notifications on Linux (osascript doesn't exist)
import sys
if sys.platform != "darwin":
    _DEFAULT_CONFIG["notify_macos"] = False

_active_config = deepcopy(_DEFAULT_CONFIG)


def get_config() -> dict:
    """Return the active configuration singleton."""
    return _active_config


def make_test_config(**overrides) -> dict:
    """Create an isolated config dict for tests — no global mutation."""
    cfg = deepcopy(_DEFAULT_CONFIG)
    cfg.update(overrides)
    return cfg


def reset_config():
    """Reset active config to defaults."""
    global _active_config
    _active_config = deepcopy(_DEFAULT_CONFIG)
