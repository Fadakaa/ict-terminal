"""Bayesian belief updater — running win rate + drawdown beliefs.

Pure functions: update state, never mutate inputs. Persisted via SQLite.
Uses Beta distribution for win rate estimation with credible intervals.
"""
from copy import deepcopy

from scipy.stats import beta as beta_dist

from ml.config import get_config

_WIN_OUTCOMES = {"tp1_hit", "tp2_hit", "tp3_hit"}


def get_default_prior(config: dict = None) -> dict:
    """Return the default uninformative prior state."""
    cfg = config or get_config()
    return {
        "alpha": cfg.get("bayesian_prior_alpha", 1),
        "beta_param": cfg.get("bayesian_prior_beta", 1),
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


def update_beliefs(prior_state: dict, outcome: str, pnl: float,
                   config: dict = None) -> dict:
    """Pure function: apply a single trade outcome to prior beliefs.

    Args:
        prior_state: current Bayesian state dict
        outcome: "stopped_out" | "tp1_hit" | "tp2_hit" | "tp3_hit"
        pnl: profit/loss in pips for drawdown tracking

    Returns: new posterior state dict (does NOT mutate prior_state)
    """
    s = deepcopy(prior_state)
    s["total_trades"] += 1
    s["cumulative_pnl"] += pnl

    is_win = outcome in _WIN_OUTCOMES

    if is_win:
        s["alpha"] += 1
        s["total_wins"] += 1
        s["consecutive_losses"] = 0
    else:
        s["beta_param"] += 1
        s["total_losses"] += 1
        s["consecutive_losses"] += 1
        s["max_consecutive_losses"] = max(
            s["max_consecutive_losses"], s["consecutive_losses"]
        )

    # Drawdown tracking
    s["peak_pnl"] = max(s["peak_pnl"], s["cumulative_pnl"])
    s["current_drawdown"] = max(0.0, s["peak_pnl"] - s["cumulative_pnl"])
    s["max_drawdown"] = max(s["max_drawdown"], s["current_drawdown"])

    return s


def get_beliefs(state: dict, config: dict = None) -> dict:
    """Derive human-readable belief summary from raw state.

    Returns dict with win rate mean, 95% credible interval, and drawdown info.
    """
    a = state["alpha"]
    b = state["beta_param"]

    mean = a / (a + b)
    lower = float(beta_dist.ppf(0.025, a, b))
    upper = float(beta_dist.ppf(0.975, a, b))

    return {
        "win_rate_mean": round(mean, 4),
        "win_rate_lower_95": round(lower, 4),
        "win_rate_upper_95": round(upper, 4),
        "consecutive_losses": state["consecutive_losses"],
        "max_consecutive_losses": state["max_consecutive_losses"],
        "max_drawdown": state["max_drawdown"],
        "current_drawdown": state["current_drawdown"],
        "total_trades": state["total_trades"],
    }


def adjust_confidence(ag_confidence: float, beliefs: dict | None,
                      config: dict = None) -> float:
    """Blend AutoGluon confidence with Bayesian win rate belief.

    Returns: weighted average, capped to [0, 1].
    Default: 0.7 * ag_confidence + 0.3 * win_rate_mean
    """
    if beliefs is None:
        return ag_confidence

    cfg = config or get_config()
    bw = cfg.get("bayesian_confidence_weight", 0.3)
    aw = 1.0 - bw

    blended = aw * ag_confidence + bw * beliefs["win_rate_mean"]
    return max(0.0, min(1.0, round(blended, 4)))
