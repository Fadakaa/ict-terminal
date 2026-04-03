"""Training dataset manager — persistent parquet + regime balance + Bayesian safeguards.

Central hub for blending WFO-simulated and live trades into a weighted dataset
for AutoGluon training. Includes negative example generation, regime balancing,
prior validation, and drift detection.
"""
import math
import os
import logging
import random

import numpy as np
import pandas as pd

from ml.config import get_config

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Training Dataset Manager
# ═══════════════════════════════════════════════════════════════════════


class TrainingDatasetManager:
    """Persistent CSV-based dataset combining WFO + live trades."""

    def __init__(self, config: dict = None):
        self.cfg = config or get_config()
        configured_path = self.cfg.get(
            "dataset_parquet_path", "ml/models/training_dataset.parquet"
        )
        # Use CSV for portability (no pyarrow dependency)
        self.data_path = configured_path.replace(".parquet", ".csv")
        self._df = self._load()

    def _load(self) -> pd.DataFrame:
        """Load existing CSV or return empty DataFrame."""
        if os.path.exists(self.data_path):
            try:
                return pd.read_csv(self.data_path)
            except Exception:
                pass
        return pd.DataFrame()

    def _save(self):
        """Persist current dataset to CSV."""
        os.makedirs(os.path.dirname(self.data_path) or ".", exist_ok=True)
        self._df.to_csv(self.data_path, index=False)

    def ingest_wfo_trades(self, trades: list[dict]) -> int:
        """Clear old WFO rows and ingest new ones.

        Each trade dict should have 38 features + outcome + metadata
        from ICTSetupDetector.detect_setups().
        """
        # Remove old WFO rows
        if not self._df.empty and "source" in self._df.columns:
            self._df = self._df[self._df["source"] != "wfo"]

        if not trades:
            self._save()
            return 0

        new_rows = []
        for i, trade in enumerate(trades):
            row = dict(trade)
            row["source"] = "wfo"
            row["setup_id"] = f"wfo-{i:04d}"
            new_rows.append(row)

        new_df = pd.DataFrame(new_rows)
        self._df = pd.concat([self._df, new_df], ignore_index=True)
        self._save()
        return len(new_rows)

    def ingest_live_trade(self, features: dict, outcome: str,
                          mfe: float, mae: float, pnl: float,
                          setup_id: str | None = None) -> None:
        """Append a single live trade to the dataset."""
        row = dict(features)
        row["source"] = "live"
        row["outcome"] = outcome
        row["max_favorable_atr"] = mfe
        row["max_drawdown_atr"] = mae
        row["pnl"] = pnl
        row["setup_id"] = setup_id or f"live-{len(self._df):04d}"

        # Feature quality label: count non-NaN feature columns
        from ml.feature_schema import FEATURE_COLUMNS, RICH_FEATURE_THRESHOLD
        filled = sum(1 for col in FEATURE_COLUMNS
                     if col in row and row[col] is not None
                     and not (isinstance(row[col], float) and math.isnan(row[col])))
        row["feature_quality"] = "full" if filled >= RICH_FEATURE_THRESHOLD else "partial"

        new_df = pd.DataFrame([row])
        self._df = pd.concat([self._df, new_df], ignore_index=True)
        self._save()

    def get_blended_dataset(self) -> pd.DataFrame:
        """Return all rows with computed sample_weight column.

        WFO weight decays as live trades grow:
            wfo_weight = 1.0 * max(0.2, 1.0 - live_count / decay_rate)
            live_weight = cfg["live_weight_multiplier"] (default 5.0)
        """
        if self._df.empty:
            return pd.DataFrame()

        df = self._df.copy()
        live_count = len(df[df["source"] == "live"]) if "source" in df.columns else 0
        decay_rate = self.cfg.get("wfo_weight_decay_rate", 200)
        live_weight = self.cfg.get("live_weight_multiplier", 5.0)

        wfo_base_weight = max(0.2, 1.0 - live_count / decay_rate)

        weights = []
        for _, row in df.iterrows():
            source = row.get("source", "")
            quality = row.get("feature_quality", "full")
            if source == "live":
                w = live_weight
            else:
                w = wfo_base_weight
            # Partial-feature rows get half weight
            if quality == "partial":
                w *= 0.5
            weights.append(w)
        df["sample_weight"] = weights

        return df

    def clear_live_trades(self):
        """Remove all live-sourced rows, keeping WFO rows. For backfill."""
        if self._df.empty:
            return
        if "source" in self._df.columns:
            self._df = self._df[self._df["source"] == "wfo"].copy()
        else:
            self._df = pd.DataFrame()
        self._save()

    def get_stats(self) -> dict:
        """Return dataset statistics."""
        if self._df.empty:
            return {
                "total": 0, "wfo_count": 0, "live_count": 0,
                "regime_distribution": {}, "outcome_distribution": {},
            }

        source_counts = self._df["source"].value_counts().to_dict() if "source" in self._df.columns else {}
        regime_counts = self._df["regime"].value_counts().to_dict() if "regime" in self._df.columns else {}
        outcome_counts = self._df["outcome"].value_counts().to_dict() if "outcome" in self._df.columns else {}

        return {
            "total": len(self._df),
            "wfo_count": source_counts.get("wfo", 0),
            "live_count": source_counts.get("live", 0),
            "regime_distribution": regime_counts,
            "outcome_distribution": outcome_counts,
        }


# ═══════════════════════════════════════════════════════════════════════
# Negative Example Generation
# ═══════════════════════════════════════════════════════════════════════


def generate_negative_examples(candles: list[dict], positive_trades: list[dict],
                                target_ratio: float = 0.3,
                                config: dict = None) -> list[dict]:
    """Generate 'no_trade' examples from low-confluence candle positions.

    Samples random candle indices NOT near any positive setup, engineers
    38 features, and labels as outcome='no_trade'.

    Args:
        candles: full candle list
        positive_trades: trades already detected (to avoid overlap)
        target_ratio: fraction of total dataset that should be no_trade
        config: optional config dict

    Returns:
        List of trade dicts with outcome='no_trade'
    """
    from ml.features import (
        compute_atr, detect_order_blocks, detect_fvgs,
        detect_liquidity, compute_market_structure,
        engineer_features_from_candles,
    )
    from ml.wfo import detect_regime

    cfg = config or get_config()

    if len(candles) < 60:
        return []

    # Calculate how many negatives we need
    n_positives = len(positive_trades)
    if n_positives == 0:
        return []
    n_negatives = max(1, int(n_positives * target_ratio / (1 - target_ratio)))

    # Get indices used by positive setups
    positive_indices = {t.get("candle_index", -1) for t in positive_trades}
    # Exclude within 5 bars of any positive setup
    excluded = set()
    for idx in positive_indices:
        for offset in range(-5, 6):
            excluded.add(idx + offset)

    # Candidate indices: 50 to len-10, not near positive setups
    candidates = [i for i in range(50, len(candles) - 10) if i not in excluded]
    if not candidates:
        return []

    atr = compute_atr(candles, 14)
    if atr <= 0:
        return []

    disp_threshold = cfg.get("wfo_displacement_threshold", 1.5)
    obs = detect_order_blocks(candles, atr, disp_threshold)
    fvgs_all = detect_fvgs(candles)
    liqs = detect_liquidity(candles, window=20)

    rng = random.Random(42)
    selected = rng.sample(candidates, min(n_negatives * 3, len(candidates)))

    negatives = []
    for idx in selected:
        if len(negatives) >= n_negatives:
            break

        ms_score = compute_market_structure(candles[:idx + 1], lookback=20)
        direction = rng.choice(["long", "short"])

        features = engineer_features_from_candles(
            candles, idx, direction, atr, obs, fvgs_all, liqs, ms_score
        )

        # Only keep if confluence is low (these are NOT real setups)
        confluence = features.get("ob_alignment", 0) + features.get("fvg_alignment", 0)
        if confluence >= cfg.get("wfo_min_confluence_score", 2):
            continue

        row = {**features}
        row["outcome"] = "no_trade"
        row["max_drawdown_atr"] = 0
        row["max_favorable_atr"] = 0
        row["bars_held"] = 0
        row["won"] = False
        row["setup_type"] = "no_trade"
        row["direction"] = direction
        row["confluence_score"] = confluence
        row["candle_index"] = idx
        row["regime"] = detect_regime(candles, idx)

        negatives.append(row)

    return negatives


# ═══════════════════════════════════════════════════════════════════════
# Regime Balancer
# ═══════════════════════════════════════════════════════════════════════


class RegimeBalancer:
    """Balance training data across market regimes."""

    def __init__(self, config: dict = None):
        self.cfg = config or get_config()
        self._regime_counts = {}

    def balance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Balance dataset by regime via up/downsampling.

        Upsamples minority regimes with small Gaussian noise on features.
        Downsamples majority regimes to median count.
        """
        if df.empty or "regime" not in df.columns:
            return df

        regime_groups = df.groupby("regime")
        counts = regime_groups.size()
        self._regime_counts = counts.to_dict()

        if len(counts) <= 1:
            return df

        median_count = int(counts.median())
        target = max(median_count, 5)

        balanced_parts = []
        feature_cols = [c for c in df.columns if c not in (
            "source", "setup_id", "outcome", "regime", "sample_weight",
            "direction", "setup_type", "candle_index", "won",
            "confluence_score", "fold", "quality",
            "max_drawdown_atr", "max_favorable_atr", "bars_held",
            "execution_cost_atr", "session_spread", "adjusted_entry", "pnl",
        )]

        for regime, group in regime_groups:
            if len(group) >= target:
                # Downsample
                balanced_parts.append(group.sample(n=target, random_state=42))
            else:
                # Keep originals + upsample with noise
                balanced_parts.append(group)
                n_needed = target - len(group)
                if n_needed > 0 and len(group) > 0:
                    upsampled = group.sample(n=n_needed, replace=True, random_state=42).copy()
                    # Add Gaussian noise to numeric feature columns
                    for col in feature_cols:
                        if col in upsampled.columns and upsampled[col].dtype in (
                            np.float64, np.int64, float, int
                        ):
                            col_range = df[col].std() if df[col].std() > 0 else 1.0
                            noise = np.random.RandomState(42).normal(
                                0, 0.01 * col_range, size=len(upsampled)
                            )
                            upsampled[col] = upsampled[col].astype(float) + noise
                    balanced_parts.append(upsampled)

        return pd.concat(balanced_parts, ignore_index=True)

    def get_regime_coverage(self) -> dict:
        """Return regime counts from last balance() call."""
        return dict(self._regime_counts)

    def get_defensive_adjustment(self, regime: str) -> float:
        """Return confidence multiplier for underrepresented regimes.

        Returns 0.7-1.0 based on how well the regime is represented.
        """
        if not self._regime_counts:
            return 1.0

        total = sum(self._regime_counts.values())
        if total == 0:
            return 1.0

        regime_count = self._regime_counts.get(regime, 0)
        ratio = regime_count / total

        # 5 regimes → fair share is 0.2
        # If ratio < 0.1, heavily underrepresented → 0.7
        # If ratio >= 0.15, well represented → 1.0
        if ratio < 0.05:
            return 0.7
        elif ratio < 0.10:
            return 0.8
        elif ratio < 0.15:
            return 0.9
        return 1.0


# ═══════════════════════════════════════════════════════════════════════
# Bayesian Safeguards
# ═══════════════════════════════════════════════════════════════════════


class PriorValidator:
    """Validate Bayesian priors before anchoring from WFO data."""

    def __init__(self, config: dict = None):
        self.cfg = config or get_config()

    def stress_test_priors(self, state: dict) -> dict:
        """Simulate consecutive losses and check if posterior remains sane.

        Returns dict with {passed, breakdown_after_n_losses, recommendation}.
        """
        alpha = state.get("alpha", 1.0)
        beta_param = state.get("beta_param", 1.0)

        # Simulate 50 consecutive losses
        test_alpha = alpha
        test_beta = beta_param
        breakdown_at = None

        for i in range(1, 51):
            test_beta += 1
            # Check posterior mean
            posterior_mean = test_alpha / (test_alpha + test_beta)
            if posterior_mean < 0.05:
                breakdown_at = i
                break

        passed = breakdown_at is None or breakdown_at >= 20

        recommendation = "Priors are robust" if passed else (
            f"Priors break down after {breakdown_at} losses — "
            "consider wider prior (lower alpha/beta)"
        )

        return {
            "passed": passed,
            "breakdown_after_n_losses": breakdown_at,
            "recommendation": recommendation,
            "initial_alpha": alpha,
            "initial_beta": beta_param,
        }

    def validate_before_anchoring(self, wfo_report, db) -> bool:
        """Check if WFO-derived priors would be sane before applying.

        Returns True if safe to anchor, False if priors would be fragile.
        """
        oos_trades = getattr(wfo_report, "total_oos_trades", 0)
        oos_wr = getattr(wfo_report, "oos_win_rate", 0.5)

        if oos_trades < 20:
            return True  # Too few trades to be dangerous

        # Simulate what the prior would look like
        oos_wins = int(oos_wr * oos_trades)
        oos_losses = oos_trades - oos_wins

        simulated_state = {
            "alpha": max(1, 1 + oos_wins),
            "beta_param": max(1, 1 + oos_losses),
        }

        result = self.stress_test_priors(simulated_state)
        if not result["passed"]:
            logger.warning(
                f"WFO priors would be fragile: {result['recommendation']}"
            )
        return result["passed"]


class DriftAlarm:
    """Detect when posterior beliefs drift significantly from priors."""

    def __init__(self, config: dict = None):
        self.cfg = config or get_config()
        self.significant_threshold = self.cfg.get("drift_significant_threshold", 2.0)
        self.critical_threshold = self.cfg.get("drift_critical_threshold", 3.0)
        self.kappa_cap_val = self.cfg.get("kappa_cap", 30)

    def check_drift(self, current_state: dict,
                    reference_state: dict = None) -> dict:
        """Compute drift between current posterior and reference prior.

        Uses absolute difference in win-rate means (percentage points) as the
        primary metric.  The old posterior_std-normalised formula produced
        spuriously high values whenever kappa grew (tight posterior → tiny std
        → inflated SD even for small mean shifts).

        Thresholds (percentage-point drift in win rate):
          - significant: >= 10 pp  (e.g. 44.6% → 34.6%)
          - critical:    >= 20 pp  (e.g. 44.6% → 24.6%)

        Args:
            current_state: current Bayesian state dict
            reference_state: reference/prior state (defaults to V1 priors)

        Returns:
            {drift_pp, level, recommendation, posterior_wr, reference_wr}
        """
        if current_state is None:
            return {"drift_pp": 0.0, "drift_sd": 0.0, "level": "none",
                    "recommendation": "No state yet"}

        if reference_state is None:
            reference_state = {"alpha": 1.0, "beta_param": 1.0}

        # Reference (prior) mean
        ref_a = reference_state.get("alpha", 1.0)
        ref_b = reference_state.get("beta_param", 1.0)
        prior_mu = ref_a / (ref_a + ref_b)

        # Posterior mean
        post_a = current_state.get("alpha", 1.0)
        post_b = current_state.get("beta_param", 1.0)
        posterior_mu = post_a / (post_a + post_b)

        # Drift in percentage points
        drift_pp = abs(posterior_mu - prior_mu) * 100

        # Also compute legacy SD metric for backwards compat
        posterior_var = (post_a * post_b) / (
            (post_a + post_b) ** 2 * (post_a + post_b + 1))
        posterior_std = math.sqrt(posterior_var) if posterior_var > 0 else 1e-6
        drift_sd = abs(posterior_mu - prior_mu) / posterior_std

        # Thresholds in percentage points
        sig_pp = self.cfg.get("drift_significant_pp", 10.0)
        crit_pp = self.cfg.get("drift_critical_pp", 20.0)

        if drift_pp >= crit_pp:
            level = "critical"
            recommendation = (
                f"Win rate has shifted {drift_pp:.1f}pp from reference "
                f"({prior_mu:.1%} → {posterior_mu:.1%}) — consider "
                "investigating or resetting priors"
            )
        elif drift_pp >= sig_pp:
            level = "significant"
            recommendation = (
                f"Win rate drifting {drift_pp:.1f}pp from reference "
                f"({prior_mu:.1%} → {posterior_mu:.1%}) — monitor closely"
            )
        else:
            level = "none"
            recommendation = (
                f"Win rate stable at {posterior_mu:.1%} "
                f"(reference {prior_mu:.1%}, drift {drift_pp:.1f}pp)"
            )

        return {
            "drift_pp": round(drift_pp, 1),
            "drift_sd": round(drift_sd, 3),
            "level": level,
            "recommendation": recommendation,
            "posterior_wr": round(posterior_mu, 4),
            "reference_wr": round(prior_mu, 4),
        }

    def cap_kappa(self, state: dict) -> dict:
        """Cap kappa (alpha + beta) to prevent prior from becoming too rigid.

        If kappa > kappa_cap, scale both alpha and beta proportionally.
        Returns the (possibly modified) state dict.
        """
        if state is None:
            return state

        alpha = state.get("alpha", 1.0)
        beta_param = state.get("beta_param", 1.0)
        kappa = alpha + beta_param

        if kappa > self.kappa_cap_val:
            scale = self.kappa_cap_val / kappa
            state["alpha"] = alpha * scale
            state["beta_param"] = beta_param * scale

        return state
