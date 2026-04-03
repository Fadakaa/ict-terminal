"""V1 Data Harvester — seeds ML models with baseline distributions from WFO simulations.

The mechanical ICT detector cannot pick winners (30.8% WR, Grade D), but its 422
simulated trades contain real drawdown/excursion distributions across sessions and
regimes on XAU/USD. This module harvests that data to seed:
  - Bayesian priors (drawdown/excursion/win rate beliefs)
  - Per-session statistics (for the calibrator)
  - The training dataset (for AutoGluon quantile regressors)

All V1 seed data is tagged with source="v1_seed" and weight=0.5.
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd

from ml.config import get_config
from ml.features import compute_atr, _extract_hour
from ml.wfo import WFOConfig, ICTSetupDetector, detect_regime


class V1DataHarvester:
    """Harvest calibration data from V1 ICT setup detector runs."""

    def __init__(self, config: dict = None):
        self.cfg = config or get_config()

    def harvest_v1_data(self, candles_df: pd.DataFrame,
                        config: dict = None) -> pd.DataFrame:
        """Run original V1 detector on candle data and extract calibration columns.

        Args:
            candles_df: DataFrame with open/high/low/close/datetime columns
            config: optional config override

        Returns:
            DataFrame with features + calibration metadata, tagged source=v1_seed
        """
        cfg = config or self.cfg

        # Convert DataFrame to list of dicts for detector
        candles = candles_df.to_dict("records")

        # V1 baseline config — exactly what the original diagnostic used
        v1_cfg = WFOConfig(
            sl_atr_mult=1.5,
            tp_atr_mults=[1.0, 2.0, 3.5],
            min_confluence_score=2,
            use_mtf=False,
            use_retracement_entry=False,
            use_quality_scoring=False,
            use_narrative_filter=False,
            use_rejection_entry=False,
            filter_counter_trend=False,
            max_bars_in_trade=20,
            displacement_threshold=1.5,
            min_setups_per_fold=5,
        )

        detector = ICTSetupDetector(v1_cfg)
        setups_df = detector.detect_setups(candles, timeframe="1h")

        if setups_df.empty:
            return pd.DataFrame()

        atr = compute_atr(candles, 14)

        # Enrich each setup with session and regime info
        rows = []
        for _, row in setups_df.iterrows():
            entry = dict(row)
            idx = int(row.get("candle_index", 0))

            # Session classification from candle hour
            session = self._classify_session(candles, idx)
            entry["session"] = session

            # Regime at entry
            regime = detect_regime(candles, idx)
            entry["regime"] = regime

            # ATR at entry
            local_atr = compute_atr(candles[max(0, idx - 15):idx + 1], 14)
            entry["atr_14"] = local_atr if local_atr > 0 else atr

            # Vol ratio at entry
            if idx >= 5:
                returns_5 = [abs(candles[j]["close"] - candles[j - 1]["close"])
                             for j in range(max(1, idx - 4), idx + 1)]
                std_5 = np.std(returns_5) if len(returns_5) > 1 else 0
            else:
                std_5 = 0
            if idx >= 30:
                returns_30 = [abs(candles[j]["close"] - candles[j - 1]["close"])
                              for j in range(max(1, idx - 29), idx + 1)]
                std_30 = np.std(returns_30) if len(returns_30) > 1 else 0
            else:
                std_30 = 0
            entry["vol_ratio_5_30"] = std_5 / std_30 if std_30 > 0 else 1.0

            # Tag as V1 seed data
            entry["source"] = "v1_seed"
            entry["sample_weight"] = 0.5

            rows.append(entry)

        result_df = pd.DataFrame(rows)
        return result_df

    def seed_bayesian(self, harvested_df: pd.DataFrame) -> dict:
        """Compute Bayesian priors from harvested V1 data.

        Groups by session, computes drawdown/excursion/win rate stats,
        then sets aggregate priors with moderate strength (kappa=15).

        Returns:
            Dict with priors and session stats
        """
        if harvested_df.empty:
            return {"priors": {}, "session_stats": {}}

        win_outcomes = {"tp1_hit", "tp2_hit", "tp3_hit"}

        # Per-session breakdown
        session_stats = {}
        sessions = harvested_df["session"].unique() if "session" in harvested_df.columns else []

        for session in sessions:
            session_df = harvested_df[harvested_df["session"] == session]
            winners = session_df[session_df["outcome"].isin(win_outcomes)]
            total = len(session_df)
            win_count = len(winners)

            stats = {
                "trades": total,
                "wins": win_count,
                "win_rate": win_count / total if total > 0 else 0.0,
                "median_drawdown": float(winners["max_drawdown_atr"].median()) if len(winners) > 0 else 0.0,
                "p95_drawdown": float(winners["max_drawdown_atr"].quantile(0.95)) if len(winners) > 0 else 0.0,
                "median_favorable": float(winners["max_favorable_atr"].median()) if len(winners) > 0 else 0.0,
                "median_bars_held": float(session_df["bars_held"].median()) if total > 0 else 0,
            }
            session_stats[session] = stats

        # All-sessions aggregate
        all_winners = harvested_df[harvested_df["outcome"].isin(win_outcomes)]
        total_trades = len(harvested_df)
        total_wins = len(all_winners)
        overall_wr = total_wins / total_trades if total_trades > 0 else 0.5

        agg_median_dd = float(all_winners["max_drawdown_atr"].median()) if len(all_winners) > 0 else 1.0
        agg_median_fav = float(all_winners["max_favorable_atr"].median()) if len(all_winners) > 0 else 1.5

        # Bayesian priors with kappa=15
        kappa = 15
        priors = {
            "drawdown_mu": agg_median_dd,
            "drawdown_kappa": kappa,
            "favorable_mu": agg_median_fav,
            "favorable_kappa": kappa,
            # Win rate prior: effective sample size of 20
            "win_alpha": round(overall_wr * 20, 2),
            "win_beta": round((1 - overall_wr) * 20, 2),
            "overall_win_rate": overall_wr,
            "total_trades": total_trades,
            "total_wins": total_wins,
        }

        # Save session stats
        models_dir = os.path.join(os.path.dirname(__file__), "models")
        os.makedirs(models_dir, exist_ok=True)

        stats_path = os.path.join(models_dir, "v1_session_stats.json")
        with open(stats_path, "w") as f:
            json.dump(session_stats, f, indent=2)

        # Save priors
        priors_path = os.path.join(models_dir, "v1_bayesian_priors.json")
        with open(priors_path, "w") as f:
            json.dump(priors, f, indent=2)

        return {"priors": priors, "session_stats": session_stats}

    def seed_training_dataset(self, harvested_df: pd.DataFrame,
                              dataset_manager=None) -> int:
        """Ingest harvested V1 data into the training dataset.

        Args:
            harvested_df: DataFrame from harvest_v1_data()
            dataset_manager: TrainingDatasetManager instance

        Returns:
            Number of rows ingested
        """
        if dataset_manager is None:
            from ml.dataset import TrainingDatasetManager
            dataset_manager = TrainingDatasetManager()

        if harvested_df.empty:
            return 0

        # Convert to list of dicts for ingestion
        trades = harvested_df.to_dict("records")

        # Tag all as v1_seed source
        for t in trades:
            t["source"] = "v1_seed"

        # Use a custom ingest that preserves v1_seed source
        # Clear old v1_seed rows first
        if not dataset_manager._df.empty and "source" in dataset_manager._df.columns:
            dataset_manager._df = dataset_manager._df[
                dataset_manager._df["source"] != "v1_seed"
            ]

        new_df = pd.DataFrame(trades)
        dataset_manager._df = pd.concat(
            [dataset_manager._df, new_df], ignore_index=True
        )
        dataset_manager._save()

        return len(trades)

    def _classify_session(self, candles: list[dict], idx: int) -> str:
        """Classify trading session from candle timestamp at index."""
        if idx < 0 or idx >= len(candles):
            return "off"

        dt_str = candles[idx].get("datetime", "")
        hour = _extract_hour(dt_str)

        if 7 <= hour < 10:
            return "london"
        elif 10 <= hour < 12:
            return "london"
        elif 12 <= hour < 16:
            return "ny_am"
        elif 16 <= hour < 21:
            return "ny_pm"
        elif 0 <= hour < 7:
            return "asia"
        else:
            return "off"


def fetch_candles(td_key: str, count: int = 2000) -> pd.DataFrame:
    """Fetch XAU/USD 1H candles from Twelve Data API."""
    import requests

    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": "XAU/USD",
        "interval": "1h",
        "outputsize": min(count, 5000),
        "apikey": td_key,
    }

    resp = requests.get(url, params=params, timeout=30)
    data = resp.json()

    if "values" not in data:
        raise ValueError(f"Twelve Data error: {data.get('message', data)}")

    df = pd.DataFrame(data["values"])
    df = df.iloc[::-1].reset_index(drop=True)  # chronological order

    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)

    return df


def main():
    parser = argparse.ArgumentParser(description="Seed ML models with V1 WFO data")
    parser.add_argument("--td-key", required=True, help="Twelve Data API key")
    parser.add_argument("--candles", type=int, default=2000, help="Number of candles")
    args = parser.parse_args()

    print("═" * 60)
    print("  V1 DATA HARVESTER — Seeding ML Models")
    print("═" * 60)

    # Fetch candles
    print(f"\n  Fetching {args.candles} candles from Twelve Data...")
    df = fetch_candles(args.td_key, args.candles)
    print(f"  Loaded {len(df)} candles")
    print(f"  Date range: {df['datetime'].iloc[0]} → {df['datetime'].iloc[-1]}")

    # Harvest
    harvester = V1DataHarvester()
    print("\n  Running V1 detector...")
    harvested = harvester.harvest_v1_data(df)
    print(f"  Harvested {len(harvested)} trades from V1 detector")

    if harvested.empty:
        print("  ⚠ No trades detected — check candle data")
        return

    # Seed Bayesian priors
    print("\n  Seeding Bayesian priors...")
    result = harvester.seed_bayesian(harvested)
    priors = result["priors"]
    print(f"  Bayesian priors seeded:")
    print(f"    drawdown_mu={priors['drawdown_mu']:.2f} ATR")
    print(f"    favorable_mu={priors['favorable_mu']:.2f} ATR")
    print(f"    win_alpha={priors['win_alpha']:.2f}, win_beta={priors['win_beta']:.2f}")
    print(f"    overall_win_rate={priors['overall_win_rate']:.1%}")

    # Print session stats
    print("\n  Session stats saved:")
    for session, stats in result["session_stats"].items():
        print(f"    {session}: {stats['trades']} trades, "
              f"WR={stats['win_rate']:.1%}, "
              f"p95_dd={stats['p95_drawdown']:.2f} ATR")

    # Seed training dataset
    print("\n  Seeding training dataset...")
    from ml.dataset import TrainingDatasetManager
    dm = TrainingDatasetManager()
    count = harvester.seed_training_dataset(harvested, dm)
    print(f"  Training dataset seeded: {count} rows at weight 0.5")

    stats = dm.get_stats()
    print(f"  Total dataset: {stats['total']} rows")
    print(f"  Outcome distribution: {stats['outcome_distribution']}")

    print("\n" + "═" * 60)
    print("  SEEDING COMPLETE")
    print("═" * 60)


if __name__ == "__main__":
    main()
