"""Tests for V1 Data Harvester (seed.py)."""
import json
import os
import tempfile

import pandas as pd
import pytest

from ml.seed import V1DataHarvester
from ml.config import make_test_config


def _make_candles(n=200, base_price=2900.0):
    """Generate synthetic candle data."""
    import random
    rng = random.Random(42)
    candles = []
    price = base_price
    for i in range(n):
        move = rng.gauss(0, 3)
        o = price
        c = price + move
        h = max(o, c) + abs(rng.gauss(0, 1.5))
        l = min(o, c) - abs(rng.gauss(0, 1.5))
        hour = i % 24
        candles.append({
            "datetime": f"2026-03-{10 + i // 24:02d} {hour:02d}:00:00",
            "open": round(o, 2),
            "high": round(h, 2),
            "low": round(l, 2),
            "close": round(c, 2),
        })
        price = c
    return candles


@pytest.fixture
def candles_df():
    return pd.DataFrame(_make_candles(200))


@pytest.fixture
def test_config():
    return make_test_config(
        dataset_parquet_path="ml/models/test_training_dataset.parquet"
    )


class TestV1DataHarvester:

    def test_harvest_returns_dataframe(self, candles_df, test_config):
        harvester = V1DataHarvester(config=test_config)
        result = harvester.harvest_v1_data(candles_df)
        assert isinstance(result, pd.DataFrame)

    def test_harvest_has_required_columns(self, candles_df, test_config):
        harvester = V1DataHarvester(config=test_config)
        result = harvester.harvest_v1_data(candles_df)
        if not result.empty:
            required = ["session", "regime", "atr_14", "source",
                        "sample_weight", "outcome", "max_drawdown_atr",
                        "max_favorable_atr", "direction"]
            for col in required:
                assert col in result.columns, f"Missing column: {col}"

    def test_harvest_tags_v1_seed(self, candles_df, test_config):
        harvester = V1DataHarvester(config=test_config)
        result = harvester.harvest_v1_data(candles_df)
        if not result.empty:
            assert (result["source"] == "v1_seed").all()
            assert (result["sample_weight"] == 0.5).all()

    def test_harvest_empty_with_short_data(self, test_config):
        short_df = pd.DataFrame(_make_candles(10))
        harvester = V1DataHarvester(config=test_config)
        result = harvester.harvest_v1_data(short_df)
        assert result.empty


class TestSeedBayesian:

    def test_seed_bayesian_returns_priors_and_stats(self, candles_df, test_config):
        harvester = V1DataHarvester(config=test_config)
        harvested = harvester.harvest_v1_data(candles_df)
        if harvested.empty:
            pytest.skip("No trades detected on synthetic data")

        result = harvester.seed_bayesian(harvested)
        assert "priors" in result
        assert "session_stats" in result

    def test_priors_have_required_fields(self, candles_df, test_config):
        harvester = V1DataHarvester(config=test_config)
        harvested = harvester.harvest_v1_data(candles_df)
        if harvested.empty:
            pytest.skip("No trades detected")

        result = harvester.seed_bayesian(harvested)
        priors = result["priors"]
        for key in ["drawdown_mu", "drawdown_kappa", "favorable_mu",
                     "favorable_kappa", "win_alpha", "win_beta"]:
            assert key in priors, f"Missing prior: {key}"

    def test_kappa_is_15(self, candles_df, test_config):
        harvester = V1DataHarvester(config=test_config)
        harvested = harvester.harvest_v1_data(candles_df)
        if harvested.empty:
            pytest.skip("No trades detected")

        result = harvester.seed_bayesian(harvested)
        assert result["priors"]["drawdown_kappa"] == 15
        assert result["priors"]["favorable_kappa"] == 15

    def test_session_stats_saved_to_disk(self, candles_df, test_config):
        harvester = V1DataHarvester(config=test_config)
        harvested = harvester.harvest_v1_data(candles_df)
        if harvested.empty:
            pytest.skip("No trades detected")

        harvester.seed_bayesian(harvested)
        path = os.path.join(os.path.dirname(__file__), "..", "models",
                            "v1_session_stats.json")
        assert os.path.exists(path)

    def test_empty_data_returns_empty(self, test_config):
        harvester = V1DataHarvester(config=test_config)
        result = harvester.seed_bayesian(pd.DataFrame())
        assert result["priors"] == {}


class TestSeedTrainingDataset:

    def test_seed_training_dataset_returns_count(self, candles_df, test_config):
        harvester = V1DataHarvester(config=test_config)
        harvested = harvester.harvest_v1_data(candles_df)
        if harvested.empty:
            pytest.skip("No trades detected")

        from ml.dataset import TrainingDatasetManager
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = make_test_config(
                dataset_parquet_path=os.path.join(tmpdir, "ds.parquet")
            )
            dm = TrainingDatasetManager(config=cfg)
            count = harvester.seed_training_dataset(harvested, dm)
            assert count == len(harvested)
            assert dm.get_stats()["total"] == count


class TestSessionClassification:

    def test_london_session(self, test_config):
        harvester = V1DataHarvester(config=test_config)
        candles = [{"datetime": "2026-03-10 08:00:00"}]
        assert harvester._classify_session(candles, 0) == "london"

    def test_ny_am_session(self, test_config):
        harvester = V1DataHarvester(config=test_config)
        candles = [{"datetime": "2026-03-10 13:00:00"}]
        assert harvester._classify_session(candles, 0) == "ny_am"

    def test_asia_session(self, test_config):
        harvester = V1DataHarvester(config=test_config)
        candles = [{"datetime": "2026-03-10 03:00:00"}]
        assert harvester._classify_session(candles, 0) == "asia"

    def test_off_session(self, test_config):
        harvester = V1DataHarvester(config=test_config)
        candles = [{"datetime": "2026-03-10 22:00:00"}]
        assert harvester._classify_session(candles, 0) == "off"
