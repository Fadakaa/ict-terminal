"""Tests for ML training pipeline — TDD."""
import os
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

from ml.training import train_classifier, train_quantile, should_retrain
from ml.database import TradeLogger
from ml.config import make_test_config


@pytest.fixture
def seeded_db(tmp_db, tmp_path):
    """DB with enough completed trades for training.

    Needs ≥25 samples so holdout eval works (20 train + 5 holdout at 80/20 split).
    """
    model_dir = str(tmp_path / "models")
    os.makedirs(model_dir, exist_ok=True)
    cfg = make_test_config(db_path=tmp_db, min_training_samples=3,
                           min_training_samples_quantile=5,
                           model_dir=model_dir)
    db = TradeLogger(db_path=tmp_db, config=cfg)
    # Insert 30 setups + outcomes (need ≥25 for holdout eval)
    for i in range(30):
        setup_id = f"test_{i}"
        db.log_setup(setup_id, {"ob_count": i % 5, "fvg_count": i % 3},
                     '{}', '[]', {"timeframe": "1h", "bias": "bullish",
                                   "direction": "long", "entry_price": 2650 + i,
                                   "sl_price": 2640 + i, "tp1_price": 2680 + i})
        result = "tp1_hit" if i % 2 == 0 else "stopped_out"
        db.log_outcome(setup_id, result, 10.0 + i, 3.0, 7.0 + i)
    return db, cfg


class TestTrainClassifier:
    def test_insufficient_data(self, tmp_db):
        cfg = make_test_config(db_path=tmp_db, min_training_samples=30)
        db = TradeLogger(db_path=tmp_db, config=cfg)
        result = train_classifier(db, config=cfg)
        assert result["status"] == "insufficient_data"

    def test_returns_trained_with_mock(self, seeded_db):
        db, cfg = seeded_db
        mock_predictor = MagicMock()
        mock_predictor.evaluate.return_value = {"accuracy": 0.85}

        with patch("ml.training.TabularPredictor", create=True, return_value=mock_predictor) as MockTP:
            MockTP.return_value.fit.return_value = mock_predictor
            result = train_classifier(db, config=cfg)

        assert result["status"] == "trained"
        assert result["model_type"] in ("classifier", "classifier_binary", "classifier_multi3")
        assert result["samples"] == 30
        assert result["accuracy"] == 0.85

    def test_logs_training_run(self, seeded_db):
        db, cfg = seeded_db
        mock_predictor = MagicMock()
        mock_predictor.evaluate.return_value = {"accuracy": 0.75}

        with patch("ml.training.TabularPredictor", create=True, return_value=mock_predictor) as MockTP:
            MockTP.return_value.fit.return_value = mock_predictor
            train_classifier(db, config=cfg)

        # Training logs as "classifier_binary" or "classifier_multi3"
        last = db.get_last_training("classifier_binary")
        if last is None:
            last = db.get_last_training("classifier_multi3")
        assert last is not None
        assert last["samples_used"] == 30
        assert last["accuracy"] == 0.75


class TestTrainQuantile:
    def test_insufficient_data(self, tmp_db):
        cfg = make_test_config(db_path=tmp_db, min_training_samples_quantile=50)
        db = TradeLogger(db_path=tmp_db, config=cfg)
        result = train_quantile(db, config=cfg)
        assert result["status"] == "insufficient_data"

    def test_returns_trained_with_mock(self, seeded_db):
        db, cfg = seeded_db
        mock_predictor = MagicMock()

        with patch("ml.training.TabularPredictor", create=True, return_value=mock_predictor) as MockTP:
            MockTP.return_value.fit.return_value = mock_predictor
            result = train_quantile(db, config=cfg)

        assert result["status"] == "trained"
        assert result["model_type"] == "quantile_mfe"


class TestShouldRetrain:
    def test_false_when_insufficient_data(self, tmp_db):
        cfg = make_test_config(db_path=tmp_db, min_training_samples=30)
        db = TradeLogger(db_path=tmp_db, config=cfg)
        assert should_retrain(db, config=cfg) is False

    def test_true_when_no_prior_training(self, seeded_db):
        db, cfg = seeded_db
        assert should_retrain(db, config=cfg) is True

    def test_false_after_recent_training(self, seeded_db):
        db, cfg = seeded_db
        # Simulate a training run that used all 30 samples
        db.log_training_run("classifier", 30, 0.85, 1)
        # retrain_on_n_new_trades defaults to 10, so no retrain needed
        assert should_retrain(db, config=cfg) is False

    def test_true_when_enough_new_trades(self, tmp_db):
        cfg = make_test_config(db_path=tmp_db, min_training_samples=3,
                               retrain_on_n_new_trades=2)
        db = TradeLogger(db_path=tmp_db, config=cfg)
        # Create 5 trades
        for i in range(5):
            db.log_setup(f"s_{i}", {}, '{}', '[]',
                         {"timeframe": "1h", "entry_price": 2650 + i,
                          "sl_price": 2640, "tp1_price": 2680})
            db.log_outcome(f"s_{i}", "tp1_hit", 10.0, 3.0, 7.0)
        # Log training at 3 samples
        db.log_training_run("classifier", 3, 0.7, 1)
        # 5 - 3 = 2 new trades >= retrain_on_n_new_trades (2)
        assert should_retrain(db, config=cfg) is True


class TestTrainWithDatasetManager:
    """Step 8: Tests for dataset_manager integration in train_classifier."""

    def _make_blended_df(self, n=10):
        """Create a DataFrame mimicking TrainingDatasetManager.get_blended_dataset()."""
        rows = []
        outcomes = ["tp1_hit", "stopped_out", "tp2_hit", "tp3_hit"]
        for i in range(n):
            row = {
                "ob_count": i % 3,
                "fvg_count": i % 2,
                "atr": 5.0 + i * 0.1,
                "body_ratio": 0.5 + i * 0.01,
                "outcome": outcomes[i % len(outcomes)],
                "source": "wfo" if i < 7 else "live",
                "sample_weight": 1.0 if i < 7 else 5.0,
                "mfe": 10.0 + i,
                "mae": 3.0,
                "pnl": 7.0 + i,
                "regime": "trending_up",
                "fold": 1,
                "setup_id": f"wfo-{i:04d}",
            }
            rows.append(row)
        return pd.DataFrame(rows)

    def test_uses_blended_dataset_when_dm_provided(self, tmp_db):
        """train_classifier uses dataset_manager.get_blended_dataset() when provided."""
        cfg = make_test_config(db_path=tmp_db, min_training_samples=3)
        db = TradeLogger(db_path=tmp_db, config=cfg)

        dm = MagicMock()
        dm.get_blended_dataset.return_value = self._make_blended_df(10)

        # AutoGluon not installed → returns early but after getting data
        result = train_classifier(db, config=cfg, dataset_manager=dm)

        dm.get_blended_dataset.assert_called_once()
        assert result["samples"] == 10

    def test_uses_outcome_label_with_dm(self, tmp_db, tmp_path):
        """When dataset_manager provided, training derives __binary_outcome from 'outcome'."""
        model_dir = str(tmp_path / "models")
        os.makedirs(model_dir, exist_ok=True)
        cfg = make_test_config(db_path=tmp_db, min_training_samples=3, model_dir=model_dir)
        db = TradeLogger(db_path=tmp_db, config=cfg)

        dm = MagicMock()
        dm.get_blended_dataset.return_value = self._make_blended_df(10)

        mock_predictor = MagicMock()
        mock_predictor.evaluate.return_value = {"accuracy": 0.80}

        with patch("ml.training.TabularPredictor", create=True, return_value=mock_predictor) as MockTP:
            MockTP.return_value.fit.return_value = mock_predictor
            result = train_classifier(db, config=cfg, dataset_manager=dm)

        # Training derives __binary_outcome from 'outcome' for binary classifier
        call_kwargs = MockTP.call_args
        assert call_kwargs.kwargs["label"] == "__binary_outcome"
        assert result["status"] == "trained"

    def test_sample_weight_passed_to_fit(self, tmp_db, tmp_path):
        """sample_weight column is extracted and passed to AutoGluon .fit()."""
        model_dir = str(tmp_path / "models")
        os.makedirs(model_dir, exist_ok=True)
        cfg = make_test_config(db_path=tmp_db, min_training_samples=3, model_dir=model_dir)
        db = TradeLogger(db_path=tmp_db, config=cfg)

        dm = MagicMock()
        dm.get_blended_dataset.return_value = self._make_blended_df(10)

        mock_predictor = MagicMock()
        mock_predictor.evaluate.return_value = {"accuracy": 0.80}

        with patch("ml.training.TabularPredictor", create=True, return_value=mock_predictor) as MockTP:
            mock_instance = MagicMock()
            mock_instance.evaluate.return_value = {"accuracy": 0.80}
            MockTP.return_value.fit.return_value = mock_instance
            train_classifier(db, config=cfg, dataset_manager=dm)

        # In AutoGluon 1.5+, sample_weight is passed to TabularPredictor
        # constructor, not .fit(). Check constructor got it.
        constructor_call = MockTP.call_args
        assert constructor_call.kwargs.get("sample_weight") == "sample_weight"
        # Also check sample_weight column is kept in the training DataFrame
        fit_call = MockTP.return_value.fit.call_args
        train_df = fit_call.args[0]
        assert "sample_weight" in train_df.columns

    def test_non_feature_cols_dropped_with_dm(self, tmp_db, tmp_path):
        """Non-feature columns (source, regime, fold, etc.) are dropped before training."""
        model_dir = str(tmp_path / "models")
        os.makedirs(model_dir, exist_ok=True)
        cfg = make_test_config(db_path=tmp_db, min_training_samples=3, model_dir=model_dir)
        db = TradeLogger(db_path=tmp_db, config=cfg)

        dm = MagicMock()
        dm.get_blended_dataset.return_value = self._make_blended_df(10)

        mock_predictor = MagicMock()
        mock_predictor.evaluate.return_value = {"accuracy": 0.80}

        with patch("ml.training.TabularPredictor", create=True, return_value=mock_predictor) as MockTP:
            mock_instance = MagicMock()
            mock_instance.evaluate.return_value = {"accuracy": 0.80}
            MockTP.return_value.fit.return_value = mock_instance
            train_classifier(db, config=cfg, dataset_manager=dm)

        # Get the DataFrame passed to .fit()
        fit_call = MockTP.return_value.fit.call_args
        train_df = fit_call.args[0]
        # These should be dropped (sample_weight is KEPT for AutoGluon 1.5+)
        for col in ["source", "mfe", "mae", "pnl",
                     "regime", "fold", "setup_id"]:
            assert col not in train_df.columns, f"{col} should be dropped"
        # Derived binary label and sample_weight should remain
        assert "__binary_outcome" in train_df.columns
        assert "sample_weight" in train_df.columns

    def test_falls_back_to_db_without_dm(self, seeded_db):
        """Without dataset_manager, falls back to db.get_training_data()."""
        db, cfg = seeded_db
        # No dataset_manager → uses db
        result = train_classifier(db, config=cfg)
        # AutoGluon not installed → returns with samples from DB
        assert result["samples"] == 30

    def test_insufficient_data_with_dm(self, tmp_db):
        """Returns insufficient_data when dataset_manager has too few rows."""
        cfg = make_test_config(db_path=tmp_db, min_training_samples=30)
        db = TradeLogger(db_path=tmp_db, config=cfg)

        dm = MagicMock()
        dm.get_blended_dataset.return_value = self._make_blended_df(5)

        result = train_classifier(db, config=cfg, dataset_manager=dm)
        assert result["status"] == "insufficient_data"
        assert result["samples"] == 5


def test_train_classifier_live_only_passes_through(monkeypatch):
    """live_only=True should be forwarded to dataset_manager.get_blended_dataset()."""
    captured = {}

    class MockDM:
        def get_blended_dataset(self, live_only=False):
            captured["live_only"] = live_only
            import pandas as pd
            return pd.DataFrame()  # empty triggers insufficient_data

    from ml import training
    result = training.train_classifier(
        db=None, dataset_manager=MockDM(), live_only=True
    )
    assert captured["live_only"] is True
    assert result["status"] == "insufficient_data"
