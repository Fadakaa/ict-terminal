"""Tests for TradeLogger SQLite persistence."""
import pytest
import json
from ml.database import TradeLogger


class TestTradeLogger:
    def test_init_creates_tables(self, tmp_db, test_config):
        db = TradeLogger(db_path=tmp_db, config=test_config)
        # Should not raise — tables exist
        assert db.get_completed_trade_count() == 0

    def test_log_setup(self, tmp_db, test_config, sample_analysis):
        db = TradeLogger(db_path=tmp_db, config=test_config)
        db.log_setup("setup-001", {"ob_count": 2}, json.dumps(sample_analysis), "[]", {
            "timeframe": "1h", "bias": "bullish", "direction": "long",
            "entry_price": 2650.0, "sl_price": 2643.0, "tp1_price": 2680.0,
        })
        pending = db.get_setups_without_outcomes()
        assert len(pending) == 1
        assert pending[0]["setup_id"] == "setup-001"

    def test_log_outcome(self, tmp_db, test_config, sample_analysis):
        db = TradeLogger(db_path=tmp_db, config=test_config)
        db.log_setup("setup-002", {"ob_count": 1}, json.dumps(sample_analysis), "[]", {
            "timeframe": "1h", "bias": "bullish", "direction": "long",
            "entry_price": 2650.0, "sl_price": 2643.0, "tp1_price": 2680.0,
        })
        result = db.log_outcome("setup-002", "tp1_hit", 35.0, 5.0, 30.0)
        assert result is True
        assert db.get_completed_trade_count() == 1

    def test_log_outcome_unknown_setup(self, tmp_db, test_config):
        db = TradeLogger(db_path=tmp_db, config=test_config)
        result = db.log_outcome("nonexistent", "stopped_out", 0, 10, -10)
        assert result is False

    def test_get_training_data(self, tmp_db, test_config, sample_analysis):
        db = TradeLogger(db_path=tmp_db, config=test_config)
        # Log 3 setups with outcomes
        for i in range(3):
            sid = f"setup-{i:03d}"
            db.log_setup(sid, {"ob_count": i + 1, "fvg_count": i}, json.dumps(sample_analysis), "[]", {
                "timeframe": "1h", "bias": "bullish", "direction": "long",
                "entry_price": 2650.0, "sl_price": 2643.0, "tp1_price": 2680.0,
            })
            db.log_outcome(sid, "tp1_hit" if i % 2 == 0 else "stopped_out", 30.0, 5.0, 25.0)
        df = db.get_training_data()
        assert len(df) == 3
        assert "actual_result" in df.columns

    def test_get_training_data_excludes_incomplete(self, tmp_db, test_config, sample_analysis):
        db = TradeLogger(db_path=tmp_db, config=test_config)
        db.log_setup("s1", {"ob_count": 1}, json.dumps(sample_analysis), "[]", {
            "timeframe": "1h", "bias": "bullish", "direction": "long",
            "entry_price": 2650.0, "sl_price": 2643.0, "tp1_price": 2680.0,
        })
        db.log_setup("s2", {"ob_count": 2}, json.dumps(sample_analysis), "[]", {
            "timeframe": "1h", "bias": "bullish", "direction": "long",
            "entry_price": 2650.0, "sl_price": 2643.0, "tp1_price": 2680.0,
        })
        db.log_outcome("s1", "tp1_hit", 30, 5, 25)
        # s2 has no outcome
        df = db.get_training_data()
        assert len(df) == 1

    def test_get_completed_trade_count(self, tmp_db, test_config, sample_analysis):
        db = TradeLogger(db_path=tmp_db, config=test_config)
        assert db.get_completed_trade_count() == 0
        db.log_setup("s1", {"ob_count": 1}, json.dumps(sample_analysis), "[]", {
            "timeframe": "1h", "bias": "bullish", "direction": "long",
            "entry_price": 2650.0, "sl_price": 2643.0, "tp1_price": 2680.0,
        })
        db.log_outcome("s1", "stopped_out", 2, 8, -7)
        assert db.get_completed_trade_count() == 1

    def test_get_setups_without_outcomes(self, tmp_db, test_config, sample_analysis):
        db = TradeLogger(db_path=tmp_db, config=test_config)
        db.log_setup("s1", {}, json.dumps(sample_analysis), "[]", {
            "timeframe": "1h", "bias": "bullish", "direction": "long",
            "entry_price": 2650.0, "sl_price": 2643.0, "tp1_price": 2680.0,
        })
        db.log_setup("s2", {}, json.dumps(sample_analysis), "[]", {
            "timeframe": "1h", "bias": "bullish", "direction": "long",
            "entry_price": 2650.0, "sl_price": 2643.0, "tp1_price": 2680.0,
        })
        db.log_outcome("s1", "tp1_hit", 30, 5, 25)
        pending = db.get_setups_without_outcomes()
        assert len(pending) == 1
        assert pending[0]["setup_id"] == "s2"

    def test_log_training_run(self, tmp_db, test_config):
        db = TradeLogger(db_path=tmp_db, config=test_config)
        db.log_training_run("classifier", 30, 0.85, 1)
        last = db.get_last_training("classifier")
        assert last is not None
        assert last["samples_used"] == 30
        assert last["accuracy"] == 0.85

    def test_clear(self, tmp_db, test_config, sample_analysis):
        db = TradeLogger(db_path=tmp_db, config=test_config)
        db.log_setup("s1", {}, json.dumps(sample_analysis), "[]", {
            "timeframe": "1h", "bias": "bullish", "direction": "long",
            "entry_price": 2650.0, "sl_price": 2643.0, "tp1_price": 2680.0,
        })
        db.clear()
        assert db.get_completed_trade_count() == 0
        assert len(db.get_setups_without_outcomes()) == 0

    # ── Bayesian state persistence ──────────────────────────────

    def test_bayesian_state_none_initially(self, tmp_db, test_config):
        db = TradeLogger(db_path=tmp_db, config=test_config)
        assert db.get_bayesian_state() is None

    def test_save_and_get_bayesian_state(self, tmp_db, test_config):
        db = TradeLogger(db_path=tmp_db, config=test_config)
        state = {
            "alpha": 5.0, "beta_param": 3.0,
            "consecutive_losses": 0, "max_consecutive_losses": 2,
            "current_drawdown": 10.0, "max_drawdown": 50.0,
            "total_trades": 7, "total_wins": 4,
            "total_losses": 3, "cumulative_pnl": 120.0,
            "peak_pnl": 130.0,
        }
        db.save_bayesian_state(state)
        loaded = db.get_bayesian_state()
        assert loaded is not None
        assert loaded["alpha"] == 5.0
        assert loaded["total_trades"] == 7

    def test_bayesian_state_upsert(self, tmp_db, test_config):
        db = TradeLogger(db_path=tmp_db, config=test_config)
        state1 = {
            "alpha": 2.0, "beta_param": 1.0,
            "consecutive_losses": 0, "max_consecutive_losses": 0,
            "current_drawdown": 0.0, "max_drawdown": 0.0,
            "total_trades": 1, "total_wins": 1,
            "total_losses": 0, "cumulative_pnl": 50.0,
            "peak_pnl": 50.0,
        }
        db.save_bayesian_state(state1)
        state2 = {
            "alpha": 2.0, "beta_param": 2.0,
            "consecutive_losses": 1, "max_consecutive_losses": 1,
            "current_drawdown": 30.0, "max_drawdown": 30.0,
            "total_trades": 2, "total_wins": 1,
            "total_losses": 1, "cumulative_pnl": 20.0,
            "peak_pnl": 50.0,
        }
        db.save_bayesian_state(state2)
        loaded = db.get_bayesian_state()
        assert loaded["total_trades"] == 2
        assert loaded["consecutive_losses"] == 1

    # ── Trade history ──────────────────────────────────────────

    def test_trade_history_empty(self, tmp_db, test_config):
        db = TradeLogger(db_path=tmp_db, config=test_config)
        assert db.get_trade_history() == []

    def test_trade_history_shape(self, tmp_db, test_config, sample_analysis):
        db = TradeLogger(db_path=tmp_db, config=test_config)
        db.log_setup("h-001", {"ob_count": 1}, json.dumps(sample_analysis), "[]", {
            "timeframe": "1h", "bias": "bullish", "direction": "long",
            "entry_price": 2650.0, "sl_price": 2643.0, "tp1_price": 2680.0,
        })
        db.log_outcome("h-001", "tp1_hit", 35.0, 5.0, 30.0)
        history = db.get_trade_history()
        assert len(history) == 1
        row = history[0]
        assert row["setup_id"] == "h-001"
        assert row["actual_result"] == "tp1_hit"
        assert row["pnl_pips"] == 30.0
        assert row["cumulative_pnl"] == 30.0
        assert "entry_price" in row
        assert "outcome_timestamp" in row

    def test_trade_history_cumulative_pnl(self, tmp_db, test_config, sample_analysis):
        db = TradeLogger(db_path=tmp_db, config=test_config)
        pnls = [30.0, -10.0, 20.0]
        for i, pnl in enumerate(pnls):
            sid = f"h-{i:03d}"
            db.log_setup(sid, {"ob_count": 1}, json.dumps(sample_analysis), "[]", {
                "timeframe": "1h", "bias": "bullish", "direction": "long",
                "entry_price": 2650.0, "sl_price": 2643.0, "tp1_price": 2680.0,
            })
            result = "tp1_hit" if pnl > 0 else "stopped_out"
            db.log_outcome(sid, result, abs(pnl), abs(pnl) * 0.2, pnl)
        history = db.get_trade_history()
        assert len(history) == 3
        assert history[0]["cumulative_pnl"] == 30.0
        assert history[1]["cumulative_pnl"] == 20.0   # 30 + (-10)
        assert history[2]["cumulative_pnl"] == 40.0   # 30 + (-10) + 20

    def test_clear_wipes_bayesian_state(self, tmp_db, test_config):
        db = TradeLogger(db_path=tmp_db, config=test_config)
        state = {
            "alpha": 3.0, "beta_param": 2.0,
            "consecutive_losses": 1, "max_consecutive_losses": 1,
            "current_drawdown": 0.0, "max_drawdown": 20.0,
            "total_trades": 4, "total_wins": 2,
            "total_losses": 2, "cumulative_pnl": 60.0,
            "peak_pnl": 80.0,
        }
        db.save_bayesian_state(state)
        db.clear()
        assert db.get_bayesian_state() is None
