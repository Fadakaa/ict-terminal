"""Trade setup logger with SQLite persistence.

Follows the DealTracker pattern from src/database.py:
constructor with optional path, _init_db, CRUD methods, clear.
"""
import sqlite3
import json
from datetime import datetime, timezone

import pandas as pd

from ml.config import get_config


class TradeLogger:
    def __init__(self, db_path: str = None, config: dict = None):
        cfg = config or get_config()
        self.db_path = db_path or cfg["db_path"]
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS trade_setups (
                setup_id    TEXT PRIMARY KEY,
                timestamp   TEXT NOT NULL,
                timeframe   TEXT,
                bias        TEXT,
                direction   TEXT,
                entry_price REAL,
                sl_price    REAL,
                tp1_price   REAL,
                tp2_price   REAL,
                tp3_price   REAL,
                features_json TEXT,
                analysis_json TEXT,
                candles_json  TEXT
            )""")
            conn.execute("""CREATE TABLE IF NOT EXISTS trade_outcomes (
                setup_id              TEXT PRIMARY KEY,
                outcome_timestamp     TEXT NOT NULL,
                actual_result         TEXT NOT NULL,
                max_favorable_excursion REAL,
                max_adverse_excursion   REAL,
                pnl_pips              REAL,
                FOREIGN KEY (setup_id) REFERENCES trade_setups(setup_id)
            )""")
            conn.execute("""CREATE TABLE IF NOT EXISTS training_log (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp       TEXT NOT NULL,
                model_type      TEXT NOT NULL,
                samples_used    INTEGER,
                accuracy        REAL,
                feature_version INTEGER
            )""")
            conn.execute("""CREATE TABLE IF NOT EXISTS bayesian_state (
                id                     INTEGER PRIMARY KEY CHECK (id = 1),
                alpha                  REAL NOT NULL DEFAULT 1.0,
                beta_param             REAL NOT NULL DEFAULT 1.0,
                consecutive_losses     INTEGER NOT NULL DEFAULT 0,
                max_consecutive_losses INTEGER NOT NULL DEFAULT 0,
                current_drawdown       REAL NOT NULL DEFAULT 0.0,
                max_drawdown           REAL NOT NULL DEFAULT 0.0,
                total_trades           INTEGER NOT NULL DEFAULT 0,
                total_wins             INTEGER NOT NULL DEFAULT 0,
                total_losses           INTEGER NOT NULL DEFAULT 0,
                cumulative_pnl         REAL NOT NULL DEFAULT 0.0,
                peak_pnl               REAL NOT NULL DEFAULT 0.0,
                updated_at             TEXT
            )""")

    def log_setup(self, setup_id: str, features: dict, analysis_json: str,
                  candles_json: str, metadata: dict):
        """Log a new trade setup for future training."""
        now = datetime.now(timezone.utc).isoformat()
        tps = []
        if "takeProfits" in (json.loads(analysis_json) if isinstance(analysis_json, str) else {}):
            pass  # TP prices come from metadata

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO trade_setups
                   (setup_id, timestamp, timeframe, bias, direction,
                    entry_price, sl_price, tp1_price, tp2_price, tp3_price,
                    features_json, analysis_json, candles_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (setup_id, now,
                 metadata.get("timeframe", ""),
                 metadata.get("bias", ""),
                 metadata.get("direction", ""),
                 metadata.get("entry_price", 0),
                 metadata.get("sl_price", 0),
                 metadata.get("tp1_price", 0),
                 metadata.get("tp2_price"),
                 metadata.get("tp3_price"),
                 json.dumps(features),
                 analysis_json,
                 candles_json))

    def log_outcome(self, setup_id: str, result: str, mfe: float,
                    mae: float, pnl: float) -> bool:
        """Record the outcome for a previously logged setup. Returns False if setup not found."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT setup_id FROM trade_setups WHERE setup_id = ?",
                (setup_id,)).fetchone()
            if not row:
                return False
            conn.execute(
                """INSERT OR REPLACE INTO trade_outcomes
                   (setup_id, outcome_timestamp, actual_result,
                    max_favorable_excursion, max_adverse_excursion, pnl_pips)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (setup_id, datetime.now(timezone.utc).isoformat(),
                 result, mfe, mae, pnl))
            return True

    def get_training_data(self) -> pd.DataFrame:
        """Return completed trades as a DataFrame with features + outcome."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("""
                SELECT s.features_json, o.actual_result,
                       o.max_favorable_excursion, o.max_adverse_excursion, o.pnl_pips
                FROM trade_setups s
                JOIN trade_outcomes o ON s.setup_id = o.setup_id
            """).fetchall()

        if not rows:
            return pd.DataFrame()

        records = []
        for features_json, result, mfe, mae, pnl in rows:
            features = json.loads(features_json) if features_json else {}
            features["actual_result"] = result
            features["mfe"] = mfe
            features["mae"] = mae
            features["pnl"] = pnl
            records.append(features)

        return pd.DataFrame(records)

    def get_completed_trade_count(self) -> int:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT COUNT(*) FROM trade_outcomes").fetchone()
            return row[0] if row else 0

    def get_setups_without_outcomes(self) -> list[dict]:
        """Return setups awaiting outcome resolution."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT s.* FROM trade_setups s
                LEFT JOIN trade_outcomes o ON s.setup_id = o.setup_id
                WHERE o.setup_id IS NULL
                ORDER BY s.timestamp DESC
            """).fetchall()
            return [dict(r) for r in rows]

    def log_training_run(self, model_type: str, samples: int,
                         accuracy: float, feature_version: int):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO training_log
                   (timestamp, model_type, samples_used, accuracy, feature_version)
                   VALUES (?, ?, ?, ?, ?)""",
                (datetime.now(timezone.utc).isoformat(),
                 model_type, samples, accuracy, feature_version))

    def get_last_training(self, model_type: str) -> dict | None:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """SELECT * FROM training_log
                   WHERE model_type = ?
                   ORDER BY id DESC LIMIT 1""",
                (model_type,)).fetchone()
            return dict(row) if row else None

    def get_bayesian_state(self) -> dict | None:
        """Return the current Bayesian state dict, or None if no state yet."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM bayesian_state WHERE id = 1"
            ).fetchone()
            if not row:
                return None
            d = dict(row)
            d.pop("id", None)
            d.pop("updated_at", None)
            return d

    def save_bayesian_state(self, state: dict):
        """Upsert the Bayesian state row (singleton, id=1)."""
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO bayesian_state
                   (id, alpha, beta_param, consecutive_losses, max_consecutive_losses,
                    current_drawdown, max_drawdown, total_trades, total_wins,
                    total_losses, cumulative_pnl, peak_pnl, updated_at)
                   VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (state["alpha"], state["beta_param"],
                 state["consecutive_losses"], state["max_consecutive_losses"],
                 state["current_drawdown"], state["max_drawdown"],
                 state["total_trades"], state["total_wins"],
                 state["total_losses"], state["cumulative_pnl"],
                 state["peak_pnl"], now))

    def get_trade_history(self) -> list[dict]:
        """Return completed trades with running cumulative PnL, ordered by outcome time."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT s.setup_id, s.timestamp, s.timeframe, s.bias, s.direction,
                       s.entry_price, s.sl_price, s.tp1_price,
                       o.outcome_timestamp, o.actual_result,
                       o.max_favorable_excursion, o.max_adverse_excursion, o.pnl_pips
                FROM trade_setups s
                JOIN trade_outcomes o ON s.setup_id = o.setup_id
                ORDER BY o.outcome_timestamp ASC
            """).fetchall()

        history = []
        cumulative = 0.0
        for row in rows:
            d = dict(row)
            cumulative += d.get("pnl_pips") or 0
            d["cumulative_pnl"] = round(cumulative, 2)
            history.append(d)
        return history

    def delete_setup(self, setup_id: str) -> bool:
        """Delete a trade setup and its outcome. Returns False if setup not found."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT setup_id FROM trade_setups WHERE setup_id = ?",
                (setup_id,)).fetchone()
            if not row:
                return False
            conn.execute("DELETE FROM trade_outcomes WHERE setup_id = ?", (setup_id,))
            conn.execute("DELETE FROM trade_setups WHERE setup_id = ?", (setup_id,))
            return True

    def clear(self):
        """Wipe all data from all tables."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM trade_outcomes")
            conn.execute("DELETE FROM trade_setups")
            conn.execute("DELETE FROM training_log")
            conn.execute("DELETE FROM bayesian_state")
