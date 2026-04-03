"""SQLite persistence for scanner setups.

Stores auto-detected ICT setups from the headless scanner.
Follows the same pattern as ml/database.py (TradeLogger).
"""
import json
import os
import sqlite3
import uuid
from datetime import datetime, timedelta


class ScannerDB:
    """CRUD for scanner_setups table."""

    def __init__(self, db_path: str = None):
        if db_path is None:
            from ml.config import get_config
            db_path = get_config().get("db_path")
        self.db_path = db_path or os.path.join(
            os.path.dirname(__file__), "models", "scanner.db"
        )
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        with self._conn() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS scanner_setups (
                    id              TEXT PRIMARY KEY,
                    created_at      TEXT NOT NULL,
                    resolved_at     TEXT,
                    status          TEXT NOT NULL DEFAULT 'pending',
                    timeframe       TEXT NOT NULL DEFAULT '1h',
                    direction       TEXT NOT NULL,
                    bias            TEXT,
                    entry_price     REAL,
                    sl_price        REAL,
                    calibrated_sl   REAL,
                    tp1             REAL,
                    tp2             REAL,
                    tp3             REAL,
                    setup_quality   TEXT,
                    killzone        TEXT,
                    rr_ratios       TEXT,
                    analysis_json   TEXT,
                    calibration_json TEXT,
                    outcome         TEXT,
                    resolved_price  REAL,
                    pnl_rr          REAL,
                    auto_resolved   INTEGER DEFAULT 0,
                    candle_hash     TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_scanner_status
                ON scanner_setups(status)
            """)
            # Add columns if upgrading from older schema
            for col, col_type in [("gross_rr", "REAL"), ("cost_rr", "REAL"),
                                   ("notified", "INTEGER DEFAULT 0"),
                                   ("mfe_atr", "REAL"),
                                   ("mae_atr", "REAL"),
                                   ("entry_zone_type", "TEXT"),
                                   ("entry_zone_high", "REAL"),
                                   ("entry_zone_low", "REAL"),
                                   ("entry_zone_position", "REAL"),
                                   ("api_cost_usd", "REAL"),
                                   ("detection_notified", "INTEGER DEFAULT 0"),
                                   ("thesis_id", "TEXT")]:
                try:
                    conn.execute(f"ALTER TABLE scanner_setups ADD COLUMN {col} {col_type}")
                except sqlite3.OperationalError:
                    pass  # Column already exists

            # Session recaps table (Opus killzone summaries)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS session_recaps (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    killzone        TEXT NOT NULL,
                    date            TEXT NOT NULL,
                    recap_json      TEXT NOT NULL,
                    created_at      TEXT NOT NULL,
                    UNIQUE(killzone, date)
                )
            """)

            # Killzone prospects table (anticipatory zone alerts)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS killzone_prospects (
                    id              TEXT PRIMARY KEY,
                    killzone        TEXT NOT NULL,
                    date            TEXT NOT NULL,
                    prospect_json   TEXT NOT NULL,
                    status          TEXT NOT NULL DEFAULT 'active',
                    created_at      TEXT NOT NULL,
                    resolved_at     TEXT,
                    trigger_result  TEXT
                )
            """)

            # Notification lifecycle table — dedup + threading for thesis-aware alerts
            conn.execute("""
                CREATE TABLE IF NOT EXISTS notification_lifecycle (
                    id              TEXT PRIMARY KEY,
                    thesis_id       TEXT NOT NULL,
                    timeframe       TEXT NOT NULL,
                    stage           INTEGER NOT NULL,
                    stage_name      TEXT NOT NULL,
                    sent_at         TEXT NOT NULL,
                    telegram_msg_id TEXT,
                    setup_id        TEXT,
                    payload_json    TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_notification_thesis
                ON notification_lifecycle(thesis_id, stage)
            """)

            # Zone cooldowns — block re-entry at stopped zone for rest of killzone
            conn.execute("""
                CREATE TABLE IF NOT EXISTS zone_cooldowns (
                    id           TEXT PRIMARY KEY,
                    zone_key     TEXT NOT NULL,
                    killzone     TEXT NOT NULL,
                    date         TEXT NOT NULL,
                    blocked_at   TEXT NOT NULL,
                    prospect_id  TEXT,
                    setup_id     TEXT
                )
            """)
            conn.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_zone_cooldown_lookup
                ON zone_cooldowns(zone_key, killzone, date)
            """)

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def store_setup(self, direction: str, bias: str, entry_price: float,
                    sl_price: float, calibrated_sl: float,
                    tps: list, setup_quality: str, killzone: str,
                    rr_ratios: list, analysis_json: dict,
                    calibration_json: dict, candle_hash: str = "",
                    timeframe: str = "1h",
                    status: str = "pending",
                    entry_zone_type: str = None,
                    entry_zone_high: float = None,
                    entry_zone_low: float = None,
                    entry_zone_position: float = None,
                    thesis_id: str = None) -> str:
        """Store a new setup. Returns the setup ID.

        Args:
            status: 'pending' (normal) or 'shadow' (Opus-rejected, tracked for outcome)
            entry_zone_type: "ob" or "fvg" — the zone type the entry is in
            entry_zone_high: High of the entry zone
            entry_zone_low: Low of the entry zone
            entry_zone_position: 0.0 (shallow) to 1.0 (deep) within the zone
            thesis_id: Narrative thesis ID for lifecycle tracking
        """
        setup_id = str(uuid.uuid4())[:8]
        now = datetime.utcnow().isoformat()
        tp1 = tps[0] if len(tps) > 0 else None
        tp2 = tps[1] if len(tps) > 1 else None
        tp3 = tps[2] if len(tps) > 2 else None

        with self._conn() as conn:
            conn.execute("""
                INSERT INTO scanner_setups
                (id, created_at, status, timeframe, direction, bias, entry_price,
                 sl_price, calibrated_sl, tp1, tp2, tp3, setup_quality,
                 killzone, rr_ratios, analysis_json, calibration_json, candle_hash,
                 entry_zone_type, entry_zone_high, entry_zone_low, entry_zone_position,
                 thesis_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                setup_id, now, status, timeframe, direction, bias, entry_price,
                sl_price, calibrated_sl, tp1, tp2, tp3, setup_quality,
                killzone, json.dumps(rr_ratios),
                json.dumps(analysis_json), json.dumps(calibration_json),
                candle_hash,
                entry_zone_type, entry_zone_high, entry_zone_low,
                entry_zone_position,
                thesis_id,
            ))

        return setup_id

    def get_pending(self, include_shadow: bool = False) -> list:
        """Return pending setups. Optionally include shadow (Opus-rejected) setups."""
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            if include_shadow:
                rows = conn.execute(
                    "SELECT * FROM scanner_setups WHERE status IN ('pending', 'shadow') ORDER BY created_at DESC"
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM scanner_setups WHERE status = 'pending' ORDER BY created_at DESC"
                ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_monitoring_setups(self) -> list:
        """Return setups with status='monitoring' (C/D grade, awaiting promotion)."""
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM scanner_setups WHERE status = 'monitoring' "
                "ORDER BY created_at DESC"
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def promote_setup(self, setup_id: str):
        """Promote a monitoring setup to pending (C/D met displacement criteria).

        Only affects setups with status='monitoring'. No-op for other statuses.
        """
        with self._conn() as conn:
            conn.execute(
                "UPDATE scanner_setups SET status = 'pending' "
                "WHERE id = ? AND status = 'monitoring'",
                (setup_id,))

    def get_unnotified_setups(self) -> list:
        """Return pending A/B setups that haven't been notified yet.

        Only A and B grade setups are worth notifying — C/D are stored
        for outcome tracking but shouldn't trigger Telegram alerts.
        """
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM scanner_setups WHERE status = 'pending' "
                "AND (notified IS NULL OR notified = 0) "
                "AND setup_quality IN ('A', 'B') "
                "ORDER BY created_at DESC"
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def mark_notified(self, setup_id: str):
        """Mark a setup as notified (entry proximity alert sent)."""
        with self._conn() as conn:
            conn.execute(
                "UPDATE scanner_setups SET notified = 1 WHERE id = ?",
                (setup_id,))

    def mark_detection_notified(self, setup_id: str):
        """Mark a setup as detection-notified (initial detection alert sent)."""
        with self._conn() as conn:
            conn.execute(
                "UPDATE scanner_setups SET detection_notified = 1 WHERE id = ?",
                (setup_id,))

    def get_history(self) -> list:
        """Return all resolved setups, most recent first."""
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM scanner_setups WHERE status != 'pending' ORDER BY resolved_at DESC"
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def resolve_setup(self, setup_id: str, outcome: str,
                      resolved_price: float = None, pnl_rr: float = None,
                      auto: bool = False,
                      gross_rr: float = None, cost_rr: float = None,
                      mfe_atr: float = None, mae_atr: float = None):
        """Mark a setup as resolved with optional cost breakdown and MFE/MAE."""
        now = datetime.utcnow().isoformat()
        with self._conn() as conn:
            conn.execute("""
                UPDATE scanner_setups
                SET status = 'resolved', outcome = ?, resolved_at = ?,
                    resolved_price = ?, pnl_rr = ?, auto_resolved = ?,
                    gross_rr = ?, cost_rr = ?,
                    mfe_atr = ?, mae_atr = ?
                WHERE id = ?
            """, (outcome, now, resolved_price, pnl_rr, int(auto),
                  gross_rr, cost_rr, mfe_atr, mae_atr, setup_id))

    def expire_old(self, hours: int = 24) -> int:
        """Expire pending setups older than N hours. Returns count expired."""
        cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
        with self._conn() as conn:
            cursor = conn.execute("""
                UPDATE scanner_setups
                SET status = 'expired', outcome = 'expired', resolved_at = ?
                WHERE status = 'pending' AND created_at < ?
            """, (datetime.utcnow().isoformat(), cutoff))
            return cursor.rowcount

    def expire_by_timeframe(self, tf_hours: dict, status: str = "pending") -> int:
        """Expire setups based on per-timeframe max age.

        Args:
            tf_hours: dict mapping timeframe -> max hours before expiry
                      e.g. {"5min": 4, "1h": 48, "1day": 336}
            status: which status to expire ('pending' or 'monitoring')
        """
        total = 0
        now = datetime.utcnow().isoformat()
        with self._conn() as conn:
            for tf, hours in tf_hours.items():
                cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
                cursor = conn.execute("""
                    UPDATE scanner_setups
                    SET status = 'expired', outcome = 'expired', resolved_at = ?
                    WHERE status = ? AND timeframe = ? AND created_at < ?
                """, (now, status, tf, cutoff))
                total += cursor.rowcount
        return total

    def find_duplicate(self, direction: str, entry_price: float,
                       timeframe: str = "1h",
                       minutes: int = 30, price_tolerance: float = 1.0) -> bool:
        """Check if a similar setup was recently stored on the same timeframe."""
        cutoff = (datetime.utcnow() - timedelta(minutes=minutes)).isoformat()
        with self._conn() as conn:
            row = conn.execute("""
                SELECT COUNT(*) FROM scanner_setups
                WHERE direction = ? AND timeframe = ? AND created_at > ?
                AND ABS(entry_price - ?) < ?
            """, (direction, timeframe, cutoff, entry_price, price_tolerance)).fetchone()
            return row[0] > 0

    def get_stats(self) -> dict:
        """Return scanner statistics."""
        with self._conn() as conn:
            pending = conn.execute(
                "SELECT COUNT(*) FROM scanner_setups WHERE status = 'pending'"
            ).fetchone()[0]
            total = conn.execute(
                "SELECT COUNT(*) FROM scanner_setups"
            ).fetchone()[0]
            resolved = conn.execute(
                "SELECT COUNT(*) FROM scanner_setups WHERE status = 'resolved'"
            ).fetchone()[0]
            wins = conn.execute(
                "SELECT COUNT(*) FROM scanner_setups WHERE outcome LIKE 'tp%'"
            ).fetchone()[0]
            last_row = conn.execute(
                "SELECT created_at FROM scanner_setups ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
            last_scan = last_row[0] if last_row else None

        return {
            "pending": pending,
            "total": total,
            "resolved": resolved,
            "wins": wins,
            "win_rate": round(wins / resolved, 3) if resolved > 0 else 0,
            "last_scan": last_scan,
        }

    def update_api_cost(self, setup_id: str, api_cost_usd: float):
        """Store the total API cost spent to create/calibrate this setup."""
        with self._conn() as conn:
            conn.execute(
                "UPDATE scanner_setups SET api_cost_usd = ? WHERE id = ?",
                (round(api_cost_usd, 6), setup_id))

    def get_resolved_with_costs(self) -> list:
        """Return resolved setups that have api_cost_usd tracked."""
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT id, timeframe, killzone, setup_quality, outcome,
                       pnl_rr, gross_rr, cost_rr, api_cost_usd,
                       mfe_atr, mae_atr, direction, created_at, resolved_at
                FROM scanner_setups
                WHERE status = 'resolved' AND api_cost_usd IS NOT NULL
                ORDER BY resolved_at ASC
            """).fetchall()
        return [dict(r) for r in rows]

    def get_resolved_setups(self) -> list:
        """Return all resolved setups with full JSON for backfilling."""
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM scanner_setups WHERE status = 'resolved' ORDER BY resolved_at"
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_pnl_history(self) -> list:
        """Return ALL resolved setups with pnl_rr, oldest first (for equity curve)."""
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT id, created_at, resolved_at, direction, outcome, pnl_rr,
                          gross_rr, cost_rr,
                          entry_price, resolved_price, setup_quality, timeframe
                   FROM scanner_setups
                   WHERE status != 'pending' AND pnl_rr IS NOT NULL
                   ORDER BY resolved_at ASC"""
            ).fetchall()
        return [dict(r) for r in rows]

    def get_setups_by_killzone(self, killzone: str, date: str) -> list:
        """Return setups created during a specific killzone and date."""
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT * FROM scanner_setups
                   WHERE killzone = ? AND created_at LIKE ?
                   ORDER BY created_at ASC""",
                (killzone, f"{date}%"),
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def store_session_recap(self, killzone: str, date: str, recap_json: dict):
        """Store an Opus session recap. Uses INSERT OR REPLACE for idempotency."""
        now = datetime.utcnow().isoformat()
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO session_recaps
                   (killzone, date, recap_json, created_at)
                   VALUES (?, ?, ?, ?)""",
                (killzone, date, json.dumps(recap_json), now),
            )

    def get_latest_session_recap(self) -> dict | None:
        """Return the most recent session recap."""
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM session_recaps ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
        if not row:
            return None
        d = dict(row)
        if d.get("recap_json") and isinstance(d["recap_json"], str):
            try:
                d["recap_json"] = json.loads(d["recap_json"])
            except (json.JSONDecodeError, TypeError):
                pass
        return d

    def store_prospect(self, killzone: str, prospect_json: dict) -> str:
        """Store a killzone prospect. Returns the prospect ID."""
        prospect_id = str(uuid.uuid4())[:8]
        now = datetime.utcnow().isoformat()
        date = datetime.utcnow().strftime("%Y-%m-%d")
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO killzone_prospects
                (id, killzone, date, prospect_json, status, created_at)
                VALUES (?, ?, ?, ?, 'active', ?)
            """, (prospect_id, killzone, date, json.dumps(prospect_json), now))
        return prospect_id

    def get_active_prospects(self, include_displaced: bool = False) -> list:
        """Return active prospects. Optionally include displaced (waiting for retrace)."""
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            if include_displaced:
                rows = conn.execute(
                    "SELECT * FROM killzone_prospects WHERE status IN ('active', 'displaced') "
                    "ORDER BY created_at DESC"
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM killzone_prospects WHERE status = 'active' "
                    "ORDER BY created_at DESC"
                ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            if d.get("prospect_json") and isinstance(d["prospect_json"], str):
                try:
                    d["prospect_json"] = json.loads(d["prospect_json"])
                except (json.JSONDecodeError, TypeError):
                    pass
            if d.get("trigger_result") and isinstance(d["trigger_result"], str):
                try:
                    d["trigger_result"] = json.loads(d["trigger_result"])
                except (json.JSONDecodeError, TypeError):
                    pass
            result.append(d)
        return result

    def get_displaced_prospects(self) -> list:
        """Return prospects in 'displaced' state (waiting for retracement)."""
        return [p for p in self.get_active_prospects(include_displaced=True)
                if p.get("status") == "displaced"]

    def mark_prospect_triggered(self, prospect_id: str, setup_id: str):
        """Mark a prospect as triggered and link it to the created scanner setup.

        Args:
            prospect_id: The killzone_prospects row ID.
            setup_id: The scanner_setups row ID that was created from this prospect.
        """
        now = datetime.utcnow().isoformat()
        with self._conn() as conn:
            conn.execute(
                "UPDATE killzone_prospects SET status='triggered', "
                "trigger_result=?, resolved_at=? WHERE id=?",
                (setup_id, now, prospect_id))

    def resolve_prospect(self, prospect_id: str, status: str, trigger_result: str = None):
        """Mark a prospect as triggered or expired."""
        now = datetime.utcnow().isoformat()
        with self._conn() as conn:
            conn.execute(
                """UPDATE killzone_prospects
                   SET status = ?, resolved_at = ?, trigger_result = ?
                   WHERE id = ?""",
                (status, now, trigger_result, prospect_id))

    def update_prospect_json(self, prospect_id: str, prospect_json: dict):
        """Update the prospect JSON (e.g. per-setup status changes) without changing row status."""
        with self._conn() as conn:
            conn.execute(
                "UPDATE killzone_prospects SET prospect_json = ? WHERE id = ?",
                (json.dumps(prospect_json), prospect_id))

    def expire_killzone_prospects(self, killzone: str):
        """Expire all active prospects for a killzone that's ending."""
        now = datetime.utcnow().isoformat()
        with self._conn() as conn:
            conn.execute(
                """UPDATE killzone_prospects
                   SET status = 'expired', resolved_at = ?
                   WHERE killzone = ? AND status = 'active'""",
                (now, killzone))

    # ── Zone cooldown methods ────────────────────────────────────────

    def _make_zone_key(self, direction: str, timeframe: str,
                       zone_high: float, zone_low: float) -> str:
        """Stable key for a zone. e.g. 'short|1h|4539.0|4505.0'."""
        return f"{direction}|{timeframe}|{round(zone_high, 1)}|{round(zone_low, 1)}"

    def block_zone_for_killzone(self, zone_key: str, killzone: str,
                                prospect_id: str = None, setup_id: str = None):
        """Block a zone for the rest of today's killzone after a stop-out."""
        record_id = str(uuid.uuid4())[:8]
        today = datetime.utcnow().strftime("%Y-%m-%d")
        now = datetime.utcnow().isoformat()
        with self._conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO zone_cooldowns
                (id, zone_key, killzone, date, blocked_at, prospect_id, setup_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (record_id, zone_key, killzone, today, now, prospect_id, setup_id))

    def is_zone_blocked(self, zone_key: str, killzone: str) -> bool:
        """Return True if this zone was stopped out in today's killzone."""
        today = datetime.utcnow().strftime("%Y-%m-%d")
        try:
            with self._conn() as conn:
                row = conn.execute(
                    "SELECT 1 FROM zone_cooldowns "
                    "WHERE zone_key = ? AND killzone = ? AND date = ?",
                    (zone_key, killzone, today)).fetchone()
            return row is not None
        except Exception:
            return False

    def expire_zone_cooldowns(self, hours_back: int = 20):
        """Delete cooldown rows older than hours_back hours."""
        cutoff = (datetime.utcnow() - timedelta(hours=hours_back)).isoformat()
        with self._conn() as conn:
            conn.execute("DELETE FROM zone_cooldowns WHERE blocked_at < ?", (cutoff,))

    # ── Notification lifecycle methods ──────────────────────────────

    def record_lifecycle_notification(self, thesis_id: str, timeframe: str,
                                       stage: int, stage_name: str,
                                       telegram_msg_id: str = None,
                                       setup_id: str = None,
                                       payload_json: dict = None) -> str:
        """Record a lifecycle notification for dedup + threading."""
        notif_id = str(uuid.uuid4())[:8]
        now = datetime.utcnow().isoformat()
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO notification_lifecycle
                (id, thesis_id, timeframe, stage, stage_name, sent_at,
                 telegram_msg_id, setup_id, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (notif_id, thesis_id, timeframe, stage, stage_name, now,
                  telegram_msg_id, setup_id,
                  json.dumps(payload_json) if payload_json else None))
        return notif_id

    def lifecycle_already_sent(self, thesis_id: str, stage: int) -> bool:
        """Check if a notification has already been sent for this thesis + stage."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM notification_lifecycle WHERE thesis_id = ? AND stage = ?",
                (thesis_id, stage)).fetchone()
        return row is not None

    def lifecycle_max_stage_sent(self, thesis_id: str) -> int:
        """Return the highest stage already sent for this thesis, or 0."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT MAX(stage) FROM notification_lifecycle WHERE thesis_id = ?",
                (thesis_id,)).fetchone()
        return row[0] if row and row[0] is not None else 0

    def get_lifecycle_thread_msg_id(self, thesis_id: str) -> str | None:
        """Get the first Telegram message ID for this thesis (for reply threading)."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT telegram_msg_id FROM notification_lifecycle "
                "WHERE thesis_id = ? AND telegram_msg_id IS NOT NULL "
                "ORDER BY stage ASC LIMIT 1",
                (thesis_id,)).fetchone()
        return row[0] if row else None

    def get_lifecycle_history(self, thesis_id: str) -> list:
        """Return all lifecycle notifications for a thesis, ordered by stage."""
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM notification_lifecycle WHERE thesis_id = ? ORDER BY stage",
                (thesis_id,)).fetchall()
        return [dict(r) for r in rows]

    def get_recent_lifecycle(self, limit: int = 50, timeframe: str = None) -> list:
        """Return recent lifecycle notifications across all theses."""
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            if timeframe:
                rows = conn.execute(
                    "SELECT * FROM notification_lifecycle WHERE timeframe = ? ORDER BY sent_at DESC LIMIT ?",
                    (timeframe, limit)).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM notification_lifecycle ORDER BY sent_at DESC LIMIT ?",
                    (limit,)).fetchall()
        results = [dict(r) for r in rows]
        for r in results:
            if r.get("payload_json") and isinstance(r["payload_json"], str):
                try:
                    r["payload_json"] = json.loads(r["payload_json"])
                except (json.JSONDecodeError, TypeError):
                    pass
        return results

    def clear(self):
        """Delete all setups."""
        with self._conn() as conn:
            conn.execute("DELETE FROM scanner_setups")

    def _row_to_dict(self, row) -> dict:
        d = dict(row)
        for key in ("rr_ratios", "analysis_json", "calibration_json"):
            if d.get(key) and isinstance(d[key], str):
                try:
                    d[key] = json.loads(d[key])
                except (json.JSONDecodeError, TypeError):
                    pass
        d["auto_resolved"] = bool(d.get("auto_resolved", 0))
        return d
