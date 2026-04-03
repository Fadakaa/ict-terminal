"""Narrative State Engine — persistent per-timeframe thesis tracking for Claude.

Every scan cycle, Claude returns a structured narrative_state alongside its
analysis JSON.  This module persists those states, detects continuation vs
revision, scores predictions, runs anti-anchoring safeguards, and provides
the previous thesis back into the next scan's prompt.

Components:
  NarrativeStore  — SQLite persistence + continuation/revision detection
  check_invalidation()  — pre-prompt price vs invalidation check (zero API cost)
  THESIS_MAX_AGE  — per-timeframe stale thresholds
"""
import json
import os
import sqlite3
import uuid
from datetime import datetime, timedelta


# ── Per-timeframe stale thesis thresholds (minutes) ──────────────────
THESIS_MAX_AGE = {
    "15min": 120,    # 2 hours
    "1h": 480,       # 8 hours
    "4h": 1440,      # 24 hours
    "1day": 4320,    # 72 hours
}

# Confidence decay per scan when expected_next_move doesn't happen
CONFIDENCE_DECAY_RATE = 0.15


class NarrativeStore:
    """Per-timeframe narrative state persistence."""

    def __init__(self, db_path: str = None):
        if db_path is None:
            from ml.config import get_config
            db_path = get_config().get("db_path")
        self.db_path = db_path or os.path.join(
            os.path.dirname(__file__), "models", "scanner.db"
        )
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._ensure_table()

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def _ensure_table(self):
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS narrative_states (
                    id                  TEXT PRIMARY KEY,
                    timeframe           TEXT NOT NULL,
                    created_at          TEXT NOT NULL,
                    thesis              TEXT,
                    p3_phase            TEXT,
                    p3_progress         TEXT,
                    directional_bias    TEXT,
                    bias_confidence     REAL,
                    key_levels_json     TEXT,
                    expected_next_move  TEXT,
                    invalidation_json   TEXT,
                    watching_for_json   TEXT,
                    scan_count          INTEGER DEFAULT 1,
                    thesis_age_minutes  INTEGER DEFAULT 0,
                    last_revision       TEXT,
                    is_revision         INTEGER DEFAULT 0,
                    prediction_hit      INTEGER,
                    invalidation_hit    INTEGER,
                    status              TEXT DEFAULT 'active',
                    scan_id             TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_narrative_tf_status
                ON narrative_states(timeframe, status)
            """)
            # Phase 6: killzone handoff summary column
            try:
                conn.execute("ALTER TABLE narrative_states ADD COLUMN killzone_summary TEXT")
            except sqlite3.OperationalError:
                pass  # Column already exists
            # Phase C: displacement zone context for cross-cycle awareness
            try:
                conn.execute(
                    "ALTER TABLE narrative_states ADD COLUMN displacement_confirmed_zones TEXT")
            except sqlite3.OperationalError:
                pass  # Column already exists

    def get_current(self, timeframe: str) -> dict | None:
        """Get the latest active narrative state for a timeframe."""
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """SELECT * FROM narrative_states
                   WHERE timeframe = ? AND status = 'active'
                   ORDER BY created_at DESC LIMIT 1""",
                (timeframe,),
            ).fetchone()
        if not row:
            return None
        return self._row_to_dict(row)

    def update_killzone_summary(self, state_id: str, summary: str):
        """Store a killzone handoff summary on an existing narrative state.

        Called during killzone transitions to pass session context forward.
        """
        with self._conn() as conn:
            conn.execute(
                "UPDATE narrative_states SET killzone_summary = ? WHERE id = ?",
                (summary, state_id))

    def update_displacement_zones(self, timeframe: str, zone: dict):
        """Append a displacement-confirmed zone to the current narrative state.

        Called by Phase 2 prospect monitor when a retrace is detected, so the
        next regular scan cycle knows that zone has been consumed.
        """
        current = self.get_current(timeframe)
        if not current:
            return
        existing = json.loads(current.get("displacement_confirmed_zones") or "[]")
        existing.append({**zone, "recorded_at": datetime.utcnow().isoformat()})
        with self._conn() as conn:
            conn.execute(
                "UPDATE narrative_states SET displacement_confirmed_zones = ? WHERE id = ?",
                (json.dumps(existing), current["id"]))

    def save(self, timeframe: str, state: dict, scan_id: str = None):
        """Save a new narrative state.

        Compares with current state to determine continuation vs revision.
        Increments scan_count on continuation, resets to 1 on revision.
        Scores the previous state's predictions if applicable.
        """
        # Validate: mandatory invalidation field
        inv = state.get("invalidation") or {}
        if not inv.get("price_level") or not inv.get("direction"):
            # No concrete invalidation → don't persist (Safeguard 1)
            return None

        prev = self.get_current(timeframe)

        is_revision = False
        scan_count = 1
        thesis_age_minutes = 0

        if prev:
            # Determine continuation vs revision
            same_bias = (state.get("directional_bias") == prev.get("directional_bias"))
            same_phase = (state.get("p3_phase") == prev.get("p3_phase"))

            # Neutral → directional = revision (new conviction forming)
            prev_neutral = prev.get("directional_bias") == "neutral"
            now_directional = state.get("directional_bias") in ("bullish", "bearish")

            if (same_bias and same_phase) and not (prev_neutral and now_directional):
                # CONTINUATION — same thesis evolving
                is_revision = False
                scan_count = prev.get("scan_count", 1) + 1
                # Compute age from original thesis creation
                try:
                    first_created = prev.get("created_at", "")
                    if first_created:
                        created_dt = datetime.fromisoformat(first_created)
                        age_delta = datetime.utcnow() - created_dt
                        # On continuation, thesis_age grows from the ORIGINAL creation
                        # The original creation is: now minus the previous age minus the scan interval
                        prev_age = prev.get("thesis_age_minutes", 0)
                        thesis_age_minutes = prev_age + max(1, int(age_delta.total_seconds() / 60) - prev_age)
                except (ValueError, TypeError):
                    thesis_age_minutes = prev.get("thesis_age_minutes", 0) + 15
            else:
                # REVISION — thesis changed
                is_revision = True
                scan_count = 1
                thesis_age_minutes = 0

            # Score previous thesis predictions
            self._score_predictions(prev, state, is_revision)

            # Mark previous state as superseded
            with self._conn() as conn:
                conn.execute(
                    "UPDATE narrative_states SET status = 'superseded' WHERE id = ?",
                    (prev["id"],))

        # Store new state
        state_id = str(uuid.uuid4())[:8]
        now = datetime.utcnow().isoformat()

        with self._conn() as conn:
            conn.execute("""
                INSERT INTO narrative_states
                (id, timeframe, created_at, thesis, p3_phase, p3_progress,
                 directional_bias, bias_confidence, key_levels_json,
                 expected_next_move, invalidation_json, watching_for_json,
                 scan_count, thesis_age_minutes, last_revision, is_revision,
                 status, scan_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?)
            """, (
                state_id, timeframe, now,
                state.get("thesis"),
                state.get("p3_phase"),
                state.get("p3_progress"),
                state.get("directional_bias"),
                state.get("bias_confidence"),
                json.dumps(state.get("key_levels", [])),
                state.get("expected_next_move"),
                json.dumps(state.get("invalidation", {})),
                json.dumps(state.get("watching_for", [])),
                scan_count,
                thesis_age_minutes,
                state.get("last_revision"),
                int(is_revision),
                scan_id,
            ))

        return state_id

    def get_history(self, timeframe: str, limit: int = 10) -> list:
        """Get recent narrative states for a timeframe, most recent first."""
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT * FROM narrative_states
                   WHERE timeframe = ?
                   ORDER BY created_at DESC LIMIT ?""",
                (timeframe, limit),
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def expire_stale(self, timeframe: str = None):
        """Mark narratives older than per-TF max age as expired.

        If timeframe is None, expires stale across all timeframes.
        """
        now = datetime.utcnow()
        timeframes = [timeframe] if timeframe else list(THESIS_MAX_AGE.keys())

        with self._conn() as conn:
            for tf in timeframes:
                max_age = THESIS_MAX_AGE.get(tf, 480)
                cutoff = (now - timedelta(minutes=max_age)).isoformat()
                conn.execute(
                    """UPDATE narrative_states
                       SET status = 'expired'
                       WHERE timeframe = ? AND status = 'active'
                       AND created_at < ?""",
                    (tf, cutoff))

    def get_revision_rate(self, timeframe: str, window: int = 20) -> float:
        """Percentage of recent scans that were revisions vs continuations.

        High rate = unstable thesis.  Very low = potential anchoring.
        """
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT is_revision FROM narrative_states
                   WHERE timeframe = ?
                   ORDER BY created_at DESC LIMIT ?""",
                (timeframe, window),
            ).fetchall()
        if not rows:
            return 0.0
        revisions = sum(1 for r in rows if r[0])
        return revisions / len(rows)

    def get_accuracy_metrics(self, timeframe: str = None) -> dict:
        """Compute aggregate prediction scoring metrics.

        Returns:
            prediction_accuracy, thesis_stability, revision_rate,
            invalidation_respect, avg_thesis_age, thesis_to_setup_conversion
        """
        where = "WHERE timeframe = ?" if timeframe else ""
        params = (timeframe,) if timeframe else ()

        with self._conn() as conn:
            # Prediction accuracy
            total_scored = conn.execute(
                f"SELECT COUNT(*) FROM narrative_states {where} AND prediction_hit IS NOT NULL"
                if where else
                "SELECT COUNT(*) FROM narrative_states WHERE prediction_hit IS NOT NULL",
                params).fetchone()[0]
            hits = conn.execute(
                f"SELECT COUNT(*) FROM narrative_states {where} AND prediction_hit = 1"
                if where else
                "SELECT COUNT(*) FROM narrative_states WHERE prediction_hit = 1",
                params).fetchone()[0]

            # Thesis stability
            total_states = conn.execute(
                f"SELECT COUNT(*) FROM narrative_states {where}"
                if where else
                "SELECT COUNT(*) FROM narrative_states",
                params).fetchone()[0]
            revisions = conn.execute(
                f"SELECT COUNT(*) FROM narrative_states {where} AND is_revision = 1"
                if where else
                "SELECT COUNT(*) FROM narrative_states WHERE is_revision = 1",
                params).fetchone()[0]

            # Invalidation respect
            inv_triggered = conn.execute(
                f"SELECT COUNT(*) FROM narrative_states {where} AND invalidation_hit = 1"
                if where else
                "SELECT COUNT(*) FROM narrative_states WHERE invalidation_hit = 1",
                params).fetchone()[0]
            # How many of those were followed by a revision?
            # (Approximation: next state for same TF is a revision)

            # Average thesis age
            avg_age = conn.execute(
                f"SELECT AVG(thesis_age_minutes) FROM narrative_states {where}"
                if where else
                "SELECT AVG(thesis_age_minutes) FROM narrative_states",
                params).fetchone()[0] or 0

        continuations = total_states - revisions

        return {
            "prediction_accuracy": round(hits / total_scored, 3) if total_scored else 0,
            "predictions_scored": total_scored,
            "predictions_hit": hits,
            "thesis_stability": round(continuations / total_states, 3) if total_states else 0,
            "total_states": total_states,
            "revisions": revisions,
            "continuations": continuations,
            "revision_rate": round(revisions / total_states, 3) if total_states else 0,
            "invalidations_triggered": inv_triggered,
            "avg_thesis_age_minutes": round(avg_age, 1),
        }

    def apply_confidence_decay(self, timeframe: str) -> float | None:
        """Decay bias_confidence by CONFIDENCE_DECAY_RATE if expected_next_move
        hasn't materialised.  Returns new confidence or None if no active thesis.

        Called when save() determines continuation but predicted move didn't happen.
        """
        current = self.get_current(timeframe)
        if not current:
            return None
        conf = current.get("bias_confidence", 0.5)
        new_conf = max(0.0, conf - CONFIDENCE_DECAY_RATE)
        with self._conn() as conn:
            conn.execute(
                "UPDATE narrative_states SET bias_confidence = ? WHERE id = ?",
                (round(new_conf, 3), current["id"]))
        return new_conf

    # ── Internal helpers ─────────────────────────────────────────────

    def _score_predictions(self, prev: dict, new_state: dict, is_revision: bool):
        """Score the previous state's predictions against what happened."""
        # If thesis was revised, the old thesis's predictions didn't play out
        if is_revision and prev.get("expected_next_move"):
            with self._conn() as conn:
                conn.execute(
                    "UPDATE narrative_states SET prediction_hit = 0 WHERE id = ?",
                    (prev["id"],))
            return

        # On continuation: check if the expected_next_move direction materialised
        # This is a heuristic — proper scoring would use price data
        expected = (prev.get("expected_next_move") or "").lower()
        new_thesis = (new_state.get("thesis") or "").lower()

        if not expected:
            return

        # Simple keyword matching for direction
        bullish_kw = {"bullish", "long", "above", "up", "higher", "rally"}
        bearish_kw = {"bearish", "short", "below", "down", "lower", "sell"}

        expected_bullish = any(w in expected for w in bullish_kw)
        expected_bearish = any(w in expected for w in bearish_kw)

        thesis_bullish = new_state.get("directional_bias") == "bullish"
        thesis_bearish = new_state.get("directional_bias") == "bearish"

        hit = None
        if expected_bullish and thesis_bullish:
            hit = 1
        elif expected_bearish and thesis_bearish:
            hit = 1
        elif expected_bullish and thesis_bearish:
            hit = 0
        elif expected_bearish and thesis_bullish:
            hit = 0
        # If unclear, leave as NULL

        if hit is not None:
            with self._conn() as conn:
                conn.execute(
                    "UPDATE narrative_states SET prediction_hit = ? WHERE id = ?",
                    (hit, prev["id"]))

    def _row_to_dict(self, row) -> dict:
        d = dict(row)
        for key, field in [("key_levels_json", "key_levels"),
                           ("invalidation_json", "invalidation"),
                           ("watching_for_json", "watching_for")]:
            if d.get(key) and isinstance(d[key], str):
                try:
                    d[field] = json.loads(d[key])
                except (json.JSONDecodeError, TypeError):
                    d[field] = [] if "levels" in key or "watching" in key else {}
            elif key in d:
                d[field] = d.pop(key, None)
        return d


def check_invalidation(state: dict, current_price: float,
                       recent_candles: list) -> str:
    """Check if narrative invalidation condition has been met.

    Returns:
        'TRIGGERED' — price closed beyond invalidation level
        'APPROACHING' — price within 0.5 ATR of invalidation
        'CLEAR' — no threat to thesis
    """
    inv = state.get("invalidation", {})
    level = inv.get("price_level")
    direction = inv.get("direction")  # 'above' or 'below'

    if not level or not direction:
        return "CLEAR"

    # Check candle CLOSES, not just wicks (avoid false triggers)
    last_close = current_price
    if recent_candles:
        last_close = recent_candles[-1].get("close", current_price)

    if direction == "above" and last_close > level:
        # Confirm with at least 1 close above
        closes_above = sum(1 for c in recent_candles[-3:]
                          if c.get("close", 0) > level) if recent_candles else 0
        if closes_above >= 1:
            return "TRIGGERED"
    elif direction == "below" and last_close < level:
        closes_below = sum(1 for c in recent_candles[-3:]
                          if c.get("close", 0) < level) if recent_candles else 0
        if closes_below >= 1:
            return "TRIGGERED"

    # Approaching check — within 0.5 ATR
    try:
        from ml.features import compute_atr
        atr = compute_atr(recent_candles) if recent_candles and len(recent_candles) >= 14 else 1.0
    except ImportError:
        atr = 1.0
    distance = abs(current_price - level)
    if distance < 0.5 * atr:
        return "APPROACHING"

    return "CLEAR"
