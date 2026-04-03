"""Priority 5: Haiku False Negative Detection.

Tracks what price does after Haiku screens out a timeframe. Classifies
rejections as true negatives vs false negatives using a two-gate system:

    Gate 1 (price gate): Did price move more than normal noise?
        Uses P90 baseline thresholds from 6 months of gold data, per forward
        window. A move must exceed the 90th percentile of random windows to
        even be a candidate FN — otherwise it's just gold being gold.

    Gate 2 (structure gate): Were ICT structural elements present?
        A big move on news with no OB/FVG/sweep is not a missed setup.
        The FN score is weighted by confluence_count. Only rejections where
        structural elements were present AND price moved abnormally are
        classified as false negatives.

Feedback loop:
    - Adjustments (bypass/loosen) expire after ADJUSTMENT_TTL_DAYS and must
      be re-earned from fresh data. No one-way ratchets.
    - Adjustment lookups are cached with a TTL to avoid DB queries on every
      scan cycle.

Storage: 'haiku_rejections' table in scanner.db (same SQLite file).
"""

import logging
import os
import sqlite3
import time
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Forward window sizes (number of candles to check after rejection)
_FORWARD_BARS = {
    "15min": 8,    # 2 hours
    "1h": 4,       # 4 hours
    "4h": 3,       # 12 hours
    "1day": 2,     # 2 days
}

# How long to wait before resolving (must have enough forward data)
_RESOLVE_DELAY = {
    "15min": timedelta(hours=2, minutes=15),
    "1h": timedelta(hours=4, minutes=15),
    "4h": timedelta(hours=12, minutes=15),
    "1day": timedelta(days=2, hours=1),
}

# ── Baseline-calibrated thresholds ──
# P90 of max(up_move, down_move) / ATR in random windows (from 6 months
# of XAU/USD 1H data, March 2026). A move must exceed this to even be
# considered a candidate FN — otherwise it's normal gold volatility.
_BASELINE_P90_ATR = {
    "15min": 2.16,   # 2h forward on 1H candles ≈ P90
    "1h": 3.23,      # 4h forward P90
    "4h": 5.73,      # 12h forward P90
    "1day": 7.0,     # 2-day forward (estimated, conservative)
}

# A strong FN requires BOTH exceeding baseline AND having structure
# Strong FN = baseline_exceeded AND confluence >= 2
FN_THRESHOLD_ATR = 3.23     # default (1h P90), overridden per-TF from _BASELINE_P90_ATR
STRONG_FN_THRESHOLD_ATR = 5.0  # obvious miss (well above any noise floor)

# Structural weight: FN requires confluence_count >= this value
FN_MIN_CONFLUENCE = 1       # At least 1 ICT element must be present

# Minimum rejections before a segment's FN rate is actionable
MIN_SEGMENT_SAMPLES = 15

# FN rate thresholds for screening adjustments (higher than before —
# must be clearly above random expectation to be actionable)
FN_RATE_LOOSEN = 0.35    # >= 35% structure-weighted FN rate → loosen
FN_RATE_BYPASS = 0.50    # >= 50% structure-weighted FN rate → bypass

# Adjustment freshness — bypass/loosen decisions expire after this many
# days and must be re-earned from recent data. Prevents one-way ratchet.
ADJUSTMENT_TTL_DAYS = 14
ADJUSTMENT_DATA_WINDOW_DAYS = 30  # Only use last 30 days of data for adjustments

# Cache TTL for adjustment lookups (seconds)
_ADJUSTMENT_CACHE_TTL = 300  # 5 minutes


class HaikuFNTracker:
    """Tracks and analyses Haiku false negatives."""

    def __init__(self, db_path: str = None):
        if db_path is None:
            from ml.config import get_config
            db_path = get_config().get("db_path")
        self.db_path = db_path or os.path.join(
            os.path.dirname(__file__), "models", "scanner.db"
        )
        self._init_table()
        # Cached adjustments to avoid DB queries every scan cycle
        self._adj_cache: list[dict] | None = None
        self._adj_cache_time: float = 0

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def _init_table(self):
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS haiku_rejections (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at      TEXT NOT NULL,
                    resolved_at     TEXT,
                    timeframe       TEXT NOT NULL,
                    killzone        TEXT,
                    last_close      REAL NOT NULL,
                    atr             REAL,
                    reason          TEXT,
                    reason_category TEXT,
                    structural_score REAL DEFAULT 0,
                    confluence_count INTEGER DEFAULT 0,

                    -- Resolution fields (filled by resolve_rejections)
                    status          TEXT NOT NULL DEFAULT 'pending',
                    forward_high    REAL,
                    forward_low     REAL,
                    max_up_move     REAL,
                    max_down_move   REAL,
                    mfe_atr         REAL,
                    is_false_negative INTEGER DEFAULT 0,
                    is_strong_fn    INTEGER DEFAULT 0,
                    move_direction  TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_haiku_rej_status
                ON haiku_rejections(status)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_haiku_rej_tf_kz
                ON haiku_rejections(timeframe, killzone)
            """)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log_rejection(
        self,
        timeframe: str,
        killzone: str,
        last_close: float,
        atr: float,
        reason: str,
        structural_score: float = 0,
        confluence_count: int = 0,
    ) -> int:
        """Log a Haiku rejection for later false-negative analysis.

        Returns:
            Row ID of the logged rejection.
        """
        reason_cat = self._categorize_reason(reason)

        with self._conn() as conn:
            cursor = conn.execute("""
                INSERT INTO haiku_rejections
                    (created_at, timeframe, killzone, last_close, atr,
                     reason, reason_category, structural_score, confluence_count,
                     status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')
            """, (
                datetime.utcnow().isoformat(),
                timeframe,
                killzone or "unknown",
                last_close,
                atr or 0,
                reason or "",
                reason_cat,
                structural_score,
                confluence_count,
            ))
            row_id = cursor.lastrowid

        logger.debug(
            "Logged Haiku rejection #%d: %s %s close=%.2f atr=%.2f "
            "confluence=%d reason=%s",
            row_id, timeframe, killzone, last_close, atr or 0,
            confluence_count, reason_cat,
        )
        return row_id

    # ------------------------------------------------------------------
    # Resolution (two-gate FN classification)
    # ------------------------------------------------------------------

    def resolve_rejections(self, candles_5min: list) -> dict:
        """Check pending rejections against forward price action.

        Two-gate FN classification:
          Gate 1: mfe_atr >= baseline P90 for this timeframe's forward window
          Gate 2: confluence_count >= FN_MIN_CONFLUENCE (structural elements present)

        Both gates must pass for is_false_negative = True.
        """
        if not candles_5min:
            return {"checked": 0, "resolved": 0, "false_negatives": 0}

        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            pending = conn.execute("""
                SELECT * FROM haiku_rejections
                WHERE status = 'pending'
                ORDER BY created_at ASC
            """).fetchall()

        if not pending:
            return {"checked": 0, "resolved": 0, "false_negatives": 0}

        now = datetime.utcnow()
        resolved = 0
        false_negatives = 0
        strong_fns = 0

        for rej in pending:
            tf = rej["timeframe"]
            delay = _RESOLVE_DELAY.get(tf, timedelta(hours=4))
            created = datetime.fromisoformat(rej["created_at"])

            if now - created < delay:
                continue

            forward = self._get_forward_candles(candles_5min, created, tf)
            if len(forward) < 2:
                continue

            last_close = rej["last_close"]
            atr = rej["atr"] or 1.0

            highs = [c["high"] for c in forward]
            lows = [c["low"] for c in forward]
            forward_high = max(highs)
            forward_low = min(lows)

            max_up_move = forward_high - last_close
            max_down_move = last_close - forward_low
            mfe = max(max_up_move, max_down_move)
            mfe_atr = round(mfe / atr, 4) if atr > 0 else 0

            # ── Gate 1: Price exceeded baseline noise? ──
            baseline = _BASELINE_P90_ATR.get(tf, 3.23)
            price_gate = mfe_atr >= baseline

            # ── Gate 2: Structural elements were present? ──
            confluence = rej["confluence_count"] or 0
            structure_gate = confluence >= FN_MIN_CONFLUENCE

            # Both gates must pass
            is_fn = 1 if (price_gate and structure_gate) else 0
            is_strong = 1 if (
                is_fn and mfe_atr >= STRONG_FN_THRESHOLD_ATR
                and confluence >= 2
            ) else 0

            if max_up_move > max_down_move:
                move_dir = "long"
            elif max_down_move > max_up_move:
                move_dir = "short"
            else:
                move_dir = "neutral"

            with self._conn() as conn:
                conn.execute("""
                    UPDATE haiku_rejections SET
                        status = 'resolved',
                        resolved_at = ?,
                        forward_high = ?,
                        forward_low = ?,
                        max_up_move = ?,
                        max_down_move = ?,
                        mfe_atr = ?,
                        is_false_negative = ?,
                        is_strong_fn = ?,
                        move_direction = ?
                    WHERE id = ?
                """, (
                    now.isoformat(),
                    round(forward_high, 2),
                    round(forward_low, 2),
                    round(max_up_move, 2),
                    round(max_down_move, 2),
                    mfe_atr,
                    is_fn,
                    is_strong,
                    move_dir,
                    rej["id"],
                ))

            resolved += 1
            if is_fn:
                false_negatives += 1
                if is_strong:
                    strong_fns += 1
                logger.info(
                    "Haiku FN: %s %s — %.1f ATR %s (baseline=%.1f, "
                    "confluence=%d, reason=%s)",
                    tf, rej["killzone"], mfe_atr, move_dir, baseline,
                    confluence, rej["reason_category"],
                )

        # Invalidate adjustment cache after resolutions
        if resolved:
            self._adj_cache = None
            logger.info(
                "Haiku FN resolution: %d resolved, %d FN (%d strong)",
                resolved, false_negatives, strong_fns,
            )

        return {
            "checked": len(pending),
            "resolved": resolved,
            "false_negatives": false_negatives,
            "strong_fns": strong_fns,
        }

    def _get_forward_candles(
        self,
        candles_5min: list,
        rejection_time: datetime,
        timeframe: str,
    ) -> list:
        """Extract candles in the forward window after a rejection timestamp."""
        bars_needed = _FORWARD_BARS.get(timeframe, 4)
        tf_minutes = {"15min": 15, "1h": 60, "4h": 240, "1day": 1440}
        minutes_forward = bars_needed * tf_minutes.get(timeframe, 60)
        end_time = rejection_time + timedelta(minutes=minutes_forward)

        forward = []
        for c in candles_5min:
            try:
                c_time = datetime.fromisoformat(
                    c.get("datetime", c.get("time", ""))
                )
            except (ValueError, TypeError):
                continue
            if rejection_time < c_time <= end_time:
                forward.append(c)
        return forward

    # ------------------------------------------------------------------
    # Expiry
    # ------------------------------------------------------------------

    def expire_stale(self, max_age_hours: int = 72) -> int:
        """Expire pending rejections older than max_age_hours."""
        cutoff = (datetime.utcnow() - timedelta(hours=max_age_hours)).isoformat()
        with self._conn() as conn:
            cursor = conn.execute("""
                UPDATE haiku_rejections
                SET status = 'expired'
                WHERE status = 'pending' AND created_at < ?
            """, (cutoff,))
            count = cursor.rowcount
        if count:
            logger.info("Expired %d stale Haiku rejections (>%dh old)",
                        count, max_age_hours)
        return count

    # ------------------------------------------------------------------
    # Reporting & Analysis
    # ------------------------------------------------------------------

    def get_fn_report(self) -> dict:
        """Aggregate false negative rates by segment.

        Only uses data from the last ADJUSTMENT_DATA_WINDOW_DAYS to keep
        the signal fresh (gold's volatility regime changes over time).
        """
        cutoff = (
            datetime.utcnow() - timedelta(days=ADJUSTMENT_DATA_WINDOW_DAYS)
        ).isoformat()

        with self._conn() as conn:
            conn.row_factory = sqlite3.Row

            total = conn.execute("""
                SELECT COUNT(*) as n,
                       SUM(is_false_negative) as fn,
                       SUM(is_strong_fn) as strong_fn,
                       AVG(mfe_atr) as avg_mfe
                FROM haiku_rejections
                WHERE status = 'resolved' AND created_at >= ?
            """, (cutoff,)).fetchone()

            overall = {
                "total_resolved": total["n"],
                "false_negatives": total["fn"] or 0,
                "strong_fns": total["strong_fn"] or 0,
                "fn_rate": round((total["fn"] or 0) / max(total["n"], 1), 4),
                "avg_mfe_atr": round(total["avg_mfe"] or 0, 4),
                "data_window_days": ADJUSTMENT_DATA_WINDOW_DAYS,
            }

            by_tf = {}
            rows = conn.execute("""
                SELECT timeframe,
                       COUNT(*) as n,
                       SUM(is_false_negative) as fn,
                       SUM(is_strong_fn) as strong_fn,
                       AVG(mfe_atr) as avg_mfe
                FROM haiku_rejections
                WHERE status = 'resolved' AND created_at >= ?
                GROUP BY timeframe
            """, (cutoff,)).fetchall()
            for r in rows:
                by_tf[r["timeframe"]] = {
                    "total": r["n"],
                    "fn": r["fn"] or 0,
                    "fn_rate": round((r["fn"] or 0) / max(r["n"], 1), 4),
                    "strong_fn": r["strong_fn"] or 0,
                    "avg_mfe": round(r["avg_mfe"] or 0, 4),
                    "baseline_p90": _BASELINE_P90_ATR.get(r["timeframe"], 3.23),
                }

            by_kz = {}
            rows = conn.execute("""
                SELECT killzone,
                       COUNT(*) as n,
                       SUM(is_false_negative) as fn,
                       AVG(mfe_atr) as avg_mfe
                FROM haiku_rejections
                WHERE status = 'resolved' AND created_at >= ?
                GROUP BY killzone
            """, (cutoff,)).fetchall()
            for r in rows:
                by_kz[r["killzone"]] = {
                    "total": r["n"],
                    "fn": r["fn"] or 0,
                    "fn_rate": round((r["fn"] or 0) / max(r["n"], 1), 4),
                    "avg_mfe": round(r["avg_mfe"] or 0, 4),
                }

            segments = {}
            rows = conn.execute("""
                SELECT timeframe, killzone,
                       COUNT(*) as n,
                       SUM(is_false_negative) as fn,
                       SUM(is_strong_fn) as strong_fn,
                       AVG(mfe_atr) as avg_mfe
                FROM haiku_rejections
                WHERE status = 'resolved' AND created_at >= ?
                GROUP BY timeframe, killzone
                HAVING COUNT(*) >= ?
            """, (cutoff, MIN_SEGMENT_SAMPLES)).fetchall()
            for r in rows:
                key = f"{r['timeframe']}|{r['killzone']}"
                fn_rate = round((r["fn"] or 0) / max(r["n"], 1), 4)
                segments[key] = {
                    "total": r["n"],
                    "fn": r["fn"] or 0,
                    "fn_rate": fn_rate,
                    "strong_fn": r["strong_fn"] or 0,
                    "avg_mfe": round(r["avg_mfe"] or 0, 4),
                    "action": self._segment_action(fn_rate, r["n"]),
                }

            by_reason = {}
            rows = conn.execute("""
                SELECT reason_category,
                       COUNT(*) as n,
                       SUM(is_false_negative) as fn,
                       AVG(mfe_atr) as avg_mfe
                FROM haiku_rejections
                WHERE status = 'resolved' AND created_at >= ?
                GROUP BY reason_category
                HAVING COUNT(*) >= 5
            """, (cutoff,)).fetchall()
            for r in rows:
                by_reason[r["reason_category"]] = {
                    "total": r["n"],
                    "fn": r["fn"] or 0,
                    "fn_rate": round((r["fn"] or 0) / max(r["n"], 1), 4),
                    "avg_mfe": round(r["avg_mfe"] or 0, 4),
                }

            struct_analysis = {}
            for label, min_s in [("low", 0), ("medium", 2), ("high", 3)]:
                max_s = {"low": 2, "medium": 3, "high": 999}[label]
                row = conn.execute("""
                    SELECT COUNT(*) as n,
                           SUM(is_false_negative) as fn,
                           AVG(mfe_atr) as avg_mfe
                    FROM haiku_rejections
                    WHERE status = 'resolved' AND created_at >= ?
                      AND structural_score >= ? AND structural_score < ?
                """, (cutoff, min_s, max_s)).fetchone()
                if row["n"] > 0:
                    struct_analysis[label] = {
                        "total": row["n"],
                        "fn": row["fn"] or 0,
                        "fn_rate": round((row["fn"] or 0) / max(row["n"], 1), 4),
                        "avg_mfe": round(row["avg_mfe"] or 0, 4),
                    }

        return {
            "overall": overall,
            "by_timeframe": by_tf,
            "by_killzone": by_kz,
            "segments": segments,
            "by_reason": by_reason,
            "by_structural_score": struct_analysis,
            "baseline_thresholds": dict(_BASELINE_P90_ATR),
        }

    def get_screening_adjustments(self) -> list[dict]:
        """Return per-segment screening adjustments based on FN data.

        Uses only recent data (ADJUSTMENT_DATA_WINDOW_DAYS) and decisions
        must be re-earned — no permanent bypasses.
        """
        report = self.get_fn_report()
        adjustments = []

        for key, seg in report.get("segments", {}).items():
            action = seg.get("action", "normal")
            if action == "normal":
                continue

            tf, kz = key.split("|", 1)
            adjustments.append({
                "timeframe": tf,
                "killzone": kz,
                "action": action,
                "fn_rate": seg["fn_rate"],
                "samples": seg["total"],
                "strong_fn": seg.get("strong_fn", 0),
                "reason": (
                    f"FN rate {seg['fn_rate']:.0%} ({seg['total']} samples, "
                    f"last {ADJUSTMENT_DATA_WINDOW_DAYS}d) — "
                    f"Haiku too conservative for {tf} {kz}"
                ),
            })

        adjustments.sort(key=lambda a: a["fn_rate"], reverse=True)
        return adjustments

    def _get_cached_adjustments(self) -> list[dict]:
        """Return cached adjustments, refreshing if TTL expired."""
        now = time.time()
        if (self._adj_cache is not None
                and now - self._adj_cache_time < _ADJUSTMENT_CACHE_TTL):
            return self._adj_cache
        self._adj_cache = self.get_screening_adjustments()
        self._adj_cache_time = now
        return self._adj_cache

    def should_bypass_haiku(self, timeframe: str, killzone: str) -> bool:
        """Quick check: should Haiku screening be bypassed for this segment?

        Uses cached adjustments (5-min TTL) to avoid DB queries every scan.
        """
        for adj in self._get_cached_adjustments():
            if adj["timeframe"] == timeframe and adj["killzone"] == killzone:
                return adj["action"] == "bypass"
        return False

    def should_loosen_haiku(self, timeframe: str, killzone: str) -> bool:
        """Check if Haiku screening should be loosened for this segment."""
        for adj in self._get_cached_adjustments():
            if adj["timeframe"] == timeframe and adj["killzone"] == killzone:
                return adj["action"] in ("loosen", "bypass")
        return False

    # ------------------------------------------------------------------
    # Stats (for API/UI)
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Quick stats for the scanner status endpoint."""
        with self._conn() as conn:
            row = conn.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN status='pending' THEN 1 ELSE 0 END) as pending,
                    SUM(CASE WHEN status='resolved' THEN 1 ELSE 0 END) as resolved,
                    SUM(CASE WHEN is_false_negative=1 THEN 1 ELSE 0 END) as fn,
                    SUM(CASE WHEN is_strong_fn=1 THEN 1 ELSE 0 END) as strong_fn
                FROM haiku_rejections
            """).fetchone()
        total = row[0] or 0
        resolved = row[2] or 0
        fn = row[3] or 0
        return {
            "total_rejections": total,
            "pending": row[1] or 0,
            "resolved": resolved,
            "false_negatives": fn,
            "strong_fns": row[4] or 0,
            "fn_rate": round(fn / max(resolved, 1), 4),
        }

    def get_recent(self, limit: int = 20) -> list[dict]:
        """Get recent resolved rejections for display."""
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT * FROM haiku_rejections
                WHERE status = 'resolved'
                ORDER BY created_at DESC
                LIMIT ?
            """, (limit,)).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _categorize_reason(reason: str) -> str:
        """Bin Haiku's free-text reason into a category for aggregation."""
        if not reason:
            return "unknown"
        reason_lower = reason.lower()

        if any(w in reason_lower for w in ["rang", "consolidat", "sideways", "chop"]):
            return "ranging"
        if any(w in reason_lower for w in ["no structure", "no clear", "unclear"]):
            return "no_structure"
        if any(w in reason_lower for w in ["no sweep", "no liquidity"]):
            return "no_sweep"
        if any(w in reason_lower for w in ["no order block", "no ob", "no fvg"]):
            return "no_zones"
        if any(w in reason_lower for w in ["overextend", "exhaust", "climax"]):
            return "overextended"
        if any(w in reason_lower for w in ["retrac", "pullback"]):
            return "retracement"
        if any(w in reason_lower for w in ["weak", "low probability", "insufficient"]):
            return "weak_setup"
        return "other"

    @staticmethod
    def _segment_action(fn_rate: float, samples: int) -> str:
        """Determine what screening action a segment warrants."""
        if samples < MIN_SEGMENT_SAMPLES:
            return "normal"
        if fn_rate >= FN_RATE_BYPASS:
            return "bypass"
        if fn_rate >= FN_RATE_LOOSEN:
            return "loosen"
        return "normal"
