"""System evolution snapshot recorder.

Periodically captures a snapshot of the entire ML brain state — narrative
weights, Bayesian beliefs, bandit arms, setup profile stats, thesis accuracy,
loss type breakdown — and stores it as a time series in scanner.db.

This enables:
  - Trend detection (is a field weight rising or falling?)
  - Cross-system diagnostics (Bayesian plateaued despite weight improvement?)
  - Meta-learning prompt injection (tell Sonnet which signals are improving)
  - Weekly progress reports

The snapshot is intentionally cheap: it reads from existing stores, computes
no new ML, and writes a single row.
"""
import json
import logging
import os
import sqlite3
from datetime import datetime, timezone, timedelta

from ml.config import get_config

logger = logging.getLogger(__name__)

_DEFAULT_DB = os.path.join(os.path.dirname(__file__), "models", "scanner.db")

# Minimum seconds between snapshots (avoid spamming on rapid resolutions)
_MIN_INTERVAL = 3600  # 1 hour


class SystemSnapshotRecorder:
    """Records and queries point-in-time snapshots of the ML brain state."""

    def __init__(self, db_path: str = None, config: dict = None):
        self.db_path = db_path or _DEFAULT_DB
        self.cfg = config or get_config()
        self._ensure_table()

    def _conn(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        return sqlite3.connect(self.db_path)

    def _ensure_table(self):
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS system_snapshots (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp       TEXT NOT NULL,
                    snapshot_json   TEXT NOT NULL,
                    trigger         TEXT DEFAULT 'auto'
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_snapshot_ts
                ON system_snapshots(timestamp)
            """)

    # ── Core: take a snapshot ─────────────────────────────────────

    def take_snapshot(self, trigger: str = "auto") -> dict:
        """Capture current state of all ML subsystems into one row.

        Args:
            trigger: what caused this snapshot ('auto', 'trade_resolved',
                     'daily', 'manual', 'backfill')

        Returns: the snapshot dict that was stored.
        """
        snap = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trigger": trigger,
        }

        # 1. Narrative field weights
        try:
            from ml.claude_bridge import ClaudeAnalysisBridge
            bridge = ClaudeAnalysisBridge()
            raw_weights = bridge._narrative_weights
            snap["narrative_weights"] = {}
            for field, val in raw_weights.items():
                if isinstance(val, dict):
                    snap["narrative_weights"][field] = {
                        "weight": val.get("weight", 0.5),
                        "total": val.get("total", 0),
                    }
                else:
                    snap["narrative_weights"][field] = {
                        "weight": val, "total": 0
                    }
        except Exception as e:
            logger.debug("Snapshot: narrative weights unavailable: %s", e)
            snap["narrative_weights"] = {}

        # 2. Bayesian beliefs
        try:
            from ml.database import TradeDatabase
            from ml.bayesian import get_beliefs
            db = TradeDatabase()
            state = db.get_bayesian_state()
            if state:
                beliefs = get_beliefs(state)
                snap["bayesian"] = {
                    "win_rate_mean": beliefs["win_rate_mean"],
                    "win_rate_lower_95": beliefs["win_rate_lower_95"],
                    "win_rate_upper_95": beliefs["win_rate_upper_95"],
                    "total_trades": state.get("total_trades", 0),
                    "total_wins": state.get("total_wins", 0),
                    "cumulative_pnl": state.get("cumulative_pnl", 0),
                    "max_drawdown": state.get("max_drawdown", 0),
                }
            else:
                snap["bayesian"] = {}
        except Exception as e:
            logger.debug("Snapshot: Bayesian state unavailable: %s", e)
            snap["bayesian"] = {}

        # 3. Bandit state
        try:
            from ml.narrative_bandit import NarrativeBandit
            bandit = NarrativeBandit()
            arms = bandit._state.get("arms", [])
            best_arm = max(arms, key=lambda a: a.get("wins", 0) /
                          max(a.get("trials", 1), 1)) if arms else {}
            snap["bandit"] = {
                "total_trades": bandit._state.get("total_trades", 0),
                "num_arms": len(arms),
                "best_arm_id": best_arm.get("arm_id", "default"),
                "best_arm_win_rate": (best_arm.get("wins", 0) /
                                      max(best_arm.get("trials", 1), 1)),
                "is_active": bandit.is_active(),
            }
        except Exception as e:
            logger.debug("Snapshot: bandit state unavailable: %s", e)
            snap["bandit"] = {}

        # 4. Setup profile stats
        try:
            from ml.setup_profiles import SetupProfileStore
            store = SetupProfileStore()
            profiles = store._profiles
            wins = sum(1 for p in profiles
                       if p.get("outcome", "").startswith("tp"))
            losses = len(profiles) - wins
            avg_mfe = 0
            mfe_vals = [p.get("mfe") for p in profiles
                        if p.get("mfe") is not None]
            avg_mfe = sum(mfe_vals) / len(mfe_vals) if mfe_vals else 0
            snap["setup_profiles"] = {
                "total": len(profiles),
                "wins": wins,
                "losses": losses,
                "win_rate": wins / len(profiles) if profiles else 0,
                "avg_mfe": round(avg_mfe, 3),
            }
        except Exception as e:
            logger.debug("Snapshot: setup profiles unavailable: %s", e)
            snap["setup_profiles"] = {}

        # 5. Narrative state thesis accuracy
        try:
            from ml.narrative_state import NarrativeStateEngine
            nse = NarrativeStateEngine(db_path=self.db_path)
            metrics = nse.get_accuracy_metrics()
            snap["thesis_accuracy"] = {
                "prediction_accuracy": metrics.get("prediction_accuracy", 0),
                "predictions_scored": metrics.get("predictions_scored", 0),
                "thesis_stability": metrics.get("thesis_stability", 0),
                "revisions": metrics.get("revisions", 0),
            }
        except Exception as e:
            logger.debug("Snapshot: thesis accuracy unavailable: %s", e)
            snap["thesis_accuracy"] = {}

        # 6. Loss type breakdown (from setup profiles with MFE data)
        try:
            from ml.setup_profiles import SetupProfileStore
            store = SetupProfileStore()
            loss_profiles = [p for p in store._profiles
                            if not p.get("outcome", "").startswith("tp")]
            type1 = sum(1 for p in loss_profiles
                        if (p.get("mfe") or 0) < 0.5)
            type2 = sum(1 for p in loss_profiles
                        if (p.get("mfe") or 0) >= 1.0)
            ambiguous = len(loss_profiles) - type1 - type2
            snap["loss_types"] = {
                "total_losses": len(loss_profiles),
                "type1_wrong_narrative": type1,
                "type2_execution_failure": type2,
                "ambiguous": ambiguous,
            }
        except Exception as e:
            logger.debug("Snapshot: loss types unavailable: %s", e)
            snap["loss_types"] = {}

        # 7. Learned rules count
        try:
            from ml.setup_profiles import SetupProfileStore
            rules = SetupProfileStore().get_learned_rules(min_samples=10)
            snap["learned_rules_count"] = len(rules)
        except Exception:
            snap["learned_rules_count"] = 0

        # Store it
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO system_snapshots (timestamp, snapshot_json, trigger) "
                "VALUES (?, ?, ?)",
                (snap["timestamp"], json.dumps(snap), trigger)
            )

        logger.info("System snapshot recorded (trigger=%s)", trigger)
        return snap

    # ── Throttled snapshot (for use in resolution flow) ───────────

    def maybe_take_snapshot(self, trigger: str = "trade_resolved") -> dict | None:
        """Take a snapshot only if enough time has passed since the last one."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT timestamp FROM system_snapshots "
                "ORDER BY id DESC LIMIT 1"
            ).fetchone()
        if row:
            last_ts = datetime.fromisoformat(row[0])
            if last_ts.tzinfo is None:
                last_ts = last_ts.replace(tzinfo=timezone.utc)
            if (datetime.now(timezone.utc) - last_ts).total_seconds() < _MIN_INTERVAL:
                return None  # too soon
        return self.take_snapshot(trigger=trigger)

    # ── Query: get snapshots ──────────────────────────────────────

    def get_snapshots(self, days: int = 30, limit: int = 500) -> list[dict]:
        """Return recent snapshots as parsed dicts, newest first."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT snapshot_json FROM system_snapshots "
                "WHERE timestamp >= ? ORDER BY id DESC LIMIT ?",
                (cutoff, limit)
            ).fetchall()
        return [json.loads(r[0]) for r in rows]

    # ── Trend computation ─────────────────────────────────────────

    def compute_trends(self, days: int = 14) -> dict:
        """Compute trend direction for each tracked metric.

        Compares the average of the most recent 3 snapshots against the
        average of the 3 oldest snapshots within the window. Returns a
        dict with trend direction ('improving', 'declining', 'stable')
        and the delta for each metric.
        """
        snaps = self.get_snapshots(days=days)
        if len(snaps) < 4:
            return {"status": "insufficient_data", "snapshots_available": len(snaps)}

        # Newest and oldest clusters
        recent = snaps[:3]
        old = snaps[-3:]

        trends = {}

        # Narrative field weight trends
        all_fields = set()
        for s in recent + old:
            all_fields.update(s.get("narrative_weights", {}).keys())

        for field in all_fields:
            recent_vals = [s.get("narrative_weights", {}).get(field, {}).get("weight", 0.5)
                          for s in recent]
            old_vals = [s.get("narrative_weights", {}).get(field, {}).get("weight", 0.5)
                       for s in old]
            recent_avg = sum(recent_vals) / len(recent_vals)
            old_avg = sum(old_vals) / len(old_vals)
            delta = recent_avg - old_avg
            direction = "improving" if delta > 0.05 else ("declining" if delta < -0.05 else "stable")
            trends[f"narrative_{field}"] = {
                "current": round(recent_avg, 3),
                "previous": round(old_avg, 3),
                "delta": round(delta, 3),
                "direction": direction,
            }

        # Bayesian win rate trend
        recent_wr = [s.get("bayesian", {}).get("win_rate_mean", 0)
                     for s in recent if s.get("bayesian", {}).get("win_rate_mean")]
        old_wr = [s.get("bayesian", {}).get("win_rate_mean", 0)
                  for s in old if s.get("bayesian", {}).get("win_rate_mean")]
        if recent_wr and old_wr:
            r_avg = sum(recent_wr) / len(recent_wr)
            o_avg = sum(old_wr) / len(old_wr)
            delta = r_avg - o_avg
            trends["bayesian_win_rate"] = {
                "current": round(r_avg, 3),
                "previous": round(o_avg, 3),
                "delta": round(delta, 3),
                "direction": "improving" if delta > 0.03 else (
                    "declining" if delta < -0.03 else "stable"),
            }

        # Loss type ratio trend (Type 2 / total losses)
        recent_t2 = [s.get("loss_types", {}).get("type2_execution_failure", 0)
                     for s in recent]
        recent_total = [s.get("loss_types", {}).get("total_losses", 1)
                       for s in recent]
        old_t2 = [s.get("loss_types", {}).get("type2_execution_failure", 0)
                  for s in old]
        old_total = [s.get("loss_types", {}).get("total_losses", 1)
                    for s in old]
        r_ratio = sum(recent_t2) / max(sum(recent_total), 1)
        o_ratio = sum(old_t2) / max(sum(old_total), 1)
        delta = r_ratio - o_ratio
        trends["type2_loss_ratio"] = {
            "current": round(r_ratio, 3),
            "previous": round(o_ratio, 3),
            "delta": round(delta, 3),
            "direction": "improving" if delta > 0.05 else (
                "declining" if delta < -0.05 else "stable"),
            "note": "Higher = more losses are execution failures (narrative was right)",
        }

        # Setup profile win rate trend
        recent_spwr = [s.get("setup_profiles", {}).get("win_rate", 0)
                       for s in recent if s.get("setup_profiles", {}).get("total")]
        old_spwr = [s.get("setup_profiles", {}).get("win_rate", 0)
                    for s in old if s.get("setup_profiles", {}).get("total")]
        if recent_spwr and old_spwr:
            r_avg = sum(recent_spwr) / len(recent_spwr)
            o_avg = sum(old_spwr) / len(old_spwr)
            delta = r_avg - o_avg
            trends["profile_win_rate"] = {
                "current": round(r_avg, 3),
                "previous": round(o_avg, 3),
                "delta": round(delta, 3),
                "direction": "improving" if delta > 0.03 else (
                    "declining" if delta < -0.03 else "stable"),
            }

        return trends

    # ── Prompt context builder ────────────────────────────────────

    def build_prompt_context(self, days: int = 14) -> str:
        """Build a compact trend summary suitable for prompt injection.

        Returns a multi-line string describing which signals are improving,
        declining, or stable — so Claude can calibrate its confidence.
        """
        trends = self.compute_trends(days=days)
        if trends.get("status") == "insufficient_data":
            return ""  # not enough history yet

        lines = []
        direction_labels = {
            "improving": "↑ improving",
            "declining": "↓ declining",
            "stable": "→ stable",
        }
        field_labels = {
            "narrative_directional_bias": "Directional bias",
            "narrative_p3_phase": "P3 phase calls",
            "narrative_premium_discount": "Premium/discount",
            "narrative_confidence_calibration": "Confidence calibration",
            "narrative_intermarket_synthesis": "Intermarket synthesis",
            "narrative_key_levels": "Key levels",
            "bayesian_win_rate": "Overall win rate",
            "profile_win_rate": "Setup profile win rate",
            "type2_loss_ratio": "Execution-fail loss ratio",
        }

        for key, label in field_labels.items():
            t = trends.get(key)
            if not t:
                continue
            pct = t["current"] * 100
            d = direction_labels.get(t["direction"], "→ stable")
            delta_pct = t["delta"] * 100
            sign = "+" if delta_pct >= 0 else ""
            lines.append(f"- {label}: {pct:.0f}% ({d}, {sign}{delta_pct:.0f}pp)")

        if not lines:
            return ""

        # Add interpretation
        improving = [k for k, t in trends.items()
                     if isinstance(t, dict) and t.get("direction") == "improving"]
        declining = [k for k, t in trends.items()
                     if isinstance(t, dict) and t.get("direction") == "declining"]

        interpretation = ""
        if declining and not improving:
            interpretation = "System accuracy is declining across the board — increase caution, raise quality bar."
        elif improving and not declining:
            interpretation = "System is learning well — trust your signals, especially improving fields."
        elif declining:
            weak = [field_labels.get(k, k) for k in declining if k in field_labels]
            if weak:
                interpretation = (f"Declining signals ({', '.join(weak[:2])}) — "
                                 "reduce weight on these in your analysis.")

        block = "SYSTEM LEARNING STATUS (last 14 days):\n" + "\n".join(lines)
        if interpretation:
            block += "\n" + interpretation
        return block

    # ── Backfill from resolved trades ─────────────────────────────

    def backfill_from_trades(self, db_path: str = None) -> int:
        """Reconstruct approximate historical snapshots by replaying resolved trades.

        This replays trade outcomes in chronological order, reconstructing what
        the narrative weights and Bayesian beliefs *would have been* at each
        point. Snapshots are created at daily boundaries.

        Returns the number of snapshots backfilled.
        """
        scan_db = db_path or self.db_path
        try:
            conn = sqlite3.connect(scan_db)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM scanner_setups "
                "WHERE outcome IS NOT NULL AND resolved_at IS NOT NULL "
                "ORDER BY resolved_at ASC"
            ).fetchall()
            conn.close()
        except Exception as e:
            logger.warning("Backfill: cannot read scanner_setups: %s", e)
            return 0

        if not rows:
            return 0

        # Replay: simulate Bayesian updates + count outcomes per day
        from ml.bayesian import get_default_prior, update_beliefs, get_beliefs

        bayes_state = get_default_prior(self.cfg)
        # Use V1 priors if configured
        bayes_state["alpha"] = self.cfg.get("bayesian_prior_alpha",
                                            bayes_state["alpha"])
        bayes_state["beta_param"] = self.cfg.get("bayesian_prior_beta",
                                                  bayes_state["beta_param"])

        daily_buckets = {}  # date_str -> list of trade dicts
        for row in rows:
            rd = dict(row)
            resolved = rd.get("resolved_at", "")
            if not resolved:
                continue
            day = resolved[:10]  # YYYY-MM-DD
            daily_buckets.setdefault(day, []).append(rd)

        count = 0
        cumulative_wins = 0
        cumulative_losses = 0
        cumulative_type1 = 0
        cumulative_type2 = 0

        for day in sorted(daily_buckets.keys()):
            trades = daily_buckets[day]

            for t in trades:
                outcome = t.get("outcome", "")
                is_win = outcome.startswith("tp")
                pnl = t.get("pnl_rr") or (1.0 if is_win else -1.0)

                # Map outcome to Bayesian format
                mapped = "tp1_hit" if is_win else "stopped_out"
                bayes_state = update_beliefs(bayes_state, mapped, pnl)

                if is_win:
                    cumulative_wins += 1
                else:
                    cumulative_losses += 1
                    mfe = t.get("mfe_atr") or 0
                    if mfe < 0.5:
                        cumulative_type1 += 1
                    elif mfe >= 1.0:
                        cumulative_type2 += 1

            # Create a snapshot for this day
            beliefs = get_beliefs(bayes_state)
            total = cumulative_wins + cumulative_losses
            snap = {
                "timestamp": f"{day}T23:59:00+00:00",
                "trigger": "backfill",
                "narrative_weights": {},  # can't reconstruct without replaying EMA
                "bayesian": {
                    "win_rate_mean": beliefs["win_rate_mean"],
                    "win_rate_lower_95": beliefs["win_rate_lower_95"],
                    "win_rate_upper_95": beliefs["win_rate_upper_95"],
                    "total_trades": bayes_state["total_trades"],
                    "total_wins": bayes_state["total_wins"],
                    "cumulative_pnl": bayes_state.get("cumulative_pnl", 0),
                    "max_drawdown": bayes_state.get("max_drawdown", 0),
                },
                "setup_profiles": {
                    "total": total,
                    "wins": cumulative_wins,
                    "losses": cumulative_losses,
                    "win_rate": cumulative_wins / total if total else 0,
                },
                "loss_types": {
                    "total_losses": cumulative_losses,
                    "type1_wrong_narrative": cumulative_type1,
                    "type2_execution_failure": cumulative_type2,
                    "ambiguous": cumulative_losses - cumulative_type1 - cumulative_type2,
                },
                "thesis_accuracy": {},
                "bandit": {},
                "learned_rules_count": 0,
            }

            with self._conn() as c:
                c.execute(
                    "INSERT INTO system_snapshots (timestamp, snapshot_json, trigger) "
                    "VALUES (?, ?, ?)",
                    (snap["timestamp"], json.dumps(snap), "backfill")
                )
            count += 1

        logger.info("Backfilled %d daily snapshots from %d resolved trades",
                    count, len(rows))
        return count

    # ── Weekly report ─────────────────────────────────────────────

    def generate_weekly_report(self) -> dict:
        """Generate a weekly system evolution report.

        Compares this week's snapshots to last week's, producing a
        component-by-component analysis with interpretations.
        """
        now = datetime.now(timezone.utc)
        this_week = self.get_snapshots(days=7)
        last_week_snaps = self._get_snapshots_between(
            now - timedelta(days=14), now - timedelta(days=7))

        if not this_week:
            return {"status": "no_data", "message": "No snapshots this week."}

        report = {
            "generated_at": now.isoformat(),
            "period": f"{(now - timedelta(days=7)).strftime('%b %d')} — {now.strftime('%b %d, %Y')}",
            "snapshots_this_week": len(this_week),
            "snapshots_last_week": len(last_week_snaps),
            "components": {},
        }

        # Helper to average a nested field across snapshots
        def avg_field(snaps, path_fn, default=0):
            vals = [path_fn(s) for s in snaps if path_fn(s) is not None]
            return sum(vals) / len(vals) if vals else default

        # ── Bayesian beliefs ──
        curr_wr = avg_field(this_week,
                           lambda s: s.get("bayesian", {}).get("win_rate_mean"))
        prev_wr = avg_field(last_week_snaps,
                           lambda s: s.get("bayesian", {}).get("win_rate_mean"))
        curr_trades = max((s.get("bayesian", {}).get("total_trades", 0)
                          for s in this_week), default=0)
        report["components"]["bayesian"] = {
            "win_rate": round(curr_wr, 3),
            "previous_win_rate": round(prev_wr, 3) if last_week_snaps else None,
            "delta": round(curr_wr - prev_wr, 3) if last_week_snaps else None,
            "total_trades": curr_trades,
            "interpretation": self._interpret_bayesian(curr_wr, prev_wr, last_week_snaps),
        }

        # ── Narrative field weights ──
        field_report = {}
        all_fields = set()
        for s in this_week + last_week_snaps:
            all_fields.update(s.get("narrative_weights", {}).keys())

        for field in sorted(all_fields):
            curr = avg_field(this_week,
                            lambda s, f=field: s.get("narrative_weights", {}).get(f, {}).get("weight"))
            prev = avg_field(last_week_snaps,
                            lambda s, f=field: s.get("narrative_weights", {}).get(f, {}).get("weight"))
            total = max((s.get("narrative_weights", {}).get(field, {}).get("total", 0)
                        for s in this_week), default=0)
            delta = curr - prev if last_week_snaps else None
            field_report[field] = {
                "weight": round(curr, 3),
                "previous": round(prev, 3) if last_week_snaps else None,
                "delta": round(delta, 3) if delta is not None else None,
                "total_trades": total,
                "direction": ("improving" if delta and delta > 0.05 else
                             "declining" if delta and delta < -0.05 else "stable")
                             if delta is not None else "unknown",
            }
        report["components"]["narrative_weights"] = field_report

        # ── Loss type breakdown ──
        curr_losses = avg_field(this_week,
                               lambda s: s.get("loss_types", {}).get("total_losses"))
        curr_t1 = avg_field(this_week,
                           lambda s: s.get("loss_types", {}).get("type1_wrong_narrative"))
        curr_t2 = avg_field(this_week,
                           lambda s: s.get("loss_types", {}).get("type2_execution_failure"))
        report["components"]["loss_types"] = {
            "total_losses": round(curr_losses),
            "type1_wrong_narrative": round(curr_t1),
            "type2_execution_failure": round(curr_t2),
            "type2_ratio": round(curr_t2 / max(curr_losses, 1), 3),
            "interpretation": (
                "Most losses are execution failures (narrative was right) — "
                "focus on entry timing and SL placement."
                if curr_t2 > curr_t1 else
                "Most losses are narrative failures — "
                "the HTF story is being read wrong. Focus on improving directional bias."
            ) if curr_losses > 0 else "No losses recorded yet.",
        }

        # ── Setup profiles ──
        curr_total = max((s.get("setup_profiles", {}).get("total", 0)
                         for s in this_week), default=0)
        prev_total = max((s.get("setup_profiles", {}).get("total", 0)
                         for s in last_week_snaps), default=0) if last_week_snaps else 0
        report["components"]["setup_profiles"] = {
            "total_profiles": curr_total,
            "new_this_week": curr_total - prev_total if last_week_snaps else curr_total,
            "interpretation": (
                f"{curr_total - prev_total} new profiles added this week. "
                f"Pattern matching gets more reliable with each resolved trade."
                if last_week_snaps and curr_total > prev_total else
                f"{curr_total} profiles total. Building up pattern matching database."
            ),
        }

        # ── Bandit ──
        latest_bandit = next((s.get("bandit", {}) for s in this_week
                             if s.get("bandit")), {})
        report["components"]["bandit"] = {
            "is_active": latest_bandit.get("is_active", False),
            "best_arm": latest_bandit.get("best_arm_id", "default"),
            "total_trades": latest_bandit.get("total_trades", 0),
            "interpretation": (
                f"Bandit is active, best arm: {latest_bandit.get('best_arm_id', 'default')} "
                f"with {latest_bandit.get('best_arm_win_rate', 0) * 100:.0f}% win rate."
                if latest_bandit.get("is_active") else
                f"Bandit not yet active ({latest_bandit.get('total_trades', 0)} trades, needs 100). "
                "Using default prompt configuration."
            ),
        }

        # ── Overall interpretation ──
        improving = [f for f, d in field_report.items() if d.get("direction") == "improving"]
        declining = [f for f, d in field_report.items() if d.get("direction") == "declining"]
        stable = [f for f, d in field_report.items() if d.get("direction") == "stable"]

        if improving and not declining:
            overall = "System is learning well across all narrative dimensions."
        elif declining and not improving:
            overall = "System accuracy is regressing — may indicate a market regime change."
        elif improving and declining:
            overall = (f"Mixed progress: {', '.join(improving[:2])} improving, "
                      f"{', '.join(declining[:2])} declining.")
        else:
            overall = "System is stable — no significant changes this week."

        report["overall_interpretation"] = overall

        return report

    # ── Helpers ────────────────────────────────────────────────────

    def _get_snapshots_between(self, start: datetime, end: datetime) -> list[dict]:
        """Get snapshots between two datetimes."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT snapshot_json FROM system_snapshots "
                "WHERE timestamp >= ? AND timestamp < ? ORDER BY id DESC",
                (start.isoformat(), end.isoformat())
            ).fetchall()
        return [json.loads(r[0]) for r in rows]

    def _interpret_bayesian(self, curr: float, prev: float,
                           has_prev: bool) -> str:
        if not has_prev or prev == 0:
            return f"Current win rate belief: {curr * 100:.1f}%. No previous week to compare."
        delta = curr - prev
        if delta > 0.05:
            return (f"Win rate belief improved significantly: "
                   f"{prev * 100:.1f}% → {curr * 100:.1f}% (+{delta * 100:.1f}pp).")
        elif delta < -0.05:
            return (f"Win rate belief declined: "
                   f"{prev * 100:.1f}% → {curr * 100:.1f}% ({delta * 100:.1f}pp). "
                   "Monitor for continued decline.")
        return (f"Win rate belief stable at {curr * 100:.1f}% "
               f"(was {prev * 100:.1f}% last week).")

    def get_snapshot_count(self) -> int:
        """Return total number of snapshots stored."""
        with self._conn() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM system_snapshots"
            ).fetchone()[0]
