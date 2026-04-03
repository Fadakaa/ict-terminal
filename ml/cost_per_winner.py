"""Priority 8: Cost-Per-Winner Optimization.

Correlates API spend with trade outcomes by timeframe × killzone segments.
Learns which scanning windows are cost-effective and recommends budget allocation.

Design:
  - Each setup's API cost is tagged at creation (Haiku screen + Sonnet analysis +
    Opus narrative) via cost_tracker.log_call(setup_id=...).
  - On resolution, the total api_cost_usd is stored on the setup row.
  - This module aggregates resolved setups into segments and computes:
      cost_per_winner = total_api_spend / wins
      roi = total_pnl_rr / total_api_spend
  - Segments with high cost-per-winner and low ROI get scan frequency reduced.
  - Segments with low cost-per-winner get prioritised.

Follows the same pattern as LayerPerformanceTracker and KillzoneProfiler.
"""
import json
import logging
import os
from collections import defaultdict
from datetime import datetime, timedelta

from ml.config import get_config

logger = logging.getLogger(__name__)

# Minimum resolved setups per segment before we make recommendations
MIN_SEGMENT_SAMPLES = 10

# Data window — only consider setups resolved in the last N days
DATA_WINDOW_DAYS = 30

# Cost-per-winner thresholds (USD)
# If a segment costs more than this per win, it's flagged for reduction
CPW_WARNING_USD = 1.50
CPW_CRITICAL_USD = 3.00

# ROI thresholds — pnl_rr earned per $1 of API spend
# Below this, the segment isn't paying for itself
ROI_MIN_THRESHOLD = 0.5

# Scan frequency adjustments
FREQ_REDUCE = "reduce"     # Scan less often (e.g. every other cycle)
FREQ_NORMAL = "normal"     # Default scan frequency
FREQ_BOOST = "boost"       # Scan more aggressively (higher priority)

WIN_OUTCOMES = ("tp1", "tp2", "tp3", "runner")


class CostPerWinnerTracker:
    """Aggregate API costs by segment, compute cost-per-winner, recommend budget allocation."""

    def __init__(self, config: dict = None):
        self.cfg = config or get_config()
        self._path = os.path.join(
            self.cfg.get("model_dir", "ml/models"), "cost_per_winner.json"
        )
        self._stats = self._load()

    def _load(self) -> dict:
        if os.path.exists(self._path):
            try:
                with open(self._path) as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "segments": {},
            "global": self._empty_global(),
            "recommendations": {},
            "updated_at": None,
        }

    def _save(self):
        self._stats["updated_at"] = datetime.utcnow().isoformat()
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        with open(self._path, "w") as f:
            json.dump(self._stats, f, indent=2)

    @staticmethod
    def _empty_segment() -> dict:
        return {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "total_api_cost_usd": 0.0,
            "total_pnl_rr": 0.0,
            "cost_per_winner_usd": None,
            "cost_per_trade_usd": None,
            "roi_per_dollar": None,
            "win_rate": None,
        }

    @staticmethod
    def _empty_global() -> dict:
        return {
            "total_trades": 0,
            "total_wins": 0,
            "total_api_cost_usd": 0.0,
            "total_pnl_rr": 0.0,
            "avg_cost_per_winner_usd": None,
            "avg_cost_per_trade_usd": None,
            "overall_roi_per_dollar": None,
        }

    def _segment_key(self, timeframe: str, killzone: str) -> str:
        """Create a segment key from timeframe and killzone."""
        tf = (timeframe or "unknown").lower()
        kz = (killzone or "unknown").lower()
        return f"{tf}_{kz}"

    def ingest_trade(self, setup: dict):
        """Ingest a resolved trade with its API cost.

        Args:
            setup: dict with keys: timeframe, killzone, setup_quality, outcome,
                   pnl_rr, api_cost_usd. All from the scanner_setups row.
        """
        api_cost = setup.get("api_cost_usd")
        if api_cost is None or api_cost <= 0:
            return  # No cost data — skip

        tf = setup.get("timeframe", "unknown")
        kz = setup.get("killzone", "unknown")
        outcome = setup.get("outcome", "")
        pnl_rr = setup.get("pnl_rr") or 0.0
        is_win = outcome in WIN_OUTCOMES

        key = self._segment_key(tf, kz)

        # Update segment
        if key not in self._stats["segments"]:
            self._stats["segments"][key] = self._empty_segment()
        seg = self._stats["segments"][key]

        seg["total_trades"] += 1
        seg["total_api_cost_usd"] = round(seg["total_api_cost_usd"] + api_cost, 6)
        seg["total_pnl_rr"] = round(seg["total_pnl_rr"] + pnl_rr, 4)
        if is_win:
            seg["wins"] += 1
        else:
            seg["losses"] += 1

        # Recompute derived metrics
        self._recompute_segment(seg)

        # Update global
        g = self._stats["global"]
        g["total_trades"] += 1
        g["total_api_cost_usd"] = round(g["total_api_cost_usd"] + api_cost, 6)
        g["total_pnl_rr"] = round(g["total_pnl_rr"] + pnl_rr, 4)
        if is_win:
            g["total_wins"] += 1
        self._recompute_global(g)

        self._save()

    @staticmethod
    def _recompute_segment(seg: dict):
        """Recompute derived metrics for a segment."""
        t = seg["total_trades"]
        w = seg["wins"]
        cost = seg["total_api_cost_usd"]
        pnl = seg["total_pnl_rr"]

        seg["win_rate"] = round(w / t, 4) if t > 0 else None
        seg["cost_per_trade_usd"] = round(cost / t, 6) if t > 0 else None
        seg["cost_per_winner_usd"] = round(cost / w, 6) if w > 0 else None
        seg["roi_per_dollar"] = round(pnl / cost, 4) if cost > 0 else None

    @staticmethod
    def _recompute_global(g: dict):
        """Recompute derived metrics for global stats."""
        t = g["total_trades"]
        w = g["total_wins"]
        cost = g["total_api_cost_usd"]
        pnl = g["total_pnl_rr"]

        g["avg_cost_per_trade_usd"] = round(cost / t, 6) if t > 0 else None
        g["avg_cost_per_winner_usd"] = round(cost / w, 6) if w > 0 else None
        g["overall_roi_per_dollar"] = round(pnl / cost, 4) if cost > 0 else None

    def recompute_from_db(self, scanner_db) -> dict:
        """Full recompute from resolved setups in the database.

        Uses DATA_WINDOW_DAYS to only consider recent data.
        Returns the updated stats.
        """
        cutoff = (datetime.utcnow() - timedelta(days=DATA_WINDOW_DAYS)).isoformat()
        setups = scanner_db.get_resolved_with_costs()

        # Reset
        self._stats = {
            "segments": {},
            "global": self._empty_global(),
            "recommendations": {},
            "updated_at": None,
        }

        for s in setups:
            if (s.get("resolved_at") or "") < cutoff:
                continue  # Outside data window
            self.ingest_trade(s)

        # Generate recommendations after recompute
        self._stats["recommendations"] = self.get_recommendations()
        self._save()
        return self._stats

    def get_recommendations(self) -> dict:
        """Generate scan frequency recommendations per segment.

        Returns:
            dict mapping segment_key -> {"action": "reduce"|"normal"|"boost",
                                          "reason": str, "cpw": float, "roi": float}
        """
        recs = {}
        for key, seg in self._stats["segments"].items():
            if seg["total_trades"] < MIN_SEGMENT_SAMPLES:
                recs[key] = {
                    "action": FREQ_NORMAL,
                    "reason": f"Insufficient data ({seg['total_trades']}/{MIN_SEGMENT_SAMPLES})",
                    "cpw": seg["cost_per_winner_usd"],
                    "roi": seg["roi_per_dollar"],
                }
                continue

            cpw = seg["cost_per_winner_usd"]
            roi = seg["roi_per_dollar"]

            if cpw is not None and cpw >= CPW_CRITICAL_USD:
                recs[key] = {
                    "action": FREQ_REDUCE,
                    "reason": f"Cost/winner ${cpw:.2f} >= ${CPW_CRITICAL_USD:.2f} critical threshold",
                    "cpw": cpw,
                    "roi": roi,
                }
            elif roi is not None and roi < ROI_MIN_THRESHOLD:
                recs[key] = {
                    "action": FREQ_REDUCE,
                    "reason": f"ROI {roi:.2f}R/$ < {ROI_MIN_THRESHOLD:.1f} minimum",
                    "cpw": cpw,
                    "roi": roi,
                }
            elif cpw is not None and cpw >= CPW_WARNING_USD:
                recs[key] = {
                    "action": FREQ_NORMAL,
                    "reason": f"Cost/winner ${cpw:.2f} approaching warning (${CPW_WARNING_USD:.2f})",
                    "cpw": cpw,
                    "roi": roi,
                }
            elif roi is not None and roi >= ROI_MIN_THRESHOLD * 3:
                recs[key] = {
                    "action": FREQ_BOOST,
                    "reason": f"High ROI {roi:.2f}R/$ — cost-effective segment",
                    "cpw": cpw,
                    "roi": roi,
                }
            else:
                recs[key] = {
                    "action": FREQ_NORMAL,
                    "reason": "Within acceptable cost parameters",
                    "cpw": cpw,
                    "roi": roi,
                }

        return recs

    def should_reduce_scan(self, timeframe: str, killzone: str) -> bool:
        """Check if a segment should reduce scan frequency.

        Called by the scanner before each cycle to decide whether to skip.
        """
        key = self._segment_key(timeframe, killzone)
        rec = self._stats.get("recommendations", {}).get(key)
        if rec and rec.get("action") == FREQ_REDUCE:
            return True
        return False

    def should_boost_scan(self, timeframe: str, killzone: str) -> bool:
        """Check if a segment deserves boosted scan frequency."""
        key = self._segment_key(timeframe, killzone)
        rec = self._stats.get("recommendations", {}).get(key)
        if rec and rec.get("action") == FREQ_BOOST:
            return True
        return False

    def get_stats(self) -> dict:
        """Return full stats for dashboard/API."""
        return {
            "global": self._stats["global"],
            "segments": self._stats["segments"],
            "recommendations": self._stats.get("recommendations", {}),
            "updated_at": self._stats.get("updated_at"),
        }

    def get_segment_ranking(self) -> list:
        """Return segments ranked by cost-effectiveness (best ROI first)."""
        ranking = []
        for key, seg in self._stats["segments"].items():
            if seg["total_trades"] < MIN_SEGMENT_SAMPLES:
                continue
            ranking.append({
                "segment": key,
                "roi_per_dollar": seg["roi_per_dollar"],
                "cost_per_winner_usd": seg["cost_per_winner_usd"],
                "win_rate": seg["win_rate"],
                "total_trades": seg["total_trades"],
                "total_api_cost_usd": seg["total_api_cost_usd"],
                "total_pnl_rr": seg["total_pnl_rr"],
            })
        ranking.sort(key=lambda x: x["roi_per_dollar"] or -999, reverse=True)
        return ranking
