"""P6+P8 — Killzone Performance Profiling + Adaptive Scan Config.

Learns per-killzone quality gates and scan frequency adjustments from
resolved trade data. Replaces universal quality thresholds with
data-driven per-killzone bars.
"""
import json
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def normalize_killzone(kz: str) -> str:
    """Map freeform Sonnet killzone strings to canonical names.

    Canonical: Asian, London, NY_AM, NY_PM, Off
    (matches prompts.py KILLZONES and get_current_killzone() output)
    """
    if not kz:
        return "Off"
    lower = kz.lower().strip()

    # NY_PM must be checked before NY_AM (both contain "new york")
    if any(x in lower for x in ["ny_pm", "ny pm", "new york pm", "ny afternoon"]):
        return "NY_PM"
    if any(x in lower for x in ["ny_am", "ny am", "new york am", "ny open",
                                  "new york", "ny session"]):
        return "NY_AM"
    if any(x in lower for x in ["london", "ldn"]):
        return "London"
    if any(x in lower for x in ["asian", "asia", "tokyo", "sydney"]):
        return "Asian"
    if any(x in lower for x in ["off", "outside"]):
        return "Off"

    return "Off"


# WR thresholds for quality gates
_WR_REQUIRE_A = 0.50     # Below 50% WR → require A-grade only
_WR_ACCEPT_B = 0.60      # 50-65% WR → accept B+
_WR_ACCEPT_C = 0.70      # 65%+ WR → accept C+

# Default quality gate when insufficient data
_DEFAULT_MIN_QUALITY = "B"


class KillzoneProfiler:
    """Learn per-killzone quality gates and scan configuration."""

    def __init__(self, model_dir: str = None):
        self._model_dir = model_dir or os.path.join(
            os.path.dirname(__file__), "models")
        self._path = os.path.join(self._model_dir, "killzone_profile.json")
        self._gates = {}  # killzone → {"min_quality": "A"|"B"|"C"}
        self._stats = {}

    def compute_stats(self, trades: list[dict]) -> dict:
        """Compute per-killzone performance statistics.

        Args:
            trades: List of resolved trade dicts with 'outcome', 'killzone',
                    'setup_quality', 'timeframe', 'analysis_json'.

        Returns:
            Dict of killzone → stats.
        """
        if not trades:
            self._stats = {}
            return {}

        by_kz = {}
        for t in trades:
            kz = normalize_killzone(t.get("killzone", ""))
            by_kz.setdefault(kz, []).append(t)

        stats = {}
        for kz, kz_trades in by_kz.items():
            is_win = lambda t: t.get("outcome", "").startswith("tp") or t.get("outcome") == "runner"
            total = len(kz_trades)
            wins = sum(1 for t in kz_trades if is_win(t))

            # By quality grade
            by_quality = {}
            for t in kz_trades:
                q = t.get("setup_quality", "?")
                by_quality.setdefault(q, {"total": 0, "wins": 0})
                by_quality[q]["total"] += 1
                if is_win(t):
                    by_quality[q]["wins"] += 1
            for q_stats in by_quality.values():
                q_stats["win_rate"] = round(
                    q_stats["wins"] / q_stats["total"], 3) if q_stats["total"] > 0 else 0

            # By timeframe
            by_tf = {}
            for t in kz_trades:
                tf = t.get("timeframe", "?")
                by_tf.setdefault(tf, {"total": 0, "wins": 0})
                by_tf[tf]["total"] += 1
                if is_win(t):
                    by_tf[tf]["wins"] += 1
            for tf_stats in by_tf.values():
                tf_stats["win_rate"] = round(
                    tf_stats["wins"] / tf_stats["total"], 3) if tf_stats["total"] > 0 else 0

            stats[kz] = {
                "total": total,
                "wins": wins,
                "losses": total - wins,
                "win_rate": round(wins / total, 3) if total > 0 else 0,
                "by_quality": by_quality,
                "by_timeframe": by_tf,
            }

        self._stats = stats
        self._save()
        return stats

    def compute_quality_gates(self, trades: list[dict],
                               min_trades: int = 10) -> dict:
        """Compute minimum quality grade per killzone based on WR.

        Logic:
        - WR < 50% → require A-grade only
        - WR 50-65% → accept B+
        - WR > 65% → accept C+

        Args:
            trades: Resolved trade list.
            min_trades: Minimum trades in killzone before setting a gate.

        Returns:
            Dict of killzone → {"min_quality": grade, "win_rate": float, "total": int}
        """
        stats = self.compute_stats(trades)

        gates = {}
        for kz, s in stats.items():
            if s["total"] < min_trades:
                continue

            wr = s["win_rate"]
            if wr < _WR_REQUIRE_A:
                min_q = "A"
            elif wr < _WR_ACCEPT_B:
                min_q = "B"
            else:
                min_q = "C" if wr >= _WR_ACCEPT_C else "B"

            gates[kz] = {
                "min_quality": min_q,
                "win_rate": wr,
                "total": s["total"],
            }

        self._gates = gates
        return gates

    def should_skip(self, killzone: str, quality: str) -> bool:
        """Check if a setup should be skipped based on killzone quality gate.

        Args:
            killzone: Current killzone name.
            quality: Setup quality grade (A/B/C/D).

        Returns:
            True if the setup doesn't meet the killzone's quality bar.
        """
        if killzone not in self._gates:
            return False

        min_q = self._gates[killzone].get("min_quality", _DEFAULT_MIN_QUALITY)
        grade_order = {"A": 0, "B": 1, "C": 2, "D": 3}
        return grade_order.get(quality, 3) > grade_order.get(min_q, 1)

    def get_scan_config(self, trades: list[dict],
                         min_trades: int = 10) -> dict:
        """Generate adaptive scan configuration per killzone.

        Low-WR killzones get restricted scanning:
        - Skip 15min timeframe (most noise)
        - Higher quality bar

        Args:
            trades: Resolved trade list.
            min_trades: Minimum trades before restricting.

        Returns:
            Dict of killzone → scan config adjustments.
        """
        stats = self.compute_stats(trades)
        config = {}

        for kz, s in stats.items():
            if s["total"] < min_trades:
                continue

            wr = s["win_rate"]
            entry = {"skip_timeframes": [], "interval_overrides": {},
                     "restricted": False}

            if wr < _WR_REQUIRE_A:
                # Low WR: skip 15min (noisy), double 1h interval
                entry["skip_timeframes"] = ["15min"]
                entry["interval_overrides"] = {"1h": 120}
                entry["restricted"] = True
            elif wr < _WR_ACCEPT_B:
                # Medium WR: increase 15min interval
                entry["interval_overrides"] = {"15min": 30}

            # High WR: no restrictions
            config[kz] = entry

        self._save(scan_config=config)
        return config

    def _save(self, scan_config: dict = None):
        os.makedirs(self._model_dir, exist_ok=True)
        data = {
            "stats": self._stats,
            "gates": self._gates,
            "scan_config": scan_config or {},
            "updated_at": datetime.utcnow().isoformat(),
        }
        with open(self._path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self) -> dict:
        """Load saved profile from disk."""
        if os.path.exists(self._path):
            try:
                with open(self._path) as f:
                    data = json.load(f)
                self._stats = data.get("stats", {})
                self._gates = data.get("gates", {})
                return data
            except Exception:
                pass
        return {}
