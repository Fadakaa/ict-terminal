"""P7 — Intermarket Signal Validation.

Tracks whether DXY/US10Y context actually improves trade outcomes.
Stratified analysis by divergence, killzone, and yield direction.
"""
import json
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class IntermarketValidator:
    """Analyze whether intermarket signals predict trade outcomes."""

    def __init__(self, model_dir: str = None):
        self._model_dir = model_dir or os.path.join(
            os.path.dirname(__file__), "models")
        self._path = os.path.join(self._model_dir, "intermarket_validation.json")
        self._last_result = None

    @staticmethod
    def score_intermarket_signal(diverging: int, is_win: bool,
                                  corr: float, yield_dir: int,
                                  direction: str) -> float:
        """Score how correct the intermarket signal was for this trade.

        Returns 0.0 (intermarket was wrong) to 1.0 (intermarket was right).

        Key logic:
        - Divergence warning + loss = correct warning (high score)
        - Divergence warning + win = false alarm (low score)
        - No divergence = neutral (0.5 base, adjusted by yield alignment)
        """
        if diverging:
            # Divergence was flagged — was it a useful warning?
            if not is_win:
                return 0.9  # Correct: warned and trade lost
            else:
                return 0.2  # False alarm: warned but trade won anyway

        # No divergence — check yield alignment
        # Falling yields support longs, rising yields support shorts
        yield_aligned = (
            (yield_dir == -1 and direction == "long") or
            (yield_dir == 1 and direction == "short")
        )
        yield_misaligned = (
            (yield_dir == 1 and direction == "long") or
            (yield_dir == -1 and direction == "short")
        )

        if yield_aligned and is_win:
            return 0.6  # Yield signal aligned and won — slight positive
        elif yield_misaligned and not is_win:
            return 0.4  # Yield warned against, and lost — mildly correct
        else:
            return 0.5  # Neutral — no strong signal either way

    def analyze(self, trades: list[dict]) -> dict:
        """Run stratified analysis on resolved trades with intermarket data.

        Args:
            trades: List of dicts with 'outcome', 'killzone', 'direction',
                    'calibration_json' (containing intermarket block).

        Returns:
            Comprehensive analysis dict with per-segment stats and recommendation.
        """
        if not trades:
            return {"total_trades": 0, "recommendation": "insufficient_data",
                    "by_divergence": {}, "by_killzone": {}, "by_yield_direction": {}}

        # Parse intermarket data from each trade
        parsed = []
        for t in trades:
            try:
                cal = json.loads(t["calibration_json"]) if isinstance(
                    t["calibration_json"], str) else t.get("calibration_json", {})
            except (json.JSONDecodeError, TypeError):
                continue

            im = cal.get("intermarket", {})
            if not im:
                continue

            is_win = t.get("outcome", "").startswith("tp") or t.get("outcome") == "runner"
            from ml.killzone_profiler import normalize_killzone
            parsed.append({
                "is_win": is_win,
                "killzone": normalize_killzone(t.get("killzone", "")),
                "direction": t.get("direction", ""),
                "diverging": im.get("gold_dxy_diverging", 0),
                "corr": im.get("gold_dxy_corr_20", 0),
                "yield_dir": im.get("yield_direction", 0),
                "dxy_range": im.get("dxy_range_position", 0.5),
            })

        if not parsed:
            return {"total_trades": 0, "recommendation": "insufficient_data",
                    "by_divergence": {}, "by_killzone": {}, "by_yield_direction": {}}

        result = {
            "total_trades": len(parsed),
            "by_divergence": self._stratify_divergence(parsed),
            "by_killzone": self._stratify_killzone(parsed),
            "by_yield_direction": self._stratify_yield(parsed),
            "analyzed_at": datetime.utcnow().isoformat(),
        }

        # Determine recommendation
        result["recommendation"] = self._compute_recommendation(result)

        # Persist
        self._last_result = result
        self._save(result)
        return result

    def _stratify_divergence(self, parsed: list) -> dict:
        div = [t for t in parsed if t["diverging"]]
        non = [t for t in parsed if not t["diverging"]]
        return {
            "diverging": self._compute_segment(div),
            "not_diverging": self._compute_segment(non),
        }

    def _stratify_killzone(self, parsed: list) -> dict:
        kzs = {}
        for t in parsed:
            kz = t["killzone"]
            kzs.setdefault(kz, []).append(t)
        return {kz: self._compute_segment(trades) for kz, trades in kzs.items()}

    def _stratify_yield(self, parsed: list) -> dict:
        buckets = {"falling": [], "flat": [], "rising": []}
        for t in parsed:
            if t["yield_dir"] == -1:
                buckets["falling"].append(t)
            elif t["yield_dir"] == 1:
                buckets["rising"].append(t)
            else:
                buckets["flat"].append(t)
        return {k: self._compute_segment(v) for k, v in buckets.items() if v}

    @staticmethod
    def _compute_segment(trades: list) -> dict:
        if not trades:
            return {"total": 0, "wins": 0, "losses": 0, "win_rate": 0}
        wins = sum(1 for t in trades if t["is_win"])
        total = len(trades)
        return {
            "total": total,
            "wins": wins,
            "losses": total - wins,
            "win_rate": round(wins / total, 3),
        }

    def _compute_recommendation(self, result: dict) -> str:
        """Determine if intermarket is useful, noise, or insufficient."""
        div_data = result["by_divergence"]
        div = div_data.get("diverging", {})
        non = div_data.get("not_diverging", {})

        # Need at least 5 diverging trades to judge
        if div.get("total", 0) < 5 or non.get("total", 0) < 5:
            return "insufficient_data"

        # If divergence WR is significantly lower than non-divergence → useful
        wr_diff = non.get("win_rate", 0) - div.get("win_rate", 0)

        if wr_diff >= 0.15:
            return "useful"  # Divergence predicts losses
        else:
            return "noise"   # No meaningful difference

    def _save(self, result: dict):
        os.makedirs(self._model_dir, exist_ok=True)
        with open(self._path, "w") as f:
            json.dump(result, f, indent=2)

    def get_last_result(self) -> dict | None:
        if self._last_result:
            return self._last_result
        if os.path.exists(self._path):
            try:
                with open(self._path) as f:
                    return json.load(f)
            except Exception:
                pass
        return None
