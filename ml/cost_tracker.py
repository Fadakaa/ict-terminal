"""API cost tracker — logs every Claude call with estimated cost.

Enforces a daily budget cap. When budget exceeded, all Claude calls
are skipped and cached data used instead.
"""
import json
import os
import logging
from collections import defaultdict
from datetime import date, datetime

from ml.config import get_config

logger = logging.getLogger(__name__)

# Pricing per 1M tokens (as of 2026)
MODEL_PRICING = {
    "haiku": {"input": 0.25, "output": 1.25},
    "claude-3-5-haiku-20241022": {"input": 0.25, "output": 1.25},
    "sonnet": {"input": 3.0, "output": 15.0},
    "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
    "opus": {"input": 15.0, "output": 75.0},
    "claude-opus-4-20250514": {"input": 15.0, "output": 75.0},
}


def _compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Compute estimated cost in USD."""
    # Normalize model name
    pricing = MODEL_PRICING.get(model)
    if not pricing:
        for key in MODEL_PRICING:
            if key in model:
                pricing = MODEL_PRICING[key]
                break
    if not pricing:
        pricing = MODEL_PRICING["sonnet"]  # default

    cost = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000
    return round(cost, 6)


class CostTracker:

    def __init__(self, config: dict = None):
        self.cfg = config or get_config()
        self._path = os.path.join(
            os.path.dirname(__file__), "models", "cost_log.json")
        self._log = self._load()

    def _load(self) -> list:
        if os.path.exists(self._path):
            try:
                with open(self._path) as f:
                    data = json.load(f)
                # Only keep last 30 days
                cutoff = (datetime.utcnow().replace(day=1)).isoformat()[:10]
                return [e for e in data if e.get("timestamp", "")[:10] >= cutoff]
            except Exception:
                pass
        return []

    def _save(self):
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        with open(self._path, "w") as f:
            json.dump(self._log, f)

    def log_call(self, model: str, input_tokens: int, output_tokens: int,
                 purpose: str, setup_id: str = None) -> float:
        """Log an API call and return its estimated cost.

        Args:
            setup_id: Optional scanner setup ID to attribute this cost to.
                      Enables per-setup cost aggregation for cost-per-winner analysis.
        """
        cost = _compute_cost(model, input_tokens, output_tokens)
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "model": model,
            "purpose": purpose,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost,
        }
        if setup_id:
            entry["setup_id"] = setup_id
        self._log.append(entry)

        # Save every 10 entries (batch for performance)
        if len(self._log) % 10 == 0:
            self._save()

        return cost

    def flush(self):
        """Force save to disk."""
        self._save()

    def get_daily_summary(self, day: str = None) -> dict:
        """Get cost summary for a specific day (default: today)."""
        target = day or date.today().isoformat()
        today_entries = [e for e in self._log
                         if e.get("timestamp", "")[:10] == target]

        by_purpose = defaultdict(float)
        by_model = defaultdict(float)
        for e in today_entries:
            by_purpose[e.get("purpose", "unknown")] += e.get("cost_usd", 0)
            by_model[e.get("model", "unknown")] += e.get("cost_usd", 0)

        total = sum(e.get("cost_usd", 0) for e in today_entries)

        return {
            "date": target,
            "total_usd": round(total, 4),
            "by_purpose": {k: round(v, 4) for k, v in by_purpose.items()},
            "by_model": {k: round(v, 4) for k, v in by_model.items()},
            "call_count": len(today_entries),
        }

    def check_budget(self) -> bool:
        """Return True if within daily budget, False if exceeded."""
        limit = self.cfg.get("daily_api_budget_usd", 5.0)
        summary = self.get_daily_summary()
        return summary["total_usd"] < limit

    def is_warning(self) -> bool:
        """Return True if approaching daily budget (at warning threshold)."""
        limit = self.cfg.get("daily_api_budget_usd", 5.0)
        threshold = self.cfg.get("budget_warning_threshold", 0.80)
        summary = self.get_daily_summary()
        return summary["total_usd"] >= limit * threshold

    def get_remaining_budget(self) -> float:
        """Return remaining daily budget in USD."""
        limit = self.cfg.get("daily_api_budget_usd", 5.0)
        summary = self.get_daily_summary()
        return round(max(0, limit - summary["total_usd"]), 4)

    def get_setup_cost(self, setup_id: str) -> dict:
        """Get total API cost attributed to a specific scanner setup.

        Returns dict with total_usd, by_purpose breakdown, and call_count.
        """
        entries = [e for e in self._log if e.get("setup_id") == setup_id]
        by_purpose = defaultdict(float)
        for e in entries:
            by_purpose[e.get("purpose", "unknown")] += e.get("cost_usd", 0)
        total = sum(e.get("cost_usd", 0) for e in entries)
        return {
            "setup_id": setup_id,
            "total_usd": round(total, 6),
            "by_purpose": {k: round(v, 6) for k, v in by_purpose.items()},
            "call_count": len(entries),
        }

    def get_history(self, days: int = 30) -> list:
        """Return daily totals for the last N days."""
        from datetime import timedelta
        result = []
        today = date.today()
        for i in range(days):
            day = (today - timedelta(days=i)).isoformat()
            summary = self.get_daily_summary(day)
            if summary["call_count"] > 0:
                result.append(summary)
        return result


# Singleton for use across the scanner
_tracker: CostTracker | None = None


def get_cost_tracker() -> CostTracker:
    global _tracker
    if _tracker is None:
        _tracker = CostTracker()
    return _tracker
