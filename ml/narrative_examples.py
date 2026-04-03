"""Gold narrative example store — curates high-quality Opus narratives for prompt injection.

Stores up to 8 examples of narratives that led to winning trades.
Enforces diversity (max 2 per session, max 2 per direction) and evicts
the lowest-quality examples using score × recency decay.
"""
import json
import os
import logging
from datetime import datetime

from ml.config import get_config

logger = logging.getLogger(__name__)

WIN_OUTCOMES = {"tp1", "tp2", "tp3", "tp1_hit", "tp2_hit", "tp3_hit"}
TP2_PLUS = {"tp2", "tp3", "tp2_hit", "tp3_hit"}


class NarrativeExampleStore:

    def __init__(self, config: dict = None):
        self.cfg = config or get_config()
        self._path = os.path.join(
            self.cfg.get("model_dir", os.path.join(os.path.dirname(__file__), "models")),
            "gold_narratives.json")
        self._examples = self._load()

    def _load(self) -> list:
        if os.path.exists(self._path):
            try:
                with open(self._path) as f:
                    return json.load(f)
            except Exception:
                pass
        return []

    def _save(self):
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        with open(self._path, "w") as f:
            json.dump(self._examples, f, indent=2)

    def add_example(self, narrative_json: dict, outcome: str,
                    session: str, direction: str, entry_price: float,
                    key_levels: list) -> bool:
        """Score and potentially add a narrative example. Returns True if stored."""
        if not narrative_json or not outcome:
            return False

        is_win = outcome in WIN_OUTCOMES
        if not is_win:
            return False  # only store winning narratives

        score = self._score(narrative_json, outcome, direction,
                            key_levels, entry_price)

        min_score = self.cfg.get("gold_example_min_score", 0.3)
        if score < min_score:
            return False

        example = {
            "narrative_json": narrative_json,
            "outcome": outcome,
            "session": session or "Off",
            "direction": direction,
            "date": datetime.utcnow().strftime("%Y-%m-%d"),
            "score": round(score, 3),
            "entry_price": entry_price,
        }

        self._evict_if_full()
        self._examples.append(example)
        self._save()
        logger.info("Gold example added: %s %s %s (score=%.2f)",
                    session, direction, outcome, score)
        return True

    def get_examples(self, session: str, bias_hint: str | None = None,
                     n: int = None) -> list:
        """Return top-n relevant examples with diversity enforcement."""
        n = n or self.cfg.get("gold_example_retrieve_n", 3)
        if not self._examples:
            return []

        candidates = sorted(self._examples, key=lambda e: e["score"], reverse=True)

        # Prioritize session match
        session_matched = [e for e in candidates if e["session"] == session]
        others = [e for e in candidates if e["session"] != session]

        result = []
        session_counts = {}
        direction_counts = {}

        for e in session_matched + others:
            if len(result) >= n:
                break
            s = e["session"]
            d = e["direction"]
            # Diversity: max 2 per session, max 2 per direction
            if session_counts.get(s, 0) >= 2:
                continue
            if direction_counts.get(d, 0) >= 2:
                continue
            result.append(e)
            session_counts[s] = session_counts.get(s, 0) + 1
            direction_counts[d] = direction_counts.get(d, 0) + 1

        return result

    def _score(self, narrative_json: dict, outcome: str,
               direction: str, key_levels: list, entry_price: float) -> float:
        """Composite score for a narrative-outcome pair."""
        score = 0.0

        # Direction correct (0.4): bias aligned with direction AND won
        bias = narrative_json.get("directional_bias", "")
        direction_aligned = (
            (bias == "bullish" and direction == "long") or
            (bias == "bearish" and direction == "short")
        )
        if direction_aligned:
            score += 0.4

        # Phase correct (0.3): distribution + short win, or accumulation + long win
        phase = narrative_json.get("power_of_3_phase", "")
        phase_correct = (
            (phase == "distribution" and direction == "short") or
            (phase == "accumulation" and direction == "long") or
            (phase == "manipulation")  # manipulation is always transitional
        )
        if phase_correct:
            score += 0.3

        # Key level touched (0.2): any key level within 0.3% of entry price
        if entry_price and key_levels:
            for level in key_levels:
                price = level.get("price", 0) if isinstance(level, dict) else 0
                if price and abs(price - entry_price) / entry_price < 0.003:
                    score += 0.2
                    break

        # TP2+ bonus (0.1)
        if outcome in TP2_PLUS:
            score += 0.1

        return score

    def _evict_if_full(self):
        """Remove lowest effective-score example when store is full."""
        max_store = self.cfg.get("gold_example_max_store", 8)
        if len(self._examples) < max_store:
            return

        decay = self.cfg.get("gold_example_recency_decay", 0.95)
        now = datetime.utcnow()

        worst_idx = 0
        worst_eff = float("inf")

        for i, e in enumerate(self._examples):
            try:
                d = datetime.strptime(e["date"], "%Y-%m-%d")
                days_old = (now - d).days
            except (ValueError, KeyError):
                days_old = 30

            effective = e.get("score", 0) * (decay ** days_old)
            if effective < worst_eff:
                worst_eff = effective
                worst_idx = i

        self._examples.pop(worst_idx)
