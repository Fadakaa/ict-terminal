"""Thompson sampling bandit for narrative prompt variant testing.

Each arm is a set of prompt parameters (not full rewrites). The bandit
selects which variant to use on each scan, then updates based on trade outcome.
Only activates after a configurable minimum trade count (default 100).
"""
import json
import os
import random
import logging

from ml.config import get_config

logger = logging.getLogger(__name__)

EXAMPLE_STRATEGIES = ["top_scoring", "session_matched", "diverse"]
WEIGHT_DISPLAYS = ["percentage", "bar_description", "none"]
EMPHASIS_OPTIONS = ["weak_fields", "strong_fields", "balanced"]

DEFAULT_ARM = {
    "arm_id": "default",
    "alpha": 1,
    "beta_param": 1,
    "trials": 0,
    "wins": 0,
    "params": {
        "example_strategy": "top_scoring",
        "weight_display": "percentage",
        "emphasis": "weak_fields",
    },
}


class NarrativeBandit:

    def __init__(self, config: dict = None):
        self.cfg = config or get_config()
        self._path = os.path.join(
            self.cfg.get("model_dir", os.path.join(os.path.dirname(__file__), "models")),
            "narrative_bandit.json")
        self._state = self._load()

    def _load(self) -> dict:
        if os.path.exists(self._path):
            try:
                with open(self._path) as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "total_trades": 0,
            "arms": [dict(DEFAULT_ARM)],
        }

    def _save(self):
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        with open(self._path, "w") as f:
            json.dump(self._state, f, indent=2)

    def is_active(self) -> bool:
        """Only active after minimum trade threshold."""
        min_trades = self.cfg.get("narrative_bandit_min_trades", 100)
        return self._state["total_trades"] >= min_trades

    def select_arm(self) -> dict:
        """Thompson sampling: sample from Beta(alpha, beta) per arm, pick highest."""
        arms = self._state["arms"]
        if not arms:
            return dict(DEFAULT_ARM)

        if len(arms) == 1:
            return dict(arms[0])

        best_arm = None
        best_sample = -1

        for arm in arms:
            a = max(arm.get("alpha", 1), 1)
            b = max(arm.get("beta_param", 1), 1)
            sample = random.betavariate(a, b)
            if sample > best_sample:
                best_sample = sample
                best_arm = arm

        return dict(best_arm) if best_arm else dict(arms[0])

    def update_arm(self, arm_id: str, is_win: bool):
        """Update Beta distribution for the selected arm."""
        for arm in self._state["arms"]:
            if arm["arm_id"] == arm_id:
                arm["trials"] += 1
                if is_win:
                    arm["alpha"] = arm.get("alpha", 1) + 1
                    arm["wins"] = arm.get("wins", 0) + 1
                else:
                    arm["beta_param"] = arm.get("beta_param", 1) + 1
                break

        self._state["total_trades"] = self._state.get("total_trades", 0) + 1
        self._save()

    def retire_underperformers(self):
        """Remove arms with WR < threshold after min trials."""
        min_trials = self.cfg.get("narrative_bandit_retire_min_trials", 30)
        retire_wr = self.cfg.get("narrative_bandit_retire_wr", 0.35)

        to_remove = []
        for arm in self._state["arms"]:
            if arm["arm_id"] == "default":
                continue  # never retire default
            if arm["trials"] >= min_trials:
                wr = arm["wins"] / arm["trials"] if arm["trials"] > 0 else 0
                if wr < retire_wr:
                    to_remove.append(arm["arm_id"])

        if to_remove:
            self._state["arms"] = [
                a for a in self._state["arms"] if a["arm_id"] not in to_remove]
            logger.info("Bandit retired arms: %s", to_remove)
            self._save()

    def maybe_generate_variant(self):
        """Every N trades, generate a new variant by mutating worst arm."""
        interval = self.cfg.get("narrative_bandit_new_variant_every", 100)
        max_arms = self.cfg.get("narrative_bandit_max_arms", 3)
        total = self._state["total_trades"]

        if total == 0 or total % interval != 0:
            return

        if len(self._state["arms"]) >= max_arms:
            # Find worst arm (excluding default) to replace
            arms = [a for a in self._state["arms"] if a["arm_id"] != "default"]
            if not arms:
                return
            worst = min(arms, key=lambda a: a["wins"] / max(a["trials"], 1))
            self._state["arms"] = [
                a for a in self._state["arms"] if a["arm_id"] != worst["arm_id"]]

        # Find the worst arm to mutate from
        source = min(self._state["arms"],
                     key=lambda a: a["wins"] / max(a["trials"], 1))
        params = dict(source.get("params", DEFAULT_ARM["params"]))

        # Randomly mutate one parameter
        param_key = random.choice(["example_strategy", "weight_display", "emphasis"])
        options = {
            "example_strategy": EXAMPLE_STRATEGIES,
            "weight_display": WEIGHT_DISPLAYS,
            "emphasis": EMPHASIS_OPTIONS,
        }
        current = params.get(param_key, options[param_key][0])
        choices = [o for o in options[param_key] if o != current]
        if choices:
            params[param_key] = random.choice(choices)

        new_arm = {
            "arm_id": f"v{len(self._state['arms'])}_t{total}",
            "alpha": 1,
            "beta_param": 1,
            "trials": 0,
            "wins": 0,
            "params": params,
        }

        self._state["arms"].append(new_arm)
        self._save()
        logger.info("Bandit generated new arm: %s with params %s",
                    new_arm["arm_id"], params)

    def get_state(self) -> dict:
        """Return full bandit state for API."""
        arms_summary = []
        for arm in self._state["arms"]:
            wr = arm["wins"] / arm["trials"] if arm["trials"] > 0 else 0
            arms_summary.append({
                "arm_id": arm["arm_id"],
                "trials": arm["trials"],
                "wins": arm["wins"],
                "win_rate": round(wr, 3),
                "params": arm.get("params", {}),
            })

        return {
            "active": self.is_active(),
            "total_trades": self._state["total_trades"],
            "arms": arms_summary,
            "num_arms": len(self._state["arms"]),
        }
