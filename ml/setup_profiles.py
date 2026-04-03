"""Setup profile store — historical DNA profiles with conditional statistics.

Stores resolved trade profiles keyed by setup DNA. Provides similarity search,
conditional win rate computation, and learned rule generation for prompt injection.

Storage: ml/models/setup_profiles.json (append-only, deduped by setup_id).
"""
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone

from ml.setup_dna import compute_similarity


_DEFAULT_PATH = os.path.join(
    os.path.dirname(__file__), "models", "setup_profiles.json"
)


class SetupProfileStore:
    """Manages historical setup DNA profiles for pattern matching."""

    def __init__(self, path: str = None):
        if path is None:
            from ml.config import get_config
            path = os.path.join(
                get_config().get("model_dir", os.path.join(os.path.dirname(__file__), "models")),
                "setup_profiles.json"
            )
        self._path = path
        self._profiles = self._load()
        self._id_set = {p["setup_id"] for p in self._profiles}

    # ── Public API ──────────────────────────────────────────────

    def add_profile(self, setup_id: str, dna: dict, outcome: str,
                    pnl_rr: float, mfe: float = None, mae: float = None) -> None:
        """Add a resolved trade profile. Deduplicates by setup_id."""
        if setup_id in self._id_set:
            return

        profile = {
            "setup_id": setup_id,
            "dna": dna,
            "outcome": outcome,
            "pnl_rr": pnl_rr,
            "mfe": mfe,
            "mae": mae,
            "added_at": datetime.now(timezone.utc).isoformat(),
        }
        self._profiles.append(profile)
        self._id_set.add(setup_id)
        self._save()

    def find_similar(self, dna: dict, top_k: int = 5,
                     min_similarity: float = 0.5) -> list[tuple[dict, float]]:
        """Find the top-k most similar historical profiles.

        Returns:
            List of (profile_dict, similarity_score) sorted by score descending.
        """
        scored = []
        for profile in self._profiles:
            sim = compute_similarity(dna, profile["dna"])
            if sim >= min_similarity:
                scored.append((profile, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def get_conditional_stats(self, dna: dict) -> dict:
        """Compute conditional statistics from similar setups.

        Uses top-20 similar profiles (min_similarity=0.5) for statistical
        relevance. Returns empty stats if fewer than 3 matches.

        Returns:
            Dict with match_count, win_rate, avg_rr, best_outcome,
            avg_mfe, avg_mae, top_matches.
        """
        matches = self.find_similar(dna, top_k=20, min_similarity=0.5)

        if len(matches) < 3:
            return {
                "match_count": len(matches),
                "win_rate": 0.0,
                "avg_rr": 0.0,
                "best_outcome": None,
                "avg_mfe": None,
                "avg_mae": None,
                "top_matches": [],
            }

        outcomes = [m[0]["outcome"] for m in matches]
        wins = sum(1 for o in outcomes if o.startswith("tp"))
        win_rate = wins / len(outcomes) if outcomes else 0.0

        rrs = [m[0]["pnl_rr"] for m in matches if m[0].get("pnl_rr") is not None]
        avg_rr = sum(rrs) / len(rrs) if rrs else 0.0

        mfes = [m[0]["mfe"] for m in matches if m[0].get("mfe") is not None]
        avg_mfe = sum(mfes) / len(mfes) if mfes else None

        maes = [m[0]["mae"] for m in matches if m[0].get("mae") is not None]
        avg_mae = sum(maes) / len(maes) if maes else None

        outcome_counts = Counter(outcomes)
        best_outcome = outcome_counts.most_common(1)[0][0] if outcome_counts else None

        top_matches = []
        for profile, sim in matches[:3]:
            top_matches.append({
                "direction": profile["dna"].get("direction", "?"),
                "killzone": profile["dna"].get("killzone", "?"),
                "timeframe": profile["dna"].get("timeframe", "?"),
                "outcome": profile["outcome"],
                "pnl_rr": profile.get("pnl_rr", 0),
                "similarity": round(sim * 100, 0),
            })

        return {
            "match_count": len(matches),
            "win_rate": round(win_rate, 4),
            "avg_rr": round(avg_rr, 2),
            "best_outcome": best_outcome,
            "avg_mfe": round(avg_mfe, 2) if avg_mfe is not None else None,
            "avg_mae": round(avg_mae, 2) if avg_mae is not None else None,
            "top_matches": top_matches,
        }

    def get_learned_rules(self, min_samples: int = 20) -> list[str]:
        """Generate human-readable rules from profile data.

        Groups profiles by key feature combinations and surfaces groups
        where the win rate significantly differs (>15pp) from baseline.

        Returns:
            List of rule strings sorted by |win_rate difference| descending.
        """
        if len(self._profiles) < min_samples:
            return []

        # Baseline win rate
        total = len(self._profiles)
        total_wins = sum(1 for p in self._profiles if p["outcome"].startswith("tp"))
        baseline_wr = total_wins / total if total else 0.0

        # Feature combinations to group by
        groupings = [
            (("killzone", "has_sweep"), "{killzone} + sweep={has_sweep}"),
            (("killzone", "timeframe"), "{killzone} {timeframe}"),
            (("ob_strength", "has_sweep"), "{ob_strength} OB + sweep={has_sweep}"),
            (("structure_type", "premium_discount"), "{structure_type} in {premium_discount}"),
            (("killzone", "ob_strength", "has_fvg"), "{killzone} + {ob_strength} OB + FVG={has_fvg}"),
        ]

        rules = []
        for keys, template in groupings:
            groups = defaultdict(list)
            for p in self._profiles:
                dna = p["dna"]
                group_key = tuple(str(dna.get(k, "?")) for k in keys)
                groups[group_key].append(p)

            for group_key, members in groups.items():
                if len(members) < min_samples:
                    continue

                group_wins = sum(1 for m in members if m["outcome"].startswith("tp"))
                group_wr = group_wins / len(members)
                diff_pp = (group_wr - baseline_wr) * 100

                if abs(diff_pp) < 15:
                    continue

                # Format rule
                values = dict(zip(keys, group_key))
                label = template.format(**values)
                direction = "above" if diff_pp > 0 else "below"
                rules.append((
                    abs(diff_pp),
                    f"{label}: {group_wr:.0%} WR ({len(members)} trades) "
                    f"— {abs(diff_pp):.0f}pp {direction} {baseline_wr:.0%} baseline"
                ))

        rules.sort(key=lambda x: x[0], reverse=True)
        return [r[1] for r in rules]

    def profile_count(self) -> int:
        """Return number of stored profiles."""
        return len(self._profiles)

    # ── Persistence ─────────────────────────────────────────────

    def _load(self) -> list[dict]:
        if not os.path.exists(self._path):
            return []
        try:
            with open(self._path) as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, IOError):
            return []

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        with open(self._path, "w") as f:
            json.dump(self._profiles, f, indent=None, separators=(",", ":"))
