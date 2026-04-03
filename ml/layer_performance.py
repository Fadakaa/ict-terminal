"""Per-layer calibration performance tracking.

For every resolved trade, retroactively checks what each layer's SL
would have done. Learns optimal SL distance by setup quality, killzone,
and volatility regime.
"""
import json
import logging
import os
from datetime import datetime

from ml.config import get_config

logger = logging.getLogger(__name__)


class LayerPerformanceTracker:
    """Track and score each calibration layer's actual value-add."""

    LAYERS = ("claude", "volatility", "v1_session", "bayesian",
              "autogluon", "historical", "floor")

    def __init__(self, config: dict = None):
        self.cfg = config or get_config()
        self._path = os.path.join(
            self.cfg.get("model_dir", "ml/models"), "layer_performance.json"
        )
        self._stats = self._load()

    def _load(self) -> dict:
        if os.path.exists(self._path):
            try:
                with open(self._path) as f:
                    return json.load(f)
            except Exception:
                pass
        return {"layers": {}, "segments": {}, "updated_at": None, "total_trades": 0}

    def _save(self):
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        with open(self._path, "w") as f:
            json.dump(self._stats, f, indent=2)

    def ingest_trade(self, calibration_json: dict, outcome: str,
                     mae_atr: float, mfe_atr: float, entry_price: float,
                     atr: float, setup_grade: str = "", killzone: str = ""):
        """Score a resolved trade against all calibration layers.

        For each layer, records:
        - Whether its SL would have survived (MAE < SL distance)
        - Whether it was the tightest survivor (best efficiency)
        - The actual SL distance it suggested
        """
        if not calibration_json or atr <= 0 or not entry_price:
            return

        is_win = outcome in ("tp1", "tp2", "tp3", "runner")

        # Reconstruct per-layer SL distances from calibration_json
        layers = self._extract_layer_distances(
            calibration_json, entry_price, atr
        )

        # Score each layer
        survivors = []
        for layer, sl_dist in layers.items():
            if sl_dist <= 0:
                continue

            survived = mae_atr < sl_dist

            # Update layer stats
            if layer not in self._stats["layers"]:
                self._stats["layers"][layer] = self._empty_layer_stats()

            ls = self._stats["layers"][layer]
            ls["total"] += 1
            ls["sl_dist_sum"] += sl_dist
            if is_win:
                ls["wins"] += 1
            if survived:
                ls["survived"] += 1
                survivors.append((layer, sl_dist))
                if is_win:
                    ls["survived_wins"] += 1

            # Segment stats: grade × killzone
            seg_key = f"{setup_grade}_{killzone}" if setup_grade and killzone else None
            if seg_key:
                segs = self._stats.setdefault("segments", {})
                if seg_key not in segs:
                    segs[seg_key] = {}
                if layer not in segs[seg_key]:
                    segs[seg_key][layer] = self._empty_layer_stats()
                seg_ls = segs[seg_key][layer]
                seg_ls["total"] += 1
                seg_ls["sl_dist_sum"] += sl_dist
                if is_win:
                    seg_ls["wins"] += 1
                if survived:
                    seg_ls["survived"] += 1
                    if is_win:
                        seg_ls["survived_wins"] += 1

        # Mark tightest survivor (both global and per-segment)
        if survivors:
            tightest = min(survivors, key=lambda x: x[1])
            self._stats["layers"][tightest[0]]["tightest_survivor"] += 1

            seg_key = f"{setup_grade}_{killzone}" if setup_grade and killzone else None
            if seg_key and seg_key in self._stats.get("segments", {}):
                seg = self._stats["segments"][seg_key]
                if tightest[0] in seg:
                    seg[tightest[0]]["tightest_survivor"] += 1

        self._stats["total_trades"] += 1
        self._stats["updated_at"] = datetime.utcnow().isoformat()

    def _extract_layer_distances(self, cal: dict, entry_price: float,
                                 atr: float) -> dict:
        """Extract per-layer SL distances in ATR from calibration_json."""
        layers = {}

        # From layer_candidates (new format, Part D)
        for name, data in cal.get("layer_candidates", {}).items():
            dist = data.get("sl_distance_atr", 0)
            if dist > 0:
                layers[name] = dist

        # If no layer_candidates, reconstruct from legacy fields
        if not layers:
            # Claude
            claude_sl = cal.get("claude_original", {}).get("sl", 0)
            if claude_sl and entry_price:
                layers["claude"] = abs(entry_price - claude_sl) / atr

            # Calibrated (winning layer)
            cal_data = cal.get("calibrated", {})
            src = cal_data.get("sl_source", "")
            dist = cal_data.get("sl_distance_atr", 0)
            if src and dist > 0:
                layers[src] = dist

            # Volatility
            vol_ctx = cal.get("volatility_context", {})
            eff_atr = vol_ctx.get("effective_atr", 0)
            if eff_atr > 0 and atr > 0:
                layers.setdefault("volatility", 2.5 * eff_atr / atr)

            # V1 Session
            sess_ctx = cal.get("session_context", {})
            v1_p95 = sess_ctx.get("v1_p95_drawdown", 0)
            if v1_p95 > 0:
                layers.setdefault("v1_session", v1_p95 * 1.1)

        # Always include the floor for comparison
        floor = self.cfg.get("sl_floor_atr", 3.0)
        layers.setdefault("floor", floor)

        return layers

    @staticmethod
    def _empty_layer_stats() -> dict:
        return {
            "total": 0, "wins": 0, "survived": 0, "survived_wins": 0,
            "tightest_survivor": 0, "sl_dist_sum": 0,
        }

    def get_layer_report(self) -> dict:
        """Generate human-readable layer performance report."""
        report = {}
        for layer, ls in self._stats.get("layers", {}).items():
            if ls["total"] < 5:
                continue
            report[layer] = {
                "trades": ls["total"],
                "survival_rate": round(ls["survived"] / ls["total"], 3),
                "win_survival_rate": (
                    round(ls["survived_wins"] / ls["wins"], 3)
                    if ls["wins"] > 0 else 0
                ),
                "avg_sl_atr": round(ls["sl_dist_sum"] / ls["total"], 2),
                "efficiency_rate": round(
                    ls["tightest_survivor"] / ls["total"], 3
                ),
            }
        return report

    def get_adaptive_floor(self, grade: str, killzone: str,
                           default: float = 3.0) -> float:
        """Get the learned SL floor for a specific segment.

        Finds the layer with the highest tightest-survivor rate in this
        segment and returns its average SL distance.
        Falls back to default if insufficient data.
        """
        seg_key = f"{grade}_{killzone}"
        seg_data = self._stats.get("segments", {}).get(seg_key, {})

        best_layer = None
        best_efficiency = 0

        for layer, ls in seg_data.items():
            if ls["total"] < 20:
                continue
            eff = ls["tightest_survivor"] / ls["total"]
            if eff > best_efficiency:
                best_efficiency = eff
                best_layer = layer

        if best_layer:
            avg_sl = seg_data[best_layer]["sl_dist_sum"] / seg_data[best_layer]["total"]
            return max(2.0, min(6.0, avg_sl))

        return default

    def full_recompute(self, db) -> dict:
        """Recompute all stats from scanner_db resolved trades."""
        import sqlite3

        self._stats = {
            "layers": {}, "segments": {},
            "updated_at": None, "total_trades": 0,
        }

        with db._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT calibration_json, outcome, mae_atr, mfe_atr, "
                "entry_price, setup_quality, killzone "
                "FROM scanner_setups "
                "WHERE outcome IS NOT NULL AND calibration_json IS NOT NULL "
                "AND mae_atr IS NOT NULL AND mae_atr > 0"
            ).fetchall()

        for row in rows:
            try:
                cal = json.loads(row["calibration_json"])
                # ATR from volatility context
                atr = cal.get("volatility_context", {}).get("atr_14", 0)
                if atr <= 0:
                    # Fallback: estimate from calibrated SL
                    cal_data = cal.get("calibrated", {})
                    sl_dist_atr = cal_data.get("sl_distance_atr", 0)
                    sl_price = cal_data.get("sl", 0)
                    entry = cal_data.get("entry", 0) or row["entry_price"]
                    if sl_dist_atr > 0 and sl_price and entry:
                        atr = abs(entry - sl_price) / sl_dist_atr
                    else:
                        continue

                self.ingest_trade(
                    calibration_json=cal,
                    outcome=row["outcome"],
                    mae_atr=row["mae_atr"],
                    mfe_atr=row["mfe_atr"] or 0,
                    entry_price=row["entry_price"],
                    atr=atr,
                    setup_grade=row["setup_quality"] or "",
                    killzone=row["killzone"] or "",
                )
            except Exception:
                continue

        self._save()
        return {
            "trades_analyzed": self._stats["total_trades"],
            "layer_report": self.get_layer_report(),
        }

    def flush(self):
        self._save()
