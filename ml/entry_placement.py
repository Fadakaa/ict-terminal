"""Entry Placement Refinement — V3 Priority 2.

Learns where within OB/FVG zones entries perform best using MFE/MAE data
from resolved trades. Feeds learned optimal placement back into Sonnet's
prompt so entries are placed at statistically proven sweet spots.

Key concepts:
  - entry_position: 0.0 (shallow/worst edge) to 1.0 (deep/best edge)
  - For longs: deep = near zone_low (discount)
  - For shorts: deep = near zone_high (premium)
"""

import json
import logging
import os
from datetime import datetime

from ml.config import get_config

logger = logging.getLogger(__name__)


# ── Zone position calculation ────────────────────────────────────────

def compute_entry_position(entry_price: float, zone_high: float,
                           zone_low: float, direction: str) -> float:
    """Compute normalized entry position within a zone.

    Returns 0.0 (worst/shallow edge) to 1.0 (best/deep edge).
    For longs, deeper = closer to zone low (better fill in discount).
    For shorts, deeper = closer to zone high (better fill in premium).
    """
    zone_size = zone_high - zone_low
    if zone_size <= 0:
        return 0.5

    raw = (entry_price - zone_low) / zone_size
    raw = max(0.0, min(1.0, raw))

    if direction == "long":
        return 1.0 - raw
    else:
        return raw


def identify_entry_zone(entry_price: float, analysis_json: dict,
                        atr: float) -> dict | None:
    """Find the OB or FVG zone that the entry price is inside or nearest to.

    Priority: OB containing > FVG containing > nearest OB within 0.5 ATR >
              nearest FVG within 0.5 ATR > None.
    """
    obs = analysis_json.get("orderBlocks") or []
    fvgs = analysis_json.get("fvgs") or []

    def _make_zone(item, zone_type):
        high = item.get("high", 0)
        low = item.get("low", 0)
        return {
            "zone_type": zone_type,
            "zone_subtype": item.get("type", "unknown"),
            "zone_high": high,
            "zone_low": low,
            "zone_size": high - low,
            "zone_size_atr": (high - low) / atr if atr > 0 else 0,
        }

    # Priority 1: OB containing entry
    for ob in obs:
        if ob.get("low", 0) <= entry_price <= ob.get("high", 0):
            z = _make_zone(ob, "ob")
            z["contains_entry"] = True
            return z

    # Priority 2: FVG containing entry
    for fvg in fvgs:
        if fvg.get("low", 0) <= entry_price <= fvg.get("high", 0):
            z = _make_zone(fvg, "fvg")
            z["contains_entry"] = True
            return z

    # Priority 3-4: Nearest zone within 0.5 ATR
    best_zone = None
    best_dist = float("inf")
    threshold = 0.5 * atr if atr > 0 else 0

    for ob in obs:
        mid = (ob.get("high", 0) + ob.get("low", 0)) / 2
        dist = abs(entry_price - mid)
        if dist < best_dist and dist < threshold:
            best_dist = dist
            z = _make_zone(ob, "ob")
            z["contains_entry"] = False
            best_zone = z

    for fvg in fvgs:
        mid = (fvg.get("high", 0) + fvg.get("low", 0)) / 2
        dist = abs(entry_price - mid)
        if dist < best_dist and dist < threshold:
            best_dist = dist
            z = _make_zone(fvg, "fvg")
            z["contains_entry"] = False
            best_zone = z

    return best_zone


# ── Metrics extraction ───────────────────────────────────────────────

def extract_entry_zone_metrics(setup_row: dict) -> dict | None:
    """Extract entry zone metrics from a resolved scanner setup.

    ATR source priority:
    1. calibration_json["volatility_context"]["atr"]
    2. sl_dist / 2.0 (fallback estimate)
    """
    entry_price = setup_row.get("entry_price", 0)
    analysis_raw = setup_row.get("analysis_json")
    if not entry_price or not analysis_raw:
        return None

    try:
        analysis = json.loads(analysis_raw) if isinstance(analysis_raw, str) else analysis_raw
    except Exception:
        return None

    direction = setup_row.get("direction", "long")

    # ATR: prefer calibration, fall back to SL estimate
    atr = 1.0
    cal_raw = setup_row.get("calibration_json", "{}")
    try:
        cal = json.loads(cal_raw) if isinstance(cal_raw, str) else (cal_raw or {})
        vol_ctx = cal.get("volatility_context") or {}
        if vol_ctx.get("atr"):
            atr = float(vol_ctx["atr"])
    except Exception:
        pass

    if atr <= 0:
        sl_price = setup_row.get("sl_price", 0)
        sl_dist = abs(entry_price - sl_price) if sl_price else 0
        atr = sl_dist / 2.0 if sl_dist > 0 else 1.0

    zone = identify_entry_zone(entry_price, analysis, atr)
    if zone is None:
        return None

    position = compute_entry_position(
        entry_price, zone["zone_high"], zone["zone_low"], direction
    )

    dist_to_high = abs(entry_price - zone["zone_high"])
    dist_to_low = abs(entry_price - zone["zone_low"])
    depth_from_edge = min(dist_to_high, dist_to_low)
    depth_atr = depth_from_edge / atr if atr > 0 else 0

    return {
        "setup_id": setup_row.get("id", ""),
        "entry_price": entry_price,
        "zone_type": zone["zone_type"],
        "zone_subtype": zone["zone_subtype"],
        "zone_high": zone["zone_high"],
        "zone_low": zone["zone_low"],
        "zone_size": zone["zone_size"],
        "zone_size_atr": zone["zone_size_atr"],
        "entry_position": round(position, 4),
        "entry_depth_atr": round(depth_atr, 4),
        "contains_entry": zone["contains_entry"],
        "outcome": setup_row.get("outcome", ""),
        "mfe_atr": setup_row.get("mfe_atr", 0),
        "mae_atr": setup_row.get("mae_atr", 0),
        "pnl_rr": setup_row.get("pnl_rr", 0),
        "killzone": setup_row.get("killzone", ""),
        "timeframe": setup_row.get("timeframe", "1h"),
        "direction": direction,
    }


# ── Live MFE/MAE computation ────────────────────────────────────────

def compute_live_mfe_mae(candles: list[dict], entry_price: float,
                         direction: str, atr: float) -> dict:
    """Compute MFE/MAE from candles between entry and resolution.

    Args:
        candles: Candles covering the trade duration.
        entry_price: The entry price.
        direction: "long" or "short".
        atr: ATR for normalization.

    Returns:
        Dict with mfe, mae, mfe_atr, mae_atr.
    """
    if atr <= 0:
        atr = 1.0

    max_fav = 0.0
    max_adv = 0.0

    for c in candles:
        if direction == "long":
            fav = c["high"] - entry_price
            adv = entry_price - c["low"]
        else:
            fav = entry_price - c["low"]
            adv = c["high"] - entry_price

        max_fav = max(max_fav, fav)
        max_adv = max(max_adv, adv)

    return {
        "mfe": round(max_fav, 2),
        "mae": round(max_adv, 2),
        "mfe_atr": round(max_fav / atr, 4),
        "mae_atr": round(max_adv / atr, 4),
    }


# ── Placement statistics analyzer ────────────────────────────────────

class EntryPlacementAnalyzer:
    """Aggregates entry placement metrics and computes optimal zones."""

    def __init__(self, config: dict = None):
        self.cfg = config or get_config()
        self._stats_path = os.path.join(
            self.cfg.get("model_dir", "ml/models"), "entry_placement_stats.json"
        )
        self._stats = self._load_stats()

    def _load_stats(self) -> dict:
        if os.path.exists(self._stats_path):
            try:
                with open(self._stats_path) as f:
                    return json.load(f)
            except Exception:
                pass
        return {"metrics": [], "summary": {}, "updated_at": None}

    def _save_stats(self):
        os.makedirs(os.path.dirname(self._stats_path), exist_ok=True)
        with open(self._stats_path, "w") as f:
            json.dump(self._stats, f, indent=2)

    def ingest_metric(self, metric: dict):
        self._stats["metrics"].append(metric)

    def compute_summary(self, min_trades: int = 15) -> dict:
        """Compute placement statistics across all ingested metrics.

        Adaptive binning: 3 tertiles when < 200 trades, 5 quintiles when >= 200.
        """
        metrics = self._stats["metrics"]
        if len(metrics) < min_trades:
            return {"error": "insufficient_data", "count": len(metrics)}

        n_bins = 5 if len(metrics) >= 200 else 3

        # Overall bins
        bins = self._bin_by_position(metrics, n_bins=n_bins)

        # By direction
        long_metrics = [m for m in metrics if m.get("direction") == "long"]
        short_metrics = [m for m in metrics if m.get("direction") == "short"]
        by_direction = {}
        if len(long_metrics) >= min_trades:
            by_direction["long"] = self._bin_by_position(long_metrics, n_bins=n_bins)
        if len(short_metrics) >= min_trades:
            by_direction["short"] = self._bin_by_position(short_metrics, n_bins=n_bins)

        # By zone type
        ob_metrics = [m for m in metrics if m.get("zone_type") == "ob"]
        fvg_metrics = [m for m in metrics if m.get("zone_type") == "fvg"]
        by_zone_type = {
            "ob": self._bin_by_position(ob_metrics, n_bins=n_bins) if len(ob_metrics) >= min_trades else None,
            "fvg": self._bin_by_position(fvg_metrics, n_bins=n_bins) if len(fvg_metrics) >= min_trades else None,
        }

        # By killzone
        by_killzone = {}
        for kz in ["London", "NY_AM", "NY_PM"]:
            kz_metrics = [m for m in metrics if m.get("killzone") == kz]
            if len(kz_metrics) >= min_trades:
                by_killzone[kz] = self._bin_by_position(kz_metrics, n_bins=n_bins)

        # Optimal bin
        optimal = self._find_optimal_bin(bins)

        # Zone containment
        contained = [m for m in metrics if m.get("contains_entry")]
        outside = [m for m in metrics if not m.get("contains_entry")]
        containment = {
            "inside_zone": self._compute_group_stats(contained) if contained else None,
            "outside_zone": self._compute_group_stats(outside) if outside else None,
        }

        summary = {
            "total_trades": len(metrics),
            "n_bins": n_bins,
            "overall_bins": bins,
            "by_direction": by_direction,
            "by_zone_type": by_zone_type,
            "by_killzone": by_killzone,
            "optimal_position": optimal,
            "containment": containment,
            "updated_at": datetime.utcnow().isoformat(),
        }

        self._stats["summary"] = summary
        self._save_stats()
        return summary

    @staticmethod
    def _bin_by_position(metrics: list, n_bins: int = 3) -> list[dict]:
        if not metrics:
            return []

        bin_width = 1.0 / n_bins
        bins = []

        for i in range(n_bins):
            low = i * bin_width
            high = (i + 1) * bin_width
            label = f"{low:.2f}-{high:.2f}"

            in_bin = [m for m in metrics if low <= m.get("entry_position", -1) < high]
            if i == n_bins - 1:
                in_bin += [m for m in metrics
                           if m.get("entry_position", -1) == 1.0
                           and m not in in_bin]

            if not in_bin:
                bins.append({"range": label, "count": 0})
                continue

            wins = [m for m in in_bin
                    if m.get("outcome") in ("tp1", "tp2", "tp3", "runner")]
            avg_mfe = sum(m.get("mfe_atr", 0) for m in in_bin) / len(in_bin)
            avg_mae = sum(m.get("mae_atr", 0) for m in in_bin) / len(in_bin)
            avg_pnl = sum(m.get("pnl_rr", 0) for m in in_bin) / len(in_bin)

            bins.append({
                "range": label,
                "count": len(in_bin),
                "win_rate": round(len(wins) / len(in_bin), 4),
                "avg_mfe_atr": round(avg_mfe, 4),
                "avg_mae_atr": round(avg_mae, 4),
                "avg_pnl_rr": round(avg_pnl, 4),
                "mfe_mae_ratio": round(avg_mfe / avg_mae, 4) if avg_mae > 0 else 0,
            })

        return bins

    @staticmethod
    def _compute_group_stats(metrics: list) -> dict:
        if not metrics:
            return {}
        wins = [m for m in metrics
                if m.get("outcome") in ("tp1", "tp2", "tp3", "runner")]
        return {
            "count": len(metrics),
            "win_rate": round(len(wins) / len(metrics), 4),
            "avg_mfe_atr": round(sum(m.get("mfe_atr", 0) for m in metrics) / len(metrics), 4),
            "avg_mae_atr": round(sum(m.get("mae_atr", 0) for m in metrics) / len(metrics), 4),
        }

    @staticmethod
    def _find_optimal_bin(bins: list) -> dict:
        valid = [b for b in bins
                 if b.get("count", 0) >= 5 and b.get("mfe_mae_ratio", 0) > 0]
        if not valid:
            return {"range": "0.33-0.67", "confidence": "low",
                    "reason": "insufficient_data"}

        best = max(valid, key=lambda b: (b["mfe_mae_ratio"], b.get("win_rate", 0)))
        count = best["count"]
        return {
            "range": best["range"],
            "mfe_mae_ratio": best["mfe_mae_ratio"],
            "win_rate": best.get("win_rate", 0),
            "count": count,
            "confidence": "high" if count >= 20 else "medium" if count >= 10 else "low",
        }

    def get_placement_guidance(self) -> dict:
        """Generate prompt-injectable guidance from placement statistics."""
        summary = self._stats.get("summary", {})
        if not summary or summary.get("total_trades", 0) < 15:
            return {"status": "insufficient_data", "rules": []}

        rules = []
        optimal = summary.get("optimal_position", {})

        # Rule 1: Optimal position range
        if optimal.get("confidence") in ("high", "medium"):
            pos_range = optimal["range"]
            parts = pos_range.split("-")
            low, high = float(parts[0]), float(parts[1])
            midpoint = (low + high) / 2

            if midpoint < 0.3:
                desc = "near the zone edge (shallow)"
            elif midpoint < 0.5:
                desc = "in the outer half of the zone"
            elif midpoint < 0.7:
                desc = "at the zone midpoint"
            else:
                desc = "deep in the zone (toward SL side)"

            rules.append(
                f"Historical data ({optimal.get('count', 0)} trades): entries {desc} "
                f"(position {pos_range}) have the best risk-adjusted performance "
                f"(MFE/MAE ratio: {optimal.get('mfe_mae_ratio', 0):.2f}, "
                f"WR: {optimal.get('win_rate', 0):.0%}). "
                f"Place entries in this region of the OB/FVG zone."
            )

        # Rule 2: Zone containment
        containment = summary.get("containment", {})
        inside = containment.get("inside_zone")
        outside = containment.get("outside_zone")
        if (inside and outside
                and inside.get("count", 0) >= 10
                and outside.get("count", 0) >= 10):
            inside_wr = inside.get("win_rate", 0)
            outside_wr = outside.get("win_rate", 0)
            diff = inside_wr - outside_wr
            if abs(diff) > 0.05:
                better = "inside" if diff > 0 else "near but outside"
                rules.append(
                    f"Entries {better} the zone perform {abs(diff):.0%}pp better "
                    f"({inside_wr:.0%} vs {outside_wr:.0%} WR). "
                    f"{'Wait for price to enter the zone before triggering.' if diff > 0 else 'Entries just outside the zone edge are acceptable.'}"
                )

        # Rule 3: Zone type preference
        by_zone = summary.get("by_zone_type", {})
        ob_stats = by_zone.get("ob")
        fvg_stats = by_zone.get("fvg")
        if ob_stats and fvg_stats:
            ob_best = max((b for b in ob_stats if b.get("count", 0) >= 5),
                          key=lambda b: b.get("mfe_mae_ratio", 0), default=None)
            fvg_best = max((b for b in fvg_stats if b.get("count", 0) >= 5),
                           key=lambda b: b.get("mfe_mae_ratio", 0), default=None)
            if ob_best and fvg_best:
                ob_r = ob_best.get("mfe_mae_ratio", 0)
                fvg_r = fvg_best.get("mfe_mae_ratio", 0)
                if ob_r > fvg_r * 1.15:
                    rules.append("OB entries outperform FVG entries — prefer OB zones when both are available.")
                elif fvg_r > ob_r * 1.15:
                    rules.append("FVG entries outperform OB entries — prefer FVG zones when both are available.")

        # Rule 4: Killzone-specific placement
        by_kz = summary.get("by_killzone", {})
        overall_range = optimal.get("range", "")
        for kz_name, kz_bins in by_kz.items():
            if not kz_bins:
                continue
            kz_best = max((b for b in kz_bins if b.get("count", 0) >= 5),
                          key=lambda b: b.get("mfe_mae_ratio", 0), default=None)
            if kz_best and kz_best["range"] != overall_range:
                rules.append(
                    f"During {kz_name}: optimal entry position shifts to {kz_best['range']} "
                    f"({kz_best.get('win_rate', 0):.0%} WR, {kz_best['count']} trades)."
                )

        # Rule 5: MAE warning zones
        bins = summary.get("overall_bins", [])
        high_mae_bins = [b for b in bins
                         if b.get("avg_mae_atr", 0) > 2.0 and b.get("count", 0) >= 5]
        if high_mae_bins:
            worst = max(high_mae_bins, key=lambda b: b["avg_mae_atr"])
            rules.append(
                f"AVOID entries in the {worst['range']} position range — "
                f"average adverse excursion is {worst['avg_mae_atr']:.1f} ATR "
                f"(high drawdown risk before trade works)."
            )

        return {
            "status": "active",
            "total_trades": summary.get("total_trades", 0),
            "rules": rules[:5],
            "optimal_position": optimal,
        }
