"""Compare Claude's OB/FVG annotations vs mechanical detectors on resolved trades.

Reads all resolved setups from scanner.db, checks if entry was at a Claude-
identified OB/FVG zone, and computes win rates & P&L by annotation presence.

Can run standalone:  python -m ml.analyse_annotations
Or via API:          GET /annotations/analysis
"""
import json
import os
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.config import get_config


def _entry_in_zone(entry_price: float, zones: list[dict], tolerance_atr: float = 0.0,
                   atr: float = 1.0) -> dict | None:
    """Check if entry price is inside (or within tolerance of) any zone."""
    tol = tolerance_atr * atr
    for z in zones:
        high = z.get("high", 0)
        low = z.get("low", 0)
        if low - tol <= entry_price <= high + tol:
            return z
    return None


def _is_win(outcome: str) -> bool:
    return outcome and outcome.startswith("tp")


def _avg(lst):
    return sum(lst) / len(lst) if lst else 0


def _wr(wins, total):
    return round(wins / total * 100, 1) if total else 0


def analyse(db_path: str | None = None) -> dict:
    """Run the full annotation analysis and return structured results.

    Returns a dict with all stats, breakdowns, and a recommendation string.
    """
    if db_path is None:
        cfg = get_config()
        db_path = cfg.get("db_path")
    if not db_path or not os.path.exists(db_path):
        return {"error": f"Database not found at {db_path}", "total": 0}

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    rows = conn.execute("""
        SELECT id, direction, entry_price, sl_price, calibrated_sl,
               outcome, pnl_rr, analysis_json, calibration_json,
               timeframe, killzone, setup_quality, created_at, status
        FROM scanner_setups
        WHERE outcome IS NOT NULL AND outcome != 'expired'
              AND entry_price IS NOT NULL AND analysis_json IS NOT NULL
        ORDER BY created_at
    """).fetchall()

    if not rows:
        conn.close()
        return {"error": "No resolved trades found", "total": 0}

    # --- Aggregate stats ---
    stats = {
        "claude_ob": {"at_zone": 0, "wins": 0, "losses": 0},
        "claude_fvg": {"at_zone": 0, "wins": 0, "losses": 0},
        "claude_ob_or_fvg": {"at_zone": 0, "wins": 0, "losses": 0},
        "no_claude_zone": {"count": 0, "wins": 0, "losses": 0},
        "total_wins": 0,
        "total_losses": 0,
        "total": 0,
    }

    trade_details = []

    for row in rows:
        try:
            analysis = json.loads(row["analysis_json"]) if row["analysis_json"] else {}
        except (json.JSONDecodeError, TypeError):
            continue

        entry_price = row["entry_price"]
        outcome = row["outcome"]
        direction = row["direction"]
        is_w = _is_win(outcome)

        stats["total"] += 1
        if is_w:
            stats["total_wins"] += 1
        else:
            stats["total_losses"] += 1

        claude_obs = analysis.get("orderBlocks") or []
        claude_fvgs = analysis.get("fvgs") or []

        claude_ob_match = _entry_in_zone(entry_price, claude_obs)
        claude_fvg_match = _entry_in_zone(entry_price,
                                          [f for f in claude_fvgs if not f.get("filled")])

        if claude_ob_match:
            stats["claude_ob"]["at_zone"] += 1
            if is_w:
                stats["claude_ob"]["wins"] += 1
            else:
                stats["claude_ob"]["losses"] += 1

        if claude_fvg_match:
            stats["claude_fvg"]["at_zone"] += 1
            if is_w:
                stats["claude_fvg"]["wins"] += 1
            else:
                stats["claude_fvg"]["losses"] += 1

        at_any = claude_ob_match or claude_fvg_match
        if at_any:
            stats["claude_ob_or_fvg"]["at_zone"] += 1
            if is_w:
                stats["claude_ob_or_fvg"]["wins"] += 1
            else:
                stats["claude_ob_or_fvg"]["losses"] += 1
        else:
            stats["no_claude_zone"]["count"] += 1
            if is_w:
                stats["no_claude_zone"]["wins"] += 1
            else:
                stats["no_claude_zone"]["losses"] += 1

        trade_details.append({
            "id": row["id"],
            "outcome": outcome,
            "win": is_w,
            "pnl_rr": row["pnl_rr"] or 0,
            "entry": entry_price,
            "direction": direction,
            "claude_obs": len(claude_obs),
            "claude_fvgs": len(claude_fvgs),
            "at_claude_ob": bool(claude_ob_match),
            "at_claude_fvg": bool(claude_fvg_match),
            "quality": row["setup_quality"],
            "killzone": row["killzone"],
        })

    conn.close()

    total = stats["total"]

    # --- Build breakdowns ---
    by_quality = defaultdict(lambda: {"wins": 0, "losses": 0})
    for t in trade_details:
        q = t["quality"] or "?"
        if t["win"]:
            by_quality[q]["wins"] += 1
        else:
            by_quality[q]["losses"] += 1

    by_killzone = defaultdict(lambda: {"wins": 0, "losses": 0})
    for t in trade_details:
        kz = t["killzone"] or "Off"
        if t["win"]:
            by_killzone[kz]["wins"] += 1
        else:
            by_killzone[kz]["losses"] += 1

    by_ob_count = defaultdict(lambda: {"wins": 0, "losses": 0})
    for t in trade_details:
        n = min(t["claude_obs"], 4)
        label = f"{n}+" if n >= 4 else str(n)
        if t["win"]:
            by_ob_count[label]["wins"] += 1
        else:
            by_ob_count[label]["losses"] += 1

    # P&L by annotation presence
    at_ob_pnl = [t["pnl_rr"] for t in trade_details if t["at_claude_ob"]]
    no_ob_pnl = [t["pnl_rr"] for t in trade_details if not t["at_claude_ob"]]
    at_fvg_pnl = [t["pnl_rr"] for t in trade_details if t["at_claude_fvg"]]
    no_fvg_pnl = [t["pnl_rr"] for t in trade_details if not t["at_claude_fvg"]]

    # --- Recommendation ---
    zone_s = stats["claude_ob_or_fvg"]
    zone_n = zone_s["at_zone"]
    zone_wr = _wr(zone_s["wins"], zone_n)
    nz = stats["no_claude_zone"]
    nozone_n = nz["count"]
    nozone_wr = _wr(nz["wins"], nozone_n)

    if zone_n >= 5 and nozone_n >= 5:
        if zone_wr > nozone_wr + 10:
            recommendation = ("KEEP_CLAUDE",
                              "Claude's zones ADD value — trades at Claude OB/FVG win "
                              f"{zone_wr:.0f}% vs {nozone_wr:.0f}% without. "
                              "Supplement with mechanical detectors to catch misses.")
        elif nozone_wr > zone_wr + 10:
            recommendation = ("REPLACE_CLAUDE",
                              "Claude's zones show NEGATIVE correlation — trades NOT at "
                              f"Claude zones win {nozone_wr:.0f}% vs {zone_wr:.0f}% at zones. "
                              "Replace with mechanical detectors.")
        else:
            recommendation = ("MERGE",
                              "No significant difference in zone vs no-zone win rates "
                              f"({zone_wr:.0f}% vs {nozone_wr:.0f}%). "
                              "Use mechanical detectors as primary, keep Claude for context.")
    else:
        recommendation = ("INSUFFICIENT_DATA",
                          f"Only {zone_n} zone trades and {nozone_n} no-zone trades. "
                          "Default to MERGE strategy until more data.")

    result = {
        "total": total,
        "overall": {
            "wins": stats["total_wins"],
            "losses": stats["total_losses"],
            "win_rate": _wr(stats["total_wins"], total),
        },
        "claude_annotations": {
            "at_ob": {
                "count": stats["claude_ob"]["at_zone"],
                "wins": stats["claude_ob"]["wins"],
                "losses": stats["claude_ob"]["losses"],
                "win_rate": _wr(stats["claude_ob"]["wins"], stats["claude_ob"]["at_zone"]),
                "avg_pnl_rr": round(_avg(at_ob_pnl), 2),
            },
            "at_fvg": {
                "count": stats["claude_fvg"]["at_zone"],
                "wins": stats["claude_fvg"]["wins"],
                "losses": stats["claude_fvg"]["losses"],
                "win_rate": _wr(stats["claude_fvg"]["wins"], stats["claude_fvg"]["at_zone"]),
                "avg_pnl_rr": round(_avg(at_fvg_pnl), 2),
            },
            "at_any_zone": {
                "count": zone_n,
                "wins": zone_s["wins"],
                "losses": zone_s["losses"],
                "win_rate": zone_wr,
            },
            "no_zone": {
                "count": nozone_n,
                "wins": nz["wins"],
                "losses": nz["losses"],
                "win_rate": nozone_wr,
            },
        },
        "pnl_by_annotation": {
            "at_ob_avg_rr": round(_avg(at_ob_pnl), 2),
            "not_at_ob_avg_rr": round(_avg(no_ob_pnl), 2),
            "at_fvg_avg_rr": round(_avg(at_fvg_pnl), 2),
            "not_at_fvg_avg_rr": round(_avg(no_fvg_pnl), 2),
        },
        "by_quality": {q: {"wins": s["wins"], "losses": s["losses"],
                           "win_rate": _wr(s["wins"], s["wins"] + s["losses"])}
                       for q, s in sorted(by_quality.items())},
        "by_killzone": {kz: {"wins": s["wins"], "losses": s["losses"],
                             "win_rate": _wr(s["wins"], s["wins"] + s["losses"])}
                        for kz, s in sorted(by_killzone.items())},
        "by_ob_count": {k: {"wins": s["wins"], "losses": s["losses"],
                            "win_rate": _wr(s["wins"], s["wins"] + s["losses"])}
                        for k, s in sorted(by_ob_count.items())},
        "recommendation": {
            "action": recommendation[0],
            "detail": recommendation[1],
        },
        "trades": trade_details,
    }
    return result


def main():
    """CLI entry point — prints a human-readable report."""
    result = analyse()

    if result.get("error"):
        print(result["error"])
        return

    total = result["total"]
    o = result["overall"]
    print(f"\n{'='*70}")
    print(f"  OB/FVG Annotation Analysis — {total} resolved trades")
    print(f"{'='*70}\n")
    print(f"Overall: {o['wins']}W / {o['losses']}L ({o['win_rate']}% WR)\n")

    print("─" * 50)
    print("CLAUDE'S ANNOTATIONS")
    print("─" * 50)
    ca = result["claude_annotations"]
    for label, key in [("Entry at Claude OB", "at_ob"),
                        ("Entry at Claude FVG", "at_fvg"),
                        ("Entry at Claude OB or FVG", "at_any_zone")]:
        s = ca[key]
        n = s["count"]
        if n:
            print(f"  {label}: {n}/{total} ({n/total*100:.0f}%) — "
                  f"{s['wins']}W/{s['losses']}L ({s['win_rate']}% WR)")
        else:
            print(f"  {label}: 0 trades")

    nz = ca["no_zone"]
    if nz["count"]:
        print(f"  NOT at any Claude zone: {nz['count']}/{total} — "
              f"{nz['wins']}W/{nz['losses']}L ({nz['win_rate']}% WR)")

    print(f"\n{'─'*50}")
    print("WIN RATE BY SETUP QUALITY")
    print("─" * 50)
    for q, s in result["by_quality"].items():
        n = s["wins"] + s["losses"]
        print(f"  Grade {q}: {s['wins']}W/{s['losses']}L ({s['win_rate']}% WR) — {n} trades")

    print(f"\n{'─'*50}")
    print("WIN RATE BY KILLZONE")
    print("─" * 50)
    for kz, s in result["by_killzone"].items():
        n = s["wins"] + s["losses"]
        print(f"  {kz}: {s['wins']}W/{s['losses']}L ({s['win_rate']}% WR) — {n} trades")

    print(f"\n{'─'*50}")
    print("AVERAGE P&L (R) BY ANNOTATION PRESENCE")
    print("─" * 50)
    p = result["pnl_by_annotation"]
    print(f"  At Claude OB:      avg {p['at_ob_avg_rr']:+.2f}R")
    print(f"  NOT at Claude OB:  avg {p['not_at_ob_avg_rr']:+.2f}R")
    print(f"  At Claude FVG:     avg {p['at_fvg_avg_rr']:+.2f}R")
    print(f"  NOT at Claude FVG: avg {p['not_at_fvg_avg_rr']:+.2f}R")

    print(f"\n{'─'*50}")
    print("WIN RATE BY NUMBER OF CLAUDE OBs DETECTED")
    print("─" * 50)
    for k, s in result["by_ob_count"].items():
        n = s["wins"] + s["losses"]
        print(f"  {k} OBs: {s['wins']}W/{s['losses']}L ({s['win_rate']}% WR) — {n} trades")

    rec = result["recommendation"]
    print(f"\n{'='*70}")
    print(f"  RECOMMENDATION: {rec['action']}")
    print(f"{'='*70}")
    print(f"\n  {rec['detail']}\n")


if __name__ == "__main__":
    main()
