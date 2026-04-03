"""One-time backfill: compute entry zone position for existing scanner setups.

Run: python -m ml.backfill_entry_zones
"""

import json
import logging
import sqlite3

from ml.entry_placement import identify_entry_zone, compute_entry_position
from ml.scanner_db import ScannerDB

logger = logging.getLogger(__name__)


def backfill_entry_zones(db: ScannerDB) -> dict:
    """Backfill entry_zone_position for all existing setups with analysis_json."""
    with db._conn() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, entry_price, direction, sl_price, "
            "analysis_json, calibration_json "
            "FROM scanner_setups WHERE analysis_json IS NOT NULL"
        ).fetchall()

    updated = 0
    skipped = 0
    no_zone = 0

    for row in rows:
        entry_price = row["entry_price"]
        direction = row["direction"] or "long"
        if not entry_price:
            skipped += 1
            continue

        try:
            analysis = json.loads(row["analysis_json"])
        except Exception:
            skipped += 1
            continue

        # ATR: prefer calibration_json, fall back to SL estimate
        atr = 1.0
        try:
            cal = json.loads(row["calibration_json"] or "{}")
            vol_ctx = cal.get("volatility_context") or {}
            if vol_ctx.get("atr"):
                atr = float(vol_ctx["atr"])
        except Exception:
            pass

        if atr <= 0:
            sl_price = row["sl_price"] or 0
            sl_dist = abs(entry_price - sl_price) if sl_price else 0
            atr = sl_dist / 2.0 if sl_dist > 0 else 1.0

        zone = identify_entry_zone(entry_price, analysis, atr)
        if not zone:
            no_zone += 1
            continue

        position = compute_entry_position(
            entry_price, zone["zone_high"], zone["zone_low"], direction
        )

        with db._conn() as conn:
            conn.execute(
                "UPDATE scanner_setups "
                "SET entry_zone_type=?, entry_zone_high=?, "
                "entry_zone_low=?, entry_zone_position=? "
                "WHERE id=?",
                (zone["zone_type"], zone["zone_high"], zone["zone_low"],
                 round(position, 4), row["id"])
            )
            updated += 1

    result = {
        "total": len(rows),
        "updated": updated,
        "skipped": skipped,
        "no_zone": no_zone,
    }
    logger.info("Backfill complete: %s", result)
    return result


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s %(levelname)s: %(message)s",
    )

    db = ScannerDB()
    result = backfill_entry_zones(db)
    print(f"Backfill results: {json.dumps(result, indent=2)}")

    # Print zone distribution
    with db._conn() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT entry_zone_type, COUNT(*) as cnt "
            "FROM scanner_setups WHERE entry_zone_type IS NOT NULL "
            "GROUP BY entry_zone_type"
        ).fetchall()
    print("\nZone type distribution:")
    for r in rows:
        print(f"  {r['entry_zone_type']}: {r['cnt']}")

    # Print position distribution
    with db._conn() as conn:
        rows = conn.execute(
            "SELECT ROUND(entry_zone_position, 1) as pos_bin, COUNT(*) as cnt "
            "FROM scanner_setups WHERE entry_zone_position IS NOT NULL "
            "GROUP BY pos_bin ORDER BY pos_bin"
        ).fetchall()
    print("\nPosition distribution (0=shallow, 1=deep):")
    for r in rows:
        print(f"  {r[0]:.1f}: {r[1]}")
