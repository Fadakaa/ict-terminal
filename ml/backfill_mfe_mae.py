"""Backfill MFE/MAE for resolved scanner setups using historical 5min candles.

Fetches 5min candles from OANDA for the full date range once, then slices
per-setup by timestamp — avoids making hundreds of individual API calls.

Run: python -m ml.backfill_mfe_mae
"""

import json
import logging
import sqlite3
from datetime import datetime, timedelta

from ml.data_providers import OandaProvider
from ml.entry_placement import compute_live_mfe_mae
from ml.scanner_db import ScannerDB
from ml.config import get_config

logger = logging.getLogger(__name__)


def _get_atr(row: sqlite3.Row) -> float:
    """Extract ATR from calibration_json or estimate from SL distance."""
    atr = 1.0
    try:
        cal = json.loads(row["calibration_json"] or "{}")
        vol_ctx = cal.get("volatility_context") or {}
        if vol_ctx.get("atr"):
            atr = float(vol_ctx["atr"])
    except Exception:
        pass

    if atr <= 0 or atr == 1.0:
        entry = row["entry_price"] or 0
        sl = row["sl_price"] or 0
        sl_dist = abs(entry - sl) if (entry and sl) else 0
        if sl_dist > 0:
            atr = sl_dist / 2.0

    return atr


def backfill_mfe_mae(db: ScannerDB) -> dict:
    """Backfill mfe_atr and mae_atr for all resolved setups missing them."""
    cfg = get_config()

    # 1. Get setups needing backfill
    with db._conn() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, created_at, resolved_at, entry_price, sl_price, "
            "direction, calibration_json, outcome "
            "FROM scanner_setups "
            "WHERE outcome IS NOT NULL AND mfe_atr IS NULL "
            "AND entry_price IS NOT NULL AND resolved_at IS NOT NULL "
            "ORDER BY created_at"
        ).fetchall()

    if not rows:
        return {"total": 0, "updated": 0, "skipped": 0, "error": 0}

    # 2. Determine date range for bulk candle fetch
    def parse_ts(ts_str):
        for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
            try:
                return datetime.strptime(ts_str[:26], fmt)
            except ValueError:
                continue
        return None

    earliest = min(parse_ts(r["created_at"]) for r in rows)
    latest = max(parse_ts(r["resolved_at"]) for r in rows)

    # Add buffer
    fetch_start = earliest - timedelta(hours=1)
    fetch_end = latest + timedelta(hours=1)

    print(f"Fetching 5min candles: {fetch_start.isoformat()} to {fetch_end.isoformat()}")
    print(f"Setups to backfill: {len(rows)}")

    # 3. Fetch all 5min candles in one batch
    provider = OandaProvider(
        account_id=cfg["oanda_account_id"],
        access_token=cfg["oanda_access_token"],
    )

    candles = provider.fetch_candles(
        symbol="XAU/USD",
        interval="5min",
        start_date=fetch_start,
        end_date=fetch_end,
    )

    if not candles:
        print("ERROR: No candles returned from OANDA")
        return {"total": len(rows), "updated": 0, "skipped": 0, "error": len(rows)}

    print(f"Fetched {len(candles)} 5min candles")

    # 4. Index candles by datetime for fast slicing
    # Parse candle datetimes once
    candle_times = []
    for c in candles:
        dt_str = c["datetime"]
        # Format: "2026-03-16 00:30:00"
        dt = datetime.strptime(dt_str[:19], "%Y-%m-%d %H:%M:%S")
        candle_times.append(dt)

    # 5. For each setup, slice candles and compute MFE/MAE
    updated = 0
    skipped = 0
    errors = 0

    for row in rows:
        try:
            entry_ts = parse_ts(row["created_at"])
            resolution_ts = parse_ts(row["resolved_at"])
            entry_price = row["entry_price"]
            direction = row["direction"] or "long"

            if not entry_ts or not resolution_ts or not entry_price:
                skipped += 1
                continue

            atr = _get_atr(row)

            # Find candles in [entry_ts, resolution_ts] window
            trade_candles = []
            for i, ct in enumerate(candle_times):
                if ct >= entry_ts and ct <= resolution_ts:
                    trade_candles.append(candles[i])

            if not trade_candles:
                # Try wider window — entry might be between candles
                for i, ct in enumerate(candle_times):
                    if ct >= (entry_ts - timedelta(minutes=10)) and ct <= (resolution_ts + timedelta(minutes=10)):
                        trade_candles.append(candles[i])

            if not trade_candles:
                skipped += 1
                continue

            result = compute_live_mfe_mae(trade_candles, entry_price, direction, atr)

            with db._conn() as conn:
                conn.execute(
                    "UPDATE scanner_setups SET mfe_atr=?, mae_atr=? WHERE id=?",
                    (result["mfe_atr"], result["mae_atr"], row["id"])
                )
            updated += 1

        except Exception as e:
            logger.warning("Failed to backfill setup %s: %s", row["id"], e)
            errors += 1

    return {
        "total": len(rows),
        "updated": updated,
        "skipped": skipped,
        "error": errors,
        "candles_fetched": len(candles),
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s %(levelname)s: %(message)s",
    )

    db = ScannerDB()
    result = backfill_mfe_mae(db)
    print(f"\nBackfill results: {json.dumps(result, indent=2)}")

    # Print MFE/MAE distribution
    with db._conn() as conn:
        conn.row_factory = sqlite3.Row
        stats = conn.execute(
            "SELECT outcome, COUNT(*) as cnt, "
            "ROUND(AVG(mfe_atr), 2) as avg_mfe, "
            "ROUND(AVG(mae_atr), 2) as avg_mae "
            "FROM scanner_setups "
            "WHERE mfe_atr IS NOT NULL "
            "GROUP BY outcome ORDER BY cnt DESC"
        ).fetchall()

    print("\nMFE/MAE by outcome:")
    print(f"  {'outcome':<12} {'count':>5} {'avg_mfe':>8} {'avg_mae':>8} {'mfe/mae':>8}")
    for r in stats:
        ratio = r["avg_mfe"] / r["avg_mae"] if r["avg_mae"] > 0 else 0
        print(f"  {r['outcome']:<12} {r['cnt']:>5} {r['avg_mfe']:>8.2f} {r['avg_mae']:>8.2f} {ratio:>8.2f}")

    # Show how many now have both zone + MFE/MAE
    with db._conn() as conn:
        both = conn.execute(
            "SELECT COUNT(*) FROM scanner_setups "
            "WHERE outcome IS NOT NULL AND mfe_atr IS NOT NULL "
            "AND entry_zone_position IS NOT NULL"
        ).fetchone()[0]
    print(f"\nSetups with both zone position + MFE/MAE: {both}")
