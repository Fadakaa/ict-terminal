"""Backfill setup DNA profiles from resolved scanner trades.

One-time seed script that reads all resolved setups from scanner_db,
encodes their DNA, and stores profiles for pattern matching.

Usage:
    python -m ml.backfill_profiles
"""
import json
import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.scanner_db import ScannerDB
from ml.setup_dna import encode_setup_dna
from ml.setup_profiles import SetupProfileStore


def backfill_profiles() -> dict:
    """Backfill setup DNA profiles from all resolved scanner trades.

    Returns:
        Summary dict with counts and any learned rules.
    """
    sdb = ScannerDB()
    trades = sdb.get_resolved_setups()

    store = SetupProfileStore()
    initial_count = store.profile_count()

    added = 0
    skipped = 0
    errors = 0

    for trade in trades:
        setup_id = trade.get("id")
        if not setup_id:
            skipped += 1
            continue

        # Parse stored JSON
        try:
            analysis = json.loads(trade.get("analysis_json") or "{}")
        except (json.JSONDecodeError, TypeError):
            analysis = {}

        try:
            calibration = json.loads(trade.get("calibration_json") or "{}")
        except (json.JSONDecodeError, TypeError):
            calibration = {}

        outcome = trade.get("outcome", "")
        pnl_rr = trade.get("pnl_rr") or 0.0
        timeframe = trade.get("timeframe", "1h")
        killzone = trade.get("killzone", "Off")

        # Skip expired/invalid outcomes
        if not outcome or outcome == "expired":
            skipped += 1
            continue

        try:
            dna = encode_setup_dna(analysis, calibration, timeframe, killzone)
            store.add_profile(setup_id, dna, outcome, pnl_rr)
            added += 1
        except Exception as e:
            errors += 1
            print(f"  Error encoding {setup_id}: {e}")

    final_count = store.profile_count()
    new_profiles = final_count - initial_count

    # Generate learned rules from the seeded data
    rules = store.get_learned_rules(min_samples=20)

    return {
        "trades_found": len(trades),
        "added": added,
        "new_profiles": new_profiles,
        "skipped": skipped,
        "errors": errors,
        "total_profiles": final_count,
        "learned_rules": rules,
    }


def main():
    print("=" * 60)
    print("Setup DNA Profile Backfill")
    print("=" * 60)

    result = backfill_profiles()

    print(f"\nResolved trades found: {result['trades_found']}")
    print(f"Processed:            {result['added']}")
    print(f"New profiles added:   {result['new_profiles']}")
    print(f"Skipped (expired/no outcome): {result['skipped']}")
    print(f"Errors:               {result['errors']}")
    print(f"Total profiles in store: {result['total_profiles']}")

    rules = result["learned_rules"]
    if rules:
        print(f"\nLearned Rules ({len(rules)}):")
        for rule in rules:
            print(f"  - {rule}")
    else:
        print("\nNo learned rules yet (need more data or wider WR divergence).")

    print("\nDone.")


if __name__ == "__main__":
    main()
