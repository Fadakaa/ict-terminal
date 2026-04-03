"""Recent context builder — gives Claude memory of what just happened.

Queries scanner_db for recent resolutions, consumed zones, swept liquidity,
and active setups on a given timeframe. Formats into a prompt-ready text block
so Claude analyses with awareness of recent events rather than starting from zero.
"""
import json
import logging
import sqlite3
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


def build_recent_context(timeframe: str, db) -> dict:
    """Build recent context for a timeframe from scanner DB.

    Returns dict with four keys:
        recent_resolutions — Last 3 resolved setups (within 24h)
        consumed_zones     — Entry zones used by resolved setups
        swept_liquidity    — SL levels from stopped-out setups (swept)
        active_setups      — Currently pending setups on this TF

    Args:
        timeframe: e.g. "15min", "1h", "4h", "1day"
        db: ScannerDB instance
    """
    ctx = {
        "recent_resolutions": [],
        "consumed_zones": [],
        "swept_liquidity": [],
        "active_setups": [],
    }

    try:
        cutoff = (datetime.utcnow() - timedelta(hours=24)).isoformat()
        resolutions = _query_recent_resolutions(db, timeframe, cutoff, limit=3)
        ctx["recent_resolutions"] = resolutions

        for res in resolutions:
            # Consumed zones: entry zones from resolved setups
            if res.get("entry_zone_type") and res.get("entry_zone_high"):
                ctx["consumed_zones"].append({
                    "zone_type": res["entry_zone_type"],
                    "high": res["entry_zone_high"],
                    "low": res["entry_zone_low"],
                    "setup_id": res.get("id", ""),
                    "outcome": res.get("outcome", ""),
                })

            # Swept liquidity: SL levels from stopped-out setups
            if res.get("outcome") == "stopped_out" and res.get("sl_price"):
                direction = res.get("direction", "long")
                ctx["swept_liquidity"].append({
                    "level": res["sl_price"],
                    "type": "SSL" if direction == "long" else "BSL",
                    "swept_at": res.get("resolved_at", ""),
                    "setup_id": res.get("id", ""),
                })

        # Active setups on this timeframe
        pending = db.get_pending()
        ctx["active_setups"] = [
            {
                "id": s.get("id", ""),
                "direction": s.get("direction", ""),
                "entry_price": s.get("entry_price"),
                "sl_price": s.get("sl_price"),
                "tp1": s.get("tp1"),
                "setup_quality": s.get("setup_quality", ""),
                "killzone": s.get("killzone", ""),
            }
            for s in pending
            if s.get("timeframe") == timeframe
        ]

        # Phase 7: Missed/expired setups for recycling
        missed = _query_missed_setups(db, timeframe, cutoff, limit=3)
        ctx["missed_setups"] = missed

        # Phase C: Displacement-confirmed zones from narrative store
        try:
            from ml.narrative_state import NarrativeStore
            ns = NarrativeStore(db.db_path).get_current(timeframe)
            if ns and ns.get("displacement_confirmed_zones"):
                ctx["displacement_confirmed_zones"] = json.loads(
                    ns["displacement_confirmed_zones"])
        except Exception:
            pass

    except Exception as e:
        logger.warning("Failed to build recent context for %s: %s", timeframe, e)

    return ctx


def _query_recent_resolutions(db, timeframe: str, cutoff: str,
                               limit: int = 3) -> list:
    """Query recently resolved setups on a timeframe since cutoff."""
    try:
        conn = sqlite3.connect(db.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT id, direction, outcome, entry_price, sl_price,
                   calibrated_sl, tp1, tp2, tp3, resolved_at, pnl_rr,
                   entry_zone_type, entry_zone_high, entry_zone_low,
                   setup_quality, killzone
            FROM scanner_setups
            WHERE timeframe = ? AND status = 'resolved'
              AND resolved_at > ?
            ORDER BY resolved_at DESC
            LIMIT ?
        """, (timeframe, cutoff, limit)).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.warning("Recent resolutions query failed: %s", e)
        return []


def _query_missed_setups(db, timeframe: str, cutoff: str,
                         limit: int = 3) -> list:
    """Query expired or entry-missed setups on a timeframe since cutoff.

    These feed back into the prompt for thesis reassessment (Phase 7).
    """
    try:
        conn = sqlite3.connect(db.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT id, direction, outcome, entry_price, sl_price,
                   entry_zone_type, entry_zone_high, entry_zone_low,
                   setup_quality, killzone, resolved_at
            FROM scanner_setups
            WHERE timeframe = ? AND status IN ('resolved', 'expired')
              AND outcome IN ('expired', 'entry_missed')
              AND resolved_at > ?
            ORDER BY resolved_at DESC
            LIMIT ?
        """, (timeframe, cutoff, limit)).fetchall()
        conn.close()

        missed = []
        for r in rows:
            rd = dict(r)
            # Build zone description
            if rd.get("entry_zone_type") and rd.get("entry_zone_high"):
                desc = (f"{rd['entry_zone_type'].upper()} at "
                        f"{rd['entry_zone_high']:.2f}-{rd['entry_zone_low']:.2f}")
            elif rd.get("entry_price"):
                desc = f"Entry zone at {rd['entry_price']:.2f}"
            else:
                desc = "Unknown zone"
            rd["zone_description"] = desc
            missed.append(rd)

        return missed
    except Exception as e:
        logger.warning("Missed setups query failed: %s", e)
        return []


def format_recent_context(ctx: dict) -> str:
    """Format recent context dict into prompt-ready text block.

    Returns empty string if no recent activity (no noise in the prompt).
    Capped at ~500 tokens to stay within budget.
    """
    if not ctx:
        return ""

    resolutions = ctx.get("recent_resolutions", [])
    consumed = ctx.get("consumed_zones", [])
    swept = ctx.get("swept_liquidity", [])
    active = ctx.get("active_setups", [])
    missed = ctx.get("missed_setups", [])

    # Nothing to report
    if not resolutions and not active and not missed:
        return ""

    lines = ["=== RECENT CONTEXT ===", ""]

    # Recent resolutions
    if resolutions:
        for res in resolutions:
            ago = _time_ago(res.get("resolved_at", ""))
            direction = (res.get("direction") or "").upper()
            outcome = (res.get("outcome") or "").upper().replace("_", " ")
            entry = res.get("entry_price", 0)
            sl = res.get("sl_price", 0)
            rr = res.get("pnl_rr", 0)
            lines.append(f"LAST RESOLUTION: {ago}")
            lines.append(
                f"  Direction: {direction} | Outcome: {outcome} | "
                f"Entry: {_fmt_price(entry)} | SL: {_fmt_price(sl)} | "
                f"R:R: {rr:+.1f}R" if rr else
                f"  Direction: {direction} | Outcome: {outcome} | "
                f"Entry: {_fmt_price(entry)} | SL: {_fmt_price(sl)}"
            )
            break  # Only show the most recent in detail to save tokens

        if len(resolutions) > 1:
            others = resolutions[1:]
            summaries = []
            for r in others:
                d = (r.get("direction") or "?")[0].upper()
                o = (r.get("outcome") or "?").replace("stopped_out", "SL").replace("_", "")
                summaries.append(f"{d} {o}")
            lines.append(f"  Prior: {', '.join(summaries)}")

    # Consumed zones
    if consumed:
        lines.append("")
        for z in consumed:
            zt = (z.get("zone_type") or "zone").upper()
            lines.append(
                f"  CONSUMED {zt}: {_fmt_price(z.get('high'))}–"
                f"{_fmt_price(z.get('low'))} "
                f"(used by setup {z.get('setup_id', '?')}, {z.get('outcome', '?')})"
            )

    # Swept liquidity
    if swept:
        lines.append("")
        for s in swept:
            ago = _time_ago(s.get("swept_at", ""))
            lines.append(
                f"  {s.get('type', 'LIQ')} SWEPT at {_fmt_price(s.get('level'))} "
                f"({ago})"
            )

    # Implications
    if resolutions:
        latest = resolutions[0]
        if latest.get("outcome") == "stopped_out":
            lines.append("")
            lines.append("IMPLICATION: Bearish pressure broke through the entry zone. Look for:")
            lines.append("  - New displacement below the swept level (continuation)")
            lines.append("  - Failed breakdown + reclaim above zone (reversal with fresh structure)")
        elif latest.get("outcome", "").startswith("tp"):
            lines.append("")
            lines.append("IMPLICATION: Setup played out. The entry zone is now consumed. Look for:")
            lines.append("  - Fresh structure forming at different level")
            lines.append("  - Continuation if trend intact, or reversal if distribution complete")

    # Active setups
    if active:
        lines.append("")
        dirs = [f"{s.get('direction', '?').upper()} @ {_fmt_price(s.get('entry_price'))}"
                for s in active]
        lines.append(f"ACTIVE SETUPS: {', '.join(dirs)}")
    else:
        lines.append("")
        lines.append("ACTIVE SETUPS: None on this timeframe.")

    # Phase 7: Missed/expired setups — reassessment context
    if missed:
        lines.append("")
        for m in missed:
            direction = (m.get("direction") or "?").upper()
            outcome = (m.get("outcome") or "?").replace("_", " ")
            zone_desc = m.get("zone_description", f"at {_fmt_price(m.get('entry_price'))}")
            quality = m.get("setup_quality", "?")
            lines.append(
                f"MISSED SETUP: {direction} {zone_desc} (Grade {quality}, {outcome})")
        lines.append("  Reassess whether the thesis still holds at current price.")
        lines.append("  If the zone was never tested, it may still be valid.")

    # Phase C: Displacement-confirmed zones (consumed this session)
    disp_zones = ctx.get("displacement_confirmed_zones", [])
    if disp_zones:
        lines.append("")
        lines.append("DISPLACEMENT-CONFIRMED ZONES (already consumed — do not re-enter):")
        for dz in disp_zones[-3:]:  # cap at 3 to keep prompt tight
            direction = (dz.get("direction") or "?").upper()
            lines.append(
                f"  {direction} OB {_fmt_price(dz.get('zone_low'))}–"
                f"{_fmt_price(dz.get('zone_high'))} — zone triggered this session")

    lines.append("=== END RECENT CONTEXT ===")
    lines.append("")

    return "\n".join(lines)


def _time_ago(iso_str: str) -> str:
    """Convert ISO timestamp to human-readable 'X min/h ago' string."""
    if not iso_str:
        return "unknown"
    try:
        resolved = datetime.fromisoformat(iso_str)
        delta = datetime.utcnow() - resolved
        mins = int(delta.total_seconds() / 60)
        if mins < 1:
            return "just now"
        if mins < 60:
            return f"{mins} min ago"
        hours = mins // 60
        if hours < 24:
            return f"{hours}h ago"
        return f"{hours // 24}d ago"
    except (ValueError, TypeError):
        return "unknown"


def _fmt_price(price) -> str:
    """Format a price for display, handling None."""
    if price is None:
        return "?"
    try:
        return f"{float(price):,.2f}"
    except (ValueError, TypeError):
        return str(price)
