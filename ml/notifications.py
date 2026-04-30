"""Trade notifications — macOS native + Telegram.

Unified lifecycle notification system (6 stages):
  💭 Stage 1: THESIS_FORMING    — Claude starts tracking something
  🔬 Stage 2: THESIS_CONFIRMED  — thesis survived 2+ scans, rising confidence
  🎯 Stage 3: SETUP_DETECTED    — concrete entry/SL/TPs identified
  ✅ Stage 4: ENTRY_READY        — price at entry, execute now
  📊 Stage 5: TRADE_RESOLVED     — outcome + post-resolution thesis
  ⚠️ Stage 6: THESIS_REVISED     — thesis changed direction

Legacy functions (notify_zone_prospect, notify_setup_detected, etc.) are
backward-compatible wrappers that route to notify_lifecycle() internally.
"""
import html
import json
import os
import subprocess
import logging
from pathlib import Path
from uuid import uuid4

import requests
from dotenv import load_dotenv

# Load .env BEFORE reading env vars — ensures Telegram creds are available
# regardless of which module imports notifications.py first
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)

from ml.config import get_config
from ml.env_utils import sanitize_env_secret

logger = logging.getLogger(__name__)

# Telegram bot token is embedded in the API URL (api.telegram.org/bot<TOKEN>/...)
# which goes through the same ASCII encoding path as HTTP headers — invisible
# Unicode in the env var would crash the request. See ml/env_utils.py.
TELEGRAM_TOKEN = sanitize_env_secret(os.getenv("TELEGRAM_BOT_TOKEN"))
TELEGRAM_CHAT_ID = sanitize_env_secret(os.getenv("TELEGRAM_CHAT_ID"))

if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
    logger.warning("Telegram credentials missing — check .env for TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID")

# Lot size calculation defaults (XAU/USD)
ACCOUNT_BALANCE = float(os.getenv("ACCOUNT_BALANCE", "100000"))
RISK_PCT = float(os.getenv("RISK_PCT", "0.5"))
DAILY_DD_LIMIT = float(os.getenv("DAILY_DD_LIMIT", "0.04"))  # 4% daily drawdown limit
XAUUSD_PIP_VALUE = 100  # $100 per $1 move per standard lot

# Daily drawdown tracker — DB-persisted, resets at 8pm GMT (trading day boundary)
DD_RESET_HOUR_GMT = 20  # 8pm GMT — new trading day starts here


def _get_trading_day() -> str:
    """Return the current trading day label based on 8pm GMT reset.

    Before 8pm GMT → today's date (e.g. "2026-04-07")
    After 8pm GMT  → tomorrow's date (the new trading day has started)
    """
    from datetime import datetime, timezone, timedelta
    now = datetime.now(timezone.utc)
    if now.hour >= DD_RESET_HOUR_GMT:
        return (now + timedelta(days=1)).strftime("%Y-%m-%d")
    return now.strftime("%Y-%m-%d")


def _get_dd_db_path() -> str:
    """Resolve the scanner DB path for drawdown persistence."""
    from ml.config import get_config
    return get_config().get("db_path")


def _daily_dd_remaining() -> dict:
    """Return daily drawdown budget status (DB-persisted).

    Returns dict with:
        limit_dollars: absolute daily DD limit in dollars
        used_dollars: realised P&L consumed today (negative = losses used)
        remaining_dollars: how much room left before breach
        remaining_pct: remaining as % of account
        used_pct: used as % of account
        warning: True if ≥50% of daily DD consumed
        critical: True if ≥75% of daily DD consumed
    """
    import sqlite3
    trading_day = _get_trading_day()
    realised_pnl = 0.0

    try:
        db_path = _get_dd_db_path()
        with sqlite3.connect(db_path) as conn:
            row = conn.execute(
                "SELECT trading_day, realised_pnl FROM daily_drawdown WHERE id = 1"
            ).fetchone()
            if row and row[0] == trading_day:
                realised_pnl = row[1]
            elif row and row[0] != trading_day:
                # Trading day has rolled over — reset
                conn.execute(
                    "UPDATE daily_drawdown SET trading_day = ?, realised_pnl = 0.0, "
                    "trade_count = 0, updated_at = datetime('now') WHERE id = 1",
                    (trading_day,))
                realised_pnl = 0.0
    except Exception as e:
        logger.warning("DD read from DB failed, using 0: %s", e)

    limit = ACCOUNT_BALANCE * DAILY_DD_LIMIT
    used = min(0, realised_pnl)  # Only losses count
    remaining = limit + used  # used is negative, so this subtracts
    return {
        "limit_dollars": limit,
        "used_dollars": used,
        "remaining_dollars": remaining,
        "remaining_pct": round((remaining / ACCOUNT_BALANCE) * 100, 2),
        "used_pct": round((abs(used) / ACCOUNT_BALANCE) * 100, 2),
        "warning": abs(used) >= limit * 0.50,
        "critical": abs(used) >= limit * 0.75,
    }


def record_daily_pnl(dollar_pnl: float):
    """Record a trade's P&L against the daily drawdown tracker (DB-persisted).

    Persists to the daily_drawdown table so DD survives server restarts.
    Resets automatically at 8pm GMT (new trading day boundary).
    """
    import math
    import sqlite3

    if not isinstance(dollar_pnl, (int, float)) or math.isnan(dollar_pnl) or math.isinf(dollar_pnl):
        logger.error("DD record rejected — invalid dollar_pnl: %r", dollar_pnl)
        return

    trading_day = _get_trading_day()

    try:
        db_path = _get_dd_db_path()
        if not db_path:
            logger.error("DD record skipped — no db_path configured")
            return

        with sqlite3.connect(db_path) as conn:
            row = conn.execute(
                "SELECT trading_day, realised_pnl FROM daily_drawdown WHERE id = 1"
            ).fetchone()

            if row and row[0] == trading_day:
                # Same trading day — accumulate
                conn.execute(
                    "UPDATE daily_drawdown SET realised_pnl = realised_pnl + ?, "
                    "trade_count = trade_count + 1, updated_at = datetime('now') WHERE id = 1",
                    (dollar_pnl,))
            elif row:
                # New trading day — reset and record
                conn.execute(
                    "UPDATE daily_drawdown SET trading_day = ?, realised_pnl = ?, "
                    "trade_count = 1, updated_at = datetime('now') WHERE id = 1",
                    (trading_day, dollar_pnl))
            else:
                # Row missing entirely — insert it
                conn.execute(
                    "INSERT INTO daily_drawdown (id, trading_day, realised_pnl, trade_count) "
                    "VALUES (1, ?, ?, 1)",
                    (trading_day, dollar_pnl))

            after = conn.execute(
                "SELECT realised_pnl FROM daily_drawdown WHERE id = 1"
            ).fetchone()
            logger.info(
                "DD recorded: $%+.0f → cumulative $%.0f (trading day %s)",
                dollar_pnl, after[0] if after else 0, trading_day,
            )
    except Exception as e:
        logger.error("DD write to DB failed: %s (pnl=$%.0f)", e, dollar_pnl)


def _calc_lot_size(entry: float, sl: float, balance: float = None,
                   risk_pct: float = None) -> dict:
    """Calculate recommended lot size for XAU/USD."""
    bal = balance or ACCOUNT_BALANCE
    pct = risk_pct or RISK_PCT
    risk_dollars = bal * (pct / 100)
    sl_distance = abs(entry - sl) if entry and sl else 0

    if sl_distance <= 0:
        return {"lot_size": 0, "risk_dollars": risk_dollars,
                "sl_distance": 0, "risk_pct": pct, "balance": bal}

    lot_size = risk_dollars / (sl_distance * XAUUSD_PIP_VALUE)
    return {
        "lot_size": round(lot_size, 2),
        "risk_dollars": round(risk_dollars, 2),
        "sl_distance": round(sl_distance, 2),
        "risk_pct": pct,
        "balance": bal,
    }


def _esc(text) -> str:
    """Escape text for Telegram HTML parse mode."""
    return html.escape(str(text))


# ═══════════════════════════════════════════════════════════════════════
# Phase 1: Zone Prospect Alert (informational)
# ═══════════════════════════════════════════════════════════════════════

def notify_zone_prospect(prospect_data: dict):
    """Zone prospect alert — demoted to log-only in Phase 6.

    Previously pushed to Telegram/macOS. Now logs to Intelligence Panel only.

    Args:
        prospect_data: dict with 'killzone' and 'setups' list
    """
    cfg = get_config()
    if not cfg.get("notify_zone_alerts", True):
        return

    kz = prospect_data.get("killzone", "?")
    setups = prospect_data.get("setups", [])
    n = len(setups)

    # Phase 6: Log only — no Telegram or macOS push
    logger.info("Zone prospect [%s]: %d conditional setup(s) identified", kz, n)
    for s in setups[:4]:
        bias = s.get("bias", "?")
        trigger = s.get("trigger_condition", "?")[:80]
        conf = s.get("confidence", "?")
        logger.info("  %s %s — confidence: %s", bias, trigger, conf)


# ═══════════════════════════════════════════════════════════════════════
# Phase 2: Displacement Confirmed Alert (get ready)
# ═══════════════════════════════════════════════════════════════════════

def notify_displacement_confirmed(setup_data: dict, displacement_data: dict):
    """Send displacement alert — sweep confirmed, waiting for retrace.

    Args:
        setup_data: the conditional setup dict from the prospect
        displacement_data: dict with ob_zone, fvg_zone, sweep_level, etc.
    """
    cfg = get_config()
    if not cfg.get("notify_displacement_alerts", True):
        return

    bias = setup_data.get("bias", "?").upper()
    sweep = displacement_data.get("sweep_level", 0)
    ob = displacement_data.get("ob_zone", {})

    # Phase 6: Demoted to log-only — no Telegram or macOS push
    logger.info("Displacement confirmed: %s sweep at %.2f, OB: %.2f-%.2f",
                bias, sweep, ob.get("low", 0), ob.get("high", 0))


# ═══════════════════════════════════════════════════════════════════════
# Phase 3: Entry Trigger Alert (execute now)
# ═══════════════════════════════════════════════════════════════════════

def notify_entry_trigger(entry_data: dict, calibration: dict = None):
    """Send entry alert — the actual trade signal with calibrated levels.

    Args:
        entry_data: analysis dict with entry, stopLoss, takeProfits, etc.
        calibration: calibration result dict (optional)
    """
    cfg = get_config()
    if not cfg.get("notify_entry_alerts", True):
        return

    entry_info = entry_data.get("entry", {})
    direction = entry_info.get("direction", "?")
    entry_price = entry_info.get("price", 0)
    sl_price = (entry_data.get("stopLoss") or {}).get("price", 0)
    tps_raw = entry_data.get("takeProfits") or []
    tps = [tp.get("price", tp) if isinstance(tp, dict) else tp for tp in tps_raw]

    # Use calibrated SL if available
    cal = calibration or {}
    cal_sl = (cal.get("calibrated") or {}).get("sl")
    effective_sl = cal_sl or sl_price

    lot = _calc_lot_size(entry_price, effective_sl)
    sl_dist = abs(entry_price - effective_sl) if entry_price and effective_sl else 1

    # macOS
    if cfg.get("notify_macos", True):
        sound = cfg.get("macos_sound_entry", "Glass")
        rr1 = f"{abs(tps[0] - entry_price) / sl_dist:.1f}R" if tps and sl_dist > 0 else "?"
        _send_macos(
            f"🎯 ENTER {direction.upper()} XAU/USD",
            f"${entry_price:.2f} | SL ${effective_sl:.2f} | TP1 {rr1} | {lot['lot_size']} lots",
            sound=sound)

    # Telegram
    if cfg.get("notify_telegram", True):
        lines = [
            f"<b>🎯 ENTRY SIGNAL — {direction.upper()} XAU/USD</b>",
            "",
            f"Entry: <b>${entry_price:.2f}</b>",
            f"Stop Loss: ${effective_sl:.2f}"
            + (f" (Claude: ${sl_price:.2f})" if cal_sl and cal_sl != sl_price else ""),
        ]

        for i, tp in enumerate(tps[:3], 1):
            rr = abs(tp - entry_price) / sl_dist if sl_dist > 0 else 0
            lines.append(f"TP{i}: ${tp:.2f} ({rr:.1f}R)")

        grade = (cal.get("confidence") or {}).get("grade", "?")
        conf_pct = (cal.get("confidence") or {}).get("autogluon_win_prob")
        kz = entry_data.get("killzone", "?")
        quality = entry_data.get("setup_quality", "?")

        lines.extend([
            "",
            f"Quality: {quality} | Grade: {grade}"
            + (f" | ML: {conf_pct * 100:.0f}%" if conf_pct else ""),
            f"Killzone: {kz}",
            "",
        ])

        # Multi3 enhancements: Kelly sizing + optimal split + EV
        multi3 = cal.get("multi3", {})
        pos_mult = multi3.get("position_multiplier", 1.0)
        adjusted_lots = round(lot["lot_size"] * pos_mult, 2)
        adjusted_risk = round(lot["risk_dollars"] * pos_mult, 2)

        if pos_mult > 1.0:
            lines.append(f"<b>LOT SIZE: {adjusted_lots} lots</b> (📐 {pos_mult}× Kelly)")
        else:
            lines.append(f"<b>LOT SIZE: {lot['lot_size']} lots</b>")
        dd = _daily_dd_remaining()
        dd_icon = "🛑" if dd["critical"] else "⚠️" if dd["warning"] else "🟢"
        lines.extend([
            f"Risk: ${adjusted_risk:.0f} ({lot['risk_pct']}% of ${lot['balance']:,.0f})",
            f"SL Distance: ${lot['sl_distance']:.2f}",
            f"{dd_icon} Daily DD: ${abs(dd['used_dollars']):,.0f} / "
            f"${dd['limit_dollars']:,.0f} ({dd['remaining_pct']:.1f}% left)",
        ])
        if dd["critical"]:
            lines.append("🛑 >75% daily drawdown consumed — consider stopping")

        # Class probabilities + management
        probs = multi3.get("class_probabilities", {})
        if probs:
            lines.append(
                f"\n📊 TP1: {probs.get('tp1', 0)*100:.0f}% | "
                f"Runner: {probs.get('runner', 0)*100:.0f}% | "
                f"SL: {probs.get('stopped_out', 0)*100:.0f}%")

        split = multi3.get("optimal_split", {})
        if split.get("advice"):
            lines.append(f"📋 {split['style'].upper()}: {split['advice']}")

        ev = multi3.get("ev", {})
        adj_ev = multi3.get("adjusted_ev_per_unit")
        if adj_ev is not None:
            ev_emoji = "✅" if adj_ev >= 0.15 else "⚠️"
            lines.append(f"{ev_emoji} EV: {adj_ev:.2f}R (${ev.get('ev_dollars', 0):.0f})")

        # Confluences
        confluences = entry_data.get("confluences", [])
        if confluences:
            conf_str = " ".join(f"✓{c}" for c in confluences[:5])
            lines.append(f"\n{conf_str}")

        # LTF signal
        ltf = entry_data.get("ltf_signal")
        if ltf:
            lines.append(f"\nLTF: {_esc(str(ltf)[:80])}")

        _send_telegram_html("\n".join(lines))


# ═══════════════════════════════════════════════════════════════════════
# Layer 1: Immediate detection alert (lightweight, fires on every A/B)
# ═══════════════════════════════════════════════════════════════════════

def notify_setup_detected(setup_data: dict):
    """Setup detection alert — demoted to log-only in Phase 6.

    Previously pushed to Telegram/macOS. Now logs to Intelligence Panel.
    Entry alerts (ENTRY_READY lifecycle stage 4) still push to Telegram.
    """
    cfg = get_config()
    if not cfg.get("notify_detection_alerts", True):
        return

    d = setup_data
    direction = d.get("direction", "?").upper()
    entry = d.get("entry_price", 0)
    current = d.get("current_price", 0)
    grade = d.get("setup_quality", "?")
    kz = d.get("killzone", "?")
    tf = d.get("timeframe", "?")
    promoted = " (promoted C/D)" if d.get("promoted_from_cd") else ""

    # Phase 6: Log-only — no Telegram or macOS push
    logger.info("Setup detected: %s %s [%s] Grade %s @ %.2f (price %.2f)%s",
                direction, kz, tf, grade, entry, current, promoted)


def notify_entry_missed(setup_data: dict):
    """Alert when entry level was passed — user can decide on market entry.

    Shows original SL/TPs (structural, unchanged) with recalculated R:R
    and lot size based on current price as potential entry.
    """
    cfg = get_config()
    d = setup_data
    direction = d.get("direction", "?").upper()
    entry = d.get("entry_price", 0)
    current = d.get("current_price", 0)
    sl = d.get("sl_price", 0)
    grade = d.get("setup_quality", "?")
    kz = d.get("killzone", "?")
    tf = d.get("timeframe", "?")
    tps = d.get("tps") or []

    distance = abs(current - entry)
    distance_pct = (distance / entry * 100) if entry else 0

    # Recalculate R:R using current price as potential secondary entry
    new_sl_distance = abs(current - sl) if sl else 0
    rr_lines = []
    for i, tp in enumerate(tps, 1):
        if new_sl_distance > 0:
            new_rr = abs(tp - current) / new_sl_distance
            rr_lines.append(f"TP{i}: ${tp:.2f} ({new_rr:.1f}R at market)")
        else:
            rr_lines.append(f"TP{i}: ${tp:.2f}")

    # Lot size at current price with original SL
    lot = _calc_lot_size(current, sl)

    title = f"⚠️ ENTRY MISSED {direction} [{tf}]"
    body_lines = [
        f"Grade: {grade} | KZ: {kz}",
        f"Original Entry: ${entry:.2f}",
        f"📍 Price Now: ${current:.2f} (+${distance:.1f} / {distance_pct:.1f}% past)",
        f"SL: ${sl:.2f}",
    ] + rr_lines + [
        f"---",
        f"If market entry: {lot['lot_size']} lots",
        f"Risk: ${lot['risk_dollars']:.0f} ({lot['risk_pct']}% of ${lot['balance']:,.0f})",
    ]
    body = "\n".join(body_lines)

    if cfg.get("notify_macos", True):
        _send_macos(title, body, sound=cfg.get("macos_sound_entry", "Glass"))
    if cfg.get("notify_telegram", True):
        _send_telegram_html(f"<b>{_esc(title)}</b>\n<pre>{_esc(body)}</pre>")


# ═══════════════════════════════════════════════════════════════════════
# Existing: Setup detection + Trade resolution (backward compatible)
# ═══════════════════════════════════════════════════════════════════════

def _build_calendar_warning_line(setup_data: dict) -> str:
    """Render a warning line for setups inside a caution/imminent calendar
    window. Returns ``""`` when the calendar is clear or no proximity payload
    is attached. Spec: warnings only — no suppression, no grade adjustment.
    """
    prox = setup_data.get("calendar_proximity") or {}
    state = prox.get("state")
    if state not in ("caution", "imminent"):
        return ""
    label = "WARNING" if state == "imminent" else "CAUTION"
    title = prox.get("next_event_title") or "high-impact news"
    minutes = prox.get("next_event_minutes")
    if isinstance(minutes, (int, float)):
        return f"⚠ {label}: {title} in ~{int(round(minutes))}m"
    return f"⚠ {label}: {title} approaching"


def build_notification_message(setup_data: dict) -> str:
    """Compose the body of a 'new setup' notification.

    Pure function (no I/O) so the body can be unit-tested without firing
    macOS / Telegram side-effects. Used by ``notify_new_setup``.
    """
    d = setup_data
    cal = d.get("calibration_json") or {}
    conf = cal.get("confidence", {})
    grade = d.get("setup_quality", conf.get("grade", "?"))
    cal_grade = conf.get("grade", "?")
    win_prob = conf.get("autogluon_win_prob")
    sl_src = cal.get("calibrated", {}).get("sl_source", "claude")
    defensive = conf.get("defensive_mode") or cal.get("defensive_mode")

    effective_sl = d.get("calibrated_sl") or d.get("sl_price", 0)
    lot = _calc_lot_size(d["entry_price"], effective_sl)

    current_price = d.get("current_price")
    price_line = ""
    if current_price and d["entry_price"]:
        dist = abs(current_price - d["entry_price"])
        price_line = f"📍 Live Price: ${current_price:.2f} (${dist:.1f} from entry)"

    body_lines = [
        _build_calendar_warning_line(d),
        f"Direction: {d['direction'].upper()} | Bias: {d.get('bias', '?')}",
        f"Entry: ${d['entry_price']:.2f}",
        price_line,
        f"SL: ${d.get('sl_price', 0):.2f}"
        + (f" -> Cal: ${d['calibrated_sl']:.2f}" if d.get("calibrated_sl") else ""),
        f"TP1: ${d['tps'][0]:.2f}" if d.get("tps") and len(d["tps"]) > 0 else "",
        f"TP2: ${d['tps'][1]:.2f}" if d.get("tps") and len(d["tps"]) > 1 else "",
        f"TP3: ${d['tps'][2]:.2f}" if d.get("tps") and len(d["tps"]) > 2 else "",
        f"RR: {d.get('rr_ratios', [])}",
        f"Grade: {grade} | KZ: {d.get('killzone', '?')}",
        f"Cal Grade: {cal_grade}" + (f" | Win Prob: {win_prob * 100:.0f}%" if win_prob else "")
        + (" [defensive mode]" if defensive else ""),
        f"SL Source: {sl_src}",
        f"---",
        f"LOT SIZE: {lot['lot_size']} lots",
        f"Risk: ${lot['risk_dollars']:.0f} ({lot['risk_pct']}% of ${lot['balance']:,.0f})",
        f"SL Distance: ${lot['sl_distance']:.2f}",
    ]
    if d.get("opus_validated"):
        body_lines.insert(0, "✅ Opus Validated")
    return "\n".join(line for line in body_lines if line)


def notify_new_setup(setup_data: dict, is_prospected: bool = False):
    """Send notification for a newly detected setup (standard scanner flow).

    If is_prospected=True, skip — entry_trigger already sent.
    """
    if is_prospected:
        return

    d = setup_data
    title = f"NEW {d['direction'].upper()} Setup [{d['timeframe']}]"
    body = build_notification_message(d)

    cfg = get_config()
    if cfg.get("notify_macos", True):
        _send_macos(title, body)
    if cfg.get("notify_telegram", True):
        _send_telegram_html(f"<b>{_esc(title)}</b>\n<pre>{_esc(body)}</pre>")


def notify_budget_warning(spent: float, limit: float):
    """Alert when approaching daily API budget limit."""
    pct = spent / limit * 100 if limit > 0 else 100
    title = f"⚠️ API Budget: ${spent:.2f} / ${limit:.2f} ({pct:.0f}%)"
    body = f"Approaching daily limit. Scanner will skip Claude calls if exceeded."
    cfg = get_config()
    if cfg.get("notify_macos", True):
        _send_macos(title, body, sound="Basso")
    if cfg.get("notify_telegram", True):
        _send_telegram_html(f"<b>{_esc(title)}</b>\n{_esc(body)}")


def notify_trade_resolved(setup: dict, result: dict):
    """DEPRECATED — kept for backward compatibility.

    Trade resolution notifications are now handled exclusively by
    notify_lifecycle(5, ...) / _build_stage_5() which owns:
      - Telegram notification
      - record_daily_pnl()
      - DD budget display
      - Post-resolution thesis

    This function no longer calls record_daily_pnl() to avoid
    double-counting when both paths fire.
    """
    logger.debug("notify_trade_resolved called (deprecated) for setup %s", setup.get("id"))


# ═══════════════════════════════════════════════════════════════════════
# Unified Lifecycle Notification System (6 Stages)
# ═══════════════════════════════════════════════════════════════════════

STAGE_NAMES = {
    1: "THESIS_FORMING",
    2: "THESIS_CONFIRMED",
    3: "SETUP_DETECTED",
    4: "ENTRY_READY",
    5: "TRADE_RESOLVED",
    6: "THESIS_REVISED",
}

STAGE_SOUNDS = {
    1: None,        # Telegram only — no macOS sound
    2: "Tink",      # Gentle awareness
    3: "Ping",      # Preparation
    4: "Glass",     # Action required
    5: "Glass",     # Outcome
    6: "Basso",     # Warning
}

STAGE_EMOJIS = {
    1: "\U0001f4ad",  # 💭
    2: "\U0001f52c",  # 🔬
    3: "\U0001f3af",  # 🎯
    4: "\u2705",      # ✅
    5: "\U0001f4ca",  # 📊
    6: "\u26a0\ufe0f",  # ⚠️
}


def notify_lifecycle(stage: int, thesis_id: str, timeframe: str,
                     narrative_state: dict,
                     setup_data: dict = None,
                     resolution_data: dict = None,
                     calibration: dict = None,
                     post_resolution_thesis: dict = None,
                     db=None):
    """Unified notification for all trade lifecycle events.

    Stages:
        1 = THESIS_FORMING    — first scan with bias_confidence >= 0.5
        2 = THESIS_CONFIRMED  — thesis survived 2+ scans, confidence >= 0.7
        3 = SETUP_DETECTED    — concrete entry/SL/TPs identified
        4 = ENTRY_READY       — price at entry, execute now
        5 = TRADE_RESOLVED    — outcome + post-resolution thesis
        6 = THESIS_REVISED    — thesis direction changed

    Args:
        stage: Lifecycle stage number (1-6)
        thesis_id: UUID linking all notifications for this thesis
        timeframe: e.g. "1h", "4h"
        narrative_state: Current thesis dict from NarrativeStore
        setup_data: Setup dict (stages 3-5)
        resolution_data: Resolution result dict (stage 5)
        calibration: Calibration dict (stages 3-4)
        post_resolution_thesis: New thesis after resolution (stage 5)
        db: ScannerDB instance for dedup tracking
    """
    cfg = get_config()

    # Config gate check
    stage_config_map = {
        1: "notify_thesis_forming",
        2: "notify_thesis_confirmed",
        6: "notify_thesis_revised",
    }
    config_key = stage_config_map.get(stage)
    if config_key and not cfg.get(config_key, True):
        return

    # Dedup check: skip if already sent for this thesis + stage
    if db:
        try:
            if db.lifecycle_already_sent(thesis_id, stage):
                logger.debug("Lifecycle stage %d already sent for thesis %s",
                             stage, thesis_id)
                return
            # Skip lower stages if higher already sent
            max_sent = db.lifecycle_max_stage_sent(thesis_id)
            if stage < max_sent and stage not in (5, 6):  # 5/6 always allowed
                logger.debug("Skipping stage %d — stage %d already sent for %s",
                             stage, max_sent, thesis_id)
                return
        except Exception as e:
            logger.debug("Lifecycle dedup check failed: %s", e)

    # Build message
    stage_name = STAGE_NAMES.get(stage, f"STAGE_{stage}")
    emoji = STAGE_EMOJIS.get(stage, "")
    title, body = _build_lifecycle_message(
        stage, emoji, stage_name, timeframe,
        narrative_state, setup_data, resolution_data,
        calibration, post_resolution_thesis)

    # Send via transports
    # Phase 6: Notification tiers — only push important stages
    # Stages 2 (CONFIRMED), 4 (ENTRY_READY), 5 (RESOLVED), 6 (REVISED)
    # push to Telegram/macOS. Stages 1 (FORMING), 3 (DETECTED) are log-only.
    PUSH_STAGES = {2, 4, 5, 6}
    is_push = stage in PUSH_STAGES

    telegram_msg_id = None
    sound = STAGE_SOUNDS.get(stage)
    reply_to = None
    if db:
        try:
            reply_to = db.get_lifecycle_thread_msg_id(thesis_id)
        except Exception:
            pass

    if is_push and cfg.get("notify_telegram", True):
        telegram_msg_id = _send_telegram_html(
            f"<b>{_esc(title)}</b>\n<pre>{_esc(body)}</pre>",
            reply_to_message_id=reply_to)
    elif not is_push:
        logger.info("Lifecycle [%s] stage %d (%s): %s — %s",
                     thesis_id[:8], stage, stage_name, title, body[:100])

    # macOS for push stages only (Phase 6 simplification)
    if is_push and sound and cfg.get("notify_macos", True):
        _send_macos(title, body[:200], sound=sound)

    # Record for dedup + threading
    if db:
        try:
            db.record_lifecycle_notification(
                thesis_id=thesis_id,
                timeframe=timeframe,
                stage=stage,
                stage_name=stage_name,
                telegram_msg_id=str(telegram_msg_id) if telegram_msg_id else None,
                setup_id=(setup_data or {}).get("id"),
                payload_json={"title": title, "body_preview": body[:200]})
        except Exception as e:
            logger.debug("Lifecycle notification record failed: %s", e)


def _build_lifecycle_message(stage: int, emoji: str, stage_name: str,
                              timeframe: str, narrative: dict,
                              setup_data: dict = None,
                              resolution_data: dict = None,
                              calibration: dict = None,
                              post_thesis: dict = None) -> tuple[str, str]:
    """Build title + body for a lifecycle notification.

    Returns (title, body) tuple.
    """
    ns = narrative or {}
    bias = (ns.get("directional_bias") or "?").upper()
    confidence = ns.get("bias_confidence", 0)
    thesis_text = ns.get("thesis", "")
    scan_count = ns.get("scan_count", 1)

    if stage == 1:
        return _build_stage_1(emoji, timeframe, bias, confidence, thesis_text, ns)
    elif stage == 2:
        return _build_stage_2(emoji, timeframe, bias, confidence, thesis_text,
                               scan_count, ns)
    elif stage == 3:
        return _build_stage_3(emoji, timeframe, bias, confidence, thesis_text,
                               scan_count, setup_data, calibration)
    elif stage == 4:
        return _build_stage_4(emoji, timeframe, setup_data, calibration,
                               bias, confidence, scan_count, thesis_text)
    elif stage == 5:
        return _build_stage_5(emoji, timeframe, setup_data, resolution_data,
                               post_thesis)
    elif stage == 6:
        return _build_stage_6(emoji, timeframe, bias, thesis_text, ns,
                               setup_data)
    else:
        return f"{emoji} LIFECYCLE [{timeframe}]", "Unknown stage"


def _build_stage_1(emoji, tf, bias, conf, thesis, ns):
    """Stage 1: THESIS FORMING"""
    title = f"{emoji} THESIS FORMING [{tf}] — {bias}"
    levels = ns.get("key_levels", [])
    levels_str = " | ".join(str(l) for l in levels[:3]) if levels else "—"
    watching = ns.get("watching_for", [])
    watching_str = "; ".join(str(w) for w in watching[:2]) if watching else "—"
    body_lines = [
        f"P3 Phase: {ns.get('p3_phase', '?')}",
        f"Confidence: {conf * 100:.0f}%" if conf else "",
        "",
        thesis[:200] if thesis else "Thesis forming...",
        "",
        f"Key levels: {levels_str}",
        f"Watching for: {watching_str}",
    ]
    return title, "\n".join(line for line in body_lines if line is not None)


def _build_stage_2(emoji, tf, bias, conf, thesis, scan_count, ns):
    """Stage 2: THESIS CONFIRMED"""
    title = f"{emoji} THESIS CONFIRMED [{tf}] — {bias} ({conf * 100:.0f}%)"
    body_lines = [
        f"Scan {scan_count} — thesis held",
        "",
        thesis[:200] if thesis else "",
        "",
        f"Invalidation: {json.dumps(ns.get('invalidation', {}))}" if ns.get("invalidation") else "",
        "",
        "Prepare for potential trade signal on next scan.",
    ]
    return title, "\n".join(line for line in body_lines if line)


def _build_stage_3(emoji, tf, bias, conf, thesis, scan_count,
                    setup_data, calibration):
    """Stage 3: SETUP DETECTED"""
    d = setup_data or {}
    direction = (d.get("direction") or "?").upper()
    entry = d.get("entry_price", 0)
    sl = d.get("sl_price", 0)
    cal_sl = d.get("calibrated_sl") or sl
    grade = d.get("setup_quality", "?")
    kz = d.get("killzone", "?")
    tps = d.get("tps") or [d.get("tp1"), d.get("tp2"), d.get("tp3")]
    tps = [t for t in tps if t]
    current = d.get("current_price", 0)

    title = f"{emoji} SETUP DETECTED [{tf}] — {direction} Grade {grade}"
    body_lines = [
        f"Thesis: \"{thesis[:100]}\" ({scan_count} scans, {conf * 100:.0f}%)"
        if thesis else "",
        "",
        f"Entry: ${entry:.2f} | SL: ${sl:.2f} | Cal SL: ${cal_sl:.2f}",
    ]
    for i, tp in enumerate(tps[:3], 1):
        rr = abs(tp - entry) / abs(sl - entry) if abs(sl - entry) > 0 else 0
        body_lines.append(f"TP{i}: ${tp:.2f} ({rr:.1f}R)")
    body_lines.append(f"KZ: {kz}")

    if current:
        dist = abs(current - entry)
        body_lines.append(f"📍 Price now: ${current:.2f} (${dist:.1f} from entry)")

    return title, "\n".join(line for line in body_lines if line)


def _build_stage_4(emoji, tf, setup_data, calibration,
                    bias, conf, scan_count, thesis):
    """Stage 4: ENTRY READY"""
    d = setup_data or {}
    direction = (d.get("direction") or "?").upper()
    entry = d.get("entry_price", 0)
    sl = d.get("calibrated_sl") or d.get("sl_price", 0)
    grade = d.get("setup_quality", "?")
    tps = d.get("tps") or [d.get("tp1"), d.get("tp2"), d.get("tp3")]
    tps = [t for t in tps if t]

    lot = _calc_lot_size(entry, sl)

    title = f"{emoji} ENTRY READY — {direction} XAU/USD [{tf}]"
    body_lines = [
        f"Entry: ${entry:.2f} | SL: ${sl:.2f}",
    ]
    for i, tp in enumerate(tps[:3], 1):
        rr = abs(tp - entry) / abs(sl - entry) if abs(sl - entry) > 0 else 0
        body_lines.append(f"TP{i}: ${tp:.2f} ({rr:.1f}R)")

    body_lines.extend([
        "",
        f"Grade: {grade}",
        f"LOT SIZE: {lot['lot_size']} lots",
        f"Risk: ${lot['risk_dollars']:.0f} ({lot['risk_pct']}% of ${lot['balance']:,.0f})",
    ])

    # Daily drawdown budget
    dd = _daily_dd_remaining()
    dd_icon = "🛑" if dd["critical"] else "⚠️" if dd["warning"] else "🟢"
    body_lines.append(
        f"{dd_icon} Daily DD: ${abs(dd['used_dollars']):,.0f} used / "
        f"${dd['remaining_dollars']:,.0f} left ({dd['remaining_pct']:.1f}%)"
    )
    if dd["critical"]:
        body_lines.append("🛑 CAUTION: >75% of daily drawdown consumed")

    if thesis:
        body_lines.extend(["", f"Thesis: {thesis[:100]} ({scan_count} scans)"])

    cal = calibration or {}
    win_prob = cal.get("confidence", {}).get("autogluon_win_prob")
    if win_prob:
        body_lines.append(f"ML Win Prob: {win_prob * 100:.0f}%")

    return title, "\n".join(line for line in body_lines if line)


def _build_stage_5(emoji, tf, setup_data, resolution_data, post_thesis):
    """Stage 5: TRADE RESOLVED"""
    d = setup_data or {}
    r = resolution_data or {}
    outcome = r.get("outcome") or d.get("outcome", "?")
    emoji_outcome = {"tp1": "✅", "tp2": "✅✅", "tp3": "✅✅✅",
                     "stopped_out": "❌", "expired": "⏰"}.get(outcome, "📊")

    entry = d.get("entry_price", 0)
    exit_price = r.get("price", 0)
    gross_rr = r.get("gross_rr", 0)
    cost_rr = r.get("cost_rr", 0)
    net_rr = r.get("rr") or d.get("rr", 0)
    direction = (d.get("direction") or "?").upper()

    title = f"{emoji_outcome} {outcome.upper().replace('_', ' ')} [{tf}] {direction}"
    body_lines = [
        f"Entry: ${entry:.2f} → Exit: ${exit_price:.2f}",
        f"Gross: {gross_rr:+.1f}R | Cost: {cost_rr:.2f}R | Net: {net_rr:+.1f}R"
        if cost_rr else f"Net: {net_rr:+.1f}R",
    ]

    sl = d.get("calibrated_sl") or d.get("sl_price") or 0
    lot = _calc_lot_size(entry, sl) if entry and sl else {"lot_size": 0, "risk_dollars": 0}

    # Decouple P&L from lot_size — risk_dollars is always computable
    risk_dollars = lot.get("risk_dollars") or (ACCOUNT_BALANCE * RISK_PCT / 100)
    pnl_dollars = net_rr * risk_dollars if net_rr else 0.0

    if lot["lot_size"]:
        body_lines.append(f"P&L: ${pnl_dollars:+,.0f} | Lot: {lot['lot_size']}")
    elif pnl_dollars:
        body_lines.append(f"P&L: ${pnl_dollars:+,.0f}")

    # Track daily DD and show remaining budget
    if pnl_dollars:
        record_daily_pnl(pnl_dollars)
    dd = _daily_dd_remaining()
    dd_icon = "🛑" if dd["critical"] else "⚠️" if dd["warning"] else "🟢"
    body_lines.append(
        f"{dd_icon} Daily DD: ${abs(dd['used_dollars']):,.0f} used / "
        f"${dd['remaining_dollars']:,.0f} left ({dd['remaining_pct']:.1f}%)"
    )
    if dd["critical"]:
        body_lines.append("🛑 >75% daily drawdown consumed — consider stopping")

    if post_thesis:
        post_bias = (post_thesis.get("directional_bias") or "?").upper()
        post_text = post_thesis.get("thesis", "")
        body_lines.extend([
            "",
            f"🔄 POST-RESOLUTION:",
            f"Thesis → {post_bias} ({post_thesis.get('bias_confidence', 0) * 100:.0f}%)",
            post_text[:150] if post_text else "Re-scan in progress.",
        ])

    return title, "\n".join(body_lines)


def _build_stage_6(emoji, tf, new_bias, thesis, ns, setup_data):
    """Stage 6: THESIS REVISED"""
    title = f"{emoji} THESIS REVISED [{tf}] — {new_bias}"
    body_lines = [
        f"Revised: \"{thesis[:150]}\"" if thesis else "Direction changed.",
    ]

    inv = ns.get("invalidation", {})
    if inv:
        body_lines.append(f"Reason: Invalidation at {inv.get('price_level', '?')}")

    if setup_data:
        body_lines.extend([
            "",
            f"Active setup affected: {(setup_data.get('direction') or '?').upper()} "
            f"entry at ${setup_data.get('entry_price', 0):.2f}",
            "",
            "The analytical basis for your position has changed.",
            "Review whether to hold, tighten SL, or exit manually.",
        ])

    return title, "\n".join(body_lines)


def _get_current_thesis(timeframe: str) -> dict | None:
    """Helper: fetch current thesis from NarrativeStore for backward compat wrappers."""
    try:
        from ml.narrative_state import NarrativeStore
        store = NarrativeStore()
        return store.get_current(timeframe)
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════
# Transport layer
# ═══════════════════════════════════════════════════════════════════════

def _send_macos(title: str, body: str, sound: str = "Glass"):
    """macOS notification center via osascript."""
    cfg = get_config()
    if not cfg.get("notify_macos", True):
        return
    try:
        safe_title = title.replace('"', '\\"')
        safe_body = body.replace('"', '\\"').replace("\n", "\\n")
        subprocess.run(
            [
                "osascript", "-e",
                f'display notification "{safe_body}" with title "{safe_title}" sound name "{sound}"',
            ],
            timeout=5,
            capture_output=True,
        )
    except Exception as e:
        logger.warning("macOS notification failed: %s", e)


def _send_telegram(title: str, body: str):
    """Legacy Telegram — plain Markdown. Kept for backward compat."""
    _send_telegram_html(f"<b>{_esc(title)}</b>\n<pre>{_esc(body)}</pre>")


def _send_telegram_html(text: str, reply_to_message_id: str = None) -> str | None:
    """Send Telegram message with HTML formatting.

    Args:
        text: HTML-formatted message text
        reply_to_message_id: Optional message ID to reply to (threading)

    Returns:
        message_id from Telegram response (for threading), or None
    """
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram skipped — credentials not configured")
        return None
    cfg = get_config()
    if not cfg.get("notify_telegram", True):
        return None
    try:
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": text,
            "parse_mode": "HTML",
        }
        if reply_to_message_id:
            payload["reply_to_message_id"] = reply_to_message_id
        resp = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json=payload,
            timeout=10,
        )
        if not resp.ok:
            logger.warning("Telegram send failed: %s %s", resp.status_code, resp.text[:200])
            return None
        # Return message_id for threading
        data = resp.json()
        return str(data.get("result", {}).get("message_id", ""))
    except Exception as e:
        logger.warning("Telegram notification failed: %s", e)
        return None
