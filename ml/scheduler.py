"""APScheduler integration for headless multi-timeframe ICT scanning.

Five jobs:
  1. Scan job — runs every 5 minutes, Mon-Fri. Checks all 7 timeframes
     but only calls Claude where candle data has changed (hash detection).
  2. Monitor job — runs every 5 minutes, Mon-Fri. Checks pending setups
     against current price for SL/TP auto-resolution.
  3. Retrain job — runs every 6 hours, Mon-Fri. Retrains AutoGluon
     classifier if 10+ new trades have accumulated since last training.
  4. Trigger monitor — runs every 90 seconds, Mon-Fri. Checks active
     prospects against current price for trigger detection. No Claude
     calls unless a trigger is detected.

Only starts if OANDA credentials and ANTHROPIC_API_KEY are set.
"""
import asyncio
import logging
import os
from datetime import datetime

from apscheduler.schedulers.asyncio import AsyncIOScheduler

logger = logging.getLogger(__name__)

_scheduler: AsyncIOScheduler | None = None
_engine = None

# Prospect schedule — 15 min before each killzone opens (UTC)
PROSPECT_SCHEDULE_UTC = {
    "Asian":  {"hour": 0, "minute": 45},
    "London": {"hour": 7, "minute": 45},
    "NY_AM":  {"hour": 12, "minute": 45},
    "NY_PM":  {"hour": 17, "minute": 45},
}


def _is_market_active() -> bool:
    """Check if XAU/USD market is open (Sun 22:00 UTC to Fri 22:00 UTC)."""
    now = datetime.utcnow()
    wd = now.weekday()  # 0=Mon
    h = now.hour
    if wd == 5:  # Saturday — closed all day
        return False
    if wd == 6 and h < 22:  # Sunday before futures open
        return False
    if wd == 4 and h >= 22:  # Friday after NY close
        return False
    return True


def _get_engine():
    global _engine
    if _engine is None:
        from ml.scanner import ScannerEngine
        _engine = ScannerEngine()
    return _engine


async def _scan_job():
    """Multi-timeframe scan — runs sync engine in thread pool."""
    if not _is_market_active():
        return
    try:
        print("[SCAN] Starting scan tick...")
        engine = _get_engine()
        result = await asyncio.get_event_loop().run_in_executor(
            None, engine.scan_all_timeframes
        )
        scanned = result.get("scanned", 0)
        skipped = result.get("skipped", 0)
        gated = result.get("gated", 0)
        print(f"[SCAN] Done: {scanned} analyzed, {skipped} unchanged, {gated} gated")

        # Backup: auto-prospect on killzone change if no prospects exist
        try:
            from ml.prompts import get_current_killzone
            current_kz = get_current_killzone()
            last_kz = getattr(engine, '_scheduler_last_kz', None)
            engine._scheduler_last_kz = current_kz
            if current_kz != last_kz and current_kz != "Off" and last_kz is not None:
                prospects = engine.db.get_active_prospects()
                if not prospects:
                    print(f"[SCAN] Killzone changed to {current_kz} with no prospects — generating...")
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(
                        None, engine._prospect_killzone_zones, current_kz)
        except Exception as e:
            logger.debug("Auto-prospect on KZ change failed: %s", e)
        logger.info("Scanner tick: %d analyzed, %d unchanged, %d gated", scanned, skipped, gated)
    except Exception as e:
        print(f"[SCAN] CRASHED: {e}")
        logger.error("Scanner job crashed: %s", e, exc_info=True)


async def _monitor_job():
    """Price monitor — runs sync engine in thread pool.
    Legacy fallback — prefer unified_monitor_job().
    """
    if not _is_market_active():
        return
    try:
        print("[MONITOR] Checking pending setups...")
        engine = _get_engine()
        result = await asyncio.get_event_loop().run_in_executor(
            None, engine.monitor_pending
        )
        resolved = result.get("resolved", 0)
        checked = result.get("checked", 0)
        print(f"[MONITOR] Checked {checked}, resolved {resolved}")
        if resolved > 0:
            logger.info("Monitor resolved %d setups", resolved)
    except Exception as e:
        print(f"[MONITOR] CRASHED: {e}")
        logger.error("Monitor job crashed: %s", e, exc_info=True)


async def _unified_monitor_job():
    """Unified monitoring loop — replaces separate monitor/trigger/CD jobs.

    Single 5-min candle fetch serves all monitor types with priority ordering:
    1. A/B pending SL/TP resolution
    2. A/B entry proximity alerts
    3. C/D displacement monitoring
    4. Prospect zone triggers
    """
    if not _is_market_active():
        return
    try:
        engine = _get_engine()
        result = await asyncio.get_event_loop().run_in_executor(
            None, engine.unified_monitor
        )
        pending = result.get("pending", {})
        cd = result.get("cd_monitoring", {})
        prospects = result.get("prospects", {})
        prox = result.get("proximity", {})

        resolved = pending.get("resolved", 0)
        promoted = cd.get("promoted", 0)
        triggered = prospects.get("triggered", 0)
        notified = prox.get("notified", 0)

        if resolved or promoted or triggered or notified:
            print(f"[UNIFIED] Resolved={resolved} Promoted={promoted} "
                  f"Triggered={triggered} Notified={notified}")
            logger.info("Unified monitor: resolved=%d promoted=%d triggered=%d notified=%d",
                        resolved, promoted, triggered, notified)
    except Exception as e:
        print(f"[UNIFIED] CRASHED: {e}")
        logger.error("Unified monitor crashed: %s", e, exc_info=True)


async def _prospect_job():
    """Generate zone prospects before a killzone opens."""
    if not _is_market_active():
        return
    try:
        engine = _get_engine()
        # Determine upcoming killzone based on current time
        hour = datetime.utcnow().hour
        if 0 <= hour < 1:
            upcoming = "Asian"
        elif 7 <= hour < 8:
            upcoming = "London"
        elif 12 <= hour < 13:
            upcoming = "NY_AM"
        elif 17 <= hour < 18:
            upcoming = "NY_PM"
        else:
            return  # Not near a killzone boundary

        print(f"[PROSPECT] Generating zones for {upcoming}...")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, engine._prospect_killzone_zones, upcoming)
        print(f"[PROSPECT] Done — zones generated for {upcoming}")
    except Exception as e:
        print(f"[PROSPECT] Failed: {e}")
        logger.error("Prospect generation failed: %s", e, exc_info=True)


async def _cd_monitor_job():
    """Check C/D grade monitoring setups for promotion. Runs every 5 minutes."""
    if not _is_market_active():
        return
    try:
        engine = _get_engine()
        result = await asyncio.get_event_loop().run_in_executor(
            None, engine.monitor_cd_setups
        )
        promoted = result.get("promoted", 0)
        expired_cd = result.get("expired", 0)
        if promoted > 0:
            print(f"[CD-MONITOR] Promoted {promoted} C/D setup(s) to pending!")
            logger.info("CD monitor: promoted %d setups", promoted)
        if expired_cd > 0:
            logger.info("CD monitor: expired %d stale monitoring setups", expired_cd)
    except Exception as e:
        logger.debug("CD monitor error: %s", e)


async def _trigger_monitor_job():
    """Check active prospects + entry proximity. Runs every 90 seconds."""
    if not _is_market_active():
        return
    try:
        engine = _get_engine()

        # Entry proximity alerts — check if price is near unnotified setups
        try:
            prox = await asyncio.get_event_loop().run_in_executor(
                None, engine.check_entry_proximity
            )
            if prox.get("notified", 0) > 0:
                print(f"[PROXIMITY] Sent {prox['notified']} entry alert(s)")
            if prox.get("missed", 0) > 0:
                print(f"[PROXIMITY] {prox['missed']} entry(ies) missed (price moved past)")
        except Exception as pe:
            logger.debug("Entry proximity check error: %s", pe)

        # Prospect trigger monitoring
        prospects = engine.db.get_active_prospects()
        if not prospects:
            return

        result = await asyncio.get_event_loop().run_in_executor(
            None, engine.monitor_prospect_triggers
        )
        triggered = result.get("triggered", 0)
        if triggered > 0:
            print(f"[TRIGGER] {triggered} prospect trigger(s) fired!")
            logger.info("Trigger monitor: %d triggers fired", triggered)
    except Exception as e:
        logger.debug("Trigger monitor error: %s", e)


async def _retrain_job():
    """Auto-retrain AutoGluon if enough new trades accumulated."""
    try:
        from ml.training import train_classifier
        from ml.database import TradeLogger
        from ml.dataset import TrainingDatasetManager
        from ml.config import get_config

        cfg = get_config()
        db = TradeLogger(config=cfg)
        dm = TrainingDatasetManager()
        stats = dm.get_stats()
        total = stats.get("total", 0)

        min_samples = cfg.get("min_training_samples", 30)
        if total < min_samples:
            print(f"[RETRAIN] Only {total} trades, need {min_samples}. Skipping.")
            return

        last = db.get_last_training("classifier")
        last_count = last["samples_used"] if last else 0
        new_trades = total - last_count

        retrain_threshold = cfg.get("retrain_on_n_new_trades", 10)
        if new_trades >= retrain_threshold:
            print(f"[RETRAIN] {new_trades} new trades since last train. Retraining...")
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, lambda: train_classifier(db, config=cfg, dataset_manager=dm)
            )
            acc = result.get("accuracy", "?")
            samples = result.get("samples_used", "?")
            print(f"[RETRAIN] Done — accuracy: {acc}, samples: {samples}")
            logger.info("Auto-retrain completed: accuracy=%s, samples=%s", acc, samples)
        else:
            print(f"[RETRAIN] Only {new_trades} new trades (need {retrain_threshold}). Skipping.")
    except ImportError:
        print("[RETRAIN] AutoGluon not installed, skipping.")
    except Exception as e:
        print(f"[RETRAIN] FAILED: {e}")
        logger.error("Retrain job failed: %s", e, exc_info=True)


async def _recompute_placement_job():
    """Periodic recompute of entry placement statistics from resolved setups."""
    try:
        from ml.entry_placement import EntryPlacementAnalyzer
        from ml.scanner_db import ScannerDB

        db = ScannerDB()
        rows = db.conn.execute(
            "SELECT * FROM scanner_setups WHERE outcome IS NOT NULL "
            "AND entry_zone_position IS NOT NULL"
        ).fetchall()
        cols = [d[0] for d in db.conn.execute("SELECT * FROM scanner_setups LIMIT 0").description]

        analyzer = EntryPlacementAnalyzer()
        count = 0
        for row in rows:
            rd = dict(zip(cols, row))
            if rd.get("mfe_atr") is not None and rd.get("mae_atr") is not None:
                analyzer.ingest_metric({
                    "entry_zone_position": rd["entry_zone_position"],
                    "entry_zone_type": rd.get("entry_zone_type", "unknown"),
                    "entry_zone_size_atr": (rd.get("entry_zone_high", 0) - rd.get("entry_zone_low", 0)) / max(rd.get("mfe_atr", 1), 0.01),
                    "direction": rd.get("direction", "long"),
                    "killzone": rd.get("killzone", "unknown"),
                    "outcome": rd.get("outcome", "loss"),
                    "mfe_atr": rd["mfe_atr"],
                    "mae_atr": rd["mae_atr"],
                })
                count += 1

        if count > 0:
            summary = analyzer.compute_summary()
            print(f"[PLACEMENT] Recomputed from {count} resolved setups")
            logger.info("Placement recompute: %d setups, optimal=%s", count, summary.get("optimal_position"))
        else:
            print("[PLACEMENT] No resolved setups with MFE/MAE + zone data. Skipping.")
    except Exception as e:
        print(f"[PLACEMENT] Recompute failed: {e}")
        logger.error("Placement recompute failed: %s", e, exc_info=True)


async def _recompute_killzone_job():
    """Periodic killzone profile recompute (P6/P8) — updates quality gates and scan config."""
    try:
        from ml.killzone_profiler import KillzoneProfiler
        from ml.scanner_db import ScannerDB
        import sqlite3

        db = ScannerDB()
        with db._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT outcome, killzone, setup_quality, timeframe, analysis_json "
                "FROM scanner_setups WHERE outcome IS NOT NULL AND killzone IS NOT NULL"
            ).fetchall()
        trades = [dict(r) for r in rows]

        kp = KillzoneProfiler()
        kp.compute_quality_gates(trades)
        kp.get_scan_config(trades)
        print(f"[KILLZONE] Profile recomputed from {len(trades)} trades")
        logger.info("Killzone profile recomputed: %d trades", len(trades))
    except Exception as e:
        print(f"[KILLZONE] Recompute failed: {e}")
        logger.error("Killzone recompute failed: %s", e, exc_info=True)


async def _recompute_intermarket_job():
    """Periodic intermarket validation recompute (P7)."""
    try:
        from ml.intermarket_validator import IntermarketValidator
        from ml.scanner_db import ScannerDB
        import sqlite3

        db = ScannerDB()
        with db._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT outcome, killzone, direction, calibration_json "
                "FROM scanner_setups "
                "WHERE outcome IS NOT NULL AND calibration_json IS NOT NULL"
            ).fetchall()
        trades = [dict(r) for r in rows]

        v = IntermarketValidator()
        result = v.analyze(trades)
        print(f"[INTERMARKET] Recomputed: {result.get('total_trades', 0)} trades, "
              f"recommendation={result.get('recommendation')}")
        logger.info("Intermarket recomputed: %d trades, rec=%s",
                     result.get("total_trades", 0), result.get("recommendation"))
    except Exception as e:
        print(f"[INTERMARKET] Recompute failed: {e}")
        logger.error("Intermarket recompute failed: %s", e, exc_info=True)


def start_scheduler():
    """Start the scanner scheduler. Only if API keys are configured."""
    global _scheduler

    from ml.config import get_config
    cfg = get_config()
    claude_key = os.environ.get("ANTHROPIC_API_KEY", "")
    oanda_ok = bool(cfg.get("oanda_account_id") and cfg.get("oanda_access_token"))

    if not oanda_ok or not claude_key:
        missing = []
        if not oanda_ok:
            missing.append("OANDA credentials")
        if not claude_key:
            missing.append("ANTHROPIC_API_KEY")
        msg = ", ".join(missing)
        print(f"Scanner not started — missing: {msg}")
        logger.warning("Scanner not started — missing: %s", msg)
        return

    try:
        _scheduler = AsyncIOScheduler()

        # Scan all timeframes every 15 minutes, Mon-Fri
        # (reduced from 5min to cut API costs ~60%; no accuracy loss
        # since 15min is the fastest candle close we analyse)
        _scheduler.add_job(
            _scan_job, "cron",
            minute="*/15", day_of_week="mon-fri",
            id="scanner_scan",
            replace_existing=True,
        )

        # Unified monitor — single loop for pending SL/TP, entry proximity,
        # C/D promotion, and prospect triggers. Shares one 5-min candle fetch.
        # Runs every 60 seconds (fastest required interval).
        _scheduler.add_job(
            _unified_monitor_job, "interval",
            seconds=60,
            id="unified_monitor",
            replace_existing=True,
        )

        # Auto-retrain AutoGluon every 6 hours if 10+ new trades accumulated
        _scheduler.add_job(
            _retrain_job, "cron",
            hour="*/6", minute=30, day_of_week="mon-fri",
            id="retrain_check",
            replace_existing=True,
        )

        # Recompute entry placement stats every 6 hours (offset from retrain)
        _scheduler.add_job(
            _recompute_placement_job, "cron",
            hour="*/6", minute=45, day_of_week="mon-fri",
            id="placement_recompute",
            replace_existing=True,
        )

        # Recompute killzone profile + scan config every 6 hours (P6/P8)
        _scheduler.add_job(
            _recompute_killzone_job, "cron",
            hour="*/6", minute=50, day_of_week="mon-fri",
            id="killzone_recompute",
            replace_existing=True,
        )

        # Recompute intermarket validation every 6 hours (P7)
        _scheduler.add_job(
            _recompute_intermarket_job, "cron",
            hour="*/6", minute=55, day_of_week="mon-fri",
            id="intermarket_recompute",
            replace_existing=True,
        )

        # Prospect jobs — generate zones 15 min before each killzone opens
        for kz, schedule in PROSPECT_SCHEDULE_UTC.items():
            _scheduler.add_job(
                _prospect_job, "cron",
                hour=schedule["hour"], minute=schedule["minute"],
                day_of_week="mon-fri",
                id=f"prospect_{kz}",
                replace_existing=True,
            )

        _scheduler.start()
        print("[SCANNER] Analysis timeframes: 15min, 1h, 4h, 1day (5min reserved for entry refinement)")
        print("Scanner started — 4 TFs every 15min + unified monitor every 60s + retrain/placement/killzone/intermarket every 6h, Mon-Fri")
        logger.info(
            "Scanner started — 4 analysis timeframes (15min, 1h, 4h, 1day), "
            "scan every 15min, unified monitor every 60s, Mon-Fri"
        )
    except Exception as e:
        print(f"Scanner start FAILED: {e}")
        logger.error("Scanner start failed: %s", e, exc_info=True)
        _scheduler = None


def stop_scheduler():
    """Gracefully stop the scheduler — waits for running jobs to finish."""
    global _scheduler
    if _scheduler:
        try:
            _scheduler.shutdown(wait=True)
        except Exception:
            # Fallback: force shutdown if graceful fails
            try:
                _scheduler.shutdown(wait=False)
            except Exception:
                pass
        _scheduler = None
        logger.info("Scanner scheduler stopped")


def is_running() -> bool:
    return _scheduler is not None and _scheduler.running


def get_next_scan_time() -> str | None:
    if not _scheduler:
        return None
    job = _scheduler.get_job("scanner_scan")
    if job and job.next_run_time:
        return job.next_run_time.isoformat()
    return None
