"""Headless multi-timeframe ICT scanner engine.

Scans all timeframes (5M → 1D) for ICT setups. Uses candle hash caching
to only call Claude when new candle data appears on each timeframe.

Schedule: runs every 5 minutes. Each tick checks which timeframes have
new data. Only calls Claude for timeframes where candles actually changed.

Environment variables required:
    ANTHROPIC_API_KEY   — for Claude analysis
    OANDA credentials   — configured in ml/config.py (oanda_account_id, oanda_access_token)
"""
import hashlib
import json
import logging
import os
import time
from datetime import datetime, timedelta

import httpx

from ml.config import get_config
from ml.env_utils import sanitize_env_secret
from ml.notifications import (
    notify_new_setup,
    notify_setup_detected, notify_entry_missed,
)
from ml.prompts import get_current_killzone
from ml.scanner_db import ScannerDB

logger = logging.getLogger(__name__)


def _safe_load_claude_json(clean: str, source: str = "claude") -> dict:
    """Parse JSON from a Claude response, recovering from common LLM
    output errors via json_repair fallback.

    Background — 2026-04-28: Sonnet 4.6 was producing JSON that strict
    json.loads couldn't parse (failures around char 7000+ in long
    analyses, 'Expecting , delimiter' at line ~115), most often because
    the model embedded an unescaped quote or literal newline inside a
    long narrative string field. The structure was logically intact —
    just unparseable to a strict parser.

    Strict json.loads runs first so clean responses cost nothing extra.
    json_repair only kicks in on failure. If repair also fails we
    re-raise the original JSONDecodeError so callers' existing error
    handling fires with their regular message.
    """
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        try:
            from json_repair import repair_json
        except ImportError:
            # json_repair not installed — re-raise so caller logs the
            # original parse error and we know to deploy the dep.
            raise
        repaired = repair_json(clean, return_objects=True)
        if isinstance(repaired, dict) and repaired:
            logger.info("Scanner: %s JSON repaired via json_repair "
                        "(strict parse failed, recovered)", source)
            return repaired
        # repair_json returns "" or [] on hopeless input — re-raise the
        # original error so the caller's _last_error captures the snippet.
        raise

ICT_SYSTEM_MESSAGE = """You are an elite ICT (Inner Circle Trader) analyst specialising in XAU/USD (Gold). You follow Michael J. Huddleston's ICT methodology precisely.

Core ICT Principles You Apply:
1. MARKET STRUCTURE — Identify Break of Structure (BOS) and Change of Character (CHoCH/MSS). Bullish structure = higher highs + higher lows. Bearish = lower highs + lower lows. A CHoCH signals potential reversal.
2. ORDER BLOCKS — The last opposing candle before displacement (a strong impulsive move). Bullish OB = last bearish candle before a strong rally with displacement. Bearish OB = last bullish candle before a strong sell-off. Validate with: displacement must exceed the OB range, price must break structure after the OB forms.
3. FAIR VALUE GAPS — 3-candle imbalance where candle 1 wick and candle 3 wick don't overlap. Bullish FVG: candle 3 low > candle 1 high. Bearish FVG: candle 3 high < candle 1 low.
4. LIQUIDITY — Resting orders at swing highs (BSL) and swing lows (SSL). Smart money hunts these pools before reversing.
5. PREMIUM/DISCOUNT — Above equilibrium = premium (shorts). Below = discount (longs). OTE at 62-79% retracement.
6. KILLZONES (all times UTC/GMT) — Asian (00:00-06:59), London (07:00-11:59), NY_AM (12:00-15:59), NY_PM (16:00-19:59), Off (20:00-23:59). Use EXACTLY one of these values for the killzone field: "Asian", "London", "NY_AM", "NY_PM", "Off".

CRITICAL — XAU/USD STOP LOSS REQUIREMENTS:
Gold is extremely volatile on intraday timeframes. Normal noise on 1H candles is 3-5 ATR.
Historical analysis of 450+ resolved trades shows:
- SLs at 1.0 ATR: stopped 79% of the time (inside noise)
- SLs at 2.0 ATR: stopped 65% of the time (still inside noise)
- SLs at 3.0 ATR: stopped 48% of the time (minimum acceptable)
- SLs at 5.0 ATR: stopped 30% of the time (good survival)
MINIMUM SL: 3.0 ATR from entry. Place SL below/above the structural level (OB low/high) but if the structural level gives < 3.0 ATR, widen to 3.0 ATR.
When calculating TPs, maintain minimum 2:1 R:R against the ACTUAL SL distance (not the structural distance).

Trade Setup Quality Standards:
- Only recommend entries where bias aligns with direction
- Entry at/near validated order block or FVG in appropriate zone
- Stop loss below/above the order block with buffer (minimum 3.0 ATR from entry)
- Minimum 2:1 RR for TP1; aim for 3:1+ for TP2/TP3
- If no high-quality setup exists, set entry to null

You respond with precise, actionable JSON only. No explanation text outside the JSON."""

# Timeframe configs: analysis candle count, HTF for context, API fetch size,
# and check_interval_min = how often (in minutes) to actually fetch this TF.
# Analysis timeframes — 5min/30min/2h removed from Claude analysis loop.
# 5min reserved for entry refinement only (trigger monitoring + LTF precision).
TIMEFRAMES = {
    "15min": {"count": 96,  "fetch": 200, "htf": "1h",   "htf_count": 48, "check_every": 15},
    "1h":    {"count": 120, "fetch": 250, "htf": "4h",   "htf_count": 20, "check_every": 60},
    "4h":    {"count": 40,  "fetch": 100, "htf": "1day", "htf_count": 12, "check_every": 240},
    "1day":  {"count": 40,  "fetch": 100, "htf": None,   "htf_count": 0,  "check_every": 1440},
}

# Per-timeframe expiry
EXPIRY_HOURS = {
    "15min": 8,      # 8 hours
    "1h": 48,        # 2 days
    "4h": 168,       # 7 days
    "1day": 336,     # 14 days
}

# Trading cost assumptions (XAU/USD)
# Round-trip spread cost in price units (entry + exit)
SPREAD_ROUND_TRIP = 0.50



class ScannerEngine:
    """Multi-timeframe headless scanner with candle-change detection."""

    def __init__(self, db: ScannerDB = None):
        # sanitize_env_secret strips invisible Unicode (U+2028/U+2029, ZWSP, BOM,
        # CR/LF) that survive copy-paste into Railway's Variables UI and would
        # otherwise crash httpx header encoding.
        self.claude_key = sanitize_env_secret(os.environ.get("ANTHROPIC_API_KEY"))
        self.db = db or ScannerDB()
        # Priority 5: Haiku false negative tracker (shares scanner.db)
        from ml.haiku_fn_tracker import HaikuFNTracker
        self._fn_tracker = HaikuFNTracker(db_path=self.db.db_path)
        # Priority 8: Cost-per-winner tracker
        from ml.cost_per_winner import CostPerWinnerTracker
        self._cpw_tracker = CostPerWinnerTracker()
        self._last_scan_time = None
        self._last_error = None
        self._total_scans = 0
        self._scans_by_tf = {}

        # Filter rejection counters — tracks why setups are being blocked
        self._filter_stats = {
            "haiku_screened_out": 0,
            "no_trade": 0,
            "duplicate": 0,
            "entry_passed": 0,
            "rr_too_low": 0,
            "opus_rejected": 0,
            "setup_found": 0,
            "total_analyses": 0,
        }

        # Candle hash cache per timeframe — only call Claude when hash changes
        self._candle_hashes: dict[str, str] = {}

        # Post-resolution re-scan cooldown tracker
        self._post_resolution_scans: dict[str, datetime] = {}

        # Last fetch time per timeframe — skip TFs that can't have new data yet
        self._last_fetch_time: dict[str, datetime] = {}

        # HTF candle cache — avoid re-fetching HTF every tick
        self._htf_cache: dict[str, dict] = {}  # tf -> {hash, candles, fetched_at}
        self._corr_cache: dict[str, dict] = {}  # "DXY|interval" -> {candles, fetched_at}

        # Unified candle store — ALL components read from here
        self._candle_store: dict[str, dict] = {}  # "SYMBOL|interval" -> {candles, fetched_at, hash}
        self._CANDLE_TTL = {
            "5min": 240, "15min": 840, "1h": 3540,
            "4h": 14340, "1day": 43200, "1week": 604800,
        }

        # Haiku screen result cache — avoids re-screening unchanged candles
        self._screen_cache: dict[str, dict] = {}

        # Narrative cache with killzone + candle hash invalidation
        self._narrative_cache = {
            "narrative": None, "timestamp": None,
            "killzone": None, "candle_hash_4h": None,
        }

        # Killzone profiler cache (P6/P8) — reloaded by scheduler job
        self._kz_profiler = None
        try:
            from ml.killzone_profiler import KillzoneProfiler
            self._kz_profiler = KillzoneProfiler()
            self._kz_profiler.load()
        except Exception:
            pass

        # Adaptive trigger polling
        self._last_trigger_check = datetime.min

        # Weekly macro narrative cache — 7-day TTL, cleared on weekly close (Sunday 21:00 UTC)
        self._weekly_narrative_cache: dict | None = None
        self._weekly_narrative_fetched_at: datetime | None = None

        # Prospect regeneration tracking (resets daily via killzone transition)
        self._prospect_regen_count = {}  # {killzone: int}
        self._prospect_regen_date = None  # date string for daily reset
        self._prospect_regen_last = {}   # {killzone: datetime} — cooldown tracking

        # Killzone transition tracking
        self._last_killzone = None

    def is_configured(self) -> bool:
        cfg = get_config()
        oanda_ok = bool(cfg.get("oanda_account_id") and cfg.get("oanda_access_token"))
        return bool(oanda_ok and self.claude_key)

    def get_status(self) -> dict:
        stats = self.db.get_stats()
        try:
            fn_stats = self._fn_tracker.get_stats()
        except Exception:
            fn_stats = {}
        try:
            cpw_stats = self._cpw_tracker.get_stats()
        except Exception:
            cpw_stats = {}
        return {
            "running": self.is_configured(),
            "last_scan": self._last_scan_time,
            "total_scans": self._total_scans,
            "scans_by_tf": dict(self._scans_by_tf),
            "last_error": self._last_error,
            "pending_count": stats["pending"],
            "timeframes": list(TIMEFRAMES.keys()),
            "filter_stats": dict(self._filter_stats),
            "haiku_fn_tracking": fn_stats,
            "cost_per_winner": cpw_stats,
        }

    def scan_all_timeframes(self) -> dict:
        """Check all timeframes for new candle data. Only calls Claude where data changed.

        Time-gated: only fetches a TF from Twelve Data when enough time has passed
        for a new candle to exist. This keeps TD API usage under 800 calls/day.

        Per hour TD calls: 12(5M) + 4(15M) + 2(30M) + 1(1H) + 0.5(2H) + 0.25(4H) ≈ 20
        Plus monitor: 12/hr. Plus HTF (cached). Total ≈ 35/hr = 840/day.
        With HTF caching: ~30/hr = 720/day — safely under 800.
        """
        if not self.is_configured():
            return {"status": "not_configured"}

        self._last_scan_time = datetime.utcnow().isoformat()
        now = datetime.utcnow()
        results = {}
        scanned = 0
        skipped = 0
        gated = 0
        td_calls = 0

        # Check for killzone transition — triggers session recap
        try:
            self._check_killzone_transition()
        except Exception as e:
            logger.warning("Killzone transition check failed: %s", e)

        # Load adaptive scan config (P8) — killzone-specific timeframe skips
        _scan_config = {}
        _current_kz = "Off"
        try:
            from ml.prompts import get_current_killzone
            if self._kz_profiler:
                self._kz_profiler.load()  # Refresh from disk (updated by scheduler)
            _current_kz = get_current_killzone()
            data = self._kz_profiler.load() if self._kz_profiler else {}
            _scan_config = data.get("scan_config", {}).get(_current_kz, {})
        except Exception:
            pass

        for tf, cfg in TIMEFRAMES.items():
            # P8: Skip timeframes restricted for current killzone
            if tf in _scan_config.get("skip_timeframes", []):
                results[tf] = {"status": "kz_skipped", "killzone": _current_kz}
                skipped += 1
                continue

            # Time gate: skip if not enough time has passed for a new candle
            # P8: Use killzone-specific interval override if available
            check_every = _scan_config.get("interval_overrides", {}).get(
                tf, cfg.get("check_every", 5))
            last_fetch = self._last_fetch_time.get(tf)
            if last_fetch:
                elapsed = (now - last_fetch).total_seconds() / 60
                if elapsed < check_every - 1:  # 1 min tolerance
                    results[tf] = {"status": "gated", "next_in": f"{check_every - elapsed:.0f}m"}
                    gated += 1
                    continue

            # Fetch candles for this timeframe
            candles = self._fetch_candles(tf, cfg["fetch"])
            td_calls += 1
            self._last_fetch_time[tf] = now
            if not candles:
                results[tf] = {"status": "fetch_failed"}
                continue

            trimmed = candles[-cfg["count"]:]

            # Check if candle data actually changed
            new_hash = self._hash_candles(trimmed)
            old_hash = self._candle_hashes.get(tf)

            if new_hash == old_hash:
                results[tf] = {"status": "unchanged"}
                skipped += 1
                continue

            # New candle data — run analysis
            self._candle_hashes[tf] = new_hash
            scanned += 1

            # Get HTF context candles (cached if recent)
            htf_candles = self._get_htf_candles(cfg["htf"], cfg["htf_count"])
            td_calls += 1 if htf_candles and tf != cfg.get("htf") else 0

            # Build recent context — gives Claude memory of what just happened
            recent_ctx = None
            try:
                from ml.recent_context import build_recent_context
                recent_ctx = build_recent_context(tf, self.db)
            except Exception as e:
                logger.debug("Recent context build failed (proceeding without): %s", e)

            # Run Claude analysis + calibration + store
            result = self._analyze_and_store(tf, trimmed, htf_candles, candles,
                                              recent_context=recent_ctx)
            results[tf] = result

            self._total_scans += 1
            self._scans_by_tf[tf] = self._scans_by_tf.get(tf, 0) + 1

        logger.info(
            "Scanner tick: %d analyzed, %d unchanged, %d gated, %d TD calls",
            scanned, skipped, gated, td_calls,
        )
        return {
            "status": "complete",
            "scanned": scanned,
            "skipped": skipped,
            "gated": gated,
            "td_calls": td_calls,
            "results": results,
        }

    def scan_once(self, timeframe: str = "1h",
                  displacement_context: dict = None) -> dict:
        """Run a single scan on one timeframe (for manual trigger / backward compat).

        Args:
            timeframe: Timeframe to scan ("1h", "4h", etc.)
            displacement_context: Optional displacement zone context injected by
                prospect monitor (Phase B). Keys: zone_high, zone_low, zone_type,
                direction, displacement_confirmed, prospect_id.
        """
        if not self.is_configured():
            return {"status": "not_configured", "error": "Missing API keys"}

        cfg = TIMEFRAMES.get(timeframe)
        if not cfg:
            return {"status": "error", "error": f"Unknown timeframe: {timeframe}"}

        self._total_scans += 1
        self._last_scan_time = datetime.utcnow().isoformat()

        candles = self._fetch_candles(timeframe, cfg["fetch"])
        if not candles:
            self._last_error = f"Failed to fetch {timeframe} candles"
            return {"status": "error", "error": self._last_error}

        trimmed = candles[-cfg["count"]:]

        htf_candles = self._get_htf_candles(cfg["htf"], cfg["htf_count"])

        # Build recent context — same as scan_all_timeframes
        recent_ctx = None
        try:
            from ml.recent_context import build_recent_context
            recent_ctx = build_recent_context(timeframe, self.db)
        except Exception as e:
            logger.debug("Recent context build failed (proceeding without): %s", e)

        # Merge displacement context if provided (Phase B)
        if displacement_context:
            recent_ctx = recent_ctx or {}
            recent_ctx["displacement_context"] = displacement_context

        return self._analyze_and_store(timeframe, trimmed, htf_candles, candles,
                                        recent_context=recent_ctx)

    def _analyze_and_store(self, timeframe: str, candles: list,
                           htf_candles: list, full_candles: list,
                           recent_context: dict = None) -> dict:
        """Screen with Haiku first, then full Sonnet analysis if setup detected."""
        try:
            self._filter_stats["total_analyses"] += 1
            # Priority 8: Accumulate API cost during this analysis pipeline
            self._pending_api_cost = 0.0

            # ── Narrative State Engine: fetch previous thesis + invalidation check ──
            prev_narrative = None
            invalidation_status = "CLEAR"
            try:
                from ml.narrative_state import NarrativeStore, check_invalidation
                ns_store = NarrativeStore(self.db.db_path)
                ns_store.expire_stale(timeframe)
                prev_narrative = ns_store.get_current(timeframe)
                if prev_narrative and candles:
                    current_price = float(candles[-1].get("close", 0))
                    invalidation_status = check_invalidation(
                        prev_narrative, current_price, candles)

                    # Safeguard 5: Structural contradiction detection
                    if prev_narrative.get("directional_bias") in ("bullish", "bearish"):
                        from ml.features import compute_market_structure
                        ms_score = compute_market_structure(candles, lookback=20)
                        thesis_bullish = prev_narrative["directional_bias"] == "bullish"
                        struct_bullish = ms_score > 0.2
                        struct_bearish = ms_score < -0.2
                        if thesis_bullish and struct_bearish:
                            prev_narrative = dict(prev_narrative)
                            prev_narrative["_structural_contradiction"] = (
                                "Your thesis says BULLISH but current structure shows "
                                f"bearish tendency (score: {ms_score:.2f}). Re-evaluate.")
                        elif not thesis_bullish and struct_bullish:
                            prev_narrative = dict(prev_narrative)
                            prev_narrative["_structural_contradiction"] = (
                                "Your thesis says BEARISH but current structure shows "
                                f"bullish tendency (score: {ms_score:.2f}). Re-evaluate.")
            except Exception as e:
                logger.debug("Narrative state fetch failed (proceeding without): %s", e)

            # Step 0.5: Check if Haiku should be bypassed for this segment (P5)
            from ml.prompts import get_current_killzone
            current_kz = get_current_killzone()
            haiku_bypassed = False
            if self._fn_tracker.should_bypass_haiku(timeframe, current_kz):
                logger.info(
                    "Scanner [%s]: Haiku BYPASSED for %s %s (high FN rate)",
                    timeframe, timeframe, current_kz,
                )
                haiku_bypassed = True
                self._filter_stats.setdefault("haiku_bypassed", 0)
                self._filter_stats["haiku_bypassed"] += 1

            # Step 1: Fetch correlated instruments — only for 1H+ timeframes
            # (5min/15min/30min correlations are noise — 20-bar lookback too short)
            # P7: Skip if intermarket recommendation is "noise" (except 4h/1day where sample grows)
            # Moved BEFORE Haiku so Opus has intermarket context, and Haiku gets watch zones.
            INTERMARKET_TIMEFRAMES = {"1h", "2h", "4h", "1day"}
            intermarket_ctx = None
            if timeframe in INTERMARKET_TIMEFRAMES:
                _skip_intermarket = False
                try:
                    from ml.intermarket_validator import IntermarketValidator
                    _im_rec = (IntermarketValidator().get_last_result() or {}).get("recommendation")
                    if _im_rec == "noise" and timeframe not in ("4h", "1day"):
                        _skip_intermarket = True
                        logger.debug("Scanner [%s]: skipping intermarket (recommendation=noise)", timeframe)
                except Exception:
                    pass

                if not _skip_intermarket:
                    try:
                        from ml.intermarket import compute_intermarket_context
                        from ml.prompts import get_current_killzone
                        corr = self._fetch_correlated_candles(timeframe, count=30)
                        session = get_current_killzone()
                        intermarket_ctx = compute_intermarket_context(
                            gold_candles=candles,
                            dxy_candles=corr.get("DXY"),
                            us10y_candles=corr.get("US10Y"),
                            session=session,
                        )
                    except Exception as e:
                        logger.warning("Scanner [%s]: intermarket fetch failed (proceeding without): %s",
                                       timeframe, e)

            # Step 1.5: Opus HTF narrative (cached for 1 hour)
            # Moved BEFORE Haiku so watch zones from Opus can inform the screen.
            htf_narrative = None
            try:
                # Fetch daily + weekly candles for Opus macro context
                daily_candles = self._get_htf_candles("1day", 45)
                weekly_candles = self._get_htf_candles("1week", 12)
                # Opus gets expanded 4H window (40 candles = ~7 days)
                opus_4h = self._get_htf_candles("4h", 40) or htf_candles or []
                htf_narrative = self._call_opus_narrative(
                    opus_4h, daily_candles, intermarket_ctx,
                    weekly_candles=weekly_candles)
            except Exception as e:
                logger.warning("Scanner [%s]: Opus narrative failed (proceeding without): %s",
                               timeframe, e)

            # Extract watch zones from Opus response for Haiku context.
            # Prefer explicit watch_zones field (Phase 3); fall back to key_levels.
            _watch_zones = None
            if htf_narrative:
                if htf_narrative.get("watch_zones"):
                    _watch_zones = [
                        {"level": wz.get("level", wz.get("price")),
                         "type": wz.get("type", "zone"),
                         "status": wz.get("status", "untested")}
                        for wz in htf_narrative["watch_zones"]
                        if wz.get("level") or wz.get("price")
                    ]
                elif htf_narrative.get("key_levels"):
                    _watch_zones = [
                        {"level": kl.get("price", kl.get("level")),
                         "type": kl.get("type", "zone"),
                         "status": kl.get("status", "untested")}
                        for kl in htf_narrative["key_levels"]
                        if kl.get("price") or kl.get("level")
                    ]

            # Prepend displacement zone if provided via scan_once() (Phase B)
            _disp_ctx = (recent_context or {}).get("displacement_context")
            if _disp_ctx and _disp_ctx.get("displacement_confirmed"):
                _disp_zone = {
                    "level": (_disp_ctx["zone_high"] + _disp_ctx["zone_low"]) / 2,
                    "type": _disp_ctx.get("zone_type", "ob"),
                    "status": "displacement_confirmed",
                    "zone_high": _disp_ctx["zone_high"],
                    "zone_low": _disp_ctx["zone_low"],
                    "direction": _disp_ctx["direction"],
                }
                _watch_zones = [_disp_zone] + (_watch_zones or [])

            # Step 2: Context-aware Haiku pre-screen (cheap — ~$0.001/call)
            # Now receives thesis + Opus watch zones so it knows WHAT to look for.
            _pending_for_tf = None
            if not haiku_bypassed:
                try:
                    _pending_for_tf = [
                        s for s in self.db.get_pending()
                        if s.get("timeframe") == timeframe
                    ]
                except Exception:
                    pass
                screen = self._call_claude_screen(
                    candles, htf_candles or [], timeframe,
                    prev_narrative=prev_narrative,
                    watch_zones=_watch_zones,
                    pending_setups=_pending_for_tf)
            else:
                screen = None  # Skip Haiku, proceed to Sonnet

            if screen and not screen.get("setup_possible", False):
                logger.info("Scanner [%s]: Haiku screen — no setup (%s)",
                            timeframe, screen.get("reason", ""))
                self._filter_stats["haiku_screened_out"] += 1

                # Priority 5: Log rejection for false negative tracking
                try:
                    from ml.features import compute_atr, detect_order_blocks, \
                        detect_fvgs, detect_liquidity
                    _atr = compute_atr(candles) if len(candles) >= 14 else 0
                    _last_close = candles[-1].get("close", 0) if candles else 0
                    # Quick structural score (reuse features.py detectors)
                    _obs = detect_order_blocks(candles, _atr) if len(candles) >= 5 and _atr > 0 else []
                    _fvgs = detect_fvgs(candles, _atr) if len(candles) >= 3 else []
                    _sweeps = detect_liquidity(candles) if len(candles) >= 10 else []
                    _struct_score = len(_obs) + len(_fvgs) + len(_sweeps)
                    _confluence = min(
                        int(bool(_obs)) + int(bool(_fvgs)) + int(bool(_sweeps)), 3
                    )
                    self._fn_tracker.log_rejection(
                        timeframe=timeframe,
                        killzone=current_kz,
                        last_close=_last_close,
                        atr=_atr,
                        reason=screen.get("reason", ""),
                        structural_score=_struct_score,
                        confluence_count=_confluence,
                    )
                except Exception as e:
                    logger.debug("FN tracker log failed: %s", e)

                # Priority 5: If Haiku said no but structural score is high
                # and this segment is flagged for loosening, override Haiku
                if (self._fn_tracker.should_loosen_haiku(timeframe, current_kz)
                        and _confluence >= 2):
                    logger.info(
                        "Scanner [%s]: Haiku overridden — loosened for %s %s "
                        "(confluence=%d, struct_score=%d)",
                        timeframe, timeframe, current_kz,
                        _confluence, _struct_score,
                    )
                    self._filter_stats.setdefault("haiku_overridden", 0)
                    self._filter_stats["haiku_overridden"] += 1
                else:
                    self._last_error = None
                    return {"status": "no_setup", "screened": True}

            # Step 2.5: Gather learned rules + placement guidance for Sonnet (zero API cost)
            setup_context = None
            try:
                from ml.setup_profiles import SetupProfileStore
                from ml.entry_placement import EntryPlacementAnalyzer
                rules = SetupProfileStore().get_learned_rules(min_samples=20)
                placement = EntryPlacementAnalyzer().get_placement_guidance()
                if rules or (placement and placement.get("status") == "active"):
                    setup_context = {
                        "learned_rules": rules or [],
                        "placement_guidance": placement if placement.get("status") == "active" else None,
                    }
            except Exception:
                pass

            # Capture Haiku's zone interaction hint for Sonnet
            haiku_zone_hint = None
            if screen and screen.get("zone_interaction"):
                haiku_zone_hint = screen["zone_interaction"]

            # ── ML enrichment: build statistical memory for Sonnet ──
            ml_context = None
            try:
                from ml.calibrate import MLCalibrator
                calibrator = MLCalibrator()
                # Build thesis type from narrative
                thesis_type = None
                if prev_narrative:
                    bias = prev_narrative.get("directional_bias", "")
                    phase = prev_narrative.get("p3_phase", "")
                    if bias and phase:
                        thesis_type = f"{bias}_{phase}"

                # Build DNA pattern from available info
                dna_pattern = None
                try:
                    from ml.setup_dna import encode_setup_dna
                    direction_hint = ((screen or {}).get("direction")
                                      or (prev_narrative or {}).get("directional_bias"))
                    if direction_hint:
                        # Minimal analysis dict for DNA encoding
                        dna_analysis = {
                            "entry": {"direction": direction_hint},
                            "killzone": current_kz,
                        }
                        dna_pattern = encode_setup_dna(
                            dna_analysis, {}, timeframe, current_kz)
                except Exception:
                    pass

                ml_context = calibrator.build_ml_context(
                    thesis_type=thesis_type,
                    timeframe=timeframe,
                    killzone=current_kz,
                    candles=candles,
                    setup_dna_pattern=dna_pattern,
                )
                if ml_context:
                    logger.debug("Scanner [%s]: ML context built — regime=%s, sl_floor=%.1f ATR",
                                 timeframe, ml_context.get("regime", "?"),
                                 ml_context.get("sl_floor_atr", 0))
            except Exception as e:
                logger.debug("ML context build failed (proceeding without): %s", e)

            # Step 3: Haiku said yes (or screen failed) — escalate to Sonnet
            logger.info("Scanner [%s]: Haiku screen passed — running Sonnet analysis%s%s",
                        timeframe,
                        " (with Opus narrative)" if htf_narrative else "",
                        f" (zone hint: {haiku_zone_hint})" if haiku_zone_hint else "")
            analysis = self._call_claude(candles, htf_candles or [], timeframe,
                                         intermarket=intermarket_ctx,
                                         htf_narrative=htf_narrative,
                                         setup_context=setup_context,
                                         prev_narrative=prev_narrative,
                                         invalidation_status=invalidation_status,
                                         recent_context=recent_context,
                                         haiku_zone_hint=haiku_zone_hint,
                                         ml_context=ml_context)
            if not analysis:
                # Preserve specific API error set inside _call_claude (e.g. 401, 400)
                # Only fall back to generic message if no specific error was recorded
                if not self._last_error:
                    self._last_error = f"Claude returned no result for {timeframe}"
                return {"status": "no_result"}

            # ── Save narrative state (even if no setup found) ──
            saved_ns_id = None
            is_revision = False
            current_ns = None
            try:
                from ml.narrative_state import NarrativeStore
                ns = analysis.get("narrative_state")
                if ns and isinstance(ns, dict):
                    ns_store = NarrativeStore(self.db.db_path)
                    saved_ns_id = ns_store.save(timeframe, ns)
                    if saved_ns_id:
                        current_ns = ns_store.get_current(timeframe)
                        is_revision = bool(current_ns and current_ns.get("is_revision"))
                    logger.debug("Scanner [%s]: narrative state saved (bias=%s, p3=%s)",
                                 timeframe, ns.get("directional_bias"), ns.get("p3_phase"))
            except Exception as e:
                logger.debug("Narrative state save failed: %s", e)

            # ── Lifecycle notifications: Stage 1 / 2 / 6 ──
            try:
                from ml.notifications import notify_lifecycle
                ns_data = current_ns or analysis.get("narrative_state") or {}
                thesis_id = ns_data.get("id") or saved_ns_id or str(ns_data.get("thesis", ""))[:8]
                bias_conf = ns_data.get("bias_confidence", 0) or 0
                key_levels = ns_data.get("key_levels") or []
                scan_count = ns_data.get("scan_count", 1) or 1

                # Stage 6: THESIS_REVISED — direction changed + pending setup exists
                if is_revision and prev_narrative:
                    prev_conf = prev_narrative.get("bias_confidence", 0) or 0
                    if prev_conf >= 0.5:
                        # Check if there's a pending setup on this TF
                        pending_on_tf = [
                            s for s in self.db.get_pending()
                            if s.get("timeframe") == timeframe
                        ]
                        if pending_on_tf:
                            notify_lifecycle(
                                6, thesis_id, timeframe, ns_data,
                                setup_data=pending_on_tf[0], db=self.db)

                # Stage 1: THESIS_FORMING — new thesis with decent confidence
                elif scan_count == 1 and bias_conf >= 0.5 and key_levels:
                    notify_lifecycle(1, thesis_id, timeframe, ns_data, db=self.db)

                # Stage 2: THESIS_CONFIRMED — survived 2+ scans, rising confidence
                elif scan_count >= 2 and bias_conf >= 0.7:
                    prev_conf = (prev_narrative or {}).get("bias_confidence", 0) or 0
                    if bias_conf > prev_conf:
                        notify_lifecycle(2, thesis_id, timeframe, ns_data,
                                         db=self.db)

            except Exception as e:
                logger.debug("Lifecycle notification failed: %s", e)

            entry = analysis.get("entry")
            if not entry or not entry.get("price"):
                logger.info("Scanner [%s]: no setup (quality: %s)",
                            timeframe, analysis.get("setup_quality", "?"))
                self._filter_stats["no_trade"] += 1
                self._last_error = None
                return {"status": "no_setup", "quality": analysis.get("setup_quality")}

            direction = entry.get("direction", "long")
            entry_price = entry["price"]

            # Normalize killzone to canonical names (Asian/London/NY_AM/NY_PM/Off)
            # so profiler gates, intermarket stratification, and DB all use same keys
            from ml.killzone_profiler import normalize_killzone
            analysis["killzone"] = normalize_killzone(analysis.get("killzone", ""))

            # Timeframe-scaled duplicate detection:
            # Higher TFs need wider price tolerance (gold moves more in 4H than 15min)
            # and longer lookback windows
            dup_params = {
                "15min": {"minutes": 20, "price_tolerance": 2.0},
                "1h":    {"minutes": 45, "price_tolerance": 3.0},
                "4h":    {"minutes": 180, "price_tolerance": 5.0},
                "1day":  {"minutes": 720, "price_tolerance": 10.0},
            }.get(timeframe, {"minutes": 30, "price_tolerance": 1.0})

            if self.db.find_duplicate(direction, entry_price, timeframe=timeframe,
                                       **dup_params):
                logger.info("Scanner [%s]: duplicate skipped (%s @ %.2f, window=%dm, tol=$%.1f)",
                            timeframe, direction, entry_price,
                            dup_params["minutes"], dup_params["price_tolerance"])
                self._filter_stats["duplicate"] += 1
                return {"status": "duplicate"}

            # Validate entry hasn't already been passed — reject if current
            # price is well beyond entry (look-ahead bias prevention).
            # 0.5% threshold (~$15 on gold at $3000) allows for normal pullbacks
            # to valid entry zones. Previous 0.2% was too tight for gold's volatility.
            entry_passed_pct = get_config().get("scanner_entry_passed_pct", 0.005)
            last_close = float(candles[-1].get("close", 0)) if candles else 0
            if last_close and entry_price:
                if direction == "long" and last_close > entry_price * (1 + entry_passed_pct):
                    logger.info(
                        "Scanner [%s]: REJECTED — entry %.2f already passed (close=%.2f, threshold=%.1f%%)",
                        timeframe, entry_price, last_close, entry_passed_pct * 100)
                    print(f"[SCANNER] Rejected {timeframe}: long entry {entry_price} already passed (close={last_close})")
                    self._filter_stats["entry_passed"] += 1
                    return {"status": "entry_passed", "entry": entry_price, "close": last_close}
                if direction == "short" and last_close < entry_price * (1 - entry_passed_pct):
                    logger.info(
                        "Scanner [%s]: REJECTED — entry %.2f already passed (close=%.2f, threshold=%.1f%%)",
                        timeframe, entry_price, last_close, entry_passed_pct * 100)
                    print(f"[SCANNER] Rejected {timeframe}: short entry {entry_price} already passed (close={last_close})")
                    self._filter_stats["entry_passed"] += 1
                    return {"status": "entry_passed", "entry": entry_price, "close": last_close}

            calibration = self._calibrate(analysis, full_candles)

            # ── Dual-mode tracking: compare ML-informed Sonnet vs old calibration ──
            if calibration and ml_context:
                try:
                    sonnet_sl_atr = (analysis.get("stopLoss") or {}).get("atr_distance", 0)
                    old_consensus_sl = calibration.get("calibrated", {}).get("sl_atr", 0)
                    ml_floor = ml_context.get("sl_floor_atr", 0)
                    safety_net_fired = (sonnet_sl_atr > 0 and ml_floor > 0
                                        and sonnet_sl_atr < ml_floor)
                    logger.info(
                        "Scanner [%s] DUAL-MODE: Sonnet SL=%.1f ATR, "
                        "old consensus=%.1f ATR, ML floor=%.1f ATR, "
                        "safety_net_fired=%s",
                        timeframe, sonnet_sl_atr, old_consensus_sl,
                        ml_floor, safety_net_fired)
                    self._filter_stats.setdefault("ml_safety_net_checks", 0)
                    self._filter_stats.setdefault("ml_safety_net_fired", 0)
                    self._filter_stats["ml_safety_net_checks"] += 1
                    if safety_net_fired:
                        self._filter_stats["ml_safety_net_fired"] += 1
                except Exception:
                    pass

            sl_price = (analysis.get("stopLoss") or {}).get("price")
            tps_data = analysis.get("takeProfits") or []
            tps = [tp.get("price") for tp in tps_data if tp.get("price")]
            rr_ratios = [tp.get("rr", 0) for tp in tps_data]

            cal_sl = None
            if calibration:
                cal_sl = calibration.get("calibrated", {}).get("sl")
                cal_tps = calibration.get("calibrated", {}).get("tps")
                cal_rr = calibration.get("calibrated", {}).get("rr_ratios")
                if cal_tps:
                    tps = cal_tps
                if cal_rr:
                    rr_ratios = cal_rr

            # Reject setups where TP1 R:R is too small to be worth trading
            min_rr = get_config().get("scanner_min_tp1_rr", 0.6)
            tp1_rr = rr_ratios[0] if rr_ratios else 0
            if tp1_rr < min_rr:
                logger.info(
                    "Scanner [%s]: REJECTED — TP1 R:R %.2f below minimum %.1f",
                    timeframe, tp1_rr, min_rr)
                print(f"[SCANNER] Rejected {timeframe}: TP1 R:R {tp1_rr:.2f} < {min_rr} minimum")
                self._filter_stats["rr_too_low"] += 1
                return {"status": "rr_too_low", "tp1_rr": tp1_rr, "min_rr": min_rr}

            # Merge intermarket context into calibration JSON for ML features
            cal_json = calibration or {}
            if intermarket_ctx:
                cal_json["intermarket"] = {
                    "gold_dxy_corr_20": intermarket_ctx.get("gold_dxy_corr_20", 0),
                    "gold_dxy_diverging": intermarket_ctx.get("gold_dxy_diverging", 0),
                    "dxy_range_position": intermarket_ctx.get("dxy_range_position", 0.5),
                    "yield_direction": intermarket_ctx.get("yield_direction", 0),
                    "narrative": intermarket_ctx.get("narrative", ""),
                }

            # Store Opus narrative in calibration JSON for tracking
            if htf_narrative:
                cal_json["opus_narrative"] = {
                    "directional_bias": htf_narrative.get("directional_bias"),
                    "power_of_3_phase": htf_narrative.get("power_of_3_phase"),
                    "phase_confidence": htf_narrative.get("phase_confidence"),
                    "p3_progress": htf_narrative.get("p3_progress"),
                    "bias_confidence": htf_narrative.get("bias_confidence"),
                    "invalidation_level": htf_narrative.get("invalidation_level"),
                    "macro_narrative": htf_narrative.get("macro_narrative", ""),
                }

            # ── Setup DNA: conditional probability + quality adjustment ──
            try:
                from ml.setup_dna import encode_setup_dna
                from ml.setup_profiles import SetupProfileStore

                dna = encode_setup_dna(analysis, calibration, timeframe,
                                       analysis.get("killzone", ""))
                dna_stats = SetupProfileStore().get_conditional_stats(dna)

                if dna_stats and dna_stats["match_count"] >= 15:
                    q = analysis.get("setup_quality", "")
                    # Downgrade: <35% WR with 15+ matches
                    if dna_stats["win_rate"] < 0.35 and q in ("A", "B"):
                        new_q = {"A": "B", "B": "C"}[q]
                        analysis["setup_quality"] = new_q
                        analysis.setdefault("warnings", []).append(
                            f"DNA downgraded {q}->{new_q}: "
                            f"{dna_stats['win_rate']:.0%} WR across {dna_stats['match_count']} similar setups")
                        logger.info("Scanner [%s]: DNA downgrade %s->%s (%.0f%% WR, %d matches)",
                                    timeframe, q, new_q, dna_stats["win_rate"] * 100, dna_stats["match_count"])
                    # Upgrade: >70% WR with 15+ matches
                    elif dna_stats["win_rate"] > 0.70 and q in ("C", "D"):
                        new_q = {"C": "B", "D": "C"}[q]
                        analysis["setup_quality"] = new_q
                        analysis.setdefault("warnings", []).append(
                            f"DNA upgraded {q}->{new_q}: "
                            f"{dna_stats['win_rate']:.0%} WR across {dna_stats['match_count']} similar setups")
                        logger.info("Scanner [%s]: DNA upgrade %s->%s (%.0f%% WR, %d matches)",
                                    timeframe, q, new_q, dna_stats["win_rate"] * 100, dna_stats["match_count"])

                # Store DNA stats in calibration JSON for frontend
                if dna_stats:
                    cal_json["dna_stats"] = {
                        "match_count": dna_stats["match_count"],
                        "win_rate": dna_stats["win_rate"],
                        "avg_rr": dna_stats["avg_rr"],
                    }
            except Exception as e:
                logger.debug("Scanner [%s]: DNA stats skipped: %s", timeframe, e)

            # ── Opus validation — always validate A and B grade setups ──
            is_shadow = False
            quality = analysis.get("setup_quality", "")

            # Always send A and B setups to Opus for validation.
            # Previously had a dynamic gate that narrowed to A-only when
            # Opus reject rate < 5%, but this was counterproductive:
            # it silently skipped B setups without validation, AND those
            # setups lost the "Opus validated" confidence badge.
            validate_qualities = {"A", "B"}

            if quality in validate_qualities:
                try:
                    validation = self._call_claude_validate(
                        analysis, candles, htf_candles, intermarket_ctx, timeframe)
                except Exception as e:
                    logger.warning("Opus validation failed (treating as validated): %s", e)
                    validation = None

                if validation:
                    analysis["opus_validation"] = validation

                    if validation["verdict"] == "rejected":
                        # Check if Opus is too conservative (high false negative rate)
                        try:
                            from ml.claude_bridge import ClaudeAnalysisBridge
                            _policy = ClaudeAnalysisBridge().get_opus_rejection_policy(
                                killzone=killzone, timeframe=timeframe)
                        except Exception:
                            _policy = {"action": "reject"}

                        if _policy.get("action") == "downgrade":
                            # Override: too many rejected setups were winners
                            dg_to = _policy.get("downgrade_to", "C")
                            analysis["setup_quality"] = dg_to
                            warnings = analysis.get("warnings") or []
                            warnings.append(
                                f"Opus rejected but overridden to {dg_to} "
                                f"(FN rate {_policy['false_negative_rate']:.0%})")
                            analysis["warnings"] = warnings
                            logger.info(
                                "Scanner [%s]: Opus REJECTED → overridden to %s "
                                "(FN rate %.0f%%)",
                                timeframe, dg_to,
                                _policy["false_negative_rate"] * 100)
                        elif _policy.get("action") == "allow":
                            # FN rate so high we treat Opus rejection as noise
                            analysis["setup_quality"] = quality  # keep original grade
                            analysis["opus_validated"] = True    # treat as validated
                            warnings = analysis.get("warnings") or []
                            warnings.append(
                                f"Opus rejection overridden to ALLOW "
                                f"(FN rate {_policy['false_negative_rate']:.0%} in {killzone})")
                            analysis["warnings"] = warnings
                            logger.info(
                                "Scanner [%s]: Opus REJECTED → ALLOWED "
                                "(FN rate %.0f%% in %s)",
                                timeframe, _policy["false_negative_rate"] * 100, killzone)
                        else:
                            logger.info(
                                "Scanner [%s]: Opus REJECTED — %s",
                                timeframe, validation.get("validation_note", ""))
                            print(f"[SCANNER] Opus REJECTED {timeframe} {direction} setup: "
                                  f"{validation.get('validation_note', '')}")
                            # Store as shadow for outcome tracking
                            analysis["setup_quality"] = "no_trade"
                            is_shadow = True

                    elif validation["verdict"] == "downgraded":
                        analysis["setup_quality"] = validation.get(
                            "adjusted_quality", quality)
                        warnings = analysis.get("warnings") or []
                        warnings.append(
                            f"Opus downgraded: {validation.get('validation_note', '')}")
                        analysis["warnings"] = warnings
                        logger.info(
                            "Scanner [%s]: Opus DOWNGRADED %s->%s — %s",
                            timeframe, quality, analysis["setup_quality"],
                            validation.get("validation_note", ""))

                    elif validation["verdict"] == "validated":
                        analysis["opus_validated"] = True
                        analysis["validation_note"] = validation.get(
                            "validation_note", "")
                        logger.info(
                            "Scanner [%s]: Opus VALIDATED — %s",
                            timeframe, validation.get("validation_note", ""))
                else:
                    # Opus failed — graceful degradation, treat as validated
                    logger.info("Scanner [%s]: Opus unavailable — proceeding with setup",
                                timeframe)

            # ── Killzone quality gate (P6) — skip if below learned bar ──
            if not is_shadow:
                try:
                    _kz_prof = self._kz_profiler
                    if _kz_prof:
                        _setup_kz = analysis.get("killzone", "")
                        _setup_q = analysis.get("setup_quality", "")
                        if _kz_prof.should_skip(_setup_kz, _setup_q):
                            gate = _kz_prof._gates.get(_setup_kz, {})
                            logger.info(
                                "Scanner [%s]: Killzone gate SKIPPED %s %s (requires %s, kz WR %.0f%%)",
                                timeframe, _setup_kz, _setup_q,
                                gate.get("min_quality", "?"),
                                gate.get("win_rate", 0) * 100)
                            analysis["setup_quality"] = "no_trade"
                            is_shadow = True
                except Exception as e:
                    logger.debug("Killzone profiler check failed: %s", e)

            # ── Entry zone placement tracking ──
            ez_type = ez_high = ez_low = ez_pos = None
            try:
                from ml.entry_placement import identify_entry_zone, compute_entry_position
                from ml.features import compute_atr
                _atr = compute_atr(candles, 14) if len(candles) >= 15 else 1.0
                _zone = identify_entry_zone(entry_price, analysis, _atr)
                if _zone:
                    ez_type = _zone["zone_type"]
                    ez_high = _zone["zone_high"]
                    ez_low = _zone["zone_low"]
                    ez_pos = round(compute_entry_position(
                        entry_price, ez_high, ez_low, direction
                    ), 4)
            except Exception as e:
                logger.debug("Entry zone computation failed: %s", e)

            # Derive thesis_id for lifecycle tracking (same logic as line ~634)
            _store_thesis_id = None
            try:
                _ns = current_ns or analysis.get("narrative_state") or {}
                _store_thesis_id = _ns.get("id") or saved_ns_id or str(_ns.get("thesis", ""))[:8] or None
            except Exception:
                pass

            setup_id = self.db.store_setup(
                direction=direction,
                bias=analysis.get("bias", ""),
                entry_price=entry_price,
                sl_price=sl_price,
                calibrated_sl=cal_sl,
                tps=tps,
                setup_quality=analysis.get("setup_quality", ""),
                killzone=analysis.get("killzone", ""),
                rr_ratios=rr_ratios,
                analysis_json=analysis,
                calibration_json=cal_json,
                candle_hash=self._hash_candles(candles),
                timeframe=timeframe,
                status=("shadow" if is_shadow
                        else "monitoring" if analysis.get("setup_quality", "") in ("C", "D")
                        else "pending"),
                entry_zone_type=ez_type,
                entry_zone_high=ez_high,
                entry_zone_low=ez_low,
                entry_zone_position=ez_pos,
                thesis_id=_store_thesis_id,
            )

            # Priority 8: Store accumulated API cost on this setup
            if self._pending_api_cost > 0:
                try:
                    self.db.update_api_cost(setup_id, self._pending_api_cost)
                except Exception as e:
                    logger.debug("Failed to store api_cost_usd: %s", e)

            self._last_error = None

            if is_shadow:
                self._filter_stats["opus_rejected"] += 1
                logger.info(
                    "Scanner [%s]: SHADOW — %s @ %.2f (Opus rejected, tracking outcome) | ID %s",
                    timeframe, direction.upper(), entry_price, setup_id)
                return {
                    "status": "opus_rejected",
                    "setup_id": setup_id,
                    "timeframe": timeframe,
                    "direction": direction,
                    "entry": entry_price,
                    "quality": quality,
                    "opus_note": (analysis.get("opus_validation") or {}).get(
                        "validation_note", ""),
                }

            # ── C/D grade → monitoring pipeline (Phase 4) ──
            _setup_quality = analysis.get("setup_quality", "")
            if _setup_quality in ("C", "D"):
                self._filter_stats.setdefault("cd_monitoring", 0)
                self._filter_stats["cd_monitoring"] += 1
                logger.info(
                    "Scanner [%s]: MONITORING — %s @ %.2f | Grade %s | ID %s",
                    timeframe, direction.upper(), entry_price,
                    _setup_quality, setup_id)
                return {
                    "status": "monitoring",
                    "setup_id": setup_id,
                    "timeframe": timeframe,
                    "direction": direction,
                    "entry": entry_price,
                    "quality": _setup_quality,
                }

            logger.info(
                "Scanner [%s]: SETUP — %s @ %.2f | SL %.2f | TPs %s | %s%s | ID %s",
                timeframe, direction.upper(), entry_price, sl_price or 0,
                [f"{tp:.2f}" for tp in tps],
                analysis.get("setup_quality", "?"),
                " (Opus ✓)" if analysis.get("opus_validated") else "",
                setup_id,
            )

            # Regime-aware quality gate — stricter in VOLATILE_CHOPPY/QUIET_DRIFT
            _grade_order = {"A": 4, "B": 3, "C": 2, "D": 1}
            _current_grade = analysis.get("setup_quality", "")
            _current_confluences = len(analysis.get("confluences", []))
            _regime_gates = get_config().get("regime_quality_gates", {})
            _struct_regime = (cal_json or {}).get("volatility_context", {}).get(
                "structural_regime", "RANGING")
            _regime_gate = _regime_gates.get(_struct_regime, {"min_grade": "B", "min_confluences": 2})
            _regime_min_grade = _regime_gate.get("min_grade", "B")
            _passes_regime_gate = (
                _grade_order.get(_current_grade, 0) >= _grade_order.get(_regime_min_grade, 3)
                and _current_confluences >= _regime_gate.get("min_confluences", 2)
            )
            if not _passes_regime_gate and _current_grade in ("A", "B"):
                logger.info(
                    "Scanner [%s]: Grade %s with %d confluences blocked by "
                    "%s regime gate (needs %s, %d confluences)",
                    timeframe, _current_grade, _current_confluences,
                    _struct_regime, _regime_min_grade,
                    _regime_gate.get("min_confluences", 2))
            notify_quality = _current_grade in ("A", "B") and _passes_regime_gate

            last_close = candles[-1].get("close", 0) if candles else 0
            proximity_pct = abs(last_close - entry_price) / entry_price if entry_price else 1

            # Layer 1: Always send immediate detection alert for A/B setups
            if notify_quality:
                notify_setup_detected({
                    "direction": direction,
                    "entry_price": entry_price,
                    "current_price": last_close,
                    "setup_quality": analysis.get("setup_quality", ""),
                    "killzone": analysis.get("killzone", ""),
                    "timeframe": timeframe,
                    "tps": tps,
                    "sl_price": sl_price,
                })
                self.db.mark_detection_notified(setup_id)

                # Stage 3: SETUP_DETECTED lifecycle (log-only, completes audit trail)
                try:
                    from ml.notifications import notify_lifecycle
                    _tid = _store_thesis_id or thesis_id
                    if _tid:
                        notify_lifecycle(3, _tid, timeframe, ns_data or {},
                                         setup_data={"id": setup_id}, db=self.db)
                except Exception as e:
                    logger.debug("Lifecycle stage 3 failed: %s", e)

            # Layer 2: Full entry alert if price is already at entry
            # Otherwise defer to proximity monitor (check_entry_proximity, 90s)
            if notify_quality and proximity_pct <= 0.003:
                # Price already at entry — Stage 4 lifecycle is the single
                # entry notification (notify_new_setup suppressed to dedup)
                self.db.mark_notified(setup_id)

                # Stage 4: ENTRY_READY lifecycle
                try:
                    from ml.notifications import notify_lifecycle
                    _tid = _store_thesis_id or thesis_id
                    if _tid:
                        notify_lifecycle(4, _tid, timeframe, ns_data or {},
                                         setup_data={"id": setup_id,
                                                      "entry_price": entry_price,
                                                      "sl_price": sl_price,
                                                      "calibrated_sl": cal_sl,
                                                      "direction": direction,
                                                      "setup_quality": analysis.get("setup_quality", ""),
                                                      "killzone": analysis.get("killzone", ""),
                                                      "tps": tps,
                                                      "timeframe": timeframe},
                                         calibration=calibration or {},
                                         db=self.db)
                except Exception as e:
                    logger.debug("Lifecycle stage 4 failed: %s", e)

            elif notify_quality:
                # Price is away from entry — defer full alert to proximity monitor
                logger.info("Scanner [%s]: detection alert sent, deferring entry alert — price $%.2f is %.1f%% from entry $%.2f",
                            timeframe, last_close, proximity_pct * 100, entry_price)
            else:
                # C/D grade — store for tracking but don't notify
                self.db.mark_notified(setup_id)
                logger.info("Scanner [%s]: %s grade — stored but not notified",
                            timeframe, analysis.get("setup_quality", "?"))

            self._filter_stats["setup_found"] += 1
            return {
                "status": "setup_found",
                "setup_id": setup_id,
                "timeframe": timeframe,
                "direction": direction,
                "entry": entry_price,
                "sl": sl_price,
                "calibrated_sl": cal_sl,
                "tps": tps,
                "quality": analysis.get("setup_quality"),
                "opus_validated": analysis.get("opus_validated", False),
            }

        except Exception as e:
            self._last_error = str(e)
            logger.error("Scanner [%s]: failed — %s", timeframe, e, exc_info=True)
            return {"status": "error", "error": str(e)}

    def check_entry_proximity(self) -> dict:
        """Check unnotified setups — send alert when price approaches entry.

        Runs every 90 seconds via the trigger monitor. No Claude API calls.
        Only needs current price from the candle store.

        Returns: {"checked": int, "notified": int, "missed": int}
        """
        unnotified = self.db.get_unnotified_setups()
        if not unnotified:
            return {"checked": 0, "notified": 0, "missed": 0}

        # Get latest price from candle store or fetch fresh 5min
        cached = self._candle_store.get("XAU_USD|5min")
        if cached and cached.get("candles"):
            current_price = float(cached["candles"][-1].get("close", 0))
        else:
            # Try to get any recent candle
            for key, val in self._candle_store.items():
                if key.startswith("XAU_USD") and val.get("candles"):
                    current_price = float(val["candles"][-1].get("close", 0))
                    break
            else:
                # Fetch fresh
                candles = self._get_candles("XAU_USD", "5min", 2)
                current_price = float(candles[-1].get("close", 0)) if candles else 0

        if not current_price:
            return {"checked": len(unnotified), "notified": 0, "missed": 0}

        from ml.config import get_config
        cfg = get_config()
        proximity_pct = cfg.get("entry_proximity_pct", 0.003)  # 0.3%
        entry_passed_pct = cfg.get("scanner_entry_passed_pct", 0.005)  # 0.5%

        notified_count = 0
        missed_count = 0

        for setup in unnotified:
            entry_price = setup.get("entry_price", 0)
            direction = setup.get("direction", "")
            if not entry_price:
                continue

            distance_pct = abs(current_price - entry_price) / entry_price

            # Check if price is approaching entry zone
            approaching = False
            if direction == "long" and current_price <= entry_price * (1 + proximity_pct):
                approaching = True
            elif direction == "short" and current_price >= entry_price * (1 - proximity_pct):
                approaching = True

            # Check if entry was missed (price moved too far past)
            missed = False
            if direction == "long" and current_price > entry_price * (1 + entry_passed_pct):
                missed = True
            elif direction == "short" and current_price < entry_price * (1 - entry_passed_pct):
                missed = True

            if missed:
                # Entry level was passed — notify with recalculated R:R
                # so user can decide whether to enter at market
                tps = []
                for k in ("tp1", "tp2", "tp3"):
                    if setup.get(k):
                        tps.append(setup[k])
                notify_entry_missed({
                    "direction": direction,
                    "entry_price": entry_price,
                    "current_price": current_price,
                    "setup_quality": setup.get("setup_quality", ""),
                    "killzone": setup.get("killzone", ""),
                    "timeframe": setup.get("timeframe", ""),
                    "tps": tps,
                    "sl_price": setup.get("sl_price", 0),
                })
                self.db.mark_notified(setup["id"])
                missed_count += 1
                logger.info("Entry proximity: MISSED (notified) %s %s @ %.2f (price now %.2f)",
                            direction, setup["id"], entry_price, current_price)
                continue

            if approaching:
                # Send the notification now
                cal_json = setup.get("calibration_json", {})
                if isinstance(cal_json, str):
                    import json as _json
                    cal_json = _json.loads(cal_json) if cal_json else {}

                tps = []
                for k in ("tp1", "tp2", "tp3"):
                    if setup.get(k):
                        tps.append(setup[k])

                # Stage 4 lifecycle is the single entry notification
                # (notify_new_setup suppressed to dedup)
                self.db.mark_notified(setup["id"])
                notified_count += 1
                logger.info("Entry proximity: NOTIFIED %s %s @ %.2f (price %.2f, %.1f%% away)",
                            direction, setup["id"], entry_price, current_price, distance_pct * 100)

                # ── Stage 4: ENTRY_READY (proximity-triggered) ──
                try:
                    from ml.notifications import notify_lifecycle
                    _thesis_id = setup.get("thesis_id")
                    if _thesis_id:
                        notify_lifecycle(4, _thesis_id, setup.get("timeframe", ""), {},
                                        setup_data=setup, calibration=cal_json,
                                        db=self.db)
                except Exception as e:
                    logger.debug("Lifecycle stage 4 (proximity) failed: %s", e)

        return {"checked": len(unnotified), "notified": notified_count, "missed": missed_count}

    def unified_monitor(self) -> dict:
        """Single monitoring loop for all active items.

        Replaces separate monitor_pending(), monitor_cd_setups(), and
        monitor_prospect_triggers() with one loop sharing a single 5-min
        candle fetch. Priority order:
          1. A/B pending setups — SL/TP resolution
          2. A/B entry proximity alerts
          3. C/D monitoring setups — displacement watch
          4. Prospect watch zones — trigger detection

        Cross-references: when a prospect zone displacement is detected
        near a C/D monitoring setup, fast-tracks promotion.
        """
        result = {
            "pending": {"checked": 0, "resolved": 0},
            "proximity": {"checked": 0, "notified": 0, "missed": 0},
            "cd_monitoring": {"checked": 0, "promoted": 0, "expired": 0},
            "prospects": {"checked": 0, "triggered": 0, "displaced": 0},
        }

        # ── Shared 5-min candle fetch ──
        # Calculate optimal count: enough for oldest pending setup, minimum 100
        candle_count = 100
        pending = self.db.get_pending(include_shadow=True)
        if pending:
            oldest_created = min(s.get("created_at", "") for s in pending)
            if oldest_created:
                try:
                    oldest_dt = datetime.fromisoformat(oldest_created)
                    hours_back = (datetime.utcnow() - oldest_dt).total_seconds() / 3600
                    candle_count = max(min(int(hours_back * 12) + 12, 500), 100)
                except (ValueError, TypeError):
                    pass

        try:
            raw_candles = self._fetch_candles("5min", candle_count)
            if not raw_candles:
                logger.warning("Unified monitor: no 5-min candle data")
                return result
            # Ensure chronological order (oldest first)
            if (len(raw_candles) >= 2
                    and raw_candles[0].get("datetime", "") > raw_candles[-1].get("datetime", "")):
                candles_chrono = list(reversed(raw_candles))
            else:
                candles_chrono = raw_candles
        except Exception as e:
            logger.error("Unified monitor: candle fetch failed: %s", e)
            return result

        # ── Priority 1: A/B pending SL/TP resolution ──
        try:
            result["pending"] = self.monitor_pending(candles_5m=candles_chrono)
        except Exception as e:
            logger.error("Unified monitor: pending check failed: %s", e)
            result["pending"] = {"error": str(e)}

        # ── Priority 2: A/B entry proximity alerts ──
        try:
            result["proximity"] = self.check_entry_proximity()
        except Exception as e:
            logger.debug("Unified monitor: proximity check failed: %s", e)
            result["proximity"] = {"error": str(e)}

        # ── Priority 3: C/D displacement monitoring ──
        try:
            result["cd_monitoring"] = self.monitor_cd_setups(candles_5m=candles_chrono)
        except Exception as e:
            logger.debug("Unified monitor: CD monitor failed: %s", e)
            result["cd_monitoring"] = {"error": str(e)}

        # ── Priority 4: Prospect zone triggers ──
        try:
            result["prospects"] = self.monitor_prospect_triggers(candles_5m=candles_chrono)
        except Exception as e:
            logger.debug("Unified monitor: prospect triggers failed: %s", e)
            result["prospects"] = {"error": str(e)}

        # ── Cross-reference: zone displacement → fast-track nearby C/D ──
        try:
            self._cross_reference_zones_with_cd(result, candles_chrono)
        except Exception as e:
            logger.debug("Unified monitor: cross-reference failed: %s", e)

        # Priority 5: Resolve Haiku false negatives
        try:
            self._fn_tracker.expire_stale(max_age_hours=72)
            fn_result = self._fn_tracker.resolve_rejections(candles_chrono)
            result["fn_tracking"] = fn_result
        except Exception as e:
            logger.debug("FN tracker resolution failed: %s", e)

        return result

    def _cross_reference_zones_with_cd(self, monitor_result: dict,
                                        candles_chrono: list):
        """When prospect triggers detect displacement, check if any C/D
        monitoring setups are near that zone and fast-track promotion.

        This bridges the prospect system with the C/D monitoring pipeline.
        """
        prospects_result = monitor_result.get("prospects", {})
        newly_displaced = prospects_result.get("displaced", 0)
        if not newly_displaced:
            return

        # Get recently displaced prospects for their zone info
        displaced = self.db.get_displaced_prospects()
        if not displaced:
            return

        monitoring = self.db.get_monitoring_setups()
        if not monitoring:
            return

        current_price = candles_chrono[-1].get("close", 0) if candles_chrono else 0

        for prospect in displaced:
            disp_data = prospect.get("trigger_result", {})
            if not isinstance(disp_data, dict):
                continue
            setup = disp_data.get("setup", {})
            displacement = disp_data.get("displacement", {})
            ob = displacement.get("ob_zone", {})
            zone_mid = (ob.get("high", 0) + ob.get("low", 0)) / 2
            zone_bias = setup.get("bias", "")

            if not zone_mid:
                continue

            for cd_setup in monitoring:
                cd_entry = cd_setup.get("entry_price", 0)
                cd_direction = cd_setup.get("direction", "")

                # Check direction match
                directions_match = (
                    (cd_direction == "long" and zone_bias == "bullish") or
                    (cd_direction == "short" and zone_bias == "bearish")
                )
                if not directions_match:
                    continue

                # Check proximity (within 1.5 ATR)
                try:
                    from ml.features import compute_atr
                    atr = compute_atr(candles_chrono, 14) if len(candles_chrono) >= 15 else 5.0
                except Exception:
                    atr = 5.0

                if abs(cd_entry - zone_mid) <= 1.5 * atr:
                    # Fast-track promotion!
                    self.db.promote_setup(cd_setup["id"])
                    logger.info(
                        "Cross-ref: PROMOTED C/D %s — near displaced zone %.2f-%.2f",
                        cd_setup["id"], ob.get("low", 0), ob.get("high", 0))

                    # Notify
                    try:
                        from ml.notifications import notify_setup_detected
                        tps = [cd_setup.get(k) for k in ("tp1", "tp2", "tp3")
                               if cd_setup.get(k) is not None]
                        notify_setup_detected({
                            "direction": cd_direction,
                            "entry_price": cd_entry,
                            "current_price": current_price,
                            "setup_quality": cd_setup.get("setup_quality", ""),
                            "killzone": cd_setup.get("killzone", ""),
                            "timeframe": cd_setup.get("timeframe", ""),
                            "tps": tps,
                            "sl_price": cd_setup.get("sl_price", 0),
                            "promoted_from_cd": True,
                            "cross_referenced": True,
                        })
                        self.db.mark_detection_notified(cd_setup["id"])
                    except Exception:
                        pass

                    # Update result counter
                    cd_result = monitor_result.get("cd_monitoring", {})
                    cd_result["promoted"] = cd_result.get("promoted", 0) + 1
                    break  # One promotion per prospect displacement

    def monitor_pending(self, candles_5m: list | None = None) -> dict:
        """Check pending setups against price history for auto-resolution.

        Fetches recent 5-min candles and walks through them chronologically
        for each setup. This ensures that if SL was hit on an earlier candle
        before TP was hit on a later candle, it correctly counts as a loss.

        Args:
            candles_5m: Optional pre-fetched 5-min candles (chronological order).
                        If provided, skips internal candle fetch.
        """
        # Include shadow setups for outcome tracking (Opus value measurement)
        pending = self.db.get_pending(include_shadow=True)
        if not pending:
            return {"checked": 0, "resolved": 0}

        expired = self.db.expire_by_timeframe(EXPIRY_HOURS)
        if expired:
            logger.info("Scanner: expired %d stale setups", expired)

        if candles_5m is not None:
            candle_count = len(candles_5m)
        else:
            # Fetch enough 5-min candles to cover the oldest pending setup.
            # Each candle = 5 min, so hours_back * 12 = candles needed.
            # Cap at 500 (≈42 hours) to stay within API limits.
            oldest_created = min(s.get("created_at", "") for s in pending)
            if oldest_created:
                from datetime import datetime
                try:
                    oldest_dt = datetime.fromisoformat(oldest_created)
                    hours_back = (datetime.utcnow() - oldest_dt).total_seconds() / 3600
                    candle_count = min(int(hours_back * 12) + 12, 500)  # +1hr buffer
                except (ValueError, TypeError):
                    candle_count = 100
            else:
                candle_count = 100
        if candles_5m is not None:
            candles_chrono = candles_5m  # Already chronological
        else:
            candle_count = max(candle_count, 24)  # minimum 2 hours

            try:
                candles = self._fetch_candles("5min", candle_count)
                if not candles:
                    return {"checked": 0, "resolved": 0, "error": "No candle data"}
            except Exception as e:
                return {"checked": 0, "resolved": 0, "error": str(e)}

            # Candles come newest-first from OANDA; reverse for chronological walk.
            candles_chrono = list(reversed(candles))

        resolved_count = 0
        for setup in pending:
            created_at = setup.get("created_at", "")
            result = self._check_setup_against_history(setup, candles_chrono, created_at)
            if result:
                # Compute MFE/MAE from the candles used for resolution
                _mfe_atr = _mae_atr = None
                try:
                    from ml.entry_placement import compute_live_mfe_mae
                    from ml.features import compute_atr
                    _ep = setup.get("entry_price", 0)
                    _dir = setup.get("direction", "long")
                    _atr = compute_atr(candles_chrono, 14) if len(candles_chrono) >= 15 else 1.0
                    if _ep and _atr > 0:
                        # Use candles from setup creation onward
                        _entry_candles = [
                            c for c in candles_chrono
                            if c.get("datetime", "") >= created_at
                        ]
                        if _entry_candles:
                            _mm = compute_live_mfe_mae(_entry_candles, _ep, _dir, _atr)
                            _mfe_atr = _mm["mfe_atr"]
                            _mae_atr = _mm["mae_atr"]
                except Exception:
                    pass

                # Attach MFE/MAE to result for downstream use
                result["mfe_atr"] = _mfe_atr
                result["mae_atr"] = _mae_atr

                # Layer 3 safety net: send entry alert before resolving
                # if setup was never notified (race condition prevention)
                if (setup.get("notified") in (None, 0)
                        and setup.get("setup_quality") in ("A", "B")
                        and setup.get("status") != "shadow"):
                    tps_for_notify = [
                        setup.get(k) for k in ("tp1", "tp2", "tp3")
                        if setup.get(k) is not None
                    ]
                    cal_json = setup.get("calibration_json") or {}
                    if isinstance(cal_json, str):
                        try:
                            cal_json = json.loads(cal_json) if cal_json else {}
                        except (json.JSONDecodeError, TypeError):
                            cal_json = {}
                    notify_new_setup({
                        "direction": setup.get("direction", ""),
                        "bias": setup.get("bias", ""),
                        "entry_price": setup.get("entry_price", 0),
                        "sl_price": setup.get("sl_price", 0),
                        "calibrated_sl": setup.get("calibrated_sl"),
                        "tps": tps_for_notify,
                        "rr_ratios": setup.get("rr_ratios") or [],
                        "setup_quality": setup.get("setup_quality", ""),
                        "killzone": setup.get("killzone", ""),
                        "timeframe": setup.get("timeframe", ""),
                        "calibration_json": cal_json,
                        "opus_validated": (setup.get("analysis_json") or {}).get("opus_validated", False),
                        "current_price": result.get("price", 0),
                    })
                    self.db.mark_notified(setup["id"])
                    logger.info("Safety net: sent entry alert for %s before resolution",
                                setup["id"])

                self.db.resolve_setup(
                    setup["id"], result["outcome"],
                    resolved_price=result.get("price"),
                    pnl_rr=result.get("rr"),
                    auto=True,
                    gross_rr=result.get("gross_rr"),
                    cost_rr=result.get("cost_rr"),
                    mfe_atr=_mfe_atr,
                    mae_atr=_mae_atr,
                )
                resolved_count += 1
                self._log_trade_complete(setup, result, candles=candles_chrono)
                logger.info(
                    "Scanner: auto-resolved %s [%s] — %s @ %.2f (%.1fR)",
                    setup["id"], setup.get("timeframe", "?"),
                    result["outcome"], result.get("price", 0), result.get("rr", 0),
                )

                # Block prospect zone for remainder of killzone on stop-out
                if (result["outcome"] == "stopped_out"
                        and (setup.get("analysis_json") or {}).get("prospect_triggered")
                        and setup.get("entry_zone_high") and setup.get("entry_zone_low")):
                    try:
                        _zone_key = self.db._make_zone_key(
                            setup["direction"],
                            setup.get("timeframe", "1h"),
                            setup["entry_zone_high"],
                            setup["entry_zone_low"])
                        self.db.block_zone_for_killzone(
                            _zone_key, get_current_killzone(),
                            setup_id=setup["id"])
                        logger.info("Zone blocked for killzone: %s (stop-out on %s)",
                                    _zone_key, setup["id"])
                    except Exception as _e:
                        logger.debug("Zone cooldown block failed: %s", _e)

                # Phase 2: Immediate re-analysis with enriched context
                # Only on real resolutions (SL/TP), not expired setups
                if result["outcome"] in ("stopped_out", "tp1", "tp2", "tp3"):
                    try:
                        self._trigger_post_resolution_scan(
                            setup.get("timeframe", "1h"), result["outcome"])
                    except Exception as e:
                        logger.warning("Post-resolution scan failed: %s", e)

        # Priority 5: Resolve Haiku false negatives using the same 5-min candles
        fn_result = {"checked": 0, "resolved": 0, "false_negatives": 0}
        try:
            self._fn_tracker.expire_stale(max_age_hours=72)
            fn_result = self._fn_tracker.resolve_rejections(candles_chrono)
        except Exception as e:
            logger.debug("FN tracker resolution failed: %s", e)

        return {
            "checked": len(pending),
            "resolved": resolved_count,
            "expired": expired,
            "fn_tracking": fn_result,
        }

    def monitor_cd_setups(self, candles_5m: list | None = None) -> dict:
        """Check monitoring (C/D grade) setups for promotion criteria.

        Promotion requires ALL of:
        1. Current price within 1.0 ATR of entry
        2. A displacement candle (>= 2.0 ATR body) in setup direction
        3. Liquidity sweep in the correct direction

        Promoted setups become 'pending' and get notified.

        Args:
            candles_5m: Optional pre-fetched 5-min candles (chronological order).
        """
        monitoring = self.db.get_monitoring_setups()
        if not monitoring:
            return {"checked": 0, "promoted": 0, "expired": 0}

        # Expire old monitoring setups using same windows as pending
        expired = self.db.expire_by_timeframe(EXPIRY_HOURS, status="monitoring")
        if expired:
            logger.info("Scanner: expired %d stale monitoring setups", expired)

        # Re-fetch after expiry (some may have been expired)
        monitoring = self.db.get_monitoring_setups()
        if not monitoring:
            return {"checked": 0, "promoted": 0, "expired": expired}

        # Use pre-fetched candles or fetch fresh
        if candles_5m is not None:
            current_price = candles_5m[-1].get("close", 0) if candles_5m else 0
        else:
            try:
                candles_5m = self._fetch_candles("5min", 100)
                if not candles_5m:
                    return {"checked": len(monitoring), "promoted": 0, "expired": expired}
                # Reverse if newest-first
                if (len(candles_5m) >= 2
                        and candles_5m[0].get("datetime", "") > candles_5m[-1].get("datetime", "")):
                    candles_5m = list(reversed(candles_5m))
                current_price = candles_5m[-1].get("close", 0)
            except Exception as e:
                logger.debug("CD monitor: failed to fetch 5m candles: %s", e)
                return {"checked": len(monitoring), "promoted": 0, "expired": expired}

        if not candles_5m or not current_price:
            return {"checked": len(monitoring), "promoted": 0, "expired": expired}

        promoted = 0
        tf_candle_cache = {}  # Cache per-TF candles to avoid redundant API calls

        for setup in monitoring:
            entry_price = setup.get("entry_price", 0)
            direction = setup.get("direction", "long")
            tf = setup.get("timeframe", "1h")

            if not entry_price:
                continue

            # Get setup-timeframe candles (cached per TF)
            if tf not in tf_candle_cache:
                try:
                    tf_cfg = TIMEFRAMES.get(tf, {"fetch": 180})
                    raw = self._fetch_candles(tf, min(tf_cfg.get("fetch", 180), 50))
                    if raw and len(raw) >= 2:
                        if raw[0].get("datetime", "") > raw[-1].get("datetime", ""):
                            raw = list(reversed(raw))
                        tf_candle_cache[tf] = raw
                    else:
                        tf_candle_cache[tf] = None
                except Exception:
                    tf_candle_cache[tf] = None

            tf_candles = tf_candle_cache.get(tf)
            if not tf_candles or len(tf_candles) < 15:
                continue

            # Compute ATR on setup timeframe
            try:
                from ml.features import compute_atr
                atr = compute_atr(tf_candles, 14)
            except Exception:
                continue
            if atr <= 0:
                continue

            # ── Criterion 1: Price within 1.0 ATR of entry ──
            price_distance = abs(current_price - entry_price)
            if price_distance > atr:
                continue

            # ── Criterion 2: Displacement candle (>= 2.0 ATR body) on setup TF ──
            has_displacement = False
            for c in tf_candles[-3:]:  # Last 3 candles on this TF
                body = abs(c.get("close", 0) - c.get("open", 0))
                if body >= 2.0 * atr:
                    if direction == "long" and c.get("close", 0) > c.get("open", 0):
                        has_displacement = True
                        break
                    elif direction == "short" and c.get("close", 0) < c.get("open", 0):
                        has_displacement = True
                        break
            if not has_displacement:
                continue

            # ── Criterion 3: Liquidity sweep on 5-min candles ──
            has_sweep = self._check_liquidity_sweep(candles_5m, direction)
            if not has_sweep:
                continue

            # ── All criteria met — promote! ──
            self.db.promote_setup(setup["id"])
            promoted += 1

            # Send detection notification for promoted setup
            try:
                from ml.notifications import notify_setup_detected
                tps = [setup.get(k) for k in ("tp1", "tp2", "tp3")
                       if setup.get(k) is not None]
                notify_setup_detected({
                    "direction": direction,
                    "entry_price": entry_price,
                    "current_price": current_price,
                    "setup_quality": setup.get("setup_quality", ""),
                    "killzone": setup.get("killzone", ""),
                    "timeframe": tf,
                    "tps": tps,
                    "sl_price": setup.get("sl_price", 0),
                    "promoted_from_cd": True,
                })
                self.db.mark_detection_notified(setup["id"])
            except Exception as e:
                logger.debug("CD promotion notification failed: %s", e)

            logger.info(
                "Scanner: PROMOTED %s [%s] %s @ %.2f — C/D met displacement criteria",
                setup["id"], tf, direction.upper(), entry_price)

        return {
            "checked": len(monitoring),
            "promoted": promoted,
            "expired": expired,
        }

    @staticmethod
    def _check_liquidity_sweep(candles_chrono: list, direction: str,
                                lookback: int = 5, check_window: int = 12) -> bool:
        """Check if recent candles show a liquidity sweep.

        For longs: price wicked below a prior swing low, then closed above it (SSL sweep).
        For shorts: price wicked above a prior swing high, then closed below it (BSL sweep).

        Args:
            candles_chrono: Candles in chronological order (oldest first).
            direction: 'long' or 'short'.
            lookback: Number of candles on each side to define a swing point.
            check_window: Number of recent candles to check for sweep.
        """
        if len(candles_chrono) < lookback * 2 + check_window:
            return False

        # Find swing points in the candles before the check window
        history = candles_chrono[:-check_window]
        recent = candles_chrono[-check_window:]

        if direction == "long":
            # Find swing lows in history
            swing_lows = []
            for i in range(lookback, len(history) - lookback):
                is_low = all(
                    history[i]["low"] <= history[i + j]["low"]
                    for j in range(-lookback, lookback + 1)
                    if j != 0 and 0 <= i + j < len(history)
                )
                if is_low:
                    swing_lows.append(history[i]["low"])

            if not swing_lows:
                return False

            # Check if any recent candle swept below the most recent swing low
            target = swing_lows[-1]  # Most recent swing low
            for c in recent:
                if c["low"] < target and c["close"] > target:
                    return True

        elif direction == "short":
            # Find swing highs in history
            swing_highs = []
            for i in range(lookback, len(history) - lookback):
                is_high = all(
                    history[i]["high"] >= history[i + j]["high"]
                    for j in range(-lookback, lookback + 1)
                    if j != 0 and 0 <= i + j < len(history)
                )
                if is_high:
                    swing_highs.append(history[i]["high"])

            if not swing_highs:
                return False

            target = swing_highs[-1]
            for c in recent:
                if c["high"] > target and c["close"] < target:
                    return True

        return False

    def _check_setup_against_history(self, setup: dict, candles_chrono: list,
                                      created_at: str) -> dict | None:
        """Walk candles chronologically. SL hit on any candle before TP = loss.

        Returns dict with gross_rr, cost_rr, and net rr (after spread).
        """
        is_long = setup["direction"] == "long"
        entry = setup.get("entry_price")
        sl = setup.get("calibrated_sl") or setup.get("sl_price")
        tps = [setup.get("tp1"), setup.get("tp2"), setup.get("tp3")]
        tps = [tp for tp in tps if tp is not None]
        rr_ratios = setup.get("rr_ratios") or []
        if isinstance(rr_ratios, str):
            try:
                rr_ratios = json.loads(rr_ratios)
            except (json.JSONDecodeError, TypeError):
                rr_ratios = []

        # Calculate spread cost in R-terms:
        # cost_rr = round-trip spread / SL distance
        sl_distance = abs(entry - sl) if entry and sl else 1
        cost_rr = round(SPREAD_ROUND_TRIP / sl_distance, 4) if sl_distance > 0 else 0.03

        # Round created_at UP to the next 5-minute candle boundary so we only
        # check candles that opened AFTER the setup was created.  This prevents
        # look-ahead bias where the candle containing the setup's creation time
        # already includes the price move that hits TP.
        created_norm = ""
        if created_at:
            try:
                raw = created_at.replace("Z", "+00:00")
                created_dt = datetime.fromisoformat(raw)
                # Ceil to next 5-min boundary
                minutes = created_dt.minute
                next_boundary = minutes + (5 - minutes % 5)
                created_dt = created_dt.replace(minute=0, second=0, microsecond=0) + timedelta(minutes=next_boundary)
                created_norm = created_dt.strftime("%Y-%m-%d %H:%M")
            except (ValueError, TypeError):
                created_norm = created_at[:16].replace("T", " ")

        first_eligible = True
        for candle in candles_chrono:
            # Skip candles that opened before the next boundary after setup creation
            candle_time = candle.get("datetime", "")[:16]
            if candle_time and created_norm and candle_time < created_norm:
                continue

            high = float(candle["high"])
            low = float(candle["low"])

            # On each candle, check SL first — if both SL and TP hit on same candle,
            # SL is assumed hit first (conservative / realistic).
            if sl is not None:
                if (is_long and low <= sl) or (not is_long and high >= sl):
                    gross_rr = -1
                    return {
                        "outcome": "stopped_out", "price": sl,
                        "gross_rr": gross_rr, "cost_rr": cost_rr,
                        "rr": round(gross_rr - cost_rr, 4),
                        "first_candle_resolve": first_eligible,
                    }

            # Check TPs from highest (TP3) to lowest (TP1)
            for i in range(len(tps) - 1, -1, -1):
                tp = tps[i]
                if (is_long and high >= tp) or (not is_long and low <= tp):
                    gross_rr = rr_ratios[i] if i < len(rr_ratios) else 0
                    return {
                        "outcome": f"tp{i + 1}", "price": tp,
                        "gross_rr": gross_rr, "cost_rr": cost_rr,
                        "rr": round(gross_rr - cost_rr, 4),
                        "first_candle_resolve": first_eligible,
                    }

            first_eligible = False

        return None

    def _get_htf_candles(self, htf: str, count: int) -> list | None:
        """Get higher-timeframe candles, using cache if fresh enough."""
        if not htf:
            return None

        cached = self._htf_cache.get(htf)
        if cached:
            age = (datetime.utcnow() - datetime.fromisoformat(cached["fetched_at"])).total_seconds()
            # Cache HTF candles for longer periods (1H=30min, 4H=2hr, 1D=6hr, 1W=24hr)
            max_age = {"1h": 1800, "4h": 7200, "1day": 21600, "1week": 86400}.get(htf, 3600)
            if age < max_age:
                return cached["candles"]

        candles = self._fetch_candles(htf, count)
        if candles:
            self._htf_cache[htf] = {
                "candles": candles,
                "fetched_at": datetime.utcnow().isoformat(),
            }
        return candles

    # _check_setup_resolution removed — replaced by _check_setup_against_history
    # which walks all candles chronologically to ensure SL-before-TP is caught.

    def _get_candles(self, symbol: str, interval: str, count: int) -> list | None:
        """Unified candle accessor — returns cached candles or fetches fresh.

        ALL components should use this instead of _fetch_candles directly.
        TTL scales with interval: 5min=4min, 1h=59min, 4h=4hr, 1day=12hr.
        """
        import hashlib as _hl
        cache_key = f"{symbol}|{interval}"
        cached = self._candle_store.get(cache_key)
        now = datetime.utcnow()
        ttl = self._CANDLE_TTL.get(interval, 300)

        if cached and (now - cached["fetched_at"]).total_seconds() < ttl:
            return cached["candles"]

        # All instruments fetched via OANDA (free, no daily limits)
        candles = self._fetch_candles_oanda(symbol, interval, count)
        if candles:
            self._candle_store[cache_key] = {
                "candles": candles,
                "fetched_at": now,
                "hash": _hl.md5(str(candles[-3:]).encode()).hexdigest(),
            }
        return candles

    def _fetch_candles_oanda(self, symbol: str, interval: str, count: int) -> list | None:
        """Fetch any instrument via OANDA — free, no daily limits.

        Handles XAU/USD, DXY (via EUR/USD inverted proxy), and US10Y.
        """
        try:
            from ml.data_providers import OandaProvider
            cfg = get_config()
            provider = OandaProvider(
                account_id=cfg.get("oanda_account_id", ""),
                access_token=cfg.get("oanda_access_token", ""),
            )
            # Calculate start_date from count + interval
            # CRITICAL: use full datetime format, NOT date-only strings.
            # Date-only "2026-03-27" gets parsed as midnight UTC, causing all
            # candles after midnight to be dropped — the same bug that was
            # fixed in server.py /candles endpoint.
            interval_hours = {"5min": 0.083, "15min": 0.25, "30min": 0.5,
                              "1h": 1, "4h": 4, "1day": 24}
            hours_back = count * interval_hours.get(interval, 1) * 1.5  # 1.5x buffer
            start = (datetime.utcnow() - timedelta(hours=hours_back)).strftime("%Y-%m-%dT%H:%M:%S")
            end = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")

            candles = provider.fetch_candles(symbol, interval, start, end)

            # DXY uses EUR/USD as proxy — invert the price direction for correlation
            # (EUR/USD goes up when dollar weakens, DXY goes down)
            if symbol == "DXY" and candles:
                for c in candles:
                    o, h, l, cl = c["open"], c["high"], c["low"], c["close"]
                    # Invert: approximate DXY = 1/EUR_USD scaled to ~104 range
                    if o > 0 and cl > 0:
                        c["open"] = round(1.0 / o * 104, 4)
                        c["high"] = round(1.0 / l * 104, 4)   # inverted: low EUR = high DXY
                        c["low"] = round(1.0 / h * 104, 4)    # inverted: high EUR = low DXY
                        c["close"] = round(1.0 / cl * 104, 4)

            # Trim to requested count
            if candles and len(candles) > count:
                candles = candles[-count:]

            return candles if candles else None

        except Exception as e:
            logger.error("OANDA fetch failed [%s %s]: %s", symbol, interval, e)
            return None

    def _fetch_candles(self, interval: str, count: int) -> list | None:
        """Fetch XAU/USD candles — uses unified cache (OANDA)."""
        return self._get_candles("XAU/USD", interval, count)

    # ── Correlated instruments (DXY, US10Y) ──────────────────────
    CORRELATED_INSTRUMENTS = {
        "DXY": "DXY",       # US Dollar Index
        "US10Y": "US10Y",   # US 10-Year Treasury Yield
    }
    # Sub-1H timeframes use a 30-min cache to save API credits
    _CORR_CACHE_SECS = 1800  # 30 minutes

    def _fetch_correlated_candles(self, interval: str, count: int = 30
                                  ) -> dict[str, list[dict] | None]:
        """Fetch DXY and US10Y candles — uses unified candle cache.

        Returns: {"DXY": [...candles...] | None, "US10Y": [...] | None}
        """
        result = {}
        for label, symbol in self.CORRELATED_INSTRUMENTS.items():
            candles = self._get_candles(symbol, interval, count)
            result[label] = candles
        return result

    def _call_opus_narrative(self, htf_candles: list, daily_candles: list = None,
                              intermarket: dict | None = None,
                              weekly_candles: list | None = None) -> dict | None:
        """Call Opus to build the HTF narrative. Cached for 1 hour.

        Returns structured narrative dict or None on failure.
        Cost: ~$0.06/call (wider windows), cached so ~4-6 calls/day during active killzones.
        """
        # Check cache — reuse if killzone + candles unchanged and within TTL
        import hashlib
        cfg = get_config()
        cache = self._narrative_cache
        current_kz = get_current_killzone()
        candle_hash = hashlib.md5(str(htf_candles[-3:]).encode()).hexdigest() if htf_candles else ""
        ttl = cfg.get("narrative_cache_ttl_seconds", 3600)

        cache_valid = (
            cache.get("narrative") is not None
            and cache.get("killzone") == current_kz
            and cache.get("candle_hash_4h") == candle_hash
            and cache.get("timestamp") is not None
            and (datetime.utcnow() - cache["timestamp"]).total_seconds() < ttl
        )
        if cache_valid:
            age_min = (datetime.utcnow() - cache["timestamp"]).total_seconds() / 60
            logger.debug("Opus narrative cache hit (%.0f min old, same KZ + candles)", age_min)
            return cache["narrative"]

        # Budget check — skip Opus if over daily limit
        try:
            from ml.cost_tracker import get_cost_tracker
            tracker = get_cost_tracker()
            if not tracker.check_budget():
                logger.warning("Opus narrative skipped — daily budget exceeded")
                if cache.get("narrative"):
                    return cache["narrative"]  # return stale cache
                return None
        except Exception:
            pass

        from ml.prompts import build_opus_narrative_prompt, OPUS_NARRATIVE_SYSTEM

        # Load most recent session recap for context
        session_recap = None
        try:
            recap = self.db.get_latest_session_recap()
            if recap:
                session_recap = recap.get("recap_json")
        except Exception:
            pass

        # Load narrative feedback for prompt self-improvement
        narrative_feedback = None
        try:
            from ml.claude_bridge import ClaudeAnalysisBridge
            bridge = ClaudeAnalysisBridge()

            session = get_current_killzone()

            # Start with killzone-specific EMA weights (falls back to _global)
            ema_raw = bridge.get_narrative_weights(killzone=session)
            weights = dict(ema_raw)

            # Blend outcome-based (AG) weights with EMA weights.
            # EMA measures accuracy (how often correct), AG measures impact
            # (correlation with winning). Blend = 50/50 geometric mean so
            # neither dominates and both dimensions contribute.
            ag_kz_weights = None
            ag_path = os.path.join(
                self.cfg.get("model_dir", os.path.join(os.path.dirname(__file__), "models")),
                "narrative_weights_ag.json")
            if os.path.exists(ag_path):
                try:
                    from ml.training import get_active_model_type
                    from ml.config import get_config as _gc
                    if get_active_model_type(_gc()["model_dir"]) in ("binary", "multi3"):
                        with open(ag_path) as f:
                            ag = json.load(f)
                        if ag:
                            ag_kz_weights = ag.get(session, ag.get("_global", {}))
                            for field in weights:
                                ema_w = weights[field]
                                ag_w = ag_kz_weights.get(field, 0.5)
                                weights[field] = round((max(0.01, ema_w) * max(0.01, ag_w)) ** 0.5, 4)
                except Exception:
                    pass

            from ml.narrative_examples import NarrativeExampleStore
            store = NarrativeExampleStore()
            bias_hint = (session_recap or {}).get("dominant_direction")
            examples = store.get_examples(session, bias_hint=bias_hint)

            # Bandit arm selection
            arm_params = None
            try:
                from ml.narrative_bandit import NarrativeBandit
                bandit = NarrativeBandit()
                if bandit.is_active():
                    arm = bandit.select_arm()
                    arm_params = arm.get("params")
                    self._current_bandit_arm = arm.get("arm_id")
            except Exception:
                pass

            narrative_feedback = {
                "weights": weights,
                "ema_raw": ema_raw,
                "ag_weights": ag_kz_weights,
                "examples": examples,
            }
            if arm_params:
                narrative_feedback["arm_params"] = arm_params
        except Exception as e:
            logger.debug("Narrative feedback loading failed (non-fatal): %s", e)

        prompt = build_opus_narrative_prompt(
            htf_candles, daily_candles, intermarket, session_recap,
            narrative_feedback=narrative_feedback,
            weekly_candles=weekly_candles)

        for attempt in range(3):
            try:
                resp = httpx.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": self.claude_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": "claude-opus-4-6",
                        "max_tokens": 800,
                        "temperature": 0,
                        "system": OPUS_NARRATIVE_SYSTEM,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                    timeout=90,
                )

                if resp.status_code in (429, 529):
                    wait = (2 ** attempt) * 2
                    logger.warning("Opus narrative rate limited, waiting %ds", wait)
                    time.sleep(wait)
                    continue

                if resp.status_code != 200:
                    logger.error("Opus narrative API error %d: %s",
                                 resp.status_code, resp.text[:200])
                    if attempt < 2:
                        time.sleep(2 ** attempt * 2)
                        continue
                    return None

                data = resp.json()
                text = ""
                for block in data.get("content", []):
                    if block.get("type") == "text":
                        text = block["text"]
                        break

                if not text:
                    return None

                clean = text.replace("```json", "").replace("```", "").strip()
                json_start = clean.find("{")
                json_end = clean.rfind("}")
                if json_start >= 0 and json_end > json_start:
                    clean = clean[json_start:json_end + 1]

                narrative = _safe_load_claude_json(clean, "opus_narrative")

                # Validate required fields
                if not narrative.get("directional_bias"):
                    logger.warning("Opus narrative missing directional_bias")
                    return None

                # Cache with killzone + candle hash for smart invalidation
                self._narrative_cache = {
                    'narrative': narrative,
                    'timestamp': datetime.utcnow(),
                    'killzone': current_kz,
                    'candle_hash_4h': candle_hash,
                }

                # Log cost
                try:
                    from ml.cost_tracker import get_cost_tracker
                    usage = data.get("usage", {})
                    _cost = get_cost_tracker().log_call(
                        "claude-opus-4-6",
                        usage.get("input_tokens", 2000),
                        usage.get("output_tokens", 500),
                        "narrative")
                    self._pending_api_cost += _cost  # P8
                except Exception:
                    pass

                logger.info("Opus narrative: %s bias (%s phase, %s confidence) — %s",
                            narrative.get("directional_bias"),
                            narrative.get("power_of_3_phase"),
                            narrative.get("phase_confidence"),
                            (narrative.get("macro_narrative") or "")[:80])
                return narrative

            except json.JSONDecodeError as e:
                logger.error("Opus narrative JSON parse failed: %s", e)
                if attempt < 2:
                    time.sleep(2)
                    continue
                return None
            except httpx.TimeoutException:
                logger.error("Opus narrative timeout (attempt %d)", attempt + 1)
                if attempt < 2:
                    time.sleep(2 ** attempt * 2)
                    continue
                return None
            except Exception as e:
                logger.error("Opus narrative failed: %s", e)
                return None

        return None

    def _is_weekly_cache_stale(self) -> bool:
        """Return True if the weekly narrative cache needs regeneration."""
        if self._weekly_narrative_cache is None or self._weekly_narrative_fetched_at is None:
            return True
        now = datetime.utcnow()
        fetched = self._weekly_narrative_fetched_at
        if (now - fetched).total_seconds() > 7 * 24 * 3600:
            return True
        # Stale if fetched in a different ISO week (handles server restarts mid-week)
        if now.year != fetched.year or now.isocalendar()[1] != fetched.isocalendar()[1]:
            return True
        return False

    def _is_near_weekly_level(self, price: float, atr: float,
                               weekly_levels: list) -> tuple[bool, dict | None]:
        """Return (True, matched_level) if price is within 3×ATR of any weekly level."""
        if not weekly_levels or atr <= 0:
            return False, None
        for level in weekly_levels:
            level_price = level.get("price")
            if level_price is None:
                continue
            if abs(price - level_price) <= 3.0 * atr:
                return True, level
        return False, None

    def _call_opus_weekly_narrative(self) -> dict | None:
        """Call Opus to generate the weekly macro narrative. Caches result for 7 days.

        Returns structured weekly narrative dict or None on failure.
        Cost: ~$0.05/call, called at most once per week (Sunday close clears cache).
        """
        from ml.prompts import build_opus_weekly_narrative_prompt, OPUS_WEEKLY_SYSTEM

        weekly_candles = self._get_htf_candles("1week", 24)
        daily_candles = self._get_htf_candles("1day", 20)

        if not weekly_candles:
            logger.warning("Weekly narrative: no weekly candles available from OANDA")
            return None

        prompt = build_opus_weekly_narrative_prompt(weekly_candles, daily_candles)

        for attempt in range(3):
            try:
                resp = httpx.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": self.claude_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": "claude-opus-4-6",
                        "max_tokens": 600,
                        "temperature": 0,
                        "system": OPUS_WEEKLY_SYSTEM,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                    timeout=90,
                )

                if resp.status_code in (429, 529):
                    wait = (2 ** attempt) * 2
                    logger.warning("Opus weekly narrative rate limited, waiting %ds", wait)
                    time.sleep(wait)
                    continue

                if resp.status_code != 200:
                    logger.error("Opus weekly narrative API error %d: %s",
                                 resp.status_code, resp.text[:200])
                    if attempt < 2:
                        time.sleep(2 ** attempt * 2)
                        continue
                    return None

                data = resp.json()
                text = ""
                for block in data.get("content", []):
                    if block.get("type") == "text":
                        text = block["text"]
                        break

                if not text:
                    return None

                clean = text.replace("```json", "").replace("```", "").strip()
                json_start = clean.find("{")
                json_end = clean.rfind("}")
                if json_start >= 0 and json_end > json_start:
                    clean = clean[json_start:json_end + 1]

                narrative = _safe_load_claude_json(clean, "opus_weekly_narrative")

                if not narrative.get("directional_bias"):
                    logger.warning("Weekly narrative missing directional_bias — discarding")
                    return None

                self._weekly_narrative_cache = narrative
                self._weekly_narrative_fetched_at = datetime.utcnow()

                try:
                    from ml.cost_tracker import get_cost_tracker
                    usage = data.get("usage", {})
                    get_cost_tracker().log_call(
                        "claude-opus-4-6",
                        usage.get("input_tokens", 1500),
                        usage.get("output_tokens", 400),
                        "weekly_narrative")
                except Exception:
                    pass

                logger.info("Opus weekly narrative: %s bias, %s P3 phase",
                            narrative.get("directional_bias"),
                            narrative.get("p3_phase", "?"))
                return narrative

            except Exception as e:
                logger.warning("Opus weekly narrative attempt %d failed: %s", attempt + 1, e)
                if attempt < 2:
                    time.sleep(2 ** attempt * 2)

        return None

    def _get_weekly_narrative(self) -> dict | None:
        """Return weekly narrative from cache, regenerating if stale."""
        if self._is_weekly_cache_stale():
            try:
                self._call_opus_weekly_narrative()
            except Exception as e:
                logger.warning("Weekly narrative call failed: %s", e)
        return self._weekly_narrative_cache

    def _generate_session_recap(self, ending_killzone: str):
        """Generate an Opus session recap when a killzone ends.

        Queries all setups from the ending killzone and asks Opus to summarize.
        """
        from ml.prompts import build_session_recap_prompt, OPUS_NARRATIVE_SYSTEM_RECAP

        today = datetime.utcnow().strftime("%Y-%m-%d")

        # Get setups from this killzone today
        try:
            setups = self.db.get_setups_by_killzone(ending_killzone, today)
        except Exception:
            setups = []

        if not setups:
            logger.debug("No setups for %s recap on %s", ending_killzone, today)
            return

        prompt = build_session_recap_prompt(setups, ending_killzone, today)

        try:
            resp = httpx.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.claude_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-opus-4-6",
                    "max_tokens": 600,
                    "temperature": 0,
                    "system": OPUS_NARRATIVE_SYSTEM_RECAP,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=60,
            )

            if resp.status_code != 200:
                logger.warning("Session recap API error %d", resp.status_code)
                return

            data = resp.json()
            text = ""
            for block in data.get("content", []):
                if block.get("type") == "text":
                    text = block["text"]
                    break

            if not text:
                return

            clean = text.replace("```json", "").replace("```", "").strip()
            json_start = clean.find("{")
            json_end = clean.rfind("}")
            if json_start >= 0 and json_end > json_start:
                clean = clean[json_start:json_end + 1]

            recap_json = _safe_load_claude_json(clean, "opus_recap")
            self.db.store_session_recap(ending_killzone, today, recap_json)
            logger.info("Session recap stored: %s on %s — %s",
                        ending_killzone, today,
                        (recap_json.get("narrative_summary") or "")[:60])

        except Exception as e:
            logger.warning("Session recap generation failed: %s", e)

    def _update_prospect_tracker(self, event: str, killzone: str = "",
                                  zone_type: str = "", is_win: bool = False):
        """Update ml/models/prospect_tracker.json with prospect lifecycle events."""
        tracker_path = os.path.join(
            self.cfg.get("model_dir", os.path.join(os.path.dirname(__file__), "models")),
            "prospect_tracker.json")
        try:
            if os.path.exists(tracker_path):
                with open(tracker_path) as f:
                    tracker = json.load(f)
            else:
                tracker = {"total_prospects": 0, "zones_reached": 0, "triggered": 0,
                           "triggered_wins": 0, "triggered_losses": 0,
                           "by_killzone": {}, "by_zone_type": {}}

            if event == "generated":
                tracker["total_prospects"] += 1
                kz_data = tracker["by_killzone"].setdefault(killzone, {"generated": 0, "triggered": 0})
                kz_data["generated"] = kz_data.get("generated", 0) + 1
            elif event == "displaced":
                tracker["zones_reached"] += 1
            elif event == "triggered":
                tracker["triggered"] += 1
                kz_data = tracker["by_killzone"].setdefault(killzone, {"generated": 0, "triggered": 0})
                kz_data["triggered"] = kz_data.get("triggered", 0) + 1
            elif event == "resolved":
                if is_win:
                    tracker["triggered_wins"] += 1
                else:
                    tracker["triggered_losses"] += 1

            with open(tracker_path, "w") as f:
                json.dump(tracker, f, indent=2)
            logger.info("[TRACKER] Updated prospect_tracker: %s (kz=%s)", event, killzone)
        except Exception as e:
            logger.warning("Prospect tracker update failed: %s", e)

    def _generate_prospect_json(self, upcoming_kz: str) -> dict | None:
        """Call Opus to generate zones and conditional setups for a killzone.

        Returns the parsed prospect dict (with per-setup status injected),
        or None on failure. Does NOT store in DB — caller decides what to do.
        Cost: ~$0.04/call.
        """
        from ml.prompts import build_prospect_prompt, OPUS_PROSPECT_SYSTEM

        # Fetch candles for prospecting
        htf_candles = self._get_htf_candles("4h", 20)
        candles_1h = self._get_htf_candles("1h", 50)
        if not htf_candles or not candles_1h:
            logger.warning("Prospect: insufficient candle data for %s", upcoming_kz)
            return None

        # Current price — so Opus draws zones relative to where price actually is
        current_price = None
        if candles_1h:
            current_price = float(candles_1h[-1].get("close", 0)) or None

        # Get intermarket + session recap
        intermarket_ctx = None
        try:
            from ml.intermarket import compute_intermarket_context
            corr = self._fetch_correlated_candles("1h", count=30)
            intermarket_ctx = compute_intermarket_context(
                gold_candles=candles_1h, dxy_candles=corr.get("DXY"),
                us10y_candles=corr.get("US10Y"), session=upcoming_kz)
        except Exception:
            pass

        session_recap = None
        try:
            recap = self.db.get_latest_session_recap()
            if recap:
                session_recap = recap.get("recap_json")
        except Exception:
            pass

        # Prior session stats — trend context so Opus knows if structure is extended
        prior_session_stats = None
        try:
            prior_session_stats = self._get_prior_session_stats(upcoming_kz)
        except Exception:
            pass

        prompt = build_prospect_prompt(
            htf_candles, candles_1h, intermarket_ctx, session_recap, upcoming_kz,
            current_price=current_price,
            prior_session_stats=prior_session_stats)

        for attempt in range(3):
            try:
                resp = httpx.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": self.claude_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": "claude-opus-4-6",
                        "max_tokens": 1200,
                        "temperature": 0,
                        "system": OPUS_PROSPECT_SYSTEM,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                    timeout=90,
                )

                if resp.status_code in (429, 529):
                    time.sleep((2 ** attempt) * 2)
                    continue
                if resp.status_code != 200:
                    logger.error("Prospect API error %d", resp.status_code)
                    if attempt < 2:
                        time.sleep(2 ** attempt * 2)
                        continue
                    return None

                data = resp.json()
                text = ""
                for block in data.get("content", []):
                    if block.get("type") == "text":
                        text = block["text"]
                        break

                if not text:
                    return None

                clean = text.replace("```json", "").replace("```", "").strip()
                json_start = clean.find("{")
                json_end = clean.rfind("}")
                if json_start >= 0 and json_end > json_start:
                    clean = clean[json_start:json_end + 1]

                prospect = _safe_load_claude_json(clean, "opus_prospect")
                setups = prospect.get("conditional_setups", [])

                if not setups:
                    logger.info("Prospect [%s]: no conditional setups from Opus", upcoming_kz)
                    return None

                # Inject per-setup tracking status
                for s in setups:
                    s.setdefault("status", "active")

                return prospect

            except json.JSONDecodeError:
                if attempt < 2:
                    time.sleep(2)
                    continue
                return None
            except Exception as e:
                logger.error("Prospect call failed: %s", e)
                return None

        return None

    def _prospect_killzone_zones(self, upcoming_kz: str):
        """Generate, store, and notify a new prospect for an upcoming killzone.

        Calls Opus via _generate_prospect_json(), stores result in DB,
        updates tracker, and sends pre-alert notification.
        """
        from ml.notifications import notify_zone_prospect

        prospect = self._generate_prospect_json(upcoming_kz)
        if not prospect:
            return

        setups = prospect.get("conditional_setups", [])

        # Store in DB
        prospect_id = self.db.store_prospect(upcoming_kz, prospect)
        logger.info("Prospect [%s]: %d conditional setups stored (ID: %s)",
                    upcoming_kz, len(setups), prospect_id)

        # Track
        self._update_prospect_tracker("generated", killzone=upcoming_kz)

        # Send pre-alert notification
        notify_zone_prospect({"killzone": upcoming_kz, "setups": setups})

    def _get_prior_session_stats(self, upcoming_kz: str) -> dict | None:
        """Get win/loss stats from the prior killzone for trend context.

        Returns dict with killzone, wins, losses, dominant_direction, price_move
        or None if insufficient data.
        """
        from ml.prompts import KILLZONE_HOURS
        # Map upcoming killzone to prior one
        kz_order = ["Asian", "London", "NY_AM", "NY_PM"]
        try:
            idx = kz_order.index(upcoming_kz)
            prior_kz = kz_order[idx - 1] if idx > 0 else kz_order[-1]
        except ValueError:
            return None

        # Get today's resolved setups from the prior killzone
        today = datetime.utcnow().strftime("%Y-%m-%d")
        history = self.db.get_history()
        prior_setups = [
            s for s in history
            if s.get("killzone") == prior_kz
            and (s.get("created_at") or "").startswith(today)
            and s.get("outcome")
        ]
        if len(prior_setups) < 2:
            return None

        wins = sum(1 for s in prior_setups if (s.get("outcome") or "").startswith("tp"))
        losses = len(prior_setups) - wins
        directions = [s.get("direction", "") for s in prior_setups]
        long_count = directions.count("long")
        short_count = directions.count("short")
        dominant = "long" if long_count >= short_count else "short"

        # Approximate price move from first to last setup entry prices
        prices = [s.get("entry_price", 0) for s in prior_setups if s.get("entry_price")]
        price_move = prices[-1] - prices[0] if len(prices) >= 2 else 0

        return {
            "killzone": prior_kz,
            "wins": wins,
            "losses": losses,
            "dominant_direction": dominant,
            "price_move": price_move,
        }

    def _get_trigger_poll_interval(self, current_price: float,
                                     prospects: list) -> float:
        """Calculate adaptive polling interval based on price distance to zones."""
        if not current_price or not prospects:
            return 90  # default

        nearest = float("inf")
        for p in prospects:
            pj = p.get("prospect_json", {}) if p.get("status") == "active" else {}
            for setup in pj.get("conditional_setups", []):
                sweep = setup.get("sweep_level") or setup.get("entry_zone", {}).get("low", 0)
                if sweep:
                    nearest = min(nearest, abs(current_price - sweep))

            # For displaced prospects, distance to OB zone matters
            if p.get("status") == "displaced":
                tr = p.get("trigger_result", {})
                if isinstance(tr, dict):
                    ob = tr.get("displacement", {}).get("ob_zone", {})
                    if ob:
                        mid = (ob.get("high", 0) + ob.get("low", 0)) / 2
                        if mid:
                            nearest = min(nearest, abs(current_price - mid))

        if nearest < 3:
            return 60
        elif nearest < 10:
            return 180
        elif nearest < 30:
            return 300
        else:
            return 600

    def monitor_prospect_triggers(self, candles_5m: list | None = None) -> dict:
        """Three-phase prospect monitoring: active → displaced → retrace → entry.

        Phase 1 (active): Price comparison against zones — no Claude calls.
                          When price reaches zone → check displacement via Sonnet.
        Phase 2 (displaced): Monitor retrace into OB/FVG zone — no Claude calls.
                             When retrace reaches zone → confirm entry via Sonnet.
        Phase 3 (entry): Sonnet confirms, calibration runs, trade alert fires.

        Args:
            candles_5m: Optional pre-fetched 5-min candles (chronological order).
        """
        from ml.notifications import notify_displacement_confirmed

        # Get both active AND displaced prospects
        all_prospects = self.db.get_active_prospects(include_displaced=True)
        if not all_prospects:
            return {"checked": 0, "triggered": 0, "displaced": 0, "has_active": False}

        # Adaptive polling — skip if not due yet based on price distance
        now = datetime.utcnow()
        elapsed = (now - self._last_trigger_check).total_seconds()
        # Quick pre-check: get last known price from candle store
        last_price = 0
        gold_5m = self._candle_store.get("XAU/USD|5min")
        if gold_5m and gold_5m.get("candles"):
            last_price = gold_5m["candles"][-1].get("close", 0)
        poll_interval = self._get_trigger_poll_interval(last_price, all_prospects)
        if elapsed < poll_interval:
            return {"checked": 0, "triggered": 0, "displaced": 0,
                    "has_active": True, "skipped": True,
                    "next_check_in": int(poll_interval - elapsed)}

        self._last_trigger_check = now

        cfg = get_config()
        if candles_5m is not None:
            candles = candles_5m[-10:] if len(candles_5m) >= 10 else candles_5m
        else:
            tf = cfg.get("trigger_candle_timeframe", "5min")
            candles = self._fetch_candles(tf, 10)
        if not candles:
            return {"checked": 0, "triggered": 0, "displaced": 0, "has_active": True}

        current_price = float(candles[-1].get("close", 0))
        if not current_price:
            return {"checked": 0, "triggered": 0, "displaced": 0, "has_active": True}

        tolerance = cfg.get("trigger_price_tolerance_pips", 3.0)
        retrace_timeout = cfg.get("retrace_timeout_candles", 15)
        triggered = 0
        newly_displaced = 0

        active_prospects = [p for p in all_prospects if p["status"] == "active"]
        displaced_prospects = [p for p in all_prospects if p["status"] == "displaced"]

        # ── PHASE 1: Per-setup invalidation + displacement check ──────────
        invalidated_killzones = set()   # Killzones where ALL setups in a prospect are gone
        prospects_needing_refill = []   # Prospects with some (not all) setups invalidated

        for prospect in active_prospects:
            pj = prospect.get("prospect_json", {})
            setups = pj.get("conditional_setups", [])
            newly_invalidated = 0
            displacement_found = False

            for setup in setups:
                # Skip setups already marked invalidated on a previous cycle
                if setup.get("status") == "invalidated":
                    continue

                entry_zone = setup.get("entry_zone", {})
                zone_high = entry_zone.get("high", 0)
                zone_low = entry_zone.get("low", 0)
                invalidation = setup.get("invalidation", 0)

                # Check per-setup invalidation
                if invalidation:
                    if (setup.get("bias") == "bullish" and current_price < invalidation) or \
                       (setup.get("bias") == "bearish" and current_price > invalidation):
                        setup["status"] = "invalidated"
                        setup["invalidated_at"] = datetime.utcnow().isoformat()
                        newly_invalidated += 1
                        logger.info("Prospect %s setup %s invalidated (price=%.2f, inv=%.2f)",
                                    prospect["id"], setup.get("id", "?"),
                                    current_price, invalidation)
                        continue  # Setup gone — skip displacement check for it

                # Check if price is near entry zone → check for displacement
                if zone_low and zone_high:
                    near_zone = (zone_low - tolerance * 3) <= current_price <= (zone_high + tolerance * 3)
                    logger.info(
                        "[PROSPECT] Phase 1 check: prospect=%s setup=%s "
                        "price=%.2f zone=[%.2f-%.2f] near_zone=%s",
                        prospect["id"], setup.get("id", "?"),
                        current_price, zone_low, zone_high, near_zone)
                    if near_zone:
                        displacement = self._check_displacement(setup, candles, current_price)
                        disp_confirmed = bool(displacement and displacement.get("displacement_confirmed"))
                        logger.info(
                            "[PROSPECT] Phase 1 displacement check: prospect=%s "
                            "displacement_confirmed=%s ob_zone=%s",
                            prospect["id"], disp_confirmed,
                            (displacement or {}).get("ob_zone"))
                        if displacement and displacement.get("displacement_confirmed"):
                            displacement_data = {
                                "setup": setup,
                                "displacement": displacement,
                                "displaced_at": datetime.utcnow().isoformat(),
                                "candles_waited": 0,
                            }
                            self.db.resolve_prospect(
                                prospect["id"], "displaced",
                                json.dumps(displacement_data))
                            newly_displaced += 1
                            self._update_prospect_tracker("displaced",
                                                          killzone=prospect.get("killzone", ""))
                            ob = displacement.get("ob_zone", {})
                            notify_displacement_confirmed(setup, displacement)
                            logger.info("Prospect %s DISPLACED: OB at %.2f-%.2f",
                                        prospect["id"], ob.get("low", 0), ob.get("high", 0))
                            displacement_found = True
                            break  # Prospect now displaced — stop checking setups

            if displacement_found:
                continue  # Prospect handled by displacement flow

            # Tally remaining active setups
            active_setups = [s for s in setups if s.get("status", "active") == "active"]

            if not active_setups:
                # ALL setups invalidated → expire the entire prospect
                self.db.resolve_prospect(prospect["id"], "expired", "all_setups_invalidated")
                invalidated_killzones.add(prospect.get("killzone", ""))
                logger.info("Prospect %s fully invalidated (all %d setups gone)",
                            prospect["id"], len(setups))
            elif newly_invalidated > 0:
                # Some invalidated, some still active → persist updated JSON + queue for refill
                pj["conditional_setups"] = setups
                self.db.update_prospect_json(prospect["id"], pj)
                num_dead = len([s for s in setups if s.get("status") == "invalidated"])
                prospects_needing_refill.append({
                    "prospect_id": prospect["id"],
                    "killzone": prospect.get("killzone", ""),
                    "slots_invalidated": num_dead,
                    "slots_active": len(active_setups),
                })
                logger.info("Prospect %s: %d/%d setups invalidated, %d active — queued for refill",
                            prospect["id"], num_dead, len(setups), len(active_setups))

        # ── Shared regen budget helpers ──────────
        today = datetime.utcnow().strftime("%Y-%m-%d")
        if self._prospect_regen_date != today:
            self._prospect_regen_count = {}
            self._prospect_regen_date = today

        regen_cfg = cfg.get("prospect_max_regen", 4)
        if isinstance(regen_cfg, dict):
            _kz_max = regen_cfg
        else:
            _kz_max = {"Asian": regen_cfg, "London": regen_cfg,
                       "NY_AM": regen_cfg, "NY_PM": regen_cfg}

        _kz_duration_hours = {
            "Asian": 7, "London": 5, "NY_AM": 4, "NY_PM": 4,
        }

        def _regen_allowed(kz: str) -> bool:
            """Check regen cap + cooldown for a killzone. Returns True if allowed."""
            max_r = _kz_max.get(kz, 4)
            count = self._prospect_regen_count.get(kz, 0)
            if count >= max_r:
                return False
            kz_h = _kz_duration_hours.get(kz, 5)
            cd_min = (kz_h * 60) / max_r if max_r > 1 else 0
            if count > 0 and cd_min > 0:
                last = self._prospect_regen_last.get(kz)
                if last and (now - last).total_seconds() / 60 < cd_min:
                    return False
            return True

        def _record_regen(kz: str):
            """Bump regen counter + cooldown timestamp."""
            self._prospect_regen_count[kz] = self._prospect_regen_count.get(kz, 0) + 1
            self._prospect_regen_last[kz] = now

        # ── Auto-regenerate for killzones that lost ALL coverage ──
        if invalidated_killzones:
            remaining = self.db.get_active_prospects(include_displaced=True)
            remaining_kzs = {p.get("killzone") for p in remaining}

            for kz in invalidated_killzones:
                if kz in remaining_kzs:
                    continue  # Other prospects still cover this killzone
                if not _regen_allowed(kz):
                    rc = self._prospect_regen_count.get(kz, 0)
                    mx = _kz_max.get(kz, 4)
                    logger.info("Prospect [%s]: fully invalidated but regen blocked (%d/%d)",
                                kz, rc, mx)
                    continue

                kz_h = _kz_duration_hours.get(kz, 5)
                mx = _kz_max.get(kz, 4)
                cd = (kz_h * 60) / mx if mx > 1 else 0
                rc = self._prospect_regen_count.get(kz, 0)
                logger.info("Prospect [%s]: all invalidated — full regen (%d/%d, cd %.0fmin)",
                            kz, rc + 1, mx, cd)
                self._prospect_killzone_zones(kz)
                _record_regen(kz)

        # ── Refill partially invalidated prospects with highest-confidence setups ──
        if prospects_needing_refill:
            from ml.notifications import notify_zone_prospect

            for info in prospects_needing_refill:
                kz = info["killzone"]
                pid = info["prospect_id"]

                if not _regen_allowed(kz):
                    logger.info("Prospect %s: refill skipped — regen cap/cooldown for %s", pid, kz)
                    continue

                # Call Opus for fresh setups
                new_prospect = self._generate_prospect_json(kz)
                if not new_prospect:
                    logger.warning("Prospect %s: refill Opus call returned nothing", pid)
                    continue

                new_setups = new_prospect.get("conditional_setups", [])
                if not new_setups:
                    continue

                # Sort new setups by confidence: high > medium > low
                _conf_order = {"high": 0, "medium": 1, "low": 2}
                new_setups.sort(key=lambda s: _conf_order.get(s.get("confidence", "low"), 3))

                # Re-read prospect from DB (may have changed since phase-1 loop)
                fresh_prospects = self.db.get_active_prospects()
                target = next((p for p in fresh_prospects if p["id"] == pid), None)
                if not target:
                    continue  # Prospect was displaced/expired between checks

                pj = target.get("prospect_json", {})
                setups = pj.get("conditional_setups", [])

                # Replace invalidated slots with best new setups
                filled = 0
                for i, s in enumerate(setups):
                    if s.get("status") == "invalidated" and filled < len(new_setups):
                        replacement = new_setups[filled]
                        replacement["status"] = "active"
                        replacement["refilled_at"] = datetime.utcnow().isoformat()
                        setups[i] = replacement
                        filled += 1

                if filled > 0:
                    pj["conditional_setups"] = setups
                    self.db.update_prospect_json(pid, pj)
                    _record_regen(kz)

                    active_now = [s for s in setups if s.get("status", "active") == "active"]
                    logger.info("Prospect %s: refilled %d invalidated slots (%d active now, regen %d/%d for %s)",
                                pid, filled, len(active_now),
                                self._prospect_regen_count.get(kz, 0), _kz_max.get(kz, 4), kz)

                    # Notify about refreshed prospect
                    notify_zone_prospect({"killzone": kz, "setups": active_now, "refill": True})

        # ── PHASE 2: Check displaced prospects for retracement ────
        for prospect in displaced_prospects:
            disp_data = prospect.get("trigger_result", {})
            if not isinstance(disp_data, dict):
                continue

            setup = disp_data.get("setup", {})
            displacement = disp_data.get("displacement", {})
            ob = displacement.get("ob_zone", {})
            fvg = displacement.get("fvg_zone", {})
            ob_high = ob.get("high", 0)
            ob_low = ob.get("low", 0)
            bias = setup.get("bias", "")

            if not ob_high or not ob_low:
                continue

            # Track candle count for timeout
            candles_waited = disp_data.get("candles_waited", 0) + 1
            disp_data["candles_waited"] = candles_waited

            # Timeout check
            if candles_waited > retrace_timeout:
                self.db.resolve_prospect(prospect["id"], "expired", "retrace_timeout")
                logger.info("Prospect %s expired — no retrace after %d candles",
                            prospect["id"], candles_waited)
                continue

            # Invalidation: price goes through OB and takes out sweep level
            sweep_level = displacement.get("sweep_level", 0)
            if sweep_level:
                if (bias == "bullish" and current_price < sweep_level) or \
                   (bias == "bearish" and current_price > sweep_level):
                    self.db.resolve_prospect(prospect["id"], "expired", "retrace_invalidation")
                    logger.info("Prospect %s invalidated — price breached sweep level", prospect["id"])
                    continue

            # Check if price has retraced INTO the OB zone
            in_ob = False
            if bias == "bullish":
                # For longs: candle low touches or enters OB zone
                candle_low = float(candles[-1].get("low", 0))
                in_ob = candle_low <= ob_high and current_price >= ob_low
                logger.info(
                    "[PROSPECT] Phase 2 retrace check (bullish): prospect=%s "
                    "candle_low=%.2f ob_high=%.2f current=%.2f ob_low=%.2f in_ob=%s "
                    "candles_waited=%d/%d",
                    prospect["id"], candle_low, ob_high, current_price, ob_low,
                    in_ob, candles_waited, retrace_timeout)
            elif bias == "bearish":
                # For shorts: candle high touches or enters OB zone
                candle_high = float(candles[-1].get("high", 0))
                in_ob = candle_high >= ob_low and current_price <= ob_high
                logger.info(
                    "[PROSPECT] Phase 2 retrace check (bearish): prospect=%s "
                    "candle_high=%.2f ob_low=%.2f current=%.2f ob_high=%.2f in_ob=%s "
                    "candles_waited=%d/%d",
                    prospect["id"], candle_high, ob_low, current_price, ob_high,
                    in_ob, candles_waited, retrace_timeout)

            if in_ob:
                # Check zone cooldown — skip if this zone stopped out earlier today
                try:
                    _zone_key = self.db._make_zone_key(
                        "long" if bias == "bullish" else "short", "1h", ob_high, ob_low)
                    if self.db.is_zone_blocked(_zone_key, get_current_killzone()):
                        logger.info(
                            "Prospect %s BLOCKED — zone %.2f-%.2f already stopped "
                            "out this killzone", prospect["id"], ob_low, ob_high)
                        in_ob = False
                except Exception as _e:
                    logger.debug("Zone cooldown check failed (non-blocking): %s", _e)

            if in_ob:
                # RETRACE DETECTED — run entry confirmation
                logger.info("Prospect %s RETRACE into OB (%.2f-%.2f) at price %.2f",
                            prospect["id"], ob_low, ob_high, current_price)

                _use_scan_once = get_config().get("prospect_use_scan_once", False)
                if _use_scan_once:
                    _prospect_tf = setup.get("timeframe", "1h")
                    _disp_ctx = {
                        "zone_high": ob_high,
                        "zone_low": ob_low,
                        "zone_type": "ob",
                        "direction": "long" if bias == "bullish" else "short",
                        "displacement_confirmed": True,
                        "prospect_id": prospect["id"],
                    }
                    # Flag displacement zone in narrative so next regular cycle
                    # knows it was consumed (Phase C)
                    try:
                        from ml.narrative_state import NarrativeStore
                        NarrativeStore(self.db.db_path).update_displacement_zones(
                            _prospect_tf, _disp_ctx)
                    except Exception as _ne:
                        logger.debug("Displacement zone narrative update failed: %s", _ne)

                    try:
                        logger.info("Prospect %s triggering out-of-cycle scan_once(%s)",
                                    prospect["id"], _prospect_tf)
                        _result = self.scan_once(_prospect_tf,
                                                 displacement_context=_disp_ctx)
                        if _result.get("status") == "setup_found":
                            triggered += 1
                            self.db.mark_prospect_triggered(
                                prospect["id"], _result.get("setup_id", ""))
                            self.db.resolve_prospect(
                                prospect["id"], "triggered",
                                json.dumps({"scan_result": _result}))
                            self._update_prospect_tracker(
                                "triggered", killzone=prospect.get("killzone", ""))
                        else:
                            logger.info("Prospect %s scan_once returned %s — no trigger",
                                        prospect["id"], _result.get("status"))
                    except Exception as _e:
                        logger.error("Prospect %s scan_once failed: %s",
                                     prospect["id"], _e)
                else:
                    # Legacy path — kept until prospect_use_scan_once=True validated
                    entry_result = self._confirm_retrace_entry(
                        setup, displacement, candles, current_price, cfg,
                        prospect_id=prospect["id"])

                    if entry_result:
                        triggered += 1
                        # mark_prospect_triggered already called inside _confirm_retrace_entry
                        # when prospect_id is provided; call resolve_prospect only for the
                        # trigger_result payload (preserve backward-compat JSON record).
                        self.db.resolve_prospect(
                            prospect["id"], "triggered",
                            json.dumps({"entry": entry_result,
                                        "setup_id": entry_result.get("setup_id")}))
                        self._update_prospect_tracker(
                            "triggered", killzone=prospect.get("killzone", ""))

            # Update candle count in DB
            self.db.resolve_prospect(
                prospect["id"], "displaced", json.dumps(disp_data))

        return {
            "checked": len(all_prospects),
            "triggered": triggered,
            "displaced": newly_displaced,
            "awaiting_retrace": len(displaced_prospects) - triggered,
            "has_active": len(all_prospects) > 0,
            "current_price": current_price,
        }

    def _check_displacement(self, prospect_setup: dict, candles: list,
                              current_price: float) -> dict | None:
        """Check if displacement has occurred at a prospect zone. Sonnet call."""
        from ml.prompts import build_displacement_check_prompt

        prompt = build_displacement_check_prompt(prospect_setup, candles[-8:], current_price)
        return self._call_sonnet_short(prompt)

    def _refine_entry_5min(self, direction: str, zone_high: float,
                           zone_low: float, entry_price: float) -> dict:
        """Use recent 5min candles to find a tighter entry within the HTF zone.

        Returns: {"refined_entry": float, "refinement_type": str, "5min_structure": str}
        """
        candles_5m = self._get_candles("XAU/USD", "5min", 12)
        if not candles_5m or len(candles_5m) < 3:
            return {"refined_entry": entry_price, "refinement_type": "none",
                    "5min_structure": "insufficient data"}

        best_entry = None
        best_type = "none"
        best_structure = "none"

        for i in range(len(candles_5m) - 1):
            c = candles_5m[i]
            c_next = candles_5m[i + 1]
            c_open = float(c.get("open", 0))
            c_close = float(c.get("close", 0))
            c_high = float(c.get("high", 0))
            c_low = float(c.get("low", 0))
            n_open = float(c_next.get("open", 0))
            n_close = float(c_next.get("close", 0))

            body_mid = (c_open + c_close) / 2

            # Check if candle body is within the zone
            if not (min(c_open, c_close) <= zone_high and max(c_open, c_close) >= zone_low):
                continue

            if direction == "long":
                # Bullish OB: down candle engulfed by next up candle
                is_down = c_close < c_open
                is_up_next = n_close > n_open
                if is_down and is_up_next and n_close > c_open:
                    candidate = body_mid
                    if best_entry is None or candidate < best_entry:  # deeper = better for longs
                        best_entry = candidate
                        best_type = "5min_ob"
                        best_structure = f"bullish OB at {candidate:.2f}"
            else:
                # Bearish OB: up candle engulfed by next down candle
                is_up = c_close > c_open
                is_down_next = n_close < n_open
                if is_up and is_down_next and n_close < c_open:
                    candidate = body_mid
                    if best_entry is None or candidate > best_entry:  # higher = better for shorts
                        best_entry = candidate
                        best_type = "5min_ob"
                        best_structure = f"bearish OB at {candidate:.2f}"

            # FVG detection (need i+2)
            if i + 2 < len(candles_5m):
                c2 = candles_5m[i + 2]
                c2_low = float(c2.get("low", 0))
                c2_high = float(c2.get("high", 0))

                if direction == "long" and c_high < c2_low:  # bullish FVG
                    fvg_mid = (c_high + c2_low) / 2
                    if zone_low <= fvg_mid <= zone_high:
                        if best_entry is None or fvg_mid < best_entry:
                            best_entry = fvg_mid
                            best_type = "5min_fvg"
                            best_structure = f"bullish FVG {c_high:.2f}-{c2_low:.2f}"
                elif direction == "short" and c_low > c2_high:  # bearish FVG
                    fvg_mid = (c_low + c2_high) / 2
                    if zone_low <= fvg_mid <= zone_high:
                        if best_entry is None or fvg_mid > best_entry:
                            best_entry = fvg_mid
                            best_type = "5min_fvg"
                            best_structure = f"bearish FVG {c2_high:.2f}-{c_low:.2f}"

        return {
            "refined_entry": best_entry or entry_price,
            "refinement_type": best_type,
            "5min_structure": best_structure,
        }

    def _confirm_retrace_entry(self, setup: dict, displacement: dict,
                                 candles: list, current_price: float,
                                 cfg: dict,
                                 prospect_id: str = None) -> dict | None:
        """Confirm entry when price retraces into the OB zone.

        Optionally fetches LTF candles for tighter entry.
        This is Alert 3 — the actual trade signal.

        Args:
            prospect_id: ID of the killzone_prospects row that triggered this entry.
                         When provided, a scanner_setup is created for monitoring.
        """
        from ml.prompts import build_retrace_confirmation_prompt
        from ml.notifications import notify_entry_trigger, _calc_lot_size

        # Optional LTF refinement
        ltf_candles = None
        if cfg.get("ltf_refinement_enabled", True):
            ltf_tf = cfg.get("ltf_refinement_timeframe", "5min")
            ltf_candles = self._fetch_candles(ltf_tf, 10)

        prompt = build_retrace_confirmation_prompt(
            displacement, setup, candles[-5:], ltf_candles, current_price)

        result = self._call_sonnet_short(prompt)
        if not result or not result.get("confirmed"):
            logger.info("Retrace entry NOT confirmed: %s", (result or {}).get("reason", ""))
            return None

        entry_price = result.get("entry", current_price)
        sl_price = result.get("sl", setup.get("preliminary_sl"))
        tps = result.get("tps", setup.get("preliminary_tps", []))

        if not entry_price or not sl_price:
            return None

        # Build full analysis for the normal calibration pipeline
        direction = "long" if setup.get("bias") == "bullish" else "short"
        killzone = get_current_killzone()
        analysis = {
            "bias": setup.get("bias", "neutral"),
            "entry": {"price": entry_price, "direction": direction},
            "stopLoss": {"price": sl_price},
            "takeProfits": [{"price": tp} for tp in (tps or [])],
            "setup_quality": "A",
            "killzone": killzone,
            "confluences": ["pre-prospected", "sweep_confirmed", "displacement_confirmed",
                            "retrace_confirmed", "ob_holding"],
            "warnings": [],
            "prospect_triggered": True,
            "retrace_entry": True,
            "sl_type": result.get("sl_type", "below_sweep"),
            "ltf_signal": result.get("ltf_signal"),
        }
        if prospect_id:
            analysis["prospect_id"] = prospect_id

        # ── Resolve thesis_id for lifecycle tracking ──────────────────────
        _prospect_thesis_id = None
        try:
            from ml.narrative_state import NarrativeStore
            _ns_engine = NarrativeStore(self.db.db_path)
            _current_ns = _ns_engine.get_current("1h")
            if _current_ns:
                _prospect_thesis_id = (_current_ns.get("id")
                                       or str(_current_ns.get("thesis", ""))[:8] or None)
        except Exception as e:
            logger.debug("Prospect thesis_id lookup failed: %s", e)

        # Fallback: generate a synthetic thesis_id so Stage 4/5 lifecycle
        # notifications always fire, even when NarrativeStore has no current state
        if not _prospect_thesis_id:
            import hashlib
            _fallback_src = f"prospect-{prospect_id or 'unknown'}-{direction}-{entry_price}"
            _prospect_thesis_id = "p-" + hashlib.md5(_fallback_src.encode()).hexdigest()[:6]
            logger.info("Prospect thesis_id fallback generated: %s", _prospect_thesis_id)

        # ── Step 2: Persist setup BEFORE sending alert ────────────────────
        # Ensures monitor_pending() can track SL/TP resolution and feed Bayesian.
        setup_id = None
        if prospect_id:
            try:
                rr_list = []
                if sl_price and entry_price and tps:
                    sl_dist = abs(entry_price - sl_price)
                    if sl_dist > 0:
                        rr_list = [round(abs(tp - entry_price) / sl_dist, 2) for tp in tps]
                setup_id = self.db.store_setup(
                    direction=direction,
                    bias=setup.get("bias", "neutral"),
                    entry_price=entry_price,
                    sl_price=sl_price,
                    calibrated_sl=sl_price,  # no calibration at this stage
                    tps=tps or [],
                    setup_quality="A",
                    killzone=killzone,
                    rr_ratios=rr_list,
                    analysis_json=analysis,
                    calibration_json={},
                    timeframe="1h",
                    status="pending",
                    thesis_id=_prospect_thesis_id,
                )
                # Link prospect row → setup row
                self.db.mark_prospect_triggered(prospect_id, setup_id)
                logger.info("PROSPECT setup stored: %s (prospect=%s, thesis=%s)",
                            setup_id, prospect_id, _prospect_thesis_id)
            except Exception as e:
                logger.error("Failed to store prospect setup: %s", e)

        # Calculate lot size for the alert
        lot = _calc_lot_size(entry_price, sl_price)

        # Alert 3: ENTRY — full trade signal
        notify_entry_trigger(analysis)

        # ── Step 4: Stage 4 lifecycle notification ────────────────────────
        try:
            from ml.notifications import notify_lifecycle
            if _prospect_thesis_id and setup_id:
                _setup_dict = {}
                try:
                    import sqlite3 as _sqlite3
                    with _sqlite3.connect(self.db.db_path) as _conn:
                        _conn.row_factory = _sqlite3.Row
                        _row = _conn.execute(
                            "SELECT * FROM scanner_setups WHERE id=?", (setup_id,)
                        ).fetchone()
                        if _row:
                            _setup_dict = dict(_row)
                except Exception:
                    _setup_dict = {"id": setup_id}
                notify_lifecycle(4, _prospect_thesis_id, "1h", {},
                                 setup_data=_setup_dict, db=self.db)
        except Exception as e:
            logger.debug("Prospect lifecycle stage 4 failed: %s", e)

        logger.info("RETRACE ENTRY: %s @ %.2f SL %.2f TPs %s — %s (setup=%s)",
                    direction, entry_price, sl_price, tps,
                    result.get("reason", ""), setup_id)

        analysis["setup_id"] = setup_id
        return analysis

    def _call_sonnet_short(self, prompt: str) -> dict | None:
        """Lightweight Sonnet call for displacement/retrace checks. Returns parsed JSON."""
        try:
            resp = httpx.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.claude_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-sonnet-4-6",
                    "max_tokens": 400,
                    "temperature": 0,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=30,
            )
            if resp.status_code != 200:
                return None

            data = resp.json()
            text = ""
            for block in data.get("content", []):
                if block.get("type") == "text":
                    text = block["text"]
                    break
            if not text:
                return None

            clean = text.replace("```json", "").replace("```", "").strip()
            json_start = clean.find("{")
            json_end = clean.rfind("}")
            if json_start >= 0 and json_end > json_start:
                clean = clean[json_start:json_end + 1]
            return _safe_load_claude_json(clean, "opus_validate")

        except Exception as e:
            logger.warning("Sonnet short call failed: %s", e)
            return None

    def _check_killzone_transition(self):
        """Detect killzone transitions and trigger session recaps + prospects.

        Phase 6: Also generates handoff summaries for narrative continuity.
        """
        from ml.prompts import get_current_killzone
        current_kz = get_current_killzone()
        prev_kz = getattr(self, '_last_killzone', None)
        self._last_killzone = current_kz

        if prev_kz and prev_kz != current_kz:
            # Expire prospects from ending killzone
            if prev_kz != "Off":
                try:
                    self.db.expire_killzone_prospects(prev_kz)
                    self._generate_session_recap(prev_kz)
                except Exception as e:
                    logger.warning("Killzone transition cleanup failed: %s", e)

                # Phase 6: Generate killzone handoff summaries for each TF
                try:
                    self._generate_kz_handoff(prev_kz)
                except Exception as e:
                    logger.debug("Killzone handoff generation failed: %s", e)

            # Generate prospects for the new killzone
            if current_kz != "Off":
                try:
                    self._prospect_killzone_zones(current_kz)
                except Exception as e:
                    logger.warning("Killzone prospecting failed: %s", e)

    def _generate_kz_handoff(self, outgoing_kz: str):
        """Generate killzone handoff summaries for narrative continuity.

        Stores a session summary on each active narrative state so the next
        scan has context from the prior killzone.
        """
        from ml.narrative_state import NarrativeStore
        ns = NarrativeStore(self.db.db_path)

        for tf in TIMEFRAMES:
            try:
                thesis = ns.get_current(tf)
                if not thesis:
                    continue
                summary = (
                    f"{outgoing_kz} session summary: {thesis.get('thesis', 'no thesis')}. "
                    f"Bias was {thesis.get('directional_bias', '?')} "
                    f"({(thesis.get('bias_confidence') or 0):.0%} confidence)."
                )
                ns.update_killzone_summary(thesis["id"], summary)
                logger.info("KZ handoff [%s]: %s → stored summary for thesis %s",
                            tf, outgoing_kz, thesis["id"])
            except Exception as e:
                logger.debug("KZ handoff [%s] failed: %s", tf, e)

    def _call_claude_validate(self, analysis: dict, candles: list,
                              htf_candles: list, intermarket: dict | None,
                              timeframe: str) -> dict | None:
        """Opus validation of a Sonnet-detected A/B setup.

        Returns {"verdict": "validated|downgraded|rejected",
                 "adjusted_quality": str, "validation_note": str,
                 "narrative_coherence": str, "confidence_adjustment": float}
        Cost: ~$0.15/call. Only called for A/B quality setups.
        """
        from ml.prompts import build_validation_prompt, OPUS_VALIDATION_SYSTEM

        prompt = build_validation_prompt(analysis, candles, htf_candles,
                                         intermarket, timeframe)

        for attempt in range(3):
            try:
                resp = httpx.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": self.claude_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": "claude-opus-4-6",
                        "max_tokens": 500,
                        "temperature": 0,
                        "system": OPUS_VALIDATION_SYSTEM,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                    timeout=90,
                )

                if resp.status_code in (429, 529):
                    wait = (2 ** attempt) * 2
                    logger.warning("Opus rate limited (%d), waiting %ds",
                                   resp.status_code, wait)
                    time.sleep(wait)
                    continue

                if resp.status_code != 200:
                    logger.error("Opus API error %d: %s",
                                 resp.status_code, resp.text[:200])
                    if attempt < 2:
                        time.sleep(2 ** attempt * 2)
                        continue
                    return None

                data = resp.json()
                text = ""
                for block in data.get("content", []):
                    if block.get("type") == "text":
                        text = block["text"]
                        break

                if not text:
                    return None

                clean = text.replace("```json", "").replace("```", "").strip()
                json_start = clean.find("{")
                json_end = clean.rfind("}")
                if json_start >= 0 and json_end > json_start:
                    clean = clean[json_start:json_end + 1]

                result = _safe_load_claude_json(clean, "opus_validate")

                # Validate the response structure
                verdict = result.get("verdict", "")
                if verdict not in ("validated", "downgraded", "rejected"):
                    logger.warning("Opus returned invalid verdict: %s", verdict)
                    return None

                # Clamp confidence adjustment
                adj = result.get("confidence_adjustment", 0)
                result["confidence_adjustment"] = max(-0.3, min(0.1, float(adj)))

                logger.info("Opus [%s]: %s — %s (quality: %s, conf adj: %+.2f)",
                            timeframe, verdict.upper(),
                            result.get("validation_note", "")[:80],
                            result.get("adjusted_quality", "?"),
                            result["confidence_adjustment"])
                return result

            except json.JSONDecodeError as e:
                logger.error("Opus JSON parse failed: %s", e)
                if attempt < 2:
                    time.sleep(2)
                    continue
                return None
            except httpx.TimeoutException:
                logger.error("Opus API timeout (attempt %d)", attempt + 1)
                if attempt < 2:
                    time.sleep(2 ** attempt * 2)
                    continue
                return None
            except Exception as e:
                logger.error("Opus call failed: %s", e)
                return None

        return None

    def _call_claude_screen(self, candles: list, htf_candles: list,
                            timeframe: str,
                            prev_narrative: dict | None = None,
                            watch_zones: list | None = None,
                            pending_setups: list | None = None) -> dict | None:
        """Fast Haiku pre-screen — checks if an ICT setup is forming.

        Returns {"setup_possible": bool, "direction": str|None, "reason": str,
                 "zone_interaction": str|None (when context-aware)}
        Cost: ~$0.001/call vs ~$0.03 for full Sonnet analysis.
        Cached: result reused if candles unchanged within TTL.

        When prev_narrative/watch_zones are provided, Haiku gets a context-aware
        prompt that tells it WHAT to look for instead of blind-screening.
        """
        import hashlib
        from ml.prompts import build_screen_prompt

        # Screen cache — avoid re-screening unchanged candles
        # Key uses last 5 candles hash, so new candle data auto-invalidates.
        # TTL reduced from 1800s (30min) to 600s (10min) so "no setup" results
        # don't stick too long — market structure can shift quickly on gold.
        # Include watch_zones hash so new Opus zones invalidate cache.
        zones_hash = hashlib.md5(
            json.dumps(watch_zones or [], sort_keys=True, default=str).encode()
        ).hexdigest()[:8]
        cache_key = f"{timeframe}_{hashlib.md5(str(candles[-5:]).encode()).hexdigest()}_{zones_hash}"
        cached = self._screen_cache.get(cache_key)
        cfg = get_config()
        screen_ttl = cfg.get("screen_cache_ttl_seconds", 600)
        if cached and (datetime.utcnow() - cached["timestamp"]).total_seconds() < screen_ttl:
            logger.debug("Haiku screen cache hit [%s]", timeframe)
            return cached["result"]

        # Budget check
        try:
            from ml.cost_tracker import get_cost_tracker
            if not get_cost_tracker().check_budget():
                logger.warning("Haiku screen skipped — daily budget exceeded")
                return cached["result"] if cached else None
        except Exception:
            pass

        prompt = build_screen_prompt(candles, htf_candles, timeframe,
                                     prev_narrative=prev_narrative,
                                     watch_zones=watch_zones,
                                     pending_setups=pending_setups)

        try:
            resp = httpx.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.claude_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-haiku-4-5-20251001",
                    "max_tokens": 200,
                    "temperature": 0,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=30,
            )

            if resp.status_code != 200:
                logger.warning("Haiku screen failed (%d) — falling through to Sonnet",
                               resp.status_code)
                return None

            data = resp.json()
            text = ""
            for block in data.get("content", []):
                if block.get("type") == "text":
                    text = block["text"]
                    break

            if not text:
                return None

            # Parse JSON from Haiku response
            text = text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            result = json.loads(text)

            # Cache the screen result
            self._screen_cache[cache_key] = {
                "result": result, "timestamp": datetime.utcnow()}
            # Trim cache to 50 entries max
            if len(self._screen_cache) > 50:
                oldest = min(self._screen_cache, key=lambda k: self._screen_cache[k]["timestamp"])
                del self._screen_cache[oldest]

            # Log cost
            try:
                from ml.cost_tracker import get_cost_tracker
                usage = data.get("usage", {})
                _cost = get_cost_tracker().log_call(
                    "haiku", usage.get("input_tokens", 500),
                    usage.get("output_tokens", 50), "screen")
                self._pending_api_cost += _cost  # P8
            except Exception:
                pass

            return result

        except (json.JSONDecodeError, httpx.TimeoutException) as e:
            logger.warning("Haiku screen parse failed: %s — falling through to Sonnet", e)
            return None
        except Exception as e:
            logger.warning("Haiku screen error: %s — falling through to Sonnet", e)
            return None

    def _call_claude(self, candles: list, htf_candles: list,
                     timeframe: str = "1h",
                     intermarket: dict | None = None,
                     htf_narrative: dict | None = None,
                     setup_context: dict | None = None,
                     prev_narrative: dict | None = None,
                     invalidation_status: str | None = None,
                     recent_context: dict | None = None,
                     haiku_zone_hint: str | None = None,
                     ml_context: dict | None = None) -> dict | None:
        """Call Claude API with enhanced ICT prompt + chart image + intermarket + narrative."""
        # Budget check
        try:
            from ml.cost_tracker import get_cost_tracker
            if not get_cost_tracker().check_budget():
                logger.warning("Sonnet analysis skipped — daily budget exceeded")
                return None
        except Exception:
            pass

        from ml.prompts import build_enhanced_ict_prompt

        # Always send full candle windows — Sonnet needs to verify Opus, not just trust it.
        # Previously compressed mode stripped 4H candles and halved execution candles
        # when Opus narrative was present, but this prevented Sonnet from catching
        # narrative errors and reduced its ability to see complete P3 cycles.
        _nw = None
        if htf_narrative:
            try:
                from ml.claude_bridge import ClaudeAnalysisBridge
                _nw = ClaudeAnalysisBridge().get_narrative_weights(killzone=get_current_killzone())
            except Exception:
                pass

        # Compute 5-state structural regime for prompt context
        _regime_ctx = None
        try:
            from ml.volatility import classify_regime
            _regime_ctx = classify_regime(candles)
        except Exception as e:
            logger.debug("Regime classification failed: %s", e)

        # Compute deterministic ICT key levels (PDH/PDL, PWH/PWL, PMH/PML, Asia H/L)
        _key_levels = None
        try:
            from ml.key_levels import compute_all_key_levels
            _intraday_for_levels = self._get_candles("XAU/USD", "15min", 200)
            _key_levels = compute_all_key_levels(
                daily_candles=daily_candles,
                weekly_candles=weekly_candles,
                intraday_candles=_intraday_for_levels,
                current_killzone=current_kz,
            )
            if _key_levels:
                logger.debug("Key levels computed: %d/%d non-None",
                             _key_levels.get("levels_computed", 0), 15)
        except Exception as e:
            logger.debug("Key levels computation failed (proceeding without): %s", e)

        htf_tf = TIMEFRAMES.get(timeframe, {}).get("htf", "4H") if htf_candles else None
        prompt = build_enhanced_ict_prompt(
            candles,        # full execution candles — never truncate
            htf_candles,    # always send HTF — Sonnet must verify Opus
            intermarket=intermarket,
            htf_narrative=htf_narrative,
            setup_context=setup_context,
            narrative_weights=_nw,
            prev_narrative=prev_narrative,
            invalidation_status=invalidation_status,
            recent_context=recent_context,
            regime_context=_regime_ctx,
            ml_context=ml_context,
            htf_label=htf_tf,
            key_levels=_key_levels)
        # Prepend timeframe context so Claude knows what it's analyzing
        tf_note = f"You are analyzing {timeframe} candles for XAU/USD. "
        if htf_candles:
            tf_note += f"Higher timeframe context is provided on {htf_tf}. "
        tf_note += "Identify any ICT setup on this timeframe.\n\n"
        if haiku_zone_hint:
            tf_note += (f"Pre-screen zone interaction: {haiku_zone_hint}\n"
                        "Investigate this zone interaction as a priority.\n\n")
        prompt = tf_note + prompt

        # Strip Unicode line/paragraph separators that OANDA data occasionally injects —
        # these cause httpx to fail with 'ascii' codec errors.
        prompt = prompt.replace('\u2028', '\n').replace('\u2029', '\n')

        # Chart images disabled — saves ~$10/day in image token costs.
        # Claude identifies ICT setups from raw candle data alone.
        # To re-enable: set scanner_send_charts=True in config.
        chart_b64 = None
        cfg_local = get_config()
        if cfg_local.get("scanner_send_charts", False):
            try:
                from ml.chart_renderer import render_chart_base64
                chart_b64 = render_chart_base64(candles, htf_candles, timeframe)
                logger.debug("Scanner [%s]: chart rendered (%d bytes)",
                             timeframe, len(chart_b64))
            except Exception as e:
                logger.warning("Scanner: chart render failed (text-only fallback): %s", e)

        # Build message content (text-only unless charts enabled)
        if chart_b64:
            user_content = [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": chart_b64,
                    },
                },
                {
                    "type": "text",
                    "text": (
                        "Above is the annotated candlestick chart showing detected "
                        "Order Blocks, Fair Value Gaps, liquidity levels (BSL/SSL), "
                        "and the 4H dealing range. Use this visual context alongside "
                        "the raw candle data below to identify ICT setups.\n\n"
                        + prompt
                    ),
                },
            ]
        else:
            user_content = prompt

        for attempt in range(3):
            try:
                resp = httpx.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": self.claude_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": "claude-sonnet-4-6",
                        "max_tokens": 3000,
                        "temperature": 0,
                        "system": ICT_SYSTEM_MESSAGE,
                        "messages": [{"role": "user", "content": user_content}],
                    },
                    timeout=90,
                )

                if resp.status_code == 401:
                    err_msg = "Claude API key invalid (401 Unauthorized)"
                    logger.critical("Scanner: %s", err_msg)
                    self._last_error = err_msg
                    return None

                if resp.status_code in (429, 529):
                    wait = (2 ** attempt) * 2
                    logger.warning("Scanner: Claude rate limited (%d), waiting %ds",
                                   resp.status_code, wait)
                    time.sleep(wait)
                    continue

                if resp.status_code != 200:
                    err_detail = resp.text[:150].strip()
                    logger.error("Scanner: Claude API error %d: %s",
                                 resp.status_code, err_detail)
                    if attempt < 2:
                        time.sleep(2 ** attempt * 2)
                        continue
                    self._last_error = f"Claude API error {resp.status_code}: {err_detail}"
                    return None

                data = resp.json()
                text = ""
                for block in data.get("content", []):
                    if block.get("type") == "text":
                        text = block["text"]
                        break

                if not text:
                    # Surface the actual reason in _last_error so /scanner/status
                    # tells the truth instead of "Claude returned no result".
                    logger.warning("Scanner: Claude returned empty response")
                    self._last_error = f"Claude returned empty response for {timeframe}"
                    return None

                clean = text.replace("```json", "").replace("```", "").strip()
                json_start = clean.find("{")
                json_end = clean.rfind("}")
                if json_start >= 0 and json_end > json_start:
                    clean = clean[json_start:json_end + 1]

                result = _safe_load_claude_json(clean, f"sonnet[{timeframe}]")

                # Log cost
                try:
                    from ml.cost_tracker import get_cost_tracker
                    usage = data.get("usage", {})
                    _cost = get_cost_tracker().log_call(
                        "sonnet", usage.get("input_tokens", 3000),
                        usage.get("output_tokens", 1500), "analysis")
                    self._pending_api_cost += _cost  # P8
                except Exception:
                    pass

                return result

            except json.JSONDecodeError as e:
                logger.error("Scanner: failed to parse Claude JSON: %s", e)
                if attempt < 2:
                    time.sleep(2)
                    continue
                # Capture a snippet of the bad payload so we can see what
                # Sonnet actually returned (truncated JSON? prose? unicode?)
                _snippet = (clean[:200] if "clean" in dir() else text[:200]) if "text" in dir() else ""
                self._last_error = (
                    f"Claude JSON parse error for {timeframe}: {e}"
                    + (f" | first 200 chars: {_snippet!r}" if _snippet else "")
                )
                return None
            except httpx.TimeoutException:
                logger.error("Scanner: Claude API timeout (attempt %d)", attempt + 1)
                if attempt < 2:
                    time.sleep(2 ** attempt * 2)
                    continue
                self._last_error = f"Claude API timeout for {timeframe} after 3 attempts"
                return None
            except Exception as e:
                logger.error("Scanner: Claude call failed: %s", e)
                self._last_error = (
                    f"Claude call exception for {timeframe}: "
                    f"{type(e).__name__}: {str(e)[:200]}"
                )
                return None

        return None

    def _calibrate(self, analysis: dict, candles: list) -> dict | None:
        """Run ML calibration on the analysis."""
        try:
            from ml.claude_bridge import ClaudeAnalysisBridge
            from ml.calibrate import MLCalibrator

            bridge = ClaudeAnalysisBridge()
            parsed = bridge.parse_analysis(analysis, candles)

            calibrator = MLCalibrator()
            return calibrator.calibrate_trade(parsed, candles)
        except Exception as e:
            logger.warning("Scanner: calibration failed: %s", e)
            return None

    def _log_trade_complete(self, setup: dict, result: dict, candles: list = None):
        """Post resolved trade to the existing trade/complete flow.

        Updates: accuracy tracker, Bayesian beliefs, training dataset.
        If candles are provided, extracts full features (not minimal zeros).
        Auto-retrains every N trades (configurable).
        Shadow trades (Opus-rejected) are tracked for outcome but don't
        update Bayesian beliefs or trigger notifications.
        """
        is_shadow = setup.get("status") == "shadow"

        try:
            from ml.claude_bridge import ClaudeAnalysisBridge
            bridge = ClaudeAnalysisBridge()

            # Track Opus validation + narrative outcomes
            raw_analysis = setup.get("analysis_json", {})
            is_win = result["outcome"].startswith("tp")

            opus_val = raw_analysis.get("opus_validation")
            if opus_val:
                bridge.update_opus_tracker(
                    opus_val["verdict"], is_win,
                    killzone=setup.get("killzone", ""),
                    timeframe=setup.get("timeframe", ""),
                    confidence=opus_val.get("confidence", 0.5),
                    direction=setup.get("direction", ""),
                    pnl_rr=result.get("pnl_rr", 0.0),
                )

            # Track prospect-triggered trade outcomes
            if raw_analysis.get("prospect_triggered"):
                self._update_prospect_tracker("resolved",
                                              killzone=setup.get("killzone", ""),
                                              is_win=is_win)

            # Track narrative alignment
            cal_json = setup.get("calibration_json") or {}
            narrative_bias = cal_json.get("opus_narrative", {}).get("directional_bias")
            entry_dir = setup.get("direction", "")
            _kz = setup.get("killzone", "")
            _phase = cal_json.get("opus_narrative", {}).get("power_of_3_phase", "")
            bridge.update_narrative_tracker(
                narrative_bias, entry_dir, is_win,
                killzone=_kz or None, phase=_phase or None)

            # Update per-field EMA weights + gold examples + bandit
            narrative_json = cal_json.get("opus_narrative", {})
            if narrative_json:
                bridge.update_narrative_field_weights(
                    narrative_json, entry_dir, is_win, result["outcome"], setup,
                    mfe_atr=result.get("mfe_atr"),
                    killzone=_kz or None)
                try:
                    from ml.narrative_examples import NarrativeExampleStore
                    store = NarrativeExampleStore()
                    store.add_example(
                        narrative_json=narrative_json,
                        outcome=result["outcome"],
                        session=setup.get("killzone", "Off"),
                        direction=entry_dir,
                        entry_price=setup.get("entry_price", 0),
                        key_levels=narrative_json.get("key_levels", []),
                    )
                except Exception as e:
                    logger.debug("Gold example store update failed: %s", e)

            # Update bandit arm
            try:
                arm_id = getattr(self, '_current_bandit_arm', None)
                if arm_id:
                    from ml.narrative_bandit import NarrativeBandit
                    bandit = NarrativeBandit()
                    bandit.update_arm(arm_id, is_win)
                    bandit.retire_underperformers()
                    bandit.maybe_generate_variant()
                    self._current_bandit_arm = None
            except Exception:
                pass

            # Shadow trades: only track Opus outcomes, skip full pipeline
            if is_shadow:
                print(f"[MONITOR] Shadow trade resolved: {setup['id']} "
                      f"[{setup.get('timeframe', '?')}] {result['outcome']} "
                      f"(Opus-rejected, would have {'WON' if result['outcome'].startswith('tp') else 'LOST'})")
                logger.info("Shadow trade %s resolved: %s", setup["id"], result["outcome"])
                return

            # ── Setup DNA profiling (zero API cost) ──
            try:
                from ml.setup_dna import encode_setup_dna
                from ml.setup_profiles import SetupProfileStore
                dna = encode_setup_dna(raw_analysis, cal_json,
                                       setup.get("timeframe", "1h"),
                                       setup.get("killzone", "Off"))
                _mfe_atr = result.get("mfe_atr")
                _mae_atr = result.get("mae_atr")
                SetupProfileStore().add_profile(
                    setup["id"], dna, result["outcome"],
                    result.get("pnl_rr", result.get("rr", 0)),
                    mfe=_mfe_atr, mae=_mae_atr)
            except Exception as e:
                logger.debug("Setup DNA profiling failed: %s", e)

            parsed_analysis = bridge.parse_analysis(raw_analysis)

            # ── Rich feature extraction: correct-timeframe candles + mechanical detectors ──
            try:
                from ml.features import (extract_features, compute_atr,
                                          detect_order_blocks, detect_fvgs,
                                          detect_liquidity)
                tf = setup.get("timeframe", "1h")
                tf_cfg = TIMEFRAMES.get(tf, {"fetch": 180})

                # Fetch correct-timeframe candles (not the 5-min monitoring candles)
                tf_candles = self._fetch_candles(tf, tf_cfg.get("fetch", 180))
                if tf_candles and len(tf_candles) >= 15:
                    # Reverse if newest-first (OANDA returns newest-first)
                    if (len(tf_candles) >= 2
                            and tf_candles[0].get("datetime", "") > tf_candles[-1].get("datetime", "")):
                        tf_candles = list(reversed(tf_candles))

                    atr = compute_atr(tf_candles, 14)

                    # ── MERGE strategy: mechanical detectors primary, Claude supplementary ──
                    # Analysis showed no-zone trades (73.3% WR) slightly outperform zone
                    # trades (68.6% WR) across 528 trades — so use precise mechanical
                    # detectors as primary source, keep Claude zones that don't overlap.
                    enriched = dict(raw_analysis)
                    mech_obs = detect_order_blocks(tf_candles, atr) if atr > 0 else []
                    mech_fvgs = detect_fvgs(tf_candles, atr) if atr > 0 else []
                    mech_liq = detect_liquidity(tf_candles)

                    claude_obs = enriched.get("orderBlocks") or []
                    claude_fvgs = enriched.get("fvgs") or []

                    # Tag sources for chart rendering
                    for ob in mech_obs:
                        ob["source"] = "mechanical"
                    for ob in claude_obs:
                        ob["source"] = "claude"
                    for fvg in mech_fvgs:
                        fvg["source"] = "mechanical"
                    for fvg in claude_fvgs:
                        fvg["source"] = "claude"

                    # Merge OBs: mechanical first, then Claude zones that don't overlap
                    merged_obs = list(mech_obs)
                    for c_ob in claude_obs:
                        c_lo, c_hi = c_ob.get("low", 0), c_ob.get("high", 0)
                        overlaps = any(
                            m.get("low", 0) <= c_hi and m.get("high", 0) >= c_lo
                            for m in mech_obs
                        )
                        if not overlaps:
                            merged_obs.append(c_ob)

                    # Merge FVGs: mechanical first, then non-overlapping Claude FVGs
                    merged_fvgs = list(mech_fvgs)
                    for c_fvg in claude_fvgs:
                        c_lo, c_hi = c_fvg.get("low", 0), c_fvg.get("high", 0)
                        overlaps = any(
                            m.get("low", 0) <= c_hi and m.get("high", 0) >= c_lo
                            for m in mech_fvgs
                        )
                        if not overlaps:
                            merged_fvgs.append(c_fvg)

                    enriched["orderBlocks"] = merged_obs
                    enriched["fvgs"] = merged_fvgs
                    if not enriched.get("liquidity"):
                        enriched["liquidity"] = mech_liq

                    # Populate entry zone fields from setup metadata
                    entry_price = setup.get("entry_price", 0)
                    sl_price = setup.get("sl_price", 0)
                    if entry_price and sl_price and atr > 0:
                        # Find the nearest OB/FVG zone containing entry
                        zone_high = zone_low = None
                        for ob in (enriched.get("orderBlocks") or []):
                            if ob.get("low", 0) <= entry_price <= ob.get("high", 0):
                                zone_high = ob["high"]
                                zone_low = ob["low"]
                                break
                        if zone_high is None:
                            for fvg in (enriched.get("fvgs") or []):
                                if fvg.get("low", 0) <= entry_price <= fvg.get("high", 0):
                                    zone_high = fvg["high"]
                                    zone_low = fvg["low"]
                                    break
                        if zone_high is not None and zone_low is not None and zone_high != zone_low:
                            enriched["entry_zone_position"] = (entry_price - zone_low) / (zone_high - zone_low)
                            enriched["entry_zone_size_atr"] = (zone_high - zone_low) / atr

                    intermarket_ctx = (setup.get("calibration_json") or {}).get("intermarket")
                    # Compute key levels at resolution time for feature extraction
                    _res_key_levels = None
                    try:
                        from ml.key_levels import compute_all_key_levels
                        _daily_for_kl = self._get_candles("XAU/USD", "1day", 45)
                        _weekly_for_kl = self._get_candles("XAU/USD", "1week", 12)
                        _intra_for_kl = self._get_candles("XAU/USD", "15min", 200)
                        _res_key_levels = compute_all_key_levels(
                            daily_candles=_daily_for_kl,
                            weekly_candles=_weekly_for_kl,
                            intraday_candles=_intra_for_kl,
                        )
                    except Exception:
                        pass
                    features = extract_features(enriched, tf_candles, tf,
                                                intermarket=intermarket_ctx,
                                                calibration=cal_json,
                                                key_levels=_res_key_levels)
                    # Populate volatility_regime from 5-state classifier
                    try:
                        from ml.volatility import classify_regime
                        regime_result = classify_regime(tf_candles)
                        features["volatility_regime"] = regime_result["regime"]
                        features["regime_confidence"] = regime_result["confidence"]
                        features["vol_ratio_5_30"] = regime_result["metrics"]["vol_ratio_5_30"]
                        features["net_movement_atr"] = regime_result["metrics"]["net_movement_atr"]
                    except Exception:
                        pass
                    parsed_analysis["features"] = features
                    logger.info("Rich features extracted for %s [%s]: %d non-zero",
                                setup.get("id", "?"), tf,
                                sum(1 for v in features.values() if v))
                else:
                    logger.warning("Could not fetch %s candles for feature extraction", tf)
            except Exception as fe:
                logger.warning("Rich feature extraction failed: %s", fe)

            bridge.log_completed_trade(
                original_analysis=parsed_analysis,
                calibrated_result=setup.get("calibration_json", {}),
                actual_outcome=result["outcome"],
                actual_pnl_atr=result.get("rr", 0),
                used_calibrated_sl=setup.get("calibrated_sl") is not None,
                notes=f"Scanner auto-resolved [{setup.get('timeframe', '?')}]: {result['outcome']}",
                source="scanner",
            )
            print(f"[MONITOR] Logged trade: {setup['id']} [{setup.get('timeframe', '?')}] "
                  f"{result['outcome']} → Accuracy+Dataset updated (Bayes skipped — scanner)")
            logger.info("Trade logged: %s [%s] %s — models updated",
                        setup["id"], setup.get("timeframe", "?"), result["outcome"])

            # ── Stage 5: RESOLVED lifecycle (single notification path) ──
            # Stage 5 handles: Telegram notification, record_daily_pnl, DD display
            try:
                from ml.notifications import notify_lifecycle
                _thesis_id = setup.get("thesis_id")
                # Generate fallback thesis_id so Stage 5 always fires
                # (prospect-triggered trades may have NULL thesis_id)
                if not _thesis_id:
                    import hashlib
                    _fb_src = f"resolve-{setup.get('id', 'unknown')}-{setup.get('direction', '')}"
                    _thesis_id = "r-" + hashlib.md5(_fb_src.encode()).hexdigest()[:6]
                    logger.info("Stage 5 fallback thesis_id: %s (setup %s)", _thesis_id, setup.get("id"))
                notify_lifecycle(5, _thesis_id, setup.get("timeframe", ""), {},
                                setup_data=setup,
                                resolution_data=result,
                                db=self.db)
            except Exception as e:
                logger.warning("Lifecycle stage 5 failed: %s", e)

            # ── Hash invalidation: force re-analysis on next scan ──
            tf = setup.get('timeframe', '1h')
            if tf in self._candle_hashes:
                del self._candle_hashes[tf]
                logger.info('Hash invalidated for %s after resolution', tf)

            # ── Layer performance tracking (P3 feedback loop) ──
            try:
                from ml.layer_performance import LayerPerformanceTracker
                from ml.features import compute_atr
                _lpt = LayerPerformanceTracker()
                _lp_atr = compute_atr(candles, 14) if candles and len(candles) >= 15 else 1.0
                _lpt.ingest_trade(
                    calibration_json=cal_json,
                    outcome=result["outcome"],
                    mae_atr=result.get("mae_atr") or 0,
                    mfe_atr=result.get("mfe_atr") or 0,
                    entry_price=setup.get("entry_price", 0),
                    atr=_lp_atr,
                    setup_grade=setup.get("setup_quality", ""),
                    killzone=setup.get("killzone", ""),
                )
                _lpt.flush()
            except Exception as e:
                logger.debug("Layer performance ingestion failed: %s", e)

            # Priority 8: Cost-per-winner tracking
            try:
                setup_with_outcome = dict(setup)
                setup_with_outcome["outcome"] = result["outcome"]
                setup_with_outcome["pnl_rr"] = result.get("rr", 0)
                self._cpw_tracker.ingest_trade(setup_with_outcome)
            except Exception as e:
                logger.debug("Cost-per-winner ingestion failed: %s", e)

            # System evolution snapshot (throttled — max 1 per hour)
            try:
                from ml.system_snapshot import SystemSnapshotRecorder
                SystemSnapshotRecorder(db_path=self.db.db_path).maybe_take_snapshot(
                    trigger="trade_resolved")
            except Exception as e:
                logger.debug("System snapshot failed: %s", e)

            # Auto-retrain check
            self._maybe_auto_retrain()

        except Exception as e:
            print(f"[MONITOR] Trade logging FAILED for {setup.get('id', '?')}: {e}")
            logger.error("Scanner: trade logging failed: %s", e, exc_info=True)

            # ── CRITICAL: Ensure Stage 5 notification + DD update even if logging fails ──
            # The try block above covers ClaudeAnalysisBridge, feature extraction,
            # and ML pipeline. If ANY crash before Stage 5, the user gets no
            # notification and DD is never updated. This safety net fires Stage 5.
            try:
                from ml.notifications import notify_lifecycle
                import hashlib
                _thesis_id = setup.get("thesis_id")
                if not _thesis_id:
                    _fb_src = f"resolve-{setup.get('id', 'unknown')}-{setup.get('direction', '')}"
                    _thesis_id = "r-" + hashlib.md5(_fb_src.encode()).hexdigest()[:6]
                notify_lifecycle(5, _thesis_id, setup.get("timeframe", ""), {},
                                setup_data={**setup, "outcome": result["outcome"],
                                            "rr": result.get("rr", 0)},
                                db=self.db)
                logger.info("Safety-net Stage 5 fired for %s after logging failure", setup.get("id"))
            except Exception as notify_err:
                logger.error("Safety-net Stage 5 ALSO failed: %s", notify_err)
                # Last resort: at least update DD in the database
                try:
                    from ml.notifications import record_daily_pnl, _calc_lot_size
                    _eff_sl = setup.get("calibrated_sl") or setup.get("sl_price", 0)
                    _lot = _calc_lot_size(setup.get("entry_price", 0), _eff_sl)
                    _pnl = result.get("rr", 0) * _lot["risk_dollars"] if _lot["risk_dollars"] else 0
                    if _pnl:
                        record_daily_pnl(_pnl)
                        logger.info("Last-resort DD update: $%.0f for setup %s", _pnl, setup.get("id"))
                except Exception as dd_err:
                    logger.error("Last-resort DD update failed: %s", dd_err)

    def _trigger_post_resolution_scan(self, timeframe: str, outcome: str):
        """Immediate re-analysis after trade resolution.

        Closes the gap between resolution and next scheduled scan.
        Uses enriched context from Phase 1 so Claude knows what just happened.
        """
        # 2-minute cooldown per timeframe — prevents cascade if multiple resolve
        last = self._post_resolution_scans.get(timeframe)
        if last and (datetime.utcnow() - last).total_seconds() < 120:
            logger.debug("Post-resolution scan skipped for %s — cooldown", timeframe)
            return
        self._post_resolution_scans[timeframe] = datetime.utcnow()

        cfg = TIMEFRAMES.get(timeframe)
        if not cfg:
            return

        candles = self._fetch_candles(timeframe, cfg["fetch"])
        if not candles:
            logger.warning("Post-resolution scan: no candles for %s", timeframe)
            return

        htf_candles = self._get_htf_candles(cfg["htf"], cfg["htf_count"])

        # Build enriched context (Phase 1)
        recent_ctx = None
        try:
            from ml.recent_context import build_recent_context
            recent_ctx = build_recent_context(timeframe, self.db)
        except Exception:
            pass

        trimmed = candles[-cfg["count"]:]
        result = self._analyze_and_store(
            timeframe, trimmed, htf_candles, candles,
            recent_context=recent_ctx)
        logger.info("Post-resolution re-scan [%s] after %s: %s",
                    timeframe, outcome, result.get("status"))

    def _maybe_auto_retrain(self):
        """Auto-retrain AutoGluon + run evaluation every N resolved trades."""
        try:
            from ml.config import get_config
            cfg = get_config()
            retrain_n = cfg.get("retrain_every_n_trades", 50)

            # Count resolved trades
            resolved = self.db.get_stats().get("resolved", 0)

            # Check last retrain count from history file
            import json
            history_path = os.path.join(cfg["model_dir"], "retrain_history.json")
            history = []
            if os.path.exists(history_path):
                with open(history_path) as f:
                    history = json.load(f)

            last_retrain_trades = history[-1]["trades_used"] if history else 0
            new_trades = resolved - last_retrain_trades

            # Check for narrative accuracy improvement that warrants early retrain
            # If a high-impact field improved significantly since last retrain,
            # the model should learn from the now-more-reliable features sooner
            narrative_improved = False
            if new_trades >= 10 and history:
                try:
                    prev_narr = history[-1].get("narrative_accuracy", {})
                    from ml.claude_bridge import ClaudeAnalysisBridge
                    curr_narr = ClaudeAnalysisBridge().get_narrative_weights()
                    for field in ("premium_discount", "directional_bias", "p3_phase"):
                        prev_acc = prev_narr.get(field, 0.5)
                        curr_acc = curr_narr.get(field, 0.5)
                        if curr_acc - prev_acc >= 0.15:  # 15% improvement
                            narrative_improved = True
                            print(f"[RETRAIN] Narrative improvement detected: "
                                  f"{field} {prev_acc:.0%} → {curr_acc:.0%}")
                except Exception:
                    pass

            if not narrative_improved and new_trades < retrain_n:
                return

            print(f"[RETRAIN] Auto-retrain triggered: {resolved} trades ({resolved - last_retrain_trades} new)")

            from ml.training import train_classifier
            from ml.database import TradeLogger
            from ml.dataset import TrainingDatasetManager
            from ml.evaluation import evaluate_classifier_walkforward

            db = TradeLogger()
            dm = TrainingDatasetManager()

            # Get before accuracy
            eval_path = os.path.join(cfg["model_dir"], "classifier_evaluation.json")
            oos_before = 0
            weaknesses_before = []
            if os.path.exists(eval_path):
                with open(eval_path) as f:
                    prev = json.load(f)
                oos_before = prev.get("oos_accuracy", 0)
                weaknesses_before = prev.get("weaknesses", [])

            # Retrain
            train_result = train_classifier(db, dataset_manager=dm, live_only=True)
            print(f"[RETRAIN] Training complete: {train_result.get('status')}, "
                  f"OOS={train_result.get('oos_accuracy', 0):.1%}")

            # Evaluate
            eval_result = evaluate_classifier_walkforward(dm)
            oos_after = eval_result.get("oos_accuracy", 0)
            weaknesses_after = eval_result.get("weaknesses", [])

            # ── Accuracy gate: reject models below 55% OOS ──
            min_oos = 0.55
            if oos_after < min_oos:
                print(f"[RETRAIN] REJECTED: OOS {oos_after:.1%} below {min_oos:.0%} gate")
                import shutil
                for subdir in ("classifier_binary", "classifier"):
                    p = os.path.join(cfg["model_dir"], subdir)
                    if os.path.exists(p):
                        shutil.rmtree(p)
                return

            # Reject if regression from a good model
            if oos_before >= min_oos and oos_after < oos_before:
                print(f"[RETRAIN] REJECTED: OOS regressed {oos_before:.1%} -> {oos_after:.1%}")
                import shutil
                for subdir in ("classifier_binary", "classifier"):
                    p = os.path.join(cfg["model_dir"], subdir)
                    if os.path.exists(p):
                        shutil.rmtree(p)
                return

            # Feature quality
            full_pct = 1 - eval_result.get("feature_quality", {}).get("minimal_feature_pct", 1.0)

            # Capture narrative accuracy snapshot for tracking improvement over time
            narrative_snapshot = {}
            try:
                from ml.claude_bridge import ClaudeAnalysisBridge
                nw = ClaudeAnalysisBridge().get_narrative_weights()
                narrative_snapshot = {k: round(v, 3) for k, v in nw.items()
                                      if isinstance(v, (int, float))}
            except Exception:
                pass

            # Log to history
            from datetime import datetime
            entry = {
                "retrained_at": datetime.utcnow().isoformat(),
                "trades_used": resolved,
                "oos_accuracy_before": round(oos_before, 4),
                "oos_accuracy_after": round(oos_after, 4),
                "improvement": f"{(oos_after - oos_before):+.1%}",
                "weaknesses_before": weaknesses_before[:5],
                "weaknesses_after": weaknesses_after[:5],
                "full_feature_pct": round(full_pct, 3),
                "narrative_accuracy": narrative_snapshot,
            }
            history.append(entry)
            history = history[-20:]  # keep last 20
            with open(history_path, "w") as f:
                json.dump(history, f, indent=2)

            print(f"[RETRAIN] Evaluation: OOS {oos_before:.1%} → {oos_after:.1%} "
                  f"({entry['improvement']}), full features: {full_pct:.0%}")

        except Exception as e:
            logger.warning("Auto-retrain failed (non-fatal): %s", e)

    def _hash_candles(self, candles: list) -> str:
        """Hash candle data for change detection."""
        raw = json.dumps([(c["datetime"], c["close"]) for c in candles[-5:]])
        return hashlib.md5(raw.encode()).hexdigest()[:12]


# ── Shared singleton ──────────────────────────────────────────────────────────
# Both server.py and scheduler.py must use this so they share one engine
# instance. Having two separate instances means scheduler increments _total_scans
# on one object while the status endpoint reads zero from the other.

_shared_engine: ScannerEngine | None = None


def get_shared_engine() -> ScannerEngine:
    """Return the process-wide singleton ScannerEngine, creating it if needed."""
    global _shared_engine
    if _shared_engine is None:
        _shared_engine = ScannerEngine()
    return _shared_engine
