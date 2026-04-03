"""Automated backtest generator V2 — three-pass architecture.

Pass 1: Structural pre-filter (FREE) — runs existing pure-Python ICT detection
         from ml/features.py over entire candle history, produces candidate windows
         with confluence scores >= 2.

Pass 2: Haiku batch validation (~$0.001/call) — sends each candidate to Haiku
         with pre-detected structural elements. Haiku validates rather than discovers.

Pass 3: Sonnet enrichment (~$0.076/call) — only Haiku-approved candidates get
         the full ICT system prompt. Includes Haiku's direction as a hint.

Target: 150-300 labelled setups from 12 months for under $15.

Also includes:
  - Regime classification (trending/ranging/volatile) via daily ATR(14)
  - Killzone filtering (skips Asia + off-hours)
  - Regime cap (no regime > 45%)
  - Entry noise jitter for data augmentation
  - Budget tracking + checkpointing (per-pass)
  - Fidelity checks (backtest vs live distribution comparison)
"""

import bisect
import json
import logging
import math
import os
import random
import time
from datetime import datetime, timedelta

import httpx

from ml.claude_bridge import ClaudeAnalysisBridge
from ml.config import get_config
from ml.cost_tracker import get_cost_tracker
from ml.data_providers import get_provider
from ml.dataset import TrainingDatasetManager
from ml.feature_schema import FEATURE_COLUMNS
from ml.features import extract_features
from ml.prompts import build_enhanced_ict_prompt, build_haiku_backtest_prompt

logger = logging.getLogger(__name__)

# Killzone definitions (UTC hours) — must match ml/prompts.py
_KILLZONES = {
    "London": (7, 12),
    "NY_AM": (12, 16),
    "NY_PM": (16, 20),
}

_SKIP_KILLZONES = {"Off"}

# Rate limits
_RATE_LIMIT_CALLS = 10
_RATE_LIMIT_WINDOW = 60  # seconds

_HAIKU_RATE_LIMIT_CALLS = 50
_HAIKU_RATE_LIMIT_WINDOW = 60  # seconds


class BacktestGenerator:
    """Generate labelled training data from historical candles + Claude analysis."""

    def __init__(self, config: dict = None):
        from ml.config import get_config
        self.cfg = config or get_config()
        self._checkpoint_path = os.path.join(
            self.cfg.get("model_dir", "ml/models"), "backtest_checkpoint.json"
        )
        self._meta_path = os.path.join(
            self.cfg.get("model_dir", "ml/models"), "backtest_meta.json"
        )
        self._pass2_cache_path = os.path.join(
            self.cfg.get("model_dir", "ml/models"), "backtest_pass2_cache.json"
        )
        self._call_timestamps: list[float] = []
        self._haiku_call_timestamps: list[float] = []
        self._rng = random.Random(42)

    # ------------------------------------------------------------------
    # Pass 1: Structural pre-filter (FREE)
    # ------------------------------------------------------------------

    def structural_scan(self, candles_1h: list[dict],
                        regime_map: dict) -> list[dict]:
        """Pass 1: Free structural pre-filter using existing ICT detection.

        Runs detect_order_blocks, detect_fvgs, detect_liquidity, and
        compute_market_structure from ml/features.py over each killzone-eligible
        candle. Scores confluence (0-6) and returns candidates with score >= 2.

        Returns list of candidate dicts sorted by candle index.
        """
        from ml.features import (
            compute_atr, detect_order_blocks, detect_fvgs,
            detect_liquidity, compute_market_structure,
        )

        candidates = []

        for i in range(40, len(candles_1h) - 50):
            dt_str = candles_1h[i].get("datetime", "")
            kz = self._get_killzone(dt_str)
            if kz in _SKIP_KILLZONES:
                continue

            # Regime for this candle's month
            month_key = dt_str[:7] if len(dt_str) >= 7 else "unknown"
            regime = regime_map.get(month_key, "ranging")

            window = candles_1h[max(0, i - 40):i + 1]
            atr = compute_atr(window, min(14, len(window) - 1))
            if atr <= 0:
                continue

            price = candles_1h[i]["close"]
            score = 0
            elements = {}

            # 1. Order blocks in last 10 candles
            obs = detect_order_blocks(window, atr)
            recent_obs = [ob for ob in obs if ob["index"] > len(window) - 11]
            if recent_obs:
                score += 1
            elements["ob_count"] = len(recent_obs)
            elements["ob_types"] = [ob["type"] for ob in recent_obs]

            # 2. FVGs near price (within 2x ATR)
            fvgs = detect_fvgs(window)
            nearby_fvgs = [
                f for f in fvgs
                if abs((f["high"] + f["low"]) / 2 - price) < 2 * atr
            ]
            if nearby_fvgs:
                score += 1
            elements["fvg_count"] = len(nearby_fvgs)
            elements["fvg_types"] = [f["type"] for f in nearby_fvgs]

            # 3. Liquidity sweep in last 5 candles
            liq = detect_liquidity(window, window=8)
            sweep = False
            for level in liq:
                if level["index"] > len(window) - 6:
                    for j in range(max(0, len(window) - 5), len(window)):
                        if (level["type"] == "buyside"
                                and window[j]["high"] > level["price"]):
                            sweep = True
                        elif (level["type"] == "sellside"
                              and window[j]["low"] < level["price"]):
                            sweep = True
            if sweep:
                score += 1
            elements["sweep_detected"] = sweep

            # 4. Clear market structure (|score| > 0.3)
            struct_score = compute_market_structure(window, lookback=20)
            if abs(struct_score) > 0.3:
                score += 1
            elements["structure_score"] = round(struct_score, 3)

            # 5. Displacement candle in last 3
            displacement = False
            for j in range(max(0, len(window) - 3), len(window)):
                body = abs(window[j]["close"] - window[j]["open"])
                if body > 1.5 * atr:
                    displacement = True
                    break
            if displacement:
                score += 1
            elements["displacement"] = displacement

            # 6. Price at OB/FVG zone
            price_in_zone = False
            for ob in recent_obs:
                if ob["low"] <= price <= ob["high"]:
                    price_in_zone = True
                    break
            if not price_in_zone:
                for fvg in nearby_fvgs:
                    if fvg["low"] <= price <= fvg["high"]:
                        price_in_zone = True
                        break
            if price_in_zone:
                score += 1
            elements["price_in_zone"] = price_in_zone

            if score >= 2:
                candidates.append({
                    "candle_idx": i,
                    "score": score,
                    "killzone": kz,
                    "regime": regime,
                    "structural_elements": elements,
                    "atr": round(atr, 4),
                })

        # Deduplicate: cluster within 4 candles, keep highest score
        candidates = self._deduplicate_candidates(candidates, gap=4)

        logger.info(
            "Structural scan: %d candidates from %d candles (score >= 2)",
            len(candidates), len(candles_1h),
        )
        return candidates

    @staticmethod
    def _deduplicate_candidates(candidates: list[dict],
                                gap: int = 4) -> list[dict]:
        """Cluster adjacent candidates and keep highest-scoring per cluster."""
        if not candidates:
            return []

        sorted_cands = sorted(candidates, key=lambda c: c["candle_idx"])
        clusters = []
        current_cluster = [sorted_cands[0]]

        for c in sorted_cands[1:]:
            if c["candle_idx"] - current_cluster[-1]["candle_idx"] <= gap:
                current_cluster.append(c)
            else:
                clusters.append(current_cluster)
                current_cluster = [c]
        clusters.append(current_cluster)

        return [max(cluster, key=lambda c: c["score"]) for cluster in clusters]

    def _apply_regime_cap(self, candidates: list[dict],
                          max_pct: float = 0.45) -> list[dict]:
        """Filter candidates so no single regime exceeds max_pct of total.

        Only caps when there are at least 2 regimes present — if all
        candidates are the same regime, they all pass through.
        """
        # Count regimes present
        regimes_present = set(c.get("regime", "ranging") for c in candidates)
        if len(regimes_present) <= 1:
            logger.info(
                "Regime cap skipped: only %d regime(s) present in %d candidates",
                len(regimes_present), len(candidates),
            )
            return candidates

        regime_counts = {"trending": 0, "ranging": 0, "volatile": 0}
        filtered = []

        for cand in candidates:
            regime = cand.get("regime", "ranging")
            total = max(sum(regime_counts.values()), 1)
            if regime_counts.get(regime, 0) / total > max_pct and total > 20:
                continue
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
            filtered.append(cand)

        logger.info(
            "Regime cap applied: %d → %d candidates. Distribution: %s",
            len(candidates), len(filtered), regime_counts,
        )
        return filtered

    # ------------------------------------------------------------------
    # Main generation (three-pass pipeline)
    # ------------------------------------------------------------------

    def generate(
        self,
        months_back: int = 12,
        max_setups: int = 300,
        budget_limit_usd: float = 20.0,
        dry_run: bool = False,
    ) -> dict:
        """Generate backtest training data using three-pass architecture.

        Pass 1: structural_scan() — free Python pre-filter
        Pass 2: Haiku validation — ~$0.001/call
        Pass 3: Sonnet enrichment — ~$0.076/call (only on Haiku-approved)

        Args:
            months_back: How many months of history to scan.
            max_setups: Stop after this many labelled setups.
            budget_limit_usd: Stop when cumulative cost exceeds this.
            dry_run: If True, run Pass 1 only and return candidate count.

        Returns:
            Summary dict with pass1_candidates, pass2_haiku_approved,
            pass3_sonnet_analysed, costs, etc.
        """
        # ── Check for Pass 2 checkpoint (resume Pass 3 only) ──
        checkpoint = self._load_checkpoint()
        resume_pass3 = (
            checkpoint.get("pass") == 2
            and os.path.exists(self._pass2_cache_path)
        )

        # dry_run applies regardless of checkpoint — never spend API money
        if dry_run and resume_pass3:
            return {
                "pass1_candidates": checkpoint.get("pass1_candidates", 0),
                "pass2_haiku_approved": checkpoint.get("pass2_haiku_approved", 0),
                "dry_run": True,
                "resumed_from_checkpoint": True,
            }

        provider = get_provider(
            self.cfg.get("backtest_data_source", "oanda"), self.cfg
        )

        end_date = datetime.utcnow().strftime("%Y-%m-%d")
        start_date = (
            datetime.utcnow() - timedelta(days=months_back * 30)
        ).strftime("%Y-%m-%d")

        # Fetch candles (always needed for Pass 3 labelling)
        logger.info(
            "Fetching 1H candles %s → %s from %s",
            start_date, end_date, provider.name(),
        )
        candles_1h = provider.fetch_candles("XAU/USD", "1h", start_date, end_date)
        if len(candles_1h) < 250:
            logger.error(
                "Insufficient 1H candles (%d), need at least 250", len(candles_1h)
            )
            return {"error": "insufficient_candles", "count": len(candles_1h)}

        # Fetch 4H candles for HTF context (matches live scanner)
        logger.info("Fetching 4H candles for HTF context")
        try:
            candles_4h = provider.fetch_candles(
                "XAU/USD", "4h", start_date, end_date
            )
            logger.info("Fetched %d 4H candles", len(candles_4h))
        except Exception as e:
            logger.warning("4H candle fetch failed: %s — continuing without", e)
            candles_4h = []

        # Build 4H datetime index for fast alignment with 1H candles
        _4h_timestamps = []
        for c in candles_4h:
            dt_str = c.get("datetime", "")
            _4h_timestamps.append(dt_str[:19].replace("T", " "))

        # Fetch intermarket data (DXY proxy + US10Y) — free OANDA calls
        dxy_candles_1h = []
        us10y_candles_1h = []
        try:
            logger.info("Fetching intermarket candles (EUR_USD, USB10Y)")
            dxy_candles_1h = provider.fetch_candles(
                "EUR/USD", "1h", start_date, end_date
            )
            us10y_candles_1h = provider.fetch_candles(
                "US10Y", "1h", start_date, end_date
            )
            logger.info(
                "Intermarket: %d DXY candles, %d US10Y candles",
                len(dxy_candles_1h), len(us10y_candles_1h),
            )
        except Exception as e:
            logger.warning(
                "Intermarket fetch failed: %s — continuing without", e
            )

        claude_key = os.getenv("ANTHROPIC_API_KEY", "")
        tracker = get_cost_tracker()
        bridge = ClaudeAnalysisBridge(config=self.cfg)
        dataset_mgr = TrainingDatasetManager(config=self.cfg)

        if resume_pass3:
            # ── Resume: load cached Pass 2 results ──
            logger.info("Resuming from Pass 2 checkpoint — skipping Pass 1 & 2")
            with open(self._pass2_cache_path) as f:
                cache = json.load(f)
            haiku_approved = cache["haiku_approved"]
            haiku_cost = cache["haiku_cost"]
            candidates_count = checkpoint.get("pass1_candidates", 0)
            total_haiku_approved = len(haiku_approved)
            # Skip already-processed candidates from prior Pass 3 attempt
            already_done = cache.get("pass3_done_indices", [])
            if already_done:
                haiku_approved = [
                    c for c in haiku_approved
                    if c["candle_idx"] not in already_done
                ]
                logger.info(
                    "Skipping %d already-processed candidates, %d remaining",
                    len(already_done), len(haiku_approved),
                )
        else:
            # ── Full run: Pass 1 + 2 ──
            logger.info("Fetching daily candles for regime classification")
            candles_daily = provider.fetch_candles(
                "XAU/USD", "1day", start_date, end_date
            )

            regime_map = self._classify_regime(candles_daily)
            logger.info("Regime classification: %s", regime_map)

            # ── Pass 1: Structural scan (FREE) ──
            candidates = self.structural_scan(candles_1h, regime_map)
            logger.info("Pass 1 complete: %d structural candidates", len(candidates))

            if dry_run:
                return {
                    "pass1_candidates": len(candidates),
                    "dry_run": True,
                }

            # Apply regime cap across candidates (relaxed for backtest: 70%)
            candidates = self._apply_regime_cap(candidates, max_pct=0.70)
            candidates_count = len(candidates)

            # Save Pass 1 checkpoint
            self._save_checkpoint({
                "pass": 1,
                "pass1_candidates": candidates_count,
                "cost": 0.0,
            })

            # ── Pass 2: Haiku validation (~$0.001/call) ──
            haiku_approved = []
            haiku_cost = 0.0

            for cand in candidates:
                idx = cand["candle_idx"]
                window = candles_1h[max(0, idx - 40):idx + 1]
                stripped = self._strip_candle_dates(window)

                prompt = build_haiku_backtest_prompt(
                    stripped, cand["structural_elements"]
                )

                self._rate_limit_haiku()
                result = self._call_haiku(claude_key, prompt)

                call_cost = tracker.log_call(
                    "haiku", 1500, 200, "backtest_haiku_screen"
                )
                haiku_cost += call_cost

                if result and result.get("valid"):
                    cand["haiku_result"] = result
                    haiku_approved.append(cand)

            logger.info(
                "Pass 2 complete: %d Haiku-approved from %d candidates ($%.2f)",
                len(haiku_approved), len(candidates), haiku_cost,
            )

            # Save Pass 2 checkpoint + cache haiku_approved for resumption
            self._save_checkpoint({
                "pass": 2,
                "pass1_candidates": candidates_count,
                "pass2_haiku_approved": len(haiku_approved),
                "cost": haiku_cost,
            })
            self._save_pass2_cache(haiku_approved, haiku_cost)
            total_haiku_approved = len(haiku_approved)

        # ── Pass 3: Sonnet enrichment (~$0.076/call) ──
        ingested_rows = []
        total_ingested = 0  # Total across all flushes (not reset)
        first_flush = True  # Purge old backtest rows only on first flush
        sonnet_cost = 0.0
        regime_counts = {"trending": 0, "ranging": 0, "volatile": 0}
        pass3_done_indices = []  # Track processed candle indices for resumption

        for cand in haiku_approved:
            if sonnet_cost + haiku_cost >= budget_limit_usd:
                logger.info(
                    "Budget limit reached ($%.2f)", sonnet_cost + haiku_cost
                )
                break
            if len(ingested_rows) >= max_setups:
                break

            idx = cand["candle_idx"]
            window = candles_1h[max(0, idx - 200):idx + 1]
            stripped = self._strip_candle_dates(window)

            # Build prompt with Haiku hint
            haiku = cand["haiku_result"]
            elements = cand["structural_elements"]
            hint = (
                f"Pre-screening identified a potential "
                f"{haiku.get('direction', '?')} setup "
                f"near {haiku.get('entry_price', 0):.2f}. "
                f"Structural elements: "
                f"OBs={elements['ob_count']}, "
                f"FVGs={elements['fvg_count']}, "
                f"Sweep={'yes' if elements['sweep_detected'] else 'no'}, "
                f"Structure={elements['structure_score']:+.2f}. "
                f"Validate and provide full ICT analysis.\n\n"
            )

            # Align 4H candles to this candidate's timestamp (matches live scanner)
            cand_dt = candles_1h[idx].get("datetime", "")
            htf_window = self._find_4h_window(
                candles_4h, _4h_timestamps, cand_dt, count=20
            )
            htf_stripped = self._strip_candle_dates(htf_window) if htf_window else []

            # Compute intermarket context for this point in time
            _intermarket = None
            if dxy_candles_1h or us10y_candles_1h:
                try:
                    from ml.intermarket import compute_intermarket_context
                    _gold_window = candles_1h[max(0, idx - 30):idx + 1]
                    _dxy_window = self._find_intermarket_window(
                        dxy_candles_1h, cand_dt, count=30
                    )
                    _us10y_window = self._find_intermarket_window(
                        us10y_candles_1h, cand_dt, count=30
                    )
                    _intermarket = compute_intermarket_context(
                        _gold_window,
                        _dxy_window or None,
                        _us10y_window or None,
                        session=cand.get("killzone", "Off"),
                    )
                except Exception as e:
                    logger.debug("Intermarket context failed for idx=%d: %s", idx, e)

            prompt = build_enhanced_ict_prompt(
                stripped, htf_stripped, intermarket=_intermarket,
                htf_label="4H"
            )
            prompt = hint + prompt

            self._rate_limit()
            analysis = self._call_claude(claude_key, prompt)

            call_cost = tracker.log_call(
                "sonnet", 3000, 1500, "backtest_sonnet_enrich"
            )
            sonnet_cost += call_cost

            if analysis is None:
                logger.debug(
                    "Sonnet returned None for candidate idx=%d", cand["candle_idx"]
                )
                pass3_done_indices.append(idx)
                continue

            if isinstance(analysis, dict) and analysis.get("_credits_exhausted"):
                logger.error("Credits exhausted — saving progress and stopping")
                self._update_pass2_cache_progress(pass3_done_indices)
                break

            # Parse through bridge — require genuine Sonnet entry/SL
            # (Haiku fallback disabled: mechanical entries have different
            #  feature distributions and outcome profiles than live trades)
            parsed = bridge.parse_analysis(analysis, window)
            if (not parsed or not parsed.get("claude_entry_price")
                    or not parsed.get("claude_sl_price")):
                logger.debug(
                    "Sonnet parse fail idx=%d: no valid entry/SL, skipping",
                    cand["candle_idx"],
                )
                pass3_done_indices.append(idx)
                continue

            entry_price = parsed["claude_entry_price"]
            sl_price = parsed["claude_sl_price"]
            direction = parsed.get("claude_direction", "long")
            tp_prices = parsed.get("claude_tp_prices", [])

            if not entry_price or not sl_price:
                pass3_done_indices.append(idx)
                continue

            # Label outcome from future candles using Claude's TP levels
            forward = candles_1h[idx + 1:idx + 51]
            outcome = self._label_outcome(
                entry_price, sl_price, tp_prices, direction, forward
            )

            # MFE/MAE from create_trade_labels
            try:
                from ml.features import create_trade_labels
                labels = create_trade_labels(
                    candles_1h, idx, direction, cand["atr"]
                )
            except Exception:
                labels = {}

            # Extract features
            features = extract_features(analysis, window, "1h")

            # Entry noise augmentation
            noisy = self._add_entry_noise(
                entry_price, abs(entry_price - sl_price),
                direction, candles_1h, idx,
            )

            # Entry zone placement
            try:
                from ml.entry_placement import identify_entry_zone, compute_entry_position
                _zone = identify_entry_zone(entry_price, analysis, cand["atr"])
                if _zone:
                    _ez_pos = round(compute_entry_position(
                        entry_price, _zone["zone_high"], _zone["zone_low"], direction
                    ), 4)
                    _ez_type = _zone["zone_type"]
                    _ez_size = _zone["zone_size_atr"]
                else:
                    _ez_pos, _ez_type, _ez_size = 0.5, "none", 0
            except Exception:
                _ez_pos, _ez_type, _ez_size = 0.5, "none", 0

            # Encode killzone to match live scanner's killzone_encoded
            _kz_encode = {"London": 1, "NY_AM": 2, "NY_PM": 2, "Asian": 3}
            kz_encoded = _kz_encode.get(cand["killzone"], 0)

            # Build training row
            row = dict(features)
            row["outcome"] = outcome
            row["source"] = "backtest"
            row["source_weight"] = 0.7
            row["regime"] = cand["regime"]
            row["killzone_name"] = cand["killzone"]
            row["killzone_encoded"] = kz_encoded
            row["setup_id"] = f"bt2-{idx:05d}"
            row["entry_price"] = noisy.get("entry", entry_price)
            row["sl_price"] = sl_price
            row["direction"] = direction
            row["confluence_score"] = cand["score"]
            row["mfe_atr"] = round(labels.get("max_favorable_atr", 0), 4)
            row["mae_atr"] = round(labels.get("max_drawdown_atr", 0), 4)
            row["bars_held"] = labels.get("bars_held", 0)
            row["haiku_agreed"] = True
            row["entry_zone_position"] = _ez_pos
            row["entry_zone_type"] = _ez_type
            row["entry_zone_size_atr"] = _ez_size
            row["entry_source"] = "sonnet"

            ingested_rows.append(row)
            pass3_done_indices.append(idx)
            regime = cand.get("regime", "ranging")
            regime_counts[regime] = regime_counts.get(regime, 0) + 1

            # Update Pass 2 cache with done indices every 5 calls (for resumption)
            if len(pass3_done_indices) % 5 == 0:
                self._update_pass2_cache_progress(pass3_done_indices)

            # Progress report every 10 setups
            if len(ingested_rows) % 10 == 0:
                logger.info(
                    "Backtest progress: %d/%d setups | $%.2f spent | "
                    "regimes: %s",
                    len(ingested_rows), max_setups,
                    haiku_cost + sonnet_cost, regime_counts,
                )

        # Single flush: all rows at once (avoids incremental save bugs)
        total_ingested = len(ingested_rows)
        if ingested_rows:
            self._ingest_rows(dataset_mgr, ingested_rows,
                              purge_old=True)

        # Final checkpoint
        total_cost = round(haiku_cost + sonnet_cost, 4)
        self._save_checkpoint({
            "pass": 3,
            "pass1_candidates": candidates_count,
            "pass2_haiku_approved": total_haiku_approved,
            "pass3_sonnet_analysed": total_ingested,
            "cost": total_cost,
            "regime_counts": regime_counts,
        })

        # Clean up Pass 2 cache on successful completion
        if os.path.exists(self._pass2_cache_path):
            try:
                os.remove(self._pass2_cache_path)
            except OSError:
                pass

        tracker.flush()

        return {
            "pass1_candidates": candidates_count,
            "pass2_haiku_approved": total_haiku_approved,
            "pass3_sonnet_analysed": total_ingested,
            "setups_found": total_ingested,
            "total_cost": total_cost,
            "haiku_cost": round(haiku_cost, 4),
            "sonnet_cost": round(sonnet_cost, 4),
            "cost_per_setup": round(
                total_cost / max(total_ingested, 1), 4
            ),
            "regime_counts": regime_counts,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_4h_window(candles_4h: list[dict], timestamps_4h: list[str],
                        target_dt: str, count: int = 20) -> list[dict]:
        """Find the 4H candle window ending at or before the target 1H datetime.

        Uses bisect on pre-sorted timestamps for O(log n) lookup.

        Args:
            candles_4h: Full 4H candle list (chronological).
            timestamps_4h: Pre-extracted datetime strings (YYYY-MM-DD HH:MM:SS).
            target_dt: Target datetime string from the 1H candle.
            count: How many 4H candles to include in the window.

        Returns:
            List of up to `count` 4H candles ending at or before target_dt.
        """
        if not candles_4h or not timestamps_4h:
            return []

        target_norm = target_dt[:19].replace("T", " ")
        # bisect_right gives insertion point — index before it is the last <= target
        pos = bisect.bisect_right(timestamps_4h, target_norm)
        if pos == 0:
            return []

        start = max(0, pos - count)
        return candles_4h[start:pos]

    @staticmethod
    def _find_intermarket_window(candles: list[dict], target_dt: str,
                                 count: int = 30) -> list[dict]:
        """Find intermarket candle window ending at or before target datetime.

        Simpler linear scan from the end since we call this per-candidate and
        intermarket candles are smaller datasets.
        """
        if not candles:
            return []

        target_norm = target_dt[:19].replace("T", " ")
        # Find last candle at or before target
        end_idx = 0
        for i, c in enumerate(candles):
            c_dt = c.get("datetime", "")[:19].replace("T", " ")
            if c_dt <= target_norm:
                end_idx = i
            else:
                break

        start = max(0, end_idx - count + 1)
        return candles[start:end_idx + 1]

    @staticmethod
    def _strip_candle_dates(candles: list[dict]) -> list[dict]:
        """Remove datetime field, replace with sequential index. Keep OHLCV only."""
        stripped = []
        for i, c in enumerate(candles):
            stripped.append({
                "index": i,
                "open": c["open"],
                "high": c["high"],
                "low": c["low"],
                "close": c["close"],
                "volume": c.get("volume", 0),
            })
        return stripped

    @staticmethod
    def _classify_regime(daily_candles: list[dict]) -> dict:
        """Classify each month's volatility regime using ATR(14) on daily candles.

        Returns dict like {"2025-09": "trending", "2025-10": "ranging", ...}
        """
        from ml.features import compute_atr

        if len(daily_candles) < 20:
            return {}

        # Group candles by month
        months: dict[str, list[dict]] = {}
        for c in daily_candles:
            month_key = c.get("datetime", "")[:7]
            if month_key:
                months.setdefault(month_key, []).append(c)

        result = {}
        for month_key, month_candles in sorted(months.items()):
            if len(month_candles) < 10:
                result[month_key] = "ranging"
                continue

            atr = compute_atr(month_candles, min(14, len(month_candles) - 1))

            # Directional metric: net change / total range
            opens = [c["open"] for c in month_candles]
            closes = [c["close"] for c in month_candles]
            net_change = abs(closes[-1] - opens[0])
            total_range = sum(c["high"] - c["low"] for c in month_candles)

            if total_range == 0:
                result[month_key] = "ranging"
                continue

            directional_ratio = net_change / total_range

            # ATR relative to average price
            avg_price = sum(c["close"] for c in month_candles) / len(month_candles)
            atr_pct = (atr / avg_price * 100) if avg_price > 0 else 0

            if directional_ratio > 0.3 and atr_pct > 0.8:
                result[month_key] = "trending"
            elif atr_pct > 1.2:
                result[month_key] = "volatile"
            else:
                result[month_key] = "ranging"

        return result

    @staticmethod
    def _label_outcome(
        entry_price: float,
        sl_price: float,
        tp_prices: list[float] | float | None,
        direction: str,
        forward_candles: list[dict],
    ) -> str:
        """Look forward through candles to determine SL/TP hits.

        Uses Claude's actual TP levels to match live scanner categories.
        Falls back to SL-multiple levels when Claude doesn't provide TPs.

        Args:
            tp_prices: List of [tp1, tp2, tp3] prices, a single tp1 float,
                       or None. Missing levels filled with SL-distance multiples.

        Returns one of: "stopped_out", "tp1", "tp2", "tp3", "tp3_hit".
        """
        if not forward_candles or not entry_price or not sl_price:
            return "stopped_out"

        sl_distance = abs(entry_price - sl_price)

        # Normalize tp_prices to a list of up to 3 levels
        if tp_prices is None:
            tp_list = []
        elif isinstance(tp_prices, (int, float)):
            tp_list = [tp_prices] if tp_prices else []
        else:
            tp_list = [p for p in tp_prices if p]

        # Fill missing TP levels with SL-distance multiples (2R, 3R, 4R)
        if direction == "long":
            sl_level = entry_price - sl_distance
            defaults = [
                entry_price + sl_distance * 2,
                entry_price + sl_distance * 3,
                entry_price + sl_distance * 4,
            ]
        else:
            sl_level = entry_price + sl_distance
            defaults = [
                entry_price - sl_distance * 2,
                entry_price - sl_distance * 3,
                entry_price - sl_distance * 4,
            ]

        # Merge: use Claude's prices where available, defaults for the rest
        levels = []
        for i in range(3):
            if i < len(tp_list) and tp_list[i]:
                levels.append(tp_list[i])
            else:
                levels.append(defaults[i])

        tp1_level, tp2_level, tp3_level = levels

        # Outcome names by level index
        tp_names = ["tp1", "tp2", "tp3"]
        best_hit = None

        for candle in forward_candles:
            high = candle["high"]
            low = candle["low"]

            if direction == "long":
                if low <= sl_level:
                    return best_hit or "stopped_out"
                if high >= tp3_level:
                    if best_hit in ("tp3",):
                        return "tp3_hit"
                    best_hit = "tp3"
                elif high >= tp2_level:
                    if best_hit not in ("tp3",):
                        best_hit = "tp2"
                elif high >= tp1_level and best_hit is None:
                    best_hit = "tp1"
            else:
                if high >= sl_level:
                    return best_hit or "stopped_out"
                if low <= tp3_level:
                    if best_hit in ("tp3",):
                        return "tp3_hit"
                    best_hit = "tp3"
                elif low <= tp2_level:
                    if best_hit not in ("tp3",):
                        best_hit = "tp2"
                elif low <= tp1_level and best_hit is None:
                    best_hit = "tp1"

        return best_hit or "stopped_out"

    @staticmethod
    def _add_entry_noise(
        entry_price: float,
        sl_distance: float,
        direction: str,
        candles: list[dict],
        idx: int,
    ) -> dict:
        """Jitter entry by +/- 1-3 candles, recalculate entry from jittered candle close.

        Keeps SL/TP distances the same relative to new entry.
        """
        rng = random.Random(idx)  # Deterministic per candle
        jitter = rng.randint(-3, 3)
        jittered_idx = max(0, min(len(candles) - 1, idx + jitter))
        new_entry = candles[jittered_idx]["close"]

        # Preserve SL/TP distances
        if direction == "long":
            new_sl = new_entry - sl_distance
        else:
            new_sl = new_entry + sl_distance

        return {
            "entry": new_entry,
            "sl": new_sl,
            "jitter_candles": jitter,
            "original_idx": idx,
            "jittered_idx": jittered_idx,
        }

    @staticmethod
    def _get_killzone(dt_str: str) -> str:
        """Determine killzone name from datetime string."""
        try:
            time_part = (
                dt_str.split("T")[-1] if "T" in dt_str
                else dt_str.split(" ")[-1]
            )
            hour = int(time_part.split(":")[0])
        except (ValueError, IndexError):
            return "Off"

        for kz_name, (start, end) in _KILLZONES.items():
            if start <= hour < end:
                return kz_name

        if 0 <= hour < 7:
            return "Asian"
        return "Off"

    # ------------------------------------------------------------------
    # API calls
    # ------------------------------------------------------------------

    def _rate_limit(self):
        """Enforce 10 calls/min rate limit for Sonnet."""
        now = time.time()
        self._call_timestamps = [
            t for t in self._call_timestamps if now - t < _RATE_LIMIT_WINDOW
        ]
        if len(self._call_timestamps) >= _RATE_LIMIT_CALLS:
            sleep_for = (
                _RATE_LIMIT_WINDOW
                - (now - self._call_timestamps[0])
                + 0.5
            )
            if sleep_for > 0:
                logger.info("Backtest rate limit: waiting %.1fs", sleep_for)
                time.sleep(sleep_for)
        self._call_timestamps.append(time.time())

    def _rate_limit_haiku(self):
        """Enforce 50 calls/min rate limit for Haiku."""
        now = time.time()
        self._haiku_call_timestamps = [
            t for t in self._haiku_call_timestamps
            if now - t < _HAIKU_RATE_LIMIT_WINDOW
        ]
        if len(self._haiku_call_timestamps) >= _HAIKU_RATE_LIMIT_CALLS:
            sleep_for = (
                _HAIKU_RATE_LIMIT_WINDOW
                - (now - self._haiku_call_timestamps[0])
                + 0.5
            )
            if sleep_for > 0:
                logger.info("Haiku rate limit: waiting %.1fs", sleep_for)
                time.sleep(sleep_for)
        self._haiku_call_timestamps.append(time.time())

    @staticmethod
    def _call_haiku(api_key: str, prompt: str) -> dict | None:
        """Call Haiku for quick setup validation. Returns parsed JSON or None."""
        if not api_key:
            return None

        for attempt in range(2):
            try:
                resp = httpx.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": "claude-haiku-4-5-20251001",
                        "max_tokens": 300,
                        "temperature": 0,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                    timeout=30,
                )

                if resp.status_code == 400 and "credit balance" in resp.text:
                    logger.error("API credits exhausted — stopping")
                    return {"_credits_exhausted": True}
                if resp.status_code in (429, 529):
                    time.sleep(2 ** attempt)
                    continue
                if resp.status_code != 200:
                    continue

                data = resp.json()
                text = data.get("content", [{}])[0].get("text", "")
                clean = text.replace("```json", "").replace("```", "").strip()
                json_start = clean.find("{")
                json_end = clean.rfind("}")
                if json_start >= 0 and json_end > json_start:
                    return json.loads(clean[json_start:json_end + 1])

            except Exception:
                continue

        return None

    @staticmethod
    def _call_claude(api_key: str, prompt: str) -> dict | None:
        """Call Claude Sonnet for ICT analysis. Returns parsed JSON or None."""
        from ml.scanner import ICT_SYSTEM_MESSAGE

        if not api_key:
            logger.error("No ANTHROPIC_API_KEY set")
            return None

        for attempt in range(3):
            try:
                resp = httpx.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": "claude-sonnet-4-20250514",
                        "max_tokens": 3000,
                        "temperature": 0,
                        "system": ICT_SYSTEM_MESSAGE,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                    timeout=90,
                )

                if resp.status_code in (429, 529):
                    wait = (2 ** attempt) * 2
                    logger.warning(
                        "Backtest Claude rate limited (%d), waiting %ds",
                        resp.status_code, wait,
                    )
                    time.sleep(wait)
                    continue

                if resp.status_code == 400 and "credit balance" in resp.text:
                    logger.error(
                        "API credits exhausted — stopping backtest"
                    )
                    return {"_credits_exhausted": True}

                if resp.status_code != 200:
                    logger.error(
                        "Backtest Claude API error %d: %s",
                        resp.status_code, resp.text[:200],
                    )
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

                return json.loads(clean)

            except json.JSONDecodeError as e:
                logger.warning(
                    "Sonnet JSON parse error (attempt %d): %s | text[:200]=%s",
                    attempt, e, clean[:200] if 'clean' in dir() else "?"
                )
                if attempt < 2:
                    time.sleep(2)
                    continue
                return None
            except httpx.TimeoutException:
                if attempt < 2:
                    time.sleep(2 ** attempt * 2)
                    continue
                return None
            except Exception as e:
                logger.error("Backtest Claude call failed: %s", e)
                return None

        return None

    # ------------------------------------------------------------------
    # Dataset ingestion
    # ------------------------------------------------------------------

    @staticmethod
    def _ingest_rows(dataset_mgr, rows: list[dict], purge_old: bool = False):
        """Ingest backtest rows into dataset with source='backtest'.

        Args:
            purge_old: If True, remove ALL existing backtest rows first.
                       Only set True on the FIRST flush of a fresh run.
                       Incremental flushes append without purging.
        """
        import pandas as pd

        if not rows:
            return

        new_df = pd.DataFrame(rows)
        if dataset_mgr._df.empty:
            dataset_mgr._df = new_df
        else:
            if purge_old and "source" in dataset_mgr._df.columns:
                dataset_mgr._df = dataset_mgr._df[
                    dataset_mgr._df["source"] != "backtest"
                ]
            dataset_mgr._df = pd.concat(
                [dataset_mgr._df, new_df], ignore_index=True
            )
        dataset_mgr._save()

    # ------------------------------------------------------------------
    # Checkpoint save/resume
    # ------------------------------------------------------------------

    def _load_checkpoint(self) -> dict:
        """Load checkpoint from disk or return defaults."""
        defaults = {
            "last_candle_idx": 200,
            "setups_found": 0,
            "cost": 0.0,
            "regime_counts": {"trending": 0, "ranging": 0, "volatile": 0},
        }
        if os.path.exists(self._checkpoint_path):
            try:
                with open(self._checkpoint_path) as f:
                    data = json.load(f)
                # Support both old format (last_candle_idx) and new (pass-level)
                if data and (data.get("last_candle_idx", 0) > 0
                             or data.get("pass", 0) > 0):
                    return data
            except Exception:
                pass
        return defaults

    def _save_checkpoint(self, state: dict):
        """Save checkpoint with pass-level granularity."""
        state["timestamp"] = datetime.utcnow().isoformat()
        os.makedirs(os.path.dirname(self._checkpoint_path), exist_ok=True)
        with open(self._checkpoint_path, "w") as f:
            json.dump(state, f, indent=2)

    def _save_pass2_cache(self, haiku_approved: list, haiku_cost: float):
        """Save Pass 2 results to disk so Pass 3 can resume after timeout."""
        cache = {
            "haiku_approved": haiku_approved,
            "haiku_cost": haiku_cost,
            "pass3_done_indices": [],
            "timestamp": datetime.utcnow().isoformat(),
        }
        os.makedirs(os.path.dirname(self._pass2_cache_path), exist_ok=True)
        with open(self._pass2_cache_path, "w") as f:
            json.dump(cache, f)
        logger.info(
            "Pass 2 cache saved: %d candidates at %s",
            len(haiku_approved), self._pass2_cache_path,
        )

    def _update_pass2_cache_progress(self, done_indices: list):
        """Update the Pass 2 cache with indices completed in Pass 3."""
        if not os.path.exists(self._pass2_cache_path):
            return
        try:
            with open(self._pass2_cache_path) as f:
                cache = json.load(f)
            cache["pass3_done_indices"] = done_indices
            with open(self._pass2_cache_path, "w") as f:
                json.dump(cache, f)
        except Exception as e:
            logger.warning("Failed to update Pass 2 cache progress: %s", e)

    # ------------------------------------------------------------------
    # Fidelity check
    # ------------------------------------------------------------------

    def run_fidelity_check(self) -> dict:
        """Compare backtest vs live rows in training dataset.

        Runs 5 statistical checks:
        1. Chi-squared test on outcome distribution
        2. KS test on top 10 features
        3. Setup characteristic comparison (SL dist, TP dist, RR)
        4. Win rate sanity check (flag if backtest > live + 15pp)
        5. Killzone distribution check

        If 2+ checks fail, reduce backtest source_weight to 0.4.
        If 4+ checks fail, reduce to 0.2.

        Returns dict with check results and adjusted weight.
        """
        from scipy.stats import chi2_contingency, ks_2samp
        from ml.dataset import TrainingDatasetManager
        import numpy as np
        import pandas as pd

        dataset_mgr = TrainingDatasetManager(config=self.cfg)
        df = dataset_mgr._df

        if df.empty or "source" not in df.columns:
            return {"error": "no_data", "checks_failed": 0}

        bt_df = df[df["source"] == "backtest"]
        live_df = df[df["source"] == "live"]

        if len(bt_df) < 10 or len(live_df) < 10:
            return {
                "error": "insufficient_data",
                "bt_count": len(bt_df),
                "live_count": len(live_df),
            }

        checks_failed = 0
        results = {}

        # 1. Chi-squared on outcome distribution
        try:
            outcomes = ["stopped_out", "tp1", "runner"]
            bt_counts = [len(bt_df[bt_df["outcome"] == o]) for o in outcomes]
            live_counts = [len(live_df[live_df["outcome"] == o]) for o in outcomes]

            # Ensure no zero counts (add 1 smoothing)
            bt_counts = [max(c, 1) for c in bt_counts]
            live_counts = [max(c, 1) for c in live_counts]

            contingency = np.array([bt_counts, live_counts])
            chi2, p_value, _, _ = chi2_contingency(contingency)
            outcome_pass = p_value > 0.05
            if not outcome_pass:
                checks_failed += 1
            results["outcome_chi2"] = {
                "chi2": round(chi2, 4),
                "p_value": round(p_value, 4),
                "passed": outcome_pass,
            }
        except Exception as e:
            results["outcome_chi2"] = {"error": str(e), "passed": True}

        # 2. KS test on representative features (balanced across categories)
        # Note: structural sub-features (ob_bullish_count, fvg_unfilled_count,
        # etc.) are excluded because the backtest pre-selects candidates with
        # high structural confluence — these will always diverge from live
        # data that includes low-structure setups. Instead, test top-level
        # structural counts + features from all other categories.
        _fidelity_features = [
            # Structural (top-level counts + alignment only)
            "ob_count", "ob_alignment", "fvg_count",
            # Price action (computed from candles — same in both pipelines)
            "atr_14", "trend_strength", "recent_volatility_ratio",
            # Trade setup
            "sl_distance_atr", "risk_reward_tp1", "bias_direction_match",
            # Context
            "num_confluences",
        ]
        top_features = [
            c for c in _fidelity_features
            if c in bt_df.columns and c in live_df.columns
        ]
        ks_results = {}
        ks_fails = 0
        for feat in top_features:
            try:
                bt_vals = bt_df[feat].dropna().astype(float)
                live_vals = live_df[feat].dropna().astype(float)
                if len(bt_vals) < 5 or len(live_vals) < 5:
                    continue
                stat, p = ks_2samp(bt_vals, live_vals)
                passed = p > 0.05
                if not passed:
                    ks_fails += 1
                ks_results[feat] = {
                    "statistic": round(stat, 4),
                    "p_value": round(p, 4),
                    "passed": passed,
                }
            except Exception:
                continue

        if ks_fails > len(top_features) * 0.5:
            checks_failed += 1
        results["feature_ks"] = {
            "tests": ks_results,
            "fails": ks_fails,
            "passed": ks_fails <= len(top_features) * 0.5,
        }

        # 3. Setup characteristics comparison (SL dist, RR)
        try:
            bt_sl = (
                bt_df["sl_distance_atr"].dropna().astype(float)
                if "sl_distance_atr" in bt_df.columns
                else pd.Series(dtype=float)
            )
            live_sl = (
                live_df["sl_distance_atr"].dropna().astype(float)
                if "sl_distance_atr" in live_df.columns
                else pd.Series(dtype=float)
            )

            if len(bt_sl) > 5 and len(live_sl) > 5:
                sl_diff = abs(bt_sl.mean() - live_sl.mean())
                sl_pass = (
                    sl_diff < live_sl.std() * 2
                    if live_sl.std() > 0
                    else True
                )
            else:
                sl_diff = 0
                sl_pass = True

            if not sl_pass:
                checks_failed += 1
            results["setup_characteristics"] = {
                "sl_mean_diff": round(sl_diff, 4),
                "passed": sl_pass,
            }
        except Exception as e:
            results["setup_characteristics"] = {
                "error": str(e), "passed": True,
            }

        # 4. Win rate sanity check
        try:
            _win_outcomes = ["tp1", "tp2", "tp3", "tp3_hit", "runner"]
            bt_wins = len(bt_df[bt_df["outcome"].isin(_win_outcomes)])
            bt_wr = bt_wins / len(bt_df) if len(bt_df) > 0 else 0
            live_wins = len(live_df[live_df["outcome"].isin(_win_outcomes)])
            live_wr = live_wins / len(live_df) if len(live_df) > 0 else 0

            # Flag if backtest WR > live + 15 percentage points
            wr_pass = bt_wr <= live_wr + 0.15
            if not wr_pass:
                checks_failed += 1
            results["win_rate"] = {
                "backtest_wr": round(bt_wr, 4),
                "live_wr": round(live_wr, 4),
                "diff": round(bt_wr - live_wr, 4),
                "passed": wr_pass,
            }
        except Exception as e:
            results["win_rate"] = {"error": str(e), "passed": True}

        # 5. Killzone distribution check
        try:
            if ("killzone_encoded" in bt_df.columns
                    and "killzone_encoded" in live_df.columns):
                bt_kz = bt_df["killzone_encoded"].value_counts(normalize=True)
                live_kz = live_df["killzone_encoded"].value_counts(
                    normalize=True
                )
                all_kz = set(bt_kz.index) | set(live_kz.index)
                max_diff = (
                    max(abs(bt_kz.get(k, 0) - live_kz.get(k, 0))
                        for k in all_kz)
                    if all_kz else 0
                )
                kz_pass = max_diff < 0.3
            else:
                kz_pass = True
                max_diff = 0

            if not kz_pass:
                checks_failed += 1
            results["killzone_distribution"] = {
                "max_diff": round(max_diff, 4),
                "passed": kz_pass,
            }
        except Exception as e:
            results["killzone_distribution"] = {
                "error": str(e), "passed": True,
            }

        # Determine adjusted weight
        if checks_failed >= 4:
            adjusted_weight = 0.2
        elif checks_failed >= 2:
            adjusted_weight = 0.4
        else:
            adjusted_weight = 0.7

        # Update backtest weights in dataset if needed
        if adjusted_weight != 0.7 and not df.empty:
            mask = df["source"] == "backtest"
            if "source_weight" in df.columns:
                df.loc[mask, "source_weight"] = adjusted_weight
                dataset_mgr._df = df
                dataset_mgr._save()

        meta = {
            "timestamp": datetime.utcnow().isoformat(),
            "checks_failed": int(checks_failed),
            "adjusted_weight": float(adjusted_weight),
            "results": self._sanitize_for_json(results),
            "bt_count": int(len(bt_df)),
            "live_count": int(len(live_df)),
        }

        # Store to backtest_meta.json
        os.makedirs(os.path.dirname(self._meta_path), exist_ok=True)
        with open(self._meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        return meta

    @staticmethod
    def _sanitize_for_json(obj):
        """Recursively convert numpy types to native Python for JSON serialization."""
        import numpy as np

        if isinstance(obj, dict):
            return {
                k: BacktestGenerator._sanitize_for_json(v)
                for k, v in obj.items()
            }
        elif isinstance(obj, list):
            return [BacktestGenerator._sanitize_for_json(v) for v in obj]
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        return obj
