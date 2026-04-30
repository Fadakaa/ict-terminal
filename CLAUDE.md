# CLAUDE.md — ICT Terminal ML Backend

## Architecture (v2 — Claude-Primary)

The mechanical ICT detector was retired after diagnostic testing showed it could not produce a profitable edge on XAU/USD (best result: 30.8% WR, 0.68 CostPF, Grade D across 12 parameter configurations).

Current architecture:
- **Claude** is the primary analyst — identifies ICT setups via narrative understanding
  (accumulation/manipulation/distribution, premium/discount arrays, Power of 3)
- **ML stack** is the calibration layer — adjusts Claude's SL/TP using:
  - V1 session statistics (baseline drawdown distributions from 1151 simulated trades)
  - Volatility calibrator (session-aware ATR + regime scaling)
  - Bayesian sequential updater (learns from live trade outcomes)
  - AutoGluon quantile regressors (once trained from accumulated data)
  - Historical pattern matching (finds similar past setups by volatility profile)
- **Consensus** takes the widest SL and median TPs across all layers
- **Claude accuracy tracker** measures how much value the ML calibration adds

Enhanced prompt sends both 1H and 4H candles so Claude does its own MTF analysis.

Flow: Candles → Claude (identifies setup) → /calibrate (adjusts levels) → User trades → /trade/complete (logs outcome) → Bayesian + dataset updated → periodic retrain

## Module Structure

| Module | Purpose |
|---|---|
| `seed.py` | V1 Data Harvester — seeds ML models with baseline distributions |
| `claude_bridge.py` | Parses Claude's ICT analysis JSON into structured ML input |
| `calibrate.py` | 6-layer SL/TP calibration engine (volatility, V1 session, Bayesian, AutoGluon, historical, consensus) |
| `prompts.py` | Enhanced multi-timeframe ICT prompt builder |
| `server.py` | FastAPI endpoints for all ML operations |
| `bayesian.py` | Beta distribution win rate + drawdown belief updater |
| `volatility.py` | EWMA volatility + session scaling + regime detection |
| `dataset.py` | Training dataset manager (CSV-based, weighted blending) |
| `haiku_fn_tracker.py` | Priority 5: Haiku false negative detection — logs rejections, resolves against forward price, learns segment adjustments |
| `cost_per_winner.py` | Priority 8: Cost-per-winner optimization — correlates API spend with trade outcomes by timeframe×killzone, recommends scan budget allocation |
| `wfo.py` | Walk-Forward Optimization engine (V1 detector, kept for seeding) |
| `features.py` | 52-feature extraction from raw candles + Claude analysis (HTF context, narrative state, intermarket, regime, entry zone) |
| `feature_schema.py` | Single source of truth for all 52 ML feature column names |
| `narrative_state.py` | Narrative State Engine — persistent per-timeframe thesis tracking across scan cycles with invalidation detection, prediction scoring, and anti-anchoring safeguards |
| `recent_context.py` | Context-Aware Scanner — builds recent context (resolutions, consumed zones, swept liquidity, active setups) for prompt enrichment |
| `backfill_features.py` | Retroactive feature enrichment — re-extracts 52-col features from stored analysis/calibration JSON, backfills regime + intermarket from OANDA. Also exposes `backfill_calendar_features()` for the calendar day-one bootstrap. |
| `calendar.py` | Forex calendar integration — FF XML source, `CalendarStore` (cache + query API + archive), `HistoricalCalendarView` (read-only history adapter), `ProximityStatus` classification (clear / caution / imminent / post_event). Single live source: ForexFactory. Hourly refresh by scheduler. Used by prompt builder, scanner metadata, notification warnings, and ML feature extraction (18 calendar columns). |
| `calendar_backfill.py` | One-shot day-one historical backfill via `market-calendar-tool`. NOT used during live ops — invoked manually once to import 6 months of FF history into `forex_calendar_history`, re-extract calendar features for stored setups, and trigger a retrain. See "Forex Calendar Integration" section below for the runbook. |

## Run Commands

```bash
source ~/dealfinder/bin/activate
cd ~/dealfinder/ict-terminal

# Seed ML models with V1 data (run once)
python -m ml.seed --td-key KEY --candles 2000

# Run tests
python -m pytest ml/tests/ -v

# Start server
python -m uvicorn ml.server:app --reload --port 8000
```

## Key Endpoints (v2)

| Endpoint | Method | Purpose |
|---|---|---|
| `/calibrate` | POST | Calibrate Claude's analysis with ML layers |
| `/trade/complete` | POST | Log completed trade, update all models |
| `/claude/accuracy` | GET | Claude accuracy tracker |
| `/calibration/value` | GET | How much value ML calibration adds |
| `/prompt/enhanced` | GET | Enhanced multi-timeframe prompt template |
| `/seed/stats` | GET | V1 session statistics |
| `/seed/run` | POST | Seed ML models with V1 data |
| `/retrain` | POST | Retrain AutoGluon from current dataset |
| `/haiku/fn/report` | GET | Full Haiku false negative analysis by segment |
| `/haiku/fn/stats` | GET | Quick FN stats for dashboard |
| `/haiku/fn/adjustments` | GET | Current screening adjustments from FN data |
| `/haiku/fn/recent` | GET | Recent resolved Haiku rejections |
| `/cost/per-winner` | GET | Cost-per-winner stats by timeframe×killzone segment |
| `/cost/per-winner/ranking` | GET | Segments ranked by cost-effectiveness (best ROI first) |
| `/cost/per-winner/recommendations` | GET | Scan frequency recommendations per segment |
| `/cost/per-winner/recompute` | POST | Recompute cost-per-winner from resolved setups |
| `/narrative/thesis/current` | GET | Current active thesis for a timeframe |
| `/narrative/thesis/history` | GET | Recent thesis history for a timeframe |
| `/narrative/thesis/accuracy` | GET | Prediction accuracy + thesis stability metrics |
| `/lifecycle/recent` | GET | Recent notification lifecycle events across all theses |
| `/lifecycle/thesis/{id}` | GET | Full lifecycle journey for a specific thesis |
| `/context/recent` | GET | Recent context (resolutions, consumed zones, swept liquidity) |
| `/calendar/upcoming` | GET | Upcoming USD high-impact events (next N hours) |
| `/calendar/recent` | GET | USD high-impact events in the last N hours (with `actual` results) |
| `/calendar/proximity` | GET | Current `ProximityStatus` snapshot (state + warning + next/last events) |
| `/calendar/refresh` | POST | Force-refresh the FF XML cache. Returns `{rate_limited: true}` on 429. |
| `/calendar/stats` | GET | Per-category event counts over the last `days` window (default 30). |

## Testing

```bash
python -m pytest ml/tests/ -v              # Full suite (~1280 tests)
python -m pytest ml/tests/test_seed.py     # Seed harvester tests
python -m pytest ml/tests/test_claude_bridge.py  # Bridge tests
python -m pytest ml/tests/test_calibrate.py      # Calibrator tests
python -m pytest ml/tests/test_prompts.py        # Prompt tests
python -m pytest ml/tests/test_calendar.py       # Calendar source/store/proximity tests
python -m pytest ml/tests/test_calendar_backfill.py  # Historical backfill tests
```

`@pytest.mark.integration` tests (e.g. `test_backfill_matches_live_xml_for_overlap_week`) are skipped by default — they hit the live FF feed and require `market-calendar-tool`. Run with `pytest -m integration` when needed.

## Completed Setup Actions

- **Bayesian state reset** (2026-03-25): Reset to V1 priors (alpha=8.92, beta=11.08, 44.6% WR). Scanner trades no longer update Bayesian beliefs (`source="scanner"` skips the update in `claude_bridge.py`). Only live trades update beliefs.
- **OANDA API token regenerated** (2026-03-25): New token generated and updated in `.env`.
- **Forex calendar day-one backfill** (2026-04-30): 6 months of FF historical events imported into `forex_calendar_history` (45 USD high-impact events across NFP/FOMC/CPI/GDP/ISM/etc.); calendar features re-extracted onto 534 stored setups. Live ops use the FF XML feed; the historical import is one-shot only (see "Forex Calendar Integration" runbook below).

## Data Provider

All market data fetched via OANDA v20 API (free with live account). Twelve Data has been fully removed.

| Instrument | OANDA Symbol | Notes |
|---|---|---|
| XAU/USD | `XAU_USD` | Direct |
| DXY | `EUR_USD` | Inverted proxy: `1/EUR_USD * 104` |
| US10Y | `USB10Y_USD` | Direct |

OANDA pagination uses `count` parameter (max 4999 per request), not `from+to`.

## Scanner Configuration

The live scanner runs automatically when uvicorn starts (Mon-Fri only). It scans 4 timeframes:

| Timeframe | Check Interval | HTF Context | Candles |
|---|---|---|---|
| 15min | Every 15 min | 1H | 40 |
| 1H | Every 60 min | 4H | 60 |
| 4H | Every 240 min | 1D | 40 |
| 1D | Every 1440 min | — | — |

Additional scheduled jobs: prospect generation 15min before each killzone, trigger monitoring every 90s, auto-retrain every 6h.

## Bayesian Update Rules

- **Live trades and scanner auto-resolves update beliefs.** Both `source="live"` and `source="scanner"` update the Beta distribution. Only `source="backtest"` or `source="wfo"` are excluded (synthetic data with inflated win rates).
- **Drift detection** uses percentage-point difference in win rate means (not the old SD-normalised metric which produced false alarms). Thresholds: 10pp = significant, 20pp = critical.
- **Reset endpoint**: `POST /bayesian/reset` resets to V1 seed priors (alpha=8.92, beta=11.08, 44.6% WR).

## Frontend

Vite + React SPA served from `dist/`. Express proxy on port 3001 forwards `/api/ml/*` to `localhost:8000`.

Build: `npm run build` (must run on Mac, not in sandbox — architecture mismatch).

Intelligence Panel tabs: ML, BAYES, SESSIONS, ACCURACY, P&L, PROP, LOG.

The ML tab displays: classifier status, AutoGluon model info, dataset composition, backtest fidelity, calibration layer performance, Bayesian drift, API cost tracking, active prospects, **forex calendar (state pill + next 3 events + manual refresh button)**, and scanner status.

## Forex Calendar Integration

Single live source: **ForexFactory weekly XML feed** (`https://nfs.faireconomy.media/ff_calendar_thisweek.xml`). Times are already UTC — no ET→UTC conversion. Encoding is `windows-1252`. Date format is US `MM-DD-YYYY`.

The system applies calendar awareness across four layers, all driven by `ml/calendar.py`:

| Layer | Where | Behavior |
|---|---|---|
| **1. Prompt context** | `prompts.py` (`_build_calendar_section`) | Adds an `ECONOMIC CALENDAR` block listing upcoming/recent USD high-impact events with ICT framing ("manipulation catalyst") so Claude weighs the manipulation interpretation more heavily inside caution/imminent windows |
| **2. Setup metadata + warnings** | `scanner.py` (`attach_calendar_proximity`), `notifications.py` (`build_notification_message`) | Attaches `calendar_proximity` to every stored setup. Notifications during caution/imminent windows render a `⚠ CAUTION/WARNING` line with the event title + minutes-to-next. **Warnings only — never suppresses or downgrades a setup.** |
| **3. Grade adjustments** | (deferred) | Static grade-downgrade rules NOT implemented. Will be added later from learned weights in `setup_profiles.py` if data shows them adding value. |
| **4. ML features** | `features.py` (`_extract_calendar_features`), `feature_schema.py` | 18 new columns: 3 magnitude (`mins_to_next_high_impact`, `mins_since_last_high_impact`, `news_density_24h`) + 4 proximity one-hot (`calendar_proximity_clear/post_event/caution/imminent`) + 11 category one-hot (`event_is_nfp/cpi/ppi/fomc/fed_speak/gdp/ism/retail_sales/unemployment/jolts/other_high`). FEATURE_COLUMNS goes 59 → 77. |

### Proximity classification

`CalendarStore.proximity(ts)` returns one of four states with this precedence:

1. **imminent** — next event ≤ 30 min away
2. **caution** — next event ≤ 90 min away
3. **post_event** — last event ≤ 90 min ago (and no next event in caution/imminent window)
4. **clear** — otherwise

Imminent eclipses post_event because upcoming manipulation risk is more actionable than residual settlement noise.

### Database tables

| Table | Source field | Purpose |
|---|---|---|
| `forex_calendar` | always `ff_xml` | Current-window cache. Upserted by `CalendarStore.refresh()`. |
| `forex_calendar_history` | `ff_xml` (live snapshots) **or** `ff_historical` (one-shot day-one backfill) | Append-only archive. PK is `(event_id, archived_at)`. Used by `HistoricalCalendarView` for feature backfill against past timestamps. |

### Scheduler

`_refresh_calendar_job` runs every 60 minutes (registered in `scheduler.py`). Uses a browser User-Agent to avoid faireconomy.media's CDN throttling Python's default `urllib` UA. 429s are logged and skipped — the next tick retries. The job only registers when `start_scheduler()` boots, which requires OANDA + Anthropic credentials.

### One-shot day-one backfill (`ml/calendar_backfill.py`)

The backfill is **not** part of live ops. It's a one-time bootstrap that imports 6 months of historical FF events (so ML features have a meaningful history) and immediately retrains the AutoGluon classifier with calendar columns active.

**Dependency caveat:** `market-calendar-tool` 0.2.x pins `pyarrow<18,>=17`, which has no Python 3.13 wheel. Install the package in a separate Python 3.12 venv just for the backfill — your live `~/dealfinder` (3.13) venv stays untouched.

**Runbook:**

```bash
# 1. Set up the py312 venv (one-time)
python3.12 -m venv ~/dealfinder-py312
source ~/dealfinder-py312/bin/activate
pip install --upgrade pip "setuptools<80"
pip install pandas requests python-dotenv "market-calendar-tool>=0.1.0"

# 2. Run stages 1+2 (historical archive + feature re-extract) in py312
cd ~/ict-terminal
python -m ml.calendar_backfill --months 6 \
    --db ~/ict-terminal/ml/models/scanner.db \
    --skip-retrain

# 3. Run stage 3 (retrain) back in py313 — AutoGluon doesn't fit in the py312 venv
source ~/dealfinder/bin/activate
curl -s -X POST http://localhost:8000/retrain
# (or run train_classifier directly if the server isn't up)
```

The backfill is idempotent — running stage 1 twice produces zero new rows. Stage 2 is also safe to re-run; it overwrites `analysis_json["calendar_features"]` with the same values.

### Notes on 0.2.x API

`market-calendar-tool` 0.2.x changed the API in three ways relative to 0.1.x:
- `site` argument now requires the `Site` enum, not a string
- `scrape_calendar` returns a `ScrapeResult` whose `.base` is the DataFrame
- Column names differ from the live XML feed (`name` vs `title`, `impactName` vs `impact`, `timeLabel` vs `time`) **and times come back in UTC+1, not UTC** — the scraper hits an `apply-settings/1` endpoint that defaults to GMT+1

`_scrape` in `calendar_backfill.py` handles all three: it imports `Site`, unwraps `result.base`, renames columns, and `_parse_row_timestamp` detects abbreviated-month dates (`"Apr 29, 2026"`) and applies a `-1h` offset so historical rows align with live UTC timestamps. The cross-source consistency test (`test_backfill_matches_live_xml_for_overlap_week`, marker `integration`) verifies this empirically against the live feed.

## V3 Roadmap — Self-Improving Feedback Loops

The system's next evolution: ICT theory remains the foundation. Outcome data creates weighted setup profiles that influence (not replace) the foundational ICT logic.

### Priority 1: Setup Profile Learning
Every resolved trade builds a profile: killzone, timeframe, ICT elements present (OB, FVG, sweep, BOS/CHoCH), entry placement within OB, Opus/Sonnet agreement, R:R, outcome. Over time, profiles create weighted rules like "when these 4 confluences are present during London on 15min, win rate is 78% — this is an A setup even if Sonnet graded it B." Profiles also promote C-grade setups that consistently win (e.g., C-grades during London with confirmed sweep) to notification-worthy. The goal is both filtering out noise AND discovering setups the system currently misses.

### Priority 2: Entry Placement Refinement
Resolved trades show MFE (max favourable excursion) and MAE (max adverse excursion). The system learns where in the OB/FVG zone entries perform best — e.g., "OB midpoint entries outperform OB edge entries by 15%." Feeds back into how Sonnet places entries, not just whether it takes the trade.

### Priority 3: SL/TP Calibration Tightening
As real scanner outcomes accumulate, the calibrator should weight live data over V1 seed data. Track which calibration layer (volatility, Bayesian, AutoGluon, historical) actually adds value and dial down the ones that don't.

### Priority 4: Opus Narrative Accuracy Loop
Track Opus directional bias, P3 phase, and premium/discount calls against outcomes. Build a track record per narrative type — "Opus is 80% accurate on distribution calls but 55% on accumulation." Feed back into how much Sonnet trusts the HTF narrative.

### Priority 5: Haiku False Negative Detection ✅ IMPLEMENTED
When Haiku screens out a timeframe, track what price did in the next 1-2 hours. If gold made a clean directional move (≥2 ATR), log it as a false negative.

**Implementation** (`haiku_fn_tracker.py`, 40 tests):
- Every Haiku rejection logged with: timestamp, timeframe, killzone, close price, ATR, reason (auto-categorised), structural score, confluence count
- Resolution piggybacks on `monitor_pending()` — uses same 5-min candles to check forward price action after rejection
- Forward windows: 15min→2h, 1h→4h, 4h→12h, 1day→2 days
- FN threshold: ≥2.0 ATR move = false negative, ≥3.5 ATR = strong FN
- Aggregates FN rates by: timeframe × killzone segment, reason category, structural score
- **Adaptive feedback**: segments with ≥60% FN rate → Haiku bypassed; ≥40% FN rate + confluence≥2 → Haiku overridden
- Stale rejections (>72h unresolved, e.g. weekend gaps) auto-expired
- Endpoints: `/haiku/fn/report`, `/haiku/fn/stats`, `/haiku/fn/adjustments`, `/haiku/fn/recent`

### Priority 6: Killzone Performance Profiling
Rather than scanning all killzones equally, learn the quality bar per killzone from data — e.g., Asian needs A-grade with 3+ confluences, London can take B-grade with 2 confluences. Adaptive, not hardcoded.

### Priority 7: Intermarket Signal Validation
Track whether DXY/US10Y context actually improved trade outcomes. If "DXY falling, supports gold long" predicts winners, lean in. If divergence warnings don't correlate with losses, stop burning tokens on them.

### Priority 8: Cost-Per-Winner Optimization ✅ IMPLEMENTED
Track API cost per winning notification by timeframe and killzone. Allocate budget to high-value scanning windows. Scan less frequently during low-quality periods.

**Implementation** (`cost_per_winner.py`, 39 tests):
- Every API call during analysis (Haiku screen, Sonnet analysis, Opus narrative) accumulates cost via `_pending_api_cost` in the scanner pipeline
- After `store_setup()`, total API cost is written to `api_cost_usd` column on `scanner_setups`
- `CostTracker.log_call()` accepts optional `setup_id` for per-setup cost attribution
- On trade resolution, `CostPerWinnerTracker.ingest_trade()` aggregates by timeframe×killzone segment
- **Per-segment metrics**: cost_per_winner_usd, cost_per_trade_usd, roi_per_dollar (R earned per $1 API spend), win_rate
- **Adaptive recommendations**: segments with CPW ≥ $3.00 or ROI < 0.5 R/$ → scan frequency reduced; segments with ROI ≥ 1.5 R/$ → boosted
- **Data windowing**: only last 30 days of resolved setups count (same pattern as P5)
- **Minimum samples**: 10 resolved setups per segment before recommendations activate
- `recompute_from_db()` for full recalculation from historical data
- Endpoints: `/cost/per-winner`, `/cost/per-winner/ranking`, `/cost/per-winner/recommendations`, `/cost/per-winner/recompute`

### Design Principle
ICT theory = the rails. Setup profiles = weighted influence on confidence thresholds, confluence requirements, and quality grading. The system never drifts away from ICT methodology — it just gets sharper at applying it.
