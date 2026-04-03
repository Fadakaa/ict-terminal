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
| `backfill_features.py` | Retroactive feature enrichment — re-extracts 52-col features from stored analysis/calibration JSON, backfills regime + intermarket from OANDA |

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

## Testing

```bash
python -m pytest ml/tests/ -v              # Full suite (842+ tests)
python -m pytest ml/tests/test_seed.py     # Seed harvester tests
python -m pytest ml/tests/test_claude_bridge.py  # Bridge tests
python -m pytest ml/tests/test_calibrate.py      # Calibrator tests
python -m pytest ml/tests/test_prompts.py        # Prompt tests
```

## Completed Setup Actions

- **Bayesian state reset** (2026-03-25): Reset to V1 priors (alpha=8.92, beta=11.08, 44.6% WR). Scanner trades no longer update Bayesian beliefs (`source="scanner"` skips the update in `claude_bridge.py`). Only live trades update beliefs.
- **OANDA API token regenerated** (2026-03-25): New token generated and updated in `.env`.

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

The ML tab displays: classifier status, AutoGluon model info, dataset composition, backtest fidelity, calibration layer performance, Bayesian drift, API cost tracking, active prospects, and scanner status.

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
