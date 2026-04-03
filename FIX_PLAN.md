# ICT Terminal ML Pipeline Fix Plan

## Context

Analysis of 249 resolved scanner trades revealed the ML self-improvement loop is partially broken. The system is profitable (+104.3R, 74.3% WR) but running on only 2 of 6 calibration layers. Four critical bugs prevent the system from learning and improving over time.

## Already Fixed (in this session)

### Fix 1: Session Detection Timestamp Parsing — `ml/volatility.py`
**Bug:** `detect_session()` couldn't parse Twelve Data timestamps (`"2026-03-16 01:30:00"` with space separator). It only handled ISO format with "T". The hour parser threw ValueError and defaulted to "off_hours" for EVERY trade. All 389 trades in the accuracy tracker were tagged session "off".
**Fix applied:** Updated the timestamp parser to handle both `"T"` and space-separated formats.

### Fix 2: Setup Type Classification — `ml/scanner.py` `_log_trade_complete()`
**Bug:** Scanner passed raw Claude JSON as `original_analysis` to `log_completed_trade()`. The bridge checked for `"claude_direction"` in the dict (a key that only exists in PARSED analysis, not raw Claude JSON). So `classify_setup_type()` never ran — all trades tagged "unknown".
**Fix applied:** `_log_trade_complete` now calls `bridge.parse_analysis(raw_analysis)` before passing to `log_completed_trade()`, so setup type classification and feature extraction work.

### Fix 3: AutoGluon Model Persistence — `ml/config.py`
**Bug:** `model_dir` was `"models/"` (relative path). AutoGluon saved trained models to whatever the current working directory was when the server process started — NOT to `ml/models/`. The classifier and quantile_mfe directories are empty.
**Fix applied:** All paths in config (`model_dir`, `db_path`, `wfo_report_path`, `dataset_parquet_path`) now resolve to absolute paths using `os.path.dirname(os.path.abspath(__file__))`.

### Fix 4: Live Trade Ingestion — `ml/claude_bridge.py`
**Bug:** `log_completed_trade()` tried `original_analysis.get("features", {})`. Raw Claude JSON has no "features" key. Features dict was always empty, `if features:` failed, and `dm.ingest_live_trade()` was never called. Only 32 of 249 scanner trades made it into the training dataset.
**Fix applied:** Added `_build_minimal_features()` fallback that constructs a feature dict from parsed analysis + calibration metadata when full candle-based features aren't available. Now ingestion always runs.

## Still Needs Doing

### Task 5: Backfill Accuracy Tracker from Existing 249 Scanner Trades

The accuracy tracker (`ml/models/claude_accuracy.json`) has stale data with all trades tagged session="off" and setup_type="unknown". It needs to be rebuilt from the scanner DB.

Write a script `ml/backfill.py` that:
1. Resets `claude_accuracy.json` to fresh empty state
2. Reads ALL 249 resolved trades from `scanner.db` (via `ScannerDB.get_history()`)
3. For each trade, in chronological order:
   - Parse the `analysis_json` using `ClaudeAnalysisBridge.parse_analysis()`
   - Call `bridge.log_completed_trade()` with the parsed analysis and `calibration_json`
   - This will correctly classify setup types and map sessions
4. Print summary showing session distribution and setup type distribution
5. Do NOT touch the Bayesian state (it's already correct with 400 trades)

Run with: `python -m ml.backfill`

### Task 6: Backfill Training Dataset from Scanner Trades

The training dataset (`ml/models/training_dataset.csv`) only has 92 rows (60 WFO + 32 live). The other 217 resolved scanner trades need ingesting.

In the same `ml/backfill.py` script (or a separate pass):
1. Read all 249 resolved scanner trades from `scanner.db`
2. For each, parse with `ClaudeAnalysisBridge` to get features
3. Use `_build_minimal_features()` for trades without full candle features
4. Call `TrainingDatasetManager.ingest_live_trade()` for each
5. Deduplicate by checking if `setup_id` already exists in the CSV
6. Print final dataset stats (total rows, source breakdown, outcome distribution)

### Task 7: Fix Tests for Config Path Changes

The config change from relative to absolute paths may break tests that depend on `model_dir` being relative. Check:
- `ml/tests/test_calibrate.py` — uses `make_test_config()`
- `ml/tests/test_training.py` — creates models in `model_dir`
- `ml/tests/test_prediction.py` — loads models from `model_dir`

Fix: Ensure `make_test_config()` overrides `model_dir` and `db_path` to use `tmp_path` in tests. The test conftest fixtures probably already do this, but verify.

Run: `python -m pytest ml/tests/ -v` — all 391 tests must pass.

### Task 8: Verify AutoGluon Retraining Works End-to-End

After backfill, the training dataset should have ~300+ rows. Trigger a retrain:
1. Run `python -m pytest ml/tests/test_training.py -v` first
2. Then manually call the retrain endpoint or run:
   ```python
   from ml.training import train_classifier
   from ml.database import TradeLogger
   from ml.dataset import TrainingDatasetManager
   db = TradeLogger()
   dm = TrainingDatasetManager()
   result = train_classifier(db, dataset_manager=dm)
   print(result)
   ```
3. Verify models are saved in `ml/models/classifier/` (should contain `learner.pkl` and other AG artifacts)
4. Verify `ml/models/quantile_mfe/` also gets populated if 50+ samples

### Task 9: End-to-End Verification

After all fixes + backfill + retrain, verify the full loop works:

1. **Session tagging:** Query scanner.db for a recent trade, check its `calibration_json` — `session_context.session` should NOT be "off" for Asian/London/NY trades
2. **Setup type:** `claude_accuracy.json` → `by_setup_type` should have entries like `"bear_ob_fvg_london"`, not just `"unknown"`
3. **AutoGluon:** The calibration endpoint should return non-zero `autogluon_win_prob` in the confidence object
4. **Dataset growth:** After the next scanner resolution, check `training_dataset.csv` row count increased by 1
5. **Confidence grades:** With AG active, confidence scores should be higher. Grades should include some B and C, not just D/F

### Task 10: Killzone Normalization in Claude's Prompt

The scanner stores wildly inconsistent killzone strings from Claude (e.g. "London Open approaching - optimal for bearish continuation", "Asian Session (00:00-04:00 UTC)", "Off-Session"). The `_map_killzone_to_session()` in `claude_bridge.py` handles this, but it would be cleaner to:

1. Update the ICT system prompt in `scanner.py` (`ICT_SYSTEM_MESSAGE`) to constrain the killzone field to exactly one of: `"Asian"`, `"London"`, `"NY_AM"`, `"NY_PM"`, `"Off"`
2. Add this to the JSON schema section of the prompt:
   ```
   "killzone": "One of: Asian, London, NY_AM, NY_PM, Off"
   ```
3. This reduces parsing fragility downstream

## Priority Order

1. **Task 7** (fix tests) — run tests FIRST to make sure the 4 already-applied fixes don't break anything
2. **Task 5** (backfill accuracy tracker) — rebuild with correct session/setup tags
3. **Task 6** (backfill training dataset) — get all 249 trades into the CSV
4. **Task 8** (retrain AutoGluon) — with 300+ samples, the models should train properly
5. **Task 9** (e2e verification) — confirm everything works together
6. **Task 10** (killzone normalization) — nice-to-have, reduces future parsing bugs
