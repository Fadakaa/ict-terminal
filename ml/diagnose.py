"""Isolated Variable Diagnostic Tool for WFO parameter testing.

Sequential, interactive diagnostic that runs each config one at a time,
isolating exactly one variable change per step. Shows incremental impact
vs baseline and vs the previous step, then pauses for user input.

Steps 1-6: Single variable changes (each resets to baseline + one tweak)
Steps 7-10: Built dynamically by layering top performers from steps 2-6

Usage:
    python -m ml.diagnose --td-key KEY --candles 1500
    python -m ml.diagnose --td-key KEY --candles 2000 --start-step 5
    python -m ml.diagnose --td-key KEY --candles 1500 --auto
"""
import argparse
import json
import os
import sys
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from typing import Optional

from ml.wfo import (
    WFOConfig, WalkForwardEngine, WFOReport, V2_CONFIG_OVERRIDES,
    _compute_wfo_grade, _fetch_candles_twelve_data,
)
from ml.execution import ExecutionSimulator
from ml.config import get_config
from ml.features import compute_atr


# ═══════════════════════════════════════════════════════════════════════
# Diagnostic sequence — Steps 1-6 (single variable changes)
# ═══════════════════════════════════════════════════════════════════════

DIAGNOSTIC_SEQUENCE = [
    # Step 1: Establish the floor
    {
        "step": 1,
        "name": "baseline",
        "description": "Current settings — establishes the floor",
        "changed": "nothing — this is the control",
        "sl": 1.5, "tps": [1.0, 2.0, 3.5], "confluence": 2,
        "retracement": False, "mtf": False, "filter_ct": False,
    },
    # Step 2: Only change SL
    {
        "step": 2,
        "name": "wider_stops",
        "description": "Wider stops only — does gold need more room?",
        "changed": "sl_atr_mult: 1.5 → 2.5",
        "sl": 2.5, "tps": [1.0, 2.0, 3.5], "confluence": 2,
        "retracement": False, "mtf": False, "filter_ct": False,
    },
    # Step 3: Reset SL, only change confluence
    {
        "step": 3,
        "name": "stricter_confluence",
        "description": "Higher confluence only — are bad setups diluting the edge?",
        "changed": "min_confluence_score: 2 → 4",
        "sl": 1.5, "tps": [1.0, 2.0, 3.5], "confluence": 4,
        "retracement": False, "mtf": False, "filter_ct": False,
    },
    # Step 4: Reset confluence, only change TPs
    {
        "step": 4,
        "name": "adjusted_targets",
        "description": "Realistic TP targets only — are targets beyond gold's actual range?",
        "changed": "tp_atr_mults: [1.0, 2.0, 3.5] → [0.75, 1.5, 2.5]",
        "sl": 1.5, "tps": [0.75, 1.5, 2.5], "confluence": 2,
        "retracement": False, "mtf": False, "filter_ct": False,
    },
    # Step 5: Reset TPs, only add MTF
    {
        "step": 5,
        "name": "mtf_alignment",
        "description": "4H alignment only — does higher-timeframe structure matter?",
        "changed": "mtf: off → on, filter_counter_trend: off → on",
        "sl": 1.5, "tps": [1.0, 2.0, 3.5], "confluence": 2,
        "retracement": False, "mtf": True, "filter_ct": True,
    },
    # Step 6: Reset MTF, only add retracement entry
    {
        "step": 6,
        "name": "retracement_entry",
        "description": "Retracement entries only — does waiting for pullback improve R:R?",
        "changed": "use_retracement_entry: off → on",
        "sl": 1.5, "tps": [1.0, 2.0, 3.5], "confluence": 2,
        "retracement": True, "mtf": False, "filter_ct": False,
    },
]


RESULTS_PATH = os.path.join(
    os.path.dirname(__file__), "models", "diagnostic_results.json"
)


# ═══════════════════════════════════════════════════════════════════════
# Step result storage
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class StepResult:
    step: int
    name: str
    description: str
    changed: str
    win_rate: float
    profit_factor: float
    cost_adjusted_pf: float
    sharpe: float
    oos_trades: int
    regime_stability: float
    grade: str
    recommended_sl_atr: float
    recommended_tp_atr: list
    config_used: dict

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "StepResult":
        return cls(**d)


# ═══════════════════════════════════════════════════════════════════════
# Impact classification
# ═══════════════════════════════════════════════════════════════════════


def _classify_wr_impact(delta: float) -> str:
    """Classify win rate change magnitude."""
    abs_d = abs(delta * 100)  # as percentage points
    if abs_d > 5:
        return "◈ SIGNIFICANT"
    if abs_d >= 2:
        return "△ MODERATE"
    return "○ MINIMAL"


def _classify_pf_impact(delta: float) -> str:
    """Classify profit factor change magnitude."""
    abs_d = abs(delta)
    if abs_d > 0.3:
        return "◈ SIGNIFICANT"
    if abs_d >= 0.1:
        return "△ MODERATE"
    return "○ MINIMAL"


def _classify_grade_change(old_grade: str, new_grade: str) -> str:
    """Classify grade transition."""
    order = {"D": 0, "C": 1, "B": 2, "A": 3}
    old_v = order.get(old_grade, 0)
    new_v = order.get(new_grade, 0)
    if new_v > old_v:
        return "◈ IMPROVED"
    if new_v < old_v:
        return "▼ DEGRADED"
    return "○ UNCHANGED"


def _compute_verdict(wr_delta: float, capf_delta: float,
                     current: StepResult, baseline: StepResult) -> str:
    """Compute overall verdict for a step."""
    wr_pct = wr_delta * 100
    # Check for degradation first
    if capf_delta < -0.2 or wr_pct < -5:
        return "NEGATIVE IMPACT — this change hurts performance"
    if capf_delta > 0.2 and wr_pct > 3:
        return "SIGNIFICANT FACTOR"
    if capf_delta > 0.1 or wr_pct > 2:
        return "MODERATE FACTOR"
    return "NOT A SIGNIFICANT FACTOR — can likely skip this variable"


# ═══════════════════════════════════════════════════════════════════════
# Output formatting
# ═══════════════════════════════════════════════════════════════════════


def _format_comparison(label: str, current: StepResult,
                       reference: StepResult, ref_name: str) -> str:
    """Format a comparison block between current and reference step."""
    lines = []
    lines.append(f"  {label}")
    lines.append("  ───────────────────────────────────")

    # Win Rate
    wr_delta = current.win_rate - reference.win_rate
    wr_tag = _classify_wr_impact(wr_delta)
    lines.append(
        f"  Win Rate:          {wr_delta:+.1%}  "
        f"({reference.win_rate:.1%} → {current.win_rate:.1%})   {wr_tag}"
    )

    # Profit Factor
    pf_delta = current.profit_factor - reference.profit_factor
    pf_tag = _classify_pf_impact(pf_delta)
    lines.append(
        f"  Profit Factor:     {pf_delta:+.2f}   "
        f"({reference.profit_factor:.2f} → {current.profit_factor:.2f})     {pf_tag}"
    )

    # Cost-Adj PF
    capf_delta = current.cost_adjusted_pf - reference.cost_adjusted_pf
    capf_tag = _classify_pf_impact(capf_delta)
    lines.append(
        f"  Cost-Adj PF:       {capf_delta:+.2f}   "
        f"({reference.cost_adjusted_pf:.2f} → {current.cost_adjusted_pf:.2f})     {capf_tag}"
    )

    # Sharpe
    sh_delta = current.sharpe - reference.sharpe
    sh_tag = _classify_pf_impact(sh_delta)
    lines.append(
        f"  Sharpe:            {sh_delta:+.2f}   "
        f"({reference.sharpe:.2f} → {current.sharpe:.2f})     {sh_tag}"
    )

    # OOS Trades
    tr_delta = current.oos_trades - reference.oos_trades
    tr_tag = "○ MINIMAL" if abs(tr_delta) < 30 else ("◈ SIGNIFICANT" if abs(tr_delta) > 100 else "△ MODERATE")
    lines.append(
        f"  OOS Trades:        {tr_delta:+d}     "
        f"({reference.oos_trades} → {current.oos_trades})        {tr_tag}"
    )

    # Grade
    grade_tag = _classify_grade_change(reference.grade, current.grade)
    lines.append(
        f"  Grade:             {reference.grade} → {current.grade}"
        f"                      {grade_tag}"
    )

    return "\n".join(lines)


def _print_step_result(current: StepResult, baseline: StepResult,
                       previous: StepResult, total_steps: int,
                       atr: float = None, entry_price: float = None):
    """Print formatted step result with comparisons."""
    print()
    print("═" * 65)
    print(f"  STEP {current.step} of {total_steps}: {current.name}")
    print(f"  Changed: {current.changed}")
    print("═" * 65)

    # Results
    print()
    print("  RESULTS")
    print("  ───────────────────────────────────")
    print(f"  Win Rate:          {current.win_rate:.1%}")
    print(f"  Profit Factor:     {current.profit_factor:.2f}")
    print(f"  Cost-Adj PF:       {current.cost_adjusted_pf:.2f}")
    print(f"  Sharpe:            {current.sharpe:.2f}")
    print(f"  OOS Trades:        {current.oos_trades}")
    print(f"  Regime Stability:  {current.regime_stability:.2f}")
    print(f"  Grade:             {current.grade}")

    # VS Baseline
    print()
    print(_format_comparison("VS BASELINE", current, baseline, "baseline"))

    # VS Previous (if different from baseline)
    if previous.step != baseline.step:
        print()
        print(_format_comparison(
            f"VS PREVIOUS STEP ({previous.name})",
            current, previous, previous.name
        ))

    # Verdict
    wr_delta = current.win_rate - baseline.win_rate
    capf_delta = current.cost_adjusted_pf - baseline.cost_adjusted_pf
    verdict = _compute_verdict(wr_delta, capf_delta, current, baseline)
    print()
    print(f"  VERDICT: {current.name.upper()} — {verdict}")

    # Compute description based on step
    if current.step == 2 and capf_delta > 0.2:
        print("  The SL was the primary bottleneck — gold needs 2.5 ATR breathing room.")
    elif current.step == 3 and capf_delta > 0.1:
        print("  Higher confluence filters out noise — quality over quantity.")
    elif current.step == 5 and capf_delta > 0.1:
        print("  Higher-timeframe alignment is a significant edge multiplier.")

    # Recommended levels (if ATR known)
    if atr and entry_price and current.recommended_sl_atr:
        print()
        sl_price = entry_price - current.recommended_sl_atr * atr
        print(f"  Recommended SL:    {sl_price:.2f} ({current.recommended_sl_atr:.1f} ATR from entry)")
        for i, tp in enumerate(current.recommended_tp_atr[:3]):
            tp_price = entry_price + tp * atr
            print(f"  Recommended TP{i+1}:   {tp_price:.2f}")

    print()
    print("═" * 65)


def _print_summary(results: list[StepResult], baseline: StepResult):
    """Print the final summary ranking table."""
    print()
    print("═" * 65)
    print("  VARIABLE IMPACT RANKING — XAU/USD ICT SYSTEM")
    print("═" * 65)

    # Single variable impacts (steps 2-6)
    singles = [r for r in results if 2 <= r.step <= 6]
    if singles:
        # Rank by cost_adjusted_pf delta
        ranked = sorted(
            singles,
            key=lambda r: r.cost_adjusted_pf - baseline.cost_adjusted_pf,
            reverse=True,
        )

        print()
        print("  SINGLE VARIABLE IMPACTS (Steps 2-6)")
        print("  ───────────────────────────────────────────────────────────")
        print(f"  {'Rank':<6}{'Variable':<24}{'WR Δ':>8}{'PF Δ':>9}{'CostPF Δ':>10}  {'Verdict'}")
        for rank, r in enumerate(ranked, 1):
            wr_d = r.win_rate - baseline.win_rate
            pf_d = r.profit_factor - baseline.profit_factor
            capf_d = r.cost_adjusted_pf - baseline.cost_adjusted_pf
            verdict = _compute_verdict(wr_d, capf_d, r, baseline)
            # Shorten verdict
            short_v = verdict.split("—")[0].strip() if "—" in verdict else verdict
            print(
                f"  {rank:<6}{r.name:<24}{wr_d:>+7.1%}"
                f"{pf_d:>+9.2f}{capf_d:>+10.2f}  {short_v}"
            )

    # Compound impacts (steps 7-10)
    compounds = [r for r in results if r.step >= 7]
    if compounds:
        print()
        print("  COMPOUND IMPACTS (Steps 7-10)")
        print("  ───────────────────────────────────────────────────────────")
        print(f"  {'Step':<6}{'Variables Combined':<28}{'WR':>7}{'PF':>7}{'CostPF':>8}  {'Grade'}")
        for r in compounds:
            print(
                f"  {r.step:<6}{r.name:<28}"
                f"{r.win_rate:>6.1%}{r.profit_factor:>7.2f}"
                f"{r.cost_adjusted_pf:>8.2f}  {r.grade}"
            )

        # Detect diminishing returns
        if len(compounds) >= 2:
            last = compounds[-1]
            second_last = compounds[-2]
            if last.cost_adjusted_pf <= second_last.cost_adjusted_pf + 0.05:
                print()
                print(f"  DIMINISHING RETURNS DETECTED AT STEP {last.step}")
                print(f"  Adding variables beyond step {second_last.step} "
                      "provided no further improvement.")

    # Find best result
    all_with_data = [r for r in results if r.step >= 2]
    if all_with_data:
        best = max(all_with_data, key=lambda r: r.cost_adjusted_pf)
        print()
        print("  RECOMMENDED PRODUCTION CONFIG")
        print("  ───────────────────────────────────────────────────────────")
        cfg = best.config_used
        print(f"  sl_atr_mult:            {cfg.get('sl', 1.5)}")
        print(f"  tp_atr_mults:           {cfg.get('tps', [1.0, 2.0, 3.5])}")
        print(f"  min_confluence_score:   {cfg.get('confluence', 2)}")
        print(f"  use_retracement_entry:  {cfg.get('retracement', False)}")
        print(f"  use_mtf:                {cfg.get('mtf', False)}")
        print(f"  filter_counter_trend:   {cfg.get('filter_ct', False)}")
        print()
        wr_improvement = best.win_rate - baseline.win_rate
        pf_improvement = best.profit_factor - baseline.profit_factor
        print(f"  Expected Performance:   {best.win_rate:.1%} WR, "
              f"{best.profit_factor:.2f} PF, Grade {best.grade}")
        print(f"  Improvement over baseline: {wr_improvement:+.1%} WR, "
              f"{pf_improvement:+.2f} PF")

    print("═" * 65)


# ═══════════════════════════════════════════════════════════════════════
# Core diagnostic engine
# ═══════════════════════════════════════════════════════════════════════


class VariableDiagnostic:
    """Isolated variable testing tool for WFO parameter optimization.

    Runs configs in a specific order designed so each step changes exactly
    one variable from baseline. After each config completes, prints the
    result, compares to baseline and previous, and waits for user input.
    """

    def __init__(self, candles: list[dict], timeframe: str = "1h",
                 auto: bool = False, use_autogluon: bool = False):
        self.candles = candles
        self.timeframe = timeframe
        self.auto = auto
        self.use_autogluon = use_autogluon
        self.results: list[StepResult] = []
        self.baseline: Optional[StepResult] = None

        # Compute reference ATR and price for display
        self.atr = compute_atr(candles, 14) if len(candles) >= 14 else 0
        self.entry_price = candles[-1]["close"] if candles else 0

    def _build_wfo_config(self, step_cfg: dict) -> WFOConfig:
        """Build WFOConfig from a diagnostic step config dict."""
        return WFOConfig(
            sl_atr_mult=step_cfg["sl"],
            tp_atr_mults=list(step_cfg["tps"]),
            min_confluence_score=step_cfg["confluence"],
            use_retracement_entry=step_cfg["retracement"],
            use_mtf=step_cfg["mtf"],
            filter_counter_trend=step_cfg["filter_ct"],
            # Fixed params — same across all steps
            train_window=500,
            test_window=100,
            step_size=50,
            max_bars_in_trade=20,
            max_folds=20,
            min_setups_per_fold=5,
        )

    def _run_step(self, step_cfg: dict) -> StepResult:
        """Execute a single WFO run with the given config and return StepResult."""
        wfo_cfg = self._build_wfo_config(step_cfg)
        engine = WalkForwardEngine(wfo_cfg, use_autogluon=self.use_autogluon)

        print(f"\n  Running WFO for step {step_cfg['step']}: {step_cfg['name']}...")
        print(f"  Config: SL={step_cfg['sl']}, TPs={step_cfg['tps']}, "
              f"Confluence≥{step_cfg['confluence']}, "
              f"MTF={step_cfg['mtf']}, Retracement={step_cfg['retracement']}, "
              f"FilterCT={step_cfg['filter_ct']}")

        report = engine.run(self.candles, self.timeframe)

        # Compute cost-adjusted profit factor by running trades through
        # execution simulator and recalculating PF
        cost_adj_pf = self._compute_cost_adjusted_pf(engine, report)

        return StepResult(
            step=step_cfg["step"],
            name=step_cfg["name"],
            description=step_cfg["description"],
            changed=step_cfg["changed"],
            win_rate=report.oos_win_rate,
            profit_factor=report.oos_profit_factor,
            cost_adjusted_pf=cost_adj_pf,
            sharpe=report.oos_sharpe,
            oos_trades=report.total_oos_trades,
            regime_stability=report.regime_stability,
            grade=report.grade,
            recommended_sl_atr=report.recommended_sl_atr,
            recommended_tp_atr=report.recommended_tp_atr,
            config_used={
                "sl": step_cfg["sl"],
                "tps": step_cfg["tps"],
                "confluence": step_cfg["confluence"],
                "retracement": step_cfg["retracement"],
                "mtf": step_cfg["mtf"],
                "filter_ct": step_cfg["filter_ct"],
            },
        )

    def _compute_cost_adjusted_pf(self, engine: WalkForwardEngine,
                                   report: WFOReport) -> float:
        """Compute cost-adjusted profit factor from OOS trades."""
        oos_trades = getattr(engine, "oos_trades", [])
        if not oos_trades:
            return report.oos_profit_factor

        cfg = get_config()
        sim = ExecutionSimulator(config=cfg)
        adjusted = sim.simulate(oos_trades, self.candles)

        if not adjusted:
            return 0.0

        win_outcomes = {"tp1_hit", "tp2_hit", "tp3_hit"}
        total_win_pnl = 0.0
        total_loss_pnl = 0.0
        for t in adjusted:
            if t.get("outcome") in win_outcomes:
                total_win_pnl += t.get("max_favorable_atr", 0)
            elif t.get("outcome") == "stopped_out":
                total_loss_pnl += t.get("max_drawdown_atr", 0)

        if total_loss_pnl <= 0:
            return min(report.oos_profit_factor, 99.9)
        return round(min(total_win_pnl / total_loss_pnl, 99.9), 4)

    def _build_compound_steps(self) -> list[dict]:
        """Build steps 7-10 by layering top performers from single-variable tests."""
        if not self.baseline:
            return []

        # Rank single-variable steps (2-6) by cost_adjusted_pf improvement
        singles = [r for r in self.results if 2 <= r.step <= 6]
        if not singles:
            return []

        ranked = sorted(
            singles,
            key=lambda r: r.cost_adjusted_pf - self.baseline.cost_adjusted_pf,
            reverse=True,
        )

        # Map step names to their config deltas
        variable_map = {
            "wider_stops": {"sl": 2.5},
            "stricter_confluence": {"confluence": 4},
            "adjusted_targets": {"tps": [0.75, 1.5, 2.5]},
            "mtf_alignment": {"mtf": True, "filter_ct": True},
            "retracement_entry": {"retracement": True},
        }

        compound_steps = []
        # Baseline config as starting point
        base_cfg = {
            "sl": 1.5, "tps": [1.0, 2.0, 3.5], "confluence": 2,
            "retracement": False, "mtf": False, "filter_ct": False,
        }

        accumulated_cfg = dict(base_cfg)
        accumulated_names = []

        for i, ranked_result in enumerate(ranked):
            # Apply this variable's changes on top of accumulated config
            deltas = variable_map.get(ranked_result.name, {})
            accumulated_cfg.update(deltas)
            accumulated_names.append(ranked_result.name)

            step_num = 7 + i
            if step_num > 10:
                break

            if i == 0:
                combined_name = ranked_result.name
                changed = f"Top 1: {ranked_result.name}"
            else:
                combined_name = " + ".join(accumulated_names[:3])
                if len(accumulated_names) > 3:
                    combined_name = f"{combined_name} +{len(accumulated_names)-3} more"
                changed = f"+ {ranked_result.name}"

            compound_steps.append({
                "step": step_num,
                "name": combined_name,
                "description": f"Compound: top {i+1} variables layered together",
                "changed": changed,
                **accumulated_cfg,
            })

        return compound_steps

    def _save_results(self):
        """Persist current results to JSON file."""
        os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
        data = [r.to_dict() for r in self.results]
        with open(RESULTS_PATH, "w") as f:
            json.dump(data, f, indent=2)

    def _load_results(self) -> list[StepResult]:
        """Load previously saved results."""
        if not os.path.exists(RESULTS_PATH):
            return []
        try:
            with open(RESULTS_PATH) as f:
                data = json.load(f)
            return [StepResult.from_dict(d) for d in data]
        except (json.JSONDecodeError, TypeError, KeyError):
            return []

    def _prompt_user(self, step_num: int, total_steps: int) -> str:
        """Prompt for user input between steps. Returns action string."""
        if self.auto:
            return "continue"

        next_step = step_num + 1
        prompt = (
            f"  Press Enter to continue to Step {next_step}, "
            f"or type 'stop' to end..."
        )
        if step_num < total_steps:
            prompt = (
                f"  Press Enter to continue to Step {next_step}, "
                f"'stop' to end, 'rerun' to rerun, 'skip' to skip next..."
            )
        else:
            prompt = "  Press Enter to finish, or type 'stop'..."

        try:
            response = input(f"\n{prompt}\n  > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return "stop"
        if not response:
            return "continue"
        return response

    def run(self, start_step: int = 1):
        """Execute the full diagnostic sequence.

        Args:
            start_step: Step to resume from. If > 1, loads existing results
                       and reruns baseline first for comparison.
        """
        # Load existing results if resuming
        if start_step > 1:
            existing = self._load_results()
            if existing:
                self.results = [r for r in existing if r.step < start_step]
                # Find baseline in existing results
                for r in existing:
                    if r.step == 1:
                        self.baseline = r
                        break
            if not self.baseline:
                print("  No saved baseline found — running baseline first...")
                start_step = 1

        # Phase 1: Single-variable steps (1-6)
        all_steps = list(DIAGNOSTIC_SEQUENCE)
        total_steps = 10  # We always plan for 10

        # If resuming from a step > 1, rerun baseline for reference
        if start_step > 1 and self.baseline is None:
            baseline_result = self._run_step(DIAGNOSTIC_SEQUENCE[0])
            self.baseline = baseline_result
            self.results.insert(0, baseline_result)
            _print_step_result(baseline_result, baseline_result, baseline_result,
                             total_steps, self.atr, self.entry_price)

        skip_next = False
        for step_cfg in all_steps:
            step_num = step_cfg["step"]

            if step_num < start_step:
                continue

            if skip_next:
                skip_next = False
                continue

            # Run this step
            result = self._run_step(step_cfg)

            # Set baseline
            if step_num == 1:
                self.baseline = result

            self.results.append(result)
            self._save_results()

            # Determine previous step for comparison
            previous = self.baseline
            if len(self.results) >= 2:
                previous = self.results[-2]

            # Print
            _print_step_result(result, self.baseline, previous,
                             total_steps, self.atr, self.entry_price)

            # Pause (unless last single-variable step and auto-building compounds)
            if step_num < 6:
                action = self._prompt_user(step_num, total_steps)
                if action == "stop":
                    break
                elif action == "rerun":
                    # Remove last result and re-run
                    self.results.pop()
                    result = self._run_step(step_cfg)
                    self.results.append(result)
                    self._save_results()
                    _print_step_result(result, self.baseline, previous,
                                     total_steps, self.atr, self.entry_price)
                    action = self._prompt_user(step_num, total_steps)
                    if action == "stop":
                        break
                elif action == "skip":
                    skip_next = True
            elif step_num == 6:
                # After step 6, build compound steps
                action = self._prompt_user(step_num, total_steps)
                if action == "stop":
                    _print_summary(self.results, self.baseline)
                    return
                elif action == "skip":
                    skip_next = True

        # Phase 2: Compound steps (7-10)
        # Only if we completed all single-variable steps
        singles_done = any(r.step == 6 for r in self.results)
        if singles_done:
            compound_steps = self._build_compound_steps()

            print()
            print("  ═══════════════════════════════════════════════")
            print("  BUILDING COMPOUND CONFIGS FROM TOP PERFORMERS")
            print("  ═══════════════════════════════════════════════")

            # Print the ranking used to build compounds
            singles = [r for r in self.results if 2 <= r.step <= 6]
            ranked = sorted(
                singles,
                key=lambda r: r.cost_adjusted_pf - self.baseline.cost_adjusted_pf,
                reverse=True,
            )
            for i, r in enumerate(ranked, 1):
                delta = r.cost_adjusted_pf - self.baseline.cost_adjusted_pf
                print(f"  #{i} {r.name}: Cost-Adj PF {delta:+.2f}")
            print()

            skip_next = False
            for step_cfg in compound_steps:
                step_num = step_cfg["step"]

                if step_num < start_step:
                    continue

                if skip_next:
                    skip_next = False
                    continue

                result = self._run_step(step_cfg)
                self.results.append(result)
                self._save_results()

                previous = self.results[-2] if len(self.results) >= 2 else self.baseline
                _print_step_result(result, self.baseline, previous,
                                 total_steps, self.atr, self.entry_price)

                if step_num < 10:
                    action = self._prompt_user(step_num, total_steps)
                    if action == "stop":
                        break
                    elif action == "rerun":
                        self.results.pop()
                        result = self._run_step(step_cfg)
                        self.results.append(result)
                        self._save_results()
                        _print_step_result(result, self.baseline, previous,
                                         total_steps, self.atr, self.entry_price)
                        action = self._prompt_user(step_num, total_steps)
                        if action == "stop":
                            break
                    elif action == "skip":
                        skip_next = True

        # Final summary
        if self.baseline:
            _print_summary(self.results, self.baseline)

    def run_ab_test(self):
        """Run A/B comparison between old V1 detector and new V2 quality detector.

        Config A: V1 (current baseline, integer confluence scoring)
        Config B: V2 (quality-weighted scoring, rejection entries, narrative filter)

        Prints side-by-side comparison.
        """
        print()
        print("═" * 60)
        print("  DETECTOR A/B TEST — OLD (V1) vs NEW (V2)")
        print("═" * 60)

        # Config A: baseline V1
        cfg_a = WFOConfig(
            train_window=500, test_window=100, step_size=50,
            max_bars_in_trade=20, max_folds=20, min_setups_per_fold=5,
            sl_atr_mult=1.5, min_confluence_score=2,
            use_quality_scoring=False,
        )

        # Config B: V2 with all new features
        cfg_b = WFOConfig(
            train_window=500, test_window=100, step_size=50,
            max_bars_in_trade=20, max_folds=20, min_setups_per_fold=5,
            use_quality_scoring=True,
            **{k: v for k, v in V2_CONFIG_OVERRIDES.items()
               if k != "use_quality_scoring"},
        )

        print("\n  Running V1 (old detector)...")
        engine_a = WalkForwardEngine(cfg_a, use_autogluon=self.use_autogluon)
        report_a = engine_a.run(self.candles, self.timeframe)
        capf_a = self._compute_cost_adjusted_pf(engine_a, report_a)

        print("  Running V2 (new detector)...")
        engine_b = WalkForwardEngine(cfg_b, use_autogluon=self.use_autogluon)
        report_b = engine_b.run(self.candles, self.timeframe)
        capf_b = self._compute_cost_adjusted_pf(engine_b, report_b)

        # Compute average quality score for V2 setups
        avg_quality = 0.0
        rejection_rate = 0.0
        oos_b = getattr(engine_b, "oos_trades", [])
        if oos_b:
            quality_scores = [t.get("total_quality_score", 0) for t in oos_b
                              if "total_quality_score" in t]
            if quality_scores:
                avg_quality = sum(quality_scores) / len(quality_scores)
            rejection_count = sum(1 for t in oos_b
                                   if t.get("entry_type") == "rejection")
            rejection_rate = rejection_count / len(oos_b) if oos_b else 0

        # Print comparison
        print()
        print("═" * 60)
        print("  DETECTOR A/B TEST — OLD vs NEW")
        print("═" * 60)

        def _delta(a_val, b_val, fmt="+.1%"):
            d = b_val - a_val
            return f"{d:{fmt}}"

        header = f"  {'':24s}{'OLD':>12s}{'NEW':>12s}{'Δ':>12s}"
        print(header)

        wr_d = _delta(report_a.oos_win_rate, report_b.oos_win_rate)
        print(f"  {'Win Rate:':<24s}{report_a.oos_win_rate:>11.1%}"
              f"{report_b.oos_win_rate:>12.1%}{wr_d:>12s}")

        cpf_d = _delta(capf_a, capf_b, "+.2f")
        print(f"  {'Cost-Adj PF:':<24s}{capf_a:>11.2f}"
              f"{capf_b:>12.2f}{cpf_d:>12s}")

        pf_d = _delta(report_a.oos_profit_factor, report_b.oos_profit_factor, "+.2f")
        print(f"  {'Profit Factor:':<24s}{report_a.oos_profit_factor:>11.2f}"
              f"{report_b.oos_profit_factor:>12.2f}{pf_d:>12s}")

        t_d = _delta(report_a.total_oos_trades, report_b.total_oos_trades, "+d")
        print(f"  {'OOS Trades:':<24s}{report_a.total_oos_trades:>11d}"
              f"{report_b.total_oos_trades:>12d}{t_d:>12s}")

        print(f"  {'Avg Quality Score:':<24s}{'n/a':>11s}"
              f"{avg_quality:>12.1f}{'n/a':>12s}")

        print(f"  {'Rejection Rate:':<24s}{'0%':>11s}"
              f"{rejection_rate:>11.0%}{'n/a':>12s}")

        print(f"  {'Grade:':<24s}{report_a.grade:>11s}"
              f"{report_b.grade:>12s}"
              f"{'':>12s}")

        sh_d = _delta(report_a.oos_sharpe, report_b.oos_sharpe, "+.2f")
        print(f"  {'Sharpe:':<24s}{report_a.oos_sharpe:>11.2f}"
              f"{report_b.oos_sharpe:>12.2f}{sh_d:>12s}")

        print("═" * 60)

        # Save A/B results
        ab_results = {
            "v1": {
                "win_rate": report_a.oos_win_rate,
                "profit_factor": report_a.oos_profit_factor,
                "cost_adjusted_pf": capf_a,
                "oos_trades": report_a.total_oos_trades,
                "sharpe": report_a.oos_sharpe,
                "grade": report_a.grade,
            },
            "v2": {
                "win_rate": report_b.oos_win_rate,
                "profit_factor": report_b.oos_profit_factor,
                "cost_adjusted_pf": capf_b,
                "oos_trades": report_b.total_oos_trades,
                "sharpe": report_b.oos_sharpe,
                "grade": report_b.grade,
                "avg_quality_score": round(avg_quality, 2),
                "rejection_rate": round(rejection_rate, 4),
            },
        }
        ab_path = os.path.join(os.path.dirname(RESULTS_PATH), "ab_test_results.json")
        os.makedirs(os.path.dirname(ab_path), exist_ok=True)
        with open(ab_path, "w") as f:
            json.dump(ab_results, f, indent=2)
        print(f"\n  A/B results saved to {ab_path}")


# ═══════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Isolated Variable Diagnostic for XAU/USD ICT WFO"
    )
    parser.add_argument("--td-key", required=True,
                        help="Twelve Data API key")
    parser.add_argument("--candles", type=int, default=1500,
                        help="Number of candles to fetch (default: 1500)")
    parser.add_argument("--timeframe", default="1h",
                        help="Candle timeframe (default: 1h)")
    parser.add_argument("--interval", default="1h",
                        help="Twelve Data interval (default: 1h)")
    parser.add_argument("--start-step", type=int, default=1,
                        help="Resume from this step (reruns baseline first)")
    parser.add_argument("--auto", action="store_true",
                        help="No pausing — run all steps automatically")
    parser.add_argument("--autogluon", action="store_true",
                        help="Use AutoGluon (default: heuristic for speed)")
    parser.add_argument("--ab-test", action="store_true",
                        help="Run A/B comparison between V1 and V2 detectors")

    args = parser.parse_args()

    print("═" * 65)
    print("  ISOLATED VARIABLE DIAGNOSTIC — XAU/USD ICT SYSTEM")
    print("═" * 65)
    print(f"  Fetching {args.candles} candles from Twelve Data...")

    candles = _fetch_candles_twelve_data(
        args.td_key, args.candles, args.interval
    )
    print(f"  Loaded {len(candles)} candles")
    print(f"  Date range: {candles[0].get('datetime', '?')} → "
          f"{candles[-1].get('datetime', '?')}")
    print(f"  Latest close: {candles[-1]['close']:.2f}")

    atr = compute_atr(candles, 14)
    print(f"  Current ATR(14): {atr:.2f}")
    print()

    diagnostic = VariableDiagnostic(
        candles=candles,
        timeframe=args.timeframe,
        auto=args.auto,
        use_autogluon=args.autogluon,
    )

    if args.ab_test:
        diagnostic.run_ab_test()
    else:
        diagnostic.run(start_step=args.start_step)

    print(f"\n  Results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
