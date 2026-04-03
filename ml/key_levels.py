"""ICT Key Levels computation from candle data.

Computes objective price levels used in ICT methodology:
- PDH/PDL: Previous Day High/Low — intraday liquidity targets
- PWH/PWL: Previous Week High/Low — HTF liquidity targets
- PMH/PML: Previous Month High/Low — macro liquidity pools
- Asia Session H/L: London session sweep targets
- Previous Session H/L: Cross-session liquidity reference

All functions are pure — no side effects, no mutation, no I/O.
Candle datetime format: "YYYY-MM-DD HH:MM:SS" (UTC).
"""
from datetime import datetime, timedelta, timezone
from collections import OrderedDict

# Killzone hour ranges (UTC) — mirrors prompts.py KILLZONES
SESSION_HOURS = OrderedDict([
    ("Asian",  (0, 7)),     # 00:00-06:59 UTC
    ("London", (7, 12)),    # 07:00-11:59 UTC
    ("NY_AM",  (12, 16)),   # 12:00-15:59 UTC
    ("NY_PM",  (16, 20)),   # 16:00-19:59 UTC
    ("Off",    (20, 24)),   # 20:00-23:59 UTC
])

# Session precedence for "previous session" lookups
SESSION_ORDER = ["Asian", "London", "NY_AM", "NY_PM", "Off"]


def _compute_equilibrium(high: float | None, low: float | None) -> float | None:
    """Midpoint of a range. Returns None if either input is None."""
    if high is None or low is None:
        return None
    return round((high + low) / 2, 2)


def _parse_dt(dt_str: str) -> datetime:
    """Parse candle datetime string to datetime object."""
    return datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")


def _group_candles_by_date(candles: list[dict]) -> dict[str, list[dict]]:
    """Group candles by the date portion of their datetime string.

    Returns OrderedDict mapping "YYYY-MM-DD" -> list of candles,
    preserving chronological order.
    """
    groups: dict[str, list[dict]] = OrderedDict()
    for c in candles:
        dt_str = c.get("datetime", "")
        if not dt_str:
            continue
        date_key = dt_str.split(" ")[0]
        groups.setdefault(date_key, []).append(c)
    return groups


def _filter_candles_by_hour_range(candles: list[dict],
                                   start_hour: int, end_hour: int,
                                   target_date: str) -> list[dict]:
    """Filter candles to those within a UTC hour range on a specific date.

    Args:
        candles: Candle list with datetime strings.
        start_hour: Inclusive start hour (0-23).
        end_hour: Exclusive end hour (1-24).
        target_date: "YYYY-MM-DD" string.

    Returns:
        Filtered list of candles matching the date + hour window.
    """
    result = []
    for c in candles:
        dt_str = c.get("datetime", "")
        if not dt_str or not dt_str.startswith(target_date):
            continue
        try:
            hour = int(dt_str.split(" ")[1].split(":")[0])
        except (IndexError, ValueError):
            continue
        if start_hour <= hour < end_hour:
            result.append(c)
    return result


def _range_from_candles(candles: list[dict]) -> tuple[float | None, float | None]:
    """Extract high/low from a list of candles.

    Returns (high, low) or (None, None) if empty.
    """
    if not candles:
        return None, None
    high = max(c.get("high", 0) for c in candles)
    low = min(c.get("low", float("inf")) for c in candles)
    if low == float("inf"):
        return None, None
    return high, low


# ── PDH / PDL ──────────────────────────────────────────────

def compute_pdh_pdl(daily_candles: list[dict]) -> dict:
    """Previous Day High/Low from daily candles.

    Uses the second-to-last candle ([-2]) since [-1] is the
    current incomplete day.

    Returns:
        {"high": float|None, "low": float|None, "eq": float|None}
    """
    if not daily_candles or len(daily_candles) < 2:
        return {"high": None, "low": None, "eq": None}

    prev = daily_candles[-2]
    high = prev.get("high")
    low = prev.get("low")
    return {"high": high, "low": low, "eq": _compute_equilibrium(high, low)}


def compute_pdh_pdl_from_intraday(candles: list[dict]) -> dict:
    """Previous Day High/Low aggregated from intraday candles.

    Groups candles by date, takes the last fully complete day
    (second-to-last date group, since the last group may be partial).

    Returns:
        {"high": float|None, "low": float|None, "eq": float|None}
    """
    groups = _group_candles_by_date(candles)
    dates = list(groups.keys())

    if len(dates) < 2:
        return {"high": None, "low": None, "eq": None}

    # Last complete day = second-to-last date
    prev_date = dates[-2]
    high, low = _range_from_candles(groups[prev_date])
    return {"high": high, "low": low, "eq": _compute_equilibrium(high, low)}


# ── PWH / PWL ──────────────────────────────────────────────

def compute_pwh_pwl(weekly_candles: list[dict]) -> dict:
    """Previous Week High/Low from weekly candles.

    Uses [-2] since [-1] is the current incomplete week.

    Returns:
        {"high": float|None, "low": float|None, "eq": float|None}
    """
    if not weekly_candles or len(weekly_candles) < 2:
        return {"high": None, "low": None, "eq": None}

    prev = weekly_candles[-2]
    high = prev.get("high")
    low = prev.get("low")
    return {"high": high, "low": low, "eq": _compute_equilibrium(high, low)}


def compute_pwh_pwl_from_daily(daily_candles: list[dict]) -> dict:
    """Previous Week High/Low aggregated from daily candles.

    Groups daily candles by ISO week number, takes the last
    complete week (second-to-last week group).

    Returns:
        {"high": float|None, "low": float|None, "eq": float|None}
    """
    if not daily_candles:
        return {"high": None, "low": None, "eq": None}

    # Group by (year, iso_week)
    week_groups: dict[tuple, list[dict]] = OrderedDict()
    for c in daily_candles:
        dt_str = c.get("datetime", "")
        if not dt_str:
            continue
        try:
            dt = _parse_dt(dt_str)
            iso = dt.isocalendar()
            key = (iso[0], iso[1])  # (year, week)
        except (ValueError, IndexError):
            continue
        week_groups.setdefault(key, []).append(c)

    weeks = list(week_groups.keys())
    if len(weeks) < 2:
        return {"high": None, "low": None, "eq": None}

    # Second-to-last complete week
    prev_week_key = weeks[-2]
    high, low = _range_from_candles(week_groups[prev_week_key])
    return {"high": high, "low": low, "eq": _compute_equilibrium(high, low)}


# ── PMH / PML ──────────────────────────────────────────────

def compute_pmh_pml(daily_candles: list[dict]) -> dict:
    """Previous Month High/Low from daily candles.

    Groups daily candles by month, takes the last complete month
    (second-to-last month group).

    Returns:
        {"high": float|None, "low": float|None, "eq": float|None}
    """
    if not daily_candles:
        return {"high": None, "low": None, "eq": None}

    # Group by (year, month)
    month_groups: dict[tuple, list[dict]] = OrderedDict()
    for c in daily_candles:
        dt_str = c.get("datetime", "")
        if not dt_str:
            continue
        try:
            parts = dt_str.split(" ")[0].split("-")
            key = (int(parts[0]), int(parts[1]))
        except (IndexError, ValueError):
            continue
        month_groups.setdefault(key, []).append(c)

    months = list(month_groups.keys())
    if len(months) < 2:
        return {"high": None, "low": None, "eq": None}

    prev_month_key = months[-2]
    high, low = _range_from_candles(month_groups[prev_month_key])
    return {"high": high, "low": low, "eq": _compute_equilibrium(high, low)}


# ── Asia Session H/L ───────────────────────────────────────

def compute_asia_session_hl(intraday_candles: list[dict],
                             target_date: str | None = None) -> dict:
    """Asian session High/Low (00:00-07:00 UTC).

    Args:
        intraday_candles: 5min/15min/1h candles in chronological order.
        target_date: "YYYY-MM-DD". Defaults to today UTC.

    Returns:
        {"high": float|None, "low": float|None, "eq": float|None}
    """
    if not intraday_candles:
        return {"high": None, "low": None, "eq": None}

    if target_date is None:
        target_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    start_h, end_h = SESSION_HOURS["Asian"]
    asia_candles = _filter_candles_by_hour_range(
        intraday_candles, start_h, end_h, target_date)

    high, low = _range_from_candles(asia_candles)
    return {"high": high, "low": low, "eq": _compute_equilibrium(high, low)}


# ── Previous Session H/L ───────────────────────────────────

def compute_previous_session_hl(intraday_candles: list[dict],
                                 current_killzone: str) -> dict:
    """Previous killzone session High/Low.

    Session order: Asian -> London -> NY_AM -> NY_PM -> Off.
    If current is Asian, previous is NY_PM from yesterday.

    Returns:
        {"high": float|None, "low": float|None, "eq": float|None,
         "session": str|None}
    """
    empty = {"high": None, "low": None, "eq": None, "session": None}
    if not intraday_candles:
        return empty

    # Determine which session to look up
    if current_killzone not in SESSION_ORDER:
        current_killzone = "Off"

    idx = SESSION_ORDER.index(current_killzone)
    if idx > 0:
        prev_session = SESSION_ORDER[idx - 1]
        same_day = True
    else:
        # Asian -> look at yesterday's NY_PM (or Off)
        prev_session = "NY_PM"
        same_day = False

    # Determine target date
    groups = _group_candles_by_date(intraday_candles)
    dates = list(groups.keys())
    if not dates:
        return empty

    if same_day:
        # Use the last date that has candles
        target_date = dates[-1]
    else:
        # Need yesterday — second-to-last date
        if len(dates) < 2:
            # Try latest date anyway (might have yesterday's candles)
            target_date = dates[-1]
        else:
            target_date = dates[-2]

    start_h, end_h = SESSION_HOURS[prev_session]
    session_candles = _filter_candles_by_hour_range(
        intraday_candles, start_h, end_h, target_date)

    high, low = _range_from_candles(session_candles)
    return {
        "high": high, "low": low,
        "eq": _compute_equilibrium(high, low),
        "session": prev_session if high is not None else None,
    }


# ── Orchestrator ────────────────────────────────────────────

def compute_all_key_levels(daily_candles: list[dict] | None = None,
                            weekly_candles: list[dict] | None = None,
                            intraday_candles: list[dict] | None = None,
                            current_killzone: str = "Off") -> dict:
    """Compute all ICT key levels from available candle data.

    Top-level orchestrator. Tries each level computation with the best
    available data source, falling back gracefully when data is missing.

    Returns dict with keys:
        pdh, pdl, pd_eq, pwh, pwl, pw_eq, pmh, pml, pm_eq,
        asia_high, asia_low, asia_eq,
        prev_session_high, prev_session_low, prev_session_eq,
        prev_session_name, levels_computed
    """
    # PDH/PDL — prefer daily candles, fall back to intraday aggregation
    pd = compute_pdh_pdl(daily_candles) if daily_candles else {"high": None, "low": None, "eq": None}
    if pd["high"] is None and intraday_candles:
        pd = compute_pdh_pdl_from_intraday(intraday_candles)

    # PWH/PWL — prefer weekly candles, fall back to daily aggregation
    pw = compute_pwh_pwl(weekly_candles) if weekly_candles else {"high": None, "low": None, "eq": None}
    if pw["high"] is None and daily_candles:
        pw = compute_pwh_pwl_from_daily(daily_candles)

    # PMH/PML — from daily candles only
    pm = compute_pmh_pml(daily_candles) if daily_candles else {"high": None, "low": None, "eq": None}

    # Asia session H/L — from intraday
    asia = compute_asia_session_hl(intraday_candles) if intraday_candles else {"high": None, "low": None, "eq": None}

    # Previous session H/L — from intraday
    prev_sess = compute_previous_session_hl(intraday_candles, current_killzone) if intraday_candles else {"high": None, "low": None, "eq": None, "session": None}

    result = {
        "pdh": pd["high"], "pdl": pd["low"], "pd_eq": pd["eq"],
        "pwh": pw["high"], "pwl": pw["low"], "pw_eq": pw["eq"],
        "pmh": pm["high"], "pml": pm["low"], "pm_eq": pm["eq"],
        "asia_high": asia["high"], "asia_low": asia["low"], "asia_eq": asia["eq"],
        "prev_session_high": prev_sess["high"],
        "prev_session_low": prev_sess["low"],
        "prev_session_eq": prev_sess["eq"],
        "prev_session_name": prev_sess.get("session"),
    }

    # Count non-None levels (exclude metadata keys)
    level_keys = [k for k in result if k != "prev_session_name"]
    result["levels_computed"] = sum(1 for k in level_keys if result[k] is not None)

    return result


def format_key_levels_for_prompt(levels: dict) -> str:
    """Format key levels dict into a prompt-ready string.

    Only includes levels that were successfully computed.
    Used by prompts.py to inject into Sonnet's analysis prompt.
    """
    if not levels or levels.get("levels_computed", 0) == 0:
        return ""

    lines = ["=== KEY LEVELS (computed from price data) ==="]

    if levels.get("pdh") is not None:
        lines.append(
            f"Previous Day:  H: {levels['pdh']:.2f} | L: {levels['pdl']:.2f} | "
            f"Eq: {levels['pd_eq']:.2f}"
        )
    if levels.get("pwh") is not None:
        lines.append(
            f"Previous Week: H: {levels['pwh']:.2f} | L: {levels['pwl']:.2f} | "
            f"Eq: {levels['pw_eq']:.2f}"
        )
    if levels.get("pmh") is not None:
        lines.append(
            f"Previous Month: H: {levels['pmh']:.2f} | L: {levels['pml']:.2f} | "
            f"Eq: {levels['pm_eq']:.2f}"
        )
    if levels.get("asia_high") is not None:
        lines.append(
            f"Asia Session:  H: {levels['asia_high']:.2f} | L: {levels['asia_low']:.2f} | "
            f"Eq: {levels['asia_eq']:.2f}"
        )
    if levels.get("prev_session_high") is not None:
        name = levels.get("prev_session_name", "?")
        lines.append(
            f"Prev Session ({name}): H: {levels['prev_session_high']:.2f} | "
            f"L: {levels['prev_session_low']:.2f} | Eq: {levels['prev_session_eq']:.2f}"
        )

    lines.append(
        "Use these as liquidity targets, sweep references, and premium/discount anchors."
    )
    lines.append("=== END KEY LEVELS ===")
    return "\n".join(lines)
