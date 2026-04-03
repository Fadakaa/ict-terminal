"""Server-side annotated candlestick chart renderer for Claude's visual analysis.

Generates PNG images with ICT overlays (OBs, FVGs, liquidity, dealing range)
using matplotlib/mplfinance.  Sent to Claude as multimodal image input so it
can read chart structure the same way a human ICT trader would.
"""
import io
import base64
import logging
from datetime import datetime

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import numpy as np

from ml.features import detect_order_blocks, detect_fvgs, detect_liquidity

logger = logging.getLogger(__name__)

# ── chart style ──────────────────────────────────────────────────────
BG_COLOR = "#0a0a1a"
GRID_COLOR = "#1a1a2e"
BULL_COLOR = "#26a69a"
BEAR_COLOR = "#ef5350"
OB_BULL_COLOR = "#26a69a"
OB_BEAR_COLOR = "#ef5350"
FVG_BULL_COLOR = "#64b5f6"
FVG_BEAR_COLOR = "#ffa726"
BSL_COLOR = "#f5c842"
SSL_COLOR = "#ff6b6b"
DR_COLOR = "#ce93d8"
TEXT_COLOR = "#cccccc"


def _parse_dt(dt_str: str) -> datetime:
    """Parse candle datetime from either 'T' or space-separated format."""
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%d %H:%M:%S.%f"):
        try:
            return datetime.strptime(dt_str, fmt)
        except ValueError:
            continue
    # Fallback — strip fractional seconds
    try:
        return datetime.fromisoformat(dt_str.replace("Z", ""))
    except Exception:
        return datetime.now()


def _compute_atr(candles: list[dict], period: int = 14) -> float:
    """Simple ATR calculation from candle dicts."""
    trs = []
    for i in range(1, len(candles)):
        h = float(candles[i]["high"])
        l = float(candles[i]["low"])
        pc = float(candles[i - 1]["close"])
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    if len(trs) < period:
        return sum(trs) / len(trs) if trs else 1.0
    return sum(trs[-period:]) / period


def render_chart(candles_1h: list[dict],
                 candles_4h: list[dict] | None = None,
                 timeframe: str = "1h",
                 width: int = 1400, height: int = 800) -> bytes:
    """Render annotated ICT candlestick chart as PNG bytes.

    Args:
        candles_1h: Primary timeframe candles (OHLC dicts)
        candles_4h: Optional HTF candles for dealing range
        timeframe: Label for chart title
        width/height: Image dimensions in pixels

    Returns:
        PNG image as bytes
    """
    candles = candles_1h[-60:] if len(candles_1h) > 60 else candles_1h
    if not candles:
        return b""

    # Parse data
    dates = [_parse_dt(c.get("datetime", "")) for c in candles]
    opens = np.array([float(c["open"]) for c in candles])
    highs = np.array([float(c["high"]) for c in candles])
    lows = np.array([float(c["low"]) for c in candles])
    closes = np.array([float(c["close"]) for c in candles])

    atr = _compute_atr(candles)

    # Detect ICT structures
    obs = detect_order_blocks(candles, atr)
    fvgs = detect_fvgs(candles)
    liquidity = detect_liquidity(candles, window=5)

    # 4H dealing range
    dr_high = dr_low = None
    if candles_4h and len(candles_4h) >= 5:
        recent_4h = candles_4h[-10:]
        dr_high = max(float(c["high"]) for c in recent_4h)
        dr_low = min(float(c["low"]) for c in recent_4h)

    # ── Create figure ────────────────────────────────────────────
    dpi = 100
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    n = len(candles)
    x = np.arange(n)

    # Candle width
    cw = 0.6

    # ── Draw dealing range (behind everything) ───────────────────
    if dr_high is not None and dr_low is not None:
        ax.axhspan(dr_low, dr_high, alpha=0.06, color=DR_COLOR, zorder=0)
        ax.axhline(dr_high, color=DR_COLOR, linestyle="--", linewidth=0.8,
                    alpha=0.5, zorder=1)
        ax.axhline(dr_low, color=DR_COLOR, linestyle="--", linewidth=0.8,
                    alpha=0.5, zorder=1)
        eq = (dr_high + dr_low) / 2
        ax.axhline(eq, color=DR_COLOR, linestyle=":", linewidth=0.6,
                    alpha=0.35, zorder=1)
        ax.text(n + 0.5, dr_high, "DR HIGH", fontsize=6, color=DR_COLOR,
                alpha=0.7, va="center")
        ax.text(n + 0.5, dr_low, "DR LOW", fontsize=6, color=DR_COLOR,
                alpha=0.7, va="center")
        ax.text(n + 0.5, eq, "EQ", fontsize=6, color=DR_COLOR,
                alpha=0.5, va="center")

    # ── Draw order blocks ────────────────────────────────────────
    for ob in obs[-6:]:  # last 6 to avoid clutter
        idx = ob["index"]
        color = OB_BULL_COLOR if ob["type"] == "bullish" else OB_BEAR_COLOR
        rect = mpatches.FancyBboxPatch(
            (idx - cw / 2, ob["low"]), n - idx + cw, ob["high"] - ob["low"],
            boxstyle="round,pad=0.1", facecolor=color, alpha=0.10,
            edgecolor=color, linewidth=0.7, linestyle="--", zorder=2,
        )
        ax.add_patch(rect)
        label = "BULL OB" if ob["type"] == "bullish" else "BEAR OB"
        ax.text(idx, ob["high"] + atr * 0.1, label, fontsize=5,
                color=color, alpha=0.7, zorder=3)

    # ── Draw FVGs ────────────────────────────────────────────────
    for fvg in fvgs[-6:]:
        idx = fvg["index"]
        color = FVG_BULL_COLOR if fvg["type"] == "bullish" else FVG_BEAR_COLOR
        rect = mpatches.Rectangle(
            (idx - cw / 2, fvg["low"]), n - idx + cw, fvg["high"] - fvg["low"],
            facecolor=color, alpha=0.08, edgecolor=color,
            linewidth=0.5, linestyle="--", zorder=2,
        )
        ax.add_patch(rect)
        ax.text(idx, fvg["high"] + atr * 0.05, "FVG", fontsize=5,
                color=color, alpha=0.65, zorder=3)

    # ── Draw liquidity levels ────────────────────────────────────
    for liq in liquidity[-8:]:
        color = BSL_COLOR if liq["type"] == "buyside" else SSL_COLOR
        label = "BSL" if liq["type"] == "buyside" else "SSL"
        ax.axhline(liq["price"], color=color, linestyle="--", linewidth=0.7,
                    alpha=0.5, zorder=2)
        ax.text(n + 0.5, liq["price"], label, fontsize=5, color=color,
                alpha=0.7, va="center", zorder=3)

    # ── Draw candlesticks ────────────────────────────────────────
    for i in range(n):
        color = BULL_COLOR if closes[i] >= opens[i] else BEAR_COLOR
        # Wick
        ax.plot([x[i], x[i]], [lows[i], highs[i]], color=color,
                linewidth=0.7, alpha=0.85, zorder=4)
        # Body
        body_low = min(opens[i], closes[i])
        body_high = max(opens[i], closes[i])
        body_h = body_high - body_low
        if body_h < atr * 0.01:
            body_h = atr * 0.01  # minimum visible body
        rect = mpatches.Rectangle(
            (x[i] - cw / 2, body_low), cw, body_h,
            facecolor=color, edgecolor=color, linewidth=0.5,
            alpha=0.88, zorder=5,
        )
        ax.add_patch(rect)

    # ── Axes formatting ──────────────────────────────────────────
    # X-axis: show datetime labels
    label_step = max(1, n // 10)
    tick_positions = list(range(0, n, label_step))
    tick_labels = [dates[i].strftime("%m/%d %H:%M") for i in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=6, color=TEXT_COLOR, rotation=30)

    ax.set_xlim(-1, n + 4)
    price_range = highs.max() - lows.min()
    ax.set_ylim(lows.min() - price_range * 0.03,
                highs.max() + price_range * 0.05)

    ax.yaxis.tick_right()
    ax.tick_params(axis="y", labelsize=7, colors=TEXT_COLOR)
    ax.grid(True, alpha=0.15, color=GRID_COLOR, linewidth=0.5)

    # Title
    last_dt = dates[-1].strftime("%Y-%m-%d %H:%M UTC")
    ax.set_title(f"XAU/USD  {timeframe}  —  {last_dt}",
                 fontsize=9, color=TEXT_COLOR, pad=8)

    # Legend
    legend_items = []
    if obs:
        legend_items.append(mpatches.Patch(facecolor=OB_BULL_COLOR, alpha=0.3,
                                           label=f"Order Blocks ({len(obs)})"))
    if fvgs:
        legend_items.append(mpatches.Patch(facecolor=FVG_BULL_COLOR, alpha=0.3,
                                           label=f"FVGs ({len(fvgs)})"))
    if liquidity:
        legend_items.append(mpatches.Patch(facecolor=BSL_COLOR, alpha=0.5,
                                           label=f"Liquidity ({len(liquidity)})"))
    if dr_high is not None:
        legend_items.append(mpatches.Patch(facecolor=DR_COLOR, alpha=0.3,
                                           label="4H Dealing Range"))
    if legend_items:
        leg = ax.legend(handles=legend_items, loc="upper left", fontsize=6,
                        facecolor=BG_COLOR, edgecolor=GRID_COLOR,
                        labelcolor=TEXT_COLOR)
        leg.get_frame().set_alpha(0.8)

    plt.tight_layout()

    # ── Export to PNG bytes ───────────────────────────────────────
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=BG_COLOR, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def render_chart_base64(candles_1h: list[dict],
                        candles_4h: list[dict] | None = None,
                        timeframe: str = "1h") -> str:
    """Render chart and return as base64-encoded PNG string."""
    png_bytes = render_chart(candles_1h, candles_4h, timeframe)
    return base64.standard_b64encode(png_bytes).decode("ascii")
