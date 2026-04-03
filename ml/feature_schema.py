"""Single source of truth for ML feature column names.

Every function that produces or consumes feature dicts MUST use these names.
Imported by: features.py, claude_bridge.py, training.py, backfill.py
"""

# All 58 feature columns in canonical order
FEATURE_COLUMNS = [
    # Order Blocks (7)
    "ob_count",
    "ob_bullish_count",
    "ob_bearish_count",
    "ob_strongest_strength",
    "ob_nearest_distance_atr",
    "ob_avg_size_atr",
    "ob_alignment",
    # FVGs (5)
    "fvg_count",
    "fvg_unfilled_count",
    "fvg_nearest_distance_atr",
    "fvg_avg_size_atr",
    "fvg_alignment",
    # Liquidity (4)
    "liq_buyside_count",
    "liq_sellside_count",
    "liq_nearest_target_distance_atr",
    "liq_nearest_threat_distance_atr",
    # Trade Setup (6)
    "risk_reward_tp1",
    "risk_reward_tp2",
    "sl_distance_atr",
    "tp1_distance_atr",
    "entry_direction",
    "bias_direction_match",
    # Confluence (4)
    "num_confluences",
    "has_ob_fvg_overlap",
    "killzone_encoded",
    "timeframe_encoded",
    # Price Action from candles (6)
    "atr_14",
    "price_vs_20sma",
    "recent_volatility_ratio",
    "last_candle_body_atr",
    "trend_strength",
    "session_hour",
    # ICT context from Claude analysis (4)
    "premium_discount_encoded",
    "p3_phase_encoded",
    "setup_quality_encoded",
    "claude_direction_encoded",
    # Intermarket (4)
    "gold_dxy_corr_20",
    "gold_dxy_diverging",
    "dxy_range_position",
    "yield_direction",
    # Regime (1) — 5-state structural label. NaN for rows without it.
    "volatility_regime",
    # Entry zone placement (2) — 0.0=shallow, 1.0=deep; zone size in ATR
    "entry_zone_position",
    "entry_zone_size_atr",
    # HTF context from Claude's analysis (5) — dealing range + structure
    "htf_bias_encoded",           # bullish=1, bearish=-1, neutral=0
    "htf_sweep_encoded",          # bsl=1 (bullish signal), ssl=-1 (bearish signal), none=0
    "dealing_range_position",     # 0.0=range low, 1.0=range high (price within HTF range)
    "htf_structure_alignment",    # 1=HTF+LTF agree, 0=neutral, -1=conflict
    "htf_displacement_quality",   # 1=strong displacement formed OB, 0=weak/none
    # Narrative state (4) — thesis maturity + conviction
    "thesis_confidence",          # 0.0-1.0 bias_confidence from narrative state
    "p3_progress_encoded",        # early=1, mid=2, late=3, none=0
    "thesis_scan_count",          # how many scans this thesis has survived
    "opus_sonnet_agreement",      # 1=agree on direction, 0=disagree or no Opus
    # Key level proximity (6) — ATR-normalised distance to ICT levels
    "price_vs_pdh_atr",           # (price - PDH) / ATR; positive = above PDH
    "price_vs_pdl_atr",           # (price - PDL) / ATR; positive = above PDL
    "price_vs_pwh_atr",           # (price - PWH) / ATR
    "price_vs_pwl_atr",           # (price - PWL) / ATR
    "price_vs_asia_high_atr",     # (price - Asia H) / ATR
    "price_vs_asia_low_atr",      # (price - Asia L) / ATR
]

FEATURE_SET = set(FEATURE_COLUMNS)

# Threshold for "rich" row (used by training.py)
RICH_FEATURE_THRESHOLD = 20
RICH_FEATURE_MIN_FOR_MULTI3 = 100
