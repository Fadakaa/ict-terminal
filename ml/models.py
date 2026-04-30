"""Pydantic request/response schemas for the ML prediction API."""
from pydantic import BaseModel
from typing import Optional


class CandleData(BaseModel):
    datetime: str
    open: float
    high: float
    low: float
    close: float


class OrderBlock(BaseModel):
    type: str
    high: float
    low: float
    candleIndex: int
    tf: Optional[str] = None
    strength: str = ""
    note: str = ""


class FVG(BaseModel):
    type: str
    high: float
    low: float
    startIndex: int
    tf: Optional[str] = None
    filled: bool = False
    note: str = ""


class LiquidityLevel(BaseModel):
    type: str
    price: float
    candleIndex: int
    tf: Optional[str] = None
    note: str = ""


class EntryData(BaseModel):
    price: float
    direction: str
    rationale: str = ""


class StopLossData(BaseModel):
    price: float
    rationale: str = ""


class TakeProfitData(BaseModel):
    price: float
    rationale: str = ""
    rr: float = 0.0


class AnalysisJSON(BaseModel):
    bias: str = "neutral"
    summary: str = ""
    orderBlocks: list[OrderBlock] = []
    fvgs: list[FVG] = []
    liquidity: list[LiquidityLevel] = []
    entry: Optional[EntryData] = None
    stopLoss: Optional[StopLossData] = None
    takeProfits: list[TakeProfitData] = []
    killzone: str = ""
    confluences: list[str] = []


class PredictionRequest(BaseModel):
    analysis: AnalysisJSON
    candles: list[CandleData]
    timeframe: str


class PredictionResponse(BaseModel):
    confidence: float
    classification: dict
    suggested_sl: Optional[float] = None
    suggested_tp1: Optional[float] = None
    suggested_tp2: Optional[float] = None
    model_status: str
    training_samples: int
    feature_importances: dict = {}
    # Consensus engine fields (None until sufficient data)
    grade: Optional[str] = None
    blended_confidence: Optional[float] = None
    conservative_sl: Optional[float] = None
    volatility_regime: Optional[str] = None
    bayesian_win_rate: Optional[float] = None
    session: Optional[str] = None
    reasoning: Optional[list[str]] = None
    # Extended ML pipeline fields
    confidence_raw: Optional[float] = None
    confidence_calibrated: Optional[float] = None
    coverage_score: Optional[float] = None
    novelty_flags: Optional[list[str]] = None
    regime_coverage: Optional[str] = None
    regime_adjustment: Optional[float] = None
    defensive_mode: Optional[bool] = None
    data_maturity: Optional[str] = None
    dataset_backing: Optional[dict] = None
    take_trade: Optional[bool] = None
    prior_drift: Optional[dict] = None
    wfo_filter: Optional[dict] = None


class TradeOutcomeRequest(BaseModel):
    setup_id: str
    result: str
    max_favorable_excursion: float = 0
    max_adverse_excursion: float = 0
    pnl_pips: float = 0


class SetupLogRequest(BaseModel):
    setup_id: str
    analysis: AnalysisJSON
    candles: list[CandleData]
    timeframe: str


class ModelStatus(BaseModel):
    classifier_trained: bool
    quantile_trained: bool
    total_trades_logged: int
    completed_trades: int
    dataset_trades: int = 0
    last_trained: Optional[str] = None
    next_retrain_trigger: int


class WFORunRequest(BaseModel):
    candles: list[CandleData]
    timeframe: str = "1h"
    train_window: int = 500
    test_window: int = 100
    step_size: int = 50
    no_autogluon: bool = False
    skip_ingest: bool = False
