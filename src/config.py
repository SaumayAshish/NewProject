"""Project configuration and default parameters."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

TODAY = date.today()
END_DATE = TODAY.isoformat()
START_DATE = (TODAY - timedelta(days=365 * 5)).isoformat()

UNIVERSE_SOURCE = "sp500"
TICKER_CSV_PATH = "data/universe/tickers.csv"

MIN_MARKET_CAP = 1_000_000_000
MAX_MISSING_RATIO = 0.25
MIN_HISTORY_DAYS = 500
MIN_AVG_DOLLAR_VOLUME = 5_000_000
LOOKBACK_DAYS = 252 * 3

TOP_N = 12
RISK_FREE_RATE = 0.03
WEIGHT_BOUNDS = (0.0, 0.15)
SECTOR_LIMITS = {"default_upper": 0.35}

VALUE_WEIGHTS = {
    "trailing_pe": 0.3,
    "forward_pe": 0.2,
    "price_to_book": 0.2,
    "price_to_sales": 0.15,
    "ev_to_ebitda": 0.15,
}
QUALITY_WEIGHTS = {
    "roe": 0.45,
    "current_ratio": 0.2,
    "debt_to_equity": 0.35,
}
TREND_WEIGHTS = {
    "ret_6m": 0.4,
    "ret_12m": 0.4,
    "ma_signal": 0.2,
}

COMPOSITE_WEIGHTS = {
    "value_score": 0.40,
    "quality_score": 0.30,
    "trend_score": 0.20,
    "risk_penalty": -0.10,
}
FINAL_SCORE_WEIGHTS = {
    "composite": 0.75,
    "trust": 0.25,
}


@dataclass(frozen=True)
class RegimeWeights:
    value: float
    quality: float
    trend: float
    risk_penalty: float


REGIME_SCORE_WEIGHTS = {
    "bull": RegimeWeights(value=0.30, quality=0.25, trend=0.35, risk_penalty=-0.10),
    "bear": RegimeWeights(value=0.35, quality=0.40, trend=0.10, risk_penalty=-0.15),
    "volatile": RegimeWeights(value=0.30, quality=0.30, trend=0.15, risk_penalty=-0.25),
}
