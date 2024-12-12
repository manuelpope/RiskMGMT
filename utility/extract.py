from datetime import datetime, timedelta
from enum import Enum
from typing import Protocol, Dict, Optional

import pandas as pd
import yfinance as yf
from pydantic import BaseModel, Field, validator

from utility.risk import RiskStats


# Modelos Pydantic para validación de datos
class TimeFrame(str, Enum):
    ONE_MONTH = "1mo"
    THREE_MONTHS = "3mo"
    SIX_MONTHS = "6mo"
    ONE_YEAR = "1y"
    FIVE_YEARS = "5y"


class MarketDataRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=10)
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    timeframe: Optional[TimeFrame] = None

    @validator('ticker')
    def validate_ticker(cls, v):
        if not v.isalnum():
            raise ValueError('Ticker debe contener solo letras y números')
        return v.upper()

    class Config:
        arbitrary_types_allowed = True


class MarketStats(BaseModel):
    mean_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    positive_days: float


class AllStats(BaseModel):
    market_stats: MarketStats
    risk_stats: RiskStats


# Interfaces
class DataFetcher(Protocol):
    def fetch_data(self, request: MarketDataRequest) -> pd.DataFrame:
        pass




class StatsCalculator(Protocol):
    def calculate_stats(self, data: pd.DataFrame) -> Dict:
        pass


class YahooDataFetcher:
    """Responsable únicamente de obtener datos de Yahoo Finance"""

    def fetch_data(self, request: MarketDataRequest) -> pd.DataFrame:
        stock = yf.Ticker(request.ticker)

        if request.timeframe:
            return stock.history(period=request.timeframe.value)
        else:
            return stock.history(
                start=request.start_date or (datetime.now() - timedelta(days=365)),
                end=request.end_date or datetime.now()
            )
