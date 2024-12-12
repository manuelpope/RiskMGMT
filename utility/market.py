from typing import List, Dict

import numpy as np
import pandas as pd

from utility.extract import  AllStats, MarketDataRequest, DataFetcher, \
    StatsCalculator, MarketStats
from utility.risk import RiskStatsCalculator, RiskStats, ReturnCalculator


class MarketDataService:
    """Clase principal que coordina los servicios"""

    def __init__(
            self,
            data_fetcher: DataFetcher,
            return_calculator: ReturnCalculator,
            stats_calculators: List[StatsCalculator]
    ):
        self.data_fetcher = data_fetcher
        self.return_calculator = return_calculator
        self.stats_calculators = stats_calculators
        self.data = None

    def get_market_data(self, request: MarketDataRequest) -> pd.DataFrame:
        self.data = self.data_fetcher.fetch_data(request)
        self.data = self.return_calculator.calculate_returns(self.data)
        return self.data

    def get_all_stats(self) -> AllStats:
        if self.data is None:
            raise ValueError("Debe obtener los datos primero usando get_market_data()")

        market_stats = {}
        risk_stats = {}

        for calculator in self.stats_calculators:
            if isinstance(calculator, MarketStatsCalculator):
                market_stats = calculator.calculate_stats(self.data)
            elif isinstance(calculator, RiskStatsCalculator):
                risk_stats = calculator.calculate_stats(self.data)

        return AllStats(
            market_stats=MarketStats(**market_stats),
            risk_stats=RiskStats(**risk_stats)
        )


class MarketStatsCalculator:
    """Responsable de calcular estadísticas de mercado"""

    def calculate_stats(self, data: pd.DataFrame) -> Dict:
        daily_returns = data['Daily_Return'].dropna()

        return MarketStats(
            mean_return=float(daily_returns.mean()),
            annualized_return=float(daily_returns.mean() * 252),
            annualized_volatility=float(daily_returns.std() * np.sqrt(252)),
            sharpe_ratio=float((daily_returns.mean() * 252) /
                               (daily_returns.std() * np.sqrt(252))),
            positive_days=float((daily_returns > 0).sum() / len(daily_returns))
        ).dict()


class TotalReturnCalculator:
    """Responsable de calcular retornos incluyendo dividendos"""

    def calculate_returns(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df['Daily_Return'] = df['Close'].pct_change()
        df['Total_Return'] = df['Daily_Return'] + df['Dividends'] / df['Close'].shift(1)
        df['Cumulative_Return'] = (1 + df['Daily_Return']).cumprod()
        return df
