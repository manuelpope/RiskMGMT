from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Tuple, Protocol
import numpy as np
from pydantic import BaseModel
from scipy import stats
import pandas as pd

from utility.extract import YahooDataFetcher, MarketDataRequest, DataFetcher
from utility.risk import ReturnCalculator
from utility.time_series import MarketDataServiceWithGarch


class MarketBetaStats(BaseModel):
    beta: float
    alpha: float
    r_squared: float
    p_value: float
    std_error: float
    correlation: float
    tracking_error: float


class MarketDataProvider(Protocol):
    """Interface para proveedores de datos de mercado"""

    def get_market_data(self, start_date: Optional[datetime] = None,
                        end_date: Optional[datetime] = None) -> pd.DataFrame:
        pass


class SPYDataProvider:
    """Proveedor de datos para S&P 500 usando SPY como proxy"""

    def __init__(self):
        self.ticker = "SPY"
        self.fetcher = YahooDataFetcher()

    def get_market_data(self, start_date: Optional[datetime] = None,
                        end_date: Optional[datetime] = None) -> pd.DataFrame:
        request = MarketDataRequest(
            ticker=self.ticker,
            start_date=start_date,
            end_date=end_date
        )
        return self.fetcher.fetch_data(request)


class BetaCalculator:
    """Calcula métricas de beta respecto al mercado"""

    def __init__(self, market_provider: MarketDataProvider = None):
        """
        Inicializa el calculador de beta

        Parameters:
        -----------
        market_provider : MarketDataProvider
            Proveedor de datos de mercado (por defecto SPY)
        """
        self.market_provider = market_provider or SPYDataProvider()

    def _prepare_data(self, stock_data: pd.DataFrame,
                      market_data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Prepara y alinea los datos de acciones y mercado"""
        # Calcular retornos del mercado
        market_returns = market_data['Close'].pct_change().dropna()

        # Obtener retornos de la acción
        stock_returns = stock_data['Daily_Return'].dropna()

        # Alinear fechas
        common_dates = stock_returns.index.intersection(market_returns.index)
        stock_returns = stock_returns[common_dates]
        market_returns = market_returns[common_dates]

        return stock_returns, market_returns

    def calculate_beta_stats(self, stock_data: pd.DataFrame) -> Dict:
        """
        Calcula estadísticas completas de beta

        Parameters:
        -----------
        stock_data : pd.DataFrame
            DataFrame con los datos de la acción incluyendo 'Daily_Return'

        Returns:
        --------
        Dict con estadísticas de beta
        """
        # Obtener datos de mercado para el mismo período
        market_data = self.market_provider.get_market_data(
            start_date=stock_data.index[0],
            end_date=stock_data.index[-1]
        )

        # Preparar datos
        stock_returns, market_returns = self._prepare_data(stock_data, market_data)

        # Realizar regresión
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            market_returns, stock_returns
        )

        # Calcular tracking error (volatilidad del error de la regresión)
        predicted_returns = market_returns * slope + intercept
        tracking_error = np.std(stock_returns - predicted_returns) * np.sqrt(252)

        # Calcular correlación
        correlation = np.corrcoef(market_returns, stock_returns)[0, 1]

        return MarketBetaStats(
            beta=float(slope),
            alpha=float(intercept * 252),  # Anualizar alpha
            r_squared=float(r_value ** 2),
            p_value=float(p_value),
            std_error=float(std_err),
            correlation=float(correlation),
            tracking_error=float(tracking_error)
        ).dict()


class MarketDataServiceWithBeta(MarketDataServiceWithGarch):
    """Servicio de datos de mercado que incluye análisis de beta"""

    def __init__(
            self,
            data_fetcher: DataFetcher,
            return_calculator: ReturnCalculator,
            garch_params: Optional[Dict] = None,
            market_provider: MarketDataProvider = None
    ):
        super().__init__(data_fetcher, return_calculator, garch_params)
        self.beta_calculator = BetaCalculator(market_provider)

    def get_beta_stats(self) -> MarketBetaStats:
        """Obtiene estadísticas de beta"""
        if self.data is None:
            raise ValueError("Debe obtener los datos primero usando get_market_data()")

        return MarketBetaStats(**self.beta_calculator.calculate_beta_stats(self.data))

    def get_complete_analysis(self) -> Dict:
        """Obtiene análisis completo incluyendo beta y GARCH"""
        return {
            'beta_stats': self.get_beta_stats().dict(),
            'risk_stats': self.get_risk_stats().dict()
        }