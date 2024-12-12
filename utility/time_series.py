from arch import arch_model
import numpy as np
from pydantic import BaseModel
from typing import Dict, Optional
import pandas as pd

from utility.extract import DataFetcher
from utility.market import MarketDataService
from utility.risk import ReturnCalculator


class GarchRiskStats(BaseModel):
    garch_volatility: float
    forecast_volatility: float
    var_95: float
    worst_day: float
    best_day: float
    max_drawdown: float


class GarchRiskCalculator:
    """Calculador de riesgo usando modelo GARCH(1,1)"""

    def __init__(self, p: int = 1, q: int = 1, forecast_horizon: int = 1):
        """
        Inicializa el calculador GARCH

        Parameters:
        -----------
        p : int
            Orden del término GARCH (volatilidad)
        q : int
            Orden del término ARCH (residuos)
        forecast_horizon : int
            Horizonte de predicción en días
        """
        self.p = p
        self.q = q
        self.forecast_horizon = forecast_horizon

    def fit_garch(self, returns: np.ndarray) -> tuple:
        """
        Ajusta el modelo GARCH a los retornos

        Returns:
        --------
        tuple: (volatilidad actual, volatilidad pronosticada)
        """
        # Remover valores nulos y convertir a numpy array
        clean_returns = returns.dropna().values * 100  # convertir a porcentaje

        # Ajustar modelo GARCH
        model = arch_model(
            clean_returns,
            vol='Garch',
            p=self.p,
            q=self.q,
            mean='Zero',
            dist='normal'
        )

        # Entrenar modelo
        results = model.fit(disp='off')

        # Obtener volatilidad actual (último valor)
        current_vol = np.sqrt(results.conditional_volatility[-1])

        # Pronosticar volatilidad
        forecast = results.forecast(horizon=self.forecast_horizon)
        forecast_vol = np.sqrt(forecast.variance.values[-1, 0])

        return current_vol / 100, forecast_vol / 100  # convertir de vuelta a decimal

    def calculate_stats(self, data: pd.DataFrame) -> Dict:
        """Calcula estadísticas de riesgo usando GARCH"""
        daily_returns = data['Daily_Return'].dropna()
        cumulative_returns = data['Cumulative_Return'].dropna()

        # Calcular volatilidades GARCH
        current_vol, forecast_vol = self.fit_garch(daily_returns)

        # Anualizar volatilidades
        annualized_current_vol = current_vol * np.sqrt(252)
        annualized_forecast_vol = forecast_vol * np.sqrt(252)

        return GarchRiskStats(
            garch_volatility=float(annualized_current_vol),
            forecast_volatility=float(annualized_forecast_vol),
            var_95=float(daily_returns.quantile(0.05)),
            worst_day=float(daily_returns.min()),
            best_day=float(daily_returns.max()),
            max_drawdown=float((cumulative_returns /
                                cumulative_returns.cummax() - 1).min())
        ).dict()


# Actualizar el MarketDataService para usar GARCH
class MarketDataServiceWithGarch(MarketDataService):
    """Extensión del servicio de datos de mercado que incluye análisis GARCH"""

    def __init__(
            self,
            data_fetcher: DataFetcher,
            return_calculator: ReturnCalculator,
            garch_params: Optional[Dict] = None
    ):
        self.data_fetcher = data_fetcher
        self.return_calculator = return_calculator
        self.garch_calculator = GarchRiskCalculator(
            **(garch_params or {'p': 1, 'q': 1, 'forecast_horizon': 1})
        )
        self.data = None

    def get_risk_stats(self) -> GarchRiskStats:
        """Obtiene estadísticas de riesgo usando GARCH"""
        if self.data is None:
            raise ValueError("Debe obtener los datos primero usando get_market_data()")

        return GarchRiskStats(**self.garch_calculator.calculate_stats(self.data))