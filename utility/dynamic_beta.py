from arch import arch_model
import numpy as np
from pydantic import BaseModel
from scipy.optimize import minimize
from typing import Tuple, List, Dict, Optional
import warnings

from utility.beta import SPYDataProvider
from utility.extract import DataFetcher
from utility.risk import ReturnCalculator
from utility.time_series import MarketDataServiceWithGarch
import pandas  as pd
warnings.filterwarnings('ignore')


class DynamicBetaStats(BaseModel):
    current_beta: float
    average_beta: float
    beta_volatility: float
    min_beta: float
    max_beta: float
    current_correlation: float
    forecast_beta: float
    conditional_volatility: Dict[str, float]


class DCCGARCHBetaCalculator:
    """
    Calcula betas dinámicos usando DCC-GARCH
    """

    def __init__(self, forecast_horizon: int = 1):
        self.forecast_horizon = forecast_horizon
        self.market_provider = SPYDataProvider()

    def _fit_univariate_garch(self, returns: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ajusta modelo GARCH(1,1) univariado
        """
        model = arch_model(returns, vol='Garch', p=1, q=1, mean='Zero', dist='normal')
        result = model.fit(disp='off')

        # Obtener volatilidades condicionales y residuos estandarizados
        h = result.conditional_volatility
        eps = returns / h

        return eps, h

    def _fit_dcc(self, eps1: np.ndarray, eps2: np.ndarray,
                 n_obs: int) -> Tuple[np.ndarray, float, float]:
        """
        Ajusta el modelo DCC a los residuos estandarizados
        """

        def _log_likelihood(params):
            a, b = params
            qt = np.zeros((n_obs, 2, 2))
            rt = np.zeros((n_obs, 2, 2))

            # Matriz de correlación incondicional
            Q_bar = np.corrcoef(eps1, eps2)
            qt[0] = Q_bar

            # Iteración DCC
            for t in range(1, n_obs):
                eps_t = np.array([[eps1[t - 1]], [eps2[t - 1]]])
                qt[t] = (1 - a - b) * Q_bar + a * (eps_t @ eps_t.T) + b * qt[t - 1]
                q_diag = np.sqrt(np.diag(qt[t]))
                rt[t] = qt[t] / (q_diag @ q_diag.T)

            # Log-likelihood
            ll = -0.5 * sum(2 * np.log(np.diag(rt[t])) +
                            np.array([eps1[t], eps2[t]]) @
                            np.linalg.inv(rt[t]) @
                            np.array([[eps1[t]], [eps2[t]]])
                            for t in range(n_obs))
            return -ll

        # Optimización
        result = minimize(_log_likelihood, x0=[0.01, 0.97],
                          bounds=((0, 1), (0, 1)),
                          constraints={'type': 'ineq',
                                       'fun': lambda x: 1 - sum(x)})

        a, b = result.x

        # Calcular correlaciones dinámicas
        Q_bar = np.corrcoef(eps1, eps2)
        qt = np.zeros((n_obs, 2, 2))
        rt = np.zeros(n_obs)
        qt[0] = Q_bar

        for t in range(1, n_obs):
            eps_t = np.array([[eps1[t - 1]], [eps2[t - 1]]])
            qt[t] = (1 - a - b) * Q_bar + a * (eps_t @ eps_t.T) + b * qt[t - 1]
            q_diag = np.sqrt(np.diag(qt[t]))
            rt[t] = qt[t][0, 1] / (q_diag[0] * q_diag[1])

        return rt, a, b

    def calculate_dynamic_beta(self, stock_data: pd.DataFrame) -> Dict:
        """
        Calcula beta dinámico usando DCC-GARCH
        """
        # Obtener datos de mercado
        market_data = self.market_provider.get_market_data(
            start_date=stock_data.index[0],
            end_date=stock_data.index[-1]
        )

        # Preparar retornos
        stock_returns = stock_data['Daily_Return'].dropna().values
        market_returns = market_data['Close'].pct_change().dropna().values

        # Ajustar GARCH univariado
        eps_stock, h_stock = self._fit_univariate_garch(stock_returns)
        eps_market, h_market = self._fit_univariate_garch(market_returns)

        # Ajustar DCC
        n_obs = len(eps_stock)
        rho, a, b = self._fit_dcc(eps_stock, eps_market, n_obs)

        # Calcular betas dinámicos
        dynamic_betas = rho * (h_stock / h_market)

        # Pronosticar beta
        last_beta = dynamic_betas[-1]
        forecast_beta = last_beta  # Podríamos implementar un pronóstico más sofisticado

        return DynamicBetaStats(
            current_beta=float(last_beta),
            average_beta=float(np.mean(dynamic_betas)),
            beta_volatility=float(np.std(dynamic_betas)),
            min_beta=float(np.min(dynamic_betas)),
            max_beta=float(np.max(dynamic_betas)),
            current_correlation=float(rho[-1]),
            forecast_beta=float(forecast_beta),
            conditional_volatility={
                'stock': float(h_stock[-1]),
                'market': float(h_market[-1])
            }
        ).dict()


class MarketDataServiceWithDynamicBeta(MarketDataServiceWithGarch):
    """Servicio de datos de mercado con beta dinámico"""

    def __init__(
            self,
            data_fetcher: DataFetcher,
            return_calculator: ReturnCalculator,
            garch_params: Optional[Dict] = None
    ):
        super().__init__(data_fetcher, return_calculator, garch_params)
        self.dynamic_beta_calculator = DCCGARCHBetaCalculator()

    def get_dynamic_beta_stats(self) -> DynamicBetaStats:
        """Obtiene estadísticas de beta dinámico"""
        if self.data is None:
            raise ValueError("Debe obtener los datos primero usando get_market_data()")

        return DynamicBetaStats(**self.dynamic_beta_calculator.calculate_dynamic_beta(self.data))

    def get_complete_analysis(self) -> Dict:
        """Obtiene análisis completo incluyendo beta dinámico y GARCH"""
        return {
            'dynamic_beta_stats': self.get_dynamic_beta_stats().dict(),
            'risk_stats': self.get_risk_stats().dict()
        }