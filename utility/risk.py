from typing import Dict, Protocol

import numpy as np
import pandas as pd
from pydantic import BaseModel

class ReturnCalculator(Protocol):
    def calculate_returns(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

class RiskStats(BaseModel):
    max_drawdown: float
    volatility: float
    var_95: float
    worst_day: float
    best_day: float


class RiskStatsCalculator:
    """Responsable de calcular estadísticas de riesgo"""

    def calculate_stats(self, data: pd.DataFrame) -> Dict:
        daily_returns = data['Daily_Return'].dropna()
        cumulative_returns = data['Cumulative_Return'].dropna()

        # Calcular volatilidad como el último valor de la ventana móvil
        volatility = daily_returns.rolling(window=20).std().iloc[-1] * np.sqrt(252)

        return RiskStats(
            max_drawdown=float((cumulative_returns /
                                cumulative_returns.cummax() - 1).min()),
            volatility=float(volatility),
            var_95=float(daily_returns.quantile(0.05)),
            worst_day=float(daily_returns.min()),
            best_day=float(daily_returns.max())
        ).dict()
