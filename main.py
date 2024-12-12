# Crear instancias
from utility.beta import MarketDataServiceWithBeta
from utility.extract import MarketDataRequest, TimeFrame, YahooDataFetcher
from utility.market import MarketDataService, TotalReturnCalculator, MarketStatsCalculator
from utility.risk import RiskStatsCalculator
from utility.time_series import MarketDataServiceWithGarch

data_fetcher = YahooDataFetcher()
return_calculator = TotalReturnCalculator()
stats_calculators = [
    MarketStatsCalculator(),
    RiskStatsCalculator()
]

# Crear servicio
market_service = MarketDataService(
    data_fetcher=data_fetcher,
    return_calculator=return_calculator,
    stats_calculators=stats_calculators
)

# Crear request
request = MarketDataRequest(
    ticker="AAPL",
    timeframe=TimeFrame.ONE_YEAR
)

# Obtener datos y estadísticas
df = market_service.get_market_data(request)
stats = market_service.get_all_stats()
print(stats)



# Crear instancias
data_fetcher = YahooDataFetcher()
return_calculator = TotalReturnCalculator()

# Configurar parámetros GARCH personalizados (opcional)
garch_params = {
    'p': 1,  # orden GARCH
    'q': 1,  # orden ARCH
    'forecast_horizon': 5  # días a pronosticar
}

# Crear servicio con GARCH
market_service = MarketDataServiceWithGarch(
    data_fetcher=data_fetcher,
    return_calculator=return_calculator,
    garch_params=garch_params
)

# Crear request
request = MarketDataRequest(
    ticker="AAPL",
    timeframe=TimeFrame.ONE_YEAR
)

# Obtener datos y estadísticas
df = market_service.get_market_data(request)
garch_stats = market_service.get_risk_stats()

print(f"Volatilidad GARCH actual (anualizada): {garch_stats.garch_volatility:.2%}")
print(f"Pronóstico de volatilidad (anualizada): {garch_stats.forecast_volatility:.2%}")

# Crear instancias
data_fetcher = YahooDataFetcher()
return_calculator = TotalReturnCalculator()

# Crear servicio con GARCH y Beta
market_service = MarketDataServiceWithBeta(
    data_fetcher=data_fetcher,
    return_calculator=return_calculator
)

# Crear request
request = MarketDataRequest(
    ticker="AAPL",
    timeframe=TimeFrame.FIVE_YEARS
)

# Obtener datos y análisis completo
df = market_service.get_market_data(request)
analysis = market_service.get_complete_analysis()

# Acceder a las estadísticas
beta_stats = analysis['beta_stats']
print(f"Beta: {beta_stats['beta']:.2f}")
print(f"Alpha anualizado: {beta_stats['alpha']:.2%}")
print(f"R-cuadrado: {beta_stats['r_squared']:.2%}")
print(f"Tracking Error: {beta_stats['tracking_error']:.2%}")
