"""Market-data providers and the local store.

Provider-native structures (yfinance DataFrames, Twelve Data JSON) must not
escape this package (§3).
"""

from app.data.base import MarketDataProvider
from app.data.budget import BudgetDecision, ProviderBudget, RequestPriority
from app.data.factory import daily_provider_chain, resolve_provider
from app.data.mock_provider import MockMarketDataProvider
from app.data.normalization import (
    denominated_currency,
    infer_price_unit,
    is_minor_unit,
    major_unit_for,
    normalise_optional,
    normalise_price,
)
from app.data.store import INTERVAL_DURATION, CandleStore, annualisation_factor
from app.data.types import (
    Candle,
    Fundamentals,
    ProviderDataQualityError,
    ProviderError,
    ProviderMapping,
    ProviderQuotaExceededError,
    ProviderSymbolNotFoundError,
    ProviderUnavailableError,
    Quote,
)
from app.data.yfinance_provider import YFinanceProvider

__all__ = [
    "INTERVAL_DURATION",
    "BudgetDecision",
    "Candle",
    "CandleStore",
    "Fundamentals",
    "MarketDataProvider",
    "MockMarketDataProvider",
    "ProviderBudget",
    "ProviderDataQualityError",
    "ProviderError",
    "ProviderMapping",
    "ProviderQuotaExceededError",
    "ProviderSymbolNotFoundError",
    "ProviderUnavailableError",
    "Quote",
    "RequestPriority",
    "YFinanceProvider",
    "annualisation_factor",
    "daily_provider_chain",
    "denominated_currency",
    "infer_price_unit",
    "is_minor_unit",
    "major_unit_for",
    "normalise_optional",
    "normalise_price",
    "resolve_provider",
]
