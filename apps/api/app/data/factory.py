"""Market-data provider selection and the daily-data fallback chain (§4)."""

from __future__ import annotations

import structlog

from app.config import Settings, get_settings
from app.data.base import MarketDataProvider
from app.data.mock_provider import MockMarketDataProvider
from app.data.yfinance_provider import YFinanceProvider
from app.models.enums import ProviderKind

log = structlog.get_logger(__name__)


def resolve_provider(kind: ProviderKind, settings: Settings | None = None) -> MarketDataProvider:
    settings = settings or get_settings()

    match kind:
        case ProviderKind.MOCK:
            return MockMarketDataProvider()

        case ProviderKind.YFINANCE:
            return YFinanceProvider(
                max_concurrency=settings.yfinance_max_concurrency,
                backoff_base_seconds=settings.yfinance_backoff_base_seconds,
                max_retries=settings.yfinance_max_retries,
                batch_size=settings.yfinance_batch_size,
            )

        case ProviderKind.TWELVE_DATA:
            # Arrives in Phase 4 with the 15-minute S&P strategy, which is the
            # only consumer that needs US intraday. Raising keeps the gap
            # explicit rather than silently serving daily bars to a 15m
            # strategy — a substitution that would look like it worked.
            raise NotImplementedError(
                "TwelveDataProvider arrives in Phase 4 (S&P 500 15-minute strategy). "
                "Daily data comes from yfinance today."
            )

        case ProviderKind.EODHD:
            raise NotImplementedError(
                "EODHDProvider arrives in Phase 2 as a verification and gap-fill source."
            )

    raise ValueError(f"Unknown provider kind: {kind}")


def daily_provider_chain(settings: Settings | None = None) -> list[ProviderKind]:
    """Provider priority for broad daily data (§4).

    The local database is checked first by the caller, not listed here — this
    is the order in which we go *out* to the network once the store has missed.
    A chain that runs out marks data unavailable rather than substituting
    something unverified (§17).
    """
    settings = settings or get_settings()
    chain: list[ProviderKind] = []

    if settings.yfinance_enabled:
        chain.append(ProviderKind.YFINANCE)
    if settings.eodhd_api_key and settings.eodhd_api_key.get_secret_value():
        chain.append(ProviderKind.EODHD)

    if not chain:
        log.warning(
            "data.no_daily_provider_configured",
            detail="No daily provider is enabled; falling back to the mock provider. "
            "Data is simulated and must not inform real decisions.",
        )
        chain.append(ProviderKind.MOCK)

    return chain
