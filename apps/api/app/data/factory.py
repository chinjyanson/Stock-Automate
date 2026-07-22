"""Market-data provider selection and the daily-data priority chain (§4).

As with brokers, a provider is never substituted. The mock provider is
reachable only when explicitly named: serving invented prices to something that
asked for real ones is the quietest way to produce confident, wrong answers.
"""

from __future__ import annotations

import structlog

from app.config import Settings, get_settings
from app.data.base import MarketDataProvider
from app.data.mock_provider import MockMarketDataProvider
from app.data.twelve_data_provider import TwelveDataProvider
from app.data.yfinance_provider import YFinanceProvider
from app.models.enums import ProviderKind

log = structlog.get_logger(__name__)


class ProviderNotConfiguredError(Exception):
    """A provider was requested or required whose configuration is absent."""


def resolve_provider(kind: ProviderKind, settings: Settings | None = None) -> MarketDataProvider:
    settings = settings or get_settings()

    match kind:
        case ProviderKind.MOCK:
            # Reachable only by explicitly asking for it. Nothing falls back here.
            return MockMarketDataProvider()

        case ProviderKind.YFINANCE:
            return YFinanceProvider(
                max_concurrency=settings.yfinance_max_concurrency,
                backoff_base_seconds=settings.yfinance_backoff_base_seconds,
                max_retries=settings.yfinance_max_retries,
                batch_size=settings.yfinance_batch_size,
            )

        case ProviderKind.TWELVE_DATA:
            # The intraday source for the S&P 15-minute strategy (§8). A missing
            # key is a configuration gap, not a cue to serve daily bars to a 15m
            # strategy — that substitution would look like it worked.
            key = settings.twelve_data_api_key
            if key is None or not key.get_secret_value():
                raise ProviderNotConfiguredError(
                    "Twelve Data was requested but TWELVE_DATA_API_KEY is not configured. "
                    "Set it in .env, or request ProviderKind.MOCK explicitly to run offline "
                    "against deterministic intraday fixtures."
                )
            return TwelveDataProvider(
                api_key=key.get_secret_value(),
                base_url=settings.twelve_data_base_url,
            )

        case ProviderKind.EODHD:
            raise NotImplementedError(
                "EODHDProvider arrives in Phase 2 as a verification and gap-fill source."
            )

    raise ValueError(f"Unknown provider kind: {kind}")


def daily_provider_chain(settings: Settings | None = None) -> list[ProviderKind]:
    """Provider priority for broad daily data (§4).

    The local database is checked first by the caller, not listed here — this is
    the order in which we go *out* to the network once the store has missed. A
    chain that runs out marks data unavailable rather than substituting
    something unverified (§17).

    Raises when no real provider is configured. Previously this appended the
    mock provider, which meant a misconfigured deployment would scan, score and
    rank instruments on a random walk while reporting nothing unusual. An empty
    chain is a configuration error and says so.
    """
    settings = settings or get_settings()
    chain: list[ProviderKind] = []

    if settings.yfinance_enabled:
        chain.append(ProviderKind.YFINANCE)
    if settings.eodhd_api_key and settings.eodhd_api_key.get_secret_value():
        chain.append(ProviderKind.EODHD)

    if not chain:
        raise ProviderNotConfiguredError(
            "No daily market-data provider is configured. Set YFINANCE_ENABLED=true "
            "(no API key required) or configure EODHD_API_KEY. To run offline "
            "against deterministic fixtures, request ProviderKind.MOCK explicitly."
        )

    return chain


def intraday_provider_chain(settings: Settings | None = None) -> list[ProviderKind]:
    """Provider priority for intraday (15m) data (§4, §8).

    Twelve Data is the only real intraday source. As with the daily chain, an
    empty chain is a configuration error rather than a silent fall back to the
    mock — a 15m strategy on invented bars is a confident, wrong answer.
    """
    settings = settings or get_settings()
    if settings.twelve_data_api_key and settings.twelve_data_api_key.get_secret_value():
        return [ProviderKind.TWELVE_DATA]
    raise ProviderNotConfiguredError(
        "No intraday market-data provider is configured. Set TWELVE_DATA_API_KEY, or "
        "request ProviderKind.MOCK explicitly to run offline against intraday fixtures."
    )
