"""Provider-neutral market-data DTOs.

yfinance DataFrames, Twelve Data JSON and EODHD payloads all stop at the
adapter boundary (§3). What crosses is this.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any

from app.models.enums import Interval, PriceUnit, ProviderKind


@dataclass(frozen=True, slots=True)
class ProviderMapping:
    """A provider's answer to "what do you call this instrument?"."""

    provider: ProviderKind
    provider_symbol: str
    #: How identity was established, recorded for the provenance display (§6).
    resolution_method: str
    price_unit: PriceUnit | None = None
    confidence: float = 1.0
    #: Set when the match is below the ISIN tier and wants human confirmation.
    requires_confirmation: bool = False
    note: str | None = None


@dataclass(frozen=True, slots=True)
class Candle:
    """One OHLCV bar as returned by a provider, already unit-normalised.

    `is_closed` is not decoration. A bar still forming has a `close` that is
    merely the last trade so far; acting on it means acting on a number that
    will change. Adapters must set this honestly (§4).
    """

    symbol: str
    interval: Interval
    #: Bar OPEN time, timezone-aware UTC.
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal | None
    currency: str
    price_unit: PriceUnit
    provider: ProviderKind
    adjusted_close: Decimal | None = None
    is_adjusted: bool = False
    is_closed: bool = True

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None:
            raise ValueError(f"Candle timestamp must be timezone-aware: {self.timestamp!r}")

    @property
    def is_coherent(self) -> bool:
        """Basic OHLC sanity: the range must contain the endpoints.

        A provider that violates this has sent us something we cannot interpret,
        and quietly trading on it is worse than dropping it.
        """
        return (
            self.high >= self.low
            and self.high >= self.open
            and self.high >= self.close
            and self.low <= self.open
            and self.low <= self.close
            and self.open > 0
            and self.close > 0
        )


@dataclass(frozen=True, slots=True)
class Quote:
    symbol: str
    price: Decimal
    currency: str
    price_unit: PriceUnit
    provider: ProviderKind
    as_of: datetime
    bid: Decimal | None = None
    ask: Decimal | None = None
    volume: Decimal | None = None

    @property
    def spread(self) -> Decimal | None:
        if self.bid is None or self.ask is None:
            return None
        return self.ask - self.bid


@dataclass(frozen=True, slots=True)
class Fundamentals:
    """Optional fundamentals. Every field may be None, and None means unknown.

    Not zero, and not a failure. The scanner must not penalise absence
    (acceptance criterion 7).
    """

    symbol: str
    provider: ProviderKind
    as_of: datetime
    currency: str | None = None
    market_cap: Decimal | None = None
    trailing_pe: Decimal | None = None
    price_to_book: Decimal | None = None
    dividend_yield: Decimal | None = None
    revenue_growth: Decimal | None = None
    earnings_growth: Decimal | None = None
    profit_margin: Decimal | None = None
    debt_to_equity: Decimal | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def available_field_count(self) -> int:
        """How many optional metrics we actually have. Feeds data completeness."""
        return sum(
            1
            for value in (
                self.market_cap,
                self.trailing_pe,
                self.price_to_book,
                self.dividend_yield,
                self.revenue_growth,
                self.earnings_growth,
                self.profit_margin,
                self.debt_to_equity,
            )
            if value is not None
        )


class ProviderError(Exception):
    """Base class for market-data failures."""


class ProviderQuotaExceededError(ProviderError):
    """The configured budget is spent. Not a retryable condition (§4)."""


class ProviderUnavailableError(ProviderError):
    """Provider unreachable. Callers fail closed rather than guess (§17)."""


class ProviderSymbolNotFoundError(ProviderError):
    """The provider does not recognise this symbol."""


class ProviderDataQualityError(ProviderError):
    """The provider answered, but with data we refuse to trust."""
