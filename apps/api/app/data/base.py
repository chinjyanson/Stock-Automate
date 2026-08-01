"""The MarketDataProvider interface (§3)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from app.data.types import (
    Candle,
    Fundamentals,
    ProviderError,
    ProviderMapping,
    ProviderQuotaExceededError,
    Quote,
)
from app.models.enums import Interval, PriceUnit, ProviderKind
from app.models.instrument import Instrument


class MarketDataProvider(ABC):
    """A source of candles, quotes and optional fundamentals.

    Implementations must:

      * return DTOs from `app.data.types`, never provider-native structures;
      * normalise prices to the instrument's major unit before returning (§4);
      * mark unclosed bars `is_closed=False` rather than dropping or hiding them;
      * raise `ProviderQuotaExceededError` rather than exceeding a budget;
      * raise rather than return partial data they cannot vouch for.
    """

    kind: ProviderKind

    @property
    def supports_intraday(self) -> bool:
        return True

    @property
    def requires_api_key(self) -> bool:
        return True

    @abstractmethod
    async def resolve_instrument(self, instrument: Instrument) -> ProviderMapping | None:
        """Find this provider's symbol for `instrument`.

        Returns None when no confident match exists. Guessing is not
        acceptable: a wrong mapping silently trades a different security (§5).
        """

    @abstractmethod
    async def get_daily_candles(self, symbol: str, start: datetime, end: datetime) -> list[Candle]:
        """Daily bars in [start, end], ascending by timestamp."""

    async def get_batch_daily_candles(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
        *,
        unit_by_symbol: dict[str, PriceUnit] | None = None,
    ) -> dict[str, list[Candle]]:
        """Daily bars for several symbols, keyed by symbol.

        Part of the interface rather than one provider's extra, because the
        difference between one-symbol-per-request and many is the difference
        between refreshing a catalogue in days and refreshing it in months. A
        caller should be able to ask for a batch without first asking what kind
        of provider it holds.

        This default is the honest fallback: it loops. Providers that can fetch
        many symbols in one request override it, and for those the saving is
        roughly the batch size — a provider budget is spent per *request*, not
        per symbol.

        A symbol the provider has no data for maps to an empty list. One bad
        symbol must not fail the batch: the rest of the sweep is still useful,
        and a caller cannot tell a failed batch from a universe of delisted
        tickers.

        `unit_by_symbol` supplies each symbol's quoted price unit from the
        catalogue, letting an implementation skip a per-symbol currency lookup.
        """
        out: dict[str, list[Candle]] = {}
        for symbol in symbols:
            try:
                out[symbol] = await self.get_daily_candles(symbol, start, end)
            except ProviderQuotaExceededError:
                # Budget exhaustion is not this symbol's failure and will not
                # resolve for the next one. Stop, and let the caller see which
                # symbols were never attempted by their absence from the result.
                raise
            except ProviderError:
                out[symbol] = []
        return out

    @abstractmethod
    async def get_intraday_candles(
        self, symbol: str, interval: str, start: datetime, end: datetime
    ) -> list[Candle]:
        """Intraday bars in [start, end], ascending by timestamp."""

    @abstractmethod
    async def get_quote(self, symbol: str) -> Quote | None:
        """Latest price, or None when the provider has no current quote."""

    @abstractmethod
    async def get_basic_fundamentals(self, symbol: str) -> Fundamentals | None:
        """Optional fundamentals. None, and None fields, both mean unknown."""

    async def health_check(self) -> bool:
        """Cheap liveness probe (§18)."""
        return True

    async def close(self) -> None:
        return None

    @staticmethod
    def _parse_interval(interval: str) -> Interval:
        try:
            return Interval(interval)
        except ValueError as exc:
            raise ValueError(
                f"Unsupported interval {interval!r}; expected one of {[i.value for i in Interval]}"
            ) from exc
