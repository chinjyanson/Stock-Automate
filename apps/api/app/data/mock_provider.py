"""Deterministic fake market-data provider.

Backs the offline workflow and every test that needs candles without a network.

Determinism comes from seeding a PRNG with a hash of the symbol, so a given
symbol always yields the identical series across processes and runs. That
matters more than realism: a test asserting "this scores 78" must not become
flaky because a random walk wandered.

The series is a geometric random walk with a mild drift. It is *not* a model of
anything. Results derived from it are simulated and must be labelled as such
(§11).
"""

from __future__ import annotations

import hashlib
import random
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from app.data.base import MarketDataProvider
from app.data.types import Candle, Fundamentals, ProviderMapping, Quote
from app.models.enums import Interval, PriceUnit, ProviderKind
from app.models.instrument import Instrument

#: Starting price and denomination per fixture symbol.
_FIXTURE_SYMBOLS: dict[str, tuple[Decimal, PriceUnit, str]] = {
    "SPY": (Decimal("598.40"), PriceUnit.USD, "USD"),
    "AAPL": (Decimal("232.15"), PriceUnit.USD, "USD"),
    "MSFT": (Decimal("418.60"), PriceUnit.USD, "USD"),
    "VUAG.L": (Decimal("87.50"), PriceUnit.GBP, "GBP"),
    "VUSA.L": (Decimal("84.20"), PriceUnit.GBP, "GBP"),
    "SGLN.L": (Decimal("49.80"), PriceUnit.GBP, "GBP"),
    "CRUD.L": (Decimal("23.10"), PriceUnit.GBP, "GBP"),
}

_INTERVAL_DURATION: dict[Interval, timedelta] = {
    Interval.M1: timedelta(minutes=1),
    Interval.M5: timedelta(minutes=5),
    Interval.M15: timedelta(minutes=15),
    Interval.H1: timedelta(hours=1),
    Interval.H4: timedelta(hours=4),
    Interval.D1: timedelta(days=1),
    Interval.W1: timedelta(weeks=1),
}


def _seed_for(symbol: str, interval: Interval) -> int:
    """Stable seed. `hash()` is salted per-process, so it cannot be used here."""
    digest = hashlib.sha256(f"{symbol}:{interval}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


class MockMarketDataProvider(MarketDataProvider):
    kind = ProviderKind.MOCK

    def __init__(self, *, annual_drift: float = 0.07, annual_volatility: float = 0.18) -> None:
        self._drift = annual_drift
        self._volatility = annual_volatility
        #: Set by tests to force failures on demand.
        self.fail_with: Exception | None = None

    @property
    def requires_api_key(self) -> bool:
        return False

    def _maybe_fail(self) -> None:
        if self.fail_with is not None:
            raise self.fail_with

    async def resolve_instrument(self, instrument: Instrument) -> ProviderMapping | None:
        self._maybe_fail()
        ticker = (instrument.exchange_ticker or "").upper()
        for symbol in _FIXTURE_SYMBOLS:
            if symbol.split(".")[0] == ticker:
                return ProviderMapping(
                    provider=self.kind,
                    provider_symbol=symbol,
                    resolution_method="mock_fixture",
                    price_unit=_FIXTURE_SYMBOLS[symbol][1],
                    confidence=1.0,
                )
        return None

    def _generate(
        self, symbol: str, interval: Interval, start: datetime, end: datetime
    ) -> list[Candle]:
        base_price, unit, currency = _FIXTURE_SYMBOLS.get(
            symbol, (Decimal("100.00"), PriceUnit.USD, "USD")
        )
        rng = random.Random(_seed_for(symbol, interval))
        step = _INTERVAL_DURATION[interval]

        # Scale drift/vol from annual to this bar's duration.
        bars_per_year = timedelta(days=365) / step
        bar_drift = self._drift / bars_per_year
        bar_vol = self._volatility / (bars_per_year**0.5)

        now = datetime.now(UTC)
        candles: list[Candle] = []
        price = float(base_price)

        # Daily and weekly bars are labels for sessions, so they must land on
        # midnight UTC — exactly as real providers stamp them. Starting the walk
        # at the caller's wall-clock `start` would stamp bars at an arbitrary
        # time of day, so two ingests seconds apart would produce different
        # timestamps for the same session and upsert into *separate* rows
        # instead of converging. (A refetch test caught this.)
        cursor = (
            start.replace(hour=0, minute=0, second=0, microsecond=0)
            if interval in (Interval.D1, Interval.W1)
            else start
        )

        while cursor < end:
            # Daily and above: skip weekends. Intraday fixtures are not session-aware;
            # tests needing real session boundaries use explicit fixtures instead.
            if interval in (Interval.D1, Interval.W1) and cursor.weekday() >= 5:
                cursor += step
                continue

            shock = rng.gauss(bar_drift, bar_vol)
            open_price = price
            close_price = open_price * (1 + shock)
            high_price = max(open_price, close_price) * (1 + abs(rng.gauss(0, bar_vol / 2)))
            low_price = min(open_price, close_price) * (1 - abs(rng.gauss(0, bar_vol / 2)))
            volume = rng.randint(500_000, 5_000_000)

            candles.append(
                Candle(
                    symbol=symbol,
                    interval=interval,
                    timestamp=cursor,
                    open=_q(open_price),
                    high=_q(high_price),
                    low=_q(low_price),
                    close=_q(close_price),
                    adjusted_close=_q(close_price),
                    volume=Decimal(volume),
                    currency=currency,
                    price_unit=unit,
                    provider=self.kind,
                    is_adjusted=False,
                    is_closed=(cursor + step) <= now,
                )
            )
            price = close_price
            cursor += step

        return candles

    async def get_daily_candles(self, symbol: str, start: datetime, end: datetime) -> list[Candle]:
        self._maybe_fail()
        return self._generate(symbol, Interval.D1, start, end)

    async def get_intraday_candles(
        self, symbol: str, interval: str, start: datetime, end: datetime
    ) -> list[Candle]:
        self._maybe_fail()
        parsed = self._parse_interval(interval)
        if not parsed.is_intraday:
            raise ValueError(f"{interval} is not an intraday interval")
        return self._generate(symbol, parsed, start, end)

    async def get_quote(self, symbol: str) -> Quote | None:
        self._maybe_fail()
        base_price, unit, currency = _FIXTURE_SYMBOLS.get(
            symbol, (Decimal("100.00"), PriceUnit.USD, "USD")
        )
        return Quote(
            symbol=symbol,
            price=base_price,
            currency=currency,
            price_unit=unit,
            provider=self.kind,
            as_of=datetime.now(UTC),
            bid=base_price * Decimal("0.9995"),
            ask=base_price * Decimal("1.0005"),
            volume=Decimal(1_000_000),
        )

    async def get_basic_fundamentals(self, symbol: str) -> Fundamentals | None:
        self._maybe_fail()
        if symbol not in _FIXTURE_SYMBOLS:
            return None
        rng = random.Random(_seed_for(symbol, Interval.D1))
        # Some fields are deliberately left None: exercising the "missing
        # fundamentals must not be scored as zero" path is the point (§6).
        return Fundamentals(
            symbol=symbol,
            provider=self.kind,
            as_of=datetime.now(UTC),
            currency=_FIXTURE_SYMBOLS[symbol][2],
            market_cap=Decimal(rng.randint(1_000_000_000, 3_000_000_000_000)),
            trailing_pe=Decimal(str(round(rng.uniform(8, 40), 2))),
            price_to_book=Decimal(str(round(rng.uniform(0.8, 12), 2))),
            dividend_yield=Decimal(str(round(rng.uniform(0, 0.05), 4))),
            revenue_growth=None,
            earnings_growth=None,
            profit_margin=Decimal(str(round(rng.uniform(0.02, 0.35), 4))),
            debt_to_equity=None,
        )


def _q(value: float) -> Decimal:
    """Quantise a float price to 8dp at the boundary, once."""
    return Decimal(str(round(value, 8)))
