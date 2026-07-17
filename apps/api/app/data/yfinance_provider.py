"""yfinance adapter (§4).

The workhorse for broad daily data: no API key, global coverage including
LSE-listed ETFs, and long history. It is also unofficial, rate-limited by
opaque heuristics, and occasionally returns subtly wrong data, so this adapter
is defensive by default.

Three things it must get right, because each is a silent-corruption risk:

  1. **GBX.** yfinance reports LSE prices in pence and signals it only via
     `currency == "GBp"`. Normalised here, once.
  2. **Unclosed bars.** `history()` happily returns today's partial bar with a
     `close` that is merely the last trade. Marked `is_closed=False`; the store
     refuses to serve it to a strategy.
  3. **Blocking I/O.** yfinance is synchronous. Called directly from the event
     loop it would stall every other request, so it runs in a thread with a
     bounded semaphore.
"""

from __future__ import annotations

import asyncio
import math
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any

import pandas as pd
import structlog
import yfinance as yf

from app.data.base import MarketDataProvider
from app.data.normalization import (
    denominated_currency,
    infer_price_unit,
    major_unit_for,
    normalise_optional,
    normalise_price,
)
from app.data.types import (
    Candle,
    Fundamentals,
    ProviderMapping,
    ProviderSymbolNotFoundError,
    ProviderUnavailableError,
    Quote,
)
from app.models.enums import Interval, PriceUnit, ProviderKind
from app.models.instrument import Instrument

log = structlog.get_logger(__name__)

#: Our interval vocabulary → yfinance's.
_INTERVAL_MAP: dict[Interval, str] = {
    Interval.M1: "1m",
    Interval.M5: "5m",
    Interval.M15: "15m",
    Interval.H1: "60m",
    Interval.D1: "1d",
    Interval.W1: "1wk",
}

#: yfinance silently truncates intraday history. Requesting beyond these
#: windows returns a short series rather than an error, which would look like a
#: data gap; we refuse the request instead so the caller knows why.
_INTRADAY_MAX_LOOKBACK = {
    Interval.M1: timedelta(days=7),
    Interval.M5: timedelta(days=59),
    Interval.M15: timedelta(days=59),
    Interval.H1: timedelta(days=729),
}

#: Suffix → likely exchange MIC, used only to corroborate an ISIN match.
_SUFFIX_TO_MIC = {
    ".L": "XLON",
    ".DE": "XETR",
    ".PA": "XPAR",
    ".AS": "XAMS",
    ".MI": "XMIL",
    ".MC": "XMAD",
    ".SW": "XSWX",
    ".ST": "XSTO",
    ".HE": "XHEL",
    ".CO": "XCSE",
    ".OL": "XOSL",
    ".IR": "XDUB",
    ".LS": "XLIS",
    ".BR": "XBRU",
    ".VI": "XWBO",
    ".WA": "XWAR",
}
_MIC_TO_SUFFIX = {mic: suffix for suffix, mic in _SUFFIX_TO_MIC.items()}


def _to_decimal(value: Any) -> Decimal | None:
    """pandas floats → Decimal, rejecting NaN/inf.

    NaN is pervasive in yfinance output (holidays, halted sessions, padded
    frames). It must never reach a Decimal column: `Decimal(nan)` is a valid
    Decimal that poisons every subsequent comparison silently.
    """
    if value is None:
        return None
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(as_float) or math.isinf(as_float):
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


class YFinanceProvider(MarketDataProvider):
    kind = ProviderKind.YFINANCE

    def __init__(
        self,
        *,
        max_concurrency: int = 4,
        backoff_base_seconds: float = 2.0,
        max_retries: int = 5,
        batch_size: int = 50,
    ) -> None:
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._backoff_base = backoff_base_seconds
        self._max_retries = max_retries
        self._batch_size = batch_size

    @property
    def requires_api_key(self) -> bool:
        return False

    async def _run_blocking(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        """Run a blocking yfinance call off the event loop, with backoff.

        yfinance signals throttling by raising or by returning an empty frame;
        neither is distinguishable from "this symbol has no data", so retries
        are bounded and the caller still has to interpret an empty result.
        """
        last_error: Exception | None = None
        for attempt in range(self._max_retries):
            async with self._semaphore:
                try:
                    return await asyncio.to_thread(func, *args, **kwargs)
                except Exception as exc:
                    last_error = exc
                    delay = self._backoff_base * (2**attempt)
                    log.warning(
                        "yfinance.retry",
                        attempt=attempt + 1,
                        max_retries=self._max_retries,
                        delay_seconds=delay,
                        error=str(exc),
                    )
            if attempt < self._max_retries - 1:
                # Sleep outside the semaphore so a backing-off task does not
                # hold a concurrency slot others could use.
                await asyncio.sleep(delay)
        raise ProviderUnavailableError(
            f"yfinance failed after {self._max_retries} attempts: {last_error}"
        ) from last_error

    # -- Mapping ------------------------------------------------------------

    async def resolve_instrument(self, instrument: Instrument) -> ProviderMapping | None:
        """Resolve a yfinance symbol for `instrument`.

        yfinance has no ISIN lookup, so identity is reconstructed from exchange
        ticker + venue suffix and then *verified* by fetching the symbol and
        comparing the ISIN it reports. An unverified guess is returned with
        `requires_confirmation=True` rather than silently trusted (§5).
        """
        if not instrument.exchange_ticker:
            return None

        suffix = ""
        if instrument.exchange and instrument.exchange.mic:
            suffix = _MIC_TO_SUFFIX.get(instrument.exchange.mic, "")

        candidate = f"{instrument.exchange_ticker}{suffix}"

        try:
            info = await self._fetch_info(candidate)
        except ProviderUnavailableError:
            return None

        if not info or not info.get("symbol"):
            return None

        reported_isin = info.get("isin")
        price_unit = infer_price_unit(candidate, info.get("currency"))

        if instrument.isin and reported_isin and reported_isin == instrument.isin:
            return ProviderMapping(
                provider=self.kind,
                provider_symbol=candidate,
                resolution_method="isin_verified",
                price_unit=price_unit,
                confidence=1.0,
                requires_confirmation=False,
            )

        # We found *a* symbol, but could not prove it is the same security.
        return ProviderMapping(
            provider=self.kind,
            provider_symbol=candidate,
            resolution_method="ticker_and_exchange_suffix",
            price_unit=price_unit,
            confidence=0.6,
            requires_confirmation=True,
            note=(
                "Matched by ticker and exchange suffix; ISIN was not confirmed. "
                "Verify before trading."
            ),
        )

    async def _fetch_info(self, symbol: str) -> dict[str, Any]:
        """Fetch `.info`, augmented with the ISIN from its separate accessor.

        `info` does not carry an ISIN for non-US listings, and yfinance exposes
        it via a distinct `.isin` property that returns the sentinel `"-"`
        rather than None when it has nothing. Both quirks are absorbed here so
        callers see a plain `isin` key that is either a real ISIN or absent.
        """

        def _fetch() -> dict[str, Any]:
            ticker = yf.Ticker(symbol)
            try:
                info = dict(ticker.info or {})
            except Exception:
                return {}
            if not info.get("isin"):
                try:
                    isin = ticker.isin
                except Exception:
                    isin = None
                # "-" is yfinance's "no ISIN", and is not a valid ISIN.
                if isin and isin != "-":
                    info["isin"] = isin
            return info

        result = await self._run_blocking(_fetch)
        return result or {}

    # -- Candles ------------------------------------------------------------

    async def get_daily_candles(self, symbol: str, start: datetime, end: datetime) -> list[Candle]:
        return await self._get_candles(symbol, Interval.D1, start, end)

    async def get_intraday_candles(
        self, symbol: str, interval: str, start: datetime, end: datetime
    ) -> list[Candle]:
        parsed = self._parse_interval(interval)
        if not parsed.is_intraday:
            raise ValueError(f"{interval} is not an intraday interval")

        max_lookback = _INTRADAY_MAX_LOOKBACK.get(parsed)
        if max_lookback and (datetime.now(UTC) - start) > max_lookback:
            raise ValueError(
                f"yfinance serves at most {max_lookback.days} days of {interval} history; "
                f"requested start {start.isoformat()} is beyond that window"
            )
        return await self._get_candles(symbol, parsed, start, end)

    async def _get_candles(
        self, symbol: str, interval: Interval, start: datetime, end: datetime
    ) -> list[Candle]:
        if interval not in _INTERVAL_MAP:
            raise ValueError(f"yfinance does not support interval {interval}")

        def _fetch() -> pd.DataFrame:
            ticker = yf.Ticker(symbol)
            # yfinance ships no type stubs, so this crosses as Any.
            return ticker.history(  # type: ignore[no-any-return]
                start=start,
                end=end,
                interval=_INTERVAL_MAP[interval],
                # Keep raw OHLC and the adjusted close side by side; we store
                # both series rather than choosing for the strategy.
                auto_adjust=False,
                actions=False,
                raise_errors=False,
            )

        frame = await self._run_blocking(_fetch)
        if frame is None or frame.empty:
            return []

        info = await self._fetch_info(symbol)
        quoted_unit = infer_price_unit(symbol, info.get("currency"))
        return self._frame_to_candles(frame, symbol, interval, quoted_unit)

    def _frame_to_candles(
        self,
        frame: pd.DataFrame,
        symbol: str,
        interval: Interval,
        quoted_unit: PriceUnit,
    ) -> list[Candle]:
        normalised_unit = major_unit_for(quoted_unit)
        currency = denominated_currency(quoted_unit)
        now = datetime.now(UTC)
        candles: list[Candle] = []

        for index, row in frame.iterrows():
            timestamp = _index_to_utc(index, interval)
            if timestamp is None:
                continue

            open_ = _to_decimal(row.get("Open"))
            high = _to_decimal(row.get("High"))
            low = _to_decimal(row.get("Low"))
            close = _to_decimal(row.get("Close"))
            if None in (open_, high, low, close):
                # A row without a full OHLC is not a candle. Dropping is correct;
                # the gap detector will notice the hole and record it.
                continue
            assert open_ is not None and high is not None and low is not None and close is not None

            candle = Candle(
                symbol=symbol,
                interval=interval,
                timestamp=timestamp,
                open=normalise_price(open_, quoted_unit),
                high=normalise_price(high, quoted_unit),
                low=normalise_price(low, quoted_unit),
                close=normalise_price(close, quoted_unit),
                adjusted_close=normalise_optional(_to_decimal(row.get("Adj Close")), quoted_unit),
                volume=_to_decimal(row.get("Volume")),
                currency=currency,
                price_unit=normalised_unit,
                provider=self.kind,
                is_adjusted=False,
                is_closed=_is_bar_closed(timestamp, interval, now),
            )
            if not candle.is_coherent:
                log.warning(
                    "yfinance.incoherent_candle_dropped",
                    symbol=symbol,
                    timestamp=timestamp.isoformat(),
                    detail="OHLC failed range sanity check",
                )
                continue
            candles.append(candle)

        candles.sort(key=lambda c: c.timestamp)
        return candles

    async def get_batch_daily_candles(
        self, symbols: list[str], start: datetime, end: datetime
    ) -> dict[str, list[Candle]]:
        """Fetch several symbols per request (§4).

        Batching is the difference between a scanner rotation that finishes and
        one that gets throttled. Symbols are chunked, and a failed chunk yields
        an empty list for its members rather than failing the whole sweep — a
        partial scan is useful, a dead one is not.
        """
        results: dict[str, list[Candle]] = {}

        for chunk_start in range(0, len(symbols), self._batch_size):
            chunk = symbols[chunk_start : chunk_start + self._batch_size]

            def _fetch(tickers: list[str] = chunk) -> pd.DataFrame:
                return yf.download(  # type: ignore[no-any-return]
                    tickers=tickers,
                    start=start,
                    end=end,
                    interval="1d",
                    auto_adjust=False,
                    actions=False,
                    group_by="ticker",
                    progress=False,
                    threads=False,
                )

            try:
                frame = await self._run_blocking(_fetch)
            except ProviderUnavailableError as exc:
                log.error("yfinance.batch_failed", symbols=chunk, error=str(exc))
                for symbol in chunk:
                    results[symbol] = []
                continue

            if frame is None or frame.empty:
                for symbol in chunk:
                    results[symbol] = []
                continue

            for symbol in chunk:
                try:
                    # A single-symbol download returns a flat frame; a
                    # multi-symbol one is column-multi-indexed by ticker.
                    sub = frame[symbol] if len(chunk) > 1 else frame
                except KeyError:
                    results[symbol] = []
                    continue
                if sub is None or sub.empty:
                    results[symbol] = []
                    continue

                info = await self._fetch_info(symbol)
                quoted_unit = infer_price_unit(symbol, info.get("currency"))
                results[symbol] = self._frame_to_candles(
                    sub.dropna(how="all"), symbol, Interval.D1, quoted_unit
                )

        return results

    # -- Quote and fundamentals --------------------------------------------

    async def get_quote(self, symbol: str) -> Quote | None:
        info = await self._fetch_info(symbol)
        if not info:
            return None

        price = _to_decimal(
            info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose")
        )
        if price is None:
            return None

        quoted_unit = infer_price_unit(symbol, info.get("currency"))
        return Quote(
            symbol=symbol,
            price=normalise_price(price, quoted_unit),
            currency=denominated_currency(quoted_unit),
            price_unit=major_unit_for(quoted_unit),
            provider=self.kind,
            as_of=datetime.now(UTC),
            bid=normalise_optional(_to_decimal(info.get("bid")), quoted_unit),
            ask=normalise_optional(_to_decimal(info.get("ask")), quoted_unit),
            volume=_to_decimal(info.get("volume")),
        )

    async def get_basic_fundamentals(self, symbol: str) -> Fundamentals | None:
        info = await self._fetch_info(symbol)
        if not info:
            raise ProviderSymbolNotFoundError(f"yfinance returned nothing for {symbol}")

        # Every field stays None when absent. Absence is unknown, not zero (§6).
        return Fundamentals(
            symbol=symbol,
            provider=self.kind,
            as_of=datetime.now(UTC),
            currency=info.get("financialCurrency") or info.get("currency"),
            market_cap=_to_decimal(info.get("marketCap")),
            trailing_pe=_to_decimal(info.get("trailingPE")),
            price_to_book=_to_decimal(info.get("priceToBook")),
            dividend_yield=_to_decimal(info.get("dividendYield")),
            revenue_growth=_to_decimal(info.get("revenueGrowth")),
            earnings_growth=_to_decimal(info.get("earningsGrowth")),
            profit_margin=_to_decimal(info.get("profitMargins")),
            debt_to_equity=_to_decimal(info.get("debtToEquity")),
            raw={},
        )

    async def health_check(self) -> bool:
        try:
            # A liquid, always-present symbol. Cheap and unambiguous.
            info = await self._fetch_info("SPY")
            return bool(info.get("symbol"))
        except Exception:
            return False


def _index_to_utc(index: Any, interval: Interval) -> datetime | None:
    """Convert a pandas index entry to a timezone-aware UTC datetime.

    Daily and weekly bars are handled differently from intraday, and the
    difference is load-bearing.

    yfinance stamps a daily bar at *exchange* midnight: the LSE bar for
    2026-07-16 arrives as `2026-07-16 00:00:00+01:00`. Tz-converting that to
    UTC yields `2026-07-15 23:00Z` — the bar silently moves to the previous
    calendar day, shifting an entire LSE daily series back one day and
    misaligning it against every other series it is compared to.

    A daily bar is not an instant; it is a label for a session. So for daily
    and weekly we keep the exchange-local calendar date and re-stamp it at
    midnight UTC, preserving the label. Intraday bars *are* instants and are
    converted normally.
    """
    if not isinstance(index, pd.Timestamp):
        return None

    if interval in (Interval.D1, Interval.W1):
        # .date() reads the local calendar date, before any UTC conversion.
        session_date = index.date()
        return datetime(session_date.year, session_date.month, session_date.day, tzinfo=UTC)

    if index.tzinfo is None:
        return index.to_pydatetime().replace(tzinfo=UTC)
    return index.tz_convert("UTC").to_pydatetime()


def _is_bar_closed(timestamp: datetime, interval: Interval, now: datetime) -> bool:
    """Has this bar finished forming?

    Conservative by construction: a bar counts as closed only once its full
    duration has elapsed. Being wrong in the other direction means a strategy
    reads a `close` that is not final (§4).
    """
    durations = {
        Interval.M1: timedelta(minutes=1),
        Interval.M5: timedelta(minutes=5),
        Interval.M15: timedelta(minutes=15),
        Interval.H1: timedelta(hours=1),
        Interval.H4: timedelta(hours=4),
        Interval.D1: timedelta(days=1),
        Interval.W1: timedelta(weeks=1),
    }
    return (timestamp + durations[interval]) <= now
