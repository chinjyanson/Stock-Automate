"""Twelve Data adapter (§4).

The intraday source for the S&P 500 15-minute strategy (§8). yfinance covers
broad daily data; Twelve Data is used where reliable *intraday* US bars are
needed. It is a keyed REST API with a tight free-tier budget, so this adapter is
defensive: it normalises to the instrument's major unit, marks a still-forming bar
`is_closed=False`, and turns a rate-limit response into `ProviderQuotaExceededError`
rather than pretending it got data.

Prices are returned in the listing currency (Twelve Data does not quote LSE lines
in pence the way yfinance does), so unit inference + normalisation still runs — a
US listing is a no-op, and a non-USD listing is handled by the same path.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any

import httpx
import structlog

from app.data.base import MarketDataProvider
from app.data.normalization import (
    denominated_currency,
    infer_price_unit,
    major_unit_for,
    normalise_price,
)
from app.data.types import (
    Candle,
    Fundamentals,
    ProviderError,
    ProviderMapping,
    ProviderQuotaExceededError,
    ProviderSymbolNotFoundError,
    ProviderUnavailableError,
    Quote,
)
from app.models.enums import Interval, PriceUnit, ProviderKind
from app.models.instrument import Instrument

log = structlog.get_logger(__name__)

#: Our interval vocabulary → Twelve Data's.
_INTERVAL_MAP: dict[Interval, str] = {
    Interval.M1: "1min",
    Interval.M5: "5min",
    Interval.M15: "15min",
    Interval.H1: "1h",
    Interval.H4: "4h",
    Interval.D1: "1day",
    Interval.W1: "1week",
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

#: Twelve Data's rate-limit error code (returned in a 200 body as well as 429).
_RATE_LIMIT_CODE = 429


def _to_decimal(value: Any) -> Decimal | None:
    if value is None or value == "":
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None


class TwelveDataProvider(MarketDataProvider):
    kind = ProviderKind.TWELVE_DATA

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.twelvedata.com",
        timeout_seconds: float = 20.0,
    ) -> None:
        if not api_key:
            raise ProviderError("Twelve Data requires an API key")
        self._api_key = api_key
        self._client = httpx.AsyncClient(
            base_url=base_url, timeout=httpx.Timeout(timeout_seconds)
        )

    async def close(self) -> None:
        await self._client.aclose()

    # -- HTTP ---------------------------------------------------------------

    async def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        query = {**params, "apikey": self._api_key}
        try:
            response = await self._client.get(path, params=query)
        except httpx.TimeoutException as exc:
            raise ProviderUnavailableError(f"Twelve Data timed out on {path}") from exc
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(f"Twelve Data transport error: {exc}") from exc

        if response.status_code == _RATE_LIMIT_CODE:
            raise ProviderQuotaExceededError("Twelve Data rate limit (HTTP 429)")
        if response.status_code >= 500:
            raise ProviderUnavailableError(f"Twelve Data {response.status_code} on {path}")

        try:
            payload = response.json()
        except ValueError as exc:
            raise ProviderError(f"Twelve Data returned non-JSON on {path}") from exc

        if not isinstance(payload, dict):
            raise ProviderError(f"Twelve Data returned a non-object body on {path}")
        # Twelve Data reports logical errors in a 200 body: {"status":"error", ...}.
        if payload.get("status") == "error":
            code = int(payload.get("code", 0) or 0)
            message = str(payload.get("message", "unknown error"))
            if code == _RATE_LIMIT_CODE:
                raise ProviderQuotaExceededError(f"Twelve Data budget: {message}")
            if code == 404 or "not found" in message.lower():
                raise ProviderSymbolNotFoundError(message)
            raise ProviderError(f"Twelve Data error {code}: {message}")
        return payload

    # -- Provider interface -------------------------------------------------

    async def resolve_instrument(self, instrument: Instrument) -> ProviderMapping | None:
        """Best-effort symbol resolution via Twelve Data's symbol search.

        Below the ISIN tier, so the match is flagged for confirmation — a wrong
        symbol trades a different security (§5), so guessing silently is not
        acceptable.
        """
        ticker = instrument.exchange_ticker
        if not ticker:
            return None
        try:
            payload = await self._get("/symbol_search", {"symbol": ticker})
        except ProviderError:
            return None
        for match in payload.get("data", []):
            if str(match.get("symbol", "")).upper() == ticker.upper():
                name = match.get("instrument_name", ticker)
                exchange = match.get("exchange")
                return ProviderMapping(
                    provider=self.kind,
                    provider_symbol=ticker,
                    resolution_method="twelve_data_symbol_search",
                    confidence=0.6,
                    requires_confirmation=True,
                    note=f"Matched {name} on {exchange}",
                )
        return None

    async def get_daily_candles(self, symbol: str, start: datetime, end: datetime) -> list[Candle]:
        return await self._time_series(symbol, Interval.D1, start, end)

    async def get_intraday_candles(
        self, symbol: str, interval: str, start: datetime, end: datetime
    ) -> list[Candle]:
        parsed = self._parse_interval(interval)
        if not parsed.is_intraday:
            raise ProviderError(f"{interval} is not an intraday interval")
        return await self._time_series(symbol, parsed, start, end)

    async def _time_series(
        self, symbol: str, interval: Interval, start: datetime, end: datetime
    ) -> list[Candle]:
        td_interval = _INTERVAL_MAP.get(interval)
        if td_interval is None:
            raise ProviderError(f"Twelve Data does not support interval {interval}")
        payload = await self._get(
            "/time_series",
            {
                "symbol": symbol,
                "interval": td_interval,
                "start_date": start.strftime("%Y-%m-%d %H:%M:%S"),
                "end_date": end.strftime("%Y-%m-%d %H:%M:%S"),
                "outputsize": 5000,
                "timezone": "UTC",
                "order": "ASC",
            },
        )
        meta = payload.get("meta", {})
        currency = meta.get("currency")
        unit = infer_price_unit(symbol, currency)
        now = datetime.now(UTC)
        duration = _INTERVAL_DURATION[interval]

        candles: list[Candle] = []
        for row in payload.get("values", []):
            candle = self._row_to_candle(row, symbol, interval, unit, currency, now, duration)
            if candle is not None:
                candles.append(candle)
        candles.sort(key=lambda c: c.timestamp)
        return candles

    def _row_to_candle(
        self,
        row: dict[str, Any],
        symbol: str,
        interval: Interval,
        unit: PriceUnit,
        currency: str | None,
        now: datetime,
        duration: timedelta,
    ) -> Candle | None:
        ts = self._parse_ts(row.get("datetime"))
        o = _to_decimal(row.get("open"))
        h = _to_decimal(row.get("high"))
        low = _to_decimal(row.get("low"))
        c = _to_decimal(row.get("close"))
        if ts is None or o is None or h is None or low is None or c is None:
            return None
        volume = _to_decimal(row.get("volume"))
        # A bar whose window has not yet elapsed is still forming.
        is_closed = (ts + duration) <= now
        return Candle(
            symbol=symbol,
            interval=interval,
            timestamp=ts,
            open=normalise_price(o, unit),
            high=normalise_price(h, unit),
            low=normalise_price(low, unit),
            close=normalise_price(c, unit),
            volume=volume,
            currency=currency or denominated_currency(unit),
            price_unit=major_unit_for(unit),
            provider=self.kind,
            is_closed=is_closed,
        )

    async def get_quote(self, symbol: str) -> Quote | None:
        try:
            payload = await self._get("/quote", {"symbol": symbol})
        except ProviderSymbolNotFoundError:
            return None
        price = _to_decimal(payload.get("close"))
        if price is None:
            return None
        currency = payload.get("currency")
        unit = infer_price_unit(symbol, currency)
        return Quote(
            symbol=symbol,
            price=normalise_price(price, unit),
            currency=currency or denominated_currency(unit),
            price_unit=major_unit_for(unit),
            provider=self.kind,
            as_of=datetime.now(UTC),
        )

    async def get_basic_fundamentals(self, symbol: str) -> Fundamentals | None:
        # Fundamentals are a higher Twelve Data tier; treat as unknown (None),
        # which the scanner must never penalise (acceptance criterion 7).
        return None

    # -- Parsing helpers ----------------------------------------------------

    @staticmethod
    def _parse_ts(raw: Any) -> datetime | None:
        if not raw:
            return None
        text = str(raw)
        fmt = "%Y-%m-%d %H:%M:%S" if " " in text else "%Y-%m-%d"
        try:
            naive = datetime.strptime(text, fmt)
        except ValueError:
            return None
        # We request `timezone=UTC`, so the values already are UTC.
        return naive.replace(tzinfo=UTC)
