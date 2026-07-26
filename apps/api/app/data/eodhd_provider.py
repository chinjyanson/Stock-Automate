"""EODHD fundamentals provider (§4 — verification / gap-fill source).

A **fundamentals-only** backup for yfinance, whose fundamental coverage is thin
for non-US listings. EODHD's strength is broad global/European coverage, so this
provider exists to fill P/E, P/B, margin, growth and yield for LSE/XETR names the
scanner's value and quality factors need.

It is deliberately *not* a price source here: the candle/quote methods raise, so
nothing routes market data through it by accident. Fundamentals calls on EODHD's
free tier are credit-expensive, so callers must budget them (a slow, quarterly
gap-fill of the stocks that matter — not the whole catalogue).
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

import httpx
import structlog

from app.data.base import MarketDataProvider
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
from app.models.enums import ProviderKind
from app.models.instrument import Instrument

log = structlog.get_logger(__name__)

#: Exchange MIC → EODHD exchange code. EODHD symbols are `TICKER.CODE`
#: (`VOD.LSE`, `BMW.XETRA`, `AAPL.US`) — a namespace of its own, unrelated to the
#: yfinance suffixes, so it is derived from the instrument's ticker + venue.
_MIC_TO_EODHD: dict[str, str] = {
    "XNAS": "US",
    "XNYS": "US",
    "ARCX": "US",
    "BATS": "US",
    "XLON": "LSE",
    "XETR": "XETRA",
    "XPAR": "PA",
    "XAMS": "AS",
    "XBRU": "BR",
    "XMIL": "MI",
    "XMAD": "MC",
    "XSWX": "SW",
    "XWBO": "VI",
    "XLIS": "LS",
    "XDUB": "IR",
    "XSTO": "ST",
    "XHEL": "HE",
    "XCSE": "CO",
    "XOSL": "OL",
    "XWAR": "WAR",
}

_NOT_A_PRICE_SOURCE = "EODHD is wired as a fundamentals-only source"


def eodhd_symbol(exchange_ticker: str | None, mic: str | None) -> str | None:
    """Build an EODHD `TICKER.CODE` symbol, or None for an unsupported venue."""
    if not exchange_ticker or mic is None:
        return None
    code = _MIC_TO_EODHD.get(mic)
    if code is None:
        return None
    return f"{exchange_ticker}.{code}"


def _dec(value: Any) -> Decimal | None:
    if value is None or value == "":
        return None
    try:
        d = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None
    return d if d.is_finite() else None


class EODHDProvider(MarketDataProvider):
    kind = ProviderKind.EODHD

    def __init__(
        self, *, api_key: str, base_url: str = "https://eodhd.com/api", timeout_seconds: float = 20.0
    ) -> None:
        if not api_key:
            raise ProviderError("EODHD requires an API key")
        self._api_key = api_key
        self._client = httpx.AsyncClient(base_url=base_url, timeout=httpx.Timeout(timeout_seconds))

    @property
    def supports_intraday(self) -> bool:
        return False

    async def close(self) -> None:
        await self._client.aclose()

    # -- Fundamentals -------------------------------------------------------

    async def get_basic_fundamentals(self, symbol: str) -> Fundamentals | None:
        """Fetch fundamentals for an EODHD `TICKER.CODE` symbol.

        `filter` trims the (otherwise huge) payload to the sections we read. A
        402/429 raises `ProviderQuotaExceededError` — that is how the caller
        detects the free-tier limit.
        """
        payload = await self._get(
            f"/fundamentals/{symbol}",
            {"fmt": "json", "filter": "General,Highlights,Valuation"},
        )
        highlights = payload.get("Highlights") or {}
        valuation = payload.get("Valuation") or {}
        general = payload.get("General") or {}

        return Fundamentals(
            symbol=symbol,
            provider=self.kind,
            as_of=datetime.now(UTC),
            currency=general.get("CurrencyCode") or None,
            market_cap=_dec(highlights.get("MarketCapitalization")),
            trailing_pe=_dec(valuation.get("TrailingPE") or highlights.get("PERatio")),
            price_to_book=_dec(valuation.get("PriceBookMRQ")),
            dividend_yield=_dec(highlights.get("DividendYield")),
            revenue_growth=_dec(highlights.get("QuarterlyRevenueGrowthYOY")),
            earnings_growth=_dec(highlights.get("QuarterlyEarningsGrowthYOY")),
            profit_margin=_dec(highlights.get("ProfitMargin")),
            # Debt/equity needs the full balance sheet (a heavier call) — left
            # unknown rather than guessed.
            debt_to_equity=None,
            raw={},
        )

    # -- HTTP ---------------------------------------------------------------

    async def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        query = {**params, "api_token": self._api_key}
        try:
            response = await self._client.get(path, params=query)
        except httpx.TimeoutException as exc:
            raise ProviderUnavailableError(f"EODHD timed out on {path}") from exc
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(f"EODHD transport error: {exc}") from exc

        if response.status_code in (402, 429):
            raise ProviderQuotaExceededError(f"EODHD quota/rate limit (HTTP {response.status_code})")
        if response.status_code == 401:
            raise ProviderError("EODHD rejected the API key (HTTP 401)")
        if response.status_code == 404:
            raise ProviderSymbolNotFoundError(f"EODHD has no data for {path}")
        if response.status_code >= 500:
            raise ProviderUnavailableError(f"EODHD {response.status_code} on {path}")
        if response.status_code != 200:
            raise ProviderError(f"EODHD {response.status_code} on {path}: {response.text[:200]}")

        try:
            payload = response.json()
        except ValueError as exc:
            raise ProviderError(f"EODHD returned non-JSON on {path}") from exc
        if not isinstance(payload, dict):
            raise ProviderSymbolNotFoundError(f"EODHD returned no object for {path}")
        return payload

    async def health_check(self) -> bool:
        try:
            payload = await self._get("/fundamentals/AAPL.US", {"fmt": "json", "filter": "General"})
            return bool(payload.get("Code") or payload)
        except ProviderError:
            return False

    # -- Not a price source -------------------------------------------------

    async def resolve_instrument(self, instrument: Instrument) -> ProviderMapping | None:
        raise ProviderError(_NOT_A_PRICE_SOURCE)

    async def get_daily_candles(
        self, symbol: str, start: datetime, end: datetime
    ) -> list[Candle]:
        raise ProviderError(_NOT_A_PRICE_SOURCE)

    async def get_intraday_candles(
        self, symbol: str, interval: str, start: datetime, end: datetime
    ) -> list[Candle]:
        raise ProviderError(_NOT_A_PRICE_SOURCE)

    async def get_quote(self, symbol: str) -> Quote | None:
        raise ProviderError(_NOT_A_PRICE_SOURCE)
