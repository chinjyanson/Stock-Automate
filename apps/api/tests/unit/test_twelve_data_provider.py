"""Twelve Data adapter against mocked HTTP.

No network: `respx` serves fixture responses so the parsing, normalisation and
error-mapping rules are pinned. The dangerous cases are a rate-limit body that
looks like a 200 and a pence-quoted price — both must not reach a strategy as
silently wrong data.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import httpx
import pytest
import respx

from app.data.twelve_data_provider import TwelveDataProvider
from app.data.types import ProviderQuotaExceededError
from app.models.enums import Interval, PriceUnit

pytestmark = pytest.mark.asyncio

_BASE = "https://api.twelvedata.com"
_START = datetime(2024, 1, 1, tzinfo=UTC)
_END = datetime(2024, 1, 3, tzinfo=UTC)


def _series_body(currency: str = "USD") -> dict:
    return {
        "meta": {"symbol": "SPY", "interval": "15min", "currency": currency},
        "values": [
            {
                "datetime": "2024-01-02 15:30:00",
                "open": "100.0",
                "high": "101.0",
                "low": "99.0",
                "close": "100.5",
                "volume": "1000",
            },
            {
                "datetime": "2024-01-02 15:15:00",
                "open": "99.0",
                "high": "100.0",
                "low": "98.5",
                "close": "99.5",
                "volume": "900",
            },
        ],
        "status": "ok",
    }


class TestParsing:
    @respx.mock
    async def test_intraday_candles_are_parsed_ascending_and_normalised(self) -> None:
        route = respx.get(f"{_BASE}/time_series").mock(
            return_value=httpx.Response(200, json=_series_body())
        )
        provider = TwelveDataProvider(api_key="k")
        try:
            candles = await provider.get_intraday_candles("SPY", "15m", _START, _END)
        finally:
            await provider.close()

        assert route.called
        # Interval mapped to Twelve Data's vocabulary.
        assert route.calls.last.request.url.params["interval"] == "15min"
        # Returned oldest-first, as Decimals.
        assert [c.close for c in candles] == [Decimal("99.5"), Decimal("100.5")]
        assert candles[0].interval is Interval.M15
        assert candles[0].price_unit is PriceUnit.USD

    @respx.mock
    async def test_pence_prices_are_normalised_to_pounds(self) -> None:
        respx.get(f"{_BASE}/time_series").mock(
            return_value=httpx.Response(200, json=_series_body(currency="GBX"))
        )
        provider = TwelveDataProvider(api_key="k")
        try:
            candles = await provider.get_intraday_candles("VOD", "15m", _START, _END)
        finally:
            await provider.close()
        # 100.5 pence -> 1.005 pounds, exact.
        assert candles[-1].close == Decimal("1.005")
        assert candles[-1].price_unit is PriceUnit.GBP

    @respx.mock
    async def test_daily_uses_the_1day_interval(self) -> None:
        route = respx.get(f"{_BASE}/time_series").mock(
            return_value=httpx.Response(200, json=_series_body())
        )
        provider = TwelveDataProvider(api_key="k")
        try:
            await provider.get_daily_candles("SPY", _START, _END)
        finally:
            await provider.close()
        assert route.calls.last.request.url.params["interval"] == "1day"


class TestErrors:
    @respx.mock
    async def test_rate_limit_body_raises_quota_exceeded(self) -> None:
        # Twelve Data reports the rate limit in a 200 body, not an HTTP status.
        respx.get(f"{_BASE}/time_series").mock(
            return_value=httpx.Response(
                200, json={"code": 429, "message": "API credits limit reached", "status": "error"}
            )
        )
        provider = TwelveDataProvider(api_key="k")
        try:
            with pytest.raises(ProviderQuotaExceededError):
                await provider.get_intraday_candles("SPY", "15m", _START, _END)
        finally:
            await provider.close()

    @respx.mock
    async def test_http_429_raises_quota_exceeded(self) -> None:
        respx.get(f"{_BASE}/time_series").mock(return_value=httpx.Response(429))
        provider = TwelveDataProvider(api_key="k")
        try:
            with pytest.raises(ProviderQuotaExceededError):
                await provider.get_intraday_candles("SPY", "15m", _START, _END)
        finally:
            await provider.close()

    async def test_missing_key_is_refused(self) -> None:
        with pytest.raises(Exception, match="API key"):
            TwelveDataProvider(api_key="")
