"""yfinance adapter behaviour, exercised offline against fixture frames.

No network. The frames here reproduce the exact shapes observed from the live
API (tz-aware exchange-local daily index, GBp currency marker, NaN padding), so
these tests pin the adapter's contract without depending on Yahoo being up or
on today's prices.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pandas as pd
import pytest

from app.data.types import Candle
from app.data.yfinance_provider import (
    YFinanceProvider,
    _index_to_utc,
    _is_bar_closed,
    _to_decimal,
)
from app.models.enums import Interval, PriceUnit, ProviderKind


class TestIndexToUtc:
    def test_lse_daily_bar_keeps_its_session_date(self) -> None:
        """The date-shift bug.

        yfinance stamps the LSE daily bar for 16 July as 2026-07-16 00:00+01:00.
        A naive tz_convert to UTC yields 2026-07-15 23:00Z, moving the bar to
        the previous calendar day and misaligning the whole series.
        """
        stamp = pd.Timestamp("2026-07-16 00:00:00", tz="Europe/London")
        result = _index_to_utc(stamp, Interval.D1)
        assert result == datetime(2026, 7, 16, 0, 0, tzinfo=UTC)
        assert result is not None and result.date().day == 16

    def test_us_daily_bar_keeps_its_session_date(self) -> None:
        stamp = pd.Timestamp("2026-07-16 00:00:00", tz="America/New_York")
        result = _index_to_utc(stamp, Interval.D1)
        assert result == datetime(2026, 7, 16, 0, 0, tzinfo=UTC)

    def test_intraday_bar_is_a_real_instant_and_converts(self) -> None:
        # 14:30 New York on a summer date is 18:30 UTC. Unlike a daily bar,
        # this one denotes a moment, so conversion is correct.
        stamp = pd.Timestamp("2026-07-16 14:30:00", tz="America/New_York")
        result = _index_to_utc(stamp, Interval.M15)
        assert result == datetime(2026, 7, 16, 18, 30, tzinfo=UTC)

    def test_non_timestamp_is_rejected(self) -> None:
        assert _index_to_utc("2026-07-16", Interval.D1) is None


class TestToDecimal:
    def test_nan_becomes_none_not_a_poisoned_decimal(self) -> None:
        """NaN must never reach a Decimal column.

        Decimal('nan') is constructible and silently false-y in every
        comparison, so a NaN close would make an instrument look permanently
        below every threshold rather than raising.
        """
        assert _to_decimal(float("nan")) is None

    def test_infinity_becomes_none(self) -> None:
        assert _to_decimal(float("inf")) is None
        assert _to_decimal(float("-inf")) is None

    def test_none_stays_none(self) -> None:
        assert _to_decimal(None) is None

    def test_float_converts_via_str_to_avoid_binary_artefacts(self) -> None:
        # Decimal(0.1) is 0.1000000000000000055511151231257827.
        # Decimal(str(0.1)) is 0.1.
        assert _to_decimal(0.1) == Decimal("0.1")

    def test_garbage_becomes_none(self) -> None:
        assert _to_decimal("not-a-number") is None


class TestIsBarClosed:
    def test_bar_whose_duration_has_elapsed_is_closed(self) -> None:
        now = datetime(2026, 7, 16, 15, 0, tzinfo=UTC)
        opened = now - timedelta(minutes=20)
        assert _is_bar_closed(opened, Interval.M15, now)

    def test_bar_still_forming_is_not_closed(self) -> None:
        now = datetime(2026, 7, 16, 15, 0, tzinfo=UTC)
        opened = now - timedelta(minutes=5)
        assert not _is_bar_closed(opened, Interval.M15, now)

    def test_boundary_is_inclusive(self) -> None:
        # A bar is closed the instant its duration completes.
        now = datetime(2026, 7, 16, 15, 0, tzinfo=UTC)
        opened = now - timedelta(minutes=15)
        assert _is_bar_closed(opened, Interval.M15, now)


class TestFrameToCandles:
    @pytest.fixture
    def provider(self) -> YFinanceProvider:
        return YFinanceProvider(max_retries=1)

    def _frame(self, rows: list[dict[str, object]], tz: str = "Europe/London") -> pd.DataFrame:
        index = pd.DatetimeIndex([r.pop("ts") for r in rows], tz=tz, name="Date")
        return pd.DataFrame(rows, index=index)

    def test_gbx_frame_is_normalised_to_pounds(self, provider: YFinanceProvider) -> None:
        """A pence-quoted frame must come out in pounds, once."""
        frame = self._frame(
            [
                {
                    "ts": "2026-07-15 00:00:00",
                    "Open": 5700.0,
                    "High": 5800.0,
                    "Low": 5690.0,
                    "Close": 5758.84,
                    "Adj Close": 5758.84,
                    "Volume": 123456,
                }
            ]
        )
        candles = provider._frame_to_candles(frame, "SGLN.L", Interval.D1, PriceUnit.GBX)

        assert len(candles) == 1
        candle = candles[0]
        assert candle.close == Decimal("57.5884")
        assert candle.open == Decimal("57.00")
        assert candle.price_unit is PriceUnit.GBP
        assert candle.currency == "GBP"
        # Volume is a share count, not a price — it must not be divided by 100.
        assert candle.volume == Decimal("123456")

    def test_gbp_frame_is_left_alone(self, provider: YFinanceProvider) -> None:
        frame = self._frame(
            [
                {
                    "ts": "2026-07-16 00:00:00",
                    "Open": 108.04,
                    "High": 108.42,
                    "Low": 107.68,
                    "Close": 108.36,
                    "Adj Close": 108.36,
                    "Volume": 258941,
                }
            ]
        )
        candles = provider._frame_to_candles(frame, "VUAG.L", Interval.D1, PriceUnit.GBP)

        assert candles[0].close == Decimal("108.36")
        assert candles[0].price_unit is PriceUnit.GBP

    def test_daily_bar_retains_its_calendar_date(self, provider: YFinanceProvider) -> None:
        frame = self._frame(
            [
                {
                    "ts": "2026-07-16 00:00:00",
                    "Open": 108.04,
                    "High": 108.42,
                    "Low": 107.68,
                    "Close": 108.36,
                    "Adj Close": 108.36,
                    "Volume": 258941,
                }
            ]
        )
        candles = provider._frame_to_candles(frame, "VUAG.L", Interval.D1, PriceUnit.GBP)
        assert candles[0].timestamp == datetime(2026, 7, 16, tzinfo=UTC)

    def test_row_with_nan_ohlc_is_dropped_not_zeroed(self, provider: YFinanceProvider) -> None:
        """A padded holiday row is not a candle.

        Dropping is correct — the gap detector will notice the hole. Coercing
        NaN to 0 would invent a 100% crash.
        """
        frame = self._frame(
            [
                {
                    "ts": "2026-07-15 00:00:00",
                    "Open": 100.0,
                    "High": 101.0,
                    "Low": 99.0,
                    "Close": 100.5,
                    "Adj Close": 100.5,
                    "Volume": 1000,
                },
                {
                    "ts": "2026-07-16 00:00:00",
                    "Open": float("nan"),
                    "High": float("nan"),
                    "Low": float("nan"),
                    "Close": float("nan"),
                    "Adj Close": float("nan"),
                    "Volume": 0,
                },
            ]
        )
        candles = provider._frame_to_candles(frame, "VUAG.L", Interval.D1, PriceUnit.GBP)
        assert len(candles) == 1
        assert candles[0].timestamp == datetime(2026, 7, 15, tzinfo=UTC)

    def test_incoherent_ohlc_is_dropped(self, provider: YFinanceProvider) -> None:
        # high < low is impossible; the provider has sent something we cannot
        # interpret, and trading on it is worse than having no bar.
        frame = self._frame(
            [
                {
                    "ts": "2026-07-15 00:00:00",
                    "Open": 100.0,
                    "High": 90.0,
                    "Low": 110.0,
                    "Close": 100.5,
                    "Adj Close": 100.5,
                    "Volume": 1000,
                }
            ]
        )
        candles = provider._frame_to_candles(frame, "VUAG.L", Interval.D1, PriceUnit.GBP)
        assert candles == []

    def test_candles_come_back_ascending(self, provider: YFinanceProvider) -> None:
        frame = self._frame(
            [
                {
                    "ts": f"2026-07-{day:02d} 00:00:00",
                    "Open": 100.0,
                    "High": 101.0,
                    "Low": 99.0,
                    "Close": 100.5,
                    "Adj Close": 100.5,
                    "Volume": 1000,
                }
                for day in (16, 14, 15)
            ]
        )
        candles = provider._frame_to_candles(frame, "VUAG.L", Interval.D1, PriceUnit.GBP)
        timestamps = [c.timestamp for c in candles]
        assert timestamps == sorted(timestamps)


class TestCandleDTO:
    def test_naive_timestamp_is_rejected_at_construction(self) -> None:
        """Timezone-naive stamps are a bug, caught where they are introduced."""
        with pytest.raises(ValueError, match="timezone-aware"):
            Candle(
                symbol="SPY",
                interval=Interval.D1,
                timestamp=datetime(2026, 7, 16),
                open=Decimal("1"),
                high=Decimal("1"),
                low=Decimal("1"),
                close=Decimal("1"),
                volume=None,
                currency="USD",
                price_unit=PriceUnit.USD,
                provider=ProviderKind.MOCK,
            )

    def test_coherent_candle_passes(self) -> None:
        candle = Candle(
            symbol="SPY",
            interval=Interval.D1,
            timestamp=datetime(2026, 7, 16, tzinfo=UTC),
            open=Decimal("100"),
            high=Decimal("102"),
            low=Decimal("99"),
            close=Decimal("101"),
            volume=Decimal("1000"),
            currency="USD",
            price_unit=PriceUnit.USD,
            provider=ProviderKind.MOCK,
        )
        assert candle.is_coherent

    def test_close_outside_high_low_is_incoherent(self) -> None:
        candle = Candle(
            symbol="SPY",
            interval=Interval.D1,
            timestamp=datetime(2026, 7, 16, tzinfo=UTC),
            open=Decimal("100"),
            high=Decimal("102"),
            low=Decimal("99"),
            close=Decimal("150"),
            volume=Decimal("1000"),
            currency="USD",
            price_unit=PriceUnit.USD,
            provider=ProviderKind.MOCK,
        )
        assert not candle.is_coherent
