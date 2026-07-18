"""Convert stored candles into the float arrays indicators consume.

The boundary where Decimal prices become float arrays. It happens here, once,
and deliberately: everything downstream (indicators, scoring) is statistical and
floating-point, while everything upstream (the store) is exact Decimal. Keeping
the conversion in one place means there is a single answer to "where did the
floats come from".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.indicators.functions import FloatArray
from app.models.market_data import Candle


@dataclass(frozen=True, slots=True)
class PriceSeries:
    """Aligned OHLCV arrays for one instrument, oldest-first.

    All arrays share an index: `close[i]`, `high[i]`, `volume[i]` describe the
    same bar. Adjusted closes are carried separately because a total-return
    screen should prefer them where available.
    """

    open: FloatArray
    high: FloatArray
    low: FloatArray
    close: FloatArray
    adjusted_close: FloatArray
    volume: FloatArray

    @property
    def length(self) -> int:
        return int(self.close.size)

    def has_at_least(self, bars: int) -> bool:
        return self.length >= bars

    @property
    def preferred_close(self) -> FloatArray:
        """Adjusted close where every bar has one, else raw close.

        A partially-adjusted series is worse than a consistently raw one — mixing
        the two injects phantom jumps at the boundary — so this is all-or-nothing.
        """
        if self.adjusted_close.size == self.close.size and not np.isnan(self.adjusted_close).any():
            return self.adjusted_close
        return self.close


def candles_to_series(candles: list[Candle]) -> PriceSeries:
    """Build a `PriceSeries` from ascending, closed candles.

    Candles are assumed already ordered oldest-first and closed — the store's
    `get_candles` returns them that way by default. NaN fills any missing field
    rather than dropping the bar, so array indices stay aligned across O/H/L/C/V.
    """
    n = len(candles)
    opens = np.full(n, np.nan)
    highs = np.full(n, np.nan)
    lows = np.full(n, np.nan)
    closes = np.full(n, np.nan)
    adj = np.full(n, np.nan)
    volumes = np.full(n, np.nan)

    for i, candle in enumerate(candles):
        opens[i] = float(candle.open)
        highs[i] = float(candle.high)
        lows[i] = float(candle.low)
        closes[i] = float(candle.close)
        adj[i] = float(candle.adjusted_close) if candle.adjusted_close is not None else np.nan
        volumes[i] = float(candle.volume) if candle.volume is not None else 0.0

    return PriceSeries(
        open=opens,
        high=highs,
        low=lows,
        close=closes,
        adjusted_close=adj,
        volume=volumes,
    )
