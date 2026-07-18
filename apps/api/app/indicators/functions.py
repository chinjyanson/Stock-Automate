"""Pure technical indicators (§6).

Every function here is a pure function of numpy arrays, returns ``float | None``,
and returns ``None`` rather than guessing when there is not enough data. That
last rule is load-bearing: a momentum reading computed from 40 days of history
is not a small error, it is a different statistic, and silently returning one
would corrupt a score that looks authoritative.

Why float and not Decimal here: these are *statistical* measures — volatility,
drawdown, correlation — which are inherently floating-point and never settle
cash. Prices are stored as Decimal and converted to float arrays at the boundary
(`series.py`). Money math — position sizing, stop distances — stays in Decimal
and lives in the risk engine, not here.

Trading-day conventions (≈21/month, 252/year) match the annualisation used by
the store, so a "3-month return" and a "60-day volatility" line up with the
candle history they are computed from.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

#: Trading days, not calendar days. These are the windows §6 names.
TRADING_DAYS_PER_YEAR = 252
TRADING_DAYS_PER_MONTH = 21


def _clean(values: npt.ArrayLike) -> FloatArray:
    """Coerce to a 1-D float array. Callers pass already-cleaned closes."""
    return np.asarray(values, dtype=np.float64).ravel()


def simple_moving_average(closes: FloatArray, period: int) -> float | None:
    """Mean of the last `period` closes, or None if there are fewer."""
    closes = _clean(closes)
    if period <= 0 or closes.size < period:
        return None
    return float(np.mean(closes[-period:]))


def sma_series(closes: FloatArray, period: int) -> FloatArray:
    """Rolling SMA aligned to the input (leading positions are NaN).

    Used where a *series* of the moving average is needed — the 200-day slope,
    for instance — rather than a single latest value.
    """
    closes = _clean(closes)
    if period <= 0 or closes.size < period:
        return np.full(closes.shape, np.nan)
    kernel = np.ones(period) / period
    rolled = np.convolve(closes, kernel, mode="valid")
    out = np.full(closes.shape, np.nan)
    out[period - 1 :] = rolled
    return out


def daily_returns(closes: FloatArray) -> FloatArray:
    """Simple day-over-day returns. One shorter than the input."""
    closes = _clean(closes)
    if closes.size < 2:
        return np.array([], dtype=np.float64)
    return closes[1:] / closes[:-1] - 1.0


def trailing_return(closes: FloatArray, lookback_days: int) -> float | None:
    """Return over the last `lookback_days` bars.

    Compares the latest close with the close `lookback_days` bars earlier, so a
    21-day return uses 22 points. None if the history is shorter.
    """
    closes = _clean(closes)
    if lookback_days <= 0 or closes.size <= lookback_days:
        return None
    past = closes[-1 - lookback_days]
    if past <= 0:
        return None
    return float(closes[-1] / past - 1.0)


def annualised_volatility(closes: FloatArray, period: int) -> float | None:
    """Annualised standard deviation of daily returns over `period` bars.

    Sample standard deviation (ddof=1) — a volatility estimate from a sample,
    not a whole population — scaled by sqrt(252).
    """
    closes = _clean(closes)
    if closes.size < period + 1:
        return None
    rets = daily_returns(closes[-(period + 1) :])
    if rets.size < 2:
        return None
    return float(np.std(rets, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))


def downside_deviation(closes: FloatArray, period: int) -> float | None:
    """Annualised deviation of *negative* daily returns only (§6, Risk).

    Volatility punishes upside and downside alike; downside deviation isolates
    the losing days, which is what a long-only screen actually cares about.
    Divided by the full sample count (not just the negative days) so an
    instrument with few but severe down days is not flattered.
    """
    closes = _clean(closes)
    if closes.size < period + 1:
        return None
    rets = daily_returns(closes[-(period + 1) :])
    if rets.size < 2:
        return None
    negatives = np.minimum(rets, 0.0)
    downside = np.sqrt(np.mean(np.square(negatives)))
    return float(downside * np.sqrt(TRADING_DAYS_PER_YEAR))


def max_drawdown(closes: FloatArray, period: int | None = None) -> float | None:
    """Largest peak-to-trough decline over the window, as a positive fraction.

    0.30 means a 30% drop from a running peak. Returned positive because a
    "bigger drawdown" reading should be a bigger number; the sign is implicit.
    """
    closes = _clean(closes)
    window = closes if period is None else closes[-period:]
    if window.size < 2:
        return None
    running_peak = np.maximum.accumulate(window)
    # Guard the divide: a zero or negative peak is not a real price.
    with np.errstate(divide="ignore", invalid="ignore"):
        drawdowns = np.where(running_peak > 0, (window - running_peak) / running_peak, 0.0)
    return float(-np.min(drawdowns))


def largest_daily_loss(closes: FloatArray, period: int | None = None) -> float | None:
    """Worst single-day return over the window, as a positive fraction.

    0.12 means the worst day fell 12%. Positive for the same reason as drawdown.
    """
    closes = _clean(closes)
    window = closes if period is None else closes[-(period + 1) :]
    rets = daily_returns(window)
    if rets.size == 0:
        return None
    worst = float(np.min(rets))
    return -worst if worst < 0 else 0.0


def average_volume(volumes: FloatArray, period: int) -> float | None:
    volumes = _clean(volumes)
    if volumes.size < period:
        return None
    return float(np.mean(volumes[-period:]))


def average_traded_value(closes: FloatArray, volumes: FloatArray, period: int) -> float | None:
    """Mean of close x volume over `period` — a liquidity proxy in currency.

    Traded *value* matters more than share count: a penny stock trading millions
    of shares can still be untradeable in size.
    """
    closes = _clean(closes)
    volumes = _clean(volumes)
    n = min(closes.size, volumes.size)
    if n < period:
        return None
    value = closes[-period:] * volumes[-period:]
    return float(np.mean(value))


def zero_volume_days(volumes: FloatArray, period: int | None = None) -> int:
    """Count of zero-volume bars in the window — days nothing traded."""
    volumes = _clean(volumes)
    window = volumes if period is None else volumes[-period:]
    return int(np.sum(window == 0))


def stale_price_days(closes: FloatArray, period: int | None = None) -> int:
    """Count of bars whose close equals the previous close.

    A price that never moves usually means a stale feed or an illiquid line, not
    a genuinely flat market. Distinct from zero-volume, which it often
    accompanies but need not.
    """
    closes = _clean(closes)
    window = closes if period is None else closes[-(period + 1) :]
    if window.size < 2:
        return 0
    return int(np.sum(window[1:] == window[:-1]))


def sma_slope(closes: FloatArray, period: int, slope_window: int = 21) -> float | None:
    """Slope of the moving average over `slope_window`, as fractional change/bar.

    A least-squares fit through the last `slope_window` points of the SMA,
    normalised by the SMA's current level so it is comparable across instruments
    of different price. 0.001 means the trend line is rising ~0.1% per day.
    """
    series = sma_series(closes, period)
    tail = series[-slope_window:]
    tail = tail[~np.isnan(tail)]
    if tail.size < 2:
        return None
    x = np.arange(tail.size, dtype=np.float64)
    slope = float(np.polyfit(x, tail, 1)[0])
    level = float(tail[-1])
    if level <= 0:
        return None
    return slope / level


def relative_strength_index(closes: FloatArray, period: int = 14) -> float | None:
    """Wilder's RSI over `period` bars, 0-100.

    RSI compares the size of recent gains to recent losses. Below ~30 is
    conventionally "oversold" (a candidate for mean-reversion / potentially
    cheap), above ~70 "overbought". Uses Wilder's smoothing (an exponential
    average with alpha = 1/period), the standard definition — a plain mean would
    give a subtly different, non-comparable number.

    Returns 100.0 when there have been no losses in the window (RSI is undefined
    there; 100 is the conventional limit).
    """
    closes = _clean(closes)
    if closes.size < period + 1:
        return None
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    # Seed with the simple average of the first `period`, then Wilder-smooth.
    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))
    for i in range(period, deltas.size):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100.0 - 100.0 / (1.0 + rs))


def distance_from_high(closes: FloatArray, period: int = TRADING_DAYS_PER_YEAR) -> float | None:
    """How far below the window's high the latest close sits, as a fraction.

    0.0 means at the high; 0.25 means 25% below it. Always ≥ 0.
    """
    closes = _clean(closes)
    window = closes[-period:]
    if window.size < 1:
        return None
    high = float(np.max(window))
    if high <= 0:
        return None
    return float((high - closes[-1]) / high)


def distance_from_low(closes: FloatArray, period: int = TRADING_DAYS_PER_YEAR) -> float | None:
    """How far above the window's low the latest close sits, as a fraction."""
    closes = _clean(closes)
    window = closes[-period:]
    if window.size < 1:
        return None
    low = float(np.min(window))
    if low <= 0:
        return None
    return float((closes[-1] - low) / low)


def position_in_range(closes: FloatArray, period: int = TRADING_DAYS_PER_YEAR) -> float | None:
    """Where the latest close sits in the window's range, 0 (low) to 1 (high)."""
    closes = _clean(closes)
    window = closes[-period:]
    if window.size < 2:
        return None
    high = float(np.max(window))
    low = float(np.min(window))
    if high <= low:
        return None
    return float((closes[-1] - low) / (high - low))


def average_true_range(
    highs: FloatArray, lows: FloatArray, closes: FloatArray, period: int = 14
) -> float | None:
    """ATR over `period` — the mean true range (§8, used for stops in Phase 3).

    True range is the greatest of: today's high-low, |high-prev close|, and
    |low-prev close|. It captures gaps that a plain high-low misses.
    """
    highs = _clean(highs)
    lows = _clean(lows)
    closes = _clean(closes)
    n = min(highs.size, lows.size, closes.size)
    if n < period + 1:
        return None
    highs, lows, closes = highs[-n:], lows[-n:], closes[-n:]
    prev_close = closes[:-1]
    tr = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(np.abs(highs[1:] - prev_close), np.abs(lows[1:] - prev_close)),
    )
    if tr.size < period:
        return None
    return float(np.mean(tr[-period:]))


def relative_momentum(
    closes: FloatArray, benchmark_closes: FloatArray, lookback_days: int
) -> float | None:
    """Instrument return minus benchmark return over the lookback (§6, Momentum).

    Positive means the instrument outpaced its benchmark; a rising tide that
    lifts everything equally scores zero. None if either series is too short.
    """
    own = trailing_return(closes, lookback_days)
    bench = trailing_return(benchmark_closes, lookback_days)
    if own is None or bench is None:
        return None
    return own - bench


def rolling_correlation(returns_a: FloatArray, returns_b: FloatArray, period: int) -> float | None:
    """Pearson correlation of two return series over the last `period` bars (§9).

    Feeds the correlation-based position sizing in the risk engine. None when
    either series is degenerate (no variance), where correlation is undefined.
    """
    a = _clean(returns_a)
    b = _clean(returns_b)
    n = min(a.size, b.size)
    if n < period:
        return None
    a, b = a[-period:], b[-period:]
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])
