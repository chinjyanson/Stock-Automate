"""Point-in-time feature computation for signal research (§8).

Every feature here is a candidate input to a model that predicts forward returns.
They are computed **vectorised over a whole series at once** rather than by
replaying bar by bar, because a per-bar loop over a thousand instruments is the
difference between a run that takes minutes and one that takes an hour.

The safety property that matters is unchanged: **every value at index `i` uses
only bars 0..i.** Rolling windows end at `i`, never straddle it. A feature that
peeked would produce a spectacular and entirely fictional result, which is the
easiest way to waste weeks building a model on it.

Features are returned as arrays aligned to the input series, with `nan` wherever
there was not enough history to compute one. Callers drop those rather than
filling them — an imputed feature value is a fabricated observation, and a model
fitted on fabrications learns the fabrication.

Deliberately **not** here: anything that needs a second instrument (sector,
benchmark) is computed by the caller, which has the other series to hand; and
CAPM beta, which the user excluded.
"""

from __future__ import annotations

import numpy as np

from app.indicators.functions import FloatArray

#: Minimum bars before any feature is considered readable.
MIN_BARS = 220


def _rolling_mean(values: FloatArray, window: int) -> FloatArray:
    """Mean of the trailing `window` values, ending at each index.

    NaN-safe, and that is not a nicety. `np.cumsum` propagates NaN to every
    subsequent element, so a naive cumulative-sum implementation returns NaN for
    the *entire series* if the first value is NaN — which it is for anything
    derived from a difference, such as true range or returns. That silently
    emptied four features here (atr_pct, atr_ratio, volatility_20d, gap_share)
    and they vanished from a feature ranking without any error being raised.

    A window containing a NaN yields NaN rather than a mean over the survivors:
    averaging fewer observations than asked for is a different statistic, and
    quietly substituting it is how a subtly wrong number gets trusted.
    """
    out = np.full(values.size, np.nan)
    if values.size < window:
        return out
    finite = np.isfinite(values)
    filled = np.where(finite, values, 0.0)
    total = np.cumsum(np.insert(filled, 0, 0.0))
    count = np.cumsum(np.insert(finite.astype(np.float64), 0, 0.0))
    window_sum = total[window:] - total[:-window]
    window_count = count[window:] - count[:-window]
    complete = window_count == window
    out[window - 1 :] = np.where(complete, window_sum / window, np.nan)
    return out


def _rolling_std(values: FloatArray, window: int) -> FloatArray:
    """Sample standard deviation over the trailing `window` values."""
    out = np.full(values.size, np.nan)
    if values.size < window:
        return out
    mean = _rolling_mean(values, window)
    mean_sq = _rolling_mean(values * values, window)
    variance = np.maximum(mean_sq - mean * mean, 0.0) * window / max(window - 1, 1)
    out = np.sqrt(variance)
    return out


def _wilder_rsi(closes: FloatArray, period: int = 14) -> FloatArray:
    """Wilder's RSI as a series, matching `indicators.relative_strength_index`.

    Wilder smoothing is an EMA with alpha = 1/period, so this is a single pass
    rather than a window recomputation — the whole reason it is worth writing
    out rather than calling the scalar version once per bar.
    """
    out = np.full(closes.size, np.nan)
    if closes.size < period + 1:
        return out
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = float(gains[:period].mean())
    avg_loss = float(losses[:period].mean())
    for i in range(period, deltas.size):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            out[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            out[i + 1] = 100.0 - 100.0 / (1.0 + rs)
    return out


def _true_range(high: FloatArray, low: FloatArray, close: FloatArray) -> FloatArray:
    previous = np.concatenate(([np.nan], close[:-1]))
    stacked = np.maximum.reduce([high - low, np.abs(high - previous), np.abs(low - previous)])
    return np.asarray(stacked, dtype=np.float64)


def compute(
    open_: FloatArray,
    high: FloatArray,
    low: FloatArray,
    close: FloatArray,
    volume: FloatArray,
) -> dict[str, FloatArray]:
    """Every candidate feature, aligned to the input series.

    Grouped by the *kind* of information each carries, because that is what
    decides whether two features are worth having together. Three views of "how
    far below its average is this" are one feature wearing three hats — which is
    exactly what the strategy's entry score turned out to be.
    """
    n = close.size
    nan = np.full(n, np.nan)
    if n < MIN_BARS:
        return {}

    features: dict[str, FloatArray] = {}

    # -- Mean reversion: how stretched, relative to its own recent range -------
    features["rsi_14"] = _wilder_rsi(close, 14)
    features["rsi_2"] = _wilder_rsi(close, 2)  # a much faster read of the same idea

    sma20 = _rolling_mean(close, 20)
    sma50 = _rolling_mean(close, 50)
    sma200 = _rolling_mean(close, 200)
    with np.errstate(divide="ignore", invalid="ignore"):
        features["discount_sma20"] = (sma20 - close) / sma20
        features["discount_sma200"] = (sma200 - close) / sma200

        std20 = _rolling_std(close, 20)
        band_width = 4.0 * std20
        features["percent_b"] = np.where(
            band_width > 0, (close - (sma20 - 2.0 * std20)) / band_width, np.nan
        )

    # -- Trend: direction and structure, not distance --------------------------
    with np.errstate(divide="ignore", invalid="ignore"):
        features["sma50_over_sma200"] = sma50 / sma200 - 1.0  # the "golden cross" margin
        # Slope of the 200-day average over 21 bars, normalised by its level.
        slope = np.full(n, np.nan)
        slope[21:] = (sma200[21:] - sma200[:-21]) / (21.0 * sma200[21:])
        features["sma200_slope"] = slope

    # -- Momentum: what it has already done ------------------------------------
    for label, window in (("1m", 21), ("3m", 63), ("12m", 252)):
        out = np.full(n, np.nan)
        if n > window:
            with np.errstate(divide="ignore", invalid="ignore"):
                out[window:] = close[window:] / close[:-window] - 1.0
        features[f"return_{label}"] = out

    # -- Position in the 52-week range ----------------------------------------
    high_252 = np.full(n, np.nan)
    low_252 = np.full(n, np.nan)
    for i in range(251, n):
        recent = close[i - 251 : i + 1]
        high_252[i] = float(np.max(recent))
        low_252[i] = float(np.min(recent))
    with np.errstate(divide="ignore", invalid="ignore"):
        features["below_52w_high"] = (high_252 - close) / high_252
        features["above_52w_low"] = (close - low_252) / low_252
        span = high_252 - low_252
        features["position_in_range"] = np.where(span > 0, (close - low_252) / span, np.nan)

    # -- Volume: is the selling exhausting? A genuinely different axis. --------
    vol20 = _rolling_mean(volume, 20)
    vol60 = _rolling_mean(volume, 60)
    with np.errstate(divide="ignore", invalid="ignore"):
        features["volume_spike"] = np.where(vol20 > 0, volume / vol20, np.nan)
        features["volume_trend"] = np.where(vol60 > 0, vol20 / vol60, np.nan)

    # -- Volatility regime: expanding (danger) or contracting (settling)? ------
    true_range = _true_range(high, low, close)
    atr14 = _rolling_mean(true_range, 14)
    atr60 = _rolling_mean(true_range, 60)
    with np.errstate(divide="ignore", invalid="ignore"):
        features["atr_pct"] = np.where(close > 0, atr14 / close, np.nan)
        features["atr_ratio"] = np.where(atr60 > 0, atr14 / atr60, np.nan)

        returns = np.concatenate(([np.nan], np.diff(close) / close[:-1]))
        features["volatility_20d"] = _rolling_std(returns, 20) * np.sqrt(252.0)
        downside = np.where(returns < 0, returns, 0.0)
        features["downside_dev_20d"] = _rolling_std(downside, 20) * np.sqrt(252.0)

    # -- Event vs drift: did it gap, or bleed? --------------------------------
    with np.errstate(divide="ignore", invalid="ignore"):
        previous_close = np.concatenate(([np.nan], close[:-1]))
        features["overnight_gap"] = open_ / previous_close - 1.0
        # How much of the last 5 days' move happened overnight. High means news.
        gap_5 = _rolling_mean(np.abs(open_ / previous_close - 1.0), 5)
        move_5 = _rolling_mean(np.abs(returns), 5)
        features["gap_share"] = np.where(move_5 > 0, gap_5 / move_5, np.nan)

    # -- Streak: how many consecutive down days --------------------------------
    down = (np.diff(close, prepend=close[0]) < 0).astype(np.float64)
    streak = np.zeros(n)
    for i in range(1, n):
        streak[i] = streak[i - 1] + 1.0 if down[i] else 0.0
    features["down_streak"] = streak

    # Everything below MIN_BARS is unreadable regardless of what was computed.
    for key in features:
        features[key][:MIN_BARS] = nan[:MIN_BARS]
    return features


def forward_returns(close: FloatArray, horizon: int) -> FloatArray:
    """Return from each bar to `horizon` bars later; nan past the end.

    The label. Deliberately simple: no stop, no target, no position sizing — a
    feature either carries information about what price does next or it does
    not, and inserting trade mechanics between the two only obscures that.
    """
    out = np.full(close.size, np.nan)
    if close.size <= horizon:
        return out
    with np.errstate(divide="ignore", invalid="ignore"):
        out[: -horizon or None] = close[horizon:] / close[:-horizon] - 1.0
    return out
