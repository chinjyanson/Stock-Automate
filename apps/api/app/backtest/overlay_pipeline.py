"""The crash-overlay pipeline, shared by the backtests that need it.

`backtest_crash_overlay.py` grew this logic inline and it is now wanted by a
second script, so it lives here once: fetch the index, build the features, fit
the detector on data strictly before the traded window, score every day, and
turn those scores into a daily alarm threshold.

Nothing here decides anything about money. It stops at "how alarming was each
day, and what counted as alarming at the time", which is exactly the boundary
that lets one pipeline serve several trading rules without favouring any.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from app.indicators import functions as ind
from app.models_ml.logistic import FittedModel, Prior, fit
from app.signals.crash_features import (
    CALIBRATION_MIN,
    CALIBRATION_WINDOW,
    FEATURES,
    INSIDER_MIN_HISTORY,
)
from app.signals.crash_features import build as _build
from app.signals.crash_features import label_fall as _label_fall
from app.signals.crash_features import rows as _rows


@dataclass(frozen=True, slots=True)
class Pipeline:
    """Everything downstream of the model and upstream of a trading decision."""

    index: pd.DatetimeIndex
    close: np.ndarray
    daily: np.ndarray
    signals: dict[str, np.ndarray]
    features: tuple[str, ...]
    model: FittedModel
    #: Model probability per bar, from a fit that never saw that bar's future.
    probability: np.ndarray
    #: First bar of the traded window.
    cut: int
    #: First bar any probability exists for.
    begin: int

    def labels(self, at: np.ndarray, *, fall: float, horizon: int) -> np.ndarray:
        return np.array([_label_fall(self.close, i, fall=fall, horizon=horizon) for i in at])

    def triggers(self, fraction: float) -> np.ndarray:
        """The bar-by-bar probability above which to warn.

        A percentile of the model's own output over a **rolling** 504 days, not
        of all history: 2008's probabilities are so extreme that a bar set from
        them is one no ordinary year ever clears, which reads as caution and is
        actually a switch stuck off. Rolling asks "alarming lately", which is
        the question a trigger is for.
        """
        out = np.full(self.close.size, np.inf)
        for i in range(self.begin + CALIBRATION_MIN, self.close.size):
            past = self.probability[max(self.begin, i - CALIBRATION_WINDOW) : i]
            past = past[np.isfinite(past)]
            if past.size >= CALIBRATION_MIN:
                out[i] = float(np.quantile(past, 1.0 - fraction))
        return out


def load(
    path: Path,
    *,
    since: str,
    until: str | None,
    split_date: str,
    fall: float,
    horizon: int,
) -> Pipeline:
    """Fetch, build, fit and score. The only function here that touches a network."""
    import yfinance as yf

    frame = yf.Ticker("^GSPC").history(period="max", interval="1d")
    frame = frame[frame.index >= since]
    if until:
        frame = frame[frame.index < until]
    close = frame["Close"].to_numpy(dtype=np.float64)
    index = pd.DatetimeIndex(frame.index.tz_localize(None)).normalize()
    daily = np.concatenate([[np.nan], close[1:] / close[:-1] - 1.0])
    n = close.size

    signals = _build(path, index, close, daily)

    features = FEATURES
    if since < "2006-01-01":
        features = tuple(f for f in FEATURES if f != "insider_rank")
    begin = INSIDER_MIN_HISTORY + 1 if "insider_rank" in features else CALIBRATION_MIN

    cut = int(index.searchsorted(pd.Timestamp(split_date)))
    train = np.array([i for i in range(begin, cut - horizon - 1) if np.isfinite(daily[i])])
    if train.size == 0:
        raise SystemExit(
            f"no training bars before {split_date}: start --since earlier, or split later"
        )

    model = fit(
        _rows(signals, train, features),
        np.array([_label_fall(close, i, fall=fall, horizon=horizon) for i in train]),
        features,
        priors={f: Prior(0.0, 1.0) for f in FEATURES},
        label_definition=f"fall of {fall:.0%} within {horizon} days",
    )

    every = np.array([i for i in range(begin, n) if np.isfinite(daily[i])])
    probability = np.full(n, np.nan)
    probability[every] = [
        model.probability(
            {f: float(v) for f, v in zip(features, row, strict=True) if np.isfinite(v)}
        )
        for row in _rows(signals, every, features)
    ]

    return Pipeline(
        index=index,
        close=close,
        daily=daily,
        signals=signals,
        features=features,
        model=model,
        probability=probability,
        cut=cut,
        begin=begin,
    )


def rsi_series(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Wilder's RSI at every bar, in one pass.

    Equal at every bar to `indicators.relative_strength_index(close[: i + 1])`,
    because that function seeds on the first `period` deltas and smooths forward
    through whatever it is given — so the recursion here *is* that function,
    unrolled. `TestRsiSeriesMatchesProduction` pins the equality rather than
    trusting this paragraph.
    """
    out = np.full(close.size, np.nan)
    if close.size < period + 1:
        return out

    deltas = np.diff(close)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))
    out[period] = 100.0 if avg_loss == 0 else 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    for i in range(period, deltas.size):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        out[i + 1] = 100.0 if avg_loss == 0 else 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    return out


def context_arrays(close: np.ndarray, daily: np.ndarray) -> dict[str, np.ndarray]:
    """Per-bar readings the re-entry rules consult, all backward-looking."""
    n = close.size
    sma_ratio = np.full(n, np.nan)
    volatility = np.full(n, np.nan)
    for i in range(n):
        window = close[max(0, i - 60) : i + 1]
        average = ind.simple_moving_average(window, 10)
        if average:
            sma_ratio[i] = close[i] / average
        vol = ind.annualised_volatility(window, 10)
        if vol is not None:
            volatility[i] = vol

    streak = np.zeros(n, dtype=int)
    for i in range(1, n):
        streak[i] = streak[i - 1] + 1 if daily[i] > 0 else 0

    return {
        "rsi": rsi_series(close),
        "sma_ratio": sma_ratio,
        "volatility": volatility,
        "up_streak": streak.astype(float),
    }
