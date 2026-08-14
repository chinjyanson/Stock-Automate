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

from collections.abc import Sequence
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


@dataclass(frozen=True, slots=True)
class Gathered:
    """The index and its features, before anything has been fitted to them.

    Split out from `load` because fetching and building is the slow part and is
    identical across models, while fitting is fast and is the part an ablation
    wants to repeat. Keeping them apart means "refit without this feature" costs
    a fit rather than a download.
    """

    index: pd.DatetimeIndex
    close: np.ndarray
    daily: np.ndarray
    signals: dict[str, np.ndarray]
    #: The features this history can support. The insider series begins in 2006,
    #: so an earlier start silently has fewer of them available.
    available: tuple[str, ...]


def gather(path: Path, *, since: str, until: str | None) -> Gathered:
    """Fetch and build. The only function here that touches a network."""
    import yfinance as yf

    frame = yf.Ticker("^GSPC").history(period="max", interval="1d")
    frame = frame[frame.index >= since]
    if until:
        frame = frame[frame.index < until]
    close = frame["Close"].to_numpy(dtype=np.float64)
    index = pd.DatetimeIndex(frame.index.tz_localize(None)).normalize()
    daily = np.concatenate([[np.nan], close[1:] / close[:-1] - 1.0])

    available = FEATURES
    if since < "2006-01-01":
        available = tuple(f for f in FEATURES if f != "insider_rank")

    return Gathered(
        index=index,
        close=close,
        daily=daily,
        signals=_build(path, index, close, daily),
        available=available,
    )


def assemble(
    source: Gathered,
    *,
    split_date: str,
    fall: float,
    horizon: int,
    features: Sequence[str] | None = None,
) -> Pipeline:
    """Fit on data strictly before `split_date`, then score every bar.

    `features` selects a subset, for asking what any one of them is worth. It is
    intersected with what the history supports rather than trusted, so naming a
    feature that this date range cannot produce narrows the model instead of
    fitting it against a column of NaN.
    """
    index, close, daily = source.index, source.close, source.daily
    signals = source.signals
    n = close.size

    chosen = tuple(f for f in (features if features is not None else source.available)
                   if f in source.available)
    if not chosen:
        raise SystemExit(
            f"no usable features: asked for {tuple(features or ())}, "
            f"this history supports {source.available}"
        )
    begin = INSIDER_MIN_HISTORY + 1 if "insider_rank" in chosen else CALIBRATION_MIN

    cut = int(index.searchsorted(pd.Timestamp(split_date)))
    train = np.array([i for i in range(begin, cut - horizon - 1) if np.isfinite(daily[i])])
    if train.size == 0:
        raise SystemExit(
            f"no training bars before {split_date}: start --since earlier, or split later"
        )

    model = fit(
        _rows(signals, train, chosen),
        np.array([_label_fall(close, i, fall=fall, horizon=horizon) for i in train]),
        chosen,
        priors={f: Prior(0.0, 1.0) for f in FEATURES},
        label_definition=f"fall of {fall:.0%} within {horizon} days",
    )

    every = np.array([i for i in range(begin, n) if np.isfinite(daily[i])])
    probability = np.full(n, np.nan)
    probability[every] = [
        model.probability(
            {f: float(v) for f, v in zip(chosen, row, strict=True) if np.isfinite(v)}
        )
        for row in _rows(signals, every, chosen)
    ]

    return Pipeline(
        index=index,
        close=close,
        daily=daily,
        signals=signals,
        features=chosen,
        model=model,
        probability=probability,
        cut=cut,
        begin=begin,
    )


def load(
    path: Path,
    *,
    since: str,
    until: str | None,
    split_date: str,
    fall: float,
    horizon: int,
) -> Pipeline:
    """Fetch, build, fit and score, with every feature the history supports."""
    source = gather(path, since=since, until=until)
    return assemble(source, split_date=split_date, fall=fall, horizon=horizon)


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
