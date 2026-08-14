"""The S&P 500 in the shape Kronos expects, labelled the way production labels it.

Two rules govern this module.

**The label must be the production label.** `app.signals.crash_features.label_fall`
with `fall=0.02, horizon=1` marks bar *i* when `min(close[i:i+2]) / close[i] - 1`
is at or below `-2%`. Since `close[i] / close[i]` is zero, that reduces to "the
next close is 2% or more below this one" — and it is reimplemented here rather
than imported because this directory runs in a different virtualenv from the
API. `tests/test_data.py` pins the two definitions against each other on shared
fixtures, which is the only thing that keeps the reimplementation honest.

**A window ending at bar i may not contain bar i+1.** Every array here is
indexed by the *decision* bar. The model sees bars `i-511 .. i` inclusive and is
asked about bar `i+1`, which it has never been shown.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

#: Kronos-small and Kronos-base are trained with 512 positions. Longer input is
#: silently truncated by their predictor, so asking for more would be a lie.
LOOKBACK = 512

#: The order Kronos was pre-trained on. Do not permute.
COLUMNS = ("open", "high", "low", "close", "volume", "amount")

SYMBOL = "^GSPC"


@dataclass(frozen=True, slots=True)
class Market:
    """Daily bars, plus the label and the plain returns, all bar-aligned."""

    index: pd.DatetimeIndex
    bars: np.ndarray  # [n, 6] raw open/high/low/close/volume/amount
    label: np.ndarray  # [n] 1.0 when the next close is >=2% below this one
    forward: np.ndarray  # [n] next-day return, for diagnosis only

    @property
    def close(self) -> np.ndarray:
        return self.bars[:, COLUMNS.index("close")]

    def usable(
        self,
        *,
        since: str | None = None,
        until: str | None = None,
        lookback: int = LOOKBACK,
    ) -> np.ndarray:
        """Decision bars with a full lookback behind them and a known answer ahead.

        The final bar is excluded however recent the data is: its label depends
        on a close that has not happened.
        """
        at = np.arange(lookback - 1, self.index.size - 1)
        keep = np.isfinite(self.label[at]) & np.isfinite(self.bars[at].sum(axis=1))
        if since is not None:
            keep &= self.index[at] >= pd.Timestamp(since)
        if until is not None:
            keep &= self.index[at] < pd.Timestamp(until)
        return at[keep]

    def windows(self, at: np.ndarray, lookback: int = LOOKBACK) -> np.ndarray:
        """[len(at), lookback, 6] — bar `i` is the last row of its own window.

        Shorter windows are worth trying rather than assuming: 512 daily bars is
        two years of context, and a model pre-trained largely on faster bars may
        simply not use that much of it. The cost of finding out is one extra
        feature extraction.
        """
        offsets = np.arange(-lookback + 1, 1)
        return self.bars[at[:, None] + offsets[None, :]]

    def stamps(self, at: np.ndarray, lookback: int = LOOKBACK) -> pd.Series:
        """Timestamps for every bar of every window, flattened window-major."""
        offsets = np.arange(-lookback + 1, 1)
        return pd.Series(self.index[(at[:, None] + offsets[None, :]).reshape(-1)])


def label_fall(close: np.ndarray, *, fall: float = 0.02, horizon: int = 1) -> np.ndarray:
    """Vectorised `crash_features.label_fall` over every bar at once.

    NaN for the last `horizon` bars, where the answer is not yet known — the
    production scalar version would happily read a short slice and return 0.0
    there, which is a wrong answer rather than a missing one.
    """
    n = close.size
    out = np.full(n, np.nan)
    for i in range(n - horizon):
        ahead = close[i : i + 1 + horizon]
        out[i] = 1.0 if float(np.min(ahead) / close[i] - 1.0) <= -fall else 0.0
    return out


def fetch(*, since: str = "1990-01-01", until: str | None = None) -> Market:
    """Download `^GSPC` daily bars. The only function here that touches a network."""
    import yfinance as yf

    frame = yf.Ticker(SYMBOL).history(period="max", interval="1d")
    frame = frame[frame.index >= since]
    if until:
        frame = frame[frame.index < until]
    return _assemble(frame)


def _assemble(frame: pd.DataFrame) -> Market:
    index = pd.DatetimeIndex(frame.index.tz_localize(None)).normalize()
    price = frame[["Open", "High", "Low", "Close"]].to_numpy(dtype=np.float64)
    volume = frame["Volume"].to_numpy(dtype=np.float64)

    # Kronos was pre-trained with a turnover column beside volume. We have no
    # true turnover for an index, so we use their own documented fallback:
    # volume times the average of the four prices.
    amount = volume * price.mean(axis=1)

    bars = np.column_stack([price, volume, amount])
    close = price[:, 3]
    forward = np.full(close.size, np.nan)
    forward[:-1] = close[1:] / close[:-1] - 1.0

    return Market(index=index, bars=bars, label=label_fall(close), forward=forward)


def save(market: Market, path: Path) -> None:
    """Cache to disk so later runs neither re-download nor drift under us."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        index=market.index.values.astype("datetime64[D]"),
        bars=market.bars,
        label=market.label,
        forward=market.forward,
    )


def load(path: Path) -> Market:
    raw = np.load(path, allow_pickle=False)
    return Market(
        index=pd.DatetimeIndex(raw["index"]),
        bars=raw["bars"],
        label=raw["label"],
        forward=raw["forward"],
    )
