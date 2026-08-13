"""The crash detector's features, and the overlay's decision rule (§9).

**This module is the single definition of both.** The research script and the
nightly job import it rather than each computing "the same" features their own
way, because the failure that matters here is silent: a serving path that
computes a feature slightly differently from the fitting path produces a model
that backtests well and trades badly, and nothing raises.

## What the detector answers

Not "will the market go up" — direction failed at every horizon tried. It
answers **"is a sharp fall imminent?"**, which measurement says is a different
and easier question: on identical features the fall question scores AUC 0.61-0.72
where direction scores 0.41-0.52.

Eleven features, all market-wide. Most are credit- and volatility-stress
measures, which is worth knowing because it bounds what the model can see: it
detects *slow, credit-driven* declines, which telegraph themselves for months.
It did well on 2008 and much less well on COVID, an exogenous shock with no
credit lead time.

## What the overlay does with it

Asymmetric by construction, on one stated assumption: **the index rises over
time, so being invested is the default and selling must justify itself.**

  * A warning steps the position down to `defensive`.
  * The position returns *in full* the first day the warning clears — the
    all-clear rule. The worst days and the best days are neighbours, so the
    recovery is what must not be missed.
  * A timeout returns it anyway, so a warning that never resolves costs days
    rather than years.

## The trigger is a rolling percentile, and that is not a detail

An absolute threshold never fires: with a 3.4% base rate the model rarely emits
a probability above 0.5, which looks like caution and is a broken switch. A
percentile of the *training* output is barely better, because training spans
2008 and its 95th percentile is a bar no ordinary year clears. So the trigger is
a percentile of a **rolling window of the model's own recent output** — "alarming
lately", which is the question a trigger actually needs answered.
"""

from __future__ import annotations

import io
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

TRADING_DAYS = 252
VOL_WINDOW = 20
INSIDER_SMOOTH = 126
INSIDER_MIN_HISTORY = 504

#: Probabilities needed before a percentile of them means anything. Two years,
#: so the first live decision is ranked against a real distribution.
CALIBRATION_MIN = 504

#: How far back the trigger looks. **Rolling, not expanding** — see the module
#: docstring; an expanding window stays pinned to 2008 forever.
CALIBRATION_WINDOW = 504

#: The index the overlay times, and the series every price feature is relative
#: to. `^GSPC` rather than a tracker: it has the longest clean history.
INDEX_SYMBOL = "^GSPC"

#: History start. The insider series begins in 2006 and is the binding
#: constraint; earlier starts must drop that feature.
DEFAULT_SINCE = "2006-01-01"

FEATURES: tuple[str, ...] = (
    "vix",
    "vix_term_structure",
    "credit_spread",
    "hyg_tlt",
    "hyg_lqd",
    "skew",
    "small_cap_rs",
    "realised_vol",
    "vol_of_vol",
    "drawdown_from_high",
    "insider_rank",
)


def fred(series_id: str, index: pd.DatetimeIndex) -> np.ndarray:
    """A FRED series, forward-filled onto `index`. Keyless CSV endpoint."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd=1990-01-01"
    raw = urllib.request.urlopen(url, timeout=60).read().decode()
    frame = pd.read_csv(io.StringIO(raw))
    frame.columns = ["date", "value"]
    frame = frame[frame["value"] != "."]
    frame["date"] = pd.to_datetime(frame["date"])
    values = pd.Series(frame["value"].astype(float).to_numpy(), index=frame["date"])
    out: np.ndarray = values.reindex(index, method="ffill").to_numpy(dtype=np.float64)
    return out


def yahoo(symbol: str, index: pd.DatetimeIndex) -> np.ndarray:
    """A daily close series, forward-filled onto `index`."""
    import yfinance as yf

    closes = yf.Ticker(symbol).history(period="max", interval="1d")["Close"]
    closes.index = closes.index.tz_localize(None).normalize()
    out: np.ndarray = closes.reindex(index, method="ffill").to_numpy(dtype=np.float64)
    return out


def momentum(ratio: np.ndarray, window: int = 60) -> np.ndarray:
    out = np.full(ratio.size, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        out[window:] = ratio[window:] / ratio[:-window] - 1.0
    return out


def insider_rank(path: Path, index: pd.DatetimeIndex) -> np.ndarray:
    """Aggregate insider buying, as a percentile of its own past.

    Ranked against history rather than used raw, because the level drifts with
    how many filings the SEC receives; the percentile is what stays comparable
    across two decades. `NaN` until there are two years to rank against.
    """
    frame = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    officer = frame.reindex(index, method="ffill")["officer_buy_share"].to_numpy(dtype=np.float64)
    smooth = np.full(officer.size, np.nan)
    for i in range(INSIDER_SMOOTH, officer.size):
        chunk = officer[i - INSIDER_SMOOTH : i]
        chunk = chunk[np.isfinite(chunk)]
        if chunk.size >= INSIDER_SMOOTH // 2:
            smooth[i] = float(np.mean(chunk))
    rank = np.full(smooth.size, np.nan)
    for i in range(INSIDER_MIN_HISTORY, smooth.size):
        history = smooth[:i][np.isfinite(smooth[:i])]
        if history.size >= INSIDER_MIN_HISTORY and np.isfinite(smooth[i]):
            rank[i] = float(np.mean(history < smooth[i]))
    return rank


def build(
    path: Path, index: pd.DatetimeIndex, close: np.ndarray, daily: np.ndarray
) -> dict[str, np.ndarray]:
    """Every feature, aligned to `index`. Missing values stay `NaN`."""
    n = close.size
    realised = np.full(n, np.nan)
    for i in range(VOL_WINDOW, n):
        realised[i] = float(np.std(daily[i - VOL_WINDOW : i], ddof=1)) * np.sqrt(TRADING_DAYS)

    # How unstable the instability itself is: volatility spikes tend to be
    # preceded by volatility becoming erratic rather than merely high.
    vol_of_vol = np.full(n, np.nan)
    for i in range(VOL_WINDOW * 3, n):
        window = realised[i - VOL_WINDOW * 2 : i]
        window = window[np.isfinite(window)]
        if window.size > 10:
            vol_of_vol[i] = float(np.std(window, ddof=1))

    # Where price sits against its own recent high. Falls beget falls, and a
    # market already off its peak is in a different state from one making highs.
    running_high = np.maximum.accumulate(close)
    from_high = close / running_high - 1.0

    vix, vix3m, skew = (yahoo(s, index) for s in ("^VIX", "^VIX3M", "^SKEW"))
    hyg, lqd, tlt, iwm = (yahoo(s, index) for s in ("HYG", "LQD", "TLT", "IWM"))
    with np.errstate(divide="ignore", invalid="ignore"):
        return {
            "vix": vix,
            "vix_term_structure": vix3m / vix,
            "credit_spread": fred("DBAA", index) - fred("DAAA", index),
            "hyg_tlt": momentum(hyg / tlt),
            "hyg_lqd": momentum(hyg / lqd),
            "skew": skew,
            "small_cap_rs": momentum(iwm / close),
            "realised_vol": realised,
            "vol_of_vol": vol_of_vol,
            "drawdown_from_high": from_high,
            "insider_rank": (insider_rank(path, index) if path.exists() else np.full(n, np.nan)),
        }


def rows(
    signals: dict[str, np.ndarray], at: np.ndarray, features: tuple[str, ...] = FEATURES
) -> np.ndarray:
    """Feature matrix for the given row indices, in `features` order."""
    return np.column_stack([[float(signals[f][i]) for f in features] for i in at]).T


def label_fall(close: np.ndarray, i: int, *, fall: float, horizon: int) -> float:
    """1 when the next `horizon` days contain a fall of `fall` from day `i`."""
    ahead = close[i : i + 1 + horizon]
    return 1.0 if float(np.min(ahead) / close[i] - 1.0) <= -fall else 0.0


def trigger_from(past: np.ndarray, fraction: float) -> float | None:
    """The probability above which to warn, as a percentile of recent output.

    `None` until there is enough history for a percentile to mean anything —
    which the caller must treat as "do not warn", never as "warn".
    """
    usable = past[np.isfinite(past)]
    if usable.size < CALIBRATION_MIN:
        return None
    return float(np.quantile(usable, 1.0 - fraction))


@dataclass(frozen=True)
class OverlayState:
    """Where the overlay stands. Persisted between days, hence a value type.

    `exit_price` doubles as the in/out flag: `None` means fully invested and the
    only live question is whether to step aside.
    """

    exposure: float = 1.0
    exit_price: float | None = None
    days_out: int = 0

    @property
    def is_defensive(self) -> bool:
        return self.exit_price is not None


def step(
    state: OverlayState,
    *,
    probability: float | None,
    trigger: float | None,
    close: float,
    defensive: float,
    timeout: int,
) -> tuple[OverlayState, str]:
    """One day of the overlay. Returns the new state and why it moved.

    **Fails invested, not defensive.** With no probability or no trigger the
    position stays where a long-only index holder would want it — a missing data
    feed must not be able to sell the portfolio. The asymmetry is deliberate and
    matches the module's founding assumption.
    """
    if probability is None or trigger is None:
        if state.is_defensive:
            # Already aside and now flying blind: come back rather than sit out
            # indefinitely on no information.
            return OverlayState(exposure=1.0), "no reading; returned to fully invested"
        return OverlayState(exposure=1.0), "no reading; holding fully invested"

    warning = probability >= trigger

    if not state.is_defensive:
        if warning:
            return (
                OverlayState(exposure=defensive, exit_price=close, days_out=0),
                f"warning fired (p={probability:.3f} >= {trigger:.3f}); "
                f"exposure cut to {defensive:.0%}",
            )
        return OverlayState(exposure=1.0), f"no warning (p={probability:.3f} < {trigger:.3f})"

    days_out = state.days_out + 1
    if not warning:
        # All clear. Whatever the price has done, the reason for standing aside
        # has gone, so step fully back in — the recovery is the thing not to
        # miss.
        return OverlayState(exposure=1.0), f"all clear after {days_out}d; fully reinvested"
    if days_out >= timeout:
        return (
            OverlayState(exposure=1.0),
            f"warning stood {days_out}d without resolving; timed out, fully reinvested",
        )
    return (
        OverlayState(exposure=defensive, exit_price=state.exit_price, days_out=days_out),
        f"warning still standing (p={probability:.3f} >= {trigger:.3f}), day {days_out}",
    )
