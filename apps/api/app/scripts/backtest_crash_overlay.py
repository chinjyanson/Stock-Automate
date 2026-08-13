"""Stay invested; step aside for crashes; buy back into the fall (§9).

    python -m app.scripts.backtest_crash_overlay --capital 5000

Every strategy tested so far has been symmetric: decide each period whether to
be in or out, which forces an opinion on rises as well as falls. Direction has
failed at every horizon tried, so this stops asking.

The design is asymmetric by construction, and rests on one assumption stated
plainly: **the index rises over time, so being invested is the default and
selling must justify itself.** Two questions, neither of which is "will it go
up":

  * **Is a sharp fall imminent?** If so, step aside. Deliberately reluctant —
    the probability threshold is high, because a false alarm costs real return
    in a market that mostly rises.
  * **Has it fallen far enough to buy back?** Mechanical, not predictive: as the
    price drops below the level it was sold at, capital returns in steps. No
    forecast of the bottom is required, only a ladder.

A timeout returns the position if the feared fall never arrives, so a wrong
warning costs days rather than years.

**On measuring the fall detector.** A sharp one-day fall is rare — a few percent
of days — so an AUC can look respectable while the trigger is useless in
practice. What matters for a trigger is what happens *when it fires*: how often
a warning is followed by the fall, and how much of the market's fall the warnings
actually cover. Both are reported, because the AUC alone would flatter it.
"""

from __future__ import annotations

import argparse
import io
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from app.models_ml.logistic import FittedModel, Prior, auc, fit

TRADING_DAYS = 252
VOL_WINDOW = 20
INSIDER_SMOOTH = 126
INSIDER_MIN_HISTORY = 504

#: Probabilities needed before a percentile of them means anything. Two years,
#: so the first held-out decision is ranked against a real distribution.
CALIBRATION_MIN = 504

#: How far back the trigger looks when setting its percentile. **Rolling, not
#: expanding.** An expanding window always contains 2008, whose probabilities
#: are so extreme that its 95th percentile is a bar no ordinary year clears —
#: calibrate that way and the switch stays off through the whole decade it was
#: meant to watch. A trailing window asks "alarming *lately*", which is the
#: question a trigger actually needs answered.
CALIBRATION_WINDOW = 504
BORROW = 0.05

FEATURES = (
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


def _fred(series_id: str, index: pd.DatetimeIndex) -> np.ndarray:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd=1990-01-01"
    raw = urllib.request.urlopen(url, timeout=60).read().decode()
    frame = pd.read_csv(io.StringIO(raw))
    frame.columns = ["date", "value"]
    frame = frame[frame["value"] != "."]
    frame["date"] = pd.to_datetime(frame["date"])
    values = pd.Series(frame["value"].astype(float).to_numpy(), index=frame["date"])
    out: np.ndarray = values.reindex(index, method="ffill").to_numpy(dtype=np.float64)
    return out


def _yahoo(symbol: str, index: pd.DatetimeIndex) -> np.ndarray:
    import yfinance as yf

    closes = yf.Ticker(symbol).history(period="max", interval="1d")["Close"]
    closes.index = closes.index.tz_localize(None).normalize()
    out: np.ndarray = closes.reindex(index, method="ffill").to_numpy(dtype=np.float64)
    return out


def _momentum(ratio: np.ndarray, window: int = 60) -> np.ndarray:
    out = np.full(ratio.size, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        out[window:] = ratio[window:] / ratio[:-window] - 1.0
    return out


def _insider_rank(path: Path, index: pd.DatetimeIndex) -> np.ndarray:
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


def _drawdown(curve: np.ndarray) -> float:
    peak = np.maximum.accumulate(curve)
    return float(np.max((peak - curve) / peak)) if curve.size else 0.0


def _build(
    path: Path, index: pd.DatetimeIndex, close: np.ndarray, daily: np.ndarray
) -> dict[str, np.ndarray]:
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

    vix, vix3m, skew = (_yahoo(s, index) for s in ("^VIX", "^VIX3M", "^SKEW"))
    hyg, lqd, tlt, iwm = (_yahoo(s, index) for s in ("HYG", "LQD", "TLT", "IWM"))
    with np.errstate(divide="ignore", invalid="ignore"):
        return {
            "vix": vix,
            "vix_term_structure": vix3m / vix,
            "credit_spread": _fred("DBAA", index) - _fred("DAAA", index),
            "hyg_tlt": _momentum(hyg / tlt),
            "hyg_lqd": _momentum(hyg / lqd),
            "skew": skew,
            "small_cap_rs": _momentum(iwm / close),
            "realised_vol": realised,
            "vol_of_vol": vol_of_vol,
            "drawdown_from_high": from_high,
            "insider_rank": _insider_rank(path, index),
        }


def _rows(signals: dict[str, np.ndarray], at: np.ndarray) -> np.ndarray:
    return np.column_stack([[float(signals[f][i]) for f in FEATURES] for i in at]).T


def _simulate(
    daily: np.ndarray,
    close: np.ndarray,
    signals: dict[str, np.ndarray],
    model: FittedModel | None,
    start: int,
    *,
    capital: float,
    mode: str,
    triggers: np.ndarray,
    defensive: float,
    ladder: float,
    timeout: int,
    cost: float,
) -> tuple[np.ndarray, int, float]:
    """Invested by default; step aside on a warning; ladder back in as it falls."""
    equity = capital
    exposure = 1.0
    curve: list[float] = []
    alarms = 0
    exit_price: float | None = None
    days_out = 0
    held: list[float] = []

    for i in range(start, daily.size):
        prior = i - 1
        if mode == "hold":
            wanted = 1.0
        else:
            reading = {
                f: float(signals[f][prior]) for f in FEATURES if np.isfinite(signals[f][prior])
            }
            probability = model.probability(reading) if (model is not None and reading) else 0.0

            if exit_price is None:
                # Fully invested: the only question is whether to step aside.
                if probability >= triggers[prior]:
                    exit_price = float(close[prior])
                    days_out = 0
                    alarms += 1
                    wanted = defensive
                else:
                    wanted = 1.0
            else:
                days_out += 1
                fallen = 1.0 - float(close[prior]) / exit_price
                # Ladder back in proportionally to how far it has fallen, so
                # capital returns *into* the decline rather than waiting for a
                # bottom nobody can identify.
                recovered = float(np.clip(fallen / ladder, 0.0, 1.0))
                wanted = defensive + (1.0 - defensive) * recovered
                if recovered >= 1.0 or days_out >= timeout:
                    # Either fully back in, or the feared fall never came.
                    exit_price = None
                    wanted = 1.0

        if abs(wanted - exposure) > 0.02:
            equity -= equity * abs(wanted - exposure) * cost / 2.0
            exposure = wanted

        equity *= 1.0 + exposure * daily[i]
        curve.append(equity)
        held.append(exposure)

    return np.asarray(curve), alarms, float(np.mean(held)) if held else 0.0


def _run(
    path: Path,
    capital: float,
    split: float,
    horizon: int,
    fall: float,
    fractions: list[float],
    defensive: float,
    ladder: float,
    timeout: int,
    cost: float,
) -> None:
    warnings.filterwarnings("ignore")
    import yfinance as yf

    frame = yf.Ticker("^GSPC").history(period="max", interval="1d")
    frame = frame[frame.index >= "2006-01-01"]
    close = frame["Close"].to_numpy(dtype=np.float64)
    index = pd.DatetimeIndex(frame.index.tz_localize(None)).normalize()
    daily = np.concatenate([[np.nan], close[1:] / close[:-1] - 1.0])
    n = close.size

    signals = _build(path, index, close, daily)

    def label_fall(i: int) -> float:
        """1 when the next `horizon` days contain a fall of `fall` from here."""
        ahead = close[i : i + 1 + horizon]
        return 1.0 if float(np.min(ahead) / close[i] - 1.0) <= -fall else 0.0

    cut = int(n * split)
    begin = INSIDER_MIN_HISTORY + 1
    train = np.array([i for i in range(begin, cut - horizon - 1) if np.isfinite(daily[i])])
    model = fit(
        _rows(signals, train),
        np.array([label_fall(i) for i in train]),
        FEATURES,
        priors={f: Prior(0.0, 1.0) for f in FEATURES},
        label_definition=f"fall of {fall:.0%} within {horizon} days",
    )

    # **The threshold has to be calibrated, and it has to keep recalibrating.**
    # A 3.4% base rate means the model almost never emits a probability above
    # 0.5, so an absolute cutoff never fires — that is not caution, it is a
    # broken switch. But a percentile of the *training* output is barely better:
    # training spans 2008, so its 95th percentile is a bar the calmer years that
    # follow almost never clear, and the switch stays off through a decade it
    # was supposed to be watching.
    #
    # So the trigger is a percentile of every probability the model has emitted
    # *before today* — expanding, never forward-looking. "Fire on the most
    # alarming 5% of days" then means the same thing in 2017 as in 2008, because
    # it is always measured against the regime actually in force.
    every = np.array([i for i in range(begin, n) if np.isfinite(daily[i])])
    p_all = np.full(n, np.nan)
    p_all[every] = [
        model.probability(
            {f: float(v) for f, v in zip(FEATURES, row, strict=True) if np.isfinite(v)}
        )
        for row in _rows(signals, every)
    ]

    def make_triggers(fraction: float) -> np.ndarray:
        out = np.full(n, np.inf)
        for i in range(begin + CALIBRATION_MIN, n):
            past = p_all[max(begin, i - CALIBRATION_WINDOW) : i]
            past = past[np.isfinite(past)]
            if past.size >= CALIBRATION_MIN:
                out[i] = float(np.quantile(past, 1.0 - fraction))
        return out

    held = np.array(range(cut, n - horizon - 1))
    y = np.array([label_fall(i) for i in held])
    p = p_all[held]

    print(f"Traded:      {index[cut].date()} to {index[-1].date()}")
    print(f"Warning of:  a {fall:.0%} fall within {horizon} trading day(s)")
    print(f"Detector:    out-of-sample AUC {auc(y, p):.4f}, base rate {y.mean():.1%}\n")

    curve, _, _ = _simulate(
        daily,
        close,
        signals,
        None,
        cut,
        capital=capital,
        mode="hold",
        triggers=np.full(n, np.inf),
        defensive=defensive,
        ladder=ladder,
        timeout=timeout,
        cost=cost,
    )
    hold_final, hold_dd = float(curve[-1]), _drawdown(curve)

    print(
        f"  {'rule':<26} {'fires':>7} {'precis':>7} {'recall':>7} {'final':>9} "
        f"{'return':>8} {'drawdn':>8} {'ret/dd':>7} {'alarms':>7} {'held':>6}"
    )
    print(
        f"  {'buy and hold':<26} {'-':>7} {'-':>7} {'-':>7} {hold_final:>9,.0f} "
        f"{hold_final / capital - 1:>7.1%} {hold_dd:>7.1%} "
        f"{(hold_final / capital - 1) / hold_dd:>7.2f} {'-':>7} {'100%':>6}"
    )

    for fraction in fractions:
        triggers = make_triggers(fraction)
        fired = p >= triggers[held]
        curve, alarms, average = _simulate(
            daily,
            close,
            signals,
            model,
            cut,
            capital=capital,
            mode="overlay",
            triggers=triggers,
            defensive=defensive,
            ladder=ladder,
            timeout=timeout,
            cost=cost,
        )
        final = float(curve[-1])
        drawdown = _drawdown(curve)
        precision = y[fired].mean() if fired.any() else float("nan")
        recall = fired[y == 1].mean() if fired.any() else 0.0
        print(
            f"  {f'overlay, top {fraction:.0%} alarming':<26} {fired.mean():>7.1%} "
            f"{precision:>7.1%} {recall:>7.1%} {final:>9,.0f} {final / capital - 1:>7.1%} "
            f"{drawdown:>7.1%} {(final / capital - 1) / drawdown if drawdown else 0:>7.2f} "
            f"{alarms:>7} {average:>6.0%}"
        )

    print(
        "\n  'precis' is how often a warning was actually followed by the fall;\n"
        "  'recall' is how much of the market's falling the warnings covered.\n"
        "  A rule is only worth reading if ret/dd beats buy and hold AND it fired\n"
        "  enough times to have been tested rather than merely sampled."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Asymmetric crash overlay.")
    parser.add_argument("--path", type=Path, default=Path("data/insider_index.csv"))
    parser.add_argument("--capital", type=float, default=5000.0)
    parser.add_argument("--split", type=float, default=0.5)
    parser.add_argument("--horizon", type=int, default=1, help="Days ahead the warning covers.")
    parser.add_argument("--fall", type=float, default=0.02, help="What counts as a sharp fall.")
    parser.add_argument(
        "--sell-fraction",
        type=float,
        nargs="+",
        default=[0.02, 0.05, 0.10, 0.15, 0.20, 0.30],
        help="Fire on this fraction of the most alarming recent days. Swept.",
    )
    parser.add_argument("--defensive", type=float, default=0.3, help="Exposure kept when out.")
    parser.add_argument("--ladder", type=float, default=0.10, help="Fall over which to buy back.")
    parser.add_argument("--timeout", type=int, default=20, help="Days before returning anyway.")
    parser.add_argument("--cost", type=float, default=0.0005)
    args = parser.parse_args()
    _run(
        args.path,
        args.capital,
        args.split,
        args.horizon,
        args.fall,
        args.sell_fraction,
        args.defensive,
        args.ladder,
        args.timeout,
        args.cost,
    )


if __name__ == "__main__":
    main()
