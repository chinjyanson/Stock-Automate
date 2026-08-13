"""Do the surviving macro signals make money, combined? (§9)

    python -m app.scripts.backtest_macro_model --capital 5000

Seven signals cleared their error bars against future drawdowns across three
horizons, with a stable ordering: the VIX and its term structure lead, credit
sits in the middle, rates and the curve are absent. This asks whether a model
built from them is worth trading.

**Two questions, because the signals are much better at one than the other.**

*Direction* — a logistic regression predicting whether the next five days close
up, traded in-or-out. This is the question asked here, and the prior is poor:
direction has already failed at one day (AUC 0.500) and twenty (55% against a
65% bar), and the same signals score +0.07 or less against five-day returns
while scoring +0.35 against five-day drawdowns. They know about turbulence, not
about direction.

*Risk* — the same regression retargeted at whether the next five days contain a
sharp fall, used to scale exposure rather than to switch it. This plays to what
the signals actually measure.

Both are compared against buy-and-hold **and** against plain volatility
targeting, because beating buy-and-hold is not the bar — the volatility rule
already does that sometimes, and a macro model has to beat *it* to have earned
the extra machinery.

Fitted on the early slice, traded on the later one. Signals are read at the
close of a Friday and the position held for the following week, so nothing is
known before it could have been.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import urllib.request
import warnings
from collections.abc import Callable

import numpy as np
import pandas as pd

from app.models_ml.logistic import FittedModel, Prior, auc, fit

TRADING_DAYS = 252
VOL_WINDOW = 20
WARMUP = 300
BORROW = 0.05

#: The signals that cleared two standard errors at more than one horizon, in the
#: order the ranking put them. Rates, the yield curve and breakevens are absent
#: because they were absent at every horizon tested.
FRED_SERIES = ("DBAA", "DAAA")
YAHOO_SERIES = ("^VIX", "^VIX3M", "^SKEW", "HYG", "LQD", "TLT", "IWM")

FEATURES = (
    "vix",
    "vix_term_structure",
    "credit_spread",
    "hyg_tlt",
    "hyg_lqd",
    "skew",
    "small_cap_rs",
    "realised_vol",
)


def _fred(series_id: str, index: pd.DatetimeIndex) -> np.ndarray:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd=1990-01-01"
    raw = urllib.request.urlopen(url, timeout=30).read().decode()
    frame = pd.read_csv(io.StringIO(raw))
    frame.columns = ["date", "value"]
    frame = frame[frame["value"] != "."]
    frame["date"] = pd.to_datetime(frame["date"])
    values = pd.Series(frame["value"].astype(float).to_numpy(), index=frame["date"])
    out: np.ndarray = values.reindex(index, method="ffill").to_numpy(dtype=np.float64)
    return out


def _yahoo(symbol: str, index: pd.DatetimeIndex) -> np.ndarray:
    import yfinance as yf

    frame = yf.Ticker(symbol).history(period="max", interval="1d")
    closes = frame["Close"]
    closes.index = closes.index.tz_localize(None).normalize()
    out: np.ndarray = closes.reindex(index, method="ffill").to_numpy(dtype=np.float64)
    return out


def _momentum(ratio: np.ndarray, window: int = 60) -> np.ndarray:
    out = np.full(ratio.size, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        out[window:] = ratio[window:] / ratio[:-window] - 1.0
    return out


def _drawdown(curve: np.ndarray) -> float:
    peak = np.maximum.accumulate(curve)
    return float(np.max((peak - curve) / peak)) if curve.size else 0.0


def _build(index: pd.DatetimeIndex, close: np.ndarray, daily: np.ndarray) -> dict[str, np.ndarray]:
    raw = {s: _fred(s, index) for s in FRED_SERIES}
    raw.update({s: _yahoo(s, index) for s in YAHOO_SERIES})

    realised = np.full(close.size, np.nan)
    for i in range(VOL_WINDOW, close.size):
        realised[i] = float(np.std(daily[i - VOL_WINDOW : i], ddof=1)) * np.sqrt(TRADING_DAYS)

    with np.errstate(divide="ignore", invalid="ignore"):
        return {
            "vix": raw["^VIX"],
            "vix_term_structure": raw["^VIX3M"] / raw["^VIX"],
            "credit_spread": raw["DBAA"] - raw["DAAA"],
            "hyg_tlt": _momentum(raw["HYG"] / raw["TLT"]),
            "hyg_lqd": _momentum(raw["HYG"] / raw["LQD"]),
            "skew": raw["^SKEW"],
            "small_cap_rs": _momentum(raw["IWM"] / close),
            "realised_vol": realised,
        }


def _rows(signals: dict[str, np.ndarray], at: np.ndarray) -> np.ndarray:
    return np.column_stack([[float(signals[f][i]) for f in FEATURES] for i in at]).T


def _label_direction(close: np.ndarray, i: int, window: int) -> float:
    return 1.0 if close[i + window] > close[i] else 0.0


def _label_calm(close: np.ndarray, i: int, threshold: float, window: int) -> float:
    """1 when the coming window contains NO sharp fall — the risk question.

    **The fall is measured from today's close**, not from the highest point
    inside the future window. Measuring only within the window degenerates at
    `window = 1`: one price has no peak-to-trough, so every single day scored as
    calm, the base rate came out at 100% and the AUC was undefined. It is also
    the wrong question even where it does not degenerate — a holder cares how
    far the price falls below *where they are now*, not how far it falls from a
    peak it may reach next Tuesday.
    """
    ahead = close[i : i + 1 + window]
    peak = np.maximum.accumulate(ahead)
    return 1.0 if float(np.max((peak - ahead) / peak)) < threshold else 0.0


def _simulate(
    daily: np.ndarray,
    close: np.ndarray,
    signals: dict[str, np.ndarray],
    model: FittedModel | None,
    start: int,
    *,
    capital: float,
    mode: str,
    threshold: float,
    vol_target: float,
    max_exposure: float,
    cost: float,
    window: int,
) -> tuple[np.ndarray, int]:
    equity = capital
    exposure = 0.0
    curve: list[float] = []
    trades = 0

    for i in range(start, daily.size):
        # Decide once per window, on the previous close.
        if (i - start) % window == 0:
            prior = i - 1
            reading = {
                f: float(signals[f][prior]) for f in FEATURES if np.isfinite(signals[f][prior])
            }
            volatility = signals["realised_vol"][prior]
            base = (
                min(vol_target / volatility, max_exposure)
                if np.isfinite(volatility) and volatility > 0
                else exposure
            )

            if mode == "hold":
                wanted = 1.0
            elif mode == "vol":
                wanted = base
            else:
                probability = model.probability(reading) if (model is not None and reading) else 0.5
                if mode == "direction":
                    wanted = base if probability >= threshold else 0.0
                else:  # scale exposure by the model's confidence
                    wanted = base * float(np.clip(probability / threshold, 0.0, 1.5))
            wanted = float(np.clip(wanted, 0.0, max_exposure))

            if abs(wanted - exposure) > 1e-9:
                equity -= equity * abs(wanted - exposure) * cost / 2.0
                exposure = wanted
                trades += 1

        equity *= 1.0 + exposure * daily[i]
        if exposure > 1.0:
            equity -= equity * (exposure - 1.0) * BORROW / TRADING_DAYS
        curve.append(equity)

    return np.asarray(curve), trades


async def _run(
    capital: float,
    split: float,
    vol_target: float,
    max_exposure: float,
    cost: float,
    fall: float,
    window: int,
) -> None:
    warnings.filterwarnings("ignore")
    import yfinance as yf

    frame = yf.Ticker("^GSPC").history(period="max", interval="1d")
    frame = frame[frame.index >= "1990-01-01"]
    close = frame["Close"].to_numpy(dtype=np.float64)
    index = pd.DatetimeIndex(frame.index.tz_localize(None)).normalize()
    daily = np.concatenate([[np.nan], close[1:] / close[:-1] - 1.0])
    n = close.size

    signals = _build(index, close, daily)
    cut = int(n * split)

    fitted: dict[str, FittedModel] = {}
    labellers: tuple[tuple[str, Callable[[int], float]], ...] = (
        ("direction", lambda i: _label_direction(close, i, window)),
        ("calm", lambda i: _label_calm(close, i, fall, window)),
    )
    for name, labeller in labellers:
        usable = [i for i in range(WARMUP, cut - window - 1) if np.isfinite(daily[i])]
        x = _rows(signals, np.array(usable))
        y = np.array([labeller(i) for i in usable])
        fitted[name] = fit(
            x,
            y,
            FEATURES,
            priors={f: Prior(0.0, 1.0) for f in FEATURES},
            label_definition=name,
        )

    held = list(range(cut, n - window - 1))
    x_held = _rows(signals, np.array(held))

    print(f"Instrument:  ^GSPC, {n:,} bars from {index[0].date()}")
    print(f"Fitted on:   bars {WARMUP}-{cut}   Traded on: {cut}-{n} (never seen)")
    print(f"Signals:     {', '.join(FEATURES)}\n")

    print("1. WHAT CAN THE COMBINED MODEL PREDICT, OUT OF SAMPLE?")
    reports: tuple[tuple[str, Callable[[int], float]], ...] = (
        (f"next {window}d UP or down", lambda i: _label_direction(close, i, window)),
        (
            f"next {window}d calm (no {fall:.0%} fall)",
            lambda i: _label_calm(close, i, fall, window),
        ),
    )
    for name, labeller in reports:
        y = np.array([labeller(i) for i in held])
        key = "direction" if "UP" in name else "calm"
        scored = np.array(
            [
                fitted[key].probability(
                    {f: float(v) for f, v in zip(FEATURES, row, strict=True) if np.isfinite(v)}
                )
                for row in x_held
            ]
        )
        print(f"     {name:<34} AUC {auc(y, scored):.4f}   base rate {y.mean():.1%}")

    print("\n2. DOES IT MAKE MONEY?")
    head = f"     {'rule':<34} {'final':>10} {'return':>9}"
    print(f"{head} {'drawdown':>10} {'ret/dd':>8} {'trades':>7}")
    for label, mode, model in (
        ("buy and hold", "hold", None),
        ("volatility targeting only", "vol", None),
        ("+ direction model (in/out)", "direction", fitted["direction"]),
        ("+ risk model (scaled)", "scaled", fitted["calm"]),
    ):
        curve, trades = _simulate(
            daily,
            close,
            signals,
            model,
            cut,
            capital=capital,
            mode=mode,
            threshold=0.5 if mode == "direction" else 0.7,
            vol_target=vol_target,
            max_exposure=max_exposure,
            cost=cost,
            window=window,
        )
        final = float(curve[-1])
        drawdown = _drawdown(curve)
        print(
            f"     {label:<34} {final:>10,.0f} {final / capital - 1:>8.1%} "
            f"{drawdown:>9.1%} {(final / capital - 1) / drawdown if drawdown else 0:>8.2f} "
            f"{trades:>7}"
        )
    print(
        "\n     The bar is the volatility row, not buy-and-hold. A macro model has\n"
        "     to beat the rule it is bolted onto, or it is machinery that costs\n"
        "     turnover and returns nothing."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Trade the combined macro signals.")
    parser.add_argument("--capital", type=float, default=5000.0)
    parser.add_argument("--split", type=float, default=0.6)
    parser.add_argument("--vol-target", type=float, default=0.20)
    parser.add_argument("--max-exposure", type=float, default=1.5)
    parser.add_argument("--cost", type=float, default=0.0005)
    parser.add_argument("--fall", type=float, default=0.02, help="What counts as a sharp fall.")
    parser.add_argument(
        "--window",
        type=int,
        default=5,
        help="Trading days between decisions, and the horizon predicted.",
    )
    args = parser.parse_args()
    asyncio.run(
        _run(
            capital=args.capital,
            split=args.split,
            vol_target=args.vol_target,
            max_exposure=args.max_exposure,
            cost=args.cost,
            fall=args.fall,
            window=args.window,
        )
    )


if __name__ == "__main__":
    main()
