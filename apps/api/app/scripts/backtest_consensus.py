"""All four models together: agree to trade, size by volatility (§9).

    python -m app.scripts.backtest_consensus --capital 5000

Four things have been built and measured separately:

  * the **volatility rule**, which sizes but never has a view;
  * the **direction model**, which has a view and is wrong (AUC ~0.51);
  * the **risk model**, which predicts turbulence well (AUC ~0.66) but adds
    nothing over the volatility rule it sits on;
  * the **insider signal**, which sorts history beautifully and failed its
    walk-forward.

This asks whether they are better together. Two ideas are being tested at once
and they are worth separating, because either could carry the result:

**1. Insider buying as a feature.** It is the only input here not derived from a
market price, so if the others are redundant with each other — and they have
been, repeatedly — this is the one with a chance of adding something. It enters
both models as an expanding-window percentile, never a raw level, so the model
sees "unusually heavy buying by this history's standards" rather than a number
whose meaning drifts.

**2. Consensus.** Trade only when the direction model expects a rise *and* the
risk model expects calm; size the resulting position with the volatility rule.
The hope is that requiring agreement filters out the occasions when either is
guessing. The concern is that two models fed the same features are not
independent witnesses, and agreement between them may only mean the features
were emphatic rather than right.

Everything is walk-forward: fitted on the early slice, traded on the later one,
across several splits, with costs and financing charged.
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
WARMUP = 300
INSIDER_SMOOTH = 126
INSIDER_MIN_HISTORY = 504
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
    """Officer buy share, smoothed, then ranked against its own past.

    A percentile rather than a level: the raw share drifts with how many
    companies file and how the SEC's coverage has changed, so a reading of 25%
    means something different in 2008 and 2024. The rank is computed only
    against earlier readings, so it can never see forward.
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
        history = smooth[:i]
        history = history[np.isfinite(history)]
        if history.size >= INSIDER_MIN_HISTORY and np.isfinite(smooth[i]):
            rank[i] = float(np.mean(history < smooth[i]))
    return rank


def _drawdown(curve: np.ndarray) -> float:
    peak = np.maximum.accumulate(curve)
    return float(np.max((peak - curve) / peak)) if curve.size else 0.0


def _rows(signals: dict[str, np.ndarray], at: np.ndarray) -> np.ndarray:
    return np.column_stack([[float(signals[f][i]) for f in FEATURES] for i in at]).T


def _simulate(
    daily: np.ndarray,
    signals: dict[str, np.ndarray],
    direction: FittedModel,
    risk: FittedModel,
    start: int,
    *,
    capital: float,
    mode: str,
    window: int,
    vol_target: float,
    max_exposure: float,
    calm_threshold: float,
    cost: float,
) -> tuple[np.ndarray, int, float]:
    equity = capital
    exposure = 0.0
    curve: list[float] = []
    changes = 0
    held: list[float] = []

    for i in range(start, daily.size):
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
            p_up = direction.probability(reading) if reading else 0.5
            p_calm = risk.probability(reading) if reading else 0.5

            if mode == "hold":
                wanted = 1.0
            elif mode == "vol":
                wanted = base
            elif mode == "consensus":
                # Both must agree before any position is taken.
                wanted = base if (p_up >= 0.5 and p_calm >= calm_threshold) else 0.0
            elif mode == "either":
                wanted = base if (p_up >= 0.5 or p_calm >= calm_threshold) else 0.0
            else:  # blended — scale rather than switch
                wanted = base * float(np.clip(2.0 * p_up * (p_calm / calm_threshold), 0.0, 1.5))

            wanted = float(np.clip(wanted, 0.0, max_exposure))
            if abs(wanted - exposure) > 0.02:
                equity -= equity * abs(wanted - exposure) * cost / 2.0
                exposure = wanted
                changes += 1

        equity *= 1.0 + exposure * daily[i]
        if exposure > 1.0:
            equity -= equity * (exposure - 1.0) * BORROW / TRADING_DAYS
        curve.append(equity)
        held.append(exposure)

    return np.asarray(curve), changes, float(np.mean(held)) if held else 0.0


def _run(
    path: Path,
    capital: float,
    split: float,
    window: int,
    vol_target: float,
    max_exposure: float,
    fall: float,
    calm_threshold: float,
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

    realised = np.full(n, np.nan)
    for i in range(VOL_WINDOW, n):
        realised[i] = float(np.std(daily[i - VOL_WINDOW : i], ddof=1)) * np.sqrt(TRADING_DAYS)

    vix, vix3m, skew = (_yahoo(s, index) for s in ("^VIX", "^VIX3M", "^SKEW"))
    hyg, lqd, tlt, iwm = (_yahoo(s, index) for s in ("HYG", "LQD", "TLT", "IWM"))
    with np.errstate(divide="ignore", invalid="ignore"):
        signals = {
            "vix": vix,
            "vix_term_structure": vix3m / vix,
            "credit_spread": _fred("DBAA", index) - _fred("DAAA", index),
            "hyg_tlt": _momentum(hyg / tlt),
            "hyg_lqd": _momentum(hyg / lqd),
            "skew": skew,
            "small_cap_rs": _momentum(iwm / close),
            "realised_vol": realised,
            "insider_rank": _insider_rank(path, index),
        }

    def label_up(i: int) -> float:
        return 1.0 if close[i + window] > close[i] else 0.0

    def label_calm(i: int) -> float:
        ahead = close[i : i + 1 + window]
        peak = np.maximum.accumulate(ahead)
        return 1.0 if float(np.max((peak - ahead) / peak)) < fall else 0.0

    cut = int(n * split)
    start = max(WARMUP, INSIDER_MIN_HISTORY + 1)
    train = np.array([i for i in range(start, cut - window - 1) if np.isfinite(daily[i])])
    x_train = _rows(signals, train)

    direction = fit(
        x_train,
        np.array([label_up(i) for i in train]),
        FEATURES,
        priors={f: Prior(0.0, 1.0) for f in FEATURES},
        label_definition="up",
    )
    risk = fit(
        x_train,
        np.array([label_calm(i) for i in train]),
        FEATURES,
        priors={f: Prior(0.0, 1.0) for f in FEATURES},
        label_definition="calm",
    )

    held = np.array(range(cut, n - window - 1))
    x_held = _rows(signals, held)

    def score(model: FittedModel) -> np.ndarray:
        return np.array(
            [
                model.probability(
                    {f: float(v) for f, v in zip(FEATURES, row, strict=True) if np.isfinite(v)}
                )
                for row in x_held
            ]
        )

    p_up, p_calm = score(direction), score(risk)
    y_up = np.array([label_up(i) for i in held])
    y_calm = np.array([label_calm(i) for i in held])

    print(f"Split:       {split:.0%}   traded {index[cut].date()} to {index[-1].date()}")
    print(f"Decisions:   every {window} trading days\n")
    print(f"  direction model, out of sample   AUC {auc(y_up, p_up):.4f}")
    print(f"  risk model, out of sample        AUC {auc(y_calm, p_calm):.4f}")
    agree = float(np.mean((p_up >= 0.5) & (p_calm >= calm_threshold)))
    print(f"  both agree to be invested        {agree:.0%} of decisions")
    print(f"  insider coefficient, direction   {direction.coefficients[-1]:+.4f}")
    print(f"  insider coefficient, risk        {risk.coefficients[-1]:+.4f}\n")

    print(
        f"  {'rule':<32} {'final':>10} {'return':>9} {'drawdown':>10} {'ret/dd':>8} {'invested':>9}"
    )
    for label, mode in (
        ("buy and hold", "hold"),
        ("volatility rule", "vol"),
        ("consensus (both agree)", "consensus"),
        ("either model agrees", "either"),
        ("blended probabilities", "blended"),
    ):
        curve, _changes, average = _simulate(
            daily,
            signals,
            direction,
            risk,
            cut,
            capital=capital,
            mode=mode,
            window=window,
            vol_target=vol_target,
            max_exposure=max_exposure,
            calm_threshold=calm_threshold,
            cost=cost,
        )
        final = float(curve[-1])
        drawdown = _drawdown(curve)
        print(
            f"  {label:<32} {final:>10,.0f} {final / capital - 1:>8.1%} {drawdown:>9.1%} "
            f"{(final / capital - 1) / drawdown if drawdown else 0:>8.2f} {average:>8.0%}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine all four models.")
    parser.add_argument("--path", type=Path, default=Path("data/insider_index.csv"))
    parser.add_argument("--capital", type=float, default=5000.0)
    parser.add_argument("--split", type=float, default=0.6)
    parser.add_argument("--window", type=int, default=21)
    parser.add_argument("--vol-target", type=float, default=0.20)
    parser.add_argument("--max-exposure", type=float, default=1.5)
    parser.add_argument("--fall", type=float, default=0.03)
    parser.add_argument("--calm-threshold", type=float, default=0.6)
    parser.add_argument("--cost", type=float, default=0.0005)
    args = parser.parse_args()
    _run(
        args.path,
        args.capital,
        args.split,
        args.window,
        args.vol_target,
        args.max_exposure,
        args.fall,
        args.calm_threshold,
        args.cost,
    )


if __name__ == "__main__":
    main()
