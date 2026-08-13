"""Walk-forward test of the aggregate insider signal (§9).

    python -m app.scripts.backtest_insider_timing --capital 5000

Aggregate insider buying is the only signal in this project to predict
*direction*: sorted into quarters, the year after the heaviest buying returned
+21.9% against -4.3% after the lightest, monotonically, with t = +2.54. This is
the test that killed every other candidate — walk it forward, charge costs, and
compare against doing nothing.

**Strictly expanding window, no fixed split.** At each decision the signal is
ranked against *only its own past*: the percentile of today's reading within
every reading before it. A fixed train/test split would waste the early history
and invite the question of where to cut; ranking against the past uses
everything and can never see forward. Early decisions are made on a thin history
and are correspondingly noisy, which is honest — it is what live trading from
2006 would have felt like.

**Rebalanced monthly**, because the signal is a 126-day average and re-reading
it daily would trade noise in a series that moves over months.

Three rules are compared against buy-and-hold and against the volatility rule,
which is the incumbent and the real bar:

  * **binary** — fully invested when insider buying is above its own median,
    out when below;
  * **scaled** — exposure set to the percentile itself, so heavy buying means
    fully invested and light buying means partly;
  * **combined** — the volatility rule's position, tilted by the percentile.

The number of position changes is reported, because a rule that traded four
times in twenty years has not really been tested however good its number looks.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

TRADING_DAYS = 252
SMOOTH = 126
REBALANCE = 21
VOL_WINDOW = 20
BORROW = 0.05

#: Readings required before the percentile means anything. Two years of history
#: to rank the third against.
MIN_HISTORY = 504


def _drawdown(curve: np.ndarray) -> float:
    peak = np.maximum.accumulate(curve)
    return float(np.max((peak - curve) / peak)) if curve.size else 0.0


def _percentile_of_past(values: np.ndarray, i: int) -> float:
    """Where today's reading sits within every reading before it, 0..1."""
    history = values[:i]
    history = history[np.isfinite(history)]
    if history.size < MIN_HISTORY or not np.isfinite(values[i]):
        return float("nan")
    return float(np.mean(history < values[i]))


def _simulate(
    daily: np.ndarray,
    signal: np.ndarray,
    vol: np.ndarray,
    *,
    capital: float,
    mode: str,
    vol_target: float,
    max_exposure: float,
    cost: float,
) -> tuple[np.ndarray, int, float]:
    equity = capital
    exposure = 0.0
    curve: list[float] = []
    changes = 0
    held: list[float] = []

    for i in range(MIN_HISTORY, daily.size):
        if (i - MIN_HISTORY) % REBALANCE == 0:
            # Everything read at i - 1: the position for today is chosen with
            # yesterday's information.
            prior = i - 1
            rank = _percentile_of_past(signal, prior)
            base = (
                min(vol_target / vol[prior], max_exposure)
                if np.isfinite(vol[prior]) and vol[prior] > 0
                else exposure
            )

            if mode == "hold":
                wanted = 1.0
            elif mode == "vol":
                wanted = base
            elif not np.isfinite(rank):
                wanted = 1.0  # signal not yet rankable: hold, do not guess
            elif mode == "binary":
                wanted = 1.0 if rank >= 0.5 else 0.0
            elif mode == "scaled":
                wanted = rank
            else:  # combined
                wanted = base * (0.5 + rank)

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


def _run(path: Path, capital: float, vol_target: float, max_exposure: float, cost: float) -> None:
    warnings.filterwarnings("ignore")
    import yfinance as yf

    insider = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    insider = insider[~insider.index.duplicated(keep="last")]

    frame = yf.Ticker("^GSPC").history(period="max", interval="1d")
    frame = frame[frame.index >= "2006-01-01"]
    close = frame["Close"].to_numpy(dtype=np.float64)
    index = pd.DatetimeIndex(frame.index.tz_localize(None)).normalize()
    n = close.size
    daily = np.concatenate([[np.nan], close[1:] / close[:-1] - 1.0])

    officer = insider.reindex(index, method="ffill")["officer_buy_share"].to_numpy(dtype=np.float64)
    smooth = np.full(n, np.nan)
    for i in range(SMOOTH, n):
        chunk = officer[i - SMOOTH : i]
        chunk = chunk[np.isfinite(chunk)]
        if chunk.size >= SMOOTH // 2:
            smooth[i] = float(np.mean(chunk))

    vol = np.full(n, np.nan)
    for i in range(VOL_WINDOW, n):
        vol[i] = float(np.std(daily[i - VOL_WINDOW : i], ddof=1)) * np.sqrt(TRADING_DAYS)

    years = (index[-1] - index[MIN_HISTORY]).days / 365.25
    print(f"Traded:      {index[MIN_HISTORY].date()} to {index[-1].date()}  ({years:.1f} years)")
    print(f"Signal:      officer/director buy share, {SMOOTH}-day average")
    print("Ranking:     expanding window — today against its own past only")
    print(f"Rebalanced:  every {REBALANCE} trading days\n")

    print(
        f"  {'rule':<34} {'final':>10} {'return':>9} {'drawdown':>10} "
        f"{'ret/dd':>8} {'changes':>8} {'invested':>9}"
    )
    for label, mode in (
        ("buy and hold", "hold"),
        ("volatility rule", "vol"),
        ("insider, in/out at median", "binary"),
        ("insider, exposure = percentile", "scaled"),
        ("volatility rule x insider tilt", "combined"),
    ):
        curve, changes, average = _simulate(
            daily,
            smooth,
            vol,
            capital=capital,
            mode=mode,
            vol_target=vol_target,
            max_exposure=max_exposure,
            cost=cost,
        )
        final = float(curve[-1])
        drawdown = _drawdown(curve)
        print(
            f"  {label:<34} {final:>10,.0f} {final / capital - 1:>8.1%} {drawdown:>9.1%} "
            f"{(final / capital - 1) / drawdown if drawdown else 0:>8.2f} {changes:>8} "
            f"{average:>8.0%}"
        )

    print(
        "\n  'changes' is how many times the position actually moved. A rule that\n"
        "  moved a handful of times over twenty years has not been tested by this\n"
        "  run so much as sampled by it, however good the final number looks."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward test of the insider signal.")
    parser.add_argument("--path", type=Path, default=Path("data/insider_index.csv"))
    parser.add_argument("--capital", type=float, default=5000.0)
    parser.add_argument("--vol-target", type=float, default=0.20)
    parser.add_argument("--max-exposure", type=float, default=1.5)
    parser.add_argument("--cost", type=float, default=0.0005)
    args = parser.parse_args()
    _run(args.path, args.capital, args.vol_target, args.max_exposure, args.cost)


if __name__ == "__main__":
    main()
