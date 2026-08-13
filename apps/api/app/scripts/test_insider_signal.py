"""Does aggregate insider buying predict the market? (§9)

    python -m app.scripts.test_insider_signal

Every signal tested so far is ultimately a market price — the VIX, credit
spreads, skew, realised volatility — so they all encode the same fear from the
same participants, which is why they kept turning out to be redundant with each
other. Insider filings are different in kind: they record what company officers
and directors actually *did* with their own money, disclosed because the law
requires it rather than because a market cleared.

They are also the only signal here with a real claim to predicting **direction**
rather than turbulence. The academic work (Lakonishok and Lee 2001; Jeng,
Metrick and Zeckhauser 2003) finds aggregate insider buying predicts market
returns at one to six months — which is both a different question and a slower
horizon than anything that has failed here so far.

**Smoothing is not optional.** A single day's buy share is dominated by whichever
handful of companies happened to file, and filings cluster after earnings, so the
raw series is mostly calendar. What is tested is a rolling average over one to
six months, which is also the horizon the underlying research uses.

Scored against forward returns at several horizons, on non-overlapping windows,
with the S&P's own trailing return alongside as a control — if insider buying
merely tracks what the market just did, that shows up as the two scoring alike.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    ok = np.isfinite(x) & np.isfinite(y)
    n = int(ok.sum())
    if n < 25:
        return float("nan"), n
    rx = np.argsort(np.argsort(x[ok])).astype(float)
    ry = np.argsort(np.argsort(y[ok])).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    d = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return (float((rx * ry).sum() / d) if d > 0 else float("nan")), n


def _rolling(values: np.ndarray, window: int) -> np.ndarray:
    out = np.full(values.size, np.nan)
    for i in range(window, values.size):
        chunk = values[i - window : i]
        chunk = chunk[np.isfinite(chunk)]
        if chunk.size >= window // 2:
            out[i] = float(np.mean(chunk))
    return out


def _run(path: Path, horizons: list[int]) -> None:
    warnings.filterwarnings("ignore")
    import yfinance as yf

    insider = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()

    frame = yf.Ticker("^GSPC").history(period="max", interval="1d")
    frame = frame[frame.index >= "2006-01-01"]
    close = frame["Close"].to_numpy(dtype=np.float64)
    index = pd.DatetimeIndex(frame.index.tz_localize(None)).normalize()
    n = close.size

    aligned = insider.reindex(index, method="ffill")
    buy_share = aligned["buy_share"].to_numpy(dtype=np.float64)
    officer_share = aligned["officer_buy_share"].to_numpy(dtype=np.float64)

    # The market's own trailing return, as a control: if insider buying only
    # tracks what just happened, it will score like this does.
    trailing = np.full(n, np.nan)
    trailing[60:] = close[60:] / close[:-60] - 1.0

    candidates = {
        "insider buy share, 21d avg": _rolling(buy_share, 21),
        "insider buy share, 63d avg": _rolling(buy_share, 63),
        "insider buy share, 126d avg": _rolling(buy_share, 126),
        "officer buy share, 21d avg": _rolling(officer_share, 21),
        "officer buy share, 63d avg": _rolling(officer_share, 63),
        "officer buy share, 126d avg": _rolling(officer_share, 126),
        "(control) S&P trailing 60d": trailing,
    }

    print(
        f"Insider index:  {len(insider):,} days, {insider.index[0].date()} to "
        f"{insider.index[-1].date()}"
    )
    print(f"S&P:            {n:,} bars from {index[0].date()}")
    print("Target:         forward RETURN — the direction question\n")

    header = f"  {'signal':<30}"
    for horizon in horizons:
        header += f" {f'{horizon}d':>9}"
    print(header + f" {'n@' + str(horizons[-1]):>7}")

    rows = []
    for name, values in candidates.items():
        scores = []
        count = 0
        for horizon in horizons:
            forward = np.full(n, np.nan)
            forward[: n - horizon] = close[horizon:] / close[: n - horizon] - 1.0
            step = np.arange(130, n - horizon - 1, horizon)
            ic, count = _spearman(values[step], forward[step])
            scores.append(ic)
        rows.append((name, scores, count))

    for name, scores, count in rows:
        line = f"  {name:<30}"
        for horizon, ic in zip(horizons, scores, strict=True):
            error = 2.0 / np.sqrt(max(n // horizon - 3, 1))
            mark = "*" if np.isfinite(ic) and abs(ic) > error else " "
            line += f" {ic:>+8.3f}{mark}"
        print(line + f" {count:>7}")

    print(
        "\n  * is more than two standard errors from zero at that horizon.\n"
        "  A POSITIVE number means heavy insider buying precedes a rising market.\n"
        "\n  The control row is the S&P's own trailing return. A signal that scores\n"
        "  like the control is telling you what already happened."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Test aggregate insider buying as a signal.")
    parser.add_argument("--path", type=Path, default=Path("data/insider_index.csv"))
    parser.add_argument("--horizons", type=int, nargs="+", default=[21, 63, 126, 252])
    args = parser.parse_args()
    _run(args.path, args.horizons)


if __name__ == "__main__":
    main()
