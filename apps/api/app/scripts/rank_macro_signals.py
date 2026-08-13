"""Do macro, credit, valuation or positioning signals warn of S&P drawdowns? (§9)

    python -m app.scripts.rank_macro_signals

Everything tested so far has been derived from the S&P's own price. This tests
signals from *outside* it — credit spreads, real rates, valuation, breadth,
cross-asset relative strength — against the question a risk overlay actually
needs answered.

**The target is drawdown, not return.** A risk overlay does not need to know
where the market is going; it needs to know when the ground is about to become
unstable. So each signal is scored against three futures, and the third is the
one that matters:

  * the next 20 days' return,
  * the next 60 days' return,
  * **the worst peak-to-trough fall over the next 60 days.**

A signal can be useless for the first two and valuable for the third, which is
precisely the case for most of these.

**Sampling is non-overlapping**, so neighbouring observations do not share the
future they are judged on — 60-day windows read daily would overlap by 98% and
make everything look significant.

**What is here and what is not.** Free daily series only: FRED for rates and
credit, yfinance for the cross-asset ratios and the CBOE indices. Deliberately
absent, with reasons:

  * **GEX and charm** need per-strike open interest. The chain is published for
    today only and is gone when the session ends; `index_options_snapshots`
    holds zero rows and can be accumulated forward but never backfilled.
  * **AAII and NAAIM surveys** publish weekly to their own websites with no
    API and no history endpoint. Scraping them is possible; it is a data
    engineering job rather than a measurement, and weekly readings give ~50
    independent observations a decade.
  * **Fed funds futures** are CME-licensed. The effective rate and the 10y-2y
    curve stand in for the same information here.
  * **CAPE** publishes monthly, giving ~12 independent points a year; the
    earnings yield against the real rate is included through the real-rate
    series instead.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import urllib.request
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sqlalchemy import select

from app.data.store import CandleStore
from app.db import session_scope
from app.indicators.series import candles_to_series
from app.models.enums import Interval
from app.models.instrument import MarketDataMapping

TRADING_DAYS = 252

#: FRED series, fetched keyless as CSV. The ICE OAS series are licence-capped at
#: roughly three years through this endpoint, which is noted where they appear
#: rather than hidden — 3 years of daily data is ~40 independent 60-day windows.
FRED = {
    "BAMLH0A0HYM2": "high-yield OAS (3y only)",
    "DBAA": "Moody Baa yield",
    "DAAA": "Moody Aaa yield",
    "DFII10": "10y real rate (TIPS)",
    "T10Y2Y": "10y-2y curve",
    "DGS10": "10y nominal",
    "T10YIE": "10y breakeven inflation",
}

#: Cross-asset and CBOE series from yfinance.
YAHOO = {
    "^VIX": "implied volatility",
    "^VIX3M": "3-month implied volatility",
    "^SKEW": "tail-risk skew",
    "HYG": "high-yield credit ETF",
    "LQD": "investment-grade credit ETF",
    "TLT": "long treasuries",
    "IWM": "small caps",
    "RSP": "equal-weight S&P",
}


def _spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    ok = np.isfinite(x) & np.isfinite(y)
    n = int(ok.sum())
    if n < 30:
        return float("nan"), n
    rx = np.argsort(np.argsort(x[ok])).astype(float)
    ry = np.argsort(np.argsort(y[ok])).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    denominator = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return (float((rx * ry).sum() / denominator) if denominator > 0 else float("nan")), n


def _fred(series_id: str, index: pd.DatetimeIndex) -> np.ndarray | None:
    try:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd=1990-01-01"
        raw = urllib.request.urlopen(url, timeout=30).read().decode()
        frame = pd.read_csv(io.StringIO(raw))
        frame.columns = ["date", "value"]
        frame = frame[frame["value"] != "."]
        frame["date"] = pd.to_datetime(frame["date"])
        values = pd.Series(frame["value"].astype(float).to_numpy(), index=frame["date"])
        # Forward-filled onto the trading calendar: a series that did not print
        # on a day carries its last known value, never a later one.
        return values.reindex(index, method="ffill").to_numpy(dtype=np.float64)
    except Exception:
        return None


def _yahoo(symbol: str, index: pd.DatetimeIndex) -> np.ndarray | None:
    import yfinance as yf

    try:
        frame = yf.Ticker(symbol).history(period="max", interval="1d")
        if frame.empty:
            return None
        closes = frame["Close"]
        closes.index = closes.index.tz_localize(None).normalize()
        aligned: np.ndarray = closes.reindex(index, method="ffill").to_numpy(dtype=np.float64)
        return aligned
    except Exception:
        return None


def _build(index: pd.DatetimeIndex, spy: np.ndarray) -> dict[str, np.ndarray]:
    """Every candidate signal, aligned to the S&P calendar."""
    raw: dict[str, np.ndarray] = {}
    for series_id in FRED:
        values = _fred(series_id, index)
        if values is not None:
            raw[series_id] = values
    for symbol in YAHOO:
        values = _yahoo(symbol, index)
        if values is not None:
            raw[symbol] = values

    out: dict[str, np.ndarray] = {}

    def add(name: str, values: np.ndarray | None) -> None:
        if values is not None and np.isfinite(values).sum() > 200:
            out[name] = values

    # --- credit -----------------------------------------------------------
    if "BAMLH0A0HYM2" in raw:
        add("HY OAS level", raw["BAMLH0A0HYM2"])
        add("HY OAS, 60d change", _change(raw["BAMLH0A0HYM2"], 60))
    if "DBAA" in raw and "DAAA" in raw:
        spread = raw["DBAA"] - raw["DAAA"]
        add("Baa-Aaa spread", spread)
        add("Baa-Aaa, 60d change", _change(spread, 60))
    if "HYG" in raw and "TLT" in raw:
        add("HYG/TLT credit appetite", _ratio_momentum(raw["HYG"] / raw["TLT"], 60))
    if "HYG" in raw and "LQD" in raw:
        add("HYG/LQD credit quality", _ratio_momentum(raw["HYG"] / raw["LQD"], 60))

    # --- rates and valuation ----------------------------------------------
    if "DFII10" in raw:
        add("10y real rate", raw["DFII10"])
        add("10y real rate, 60d change", _change(raw["DFII10"], 60))
    if "T10Y2Y" in raw:
        add("10y-2y curve", raw["T10Y2Y"])
    if "T10YIE" in raw:
        add("10y breakeven inflation", raw["T10YIE"])
    if "DGS10" in raw:
        add("10y nominal, 60d change", _change(raw["DGS10"], 60))

    # --- options ----------------------------------------------------------
    if "^SKEW" in raw:
        add("CBOE SKEW", raw["^SKEW"])
    if "^VIX" in raw:
        add("VIX level", raw["^VIX"])
    if "^VIX" in raw and "^VIX3M" in raw:
        add("VIX term structure (3M/1M)", raw["^VIX3M"] / raw["^VIX"])

    # --- breadth and cross-asset -----------------------------------------
    if "RSP" in raw:
        # Equal-weight against cap-weight: when the average share lags the
        # index, the rise is carried by a few names. The closest free stand-in
        # for "% of members above their 200-day".
        add("breadth proxy (RSP/SPY, 60d)", _ratio_momentum(raw["RSP"] / spy, 60))
    if "IWM" in raw:
        add("small-cap RS (IWM/SPY, 60d)", _ratio_momentum(raw["IWM"] / spy, 60))

    return out


def _change(values: np.ndarray, window: int) -> np.ndarray:
    out = np.full(values.size, np.nan)
    out[window:] = values[window:] - values[:-window]
    return out


def _ratio_momentum(ratio: np.ndarray, window: int) -> np.ndarray:
    out = np.full(ratio.size, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        out[window:] = ratio[window:] / ratio[:-window] - 1.0
    return out


async def _load_spy(symbol: str) -> tuple[np.ndarray | None, list[datetime] | None]:
    async with session_scope() as session:
        mapping = (
            (
                await session.execute(
                    select(MarketDataMapping).where(MarketDataMapping.provider_symbol == symbol)
                )
            )
            .scalars()
            .first()
        )
        if mapping is None:
            return None, None
        candles = await CandleStore(session).get_candles(
            mapping.instrument_id, Interval.D1, limit=10_000, closed_only=True
        )
    if len(candles) < 800:
        return None, None
    return candles_to_series(candles).close, [c.timestamp for c in candles]


async def _run(symbol: str, horizon: int) -> None:
    warnings.filterwarnings("ignore")
    if symbol.startswith("^"):
        # The index itself rather than the tracker: SPY starts in 1993 and this
        # store holds it from 2011, which leaves 58 non-overlapping 60-day
        # windows — too few for any of these correlations to clear their own
        # error bars. ^GSPC reaches back to 1927, and the binding constraint
        # becomes each macro series' own start date instead of the price data's.
        import yfinance as yf

        frame = yf.Ticker(symbol).history(period="max", interval="1d")
        frame = frame[frame.index >= "1990-01-01"]
        close = frame["Close"].to_numpy(dtype=np.float64)
        index = pd.DatetimeIndex(frame.index.tz_localize(None)).normalize()
    else:
        close, timestamps = await _load_spy(symbol)
        if close is None or timestamps is None:
            print(f"{symbol}: no usable history.")
            return
        index = pd.DatetimeIndex([t.replace(tzinfo=None) for t in timestamps]).normalize()
    n = close.size

    # --- the three futures -------------------------------------------------
    forward_20 = np.full(n, np.nan)
    forward_60 = np.full(n, np.nan)
    forward_dd = np.full(n, np.nan)
    for i in range(n - horizon - 1):
        forward_20[i] = close[min(i + 20, n - 1)] / close[i] - 1.0
        # Named for the default horizon; it follows --horizon, and the column
        # header is generated from it so the two cannot disagree.
        forward_60[i] = close[i + horizon] / close[i] - 1.0
        window = close[i + 1 : i + 1 + horizon]
        peak = np.maximum.accumulate(window)
        forward_dd[i] = float(np.max((peak - window) / peak))

    signals = _build(index, close)
    step = np.arange(260, n - horizon - 1, horizon)

    print(f"Instrument:  {symbol}   {n:,} bars")
    print(f"Sampling:    {len(step)} non-overlapping {horizon}-day windows")
    print(f"Target:      the WORST peak-to-trough fall over the next {horizon} days\n")
    horizon_label = f"-> {horizon}d ret"
    print(f"  {'signal':<32} {'-> drawdown':>12} {'-> 20d ret':>11} {horizon_label:>11} {'n':>6}")

    rows = []
    for name, values in signals.items():
        dd_ic, count = _spearman(values[step], forward_dd[step])
        r20, _ = _spearman(values[step], forward_20[step])
        r60, _ = _spearman(values[step], forward_60[step])
        if np.isfinite(dd_ic):
            rows.append((name, dd_ic, r20, r60, count))

    for name, dd_ic, r20, r60, count in sorted(rows, key=lambda r: -abs(r[1])):
        # Two standard errors, roughly, for a rank correlation on `count` points.
        error = 2.0 / np.sqrt(max(count - 3, 1))
        mark = " *" if abs(dd_ic) > error else "  "
        print(f"  {name:<32} {dd_ic:>+11.3f}{mark} {r20:>+11.3f} {r60:>+11.3f} {count:>6}")

    print(
        "\n  * marks a drawdown correlation more than two standard errors from\n"
        "  zero. A POSITIVE number means a high reading precedes a deeper fall,\n"
        "  so it would be used as a warning; a negative one means the opposite.\n"
        "\n  GEX and charm are absent because they cannot be backfilled: the option\n"
        "  chain exists for today only. They can be accumulated forward at one row\n"
        "  a night, which is the only route to ever testing them."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank macro signals against S&P drawdowns.")
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--horizon", type=int, default=60, help="Forward window, trading days.")
    args = parser.parse_args()
    asyncio.run(_run(symbol=args.symbol, horizon=args.horizon))


if __name__ == "__main__":
    main()
