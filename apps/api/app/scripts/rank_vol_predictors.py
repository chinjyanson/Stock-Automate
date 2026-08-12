"""What best predicts the next 20 days of volatility? (§9)

    python -m app.scripts.rank_vol_predictors

Volatility targeting works because volatility is forecastable. The obvious next
question is whether something forecasts it *better* than the trailing realised
estimate the strategy currently uses.

**On the three candidates that get asked about.**

*Black-Scholes* is a pricing formula, not a forecast. Run backwards on a quoted
option it yields **implied volatility** — the market's own estimate of what is
coming — and for the S&P that quantity is published continuously as the VIX. So
"test Black-Scholes" and "test the VIX" are the same experiment, and this is it.
The VIX has daily history to 1990, which makes it the one option-derived measure
here that can actually be evaluated.

*GEX and charm* cannot be evaluated at all. They need per-strike open interest,
an option chain is published only for today, and the day's chain is gone once
the session ends. `index_options_snapshots` holds zero rows and cannot be
backfilled by any path — only accumulated forward, one row a night. That is a
fact about the data, not a gap in this script.

*Volume* is included because it is asked about and because it is free: turnover,
its trend, and volume-weighted range all get a column here.

Everything is scored the same way — Spearman rank correlation against the next
20 days' realised volatility, on **non-overlapping** windows so neighbours do not
share the future they are being judged on.
"""

from __future__ import annotations

import argparse
import asyncio

import numpy as np
from sqlalchemy import select

from app.data.store import CandleStore
from app.db import session_scope
from app.indicators.series import candles_to_series
from app.models.enums import Interval
from app.models.instrument import MarketDataMapping

TRADING_DAYS = 252
WINDOW = 20

#: Fetched live rather than stored: these are index levels, not tradable
#: instruments, and none is in the candle store. Read-only and free.
PROXIES = {
    "^VIX": "implied vol, 30-day (the Black-Scholes answer)",
    "^VIX3M": "implied vol, 3-month",
    "^SKEW": "tail-risk skew",
    "^VVIX": "volatility of volatility",
}


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 30:
        return float("nan")
    rx = np.argsort(np.argsort(x[ok])).astype(float)
    ry = np.argsort(np.argsort(y[ok])).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    denominator = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / denominator) if denominator > 0 else float("nan")


def _fetch(symbols: list[str], index: object) -> dict[str, np.ndarray]:
    """Daily closes for each proxy, reindexed onto the SPY calendar."""
    import warnings

    import yfinance as yf

    warnings.filterwarnings("ignore")
    out: dict[str, np.ndarray] = {}
    for symbol in symbols:
        try:
            frame = yf.Ticker(symbol).history(period="max", interval="1d")
        except Exception:
            continue
        if frame.empty:
            continue
        series = frame["Close"]
        series.index = series.index.tz_localize(None).normalize()
        # Forward-fill onto our calendar: a proxy that did not print on a day is
        # carried, never interpolated forward from the future.
        aligned = series.reindex(index, method="ffill")
        out[symbol] = aligned.to_numpy(dtype=np.float64)
    return out


async def _load_spy(symbol: str):
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
    return candles_to_series(candles), [c.timestamp for c in candles]


async def _run(symbol: str) -> None:
    import pandas as pd

    series, timestamps = await _load_spy(symbol)
    if series is None:
        print(f"{symbol}: no usable history.")
        return

    index = pd.DatetimeIndex([t.replace(tzinfo=None) for t in timestamps]).normalize()
    close = series.close
    volume = series.volume
    daily = np.concatenate([[np.nan], close[1:] / close[:-1] - 1.0])
    n = close.size

    # -- the thing being predicted -----------------------------------------
    # Strictly the FUTURE: starts at i + 1. Starting at i would put day i's
    # return on both sides of the comparison — it is already inside every
    # trailing estimator — and quietly inflate every score by one day of
    # overlap, most of all for the estimators that lean on the latest bar.
    future_vol = np.full(n, np.nan)
    for i in range(WINDOW, n - WINDOW - 1):
        future_vol[i] = float(np.std(daily[i + 1 : i + 1 + WINDOW], ddof=1)) * np.sqrt(TRADING_DAYS)

    # -- candidates ---------------------------------------------------------
    trailing = np.full(n, np.nan)
    for i in range(WINDOW + 1, n):
        trailing[i] = float(np.std(daily[i - WINDOW + 1 : i + 1], ddof=1)) * np.sqrt(TRADING_DAYS)

    candidates: dict[str, np.ndarray] = {"trailing realised vol (current)": trailing}

    # Range-based: high-low captures intraday movement a close-to-close
    # estimate cannot see, and is a classic improvement on it.
    parkinson = np.full(n, np.nan)
    log_hl = np.log(series.high / np.maximum(series.low, 1e-12)) ** 2
    for i in range(WINDOW + 1, n):
        parkinson[i] = float(
            np.sqrt(np.mean(log_hl[i - WINDOW + 1 : i + 1]) / (4 * np.log(2)))
        ) * np.sqrt(TRADING_DAYS)
    candidates["Parkinson range vol"] = parkinson

    # Volume, since it was asked about.
    vol_trend = np.full(n, np.nan)
    for i in range(60, n):
        recent = float(np.mean(volume[i - 20 : i]))
        base = float(np.mean(volume[i - 60 : i]))
        vol_trend[i] = recent / base if base > 0 else np.nan
    candidates["volume trend (20d / 60d)"] = vol_trend
    candidates["turnover (price x volume)"] = close * volume

    fetched = _fetch(list(PROXIES), index)
    for name, values in fetched.items():
        candidates[f"{name} — {PROXIES[name]}"] = values
    if "^VIX" in fetched and "^VIX3M" in fetched:
        with np.errstate(divide="ignore", invalid="ignore"):
            candidates["VIX3M / VIX term structure"] = fetched["^VIX3M"] / fetched["^VIX"]
    if "^VIX" in fetched:
        with np.errstate(divide="ignore", invalid="ignore"):
            candidates["VIX / trailing realised"] = fetched["^VIX"] / (trailing * 100.0)

    # -- score --------------------------------------------------------------
    step = np.arange(WINDOW + 1, n - WINDOW, WINDOW)
    print(f"Instrument:  {symbol}   {n:,} bars")
    print(f"Predicting:  realised volatility over the NEXT {WINDOW} trading days")
    print(f"Sampling:    {len(step)} non-overlapping windows\n")
    print(f"  {'predictor':<44} {'rank corr':>10}")

    scored = []
    for name, values in candidates.items():
        ic = _spearman(values[step], future_vol[step])
        if np.isfinite(ic):
            scored.append((name, ic))
    for name, ic in sorted(scored, key=lambda kv: -abs(kv[1])):
        marker = "  <- in use" if "current" in name else ""
        print(f"  {name:<44} {ic:>+10.3f}{marker}")

    print(
        "\n  GEX and charm are absent because they cannot be tested: they need\n"
        "  per-strike open interest, the chain exists only for today, and\n"
        "  `index_options_snapshots` holds zero rows. They can be accumulated\n"
        "  forward at one row a night, never backfilled."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank predictors of future volatility.")
    parser.add_argument("--symbol", default="SPY")
    args = parser.parse_args()
    asyncio.run(_run(symbol=args.symbol))


if __name__ == "__main__":
    main()
