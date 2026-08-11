"""Kronos on a single liquid instrument, e.g. the S&P 500 (§8).

    python -m app.scripts.measure_kronos_index --symbol SPY --horizon 10

Kronos scored no better than a two-line moving average across a universe of
small caps. The obvious objection is that it was trained across 45 global
exchanges, where liquid large caps dominate, and simply does not transfer to
thinly-traded microcaps. This tests that objection directly on the most liquid
instrument there is.

**The statistics have to change for one instrument, and this is the trap.**
Sampling a 10-day forward return at every bar gives windows that overlap by 90%,
so a thousand "observations" carry perhaps a hundred bars of independent
information. Ordinary error bars would then be roughly three times too narrow
and everything would look significant. So this samples **non-overlapping**
windows only: one observation per `horizon` bars, each looking at a stretch of
future that no other observation has seen.

The cost is sample size — fifteen years of daily bars gives ~375 independent
10-day windows, not 3,750. That is the honest number, and it is why this reports
the effective count rather than the bar count.
"""

from __future__ import annotations

import argparse
import asyncio
import math

import numpy as np
from sqlalchemy import select

from app.backtest.features import compute, forward_returns
from app.data.store import CandleStore
from app.db import session_scope
from app.indicators.series import candles_to_series
from app.models.enums import Interval
from app.models.instrument import Instrument, MarketDataMapping

BASELINE_FEATURE = "discount_sma200"
CONTEXT_BARS = 300


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 10:
        return float("nan")
    rx, ry = _rank(x), _rank(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denominator = math.sqrt(float((rx * rx).sum()) * float((ry * ry).sum()))
    return float((rx * ry).sum() / denominator) if denominator > 0 else float("nan")


def _rank(values: np.ndarray) -> np.ndarray:
    order = values.argsort()
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    return ranks


async def _run(
    symbol: str, horizon: int, samples: int, model: str, limit: int, dump: str | None
) -> None:
    from app.signals.kronos_client import KronosClient, is_available, repo_is_present

    if not (is_available() and repo_is_present()):
        print("Kronos is not set up. Run: uv sync --extra kronos --extra dev")
        print("                     then: python -m app.scripts.setup_kronos")
        return

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
            print(f"No mapping for {symbol}.")
            return
        instrument = await session.get(Instrument, mapping.instrument_id)
        store = CandleStore(session)
        bars = await store.get_candles(
            mapping.instrument_id, Interval.D1, limit=5000, closed_only=True
        )

    if len(bars) < CONTEXT_BARS + horizon * 3:
        print(f"{symbol}: only {len(bars)} bars — too few.")
        return

    series = candles_to_series(bars)
    timestamps = [c.timestamp for c in bars]
    features = compute(series.open, series.high, series.low, series.close, series.volume)
    forward = forward_returns(series.close, horizon)

    # Non-overlapping windows: step by `horizon`, so no two observations share
    # any of the future they are measuring.
    first = CONTEXT_BARS
    last = series.length - horizon - 1
    indices = list(range(first, last, horizon))
    if limit and len(indices) > limit:
        # Evenly spread rather than truncated, so the sample still spans the
        # whole history rather than only its earliest stretch.
        indices = [indices[i] for i in np.linspace(0, len(indices) - 1, limit, dtype=int)]

    name = instrument.name if instrument else symbol
    print(f"Instrument: {name} ({symbol})")
    print(f"History:    {len(bars):,} bars, {timestamps[0].date()} to {timestamps[-1].date()}")
    print(f"Sampling:   {len(indices)} NON-OVERLAPPING {horizon}-day windows")
    print(f"Model:      {model}, {samples} paths per forecast\n")

    client = KronosClient(model)
    rows: list[tuple[float, float, float, float, float]] = []
    for n, i in enumerate(indices, start=1):
        if not (np.isfinite(forward[i]) and np.isfinite(features[BASELINE_FEATURE][i])):
            continue
        window = series.head(i + 1)
        forecast = client.forecast(
            open_=window.open,
            high=window.high,
            low=window.low,
            close=window.close,
            volume=window.volume,
            timestamps=timestamps[: i + 1],
            horizon_days=horizon,
            sample_count=samples,
        )
        if forecast is None:
            continue
        rows.append(
            (
                forecast.predicted_return,
                forecast.prob_up,
                forecast.path_dispersion,
                float(features[BASELINE_FEATURE][i]),
                float(forward[i]),
            )
        )
        if n % 25 == 0:
            print(f"  {n}/{len(indices)} windows")

    if len(rows) < 30:
        print(f"\nOnly {len(rows)} usable windows — too few to read.")
        return

    block = np.asarray(rows, dtype=np.float64)
    y = block[:, 4]
    error = 2.0 / math.sqrt(max(len(rows) - 3, 1))

    print(f"\n  {len(rows)} independent windows")
    print(f"  mean {horizon}-day return over the sample: {y.mean():+.2%}")
    print(f"\n  {'feature':<24} {'IC':>8}  {'+/- 2 se':>9}")
    for label, column in (
        ("kronos_return", 0),
        ("kronos_prob_up", 1),
        ("kronos_dispersion", 2),
        (f"{BASELINE_FEATURE} (free)", 3),
    ):
        ic = _spearman(block[:, column], y)
        mark = "  *" if abs(ic) > error else ""
        print(f"  {label:<24} {ic:>+8.4f}  {error:>9.4f}{mark}")

    # A directional read as well as a rank one: an index that rises over the
    # sample makes "always long" look clever, so the comparison that matters is
    # against holding it, not against zero.
    predicted_up = block[:, 0] > 0
    if predicted_up.any() and (~predicted_up).any():
        up, down = y[predicted_up], y[~predicted_up]
        print(
            f"\n  when Kronos predicts up:   {up.mean():+.2%} "
            f"({up.size} windows, sd {up.std(ddof=1):.2%})"
        )
        print(
            f"  when Kronos predicts down: {down.mean():+.2%} "
            f"({down.size} windows, sd {down.std(ddof=1):.2%})"
        )
        print(f"  buy and hold:              {y.mean():+.2%} (all {y.size} windows)")

        # Welch's t: the two groups have different sizes and, usually, different
        # variances, so the pooled form would understate the error. A difference
        # in means is the claim being made — "following the signal beats holding"
        # — and without this the gap is just two numbers that differ.
        se = math.sqrt(up.std(ddof=1) ** 2 / up.size + down.std(ddof=1) ** 2 / down.size)
        difference = up.mean() - down.mean()
        t_stat = difference / se if se > 0 else 0.0
        verdict = "SIGNIFICANT" if abs(t_stat) > 2 else "not significant"
        print(
            f"\n  difference {difference:+.2%}, standard error {se:.2%}, "
            f"t = {t_stat:+.2f} -> {verdict}"
        )

    if dump:
        await asyncio.to_thread(_write_csv, dump, block.tolist())
        print(f"\n  raw rows written to {dump} — re-analysable without re-running Kronos")


def _write_csv(path: str, rows: list[list[float]]) -> None:
    """Off the event loop: a blocking write inside a coroutine stalls it."""
    import csv

    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["kronos_return", "prob_up", "dispersion", BASELINE_FEATURE, "forward_return"]
        )
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure Kronos on one liquid instrument.")
    parser.add_argument("--symbol", default="SPY", help="Provider symbol, e.g. SPY or VUAG.L.")
    parser.add_argument("--horizon", type=int, default=10, help="Forward-return horizon.")
    parser.add_argument("--samples", type=int, default=4, help="Paths per forecast.")
    parser.add_argument("--model", default="kronos-small", help="Variant.")
    parser.add_argument("--limit", type=int, default=150, help="Cap on windows sampled.")
    parser.add_argument("--dump", help="Write the raw rows to this CSV.")
    args = parser.parse_args()
    asyncio.run(
        _run(
            symbol=args.symbol,
            horizon=args.horizon,
            samples=args.samples,
            model=args.model,
            limit=args.limit,
            dump=args.dump,
        )
    )


if __name__ == "__main__":
    main()
