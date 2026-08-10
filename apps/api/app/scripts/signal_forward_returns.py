"""Does any signal we compute actually precede better returns? (§8)

    python -m app.scripts.signal_forward_returns --size 1000

Twenty-seven configurations of the mean-reversion strategy all returned the same
zero expectancy, which says the *entry* carries no information — but a strategy
result cannot say that cleanly, because stops, targets and trailing all sit
between the signal and the outcome. This strips every one of them away.

The question here is the simplest one available: **after a signal fires, what
happens to the price over the next N days?** No stop, no target, no position
sizing. If a signal's forward return is indistinguishable from a randomly chosen
bar on the same instrument, no exit design can rescue a strategy built on it, and
the honest move is to stop tuning and change signal.

The comparison is against **random bars drawn from the same instruments**, not
against zero, which matters: a universe that drifted up over the sample would
make any long signal look predictive against a zero baseline. What is reported is
the *edge* — signal mean minus baseline mean — with a standard error, so a
difference can be read against its own noise.

Reads the candle store only. Pure measurement: it writes nothing and changes no
behaviour.
"""

from __future__ import annotations

import argparse
import asyncio
import math
import random

import numpy as np

from app.backtest.engine import is_continuous
from app.backtest.service import BacktestService
from app.data.store import CandleStore
from app.db import session_scope
from app.indicators.series import PriceSeries, candles_to_series
from app.models.enums import Interval
from app.strategies.mean_reversion import EntryRules, read_entry

#: Horizons to measure, in trading days. The strategy's median hold is ~7 days,
#: so 5-20 brackets it; 60 is included because a value screen — if it works at
#: all — should work on a slower clock than a dip-buying trade.
HORIZONS = (5, 10, 20, 60)

#: Bars to skip so the 200-day trend gate is measurable, matching the backtest.
WARMUP = 260


def _forward_return(closes: np.ndarray, i: int, horizon: int) -> float | None:
    """Simple return from bar i to bar i+horizon, or None past the end."""
    if i + horizon >= closes.size:
        return None
    start, end = float(closes[i]), float(closes[i + horizon])
    if start <= 0:
        return None
    return end / start - 1.0


def _welford(values: list[float]) -> tuple[float, float, int]:
    """(mean, standard error, n) — the only statistics this script reports."""
    n = len(values)
    if n < 2:
        return (values[0] if values else 0.0, 0.0, n)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(variance / n), n


def _print_row(label: str, signal: list[float], baseline: list[float]) -> None:
    s_mean, s_se, s_n = _welford(signal)
    b_mean, b_se, _ = _welford(baseline)
    if s_n < 30:
        print(f"    {label:<12} {s_n:>6} signals — too few to read")
        return
    edge = s_mean - b_mean
    # Standard error of a difference of independent means.
    edge_se = math.sqrt(s_se**2 + b_se**2)
    verdict = "  *" if abs(edge) > 2 * edge_se else ""
    print(
        f"    {label:<12} {s_n:>6} signals   "
        f"signal {s_mean:>+7.2%}   baseline {b_mean:>+7.2%}   "
        f"edge {edge:>+7.2%} +/-{2 * edge_se:.2%}{verdict}"
    )


async def _run(size: int, seed: int) -> None:
    rng = random.Random(seed)
    rules = EntryRules()

    # signal name -> horizon -> returns
    signal_returns: dict[str, dict[int, list[float]]] = {
        "entry fires": {h: [] for h in HORIZONS},
        "score >= 0.8": {h: [] for h in HORIZONS},
        "RSI <= 30": {h: [] for h in HORIZONS},
        "below lower band": {h: [] for h in HORIZONS},
    }
    baseline: dict[int, list[float]] = {h: [] for h in HORIZONS}

    async with session_scope() as session:
        service = BacktestService(session)
        instruments = await service.top_ranked_instruments(size)
        if not instruments:
            instruments = await service.instruments_with_history(size)
        if not instruments:
            print("No instruments. Ingest candles or run a scan first.")
            return
        store = CandleStore(session)

        used = 0
        for instrument in instruments:
            candles = await store.get_candles(
                instrument.id, Interval.D1, limit=2000, closed_only=True
            )
            if len(candles) < WARMUP + max(HORIZONS) + 1:
                continue
            series = candles_to_series(candles)
            if not is_continuous(series):
                continue
            used += 1
            closes = series.close

            for i in range(WARMUP, series.length):
                window: PriceSeries = series.head(i + 1)
                reading = read_entry(window, rules)
                if reading is None:
                    continue

                fired: list[str] = []
                if reading.admits:
                    fired.append("entry fires")
                if reading.score >= 0.80:
                    fired.append("score >= 0.8")
                if reading.rsi <= 30.0:
                    fired.append("RSI <= 30")
                if float(closes[i]) <= reading.lower:
                    fired.append("below lower band")

                for horizon in HORIZONS:
                    forward = _forward_return(closes, i, horizon)
                    if forward is None:
                        continue
                    for name in fired:
                        signal_returns[name][horizon].append(forward)

            # Baseline: random bars from the same instrument, so the comparison
            # controls for whatever this name did over the sample. Sampled at a
            # rate that keeps the baseline comfortably larger than any signal set.
            span = series.length - WARMUP - max(HORIZONS)
            if span > 0:
                for _ in range(min(span, 200)):
                    i = rng.randrange(WARMUP, WARMUP + span)
                    for horizon in HORIZONS:
                        forward = _forward_return(closes, i, horizon)
                        if forward is not None:
                            baseline[horizon].append(forward)

    print(f"Instruments: {used} of {len(instruments)} usable")
    print("Baseline:    random bars from the same instruments (not zero)")
    print("Marked *:    edge is more than two standard errors from the baseline\n")
    for name, by_horizon in signal_returns.items():
        print(f"  {name}")
        for horizon in HORIZONS:
            _print_row(f"{horizon}d", by_horizon[horizon], baseline[horizon])
        print()
    print(
        "  A signal whose edge band spans zero carries no information at that\n"
        "  horizon, and no stop/target design can extract what is not there."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Forward returns after each signal.")
    parser.add_argument("--size", type=int, default=400, help="Instruments to scan.")
    parser.add_argument("--seed", type=int, default=20260810, help="Baseline sampling seed.")
    args = parser.parse_args()
    asyncio.run(_run(size=args.size, seed=args.seed))


if __name__ == "__main__":
    main()
