"""What would £5,000 in an S&P tracker have become, timed? (§9)

    python -m app.scripts.backtest_index_portfolio --capital 5000

The stock pipeline decides *which* names to buy. This decides only *when to be
invested at all* in one tracker — in or out, no selection. So the benchmark is
not "did it make money" but "did it beat leaving the money in the tracker",
which over a rising market is a hard bar and the only honest one.

**The split is walk-forward, and it has to be.** The index model ships fitted on
SPY's whole history, so measuring it on SPY would be measuring how well it
memorised the answers. This refits on the early slice and trades only the later
one, which the fit never saw. `--split` moves the boundary.

**Three numbers decide whether timing was worth it**, and only the first is
obvious:

  * final equity against buy-and-hold — did it beat doing nothing;
  * worst drawdown against buy-and-hold — timing usually *costs* return and
    buys a smoother ride, so a strategy that loses a little and halves the
    drawdown may still be the one you want;
  * time in market — a strategy invested 95% of the time is a tracker with
    extra steps, and one invested 20% of the time is mostly a cash holding
    whose result says more about cash than about the signal.

Costs are charged per round trip, but they matter far less here than in the
stock pipeline: a tracker trades on a penny spread, and the strategy makes a
handful of decisions a year rather than hundreds.
"""

from __future__ import annotations

import argparse
import asyncio

import numpy as np
from sqlalchemy import select

from app.backtest.features import compute, forward_returns
from app.data.store import CandleStore
from app.db import session_scope
from app.indicators.series import PriceSeries, candles_to_series
from app.models.enums import Interval
from app.models.instrument import MarketDataMapping
from app.models_ml.logistic import FittedModel, Prior, fit
from app.strategies.logistic_index import PRICE_FEATURES

HORIZON_DAYS = 20

#: Where the model may first be evaluated: the feature module returns nothing
#: below 220 bars, and the 200-day slope wants its fit window on top.
WARMUP = 300


def _fit_on(series: PriceSeries, upto: int) -> FittedModel | None:
    """Fit on bars [WARMUP, upto), using non-overlapping forward windows.

    Non-overlapping because a 20-day return read at every bar shares 95% of its
    future with its neighbour, and error bars computed on the overlapping
    version would be about three times too narrow.
    """
    columns = compute(series.open, series.high, series.low, series.close, series.volume)
    if not columns:
        return None
    forward = forward_returns(series.close, HORIZON_DAYS)

    rows: list[list[float]] = []
    labels: list[float] = []
    for i in range(WARMUP, upto - HORIZON_DAYS, HORIZON_DAYS):
        if not np.isfinite(forward[i]):
            continue
        row = [float(columns[n][i]) if n in columns else np.nan for n in PRICE_FEATURES]
        if not any(np.isfinite(v) for v in row):
            continue
        rows.append(row)
        labels.append(1.0 if forward[i] > 0 else 0.0)

    if len(rows) < 40:
        return None
    return fit(
        np.asarray(rows),
        np.asarray(labels),
        PRICE_FEATURES,
        priors={n: Prior(0.0, 1.0) for n in PRICE_FEATURES},
        label_definition="1 if the forward 20-day return is positive",
    )


def _drawdown(curve: np.ndarray) -> float:
    peak = np.maximum.accumulate(curve)
    return float(np.max((peak - curve) / peak)) if curve.size else 0.0


def _simulate(
    series: PriceSeries,
    model: FittedModel,
    start: int,
    *,
    capital: float,
    entry_p: float,
    exit_p: float,
    cost_pct: float,
) -> tuple[np.ndarray, int, float]:
    """Walk the held-out slice, in or out. Returns (curve, round trips, share invested)."""
    columns = compute(series.open, series.high, series.low, series.close, series.volume)
    cash = capital
    shares = 0.0
    curve: list[float] = []
    trips = 0
    days_in = 0

    for i in range(start, series.length):
        row = {n: float(columns[n][i]) for n in PRICE_FEATURES if n in columns}
        row = {k: v for k, v in row.items() if np.isfinite(v)}
        probability = model.probability(row) if row else 0.5

        # Decided on this close, filled at the next open — the same discipline
        # the stock replay uses, and for the same reason.
        nxt = i + 1
        if nxt >= series.length:
            break
        fill = float(series.open[nxt])

        if shares == 0.0 and probability >= entry_p and fill > 0:
            spend = cash * (1.0 - cost_pct / 2.0)
            shares = spend / fill
            cash = 0.0
            trips += 1
        elif shares > 0.0 and probability < exit_p:
            cash = shares * fill * (1.0 - cost_pct / 2.0)
            shares = 0.0

        value = cash + shares * float(series.close[i])
        curve.append(value)
        if shares > 0.0:
            days_in += 1

    if shares > 0.0:
        cash = shares * float(series.close[-1]) * (1.0 - cost_pct / 2.0)
        shares = 0.0
        curve.append(cash)

    total = max(len(curve), 1)
    return np.asarray(curve), trips, days_in / total


async def _load(symbol: str) -> PriceSeries | None:
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
            return None
        candles = await CandleStore(session).get_candles(
            mapping.instrument_id, Interval.D1, limit=10_000, closed_only=True
        )
    if len(candles) < 800:
        return None
    return candles_to_series(candles)


async def _run(
    symbol: str, capital: float, split: float, entry_p: float, exit_p: float, costs: list[float]
) -> None:
    series = await _load(symbol)
    if series is None:
        print(f"{symbol}: no usable history.")
        return

    cut = int(series.length * split)
    model = _fit_on(series, cut)
    if model is None:
        print("Not enough history before the split to fit on.")
        return

    print(f"Instrument:  {symbol}")
    print(f"Capital:     £{capital:,.0f}")
    print(f"Fitted on:   bars {WARMUP}-{cut} ({model.n_observations} independent windows)")
    print(f"Traded on:   bars {cut}-{series.length} — never seen by the fit")
    print(f"Rule:        in when P >= {entry_p:.0%}, out when P < {exit_p:.0%}")
    print(f"\n  {'feature':<20} {'coef':>9}")
    for name, coefficient in zip(model.feature_names, model.coefficients, strict=True):
        print(f"  {name:<20} {coefficient:>+9.4f}")

    held = series.close[cut:]
    hold_curve = capital * held / float(held[0])
    print("\n  BUY AND HOLD over the traded slice")
    print(f"    final equity      £{hold_curve[-1]:,.2f}")
    print(
        f"    profit / loss     £{hold_curve[-1] - capital:+,.2f}   "
        f"({(hold_curve[-1] - capital) / capital:+.1%})"
    )
    print(f"    worst drawdown    {_drawdown(hold_curve):.1%}")

    for cost in costs:
        curve, trips, invested = _simulate(
            series,
            model,
            cut,
            capital=capital,
            entry_p=entry_p,
            exit_p=exit_p,
            cost_pct=cost,
        )
        if curve.size == 0:
            continue
        final = float(curve[-1])
        print(f"\n  TIMED, round-trip cost {cost:.2%}")
        print(f"    final equity      £{final:,.2f}")
        print(
            f"    profit / loss     £{final - capital:+,.2f}   ({(final - capital) / capital:+.1%})"
        )
        print(f"    worst drawdown    {_drawdown(curve):.1%}")
        print(f"    round trips       {trips}")
        print(f"    time invested     {invested:.0%}")
        print(f"    vs buy and hold   £{final - float(hold_curve[-1]):+,.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Portfolio backtest of index timing.")
    parser.add_argument("--symbol", default="SPY", help="Tracker to time.")
    parser.add_argument("--capital", type=float, default=5000.0)
    parser.add_argument("--split", type=float, default=0.6, help="Fraction of history to fit on.")
    parser.add_argument("--entry", type=float, default=0.55)
    parser.add_argument("--exit", dest="exit_p", type=float, default=0.45)
    parser.add_argument("--costs", type=float, nargs="+", default=[0.0005, 0.002])
    args = parser.parse_args()
    asyncio.run(
        _run(
            symbol=args.symbol,
            capital=args.capital,
            split=args.split,
            entry_p=args.entry,
            exit_p=args.exit_p,
            costs=args.costs,
        )
    )


if __name__ == "__main__":
    main()
