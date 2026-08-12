"""What would £5,000 have become? A portfolio backtest in money (§8).

    python -m app.scripts.backtest_portfolio --capital 5000

Every other measurement in this repo is per-instrument and denominated in R —
one R being the distance from entry to the initial stop. That is the right unit
for judging an *entry rule*, because it is comparable across instruments and
independent of position size. It is the wrong unit for answering "how much money
would I have made", because it silently assumes you could take every trade at a
constant risk, with unlimited capital and no costs.

This answers the money question, and the differences from the R-based runs are
where the honesty lives:

  * **One shared pot.** Capital is finite. A signal on a day when the book is
    full, or the cash is committed, is simply missed — exactly as it would be
    live. The R-based replay takes every trade it sees.
  * **The real risk limits**, read from `RiskConfiguration`'s defaults rather
    than invented here: 1% of equity risked per trade, 10% ceiling per position,
    6% total open risk, at most 10 positions at once.
  * **Costs are charged.** Spread and commission come off every round trip. The
    traded universe has a median price near £1.64 and a median daily turnover
    near £90,000, where a 2-3% spread is ordinary — and 1R is about 31% of the
    share price, so that spread is 0.06-0.10R a trade against a measured
    expectancy of about -0.08R. A backtest that omits it is not approximately
    right, it is reporting a different strategy.
  * **Position sizing compounds.** Risk is 1% of *current* equity, so losses
    shrink the next position and gains grow it.

**The instruments are the ones the model never saw.** Folds split by a stable
hash of the instrument id, the model is fitted on `fit`, and this runs on
`confirm`. Testing on the training set would measure how well the fit memorised
its own sample, which is never the question.

Entries fill at the next open, stops rest intrabar and fill at the open on a
gap, targets are decided on a close and filled next open — the same pessimistic
execution the R-based replay uses, for the same reason.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass, field
from datetime import date

import numpy as np

from app.backtest.engine import is_continuous
from app.backtest.entries import ModelReader
from app.backtest.service import BacktestService
from app.data.store import CandleStore
from app.db import session_scope
from app.indicators.series import PriceSeries, candles_to_series
from app.models.enums import Interval, StrategyKind
from app.models_ml.logistic import FittedModel
from app.services.strategy_model import StrategyModelService

#: Read from `RiskConfiguration`'s defaults so this simulates the system that
#: would actually trade, not a more permissive one invented for the backtest.
RISK_PER_TRADE_PCT = 0.01
MAX_POSITION_PCT = 0.10
MAX_TOTAL_OPEN_RISK_PCT = 0.06
MAX_OPEN_POSITIONS = 10
ATR_STOP_MULTIPLIER = 5.0

#: Round-trip cost as a fraction of notional: spread plus commission, charged on
#: entry and exit. The default is deliberately modest for the universe — see the
#: module docstring — and `--cost` sweeps it, because the answer's sensitivity to
#: this number is itself a finding.
DEFAULT_COST_PCT = 0.02


@dataclass
class Position:
    instrument: str
    shares: float
    entry_price: float
    stop: float
    entry_day: int


@dataclass
class Ledger:
    capital: float
    equity: float
    cash: float
    positions: dict[str, Position] = field(default_factory=dict)
    closed: list[float] = field(default_factory=list)  # realised P/L per trade
    curve: list[tuple[date, float]] = field(default_factory=list)
    missed_no_capital: int = 0
    missed_book_full: int = 0
    costs_paid: float = 0.0


@dataclass
class Candidate:
    """One instrument's aligned series and precomputed model readings."""

    name: str
    dates: list[date]
    series: PriceSeries
    admits: np.ndarray
    target: np.ndarray
    atr: np.ndarray


async def _load(
    service: BacktestService, store: CandleStore, model: FittedModel, threshold: float, size: int
) -> list[Candidate]:
    """Confirm-fold instruments, with the model evaluated at every bar."""
    universe = await service.top_ranked_instruments(size)
    if not universe:
        universe = await service.instruments_with_history(size)
    held_out = service.split(universe, fold="confirm")

    out: list[Candidate] = []
    for instrument in held_out:
        candles = await store.get_candles(instrument.id, Interval.D1, limit=1200, closed_only=True)
        if len(candles) < 340:
            continue
        series = candles_to_series(candles)
        if not is_continuous(series):
            continue

        reader = ModelReader(
            model=model, threshold=threshold, atr_stop_multiplier=ATR_STOP_MULTIPLIER
        )
        reader.prepare(series)
        n = series.length
        admits = np.zeros(n, dtype=bool)
        target = np.full(n, np.nan)
        atr = np.full(n, np.nan)
        for i in range(300, n):
            reading = reader(series.head(i + 1))
            if reading is None:
                continue
            admits[i] = reading.admits
            target[i] = reading.target
            atr[i] = reading.atr

        out.append(
            Candidate(
                name=instrument.name or str(instrument.id),
                dates=[c.timestamp.date() for c in candles],
                series=series,
                admits=admits,
                target=target,
                atr=atr,
            )
        )
    return out


def _simulate(candidates: list[Candidate], capital: float, cost_pct: float) -> Ledger:
    """Walk the calendar once, sharing one pot of capital across everything."""
    ledger = Ledger(capital=capital, equity=capital, cash=capital)

    # A master calendar, so every instrument is seen on the same day and the
    # capital constraint is real rather than per-instrument.
    calendar = sorted({d for c in candidates for d in c.dates})
    index: dict[str, dict[date, int]] = {
        c.name: {d: i for i, d in enumerate(c.dates)} for c in candidates
    }
    by_name = {c.name: c for c in candidates}

    for day_number, today in enumerate(calendar):
        # --- 1. Manage what is open, before considering anything new. --------
        for name in list(ledger.positions):
            position = ledger.positions[name]
            candidate = by_name[name]
            i = index[name].get(today)
            if i is None:
                continue
            low = float(candidate.series.low[i])
            open_ = float(candidate.series.open[i])
            close = float(candidate.series.close[i])

            exit_price: float | None = None
            if low <= position.stop:
                # Resting order: a gap through it fills at the open, not at the
                # price you asked for.
                exit_price = min(open_, position.stop)
            elif not np.isnan(candidate.target[i]) and close >= candidate.target[i]:
                # Decided on the close, filled at the next open.
                nxt = (
                    index[name].get(calendar[day_number + 1])
                    if day_number + 1 < len(calendar)
                    else None
                )
                if nxt is not None:
                    exit_price = float(candidate.series.open[nxt])

            if exit_price is not None:
                proceeds = position.shares * exit_price
                cost = proceeds * cost_pct / 2.0
                ledger.cash += proceeds - cost
                ledger.costs_paid += cost
                ledger.closed.append(proceeds - cost - position.shares * position.entry_price)
                del ledger.positions[name]

        # --- 2. Mark the book to market. -------------------------------------
        holdings = 0.0
        for name, position in ledger.positions.items():
            i = index[name].get(today)
            price = float(by_name[name].series.close[i]) if i is not None else position.entry_price
            holdings += position.shares * price
        ledger.equity = ledger.cash + holdings
        ledger.curve.append((today, ledger.equity))

        if ledger.equity <= 0:
            break

        # --- 3. New entries, subject to every limit. -------------------------
        open_risk = sum(p.shares * (p.entry_price - p.stop) for p in ledger.positions.values())
        for candidate in candidates:
            if candidate.name in ledger.positions:
                continue
            i = index[candidate.name].get(today)
            if i is None or not candidate.admits[i] or np.isnan(candidate.atr[i]):
                continue
            nxt_day = calendar[day_number + 1] if day_number + 1 < len(calendar) else None
            nxt = index[candidate.name].get(nxt_day) if nxt_day else None
            if nxt is None:
                continue

            if len(ledger.positions) >= MAX_OPEN_POSITIONS:
                ledger.missed_book_full += 1
                break

            entry = float(candidate.series.open[nxt])
            stop_distance = candidate.atr[i] * ATR_STOP_MULTIPLIER
            if entry <= 0 or stop_distance <= 0 or entry - stop_distance <= 0:
                continue

            # Size by risk, then apply every ceiling; the smallest wins.
            risk_budget = ledger.equity * RISK_PER_TRADE_PCT
            remaining_risk = ledger.equity * MAX_TOTAL_OPEN_RISK_PCT - open_risk
            if remaining_risk <= 0:
                continue
            budget = min(risk_budget, remaining_risk)
            shares = budget / stop_distance
            shares = min(shares, ledger.equity * MAX_POSITION_PCT / entry)

            notional = shares * entry
            cost = notional * cost_pct / 2.0
            if notional + cost > ledger.cash:
                # Not enough cash left. Real, and worth counting rather than
                # quietly scaling the position down to fit.
                ledger.missed_no_capital += 1
                continue
            if shares <= 0 or notional < 1.0:
                continue

            ledger.cash -= notional + cost
            ledger.costs_paid += cost
            open_risk += shares * stop_distance
            ledger.positions[candidate.name] = Position(
                instrument=candidate.name,
                shares=shares,
                entry_price=entry,
                stop=entry - stop_distance,
                entry_day=day_number,
            )

    # Close whatever is still open at the last price, rather than dropping it:
    # silently discarding open positions flatters a strategy whose losers are
    # simply held longer than its winners.
    for name, position in list(ledger.positions.items()):
        last = float(by_name[name].series.close[-1])
        proceeds = position.shares * last
        cost = proceeds * cost_pct / 2.0
        ledger.cash += proceeds - cost
        ledger.costs_paid += cost
        ledger.closed.append(proceeds - cost - position.shares * position.entry_price)
    ledger.positions.clear()
    ledger.equity = ledger.cash
    return ledger


def _report(label: str, ledger: Ledger) -> None:
    profit = ledger.equity - ledger.capital
    trades = len(ledger.closed)
    wins = [p for p in ledger.closed if p > 0]
    curve = np.array([e for _, e in ledger.curve]) if ledger.curve else np.array([ledger.capital])
    peak = np.maximum.accumulate(curve)
    drawdown = float(np.max((peak - curve) / peak)) if curve.size else 0.0

    print(f"\n  {label}")
    print(f"    final equity      £{ledger.equity:,.2f}")
    print(f"    profit / loss     £{profit:+,.2f}   ({profit / ledger.capital:+.1%})")
    print(f"    trades taken      {trades:,}")
    if trades:
        print(f"    win rate          {len(wins) / trades:.1%}")
        print(f"    average trade     £{np.mean(ledger.closed):+,.2f}")
    print(f"    costs paid        £{ledger.costs_paid:,.2f}")
    print(f"    worst drawdown    {drawdown:.1%}")
    print(
        f"    signals missed    {ledger.missed_book_full:,} (book full), "
        f"{ledger.missed_no_capital:,} (no cash)"
    )


def _buy_and_hold(candidates: list[Candidate], capital: float) -> float:
    """Equal-weight, bought at the start and never touched.

    The benchmark that makes every other number readable. A strategy is not
    judged against zero — over a period when the market rose, "made money" is
    not evidence of skill, and over one when it fell, "lost money" is not
    evidence of its absence. This is what the same capital, in the same names,
    over the same period, would have done with no strategy at all.
    """
    stake = capital / len(candidates)
    total = 0.0
    for candidate in candidates:
        first = float(candidate.series.close[300])
        last = float(candidate.series.close[-1])
        total += stake * (last / first) if first > 0 else stake
    return total


async def _run(capital: float, size: int, threshold: float, costs: list[float]) -> None:
    async with session_scope() as session:
        model = await StrategyModelService(session).active(StrategyKind.LOGISTIC_STOCK)
        if model is None:
            print("No active model. Fit one first: python -m app.scripts.fit_stock_model --save")
            return
        service = BacktestService(session)
        store = CandleStore(session)
        candidates = await _load(service, store, model, threshold, size)

    if not candidates:
        print("No held-out instruments with enough history.")
        return

    span = sorted({d for c in candidates for d in c.dates})
    years = (span[-1] - span[0]).days / 365.25
    print(f"Capital:     £{capital:,.0f}")
    print(f"Universe:    {len(candidates)} instruments — the CONFIRM fold, never seen by the fit")
    print(f"Period:      {span[0]} to {span[-1]}  ({years:.1f} years)")
    print(f"Entry:       model probability >= {threshold:.0%}")
    print(
        f"Limits:      {RISK_PER_TRADE_PCT:.0%} risk/trade, {MAX_POSITION_PCT:.0%} max position, "
        f"{MAX_TOTAL_OPEN_RISK_PCT:.0%} open risk, {MAX_OPEN_POSITIONS} positions"
    )

    benchmark = _buy_and_hold(candidates, capital)
    print("\n  BUY AND HOLD the same names, equal weight, no trading:")
    print(f"    final equity      £{benchmark:,.2f}")
    print(
        f"    profit / loss     £{benchmark - capital:+,.2f}   "
        f"({(benchmark - capital) / capital:+.1%})"
    )

    for cost in costs:
        ledger = _simulate(candidates, capital, cost)
        _report(f"round-trip cost {cost:.1%}", ledger)

    print(
        "\n  Costs are the sweep because this universe cannot be traded for free:\n"
        "  a median price near £1.64 and £90k of daily turnover make a 2-3% round\n"
        "  trip ordinary. The zero-cost row is not a forecast — it is the ceiling\n"
        "  the strategy could reach if trading were free, shown so the gap between\n"
        "  it and the others is visible."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Portfolio backtest, in money.")
    parser.add_argument("--capital", type=float, default=5000.0, help="Starting capital.")
    parser.add_argument("--size", type=int, default=300, help="Universe to draw the fold from.")
    parser.add_argument("--threshold", type=float, default=0.55, help="Entry probability.")
    parser.add_argument(
        "--costs",
        type=float,
        nargs="+",
        default=[0.0, 0.005, DEFAULT_COST_PCT],
        help="Round-trip costs to sweep, as fractions of notional.",
    )
    args = parser.parse_args()
    asyncio.run(
        _run(
            capital=args.capital,
            size=args.size,
            threshold=args.threshold,
            costs=args.costs,
        )
    )


if __name__ == "__main__":
    main()
