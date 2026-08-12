"""How *often* does volatility targeting beat buy-and-hold? (§9)

    python -m app.scripts.vol_target_consistency --years 3

The headline result — £35,884 against £32,380 on SPY over fifteen years — is a
single path. One coin landing heads is not evidence about the coin. This runs
the same rule over many separate windows and many instruments and reports the
fraction it wins, which is the number that says whether the rule works or
whether one sample was kind.

**Three things are measured separately**, because a strategy can win on one and
lose on another and the difference matters:

  * **return** — did it end with more money;
  * **drawdown** — was the worst fall shallower;
  * **return per drawdown** — was the gain better per unit of pain.

Volatility targeting is expected to win the second reliably and the first only
sometimes, since with exposure capped it holds less than the market during calm
rises. If that expectation is wrong in either direction, this is where it shows.

**Windows are non-overlapping.** Two five-year windows starting a month apart
share 98% of their days and are very nearly one observation; counting them as
two would make any result look far more certain than it is.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass

import numpy as np
from sqlalchemy import func, select

from app.data.store import CandleStore
from app.db import session_scope
from app.indicators.series import candles_to_series
from app.models.enums import InstrumentKind, Interval
from app.models.instrument import Instrument
from app.models.market_data import Candle

TRADING_DAYS = 252
VOL_WINDOW = 20
BAND = 0.10
BORROW = 0.05
COST = 0.0005


@dataclass
class Outcome:
    instrument: str
    hold_return: float
    hold_drawdown: float
    strategy_return: float
    strategy_drawdown: float

    @property
    def beat_return(self) -> bool:
        return self.strategy_return > self.hold_return

    @property
    def beat_drawdown(self) -> bool:
        return self.strategy_drawdown < self.hold_drawdown

    @property
    def beat_ratio(self) -> bool:
        mine = self.strategy_return / self.strategy_drawdown if self.strategy_drawdown else 0.0
        theirs = self.hold_return / self.hold_drawdown if self.hold_drawdown else 0.0
        return mine > theirs


def _drawdown(curve: np.ndarray) -> float:
    peak = np.maximum.accumulate(curve)
    return float(np.max((peak - curve) / peak)) if curve.size else 0.0


def _run_window(
    daily: np.ndarray,
    vol: np.ndarray,
    start: int,
    stop: int,
    target: float,
    cap: float,
    adaptive: bool,
) -> tuple[float, float, float, float] | None:
    """Buy-and-hold and the volatility rule over one window."""
    if stop - start < 60:
        return None
    slice_daily = daily[start:stop]
    if not np.all(np.isfinite(slice_daily)):
        return None

    if adaptive:
        # Each instrument targeted at its OWN typical volatility, measured only
        # on the days before this window opens. A fixed target silently assumes
        # every asset is as volatile as the one it was chosen on: at 20% against
        # a 5% bond fund the rule is simply pinned at maximum leverage forever,
        # which is not volatility targeting at all.
        history = vol[:start]
        history = history[np.isfinite(history)]
        if history.size < 60:
            return None
        target = float(np.median(history))

    hold = np.cumprod(1.0 + slice_daily)
    equity = 1.0
    exposure = 0.0
    curve = []
    for i in range(start, stop):
        # Sized on the estimate available the day before, as live.
        prior = vol[i - 1]
        wanted = min(target / prior, cap) if np.isfinite(prior) and prior > 0 else exposure
        if abs(wanted - exposure) > BAND:
            equity -= equity * abs(wanted - exposure) * COST / 2.0
            exposure = wanted
        equity *= 1.0 + exposure * daily[i]
        if exposure > 1.0:
            equity -= equity * (exposure - 1.0) * BORROW / TRADING_DAYS
        curve.append(equity)

    strategy = np.asarray(curve)
    if strategy.size == 0:
        return None
    return (
        float(hold[-1]) - 1.0,
        _drawdown(hold),
        float(strategy[-1]) - 1.0,
        _drawdown(strategy),
    )


async def _universe(min_bars: int, kind: InstrumentKind | None) -> list[tuple[str, np.ndarray]]:
    async with session_scope() as session:
        rows = (
            await session.execute(
                select(Instrument.id, Instrument.name)
                .join(Candle, Candle.instrument_id == Instrument.id)
                .where(Instrument.kind == kind if kind else True)
                .where(Candle.interval == Interval.D1)
                .group_by(Instrument.id, Instrument.name)
                .having(func.count(Candle.id) >= min_bars)
            )
        ).all()
        store = CandleStore(session)
        out = []
        for instrument_id, name in rows:
            candles = await store.get_candles(
                instrument_id, Interval.D1, limit=10_000, closed_only=True
            )
            if len(candles) < min_bars:
                continue
            close = candles_to_series(candles).close
            if np.any(close <= 0):
                continue
            # An unadjusted split would dominate any window containing it.
            ratio = close[1:] / close[:-1]
            if np.max(ratio) > 4.0 or np.min(ratio) < 0.25:
                continue
            out.append((name or str(instrument_id), close))
    return out


async def _run(
    years: float,
    target: float,
    cap: float,
    min_bars: int,
    kind: InstrumentKind | None,
    adaptive: bool,
) -> None:
    window = int(years * TRADING_DAYS)
    universe = await _universe(min_bars, kind)
    if not universe:
        print("No instruments with enough history.")
        return

    outcomes: list[Outcome] = []
    for name, close in universe:
        daily = np.concatenate([[np.nan], close[1:] / close[:-1] - 1.0])
        vol = np.full(daily.size, np.nan)
        for i in range(VOL_WINDOW, daily.size):
            vol[i] = float(np.std(daily[i - VOL_WINDOW : i], ddof=1)) * np.sqrt(TRADING_DAYS)

        # Non-overlapping windows, starting once the estimator is warm.
        start = VOL_WINDOW + 5
        while start + window <= daily.size:
            result = _run_window(daily, vol, start, start + window, target, cap, adaptive)
            if result is not None:
                outcomes.append(Outcome(name, *result))
            start += window

    if not outcomes:
        print("No usable windows.")
        return

    n = len(outcomes)
    instruments = len({o.instrument for o in outcomes})
    shown = "each instrument's own median vol" if adaptive else f"{target:.0%}"
    print(f"Rule:        exposure = ({shown}) / volatility, capped at {cap:.0%}")
    print(
        f"Windows:     {n} non-overlapping {years:g}-year windows across {instruments} instruments"
    )
    print(f"Costs:       {COST:.2%} per unit traded, {BORROW:.0%}/yr on leverage\n")

    print(f"  {'measure':<34} {'beats buy & hold':>18}")
    for label, wins in (
        ("higher return", sum(o.beat_return for o in outcomes)),
        ("shallower drawdown", sum(o.beat_drawdown for o in outcomes)),
        ("better return per drawdown", sum(o.beat_ratio for o in outcomes)),
    ):
        print(f"  {label:<34} {wins / n:>17.1%}   ({wins}/{n})")

    hold_returns = np.array([o.hold_return for o in outcomes])
    mine = np.array([o.strategy_return for o in outcomes])
    hold_dd = np.array([o.hold_drawdown for o in outcomes])
    my_dd = np.array([o.strategy_drawdown for o in outcomes])

    print(f"\n  {'':<20} {'buy & hold':>12} {'strategy':>12}")
    print(f"  {'median return':<20} {np.median(hold_returns):>11.1%} {np.median(mine):>12.1%}")
    print(f"  {'median drawdown':<20} {np.median(hold_dd):>11.1%} {np.median(my_dd):>12.1%}")
    print(f"  {'worst window':<20} {np.min(hold_returns):>11.1%} {np.min(mine):>12.1%}")
    print(f"  {'best window':<20} {np.max(hold_returns):>11.1%} {np.max(mine):>12.1%}")

    # Where it hurts: windows the market rose hardest are where a capped rule
    # most often falls behind, and that is worth seeing rather than averaging.
    rising = hold_returns > np.median(hold_returns)
    print(
        f"\n  in the {int(rising.sum())} strongest windows it beat return "
        f"{np.mean(mine[rising] > hold_returns[rising]):.0%} of the time"
    )
    print(
        f"  in the {int((~rising).sum())} weakest windows it beat return "
        f"{np.mean(mine[~rising] > hold_returns[~rising]):.0%} of the time"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Consistency of volatility targeting.")
    parser.add_argument("--years", type=float, default=3.0, help="Window length.")
    parser.add_argument("--target", type=float, default=0.20)
    parser.add_argument("--cap", type=float, default=1.5)
    parser.add_argument("--min-bars", type=int, default=900)
    parser.add_argument("--etfs-only", action="store_true", help="Restrict to ETFs.")
    parser.add_argument(
        "--adaptive",
        action="store_true",
        help="Target each instrument's own past median volatility instead of a fixed number.",
    )
    args = parser.parse_args()
    asyncio.run(
        _run(
            years=args.years,
            target=args.target,
            cap=args.cap,
            min_bars=args.min_bars,
            kind=InstrumentKind.ETF if args.etfs_only else None,
            adaptive=args.adaptive,
        )
    )


if __name__ == "__main__":
    main()
