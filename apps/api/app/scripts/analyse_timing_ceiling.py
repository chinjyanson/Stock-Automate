"""Why is buy-and-hold so hard to beat by timing? (§9)

    python -m app.scripts.analyse_timing_ceiling --symbol SPY

"Sell before the fall, buy after the dip" is the obvious thing to want, and the
obvious question is why a model cannot do it. This decomposes that into three
measurements, none of which involve the fitted model — they are properties of
the price series itself, and they bound what *any* timing rule could achieve.

**1. The prize.** A perfect oracle, in when the next stretch rises and out when
it falls. This is the ceiling: if it were small, timing would not be worth
attempting at all. It is not small, which is why the idea is so appealing.

**2. The bar.** The oracle, corrupted — a chosen fraction of its calls flipped
to wrong. Sweeping that fraction finds the accuracy at which timing stops
beating buy-and-hold, and that number is the honest specification for any model:
below it, a timing strategy loses *by construction*, however well engineered.

**3. The reason.** Returns are not evenly spread. A handful of days carry most
of the gain, and — the part that defeats timing — the best days sit unusually
close to the worst ones, because volatility arrives in clusters. Selling before
a fall means being out during the rebound, and the rebound is where the return
is. This measures how tightly the two are packed.

Pure arithmetic on stored candles; no model, no fitting, nothing to overfit.
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

#: The horizon the index model predicts, so the oracle is judged on the same
#: question the model is asked.
HORIZON_DAYS = 20


def _compound(daily: np.ndarray) -> float:
    return float(np.prod(1.0 + daily))


def _oracle_mask(close: np.ndarray, horizon: int, rng: np.random.Generator, accuracy: float):
    """Invested-or-not per day, from a forecaster of the given accuracy.

    The oracle looks `horizon` days ahead and is right `accuracy` of the time;
    the rest of its calls are inverted. Accuracy 1.0 is perfect foresight and
    0.5 is a coin flip — which is what an uninformative model amounts to.
    """
    n = close.size
    future = np.full(n, np.nan)
    future[: n - horizon] = close[horizon:] / close[: n - horizon] - 1.0
    truth = future > 0
    correct = rng.random(n) < accuracy
    return np.where(correct, truth, ~truth)


def _timed_return(daily: np.ndarray, invested: np.ndarray) -> float:
    """Compound only the days the rule was invested; cash earns nothing."""
    return _compound(np.where(invested[: daily.size], daily, 0.0))


async def _load(symbol: str) -> np.ndarray | None:
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
    if len(candles) < 500:
        return None
    return candles_to_series(candles).close


async def _run(symbol: str, capital: float, trials: int) -> None:
    close = await _load(symbol)
    if close is None:
        print(f"{symbol}: no usable history.")
        return

    daily = close[1:] / close[:-1] - 1.0
    n = daily.size
    hold = _compound(daily)
    rng = np.random.default_rng(0)

    print(f"Instrument:  {symbol}   {n:,} trading days")
    print(f"Buy & hold:  £{capital:,.0f} -> £{capital * hold:,.0f}  ({hold - 1:+.1%})\n")

    # -- 1. The prize -------------------------------------------------------
    perfect = _oracle_mask(close, HORIZON_DAYS, rng, 1.0)
    perfect_value = capital * _timed_return(daily, perfect[:-1])
    print("1. THE PRIZE — a perfect oracle on the next 20 days")
    print(f"     £{capital:,.0f} -> £{perfect_value:,.0f}")
    print(f"     {perfect_value / (capital * hold):.1f}x what buy and hold made")
    print("     So the opportunity is real. The difficulty is entirely in the")
    print("     predicting, not in the idea.\n")

    # -- 2. The bar ---------------------------------------------------------
    print("2. THE BAR — how accurate must the forecast be to be worth having?")
    print(f"     {'accuracy':>10} {'median outcome':>16} {'beats buy & hold':>18}")
    break_even: float | None = None
    for accuracy in (0.50, 0.52, 0.55, 0.57, 0.60, 0.65, 0.70, 0.80):
        finals = []
        for _ in range(trials):
            mask = _oracle_mask(close, HORIZON_DAYS, rng, accuracy)
            finals.append(capital * _timed_return(daily, mask[:-1]))
        median = float(np.median(finals))
        beat = float(np.mean(np.array(finals) > capital * hold))
        if break_even is None and beat >= 0.5:
            break_even = accuracy
        print(f"     {accuracy:>9.0%} {median:>15,.0f} {beat:>17.0%}")
    if break_even is not None:
        print(f"\n     Break-even is around {break_even:.0%} accuracy on 20-day direction.")
    print("     The fitted index model scores about 53-57% out of sample, and")
    print("     that estimate itself moves when the data is cut differently.\n")

    # -- 3. The reason ------------------------------------------------------
    order = np.argsort(daily)
    worst = order[:10]
    best = order[-10:]
    without_best = np.delete(daily, best)
    without_worst = np.delete(daily, worst)

    print("3. THE REASON — the return lives in a handful of days")
    print(f"     all {n:,} days                  £{capital * hold:>12,.0f}")
    print(f"     missing the 10 BEST days      £{capital * _compound(without_best):>12,.0f}")
    print(f"     avoiding the 10 WORST days    £{capital * _compound(without_worst):>12,.0f}")

    # How close do the best and worst days sit to each other?
    gaps = [int(np.min(np.abs(best - w))) for w in worst]
    within_five = sum(1 for g in gaps if g <= 5)
    print(f"\n     Of the 10 worst days, {within_five} sit within 5 trading days of one")
    print(f"     of the 10 best. Median gap: {int(np.median(gaps))} trading days.")
    print("     Volatility arrives in clusters, so the violent up days are")
    print("     packed in among the violent down days — they are the same")
    print("     episode. A rule that is out for the crash is usually out for")
    print("     the rebound, and the rebound is where the return is.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bound what any timing rule could achieve.")
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--capital", type=float, default=5000.0)
    parser.add_argument("--trials", type=int, default=200, help="Draws per accuracy level.")
    args = parser.parse_args()
    asyncio.run(_run(symbol=args.symbol, capital=args.capital, trials=args.trials))


if __name__ == "__main__":
    main()
