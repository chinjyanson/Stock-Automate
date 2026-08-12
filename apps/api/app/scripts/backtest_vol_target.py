"""Scale exposure by volatility instead of switching in and out (§9).

    python -m app.scripts.backtest_vol_target --symbol SPY --capital 5000

Timing direction failed for a measurable reason: 20-day direction needs about
65% accuracy to beat buy-and-hold and is predictable at roughly 55%, and the
best and worst days sit within a day or two of each other, so being out for the
crash means being out for the rebound.

This attacks the problem from the side that is *not* a coin flip. Volatility
clusters — calm follows calm and turmoil follows turmoil — so it can be
forecast far better than direction can. The rule that follows is:

    exposure = target volatility / recent volatility     (capped)

so a calm market is held fully and a violent one is held in part. Exposure is
never zero, which is the point: the rebound is not missed, only held smaller.

**Part 1 tests the premise before relying on it.** If volatility were no more
forecastable than direction in this data, the whole idea would be unfounded and
worth abandoning rather than implementing.

**On what to expect.** Volatility targeting is not a way to make more money. Its
claim is a better ride for the return — less drawdown per unit of gain — and
with exposure capped at 1.0 it will usually *lag* buy-and-hold outright, because
it spends the calm periods fully invested and the wild ones only partly. Whether
that is a good trade is the reader's call, so this reports return, drawdown and
return-per-unit-of-risk side by side rather than picking a winner.

Everything is point-in-time: volatility on day `i` uses returns up to `i`, and
the position it implies is taken at the next open.
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

#: Window for the realised-volatility estimate. Short enough to react to a
#: regime change within days, long enough not to be noise.
VOL_WINDOW = 20

#: Rebalance only when exposure has drifted this far from target. Without a
#: band the position is adjusted every single day and turnover eats the result.
REBALANCE_BAND = 0.10

#: Annual cost of borrowing, charged on exposure above 1.0. Retail margin is
#: dearer than this; it is set low deliberately so the result is not flattered
#: by an optimistic financing assumption being buried in a constant.
DEFAULT_BORROW_RATE = 0.05


def _annualised_vol(daily: np.ndarray, window: int) -> np.ndarray:
    """Trailing realised volatility, annualised. Point-in-time by construction."""
    out = np.full(daily.size, np.nan)
    for i in range(window, daily.size):
        out[i] = float(np.std(daily[i - window : i], ddof=1)) * np.sqrt(TRADING_DAYS)
    return out


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


def _predictability(daily: np.ndarray) -> None:
    """Is volatility more forecastable than direction? The premise, tested."""
    vol = _annualised_vol(daily, VOL_WINDOW)
    n = daily.size

    # Next 20 days' realised volatility, and next 20 days' return.
    future_vol = np.full(n, np.nan)
    future_ret = np.full(n, np.nan)
    for i in range(VOL_WINDOW, n - VOL_WINDOW):
        future_vol[i] = float(np.std(daily[i : i + VOL_WINDOW], ddof=1)) * np.sqrt(TRADING_DAYS)
        future_ret[i] = float(np.prod(1.0 + daily[i : i + VOL_WINDOW]) - 1.0)

    # Non-overlapping, so neighbours do not share their future.
    step = np.arange(VOL_WINDOW, n - VOL_WINDOW, VOL_WINDOW)
    vol_ic = _spearman(vol[step], future_vol[step])
    dir_ic = _spearman(vol[step], future_ret[step])
    trend_ic = _spearman(
        np.array([float(np.prod(1.0 + daily[i - VOL_WINDOW : i]) - 1.0) for i in step]),
        future_ret[step],
    )

    print("1. IS THE PREMISE TRUE? — what today predicts about the next 20 days")
    print(f"     {len(step)} non-overlapping windows\n")
    print(f"     recent volatility -> next volatility   {vol_ic:>+7.3f}")
    print(f"     recent volatility -> next return       {dir_ic:>+7.3f}")
    print(f"     recent return     -> next return       {trend_ic:>+7.3f}")
    print()
    # Judged by comparison rather than against an absolute threshold. These are
    # rank correlations over the same windows, so they can be read against each
    # other — and for scale, the stock model's strongest price feature scores
    # about 0.09 on its own equivalent.
    if vol_ic > 3 * abs(trend_ic):
        ratio = vol_ic / max(abs(trend_ic), 1e-9)
        print(f"     Volatility predicts itself {ratio:.1f}x better than recent return")
        print("     predicts next return. That asymmetry is the whole premise:")
        print("     forecast the thing that can be forecast, and decline to guess")
        print("     the thing that cannot.\n")
    else:
        print("     Volatility is no more forecastable than direction here, so the")
        print("     premise does not hold and the result below is coincidence")
        print("     rather than mechanism.\n")


def _drawdown(curve: np.ndarray) -> float:
    peak = np.maximum.accumulate(curve)
    return float(np.max((peak - curve) / peak)) if curve.size else 0.0


def _simulate(
    daily: np.ndarray,
    vol: np.ndarray,
    *,
    capital: float,
    target_vol: float,
    max_exposure: float,
    cost_pct: float,
    borrow_rate: float,
) -> tuple[np.ndarray, float, float]:
    """Hold `target/recent` of the index, rebalanced through a band.

    Returns (equity curve, total turnover, average exposure).
    """
    equity = capital
    exposure = 0.0
    curve: list[float] = []
    turnover = 0.0
    exposures: list[float] = []

    for i in range(VOL_WINDOW + 1, daily.size):
        if np.isfinite(vol[i]) and vol[i] > 0:
            wanted = min(target_vol / vol[i], max_exposure)
        else:
            wanted = exposure
        # Only trade when the drift is worth paying the spread for.
        if abs(wanted - exposure) > REBALANCE_BAND:
            turnover += abs(wanted - exposure)
            equity -= equity * abs(wanted - exposure) * cost_pct / 2.0
            exposure = wanted

        equity *= 1.0 + exposure * daily[i]
        # Exposure above 1.0 is borrowed, and borrowed money is not free. Left
        # uncharged, leverage looks like a way to manufacture return from
        # nothing — which is exactly the error that makes a levered backtest
        # attractive and a levered account disappointing.
        if exposure > 1.0:
            equity -= equity * (exposure - 1.0) * borrow_rate / TRADING_DAYS
        curve.append(equity)
        exposures.append(exposure)

    return np.asarray(curve), turnover, float(np.mean(exposures)) if exposures else 0.0


def _simulate_regime(
    daily: np.ndarray,
    vol: np.ndarray,
    *,
    capital: float,
    threshold: float,
    rough_exposure: float,
    cost_pct: float,
    borrow_rate: float,
    hysteresis: float = 0.15,
) -> tuple[np.ndarray, int, float]:
    """Fully invested while calm, de-risked while rough. A stepped rule.

    The continuous version scales exposure every day, which means paying a
    little financing and a little turnover even in the quiet stretches where
    buy-and-hold was already the better answer. This does nothing at all until
    volatility crosses a line, then de-risks — so calm markets are held whole.

    **Hysteresis is not optional here.** A single threshold with volatility
    sitting on top of it flips the position every few days and pays a spread
    each time. Rough is entered at `threshold x 1.15` and left at
    `threshold / 1.15`, so the rule has to mean it.
    """
    equity = capital
    exposure = 1.0
    rough = False
    curve: list[float] = []
    switches = 0
    exposures: list[float] = []

    upper = threshold * (1.0 + hysteresis)
    lower = threshold / (1.0 + hysteresis)

    for i in range(VOL_WINDOW + 1, daily.size):
        current = vol[i]
        if np.isfinite(current):
            if not rough and current > upper:
                rough = True
            elif rough and current < lower:
                rough = False
        wanted = rough_exposure if rough else 1.0

        if abs(wanted - exposure) > 1e-9:
            equity -= equity * abs(wanted - exposure) * cost_pct / 2.0
            exposure = wanted
            switches += 1

        equity *= 1.0 + exposure * daily[i]
        if exposure > 1.0:
            equity -= equity * (exposure - 1.0) * borrow_rate / TRADING_DAYS
        curve.append(equity)
        exposures.append(exposure)

    return np.asarray(curve), switches, float(np.mean(exposures)) if exposures else 0.0


def _regime_table(
    title: str,
    daily: np.ndarray,
    vol: np.ndarray,
    capital: float,
    thresholds: list[float],
    cost: float,
    borrow_rate: float,
) -> None:
    if daily.size < VOL_WINDOW + 50:
        return
    hold_curve = capital * np.cumprod(1.0 + daily[VOL_WINDOW + 1 :])
    hold_final = float(hold_curve[-1])
    hold_dd = _drawdown(hold_curve)

    print(f"\n{title}")
    print(
        f"     {'rule':<28} {'final':>10} {'return':>9} {'drawdown':>10} {'ret/dd':>8} "
        f"{'switches':>9}"
    )
    print(
        f"     {'buy and hold':<28} {hold_final:>10,.0f} {hold_final / capital - 1:>8.1%} "
        f"{hold_dd:>9.1%} {(hold_final / capital - 1) / hold_dd if hold_dd else 0:>8.2f} "
        f"{0:>9}"
    )
    for threshold in thresholds:
        for rough in (0.5, 0.0):
            curve, switches, _average = _simulate_regime(
                daily,
                vol,
                capital=capital,
                threshold=threshold,
                rough_exposure=rough,
                cost_pct=cost,
                borrow_rate=borrow_rate,
            )
            if curve.size == 0:
                continue
            final = float(curve[-1])
            drawdown = _drawdown(curve)
            label = f"calm 100%, rough {rough:.0%} @ {threshold:.0%}"
            print(
                f"     {label:<28} {final:>10,.0f} {final / capital - 1:>8.1%} "
                f"{drawdown:>9.1%} "
                f"{(final / capital - 1) / drawdown if drawdown else 0:>8.2f} {switches:>9}"
            )


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


async def _run(
    symbol: str,
    capital: float,
    targets: list[float],
    max_exposure: float,
    cost: float,
    borrow_rate: float,
) -> None:
    close = await _load(symbol)
    if close is None:
        print(f"{symbol}: no usable history.")
        return

    daily = close[1:] / close[:-1] - 1.0
    vol = _annualised_vol(daily, VOL_WINDOW)

    print(f"Instrument:  {symbol}   {daily.size:,} trading days")
    print(f"Realised volatility: median {np.nanmedian(vol):.1%} annualised\n")

    _predictability(daily)

    _table(
        "2. THE RESULT — whole period",
        daily,
        vol,
        capital,
        targets,
        max_exposure,
        cost,
        borrow_rate,
    )
    regime_thresholds = [0.15, 0.20, 0.25]
    _regime_table(
        "3. REGIME SWITCH — buy and hold while calm, de-risk while rough",
        daily,
        vol,
        capital,
        regime_thresholds,
        cost,
        borrow_rate,
    )

    half = daily.size // 2
    _table(
        "4. FIRST HALF (calm bull market)",
        daily[:half],
        vol[:half],
        capital,
        targets,
        max_exposure,
        cost,
        borrow_rate,
    )
    _regime_table(
        "6. REGIME SWITCH, FIRST HALF",
        daily[:half],
        vol[:half],
        capital,
        regime_thresholds,
        cost,
        borrow_rate,
    )
    _regime_table(
        "7. REGIME SWITCH, SECOND HALF",
        daily[half:],
        vol[half:],
        capital,
        regime_thresholds,
        cost,
        borrow_rate,
    )
    _table(
        "5. SECOND HALF (2020 and 2022)",
        daily[half:],
        vol[half:],
        capital,
        targets,
        max_exposure,
        cost,
        borrow_rate,
    )
    print(
        f"\n     Costs {cost:.2%} per unit of turnover, borrowing charged at "
        f"{borrow_rate:.0%}/yr on\n     exposure above 100%, rebalanced through a "
        f"{REBALANCE_BAND:.0%} band, capped at {max_exposure:.0%}.\n"
        "\n     'ret/dd' is return divided by worst drawdown — gain bought per unit\n"
        "     of pain, and the number this strategy exists to improve. The halves\n"
        "     are there because a target that only wins on one of them was chosen\n"
        "     by looking at the answer."
    )


def _table(
    title: str,
    daily: np.ndarray,
    vol: np.ndarray,
    capital: float,
    targets: list[float],
    max_exposure: float,
    cost: float,
    borrow_rate: float,
) -> None:
    if daily.size < VOL_WINDOW + 50:
        return
    hold_curve = capital * np.cumprod(1.0 + daily[VOL_WINDOW + 1 :])
    hold_final = float(hold_curve[-1])
    hold_dd = _drawdown(hold_curve)

    print(f"\n{title}")
    print(
        f"     {'strategy':<24} {'final':>10} {'return':>9} {'drawdown':>10} {'ret/dd':>8} "
        f"{'exposure':>9}"
    )
    print(
        f"     {'buy and hold':<24} {hold_final:>10,.0f} "
        f"{hold_final / capital - 1:>8.1%} {hold_dd:>9.1%} "
        f"{(hold_final / capital - 1) / hold_dd if hold_dd else 0:>8.2f} {1.0:>8.0%}"
    )
    for target in targets:
        curve, _turnover, average = _simulate(
            daily,
            vol,
            capital=capital,
            target_vol=target,
            max_exposure=max_exposure,
            cost_pct=cost,
            borrow_rate=borrow_rate,
        )
        if curve.size == 0:
            continue
        final = float(curve[-1])
        drawdown = _drawdown(curve)
        print(
            f"     {'vol target ' + format(target, '.0%'):<24} {final:>10,.0f} "
            f"{final / capital - 1:>8.1%} {drawdown:>9.1%} "
            f"{(final / capital - 1) / drawdown if drawdown else 0:>8.2f} {average:>8.0%}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Volatility-targeted exposure.")
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--capital", type=float, default=5000.0)
    parser.add_argument(
        "--targets",
        type=float,
        nargs="+",
        default=[0.10, 0.12, 0.15, 0.20],
        help="Annualised volatility targets.",
    )
    parser.add_argument(
        "--max-exposure",
        type=float,
        default=1.0,
        help="Cap on position size. 1.0 is unlevered; above 1.0 borrows.",
    )
    parser.add_argument("--cost", type=float, default=0.0005, help="Cost per unit of turnover.")
    parser.add_argument(
        "--borrow",
        type=float,
        default=DEFAULT_BORROW_RATE,
        help="Annual financing rate charged on exposure above 100%%.",
    )
    args = parser.parse_args()
    asyncio.run(
        _run(
            symbol=args.symbol,
            capital=args.capital,
            targets=args.targets,
            max_exposure=args.max_exposure,
            cost=args.cost,
            borrow_rate=args.borrow,
        )
    )


if __name__ == "__main__":
    main()
