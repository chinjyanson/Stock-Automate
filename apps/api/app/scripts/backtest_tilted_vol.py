"""Volatility sizing, tilted by a next-day direction model (§9).

    python -m app.scripts.backtest_tilted_vol --symbol SPY --capital 5000

Volatility targeting sets *how much* to hold and never has an opinion on
direction. This adds one: a logistic regression over price and volume features
predicting whether tomorrow closes up, used to tilt the position the volatility
rule already chose.

    exposure = (target / volatility) x (1 + strength x (P - 0.5) x 2)

so P = 0.5 leaves the volatility rule untouched, and confidence either way
scales it up or down. The volatility rule remains the base because it is the
part that is known to work; the tilt can only modulate it, never replace it.

**Why a one-day horizon changes the arithmetic.** Every previous directional
model here predicted 20 days ahead, which on non-overlapping windows leaves 187
independent observations to fit on — the binding constraint on everything
attempted so far. One-day returns do not overlap at all, so the same fifteen
years yields roughly 3,750. Twenty times the data.

The cost is that one-day returns are far noisier than one-month returns, so a
tiny edge has to survive far more turnover. Whether the extra sample buys more
than the extra noise costs is exactly what this measures, and step 1 answers it
before the strategy is run.

Fitted on the early slice, traded on the later one, which the fit never saw.
"""

from __future__ import annotations

import argparse
import asyncio

import numpy as np
from sqlalchemy import select

from app.backtest.features import compute
from app.data.store import CandleStore
from app.db import session_scope
from app.indicators.series import PriceSeries, candles_to_series
from app.models.enums import Interval
from app.models.instrument import MarketDataMapping
from app.models_ml.logistic import FittedModel, Prior, auc, fit

TRADING_DAYS = 252
VOL_WINDOW = 20
WARMUP = 300
BAND = 0.05
BORROW = 0.05

#: Price and volume features for the one-day question. `rsi_2` leads on purpose:
#: it is built for short-horizon reversal, where `rsi_14` is not. Volume enters
#: as a spike and a trend because turnover level and turnover *change* are
#: different facts, and the ranking found the level far more informative.
FEATURES = (
    "rsi_2",
    "percent_b",
    "discount_sma20",
    "sma50_over_sma200",
    "discount_sma200",
    "sma200_slope",
    "volume_spike",
    "volume_trend",
    "overnight_gap",
    "down_streak",
    "atr_pct",
)


def _annualised_vol(daily: np.ndarray) -> np.ndarray:
    out = np.full(daily.size, np.nan)
    for i in range(VOL_WINDOW, daily.size):
        out[i] = float(np.std(daily[i - VOL_WINDOW : i], ddof=1)) * np.sqrt(TRADING_DAYS)
    return out


def _drawdown(curve: np.ndarray) -> float:
    peak = np.maximum.accumulate(curve)
    return float(np.max((peak - curve) / peak)) if curve.size else 0.0


def _matrix(columns: dict[str, np.ndarray], rows: range | np.ndarray) -> np.ndarray:
    return np.column_stack(
        [[float(columns[n][i]) if n in columns else np.nan for n in FEATURES] for i in rows]
    ).T


def _fit_direction(
    series: PriceSeries, daily: np.ndarray, cut: int
) -> tuple[FittedModel | None, dict[str, np.ndarray]]:
    """Fit next-day up/down on bars [WARMUP, cut)."""
    columns = compute(series.open, series.high, series.low, series.close, series.volume)
    if not columns:
        return None, {}

    rows = np.arange(WARMUP, cut - 1)
    usable = [i for i in rows if np.isfinite(daily[i + 1])]
    if len(usable) < 400:
        return None, columns

    x = _matrix(columns, np.array(usable))
    y = np.array([1.0 if daily[i + 1] > 0 else 0.0 for i in usable])
    model = fit(
        x,
        y,
        FEATURES,
        priors={n: Prior(0.0, 1.0) for n in FEATURES},
        label_definition="1 if the next day closes up",
    )
    return model, columns


def _simulate(
    daily: np.ndarray,
    vol: np.ndarray,
    columns: dict[str, np.ndarray],
    model: FittedModel | None,
    start: int,
    *,
    capital: float,
    target: float,
    max_exposure: float,
    strength: float,
    cost: float,
) -> tuple[np.ndarray, int, float]:
    equity = capital
    exposure = 0.0
    curve: list[float] = []
    trades = 0
    exposures: list[float] = []

    for i in range(start, daily.size):
        # Everything is read at i - 1 and applied to day i's return: the
        # position for tomorrow is chosen with today's information only.
        prior = i - 1
        base = (
            min(target / vol[prior], max_exposure)
            if np.isfinite(vol[prior]) and vol[prior] > 0
            else exposure
        )
        if model is not None and strength > 0:
            reading = {
                n: float(columns[n][prior])
                for n in FEATURES
                if n in columns and np.isfinite(columns[n][prior])
            }
            probability = model.probability(reading) if reading else 0.5
            base *= 1.0 + strength * (probability - 0.5) * 2.0
        wanted = float(np.clip(base, 0.0, max_exposure))

        if abs(wanted - exposure) > BAND:
            equity -= equity * abs(wanted - exposure) * cost / 2.0
            exposure = wanted
            trades += 1

        equity *= 1.0 + exposure * daily[i]
        if exposure > 1.0:
            equity -= equity * (exposure - 1.0) * BORROW / TRADING_DAYS
        curve.append(equity)
        exposures.append(exposure)

    return np.asarray(curve), trades, float(np.mean(exposures)) if exposures else 0.0


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
    return candles_to_series(candles) if len(candles) >= 900 else None


async def _run(
    symbol: str, capital: float, split: float, target: float, max_exposure: float, cost: float
) -> None:
    series = await _load(symbol)
    if series is None:
        print(f"{symbol}: no usable history.")
        return

    close = series.close
    daily = np.concatenate([[np.nan], close[1:] / close[:-1] - 1.0])
    vol = _annualised_vol(daily)
    cut = int(series.length * split)

    model, columns = _fit_direction(series, daily, cut)
    if model is None:
        print("Not enough history before the split to fit on.")
        return

    # --- 1. Is next-day direction predictable at all? ---------------------
    held = [i for i in range(cut, series.length - 1) if np.isfinite(daily[i + 1])]
    x = _matrix(columns, np.array(held))
    y = np.array([1.0 if daily[i + 1] > 0 else 0.0 for i in held])
    scored = np.array(
        [
            model.probability(
                {n: float(v) for n, v in zip(FEATURES, row, strict=True) if np.isfinite(v)}
            )
            for row in x
        ]
    )
    out_auc = auc(y, scored)
    hit = float(np.mean((scored >= 0.5) == (y == 1.0)))

    print(f"Instrument:  {symbol}")
    print(f"Fitted on:   bars {WARMUP}-{cut}  ({model.n_observations:,} days)")
    print(f"Traded on:   bars {cut}-{series.length}  ({len(held):,} days, never seen)\n")

    print("1. CAN IT PREDICT TOMORROW?")
    print(f"     out-of-sample AUC     {out_auc:.4f}   (0.50 is a coin flip)")
    print(f"     directional hit rate  {hit:.2%}")
    print(f"     base rate (up days)   {y.mean():.2%}")
    print(f"\n     {'feature':<20} {'coef':>9}")
    for name, coefficient in zip(model.feature_names, model.coefficients, strict=True):
        star = "  *" if abs(coefficient) > 2 * model.standard_errors[FEATURES.index(name)] else ""
        print(f"     {name:<20} {coefficient:>+9.4f}{star}")

    # --- 2. Does tilting the volatility rule help? ------------------------
    hold_curve = capital * np.cumprod(1.0 + daily[cut:])
    print("\n2. DOES IT HELP THE STRATEGY?")
    print(f"     {'rule':<34} {'final':>10} {'drawdown':>10} {'ret/dd':>8} {'trades':>8}")
    print(
        f"     {'buy and hold':<34} {hold_curve[-1]:>10,.0f} "
        f"{_drawdown(hold_curve):>9.1%} "
        f"{(hold_curve[-1] / capital - 1) / _drawdown(hold_curve):>8.2f} {0:>8}"
    )
    for strength in (0.0, 0.25, 0.5, 1.0):
        curve, trades, _average = _simulate(
            daily,
            vol,
            columns,
            model,
            cut,
            capital=capital,
            target=target,
            max_exposure=max_exposure,
            strength=strength,
            cost=cost,
        )
        label = "vol target only" if strength == 0 else f"vol target + tilt x{strength:.2f}"
        final = float(curve[-1])
        drawdown = _drawdown(curve)
        print(
            f"     {label:<34} {final:>10,.0f} {drawdown:>9.1%} "
            f"{(final / capital - 1) / drawdown if drawdown else 0:>8.2f} {trades:>8}"
        )
    print(
        "\n     Strength 0 is the volatility rule untouched, and every other row\n"
        "     must beat it to have earned its place. Beating buy-and-hold is not\n"
        "     the test here — the volatility rule already does that — the test is\n"
        "     whether the direction model adds anything on top of it."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Volatility sizing tilted by a direction model.")
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--capital", type=float, default=5000.0)
    parser.add_argument("--split", type=float, default=0.6)
    parser.add_argument("--target", type=float, default=0.20)
    parser.add_argument("--max-exposure", type=float, default=1.5)
    parser.add_argument("--cost", type=float, default=0.0005)
    args = parser.parse_args()
    asyncio.run(
        _run(
            symbol=args.symbol,
            capital=args.capital,
            split=args.split,
            target=args.target,
            max_exposure=args.max_exposure,
            cost=args.cost,
        )
    )


if __name__ == "__main__":
    main()
