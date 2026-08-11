"""Fit the index exposure model (§9).

    python -m app.scripts.fit_index_model --symbol SPY
    python -m app.scripts.fit_index_model --symbol SPY --save

**Label:** 1 if the forward 20-day return is positive. The index decision is
exposure rather than a discrete trade — there is no target and no stop to reach
— so "did the market go up" is the question the position is a bet on.

**Non-overlapping windows only.** A 20-day forward return read at every bar
gives windows that share 95% of their future, so a thousand rows carry perhaps
fifty bars of independent information. Ordinary error bars would then be about
three times too narrow and everything would look significant. Stepping by the
horizon costs sample size — thirty years of daily bars gives ~375 independent
windows, not 7,500 — and that is the honest number.

**Some features cannot be backfilled at all**, and this is the reason the fitter
takes priors. An option chain exists for today only; per-strike open interest is
gone once the session passes and no provider sells it back. So dealer gamma and
charm have no history, and the choice is between shipping nothing for years and
shipping a coefficient that starts as theory and converges onto evidence. This
does the second, and reports how far the convergence has got.

The priors below are **market-structure folklore with error bars, not
findings**, and the script says so in the model's stored notes. What justifies
using them is not their accuracy but the clamp: a coefficient still dominated by
its prior can contribute at most `LOW_SHRINKAGE_CLAMP` log-odds in total, which
at P=0.5 is under nine percentage points.
"""

from __future__ import annotations

import argparse
import asyncio

import numpy as np
from sqlalchemy import select

from app.backtest.features import compute as compute_features
from app.backtest.features import forward_returns
from app.data.store import CandleStore
from app.db import session_scope
from app.indicators.series import candles_to_series
from app.models.enums import Interval, StrategyKind
from app.models.instrument import MarketDataMapping
from app.models_ml.logistic import SHRINKAGE_THRESHOLD, FittedModel, Prior, auc, fit
from app.services.strategy_model import StrategyModelService
from app.strategies.logistic_index import OPTION_FEATURES, PRICE_FEATURES

HORIZON_DAYS = 20

LABEL_DEFINITION = "1 if the forward 20-day return on the index is positive"

#: Ceiling on the summed log-odds from features still carried by their priors.
#: At P = 0.5 this is +/- 8.7 percentage points — enough for a genuine signal to
#: tilt an exposure decision, never enough for a wrong assumption to make it.
LOW_SHRINKAGE_CLAMP = 0.35

#: Priors, in log-odds per standard deviation of a z-scored feature.
#:
#: `gamma_tilt`: dealer gamma positioning. Long gamma means dealers sell rallies
#: and buy dips to stay hedged, which suppresses volatility; short gamma means
#: their hedging amplifies moves and the downside tail fattens. The direction is
#: well established in market-structure work and the magnitude is not, so the
#: mean is small and tau is large enough that the prior does not exclude zero or
#: a sign flip.
#:
#: `charm_tilt`: charm flows into expiry are a weaker and much less replicated
#: story than the gamma one. Centred nearly at zero — it encodes "probably
#: slightly positive, quite possibly nothing".
#:
#: Everything else is uninformative and behaves as light ridge.
PRIORS = {
    "gamma_tilt": Prior(mean=0.15, tau=0.35),
    "charm_tilt": Prior(mean=0.05, tau=0.35),
}


def _report(model: FittedModel, names: tuple[str, ...]) -> None:
    print(f"\n  {'feature':<20} {'coef':>8} {'+/- 2se':>9} {'prior':>8} {'shrinkage':>10}  source")
    for i, name in enumerate(names):
        shrinkage = model.shrinkage[i]
        if not model.scale_known[i]:
            source = "PRIOR (no scale yet — not served)"
        elif shrinkage < SHRINKAGE_THRESHOLD:
            source = f"mostly prior — clamped to +/-{LOW_SHRINKAGE_CLAMP}"
        else:
            source = "data"
        print(
            f"  {name:<20} {model.coefficients[i]:>+8.4f} "
            f"{2 * model.standard_errors[i]:>9.4f} {model.prior_means[i]:>+8.2f} "
            f"{shrinkage:>10.2f}  {source}"
        )
    print(f"  {'(intercept)':<20} {model.intercept:>+8.4f}")


async def _run(symbol: str, save: bool) -> None:
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
            print(f"No market-data mapping for {symbol}. Ingest it first.")
            return
        candles = await CandleStore(session).get_candles(
            mapping.instrument_id, Interval.D1, limit=10_000, closed_only=True
        )
        if len(candles) < 600:
            print(f"{symbol}: only {len(candles)} bars — too few.")
            return

        series = candles_to_series(candles)
        columns = compute_features(
            series.open, series.high, series.low, series.close, series.volume
        )
        forward = forward_returns(series.close, HORIZON_DAYS)

        names = PRICE_FEATURES + OPTION_FEATURES
        # Stepping by the horizon so no two rows share any of the future they
        # measure. See the module docstring — this is the difference between 375
        # observations and a spurious 7,500.
        first = 260
        last = series.length - HORIZON_DAYS - 1
        indices = list(range(first, last, HORIZON_DAYS))

        rows: list[list[float]] = []
        labels: list[float] = []
        for i in indices:
            if not np.isfinite(forward[i]):
                continue
            row = []
            usable = False
            for name in names:
                column = columns.get(name)
                if column is None or i >= column.size or not np.isfinite(column[i]):
                    # Option features are entirely absent from history and land
                    # here on every row, which is exactly what leaves their
                    # coefficients at the prior and their scale unknown.
                    row.append(np.nan)
                    continue
                row.append(float(column[i]))
                usable = True
            if usable:
                rows.append(row)
                labels.append(1.0 if forward[i] > 0 else 0.0)

        if len(rows) < 60:
            print(f"Only {len(rows)} independent windows — too few to fit.")
            return

        x = np.asarray(rows, dtype=np.float64)
        y = np.asarray(labels, dtype=np.float64)

        # Split by date, not by instrument: there is only one instrument, so the
        # halves have to be early and late.
        cut = len(y) // 2
        x_fit, y_fit = x[:cut], y[:cut]
        x_confirm, y_confirm = x[cut:], y[cut:]

        print(f"Instrument: {symbol}")
        print(f"History:    {len(candles):,} bars")
        print(f"Sampling:   {len(rows)} NON-OVERLAPPING {HORIZON_DAYS}-day windows")
        print(f"Folds:      {len(y_fit)} early / {len(y_confirm)} late, split by date")
        print(f"Base rate:  {y.mean():.1%} of windows were positive")
        print(
            f"Clamp:      features under {SHRINKAGE_THRESHOLD:.0%} shrinkage contribute at "
            f"most +/-{LOW_SHRINKAGE_CLAMP} log-odds"
        )

        model = fit(
            x_fit,
            y_fit,
            names,
            priors=PRIORS,
            label_definition=LABEL_DEFINITION,
            low_shrinkage_clamp=LOW_SHRINKAGE_CLAMP,
        )
        print("\nFIT fold (early half)")
        _report(model, names)

        confirm_probability = np.array(
            [
                model.probability(
                    {n: float(v) for n, v in zip(names, row, strict=True) if np.isfinite(v)}
                )
                for row in x_confirm
            ]
        )
        confirm_auc = auc(y_confirm, confirm_probability)
        print(f"\nCONFIRM fold (late half) — {len(y_confirm)} windows the fit never saw")
        print(f"  AUC   {confirm_auc:.4f}   (0.50 is a coin flip)")
        print(f"  base  {y_confirm.mean():.1%} positive")
        if confirm_auc <= 0.55:
            print(
                "\n  At or below 0.55 out of sample this model does not separate up\n"
                "  periods from down ones. Note that the option features contributed\n"
                "  nothing to it — they have no history — so this is a verdict on the\n"
                "  price features alone."
            )

        unfitted = [n for i, n in enumerate(names) if not model.scale_known[i]]
        if unfitted:
            print(
                f"\n  Carried entirely by their priors, and NOT served until their scale\n"
                f"  can be estimated (~60 daily rows): {', '.join(unfitted)}.\n"
                f"  Run the index-options job nightly and refit; `shrinkage` above is\n"
                f"  the number that says whether it is working."
            )

        if save:
            stored = fit(
                x,
                y,
                names,
                priors=PRIORS,
                label_definition=LABEL_DEFINITION,
                low_shrinkage_clamp=LOW_SHRINKAGE_CLAMP,
            )
            row_saved = await StrategyModelService(session).save(
                StrategyKind.LOGISTIC_INDEX,
                stored,
                notes=(
                    f"fit_index_model on {symbol}, {len(rows)} non-overlapping "
                    f"{HORIZON_DAYS}-day windows, confirm AUC {confirm_auc:.4f}. "
                    f"gamma_tilt/charm_tilt priors are market-structure theory, not "
                    f"measurements: they cannot be backfilled and converge only as the "
                    f"nightly options job accumulates rows."
                ),
            )
            print(f"\nSaved and activated model {row_saved.id}")
        else:
            print("\nNot saved. Pass --save to store and activate this model.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit the index exposure model.")
    parser.add_argument("--symbol", default="SPY", help="Provider symbol for the index proxy.")
    parser.add_argument("--save", action="store_true", help="Store and activate the fit.")
    args = parser.parse_args()
    asyncio.run(_run(symbol=args.symbol, save=args.save))


if __name__ == "__main__":
    main()
