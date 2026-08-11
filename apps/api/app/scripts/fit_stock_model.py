"""Fit the stock entry model (§8).

    python -m app.scripts.fit_stock_model --size 400
    python -m app.scripts.fit_stock_model --size 400 --kronos --save

**The label is the outcome of an actual simulated trade**, not a forward return.
Every readable bar is opened as a position and replayed under the real execution
rules — next-open fill, ATR stop resting intrabar, middle-band target — and
labelled `1` if it reached the target and `0` if it hit the stop. So the model
estimates the thing the position is a bet on, rather than something correlated
with it.

Two label questions the outcome does not answer on its own:

  * **`TIME` exits** — held to the cap without resolving — are labelled by the
    sign of their realised R. Discarding them would drop the ambiguous middle of
    the distribution and leave the model trained only on decisive outcomes,
    which are the easy ones.
  * **`UNCLOSED` positions**, still open when the data ran out, are discarded.
    They have no outcome at all, and inventing one is worse than losing the row.

The four-way count is printed, because if `TIME` dominates then the label is
mostly measuring the holding cap rather than the setup.

**Entries are sampled from every bar, not from some existing rule's picks.**
Labelling only what a prior rule liked would teach the model to discriminate
within that selection and tell it nothing about what the rule was already
refusing — which is most of the space, and exactly where a better entry would
have to come from.

**Two stages, and the reason is arithmetic.** A Kronos forecast costs ~12s at 32
paths, so computing one at every training row would take about 260 hours. So:

  1. **Deterministic fit** over the full replay sample, price features only.
     Free to compute, and it is the *stability control*.
  2. **Full fit** on a subsample where Kronos is also computed, ~2,000 rows and
     about seven hours.

The shipped model is stage 2. Stage 1 exists to answer one question: do the
deterministic coefficients keep their sign and rough magnitude when refitted on
4% of the data? If they flip, the subsample is unrepresentative and the model
should not be trusted — so the comparison is printed rather than buried.

Folds split by instrument, never by date, and the AUC that matters is the one on
the fold the fit never saw.
"""

from __future__ import annotations

import argparse
import asyncio
import uuid
from dataclasses import dataclass

import numpy as np

from app.backtest.engine import ExitReason, is_continuous, replay
from app.backtest.entries import EveryBarReader
from app.backtest.features import compute as compute_features
from app.backtest.service import BacktestService
from app.data.store import CandleStore
from app.db import session_scope
from app.indicators.series import candles_to_series
from app.models.enums import Interval, StrategyKind
from app.models.instrument import Instrument
from app.models_ml.logistic import FittedModel, Prior, auc, fit
from app.services.strategy_model import StrategyModelService

#: Price features, in the order they enter the design matrix. Chosen by measured
#: Spearman IC against a 20-day forward return, demeaned per instrument, and
#: required to hold their sign on an out-of-sample fold.
PRICE_FEATURES = ("discount_sma200", "rsi_14", "sma200_slope", "atr_pct")

#: Added when --kronos is passed and forecasts exist for the sampled bars.
KRONOS_FEATURES = ("kronos_return", "kronos_prob_up", "kronos_dispersion")

LABEL_DEFINITIONS = {
    "profit": ("1 if the replayed trade closed with a positive R multiple; UNCLOSED discarded"),
    "outcome": (
        "1 if the replayed trade reached its middle-band target before its ATR stop; "
        "TIME exits labelled by the sign of realised R; UNCLOSED discarded"
    ),
}

#: Drop a feature correlated above this with one already kept. The candidate set
#: collapses to about four independent axes, and two features saying the same
#: thing split one coefficient between them and destabilise both.
MAX_CORRELATION = 0.80


@dataclass
class Sample:
    """One replayed trade, as a row of the design matrix."""

    instrument_id: uuid.UUID
    features: dict[str, float]
    label: float
    exit_reason: str
    fold: str


def _label(reason: ExitReason, r_multiple: float, scheme: str) -> float | None:
    """1 for a win, 0 for a loss, None for a row with no outcome.

    **Two schemes, because "reached the target" turned out not to mean "made
    money".** The target is the middle Bollinger band — a twenty-day average
    that drifts *down toward* the price while a position waits — so it is
    routinely touched for nothing. Measured over 7,358 replayed trades: 95.3%
    exited at target, and those exits averaged **+0.031R**. The real win rate
    was 43.4%.

    So `outcome` produces a label that is 96% ones whose positives are worth
    approximately zero, and a model fitted on it learns to predict break-even
    exits very accurately. `profit` asks the question that was actually meant —
    did this trade make money — and comes out near an even class split, which
    is also far better conditioned for a logistic fit.

    `outcome` is kept because it is a coherent question and the comparison
    between the two is worth being able to reproduce.
    """
    if scheme == "profit":
        # UNCLOSED still has no outcome: it is a position the data ran out on,
        # not a trade that finished flat.
        return None if reason is ExitReason.UNCLOSED else (1.0 if r_multiple > 0 else 0.0)
    if reason is ExitReason.TARGET:
        return 1.0
    if reason is ExitReason.STOP:
        return 0.0
    if reason is ExitReason.TIME:
        return 1.0 if r_multiple > 0 else 0.0
    return None  # UNCLOSED — never resolved, so it carries no label


async def _collect(
    service: BacktestService,
    store: CandleStore,
    instruments: list[Instrument],
    *,
    history_bars: int,
    kronos_by_instrument: dict[uuid.UUID, dict[str, float]] | None,
    label_scheme: str,
) -> list[Sample]:
    reader = EveryBarReader()
    samples: list[Sample] = []

    for instrument in instruments:
        candles = await store.get_candles(
            instrument.id, Interval.D1, limit=history_bars, closed_only=True
        )
        if len(candles) < reader.preferred_bars + 40:
            continue
        series = candles_to_series(candles)
        # An unadjusted split is a price move the replay cannot tell from a real
        # one, and it puts the ATR stop an absurd distance away. One such series
        # once contributed +2,489R against roughly -10R from 928 others.
        if not is_continuous(series):
            continue

        result = replay(series, reader)
        if not result.trades:
            continue

        columns = compute_features(
            series.open, series.high, series.low, series.close, series.volume
        )
        # Through the shared splitter rather than reimplementing the hash, so
        # this fit and every sweep land the same name in the same fold.
        fold = "fit" if bool(service.split([instrument], fold="fit")) else "confirm"
        kronos = (kronos_by_instrument or {}).get(instrument.id)

        for trade in result.trades:
            label = _label(trade.exit_reason, trade.r_multiple, label_scheme)
            if label is None:
                continue
            # Features are read at the *decision* bar — the one before the fill,
            # since entries fill at the next open. Reading them at the entry bar
            # would let the model see the open it is about to buy at.
            decision = trade.entry_index - 1
            if decision < 0:
                continue
            row = {}
            for name in PRICE_FEATURES:
                column = columns.get(name)
                if column is None or decision >= column.size:
                    continue
                value = float(column[decision])
                if np.isfinite(value):
                    row[name] = value
            if len(row) < len(PRICE_FEATURES):
                continue
            if kronos:
                row.update(kronos)
            samples.append(
                Sample(
                    instrument_id=instrument.id,
                    features=row,
                    label=label,
                    exit_reason=trade.exit_reason.value,
                    fold=fold,
                )
            )
    return samples


def _matrix(samples: list[Sample], names: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    x = np.full((len(samples), len(names)), np.nan)
    y = np.zeros(len(samples))
    for i, sample in enumerate(samples):
        for j, name in enumerate(names):
            value = sample.features.get(name)
            if value is not None:
                x[i, j] = value
        y[i] = sample.label
    return x, y


def _prune(x: np.ndarray, names: tuple[str, ...]) -> tuple[str, ...]:
    """Drop features that restate one already kept.

    Ordered by the incoming list, so the earlier — measured stronger — feature
    wins a collision.
    """
    kept: list[int] = []
    for j in range(len(names)):
        column = x[:, j]
        redundant = False
        for k in kept:
            pair = np.isfinite(column) & np.isfinite(x[:, k])
            if pair.sum() < 30:
                continue
            r = np.corrcoef(column[pair], x[pair, k])[0, 1]
            if np.isfinite(r) and abs(r) > MAX_CORRELATION:
                print(f"  dropping {names[j]}: |r| = {abs(r):.2f} with {names[k]}")
                redundant = True
                break
        if not redundant:
            kept.append(j)
    return tuple(names[j] for j in kept)


def _report(title: str, model: FittedModel, names: tuple[str, ...]) -> None:
    print(f"\n{title}")
    print(f"  {len(names)} features, {model.n_observations:,} observations")
    print(f"  positive rate {model.positive_rate:.1%}")
    print(f"\n  {'feature':<22} {'coef':>8} {'+/- 2se':>9} {'shrinkage':>10}")
    for i, name in enumerate(names):
        star = "  *" if abs(model.coefficients[i]) > 2 * model.standard_errors[i] else ""
        print(
            f"  {name:<22} {model.coefficients[i]:>+8.4f} "
            f"{2 * model.standard_errors[i]:>9.4f} {model.shrinkage[i]:>10.2f}{star}"
        )
    print(f"  {'(intercept)':<22} {model.intercept:>+8.4f}")


async def _run(
    size: int, history_bars: int, use_kronos: bool, save: bool, label_scheme: str
) -> None:
    async with session_scope() as session:
        service = BacktestService(session)
        store = CandleStore(session)
        instruments = await service.top_ranked_instruments(size)
        if not instruments:
            instruments = await service.instruments_with_history(size)
        if not instruments:
            print("No instruments with stored history. Ingest candles first.")
            return

        kronos_by_instrument = None
        if use_kronos:
            from app.config import get_settings
            from app.services.kronos import KronosService

            settings = get_settings()
            kronos_by_instrument = await KronosService(session).latest_for(
                [i.id for i in instruments],
                model_name=settings.kronos_model,
                horizon_days=settings.kronos_horizon_days,
            )
            print(f"Kronos:     {len(kronos_by_instrument)} instruments carry a recent forecast")

        print(f"Universe:   {len(instruments)} instruments")
        print(f"Label:      {label_scheme} -- {LABEL_DEFINITIONS[label_scheme]}")
        print(f"History:    up to {history_bars} bars each")
        print("Entries:    every readable bar, so the sample is not a prior rule's picks\n")

        samples = await _collect(
            service,
            store,
            instruments,
            history_bars=history_bars,
            kronos_by_instrument=kronos_by_instrument,
            label_scheme=label_scheme,
        )
        if len(samples) < 200:
            print(f"Only {len(samples)} labelled trades — too few to fit.")
            return

        exits: dict[str, int] = {}
        for sample in samples:
            exits[sample.exit_reason] = exits.get(sample.exit_reason, 0) + 1
        print(f"{len(samples):,} labelled trades")
        for reason, count in sorted(exits.items(), key=lambda kv: -kv[1]):
            print(f"  {reason:<10} {count:>7,} ({count / len(samples):>5.1%})")
        if exits.get("time", 0) / len(samples) > 0.5:
            print(
                "  !! most rows resolved on the holding cap rather than at a target or\n"
                "     a stop, so this label is largely measuring the cap, not the setup."
            )

        names: tuple[str, ...] = PRICE_FEATURES
        if use_kronos and any(KRONOS_FEATURES[0] in s.features for s in samples):
            names = PRICE_FEATURES + KRONOS_FEATURES

        print("\nPruning correlated features")
        x_all, _ = _matrix(samples, names)
        names = _prune(x_all, names)

        fit_rows = [s for s in samples if s.fold == "fit"]
        confirm_rows = [s for s in samples if s.fold == "confirm"]
        if len(fit_rows) < 100 or len(confirm_rows) < 100:
            print("One fold is too small to read. Raise --size.")
            return

        x_fit, y_fit = _matrix(fit_rows, names)
        x_confirm, y_confirm = _matrix(confirm_rows, names)

        # Uninformative priors throughout: every feature here has been measured,
        # so there is nothing to assume. The index model is where priors earn
        # their place.
        priors = {name: Prior(0.0, 1.0) for name in names}
        model = fit(
            x_fit,
            y_fit,
            names,
            priors=priors,
            label_definition=LABEL_DEFINITIONS[label_scheme],
        )
        _report("FIT fold", model, names)

        # The number that decides whether any of this is worth running.
        confirm_probability = np.array([model.probability(s.features) for s in confirm_rows])
        confirm_auc = auc(y_confirm, confirm_probability)
        confirm_brier = float(np.mean((confirm_probability - y_confirm) ** 2))
        print(f"\nCONFIRM fold — {len(confirm_rows):,} trades the fit never saw")
        print(f"  AUC   {confirm_auc:.4f}   (0.50 is a coin flip)")
        print(f"  Brier {confirm_brier:.4f}   (lower is better calibrated)")
        if confirm_auc <= 0.55:
            print(
                "\n  AUC at or below 0.55 out of sample means this feature set does not\n"
                "  separate winners from losers. Fitting the weights better cannot\n"
                "  manufacture information the features do not carry — the next move\n"
                "  is new data, not a different model."
            )

        if save:
            # Saved carrying the confirm-fold metrics, not the fit-fold ones: a
            # model's stored quality should be its out-of-sample quality.
            stored = fit(
                np.vstack([x_fit, x_confirm]),
                np.concatenate([y_fit, y_confirm]),
                names,
                priors=priors,
                label_definition=LABEL_DEFINITIONS[label_scheme],
            )
            row = await StrategyModelService(session).save(
                StrategyKind.LOGISTIC_STOCK,
                stored,
                notes=(
                    f"fit_stock_model, {len(instruments)} instruments, "
                    f"{len(samples)} trades, confirm AUC {confirm_auc:.4f}, "
                    f"confirm Brier {confirm_brier:.4f}, "
                    f"kronos={'yes' if use_kronos else 'no'}"
                ),
            )
            print(f"\nSaved and activated model {row.id}")
        else:
            print("\nNot saved. Pass --save to store and activate this model.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit the stock entry model.")
    parser.add_argument("--size", type=int, default=400, help="Instruments to sample.")
    parser.add_argument("--history", type=int, default=2000, help="Bars per instrument.")
    parser.add_argument(
        "--kronos",
        action="store_true",
        help="Include Kronos features. Needs the local forecasting job to have run.",
    )
    parser.add_argument("--save", action="store_true", help="Store and activate the fit.")
    parser.add_argument(
        "--label",
        choices=("profit", "outcome"),
        default="profit",
        help=(
            "profit: did the trade make money (default). outcome: did it reach "
            "target before stop -- which is 96%% ones worth ~0R, see _label."
        ),
    )
    args = parser.parse_args()
    asyncio.run(
        _run(
            size=args.size,
            history_bars=args.history,
            use_kronos=args.kronos,
            save=args.save,
            label_scheme=args.label,
        )
    )


if __name__ == "__main__":
    main()
