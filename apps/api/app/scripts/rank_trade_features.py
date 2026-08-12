"""Which indicators predict a *profitable trade*? (§8)

    python -m app.scripts.rank_trade_features --size 200

`feature_importance` ranks indicators against a forward return. This ranks them
against the thing the strategy is actually a bet on: whether a replayed trade,
opened under the real execution rules and closed at the real stop or target,
made money. Those are not the same question, and the second is the one that
decides what belongs in the model.

Two numbers per indicator, and the second matters more.

**Standalone** is how well the indicator separates winners from losers on its
own, as an AUC — 0.50 is a coin flip. **Incremental** is what it adds *on top of
the features already chosen*, measured by fitting with and without it and taking
the difference in confirm-fold AUC. An indicator can look excellent standalone
and add nothing, because it restates something already in the model; that is the
normal case rather than the exception, and standalone ranking alone would fill
the model with four ways of saying "oversold".

Everything is measured on the confirm fold, split by instrument, so an indicator
has to work on stocks the fit never saw.
"""

from __future__ import annotations

import argparse
import asyncio

import numpy as np

from app.backtest.engine import ExitReason, is_continuous, replay
from app.backtest.entries import EveryBarReader
from app.backtest.features import compute
from app.backtest.service import BacktestService
from app.data.store import CandleStore
from app.db import session_scope
from app.indicators.series import candles_to_series
from app.models.enums import Interval
from app.models_ml.logistic import Prior, auc, fit

#: The four the model ships with today, kept as the incremental baseline.
INCUMBENTS = ("discount_sma200", "rsi_14", "sma200_slope", "atr_pct")

LABEL = "1 if the replayed trade closed with a positive R multiple"


async def _collect(size: int, history: int) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Every completed trade, its features at the decision bar, and its outcome."""
    async with session_scope() as session:
        service = BacktestService(session)
        store = CandleStore(session)
        instruments = await service.top_ranked_instruments(size)
        if not instruments:
            instruments = await service.instruments_with_history(size)
        reader = EveryBarReader()

        columns: dict[str, list[float]] = {}
        labels: list[float] = []
        folds: list[float] = []

        for instrument in instruments:
            candles = await store.get_candles(
                instrument.id, Interval.D1, limit=history, closed_only=True
            )
            if len(candles) < reader.preferred_bars + 40:
                continue
            series = candles_to_series(candles)
            if not is_continuous(series):
                continue
            result = replay(series, reader)
            if not result.trades:
                continue
            computed = compute(series.open, series.high, series.low, series.close, series.volume)
            if not computed:
                continue
            is_fit = 1.0 if bool(service.split([instrument], fold="fit")) else 0.0

            for trade in result.trades:
                if trade.exit_reason is ExitReason.UNCLOSED:
                    continue
                i = trade.entry_index - 1
                if i < 0:
                    continue
                for name, column in computed.items():
                    if i >= column.size:
                        continue
                    columns.setdefault(name, []).append(float(column[i]))
                labels.append(1.0 if trade.r_multiple > 0 else 0.0)
                folds.append(is_fit)

    n = len(labels)
    usable = {k: np.asarray(v) for k, v in columns.items() if len(v) == n}
    return usable, np.asarray(labels), np.asarray(folds)


def _confirm_auc(
    x: dict[str, np.ndarray], names: tuple[str, ...], y: np.ndarray, fold: np.ndarray
) -> float:
    """Fit on one half of the instruments, score on the other."""
    matrix = np.column_stack([x[n] for n in names])
    is_fit = fold == 1.0
    model = fit(
        matrix[is_fit],
        y[is_fit],
        names,
        priors={n: Prior(0.0, 1.0) for n in names},
        label_definition=LABEL,
    )
    scored = np.array(
        [
            model.probability({n: float(v) for n, v in zip(names, row, strict=True)})
            for row in matrix[~is_fit]
        ]
    )
    return auc(y[~is_fit], scored)


def _forward_select(
    x: dict[str, np.ndarray], y: np.ndarray, fold: np.ndarray, *, rounds: int
) -> None:
    """Greedily add the indicator that helps most, then re-measure and repeat.

    Ranking every candidate against one fixed baseline, as the table above does,
    overstates a group of them: three indicators that each add +0.007 for the
    *same* reason add +0.007 between them, not +0.021. Re-measuring after each
    pick is what prices that in.

    Stops when the best remaining candidate adds nothing, because a set chosen
    by "keep going until the list runs out" is a set fitted to the confirm fold.
    """
    chosen: tuple[str, ...] = INCUMBENTS
    current = _confirm_auc(x, chosen, y, fold)
    print(f"\nForward selection, starting from the current four (AUC {current:.4f})")

    for step in range(1, rounds + 1):
        candidates = [n for n in sorted(x) if n not in chosen]
        if not candidates:
            break
        scored = [(n, _confirm_auc(x, (*chosen, n), y, fold)) for n in candidates]
        best_name, best_auc = max(scored, key=lambda kv: kv[1])
        gain = best_auc - current
        if gain <= 0.0:
            print(f"  round {step}: best candidate ({best_name}) adds {gain:+.4f} — stopping")
            break
        chosen = (*chosen, best_name)
        current = best_auc
        print(f"  round {step}: + {best_name:<20} AUC {current:.4f}  ({gain:+.4f})")

    print(f"\n  final set ({len(chosen)}): {', '.join(chosen)}")
    print(f"  confirm AUC {current:.4f}")


async def _run(size: int, history: int, rounds: int) -> None:
    x, y, fold = await _collect(size, history)
    if y.size < 500:
        print(f"Only {y.size} trades — too few. Raise --size.")
        return

    print(
        f"{y.size:,} completed trades, {int((fold == 1).sum()):,} fit / "
        f"{int((fold == 0).sum()):,} confirm"
    )
    print(f"{y.mean():.1%} of them made money\n")

    base = _confirm_auc(x, INCUMBENTS, y, fold)
    print(f"The four the model ships with today: confirm AUC {base:.4f}\n")

    rows: list[tuple[str, float, float]] = []
    for name in sorted(x):
        standalone = _confirm_auc(x, (name,), y, fold)
        if name in INCUMBENTS:
            incremental = float("nan")
        else:
            incremental = _confirm_auc(x, (*INCUMBENTS, name), y, fold) - base
        rows.append((name, standalone, incremental))

    rows.sort(key=lambda r: (-r[2] if r[2] == r[2] else -99, -r[1]))
    print(f"  {'indicator':<22} {'standalone':>11} {'adds':>9}")
    for name, standalone, incremental in rows:
        mark = "" if name not in INCUMBENTS else "   (already in)"
        added = "      —" if incremental != incremental else f"{incremental:>+9.4f}"
        print(f"  {name:<22} {standalone:>11.4f} {added}{mark}")

    print(
        "\n  'standalone' is that indicator alone; 'adds' is the change in confirm\n"
        "  AUC from putting it alongside the existing four. A high standalone with\n"
        "  a zero 'adds' means it restates something the model already has."
    )

    _forward_select(x, y, fold, rounds=rounds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank indicators against trade outcomes.")
    parser.add_argument("--size", type=int, default=200, help="Instruments to sample.")
    parser.add_argument("--history", type=int, default=1200, help="Bars per instrument.")
    parser.add_argument("--rounds", type=int, default=8, help="Max forward-selection rounds.")
    args = parser.parse_args()
    asyncio.run(_run(size=args.size, history=args.history, rounds=args.rounds))


if __name__ == "__main__":
    main()
