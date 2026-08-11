"""Which features actually predict forward returns? (§8)

    python -m app.scripts.feature_importance --size 600
    python -m app.scripts.feature_importance --size 600 --horizon 20

The groundwork for a model. Before fitting anything, this asks the prior
question: **does each candidate feature carry information about what price does
next, and is that information stable?**

Two numbers per feature per horizon:

  * **IC** — the Spearman rank correlation between the feature and the forward
    return. Rank rather than linear because a feature only has to *order*
    outcomes correctly to be useful; it need not be linearly related to them.
    An IC of 0.03 is respectable in equities, 0.05 is good, 0.10 is suspicious
    and usually means a bug.
  * **Quintile spread** — mean forward return of the top fifth minus the bottom
    fifth. Reported alongside the IC because a feature can have a real
    relationship that is not monotone, which the IC alone hides.

Every feature is measured on **both folds independently**. A feature with a
strong IC on one and nothing on the other has not been shown to work; it has
been shown to fit. That check is the whole reason this exists rather than a
single pooled number.

Reads the candle store only. Writes nothing.
"""

from __future__ import annotations

import argparse
import asyncio
import math

import numpy as np

from app.backtest.engine import is_continuous
from app.backtest.features import MIN_BARS, compute, forward_returns
from app.backtest.service import BacktestService
from app.data.store import CandleStore
from app.db import session_scope
from app.indicators.series import candles_to_series
from app.models.enums import Interval
from app.models.instrument import Instrument

#: Correlations to a whole *sector* need the sector's series, so they are built
#: here rather than in `features.compute`, which sees one instrument at a time.
RELATIVE_FEATURES = ("relative_to_sector_21d",)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Rank correlation, with ties averaged."""
    if x.size < 30:
        return float("nan")
    rx = _rank(x)
    ry = _rank(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denominator = math.sqrt(float((rx * rx).sum()) * float((ry * ry).sum()))
    return float((rx * ry).sum() / denominator) if denominator > 0 else float("nan")


def _rank(values: np.ndarray) -> np.ndarray:
    order = values.argsort()
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    return ranks


def _quintile_spread(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """(bottom fifth mean, top fifth mean, spread)."""
    if x.size < 50:
        return (float("nan"),) * 3
    order = x.argsort()
    fifth = x.size // 5
    bottom = float(y[order[:fifth]].mean())
    top = float(y[order[-fifth:]].mean())
    return bottom, top, top - bottom


async def _collect(
    store: CandleStore,
    instruments: list[Instrument],
    horizon: int,
    *,
    demean: bool = True,
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """Pooled (feature value, forward return) pairs across the universe.

    `demean` subtracts each instrument's own mean from both the feature and the
    label before pooling, and it changes what is being measured entirely.

    Without it, a stock that declined all period contributes hundreds of
    (low reading, negative return) pairs, so the correlation largely reports
    "persistently weak stocks stayed weak" — a statement about *which* stocks,
    not about *when*. That inflates any feature correlated with being a poor
    company, which is most volatility and drawdown measures.

    With it, each instrument is centred on itself, so what survives is the
    within-instrument relationship: given this stock, does a high reading today
    precede a better-than-usual move for *it*. That is the timing question, and
    it is the only one this strategy asks — the scanner has already chosen which
    stocks are worth holding.
    """
    values: dict[str, list[float]] = {}
    labels: dict[str, list[float]] = {}

    for instrument in instruments:
        candles = await store.get_candles(instrument.id, Interval.D1, limit=2000, closed_only=True)
        if len(candles) < MIN_BARS + horizon + 1:
            continue
        series = candles_to_series(candles)
        if not is_continuous(series):
            continue

        features = compute(series.open, series.high, series.low, series.close, series.volume)
        if not features:
            continue
        forward = forward_returns(series.close, horizon)

        for name, arr in features.items():
            usable = np.isfinite(arr) & np.isfinite(forward)
            if usable.sum() < 30:
                continue
            x = arr[usable]
            y = forward[usable]
            if demean:
                x = x - x.mean()
                y = y - y.mean()
            values.setdefault(name, []).extend(x.tolist())
            labels.setdefault(name, []).extend(y.tolist())

    return values, labels


def _report(
    title: str,
    values: dict[str, list[float]],
    labels: dict[str, list[float]],
) -> dict[str, float]:
    print(f"\n  {title}")
    print(
        f"    {'feature':<22} {'n':>8}  {'IC':>7}  {'bottom 5th':>11} {'top 5th':>9} {'spread':>9}"
    )
    ics: dict[str, float] = {}
    rows = []
    for name in values:
        x = np.asarray(values[name], dtype=np.float64)
        y = np.asarray(labels[name], dtype=np.float64)
        ic = _spearman(x, y)
        bottom, top, spread = _quintile_spread(x, y)
        ics[name] = ic
        rows.append((abs(ic) if math.isfinite(ic) else -1.0, name, x.size, ic, bottom, top, spread))
    for _, name, n, ic, bottom, top, spread in sorted(rows, reverse=True):
        print(f"    {name:<22} {n:>8,}  {ic:>+7.4f}  {bottom:>+10.2%} {top:>+8.2%} {spread:>+8.2%}")
    return ics


async def _run(size: int, horizon: int, demean: bool, correlations: bool = False) -> None:
    async with session_scope() as session:
        service = BacktestService(session)
        instruments = await service.top_ranked_instruments(size)
        if not instruments:
            instruments = await service.instruments_with_history(size)
        if not instruments:
            print("No instruments. Ingest candles or run a scan first.")
            return
        store = CandleStore(session)

        print(f"Universe:  {len(instruments)} instruments, {horizon}-day forward return")
        print("IC:        Spearman rank correlation. 0.03 is respectable, 0.10 suspicious.")
        print("Folds:     a feature must work on both, or it has been fitted not found.")
        print(
            "Demeaned:  "
            + (
                "yes — each instrument centred on itself, so this is a TIMING signal"
                if demean
                else "no — pooled raw, so this mixes 'which stock' with 'when'"
            )
        )

        if correlations:
            await _correlations(store, instruments)
            return

        fold_ics: dict[str, dict[str, float]] = {}
        for fold in ("fit", "confirm"):
            half = service.split(instruments, fold=fold)
            values, labels = await _collect(store, half, horizon, demean=demean)
            fold_ics[fold] = _report(
                f"{fold.upper()} fold — {len(half)} instruments", values, labels
            )

        print("\n  stability — same sign and comparable size on both folds?")
        print(f"    {'feature':<22} {'fit IC':>9} {'confirm IC':>12}  verdict")
        shared = set(fold_ics["fit"]) & set(fold_ics["confirm"])
        scored = []
        for name in shared:
            a, b = fold_ics["fit"][name], fold_ics["confirm"][name]
            if not (math.isfinite(a) and math.isfinite(b)):
                continue
            agrees = (a > 0) == (b > 0)
            weaker = min(abs(a), abs(b))
            scored.append((weaker if agrees else -1.0, name, a, b, agrees))
        for weaker, name, a, b, agrees in sorted(scored, reverse=True):
            if not agrees:
                verdict = "flips sign — discard"
            elif weaker >= 0.03:
                verdict = "KEEP"
            elif weaker >= 0.015:
                verdict = "marginal"
            else:
                verdict = "too weak"
            print(f"    {name:<22} {a:>+9.4f} {b:>+12.4f}  {verdict}")


async def _correlations(store: CandleStore, instruments: list[Instrument]) -> None:
    """How much do the surviving features actually overlap?

    Twelve features that all say "buy weakness" are not twelve pieces of
    information. Feeding correlated inputs to a logistic regression splits their
    weight between them, destabilises the coefficients and produces a model far
    more confident than the evidence supports. One representative per axis is
    worth more than all twelve.
    """
    collected: dict[str, list[float]] = {}
    for instrument in instruments:
        candles = await store.get_candles(instrument.id, Interval.D1, limit=2000, closed_only=True)
        if len(candles) < MIN_BARS + 21:
            continue
        series = candles_to_series(candles)
        if not is_continuous(series):
            continue
        features = compute(series.open, series.high, series.low, series.close, series.volume)
        if not features:
            continue
        names = sorted(features)
        stacked = np.vstack([features[n] for n in names])
        # Every feature finite on the same bar, which needs ~252 bars because
        # of the 52-week window — so the threshold is low on purpose. A median
        # instrument holds 281 bars and contributes only a couple of dozen rows.
        usable = np.isfinite(stacked).all(axis=0)
        if usable.sum() < 5:
            continue
        for i, name in enumerate(names):
            row = stacked[i][usable]
            collected.setdefault(name, []).extend((row - row.mean()).tolist())

    names = sorted(collected)
    if not names:
        print("\n  no instrument had every feature readable on the same bar")
        return
    size = min(len(v) for v in collected.values())
    matrix = np.vstack([np.asarray(collected[n][:size]) for n in names])
    ranks = np.vstack([_rank(row) for row in matrix])
    ranks = ranks - ranks.mean(axis=1, keepdims=True)
    norms = np.sqrt((ranks * ranks).sum(axis=1))
    corr = (ranks @ ranks.T) / np.outer(norms, norms)

    print(f"\n  pairwise rank correlation, |r| >= 0.6 (n={size:,})")
    seen = set()
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if j <= i:
                continue
            r = float(corr[i, j])
            if abs(r) >= 0.6:
                seen.add(a)
                seen.add(b)
                print(f"    {a:<22} {b:<22} {r:>+6.2f}")
    independent = [n for n in names if n not in seen]
    if independent:
        print("\n  not strongly correlated with anything:")
        for name in independent:
            print(f"    {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank candidate features by predictive power.")
    parser.add_argument("--size", type=int, default=600, help="Instruments to scan.")
    parser.add_argument("--horizon", type=int, default=20, help="Forward-return horizon in days.")
    parser.add_argument(
        "--correlations",
        action="store_true",
        help="Report pairwise feature correlation instead of predictive power.",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help=(
            "Pool without demeaning. Mixes 'which stock is good' with 'when to buy', "
            "which inflates anything correlated with being a poor company."
        ),
    )
    args = parser.parse_args()
    asyncio.run(
        _run(
            size=args.size,
            horizon=args.horizon,
            demean=not args.raw,
            correlations=args.correlations,
        )
    )


if __name__ == "__main__":
    main()
