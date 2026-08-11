"""Does Kronos predict better than a moving average? (§8)

    python -m app.scripts.measure_kronos --instruments 40 --points 12

The question that decides whether Kronos earns a place: `discount_sma200` scores
an IC of about 0.09 on both folds and costs two lines of arithmetic. Kronos costs
a 300MB dependency, a GPU and a nightly job. It has to beat that, or it is an
expensive way to compute something cheaper.

Measured exactly like any other feature — Spearman IC against the same forward
return, on the same folds — so the numbers are directly comparable with
`feature_importance`.

**The sampling is the constraint.** A full-history sweep would need a forecast at
every bar of every instrument, which at ~0.2s per path is weeks. So this samples
`--points` evaluation dates per instrument, spread across the usable history.
That trades precision for tractability and the error bars say how much.

**Point-in-time is enforced by construction**: each forecast sees `series.head(i
+ 1)` and nothing after it, the same slicing the backtest uses. This is the one
place a Kronos feature could silently leak the future, so it is done in one
place and stated here.
"""

from __future__ import annotations

import argparse
import asyncio
import math

import numpy as np

from app.backtest.engine import is_continuous
from app.backtest.features import compute, forward_returns
from app.backtest.service import BacktestService
from app.data.store import CandleStore
from app.db import session_scope
from app.indicators.series import candles_to_series
from app.models.enums import Interval

#: Compared against, because it is the cheapest feature that works.
BASELINE_FEATURE = "discount_sma200"

#: Bars of context handed to Kronos. Its own limit is 512 for small/base.
CONTEXT_BARS = 512


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 10:
        return float("nan")
    rx, ry = _rank(x), _rank(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denominator = math.sqrt(float((rx * rx).sum()) * float((ry * ry).sum()))
    return float((rx * ry).sum() / denominator) if denominator > 0 else float("nan")


def _rank(values: np.ndarray) -> np.ndarray:
    order = values.argsort()
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    return ranks


def _ic_error(n: int) -> float:
    """Rough standard error of a correlation: 1/sqrt(n - 3)."""
    return 1.0 / math.sqrt(max(n - 3, 1))


async def _run(instruments_wanted: int, points: int, horizon: int, samples: int) -> None:
    from app.signals.kronos_client import KronosClient, is_available, repo_is_present

    if not is_available():
        print("torch is not installed. Run:  uv sync --extra kronos")
        return
    if not repo_is_present():
        print("Kronos source not found. Run:  python -m app.scripts.setup_kronos")
        return

    client = KronosClient()
    print(f"Model:     {client.model_name} on {client.device}")
    print(f"Sampling:  {points} dates x {instruments_wanted} instruments x {samples} paths")
    print(f"Horizon:   {horizon} trading days")
    print(f"Baseline:  {BASELINE_FEATURE}, which costs nothing to compute\n")

    kronos_return: list[float] = []
    kronos_prob: list[float] = []
    kronos_dispersion: list[float] = []
    baseline: list[float] = []
    labels: list[float] = []
    used = 0

    async with session_scope() as session:
        service = BacktestService(session)
        universe = await service.top_ranked_instruments(instruments_wanted * 3)
        if not universe:
            universe = await service.instruments_with_history(instruments_wanted * 3)
        store = CandleStore(session)

        for instrument in universe:
            if used >= instruments_wanted:
                break
            candles = await store.get_candles(
                instrument.id, Interval.D1, limit=2000, closed_only=True
            )
            if len(candles) < CONTEXT_BARS + horizon + 10:
                continue
            series = candles_to_series(candles)
            if not is_continuous(series):
                continue

            features = compute(series.open, series.high, series.low, series.close, series.volume)
            if BASELINE_FEATURE not in features:
                continue
            forward = forward_returns(series.close, horizon)
            timestamps = [c.timestamp for c in candles]

            # Evaluation dates spread across the usable window, rather than
            # clustered — a run of adjacent bars is nearly one observation.
            first = CONTEXT_BARS
            last = series.length - horizon - 1
            if last <= first:
                continue
            indices = np.linspace(first, last, num=points, dtype=int)

            per_instrument: list[tuple[float, float, float, float, float]] = []
            for i in indices:
                if not (np.isfinite(forward[i]) and np.isfinite(features[BASELINE_FEATURE][i])):
                    continue
                window = series.head(i + 1)  # nothing after bar i is visible
                forecast = client.forecast(
                    open_=window.open,
                    high=window.high,
                    low=window.low,
                    close=window.close,
                    volume=window.volume,
                    timestamps=timestamps[: i + 1],
                    horizon_days=horizon,
                    sample_count=samples,
                )
                if forecast is None:
                    continue
                per_instrument.append(
                    (
                        forecast.predicted_return,
                        forecast.prob_up,
                        forecast.path_dispersion,
                        float(features[BASELINE_FEATURE][i]),
                        float(forward[i]),
                    )
                )

            if len(per_instrument) < 3:
                continue
            used += 1
            # Demeaned per instrument, for the same reason as the feature
            # ranking: otherwise this measures which stocks are good rather than
            # when to buy them, and the scanner already answers the first.
            block = np.asarray(per_instrument, dtype=np.float64)
            block = block - block.mean(axis=0)
            kronos_return.extend(block[:, 0].tolist())
            kronos_prob.extend(block[:, 1].tolist())
            kronos_dispersion.extend(block[:, 2].tolist())
            baseline.extend(block[:, 3].tolist())
            labels.extend(block[:, 4].tolist())
            print(
                f"  {used:>3}/{instruments_wanted}  {instrument.name[:38]:<38} "
                f"{len(per_instrument)} points"
            )

    y = np.asarray(labels)
    if y.size < 30:
        print(f"\nOnly {y.size} observations — too few to read. Raise --instruments.")
        return

    print(f"\n  {y.size} observations from {used} instruments")
    print(f"  {'feature':<22} {'IC':>8}  {'+/- 2 se':>9}")
    rows = (
        ("kronos_return", np.asarray(kronos_return)),
        ("kronos_prob_up", np.asarray(kronos_prob)),
        ("kronos_dispersion", np.asarray(kronos_dispersion)),
        (f"{BASELINE_FEATURE} (free)", np.asarray(baseline)),
    )
    error = 2 * _ic_error(y.size)
    for name, x in rows:
        ic = _spearman(x, y)
        mark = "  *" if abs(ic) > error else ""
        print(f"  {name:<22} {ic:>+8.4f}  {error:>9.4f}{mark}")

    print(
        "\n  Kronos has to beat the free baseline to justify a 300MB dependency,\n"
        "  a GPU and a nightly job. Marked * means the band excludes zero."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure Kronos against a free feature.")
    parser.add_argument("--instruments", type=int, default=40, help="Instruments to sample.")
    parser.add_argument("--points", type=int, default=12, help="Evaluation dates each.")
    parser.add_argument("--horizon", type=int, default=20, help="Forward-return horizon.")
    parser.add_argument("--samples", type=int, default=4, help="Kronos paths per forecast.")
    args = parser.parse_args()
    asyncio.run(
        _run(
            instruments_wanted=args.instruments,
            points=args.points,
            horizon=args.horizon,
            samples=args.samples,
        )
    )


if __name__ == "__main__":
    main()
