"""Which way of buying back in after a crash warning actually works? (§9)

    python -m app.scripts.compare_reentry

The detector decides when to step aside. This decides when to come back, and
the previous session established that the second decision moves the money more
than the first: sweeping the buy-back knob alone moved the twenty-year result
from 19,935 to 36,333 pounds, and turned the worst threshold into the best.

That finding is also the reason this script is shaped the way it is.

## The protocol, fixed before any result was looked at

Picking the best of many rules by reading their results is how the last set of
numbers stopped meaning anything. So:

  * **Two windows, split by date.** Rules are ranked on 2006-2015 and then
    re-measured on 2016-2025, with the detector refitted for each window on data
    strictly before it. A rule that wins the first and fails the second has been
    caught, not discovered.
  * **A rule is scored by its *median* across settings, never its best.** Each
    rule is run over four alarm thresholds and its own parameter sweep. The
    median says "how does this idea do if I cannot tune it perfectly", which is
    the only question a future user of the rule can answer in advance. The
    spread is printed beside it, because a wide one means the median is luck too.
  * **Two benchmarks on every row.** Buy and hold, and holding the same average
    exposure flat. The second is what separates timing from simply owning less.
  * **`wait` is in the table.** It does nothing at all until the timeout. Any
    rule that cannot beat it has bought complexity and nothing else.
"""

from __future__ import annotations

import argparse
import statistics
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.backtest import reentry
from app.backtest.overlay_pipeline import Pipeline, context_arrays, load

WINDOWS: tuple[tuple[str, str, str], ...] = (
    ("2006-2015", "2006-01-01", "2016-01-01"),
    ("2016-2025", "2016-01-01", "2026-01-01"),
)


@dataclass(frozen=True, slots=True)
class Run:
    final: float
    drawdown: float
    flat_final: float
    exposure: float


def _drawdown(curve: np.ndarray) -> float:
    peak = np.maximum.accumulate(curve)
    return float(np.max((peak - curve) / peak)) if curve.size else 0.0


def simulate(
    pipe: Pipeline,
    arrays: dict[str, np.ndarray],
    triggers: np.ndarray,
    rule: reentry.Rule,
    *,
    capital: float,
    defensive: float,
    timeout: int,
    cost: float,
) -> Run:
    """Invested by default; step aside on a warning; let `rule` decide the return.

    Every decision on bar *i* reads bar *i - 1* and is paid bar *i*'s return, so
    no rule can act on a price it could not have seen.
    """
    equity = capital
    exposure = 1.0
    curve: list[float] = []
    held: list[float] = []
    exit_price: float | None = None
    low_since = float("inf")
    vol_at_exit = float("nan")
    days_out = 0

    for i in range(pipe.cut, pipe.daily.size):
        prior = i - 1
        probability = pipe.probability[prior]
        warning = bool(np.isfinite(probability) and probability >= triggers[prior])
        here = float(pipe.close[prior])

        if exit_price is None:
            if warning:
                exit_price = here
                low_since = here
                vol_at_exit = float(arrays["volatility"][prior])
                days_out = 0
                wanted = defensive
            else:
                wanted = 1.0
        else:
            days_out += 1
            low_since = min(low_since, here)
            context = reentry.Context(
                days_out=days_out,
                price=here,
                exit_price=exit_price,
                low_since=low_since,
                warning=warning,
                rsi=float(arrays["rsi"][prior]),
                sma_ratio=float(arrays["sma_ratio"][prior]),
                volatility=float(arrays["volatility"][prior]),
                volatility_at_exit=vol_at_exit,
                up_streak=int(arrays["up_streak"][prior]),
            )
            back = rule(context)
            back = 0.0 if not np.isfinite(back) else back
            wanted = defensive + (1.0 - defensive) * back
            if back >= 1.0 or days_out >= timeout:
                exit_price = None
                wanted = 1.0

        if abs(wanted - exposure) > 0.02:
            equity -= equity * abs(wanted - exposure) * cost / 2.0
            exposure = wanted

        step = pipe.daily[i]
        equity *= 1.0 + exposure * (step if np.isfinite(step) else 0.0)
        curve.append(equity)
        held.append(exposure)

    average = float(np.mean(held)) if held else 0.0
    flat = capital * np.cumprod(1.0 + average * np.nan_to_num(pipe.daily[pipe.cut :]))
    return Run(
        final=float(curve[-1]),
        drawdown=_drawdown(np.asarray(curve)),
        flat_final=float(flat[-1]),
        exposure=average,
    )


def _hold(pipe: Pipeline, capital: float) -> tuple[float, float]:
    curve = capital * np.cumprod(1.0 + np.nan_to_num(pipe.daily[pipe.cut :]))
    return float(curve[-1]), _drawdown(curve)


def _measure(
    pipe: Pipeline,
    arrays: dict[str, np.ndarray],
    fractions: tuple[float, ...],
    *,
    capital: float,
    defensive: float,
    timeout: int,
    cost: float,
) -> dict[str, list[Run]]:
    """Every rule, over every alarm threshold and every parameter it takes."""
    cache = {f: pipe.triggers(f) for f in fractions}
    out: dict[str, list[Run]] = {}
    for name, (factory, params) in reentry.REGISTRY.items():
        runs: list[Run] = []
        for param in params:
            rule = factory(param)
            for fraction in fractions:
                runs.append(
                    simulate(
                        pipe,
                        arrays,
                        cache[fraction],
                        rule,
                        capital=capital,
                        defensive=defensive,
                        timeout=timeout,
                        cost=cost,
                    )
                )
        out[name] = runs
    return out


def _report(
    label: str,
    results: dict[str, list[Run]],
    hold_final: float,
    hold_dd: float,
    order: list[str] | None,
) -> list[str]:
    print(f"\n  {label}   buy and hold {hold_final:,.0f}, worst fall {hold_dd:.1%}\n")
    print(
        f"  {'buy-back rule':<15} {'runs':>5} {'median':>9} {'worst':>9} {'best':>9} "
        f"{'fall':>7} {'beat b&h':>9} {'beat flat':>10}"
    )
    print("  " + "-" * 82)

    rows: list[tuple[float, str, str]] = []
    for name, runs in results.items():
        finals = [r.final for r in runs]
        median = statistics.median(finals)
        beat_hold = sum(1 for r in runs if r.final > hold_final) / len(runs)
        beat_flat = sum(1 for r in runs if r.final > r.flat_final) / len(runs)
        line = (
            f"  {name:<15} {len(runs):>5} {median:>9,.0f} {min(finals):>9,.0f} "
            f"{max(finals):>9,.0f} {statistics.median(r.drawdown for r in runs):>6.1%} "
            f"{beat_hold:>9.0%} {beat_flat:>10.0%}"
        )
        rows.append((median, name, line))

    ranking = [n for _, n, _ in sorted(rows, reverse=True)]
    lookup = {n: line for _, n, line in rows}
    for name in order or ranking:
        print(lookup[name])
    return ranking


def run(
    *,
    path: Path,
    since: str,
    capital: float,
    defensive: float,
    timeout: int,
    cost: float,
    fall: float,
    horizon: int,
    fractions: tuple[float, ...],
) -> None:
    warnings.filterwarnings("ignore")
    print(
        f"\n  Ranking buy-back rules on the first window, re-measuring on the second."
        f"\n  Alarm thresholds swept: {', '.join(f'{f:.0%}' for f in fractions)}."
        f"\n  Each rule keeps the same defensive weight ({defensive:.0%}) and"
        f" {timeout}-day timeout.\n"
    )

    order: list[str] | None = None
    for label, start, end in WINDOWS:
        pipe = load(path, since=since, until=end, split_date=start, fall=fall, horizon=horizon)
        arrays = context_arrays(pipe.close, pipe.daily)
        results = _measure(
            pipe,
            arrays,
            fractions,
            capital=capital,
            defensive=defensive,
            timeout=timeout,
            cost=cost,
        )
        hold_final, hold_dd = _hold(pipe, capital)
        ranking = _report(label, results, hold_final, hold_dd, order)
        if order is None:
            order = ranking
            print("\n  (the second window below keeps this same row order, so a rule that")
            print("   slid down the table is visible as one out of place)")

    print(
        "\n  'median' is the middle result across every threshold and parameter — what"
        "\n  the idea is worth without hindsight. 'worst'/'best' bracket it: a wide"
        "\n  bracket means even the median is a coin toss. 'beat flat' compares against"
        "\n  holding the same average exposure all along, which is the row that"
        "\n  separates timing from simply owning less.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare crash-overlay buy-back rules.")
    parser.add_argument("--path", type=Path, default=Path("data/insider_index.csv"))
    parser.add_argument("--since", default="1990-01-01")
    parser.add_argument("--capital", type=float, default=5000.0)
    parser.add_argument("--defensive", type=float, default=0.3)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--cost", type=float, default=0.0005)
    parser.add_argument("--fall", type=float, default=0.02)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--sell-fraction", type=float, nargs="+", default=[0.05, 0.10, 0.15, 0.20])
    args = parser.parse_args()

    run(
        path=args.path,
        since=args.since,
        capital=args.capital,
        defensive=args.defensive,
        timeout=args.timeout,
        cost=args.cost,
        fall=args.fall,
        horizon=args.horizon,
        fractions=tuple(args.sell_fraction),
    )


if __name__ == "__main__":
    main()
