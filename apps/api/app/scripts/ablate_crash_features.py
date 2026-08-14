"""Which of the detector's ten inputs are actually earning their place? (§14)

    python -m app.scripts.ablate_crash_features

The crash detector reads ten market series and scores 0.798 on held-out days.
The 20-day spread of daily returns — one number, no model, no fitting — scores
0.749 on the same days. So the entire apparatus is worth about five points of
AUC over a line of code, and that invites an obvious question: which of the ten
is buying those five points, and which are along for the ride?

It matters because roughly half of them are volatility wearing different hats.
`vix` is implied volatility, `realised_vol` is realised volatility,
`vol_of_vol` is the volatility of that, and `vix_term_structure` is its shape.
If those four are mostly one feature, the detector is thinner than it looks and
the room for improvement is in the other six.

## Four questions, because no single one answers it

  * **Drop one.** What does the model lose without this feature? Low is not the
    same as useless — two correlated features each cover for the other, so both
    look free to remove while removing both is expensive.
  * **Keep only one.** What is this feature worth alone? High here and low in
    the drop-one table is the signature of a redundant feature.
  * **Drop a whole family.** The test the first two cannot do. If the four
    volatility features are one idea in four costumes, this is where it shows.
  * **Build it up from nothing.** Greedy forward selection, scored inside the
    training years, then measured once on the held-out ones. This is the only
    section that produces a recommendation rather than a diagnosis.

## Reading the numbers without fooling ourselves

Every comparison is a paired bootstrap against the full model (`auc_gap`),
because the held-out decade has 92 falls in it and two models judged on the same
92 events share most of their noise.

Even so: the drop-one table runs ten tests at a 90% interval, so **one crossing
is expected by chance**. A single feature whose interval clears zero is not a
finding. A family that clears it, or a feature that clears it in both the
drop-one and keep-only tables, is worth acting on.
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.backtest.overlay_pipeline import Gathered, Pipeline, assemble, gather
from app.models_ml.logistic import Gap, auc, auc_gap

#: What each feature is really measuring, for the family ablation. The point of
#: the grouping is that members of a family are near-substitutes, so dropping
#: any one of them tells you very little and dropping all of them tells you
#: everything.
FAMILIES: dict[str, tuple[str, ...]] = {
    "volatility": ("vix", "vix_term_structure", "realised_vol", "vol_of_vol"),
    "credit": ("credit_spread", "hyg_tlt", "hyg_lqd"),
    "crash pricing": ("skew",),
    "market state": ("small_cap_rs", "drawdown_from_high"),
    "insiders": ("insider_rank",),
}


@dataclass(frozen=True, slots=True)
class Result:
    name: str
    features: tuple[str, ...]
    score: float
    gap: Gap | None
    #: True when the model came out constant because its inputs had no data in
    #: the training window. Reported rather than scored — see `flat`.
    untested: bool = False


def held_out(pipe: Pipeline, *, fall: float, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Scores and labels for the bars after the split, where a fit never looked."""
    at = np.arange(pipe.cut, pipe.close.size - horizon)
    at = at[np.isfinite(pipe.probability[at])]
    return pipe.probability[at], pipe.labels(at, fall=fall, horizon=horizon)


def window(
    pipe: Pipeline, start: int, stop: int, *, fall: float, horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    """The same, over an arbitrary span — used for the inner selection window."""
    at = np.arange(start, min(stop, pipe.close.size - horizon))
    at = at[np.isfinite(pipe.probability[at])]
    return pipe.probability[at], pipe.labels(at, fall=fall, horizon=horizon)


def flat(scores: np.ndarray) -> bool:
    """Did the model come out constant — i.e. did it have nothing to fit?

    A feature whose series has not started yet is all-NaN across the training
    window, so its coefficient stays at the prior and every day gets the same
    probability. That scores exactly 0.500 and *looks* like a feature which
    carries no signal, when the truth is that it was never tested. HYG begins in
    2007 and the VIX term structure in 2007, so any split before that hits this.
    """
    usable = scores[np.isfinite(scores)]
    return usable.size > 0 and bool(np.ptp(usable) < 1e-12)


def choppiness(close: np.ndarray, span: int = 20) -> np.ndarray:
    """The floor: 20-day standard deviation of daily returns, and nothing else."""
    daily = np.concatenate([[np.nan], close[1:] / close[:-1] - 1.0])
    out = np.full(close.size, np.nan)
    for i in range(span, close.size):
        chunk = daily[i - span + 1 : i + 1]
        chunk = chunk[np.isfinite(chunk)]
        if chunk.size == span:
            out[i] = float(np.std(chunk, ddof=1))
    return out


def measure(
    source: Gathered,
    features: tuple[str, ...],
    *,
    split_date: str,
    fall: float,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    pipe = assemble(source, split_date=split_date, fall=fall, horizon=horizon, features=features)
    return held_out(pipe, fall=fall, horizon=horizon)


def table(rows: list[Result], baseline: float, *, note: str = "") -> None:
    print(f"  {'variant':<26} {'inputs':>6} {'AUC':>7} {'vs full':>9} {'90% range':>18} {'':>4}")
    print("  " + "-" * 76)
    for row in rows:
        if row.untested:
            print(
                f"  {row.name:<26} {len(row.features):>6} {'--':>7}   "
                f"no data in the training window"
            )
            continue
        if row.gap is None:
            print(f"  {row.name:<26} {len(row.features):>6} {row.score:>7.3f} {'':>9} {'':>18}")
            continue
        mark = "  <-" if row.gap.real else ""
        print(
            f"  {row.name:<26} {len(row.features):>6} {row.score:>7.3f} "
            f"{row.gap.difference:>+9.3f} {row.gap.low:>+7.3f} to {row.gap.high:<+7.3f}{mark}"
        )
    print(f"\n  full model {baseline:.3f}. {note}")


def forward_select(
    source: Gathered,
    *,
    split_date: str,
    inner_date: str,
    fall: float,
    horizon: int,
) -> list[tuple[str, float]]:
    """Add features one at a time, judged only on data the final test never sees.

    The inner window is carved out of the *training* years, so this whole
    procedure could have been run in 2015 with no knowledge of what followed.
    That is what makes the resulting shortlist a recommendation rather than a
    description of the test set.
    """
    index = source.index
    inner_cut = int(index.searchsorted(np.datetime64(inner_date)))
    outer_cut = int(index.searchsorted(np.datetime64(split_date)))

    chosen: list[str] = []
    trail: list[tuple[str, float]] = []
    best_so_far = 0.5

    while True:
        candidates = [f for f in source.available if f not in chosen]
        if not candidates:
            break
        scored = []
        for feature in candidates:
            trial = (*chosen, feature)
            pipe = assemble(
                source, split_date=inner_date, fall=fall, horizon=horizon, features=trial
            )
            scores, labels = window(pipe, inner_cut, outer_cut, fall=fall, horizon=horizon)
            scored.append((auc(labels, scores), feature))
        scored.sort(reverse=True)
        value, feature = scored[0]
        # Stop as soon as the best remaining addition stops helping. No
        # tolerance band: anything that needs one is inside the noise anyway.
        if value <= best_so_far:
            break
        best_so_far = value
        chosen.append(feature)
        trail.append((feature, value))
    return trail


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablate the crash detector's features.")
    parser.add_argument("--path", type=Path, default=Path("data/insider_index.csv"))
    parser.add_argument("--since", default="1990-01-01")
    parser.add_argument("--until", default=None)
    parser.add_argument("--split-date", default="2016-01-01")
    parser.add_argument("--inner-date", default="2010-01-01")
    parser.add_argument("--fall", type=float, default=0.02)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--skip-selection", action="store_true")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    source = gather(args.path, since=args.since, until=args.until)
    full = tuple(source.available)
    shared = {"split_date": args.split_date, "fall": args.fall, "horizon": args.horizon}

    base_scores, labels = measure(source, full, **shared)
    baseline = auc(labels, base_scores)

    print(
        f"\n  {len(labels):,} held-out days from {args.split_date}, "
        f"{int(labels.sum())} of them followed by a {args.fall:.0%} fall"
    )
    print(f"  the detector reads {len(full)} inputs: {', '.join(full)}\n")

    # The floor. If a subset of the model cannot beat this, the model is not
    # what is doing the work. Compared on exactly the days both can score, so
    # the gap is a paired one rather than two numbers from two calendars.
    pipe = assemble(source, features=full, **shared)
    rough = choppiness(source.close)
    at = np.arange(pipe.cut, pipe.close.size - args.horizon)
    at = at[np.isfinite(rough[at]) & np.isfinite(pipe.probability[at])]
    shared_labels = pipe.labels(at, fall=args.fall, horizon=args.horizon)
    floor = auc(shared_labels, rough[at])
    over_floor = auc_gap(shared_labels, pipe.probability[at], rough[at])
    print(f"  the floor — 20-day choppiness alone, no model: {floor:.3f}")
    print(f"  the full detector:                            {baseline:.3f}")
    print(
        f"  what the model buys over the floor:           {over_floor.difference:+.3f} "
        f"({over_floor.low:+.3f} to {over_floor.high:+.3f})"
        f"{'' if over_floor.real else '  — not distinguishable'}\n"
    )

    print("  1. Drop one feature and refit.\n")
    rows = []
    for feature in full:
        kept = tuple(f for f in full if f != feature)
        scores, _ = measure(source, kept, **shared)
        rows.append(
            Result(
                f"without {feature}",
                kept,
                auc(labels, scores),
                auc_gap(labels, scores, base_scores),
            )
        )
    rows.sort(key=lambda r: r.score)
    table(rows, baseline, note="Most negative at the top: those hurt most to lose.")
    print("  Ten tests at a 90% interval, so about one crossing is expected by chance.")

    print("\n  2. Keep only one feature.\n")
    rows = []
    for feature in full:
        scores, _ = measure(source, (feature,), **shared)
        if flat(scores):
            rows.append(Result(f"{feature} alone", (feature,), 0.5, None, untested=True))
            continue
        rows.append(
            Result(
                f"{feature} alone",
                (feature,),
                auc(labels, scores),
                auc_gap(labels, scores, base_scores),
            )
        )
    rows.sort(key=lambda r: (r.untested, -r.score))
    table(rows, baseline, note="High here but free to drop above means redundant.")

    print("\n  3. Drop a whole family.\n")
    rows = []
    for name, members in FAMILIES.items():
        kept = tuple(f for f in full if f not in members)
        present = [m for m in members if m in full]
        if not present or not kept:
            continue
        scores, _ = measure(source, kept, **shared)
        rows.append(
            Result(
                f"without {name}", kept, auc(labels, scores), auc_gap(labels, scores, base_scores)
            )
        )
    rows.sort(key=lambda r: r.score)
    table(rows, baseline, note="The test that survives features covering for each other.")

    if not args.skip_selection:
        print(f"\n  4. Built up from nothing, chosen on {args.inner_date}-{args.split_date}")
        print("     — inside the training years, so the held-out decade stays unseen.\n")
        trail = forward_select(
            source,
            split_date=args.split_date,
            inner_date=args.inner_date,
            fall=args.fall,
            horizon=args.horizon,
        )
        picked: list[str] = []
        rows = []
        for feature, inner in trail:
            picked.append(feature)
            scores, _ = measure(source, tuple(picked), **shared)
            rows.append(
                Result(
                    f"+ {feature}",
                    tuple(picked),
                    auc(labels, scores),
                    auc_gap(labels, scores, base_scores),
                )
            )
            print(f"    added {feature:<22} training-window AUC {inner:.3f}")
        print()
        table(rows, baseline, note="Held-out score of each step, for information only.")
        print(f"  it chose {len(picked)} of {len(full)}: {', '.join(picked)}")

    print()


if __name__ == "__main__":
    main()
