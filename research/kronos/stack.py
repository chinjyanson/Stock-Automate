"""Can a fitted combiner of the two detectors beat the better one alone? (§13)

    python stack.py

`compare.py` blends the shipped detector and Kronos by averaging their ranks —
equal weights, nothing fitted. The obvious objection is that equal weights are
arbitrary: fit the weights and Kronos might earn a small, useful share. This
script answers that in four steps — the first two measure it, the last two
explain why it comes out the way it does.

## 1. The ceiling, which settles it before any fitting happens

A logistic regression on two scores ranks days by a weighted sum of them.
Whatever it estimates, its ranking is a point on a one-parameter family. So
sweep that parameter and *read the test answers to pick the best point* — pure
cheating, and therefore an upper bound on every honest method. If the ceiling is
low, no combiner can be good, and no amount of clever fitting changes it.

## 2. The regression itself, fitted where it is allowed to be

Both detectors were fitted on data before 2016, so every score from 2016 onward
is out-of-sample for both. That makes the held-out decade safe to split again:
the combiner is fitted on its first half and measured on its second. Nothing it
sees while training has any bearing on the days it is judged on.

Doing it the other way — fitting the combiner on pre-2016 scores — would be
wrong in a way that is easy to miss. Both base models were *fitted* on those
years, so their pre-2016 output is in-sample and unrealistically sharp. A
combiner trained on it would learn how much to trust two detectors that no
longer exist, and would carry that mis-weighting into the test period.

## 3. Where they disagree, and who is right there

The part that explains the result rather than just reporting it. Two detectors
that agree everywhere cannot improve on each other. Two that disagree can — but
only if the disagreements are ones the second detector wins.

## 4. Each half on its own

Kronos does come out ahead over 2021-2026, which is where a fitted combiner
finds its small positive, and reporting that without this section would be
misleading. It is behind by more over 2016-2020, and the best weight swings from
0.08 to 0.80 between the two. A weight that moves like that across adjacent
five-year windows is not a weight that can be learned; it is whichever detector
happened to have the better half.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

import metrics

#: Where the held-out decade is split again, to give the combiner somewhere
#: honest to learn. Both base detectors are out-of-sample on both sides.
COMBINER_SPLIT = "2021-01-01"


def rank(values: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(values)) / max(len(values) - 1, 1)


def log_odds(p: np.ndarray) -> np.ndarray:
    """Probabilities on the scale a logistic regression naturally adds them on."""
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    z = np.log(p / (1.0 - p))
    return (z - z.mean()) / z.std()


def load(cache: Path, kronos: str) -> pd.DataFrame:
    ship = pd.read_csv(cache / "logistic_1990.csv", parse_dates=["date"])
    ship = ship[(ship.trained_on == 0) & ship.probability.notna()]
    ship = ship[["date", "probability", "label"]].rename(columns={"probability": "shipped"})
    other = pd.read_csv(cache / f"pred_{kronos}.csv", parse_dates=["date"])
    column = next(c for c in other.columns if c.startswith("kronos") or c.startswith("finetune"))
    other = other[["date", column]].rename(columns={column: "kronos"})
    return ship.merge(other, on="date").dropna().reset_index(drop=True)


def ceiling(frame: pd.DataFrame) -> None:
    label = frame.label.to_numpy(float)
    alone = metrics.auc(frame.shipped.to_numpy(), label)
    print("\n  1. The ceiling: every weighting, with the answers in hand.\n")
    print(
        f"  {'scale':<12} {'best weight on Kronos':>22} {'best AUC':>10} {'vs shipped alone':>18}"
    )
    print("  " + "-" * 66)
    for name, pair in (
        ("ranks", (rank(frame.shipped.to_numpy()), rank(frame.kronos.to_numpy()))),
        ("log-odds", (log_odds(frame.shipped.to_numpy()), log_odds(frame.kronos.to_numpy()))),
    ):
        left, right = pair
        best = max(
            ((metrics.auc((1 - w) * left + w * right, label), w) for w in np.linspace(0, 1, 101)),
            key=lambda t: t[0],
        )
        print(f"  {name:<12} {best[1]:>22.2f} {best[0]:>10.4f} {best[0] - alone:>+18.4f}")
    print(f"\n  shipped detector alone: {alone:.4f}. The rows above are the best a")
    print("  two-feature blend can do *while cheating*, so they bound everything below.")


def fitted(frame: pd.DataFrame, split: str) -> None:
    label = frame.label.to_numpy(float)
    train = np.flatnonzero(frame.date < pd.Timestamp(split))
    test = np.flatnonzero(frame.date >= pd.Timestamp(split))

    print(
        f"\n  2. The combiner, fitted on {frame.date[train[0]].date()}"
        f"..{frame.date[train[-1]].date()} ({int(label[train].sum())} falls)"
    )
    print(
        f"     and measured on {frame.date[test[0]].date()}"
        f"..{frame.date[test[-1]].date()} ({int(label[test].sum())} falls).\n"
    )

    features = np.column_stack(
        [log_odds(frame.shipped.to_numpy()), log_odds(frame.kronos.to_numpy())]
    )
    scores = {
        "shipped detector alone": frame.shipped.to_numpy(),
        "Kronos alone": frame.kronos.to_numpy(),
    }

    combiner = LogisticRegression(class_weight="balanced", max_iter=4000)
    combiner.fit(features[train], label[train])
    scores["logistic on the two"] = combiner.predict_proba(features)[:, 1]
    weights = combiner.coef_[0]

    # A non-linear combiner too, in case the two detectors are each right in a
    # region the other is wrong in — something a weighted sum cannot express.
    tree = GradientBoostingClassifier(random_state=0, max_depth=2, n_estimators=150)
    tree.fit(features[train], label[train])
    scores["boosted trees on the two"] = tree.predict_proba(features)[:, 1]

    print(f"  the logistic chose weights: shipped {weights[0]:+.3f}, Kronos {weights[1]:+.3f}")
    if weights[1] < 0:
        print("  — a negative weight on Kronos, i.e. it fitted best by subtracting it.")
    print()
    print(metrics.table([metrics.evaluate(n, s[test], label[test]) for n, s in scores.items()]))

    reference = scores["shipped detector alone"]
    print(f"\n  {'against shipped alone':<28} {'AUC gap':>9} {'90% range':>18} {'better in':>10}")
    print("  " + "-" * 70)
    for name, score in scores.items():
        if name == "shipped detector alone":
            continue
        point, low, high, share = metrics.duel(score[test], reference[test], label[test])
        verdict = "" if low < 0.0 < high else "   <- real"
        print(f"  {name:<28} {point:>+9.3f} {low:>+7.3f} to {high:<+7.3f} {share:>9.0%}{verdict}")


def disagreements(frame: pd.DataFrame, apart: float = 0.25) -> None:
    label = frame.label.to_numpy(float)
    ship_rank, kron_rank = rank(frame.shipped.to_numpy()), rank(frame.kronos.to_numpy())
    gap = kron_rank - ship_rank
    wide = np.abs(gap) > apart

    print("\n  3. Where they disagree, and who is right there.\n")
    print(f"  their rankings correlate {np.corrcoef(ship_rank, kron_rank)[0, 1]:.3f}")
    print(
        f"  they sit more than {apart:.0%} of the way apart on {int(wide.sum()):,} of "
        f"{len(frame):,} days\n"
    )
    for name, pick in (
        ("Kronos the more worried", wide & (gap > 0)),
        ("shipped the more worried", wide & (gap < 0)),
    ):
        print(f"    {name:<26} {int(pick.sum()):>5} days, {label[pick].mean():>5.1%} of them fell")
    print(f"    {'base rate':<26} {len(frame):>5} days, {label.mean():>5.1%}")
    print("\n  A blend can only add value on the days the two disagree. Whichever")
    print("  detector's extra worry is followed by more falls is the one to keep.")


def halves(frame: pd.DataFrame, split: str) -> None:
    """The same question asked of each half, because a flip is the tell.

    A weight that is right in one period and wrong in the next is not a weight,
    it is a coincidence. The buy-back study reached the same conclusion by the
    same route, and it is the reason this section exists at all.
    """
    label = frame.label.to_numpy(float)
    print("\n  4. Each half separately, and how the best weight moves between them.\n")
    print(
        f"  {'period':<14} {'days':>6} {'falls':>6} {'shipped':>9} {'Kronos':>8} "
        f"{'best weight on Kronos':>23}"
    )
    print("  " + "-" * 72)

    bounds: list[tuple[str, pd.Timestamp | None, pd.Timestamp | None]] = [
        ("first half", None, pd.Timestamp(split)),
        ("second half", pd.Timestamp(split), None),
        ("both", None, None),
    ]
    for name, start, stop in bounds:
        pick = np.ones(len(frame), dtype=bool)
        if start is not None:
            pick &= (frame.date >= start).to_numpy()
        if stop is not None:
            pick &= (frame.date < stop).to_numpy()
        left, right = rank(frame.shipped.to_numpy()[pick]), rank(frame.kronos.to_numpy()[pick])
        best = max(
            (
                (metrics.auc((1 - w) * left + w * right, label[pick]), w)
                for w in np.linspace(0, 1, 101)
            ),
            key=lambda t: t[0],
        )
        print(
            f"  {name:<14} {int(pick.sum()):>6,} {int(label[pick].sum()):>6} "
            f"{metrics.auc(frame.shipped.to_numpy()[pick], label[pick]):>9.3f} "
            f"{metrics.auc(frame.kronos.to_numpy()[pick], label[pick]):>8.3f} {best[1]:>23.2f}"
        )

    print("\n  If the best weight is stable across the halves, a fitted combiner has")
    print("  something to learn. If it swings, the fit is chasing whichever detector")
    print("  happened to have the better half, and will carry that into the next one.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit a combiner over the two detectors.")
    parser.add_argument("--cache", type=Path, default=Path("cache"))
    parser.add_argument("--kronos", default="kronos_small_256")
    parser.add_argument("--split", default=COMBINER_SPLIT)
    args = parser.parse_args()

    frame = load(args.cache, args.kronos)
    print(
        f"\n  shipped detector vs {args.kronos}, {len(frame):,} days, "
        f"{int(frame.label.sum())} falls"
    )

    ceiling(frame)
    fitted(frame, args.split)
    disagreements(frame)
    halves(frame, args.split)
    print()


if __name__ == "__main__":
    main()
