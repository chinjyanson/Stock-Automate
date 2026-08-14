"""How much history should Kronos be shown? Choose it without peeking. (§12)

    python features.py --size small --lookback 128
    python features.py --size small --lookback 256
    python sweep.py --size small

512 daily bars is two years of context. Kronos was pre-trained on a corpus with
a great deal of intraday data in it, so two years of daily bars may be far more
than it knows what to do with — and in fact it is: the same probe scores 0.645
cross-validated at 512 and 0.682 at 256.

That is a useful finding and a dangerous one, because "try three window lengths
and report the best" is how a result stops meaning anything. So the choice is
made the same way every other setting in this directory is made — by
cross-validated AUC inside the training years — and this script prints the two
columns side by side so anyone can check that the winner was picked on the left
one and not the right one.

If the two columns disagree, the cross-validated column is the answer and the
disagreement is the finding.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import metrics
import probe


def main() -> None:
    parser = argparse.ArgumentParser(description="Pick the lookback on training data alone.")
    parser.add_argument("--cache", type=Path, default=Path("cache"))
    parser.add_argument("--size", default="small")
    parser.add_argument("--test-from", default="2016-01-01")
    args = parser.parse_args()

    paths = sorted(args.cache.glob(f"features_{args.size}*.npz"))
    if not paths:
        raise SystemExit(f"no feature caches matching features_{args.size}*.npz")

    rows = []
    for path in paths:
        raw = np.load(path, allow_pickle=False)
        date = pd.DatetimeIndex(raw["date"])
        state, label = raw["state"].astype(np.float64), raw["label"]
        split = probe.split_by_date(date, args.test_from)

        best: tuple[tuple[float, float, int], int | None, float] | None = None
        for width in probe.WIDTHS:
            if width is not None and width > state.shape[1]:
                continue
            for strength in probe.STRENGTHS:
                value = probe.cross_validate(state, label, split, width=width, strength=strength)
                if not np.isfinite(value):
                    continue
                key = (round(value, 3), -strength, -(width or 10**6))
                if best is None or key > best[0]:
                    best = (key, width, strength)

        assert best is not None
        (chosen, _, _), width, strength = best
        scores = probe.fit_one(state, label, split.fit, width=width, strength=strength)
        held = metrics.evaluate(path.stem, scores[split.test], label[split.test])
        window = path.stem.split("_")[-1]
        rows.append((chosen, held, "512" if window == args.size else window))

    winner = max(rows, key=lambda r: r[0])[2]

    print(f"\n  Kronos-{args.size}: how much history to show it, chosen on the")
    print("  training years and then — separately — measured on the held-out ones.\n")
    print(
        f"  {'window':>8} {'cross-validated':>16} {'held-out AUC':>14} {'90% range':>14} "
        f"{'top 10%':>9}"
    )
    print("  " + "-" * 68)
    for chosen, held, window in sorted(rows, key=lambda r: int(r[2])):
        mark = "   <- chosen" if window == winner else ""
        print(
            f"  {window:>8} {chosen:>16.3f} {held.auc:>14.3f} "
            f"{held.auc_low:>6.3f}-{held.auc_high:<7.3f} {held.top_rate:>8.1%}{mark}"
        )

    ranked_by_test = max(rows, key=lambda r: r[1].auc)[2]
    if ranked_by_test != winner:
        print(f"\n  NOTE: the held-out column would have preferred {ranked_by_test}. The")
        print(f"  choice stands at {winner}, because that is the one made without looking.")
    else:
        print(f"\n  Both columns agree on {winner}, so the choice costs nothing in doubt.")
    print()


if __name__ == "__main__":
    main()
