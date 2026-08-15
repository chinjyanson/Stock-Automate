"""Write out what the shipped detector thought, day by day. (§10)

    python -m app.scripts.dump_crash_probabilities --split-date 2016-01-01

One row per bar: the date, the close, what the detector thought, and what
actually happened. It exists so that anything wanting to be compared against
the shipped detector can be, without having to import it or reconstruct it.

Its current caller is `test_gex_theory.py --detector`, which needs the
detector's own scores to ask whether dealer gamma adds anything to them. It was
written for a comparison against a foundation model that ran in a separate
virtualenv and could not import the API at all; that experiment is finished and
gone, and the seam turned out to be worth keeping on its own.

Writing the label out beside the probability is deliberate. It means the reader
never has to reconstruct what counts as a 2% fall, and any disagreement about
the target shows up as a mismatched column rather than as a quiet few-tenths
difference in someone's accuracy figure.
"""

from __future__ import annotations

import argparse
import csv
import warnings
from pathlib import Path

import numpy as np

from app.backtest.overlay_pipeline import load


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump per-bar crash probabilities.")
    parser.add_argument("--path", type=Path, default=Path("data/insider_index.csv"))
    parser.add_argument("--since", default="1990-01-01")
    parser.add_argument("--until", default=None)
    parser.add_argument("--split-date", default="2016-01-01")
    parser.add_argument("--fall", type=float, default=0.02)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    pipe = load(
        args.path,
        since=args.since,
        until=args.until,
        split_date=args.split_date,
        fall=args.fall,
        horizon=args.horizon,
    )

    at = np.arange(pipe.close.size)
    label = pipe.labels(at, fall=args.fall, horizon=args.horizon)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date", "close", "probability", "label", "trained_on"])
        for bar in at:
            # The last `horizon` bars have no answer yet; the labeller returns
            # 0.0 there rather than nothing, so they are dropped here instead.
            i = int(bar)
            if i + args.horizon >= pipe.close.size:
                continue
            writer.writerow(
                [
                    pipe.index[i].date().isoformat(),
                    f"{pipe.close[i]:.4f}",
                    "" if not np.isfinite(pipe.probability[i]) else f"{pipe.probability[i]:.8f}",
                    int(label[i]),
                    int(i < pipe.cut),
                ]
            )

    scored = np.isfinite(pipe.probability)
    held = at[scored & (at >= pipe.cut)]
    print(f"\n  features: {', '.join(pipe.features)}")
    print(f"  {int(scored.sum()):,} scored bars, {held.size:,} of them after {args.split_date}")
    print(
        f"  falls in the held-out part: {int(label[held].sum())} ({label[held].mean():.2%} of days)"
    )
    print(f"  written to {args.out}\n")


if __name__ == "__main__":
    main()
