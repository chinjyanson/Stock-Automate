"""Every detector, the same days, one table. (§11)

    python compare.py

The individual scripts each print their own result, which makes it easy to
compare two numbers that were never comparable — different date ranges,
different rows dropped for missing data, different counts of the rare event that
all the statistics hinge on. This script exists to make that mistake harder: it
joins everything on date, keeps only the days *every* detector scored, and
prints one table.

It also carries the two floors, because a detector's real competition is not the
other detectors:

  * **the base rate** — what you get by guessing;
  * **recent choppiness** — one number, no model, no fitting.

And one combination that costs nothing to try: averaging the ranks of the
shipped detector and the best Kronos variant. If two detectors are each right
about different days, the average beats both; if one of them is noise, it does
not. That is a cheap and quite hard test of whether Kronos brought anything.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import data
import metrics


def rank(values: np.ndarray) -> np.ndarray:
    """Positions from 0 to 1, so two detectors on different scales can be averaged."""
    order = np.argsort(np.argsort(values))
    return order / max(len(values) - 1, 1)


def collect(cache: Path, market: data.Market, test_from: str) -> pd.DataFrame:
    close = market.close
    ret = np.r_[np.nan, close[1:] / close[:-1] - 1.0]
    frame = pd.DataFrame(
        {
            "date": market.index,
            "label": market.label,
            "recent choppiness": pd.Series(ret).rolling(20).std().to_numpy(),
        }
    )
    frame = frame[frame.label.notna()].reset_index(drop=True)

    for name, column in (
        ("logistic_1990", "shipped detector"),
        ("logistic_2006", "shipped, with insider data"),
    ):
        path = cache / f"{name}.csv"
        if not path.exists():
            continue
        other = pd.read_csv(path, parse_dates=["date"])[["date", "probability"]]
        frame = frame.merge(other.rename(columns={"probability": column}), on="date", how="left")

    for path in sorted(cache.glob("pred_*.csv")):
        other = pd.read_csv(path, parse_dates=["date"])
        # Every probe file carries its own fitted-choppiness control, and they
        # are all the same series. Keep the raw one above and drop the copies,
        # so the table lists each distinct detector once.
        other = other.drop(columns=["label", "choppiness_fitted"], errors="ignore")
        frame = frame.merge(other, on="date", how="left")

    for path in sorted(cache.glob("zeroshot_*.npz")):
        raw = np.load(path, allow_pickle=False)
        samples = raw["samples"]
        size = path.stem.replace("zeroshot_", "")
        other = pd.DataFrame(
            {
                "date": pd.DatetimeIndex(raw["date"]),
                f"zero-shot {size}: chance of a 2% fall": (samples <= -0.02).mean(axis=1),
                f"zero-shot {size}: worst 5% of futures": -np.quantile(samples, 0.05, axis=1),
                f"zero-shot {size}: spread of futures": samples.std(axis=1),
            }
        )
        frame = frame.merge(other, on="date", how="left")

    return frame[frame.date >= pd.Timestamp(test_from)].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare every detector on identical days.")
    parser.add_argument("--cache", type=Path, default=Path("cache"))
    parser.add_argument("--market", type=Path, default=Path("cache/sp500.npz"))
    parser.add_argument("--test-from", default="2016-01-01")
    parser.add_argument("--recent-from", default="2025-01-01")
    args = parser.parse_args()

    market = data.load(args.market)
    frame = collect(args.cache, market, args.test_from)

    columns = [c for c in frame.columns if c not in ("date", "label")]
    complete = frame[columns].notna().all(axis=1)
    frame = frame[complete].reset_index(drop=True)
    label = frame.label.to_numpy(float)

    # A fitting-free ensemble: average the two rankings and see if the pair beats
    # either half. Nothing is estimated, so there is nothing here to overfit.
    best_kronos = max(
        (c for c in columns if "kronos" in c or "zero-shot" in c or "finetune" in c),
        key=lambda c: metrics.auc(frame[c].to_numpy(), label),
        default=None,
    )
    if best_kronos and "shipped detector" in columns:
        frame["shipped + best Kronos, averaged"] = (
            rank(frame["shipped detector"].to_numpy()) + rank(frame[best_kronos].to_numpy())
        ) / 2.0
        columns.append("shipped + best Kronos, averaged")

    scores = [metrics.evaluate(c, frame[c].to_numpy(), label) for c in columns]
    scores.sort(key=lambda s: -s.auc)

    print(
        f"\n  {frame.shape[0]:,} days every detector scored, "
        f"{args.test_from} to {frame.date.max().date()}"
    )
    print(
        f"  {int(label.sum())} of them were followed by a 2% fall "
        f"({label.mean():.2%}) — guessing scores 0.500\n"
    )
    print(metrics.table(scores))

    reference = "shipped detector"
    if reference in columns:
        print("\n  and the same thing asked properly — each detector against the")
        print(f"  {reference}, resampled on the same days, so the noise they share")
        print("  cancels instead of hiding the difference:\n")
        print(f"  {'detector':<36} {'AUC gap':>9} {'90% range':>16} {'better in':>10}")
        print("  " + "-" * 76)
        gaps = []
        for column in columns:
            if column == reference:
                continue
            point, low, high, share = metrics.duel(
                frame[column].to_numpy(), frame[reference].to_numpy(), label
            )
            gaps.append((point, column, low, high, share))
        for point, column, low, high, share in sorted(gaps, reverse=True):
            verdict = "" if low < 0.0 < high else "   <- real"
            print(
                f"  {column:<36} {point:>+9.3f} {low:>+7.3f} to {high:<+7.3f} "
                f"{share:>9.0%}{verdict}"
            )
        print("\n  A range straddling zero means the two are not distinguishable on")
        print("  92 events, however different their headline numbers look.")

    if best_kronos:
        print(f"\n  best Kronos variant above: {best_kronos}")

    recent = frame.date >= pd.Timestamp(args.recent_from)
    if recent.sum() > 100 and label[recent.to_numpy()].sum() >= 3:
        print(f"\n  and the {args.recent_from}-onward slice alone — small, and the only part")
        print("  a model pre-trained through 2024 cannot have seen. Read the ranges.\n")
        late = [
            metrics.evaluate(c, frame[c].to_numpy()[recent.to_numpy()], label[recent.to_numpy()])
            for c in columns
        ]
        print(metrics.table(sorted(late, key=lambda s: -s.auc)))
    print()


if __name__ == "__main__":
    main()
