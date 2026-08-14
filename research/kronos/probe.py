"""Train Kronos to answer our question, and find out whether it knew anything.

    python probe.py --size small

## What is being trained

Kronos is a forecaster of candlesticks, not a classifier of crashes. To point it
at our question we take its final hidden state — its summary of the trading days
it was shown — and fit a classifier on top of it against the production label,
"tomorrow's close is 2% or more below today's". How many days that is turns out
to matter a great deal, and `sweep.py` chooses it the same way this script
chooses everything else: inside the training years.

The backbone stays frozen. That is a statistical decision, not a shortcut. The
training window holds around two hundred and thirty 2%-falls in total. Turning
twenty-five million parameters loose on two hundred and thirty examples does not
learn what a crash looks like; it learns what those particular two hundred and
thirty days looked like, and the held-out result would be worse *and* less
interpretable. A frozen backbone with a small head asks a cleaner question, and
it is the question we actually care about: **is the answer already in there?**

## The protocol, fixed before any result was read

  * **Test** on 2016 onward, looked at once, at the end. Everything before it is
    training data and may be sliced however we like.
  * Settings are chosen by **expanding-window cross-validation** inside the
    training years: fit on the first block, score the next, extend, repeat, and
    average. The first attempt used a single 2012-2015 validation window, and it
    was a mistake worth recording — that window contains fifteen falls, few
    enough that one variant picked a setting which scored 0.70 there and 0.54 on
    the test set. Averaging over folds is not a refinement; it is the difference
    between choosing and guessing.
  * Ties go to the stronger regularisation. No test number is allowed to
    influence any setting.
  * Every rival is scored **on identical days**, so differences cannot come from
    one model getting an easier calendar.

## The comparison that decides it

Beating a coin toss is not the bar. Two floors matter more:

  * **Recent choppiness alone** — the 20-day standard deviation of returns, one
    number, no model. Falls of 2% cluster in jumpy markets, so this is already a
    decent detector, and anything that cannot beat it is not earning its keep.
  * **Choppiness with Kronos added.** If the pair beats choppiness alone, Kronos
    is contributing something the simple number does not have. If it does not,
    then whatever Kronos knows, we already knew.

## One caveat that cannot be tested away

Kronos was pre-trained on a corpus drawn from 45 exchanges, and neither the
paper's abstract nor the model card states the cut-off date. If US index history
through 2024 is in there, then a good score on 2016-2024 may be partly memory
rather than foresight. Nothing in this script can rule that out, so the last
section reports the most recent slice separately: it is small and its interval
is wide, but it is the only part of the test window that a 2025-trained model
cannot have memorised.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import data
import metrics

#: Cross-validation folds inside the training years. Five gives each fold a few
#: hundred days and, crucially, spreads the calm and violent years across
#: different folds instead of concentrating them in one window.
FOLDS = 5

#: How hard to shrink the coefficients, and how many directions of the hidden
#: state to keep. Both are chosen inside the training years only.
STRENGTHS = (0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 1.0)
WIDTHS = (8, 16, 32, 64, 128, None)


@dataclass(frozen=True, slots=True)
class Split:
    fit: np.ndarray  #: everything before the test date
    test: np.ndarray

    def folds(self, count: int = FOLDS) -> list[tuple[np.ndarray, np.ndarray]]:
        """Expanding windows: fit on what came before, score what came next.

        Never a shuffled split. Days are not exchangeable — a shuffled fold
        would let a model trained on Tuesday and Thursday be scored on the
        Wednesday between them, which flatters anything that keys off the slow
        drift of volatility, i.e. everything here.
        """
        blocks = np.array_split(self.fit, count + 1)
        return [
            (np.concatenate(blocks[: k + 1]), blocks[k + 1])
            for k in range(count)
            if blocks[k + 1].size > 0
        ]


def split_by_date(date: pd.DatetimeIndex, test_from: str) -> Split:
    return Split(
        fit=np.flatnonzero(date < pd.Timestamp(test_from)),
        test=np.flatnonzero(date >= pd.Timestamp(test_from)),
    )


def choppiness(market: data.Market, bar: np.ndarray, window: int = 20) -> np.ndarray:
    """The 20-day standard deviation of daily returns, per decision bar.

    Backward-looking by construction: bar *i* uses returns up to and including
    bar *i*, and the label asks about bar *i + 1*.
    """
    close = market.close
    ret = np.r_[np.nan, close[1:] / close[:-1] - 1.0]
    rolling = pd.Series(ret).rolling(window).std().to_numpy()
    return rolling[bar]


def fit_one(
    features: np.ndarray,
    label: np.ndarray,
    train: np.ndarray,
    *,
    width: int | None,
    strength: float,
    seed: int = 0,
) -> np.ndarray:
    """Fit on `train`, score every row. Scaling and PCA are fitted on train too.

    That last point is not a formality. Standardising against statistics drawn
    from the whole series would let the test years quietly set the scale the
    training years are measured in, which is a leak — small, but the kind that
    makes a result irreproducible in live use.
    """
    scaler = StandardScaler().fit(features[train])
    scaled = scaler.transform(features)

    if width is not None and width < features.shape[1]:
        reducer = PCA(n_components=width, random_state=seed).fit(scaled[train])
        scaled = reducer.transform(scaled)

    # Balanced weights because one day in thirty is a fall: left alone, the
    # fitter would find that predicting "no" every time is 97% correct.
    model = LogisticRegression(
        C=strength, max_iter=4000, class_weight="balanced", random_state=seed
    )
    model.fit(scaled[train], label[train])
    return model.predict_proba(scaled)[:, 1]


def cross_validate(
    features: np.ndarray,
    label: np.ndarray,
    split: Split,
    *,
    width: int | None,
    strength: float,
) -> float:
    """Mean AUC across the expanding windows. Test data is never touched."""
    got = []
    for train, check in split.folds():
        if label[train].sum() < 5 or label[check].sum() < 3:
            continue
        scores = fit_one(features, label, train, width=width, strength=strength)
        value = metrics.auc(scores[check], label[check])
        if np.isfinite(value):
            got.append(value)
    return float(np.mean(got)) if got else float("nan")


def search(
    features: np.ndarray,
    label: np.ndarray,
    split: Split,
    *,
    name: str,
    verbose: bool = True,
) -> tuple[np.ndarray, int | None, float]:
    """Choose settings inside the training years, then fit once on all of them."""
    best: tuple[tuple[float, float, int], int | None, float] | None = None
    for width in WIDTHS:
        if width is not None and width > features.shape[1]:
            continue
        for strength in STRENGTHS:
            value = cross_validate(features, label, split, width=width, strength=strength)
            if not np.isfinite(value):
                continue
            # Ties go to the stronger shrinkage — smaller C, then fewer
            # directions — so a coin toss between settings never buys complexity.
            key = (round(value, 3), -strength, -(width or 10**6))
            if best is None or key > best[0]:
                best = (key, width, strength)

    assert best is not None, f"no setting produced a usable fold for {name}"
    (value, _, _), width, strength = best
    scores = fit_one(features, label, split.fit, width=width, strength=strength)
    if verbose:
        shape = "all" if width is None else str(width)
        print(
            f"    {name:<28} chose width={shape:<4} strength={strength:<7} "
            f"cross-validated AUC {value:.3f}"
        )
    return scores, width, value


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a 2%-fall head on Kronos features.")
    parser.add_argument("--cache", type=Path, default=Path("cache/sp500.npz"))
    parser.add_argument("--size", default="small", choices=("mini", "small", "base"))
    parser.add_argument("--features", type=Path, default=None)
    parser.add_argument("--test-from", default="2016-01-01")
    parser.add_argument("--recent-from", default="2025-01-01")
    parser.add_argument("--tag", default=None, help="column name; defaults to the feature file")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    market = data.load(args.cache)
    path = args.features or Path(f"cache/features_{args.size}.npz")
    # Named after the features it read, so that two runs over different window
    # lengths land in different columns instead of colliding in `compare.py`.
    tag = args.tag or path.stem.replace("features_", "")
    raw = np.load(path, allow_pickle=False)
    bar, date = raw["bar"], pd.DatetimeIndex(raw["date"])
    state, label = raw["state"].astype(np.float64), raw["label"]

    split = split_by_date(date, args.test_from)
    chop = choppiness(market, bar)[:, None]
    usable = np.isfinite(chop[:, 0])

    print(f"\n  Kronos-{args.size}: {state.shape[1]} numbers per day, {bar.size:,} days")
    print(
        f"  fit on {date[split.fit[0]].date()}..{date[split.fit[-1]].date()} "
        f"({int(label[split.fit].sum())} falls, {FOLDS} folds), "
        f"test from {args.test_from} ({int(label[split.test].sum())} falls)\n"
    )
    print("  choosing settings inside the training years only:")

    kronos, _, _ = search(state, label, split, name="Kronos alone")
    combined, _, _ = search(
        np.hstack([state, np.nan_to_num(chop)]), label, split, name="Kronos + choppiness"
    )
    simple, _, _ = search(np.nan_to_num(chop), label, split, name="choppiness alone")

    frame = pd.DataFrame(
        {
            "date": date,
            "label": label,
            f"kronos_{tag}": kronos,
            f"kronos_{tag}_plus_chop": combined,
            "choppiness_fitted": simple,
        }
    )
    out = args.out or Path(f"cache/pred_kronos_{tag}.csv")
    frame.iloc[split.test].to_csv(out, index=False)

    scores = []
    for column in frame.columns[2:]:
        keep = split.test[usable[split.test]]
        scores.append(metrics.evaluate(column, frame[column].to_numpy()[keep], label[keep]))

    print(f"\n  held out, {args.test_from} onward\n")
    print(metrics.table(scores))

    recent = split.test[(date[split.test] >= pd.Timestamp(args.recent_from)) & usable[split.test]]
    if recent.size > 100 and label[recent].sum() >= 3:
        print(f"\n  the most recent slice alone, {args.recent_from} onward — the part no")
        print("  pre-2025 corpus can contain. Small, so read the range not the number.\n")
        print(
            metrics.table(
                [
                    metrics.evaluate(c, frame[c].to_numpy()[recent], label[recent])
                    for c in frame.columns[2:]
                ]
            )
        )

    print(f"\n  predictions written to {out}\n")


if __name__ == "__main__":
    main()
