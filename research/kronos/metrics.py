"""Scoring a rare-event detector, with the uncertainty attached.

A 2% one-day fall happens on roughly one trading day in forty. A ten-year test
window therefore contains something like sixty of them, and *every* statistic
computed from sixty events is noisy. Quoting a bare number like "AUC 0.61"
invites a decision that the data cannot support, so every headline figure here
arrives with a bootstrap interval beside it and the count it was computed from.

The measures, and why each is here:

  * **AUC** — pick a day that fell and a day that did not; how often does the
    detector score the faller higher? 0.5 is a coin toss. It ignores calibration
    entirely, which is what we want first: we need to know whether the ranking
    carries information before caring what the numbers mean.
  * **Lift in the top slice** — of the days the detector is most worried about,
    what share actually fell, against the base rate. This is the one that maps
    to the trading rule, which acts on the worst few percent of days and ignores
    the rest.
  * **Brier score** — whether the probabilities mean what they say. Reported,
    but never used to choose between models, because a model can be beautifully
    calibrated and useless for ranking.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Bootstrap resamples behind every interval. Enough for a stable 5th/95th
#: percentile without making the comparison script slow.
RESAMPLES = 2000


@dataclass(frozen=True, slots=True)
class Score:
    name: str
    days: int
    events: int
    auc: float
    auc_low: float
    auc_high: float
    lift: float
    top_rate: float
    base_rate: float
    brier: float

    @property
    def informative(self) -> bool:
        """Is the ranking distinguishable from a coin toss at all?"""
        return self.auc_low > 0.5


def auc(score: np.ndarray, label: np.ndarray) -> float:
    """Rank-based AUC, ties shared. Equal to the Mann-Whitney U statistic.

    Computed from ranks rather than by sweeping thresholds so that a detector
    which emits many identical values — a saturating one, say — is neither
    rewarded nor punished for the tie.
    """
    positive = label > 0.5
    n_pos = int(positive.sum())
    n_neg = int(label.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(score, kind="mergesort")
    ordered = score[order]

    # Average the ranks within each run of equal scores, so a tie counts as half
    # a win rather than a whole one. Done without a Python loop because the
    # bootstrap calls this thousands of times, and resampling manufactures ties
    # by the hundred — the naive version is the slowest line in the file.
    edges = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1], True])
    starts, stops = edges[:-1], edges[1:]
    shared = (starts + stops + 1) / 2.0  # mean of the 1-based ranks in each run
    ranks = np.empty(score.size, dtype=np.float64)
    ranks[order] = np.repeat(shared, stops - starts)

    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def lift(score: np.ndarray, label: np.ndarray, fraction: float = 0.10) -> tuple[float, float]:
    """How dense are real falls among the days the detector fears most?

    Returns `(lift, hit rate)`: a lift of 2.0 means the top slice contains falls
    at twice the background rate. The slice size mirrors how the overlay is
    actually used — it acts on a small tail and ignores everything else.
    """
    base = float(label.mean())
    keep = max(1, int(round(score.size * fraction)))
    worst = np.argsort(-score, kind="mergesort")[:keep]
    rate = float(label[worst].mean())
    return (rate / base if base > 0 else float("nan")), rate


def brier(score: np.ndarray, label: np.ndarray) -> float:
    """Mean squared error of the probability. Only meaningful if score is one."""
    return float(np.mean((score - label) ** 2))


def evaluate(
    name: str,
    score: np.ndarray,
    label: np.ndarray,
    *,
    fraction: float = 0.10,
    seed: int = 0,
) -> Score:
    """Every measure at once, with a bootstrap interval on the headline one."""
    finite = np.isfinite(score) & np.isfinite(label)
    score, label = score[finite], label[finite]

    point = auc(score, label)
    rng = np.random.default_rng(seed)
    draws = np.empty(RESAMPLES)
    for i in range(RESAMPLES):
        pick = rng.integers(0, score.size, score.size)
        draws[i] = auc(score[pick], label[pick])
    draws = draws[np.isfinite(draws)]

    ratio, rate = lift(score, label, fraction)
    return Score(
        name=name,
        days=int(score.size),
        events=int(label.sum()),
        auc=point,
        auc_low=float(np.quantile(draws, 0.05)),
        auc_high=float(np.quantile(draws, 0.95)),
        lift=ratio,
        top_rate=rate,
        base_rate=float(label.mean()),
        brier=brier(np.clip(score, 0.0, 1.0), label),
    )


def duel(
    left: np.ndarray,
    right: np.ndarray,
    label: np.ndarray,
    *,
    seed: int = 0,
) -> tuple[float, float, float, float]:
    """Is `left` really better than `right`, or do their intervals just overlap?

    Comparing two detectors by checking whether their separate confidence
    intervals overlap is a weak and misleading test. Both intervals are wide
    mostly because the *test window* is short — the same 92 events drive both —
    and that shared source of noise cancels when the two are resampled together.
    So the difference is bootstrapped directly, on the same resampled days.

    Returns `(difference, low, high, share of resamples where left wins)`. If
    the interval straddles zero, the two detectors are not distinguishable on
    this data, whatever their point estimates say.
    """
    point = auc(left, label) - auc(right, label)
    rng = np.random.default_rng(seed)
    draws = np.empty(RESAMPLES)
    for i in range(RESAMPLES):
        pick = rng.integers(0, label.size, label.size)
        draws[i] = auc(left[pick], label[pick]) - auc(right[pick], label[pick])
    draws = draws[np.isfinite(draws)]
    return (
        point,
        float(np.quantile(draws, 0.05)),
        float(np.quantile(draws, 0.95)),
        float((draws > 0).mean()),
    )


HEADER = (
    f"  {'detector':<26} {'days':>6} {'falls':>6} {'AUC':>6} {'90% range':>14} "
    f"{'top 10% hit':>12} {'lift':>6}"
)


def row(score: Score) -> str:
    mark = "" if score.informative else "   (coin toss)"
    return (
        f"  {score.name:<26} {score.days:>6} {score.events:>6} {score.auc:>6.3f} "
        f"{score.auc_low:>6.3f}-{score.auc_high:<7.3f} "
        f"{score.top_rate:>11.1%} {score.lift:>6.2f}{mark}"
    )


def table(scores: list[Score]) -> str:
    lines = [HEADER, "  " + "-" * 86]
    lines += [row(s) for s in scores]
    return "\n".join(lines)
