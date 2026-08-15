"""Logistic regression with a per-feature prior, fitted by IRLS.

The strategies used to sum hand-chosen weights and compare the total to a
hand-chosen threshold. This fits the weights instead, and returns a calibrated
probability rather than a score with no units.

**Why a prior, and not just a ridge penalty.** Some features have years of
history and some have none — dealer gamma exposure cannot be backfilled at all,
because an option chain is only published for today and per-strike open interest
is gone once the day passes. A plain ridge is already a Gaussian prior centred
at zero; all that is added here is the ability to centre it somewhere else, per
feature. A feature with no data keeps its prior exactly, a feature with plenty
of data overwhelms it, and everything in between is handled by the same solve.

**The honesty mechanism is `shrinkage`.** For each coefficient it reports the
fraction of the posterior precision that came from data rather than from the
prior — 0.0 means "this number is entirely something we assumed", 1.0 means
"the data decided this". Without it, a prior-driven coefficient is
indistinguishable from a fitted one once it is written to a table, which is
exactly how an assumption quietly becomes a finding.

**The clamp bounds a wrong prior.** On 20-day non-overlapping windows a
coefficient can stay prior-driven for years, so "the data will fix it" is not a
safeguard. `low_shrinkage_clamp` limits the summed log-odds that features below
`SHRINKAGE_THRESHOLD` may contribute, so a prior that is simply wrong can tilt
a decision and can never drive it. It lifts itself as those features earn their
shrinkage.

Dependency-free beyond numpy, as everything numeric here is: scipy would be
the obvious home for a fitter, and is a large dependency for one Newton solve on
a worker with a 448M ceiling.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from app.indicators.functions import FloatArray

#: Below this fraction, a coefficient is more belief than evidence. Used to
#: decide what the clamp applies to. 0.5 is the natural place to put it: it is
#: the point where the data has contributed as much precision as the prior did.
SHRINKAGE_THRESHOLD = 0.5

#: The intercept is deliberately almost unpenalised. A prior on it would be a
#: statement about the base rate, which the data always knows better than we do.
_INTERCEPT_TAU = 1e3

#: IRLS converges quadratically; 100 is a runaway guard, not a working limit.
_MAX_ITERATIONS = 100
_TOLERANCE = 1e-9

#: Floor on the IRLS weights p(1-p). Saturated probabilities drive it to zero
#: and make the Hessian singular, which is a numerical accident rather than a
#: real statement that the parameter is unidentified.
_WEIGHT_FLOOR = 1e-6


@dataclass(frozen=True, slots=True)
class Prior:
    """A Gaussian prior on one coefficient, in log-odds per standard deviation.

    `tau` is the prior standard deviation, so it says how firmly the belief is
    held: small is confident, large is barely an opinion. `Prior(0.0, 1.0)` is
    the uninformative default and behaves as light ridge.
    """

    mean: float = 0.0
    tau: float = 1.0

    @property
    def precision(self) -> float:
        return 1.0 / (self.tau * self.tau)


UNINFORMATIVE = Prior(0.0, 1.0)


@dataclass(frozen=True, slots=True)
class FittedModel:
    """Coefficients, the transform they expect, and how much was assumed.

    The standardisation is stored *with* the coefficients on purpose. Fitting on
    z-scored features and serving on raw ones is a silent, total failure, and
    keeping the two together is the only way to make it impossible.
    """

    feature_names: tuple[str, ...]
    coefficients: tuple[float, ...]
    intercept: float
    #: Standardisation learned at fit time, applied identically at serve time.
    means: tuple[float, ...]
    sds: tuple[float, ...]
    #: False when the training data carried no usable values for this feature,
    #: so its scale is unknown. Such a feature is *not served* — see `logit`.
    scale_known: tuple[bool, ...]
    prior_means: tuple[float, ...]
    prior_taus: tuple[float, ...]
    #: Fraction of each coefficient's precision that came from data, 0..1.
    shrinkage: tuple[float, ...]
    standard_errors: tuple[float, ...]
    n_observations: int
    positive_rate: float
    auc: float
    brier: float
    log_loss: float
    #: What a 1 means, in words. So a model read out of a table in a year still
    #: says what its probability is a probability *of*.
    label_definition: str
    #: Ceiling on the summed log-odds from features below SHRINKAGE_THRESHOLD.
    low_shrinkage_clamp: float | None = None

    def contributions(self, features: Mapping[str, float]) -> dict[str, float]:
        """Per-feature log-odds contribution, before any clamp.

        Exposed for provenance: a signal that carries these can be explained
        after the fact, which a bare probability cannot.
        """
        out: dict[str, float] = {}
        for i, name in enumerate(self.feature_names):
            out[name] = self.coefficients[i] * self._z(i, features.get(name))
        return out

    def logit(self, features: Mapping[str, float]) -> float:
        """Log-odds for one observation, with the clamp applied.

        A feature that is absent, non-finite, or whose scale was never learned
        imputes to the training mean and therefore contributes exactly zero.
        That is the same "missing is neutral" doctrine the scanner uses when a
        whole group is unavailable — a name is never rejected for lacking data.
        """
        free = 0.0
        clamped = 0.0
        for i, name in enumerate(self.feature_names):
            contribution = self.coefficients[i] * self._z(i, features.get(name))
            if self.shrinkage[i] < SHRINKAGE_THRESHOLD and self.low_shrinkage_clamp is not None:
                clamped += contribution
            else:
                free += contribution

        if self.low_shrinkage_clamp is not None:
            limit = abs(self.low_shrinkage_clamp)
            clamped = max(-limit, min(limit, clamped))
        return self.intercept + free + clamped

    def probability(self, features: Mapping[str, float]) -> float:
        """Calibrated probability that the label is 1."""
        return float(_sigmoid(np.array([self.logit(features)]))[0])

    def _z(self, index: int, value: float | None) -> float:
        if value is None or not math.isfinite(value) or not self.scale_known[index]:
            return 0.0
        return (value - self.means[index]) / self.sds[index]


def fit(
    features: FloatArray,
    labels: FloatArray,
    feature_names: Sequence[str],
    *,
    priors: Mapping[str, Prior] | None = None,
    label_definition: str,
    low_shrinkage_clamp: float | None = None,
) -> FittedModel:
    """Fit by iteratively reweighted least squares under a Gaussian prior.

    `features` is (n, k) raw — unstandardised — and may contain NaN, which is
    treated as "not observed" rather than as a value. `labels` is (n,) of 0/1.

    Maximises `l(w) - 0.5 (w - mu)' L (w - mu)` with `L = diag(1/tau^2)`, whose
    Newton step is `(X'WX + L) d = X'(y - p) - L(w - mu)`. Setting every prior
    to `Prior(0, tau)` recovers ordinary ridge, so this is one code path for a
    model that has data and one that partly does not.

    With no observations at all the gradient is zero at `w = mu`, so the fit
    returns the prior exactly. That is the intended behaviour, not a degenerate
    case: it is what lets a feature ship on belief and converge onto evidence.
    """
    names = tuple(feature_names)
    k = len(names)
    x = np.asarray(features, dtype=np.float64).reshape(-1, k) if k else np.zeros((0, 0))
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    if x.shape[0] != y.size:
        raise ValueError(f"features has {x.shape[0]} rows but labels has {y.size}")
    if y.size and not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("labels must be 0 or 1")

    prior_map = dict(priors or {})
    resolved = [prior_map.get(name, UNINFORMATIVE) for name in names]

    means, sds, scale_known = _standardisation(x)
    z = _standardise(x, means, sds, scale_known)

    # Column of ones for the intercept, which carries its own near-flat prior.
    design = np.hstack([np.ones((z.shape[0], 1)), z])
    prior_mean = np.array([0.0] + [p.mean for p in resolved])
    prior_precision = np.array([1.0 / (_INTERCEPT_TAU**2)] + [p.precision for p in resolved])

    weights = prior_mean.copy()  # start at the prior, so n = 0 converges at once
    data_precision = np.zeros(design.shape[1])
    hessian = np.diag(prior_precision)

    for _ in range(_MAX_ITERATIONS):
        p = _sigmoid(design @ weights)
        w = np.maximum(p * (1.0 - p), _WEIGHT_FLOOR)
        data_precision = np.einsum("ij,i,ij->j", design, w, design)
        hessian = (design.T * w) @ design + np.diag(prior_precision)
        gradient = design.T @ (y - p) - prior_precision * (weights - prior_mean)
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:  # pragma: no cover - the prior makes this unreachable
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        weights = weights + step
        if np.max(np.abs(step)) < _TOLERANCE:
            break

    covariance = np.linalg.inv(hessian)
    errors = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    shrinkage = data_precision / (data_precision + prior_precision)

    predicted = _sigmoid(design @ weights) if y.size else np.zeros(0)
    return FittedModel(
        feature_names=names,
        coefficients=tuple(weights[1:].tolist()),
        intercept=float(weights[0]),
        means=tuple(means.tolist()),
        sds=tuple(sds.tolist()),
        scale_known=tuple(bool(v) for v in scale_known),
        prior_means=tuple(p.mean for p in resolved),
        prior_taus=tuple(p.tau for p in resolved),
        shrinkage=tuple(shrinkage[1:].tolist()),
        standard_errors=tuple(errors[1:].tolist()),
        n_observations=int(y.size),
        positive_rate=float(y.mean()) if y.size else 0.0,
        auc=auc(y, predicted),
        brier=brier(y, predicted),
        log_loss=log_loss(y, predicted),
        label_definition=label_definition,
        low_shrinkage_clamp=low_shrinkage_clamp,
    )


def _standardisation(x: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Per-column mean and standard deviation, ignoring the unobserved.

    A column that is entirely missing, or constant, gets mean 0 / sd 1 and is
    marked scale-unknown. Its standardised column is then all zeros, so it
    contributes no precision and its coefficient stays at the prior — and
    `FittedModel` refuses to serve it, because a raw value divided by a
    fabricated scale would be a number with no meaning.
    """
    k = x.shape[1]
    means = np.zeros(k)
    sds = np.ones(k)
    known = np.zeros(k, dtype=bool)
    for j in range(k):
        column = x[:, j]
        finite = column[np.isfinite(column)]
        if finite.size < 2:
            continue
        sd = float(finite.std(ddof=1))
        if sd <= 0.0:
            continue
        means[j] = float(finite.mean())
        sds[j] = sd
        known[j] = True
    return means, sds, known


def _standardise(
    x: FloatArray, means: FloatArray, sds: FloatArray, known: FloatArray
) -> FloatArray:
    z = np.zeros_like(x)
    for j in range(x.shape[1]):
        if not known[j]:
            continue
        column = (x[:, j] - means[j]) / sds[j]
        z[:, j] = np.where(np.isfinite(column), column, 0.0)
    return z


def _sigmoid(x: FloatArray) -> FloatArray:
    """Numerically stable logistic function.

    The naive form overflows for large negative inputs, which a saturated
    coefficient reaches easily; branching on the sign keeps both tails exact.
    """
    out = np.empty_like(x, dtype=np.float64)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exponential = np.exp(x[~positive])
    out[~positive] = exponential / (1.0 + exponential)
    return out


def average_ranks(values: FloatArray) -> FloatArray:
    """1-indexed ranks, ties sharing their average.

    Tie handling is not a detail here: a feature that takes few distinct values
    produces many ties, and ordinal ranking would silently break them in input
    order, which is a bias rather than a rounding error.
    """
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(1, values.size + 1, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    for i in range(1, values.size + 1):
        if i == values.size or sorted_values[i] != sorted_values[start]:
            if i - start > 1:
                ranks[order[start:i]] = ranks[order[start:i]].mean()
            start = i
    return ranks


def auc(labels: FloatArray, predicted: FloatArray) -> float:
    """Area under the ROC curve, by the rank identity with Mann-Whitney U.

    0.5 is a coin flip. Reported on the *confirm* fold, where it is the number
    that decides whether the model knows anything at all.
    """
    positives = labels == 1.0
    n_pos = int(positives.sum())
    n_neg = int(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = average_ranks(np.asarray(predicted, dtype=np.float64))
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def brier(labels: FloatArray, predicted: FloatArray) -> float:
    """Mean squared error of the probability — a calibration score."""
    if labels.size == 0:
        return float("nan")
    return float(np.mean((predicted - labels) ** 2))


def log_loss(labels: FloatArray, predicted: FloatArray) -> float:
    if labels.size == 0:
        return float("nan")
    clipped = np.clip(predicted, 1e-12, 1.0 - 1e-12)
    return float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped)))


@dataclass(frozen=True, slots=True)
class Gap:
    """How much better one model ranks than another, and how sure we can be."""

    difference: float
    low: float
    high: float
    #: Share of resamples in which `left` came out ahead.
    share: float

    @property
    def real(self) -> bool:
        """Does the interval sit strictly to one side of zero?

        Written as "excludes zero" rather than "does not straddle zero". The two
        differ exactly when an endpoint *is* zero — including the degenerate
        case of comparing a model against itself, where the interval collapses
        to a single point at zero and the second phrasing calls it a real
        difference.
        """
        return self.high < 0.0 or self.low > 0.0


def auc_gap(
    labels: FloatArray,
    left: FloatArray,
    right: FloatArray,
    *,
    resamples: int = 2000,
    seed: int = 0,
) -> Gap:
    """Compare two models on the same days, by resampling the *difference*.

    Reading two separate confidence intervals and checking whether they overlap
    is the intuitive comparison and a badly underpowered one. Both intervals are
    wide largely because the evaluation window is short — and that noise is
    *shared*, since both models are being judged on the same days and the same
    handful of events. Resampling the pair together cancels it.

    The effect is not small. Two detectors whose separate 90% ranges overlap
    across most of their width can still differ with the paired interval
    entirely on one side of zero, because on any given resample the better one
    is better almost every time.

    Rare events make this matter more, not less: when a year contributes a dozen
    positives, which dozen you happened to get drives both scores in the same
    direction at once.
    """
    labels = np.asarray(labels, dtype=np.float64)
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)

    point = auc(labels, left) - auc(labels, right)
    rng = np.random.default_rng(seed)
    draws = np.empty(resamples, dtype=np.float64)
    for i in range(resamples):
        pick = rng.integers(0, labels.size, labels.size)
        draws[i] = auc(labels[pick], left[pick]) - auc(labels[pick], right[pick])

    usable = draws[np.isfinite(draws)]
    if usable.size == 0:
        return Gap(difference=point, low=float("nan"), high=float("nan"), share=float("nan"))
    return Gap(
        difference=point,
        low=float(np.quantile(usable, 0.05)),
        high=float(np.quantile(usable, 0.95)),
        share=float((usable > 0.0).mean()),
    )
