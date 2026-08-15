"""Whole-book stress testing by historical bootstrap.

The risk engine sizes every position individually — a stop distance, a risk
budget, a set of caps. What it could not do until now is ask the portfolio-level
question: *given everything already held, plus this candidate, how bad could the
next month plausibly get?* Six positions each risking 1% are not a 6% risk if
they all fall together, and nothing in per-position sizing can see that.

The method is a historical bootstrap rather than a parametric model. Daily
returns are resampled **by date**, with replacement, across a horizon. Because
the portfolio's return on a given date is a fixed weighted sum of its holdings'
returns on that date, resampling dates preserves the cross-sectional correlation
between holdings *exactly* — no covariance matrix to estimate, nothing to keep
positive semi-definite, and no assumption that returns are normal, which they
observably are not in the tail this function exists to measure.

Pure and I/O-free, like `app.scanner.scoring`: everything it needs is passed in,
so it is fully deterministic and testable without a database or a network.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

import numpy as np

from app.indicators.functions import FloatArray

#: Simulated paths per run. 2000 is enough to put a stable estimate on a 95th
#: percentile while costing single-digit milliseconds; the array it allocates is
#: a few megabytes, irrelevant against the worker's memory ceiling.
DEFAULT_PATHS = 2000

#: Horizon in trading days — roughly one month, matching the holding period the
#: mean-reversion strategy actually targets.
DEFAULT_HORIZON_DAYS = 20

#: Which tail to report. 95 means "the loss only 5% of simulated months exceed".
DEFAULT_PERCENTILE = 95.0

#: Minimum overlapping history before a bootstrap means anything. Below this the
#: resample is drawing from too few distinct days to describe a distribution, and
#: the honest answer is "unknown" rather than a confident small number.
MIN_RETURNS = 60


@dataclass(frozen=True, slots=True)
class StressResult:
    """The tail outcome of a simulated horizon, as a positive loss fraction."""

    paths: int
    horizon_days: int
    percentile: float
    #: Positive == loss. 0.12 means "5% of simulated months lost more than 12%".
    #: Floored at zero: a book whose tail outcome is still a gain has no stress
    #: loss to report, and a negative number here would invite sign errors at
    #: every call site that treats this as a magnitude.
    portfolio_loss_pct: float


def bootstrap_stress(
    position_returns: dict[str, FloatArray],
    weights: dict[str, float],
    *,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    paths: int = DEFAULT_PATHS,
    percentile: float = DEFAULT_PERCENTILE,
    rng: np.random.Generator,
) -> StressResult | None:
    """Tail loss of a weighted book over `horizon_days`, by date resampling.

    `position_returns` maps an opaque key (an instrument id) to that holding's
    daily returns, oldest-first. `weights` maps the same keys to each holding's
    share of equity — they need not sum to 1.0, and a book that is only half
    invested should pass weights summing to 0.5.

    Returns None when there is nothing to simulate or too little overlapping
    history to simulate from. That is deliberately distinct from a result of
    0.0: "we cannot say" must not read as "no risk".

    `rng` is required rather than defaulted so the caller owns determinism —
    the same book must produce the same verdict, in a test and in production.
    """
    if horizon_days <= 0 or paths <= 0 or not 0.0 < percentile < 100.0:
        return None

    keys = sorted(k for k in position_returns if weights.get(k))
    if not keys:
        return None

    # Align to the shortest available history so every holding contributes the
    # same dates — a bootstrap over ragged series would silently sample some
    # holdings from a different period than others.
    length = min(int(position_returns[k].size) for k in keys)
    if length < MIN_RETURNS:
        return None

    returns = np.vstack([np.asarray(position_returns[k], dtype=np.float64)[-length:] for k in keys])
    if not np.isfinite(returns).all():
        return None
    weight_vector = np.array([weights[k] for k in keys], dtype=np.float64)

    # The portfolio's return on date j is the weighted sum of its holdings'
    # returns on date j. Collapsing to that series first is exact — the weighted
    # sum is linear, so aggregating before resampling and after are identical —
    # and it turns an (assets x paths x horizon) sample into a 1-D lookup.
    portfolio_daily = weight_vector @ returns

    draws = rng.integers(0, length, size=(paths, horizon_days))
    # Compound rather than sum: a 20-day outcome is a product of daily factors,
    # and over a tail-sized move the difference is not academic.
    cumulative = np.prod(1.0 + portfolio_daily[draws], axis=1) - 1.0

    # The 95th-percentile *loss* is the 5th percentile of the return
    # distribution, negated.
    tail_return = float(np.percentile(cumulative, 100.0 - percentile))
    return StressResult(
        paths=paths,
        horizon_days=horizon_days,
        percentile=percentile,
        portfolio_loss_pct=max(0.0, -tail_return),
    )


def drawdown_scaled_quantity(
    proposed_quantity: Decimal,
    stressed_loss_pct: float,
    max_drawdown_pct: float,
) -> Decimal:
    """Shrink a proposed size until the book's stressed loss fits the limit.

    Scales linearly: if the simulated tail loss is 20% against a 15% limit, the
    position is cut to three quarters. That is an approximation — the candidate
    is only part of the book, so halving it does not halve the loss — but it is
    a *conservative* one in the direction that matters, deterministic, and one
    simulation rather than a bisection search. Tighten to a search only if this
    proves too coarse in practice.

    Returns the proposal untouched when the limit is not breached, or when
    either input is non-positive and the ratio would be meaningless.
    """
    if stressed_loss_pct <= 0.0 or max_drawdown_pct <= 0.0:
        return proposed_quantity
    if stressed_loss_pct <= max_drawdown_pct:
        return proposed_quantity
    scale = Decimal(str(max_drawdown_pct)) / Decimal(str(stressed_loss_pct))
    return proposed_quantity * scale
