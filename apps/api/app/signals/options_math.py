"""Black-Scholes, as an instrument for reading the options market.

Nothing here places an options order and nothing here is meant to. The point is
that a listed option chain is a published, continuously-repriced statement of
what the market expects — and Black-Scholes is the lens that turns a quoted
price into that expectation. Two readings matter downstream:

  * **Implied volatility**, solved backwards from a quoted price. The scanner's
    own volatility measures are all backward-looking; this is the only
    forward-looking one available for free.
  * **Gamma**, which drives the dealer-hedging read in `spx_options`.

Pure and dependency-free: `math` and `numpy` only. scipy would be the obvious
home for the normal CDF and for a root finder, and is deliberately not used —
it is a large dependency for two functions, on a worker with a 448M ceiling.
`math.erf` gives the CDF exactly, and bisection finds the vol robustly without
the failure modes Newton-Raphson has near expiry and deep out of the money.

Conventions: `t` is time to expiry in **years**, rates and volatilities are
annualised decimals (0.05 = 5%), and every function returns None rather than a
fabricated number when its inputs cannot support an answer.
"""

from __future__ import annotations

import math

#: Bounds for the implied-volatility search. 0.1% to 500% annualised spans
#: everything a real quote implies; a solution outside it means the quote is
#: broken (stale mark, crossed book) rather than that vol is really there.
_IV_LOW, _IV_HIGH = 1e-4, 5.0

#: Bisection halves the interval each pass, so 80 gets far below float noise.
_IV_ITERATIONS = 80


def _norm_cdf(x: float) -> float:
    """Standard normal CDF via the error function — exact, not a fit."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _d1_d2(
    spot: float, strike: float, t: float, vol: float, rate: float, dividend: float
) -> tuple[float, float] | None:
    if spot <= 0 or strike <= 0 or t <= 0 or vol <= 0:
        return None
    denominator = vol * math.sqrt(t)
    d1 = (math.log(spot / strike) + (rate - dividend + 0.5 * vol * vol) * t) / denominator
    return d1, d1 - denominator


def bs_price(
    spot: float,
    strike: float,
    t: float,
    vol: float,
    *,
    is_call: bool,
    rate: float = 0.0,
    dividend: float = 0.0,
) -> float | None:
    """Black-Scholes fair value of a European option."""
    pair = _d1_d2(spot, strike, t, vol, rate, dividend)
    if pair is None:
        return None
    d1, d2 = pair
    discounted_strike = strike * math.exp(-rate * t)
    discounted_spot = spot * math.exp(-dividend * t)
    if is_call:
        return discounted_spot * _norm_cdf(d1) - discounted_strike * _norm_cdf(d2)
    return discounted_strike * _norm_cdf(-d2) - discounted_spot * _norm_cdf(-d1)


def delta(
    spot: float,
    strike: float,
    t: float,
    vol: float,
    *,
    is_call: bool,
    rate: float = 0.0,
    dividend: float = 0.0,
) -> float | None:
    """Sensitivity of the option's value to a move in the underlying.

    Also read as a rough market-implied probability of finishing in the money,
    which is how `spx_options` uses it to locate the 25-delta wings.
    """
    pair = _d1_d2(spot, strike, t, vol, rate, dividend)
    if pair is None:
        return None
    d1, _ = pair
    discount = math.exp(-dividend * t)
    return discount * _norm_cdf(d1) if is_call else discount * (_norm_cdf(d1) - 1.0)


def gamma(
    spot: float,
    strike: float,
    t: float,
    vol: float,
    *,
    rate: float = 0.0,
    dividend: float = 0.0,
) -> float | None:
    """Rate of change of delta. Identical for calls and puts at a given strike.

    This is the quantity dealer-hedging flow is proportional to: a dealer who is
    long gamma must sell into rallies and buy into dips to stay neutral, and one
    who is short gamma must do the opposite.
    """
    pair = _d1_d2(spot, strike, t, vol, rate, dividend)
    if pair is None:
        return None
    d1, _ = pair
    return math.exp(-dividend * t) * _norm_pdf(d1) / (spot * vol * math.sqrt(t))


def implied_vol(
    price: float,
    spot: float,
    strike: float,
    t: float,
    *,
    is_call: bool,
    rate: float = 0.0,
    dividend: float = 0.0,
) -> float | None:
    """Volatility that reprices the option at `price`, by bisection.

    Bisection rather than Newton-Raphson deliberately. Vega collapses toward
    zero for deep out-of-the-money and near-expiry options — exactly the strikes
    a skew measurement cares about — and Newton divides by it, so it diverges
    on the quotes that matter most. Bisection cannot: price is monotonic in
    volatility, so halving the bracket always converges.

    None when the price is outside what any volatility can produce, which in
    practice means a stale or crossed quote rather than a real reading.
    """
    if price <= 0 or spot <= 0 or strike <= 0 or t <= 0:
        return None

    # Below intrinsic value there is no volatility that fits — the quote is
    # broken, not informative.
    discounted_strike = strike * math.exp(-rate * t)
    discounted_spot = spot * math.exp(-dividend * t)
    intrinsic = (
        max(0.0, discounted_spot - discounted_strike)
        if is_call
        else max(0.0, discounted_strike - discounted_spot)
    )
    if price < intrinsic - 1e-9:
        return None

    low, high = _IV_LOW, _IV_HIGH
    price_high = bs_price(spot, strike, t, high, is_call=is_call, rate=rate, dividend=dividend)
    if price_high is None or price > price_high:
        return None  # more expensive than 500% vol explains

    for _ in range(_IV_ITERATIONS):
        mid = 0.5 * (low + high)
        value = bs_price(spot, strike, t, mid, is_call=is_call, rate=rate, dividend=dividend)
        if value is None:
            return None
        if value > price:
            high = mid
        else:
            low = mid
    return 0.5 * (low + high)
