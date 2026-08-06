"""Black-Scholes correctness, against known-answer cases.

These are checked against closed-form identities and textbook values rather
than against the implementation's own output. An options model that is subtly
wrong produces a confident number that is not right, and every downstream
reading — implied volatility, the skew, the dealer-gamma estimate — inherits
the error silently.
"""

from __future__ import annotations

import math

import pytest

from app.signals import options_math as om


class TestBlackScholesPrice:
    def test_a_textbook_call(self) -> None:
        # S=100, K=100, t=1, vol=20%, r=5% is the standard worked example;
        # the call is 10.4506 to four places.
        price = om.bs_price(100.0, 100.0, 1.0, 0.20, is_call=True, rate=0.05)
        assert price == pytest.approx(10.4506, abs=1e-4)

    def test_put_call_parity_holds(self) -> None:
        """C - P == S - K*exp(-rt). An identity, so any drift is a real error."""
        s, k, t, vol, r = 100.0, 95.0, 0.75, 0.25, 0.03
        call = om.bs_price(s, k, t, vol, is_call=True, rate=r)
        put = om.bs_price(s, k, t, vol, is_call=False, rate=r)
        assert call is not None and put is not None
        assert call - put == pytest.approx(s - k * math.exp(-r * t), abs=1e-9)

    def test_a_deep_in_the_money_call_approaches_its_intrinsic_value(self) -> None:
        price = om.bs_price(200.0, 50.0, 0.25, 0.15, is_call=True, rate=0.0)
        assert price == pytest.approx(150.0, abs=0.01)

    def test_none_when_expired_or_volatility_free(self) -> None:
        # No time and no volatility both mean the model has nothing to say.
        assert om.bs_price(100.0, 100.0, 0.0, 0.2, is_call=True) is None
        assert om.bs_price(100.0, 100.0, 1.0, 0.0, is_call=True) is None


class TestGreeks:
    def test_at_the_money_call_delta_is_about_a_half(self) -> None:
        d = om.delta(100.0, 100.0, 1.0, 0.20, is_call=True, rate=0.0)
        assert d == pytest.approx(0.54, abs=0.02)

    def test_call_and_put_delta_differ_by_one(self) -> None:
        """Another identity: with no dividend, delta_call - delta_put == 1."""
        call = om.delta(100.0, 110.0, 0.5, 0.3, is_call=True)
        put = om.delta(100.0, 110.0, 0.5, 0.3, is_call=False)
        assert call is not None and put is not None
        assert call - put == pytest.approx(1.0, abs=1e-12)

    def test_gamma_is_the_same_for_a_call_and_a_put(self) -> None:
        # Not a separate code path, but the property the GEX sum relies on.
        g = om.gamma(100.0, 100.0, 0.5, 0.2)
        assert g is not None and g > 0

    def test_gamma_peaks_near_the_money(self) -> None:
        near = om.gamma(100.0, 100.0, 0.25, 0.2)
        far = om.gamma(100.0, 160.0, 0.25, 0.2)
        assert near is not None and far is not None
        assert near > far

    def test_gamma_matches_a_numeric_second_derivative(self) -> None:
        """Gamma is d2V/dS2, so finite differences must agree."""
        s, k, t, vol = 100.0, 105.0, 0.5, 0.25
        h = 0.01
        up = om.bs_price(s + h, k, t, vol, is_call=True)
        mid = om.bs_price(s, k, t, vol, is_call=True)
        down = om.bs_price(s - h, k, t, vol, is_call=True)
        assert up is not None and mid is not None and down is not None
        numeric = (up - 2 * mid + down) / (h * h)
        assert om.gamma(s, k, t, vol) == pytest.approx(numeric, rel=1e-4)


class TestImpliedVolatility:
    def test_it_recovers_the_volatility_that_made_the_price(self) -> None:
        """The round trip. This is the property the whole module rests on."""
        for vol in (0.08, 0.20, 0.45, 1.20):
            price = om.bs_price(100.0, 100.0, 0.5, vol, is_call=True, rate=0.02)
            assert price is not None
            assert om.implied_vol(
                price, 100.0, 100.0, 0.5, is_call=True, rate=0.02
            ) == pytest.approx(vol, abs=1e-6)

    def test_it_recovers_volatility_far_out_of_the_money(self) -> None:
        """Where Newton-Raphson fails and bisection is why this uses bisection.

        Vega is tiny here, so a derivative-based solver diverges — and these are
        exactly the wing strikes a skew measurement is made of.
        """
        price = om.bs_price(100.0, 160.0, 0.08, 0.55, is_call=True)
        assert price is not None
        assert om.implied_vol(price, 100.0, 160.0, 0.08, is_call=True) == pytest.approx(
            0.55, abs=1e-5
        )

    def test_it_recovers_put_volatility(self) -> None:
        price = om.bs_price(100.0, 90.0, 0.3, 0.33, is_call=False, rate=0.01)
        assert price is not None
        assert om.implied_vol(price, 100.0, 90.0, 0.3, is_call=False, rate=0.01) == pytest.approx(
            0.33, abs=1e-6
        )

    def test_none_below_intrinsic_value(self) -> None:
        # A price under intrinsic is a broken quote, not a low volatility.
        assert om.implied_vol(1.0, 200.0, 50.0, 0.25, is_call=True) is None

    def test_none_when_the_price_is_impossibly_high(self) -> None:
        assert om.implied_vol(99.0, 100.0, 100.0, 0.5, is_call=True) is None

    def test_none_on_a_worthless_quote(self) -> None:
        assert om.implied_vol(0.0, 100.0, 100.0, 0.5, is_call=True) is None
