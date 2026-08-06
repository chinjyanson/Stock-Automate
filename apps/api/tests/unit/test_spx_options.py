"""Index option-chain readings: dealer gamma, skew, at-the-money vol.

Built entirely from synthetic chains priced with the module's own Black-Scholes,
so every expected value is known in advance rather than reproduced from a live
provider. The sign of GEX is the thing most worth pinning: it rests on an
assumed dealer position, and a sign error would invert the meaning of the signal
the index strategy trades on.
"""

from __future__ import annotations

from datetime import date

import pytest

from app.signals import options_math as om
from app.signals import spx_options as sx

_AS_OF = date(2026, 8, 6)
_SPOT = 5000.0
_DAYS = 30


def _quote(strike: float, *, is_call: bool, oi: float, vol: float = 0.20) -> sx.OptionQuote:
    t = _DAYS / 365.0
    price = om.bs_price(_SPOT, strike, t, vol, is_call=is_call)
    assert price is not None
    return sx.OptionQuote(
        strike=strike,
        last_price=price,
        open_interest=oi,
        is_call=is_call,
        provider_iv=vol,
    )


def _read(quotes: list[sx.OptionQuote]) -> sx.IndexOptionsReading | None:
    return sx.compute_reading(
        as_of=_AS_OF, symbol="^SPX", spot=_SPOT, quotes=quotes, expiry_days=_DAYS
    )


class TestGammaExposure:
    def test_call_only_open_interest_is_positive_gamma(self) -> None:
        """Dealers are assumed long calls, so call open interest reads positive.

        If this ever flips, every entry and exit in the index strategy inverts.
        """
        reading = _read([_quote(5000, is_call=True, oi=1000)])
        assert reading is not None
        assert reading.gamma_exposure is not None
        assert reading.gamma_exposure > 0

    def test_put_only_open_interest_is_negative_gamma(self) -> None:
        reading = _read([_quote(5000, is_call=False, oi=1000)])
        assert reading is not None
        assert reading.gamma_exposure is not None
        assert reading.gamma_exposure < 0

    def test_balanced_open_interest_nets_to_about_zero(self) -> None:
        """Gamma is identical for a call and a put at the same strike, so equal
        open interest on both sides cancels under the assumed dealer position."""
        reading = _read([_quote(5000, is_call=True, oi=1000), _quote(5000, is_call=False, oi=1000)])
        assert reading is not None
        assert reading.gamma_exposure == pytest.approx(0.0, abs=1e-9)

    def test_more_open_interest_means_more_exposure(self) -> None:
        small = _read([_quote(5000, is_call=True, oi=100)])
        large = _read([_quote(5000, is_call=True, oi=1000)])
        assert small is not None and large is not None
        assert small.gamma_exposure is not None and large.gamma_exposure is not None
        assert large.gamma_exposure == pytest.approx(10 * small.gamma_exposure, rel=1e-9)

    def test_strikes_without_open_interest_are_excluded(self) -> None:
        """No open interest means no position to hedge, so no hedging flow."""
        reading = _read([_quote(5000, is_call=True, oi=0)])
        assert reading is not None
        assert reading.gamma_exposure is None
        assert reading.contracts_used == 0


class TestSkew:
    def test_equal_wing_volatility_is_a_flat_skew(self) -> None:
        reading = _read(
            [
                _quote(5600, is_call=True, oi=100, vol=0.20),
                _quote(4400, is_call=False, oi=100, vol=0.20),
            ]
        )
        assert reading is not None
        assert reading.skew_25delta == pytest.approx(0.0, abs=1e-9)

    def test_dearer_puts_give_a_positive_skew(self) -> None:
        """The normal shape for an equity index, and a steepening is the market
        paying up for crash protection."""
        reading = _read(
            [
                _quote(5600, is_call=True, oi=100, vol=0.18),
                _quote(4400, is_call=False, oi=100, vol=0.28),
            ]
        )
        assert reading is not None
        assert reading.skew_25delta == pytest.approx(0.10, abs=1e-6)

    def test_no_skew_without_both_wings(self) -> None:
        # One side quoted is not a skew, and inventing one would be worse than
        # reporting none.
        reading = _read([_quote(5600, is_call=True, oi=100)])
        assert reading is not None
        assert reading.skew_25delta is None


class TestAtmVolatility:
    def test_it_takes_the_strike_nearest_spot(self) -> None:
        reading = _read(
            [
                _quote(4000, is_call=True, oi=10, vol=0.40),
                _quote(5010, is_call=True, oi=10, vol=0.15),
                _quote(6000, is_call=True, oi=10, vol=0.35),
            ]
        )
        assert reading is not None
        assert reading.atm_iv == pytest.approx(0.15)


class TestUnusableChains:
    def test_none_on_an_empty_chain(self) -> None:
        assert _read([]) is None

    def test_none_without_a_spot_price(self) -> None:
        assert (
            sx.compute_reading(
                as_of=_AS_OF,
                symbol="^SPX",
                spot=0.0,
                quotes=[_quote(5000, is_call=True, oi=10)],
                expiry_days=_DAYS,
            )
            is None
        )

    def test_a_chain_of_broken_quotes_yields_no_readings(self) -> None:
        """Zero prices and absent provider vol leave nothing to solve from."""
        junk = [
            sx.OptionQuote(strike=5000, last_price=0.0, open_interest=10, is_call=True),
            sx.OptionQuote(strike=4500, last_price=0.0, open_interest=10, is_call=False),
        ]
        reading = _read(junk)
        assert reading is not None
        assert reading.gamma_exposure is None
        assert reading.skew_25delta is None
        assert reading.atm_iv is None


class TestExpiryChoice:
    def test_it_picks_the_expiry_nearest_the_target(self) -> None:
        picked = sx._pick_expiry(["2026-08-10", "2026-09-05", "2026-12-18"], _AS_OF)
        assert picked is not None
        assert picked[0] == "2026-09-05"  # 30 days out

    def test_it_skips_the_pinning_window(self) -> None:
        """Expiries within a week say more about the expiry than the market."""
        picked = sx._pick_expiry(["2026-08-07", "2026-09-05"], _AS_OF)
        assert picked is not None
        assert picked[0] == "2026-09-05"

    def test_none_when_everything_is_too_near(self) -> None:
        assert sx._pick_expiry(["2026-08-07", "2026-08-08"], _AS_OF) is None
