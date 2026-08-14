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


class TestScaleFreeTilt:
    """The reading that survives the `^SPX` -> `SPY` fallback.

    This is the class that matters most for the index history, because the
    corruption it guards against is permanent. The chain is read from `^SPX`
    when it answers and `SPY` when it does not, and a gamma *exposure* goes as
    `open interest x spot` — so the two proxies report several-fold different
    numbers for identical positioning. Accumulated into one series that step is
    indistinguishable from a regime shift, and a model fitted on it would learn
    the fallback schedule.

    Worth being precise about the size, because the intuitive answer is wrong.
    Gamma itself goes as `1/S`, so the `S^2` in the exposure formula does not
    survive: one factor of `S` cancels. The step is therefore the contract-count
    ratio over ten, a few-fold effect rather than the hundred-fold one the `S^2`
    suggests — small enough to pass for a market move, which is precisely what
    makes it dangerous rather than obvious.
    """

    #: SPY's option open interest against SPX's, in contracts. Nowhere near
    #: the ten-times ratio that would coincidentally cancel the spot factor.
    SCALE_SPY = 40.0

    @staticmethod
    def _chain(spot: float, scale: float) -> list[sx.OptionQuote]:
        """The same positioning, expressed at a different index level.

        `scale` multiplies open interest while spot and the strikes move
        independently, which is what changing proxy actually does: SPY trades
        near a tenth of SPX's level and carries far more contracts.

        The exact arithmetic matters here. Gamma goes as `1/S`, so a gamma
        *exposure* — which multiplies by `S^2` — goes as `open interest x S`,
        not as `S^2`. A proxy switch therefore moves the reading by the ratio
        of contract counts divided by ten, and only an open interest exactly
        ten times larger would leave it unchanged. `SCALE_SPY` below is set to a
        realistic contract ratio rather than that knife edge.
        """
        t = _DAYS / 365.0
        quotes = []
        for moneyness, oi_call, oi_put in (
            (0.95, 800.0, 1500.0),
            (1.0, 2000.0, 2000.0),
            (1.05, 1200.0, 600.0),
        ):
            strike = spot * moneyness
            for is_call, oi in ((True, oi_call), (False, oi_put)):
                price = om.bs_price(spot, strike, t, 0.20, is_call=is_call)
                assert price is not None
                quotes.append(
                    sx.OptionQuote(
                        strike=strike,
                        last_price=price,
                        open_interest=oi * scale,
                        is_call=is_call,
                        provider_iv=0.20,
                    )
                )
        return quotes

    def _reading(self, spot: float, scale: float, symbol: str) -> sx.IndexOptionsReading:
        reading = sx.compute_reading(
            as_of=_AS_OF,
            symbol=symbol,
            spot=spot,
            quotes=self._chain(spot, scale),
            expiry_days=_DAYS,
        )
        assert reading is not None
        return reading

    def test_the_raw_exposure_is_not_comparable_across_proxies(self) -> None:
        """The bug being fixed, asserted so it cannot be called a small effect."""
        spx = self._reading(5000.0, 1.0, "^SPX")
        spy = self._reading(500.0, self.SCALE_SPY, "SPY")
        assert spx.gamma_exposure is not None and spy.gamma_exposure is not None
        # Identical positioning, four-fold different number: 40x the contracts
        # against a tenth of the level. Small enough to look like a market move
        # and large enough to dominate anything fitted on the series.
        assert abs(spy.gamma_exposure / spx.gamma_exposure) == pytest.approx(4.0, rel=1e-6)

    def test_the_tilt_is_identical_across_proxies(self) -> None:
        """And the fix: a net-to-gross ratio cancels both spot and contract count."""
        spx = self._reading(5000.0, 1.0, "^SPX")
        spy = self._reading(500.0, self.SCALE_SPY, "SPY")
        assert spx.gamma_tilt is not None and spy.gamma_tilt is not None
        assert spx.gamma_tilt == pytest.approx(spy.gamma_tilt, abs=1e-9)

    def test_the_charm_tilt_is_identical_across_proxies(self) -> None:
        spx = self._reading(5000.0, 1.0, "^SPX")
        spy = self._reading(500.0, self.SCALE_SPY, "SPY")
        assert spx.charm_tilt is not None and spy.charm_tilt is not None
        assert spx.charm_tilt == pytest.approx(spy.charm_tilt, abs=1e-9)

    def test_the_tilt_is_unchanged_by_open_interest_alone(self) -> None:
        base = self._reading(5000.0, 1.0, "^SPX")
        busier = self._reading(5000.0, 37.0, "^SPX")
        assert base.gamma_tilt is not None and busier.gamma_tilt is not None
        assert base.gamma_tilt == pytest.approx(busier.gamma_tilt, abs=1e-9)

    def test_the_tilt_is_bounded(self) -> None:
        reading = self._reading(5000.0, 1.0, "^SPX")
        assert reading.gamma_tilt is not None and reading.charm_tilt is not None
        assert -1.0 <= reading.gamma_tilt <= 1.0
        assert -1.0 <= reading.charm_tilt <= 1.0

    def test_calls_only_tilts_fully_long(self) -> None:
        reading = _read([_quote(5000, is_call=True, oi=1000), _quote(5100, is_call=True, oi=500)])
        assert reading is not None
        assert reading.gamma_tilt == pytest.approx(1.0)

    def test_puts_only_tilts_fully_short(self) -> None:
        reading = _read([_quote(5000, is_call=False, oi=1000), _quote(4900, is_call=False, oi=500)])
        assert reading is not None
        assert reading.gamma_tilt == pytest.approx(-1.0)

    def test_tilt_is_none_when_nothing_carried_open_interest(self) -> None:
        reading = _read([_quote(5000, is_call=True, oi=0.0)])
        assert reading is not None
        assert reading.gamma_tilt is None
        assert reading.charm_tilt is None


class TestCharmExposure:
    def test_it_is_produced_alongside_gamma(self) -> None:
        reading = _read([_quote(5100, is_call=True, oi=1000)])
        assert reading is not None
        assert reading.charm_exposure is not None
        assert reading.charm_tilt is not None

    def test_it_follows_the_same_dealer_sign_convention_as_gamma(self) -> None:
        """One book, assembled one way — not two differently-signed readings."""
        call = _read([_quote(5100, is_call=True, oi=1000)])
        put = _read([_quote(5100, is_call=False, oi=1000)])
        assert call is not None and put is not None
        assert call.charm_exposure is not None and put.charm_exposure is not None
        # Charm is identical for a call and a put at the same strike without a
        # dividend, so only the assumed dealer side separates these.
        assert call.charm_exposure == pytest.approx(-put.charm_exposure)
