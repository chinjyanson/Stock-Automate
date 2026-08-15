"""The market-regime risk factor: what scales it, and what deliberately does not."""

from __future__ import annotations

from datetime import date

from app.services.market_regime import MIN_RISK_FACTOR, MarketRegimeService, RegimeReading


def _reading(**kwargs: object) -> RegimeReading:
    r = RegimeReading(as_of=date(2026, 7, 31))
    for key, value in kwargs.items():
        setattr(r, key, value)
    MarketRegimeService._score(r)
    return r


class TestRiskFactor:
    def test_calm_conditions_leave_risk_untouched(self) -> None:
        r = _reading(vix=15.0, breadth=0.60, sp500_uptrend=True, world_uptrend=True)
        assert r.risk_factor == 1.0
        assert r.posture == "normal"

    def test_each_fast_signal_reduces_risk(self) -> None:
        assert _reading(vix=35.0).risk_factor < 1.0
        assert _reading(breadth=0.20).risk_factor < 1.0
        assert _reading(sp500_uptrend=False).risk_factor < 1.0
        assert _reading(credit_spread=9.0).risk_factor < 1.0
        assert _reading(sp500_realised_vol=0.35).risk_factor < 1.0

    def test_the_worst_case_still_trades(self) -> None:
        """The floor is the whole design: a bad regime trades smaller, not never.

        An overlay that can reach zero stops being a risk control and becomes a
        market-timing bet.
        """
        r = _reading(
            vix=80.0,
            breadth=0.01,
            sp500_uptrend=False,
            world_uptrend=False,
            credit_spread=15.0,
            sp500_realised_vol=0.80,
        )
        assert r.risk_factor == MIN_RISK_FACTOR
        assert r.risk_factor > 0.5
        assert r.posture == "risk-off"

    def test_four_full_signals_reach_the_floor(self) -> None:
        """The calibration invariant, pinned so it is not drifted into.

        Cut sizes were reduced when credit spreads and realised volatility were
        added, precisely so that adding inputs did not quietly make the system
        more defensive. Four of the six confirming is the bar; if a seventh
        signal is ever added, this is the test that has to be re-argued.
        """
        r = _reading(vix=35.0, breadth=0.20, sp500_uptrend=False, credit_spread=9.0)
        assert r.risk_factor == MIN_RISK_FACTOR

    def test_three_full_signals_do_not_reach_the_floor(self) -> None:
        r = _reading(vix=35.0, breadth=0.20, sp500_uptrend=False)
        assert r.risk_factor > MIN_RISK_FACTOR

    def test_a_calm_vix_does_not_mask_a_grinding_selloff(self) -> None:
        """Why realised volatility earns its place next to VIX.

        VIX is the market's *expectation*. A slow, orderly decline can leave it
        untroubled while the tape is plainly deteriorating — realised vol is the
        input that sees that, and the two diverging is exactly the case worth
        catching.
        """
        calm = _reading(vix=15.0, sp500_realised_vol=0.05)
        grinding = _reading(vix=15.0, sp500_realised_vol=0.35)
        assert calm.risk_factor == 1.0
        assert grinding.risk_factor < calm.risk_factor

    def test_valuation_measures_never_move_the_factor(self) -> None:
        """Buffett and the yield curve are alert-only, by explicit decision.

        Both sit at extremes for years; letting them scale risk would mean a
        permanently suppressed multiplier rather than a responsive one — a bot
        that never trades, which is easy to mistake for a cautious bot.
        """
        calm = _reading(vix=15.0, breadth=0.60, sp500_uptrend=True, world_uptrend=True)
        extreme = _reading(
            vix=15.0,
            breadth=0.60,
            sp500_uptrend=True,
            world_uptrend=True,
            buffett_indicator=250.0,
            yield_curve_10y2y=-1.5,
            # Credit and realised vol, by contrast, *do* count — pinned here so
            # the line between context and signal stays explicit.
            credit_spread=3.0,
            sp500_realised_vol=0.10,
        )
        assert extreme.risk_factor == calm.risk_factor == 1.0

    def test_missing_inputs_are_not_treated_as_risk_off(self) -> None:
        """A failed provider call must not look like a bear market."""
        assert _reading().risk_factor == 1.0

    def test_a_single_mild_signal_barely_registers(self) -> None:
        """'Not that significant' was the requirement: one elevated reading
        should trim, not slam the brakes."""
        r = _reading(vix=22.0, breadth=0.60, sp500_uptrend=True, world_uptrend=True)
        assert 0.90 < r.risk_factor < 1.0
