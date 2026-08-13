"""Buy-back rules, and the two claims their modules make in prose.

The rules themselves are small enough that the interesting tests are about the
*shapes* they are supposed to have — particularly the ladder's blind spot, which
is the flaw that motivated the whole comparison and must not be quietly fixed by
accident.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.backtest import reentry
from app.backtest.overlay_pipeline import rsi_series
from app.indicators.functions import relative_strength_index


def _context(**overrides: float | int | bool) -> reentry.Context:
    base: dict = {
        "days_out": 5,
        "price": 95.0,
        "exit_price": 100.0,
        "low_since": 92.0,
        "warning": False,
        "rsi": 45.0,
        "sma_ratio": 1.0,
        "volatility": 0.20,
        "volatility_at_exit": 0.30,
        "up_streak": 0,
    }
    base.update(overrides)
    return reentry.Context(**base)


class TestTheLaddersBlindSpot:
    """A rally after the alarm must leave the plain ladder stuck.

    This is the defect the episode ledger traced every large loss to. It is a
    property of the rule, not a bug, so it is pinned: if someone "fixes" the
    ladder in place, the comparison between rules silently stops meaning what it
    says.
    """

    def test_a_rally_leaves_the_ladder_at_zero(self) -> None:
        risen = _context(price=112.0, low_since=100.0)
        assert reentry.ladder(0.10)(risen) == 0.0

    def test_the_bounce_rule_is_the_one_that_responds(self) -> None:
        risen = _context(price=112.0, low_since=100.0)
        assert reentry.bounce(0.05)(risen) == 1.0
        assert reentry.ladder_bounce(0.05)(risen) == 1.0


class TestRulesStayInRange:
    def test_every_rule_returns_a_fraction(self) -> None:
        contexts = [
            _context(),
            _context(price=40.0, low_since=40.0),
            _context(price=200.0, low_since=90.0),
            _context(warning=True, rsi=10.0, sma_ratio=0.8, volatility=0.9),
            _context(exit_price=0.0, low_since=0.0, volatility_at_exit=0.0),
        ]
        for name, (factory, params) in reentry.REGISTRY.items():
            for param in params:
                for context in contexts:
                    value = factory(param)(context)
                    assert 0.0 <= value <= 1.0, f"{name}({param}) returned {value}"


class TestLadderIsProportional:
    def test_deeper_falls_buy_more_back(self) -> None:
        rule = reentry.ladder(0.10)
        assert rule(_context(price=100.0)) == pytest.approx(0.0)
        assert rule(_context(price=95.0)) == pytest.approx(0.5)
        assert rule(_context(price=90.0)) == pytest.approx(1.0)
        assert rule(_context(price=80.0)) == pytest.approx(1.0)


class TestRsiSeriesMatchesProduction:
    """`rsi_series` claims to equal the production RSI at every bar.

    It is an unrolled form of the same Wilder recursion, so this is an equality
    and not an approximation — which is worth pinning, because a re-entry rule
    reading a subtly different RSI from the one the rest of the system uses
    would be a difference nobody would ever notice by reading the code.
    """

    def test_matches_at_every_checked_bar(self) -> None:
        rng = np.random.default_rng(11)
        close = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, 400)))
        series = rsi_series(close)

        for i in (20, 50, 137, 250, 399):
            expected = relative_strength_index(close[: i + 1])
            assert expected is not None
            assert series[i] == pytest.approx(expected)

    def test_is_undefined_before_enough_bars(self) -> None:
        close = np.linspace(100.0, 110.0, 10)
        assert np.isnan(rsi_series(close)).all()
