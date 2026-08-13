"""The scanner backtest's load-bearing claims, pinned.

Three of them, each corresponding to a way this harness could produce a
confident and completely wrong number:

  * it scores a 500-bar window and claims that equals scoring everything;
  * it claims a decision cannot see the bar it is executed on;
  * it claims the arithmetic mean ranks volatility rather than skill, which is
    the confound that first made `quality` look like a strong backwards signal.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.backtest import scanner_replay as replay
from app.backtest.universe import Panel
from app.scanner import scoring


def _panel(days: int = 1200, symbols: int = 4, seed: int = 7) -> Panel:
    """A synthetic panel: geometric random walks on one shared calendar."""
    rng = np.random.default_rng(seed)
    dates = np.datetime64("2010-01-01", "D") + np.arange(days)
    steps = rng.normal(0.0004, 0.015, size=(days, symbols))
    close = 100.0 * np.exp(np.cumsum(steps, axis=0))
    names = tuple(f"S{i}" for i in range(symbols))
    return Panel(
        dates=dates,
        symbols=names,
        sectors=dict.fromkeys(names, "Industrials"),
        fields={
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "adjusted_close": close,
            "volume": np.full((days, symbols), 5_000_000.0),
        },
    )


class TestWindowIsEnough:
    """500 bars must score identically to the whole history.

    Every lookback in `indicators.functions` is bounded at 252 bars, so this is
    a true statement about the code rather than a tolerance — and if someone
    adds an unbounded indicator, the backtest would silently start disagreeing
    with production. This is the test that would catch it.
    """

    def test_window_matches_full_history(self) -> None:
        panel = _panel(days=1500)
        cut = 1400
        windowed = replay._slice(panel, 0, cut, window=replay.WINDOW)
        entire = replay._slice(panel, 0, cut, window=cut + 1)
        assert windowed is not None and entire is not None
        assert entire.length > windowed.length

        a = scoring.score_series(windowed)
        b = scoring.score_series(entire)
        assert a.score == pytest.approx(b.score)
        for name in scoring.GROUP_NAMES:
            assert (a.groups[name].score is None) == (b.groups[name].score is None)
            if a.groups[name].score is not None:
                assert a.groups[name].score == pytest.approx(b.groups[name].score)


class TestNoLookAhead:
    """A decision at bar *i* may not see bar *i + 1*, and is traded at it."""

    def test_slice_ends_at_the_decision_bar(self) -> None:
        panel = _panel()
        cut = 900
        series = replay._slice(panel, 0, cut)
        assert series is not None
        assert series.close[-1] == panel.fields["close"][cut, 0]
        assert panel.fields["close"][cut + 1, 0] not in series.close

    def test_a_future_spike_cannot_change_the_score(self) -> None:
        panel = _panel()
        cut = 900
        before = scoring.score_series(replay._slice(panel, 0, cut)).score  # type: ignore[arg-type]

        tampered = {k: v.copy() for k, v in panel.fields.items()}
        for name in ("open", "high", "low", "close", "adjusted_close"):
            tampered[name][cut + 1 :, 0] *= 5.0
        future = Panel(
            dates=panel.dates,
            symbols=panel.symbols,
            sectors=panel.sectors,
            fields=tampered,
        )
        after = scoring.score_series(replay._slice(future, 0, cut)).score  # type: ignore[arg-type]
        assert before == pytest.approx(after)

    def test_the_book_is_traded_one_bar_after_the_decision(self) -> None:
        panel = _panel()
        # Hold column 0 from a decision at bar 500; the first return earned must
        # be the one from bar 501 to 502, never 500 to 501.
        result = replay.simulate(panel, [(500, [0])], name="one name", cost=0.0)
        close = panel.fields["adjusted_close"][:, 0]
        assert result.dates[0] == panel.dates[501]
        assert result.equity[0] == pytest.approx(1.0)
        assert result.equity[1] == pytest.approx(close[502] / close[501])


class TestTheMeanRanksVolatility:
    """The confound `by_decile` exists to expose.

    Two buckets with identical compounding, one far more volatile: the
    arithmetic mean prefers the volatile one and the log mean does not. A signal
    correlated with volatility therefore "predicts" on the mean while paying an
    investor nothing, which is exactly what `quality` did.
    """

    def test_equal_growth_different_volatility(self) -> None:
        calm = np.tile([0.02, -0.02], 500)
        wild = np.tile([0.40, -0.2857142857142857], 500)  # same product per pair
        scores = np.concatenate([np.zeros(calm.size), np.ones(wild.size)])
        forward = np.concatenate([calm, wild])

        rows = replay.by_decile(scores, forward, buckets=2)
        assert len(rows) == 2
        low, high = rows

        assert high["mean"] > low["mean"] + 0.02
        assert high["growth"] == pytest.approx(low["growth"], abs=2e-3)


class TestRebalanceDays:
    def test_month_ends_only(self) -> None:
        panel = _panel(days=400)
        days = replay.rebalance_days(panel.dates)
        months = panel.dates.astype("datetime64[M]")
        for i in days:
            assert months[i] != months[i + 1], "not the last bar of its month"
