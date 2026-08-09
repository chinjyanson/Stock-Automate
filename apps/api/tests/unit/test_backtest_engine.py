"""Historical replay (§8, §20).

A backtest is a measuring instrument, so the tests that matter are the ones
proving it is not optimistic. Three failure modes account for most flattering
backtests, and each has a test here that would catch it:

  * **Look-ahead** — deciding on a bar using prices from after it.
    `TestNoLookAhead`.
  * **Fills you could not have got** — buying at the close you just read, or
    getting your stop price through a gap. `TestExecutionIsPessimistic`.
  * **Quietly dropping the awkward trades** — the position still open when the
    data ends, which is disproportionately a loser. `TestUnclosedPositions`.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.backtest.engine import (
    BacktestResult,
    BacktestTrade,
    ExitReason,
    PortfolioResult,
    ReplayConfig,
    is_continuous,
    replay,
    worst_daily_ratio,
)
from app.indicators.series import PriceSeries
from app.strategies.mean_reversion import EntryRules, read_entry


def _series(
    closes: list[float],
    *,
    highs: list[float] | None = None,
    lows: list[float] | None = None,
    opens: list[float] | None = None,
) -> PriceSeries:
    c = np.array(closes, dtype=np.float64)
    return PriceSeries(
        open=np.array(opens, dtype=np.float64) if opens else c.copy(),
        high=np.array(highs, dtype=np.float64) if highs else c * 1.01,
        low=np.array(lows, dtype=np.float64) if lows else c * 0.99,
        close=c,
        adjusted_close=np.full(c.size, np.nan),
        volume=np.full(c.size, 500_000.0),
    )


#: Repeated oscillate-then-crash-then-recover cycles, long enough to clear the
#: 260-bar warmup and still leave room for several complete round trips.
def _cyclical(cycles: int = 12) -> list[float]:
    closes: list[float] = []
    for _ in range(cycles):
        closes.extend(100 + (3 if i % 2 else -3) for i in range(30))
        closes.extend([94.0, 88.0, 84.0])  # the dislocation
        closes.extend([88.0, 94.0, 99.0, 101.0])  # the reversion
    return closes


_RULES = EntryRules()
_SHORT_WARMUP = ReplayConfig(warmup_bars=40)


def _trade(entry: float, exit_: float, stop: float, **kw: object) -> BacktestTrade:
    return BacktestTrade(
        entry_index=int(kw.get("entry_index", 0)),
        exit_index=int(kw.get("exit_index", 5)),
        entry_price=entry,
        exit_price=exit_,
        initial_stop=stop,
        entry_score=float(kw.get("entry_score", 0.7)),
        exit_reason=ExitReason(kw.get("exit_reason", ExitReason.TARGET)),
    )


class TestSplitDetection:
    """An unadjusted split is a corporate action, not a price move.

    The store's raw OHLC is uncorrected — only `adjusted_close` is fixed, and the
    replay needs open/high/low too — so a reverse split reads as a genuine move
    and the ATR-derived stop ends up an absurd distance away. This is not
    hypothetical: it produced a +2,489R instrument against roughly -10R from 928
    others, turning a losing rule into an apparent 4.79 profit factor.
    """

    def test_an_ordinary_series_is_continuous(self) -> None:
        assert is_continuous(_series(_cyclical(3)))

    def test_a_reverse_split_is_caught(self) -> None:
        """The real shape that broke the first deep run: 0.0003 -> 20.97."""
        closes = [*(0.0003 for _ in range(40)), *(20.97 for _ in range(40))]
        assert not is_continuous(_series(closes))

    def test_a_forward_split_is_caught_too(self) -> None:
        """Halving is as discontinuous as doubling; the check is symmetric."""
        closes = [*(100.0 for _ in range(40)), *(5.0 for _ in range(40))]
        assert not is_continuous(_series(closes))

    def test_a_violent_but_real_move_is_kept(self) -> None:
        """A 60% single-day crash is a catastrophe, not a split, and the strategy
        should be measured on it rather than quietly excused from it."""
        closes = [*(100.0 for _ in range(40)), 40.0, *(41.0 for _ in range(20))]
        assert is_continuous(_series(closes))

    def test_the_worst_ratio_is_reported_for_diagnosis(self) -> None:
        closes = [10.0, 10.0, 100.0, 100.0]
        assert worst_daily_ratio(_series(closes)) == pytest.approx(10.0)

    def test_a_series_too_short_to_compare_is_not_flagged(self) -> None:
        assert worst_daily_ratio(_series([100.0])) == 1.0


class TestConcentration:
    """One trade must not be able to carry a headline unnoticed."""

    def _result(self, *r_multiples: float) -> BacktestResult:
        return BacktestResult(
            trades=tuple(_trade(100.0, 100.0 + 10.0 * r, 90.0) for r in r_multiples)
        )

    def test_the_median_ignores_a_single_outlier(self) -> None:
        """Mean and median disagreeing wildly is the tell.

        Four losses and one enormous win averages positive; the median says the
        typical trade lost. Reporting only the mean is how a broken series passes
        for an edge.
        """
        result = self._result(-1.0, -1.0, -1.0, -1.0, 500.0)
        assert result.expectancy_r > 90
        assert result.median_r == pytest.approx(-1.0)

    def test_the_largest_trade_share_flags_the_outlier(self) -> None:
        assert self._result(-1.0, -1.0, -1.0, -1.0, 500.0).largest_trade_share > 0.99

    def test_an_even_spread_is_not_flagged(self) -> None:
        assert self._result(1.0, -1.0, 1.5, -1.0, 0.5).largest_trade_share < 0.4


class TestRMultiples:
    def test_a_trade_that_makes_what_it_risked_is_one_r(self) -> None:
        assert _trade(100.0, 110.0, 90.0).r_multiple == pytest.approx(1.0)

    def test_a_trade_stopped_out_is_minus_one_r(self) -> None:
        assert _trade(100.0, 90.0, 90.0).r_multiple == pytest.approx(-1.0)

    def test_r_is_measured_against_the_initial_stop_not_a_trailed_one(self) -> None:
        """A stop that ratcheted up changes the outcome, not the wager.

        Measuring against the trailed stop would inflate every winner that
        trailed, which is most of them — the single easiest way to make a
        mediocre system look excellent.
        """
        trade = _trade(100.0, 112.0, 90.0)
        assert trade.risk == pytest.approx(10.0)
        assert trade.r_multiple == pytest.approx(1.2)

    def test_a_degenerate_zero_risk_trade_scores_nothing_rather_than_dividing(self) -> None:
        assert _trade(100.0, 120.0, 100.0).r_multiple == 0.0


class TestAggregates:
    def _result(self, *r_multiples: float) -> BacktestResult:
        # entry 100, stop 90 → risk 10, so exit = 100 + 10r
        return BacktestResult(
            trades=tuple(_trade(100.0, 100.0 + 10.0 * r, 90.0) for r in r_multiples)
        )

    def test_expectancy_is_the_mean_r(self) -> None:
        assert self._result(1.0, -1.0, 2.0).expectancy_r == pytest.approx(2.0 / 3.0)

    def test_win_rate_and_expectancy_are_different_questions(self) -> None:
        """One win in four can still be a good system, and this proves the
        instrument reports both rather than conflating them."""
        result = self._result(4.0, -1.0, -1.0, -1.0)
        assert result.win_rate == pytest.approx(0.25)
        assert result.expectancy_r == pytest.approx(0.25)

    def test_profit_factor_is_gross_win_over_gross_loss(self) -> None:
        assert self._result(3.0, -1.0, -1.0).profit_factor == pytest.approx(1.5)

    def test_profit_factor_is_none_rather_than_infinite_when_nothing_lost(self) -> None:
        assert self._result(1.0, 2.0).profit_factor is None

    def test_max_drawdown_is_peak_to_trough_not_worst_trade(self) -> None:
        """Three consecutive -1R losses after a +2R peak is a 3R drawdown."""
        assert self._result(2.0, -1.0, -1.0, -1.0).max_drawdown_r == pytest.approx(3.0)

    def test_an_empty_result_reports_zeroes_rather_than_dividing_by_zero(self) -> None:
        empty = BacktestResult()
        assert empty.trade_count == 0
        assert empty.win_rate == 0.0
        assert empty.expectancy_r == 0.0
        assert empty.max_drawdown_r == 0.0


class TestNoLookAhead:
    def test_a_decision_never_sees_a_price_after_it(self) -> None:
        """Truncating the future must not change any decision taken before it.

        This is *the* backtest test. If the replay peeked, appending bars would
        retroactively alter earlier entries — which is exactly what look-ahead
        looks like from the outside, and is otherwise invisible.
        """
        closes = _cyclical()
        full = replay(_series(closes), _RULES, _SHORT_WARMUP)
        truncated = replay(_series(closes[:-30]), _RULES, _SHORT_WARMUP)

        # Every trade the shorter run took must appear identically in the longer
        # one — the extra data may add trades at the end, never change earlier ones.
        assert truncated.trade_count > 0
        for short, long in zip(truncated.trades, full.trades, strict=False):
            if short.exit_reason is ExitReason.UNCLOSED:
                break  # the final open trade legitimately resolves differently
            assert short.entry_index == long.entry_index
            assert short.entry_price == pytest.approx(long.entry_price)
            assert short.exit_price == pytest.approx(long.exit_price)

    def test_head_gives_the_rules_only_the_past(self) -> None:
        closes = _cyclical(2)
        series = _series(closes)
        at_bar = read_entry(series.head(120), _RULES)
        # Rewriting everything after bar 119 cannot change what bar 119 knew.
        mutated = list(closes)
        for i in range(120, len(mutated)):
            mutated[i] = 1_000.0
        after = read_entry(_series(mutated).head(120), _RULES)
        assert at_bar is not None and after is not None
        assert at_bar.score == pytest.approx(after.score)


class TestExecutionIsPessimistic:
    def test_entries_fill_at_the_next_open_not_the_signal_close(self) -> None:
        """The strategy runs at 22:30 on closed candles; it cannot buy at the
        close it just read. Filling there is worth free money that is not real."""
        closes = _cyclical(4)
        series = _series(closes)
        result = replay(series, _RULES, _SHORT_WARMUP)
        assert result.trade_count > 0
        for trade in result.trades:
            assert trade.entry_price == pytest.approx(float(series.open[trade.entry_index]))

    def test_a_gap_through_the_stop_fills_at_the_open(self) -> None:
        """You do not get the price you asked for in a gap.

        A bar that opens far below the resting stop fills there, so the loss is
        bigger than 1R. Assuming the stop price would be honoured is a standard
        and material over-statement.
        """
        # Build a setup that enters, then gaps violently down.
        closes = [*(100 + (3 if i % 2 else -3) for i in range(60)), 94.0, 88.0, 84.0, 50.0, 49.0]
        opens = [*closes[:-2], 50.0, 49.0]
        lows = [c * 0.99 for c in closes[:-2]] + [49.0, 48.0]
        series = _series(closes, opens=opens, lows=lows)
        result = replay(series, _RULES, ReplayConfig(warmup_bars=40))

        stopped = [t for t in result.trades if t.exit_reason is ExitReason.STOP]
        assert stopped, "expected the gap to trigger the stop"
        worst = min(stopped, key=lambda t: t.r_multiple)
        assert worst.exit_price < worst.initial_stop
        assert worst.r_multiple < -1.0

    def test_stops_are_checked_before_targets(self) -> None:
        """A daily bar cannot say which came first, so assume the worse one."""
        closes = _cyclical(4)
        result = replay(_series(closes), _RULES, _SHORT_WARMUP)
        # No trade may record a target exit on a bar whose low broke its stop;
        # if the ordering were reversed some would.
        assert all(
            t.exit_reason is not ExitReason.TARGET or t.exit_price > t.initial_stop
            for t in result.trades
        )


class TestUnclosedPositions:
    def test_a_position_open_at_the_end_is_recorded_not_dropped(self) -> None:
        """Dropping it flatters a system that holds its losers.

        A rule whose winners close quickly and whose losers are still open when
        the sample ends looks wonderful if the open ones simply vanish.
        """
        # Enter near the end and never recover.
        closes = [*(100 + (3 if i % 2 else -3) for i in range(60)), 94.0, 88.0, 84.0, 83.0, 82.5]
        result = replay(_series(closes), _RULES, ReplayConfig(warmup_bars=40))
        assert any(t.exit_reason is ExitReason.UNCLOSED for t in result.trades)


class TestExits:
    def test_the_time_stop_closes_a_position_that_never_reverts(self) -> None:
        closes = [*(100 + (3 if i % 2 else -3) for i in range(60)), 94.0, 88.0, *([84.0] * 40)]
        capped = replay(_series(closes), _RULES, ReplayConfig(warmup_bars=40, max_holding_bars=5))
        assert any(t.exit_reason is ExitReason.TIME for t in capped.trades)

    def test_trailing_only_ever_raises_the_stop(self) -> None:
        """Mirrors StopService, which ratchets up and never down."""
        closes = _cyclical(6)
        trailed = replay(_series(closes), _RULES, ReplayConfig(warmup_bars=40, trail_stops=True))
        fixed = replay(_series(closes), _RULES, ReplayConfig(warmup_bars=40, trail_stops=False))
        # Trailing can only close trades earlier or at the same time, never later.
        assert trailed.trade_count >= fixed.trade_count

    def test_a_reverting_stock_exits_at_the_target(self) -> None:
        result = replay(_series(_cyclical(6)), _RULES, _SHORT_WARMUP)
        assert result.exit_breakdown().get("target", 0) > 0


class TestRulesAreShared:
    def test_the_replay_uses_the_live_entry_threshold(self) -> None:
        """Raising the threshold must reduce trades, or the rules are not shared."""
        closes = _cyclical(8)
        loose = replay(_series(closes), EntryRules(entry_threshold=0.50), _SHORT_WARMUP)
        strict = replay(_series(closes), EntryRules(entry_threshold=0.95), _SHORT_WARMUP)
        assert loose.trade_count > strict.trade_count

    def test_the_trend_gate_reaches_the_replay(self) -> None:
        closes = _cyclical(10)
        permissive = replay(_series(closes), EntryRules(trend_slope_min=-1.0), _SHORT_WARMUP)
        demanding = replay(_series(closes), EntryRules(trend_slope_min=0.05), _SHORT_WARMUP)
        assert permissive.trade_count > demanding.trade_count

    def test_every_recorded_entry_score_cleared_the_threshold(self) -> None:
        rules = EntryRules(entry_threshold=0.65)
        result = replay(_series(_cyclical(8)), rules, _SHORT_WARMUP)
        assert result.trade_count > 0
        assert all(t.entry_score >= rules.entry_threshold for t in result.trades)


class TestWarmup:
    def test_the_default_warmup_covers_the_trend_gate(self) -> None:
        """Below 221 bars the slope is unmeasurable and its gate reads as pass.

        Starting earlier would replay an *ungated* strategy over the first
        stretch and pool it with the gated one — two different strategies
        reported as a single number.
        """
        assert _RULES.preferred_bars >= 221
        short = _series(_cyclical(2))  # ~74 bars, under the default warmup
        assert replay(short, _RULES).trade_count == 0

    def test_bars_replayed_is_reported(self) -> None:
        result = replay(_series(_cyclical(8)), _RULES, _SHORT_WARMUP)
        assert result.bars_replayed > 0


class TestPortfolio:
    def test_pooling_preserves_every_trade(self) -> None:
        a = replay(_series(_cyclical(6)), _RULES, _SHORT_WARMUP)
        b = replay(_series(_cyclical(8)), _RULES, _SHORT_WARMUP)
        pooled = PortfolioResult({"A": a, "B": b}).combined
        assert pooled.trade_count == a.trade_count + b.trade_count
        assert pooled.total_r == pytest.approx(a.total_r + b.total_r)
