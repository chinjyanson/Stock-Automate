"""Historical replay of the mean-reversion strategy (§8).

Every threshold in `strategies/mean_reversion.py` was reasoned about and none of
them has ever been measured. This is what turns them from arguments into
numbers: replay the strategy bar by bar over stored candles, record the trades it
would have taken, and report whether the entry has an edge.

Pure and I/O-free, like `risk/stress.py` and `scanner/scoring.py`: everything it
needs is passed in, so it is deterministic and testable without a database.

**It calls `read_entry`, the same function the live strategy calls, on the same
`EntryRules` object.** That is not a convenience. A backtest that reimplements
the rules measures a strategy that does not trade, and the divergence is silent
because both halves look correct in isolation — which is the single most common
way a backtest comes to be confidently wrong.

Results are in **R multiples**, not currency. One R is the distance from entry to
the initial stop, so a trade that reaches its target having risked 2.30 to make
3.10 scores +1.35R whatever the position size was. That deliberately sidesteps
position sizing, which is the risk engine's business and depends on the whole
book — and it is the right unit anyway, because expectancy in R is exactly what
says whether an entry rule is worth running.

What this **cannot** tell you, and it matters:

  * **No portfolio.** Every instrument is replayed independently, so the caps,
    the correlation reductions and the whole-book stress test never apply. It
    measures the *entry rule*, not the system.
  * **PEAD is absent.** The earnings table only holds recent dates, so a
    historical replay has no drift reading and the veto never fires. The live
    strategy is therefore slightly *more* selective than this.
  * **No insider exits**, for the same reason.
  * **Survivorship.** It replays instruments that exist in the store today.
    Anything delisted is not there to lose money in the sample.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from enum import StrEnum

from app.indicators.series import PriceSeries
from app.strategies.mean_reversion import EntryRules, read_entry

#: Multiple of ATR the risk engine places the stop at. Mirrored here rather than
#: read from `RiskConfiguration` so the replay stays pure; pass the live value in
#: if it has been tuned away from the default.
DEFAULT_ATR_STOP_MULTIPLIER = 2.0

#: A single-bar close ratio beyond this is a corporate action or bad data, not a
#: price move, and the series is not a continuous record of what a holder would
#: have experienced.
#:
#: Four is deliberately loose. A real equity essentially never doubles and
#: doubles again in a day; even a 3x leveraged ETP tops out well inside it. The
#: bound exists to catch unadjusted splits, and the first run of this harness
#: shows why it has to: a 3x inverse ETP with an unadjusted reverse split
#: (0.0003 -> 20.97 in one bar, a 917-million-fold range across its history)
#: produced +2,489R on its own against roughly -10R from the other 928
#: instruments combined, and turned a losing rule into an apparent 4.79 profit
#: factor. Nothing about that was a trading result.
MAX_DAILY_PRICE_RATIO = 4.0


def worst_daily_ratio(series: PriceSeries) -> float:
    """Largest single-bar price ratio in the series, always >= 1.

    Detects unadjusted splits, which the store's raw OHLC carries because only
    `adjusted_close` is corrected and the replay needs open/high/low too. Returns
    1.0 for a series with fewer than two usable bars — nothing to compare.
    """
    closes = series.close
    if closes.size < 2:
        return 1.0
    worst = 1.0
    for previous, current in itertools.pairwise(closes):
        a, b = float(previous), float(current)
        if a <= 0 or b <= 0:
            continue
        ratio = b / a
        worst = max(worst, ratio, 1.0 / ratio)
    return worst


def is_continuous(series: PriceSeries, *, max_ratio: float = MAX_DAILY_PRICE_RATIO) -> bool:
    """Whether the series can be read as one uninterrupted price record.

    Excluding a discontinuous instrument is the honest response rather than
    trying to repair it: a split factor inferred from the jump would be a guess,
    and a wrong guess produces plausible-looking trades instead of obviously
    broken ones.
    """
    return worst_daily_ratio(series) <= max_ratio


class ExitReason(StrEnum):
    TARGET = "target"  # recovered to the middle band, as intended
    STOP = "stop"  # the ATR stop was hit
    TIME = "time"  # held longer than the cap allows
    UNCLOSED = "unclosed"  # still open when the data ran out


@dataclass(frozen=True, slots=True)
class BacktestTrade:
    """One round trip, priced in R."""

    entry_index: int
    exit_index: int
    entry_price: float
    exit_price: float
    initial_stop: float
    entry_score: float
    exit_reason: ExitReason
    #: Distance to the middle-band target divided by distance to the stop, as
    #: seen at entry. Recorded rather than inferred: extrapolating it from a
    #: handful of fixtures produced a confident and wrong claim about where this
    #: strategy's expectancy was going.
    reward_risk: float = 0.0

    @property
    def bars_held(self) -> int:
        return self.exit_index - self.entry_index

    @property
    def risk(self) -> float:
        return self.entry_price - self.initial_stop

    @property
    def r_multiple(self) -> float:
        """Profit as a multiple of what was risked at entry.

        Measured against the *initial* stop, not the trailed one: R is what the
        trade put at stake when it was taken, and a stop that later ratcheted
        up changes the outcome, not the wager.
        """
        return (self.exit_price - self.entry_price) / self.risk if self.risk > 0 else 0.0

    @property
    def is_win(self) -> bool:
        return self.r_multiple > 0


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Aggregate outcome of one replay."""

    trades: tuple[BacktestTrade, ...] = ()
    bars_replayed: int = 0
    #: Bars where the rules produced no opinion at all (too little history, a
    #: flat window). Reported so a run that found nothing can be told apart from
    #: a run that could not look.
    bars_unreadable: int = 0

    @property
    def trade_count(self) -> int:
        return len(self.trades)

    @property
    def wins(self) -> int:
        return sum(1 for t in self.trades if t.is_win)

    @property
    def win_rate(self) -> float:
        return self.wins / self.trade_count if self.trades else 0.0

    @property
    def total_r(self) -> float:
        return sum(t.r_multiple for t in self.trades)

    @property
    def expectancy_r(self) -> float:
        """Average R per trade — the number that says whether this is worth running.

        Positive expectancy with a low win rate is normal and fine for a system
        with a fixed target and a fixed stop; the win rate on its own says
        nothing without the payoff beside it.
        """
        return self.total_r / self.trade_count if self.trades else 0.0

    @property
    def median_r(self) -> float:
        """Middle trade by R. Read this beside `expectancy_r`, always.

        The mean is what compounds and is therefore the number that matters, but
        it is also the number a single broken series can capture — one
        split-corrupted instrument once contributed +2,489R against roughly -10R
        from 928 others. When mean and median disagree wildly, the mean is
        describing one trade, not a strategy.
        """
        if not self.trades:
            return 0.0
        ordered = sorted(t.r_multiple for t in self.trades)
        mid = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[mid]
        return (ordered[mid - 1] + ordered[mid]) / 2

    @property
    def largest_trade_share(self) -> float:
        """Fraction of gross absolute R contributed by the single biggest trade.

        A concentration alarm. Anything much above a few percent means the
        headline is an anecdote.
        """
        gross = sum(abs(t.r_multiple) for t in self.trades)
        if gross <= 0:
            return 0.0
        return max(abs(t.r_multiple) for t in self.trades) / gross

    @property
    def avg_bars_held(self) -> float:
        return sum(t.bars_held for t in self.trades) / self.trade_count if self.trades else 0.0

    @property
    def avg_win_r(self) -> float:
        wins = [t.r_multiple for t in self.trades if t.is_win]
        return sum(wins) / len(wins) if wins else 0.0

    @property
    def avg_loss_r(self) -> float:
        """Mean loss as a positive number, so it reads beside `avg_win_r`."""
        losses = [t.r_multiple for t in self.trades if not t.is_win]
        return -sum(losses) / len(losses) if losses else 0.0

    def r_by_exit(self) -> dict[str, tuple[int, float]]:
        """(count, mean R) per exit reason — the exit-side diagnostic.

        Where the leakage is. An entry offering ~1.3:1 at a 50% win rate should
        return about +0.15R; if the realised figure is near zero, the difference
        is being given back somewhere in this table rather than in the entry.
        """
        buckets: dict[str, list[float]] = {}
        for trade in self.trades:
            buckets.setdefault(trade.exit_reason.value, []).append(trade.r_multiple)
        return {k: (len(v), sum(v) / len(v)) for k, v in sorted(buckets.items())}

    @property
    def profit_factor(self) -> float | None:
        """Gross winning R over gross losing R. None when nothing lost."""
        losses = -sum(t.r_multiple for t in self.trades if not t.is_win)
        gains = sum(t.r_multiple for t in self.trades if t.is_win)
        return gains / losses if losses > 0 else None

    @property
    def equity_curve_r(self) -> tuple[float, ...]:
        """Cumulative R after each closed trade, in order."""
        curve: list[float] = []
        running = 0.0
        for trade in self.trades:
            running += trade.r_multiple
            curve.append(running)
        return tuple(curve)

    @property
    def max_drawdown_r(self) -> float:
        """Deepest peak-to-trough fall of the R curve. Positive == a loss."""
        peak = 0.0
        worst = 0.0
        running = 0.0
        for trade in self.trades:
            running += trade.r_multiple
            peak = max(peak, running)
            worst = max(worst, peak - running)
        return worst

    def exit_breakdown(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for trade in self.trades:
            counts[trade.exit_reason.value] = counts.get(trade.exit_reason.value, 0) + 1
        return counts


@dataclass(slots=True)
class _OpenPosition:
    entry_index: int
    entry_price: float
    initial_stop: float
    stop: float
    entry_score: float
    reward_risk: float = 0.0

    @property
    def risk_distance(self) -> float:
        return self.entry_price - self.initial_stop


@dataclass(frozen=True, slots=True)
class ReplayConfig:
    """How the replay models execution, as distinct from the entry rules."""

    atr_stop_multiplier: float = DEFAULT_ATR_STOP_MULTIPLIER
    #: 0 disables the time stop, matching `RiskConfiguration.max_holding_days`.
    max_holding_bars: int = 0
    trail_stops: bool = True
    #: Hold for exactly this many bars, ignoring the band target entirely. The
    #: stop still applies — this is "wait for the move", not "wait no matter what".
    #:
    #: Exists because the measured edge builds over ~20 days (RSI <= 30 precedes
    #: +0.77% at 5d, +1.79% at 10d, +3.08% at 20d) while the band exit closes the
    #: position after ~7. The strategy was collecting roughly a third of a real
    #: move and calling it a day.
    hold_bars: int | None = None
    #: Take profit at `entry + fixed_target_r * risk`, fixed at entry, instead of
    #: at the live middle band. None keeps the band.
    #:
    #: The band is a 20-day moving average, so it follows price *down* while a
    #: position waits to revert — the target drifts toward the entry and a winner
    #: is paid less than the setup promised. Measured: target exits average
    #: +0.73R against a structural ~1.3R at entry, which is most of the reason
    #: expectancy sits at zero. Freezing the target is the direct fix.
    fixed_target_r: float | None = None
    #: Bars to skip before trading. Defaults to `EntryRules.preferred_bars`,
    #: because below that the 200-day slope cannot be computed and its gate reads
    #: as satisfied — so an earlier start would silently replay a *different*,
    #: ungated strategy over the first stretch and pool the results.
    warmup_bars: int | None = None


def replay(
    series: PriceSeries,
    rules: EntryRules,
    config: ReplayConfig | None = None,
) -> BacktestResult:
    """Walk `series` bar by bar, taking every trade the rules would have taken.

    Execution is modelled the way the live system actually behaves, because the
    optimistic alternatives are where backtests earn their bad reputation:

      * **Entries fill at the next bar's open.** The strategy runs at 22:30 on
        closed candles and submits a market order, so it cannot buy at the close
        it just read. Filling at that close is the classic look-ahead, and it is
        worth several points of annual return that do not exist.
      * **Stops fill intrabar**, because they rest at the broker as real orders.
        A bar that gapped straight through fills at the open, not at the stop
        price — you do not get the price you asked for in a gap.
      * **Targets fill at the next open**, like entries: the middle-band exit is
        a decision taken on a close.
      * **Stops are checked before targets** within a bar. Daily bars cannot say
        which came first, so the replay assumes the worse one.
    """
    config = config or ReplayConfig()
    warmup = config.warmup_bars if config.warmup_bars is not None else rules.preferred_bars
    multiplier = config.atr_stop_multiplier

    trades: list[BacktestTrade] = []
    position: _OpenPosition | None = None
    unreadable = 0
    start = max(warmup, rules.required_bars)
    length = series.length

    for i in range(start, length):
        # Everything known as of this bar's close, and nothing after it.
        reading = read_entry(series.head(i + 1), rules)
        if reading is None:
            unreadable += 1

        if position is not None:
            # 1. The resting stop, checked first and filled intrabar.
            if float(series.low[i]) <= position.stop:
                fill = min(float(series.open[i]), position.stop)
                trades.append(
                    BacktestTrade(
                        entry_index=position.entry_index,
                        exit_index=i,
                        entry_price=position.entry_price,
                        exit_price=fill,
                        initial_stop=position.initial_stop,
                        entry_score=position.entry_score,
                        reward_risk=position.reward_risk,
                        exit_reason=ExitReason.STOP,
                    )
                )
                position = None
                continue

            # 2a. Fixed holding period — exit on schedule rather than on a price
            #     level, so a slow-building edge is given time to arrive.
            if config.hold_bars is not None:
                if (i - position.entry_index) >= config.hold_bars and i + 1 < length:
                    trades.append(
                        BacktestTrade(
                            entry_index=position.entry_index,
                            exit_index=i + 1,
                            entry_price=position.entry_price,
                            exit_price=float(series.open[i + 1]),
                            initial_stop=position.initial_stop,
                            entry_score=position.entry_score,
                            reward_risk=position.reward_risk,
                            exit_reason=ExitReason.TIME,
                        )
                    )
                    position = None
                    continue
                if config.trail_stops and reading is not None:
                    candidate = float(series.close[i]) - reading.atr * multiplier
                    if candidate > position.stop:
                        position.stop = candidate
                continue

            # 2b. Target reached — the thesis played out. Decided on this close,
            #     filled on the next open. A fixed target is frozen at entry; the
            #     band target moves with the average and is read fresh each bar.
            target = (
                position.entry_price + config.fixed_target_r * position.risk_distance
                if config.fixed_target_r is not None
                else (reading.middle if reading is not None else None)
            )
            if target is not None and float(series.close[i]) >= target:
                if i + 1 < length:
                    trades.append(
                        BacktestTrade(
                            entry_index=position.entry_index,
                            exit_index=i + 1,
                            entry_price=position.entry_price,
                            exit_price=float(series.open[i + 1]),
                            initial_stop=position.initial_stop,
                            entry_score=position.entry_score,
                            reward_risk=position.reward_risk,
                            exit_reason=ExitReason.TARGET,
                        )
                    )
                    position = None
                continue

            # 3. Held too long.
            if (
                config.max_holding_bars
                and (i - position.entry_index) >= config.max_holding_bars
                and i + 1 < length
            ):
                trades.append(
                    BacktestTrade(
                        entry_index=position.entry_index,
                        exit_index=i + 1,
                        entry_price=position.entry_price,
                        exit_price=float(series.open[i + 1]),
                        initial_stop=position.initial_stop,
                        entry_score=position.entry_score,
                        reward_risk=position.reward_risk,
                        exit_reason=ExitReason.TIME,
                    )
                )
                position = None
                continue

            # 4. Ratchet the stop up, never down — mirroring StopService.
            if config.trail_stops and reading is not None:
                candidate = float(series.close[i]) - reading.atr * multiplier
                if candidate > position.stop:
                    position.stop = candidate
            continue

        # Flat: does this bar admit an entry? PEAD is absent from a historical
        # replay (see the module docstring), so only the price-derived gates run.
        if reading is not None and reading.admits and i + 1 < length:
            entry_price = float(series.open[i + 1])
            stop = entry_price - reading.atr * multiplier
            if stop > 0 and entry_price > stop:
                position = _OpenPosition(
                    entry_index=i + 1,
                    entry_price=entry_price,
                    initial_stop=stop,
                    stop=stop,
                    entry_score=reading.score,
                    # From the reading, not recomputed: the gate and the record
                    # must be the same number or a filtered run would be measured
                    # against a subtly different quantity than it filtered on.
                    reward_risk=reading.reward_risk,
                )

    # A position still open when the data runs out is recorded rather than
    # dropped: silently discarding it would flatter a strategy whose losers are
    # simply held longer than its winners.
    if position is not None:
        trades.append(
            BacktestTrade(
                entry_index=position.entry_index,
                exit_index=length - 1,
                entry_price=position.entry_price,
                exit_price=float(series.close[length - 1]),
                initial_stop=position.initial_stop,
                entry_score=position.entry_score,
                reward_risk=position.reward_risk,
                exit_reason=ExitReason.UNCLOSED,
            )
        )

    return BacktestResult(
        trades=tuple(trades),
        bars_replayed=max(0, length - start),
        bars_unreadable=unreadable,
    )


@dataclass(frozen=True, slots=True)
class PortfolioResult:
    """Every instrument's replay, pooled."""

    per_instrument: dict[str, BacktestResult] = field(default_factory=dict)

    @property
    def combined(self) -> BacktestResult:
        """All trades pooled into one result, ordered by entry bar.

        Pooling loses the calendar — two trades on different instruments at the
        same index are not simultaneous — so `max_drawdown_r` on the combined
        result is the drawdown of a *sequence* of trades, not of a portfolio held
        through time. Read it as "how bad a run of trades did this rule produce",
        which is the honest question for an entry rule.
        """
        trades = sorted(
            (t for r in self.per_instrument.values() for t in r.trades),
            key=lambda t: t.entry_index,
        )
        return BacktestResult(
            trades=tuple(trades),
            bars_replayed=sum(r.bars_replayed for r in self.per_instrument.values()),
            bars_unreadable=sum(r.bars_unreadable for r in self.per_instrument.values()),
        )
