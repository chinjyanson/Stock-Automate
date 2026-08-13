"""Ways to buy back in after stepping aside for a crash warning.

The detector answers "is a sharp fall coming". It says nothing about when to
return, and that second decision turned out to matter more than the first: the
same detector went from losing to buy-and-hold to beating it by nine thousand
pounds purely on how quickly it bought back, and the largest single losses in
the whole backtest all shared one shape — the alarm fired, the market *rose*,
and a rule that only re-enters on further falls sat at the defensive weight
until a timer expired, standing outside the entire rebound.

So this module makes the re-entry decision a first-class thing that can be
swapped and measured, instead of a branch buried in a simulation loop.

## The shape of a rule

Every rule answers one question each day it is out of the market: **how far back
in should I be, from 0 (stay defensive) to 1 (fully invested)?** Reaching 1 ends
the episode. Everything a rule may look at is on `Context`, and nothing there is
from the future — each field is computed from the decision bar or earlier.

## The rules are deliberately of different *kinds*

Grouping them this way is the point: if the winners all come from one family,
that is evidence about what actually drives the result, and if the winner is
`wait`, the cleverness was never worth anything.

  * do nothing — `wait`
  * how far it has fallen — `ladder`, `dip`
  * how far it has bounced — `bounce`, `ladder_bounce`
  * has momentum turned — `rsi`, `up_streak`, `reclaim`
  * has the panic passed — `calm`, `all_clear`
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

#: A rule: given the day's context, how far back toward fully invested to be.
Rule = Callable[["Context"], float]


@dataclass(frozen=True, slots=True)
class Context:
    """What a re-entry rule may look at on one day out of the market.

    Every field is measured at the decision bar or before it. `price` is that
    bar's close, not the next one — the simulation acts on the following bar, so
    a rule reading `price` is not reading a price it could not have known.
    """

    #: Trading days since the alarm fired.
    days_out: int
    #: Close on the decision bar.
    price: float
    #: Close on the bar the alarm fired — the price we effectively sold at.
    exit_price: float
    #: Lowest close seen since the alarm.
    low_since: float
    #: The detector still reads today as alarming.
    warning: bool
    #: 14-day RSI of the index. Below 30 is the conventional "oversold".
    rsi: float
    #: Close divided by its own 10-day average. Above 1.0 is a reclaimed trend.
    sma_ratio: float
    #: 10-day realised volatility now, and on the day the alarm fired.
    volatility: float
    volatility_at_exit: float
    #: Consecutive up days ending on the decision bar.
    up_streak: int

    @property
    def fallen(self) -> float:
        """How far below the sale price we are. Negative means it rose instead."""
        return 1.0 - self.price / self.exit_price if self.exit_price > 0 else 0.0

    @property
    def bounced(self) -> float:
        """How far above the post-alarm low we are."""
        return self.price / self.low_since - 1.0 if self.low_since > 0 else 0.0


def _clamp(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def wait(_param: float) -> Rule:
    """Do nothing; the caller's timeout ends the episode.

    The control every other rule has to beat. It is not a strawman — it is what
    the overlay does today whenever the market fails to fall after an alarm, and
    a cleverer rule that cannot beat waiting has bought nothing but complexity.
    """

    def rule(_ctx: Context) -> float:
        return 0.0

    return rule


def ladder(depth: float) -> Rule:
    """Buy back in proportion to how far it has fallen below the sale price.

    The original design: no forecast of the bottom, just capital returning in
    steps as the decline deepens. Its flaw is structural and is why this module
    exists — `fallen` cannot go below zero, so a market that rallies off the
    alarm leaves this rule at the defensive weight indefinitely.
    """

    def rule(ctx: Context) -> float:
        return _clamp(ctx.fallen / depth)

    return rule


def dip(depth: float) -> Rule:
    """All the way back in one step, once it has fallen `depth` below the sale.

    The ladder's idea without the gradualism: wait for a real discount, then
    commit. Fewer, larger decisions, which is a genuinely different bet from
    averaging in even though both are "buy the dip".
    """

    def rule(ctx: Context) -> float:
        return 1.0 if ctx.fallen >= depth else 0.0

    return rule


def bounce(rise: float) -> Rule:
    """Back in full once it has risen `rise` off its lowest point since the alarm.

    Waits for the turn rather than trying to catch the bottom. This is the rule
    that repairs the ladder's blind spot, because a rally off the alarm is
    exactly what it responds to.
    """

    def rule(ctx: Context) -> float:
        return 1.0 if ctx.bounced >= rise else 0.0

    return rule


def ladder_bounce(rise: float) -> Rule:
    """The ladder, plus a bounce escape. Whichever fires first wins.

    Ships today as `--reentry price --rebound`. Included so the new ideas are
    measured against the incumbent rather than against a strawman.
    """

    laddered = ladder(0.10)
    bounced = bounce(rise)

    def rule(ctx: Context) -> float:
        return max(laddered(ctx), bounced(ctx))

    return rule


def rsi(level: float) -> Rule:
    """Back in full when the 14-day RSI recovers above `level`.

    The classic reversal read: RSI falls under 30 when selling is exhausted, and
    turning back up is taken as the selling being over. Unlike `bounce` this
    measures the *strength* of recent rises against recent falls rather than one
    price ratio, so a limp drift upwards does not trigger it.
    """

    def rule(ctx: Context) -> float:
        return 1.0 if ctx.rsi >= level else 0.0

    return rule


def reclaim(ratio: float) -> Rule:
    """Back in full when the price closes back above its 10-day average.

    A trend-following re-entry: do not ask whether the fall is over, ask whether
    the short-term trend has turned back up. Slower than `bounce` by
    construction, which is either patience or lateness depending on the episode.
    """

    def rule(ctx: Context) -> float:
        return 1.0 if ctx.sma_ratio >= ratio else 0.0

    return rule


def up_streak(days: float) -> Rule:
    """Back in full after `days` consecutive up days.

    The crudest possible reversal signal, included precisely because it is
    crude: if it matches the indicator-based rules, then what those rules are
    detecting is simply "the market went up a bit", and the indicator is
    decoration.
    """

    def rule(ctx: Context) -> float:
        return 1.0 if ctx.up_streak >= days else 0.0

    return rule


def calm(ratio: float) -> Rule:
    """Back in full when 10-day volatility falls back to `ratio` of its level
    at the alarm.

    Ignores price direction entirely and asks only whether the market has
    stopped thrashing. Crashes cluster in time — violent days follow violent
    days — so waiting for the violence to subside is a different claim from
    waiting for the price to recover, and can fire while still far underwater.
    """

    def rule(ctx: Context) -> float:
        if ctx.volatility_at_exit <= 0:
            return 0.0
        return 1.0 if ctx.volatility <= ratio * ctx.volatility_at_exit else 0.0

    return rule


def all_clear(_param: float) -> Rule:
    """Back in full the day the detector stops warning.

    The only rule that consults the model while out. It re-reads the same
    evidence that caused the exit, so it can return the position the day after a
    false alarm instead of waiting on price to prove anything.
    """

    def rule(ctx: Context) -> float:
        return 0.0 if ctx.warning else 1.0

    return rule


#: Every rule, with the sweep of parameter values each is tested over. A rule
#: with no parameter still lists one value so the comparison treats them alike.
REGISTRY: dict[str, tuple[Callable[[float], Rule], tuple[float, ...]]] = {
    "wait": (wait, (0.0,)),
    "ladder": (ladder, (0.05, 0.10, 0.15, 0.20)),
    "dip": (dip, (0.03, 0.05, 0.10, 0.15)),
    "bounce": (bounce, (0.02, 0.03, 0.05, 0.08)),
    "ladder_bounce": (ladder_bounce, (0.02, 0.03, 0.05, 0.08)),
    "rsi": (rsi, (30.0, 40.0, 50.0, 60.0)),
    "reclaim": (reclaim, (0.98, 1.00, 1.02)),
    "up_streak": (up_streak, (1.0, 2.0, 3.0)),
    "calm": (calm, (0.6, 0.8, 1.0, 1.2)),
    "all_clear": (all_clear, (0.0,)),
}
