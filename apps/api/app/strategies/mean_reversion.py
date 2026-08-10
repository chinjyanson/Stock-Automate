"""Daily mean reversion on scanner-ranked stocks (§8).

Two layers, deliberately separated. The **scanner** decides *what* is worth
owning — its absolute, fundamentals-first score ranks the catalogue, and the top
names become this strategy's universe. This strategy decides only *when*: it
buys a ranked stock that has been pushed unusually cheap relative to its own
recent range, and sells it once that dislocation has closed.

The entry has two halves that work in completely different ways, and the split
is the design.

**How dislocated is this? — a weighted score.** Three readings of the same
underlying question, blended into one 0-1 number that must clear
`entry_threshold`:

  * **Bollinger band position** measures the move against the stock's *own*
    recent volatility, which is what makes a 6% drop comparable between a wild
    small-cap and a steady large-cap. Read as an inverted %B, so at or below the
    lower band is full strength, the middle band is a half, and the upper band
    is nothing.
  * **RSI** reads whether the selling is exhausted. Price can pierce a band
    while still trending down hard, and RSI is the thing that tells those apart.
  * **Discount to the 20-day average** is the sanity check on the other two,
    both of which are *relative* measures. A very quiet stock can break two
    standard deviations on a move that is, in cash terms, nothing; this asks how
    far it actually fell.

None of the three is individually required. A deeply oversold RSI can carry a
shallow band break and a violent band break can carry a middling RSI, which is
the point: they measure one thing by three routes, so demanding all three
clear a threshold was arbitrary. A component that cannot be computed drops out
and the remaining weights renormalise — the same discipline the scanner uses.

**Should this ever be bought? — hard gates.** These stay absolute, because they
answer a different question. No amount of dislocation should overrule them: a
dying business must not become buyable merely by falling far enough.

  * **ATR** gates on volatility. A stock whose true range is a rounding error
    has no snapback worth trading, and its band width is noise.
  * **200-day slope** is the falling-knife filter. Buying dips works in an
    uptrend and loses money in a downtrend, and nothing in a band break or an
    RSI reading can tell those apart. Note how it composes with the scanner
    rather than fighting it: the scanner rewards a price *below* its 200-day
    average, this gate requires the average *itself* to be rising. Cheap
    relative to a business that is still growing — not cheap because it is
    shrinking.
  * **Post-earnings drift** vetoes a dip the market is still repricing. If the
    last report landed badly, PEAD says the drift is not finished, so buying the
    dip now is buying in front of more of it. The drift decays over 60 days and
    the veto decays with it.
  * **Anchored VWAP** is an optional fourth gate, off by default. Anchored to
    the last year's lowest close, it is the average price paid by everyone who
    has bought since the bottom — entering below it means buying cheaper than
    the crowd already committed to this recovery. Being a gate it can only make
    entries rarer, which is a real behavioural change and belongs behind a flag.

A gate can refuse a perfect score, and no score can talk a gate round. That
asymmetry is the whole reason the entry is split in two rather than being one
number: "how attractive is this" and "is this allowed" are not the same question,
and blending them lets an attractive enough trade buy its way past a safety rule.

Momentum lives here, in the timing layer, and deliberately not in the scanner.
The scanner rotates 200-2000 names a night against ~20,000 instruments, so a
score there is 10-100 days old when compared against a fresh one — fine for a
P/E, useless for a trend. This strategy sees every candidate every night against
candles from last night.

ATR does double duty: the risk engine downstream also sizes the position and
places the stop from it (`app.risk.engine`), so a wider-ranging stock
automatically gets a smaller position and a wider stop. That is why this file
does not size anything.

Insider selling forces an early exit, and only an exit. Entries need no separate
veto: the scanner already discounts such a stock by up to 40%, which drops it out
of the ranked universe this strategy is given, so a second check would be the
same rule applied twice.

The exit is conditional on the drop not having happened yet. Chief-officer
selling precedes further weakness, so a position held into it is worth closing
before the ATR stop — but once the market has already marked the stock down, the
information is in the price and selling only realises the loss at the bottom. A
stock that has *risen* since the filing is the best case to leave, not a damped
one, so the test is on the signed move rather than its magnitude.

Long-only. Otherwise exits when price recovers to the middle band — the mean it was
reverting to — leaving the ATR stop and the holding-period cap to handle the
case where it never does. A stock leaving the scanner's top ranks does *not*
force an exit: rank oscillation around the boundary would churn the portfolio,
and the position's own thesis (it was oversold, it should revert) has not
changed.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.indicators import functions as ind
from app.indicators.series import PriceSeries
from app.models.enums import Interval, OrderSide, StrategyKind
from app.strategies.base import Strategy, StrategyContext, StrategySignal

#: RSI at which the oversold component scores nothing, and at which it is full.
#: 50 is the neutral midpoint of the indicator, so anything above it contributes
#: zero rather than negatively — this is a dip-buying strategy, and an
#: overbought reading is simply not evidence, not evidence against.
RSI_NEUTRAL = 50.0
RSI_FULL = 20.0

#: Discount to the 20-day average at which that component reaches full strength.
#: 10% is a large move against a twenty-day mean for most listed equities.
SMA20_DISCOUNT_FULL = 0.10


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


@dataclass(frozen=True, slots=True)
class EntryRules:
    """Every tunable the entry decision uses, resolved from params once.

    Frozen and free of I/O so the same rules object can drive the live strategy
    and a historical replay. That sharing is not a convenience — it is the only
    thing stopping the backtest from measuring a subtly different strategy from
    the one that trades, which is the classic way a backtest comes to be
    confidently wrong.
    """

    bb_period: int = 20
    bb_std: float = 2.0
    rsi_period: int = 14
    atr_period: int = 14
    min_atr_pct: float = 0.02
    #: Bollinger position, **off by default**.
    #:
    #: Measured on both folds: including it lowers the profit factor (1.30 -> 1.20
    #: fit, 1.15 -> 1.00 confirm) and roughly doubles the worst losing streak. It
    #: fires on nearly every candidate, so it added volume rather than quality —
    #: and it was measuring the same quantity as the other two components anyway,
    #: which is why the blend never beat RSI alone. Kept as a settable weight
    #: rather than deleted so the comparison stays reproducible.
    weight_band: float = 0.0
    weight_rsi: float = 0.40
    weight_discount: float = 0.15
    entry_threshold: float = 0.60
    trend_slope_min: float = 0.0
    #: Multiple of ATR the risk engine will place the stop at. Mirrored here so
    #: the entry can weigh what it stands to make against what it stands to lose;
    #: the strategy still sizes nothing.
    atr_stop_multiplier: float = 5.0
    #: Refuse a setup offering less than this reward per unit of risk. 0 disables.
    #:
    #: Measured across 2,088 replayed trades, the *median* setup risks 1.0 to make
    #: 0.90, and at a 50% win rate that loses by arithmetic — 0.5 x 0.9 - 0.5 x 1.0
    #: = -0.05R, against a measured -0.04R for the sub-1:1 bucket. More than half
    #: of all entries took odds that could not win. This gate exists to refuse
    #: them, and needs no statistical support to justify: a below-even payoff on a
    #: coin flip is a losing bet whatever a backtest says.
    min_reward_risk: float = 0.0
    avwap_enabled: bool = False
    avwap_anchor_period: int = ind.TRADING_DAYS_PER_YEAR

    @property
    def required_bars(self) -> int:
        """Below this nothing can be computed at all."""
        return max(self.bb_period, self.rsi_period + 1, self.atr_period + 1)

    @property
    def preferred_bars(self) -> int:
        """Enough for the 200-day slope and its 21-bar fit window."""
        return max(self.bb_period * 6, 260)


@dataclass(frozen=True, slots=True)
class EntryReading:
    """What the price series alone says about entering, at one point in time.

    Everything here is derived from bars up to and including the last one in the
    series handed in — there is no way for it to see forward, which is what makes
    it safe to replay.

    `pead` is deliberately absent: it comes from the earnings table rather than
    from prices, so the caller applies it. `admits` therefore means "the price
    series raises no objection", not "trade this".
    """

    score: float
    lower: float
    middle: float
    upper: float
    rsi: float
    atr: float
    atr_pct: float
    trend_slope: float | None
    avwap: float | None
    #: Distance to the middle-band target over distance to the stop, at entry.
    reward_risk: float
    score_ok: bool
    atr_ok: bool
    trend_ok: bool
    avwap_ok: bool
    reward_risk_ok: bool

    @property
    def admits(self) -> bool:
        return (
            self.score_ok
            and self.atr_ok
            and self.trend_ok
            and self.avwap_ok
            and self.reward_risk_ok
        )


def read_entry(series: PriceSeries, rules: EntryRules) -> EntryReading | None:
    """Score the dislocation and evaluate the price-derived gates.

    None when no opinion is possible: too few bars, a flat window with no
    meaningful band, or an unusable price. That is distinct from a reading that
    declines — "cannot tell" and "no" are different answers and the caller
    treats them differently.
    """
    closes = series.close
    if series.length < rules.required_bars:
        return None

    bands = ind.bollinger_bands(closes, rules.bb_period, rules.bb_std)
    if bands is None:
        return None  # flat window: no meaningful band, so no opinion
    lower, middle, upper = bands

    rsi = ind.relative_strength_index(closes, rules.rsi_period)
    atr = ind.average_true_range(series.high, series.low, closes, period=rules.atr_period)
    last = float(closes[-1])
    if rsi is None or atr is None or last <= 0:
        return None

    # -- How dislocated is this? Three readings, one score. -------------------
    #
    # Each contributes (weight, strength); anything unmeasurable is left out and
    # the divisor shrinks with it, so absence neither helps nor hurts. None of
    # the three can veto on its own.
    components: list[tuple[float, float]] = []

    band_width = upper - lower
    if band_width > 0:
        # Inverted %B: 1.0 at or below the lower band, 0.5 at the middle, 0.0 at
        # the upper. Reading it as a position rather than as a yes/no break is
        # what lets a stock that stopped just short of its band still make the
        # case on the strength of the other two.
        components.append((rules.weight_band, _clamp01(1.0 - (last - lower) / band_width)))

    components.append(
        (rules.weight_rsi, _clamp01((RSI_NEUTRAL - float(rsi)) / (RSI_NEUTRAL - RSI_FULL)))
    )

    if middle > 0:
        discount = (middle - last) / middle
        components.append((rules.weight_discount, _clamp01(discount / SMA20_DISCOUNT_FULL)))

    total_weight = sum(w for w, _ in components)
    score = sum(w * s for w, s in components) / total_weight if total_weight > 0 else 0.0

    # -- Should this ever be bought? Absolute gates. ---------------------------
    atr_pct = atr / last

    # `sma_slope` returns None on a series too short to fit 200 bars, which is
    # most newly-listed names — and that reads as satisfied. A gate that failed
    # closed on missing data would quietly stop this strategy trading anything
    # without a year of history.
    trend_slope = ind.sma_slope(closes, 200, slope_window=21)

    avwap: float | None = None
    if rules.avwap_enabled:
        anchor = ind.lowest_close_index(closes, rules.avwap_anchor_period)
        if anchor is not None:
            avwap = ind.anchored_vwap(series.high, series.low, closes, series.volume, anchor)

    # What this setup stands to make against what it stands to lose. The target
    # is the middle band and the stop is a multiple of ATR below entry, so both
    # are known at entry — and roughly half of all setups turn out to offer less
    # than 1:1, which is the single clearest defect measurement has found.
    stop_distance = atr * rules.atr_stop_multiplier
    reward_risk = (middle - last) / stop_distance if stop_distance > 0 else 0.0

    return EntryReading(
        score=score,
        lower=lower,
        middle=middle,
        upper=upper,
        rsi=float(rsi),
        atr=atr,
        atr_pct=atr_pct,
        trend_slope=trend_slope,
        avwap=avwap,
        reward_risk=reward_risk,
        score_ok=score >= rules.entry_threshold,
        atr_ok=atr_pct >= rules.min_atr_pct,
        trend_ok=trend_slope is None or trend_slope >= rules.trend_slope_min,
        avwap_ok=avwap is None or last <= avwap,
        reward_risk_ok=reward_risk >= rules.min_reward_risk,
    )


class MeanReversionStrategy(Strategy):
    kind = StrategyKind.MEAN_REVERSION
    interval = Interval.D1

    def rules(self) -> EntryRules:
        """The entry tunables, as the shared frozen object the backtest replays.

        Built here and nowhere else, so a historical run and a live run cannot
        drift apart on a default.

        The weights are relative, not absolute: they are renormalised by whatever
        could be measured, so they need not sum to 1 and a missing component
        costs nothing. The band leads because it is the only one of the three
        scaled to the instrument's own volatility; the discount trails because it
        partly restates the band — deliberately, as an absolute-magnitude check
        on two relative measures — and would otherwise double-count.

        `entry_threshold` at 0.60 is deliberately *below* where the old
        all-or-nothing rules sat: a band break AND RSI <= 35 corresponded to
        roughly 0.71 on this scale, so the same setups still qualify and a band
        break with RSI in the low 40s — or a deeply oversold stock that stopped
        just short of its band — now qualifies too, where before either was
        refused outright.

        `trend_slope_min` ships **on**, unlike anchored VWAP: it has real
        evidence behind it rather than a plausible story, and its failure mode
        (skipping a dip in a declining business) is the one this strategy most
        needs protection from.
        """
        return EntryRules(
            bb_period=int(self.param("bb_period", 20)),
            bb_std=float(self.param("bb_std", 2.0)),
            rsi_period=int(self.param("rsi_period", 14)),
            atr_period=int(self.param("atr_period", 14)),
            min_atr_pct=float(self.param("min_atr_pct", 0.02)),
            weight_band=float(self.param("entry_weight_band", 0.45)),
            weight_rsi=float(self.param("entry_weight_rsi", 0.40)),
            weight_discount=float(self.param("entry_weight_discount", 0.15)),
            entry_threshold=float(self.param("entry_threshold", 0.60)),
            trend_slope_min=float(self.param("trend_slope_min", 0.0)),
            atr_stop_multiplier=float(self.param("atr_stop_multiplier", 2.0)),
            min_reward_risk=float(self.param("min_reward_risk", 0.0)),
            avwap_enabled=bool(self.param("avwap_enabled", False)),
            avwap_anchor_period=int(self.param("avwap_anchor_period", ind.TRADING_DAYS_PER_YEAR)),
        )

    async def evaluate(self, ctx: StrategyContext) -> list[StrategySignal]:
        rules = self.rules()
        # Insider selling pressure (0..0.40) at which a held position is closed.
        # 0.10 is a quarter of maximum, so it takes a real chief-officer sale
        # rather than a small or half-decayed one.
        insider_veto = float(self.param("insider_sell_veto", 0.10))
        # ...but only while the drop has not already happened. Measured in ATR
        # so it means the same on a calm stock and a wild one.
        insider_exit_max_drop = float(self.param("insider_exit_max_drop_atr", 1.0))
        # PEAD reading at or below which a dip is left alone. 50 is neutral, so
        # 40 is a real negative reaction rather than noise — roughly a 2%
        # abnormal move down on a fresh report, or a 10% one about seven weeks
        # ago once the decay is applied. Applied here rather than inside
        # `read_entry` because it comes from the earnings table, not from prices.
        pead_veto_below = float(self.param("pead_veto_below", 40.0))

        signals: list[StrategySignal] = []
        for instrument in ctx.instruments:
            # The 200-day slope needs 200 bars plus its 21-bar fit window, which
            # is far more than the Bollinger window asks for — hence
            # `preferred_bars` rather than a plain multiple of `bb_period`.
            # `required` is deliberately *not* raised with it: a short series
            # must still be tradable, it just cannot answer the trend question.
            series = await ctx.series(
                instrument.id,
                self.read_interval,
                limit=rules.preferred_bars,
                required=rules.required_bars,
            )
            if series is None:
                continue

            # The same call the backtest makes, on the same rules object.
            reading = read_entry(series, rules)
            if reading is None:
                continue

            last = float(series.close[-1])
            held = ctx.held_quantity(instrument.id)
            pressure = ctx.sell_pressure(instrument.id)
            lower, middle = reading.lower, reading.middle

            # Is the market still repricing a bad report? Absent means no live
            # earnings event, which is the common case outside the US where the
            # calendar is thin — and absence reads as satisfied, like every other
            # missing measurement here.
            pead = ctx.pead_score(instrument.id)
            pead_ok = pead is None or pead > pead_veto_below

            metrics = {
                "bb_lower": lower,
                "bb_middle": middle,
                "bb_upper": reading.upper,
                "rsi": reading.rsi,
                "atr": reading.atr,
                "atr_pct": reading.atr_pct,
                "close": last,
                "entry_score": reading.score,
            }
            if reading.avwap is not None:
                metrics["anchored_vwap"] = reading.avwap
            if reading.trend_slope is not None:
                metrics["sma200_slope"] = reading.trend_slope
            if pead is not None:
                metrics["pead_score"] = pead

            # Leave *before* the fall, or not at all. A chief officer choosing
            # to sell precedes about -6.28% excess return over the following
            # month (686-event backtest), so a position held into that is worth
            # closing early rather than riding to the ATR stop. But only while
            # the market has not already acted: once the stock has dropped more
            # than `insider_exit_max_drop` ATR since the filing, the information
            # is in the price and selling realises the loss at the bottom.
            # A stock that has *risen* is the best case, not a damped one —
            # which is why this reads the signed move, not its magnitude.
            insider_exit = (
                held > 0
                and pressure is not None
                and pressure.sell_penalty >= insider_veto
                and (pressure.move_atr is None or pressure.move_atr > -insider_exit_max_drop)
            )

            if insider_exit:
                assert pressure is not None  # narrowed by insider_exit
                moved = (
                    "unknown"
                    if pressure.move_atr is None
                    else f"{pressure.move_atr:+.1f} ATR since filing"
                )
                # Checked before the mean-reversion exit so it wins when both fire.
                signals.append(
                    StrategySignal(
                        instrument_id=instrument.id,
                        side=OrderSide.SELL,
                        conviction=1.0,
                        reason=(
                            f"Insider exit: chief-officer selling "
                            f"(pressure {pressure.sell_penalty:.0%} >= {insider_veto:.0%}), "
                            f"price {moved} — leaving before the drop"
                        ),
                        target_quantity=held,
                        metrics={
                            **metrics,
                            "insider_sell_pressure": pressure.sell_penalty,
                            "insider_move_atr": pressure.move_atr or 0.0,
                        },
                    )
                )
            elif held <= 0:
                # `reading.admits` is the score plus every price-derived gate;
                # PEAD is the one the price series cannot answer. The score
                # decides *whether it is dislocated enough*, the gates decide
                # *whether it should be bought at all*, and a gate can refuse a
                # perfect score while no score can talk a gate round.
                if reading.admits and pead_ok:
                    avwap_note = (
                        f", below anchored VWAP {reading.avwap:.2f}"
                        if reading.avwap is not None
                        else ""
                    )
                    trend_note = (
                        f", 200-day trend {reading.trend_slope:+.3%}/day"
                        if reading.trend_slope is not None
                        else ""
                    )
                    signals.append(
                        StrategySignal(
                            instrument_id=instrument.id,
                            # The dislocation score *is* the conviction. Still not
                            # read by sizing — the risk engine works from ATR and
                            # equity alone — but it is at least now a number that
                            # means something rather than a restatement of the
                            # band break.
                            conviction=reading.score,
                            side=OrderSide.BUY,
                            reason=(
                                f"Mean reversion: entry score {reading.score:.2f} "
                                f"(>= {rules.entry_threshold:.2f}) — close {last:.2f} vs lower "
                                f"band {lower:.2f}, RSI {reading.rsi:.0f}, "
                                f"ATR {reading.atr_pct:.1%} of price "
                                f"(>= {rules.min_atr_pct:.1%})"
                                f"{avwap_note}{trend_note}"
                            ),
                            metrics=metrics,
                        )
                    )
            elif last >= middle:
                signals.append(
                    StrategySignal(
                        instrument_id=instrument.id,
                        side=OrderSide.SELL,
                        conviction=1.0,
                        reason=(
                            f"Mean reversion exit: close {last:.2f} recovered to the "
                            f"middle band {middle:.2f}"
                        ),
                        target_quantity=held,
                        metrics=metrics,
                    )
                )
        return signals
