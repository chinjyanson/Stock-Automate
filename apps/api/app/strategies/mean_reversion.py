"""Daily mean reversion on scanner-ranked stocks (§8).

Two layers, deliberately separated. The **scanner** decides *what* is worth
owning — its absolute, fundamentals-first score ranks the catalogue, and the top
names become this strategy's universe. This strategy decides only *when*: it
buys a ranked stock that has been pushed unusually cheap relative to its own
recent range, and sells it once that dislocation has closed.

Three indicators, each with a distinct job — none of them redundant:

  * **Bollinger Bands** locate the dislocation. A close below the lower band is
    a move large relative to the stock's own recent volatility, which is what
    makes this comparable across a volatile small-cap and a steady large-cap.
  * **RSI** confirms it. Price alone can pierce a band while still trending
    down hard; requiring oversold momentum too filters the "cheap and still
    falling" case that a band break cannot distinguish on its own.
  * **ATR** gates on volatility. A stock whose true range is a rounding error
    has no snapback worth trading, and its band width is noise. The entry
    therefore requires a minimum ATR as a fraction of price.

  * **Anchored VWAP** is an optional fourth condition, off by default. Anchored
    to the last year's lowest close, it is the average price paid by everyone
    who has bought since the bottom — entering below it means buying cheaper
    than the crowd already committed to this recovery. It is a hard gate rather
    than a weighting because conviction never reaches sizing, so it can only
    make entries rarer; that is a real behavioural change, hence the flag.

Two further gates ask a question the three indicators above cannot: is this dip
in a company that is still fundamentally fine, or in one that is dying?

  * **200-day slope** is the falling-knife filter. Buying dips works in an
    uptrend and is a losing trade in a downtrend, and nothing in a band break or
    an RSI reading can tell those apart. Note how it composes with the scanner
    rather than fighting it: the scanner rewards a price *below* its 200-day
    average, this gate requires the average *itself* to be rising. Cheap
    relative to a business that is still growing — not cheap because it is
    shrinking.
  * **Post-earnings drift** vetoes a dip the market is still repricing. If the
    last report landed badly, PEAD says the drift is not finished, so buying the
    dip now is buying in front of more of it. The drift decays over 60 days and
    the veto decays with it.

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

from app.indicators import functions as ind
from app.models.enums import Interval, OrderSide, StrategyKind
from app.strategies.base import Strategy, StrategyContext, StrategySignal


class MeanReversionStrategy(Strategy):
    kind = StrategyKind.MEAN_REVERSION
    interval = Interval.D1

    async def evaluate(self, ctx: StrategyContext) -> list[StrategySignal]:
        bb_period = int(self.param("bb_period", 20))
        bb_std = float(self.param("bb_std", 2.0))
        rsi_period = int(self.param("rsi_period", 14))
        rsi_oversold = float(self.param("rsi_oversold", 35.0))
        atr_period = int(self.param("atr_period", 14))
        min_atr_pct = float(self.param("min_atr_pct", 0.02))
        # Insider selling pressure (0..0.40) at which a held position is closed.
        # 0.10 is a quarter of maximum, so it takes a real chief-officer sale
        # rather than a small or half-decayed one.
        insider_veto = float(self.param("insider_sell_veto", 0.10))
        # ...but only while the drop has not already happened. Measured in ATR
        # so it means the same on a calm stock and a wild one.
        insider_exit_max_drop = float(self.param("insider_exit_max_drop_atr", 1.0))
        # Anchored VWAP as a fourth entry condition, shipped **off**. It is a
        # gate rather than a conviction adjustment because conviction never
        # reaches sizing — the risk engine sizes from ATR and equity alone — so a
        # conviction-based version would change nothing that happens. Being a
        # gate, it can only ever make entries rarer, which is a real change to
        # the strategy's behaviour and belongs behind a flag that can be turned
        # on and measured rather than assumed.
        avwap_enabled = bool(self.param("avwap_enabled", False))
        avwap_anchor_period = int(self.param("avwap_anchor_period", ind.TRADING_DAYS_PER_YEAR))
        # Minimum slope of the 200-day average, as fractional change per bar.
        # Zero means "flat or rising". Ships **on**, unlike anchored VWAP: this
        # one has real evidence behind it rather than a plausible story, and its
        # failure mode (skipping a dip in a declining business) is the one this
        # strategy most needs protection from.
        trend_slope_min = float(self.param("trend_slope_min", 0.0))
        # PEAD reading at or below which a dip is left alone. 50 is neutral, so
        # 40 is a real negative reaction rather than noise — roughly a 2%
        # abnormal move down on a fresh report, or a 10% one about seven weeks
        # ago once the decay is applied.
        pead_veto_below = float(self.param("pead_veto_below", 40.0))

        signals: list[StrategySignal] = []
        for instrument in ctx.instruments:
            # The 200-day slope needs 200 bars plus its 21-bar fit window, which
            # is far more than the Bollinger window asks for — hence the floor
            # rather than a plain multiple of `bb_period`. `required` is
            # deliberately *not* raised with it: a short series must still be
            # tradable, it just cannot answer the trend question (see below).
            series = await ctx.series(
                instrument.id,
                self.read_interval,
                limit=max(bb_period * 6, 260),
                required=max(bb_period, rsi_period + 1, atr_period + 1),
            )
            if series is None:
                continue

            bands = ind.bollinger_bands(series.close, bb_period, bb_std)
            if bands is None:
                continue  # flat window: no meaningful band, so no opinion
            lower, middle, upper = bands
            rsi = ind.relative_strength_index(series.close, rsi_period)
            atr = ind.average_true_range(series.high, series.low, series.close, period=atr_period)
            last = float(series.close[-1])
            if rsi is None or atr is None or last <= 0:
                continue

            # ATR as a fraction of price, so the threshold means the same thing
            # for a £2 stock and a £200 one.
            atr_pct = atr / last
            held = ctx.held_quantity(instrument.id)
            pressure = ctx.sell_pressure(instrument.id)

            # Anchored to the lowest close of the last year: the average price
            # paid by everyone who has bought since the bottom. Entering below it
            # means buying cheaper than the crowd that already committed to this
            # recovery, rather than at the top of their range.
            avwap: float | None = None
            if avwap_enabled:
                anchor = ind.lowest_close_index(series.close, avwap_anchor_period)
                if anchor is not None:
                    avwap = ind.anchored_vwap(
                        series.high, series.low, series.close, series.volume, anchor
                    )
            # Unavailable reads as satisfied, the same discipline as everywhere
            # else here: a missing measurement must not silently block trading.
            avwap_ok = avwap is None or last <= avwap

            # Is the long-term trend still intact? `sma_slope` returns None on a
            # series too short to fit 200 bars, which is most newly-listed names
            # — and that reads as satisfied, for the same reason as above. A gate
            # that fails closed on missing data would quietly stop this strategy
            # trading anything without a year of history.
            trend_slope = ind.sma_slope(series.close, 200, slope_window=21)
            trend_ok = trend_slope is None or trend_slope >= trend_slope_min

            # Is the market still repricing a bad report? Absent means no live
            # earnings event, which is the common case outside the US where the
            # calendar is thin — again, satisfied.
            pead = ctx.pead_score(instrument.id)
            pead_ok = pead is None or pead > pead_veto_below

            metrics = {
                "bb_lower": lower,
                "bb_middle": middle,
                "bb_upper": upper,
                "rsi": float(rsi),
                "atr": atr,
                "atr_pct": atr_pct,
                "close": last,
            }
            if avwap is not None:
                metrics["anchored_vwap"] = avwap
            if trend_slope is not None:
                metrics["sma200_slope"] = trend_slope
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
                if (
                    last <= lower
                    and rsi <= rsi_oversold
                    and atr_pct >= min_atr_pct
                    and avwap_ok
                    and trend_ok
                    and pead_ok
                ):
                    # Conviction from how far below the band it closed, measured
                    # in band-widths so it stays comparable across instruments.
                    band_width = upper - lower
                    overshoot = (lower - last) / band_width if band_width > 0 else 0.0
                    conviction = min(1.0, 0.5 + overshoot)
                    avwap_note = f", below anchored VWAP {avwap:.2f}" if avwap is not None else ""
                    trend_note = (
                        f", 200-day trend {trend_slope:+.3%}/day" if trend_slope is not None else ""
                    )
                    signals.append(
                        StrategySignal(
                            instrument_id=instrument.id,
                            side=OrderSide.BUY,
                            conviction=conviction,
                            reason=(
                                f"Mean reversion: close {last:.2f} <= lower band "
                                f"{lower:.2f}, RSI {rsi:.0f} (<= {rsi_oversold:.0f}), "
                                f"ATR {atr_pct:.1%} of price (>= {min_atr_pct:.1%})"
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
