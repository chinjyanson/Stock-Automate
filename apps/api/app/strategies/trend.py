"""Gold / oil trend following (§8).

The thesis: commodities trend. Hold the instrument while it is in a confirmed
uptrend — price above a long moving average, that average rising, and a positive
trailing return — and step aside when the trend breaks. Long-only, daily bars.

This is deliberately plain: it reuses the Phase 2 indicators (`sma_slope`,
`simple_moving_average`, `trailing_return`) and adds no new maths. The universe
(gold and oil ETCs) is configuration, not baked in.
"""

from __future__ import annotations

from app.indicators import functions as ind
from app.models.enums import Interval, OrderSide, StrategyKind
from app.strategies.base import Strategy, StrategyContext, StrategySignal


class TrendFollowingStrategy(Strategy):
    kind = StrategyKind.TREND_FOLLOWING
    interval = Interval.D1

    async def evaluate(self, ctx: StrategyContext) -> list[StrategySignal]:
        sma_period = int(self.param("sma_period", 100))
        slope_window = int(self.param("slope_window", 21))
        return_lookback = int(self.param("return_lookback", 60))

        signals: list[StrategySignal] = []
        for instrument in ctx.instruments:
            series = await ctx.series(instrument.id, self.read_interval, limit=sma_period * 3)
            if series is None or series.length < sma_period + slope_window:
                continue

            sma = ind.simple_moving_average(series.close, sma_period)
            slope = ind.sma_slope(series.close, sma_period, slope_window)
            trailing = ind.trailing_return(series.close, return_lookback)
            if sma is None or slope is None or trailing is None:
                continue
            last = float(series.close[-1])
            held = ctx.held_quantity(instrument.id)

            uptrend = last > sma and slope > 0 and trailing > 0
            if held <= 0 and uptrend:
                # Conviction from the strength of the trend (capped).
                conviction = min(1.0, max(0.0, trailing * 4))
                signals.append(
                    StrategySignal(
                        instrument_id=instrument.id,
                        side=OrderSide.BUY,
                        conviction=conviction,
                        reason=(
                            f"Trend: price {last:.2f} > SMA{sma_period} {sma:.2f}, "
                            f"slope {slope:.4f}, {return_lookback}d return {trailing:.2%}"
                        ),
                        metrics={"sma": sma, "slope": slope, "trailing_return": trailing},
                    )
                )
            elif held > 0 and (last < sma or slope <= 0):
                signals.append(
                    StrategySignal(
                        instrument_id=instrument.id,
                        side=OrderSide.SELL,
                        conviction=1.0,
                        reason=(
                            f"Trend break: price {last:.2f} vs SMA{sma_period} {sma:.2f}, "
                            f"slope {slope:.4f}"
                        ),
                        target_quantity=held,
                        metrics={"sma": sma, "slope": slope},
                    )
                )
        return signals
