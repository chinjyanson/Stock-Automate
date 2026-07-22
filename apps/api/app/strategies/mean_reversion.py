"""S&P 500 15-minute mean reversion (§8).

The thesis: on a 15-minute chart, a liquid index proxy that gets stretched well
below its recent intraday mean while momentum is washed out tends to snap back.
So: go long when price is both far below its moving average (a low z-score) and
oversold (low RSI); exit when it has reverted to the mean.

Long-only. The z-score and RSI thresholds are configuration, not constants, so the
strategy can be tuned without a code change.
"""

from __future__ import annotations

import numpy as np

from app.indicators import functions as ind
from app.models.enums import Interval, OrderSide, StrategyKind
from app.strategies.base import Strategy, StrategyContext, StrategySignal


class MeanReversionStrategy(Strategy):
    kind = StrategyKind.SP500_MEAN_REVERSION
    interval = Interval.M15

    async def evaluate(self, ctx: StrategyContext) -> list[StrategySignal]:
        sma_period = int(self.param("sma_period", 20))
        entry_z = float(self.param("zscore_entry", -2.0))
        rsi_period = int(self.param("rsi_period", 14))
        rsi_oversold = float(self.param("rsi_oversold", 30.0))
        exit_z = float(self.param("zscore_exit", 0.0))

        signals: list[StrategySignal] = []
        for instrument in ctx.instruments:
            series = await ctx.series(instrument.id, self.read_interval, limit=sma_period * 6)
            if series is None or series.length < sma_period + 1:
                continue

            window = series.close[-sma_period:]
            mean = float(np.mean(window))
            std = float(np.std(window))
            if std == 0:
                continue  # a flat window has no meaningful z-score
            last = float(series.close[-1])
            z = (last - mean) / std
            rsi = ind.relative_strength_index(series.close, rsi_period)
            held = ctx.held_quantity(instrument.id)

            if held <= 0 and z <= entry_z and rsi is not None and rsi <= rsi_oversold:
                # Conviction grows with how stretched and oversold it is.
                conviction = min(1.0, abs(z) / abs(entry_z)) if entry_z else 0.5
                signals.append(
                    StrategySignal(
                        instrument_id=instrument.id,
                        side=OrderSide.BUY,
                        conviction=conviction,
                        reason=(
                            f"Mean reversion: z={z:.2f} (<= {entry_z}), "
                            f"RSI={rsi:.0f} (<= {rsi_oversold:.0f})"
                        ),
                        metrics={"zscore": z, "rsi": float(rsi), "mean": mean},
                    )
                )
            elif held > 0 and z >= exit_z:
                signals.append(
                    StrategySignal(
                        instrument_id=instrument.id,
                        side=OrderSide.SELL,
                        conviction=1.0,
                        reason=f"Mean reversion exit: reverted to mean (z={z:.2f} >= {exit_z})",
                        target_quantity=held,
                        metrics={"zscore": z, "mean": mean},
                    )
                )
        return signals
