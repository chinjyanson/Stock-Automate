"""Target-allocation pie rebalancer (§8).

A "pie" is a basket held at target weights (the Trading 212 concept). This
strategy keeps the basket near those targets: it values each holding, compares its
weight to the target, and — where a holding has drifted beyond a tolerance band —
signals a buy for the underweight and a trim for the overweight.

Long-only: a trim never sells more than is held, and an underweight is bought, not
shorted. The pie's *risk control is its allocation* — the weights bound
concentration by construction — so these signals carry an explicit target quantity
and are filled directly (still gated by halts and data staleness), rather than
being re-sized by the volatility engine, which would defeat the target.
"""

from __future__ import annotations

import uuid
from decimal import Decimal

from app.models.enums import Interval, OrderSide, StrategyKind
from app.strategies.base import Strategy, StrategyContext, StrategySignal


class PieRebalanceStrategy(Strategy):
    kind = StrategyKind.PIE_REBALANCE
    interval = Interval.D1

    async def evaluate(self, ctx: StrategyContext) -> list[StrategySignal]:
        weights = self._weights()
        if not weights:
            return []
        band = float(self.param("drift_band", 0.05))

        # Price each pie instrument from the store's latest close.
        prices: dict[uuid.UUID, float] = {}
        for instrument in ctx.instruments:
            if instrument.id not in weights:
                continue
            series = await ctx.series(instrument.id, self.read_interval, limit=30)
            if series is None:
                continue
            prices[instrument.id] = float(series.close[-1])

        # Pie capital: the configured budget, else the current holdings value
        # (rebalance-only when no budget is set).
        current_value = {
            iid: float(ctx.held_quantity(iid)) * price for iid, price in prices.items()
        }
        capital = (
            float(self.config.account_equity)
            if self.config.account_equity is not None
            else sum(current_value.values())
        )
        if capital <= 0:
            return []

        signals: list[StrategySignal] = []
        for instrument in ctx.instruments:
            iid = instrument.id
            if iid not in weights or iid not in prices:
                continue
            price = prices[iid]
            target_value = capital * weights[iid]
            drift_value = target_value - current_value.get(iid, 0.0)
            drift_fraction = drift_value / capital
            if abs(drift_fraction) <= band:
                continue

            conviction = min(1.0, abs(drift_fraction) / band) if band else 1.0
            if drift_value > 0:
                qty = Decimal(str(drift_value / price)).quantize(Decimal("0.00000001"))
                if qty <= 0:
                    continue
                signals.append(
                    StrategySignal(
                        instrument_id=iid,
                        side=OrderSide.BUY,
                        conviction=conviction,
                        reason=(
                            f"Pie underweight: target {weights[iid]:.0%}, "
                            f"drift {drift_fraction:+.1%} (> {band:.0%})"
                        ),
                        target_quantity=qty,
                        metrics={"target_weight": weights[iid], "drift": drift_fraction},
                    )
                )
            else:
                held = ctx.held_quantity(iid)
                qty = min(
                    held, Decimal(str(-drift_value / price)).quantize(Decimal("0.00000001"))
                )
                if qty <= 0:
                    continue
                signals.append(
                    StrategySignal(
                        instrument_id=iid,
                        side=OrderSide.SELL,
                        conviction=conviction,
                        reason=(
                            f"Pie overweight: target {weights[iid]:.0%}, "
                            f"drift {drift_fraction:+.1%} (trim)"
                        ),
                        target_quantity=qty,
                        metrics={"target_weight": weights[iid], "drift": drift_fraction},
                    )
                )
        return signals

    def _weights(self) -> dict[uuid.UUID, float]:
        raw = (self.config.universe or {}).get("weights", {})
        weights: dict[uuid.UUID, float] = {}
        for key, value in raw.items():
            try:
                weights[uuid.UUID(str(key))] = float(value)
            except (ValueError, TypeError):
                continue
        return weights
