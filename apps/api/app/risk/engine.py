"""Position sizing and the pre-trade gate (§9).

`evaluate` is the single chokepoint every order passes. It sizes volatility-
adjusted (constant *risk*, not constant size), then caps that by every limit and
lets the **smallest** cap win, rounding down so no cap is breached by the
rounding. Before any of that it fails closed: an active halt, stale data, or a
missing configuration all mean "no trade", stated as a reason rather than a
silent skip.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import ROUND_DOWN, Decimal

import structlog
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.broker.types import BrokerAccount, BrokerPosition
from app.data.store import CandleStore
from app.indicators import functions as ind
from app.indicators.series import candles_to_series
from app.models.enums import BrokerKind, Interval, TradeIntentStatus
from app.models.instrument import Instrument
from app.models.market_data import Candle
from app.models.risk import RiskConfiguration, TradeIntent
from app.risk.halts import HaltService

log = structlog.get_logger(__name__)

#: The venue trades daily bars; a bar older than this means the picture is stale
#: enough to refuse (weekends/holidays are why it is days, not one).
STALE_MAX_AGE = timedelta(days=5)

#: The step positions round down to. The internal paper venue is fully
#: fractional; a real broker's step would come from its BrokerInstrument.
QUANTITY_STEP = Decimal("0.00000001")

#: How much a correlated position is cut when it would pile onto existing
#: benchmark exposure. A reduction, not a warning — it must change the order.
CORRELATION_REDUCTION = Decimal("0.5")


@dataclass
class RiskDecision:
    """The engine's verdict on one candidate order."""

    approved_quantity: Decimal
    entry_price: Decimal
    stop_price: Decimal | None
    risk_amount: Decimal
    rejected: bool = False
    reason: str | None = None
    applied_caps: list[str] = field(default_factory=list)
    correlation: float | None = None

    @classmethod
    def reject(cls, reason: str) -> RiskDecision:
        return cls(
            approved_quantity=Decimal(0),
            entry_price=Decimal(0),
            stop_price=None,
            risk_amount=Decimal(0),
            rejected=True,
            reason=reason,
        )


class RiskEngine:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._store = CandleStore(session)
        self._halts = HaltService(session)

    async def evaluate(
        self,
        *,
        instrument: Instrument,
        config: RiskConfiguration | None,
        account: BrokerAccount,
        positions: list[BrokerPosition],
        candles: list[Candle],
        benchmark_candles: list[Candle] | None = None,
        broker: BrokerKind = BrokerKind.INTERNAL_PAPER,
    ) -> RiskDecision:
        # 1. Fail-closed gates, before any sizing. -------------------------
        if config is None:
            return RiskDecision.reject(
                "No active risk configuration; refusing to size a trade (fail closed)."
            )
        halt = await self._halts.blocking_halt(instrument.id)
        if halt is not None:
            return RiskDecision.reject(
                f"Trading halted ({halt.kind.value}/{halt.scope.value}): {halt.reason}"
            )
        if await self._store.is_stale(instrument.id, Interval.D1, max_age=STALE_MAX_AGE):
            return RiskDecision.reject(
                f"{instrument.name} data is stale; blocking the order (fail closed)."
            )

        # 2. Count / frequency gates. --------------------------------------
        open_position_tickers = {p.broker_ticker for p in positions if p.quantity > 0}
        is_new_instrument = str(instrument.id) not in open_position_tickers
        if is_new_instrument and len(open_position_tickers) >= config.max_open_positions:
            return RiskDecision.reject(
                f"At the open-position cap ({config.max_open_positions}); no new positions."
            )
        if await self._trades_today(broker) >= config.max_trades_per_day:
            return RiskDecision.reject(
                f"At the daily trade cap ({config.max_trades_per_day})."
            )

        # 3. Entry, stop, base size. ---------------------------------------
        series = candles_to_series(candles)
        if series.length < 20:
            return RiskDecision.reject("Insufficient recent history to size a stop.")
        entry_price = Decimal(str(float(series.close[-1])))
        if entry_price <= 0:
            return RiskDecision.reject("Invalid entry price.")

        atr = ind.average_true_range(series.high, series.low, series.close, period=14)
        if atr is None or atr <= 0:
            return RiskDecision.reject("Could not compute a stop distance (ATR unavailable).")
        stop_distance = Decimal(str(atr)) * Decimal(str(config.atr_stop_multiplier))
        stop_price = entry_price - stop_distance
        if stop_price <= 0:
            return RiskDecision.reject("Computed stop is at or below zero.")

        equity = account.total
        if equity <= 0:
            return RiskDecision.reject("Account equity is not positive.")
        risk_budget = equity * Decimal(str(config.risk_per_trade_pct))
        raw_quantity = risk_budget / stop_distance

        # 4. Caps — smallest wins. -----------------------------------------
        caps: dict[str, Decimal] = {"risk_budget": raw_quantity}

        caps["max_position_pct"] = (
            equity * Decimal(str(config.max_position_pct)) / entry_price
        )

        held_value = sum(
            (
                p.quantity * (p.current_price or p.average_price)
                for p in positions
                if p.broker_ticker == str(instrument.id)
            ),
            start=Decimal(0),
        )
        instrument_room = equity * Decimal(str(config.max_instrument_pct)) - held_value
        caps["max_instrument_pct"] = max(instrument_room, Decimal(0)) / entry_price

        caps["available_cash"] = account.free_for_trading / entry_price

        open_risk = await self._open_risk(broker)
        open_risk_room = equity * Decimal(str(config.max_total_open_risk_pct)) - open_risk
        caps["max_total_open_risk"] = max(open_risk_room, Decimal(0)) / stop_distance

        if config.monetary_position_cap is not None:
            caps["monetary_cap"] = Decimal(str(config.monetary_position_cap)) / entry_price

        # The binding cap is the smallest allowance.
        binding_cap = min(caps, key=lambda k: caps[k])
        quantity = caps[binding_cap]

        # 5. Correlation — reduce (never merely warn). ---------------------
        correlation: float | None = None
        applied_caps = [binding_cap]
        if benchmark_candles:
            correlation = self._correlation(series.close, benchmark_candles, config)
            if correlation is not None and correlation > float(config.correlation_threshold):
                invested = account.invested or Decimal(0)
                exposure = invested / equity if equity else Decimal(0)
                if exposure > Decimal(str(config.max_portfolio_sp500_pct)):
                    quantity = quantity * CORRELATION_REDUCTION
                    applied_caps.append("correlation_reduction")

        # 6. Round down to step; reject a position that rounds to nothing. --
        quantity = quantity.quantize(QUANTITY_STEP, rounding=ROUND_DOWN)
        if quantity <= 0:
            return RiskDecision.reject(
                f"Position sized to zero after caps ({binding_cap} bound it)."
            )

        risk_amount = (quantity * stop_distance).quantize(Decimal("0.0001"), rounding=ROUND_DOWN)
        return RiskDecision(
            approved_quantity=quantity,
            entry_price=entry_price,
            stop_price=stop_price.quantize(Decimal("0.00000001"), rounding=ROUND_DOWN),
            risk_amount=risk_amount,
            applied_caps=applied_caps,
            correlation=correlation,
        )

    # -- Helpers -----------------------------------------------------------

    def _correlation(
        self,
        closes: object,
        benchmark_candles: list[Candle],
        config: RiskConfiguration,
    ) -> float | None:
        bench = candles_to_series(benchmark_candles)
        window = int(config.correlation_window_short)
        if bench.length < window + 1:
            return None
        own_returns = ind.daily_returns(closes)  # type: ignore[arg-type]
        bench_returns = ind.daily_returns(bench.close)
        return ind.rolling_correlation(own_returns, bench_returns, window)

    async def _open_risk(self, broker: BrokerKind) -> Decimal:
        """Sum of (entry - stop) * filled_qty across open, stopped intents."""
        rows = (
            (
                await self._session.execute(
                    select(TradeIntent).where(
                        TradeIntent.broker == broker,
                        TradeIntent.status.in_(
                            [TradeIntentStatus.SUBMITTED, TradeIntentStatus.RECONCILED]
                        ),
                        TradeIntent.stop_price.is_not(None),
                        TradeIntent.filled_price.is_not(None),
                    )
                )
            )
            .scalars()
            .all()
        )
        total = Decimal(0)
        for intent in rows:
            # The query filters these to non-null, but keep the guard explicit.
            if intent.filled_price is None or intent.stop_price is None:
                continue
            qty = Decimal(intent.filled_quantity or 0)
            distance = Decimal(intent.filled_price) - Decimal(intent.stop_price)
            if distance > 0:
                total += distance * qty
        return total

    async def _trades_today(self, broker: BrokerKind) -> int:
        start = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        count = await self._session.execute(
            select(func.count())
            .select_from(TradeIntent)
            .where(TradeIntent.broker == broker, TradeIntent.created_at >= start)
        )
        return int(count.scalar_one())
