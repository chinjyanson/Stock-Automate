"""Position sizing and the pre-trade gate (§9).

`evaluate` is the single chokepoint every order passes. It sizes volatility-
adjusted (constant *risk*, not constant size), then caps that by every limit and
lets the **smallest** cap win, rounding down so no cap is breached by the
rounding. Before any of that it fails closed: an active halt, stale data, or a
missing configuration all mean "no trade", stated as a reason rather than a
silent skip.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import ROUND_DOWN, Decimal
from typing import Any

import numpy as np
import structlog
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.broker.types import BrokerAccount, BrokerPosition
from app.data.store import CandleStore
from app.indicators import functions as ind
from app.indicators.functions import FloatArray
from app.indicators.series import PriceSeries, candles_to_series
from app.models.enums import BrokerKind, Interval, TradeIntentStatus
from app.models.instrument import Instrument
from app.models.market_data import Candle
from app.models.risk import RiskConfiguration, TradeIntent
from app.risk import stress
from app.risk.halts import HaltService
from app.services.market_regime import MarketRegimeService

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

#: Daily bars loaded per held position for the whole-book stress test. A year is
#: enough to include a drawdown or two without making the resample a survey of
#: ancient history the current book has nothing to do with.
STRESS_HISTORY_BARS = 260

#: Key the candidate is filed under while it is simulated alongside the book. A
#: UUID is never this, so it cannot collide with a real instrument id.
_CANDIDATE_KEY = "candidate"

#: Fixed seed for the stress simulation. The same book must reach the same
#: verdict every time it is evaluated — a position approved on one run and cut on
#: the next, with nothing changed but the draw, would be impossible to audit or
#: to tune against.
STRESS_SEED = 20260806


def _stress_rng() -> np.random.Generator:
    return np.random.default_rng(STRESS_SEED)


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
    rate_correlation: float | None = None
    #: Simulated 20-day tail loss of the whole book with this position added,
    #: as a positive fraction. None when it could not be measured.
    stress_loss_pct: float | None = None
    #: The market-regime multiplier that scaled the risk budget.
    regime_factor: float | None = None

    def as_record(self) -> dict[str, Any]:
        """The verdict as a JSON-safe dict, for the audit trail.

        Which cap bound a trade, and what the correlation and stress readings
        were when it did, is the only evidence available for tuning these limits
        later — or for answering "why was this position small?" months after the
        fact. A reason string cannot carry it, so it is recorded structurally.
        """
        return {
            "approved_quantity": str(self.approved_quantity),
            "entry_price": str(self.entry_price),
            "stop_price": str(self.stop_price) if self.stop_price is not None else None,
            "risk_amount": str(self.risk_amount),
            "rejected": self.rejected,
            "reason": self.reason,
            "applied_caps": list(self.applied_caps),
            "correlation": self.correlation,
            "rate_correlation": self.rate_correlation,
            "stress_loss_pct": self.stress_loss_pct,
            "regime_factor": self.regime_factor,
        }

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
        rates_candles: list[Candle] | None = None,
        broker: BrokerKind = BrokerKind.INTERNAL_PAPER,
        equity_ceiling: Decimal | None = None,
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
            return RiskDecision.reject(f"At the daily trade cap ({config.max_trades_per_day}).")

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

        # For live, the arming session's affirmed capital ceiling bounds the
        # equity the whole sizing scales against — you cannot deploy more than
        # you said you would, even if the account holds more.
        equity = account.total
        if equity_ceiling is not None:
            equity = min(equity, equity_ceiling)
        if equity <= 0:
            return RiskDecision.reject("Account equity is not positive.")
        # Market-wide risk posture scales the per-trade budget. Floored well
        # above zero by construction (see MIN_RISK_FACTOR): a bad regime makes
        # every position smaller, it never stops the strategy trading. Absent
        # measurement means 1.0 — missing information must not masquerade as a
        # risk-off signal.
        regime = await MarketRegimeService(self._session).current_risk_factor()
        regime_factor = Decimal(str(regime))
        risk_budget = equity * Decimal(str(config.risk_per_trade_pct)) * regime_factor
        applied_regime = f"market regime x{regime_factor}" if regime_factor < 1 else None
        raw_quantity = risk_budget / stop_distance

        # 4. Caps — smallest wins. -----------------------------------------
        caps: dict[str, Decimal] = {"risk_budget": raw_quantity}

        caps["max_position_pct"] = equity * Decimal(str(config.max_position_pct)) / entry_price

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

        # Cash available, but never more than the (possibly ceiling-bounded) equity.
        caps["available_cash"] = min(account.free_for_trading, equity) / entry_price

        open_risk = await self._open_risk(broker)
        open_risk_room = equity * Decimal(str(config.max_total_open_risk_pct)) - open_risk
        caps["max_total_open_risk"] = max(open_risk_room, Decimal(0)) / stop_distance

        if config.monetary_position_cap is not None:
            caps["monetary_cap"] = Decimal(str(config.monetary_position_cap)) / entry_price

        # Total market exposure. Previously modelled and documented but never
        # enforced, so `docs/risk-model.md` claimed a bound that did not exist.
        invested = sum(
            (p.quantity * (p.current_price or p.average_price) for p in positions),
            start=Decimal(0),
        )
        exposure_room = equity * Decimal(str(config.max_portfolio_exposure_pct)) - invested
        caps["max_portfolio_exposure_pct"] = max(exposure_room, Decimal(0)) / entry_price

        # Whole-book stress. Every cap above sizes this position against this
        # position; this one asks what the *portfolio* does in a bad month with
        # the candidate added, which is the only cap that can see six positions
        # that are really one bet. Absent when it cannot be computed, in which
        # case it simply does not participate.
        stress_cap, stress_loss_pct = await self._stress_cap(
            positions, series, entry_price, equity, config, raw_quantity
        )
        if stress_cap is not None:
            caps["stress_drawdown"] = stress_cap

        # The binding cap is the smallest allowance.
        binding_cap = min(caps, key=lambda k: caps[k])
        quantity = caps[binding_cap]

        # 5. Correlation — reduce (never merely warn). ---------------------
        # Six "diversified" positions that are all really one S&P bet is the
        # failure this prevents. If the candidate is benchmark-correlated *and*
        # the book already carries too much benchmark-correlated exposure, cut
        # the size. Exposure is measured position-by-position (each holding's own
        # correlation to the benchmark), not by gross invested — that is the
        # Phase 4 refinement over the earlier approximation.
        #
        # Rate sensitivity is the same shape of mistake against a different
        # reference: a book of REITs, utilities and long-duration growth is one
        # bet on yields however uncorrelated those names look to each other.
        # Measured on magnitude, so a strongly *negatively* rate-correlated book
        # counts too — that is still a rates bet, just the other way round.
        correlation: float | None = None
        rate_correlation: float | None = None
        applied_caps = [binding_cap]
        if applied_regime:
            applied_caps.append(applied_regime)

        reductions: list[str] = []
        if benchmark_candles:
            correlation = self._correlation(series.close, benchmark_candles, config)
            if correlation is not None and correlation > float(config.correlation_threshold):
                exposure = await self._correlated_exposure(
                    positions, benchmark_candles, config, equity
                )
                if exposure > Decimal(str(config.max_portfolio_sp500_pct)):
                    reductions.append("correlation_reduction")
        if rates_candles:
            rate_correlation = self._correlation(series.close, rates_candles, config)
            if rate_correlation is not None and abs(rate_correlation) > float(
                config.correlation_threshold
            ):
                exposure = await self._correlated_exposure(
                    positions, rates_candles, config, equity, use_abs=True
                )
                if exposure > Decimal(str(config.max_portfolio_rate_sensitive_pct)):
                    reductions.append("rate_correlation_reduction")
        if reductions:
            # At most one cut, however many gates fired. Two independent x0.5
            # multipliers would give x0.25, which neither rule intends — being
            # concentrated in two ways is not twice as bad as being concentrated
            # in one. Every gate that fired is still named, so the audit trail
            # shows the full reason.
            quantity = quantity * CORRELATION_REDUCTION
            applied_caps.extend(reductions)

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
            rate_correlation=rate_correlation,
            stress_loss_pct=stress_loss_pct,
            regime_factor=regime,
        )

    # -- Helpers -----------------------------------------------------------

    def _correlation(
        self,
        closes: object,
        reference_candles: list[Candle],
        config: RiskConfiguration,
    ) -> float | None:
        bench = candles_to_series(reference_candles)
        window = int(config.correlation_window_short)
        if bench.length < window + 1:
            return None
        own_returns = ind.daily_returns(closes)  # type: ignore[arg-type]
        bench_returns = ind.daily_returns(bench.close)
        return ind.rolling_correlation(own_returns, bench_returns, window)

    async def _correlated_exposure(
        self,
        positions: list[BrokerPosition],
        reference_candles: list[Candle],
        config: RiskConfiguration,
        equity: Decimal,
        *,
        use_abs: bool = False,
    ) -> Decimal:
        """Fraction of equity held in positions that track `reference_candles`.

        For each open position, correlate its own daily returns against the
        reference series; sum the value of those above the threshold. This is the
        real "how much of the book is one bet" measure the sizing reduction acts
        on, replacing the earlier gross-invested approximation.

        The reference is the S&P benchmark for the market-beta check and the
        rates proxy for the rate-sensitivity check. `use_abs` compares on
        magnitude, which is what rates need: moving hard *against* yields is as
        much a rates position as moving with them, whereas a holding that is
        strongly negatively correlated to the market is genuine diversification
        and must not be counted as concentration.
        """
        if not reference_candles or equity <= 0:
            return Decimal(0)
        bench = candles_to_series(reference_candles)
        window = int(config.correlation_window_short)
        if bench.length < window + 1:
            return Decimal(0)
        bench_returns = ind.daily_returns(bench.close)
        threshold = float(config.correlation_threshold)

        correlated_value = Decimal(0)
        for position in positions:
            if position.quantity <= 0:
                continue
            try:
                instrument_id = uuid.UUID(position.broker_ticker)
            except ValueError:
                continue  # non-paper venues key by ticker, not instrument id
            candles = await self._store.get_candles(
                instrument_id, Interval.D1, limit=window + 5, closed_only=True
            )
            if len(candles) < window + 1:
                continue
            series = candles_to_series(candles)
            corr = ind.rolling_correlation(ind.daily_returns(series.close), bench_returns, window)
            if corr is not None and (abs(corr) if use_abs else corr) > threshold:
                price = position.current_price or position.average_price
                correlated_value += position.quantity * price
        return correlated_value / equity

    async def _stress_cap(
        self,
        positions: list[BrokerPosition],
        candidate: PriceSeries,
        entry_price: Decimal,
        equity: Decimal,
        config: RiskConfiguration,
        proposed_quantity: Decimal,
    ) -> tuple[Decimal | None, float | None]:
        """Largest quantity that keeps the whole book inside its drawdown limit.

        Returns `(cap, stressed_loss)`, either of which may be None. A None cap
        means the stress test did not bind — because it could not be computed
        (no positions, too little history) or because the book is comfortably
        inside the limit. It never means zero: an unmeasurable stress must not
        silently block a trade.

        A cap of exactly zero *is* meaningful, and is returned when the existing
        book already breaches the limit on its own. Shrinking the candidate
        cannot fix that, so the right answer is to add nothing to it.
        """
        limit = float(config.max_portfolio_drawdown_pct)
        if limit <= 0 or equity <= 0:
            return None, None

        returns: dict[str, FloatArray] = {}
        weights: dict[str, float] = {}
        for position in positions:
            if position.quantity <= 0:
                continue
            try:
                instrument_id = uuid.UUID(position.broker_ticker)
            except ValueError:
                continue  # non-paper venues key by ticker, not instrument id
            candles = await self._store.get_candles(
                instrument_id, Interval.D1, limit=STRESS_HISTORY_BARS, closed_only=True
            )
            if len(candles) < stress.MIN_RETURNS + 1:
                continue
            key = str(instrument_id)
            returns[key] = ind.daily_returns(candles_to_series(candles).close)
            price = position.current_price or position.average_price
            weights[key] = float(position.quantity * price / equity)

        held = stress.bootstrap_stress(returns, weights, rng=_stress_rng())
        if held is not None and held.portfolio_loss_pct >= limit:
            # Already over the limit before this trade. Nothing to allocate.
            return Decimal(0), held.portfolio_loss_pct

        # Add the candidate at its proposed size and re-measure.
        candidate_weight = float(proposed_quantity * entry_price / equity)
        returns[_CANDIDATE_KEY] = ind.daily_returns(candidate.close)
        weights[_CANDIDATE_KEY] = candidate_weight
        result = stress.bootstrap_stress(returns, weights, rng=_stress_rng())
        if result is None:
            return None, held.portfolio_loss_pct if held else None
        if result.portfolio_loss_pct <= limit:
            return None, result.portfolio_loss_pct
        return (
            stress.drawdown_scaled_quantity(proposed_quantity, result.portfolio_loss_pct, limit),
            result.portfolio_loss_pct,
        )

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
