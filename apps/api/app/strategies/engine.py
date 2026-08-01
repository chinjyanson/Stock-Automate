"""Strategy evaluation and execution (§8, §9).

Runs a configured strategy over its universe, turns each signal into an action,
and records what happened. The safety story is that **no signal reaches the broker
without passing a gate**:

  * A long *entry* is sized and gated by the full risk engine (via a proposal and
    `ExecutionService`), exactly like a scanner candidate. There is no route
    around it: every entry is risk-sized, without exception.
  * An *exit / trim* reduces risk, so it is always allowed (even under a halt) and
    filled directly.

Every signal produces a `StrategyDecision` recording its outcome, so "the strategy
wanted to but the risk engine said no" is visible, never silent. So does every
instrument the strategy could *not* evaluate for want of candle history — a
SKIPPED decision with no side. Without that row the two failure modes are
indistinguishable from the outside: a run that looked and found nothing, and a
run that had nothing to look at, both report zero signals.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.audit.service import AuditService
from app.broker.base import Broker
from app.broker.factory import resolve_broker
from app.broker.types import BrokerOrderRejectedError, BrokerOrderRequest
from app.data.store import CandleStore
from app.models.enums import (
    ActorKind,
    AuditEventKind,
    BrokerKind,
    Interval,
    OrderSide,
    OrderType,
    StrategyDecisionOutcome,
    StrategyRunStatus,
    TradeIntentStatus,
)
from app.models.instrument import BrokerInstrument, Instrument
from app.models.risk import TradeIntent
from app.models.scanner import ProposalStatus
from app.models.strategy import StrategyConfiguration, StrategyDecision, StrategyRun
from app.risk.execution import ExecutionError, ExecutionService
from app.risk.halts import HaltService
from app.scanner.proposals import ProposalError, ProposalInputs, ProposalService
from app.services.insider import InsiderIngestionService
from app.services.system_settings import active_broker_kind, autonomous_live_enabled
from app.strategies.base import InsiderPressure, StrategyContext, StrategySignal
from app.strategies.registry import build_strategy

log = structlog.get_logger(__name__)

#: Staleness thresholds, chosen per the interval the strategy actually reads.
#: Five days for daily bars tolerates a long weekend plus a holiday; 45 minutes
#: for intraday is three missed 15m refreshes. Applying the daily threshold to a
#: 15m strategy — as this did until the interval was threaded through — makes the
#: gate vacuous: bars days old would pass a check meant to catch minutes.
_DAILY_STALE_MAX_AGE = timedelta(days=5)
_INTRADAY_STALE_MAX_AGE = timedelta(minutes=45)


@dataclass
class StrategyRunSummary:
    run_id: uuid.UUID
    considered: int = 0
    signals: int = 0
    proposals: int = 0
    executed: int = 0
    rejected: int = 0
    #: Instruments the strategy could not evaluate for want of history. Counted
    #: separately from `rejected`: nothing was judged and refused, the data to
    #: judge it was not there.
    skipped: int = 0
    #: Instruments whose newest bar is older than the freshness threshold for the
    #: interval traded. They are still evaluated — the answer is just computed
    #: from old prices, which the caller deserves to know before trusting a
    #: quiet result.
    stale: int = 0
    errors: list[str] = field(default_factory=list)


class StrategyEngine:
    def __init__(self, session: AsyncSession, *, broker: Broker | None = None) -> None:
        self._session = session
        self._store = CandleStore(session)
        self._audit = AuditService(session)
        self._halts = HaltService(session)
        self._proposals = ProposalService(session)
        # An injected broker lets tests drive the autonomous-live path without
        # ever constructing the real live adapter. Defaults set per-run.
        self._injected_broker = broker
        self._broker_kind = BrokerKind.INTERNAL_PAPER
        self._propose_only = False

    async def run(
        self,
        config: StrategyConfiguration,
        *,
        selection_reason: str = "scheduled",
        actor_user_id: uuid.UUID | None = None,
    ) -> StrategyRunSummary:
        instruments = await self._universe(config)
        run = StrategyRun(
            configuration_id=config.id,
            kind=config.kind,
            status=StrategyRunStatus.RUNNING,
            started_at=datetime.now(UTC),
            instruments_considered=len(instruments),
            selection_reason=selection_reason,
        )
        self._session.add(run)
        await self._session.flush()
        await self._audit.record(
            kind=AuditEventKind.STRATEGY_RUN_STARTED,
            summary=f"Strategy '{config.name}' ({config.kind.value}) evaluating {len(instruments)}",
            actor_kind=ActorKind.USER if actor_user_id else ActorKind.SCHEDULER,
            actor_user_id=actor_user_id,
            subject_type="strategy_run",
            subject_id=str(run.id),
        )

        summary = StrategyRunSummary(run_id=run.id, considered=len(instruments))

        # Where do this run's orders go? The paper/live toggle picks the venue. On
        # live, a strategy may only fill without a human when autonomous is
        # separately enabled on the server; otherwise it just *proposes* and a
        # person approves each order.
        if self._injected_broker is not None:
            self._broker_kind = self._injected_broker.kind
        else:
            self._broker_kind = await active_broker_kind(self._session)
        self._propose_only = self._broker_kind.is_live and not await autonomous_live_enabled(
            self._session
        )
        broker = self._injected_broker or resolve_broker(self._broker_kind, session=self._session)
        try:
            await self._evaluate_and_act(run, config, broker, instruments, summary, actor_user_id)
        finally:
            if self._injected_broker is None:
                await broker.close()
        return summary

    async def _evaluate_and_act(
        self,
        run: StrategyRun,
        config: StrategyConfiguration,
        broker: Broker,
        instruments: list[Instrument],
        summary: StrategyRunSummary,
        actor_user_id: uuid.UUID | None,
    ) -> None:
        account = await broker.get_account()
        positions = await broker.get_positions()
        equity = (
            Decimal(config.account_equity) if config.account_equity is not None else account.total
        )

        strategy = build_strategy(config)

        # How old is the data this pass will read? Measured before evaluating and
        # regardless of the outcome, because the freshness gate in
        # `_blocked_reason` only fires on an entry attempt — a run that produces
        # no signal never reaches it, so evaluating days-old bars otherwise looks
        # identical to a healthy quiet run.
        summary.stale = await self._count_stale(instruments, strategy.read_interval)
        if summary.stale:
            log.warning(
                "strategy.stale_data",
                run_id=str(run.id),
                kind=config.kind.value,
                interval=strategy.read_interval.value,
                stale_instruments=summary.stale,
                of=len(instruments),
            )

        ctx = StrategyContext(
            config=config,
            store=self._store,
            instruments=instruments,
            positions=positions,
            insider_sell_pressure=await self._insider_pressure(instruments),
        )
        try:
            signals = await strategy.evaluate(ctx)
        except Exception as exc:  # a broken strategy must fail its run, not the worker
            log.exception("strategy.evaluate_failed", kind=config.kind.value)
            run.status = StrategyRunStatus.FAILED
            run.error = str(exc)
            run.completed_at = datetime.now(UTC)
            await self._session.flush()
            summary.errors.append(str(exc))
            return

        # Instruments the store could not serve enough history for produced no
        # signal because the strategy never saw them — record that explicitly, so
        # a silent run is distinguishable from a considered "no".
        for instrument_id, shortfall in ctx.insufficient_history.items():
            self._skip_decision(run, config, instrument_id, shortfall)
            summary.skipped += 1
        if ctx.insufficient_history:
            log.warning(
                "strategy.insufficient_history",
                run_id=str(run.id),
                kind=config.kind.value,
                instruments=len(ctx.insufficient_history),
            )

        summary.signals = len(signals)
        for signal in signals:
            await self._process(
                run, config, broker, signal, equity, summary, strategy.read_interval
            )

        run.status = StrategyRunStatus.COMPLETED
        run.completed_at = datetime.now(UTC)
        run.signals_generated = summary.signals
        run.proposals_created = summary.proposals
        run.executed = summary.executed
        run.rejected = summary.rejected
        await self._session.flush()
        await self._audit.record(
            kind=AuditEventKind.STRATEGY_RUN_COMPLETED,
            summary=(
                f"Strategy '{config.name}': {summary.signals} signals, "
                f"{summary.executed} executed, {summary.rejected} rejected, "
                f"{summary.skipped} not evaluated"
            ),
            actor_kind=ActorKind.USER if actor_user_id else ActorKind.SCHEDULER,
            actor_user_id=actor_user_id,
            subject_type="strategy_run",
            subject_id=str(run.id),
            payload={
                "signals": summary.signals,
                "executed": summary.executed,
                "rejected": summary.rejected,
                "skipped": summary.skipped,
            },
        )
        log.info(
            "strategy.run_completed",
            run_id=str(run.id),
            kind=config.kind.value,
            executed=summary.executed,
            skipped=summary.skipped,
        )

    # -- Per-signal handling ------------------------------------------------

    async def _process(
        self,
        run: StrategyRun,
        config: StrategyConfiguration,
        broker: Broker,
        signal: StrategySignal,
        equity: Decimal,
        summary: StrategyRunSummary,
        interval: Interval,
    ) -> None:
        instrument = await self._session.get(Instrument, signal.instrument_id)
        if instrument is None:
            return
        decision = self._decision(run, config, signal)

        # Approval-required arming: the engine only records; a human approves
        # each entry (which routes it live). It does not auto-act.
        if self._propose_only:
            if signal.side is OrderSide.BUY:
                await self._propose_only_entry(instrument, signal, decision, equity, summary)
            else:
                decision.outcome = StrategyDecisionOutcome.SIGNALLED
                decision.reason = f"{signal.reason} — recorded; awaiting human (armed approval)"
            return

        if signal.side is OrderSide.SELL:
            await self._exit(broker, instrument, signal, decision, summary)
            return

        # A long entry / add. Fail-closed gates first.
        blocked = await self._blocked_reason(instrument, interval)
        if blocked is not None:
            decision.outcome = StrategyDecisionOutcome.REJECTED_BY_RISK
            decision.reason = f"{signal.reason} — blocked: {blocked}"
            summary.rejected += 1
            return

        # Every entry is risk-sized. There is no targeted-quantity buy path: the
        # one strategy that had one (pie rebalancing, filling to a target weight)
        # is gone, and an un-risk-sized order is not something to keep a route
        # open for.
        await self._buy_sized(config, instrument, broker, signal, decision, equity, summary)

    async def _propose_only_entry(
        self,
        instrument: Instrument,
        signal: StrategySignal,
        decision: StrategyDecision,
        equity: Decimal,
        summary: StrategyRunSummary,
    ) -> None:
        """Create a pending proposal for a human to approve (armed, approval-required)."""
        try:
            proposal = await self._proposals.propose_from_signal(
                instrument, ProposalInputs(account_equity=equity), reason=signal.reason
            )
        except ProposalError as exc:
            decision.outcome = StrategyDecisionOutcome.SKIPPED
            decision.reason = f"{signal.reason} — skipped: {exc}"
            return
        decision.proposal_id = proposal.id
        summary.proposals += 1
        decision.outcome = StrategyDecisionOutcome.PROPOSED

    async def _buy_sized(
        self,
        config: StrategyConfiguration,
        instrument: Instrument,
        broker: Broker,
        signal: StrategySignal,
        decision: StrategyDecision,
        equity: Decimal,
        summary: StrategyRunSummary,
    ) -> None:
        try:
            proposal = await self._proposals.propose_from_signal(
                instrument, ProposalInputs(account_equity=equity), reason=signal.reason
            )
        except ProposalError as exc:
            decision.outcome = StrategyDecisionOutcome.SKIPPED
            decision.reason = f"{signal.reason} — skipped: {exc}"
            return

        decision.proposal_id = proposal.id
        summary.proposals += 1

        if not config.auto_execute:
            decision.outcome = StrategyDecisionOutcome.PROPOSED
            return

        # Auto-execute against this run's venue (paper, or live when autonomous),
        # running the full risk engine + fill. The broker is shared so positions
        # and preflight stay consistent.
        proposal.status = ProposalStatus.APPROVED
        proposal.decided_at = datetime.now(UTC)
        await self._session.flush()
        execution = ExecutionService(self._session, broker=broker)
        try:
            executed = await execution.execute_approved(proposal, actor_user_id=None)
        except ExecutionError as exc:
            decision.outcome = StrategyDecisionOutcome.REJECTED_BY_RISK
            decision.reason = f"{signal.reason} — execution refused: {exc}"
            summary.rejected += 1
            return

        if executed.status is ProposalStatus.EXECUTED:
            decision.outcome = StrategyDecisionOutcome.EXECUTED
            summary.executed += 1
        else:
            decision.outcome = StrategyDecisionOutcome.REJECTED_BY_RISK
            summary.rejected += 1

    async def _exit(
        self,
        broker: Broker,
        instrument: Instrument,
        signal: StrategySignal,
        decision: StrategyDecision,
        summary: StrategyRunSummary,
    ) -> None:
        """Sell to close or trim. Always permitted — it reduces risk."""
        ticker = await self._venue_ticker(instrument.id)
        if ticker is None:
            decision.outcome = StrategyDecisionOutcome.SKIPPED
            decision.reason = f"{signal.reason} — instrument not available on the venue"
            return
        current = await broker.get_positions()
        held = sum(
            (Decimal(p.quantity) for p in current if p.broker_ticker == ticker),
            start=Decimal(0),
        )
        qty = min(held, signal.target_quantity) if signal.target_quantity is not None else held
        if qty <= 0:
            decision.outcome = StrategyDecisionOutcome.SKIPPED
            decision.reason = f"{signal.reason} — nothing held to sell"
            return
        try:
            order = await broker.place_order(
                BrokerOrderRequest(
                    broker_ticker=ticker,
                    side=OrderSide.SELL,
                    quantity=qty,
                    order_type=OrderType.MARKET,
                )
            )
        except BrokerOrderRejectedError as exc:
            decision.outcome = StrategyDecisionOutcome.SKIPPED
            decision.reason = f"{signal.reason} — venue refused: {exc}"
            return

        # If the position is now flat, close its open intent so P/L is booked.
        if qty >= held:
            await self._close_intent(instrument.id, order.average_fill_price)
        decision.outcome = StrategyDecisionOutcome.EXECUTED
        summary.executed += 1

    # -- Helpers -----------------------------------------------------------

    def _decision(
        self, run: StrategyRun, config: StrategyConfiguration, signal: StrategySignal
    ) -> StrategyDecision:
        decision = StrategyDecision(
            run_id=run.id,
            configuration_id=config.id,
            instrument_id=signal.instrument_id,
            kind=config.kind,
            side=signal.side,
            conviction=Decimal(str(signal.conviction)),
            outcome=StrategyDecisionOutcome.SIGNALLED,
            reason=signal.reason,
            metrics={k: float(v) for k, v in signal.metrics.items()},
        )
        self._session.add(decision)
        return decision

    def _skip_decision(
        self,
        run: StrategyRun,
        config: StrategyConfiguration,
        instrument_id: uuid.UUID,
        reason: str,
    ) -> StrategyDecision:
        """Record an instrument the strategy could not evaluate at all.

        `side` is null and conviction zero because nothing was decided — there
        was no opinion to have. Without this row the only trace of a data
        outage is a run whose `signals` count is zero, which reads exactly like
        a healthy run that found no setup.
        """
        decision = StrategyDecision(
            run_id=run.id,
            configuration_id=config.id,
            instrument_id=instrument_id,
            kind=config.kind,
            side=None,
            conviction=Decimal(0),
            outcome=StrategyDecisionOutcome.SKIPPED,
            reason=f"Not evaluated: {reason}",
        )
        self._session.add(decision)
        return decision

    async def _venue_ticker(self, instrument_id: uuid.UUID) -> str | None:
        """The current venue's ticker: instrument id for paper, BrokerInstrument for live."""
        if self._broker_kind is BrokerKind.INTERNAL_PAPER:
            return str(instrument_id)
        bi = (
            (
                await self._session.execute(
                    select(BrokerInstrument).where(
                        BrokerInstrument.instrument_id == instrument_id,
                        BrokerInstrument.broker == self._broker_kind,
                    )
                )
            )
            .scalars()
            .first()
        )
        if bi is None or not bi.is_currently_available:
            return None
        return bi.broker_ticker

    async def _insider_pressure(
        self, instruments: list[Instrument]
    ) -> dict[uuid.UUID, InsiderPressure]:
        """Insider selling pressure across the universe, resolved once per run.

        Looked up here rather than inside the strategy so `Strategy.evaluate`
        keeps its contract of reading only what it is handed, and stays testable
        against fixtures with no database.

        Failures degrade to an empty map. This is an optional overlay on a
        working strategy: losing it must return the strategy to its pre-insider
        behaviour, never stop it evaluating.
        """
        service = InsiderIngestionService(self._session)
        pressure: dict[uuid.UUID, InsiderPressure] = {}
        for instrument in instruments:
            try:
                score = await service.score_instrument(instrument.id)
            except Exception as exc:
                log.warning(
                    "strategy.insider_pressure_failed",
                    instrument_id=str(instrument.id),
                    error=str(exc),
                )
                continue
            if score is not None and score.sell_penalty > 0:
                pressure[instrument.id] = InsiderPressure(
                    sell_penalty=score.sell_penalty,
                    move_atr=score.move_since_filing_atr,
                )
        return pressure

    async def _count_stale(self, instruments: list[Instrument], interval: Interval) -> int:
        """How many instruments hold no bar newer than the freshness threshold.

        Shares its threshold with `_blocked_reason`, so the count and the gate
        can never disagree about what "stale" means.
        """
        max_age = _INTRADAY_STALE_MAX_AGE if interval.is_intraday else _DAILY_STALE_MAX_AGE
        stale = 0
        for instrument in instruments:
            if await self._store.is_stale(instrument.id, interval, max_age=max_age):
                stale += 1
        return stale

    async def _blocked_reason(self, instrument: Instrument, interval: Interval) -> str | None:
        """Halt and freshness gates for an entry, against the bars actually traded.

        `interval` is the strategy's read interval, not a constant: a 15m
        strategy entering on bars that are hours old is exactly what the
        freshness gate exists to stop, and checking daily bars instead would let
        it through.
        """
        halt = await self._halts.blocking_halt(instrument.id)
        if halt is not None:
            return f"halt {halt.kind.value}"
        max_age = _INTRADAY_STALE_MAX_AGE if interval.is_intraday else _DAILY_STALE_MAX_AGE
        if await self._store.is_stale(instrument.id, interval, max_age=max_age):
            return f"stale {interval.value} data (older than {max_age})"
        return None

    async def _close_intent(self, instrument_id: uuid.UUID, exit_price: Decimal | None) -> None:
        intent = (
            (
                await self._session.execute(
                    select(TradeIntent).where(
                        TradeIntent.instrument_id == instrument_id,
                        TradeIntent.broker == self._broker_kind,
                        TradeIntent.status == TradeIntentStatus.RECONCILED,
                        TradeIntent.closed_at.is_(None),
                    )
                )
            )
            .scalars()
            .first()
        )
        if intent is not None:
            intent.closed_at = datetime.now(UTC)
            intent.exit_price = exit_price
            intent.exit_reason = "strategy_exit"

    async def _universe(self, config: StrategyConfiguration) -> list[Instrument]:
        ids: set[uuid.UUID] = set()
        universe = config.universe or {}
        for raw in universe.get("instrument_ids", []):
            try:
                ids.add(uuid.UUID(str(raw)))
            except (ValueError, TypeError):
                continue
        for raw in universe.get("weights") or {}:
            try:
                ids.add(uuid.UUID(str(raw)))
            except (ValueError, TypeError):
                continue
        if not ids:
            return []
        rows = await self._session.execute(select(Instrument).where(Instrument.id.in_(ids)))
        return list(rows.scalars().all())
