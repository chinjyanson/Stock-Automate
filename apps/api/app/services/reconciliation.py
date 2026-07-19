"""Scheduled broker reconciliation (§10, §16).

Compares a venue's truth against local `TradeIntent` records and reacts to the
result as *state*:

  * **Diverged** — raise a global reconciliation halt and fail closed. Trading
    stops until the divergence is understood; "the broker shows something we do
    not" is exactly the condition where continuing to trade compounds a mistake.
  * **Clean** — resolve intents left in RECONCILIATION_REQUIRED (their outcome is
    now known), and clear a reconciliation halt the engine raised earlier, since
    the condition behind it is demonstrably gone.

A user's kill switch is never touched here — `clear_kind` only clears halts with
no human actor.
"""

from __future__ import annotations

from datetime import UTC, datetime

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.audit.service import AuditService
from app.broker.base import Broker
from app.broker.types import ReconciliationResult
from app.models.enums import ActorKind, AuditEventKind, HaltKind, HaltScope, TradeIntentStatus
from app.models.risk import TradeIntent
from app.risk.halts import HaltService

log = structlog.get_logger(__name__)


class ReconciliationService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._audit = AuditService(session)
        self._halts = HaltService(session)

    async def run(self, broker: Broker) -> ReconciliationResult:
        result = await broker.reconcile()
        if result.is_clean:
            await self._on_clean(broker)
        else:
            await self._on_divergence(broker, result)
        await self._session.flush()
        return result

    async def _on_clean(self, broker: Broker) -> None:
        resolved = await self._resolve_pending_intents(broker)
        cleared = await self._halts.clear_kind(HaltKind.RECONCILIATION)
        if resolved or cleared:
            await self._audit.record(
                kind=AuditEventKind.RECONCILIATION_RESOLVED,
                summary=(
                    f"Reconciliation clean for {broker.kind.value}: "
                    f"{resolved} intent(s) resolved, {cleared} halt(s) cleared"
                ),
                actor_kind=ActorKind.SCHEDULER,
                actor_label="reconciliation_job",
                subject_type="broker",
                subject_id=broker.kind.value,
            )
        log.info("reconciliation.clean", broker=broker.kind.value, resolved=resolved)

    async def _on_divergence(self, broker: Broker, result: ReconciliationResult) -> None:
        detail = "; ".join(d.detail for d in result.discrepancies) or "unspecified divergence"
        await self._halts.activate(
            HaltKind.RECONCILIATION,
            f"{broker.kind.value}: {detail}",
            scope=HaltScope.GLOBAL,
        )
        await self._audit.record(
            kind=AuditEventKind.RECONCILIATION_REQUIRED,
            summary=(
                f"Reconciliation found {len(result.discrepancies)} discrepancy(ies) "
                f"for {broker.kind.value}; trading halted"
            ),
            actor_kind=ActorKind.SCHEDULER,
            actor_label="reconciliation_job",
            subject_type="broker",
            subject_id=broker.kind.value,
            payload={"discrepancies": [d.detail for d in result.discrepancies]},
        )
        log.warning(
            "reconciliation.diverged",
            broker=broker.kind.value,
            discrepancies=len(result.discrepancies),
        )

    async def _resolve_pending_intents(self, broker: Broker) -> int:
        """Move intents stuck in RECONCILIATION_REQUIRED to a terminal state.

        A clean reconciliation means the broker's view is authoritative and
        matches ours; an intent still marked "reconciliation required" whose
        order the venue now shows is reconciled, otherwise abandoned.
        """
        order_ids = {o.broker_order_id for o in await broker.get_order_history()}
        rows = (
            (
                await self._session.execute(
                    select(TradeIntent).where(
                        TradeIntent.broker == broker.kind,
                        TradeIntent.status == TradeIntentStatus.RECONCILIATION_REQUIRED,
                    )
                )
            )
            .scalars()
            .all()
        )
        now = datetime.now(UTC)
        for intent in rows:
            if intent.broker_order_id and intent.broker_order_id in order_ids:
                intent.status = TradeIntentStatus.RECONCILED
                intent.reconciled_at = now
            else:
                # The order never reached the venue; the intent is dead, not open.
                intent.status = TradeIntentStatus.ABANDONED
        return len(rows)
