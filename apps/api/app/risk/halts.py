"""Trading halts as first-class state (§9).

A halt is a row, activated and cleared explicitly and audited both ways. The
engine consults `blocking_halt` before it sizes anything; a halt that is active
means the answer is already "no" regardless of how good the trade looks.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.audit.service import AuditService
from app.models.enums import ActorKind, AuditEventKind, HaltKind, HaltScope
from app.models.risk import RiskHalt

log = structlog.get_logger(__name__)


class HaltService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._audit = AuditService(session)

    async def active_halts(self, instrument_id: uuid.UUID | None = None) -> list[RiskHalt]:
        """All active halts that would block an order for `instrument_id`.

        Global halts always apply; instrument halts apply only to their
        instrument. Strategy halts are returned too (callers that know their
        strategy can filter), but do not gate a plain instrument order.
        """
        stmt = select(RiskHalt).where(RiskHalt.is_active.is_(True))
        result = (await self._session.execute(stmt)).scalars().all()
        halts: list[RiskHalt] = []
        for halt in result:
            if halt.scope is HaltScope.GLOBAL or (
                halt.scope is HaltScope.INSTRUMENT
                and instrument_id is not None
                and halt.instrument_id == instrument_id
            ):
                halts.append(halt)
        return halts

    async def blocking_halt(self, instrument_id: uuid.UUID | None = None) -> RiskHalt | None:
        """The first halt blocking this order, or None. Global halts win."""
        halts = await self.active_halts(instrument_id)
        halts.sort(key=lambda h: 0 if h.scope is HaltScope.GLOBAL else 1)
        return halts[0] if halts else None

    async def activate(
        self,
        kind: HaltKind,
        reason: str,
        *,
        scope: HaltScope = HaltScope.GLOBAL,
        instrument_id: uuid.UUID | None = None,
        strategy_key: str | None = None,
        actor_user_id: uuid.UUID | None = None,
    ) -> RiskHalt:
        """Record and audit a new halt. Idempotent per (kind, scope, target).

        Re-activating an already-active halt of the same kind and target returns
        the existing row rather than stacking duplicates.
        """
        existing = (
            await self._session.execute(
                select(RiskHalt).where(
                    RiskHalt.is_active.is_(True),
                    RiskHalt.kind == kind,
                    RiskHalt.scope == scope,
                    RiskHalt.instrument_id == instrument_id,
                )
            )
        ).scalar_one_or_none()
        if existing is not None:
            return existing

        now = datetime.now(UTC)
        halt = RiskHalt(
            kind=kind,
            scope=scope,
            instrument_id=instrument_id,
            strategy_key=strategy_key,
            is_active=True,
            reason=reason,
            activated_by_user_id=actor_user_id,
            activated_at=now,
        )
        self._session.add(halt)
        await self._session.flush()

        event = (
            AuditEventKind.KILL_SWITCH_ACTIVATED
            if kind is HaltKind.KILL_SWITCH
            else AuditEventKind.RISK_HALT_ACTIVATED
        )
        await self._audit.record(
            kind=event,
            summary=f"Risk halt activated: {kind.value} ({scope.value}) — {reason}",
            actor_kind=ActorKind.USER if actor_user_id else ActorKind.RISK_ENGINE,
            actor_user_id=actor_user_id,
            subject_type="risk_halt",
            subject_id=str(halt.id),
            payload={"kind": kind.value, "scope": scope.value, "reason": reason},
        )
        log.warning("risk.halt_activated", kind=kind.value, scope=scope.value, reason=reason)
        return halt

    async def clear(
        self, halt_id: uuid.UUID, *, actor_user_id: uuid.UUID | None = None
    ) -> RiskHalt:
        """Clear a halt.

        A user clears the blunt and surgical halts by hand. A system actor
        (`actor_user_id=None`) is permitted only for condition-based halts the
        engine itself raised and can see resolved — a stale reconciliation that
        is now clean. This is not self-clearing on restart, which the model
        forbids; it is clearing because the condition that caused it is gone.
        """
        halt = await self._session.get(RiskHalt, halt_id)
        if halt is None:
            raise ValueError("Halt not found")
        if not halt.is_active:
            return halt
        halt.is_active = False
        halt.cleared_by_user_id = actor_user_id
        halt.cleared_at = datetime.now(UTC)
        await self._session.flush()

        await self._audit.record(
            kind=AuditEventKind.RISK_HALT_CLEARED,
            summary=f"Risk halt cleared: {halt.kind.value} ({halt.scope.value})",
            actor_kind=ActorKind.USER if actor_user_id else ActorKind.RISK_ENGINE,
            actor_user_id=actor_user_id,
            subject_type="risk_halt",
            subject_id=str(halt.id),
        )
        log.warning("risk.halt_cleared", kind=halt.kind.value, halt_id=str(halt.id))
        return halt

    async def clear_kind(self, kind: HaltKind) -> int:
        """Clear all active halts of one kind that the engine raised. Returns count.

        Used when the condition behind a system halt has demonstrably cleared —
        e.g. a reconciliation that now matches. Only touches halts with no user
        actor, so it can never silently undo a human's kill switch.
        """
        halts = (
            (
                await self._session.execute(
                    select(RiskHalt).where(
                        RiskHalt.is_active.is_(True),
                        RiskHalt.kind == kind,
                        RiskHalt.activated_by_user_id.is_(None),
                    )
                )
            )
            .scalars()
            .all()
        )
        for halt in halts:
            await self.clear(halt.id)
        return len(halts)

    async def kill_switch(self, reason: str, *, actor_user_id: uuid.UUID) -> RiskHalt:
        """The blunt override: a global halt that stops all trading at once."""
        return await self.activate(
            HaltKind.KILL_SWITCH,
            reason,
            scope=HaltScope.GLOBAL,
            actor_user_id=actor_user_id,
        )
