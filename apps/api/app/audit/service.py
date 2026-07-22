"""Audit log writer and verifier (§17, §20).

Appends are serialised through a PostgreSQL advisory lock. That is a real cost
on a hot path, and it is accepted deliberately: a hash chain is only meaningful
if every writer agrees on what "the previous event" was. Two concurrent appends
reading the same tip would produce two events claiming the same predecessor,
and the chain would fork — which is indistinguishable from tampering, and would
make the whole mechanism worthless exactly when it is needed.

Audit volume is low (orders, approvals, credential changes), so serialising is
affordable. If it ever is not, the answer is a dedicated append worker, not a
weaker chain.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence as TypingSequence
from datetime import UTC, datetime
from typing import Any

import structlog
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.audit import GENESIS_HASH, AuditEvent
from app.models.enums import ActorKind, AuditEventKind

log = structlog.get_logger(__name__)

#: Arbitrary but fixed application-wide lock id for audit appends.
_AUDIT_ADVISORY_LOCK_ID = 4_812_337_001

#: Keys scrubbed from any payload before it is written. Defence in depth: the
#: call sites are not supposed to pass these, and this ensures a careless one
#: cannot leak a credential into permanent, immutable storage.
_REDACTED_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "password",
        "secret",
        "token",
        "session_token",
        "private_key",
        "encrypted_api_key",
    }
)

_REDACTION_PLACEHOLDER = "[redacted]"


def _redact(value: Any, _depth: int = 0) -> Any:
    """Recursively strip secret-looking keys from an audit payload."""
    if _depth > 6:
        return value
    if isinstance(value, dict):
        return {
            key: (
                _REDACTION_PLACEHOLDER
                if key.lower() in _REDACTED_KEYS
                else _redact(item, _depth + 1)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact(item, _depth + 1) for item in value]
    return value


class AuditService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def _acquire_append_lock(self) -> None:
        """Serialise appends within this transaction.

        Transaction-scoped (`pg_advisory_xact_lock`), so it releases on commit
        or rollback with no unlock bookkeeping and no risk of a crashed process
        holding it forever.
        """
        bind = self._session.get_bind()
        if bind.dialect.name != "postgresql":
            # SQLite unit tests run single-threaded; there is nothing to serialise.
            return
        await self._session.execute(
            text("SELECT pg_advisory_xact_lock(:lock_id)"),
            {"lock_id": _AUDIT_ADVISORY_LOCK_ID},
        )

    async def _current_tip_hash(self) -> str:
        result = await self._session.execute(
            select(AuditEvent.event_hash).order_by(AuditEvent.sequence.desc()).limit(1)
        )
        return result.scalar_one_or_none() or GENESIS_HASH

    async def _next_sequence(self) -> int:
        """Claim the next chain position *before* inserting.

        The obvious implementation — insert, flush to let the column default
        assign `sequence`, then update the row with its hash — cannot work here:
        `sequence` is part of the hashed material, so the hash is unknown until
        the row exists, and the immutability trigger correctly rejects the
        follow-up UPDATE. (The trigger caught exactly this during development.)

        Drawing the value from the sequence up front makes the write a single
        INSERT of an already-complete, already-hashed row, which is what an
        append-only table requires.
        """
        bind = self._session.get_bind()
        if bind.dialect.name == "postgresql":
            # nextval is atomic and never reuses a value, even across rollbacks.
            # A rolled-back append therefore leaves a gap in the sequence rather
            # than a duplicate — gaps are harmless to the chain, duplicates
            # would break it.
            result = await self._session.execute(
                text("SELECT nextval('audit_events_sequence_seq')")
            )
            return int(result.scalar_one())

        # SQLite (unit tests): no sequences. The append lock is a no-op there
        # too, but tests are single-threaded so max+1 is sufficient.
        result = await self._session.execute(select(func.max(AuditEvent.sequence)))
        return int(result.scalar_one_or_none() or 0) + 1

    async def record(
        self,
        *,
        kind: AuditEventKind,
        summary: str,
        actor_kind: ActorKind = ActorKind.SYSTEM,
        actor_user_id: uuid.UUID | None = None,
        actor_label: str | None = None,
        subject_type: str | None = None,
        subject_id: str | None = None,
        payload: dict[str, Any] | None = None,
        request_id: str | None = None,
        trade_intent_id: str | None = None,
        strategy_decision_id: str | None = None,
        occurred_at: datetime | None = None,
    ) -> AuditEvent:
        """Append one immutable event.

        The caller's transaction owns the commit. An audit record and the change
        it describes must land together or not at all — an order that committed
        without its audit row, or vice versa, is a worse outcome than either
        failing.
        """
        await self._acquire_append_lock()
        previous_hash = await self._current_tip_hash()
        sequence = await self._next_sequence()

        event = AuditEvent(
            id=uuid.uuid4(),
            sequence=sequence,
            occurred_at=occurred_at or datetime.now(UTC),
            kind=kind,
            actor_kind=actor_kind,
            actor_user_id=actor_user_id,
            actor_label=actor_label,
            subject_type=subject_type,
            subject_id=subject_id,
            summary=summary,
            payload=_redact(payload) if payload else None,
            request_id=request_id,
            trade_intent_id=trade_intent_id,
            strategy_decision_id=strategy_decision_id,
            previous_hash=previous_hash,
        )
        # The row is fully determined before it is written: hash last, insert
        # once, never update.
        event.event_hash = event.compute_hash()
        self._session.add(event)
        await self._session.flush()

        log.info(
            "audit.recorded",
            kind=str(kind),
            sequence=event.sequence,
            subject_type=subject_type,
            subject_id=subject_id,
            actor_kind=str(actor_kind),
        )
        return event

    async def verify_chain(self, *, limit: int | None = None) -> tuple[bool, list[str]]:
        """Walk the chain and report breaks.

        Returns (is_intact, problems). A break means a row was altered, removed
        or inserted out of band — the trigger should make that impossible, so a
        failure here is a genuine incident, not a routine check failing.
        """
        stmt = select(AuditEvent).order_by(AuditEvent.sequence.asc())
        if limit is not None:
            stmt = stmt.limit(limit)
        result = await self._session.execute(stmt)
        events: TypingSequence[AuditEvent] = list(result.scalars().all())

        problems: list[str] = []
        expected_previous = GENESIS_HASH

        for event in events:
            if event.previous_hash != expected_previous:
                problems.append(
                    f"sequence {event.sequence}: previous_hash {event.previous_hash[:12]}… "
                    f"does not match preceding event hash {expected_previous[:12]}…"
                )
            recomputed = event.compute_hash()
            if recomputed != event.event_hash:
                problems.append(
                    f"sequence {event.sequence}: content does not match its hash "
                    f"(stored {event.event_hash[:12]}…, recomputed {recomputed[:12]}…)"
                )
            expected_previous = event.event_hash

        return (not problems), problems

    async def count(self) -> int:
        result = await self._session.execute(select(func.count()).select_from(AuditEvent))
        return int(result.scalar_one())

    async def recent(
        self,
        *,
        limit: int = 50,
        kind: AuditEventKind | None = None,
        subject_type: str | None = None,
        subject_id: str | None = None,
    ) -> list[AuditEvent]:
        stmt = select(AuditEvent).order_by(AuditEvent.sequence.desc()).limit(limit)
        if kind is not None:
            stmt = stmt.where(AuditEvent.kind == kind)
        if subject_type is not None:
            stmt = stmt.where(AuditEvent.subject_type == subject_type)
        if subject_id is not None:
            stmt = stmt.where(AuditEvent.subject_id == subject_id)
        result = await self._session.execute(stmt)
        return list(result.scalars().all())
