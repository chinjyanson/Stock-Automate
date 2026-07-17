"""Immutable audit log (§17).

Append-only, and tamper-evident: each row commits to the previous row's hash,
so altering or excising history breaks the chain at a detectable point. The
table additionally carries a database trigger rejecting UPDATE and DELETE (see
the initial migration) — application-level discipline is not a guarantee when
the whole point is to survive a compromised application.

Secrets never enter this table. Credential changes are recorded as the *fact*
of a change, never the value.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import BigInteger, DateTime, ForeignKey, Index, Sequence, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, StrEnumType, UUIDPrimaryKeyMixin
from app.models.enums import ActorKind, AuditEventKind

#: Chain seed for the first event in an empty log.
GENESIS_HASH = "0" * 64

#: Explicit sequence: `autoincrement=True` only applies to integer primary keys,
#: and this table's PK is a UUID. Without this the column would have no default
#: and every insert would fail.
AUDIT_SEQUENCE = Sequence("audit_events_sequence_seq")


class AuditEvent(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "audit_events"
    __table_args__ = (
        Index("ix_audit_events_kind_occurred", "kind", "occurred_at"),
        Index("ix_audit_events_subject", "subject_type", "subject_id"),
    )

    #: Monotonic chain position. The hash chain needs a *total* order, and
    #: `occurred_at` cannot provide one: two events can share a timestamp, and
    #: clocks move backwards. A database-assigned identity column is the only
    #: ordering both concurrent writers agree on.
    sequence: Mapped[int] = mapped_column(
        BigInteger,
        AUDIT_SEQUENCE,
        server_default=AUDIT_SEQUENCE.next_value(),
        unique=True,
        nullable=False,
        index=True,
    )

    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    kind: Mapped[AuditEventKind] = mapped_column(StrEnumType(AuditEventKind, 40), nullable=False)

    actor_kind: Mapped[ActorKind] = mapped_column(StrEnumType(ActorKind, 16), nullable=False)
    actor_user_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), index=True
    )
    #: Human-readable actor label for system/scheduler actors, e.g. the job name.
    actor_label: Mapped[str | None] = mapped_column(String(120))

    #: What the event is about, e.g. ("instrument", <uuid>) or ("order", <uuid>).
    subject_type: Mapped[str | None] = mapped_column(String(40))
    subject_id: Mapped[str | None] = mapped_column(String(64))

    summary: Mapped[str] = mapped_column(Text, nullable=False)
    #: Structured detail. Must never contain credentials or auth headers.
    payload: Mapped[dict[str, Any] | None] = mapped_column()

    #: Correlation identifiers (§18).
    request_id: Mapped[str | None] = mapped_column(String(64), index=True)
    trade_intent_id: Mapped[str | None] = mapped_column(String(64), index=True)
    strategy_decision_id: Mapped[str | None] = mapped_column(String(64), index=True)

    #: Tamper-evidence chain.
    previous_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    event_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)

    def compute_hash(self) -> str:
        """Hash over this event's content plus the previous event's hash.

        `sort_keys` and a separator-normalised dump keep the digest stable
        across Python versions and dict ordering.
        """
        material = {
            "id": str(self.id),
            "sequence": self.sequence,
            "occurred_at": self.occurred_at.isoformat(),
            "kind": str(self.kind),
            "actor_kind": str(self.actor_kind),
            "actor_user_id": str(self.actor_user_id) if self.actor_user_id else None,
            "actor_label": self.actor_label,
            "subject_type": self.subject_type,
            "subject_id": self.subject_id,
            "summary": self.summary,
            "payload": self.payload,
            "previous_hash": self.previous_hash,
        }
        encoded = json.dumps(material, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def __repr__(self) -> str:
        return f"<AuditEvent {self.kind} {self.occurred_at.isoformat()}>"
