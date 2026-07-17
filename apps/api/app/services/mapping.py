"""Instrument → provider symbol mapping (§5).

Acceptance criterion 4: map a broker instrument to one or more market-data
symbols. This is where the signal/execution split becomes concrete — the
mapping marked `is_signal_source` is what a strategy reads, while orders route
to the instrument's broker ticker, and the two need not be the same security
(SPY signal → VUAG execution).

A wrong mapping is the quietest catastrophic failure available to this system:
everything downstream works perfectly, on the wrong instrument's prices. So an
unverified match is recorded as requiring confirmation and is refused for
trading until a human agrees.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.audit.service import AuditService
from app.data.base import MarketDataProvider
from app.models.enums import ActorKind, AuditEventKind, LifecycleState, ProviderKind
from app.models.instrument import Instrument, MarketDataMapping

log = structlog.get_logger(__name__)


@dataclass
class MappingResult:
    instrument_id: uuid.UUID
    provider: ProviderKind
    mapping: MarketDataMapping | None
    created: bool = False
    requires_confirmation: bool = False
    reason: str | None = None

    @property
    def resolved(self) -> bool:
        return self.mapping is not None


class MappingService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._audit = AuditService(session)

    async def resolve(
        self,
        instrument: Instrument,
        provider: MarketDataProvider,
        *,
        is_signal_source: bool = False,
        actor_user_id: uuid.UUID | None = None,
    ) -> MappingResult:
        """Ask a provider what it calls this instrument, and record the answer."""
        existing = await self.find_mapping(instrument.id, provider.kind)
        if existing is not None and existing.confirmed_by_user:
            # A human has already ruled on this. Do not re-derive it.
            return MappingResult(
                instrument_id=instrument.id,
                provider=provider.kind,
                mapping=existing,
                reason="Already confirmed by user",
            )

        proposal = await provider.resolve_instrument(instrument)
        if proposal is None:
            log.info(
                "mapping.unresolved",
                instrument_id=str(instrument.id),
                provider=str(provider.kind),
            )
            return MappingResult(
                instrument_id=instrument.id,
                provider=provider.kind,
                mapping=None,
                reason=f"{provider.kind} could not identify this instrument",
            )

        if existing is not None:
            existing.provider_symbol = proposal.provider_symbol
            existing.price_unit = proposal.price_unit
            existing.resolution_method = proposal.resolution_method
            existing.last_verified_at = datetime.now(UTC)
            existing.last_error = None
            if is_signal_source:
                await self._clear_other_signal_sources(instrument.id, keep=existing.id)
                existing.is_signal_source = True
            mapping = existing
            created = False
        else:
            mapping = MarketDataMapping(
                instrument_id=instrument.id,
                provider=provider.kind,
                provider_symbol=proposal.provider_symbol,
                price_unit=proposal.price_unit,
                resolution_method=proposal.resolution_method,
                confirmed_by_user=False,
                is_signal_source=is_signal_source,
                last_verified_at=datetime.now(UTC),
            )
            self._session.add(mapping)
            if is_signal_source:
                await self._session.flush()
                await self._clear_other_signal_sources(instrument.id, keep=mapping.id)
            created = True

        await self._session.flush()

        if proposal.requires_confirmation:
            # Identity below the ISIN tier. Hold the instrument at
            # MAPPING_REQUIRED so it cannot advance toward tradability (§7).
            instrument.lifecycle_state = LifecycleState.MAPPING_REQUIRED
            instrument.lifecycle_note = proposal.note or (
                "Provider symbol was matched below the ISIN tier and needs confirmation."
            )

        await self._audit.record(
            kind=AuditEventKind.INSTRUMENT_MAPPED,
            summary=(
                f"Mapped {instrument.name} to {provider.kind}:{proposal.provider_symbol} "
                f"via {proposal.resolution_method}"
            ),
            actor_kind=ActorKind.USER if actor_user_id else ActorKind.SYSTEM,
            actor_user_id=actor_user_id,
            subject_type="instrument",
            subject_id=str(instrument.id),
            payload={
                "provider": str(provider.kind),
                "provider_symbol": proposal.provider_symbol,
                "resolution_method": proposal.resolution_method,
                "confidence": proposal.confidence,
                "requires_confirmation": proposal.requires_confirmation,
                "is_signal_source": is_signal_source,
            },
        )

        return MappingResult(
            instrument_id=instrument.id,
            provider=provider.kind,
            mapping=mapping,
            created=created,
            requires_confirmation=proposal.requires_confirmation,
            reason=proposal.note,
        )

    async def confirm(
        self,
        mapping_id: uuid.UUID,
        *,
        actor_user_id: uuid.UUID,
        provider_symbol: str | None = None,
    ) -> MarketDataMapping:
        """Record a human's agreement that a mapping is correct.

        Tier 5 of the §5 hierarchy. Also permits overriding the symbol, which
        is the escape hatch when automatic resolution simply cannot get there.
        """
        mapping = await self._session.get(MarketDataMapping, mapping_id)
        if mapping is None:
            raise ValueError(f"No such mapping: {mapping_id}")

        if provider_symbol and provider_symbol != mapping.provider_symbol:
            previous = mapping.provider_symbol
            mapping.provider_symbol = provider_symbol
            mapping.resolution_method = "manual_user_override"
            log.info(
                "mapping.manually_overridden",
                mapping_id=str(mapping_id),
                previous=previous,
                new=provider_symbol,
            )

        mapping.confirmed_by_user = True
        mapping.last_verified_at = datetime.now(UTC)

        instrument = await self._session.get(Instrument, mapping.instrument_id)
        if instrument is not None and instrument.lifecycle_state is LifecycleState.MAPPING_REQUIRED:
            # Mapping resolved. Next gate is history, not tradability — this
            # does not make the instrument bot-eligible.
            instrument.lifecycle_state = LifecycleState.DATA_BACKFILL
            instrument.lifecycle_note = "Mapping confirmed; awaiting history backfill."

        await self._session.flush()
        await self._audit.record(
            kind=AuditEventKind.INSTRUMENT_MAPPED,
            summary=f"Mapping confirmed: {mapping.provider}:{mapping.provider_symbol}",
            actor_kind=ActorKind.USER,
            actor_user_id=actor_user_id,
            subject_type="instrument",
            subject_id=str(mapping.instrument_id),
            payload={
                "mapping_id": str(mapping_id),
                "provider": str(mapping.provider),
                "provider_symbol": mapping.provider_symbol,
                "manual_override": mapping.resolution_method == "manual_user_override",
            },
        )
        return mapping

    async def set_signal_source(
        self, instrument_id: uuid.UUID, mapping_id: uuid.UUID, *, actor_user_id: uuid.UUID
    ) -> MarketDataMapping:
        """Choose which series drives signals for this instrument.

        Enables the SPY-signal/VUAG-execution arrangement. Exactly one mapping
        per instrument may be the signal source — two would make "the" signal
        ambiguous and the choice implicit.
        """
        mapping = await self._session.get(MarketDataMapping, mapping_id)
        if mapping is None or mapping.instrument_id != instrument_id:
            raise ValueError("Mapping does not belong to this instrument")

        await self._clear_other_signal_sources(instrument_id, keep=mapping_id)
        mapping.is_signal_source = True
        await self._session.flush()

        await self._audit.record(
            kind=AuditEventKind.INSTRUMENT_MAPPED,
            summary=(f"Signal source set to {mapping.provider}:{mapping.provider_symbol}"),
            actor_kind=ActorKind.USER,
            actor_user_id=actor_user_id,
            subject_type="instrument",
            subject_id=str(instrument_id),
            payload={
                "mapping_id": str(mapping_id),
                "provider_symbol": mapping.provider_symbol,
            },
        )
        return mapping

    async def _clear_other_signal_sources(
        self, instrument_id: uuid.UUID, *, keep: uuid.UUID
    ) -> None:
        result = await self._session.execute(
            select(MarketDataMapping).where(
                MarketDataMapping.instrument_id == instrument_id,
                MarketDataMapping.is_signal_source.is_(True),
                MarketDataMapping.id != keep,
            )
        )
        for other in result.scalars().all():
            other.is_signal_source = False

    async def find_mapping(
        self, instrument_id: uuid.UUID, provider: ProviderKind
    ) -> MarketDataMapping | None:
        """This instrument's mapping for one provider, if any."""
        result = await self._session.execute(
            select(MarketDataMapping).where(
                MarketDataMapping.instrument_id == instrument_id,
                MarketDataMapping.provider == provider,
            )
        )
        return result.scalars().first()

    async def signal_mapping_for(self, instrument_id: uuid.UUID) -> MarketDataMapping | None:
        """The mapping a strategy should read for this instrument."""
        result = await self._session.execute(
            select(MarketDataMapping).where(
                MarketDataMapping.instrument_id == instrument_id,
                MarketDataMapping.is_signal_source.is_(True),
                MarketDataMapping.is_active.is_(True),
            )
        )
        return result.scalars().first()

    async def list_for_instrument(self, instrument_id: uuid.UUID) -> list[MarketDataMapping]:
        result = await self._session.execute(
            select(MarketDataMapping)
            .where(MarketDataMapping.instrument_id == instrument_id)
            .order_by(MarketDataMapping.priority.asc())
        )
        return list(result.scalars().all())
