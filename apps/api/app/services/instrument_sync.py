"""Trading 212 instrument synchronisation (§3, §5).

Turns the broker catalogue into `BrokerInstrument` rows, and resolves each to a
canonical `Instrument` using the §5 identity hierarchy.

Two rules this service will not bend:

  * **Sync never confers tradability.** Appearing in the catalogue sets
    `lifecycle_state = DISCOVERED` and nothing else. Bot Universe membership is
    a separate, deliberate act (§7). A broker adding 500 instruments overnight
    must not enlarge what the bot can trade.

  * **Identity is never guessed.** Where ISIN is unavailable we fall down the
    hierarchy and mark the result as needing confirmation, rather than assuming
    that equal tickers mean equal securities. They frequently do not.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.audit.service import AuditService
from app.broker.base import Broker
from app.broker.types import BrokerInstrument as BrokerInstrumentDTO
from app.models.enums import (
    ActorKind,
    AuditEventKind,
    BrokerKind,
    InstrumentKind,
    LifecycleState,
    PriceUnit,
)
from app.models.instrument import BrokerInstrument, Exchange, Instrument

log = structlog.get_logger(__name__)

#: Trading 212 encodes venue in a ticker suffix rather than reporting a MIC.
#: Mapping is explicit and conservative: an unrecognised suffix yields no
#: exchange rather than a plausible-looking wrong one.
_T212_SUFFIX_TO_MIC: dict[str, str] = {
    "l_EQ": "XLON",
    "_US_EQ": "XNAS",
    "d_EQ": "XETR",
    "p_EQ": "XPAR",
    "a_EQ": "XAMS",
    "m_EQ": "XMIL",
    "e_EQ": "XMAD",
    "s_EQ": "XSWX",
    "_SW_EQ": "XSWX",
}


@dataclass
class SyncResult:
    broker: BrokerKind
    synced_at: datetime
    total_from_broker: int = 0
    broker_instruments_created: int = 0
    broker_instruments_updated: int = 0
    instruments_created: int = 0
    instruments_matched_by_isin: int = 0
    instruments_needing_confirmation: int = 0
    delisted: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        return (
            f"{self.total_from_broker} instruments from {self.broker}: "
            f"{self.broker_instruments_created} new, "
            f"{self.broker_instruments_updated} updated, "
            f"{self.instruments_created} canonical instruments created, "
            f"{self.delisted} no longer available"
        )


def _infer_mic(broker_ticker: str) -> str | None:
    """Best-effort venue from a Trading 212 ticker suffix.

    Longest suffix first: "_US_EQ" must win over "_EQ"-style shorter matches.
    """
    for suffix in sorted(_T212_SUFFIX_TO_MIC, key=len, reverse=True):
        if broker_ticker.endswith(suffix):
            return _T212_SUFFIX_TO_MIC[suffix]
    return None


def _exchange_ticker_from(broker_ticker: str) -> str:
    """Strip Trading 212's suffix to recover the exchange ticker.

    "VUAGl_EQ" → "VUAG", "AAPL_US_EQ" → "AAPL".
    """
    for suffix in sorted(_T212_SUFFIX_TO_MIC, key=len, reverse=True):
        if broker_ticker.endswith(suffix):
            return broker_ticker[: -len(suffix)]
    return broker_ticker.split("_")[0]


def _price_unit_for(currency: str | None) -> PriceUnit:
    """Trading 212 reports GBX explicitly for pence-quoted lines."""
    if not currency:
        return PriceUnit.USD
    try:
        return PriceUnit(currency.upper())
    except ValueError:
        return PriceUnit.USD


class InstrumentSyncService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._audit = AuditService(session)

    async def sync(self, broker: Broker, *, actor_label: str = "instrument_sync_job") -> SyncResult:
        """Pull the catalogue and reconcile it into local tables.

        Idempotent: re-running produces no new rows and no duplicate identities,
        which matters because this runs on a schedule and jobs get retried (§16).
        """
        result = SyncResult(broker=broker.kind, synced_at=datetime.now(UTC))

        catalogue = await broker.sync_instruments()
        result.total_from_broker = len(catalogue)

        seen_tickers: set[str] = set()
        exchange_cache: dict[str, Exchange | None] = {}

        for dto in catalogue:
            if not dto.broker_ticker:
                result.errors.append("Broker returned an instrument with no ticker; skipped")
                continue
            seen_tickers.add(dto.broker_ticker)
            try:
                await self._sync_one(dto, broker.kind, result, exchange_cache)
            except Exception as exc:
                log.exception(
                    "sync.instrument_failed", broker_ticker=dto.broker_ticker, error=str(exc)
                )
                result.errors.append(f"{dto.broker_ticker}: {exc}")

        result.delisted = await self._mark_unavailable(broker.kind, seen_tickers)
        await self._session.flush()

        await self._audit.record(
            kind=AuditEventKind.INSTRUMENT_SYNCED,
            summary=result.summary,
            actor_kind=ActorKind.SCHEDULER,
            actor_label=actor_label,
            subject_type="broker",
            subject_id=str(broker.kind),
            payload={
                "total_from_broker": result.total_from_broker,
                "created": result.broker_instruments_created,
                "updated": result.broker_instruments_updated,
                "delisted": result.delisted,
                "needing_confirmation": result.instruments_needing_confirmation,
                "error_count": len(result.errors),
            },
        )

        log.info("sync.completed", broker=str(broker.kind), summary=result.summary)
        return result

    async def _sync_one(
        self,
        dto: BrokerInstrumentDTO,
        broker_kind: BrokerKind,
        result: SyncResult,
        exchange_cache: dict[str, Exchange | None],
    ) -> None:
        mic = dto.exchange_mic or _infer_mic(dto.broker_ticker)
        exchange = await self._get_exchange(mic, exchange_cache) if mic else None

        instrument = await self._resolve_instrument(dto, exchange, result)

        existing = await self._session.execute(
            select(BrokerInstrument).where(
                BrokerInstrument.broker == broker_kind,
                BrokerInstrument.broker_ticker == dto.broker_ticker,
            )
        )
        broker_instrument = existing.scalar_one_or_none()

        if broker_instrument is None:
            self._session.add(
                BrokerInstrument(
                    instrument_id=instrument.id,
                    broker=broker_kind,
                    broker_ticker=dto.broker_ticker,
                    broker_name=dto.name,
                    broker_isin=dto.isin,
                    currency=dto.currency,
                    is_currently_available=dto.is_currently_available,
                    min_quantity=str(dto.min_quantity) if dto.min_quantity else None,
                    quantity_step=str(dto.quantity_step) if dto.quantity_step else None,
                    supports_fractional=dto.supports_fractional,
                    last_synced_at=datetime.now(UTC),
                    raw_payload=dto.raw or None,
                )
            )
            result.broker_instruments_created += 1
        else:
            broker_instrument.broker_name = dto.name
            broker_instrument.broker_isin = dto.isin
            broker_instrument.currency = dto.currency
            broker_instrument.is_currently_available = dto.is_currently_available
            broker_instrument.supports_fractional = dto.supports_fractional
            broker_instrument.last_synced_at = datetime.now(UTC)
            broker_instrument.raw_payload = dto.raw or None
            result.broker_instruments_updated += 1

    async def _resolve_instrument(
        self,
        dto: BrokerInstrumentDTO,
        exchange: Exchange | None,
        result: SyncResult,
    ) -> Instrument:
        """Find or create the canonical instrument, per the §5 hierarchy."""

        # Tier 1: ISIN (+ exchange, since one ISIN can be listed on several).
        if dto.isin:
            stmt = select(Instrument).where(Instrument.isin == dto.isin)
            if exchange is not None:
                stmt = stmt.where(Instrument.exchange_id == exchange.id)
            found = (await self._session.execute(stmt)).scalars().first()
            if found is not None:
                result.instruments_matched_by_isin += 1
                return found

        exchange_ticker = _exchange_ticker_from(dto.broker_ticker)

        # Tier 2: exchange MIC + exchange ticker.
        if exchange is not None:
            found = (
                (
                    await self._session.execute(
                        select(Instrument).where(
                            Instrument.exchange_id == exchange.id,
                            Instrument.exchange_ticker == exchange_ticker,
                        )
                    )
                )
                .scalars()
                .first()
            )
            if found is not None:
                # Backfill an ISIN we did not previously have.
                if dto.isin and not found.isin:
                    found.isin = dto.isin
                return found

        price_unit = _price_unit_for(dto.currency)
        # A GBX-quoted instrument settles in GBP. Storing "GBX" as the currency
        # would make every downstream cash calculation wrong by 100x.
        currency = "GBP" if price_unit is PriceUnit.GBX else (dto.currency or "USD")

        needs_confirmation = dto.isin is None
        if needs_confirmation:
            result.instruments_needing_confirmation += 1

        instrument = Instrument(
            id=uuid.uuid4(),
            isin=dto.isin,
            exchange_id=exchange.id if exchange else None,
            exchange_ticker=exchange_ticker,
            name=dto.name or dto.broker_ticker,
            kind=_parse_kind(dto.kind),
            currency=currency,
            price_unit=price_unit,
            identity_confirmed_by_user=False,
            identity_resolution_note=(
                "Resolved by broker ISIN"
                if dto.isin
                else "No ISIN from broker; identity inferred from ticker and inferred "
                "exchange. Confirm before trading."
            ),
            # Discovery is not eligibility. Bot Universe membership is explicit (§7).
            lifecycle_state=LifecycleState.DISCOVERED,
            is_bot_universe=False,
        )
        self._session.add(instrument)
        await self._session.flush()
        result.instruments_created += 1
        return instrument

    async def _get_exchange(self, mic: str, cache: dict[str, Exchange | None]) -> Exchange | None:
        if mic in cache:
            return cache[mic]
        found = (
            (await self._session.execute(select(Exchange).where(Exchange.mic == mic)))
            .scalars()
            .first()
        )
        cache[mic] = found
        if found is None:
            log.warning(
                "sync.unknown_exchange",
                mic=mic,
                detail="Instrument references an exchange not present in the "
                "exchanges table; run the seed script",
            )
        return found

    async def _mark_unavailable(self, broker_kind: BrokerKind, seen: set[str]) -> int:
        """Flag instruments the broker no longer lists.

        Not deleted. Historical orders and positions reference them, and an
        instrument can return to the catalogue. Marking is reversible; deleting
        loses provenance.
        """
        result = await self._session.execute(
            select(BrokerInstrument).where(
                BrokerInstrument.broker == broker_kind,
                BrokerInstrument.is_currently_available.is_(True),
            )
        )
        count = 0
        for broker_instrument in result.scalars().all():
            if broker_instrument.broker_ticker not in seen:
                broker_instrument.is_currently_available = False
                count += 1
                log.info(
                    "sync.instrument_no_longer_available",
                    broker_ticker=broker_instrument.broker_ticker,
                )
        return count


def _parse_kind(raw: str) -> InstrumentKind:
    try:
        return InstrumentKind(raw)
    except ValueError:
        return InstrumentKind.UNKNOWN
