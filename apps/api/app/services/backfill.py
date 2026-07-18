"""Bulk history backfill (§7 steps 5-6).

The scanner scores only instruments that have stored candles, and broker sync
imports the catalogue without price history. This service closes that gap: for a
batch of instruments it creates provider mappings and ingests daily candles, so
the scanner has a real universe to rank.

It exists because doing this one instrument at a time (map, then ingest) is both
slow and needs a per-instrument info call to infer the price unit. Here the unit
is already known from the catalogue, so a single batched `yf.download` covers
many instruments with no per-symbol metadata calls — the difference between
minutes and tens of minutes for a few hundred instruments.

Mappings created here are marked unconfirmed: the yfinance symbol is derived
from the exchange ticker and venue suffix, not ISIN-verified. That is acceptable
for *scanning* (a wrong symbol shows up as absent or incoherent data and is
dropped), but such a mapping must be confirmed before it is ever used to trade —
which the proposal path already enforces via the tradability and confirmation
gates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.data.store import CandleStore
from app.data.yfinance_provider import _MIC_TO_SUFFIX, YFinanceProvider
from app.models.enums import DataSeriesType, Interval, ProviderKind
from app.models.instrument import Exchange, Instrument, MarketDataMapping

log = structlog.get_logger(__name__)

#: Only these venues are backfilled by this service for now. US venues map to a
#: plain yfinance ticker; the LSE and major European venues map via a documented
#: suffix. Others are skipped rather than guessed at.
_SUPPORTED_MICS = set(_MIC_TO_SUFFIX) | {"XNAS", "XNYS", "ARCX", "BATS"}


@dataclass
class BackfillResult:
    considered: int = 0
    mapped: int = 0
    ingested: int = 0
    candles_written: int = 0
    skipped_unsupported: int = 0
    no_data: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        return (
            f"{self.considered} considered, {self.mapped} mapped, {self.ingested} ingested "
            f"({self.candles_written} candles), {self.no_data} returned no data, "
            f"{self.skipped_unsupported} unsupported venues, {len(self.errors)} errors"
        )


def _yfinance_symbol(instrument: Instrument, exchange: Exchange | None) -> str | None:
    """Derive a yfinance symbol from the exchange ticker and venue.

    US venues use the bare ticker; others append the venue suffix (`.L`, `.DE`,
    …). Returns None for venues this service does not support, so the caller can
    skip rather than fabricate a symbol that will silently fetch the wrong thing.
    """
    if not instrument.exchange_ticker or exchange is None:
        return None
    mic = exchange.mic
    if mic in {"XNAS", "XNYS", "ARCX", "BATS"}:
        return instrument.exchange_ticker
    suffix = _MIC_TO_SUFFIX.get(mic)
    if suffix is None:
        return None
    return f"{instrument.exchange_ticker}{suffix}"


class BackfillService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._store = CandleStore(session)

    async def backfill(
        self,
        instruments: list[Instrument],
        provider: YFinanceProvider,
        *,
        backfill_days: int = 400,
        batch_size: int = 40,
    ) -> BackfillResult:
        """Map and ingest daily candles for a batch of instruments."""
        result = BackfillResult(considered=len(instruments))

        # Resolve exchanges once.
        exchange_ids = {i.exchange_id for i in instruments if i.exchange_id}
        exchanges = {
            e.id: e
            for e in (
                await self._session.execute(select(Exchange).where(Exchange.id.in_(exchange_ids)))
            )
            .scalars()
            .all()
        }

        # Build symbol map and ensure a mapping row exists for each.
        symbol_to_instrument: dict[str, Instrument] = {}
        for instrument in instruments:
            exchange = exchanges.get(instrument.exchange_id) if instrument.exchange_id else None
            symbol = _yfinance_symbol(instrument, exchange)
            if symbol is None:
                result.skipped_unsupported += 1
                continue
            await self._ensure_mapping(instrument, symbol)
            result.mapped += 1
            symbol_to_instrument[symbol] = instrument
        await self._session.flush()

        if not symbol_to_instrument:
            return result

        start = datetime.now(UTC) - timedelta(days=backfill_days)
        end = datetime.now(UTC)
        symbols = list(symbol_to_instrument)
        unit_by_symbol = {s: inst.price_unit for s, inst in symbol_to_instrument.items()}

        # Batched download, converting each frame with the instrument's KNOWN
        # unit — no per-symbol info calls.
        for chunk_start in range(0, len(symbols), batch_size):
            chunk = symbols[chunk_start : chunk_start + batch_size]
            try:
                candles_by_symbol = await provider.get_batch_daily_candles(
                    chunk, start, end, unit_by_symbol=unit_by_symbol
                )
            except Exception as exc:
                log.exception("backfill.chunk_failed", symbols=chunk)
                result.errors.append(f"chunk {chunk_start}: {exc}")
                continue

            for symbol in chunk:
                instrument = symbol_to_instrument[symbol]
                candles = candles_by_symbol.get(symbol, [])
                if not candles:
                    result.no_data += 1
                    continue
                # Re-stamp currency/unit from the catalogue: the batch path does
                # not fetch per-symbol currency, so trust what sync recorded.
                written = await self._store.upsert_candles(
                    instrument.id, candles, series_type=DataSeriesType.RAW
                )
                result.candles_written += written
                result.ingested += 1

        await self._session.flush()
        log.info("backfill.completed", summary=result.summary)
        return result

    async def _ensure_mapping(self, instrument: Instrument, symbol: str) -> None:
        existing = await self._session.execute(
            select(MarketDataMapping).where(
                MarketDataMapping.instrument_id == instrument.id,
                MarketDataMapping.provider == ProviderKind.YFINANCE,
            )
        )
        if existing.scalar_one_or_none() is not None:
            return
        self._session.add(
            MarketDataMapping(
                instrument_id=instrument.id,
                provider=ProviderKind.YFINANCE,
                provider_symbol=symbol,
                price_unit=instrument.price_unit,
                is_signal_source=True,
                # Not ISIN-verified — safe for scanning, must be confirmed before
                # trading (see module docstring).
                confirmed_by_user=False,
                resolution_method="backfill_ticker_and_suffix",
            )
        )

    async def select_backfill_candidates(
        self, *, limit: int, mics: list[str] | None = None
    ) -> list[Instrument]:
        """Instruments that lack candles, on supported venues, for backfill.

        Prioritises instruments with no stored daily candles at all, so a run
        broadens coverage rather than re-fetching what is already held.
        """
        from app.models.market_data import Candle

        have_candles = select(Candle.instrument_id).where(Candle.interval == Interval.D1).distinct()
        target_mics = mics or ["XNAS", "XNYS", "ARCX"]

        stmt = (
            select(Instrument)
            .join(Exchange, Exchange.id == Instrument.exchange_id)
            .where(
                Exchange.mic.in_(target_mics),
                Instrument.exchange_ticker.is_not(None),
                Instrument.suspended_at.is_(None),
                Instrument.id.notin_(have_candles),
            )
            .order_by(Instrument.exchange_ticker.asc())
            .limit(limit)
        )
        return list((await self._session.execute(stmt)).scalars().all())
