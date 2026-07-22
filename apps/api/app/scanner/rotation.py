"""Rotating scan selection (§6).

Free data budgets mean the whole catalogue cannot be scored every day, so the
scanner rotates. The priority order is fixed by §6:

  1. Watchlist members
  2. Previously high-ranking candidates
  3. Instruments with newly refreshed data
  4. The rest of the catalogue, oldest-scanned first

`last_scanned_at ASC NULLS FIRST` does most of the work: never-scanned and
longest-unscanned instruments surface first, so over successive days the whole
universe is covered without ever exceeding the per-scan cap. Higher-priority
tiers are unioned in ahead of that sweep.

Selection is filtered to instruments that actually have enough stored history to
score — scheduling one that will only be skipped wastes a slot.
"""

from __future__ import annotations

import uuid

from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.enums import BrokerKind, Interval, LifecycleState
from app.models.instrument import BrokerInstrument, Instrument, WatchlistInstrument
from app.models.market_data import Candle
from app.models.scanner import Classification, ScannerConfiguration, ScannerResult
from app.scanner.engine import MIN_BARS_TO_SCORE

#: Upper bound on the rotation candidate pool fetched per scan, before the
#: per-scan instrument cap is applied. A safety bound on the query, not the
#: scan size.
_ROTATION_POOL_LIMIT = 2000


async def select_instruments(
    session: AsyncSession,
    *,
    configuration: ScannerConfiguration | None = None,
    limit: int | None = None,
) -> tuple[list[Instrument], str]:
    """Choose the instruments for the next scan.

    Returns the instruments and a short reason describing the selection, which
    is stored on the run so the UI can answer "why was this scanned".
    """
    max_instruments = _max_instruments(configuration, limit)

    scored_enough = (
        select(Candle.instrument_id)
        .where(Candle.interval == Interval.D1, Candle.is_closed.is_(True))
        .group_by(Candle.instrument_id)
        .having(func.count(Candle.id) >= MIN_BARS_TO_SCORE)
    )
    eligible_ids = {row[0] for row in (await session.execute(scored_enough)).all()}
    if not eligible_ids:
        return [], "no instruments have enough stored history yet"

    ordered_ids: list[uuid.UUID] = []
    seen: set[uuid.UUID] = set()

    def _extend(ids: list[uuid.UUID]) -> None:
        for iid in ids:
            if iid in eligible_ids and iid not in seen:
                seen.add(iid)
                ordered_ids.append(iid)

    # Tier 1: watchlist members.
    _extend(await _watchlist_instrument_ids(session))
    # Tier 2: previously high-ranking candidates.
    _extend(await _previous_candidate_ids(session))
    # Tiers 3+4: the rotating sweep, oldest-scanned first. Restricted to the
    # instruments that can actually be scored, so the sweep's cap and ordering
    # operate within the scannable set rather than being saturated by the far
    # larger pool of never-scanned instruments that have no stored history.
    _extend(await _rotation_ids(session, configuration, eligible_ids))

    chosen_ids = ordered_ids[:max_instruments]
    if not chosen_ids:
        return [], "no eligible instruments"

    instruments = await _load(session, chosen_ids)
    reason = (
        f"rotation: {len(chosen_ids)} instruments (watchlist + prior candidates + oldest-scanned)"
    )
    return instruments, reason


def _max_instruments(config: ScannerConfiguration | None, limit: int | None) -> int:
    if limit is not None:
        return limit
    if config is not None:
        return config.max_instruments_per_scan
    return 200


async def _watchlist_instrument_ids(session: AsyncSession) -> list[uuid.UUID]:
    result = await session.execute(select(WatchlistInstrument.instrument_id))
    return [row[0] for row in result.all()]


async def _previous_candidate_ids(session: AsyncSession) -> list[uuid.UUID]:
    """Instruments that were screening/watchlist candidates in a recent result.

    Ordered by score so the strongest prior candidates are re-verified first.
    """
    result = await session.execute(
        select(ScannerResult.instrument_id)
        .where(
            ScannerResult.classification.in_(
                [Classification.SCREENING_CANDIDATE, Classification.WATCHLIST_CANDIDATE]
            )
        )
        .order_by(ScannerResult.core_score.desc())
        .limit(500)
    )
    # Preserve order while de-duplicating (an instrument can appear in several runs).
    seen: set[uuid.UUID] = set()
    out: list[uuid.UUID] = []
    for (iid,) in result.all():
        if iid not in seen:
            seen.add(iid)
            out.append(iid)
    return out


async def _rotation_ids(
    session: AsyncSession,
    config: ScannerConfiguration | None,
    eligible_ids: set[uuid.UUID],
) -> list[uuid.UUID]:
    conditions = [
        Instrument.is_scanner_eligible.is_(True),
        Instrument.suspended_at.is_(None),
        Instrument.lifecycle_state != LifecycleState.ARCHIVED,
        # Only sweep instruments that can be scored. Without this, a catalogue
        # dominated by never-scanned, history-less instruments fills the whole
        # `last_scanned_at NULLS FIRST` window below, and the scannable ones —
        # which have been scanned and so sort last — never make the cut.
        Instrument.id.in_(eligible_ids),
    ]

    stmt = select(Instrument.id)

    if config is not None and config.trading212_only:
        stmt = stmt.join(BrokerInstrument, BrokerInstrument.instrument_id == Instrument.id).where(
            BrokerInstrument.broker.in_([BrokerKind.TRADING212_DEMO, BrokerKind.TRADING212_LIVE]),
            BrokerInstrument.is_currently_available.is_(True),
        )

    # NULLS FIRST puts never-scanned instruments at the front of the sweep.
    stmt = (
        stmt.where(and_(*conditions))
        .order_by(Instrument.last_scanned_at.asc().nulls_first())
        .limit(_ROTATION_POOL_LIMIT)
    )

    result = await session.execute(stmt)
    return [row[0] for row in result.all()]


async def _load(session: AsyncSession, ids: list[uuid.UUID]) -> list[Instrument]:
    """Load instruments and preserve the priority order of `ids`."""
    result = await session.execute(select(Instrument).where(Instrument.id.in_(ids)))
    by_id = {i.id: i for i in result.scalars().all()}
    return [by_id[iid] for iid in ids if iid in by_id]
