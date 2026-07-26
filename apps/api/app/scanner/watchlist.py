"""Scanner watchlist membership (§6).

The rotation already ranks watchlist members first (`app.scanner.rotation`,
tier 1), but until now nothing could put an instrument in one: the tables
existed and stayed empty. This module is the write side.

Scoping note. A watchlist belongs to a user, because the schema says so and a
multi-operator deployment would need that. The *rotation* reads membership
globally, without a user — it runs on a schedule, on behalf of nobody. For the
single-operator deployment this targets the two coincide exactly; if a second
operator is ever added, the rotation's query is the place that has to choose
whose watchlist governs the scan, and it should be made to say so explicitly
rather than quietly unioning them.
"""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.instrument import Instrument, Watchlist, WatchlistInstrument

#: The watchlist the scanner UI reads and writes. One per user, created on
#: first use — an operator should not have to name a list before pinning a
#: stock they are looking at.
DEFAULT_WATCHLIST_NAME = "Scanner watchlist"

_DEFAULT_DESCRIPTION = (
    "Instruments pinned from the scanner. Every member is selected for the next "
    "scan ahead of the rotating sweep."
)


async def find_default(session: AsyncSession, user_id: uuid.UUID) -> Watchlist | None:
    """The user's scanner watchlist, or None. Never writes — safe on a read path."""
    return (
        await session.execute(
            select(Watchlist).where(
                Watchlist.user_id == user_id,
                Watchlist.name == DEFAULT_WATCHLIST_NAME,
            )
        )
    ).scalar_one_or_none()


async def get_or_create_default(session: AsyncSession, user_id: uuid.UUID) -> Watchlist:
    """The user's scanner watchlist, created on first use.

    Only for write paths. A GET that creates rows it never commits is a wasted
    round trip at best, and a surprise in the audit trail at worst.
    """
    existing = await find_default(session, user_id)
    if existing is not None:
        return existing

    watchlist = Watchlist(
        user_id=user_id,
        name=DEFAULT_WATCHLIST_NAME,
        description=_DEFAULT_DESCRIPTION,
    )
    session.add(watchlist)
    await session.flush()
    return watchlist


async def add(
    session: AsyncSession,
    user_id: uuid.UUID,
    instrument_id: uuid.UUID,
    *,
    note: str | None = None,
) -> tuple[WatchlistInstrument, bool]:
    """Pin an instrument. Returns the entry and whether it was newly created.

    Idempotent: watchlisting something twice is not an error, it is the same
    request arriving twice (a double-click, a retried POST). The unique
    constraint on (watchlist, instrument) would reject the second write, so the
    existing row is returned instead of raising.
    """
    watchlist = await get_or_create_default(session, user_id)
    existing = (
        await session.execute(
            select(WatchlistInstrument).where(
                WatchlistInstrument.watchlist_id == watchlist.id,
                WatchlistInstrument.instrument_id == instrument_id,
            )
        )
    ).scalar_one_or_none()
    if existing is not None:
        if note is not None:
            existing.note = note
        return existing, False

    entry = WatchlistInstrument(watchlist_id=watchlist.id, instrument_id=instrument_id, note=note)
    session.add(entry)
    await session.flush()
    return entry, True


async def remove(session: AsyncSession, user_id: uuid.UUID, instrument_id: uuid.UUID) -> bool:
    """Unpin an instrument. Returns whether a row was actually removed."""
    watchlist = await get_or_create_default(session, user_id)
    entry = (
        await session.execute(
            select(WatchlistInstrument).where(
                WatchlistInstrument.watchlist_id == watchlist.id,
                WatchlistInstrument.instrument_id == instrument_id,
            )
        )
    ).scalar_one_or_none()
    if entry is None:
        return False
    await session.delete(entry)
    await session.flush()
    return True


async def list_entries(
    session: AsyncSession, user_id: uuid.UUID
) -> list[tuple[WatchlistInstrument, Instrument | None]]:
    """The user's watchlist, oldest pin first, with each instrument loaded.

    The eager load reaches through to `Instrument.exchange` deliberately. Loading
    only the instrument leaves `.exchange` lazy, and a lazy relationship touched
    on an asyncio session raises `MissingGreenlet` rather than emitting a query —
    so a caller that renders the venue gets a 500, not a slow response.
    """
    watchlist = await find_default(session, user_id)
    if watchlist is None:
        return []
    entries = list(
        (
            await session.execute(
                select(WatchlistInstrument)
                .where(WatchlistInstrument.watchlist_id == watchlist.id)
                .order_by(WatchlistInstrument.created_at.asc())
                .options(
                    selectinload(WatchlistInstrument.instrument).selectinload(Instrument.exchange)
                )
            )
        )
        .scalars()
        .all()
    )
    return [(e, e.instrument) for e in entries]


async def watchlisted_instrument_ids(session: AsyncSession) -> list[uuid.UUID]:
    """Every watchlisted instrument, across watchlists — what the rotation pins.

    Ordered by when it was pinned so that, in the pathological case where the
    watchlist alone exceeds a scan's instrument cap, the truncation is at least
    stable and comprehensible rather than arbitrary.
    """
    result = await session.execute(
        select(WatchlistInstrument.instrument_id).order_by(WatchlistInstrument.created_at.asc())
    )
    seen: set[uuid.UUID] = set()
    out: list[uuid.UUID] = []
    for (iid,) in result.all():
        if iid not in seen:
            seen.add(iid)
            out.append(iid)
    return out
