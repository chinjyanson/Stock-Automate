"""Persist and serve the daily index-options reading.

Split the same way the insider and regime services are: one half fetches and
records, the other half answers questions from what has been recorded. The
strategy reads only the recorded side, so evaluating a signal never touches the
network — the same store-only discipline the scanner keeps.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.index_options import IndexOptionsSnapshot
from app.signals import spx_options
from app.signals.spx_options import IndexOptionsReading

log = structlog.get_logger(__name__)

#: A reading older than this is not a description of today's market. The
#: strategy declines to act rather than trading on a stale surface — the chain
#: is a snapshot, and a week-old snapshot of dealer positioning is worse than
#: none because it still looks like a number.
MAX_READING_AGE_DAYS = 4


def _dec(value: float | None) -> Decimal | None:
    return None if value is None else Decimal(str(round(value, 6)))


class IndexOptionsService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def measure_and_record(self) -> IndexOptionsSnapshot | None:
        """Fetch a live chain and upsert today's row. Network — jobs only."""
        reading = await spx_options.fetch_reading()
        if reading is None:
            log.warning("index_options.no_reading")
            return None
        return await self.record(reading)

    async def record(self, reading: IndexOptionsReading) -> IndexOptionsSnapshot:
        """Upsert by (date, symbol), so re-running the job converges.

        Keyed on the symbol as well as the date because the reading falls back
        from `^SPX` to `SPY`. Overwriting one proxy's row with the other's would
        put two incomparable scales in one series under one timestamp, and no
        later query could tell them apart.
        """
        existing = (
            (
                await self._session.execute(
                    select(IndexOptionsSnapshot).where(
                        IndexOptionsSnapshot.as_of == reading.as_of,
                        IndexOptionsSnapshot.symbol == reading.symbol,
                    )
                )
            )
            .scalars()
            .first()
        )
        snapshot = existing or IndexOptionsSnapshot(as_of=reading.as_of, symbol=reading.symbol)
        snapshot.spot = _dec(reading.spot)
        snapshot.expiry_days = reading.expiry_days
        snapshot.gamma_exposure = _dec(reading.gamma_exposure)
        snapshot.gamma_tilt = _dec(reading.gamma_tilt)
        snapshot.charm_exposure = _dec(reading.charm_exposure)
        snapshot.charm_tilt = _dec(reading.charm_tilt)
        snapshot.skew_25delta = _dec(reading.skew_25delta)
        snapshot.atm_iv = _dec(reading.atm_iv)
        snapshot.contracts_used = reading.contracts_used
        if existing is None:
            self._session.add(snapshot)
        await self._session.flush()
        return snapshot

    async def history(self, *, symbol: str, limit: int = 500) -> list[IndexOptionsSnapshot]:
        """Readings for one proxy, oldest first.

        Filtered to a single symbol on purpose, and there is no unfiltered
        variant. A series that mixes `^SPX` and `SPY` rows carries a ten-to-one
        step wherever the fallback fired, which is indistinguishable from a
        regime shift once it reaches a model — so the option to make that
        mistake is simply not offered.
        """
        rows = (
            (
                await self._session.execute(
                    select(IndexOptionsSnapshot)
                    .where(IndexOptionsSnapshot.symbol == symbol)
                    .order_by(IndexOptionsSnapshot.as_of.desc())
                    .limit(limit)
                )
            )
            .scalars()
            .all()
        )
        return list(reversed(rows))

    async def latest(self) -> IndexOptionsSnapshot | None:
        """The most recent reading, or None if it is too old to describe today."""
        snapshot = (
            (
                await self._session.execute(
                    select(IndexOptionsSnapshot)
                    .order_by(IndexOptionsSnapshot.as_of.desc())
                    .limit(1)
                )
            )
            .scalars()
            .first()
        )
        if snapshot is None:
            return None
        age = (datetime.now(UTC).date() - snapshot.as_of).days
        if age > MAX_READING_AGE_DAYS:
            log.info("index_options.stale", as_of=str(snapshot.as_of), age_days=age)
            return None
        return snapshot
