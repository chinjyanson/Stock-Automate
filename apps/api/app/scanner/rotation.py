"""Rotating scan selection (§6).

Free data budgets mean the whole catalogue cannot be scored every day, so the
scanner rotates. The priority order is fixed by §6:

  1. Watchlist members
  2. The current top `TOP_RANKED_ALWAYS_RESCANNED` by latest score (exploit)
  3. The rest of the catalogue, oldest-scanned first (explore)

Deliberately an explore/exploit split rather than one sweep. The scan covers a
slice of a 15k catalogue per run, and the strategy draws its universe from the
latest run only — so a highly-ranked stock that is not re-scanned tonight is not
a candidate tonight, however good it was yesterday. Tier 2 guarantees the best
band is always current; tier 3 spends what is left finding new ones.

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
from app.models.instrument import BrokerInstrument, Instrument
from app.models.market_data import Candle
from app.models.scanner import ScannerConfiguration, ScannerResult
from app.scanner.engine import MIN_BARS_TO_SCORE
from app.scanner.watchlist import watchlisted_instrument_ids

#: Upper bound on the rotation candidate pool fetched per scan, before the
#: per-scan instrument cap is applied. A safety bound on the query, not the
#: scan size.
_ROTATION_POOL_LIMIT = 2000

#: How many of the best-scoring instruments are re-scored every single run,
#: before any exploration happens.
#:
#: This is the *exploit* half of the rotation. The scan covers a slice of a
#: 15k catalogue, and the strategy's universe is drawn from the latest run only
#: — so without a guaranteed re-scan a highly-ranked stock drops out of
#: consideration entirely on any night it is not swept, however good it is.
#: Holding the top band steady also means today's ranking is built from today's
#: prices for the names that matter, rather than a mix of dates.
TOP_RANKED_ALWAYS_RESCANNED = 200


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

    # Tier 1: watchlist members. Pinning an instrument is a promise that it gets
    # looked at next run, so the ways that promise can fail are counted and
    # reported below rather than being quietly absorbed.
    watchlisted = await watchlisted_instrument_ids(session)
    _extend(watchlisted)
    unscannable = [iid for iid in watchlisted if iid not in eligible_ids]
    watchlist_selected = len(ordered_ids)

    # Tier 2 — EXPLOIT: the current best, re-scored every run without exception.
    # The strategy picks its universe from the latest run alone, so a top-ranked
    # stock that is not re-scanned tonight is not a candidate tonight.
    _extend(await _top_ranked_ids(session, TOP_RANKED_ALWAYS_RESCANNED))
    exploit_count = len(ordered_ids)

    # Tier 3 — EXPLORE: the rotating sweep, oldest-scanned first, filling
    # whatever the cap leaves. Restricted to instruments that can actually be
    # scored, so the sweep is not saturated by never-scanned rows with no stored
    # history.
    _extend(await _rotation_ids(session, configuration, eligible_ids))

    chosen_ids = ordered_ids[:max_instruments]
    if not chosen_ids:
        return [], "no eligible instruments"

    instruments = await _load(session, chosen_ids)

    exploit_selected = min(exploit_count, max_instruments)
    parts = [
        f"rotation: {len(chosen_ids)} instruments "
        f"({exploit_selected} top-ranked re-scored, "
        f"{max(0, len(chosen_ids) - exploit_selected)} explored)"
    ]
    if watchlist_selected:
        parts.append(f"{min(watchlist_selected, max_instruments)} watchlisted")
    if unscannable:
        # Pinned, but with fewer than MIN_BARS_TO_SCORE stored daily bars, so
        # scoring it would only produce a skip. Named here because "I watchlisted
        # it and it never appears" is otherwise indistinguishable from a bug.
        parts.append(f"{len(unscannable)} watchlisted excluded: too little stored history to score")
    if watchlist_selected > max_instruments:
        # The cap is a data-budget guard (§6) and still wins, but a watchlist
        # bigger than a whole scan means some pins genuinely will not be honoured.
        parts.append(
            f"WARNING: watchlist ({watchlist_selected}) exceeds the per-scan cap "
            f"({max_instruments}); the newest pins were not selected"
        )
    return instruments, "; ".join(parts)


def _max_instruments(config: ScannerConfiguration | None, limit: int | None) -> int:
    if limit is not None:
        return limit
    if config is not None:
        return config.max_instruments_per_scan
    return 200


async def _top_ranked_ids(session: AsyncSession, limit: int) -> list[uuid.UUID]:
    """The best-scoring instruments, by each one's *most recent* score.

    Two details this gets right that a naive "order every result by score" does
    not, and both were wrong before:

      * **Ranks by `primary_score`**, the fundamentals-first blend that actually
        decides the ranking and the strategy's universe. The previous version
        ordered by `core_score` — the momentum core — so the tier meant to
        re-verify the best stocks was sorting them by a different number than
        the one that makes them the best.

      * **Uses each instrument's latest result only.** Ordering across all of
        history and de-duplicating keeps the *highest* score an instrument ever
        had, so a stock that scored 85 three weeks ago and 40 last night is
        pulled back in on the stale 85 — indefinitely, since re-scanning it
        cannot remove the old row.
    """
    ranked = select(
        ScannerResult.instrument_id.label("instrument_id"),
        ScannerResult.primary_score.label("primary_score"),
        func.row_number()
        .over(
            partition_by=ScannerResult.instrument_id,
            order_by=ScannerResult.created_at.desc(),
        )
        .label("recency_rank"),
    ).subquery()
    result = await session.execute(
        select(ranked.c.instrument_id)
        .where(ranked.c.recency_rank == 1)
        .order_by(ranked.c.primary_score.desc())
        .limit(limit)
    )
    return [row[0] for row in result.all()]


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
