"""Re-ingest daily candles at a longer lookback, for backtesting (§4, §8).

    python -m app.scripts.deepen_history --years 10 --size 600

The catalogue was built at `DEFAULT_BACKFILL_DAYS` (730), which is why the store
tops out around 500 daily bars. That is not enough to measure a strategy: with
the backtest's 260-bar warmup, a median instrument leaves roughly fourteen
tradeable bars, so four hundred names produced fewer than a hundred trades and
every comparison came back inside its own error bars.

This forces a full re-ingest at a deeper window. `force_full_backfill` is the
point — the ordinary incremental path only asks for bars *newer* than what is
already stored, so it can never reach backwards, and re-running the normal
refresh would do nothing at all.

Targets the scanner's top-ranked names rather than the whole catalogue, because
those are the only ones the strategy ever sees and therefore the only ones worth
replaying. Idempotent: upserts by (instrument, interval, timestamp), so
re-running costs provider calls but cannot corrupt anything.

Free yfinance, throttled by the provider adapter. Expect this to take a while.
"""

from __future__ import annotations

import argparse
import asyncio
import uuid
from collections.abc import Sequence

import structlog
from sqlalchemy import func, select

from app.config import get_settings
from app.data.factory import resolve_provider
from app.db import session_scope
from app.models.enums import Interval, ProviderKind
from app.models.market_data import Candle
from app.services.ingestion import IngestionService

log = structlog.get_logger(__name__)


async def _bar_counts(session: object, ids: Sequence[uuid.UUID]) -> dict[uuid.UUID, int]:
    rows = (
        await session.execute(  # type: ignore[attr-defined]
            select(Candle.instrument_id, func.count())
            .where(Candle.instrument_id.in_(ids), Candle.interval == Interval.D1)
            .group_by(Candle.instrument_id)
        )
    ).all()
    return {r[0]: int(r[1]) for r in rows}


async def _run(years: int, size: int, batch_size: int) -> None:
    from app.backtest.service import BacktestService

    settings = get_settings()
    provider = resolve_provider(ProviderKind.YFINANCE, settings)
    backfill_days = years * 365

    async with session_scope() as session:
        service = BacktestService(session)
        instruments = await service.top_ranked_instruments(size)
        if not instruments:
            instruments = await service.instruments_with_history(size)
        ids = [i.id for i in instruments]
        before = await _bar_counts(session, ids)

    if not instruments:
        print("No instruments to deepen. Run a scan first.")
        return

    depths = sorted(before.get(i, 0) for i in ids)
    median_before = depths[len(depths) // 2] if depths else 0
    print(f"Deepening {len(instruments)} instrument(s) to ~{years}y ({backfill_days} days)")
    print(f"Before: median {median_before} bars, max {max(depths) if depths else 0}\n")

    done = 0
    for start in range(0, len(instruments), batch_size):
        batch = instruments[start : start + batch_size]
        async with session_scope() as session:
            ingestion = IngestionService(session)
            for instrument in batch:
                try:
                    await ingestion.ingest_daily(
                        instrument,
                        provider,
                        backfill_days=backfill_days,
                        force_full_backfill=True,
                    )
                except Exception as exc:  # one bad symbol must not end the sweep
                    log.warning(
                        "deepen.instrument_failed",
                        instrument_id=str(instrument.id),
                        error=str(exc),
                    )
        done += len(batch)
        print(f"  {done}/{len(instruments)} instruments re-ingested", flush=True)

    async with session_scope() as session:
        after = await _bar_counts(session, ids)
    depths_after = sorted(after.get(i, 0) for i in ids)
    median_after = depths_after[len(depths_after) // 2] if depths_after else 0
    deep = sum(1 for d in depths_after if d >= 1000)
    print(
        f"\nAfter:  median {median_after} bars, max {max(depths_after) if depths_after else 0}, "
        f"{deep} instrument(s) with 1000+ bars"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-ingest daily candles at a deeper window.")
    parser.add_argument("--years", type=int, default=10, help="Lookback in years.")
    parser.add_argument("--size", type=int, default=600, help="Top-ranked instruments to deepen.")
    parser.add_argument("--batch-size", type=int, default=50, help="Instruments per commit.")
    args = parser.parse_args()
    asyncio.run(_run(years=args.years, size=args.size, batch_size=args.batch_size))


if __name__ == "__main__":
    main()
