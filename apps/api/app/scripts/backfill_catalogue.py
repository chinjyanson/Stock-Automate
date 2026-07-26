"""Map + backfill the Trading 212-tradable catalogue, so the scanner has a universe.

    python -m app.scripts.backfill_catalogue --loop

Runs `BackfillService` in batches: for tradable instruments that have no yfinance
mapping yet, it derives a symbol and ingests daily history. Idempotent and
resumable — re-running continues where it left off, because every instrument a
batch touches gets a mapping and is not re-selected. `--loop` keeps going until
no never-attempted candidates remain; otherwise it does a single batch.

Uses free yfinance, throttled by the provider adapter (batch 50 / concurrency 4 /
backoff). This is a deliberately slow, one-time sweep — expect it to run for a
while on a full catalogue. The daily refresh job keeps everything current after.
"""

from __future__ import annotations

import argparse
import asyncio

import structlog

from app.config import get_settings
from app.data.factory import resolve_provider
from app.db import session_scope
from app.models.enums import ProviderKind
from app.services.backfill import BackfillService

log = structlog.get_logger(__name__)


async def _run(batch_size: int, loop: bool, trading212_only: bool) -> None:
    settings = get_settings()
    provider = resolve_provider(ProviderKind.YFINANCE, settings)
    round_num = 0
    try:
        while True:
            round_num += 1
            async with session_scope() as session:
                service = BackfillService(session)
                candidates = await service.select_backfill_candidates(
                    limit=batch_size, trading212_only=trading212_only
                )
                if not candidates:
                    counts = await service.funnel_counts(trading212_only=trading212_only)
                    print(
                        f"Done — nothing left to attempt. "
                        f"tradable={counts.tradable} mapped={counts.mapped} "
                        f"candled={counts.candled} scannable={counts.scannable}"
                    )
                    return
                result = await service.backfill(candidates, provider)  # type: ignore[arg-type]
                counts = await service.funnel_counts(trading212_only=trading212_only)
            print(
                f"[round {round_num}] {result.summary} | "
                f"funnel: mapped={counts.mapped} candled={counts.candled} "
                f"scannable={counts.scannable} / tradable={counts.tradable}"
            )
            if not loop:
                return
    finally:
        await provider.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill the tradable catalogue.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Instruments per batch (default: settings.backfill_ingest_batch_size).",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Keep processing batches until the catalogue is drained.",
    )
    parser.add_argument(
        "--all-venues",
        action="store_true",
        help="Do not restrict to Trading 212-tradable instruments.",
    )
    args = parser.parse_args()

    batch_size = args.batch_size or get_settings().backfill_ingest_batch_size
    asyncio.run(_run(batch_size=batch_size, loop=args.loop, trading212_only=not args.all_venues))


if __name__ == "__main__":
    main()
