"""Tag instruments with sector/industry and set up the sector ETFs (§6).

    python -m app.scripts.enrich_sectors --loop

First ensures the 11 SPDR sector ETFs exist with candles (so the scorer has each
sector's proxy series), then tags mapped instruments with the sector/industry
yfinance assigns them. Idempotent and resumable — `--loop` runs batches until no
un-attempted instruments remain; otherwise a single batch.

Per-symbol `.info` is slow/rate-limited, so a full sweep of the scannable
universe is an overnight-scale job. The scanner works throughout: untagged stocks
simply get no sector signal until their tag lands.
"""

from __future__ import annotations

import argparse
import asyncio

from app.config import get_settings
from app.data.factory import resolve_provider
from app.db import session_scope
from app.models.enums import ProviderKind
from app.services.enrichment import EnrichmentService


async def _run(batch_size: int, loop: bool, etfs_only: bool) -> None:
    settings = get_settings()
    provider = resolve_provider(ProviderKind.YFINANCE, settings)
    try:
        async with session_scope() as session:
            etf = await EnrichmentService(session).ensure_sector_etfs(provider)  # type: ignore[arg-type]
        print(f"Sector ETFs: {etf['created']} created, {etf['candles_written']} candles ingested.")
        if etfs_only:
            return

        round_num = 0
        while True:
            round_num += 1
            async with session_scope() as session:
                result = await EnrichmentService(session).enrich_sectors(provider, batch_size)  # type: ignore[arg-type]
            print(
                f"[round {round_num}] {result.attempted} processed: "
                f"{result.classified} sector-classified, "
                f"{result.with_fundamentals} with fundamentals."
            )
            if not loop or result.attempted == 0:
                if result.attempted == 0:
                    print("Done — nothing left to tag.")
                return
    finally:
        await provider.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich instruments with sector/industry.")
    parser.add_argument("--batch-size", type=int, default=300, help="Instruments per batch.")
    parser.add_argument("--loop", action="store_true", help="Tag until the universe is done.")
    parser.add_argument(
        "--etfs-only", action="store_true", help="Only (re)ingest the sector ETFs, skip tagging."
    )
    args = parser.parse_args()
    asyncio.run(_run(batch_size=args.batch_size, loop=args.loop, etfs_only=args.etfs_only))


if __name__ == "__main__":
    main()
