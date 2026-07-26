"""One-time catalogue backfill job (§4, §7).

Drives `BackfillService` over the Trading 212-tradable universe: map a yfinance
symbol and ingest daily history for instruments that have neither. It is meant to
be *triggered* (admin endpoint or CLI), not scheduled — a one-time sweep that
lights up the scanner's universe. Afterwards the daily refresh job keeps
everything current and picks up any stragglers.

Paced and self-chaining: each run does one batch under a distributed lock, then
re-enqueues itself if work remains, so a single trigger drains the catalogue over
successive throttle-friendly batches rather than one enormous run.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import asdict
from typing import Any

import redis
import structlog
from app.config import get_settings
from app.data.factory import ProviderNotConfiguredError, resolve_provider
from app.db import session_scope
from app.models.enums import ProviderKind
from app.services.backfill import BackfillService

from worker.app import app
from worker.locks import LockNotAcquiredError, distributed_lock

log = structlog.get_logger(__name__)


def _redis() -> redis.Redis:
    return redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6380/0"))


async def _run(batch_size: int) -> dict[str, Any]:
    settings = get_settings()
    provider = resolve_provider(ProviderKind.YFINANCE, settings)
    try:
        async with session_scope() as session:
            service = BackfillService(session)
            candidates = await service.select_backfill_candidates(
                limit=batch_size, trading212_only=True
            )
            considered = len(candidates)
            if candidates:
                # `backfill` expects the concrete yfinance provider (batched download).
                await service.backfill(candidates, provider)  # type: ignore[arg-type]
            funnel = await service.funnel_counts(trading212_only=True)
            return {"considered": considered, "funnel": asdict(funnel)}
    finally:
        await provider.close()


@app.task(bind=True, name="worker.jobs.backfill.backfill_catalogue", max_retries=2)
def backfill_catalogue(  # type: ignore[no-untyped-def]
    self, batch_size: int | None = None, round_num: int = 1
):
    """Backfill one batch of the tradable catalogue, then continue if work remains."""
    settings = get_settings()
    size = batch_size or settings.backfill_ingest_batch_size
    try:
        with distributed_lock(_redis(), "backfill_catalogue", ttl_seconds=3600):
            result = asyncio.run(_run(size))
            log.info(
                "job.backfill_catalogue.completed",
                round=round_num,
                considered=result["considered"],
                **result["funnel"],
            )
            # Self-chain while a batch still finds work, bounded by a safety cap.
            if result["considered"] > 0 and round_num < settings.backfill_max_rounds:
                backfill_catalogue.apply_async(
                    kwargs={"batch_size": size, "round_num": round_num + 1},
                    countdown=settings.backfill_continuation_delay_seconds,
                )
            else:
                log.info("job.backfill_catalogue.finished", rounds=round_num, **result["funnel"])
            return result
    except LockNotAcquiredError:
        log.info("job.backfill_catalogue.skipped", reason="already running")
        return {"skipped": True, "reason": "another worker holds the lock"}
    except ProviderNotConfiguredError as exc:
        log.info("job.backfill_catalogue.skipped", reason=str(exc))
        return {"skipped": True, "reason": "yfinance not configured"}
    except Exception as exc:
        log.exception("job.backfill_catalogue.failed", error=str(exc))
        raise self.retry(exc=exc, countdown=300 * (2**self.request.retries)) from exc
