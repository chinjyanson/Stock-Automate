"""Instrument synchronisation job (§16)."""

from __future__ import annotations

import asyncio
import os
from typing import Any

import redis
import structlog
from app.broker.factory import default_paper_broker_kind, resolve_broker
from app.config import get_settings
from app.db import session_scope
from app.models.enums import BrokerKind
from app.services.instrument_sync import InstrumentSyncService

from worker.app import app
from worker.locks import LockNotAcquiredError, distributed_lock

log = structlog.get_logger(__name__)


def _redis() -> redis.Redis:
    return redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6380/0"))


async def _sync(broker_kind: BrokerKind) -> dict[str, Any]:
    settings = get_settings()
    broker = resolve_broker(broker_kind, settings)
    try:
        async with session_scope() as session:
            result = await InstrumentSyncService(session).sync(
                broker, actor_label="worker:sync_instruments"
            )
            return {
                "broker": str(result.broker),
                "total_from_broker": result.total_from_broker,
                "created": result.broker_instruments_created,
                "updated": result.broker_instruments_updated,
                "delisted": result.delisted,
                "errors": result.errors[:10],
            }
    finally:
        await broker.close()


@app.task(bind=True, name="worker.jobs.instruments.sync_instruments", max_retries=3)
def sync_instruments(self, broker: str | None = None) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    """Pull the broker catalogue into local tables.

    Idempotent on its own (it upserts), so the lock is not protecting
    correctness here — it prevents two workers from spending Trading 212's
    tight instrument-metadata rate limit on the same work and tripping a 429.

    Defaults to the paper broker: a scheduled job must never reach for a live
    venue on its own.
    """
    kind = BrokerKind(broker) if broker else default_paper_broker_kind()

    try:
        with distributed_lock(_redis(), "sync_instruments", ttl_seconds=900):
            result = asyncio.run(_sync(kind))
            log.info("job.sync_instruments.completed", **result)
            return result
    except LockNotAcquiredError:
        # Not a failure. Another worker is already doing it.
        log.info("job.sync_instruments.skipped", reason="already running")
        return {"skipped": True, "reason": "another worker holds the lock"}
    except Exception as exc:
        log.exception("job.sync_instruments.failed", error=str(exc))
        # Exponential backoff: a broker that is down stays down for a while, and
        # hammering it makes both sides worse.
        raise self.retry(exc=exc, countdown=60 * (2**self.request.retries)) from exc
