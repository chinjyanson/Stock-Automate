"""Scanner and approval jobs (§16).

The daily rotating scan and approval-expiry processing. Both are idempotent —
re-running produces no duplicate side effects — and the scan holds a distributed
lock so two workers cannot burn the same slice of the (budgeted) universe at
once.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import redis
import structlog
from app.db import session_scope
from app.models.scanner import ScannerConfiguration
from app.scanner.engine import ScannerEngine
from app.scanner.proposals import ProposalService
from app.scanner.rotation import select_instruments
from sqlalchemy import select

from worker.app import app
from worker.locks import LockNotAcquiredError, distributed_lock

log = structlog.get_logger(__name__)


def _redis() -> redis.Redis:
    return redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6380/0"))


async def _run_rotation(limit: int | None) -> dict[str, Any]:
    async with session_scope() as session:
        config = (
            await session.execute(
                select(ScannerConfiguration).where(ScannerConfiguration.is_active.is_(True))
            )
        ).scalar_one_or_none()

        instruments, reason = await select_instruments(session, configuration=config, limit=limit)
        if not instruments:
            return {"scored": 0, "note": reason}

        summary = await ScannerEngine(session).run(
            instruments, configuration=config, selection_reason=reason
        )
        return {
            "considered": summary.considered,
            "scored": summary.scored,
            "skipped": summary.skipped,
            "screening": summary.screening_candidates,
            "watchlist": summary.watchlist_candidates,
        }


@app.task(bind=True, name="worker.jobs.scanner.rotate_scan", max_retries=2)
def rotate_scan(self, limit: int | None = None) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    """Run the daily rotating scan (§6, §16).

    Locked so two workers do not scan overlapping slices — the rotation advances
    `last_scanned_at`, and a double run would waste a day's coverage and, once
    intraday verification exists, provider budget.
    """
    try:
        with distributed_lock(_redis(), "scanner_rotation", ttl_seconds=1800):
            result = asyncio.run(_run_rotation(limit))
            log.info("job.scanner_rotation.completed", **result)
            return result
    except LockNotAcquiredError:
        log.info("job.scanner_rotation.skipped", reason="already running")
        return {"skipped": True, "reason": "another worker holds the lock"}
    except Exception as exc:
        log.exception("job.scanner_rotation.failed", error=str(exc))
        raise self.retry(exc=exc, countdown=300 * (2**self.request.retries)) from exc


async def _expire_proposals() -> int:
    async with session_scope() as session:
        return await ProposalService(session).expire_stale()


@app.task(bind=True, name="worker.jobs.scanner.expire_approvals")
def expire_approvals(self) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    """Mark expired pending proposals EXPIRED (§16).

    No lock: `expire_stale` only transitions PENDING→EXPIRED, which is
    idempotent, and running it twice is harmless.
    """
    expired = asyncio.run(_expire_proposals())
    if expired:
        log.info("job.expire_approvals.completed", expired=expired)
    return {"expired": expired}
