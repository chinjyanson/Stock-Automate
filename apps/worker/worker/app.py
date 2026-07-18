"""Celery application and schedule (§16).

Jobs must be idempotent even though the broker's order endpoint is not. Two
mechanisms make that true, and both matter:

  * **Idempotent by construction.** Sync and ingestion upsert; re-running
    converges rather than duplicating. This is the primary defence — a job that
    is safe to run twice needs no lock.

  * **Distributed locks where duplication could place an order.** Celery
    guarantees at-least-once delivery, and a worker that dies mid-task has its
    task redelivered. For anything that can reach a broker, "ran twice" must be
    impossible, not merely unlikely.

Phase 1 schedules only the jobs whose dependencies exist: instrument sync and
daily candle refresh. Strategy evaluation, reconciliation and the market
summaries arrive with the phases that build them, rather than being scheduled
now as no-ops that look implemented.
"""

from __future__ import annotations

import os

import structlog
from celery import Celery
from celery.schedules import crontab

log = structlog.get_logger(__name__)

app = Celery(
    "trading_platform",
    broker=os.environ.get("CELERY_BROKER_URL", "redis://localhost:6380/1"),
    backend=os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6380/2"),
    include=[
        "worker.jobs.instruments",
        "worker.jobs.market_data",
        "worker.jobs.scanner",
    ],
)

app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    # Ack after the task returns, not on receipt. If a worker is killed
    # mid-task the message is redelivered rather than silently lost — safe
    # because our tasks are idempotent, and the alternative (losing a
    # reconciliation) is worse than running one twice.
    task_acks_late=True,
    # Do not prefetch a queue's worth of work into one process; jobs here are
    # long and I/O-bound, and hoarding them stalls the others.
    worker_prefetch_multiplier=1,
    task_reject_on_worker_lost=True,
    # A ceiling so a wedged provider call cannot hold a worker slot forever.
    task_time_limit=1800,
    task_soft_time_limit=1500,
    result_expires=3600,
)

app.conf.beat_schedule = {
    # Trading 212 rate-limits instrument metadata hard, and the catalogue
    # changes slowly. Daily, before the London open.
    "sync-instruments": {
        "task": "worker.jobs.instruments.sync_instruments",
        "schedule": crontab(hour=6, minute=0),
        "options": {"expires": 3600},
    },
    # After the US close, so the previous session's daily bars exist. This is a
    # deliberate simplification for Phase 1: a per-exchange schedule (§13, "use
    # the relevant exchange schedule rather than one hard-coded market-open
    # time") lands with the trading-schedule sync.
    "refresh-daily-candles": {
        "task": "worker.jobs.market_data.refresh_daily_candles",
        "schedule": crontab(hour=21, minute=30),
        "options": {"expires": 7200},
    },
    # After the candle refresh, so the scan reads fresh data. The rotation caps
    # itself per §6, so this covers the universe over successive days.
    "scanner-rotation": {
        "task": "worker.jobs.scanner.rotate_scan",
        "schedule": crontab(hour=22, minute=0),
        "options": {"expires": 7200},
    },
    # Frequent and cheap: a stale approval must not linger far past its expiry.
    "expire-approvals": {
        "task": "worker.jobs.scanner.expire_approvals",
        "schedule": crontab(minute="*/5"),
        "options": {"expires": 300},
    },
}


@app.task(bind=True, name="worker.health.ping")
def ping(self) -> str:  # type: ignore[no-untyped-def]
    """Liveness probe for the worker (§18)."""
    return "pong"
