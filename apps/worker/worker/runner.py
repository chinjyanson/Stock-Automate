"""The single entrypoint every Celery job uses to run async work.

`asyncio.run` builds a fresh event loop per call, and a Celery prefork worker
runs many tasks in the same process. The API's engine is a module-level
singleton holding a pooled set of asyncpg connections, and an asyncpg connection
belongs to the loop that opened it — so the second task in a process inherits
connections bound to the first task's now-closed loop. The result is the pair of
failures this exists to prevent:

    RuntimeError: Event loop is closed
    RuntimeError: Task ... got Future ... attached to a different loop

`pool_pre_ping` does not save it. The ping is itself an await on the dead loop,
so the health check becomes the thing that raises.

Disposing the engine at the end of each task keeps the pool's lifetime inside
the loop's lifetime, which is the actual invariant. The cost is a new connection
per task — negligible for jobs that run every few minutes and do real work, and
far cheaper than a scheduled job failing at 06:00 unattended.

The API is unaffected and deliberately untouched: it runs one loop for the
process lifetime and disposes on shutdown, so pooling there is both correct and
worth having.
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any

from app.db import dispose_engine


def run_job[T](coro: Coroutine[Any, Any, T]) -> T:
    """Run `coro` on a fresh loop and leave no connection behind.

    Disposal is in a `finally` so it happens on the failure path too — a task
    that raises must not strand a connection for the next task to trip over.
    """

    async def _main() -> T:
        try:
            return await coro
        finally:
            await dispose_engine()

    return asyncio.run(_main())
