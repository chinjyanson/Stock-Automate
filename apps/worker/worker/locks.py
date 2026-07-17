"""Distributed locks for jobs whose duplicate execution could place orders (§16).

Redis-backed, with two properties that matter more than they look:

  * **Fenced by a token.** Release only deletes the key if we still own it. A
    job that overran its TTL must not delete a lock a *different* worker has
    since acquired — that would let two workers run concurrently while both
    believe they hold it.

  * **Always expiring.** A worker that dies holding a lock must not block the
    job forever, so the key always carries a TTL. The trade-off is that a task
    exceeding its TTL loses exclusivity; TTLs are therefore set well above the
    expected runtime.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import contextmanager

import redis
import structlog

log = structlog.get_logger(__name__)

#: Delete only if the value still matches our token. Lua so the check and the
#: delete are atomic — a GET followed by a DEL is a race.
_RELEASE_SCRIPT = """
if redis.call("get", KEYS[1]) == ARGV[1] then
    return redis.call("del", KEYS[1])
else
    return 0
end
"""


class LockNotAcquiredError(Exception):
    """Another worker holds this lock. Usually not an error — just skip."""


@contextmanager
def distributed_lock(
    client: redis.Redis, name: str, *, ttl_seconds: int = 600, blocking: bool = False
) -> Iterator[str]:
    """Hold a named lock for the duration of the block.

    `blocking=False` by default: for a scheduled job, "someone else is already
    doing this" means skip this tick, not queue up behind them. Waiting would
    just run the same work twice in sequence.
    """
    token = str(uuid.uuid4())
    key = f"lock:{name}"

    acquired = client.set(key, token, nx=True, ex=ttl_seconds)
    if not acquired:
        if not blocking:
            raise LockNotAcquiredError(f"Lock {name!r} is held by another worker")
        raise LockNotAcquiredError(f"Lock {name!r} is held and blocking is not supported")

    log.debug("lock.acquired", lock=name, ttl_seconds=ttl_seconds)
    try:
        yield token
    finally:
        try:
            released = client.eval(_RELEASE_SCRIPT, 1, key, token)
            if not released:
                # We no longer own it: our TTL expired and someone else took it.
                # Worth knowing — it means the TTL is too short for this job.
                log.warning(
                    "lock.release_failed_not_owner",
                    lock=name,
                    detail="Lock expired mid-task and was re-acquired elsewhere; "
                    "the task may have run concurrently. Increase its TTL.",
                )
            else:
                log.debug("lock.released", lock=name)
        except Exception as exc:
            log.error("lock.release_error", lock=name, error=str(exc))
