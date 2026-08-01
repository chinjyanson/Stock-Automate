"""Daily market-regime measurement and alerting (§9, §16).

Measures the market-wide conditions, stores the reading, and emails only when
something has actually *changed*. A daily "conditions are normal" mail is a mail
nobody reads, and an alerting channel people ignore is worse than none — so the
comparison against yesterday's snapshot is the point of the job, not a detail.

Sending is best-effort throughout: the snapshot is persisted before any mail is
attempted, so a Brevo outage costs an alert but never the measurement.
"""

from __future__ import annotations

import os
from typing import Any

import redis
import structlog
from app.db import session_scope
from app.models.market_regime import MarketRegimeSnapshot
from app.models.user import User
from app.services.email import BrevoEmailService
from app.services.market_regime import MarketRegimeService, RegimeReading
from sqlalchemy import select

from worker.app import app
from worker.locks import LockNotAcquiredError, distributed_lock
from worker.runner import run_job

log = structlog.get_logger(__name__)

#: Buffett indicator levels worth a mail when crossed. Valuation moves slowly,
#: so these fire rarely by design — and never gate trading (see the service).
_BUFFETT_LEVELS = (150.0, 175.0, 200.0, 225.0, 250.0)
#: VIX levels likewise: 20 is "the market has noticed", 30 is genuine stress.
_VIX_LEVELS = (20.0, 30.0, 40.0)


def _redis() -> redis.Redis:
    return redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6380/0"))


def _crossed(previous: float | None, current: float | None, level: float) -> str | None:
    """Did a value cross `level` since the last reading, and which way?"""
    if previous is None or current is None:
        return None
    if previous < level <= current:
        return "above"
    if previous >= level > current:
        return "below"
    return None


def _changes(previous: MarketRegimeSnapshot | None, reading: RegimeReading) -> list[str]:
    """What is worth telling someone about, comparing with the last reading.

    First ever reading alerts unconditionally — there is nothing to compare
    against, and silently starting up would leave the operator unsure whether
    the job runs at all.
    """
    if previous is None:
        return [f"First market-regime reading: posture {reading.posture}"]

    changes: list[str] = []
    if previous.posture != reading.posture:
        changes.append(f"Risk posture {previous.posture} -> {reading.posture}")

    prev_sp = previous.sp500_uptrend
    if (
        prev_sp is not None
        and reading.sp500_uptrend is not None
        and prev_sp != reading.sp500_uptrend
    ):
        changes.append(
            "S&P 500 golden cross (50d above 200d)"
            if reading.sp500_uptrend
            else "S&P 500 DEATH CROSS (50d below 200d)"
        )
    prev_world = previous.world_uptrend
    if (
        prev_world is not None
        and reading.world_uptrend is not None
        and prev_world != reading.world_uptrend
    ):
        changes.append(
            "World index golden cross" if reading.world_uptrend else "World index DEATH CROSS"
        )

    # The yield curve matters at zero: inversion is the signal, not the level.
    prev_curve = (
        float(previous.yield_curve_10y2y) if previous.yield_curve_10y2y is not None else None
    )
    direction = _crossed(prev_curve, reading.yield_curve_10y2y, 0.0)
    if direction == "below":
        changes.append("Yield curve INVERTED (10y-2y went negative)")
    elif direction == "above":
        changes.append("Yield curve un-inverted (10y-2y back positive)")

    prev_vix = float(previous.vix) if previous.vix is not None else None
    for level in _VIX_LEVELS:
        if (way := _crossed(prev_vix, reading.vix, level)) is not None:
            changes.append(f"VIX crossed {way} {level:.0f} (now {reading.vix:.1f})")

    prev_buffett = (
        float(previous.buffett_indicator) if previous.buffett_indicator is not None else None
    )
    for level in _BUFFETT_LEVELS:
        if (way := _crossed(prev_buffett, reading.buffett_indicator, level)) is not None:
            changes.append(
                f"Buffett indicator crossed {way} {level:.0f}% "
                f"(now {reading.buffett_indicator:.0f}%) — valuation context, not a trade signal"
            )
    return changes


def _render(reading: RegimeReading, changes: list[str]) -> str:
    def row(label: str, value: str) -> str:
        return f"<tr><td style='padding:4px 12px 4px 0'>{label}</td><td><b>{value}</b></td></tr>"

    breadth = f"{reading.breadth:.0%} of {reading.breadth_sample:,}" if reading.breadth else "—"
    return f"""
    <h2>Market regime — {reading.as_of.isoformat()}</h2>
    <p><b>Posture: {reading.posture.upper()}</b> · risk multiplier
       &times;{reading.risk_factor}</p>
    <h3>What changed</h3>
    <ul>{"".join(f"<li>{c}</li>" for c in changes)}</ul>
    <h3>Readings</h3>
    <table>
      {row("VIX", f"{reading.vix:.1f}" if reading.vix else "—")}
      {row("Breadth above 200d", breadth)}
      {row("S&amp;P 500 trend", "up" if reading.sp500_uptrend else "DOWN")}
      {row("World index trend", "up" if reading.world_uptrend else "DOWN")}
      {
        row(
            "Yield curve 10y-2y",
            f"{reading.yield_curve_10y2y:+.2f}" if reading.yield_curve_10y2y is not None else "—",
        )
    }
      {
        row(
            "Buffett indicator",
            f"{reading.buffett_indicator:.0f}%" if reading.buffett_indicator else "—",
        )
    }
    </table>
    <p style="color:#666;font-size:12px">
      The risk multiplier scales position sizing only, and is floored — a weak
      regime trades smaller, never not at all. The Buffett indicator is reported
      for context and deliberately excluded from it: it is quarterly, months
      stale on arrival, and has been elevated for years.
    </p>
    <p style="color:#666;font-size:12px">Automated measurement, not financial advice.</p>
    """


async def _measure_and_alert() -> dict[str, Any]:
    async with session_scope() as session:
        service = MarketRegimeService(session)
        reading = await service.measure()
        previous = await service.previous(reading.as_of)
        changes = _changes(previous, reading)
        # Persist before mailing: a send failure must not lose the measurement.
        await service.record(reading)

        outcome: dict[str, Any] = {
            "as_of": reading.as_of.isoformat(),
            "posture": reading.posture,
            "risk_factor": reading.risk_factor,
            "breadth": reading.breadth,
            "vix": reading.vix,
            "changes": len(changes),
        }
        if not changes:
            return outcome

        email = BrevoEmailService()
        if not email.is_configured:
            log.info("job.market_regime.alert_skipped", reason="brevo not configured")
            outcome["alerted"] = 0
            return outcome

        users = (
            (await session.execute(select(User).where(User.is_active.is_(True)))).scalars().all()
        )
        subject = f"Market regime: {reading.posture} — {changes[0]}"
        html = _render(reading, changes)
        sent = 0
        for user in users:
            if await email.send(
                to_email=user.email, to_name=user.display_name, subject=subject, html=html
            ):
                sent += 1
        outcome["alerted"] = sent
        return outcome


@app.task(bind=True, name="worker.jobs.market_regime.measure_market_regime", max_retries=2)
def measure_market_regime(self) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    """Measure market-wide conditions, store them, and alert on any change."""
    try:
        with distributed_lock(_redis(), "measure_market_regime", ttl_seconds=900):
            result = run_job(_measure_and_alert())
            log.info("job.market_regime.completed", **result)
            return result
    except LockNotAcquiredError:
        return {"skipped": True, "reason": "another worker holds the lock"}
    except Exception as exc:
        log.exception("job.market_regime.failed", error=str(exc))
        raise self.retry(exc=exc, countdown=600 * (2**self.request.retries)) from exc
