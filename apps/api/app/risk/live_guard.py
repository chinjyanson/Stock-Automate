"""Live loss guard — halt and fall back to paper on breach (§9, §14).

The risk configuration carries a `max_daily_loss`. This service makes that number
real: it totals the day's *realised* live loss and, on breach, raises a global
daily-loss halt **and switches the trading venue back to paper**. Stopping the day
is not a suggestion — a limit that only warns is decoration
(`docs/risk-model.md`).

It runs frequently, reads only local records (no broker call, no credentials), and
is a no-op while the venue is paper — so it costs nothing until it matters.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time
from decimal import Decimal
from typing import Any

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.audit.service import AuditService
from app.models.enums import ActorKind, AuditEventKind, BrokerKind, HaltKind, HaltScope
from app.models.risk import TradeIntent
from app.risk.config import load_active_risk_config
from app.risk.halts import HaltService
from app.services.system_settings import (
    TRADING_LIVE_MODE_KEY,
    live_mode_enabled,
    set_bool_setting,
)

log = structlog.get_logger(__name__)


class LiveGuardService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._audit = AuditService(session)
        self._halts = HaltService(session)

    async def enforce(self) -> dict[str, Any]:
        """Halt and revert to paper if the day's live loss breaches the limit."""
        if not await live_mode_enabled(self._session):
            return {"live": False, "breached": False}

        config = await load_active_risk_config(self._session)
        if config is None or config.max_daily_loss is None:
            return {"live": True, "breached": False, "note": "no daily loss limit configured"}

        limit = Decimal(config.max_daily_loss)
        realised = await self._realised_live_loss_today()
        # A loss is negative P/L; the limit is a positive number.
        if realised > -limit:
            return {"live": True, "breached": False, "realised": str(realised)}

        await self._halts.activate(
            HaltKind.DAILY_LOSS,
            f"Daily live loss {realised} breached the {limit} limit",
            scope=HaltScope.GLOBAL,
        )
        # Fall back to paper: the venue should not stay pointed at real money
        # after the day's loss limit has been hit.
        await set_bool_setting(
            self._session,
            TRADING_LIVE_MODE_KEY,
            False,
            description="Whether the product trades the live venue rather than paper.",
            is_sensitive=True,
            user_id=None,
        )
        await self._audit.record(
            kind=AuditEventKind.LIVE_DISARMED,
            summary=f"Reverted to paper: daily live loss {realised} breached {limit}",
            actor_kind=ActorKind.RISK_ENGINE,
            subject_type="system_setting",
            subject_id=TRADING_LIVE_MODE_KEY,
            payload={"realised": str(realised), "max_daily_loss": str(limit)},
        )
        await self._session.flush()
        log.warning("live_guard.reverted_to_paper", realised=str(realised), limit=str(limit))
        return {"live": True, "breached": True, "realised": str(realised)}

    async def _realised_live_loss_today(self, day: date | None = None) -> Decimal:
        day = day or datetime.now(UTC).date()
        start = datetime.combine(day, time.min, tzinfo=UTC)
        end = datetime.combine(day, time.max, tzinfo=UTC)
        rows = (
            (
                await self._session.execute(
                    select(TradeIntent).where(
                        TradeIntent.broker == BrokerKind.TRADING212_LIVE,
                        TradeIntent.closed_at.is_not(None),
                        TradeIntent.closed_at >= start,
                        TradeIntent.closed_at <= end,
                    )
                )
            )
            .scalars()
            .all()
        )
        total = Decimal(0)
        for intent in rows:
            if intent.exit_price is None or intent.filled_price is None:
                continue
            qty = Decimal(intent.filled_quantity or 0)
            total += (Decimal(intent.exit_price) - Decimal(intent.filled_price)) * qty
        return total
