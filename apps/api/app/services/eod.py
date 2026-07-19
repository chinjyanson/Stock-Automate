"""End-of-day account summary (§16).

Persists one `DailyAccountSummary` per venue per day: cash, equity, what is open,
what moved. Upserts on (broker, day) so a re-run — a retried job, a manual
regeneration — converges rather than duplicating.

Realised P/L is computed from the intents *closed* on the day (a long position's
exit price minus its entry, times filled quantity); unrealised P/L is the mark on
what is still open. Keeping the two separate is deliberate: a good day that is all
unrealised is not the same as one that is banked.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time
from decimal import Decimal

import structlog
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.audit.service import AuditService
from app.broker.base import Broker
from app.models.enums import ActorKind, AuditEventKind
from app.models.reporting import DailyAccountSummary
from app.models.risk import RiskHalt, TradeIntent

log = structlog.get_logger(__name__)


class EODSummaryService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._audit = AuditService(session)

    async def generate(
        self, broker: Broker, *, summary_date: date | None = None
    ) -> DailyAccountSummary:
        summary_date = summary_date or datetime.now(UTC).date()
        account = await broker.get_account()
        positions = await broker.get_positions()

        unrealised = sum(
            (p.unrealised_pnl or Decimal(0) for p in positions), start=Decimal(0)
        )
        realised = await self._realised_pnl(broker, summary_date)
        trades = await self._trades_on(broker, summary_date)
        active_halts = await self._active_halt_count()
        previous = await self._previous_summary(broker, summary_date)
        equity = account.total
        equity_change = (
            equity - Decimal(previous.equity) if previous is not None else None
        )

        summary = await self._existing(broker, summary_date)
        if summary is None:
            summary = DailyAccountSummary(
                broker=broker.kind, summary_date=summary_date, currency=account.currency
            )
            self._session.add(summary)

        summary.currency = account.currency
        summary.cash = account.cash
        summary.equity = equity
        summary.invested = account.invested or Decimal(0)
        summary.unrealised_pnl = unrealised
        summary.realised_pnl = realised
        summary.equity_change = equity_change
        summary.open_positions = len([p for p in positions if p.quantity > 0])
        summary.trades_today = trades
        summary.active_halts = active_halts
        await self._session.flush()

        await self._audit.record(
            kind=AuditEventKind.EOD_SUMMARY_GENERATED,
            summary=(
                f"EOD {broker.kind.value} {summary_date}: equity {equity} "
                f"(realised {realised}, unrealised {unrealised}, {summary.open_positions} open)"
            ),
            actor_kind=ActorKind.SCHEDULER,
            actor_label="eod_summary_job",
            subject_type="daily_account_summary",
            subject_id=str(summary.id),
            payload={
                "equity": str(equity),
                "realised_pnl": str(realised),
                "unrealised_pnl": str(unrealised),
                "open_positions": summary.open_positions,
            },
        )
        log.info(
            "eod.summary_generated",
            broker=broker.kind.value,
            date=summary_date.isoformat(),
            equity=str(equity),
        )
        return summary

    # -- Helpers -----------------------------------------------------------

    def _day_bounds(self, summary_date: date) -> tuple[datetime, datetime]:
        start = datetime.combine(summary_date, time.min, tzinfo=UTC)
        end = datetime.combine(summary_date, time.max, tzinfo=UTC)
        return start, end

    async def _realised_pnl(self, broker: Broker, summary_date: date) -> Decimal:
        start, end = self._day_bounds(summary_date)
        rows = (
            (
                await self._session.execute(
                    select(TradeIntent).where(
                        TradeIntent.broker == broker.kind,
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

    async def _trades_on(self, broker: Broker, summary_date: date) -> int:
        start, end = self._day_bounds(summary_date)
        count = await self._session.execute(
            select(func.count())
            .select_from(TradeIntent)
            .where(
                TradeIntent.broker == broker.kind,
                TradeIntent.created_at >= start,
                TradeIntent.created_at <= end,
            )
        )
        return int(count.scalar_one())

    async def _active_halt_count(self) -> int:
        count = await self._session.execute(
            select(func.count()).select_from(RiskHalt).where(RiskHalt.is_active.is_(True))
        )
        return int(count.scalar_one())

    async def _existing(
        self, broker: Broker, summary_date: date
    ) -> DailyAccountSummary | None:
        return (
            await self._session.execute(
                select(DailyAccountSummary).where(
                    DailyAccountSummary.broker == broker.kind,
                    DailyAccountSummary.summary_date == summary_date,
                )
            )
        ).scalar_one_or_none()

    async def _previous_summary(
        self, broker: Broker, summary_date: date
    ) -> DailyAccountSummary | None:
        return (
            await self._session.execute(
                select(DailyAccountSummary)
                .where(
                    DailyAccountSummary.broker == broker.kind,
                    DailyAccountSummary.summary_date < summary_date,
                )
                .order_by(DailyAccountSummary.summary_date.desc())
                .limit(1)
            )
        ).scalar_one_or_none()
