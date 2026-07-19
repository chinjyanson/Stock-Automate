"""End-of-day summary against real PostgreSQL.

The summary captures a venue's day, upserts per (broker, day), tracks realised
P/L from the positions closed that day, and reports the equity change against the
previous day's row.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from sqlalchemy import func, select

from app.broker.internal_paper import InternalPaperBroker
from app.broker.types import BrokerOrderRequest
from app.models.enums import BrokerKind, OrderSide, OrderType, TradeIntentStatus
from app.models.instrument import Instrument
from app.models.reporting import DailyAccountSummary
from app.models.risk import TradeIntent
from app.services.eod import EODSummaryService

pytestmark = pytest.mark.asyncio


async def _buy(db: object, instrument: Instrument, qty: Decimal) -> InternalPaperBroker:
    broker = InternalPaperBroker(db, starting_cash=Decimal("100000"))  # type: ignore[arg-type]
    await broker.place_order(
        BrokerOrderRequest(
            broker_ticker=str(instrument.id),
            side=OrderSide.BUY,
            quantity=qty,
            order_type=OrderType.MARKET,
        )
    )
    return broker


class TestGenerate:
    async def test_summary_captures_the_open_book(
        self, db: object, candled_instrument: Instrument
    ) -> None:
        broker = await _buy(db, candled_instrument, Decimal("2"))
        summary = await EODSummaryService(db).generate(broker)  # type: ignore[arg-type]
        await db.commit()  # type: ignore[attr-defined]

        assert summary.broker is BrokerKind.INTERNAL_PAPER
        assert summary.open_positions == 1
        assert Decimal(summary.invested) > 0
        assert Decimal(summary.equity) > 0

    async def test_generation_is_idempotent_per_day(
        self, db: object, candled_instrument: Instrument
    ) -> None:
        broker = await _buy(db, candled_instrument, Decimal("1"))
        service = EODSummaryService(db)  # type: ignore[arg-type]
        await service.generate(broker)
        await service.generate(broker)
        await db.commit()  # type: ignore[attr-defined]

        count = (
            await db.execute(  # type: ignore[attr-defined]
                select(func.count()).select_from(DailyAccountSummary)
            )
        ).scalar_one()
        assert count == 1

    async def test_realised_pnl_counts_positions_closed_today(
        self, db: object, candled_instrument: Instrument
    ) -> None:
        broker = InternalPaperBroker(db, starting_cash=Decimal("100000"))  # type: ignore[arg-type]
        now = datetime.now(UTC)
        # A closed intent booked today: entry 100, exit 110, qty 3 -> +30 realised.
        db.add(  # type: ignore[attr-defined]
            TradeIntent(
                instrument_id=candled_instrument.id,
                broker=broker.kind,
                client_reference=uuid.uuid4(),
                status=TradeIntentStatus.RECONCILED,
                side=OrderSide.BUY,
                quantity=Decimal("3"),
                filled_quantity=Decimal("3"),
                filled_price=Decimal("100"),
                exit_price=Decimal("110"),
                exit_reason="stop",
                closed_at=now,
            )
        )
        await db.flush()  # type: ignore[attr-defined]

        summary = await EODSummaryService(db).generate(broker)  # type: ignore[arg-type]
        await db.commit()  # type: ignore[attr-defined]
        assert Decimal(summary.realised_pnl) == Decimal("30")

    async def test_equity_change_is_measured_against_the_previous_day(
        self, db: object, candled_instrument: Instrument
    ) -> None:
        broker = InternalPaperBroker(db, starting_cash=Decimal("100000"))  # type: ignore[arg-type]
        yesterday = (datetime.now(UTC) - timedelta(days=1)).date()
        db.add(  # type: ignore[attr-defined]
            DailyAccountSummary(
                broker=broker.kind,
                summary_date=yesterday,
                currency="GBP",
                cash=Decimal("90000"),
                equity=Decimal("90000"),
                invested=Decimal("0"),
                unrealised_pnl=Decimal("0"),
                realised_pnl=Decimal("0"),
                open_positions=0,
                trades_today=0,
                active_halts=0,
            )
        )
        await db.flush()  # type: ignore[attr-defined]

        summary = await EODSummaryService(db).generate(broker)  # type: ignore[arg-type]
        await db.commit()  # type: ignore[attr-defined]
        # A fresh 100k account against a 90k prior day is +10k.
        assert summary.equity_change is not None
        assert Decimal(summary.equity_change) == Decimal("10000.0000")
