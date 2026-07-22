"""Live loss guard against real PostgreSQL.

The configured daily-loss limit must be enforced: when the day's realised live
loss breaches it, the guard halts trading **and drops the venue back to paper**.
A limit that only warns is decoration.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from app.models.enums import BrokerKind, HaltKind, OrderSide, TradeIntentStatus
from app.models.instrument import Instrument
from app.models.risk import RiskConfiguration, TradeIntent
from app.risk.halts import HaltService
from app.risk.live_guard import LiveGuardService
from app.services.system_settings import (
    TRADING_LIVE_MODE_KEY,
    live_mode_enabled,
    set_bool_setting,
)

pytestmark = pytest.mark.asyncio


async def _go_live(db: object) -> None:
    await set_bool_setting(
        db,  # type: ignore[arg-type]
        TRADING_LIVE_MODE_KEY,
        True,
        description="x",
        is_sensitive=True,
        user_id=None,
    )
    await db.flush()  # type: ignore[attr-defined]


async def _config(db: object, max_daily_loss: str | None) -> None:
    db.add(  # type: ignore[attr-defined]
        RiskConfiguration(
            name="default",
            is_active=True,
            max_daily_loss=Decimal(max_daily_loss) if max_daily_loss else None,
        )
    )
    await db.flush()  # type: ignore[attr-defined]


async def _closed_live_loss(
    db: object, instrument: Instrument, *, entry: str, exit_: str, qty: str
) -> None:
    db.add(  # type: ignore[attr-defined]
        TradeIntent(
            instrument_id=instrument.id,
            broker=BrokerKind.TRADING212_LIVE,
            client_reference=uuid.uuid4(),
            status=TradeIntentStatus.RECONCILED,
            side=OrderSide.BUY,
            quantity=Decimal(qty),
            filled_quantity=Decimal(qty),
            filled_price=Decimal(entry),
            exit_price=Decimal(exit_),
            exit_reason="stop",
            closed_at=datetime.now(UTC),
        )
    )
    await db.flush()  # type: ignore[attr-defined]


class TestDailyLoss:
    async def test_a_breach_halts_and_reverts_to_paper(
        self, db: object, candled_instrument: Instrument
    ) -> None:
        await _go_live(db)
        await _config(db, "100")
        # Entry 100, exit 50, qty 5 -> -250 realised, past the 100 limit.
        await _closed_live_loss(db, candled_instrument, entry="100", exit_="50", qty="5")

        result = await LiveGuardService(db).enforce()  # type: ignore[arg-type]
        await db.commit()  # type: ignore[attr-defined]

        assert result["breached"] is True
        # The venue is dropped back to paper and trading is halted.
        assert await live_mode_enabled(db) is False  # type: ignore[arg-type]
        halts = await HaltService(db).active_halts()  # type: ignore[arg-type]
        assert any(h.kind is HaltKind.DAILY_LOSS for h in halts)

    async def test_within_the_limit_does_nothing(
        self, db: object, candled_instrument: Instrument
    ) -> None:
        await _go_live(db)
        await _config(db, "1000")
        await _closed_live_loss(db, candled_instrument, entry="100", exit_="90", qty="5")  # -50

        result = await LiveGuardService(db).enforce()  # type: ignore[arg-type]
        await db.commit()  # type: ignore[attr-defined]

        assert result["breached"] is False
        assert await live_mode_enabled(db) is True  # type: ignore[arg-type]

    async def test_paper_mode_is_a_noop(self, db: object) -> None:
        # Nothing to guard while the venue is paper.
        result = await LiveGuardService(db).enforce()  # type: ignore[arg-type]
        assert result == {"live": False, "breached": False}

    async def test_no_configured_limit_is_a_noop(
        self, db: object, candled_instrument: Instrument
    ) -> None:
        await _go_live(db)
        await _config(db, None)
        await _closed_live_loss(db, candled_instrument, entry="100", exit_="1", qty="50")

        result = await LiveGuardService(db).enforce()  # type: ignore[arg-type]
        assert result["breached"] is False
        assert await live_mode_enabled(db) is True  # type: ignore[arg-type]
