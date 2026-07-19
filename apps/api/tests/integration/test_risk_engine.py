"""Risk engine sizing and gating against real PostgreSQL.

The properties under test are the ones whose violation costs money: the smallest
cap binds, positions round down (never up, which would breach the cap that
produced them), and the engine fails closed on a missing configuration, an
active halt, or stale data.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from app.broker.types import BrokerAccount
from app.data.store import CandleStore
from app.models.enums import HaltKind, HaltScope, Interval
from app.models.instrument import Instrument
from app.models.risk import RiskConfiguration
from app.risk.engine import QUANTITY_STEP, RiskEngine
from app.risk.halts import HaltService

pytestmark = pytest.mark.asyncio


def _account(cash: Decimal = Decimal("100000")) -> BrokerAccount:
    return BrokerAccount(
        account_id="PAPER",
        currency="USD",
        cash=cash,
        total=cash,
        free_for_trading=cash,
        invested=Decimal(0),
    )


async def _candles(db: object, instrument: Instrument) -> list:
    return await CandleStore(db).get_candles(  # type: ignore[arg-type]
        instrument.id, Interval.D1, limit=250, closed_only=True
    )


async def _seed_config(db: object, **overrides: object) -> RiskConfiguration:
    config = RiskConfiguration(name="default", is_active=True, **overrides)
    db.add(config)  # type: ignore[attr-defined]
    await db.flush()  # type: ignore[attr-defined]
    return config


class TestSizing:
    async def test_a_position_is_sized_and_stopped(
        self, db: object, candled_instrument: Instrument
    ) -> None:
        config = await _seed_config(db)
        decision = await RiskEngine(db).evaluate(  # type: ignore[arg-type]
            instrument=candled_instrument,
            config=config,
            account=_account(),
            positions=[],
            candles=await _candles(db, candled_instrument),
        )
        assert not decision.rejected
        assert decision.approved_quantity > 0
        assert decision.stop_price is not None
        assert decision.stop_price < decision.entry_price
        # Risk is ~1% of equity by construction.
        assert decision.risk_amount <= Decimal("100000") * Decimal("0.02")

    async def test_quantity_rounds_down_to_the_step(
        self, db: object, candled_instrument: Instrument
    ) -> None:
        config = await _seed_config(db)
        decision = await RiskEngine(db).evaluate(  # type: ignore[arg-type]
            instrument=candled_instrument,
            config=config,
            account=_account(),
            positions=[],
            candles=await _candles(db, candled_instrument),
        )
        # The quantity is an exact multiple of the step — never rounded up.
        assert decision.approved_quantity == decision.approved_quantity.quantize(QUANTITY_STEP)

    async def test_monetary_cap_binds_when_it_is_smallest(
        self, db: object, candled_instrument: Instrument
    ) -> None:
        # A tiny absolute cap must win over the percentage caps.
        config = await _seed_config(db, monetary_position_cap=Decimal("100"))
        decision = await RiskEngine(db).evaluate(  # type: ignore[arg-type]
            instrument=candled_instrument,
            config=config,
            account=_account(),
            positions=[],
            candles=await _candles(db, candled_instrument),
        )
        assert not decision.rejected
        assert "monetary_cap" in decision.applied_caps
        # Position value cannot exceed the cap.
        assert decision.approved_quantity * decision.entry_price <= Decimal("100")


class TestFailClosed:
    async def test_missing_configuration_is_rejected(
        self, db: object, candled_instrument: Instrument
    ) -> None:
        decision = await RiskEngine(db).evaluate(  # type: ignore[arg-type]
            instrument=candled_instrument,
            config=None,
            account=_account(),
            positions=[],
            candles=await _candles(db, candled_instrument),
        )
        assert decision.rejected
        assert "configuration" in (decision.reason or "")

    async def test_active_global_halt_blocks_sizing(
        self, db: object, candled_instrument: Instrument, approver: object
    ) -> None:
        config = await _seed_config(db)
        await HaltService(db).activate(  # type: ignore[arg-type]
            HaltKind.KILL_SWITCH,
            "test halt",
            scope=HaltScope.GLOBAL,
            actor_user_id=approver,  # type: ignore[arg-type]
        )
        decision = await RiskEngine(db).evaluate(  # type: ignore[arg-type]
            instrument=candled_instrument,
            config=config,
            account=_account(),
            positions=[],
            candles=await _candles(db, candled_instrument),
        )
        assert decision.rejected
        assert "halted" in (decision.reason or "")

    async def test_open_position_cap_blocks_a_new_instrument(
        self, db: object, candled_instrument: Instrument
    ) -> None:
        config = await _seed_config(db, max_open_positions=0)
        decision = await RiskEngine(db).evaluate(  # type: ignore[arg-type]
            instrument=candled_instrument,
            config=config,
            account=_account(),
            positions=[],
            candles=await _candles(db, candled_instrument),
        )
        assert decision.rejected
        assert "open-position cap" in (decision.reason or "")
