"""Live execution routing and gating against real PostgreSQL.

No real-money order is placed, and none can be: these drive a **fake** broker that
reports `is_live=True`, injected into the execution service. The real Trading 212
live adapter is never constructed (it needs the server flag + credentials the
tests never set). What is under test is the money-adjacent logic — live ticker
resolution, the depth-in-defence preflight, and the capital ceiling.
"""

from __future__ import annotations

import types
import uuid
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from app.broker.base import Broker
from app.broker.types import (
    BrokerAccount,
    BrokerOrder,
    BrokerOrderRequest,
    BrokerPosition,
    ReconciliationResult,
)
from app.models.enums import (
    BrokerKind,
    HaltKind,
    HaltScope,
    OrderStatus,
)
from app.models.instrument import BrokerInstrument, Instrument
from app.models.risk import RiskConfiguration
from app.models.scanner import ProposalStatus, TradeProposal
from app.risk.execution import ExecutionError, ExecutionService
from app.risk.halts import HaltService
from app.scanner.proposals import ProposalInputs, ProposalService
from app.services.system_settings import TRADING_LIVE_MODE_KEY, set_bool_setting

pytestmark = pytest.mark.asyncio

_LIVE_TICKER = "AAPL_US_EQ"


class FakeLiveBroker(Broker):
    """A live-reporting broker that records orders instead of sending them."""

    kind = BrokerKind.TRADING212_LIVE

    def __init__(self) -> None:
        self.orders: list[BrokerOrderRequest] = []

    async def sync_instruments(self):  # type: ignore[no-untyped-def]
        return []

    async def get_account(self) -> BrokerAccount:
        cash = Decimal("1000000")
        return BrokerAccount(
            account_id="LIVE", currency="USD", cash=cash, total=cash, free_for_trading=cash
        )

    async def get_positions(self) -> list[BrokerPosition]:
        return []

    async def get_pending_orders(self) -> list[BrokerOrder]:
        return []

    async def get_order_history(self) -> list[BrokerOrder]:
        return []

    async def place_order(self, request: BrokerOrderRequest) -> BrokerOrder:
        self.orders.append(request)
        return BrokerOrder(
            broker_order_id=str(uuid.uuid4()),
            broker_ticker=request.broker_ticker,
            side=request.side,
            order_type=request.order_type,
            quantity=request.quantity,
            status=OrderStatus.FILLED,
            filled_quantity=request.quantity,
            average_fill_price=Decimal("100"),
        )

    async def cancel_order(self, broker_order_id: str) -> None:
        return None

    async def reconcile(self) -> ReconciliationResult:
        return ReconciliationResult(
            broker=self.kind, reconciled_at=datetime.now(UTC), positions_checked=0, orders_checked=0
        )


def _live_settings_on() -> types.SimpleNamespace:
    return types.SimpleNamespace(live_trading_enabled=True)


async def _go_live(db: object) -> None:
    """Point the venue at live — the toggle that replaced the arming session."""
    await set_bool_setting(
        db,  # type: ignore[arg-type]
        TRADING_LIVE_MODE_KEY,
        True,
        description="x",
        is_sensitive=True,
        user_id=None,
    )
    await db.flush()  # type: ignore[attr-defined]


async def _live_instrument(db: object, instrument: Instrument, available: bool = True) -> None:
    db.add(  # type: ignore[attr-defined]
        BrokerInstrument(
            instrument_id=instrument.id,
            broker=BrokerKind.TRADING212_LIVE,
            broker_ticker=_LIVE_TICKER,
            is_currently_available=available,
        )
    )
    await db.flush()  # type: ignore[attr-defined]


async def _approved_proposal(db: object, instrument: Instrument, **inputs: object) -> TradeProposal:
    proposal = await ProposalService(db).propose_from_signal(  # type: ignore[arg-type]
        instrument,
        ProposalInputs(account_equity=Decimal(str(inputs.get("account_equity", "100000")))),
        reason="test",
    )
    proposal.status = ProposalStatus.APPROVED
    proposal.decided_at = datetime.now(UTC)
    await db.flush()  # type: ignore[attr-defined]
    return proposal


async def _config(db: object, **overrides: object) -> None:
    db.add(RiskConfiguration(name="default", is_active=True, **overrides))  # type: ignore[attr-defined]
    await db.flush()  # type: ignore[attr-defined]


class TestLiveRouting:
    async def test_live_mode_routes_with_the_resolved_ticker(
        self, db: object, candled_instrument: Instrument, approver: uuid.UUID, monkeypatch
    ) -> None:
        monkeypatch.setattr("app.risk.execution.get_settings", _live_settings_on)
        await _config(db)
        await _live_instrument(db, candled_instrument)
        await _go_live(db)
        proposal = await _approved_proposal(db, candled_instrument)

        fake = FakeLiveBroker()
        executed = await ExecutionService(db, broker=fake).execute_approved(  # type: ignore[arg-type]
            proposal, actor_user_id=approver
        )
        await db.commit()  # type: ignore[attr-defined]

        assert executed.status is ProposalStatus.EXECUTED
        assert fake.orders, "an order should have reached the live broker"
        # The broker's own ticker, not our instrument UUID.
        assert fake.orders[0].broker_ticker == _LIVE_TICKER

    async def test_capital_ceiling_bounds_the_order(
        self, db: object, candled_instrument: Instrument, approver: uuid.UUID, monkeypatch
    ) -> None:
        monkeypatch.setattr("app.risk.execution.get_settings", _live_settings_on)
        # Tiny ceiling against a million-dollar account.
        await _config(db, max_live_capital=Decimal("500"))
        await _live_instrument(db, candled_instrument)
        await _go_live(db)
        proposal = await _approved_proposal(db, candled_instrument)

        fake = FakeLiveBroker()
        await ExecutionService(db, broker=fake).execute_approved(  # type: ignore[arg-type]
            proposal, actor_user_id=approver
        )
        await db.commit()  # type: ignore[attr-defined]

        assert fake.orders
        market = fake.orders[0]
        # Position value cannot exceed the affirmed ceiling.
        assert market.quantity * proposal.indicative_entry_price <= Decimal("500")


class TestLivePreflight:
    async def test_in_paper_mode_live_is_refused(
        self, db: object, candled_instrument: Instrument, approver: uuid.UUID, monkeypatch
    ) -> None:
        monkeypatch.setattr("app.risk.execution.get_settings", _live_settings_on)
        await _config(db)
        await _live_instrument(db, candled_instrument)
        proposal = await _approved_proposal(db, candled_instrument)

        fake = FakeLiveBroker()
        with pytest.raises(ExecutionError, match="venue is set to paper"):
            await ExecutionService(db, broker=fake).execute_approved(  # type: ignore[arg-type]
                proposal, actor_user_id=approver
            )
        await db.commit()  # type: ignore[attr-defined]
        assert not fake.orders
        await db.refresh(proposal)  # type: ignore[attr-defined]
        assert proposal.status is ProposalStatus.REJECTED_BY_RISK

    async def test_an_active_halt_refuses_live(
        self, db: object, candled_instrument: Instrument, approver: uuid.UUID, monkeypatch
    ) -> None:
        monkeypatch.setattr("app.risk.execution.get_settings", _live_settings_on)
        await _config(db)
        await _live_instrument(db, candled_instrument)
        await _go_live(db)
        await HaltService(db).activate(  # type: ignore[arg-type]
            HaltKind.KILL_SWITCH, "stop", scope=HaltScope.GLOBAL
        )
        proposal = await _approved_proposal(db, candled_instrument)

        fake = FakeLiveBroker()
        with pytest.raises(ExecutionError, match="risk halt"):
            await ExecutionService(db, broker=fake).execute_approved(  # type: ignore[arg-type]
                proposal, actor_user_id=approver
            )
        assert not fake.orders

    async def test_missing_live_instrument_refuses(
        self, db: object, candled_instrument: Instrument, approver: uuid.UUID, monkeypatch
    ) -> None:
        monkeypatch.setattr("app.risk.execution.get_settings", _live_settings_on)
        await _config(db)
        # No BrokerInstrument for the live venue.
        await _go_live(db)
        proposal = await _approved_proposal(db, candled_instrument)

        fake = FakeLiveBroker()
        with pytest.raises(ExecutionError, match="not available on the live broker"):
            await ExecutionService(db, broker=fake).execute_approved(  # type: ignore[arg-type]
                proposal, actor_user_id=approver
            )
        assert not fake.orders

    async def test_server_flag_off_refuses(
        self, db: object, candled_instrument: Instrument, approver: uuid.UUID, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            "app.risk.execution.get_settings",
            lambda: types.SimpleNamespace(live_trading_enabled=False),
        )
        await _config(db)
        await _live_instrument(db, candled_instrument)
        await _go_live(db)
        proposal = await _approved_proposal(db, candled_instrument)

        fake = FakeLiveBroker()
        with pytest.raises(ExecutionError, match="disabled on the server"):
            await ExecutionService(db, broker=fake).execute_approved(  # type: ignore[arg-type]
                proposal, actor_user_id=approver
            )
        assert not fake.orders
