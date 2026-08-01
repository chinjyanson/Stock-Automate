"""Strategy live routing against real PostgreSQL.

With the venue set to live: autonomous **enabled** lets a strategy fill straight
through to the (fake) live broker; autonomous **disabled** makes it only
*propose*, leaving a human to approve each order. No real live broker is ever
constructed — the venue is injected.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from sqlalchemy import select

from app.broker.base import Broker
from app.broker.types import (
    BrokerAccount,
    BrokerOrder,
    BrokerOrderRequest,
    BrokerPosition,
    ReconciliationResult,
)
from app.data.store import CandleStore
from app.data.types import Candle as CandleDTO
from app.models.enums import (
    BrokerKind,
    InstrumentKind,
    Interval,
    OrderStatus,
    PriceUnit,
    ProviderKind,
    StrategyDecisionOutcome,
    StrategyKind,
)
from app.models.instrument import BrokerInstrument, Exchange, Instrument
from app.models.risk import RiskConfiguration
from app.models.strategy import StrategyConfiguration, StrategyDecision
from app.services.system_settings import (
    AUTONOMOUS_LIVE_KEY,
    TRADING_LIVE_MODE_KEY,
    set_bool_setting,
)
from app.strategies.engine import StrategyEngine

pytestmark = pytest.mark.asyncio

_LIVE_TICKER = "RISKY_LIVE"


class FakeLiveBroker(Broker):
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


#: A stable, oscillating base then a sharp sell-off — the dislocation the
#: strategy exists to catch, and the only shape that produces a guaranteed
#: entry. A *gradual* decline does not work: the bands follow a trend down, so
#: price never breaks its own lower band.
_STABLE_BASE = [100 + (2 if i % 2 else -2) for i in range(45)]
_SELLOFF = [*_STABLE_BASE, 95.0, 90.0, 86.0]


async def _entry_instrument(db: object) -> Instrument:
    exchange = (
        await db.execute(select(Exchange).where(Exchange.mic == "XNAS"))  # type: ignore[attr-defined]
    ).scalar_one_or_none()
    if exchange is None:
        exchange = Exchange(mic="XNAS", name="Nasdaq", country="US", timezone="America/New_York")
        db.add(exchange)  # type: ignore[attr-defined]
        await db.flush()  # type: ignore[attr-defined]
    instrument = Instrument(
        id=uuid.uuid4(),
        exchange_id=exchange.id,
        exchange_ticker="RISKY",
        name="Risky Inc.",
        kind=InstrumentKind.STOCK,
        currency="USD",
        price_unit=PriceUnit.USD,
    )
    db.add(instrument)  # type: ignore[attr-defined]
    await db.flush()  # type: ignore[attr-defined]

    now = datetime.now(UTC).replace(second=0, microsecond=0)
    closes = _SELLOFF
    candles = [
        CandleDTO(
            symbol="RISKY",
            interval=Interval.D1,
            timestamp=now - timedelta(days=len(closes) - 1 - i),
            open=Decimal(str(c)),
            high=Decimal(str(c)) * Decimal("1.01"),
            low=Decimal(str(c)) * Decimal("0.99"),
            close=Decimal(str(c)),
            volume=Decimal("100000"),
            currency="USD",
            price_unit=PriceUnit.USD,
            provider=ProviderKind.MOCK,
            is_closed=True,
        )
        for i, c in enumerate(closes)
    ]
    await CandleStore(db).upsert_candles(instrument.id, candles)  # type: ignore[arg-type]
    return instrument


async def _set(db: object, key: str, value: bool) -> None:
    await set_bool_setting(
        db,  # type: ignore[arg-type]
        key,
        value,
        description="x",
        is_sensitive=True,
        user_id=None,
    )
    await db.flush()  # type: ignore[attr-defined]


async def _entry_config(db: object, instrument: Instrument) -> StrategyConfiguration:
    db.add(RiskConfiguration(name="default", is_active=True))  # type: ignore[attr-defined]
    config = StrategyConfiguration(
        kind=StrategyKind.MEAN_REVERSION,
        name="meanrev",
        is_active=True,
        interval=Interval.D1,
        auto_execute=True,
        # RSI is pinned at 40 rather than taken from the configured default so
        # that retuning the strategy cannot silently stop these tests producing
        # the entry whose *routing* is what they are about.
        params={
            "bb_period": 20,
            "bb_std": 2.0,
            "rsi_period": 14,
            "rsi_oversold": 40.0,
            "atr_period": 14,
            "min_atr_pct": 0.02,
        },
        universe={"instrument_ids": [str(instrument.id)]},
    )
    db.add(config)  # type: ignore[attr-defined]
    await db.flush()  # type: ignore[attr-defined]
    return config


class TestAutonomous:
    async def test_autonomous_routes_entries_live(
        self, db: object, approver: uuid.UUID, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            "app.risk.execution.get_settings",
            lambda: __import__("types").SimpleNamespace(live_trading_enabled=True),
        )
        instrument = await _entry_instrument(db)
        db.add(  # type: ignore[attr-defined]
            BrokerInstrument(
                instrument_id=instrument.id,
                broker=BrokerKind.TRADING212_LIVE,
                broker_ticker=_LIVE_TICKER,
                is_currently_available=True,
            )
        )
        config = await _entry_config(db, instrument)
        # Live venue + autonomous permitted: strategies fill without a human.
        await _set(db, TRADING_LIVE_MODE_KEY, True)
        await _set(db, AUTONOMOUS_LIVE_KEY, True)

        fake = FakeLiveBroker()
        summary = await StrategyEngine(db, broker=fake).run(config)  # type: ignore[arg-type]
        await db.commit()  # type: ignore[attr-defined]

        assert summary.executed == 1
        assert fake.orders and fake.orders[0].broker_ticker == _LIVE_TICKER


class TestApprovalRequired:
    async def test_live_without_autonomous_only_proposes(
        self, db: object, approver: uuid.UUID
    ) -> None:
        instrument = await _entry_instrument(db)
        config = await _entry_config(db, instrument)
        # Live venue, autonomous NOT permitted: strategies may only propose.
        await _set(db, TRADING_LIVE_MODE_KEY, True)

        fake = FakeLiveBroker()
        summary = await StrategyEngine(db, broker=fake).run(config)  # type: ignore[arg-type]
        await db.commit()  # type: ignore[attr-defined]

        assert summary.executed == 0
        assert summary.proposals == 1
        # Nothing reached the venue — a human has to approve first.
        assert not fake.orders
        decision = (
            (
                await db.execute(select(StrategyDecision))  # type: ignore[attr-defined]
            )
            .scalars()
            .one()
        )
        assert decision.outcome is StrategyDecisionOutcome.PROPOSED
        assert decision.proposal_id is not None
