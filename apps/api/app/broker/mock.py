"""Deterministic mock broker.

Selected automatically when no Trading 212 credentials are configured, so the
whole local workflow — instrument sync, mapping, backfill, scan, paper trade —
runs on a fresh clone with an empty `.env` (§23).

Determinism is the point. Fixtures are fixed, ordering is stable, and no clock
or RNG influences the catalogue, so tests can assert on it. The account balance
does move as orders fill, because code that never sees cash decrease is code
that has not been tested.

The fixture ISINs and tickers are real, because instrument *identity* is what
the mapping layer under test is resolving; the prices are invented and labelled
as such.
"""

from __future__ import annotations

import uuid
from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from app.broker.base import Broker
from app.broker.types import (
    BrokerAccount,
    BrokerInstrument,
    BrokerOrder,
    BrokerOrderRejectedError,
    BrokerOrderRequest,
    BrokerPosition,
    ReconciliationResult,
)
from app.models.enums import BrokerKind, OrderSide, OrderStatus, OrderType

#: Invented reference prices, in the instrument's quoted unit (GBX for LSE).
_FIXTURE_PRICES: dict[str, Decimal] = {
    "VUAGl_EQ": Decimal("8750.00"),
    "VUSAl_EQ": Decimal("8420.00"),
    "SGLNl_EQ": Decimal("4980.00"),
    "CRUDl_EQ": Decimal("2310.00"),
    "AAPL_US_EQ": Decimal("232.15"),
    "MSFT_US_EQ": Decimal("418.60"),
    "SPY_US_EQ": Decimal("598.40"),
}

_FIXTURE_INSTRUMENTS: list[dict[str, Any]] = [
    {
        "ticker": "VUAGl_EQ",
        "name": "Vanguard S&P 500 UCITS ETF (Acc)",
        "isin": "IE00BFMXXD54",
        "currencyCode": "GBX",
        "type": "ETF",
        "minTradeQuantity": "0.01",
    },
    {
        "ticker": "VUSAl_EQ",
        "name": "Vanguard S&P 500 UCITS ETF (Dist)",
        "isin": "IE00B3XXRP09",
        "currencyCode": "GBX",
        "type": "ETF",
        "minTradeQuantity": "0.01",
    },
    {
        "ticker": "SGLNl_EQ",
        "name": "iShares Physical Gold ETC",
        "isin": "IE00B4ND3602",
        "currencyCode": "GBX",
        "type": "ETC",
        "minTradeQuantity": "0.01",
    },
    {
        "ticker": "CRUDl_EQ",
        "name": "WisdomTree WTI Crude Oil ETC",
        "isin": "GB00B15KY765",
        "currencyCode": "GBX",
        "type": "ETC",
        "minTradeQuantity": "0.1",
    },
    {
        "ticker": "AAPL_US_EQ",
        "name": "Apple Inc.",
        "isin": "US0378331005",
        "currencyCode": "USD",
        "type": "STOCK",
        "minTradeQuantity": "0.01",
    },
    {
        "ticker": "MSFT_US_EQ",
        "name": "Microsoft Corporation",
        "isin": "US5949181045",
        "currencyCode": "USD",
        "type": "STOCK",
        "minTradeQuantity": "0.01",
    },
    {
        "ticker": "SPY_US_EQ",
        "name": "SPDR S&P 500 ETF Trust",
        "isin": "US78462F1030",
        "currencyCode": "USD",
        "type": "ETF",
        "minTradeQuantity": "0.01",
    },
]

_INSTRUMENT_KIND_MAP = {"STOCK": "stock", "ETF": "etf", "ETC": "etc"}


class MockBroker(Broker):
    """In-memory broker over fixed fixtures. Never touches the network."""

    kind = BrokerKind.MOCK

    def __init__(self, *, starting_cash: Decimal = Decimal("10000.00")) -> None:
        self._cash = starting_cash
        self._starting_cash = starting_cash
        self._positions: dict[str, tuple[Decimal, Decimal]] = {}  # ticker -> (qty, avg_price)
        self._orders: list[BrokerOrder] = []
        #: Set by tests to force a failure on the next submission.
        self.fail_next_order_with: Exception | None = None

    async def sync_instruments(self) -> list[BrokerInstrument]:
        return [
            BrokerInstrument(
                broker_ticker=row["ticker"],
                name=row["name"],
                isin=row["isin"],
                currency=row["currencyCode"],
                exchange_mic=None,
                kind=_INSTRUMENT_KIND_MAP.get(row["type"], "unknown"),
                is_currently_available=True,
                min_quantity=Decimal(row["minTradeQuantity"]),
                quantity_step=Decimal(row["minTradeQuantity"]),
                supports_fractional=Decimal(row["minTradeQuantity"]) < 1,
                raw=row,
            )
            for row in _FIXTURE_INSTRUMENTS
        ]

    async def get_account(self) -> BrokerAccount:
        invested = sum((qty * price for qty, price in self._positions.values()), start=Decimal(0))
        return BrokerAccount(
            account_id="MOCK-000000",
            currency="GBP",
            cash=self._cash,
            total=self._cash + invested,
            free_for_trading=self._cash,
            invested=invested,
            result=self._cash + invested - self._starting_cash,
            blocked=Decimal(0),
            retrieved_at=datetime.now(UTC),
        )

    async def get_positions(self) -> list[BrokerPosition]:
        return [
            BrokerPosition(
                broker_ticker=ticker,
                quantity=qty,
                average_price=avg,
                current_price=_FIXTURE_PRICES.get(ticker, avg),
                unrealised_pnl=(_FIXTURE_PRICES.get(ticker, avg) - avg) * qty,
            )
            for ticker, (qty, avg) in sorted(self._positions.items())
            if qty != 0
        ]

    async def get_pending_orders(self) -> list[BrokerOrder]:
        return [o for o in self._orders if not o.is_terminal]

    async def get_order_history(self) -> list[BrokerOrder]:
        return list(self._orders)

    async def place_order(self, request: BrokerOrderRequest) -> BrokerOrder:
        request.validate()

        if self.fail_next_order_with is not None:
            failure, self.fail_next_order_with = self.fail_next_order_with, None
            raise failure

        price = _FIXTURE_PRICES.get(request.broker_ticker)
        if price is None:
            raise BrokerOrderRejectedError(f"Unknown instrument: {request.broker_ticker}")

        fill_price = request.limit_price if request.order_type is OrderType.LIMIT else price
        assert fill_price is not None
        cost = fill_price * request.quantity

        if request.side is OrderSide.BUY:
            if cost > self._cash:
                raise BrokerOrderRejectedError(f"Insufficient cash: need {cost}, have {self._cash}")
            self._cash -= cost
            held_qty, held_avg = self._positions.get(
                request.broker_ticker, (Decimal(0), Decimal(0))
            )
            new_qty = held_qty + request.quantity
            # Weighted average cost; guards the divide when re-opening from flat.
            new_avg = ((held_qty * held_avg) + cost) / new_qty if new_qty else Decimal(0)
            self._positions[request.broker_ticker] = (new_qty, new_avg)
        else:
            held_qty, held_avg = self._positions.get(
                request.broker_ticker, (Decimal(0), Decimal(0))
            )
            if request.quantity > held_qty:
                raise BrokerOrderRejectedError(
                    f"Cannot sell {request.quantity}; position is {held_qty}"
                )
            self._cash += cost
            self._positions[request.broker_ticker] = (held_qty - request.quantity, held_avg)

        order = BrokerOrder(
            broker_order_id=str(uuid.uuid4()),
            broker_ticker=request.broker_ticker,
            side=request.side,
            order_type=request.order_type,
            quantity=request.quantity,
            status=OrderStatus.FILLED,
            filled_quantity=request.quantity,
            average_fill_price=fill_price,
            limit_price=request.limit_price,
            stop_price=request.stop_price,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
            client_reference=request.client_reference,
        )
        self._orders.append(order)
        return order

    async def cancel_order(self, broker_order_id: str) -> None:
        for index, order in enumerate(self._orders):
            if order.broker_order_id == broker_order_id and not order.is_terminal:
                self._orders[index] = replace(
                    order, status=OrderStatus.CANCELLED, updated_at=datetime.now(UTC)
                )
                return

    async def reconcile(self) -> ReconciliationResult:
        return ReconciliationResult(
            broker=self.kind,
            reconciled_at=datetime.now(UTC),
            positions_checked=len(self._positions),
            orders_checked=len(self._orders),
            discrepancies=[],
        )
