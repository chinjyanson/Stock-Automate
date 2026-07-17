"""Trading 212 Public API adapter (§3).

Scope is deliberately narrow. Trading 212 tells us what the account holds and
accepts our orders; it is never asked for OHLCV history or scanner breadth (§4)
— its rate limits make that infeasible and its candles are not the series our
strategies are validated against.

Demo and live are separate classes over a shared client. They differ only in
`kind` and base URL, but the split is not cosmetic: `is_live` derives from the
class, so no configuration mistake can make a demo adapter place a real order
or vice versa, and the two credentials can never be confused for one another.

Rate limiting: Trading 212 publishes tight per-endpoint limits (instrument
metadata in particular is roughly one request per minute). We throttle
client-side rather than discovering limits via 429s, because a 429 on an order
submission is an ambiguous outcome we would rather never provoke.
"""

from __future__ import annotations

import asyncio
import base64
import time
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

import httpx
import structlog

from app.broker.base import Broker
from app.broker.types import (
    BrokerAccount,
    BrokerAmbiguousResponseError,
    BrokerAuthError,
    BrokerError,
    BrokerInstrument,
    BrokerOrder,
    BrokerOrderRejectedError,
    BrokerOrderRequest,
    BrokerPosition,
    BrokerRateLimitError,
    BrokerUnavailableError,
    ReconciliationDiscrepancy,
    ReconciliationResult,
)
from app.models.enums import BrokerKind, OrderSide, OrderStatus, OrderType

log = structlog.get_logger(__name__)


def _to_decimal(value: Any) -> Decimal | None:
    """Parse a broker numeric without going through float.

    Trading 212 returns JSON numbers, which Python's json module has already
    made floats. Converting via `str` recovers the shortest repr rather than
    the binary artefact, which is the closest we can get to the intended
    decimal from this side of the wire.
    """
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None


#: Trading 212 order status vocabulary → our own.
_ORDER_STATUS_MAP: dict[str, OrderStatus] = {
    "LOCAL": OrderStatus.PENDING,
    "UNCONFIRMED": OrderStatus.PENDING,
    "CONFIRMED": OrderStatus.SUBMITTED,
    "NEW": OrderStatus.SUBMITTED,
    "SUBMITTED": OrderStatus.SUBMITTED,
    "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
    "FILLED": OrderStatus.FILLED,
    "CANCELLING": OrderStatus.SUBMITTED,
    "CANCELLED": OrderStatus.CANCELLED,
    "REJECTED": OrderStatus.REJECTED,
    "REPLACING": OrderStatus.SUBMITTED,
    "REPLACED": OrderStatus.SUBMITTED,
    "EXPIRED": OrderStatus.EXPIRED,
}

#: Trading 212 instrument type vocabulary → our own.
_INSTRUMENT_KIND_MAP: dict[str, str] = {
    "STOCK": "stock",
    "ETF": "etf",
    "ETC": "etc",
    "REIT": "stock",
    "TRUST": "trust",
    "CRYPTOCURRENCY": "unknown",
    "INDEX": "unknown",
    "FOREX": "unknown",
    "FUTURES": "unknown",
    "WARRANT": "unknown",
    "CVR": "unknown",
    "CORPACT": "unknown",
}


class _RateLimiter:
    """Simple token-bucket-ish throttle over a shared minute window."""

    def __init__(self, max_per_minute: int) -> None:
        self._min_interval = 60.0 / max(1, max_per_minute)
        self._last_call = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            wait = self._min_interval - (now - self._last_call)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_call = time.monotonic()


class _Trading212Client:
    """Thin HTTP client. Knows about auth, throttling and error taxonomy."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        base_url: str,
        *,
        timeout_seconds: float = 20.0,
        max_requests_per_minute: int = 30,
    ) -> None:
        if not api_key or not api_secret:
            raise BrokerAuthError("Trading 212 API key and secret are both required")
        self._base_url = base_url.rstrip("/")
        self._limiter = _RateLimiter(max_requests_per_minute)
        # Trading 212 authenticates with HTTP Basic: the API key is the
        # username, the API secret the password. `base64(key:secret)` with no
        # trailing newline (b64encode over bytes never adds one), sent in the
        # Authorization header — never a query string, which ends up in proxy
        # logs. The older scheme of sending the raw key alone is no longer
        # accepted, which is why a single-value key returns 401.
        token = base64.b64encode(f"{api_key}:{api_secret}".encode()).decode("ascii")
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(timeout_seconds),
            headers={"Authorization": f"Basic {token}", "Accept": "application/json"},
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        idempotent: bool,
    ) -> Any:
        """Perform one request.

        `idempotent` decides what a timeout means. For reads, a timeout is a
        retryable nuisance. For an order submission it is a genuine unknown —
        the order may be live — so it becomes an ambiguity the caller must
        resolve by reconciliation rather than by retrying.
        """
        await self._limiter.acquire()
        try:
            response = await self._client.request(method, path, json=json_body, params=params)
        except httpx.TimeoutException as exc:
            if idempotent:
                raise BrokerUnavailableError(f"Trading 212 timed out on {method} {path}") from exc
            raise BrokerAmbiguousResponseError(
                f"Trading 212 timed out on non-idempotent {method} {path}; "
                "order state is unknown and must be reconciled"
            ) from exc
        except httpx.HTTPError as exc:
            # A transport error on a write is equally unknown: the request may
            # have been received before the connection died.
            if idempotent:
                raise BrokerUnavailableError(f"Trading 212 transport error: {exc}") from exc
            raise BrokerAmbiguousResponseError(
                f"Trading 212 transport error on non-idempotent {method} {path}: {exc}"
            ) from exc

        return self._handle_response(response, method=method, path=path, idempotent=idempotent)

    def _handle_response(
        self, response: httpx.Response, *, method: str, path: str, idempotent: bool
    ) -> Any:
        status = response.status_code

        if status == 401 or status == 403:
            raise BrokerAuthError(
                f"Trading 212 rejected credentials on {method} {path} (HTTP {status})"
            )
        if status == 429:
            retry_after = response.headers.get("Retry-After")
            raise BrokerRateLimitError(
                f"Trading 212 rate limit hit on {method} {path}",
                retry_after_seconds=float(retry_after) if retry_after else None,
            )
        if status in (400, 422):
            # A validated refusal: nothing was placed, so this is safe.
            raise BrokerOrderRejectedError(
                f"Trading 212 rejected {method} {path}: {self._safe_body(response)}"
            )
        if status >= 500:
            if idempotent:
                raise BrokerUnavailableError(f"Trading 212 {status} on {method} {path}")
            raise BrokerAmbiguousResponseError(
                f"Trading 212 {status} on non-idempotent {method} {path}; state unknown"
            )
        if status >= 400:
            raise BrokerError(
                f"Trading 212 {status} on {method} {path}: {self._safe_body(response)}"
            )

        if not response.content:
            return None
        try:
            return response.json()
        except ValueError as exc:
            if idempotent:
                raise BrokerError(f"Trading 212 returned non-JSON on {method} {path}") from exc
            raise BrokerAmbiguousResponseError(
                f"Trading 212 returned unparseable body on {method} {path}; state unknown"
            ) from exc

    @staticmethod
    def _safe_body(response: httpx.Response) -> str:
        """Truncated body for error messages.

        Bounded because broker errors occasionally echo request context, and an
        unbounded copy into an exception message is how secrets reach logs.
        """
        text = response.text or ""
        return text[:300]


class _Trading212Broker(Broker):
    """Shared implementation. Instantiate `Trading212DemoBroker`/`...LiveBroker`."""

    kind: BrokerKind

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        base_url: str,
        *,
        timeout_seconds: float = 20.0,
        max_requests_per_minute: int = 30,
    ) -> None:
        self._client = _Trading212Client(
            api_key,
            api_secret,
            base_url,
            timeout_seconds=timeout_seconds,
            max_requests_per_minute=max_requests_per_minute,
        )
        self._log = log.bind(broker=str(self.kind))

    async def close(self) -> None:
        await self._client.close()

    # -- Reads --------------------------------------------------------------

    async def sync_instruments(self) -> list[BrokerInstrument]:
        payload = await self._client.request("GET", "/equity/metadata/instruments", idempotent=True)
        if not isinstance(payload, list):
            raise BrokerError("Trading 212 instrument metadata was not a list")
        instruments = [self._parse_instrument(row) for row in payload]
        self._log.info("trading212.instruments_synced", count=len(instruments))
        return instruments

    async def get_account(self) -> BrokerAccount:
        cash, info = await asyncio.gather(
            self._client.request("GET", "/equity/account/cash", idempotent=True),
            self._client.request("GET", "/equity/account/info", idempotent=True),
        )
        currency = (info or {}).get("currencyCode") or "GBP"
        account_id = str((info or {}).get("id") or "unknown")
        return BrokerAccount(
            account_id=account_id,
            currency=currency,
            cash=_to_decimal((cash or {}).get("free")) or Decimal(0),
            total=_to_decimal((cash or {}).get("total")) or Decimal(0),
            free_for_trading=_to_decimal((cash or {}).get("freeForStocks"))
            or _to_decimal((cash or {}).get("free"))
            or Decimal(0),
            invested=_to_decimal((cash or {}).get("invested")),
            result=_to_decimal((cash or {}).get("result")),
            blocked=_to_decimal((cash or {}).get("blocked")),
            retrieved_at=datetime.now(UTC),
        )

    async def get_positions(self) -> list[BrokerPosition]:
        payload = await self._client.request("GET", "/equity/portfolio", idempotent=True)
        if not isinstance(payload, list):
            return []
        return [self._parse_position(row) for row in payload]

    async def get_pending_orders(self) -> list[BrokerOrder]:
        payload = await self._client.request("GET", "/equity/orders", idempotent=True)
        if not isinstance(payload, list):
            return []
        return [self._parse_order(row) for row in payload]

    async def get_order_history(self) -> list[BrokerOrder]:
        payload = await self._client.request(
            "GET", "/equity/history/orders", params={"limit": 50}, idempotent=True
        )
        items = payload.get("items", []) if isinstance(payload, dict) else []
        return [self._parse_order(row) for row in items]

    # -- Writes -------------------------------------------------------------

    async def place_order(self, request: BrokerOrderRequest) -> BrokerOrder:
        """Submit exactly once. Never retried here — see `Broker.place_order`."""
        request.validate()

        path, body = self._build_order_payload(request)
        self._log.info(
            "trading212.order_submitting",
            ticker=request.broker_ticker,
            side=str(request.side),
            order_type=str(request.order_type),
            client_reference=request.client_reference,
        )
        payload = await self._client.request("POST", path, json_body=body, idempotent=False)
        if not isinstance(payload, dict):
            # We got a 2xx with an unusable body: the order probably exists.
            raise BrokerAmbiguousResponseError(
                "Trading 212 accepted the order but returned an unreadable body; "
                "state must be reconciled"
            )
        order = self._parse_order(payload)
        self._log.info(
            "trading212.order_submitted",
            broker_order_id=order.broker_order_id,
            status=str(order.status),
        )
        return order

    async def cancel_order(self, broker_order_id: str) -> None:
        try:
            await self._client.request(
                "DELETE", f"/equity/orders/{broker_order_id}", idempotent=True
            )
        except BrokerOrderRejectedError:
            # Already terminal. Cancelling a filled order is a no-op, not an error.
            self._log.info("trading212.cancel_noop", broker_order_id=broker_order_id)

    async def reconcile(self) -> ReconciliationResult:
        """Report broker-side truth.

        This adapter only surfaces what the broker says; comparison against
        local intents happens in the execution layer, which owns those records.
        Ambiguity here is reported, never silently resolved.
        """
        positions, pending = await asyncio.gather(self.get_positions(), self.get_pending_orders())
        discrepancies: list[ReconciliationDiscrepancy] = []
        for position in positions:
            if position.current_price is None:
                discrepancies.append(
                    ReconciliationDiscrepancy(
                        kind="missing_price",
                        broker_ticker=position.broker_ticker,
                        detail="Broker reported a position without a current price; "
                        "unrealised P&L and stop distance cannot be computed",
                    )
                )
        return ReconciliationResult(
            broker=self.kind,
            reconciled_at=datetime.now(UTC),
            positions_checked=len(positions),
            orders_checked=len(pending),
            discrepancies=discrepancies,
        )

    # -- Parsing ------------------------------------------------------------

    def _build_order_payload(self, request: BrokerOrderRequest) -> tuple[str, dict[str, Any]]:
        # Trading 212 encodes side as the sign of the quantity.
        signed_quantity = request.quantity if request.side is OrderSide.BUY else -request.quantity
        body: dict[str, Any] = {
            "ticker": request.broker_ticker,
            "quantity": float(signed_quantity),
        }
        match request.order_type:
            case OrderType.MARKET:
                return "/equity/orders/market", body
            case OrderType.LIMIT:
                body["limitPrice"] = float(request.limit_price or 0)
                body["timeValidity"] = request.time_in_force
                return "/equity/orders/limit", body
            case OrderType.STOP:
                body["stopPrice"] = float(request.stop_price or 0)
                body["timeValidity"] = request.time_in_force
                return "/equity/orders/stop", body
            case OrderType.STOP_LIMIT:
                body["limitPrice"] = float(request.limit_price or 0)
                body["stopPrice"] = float(request.stop_price or 0)
                body["timeValidity"] = request.time_in_force
                return "/equity/orders/stop_limit", body
        raise ValueError(f"Unsupported order type: {request.order_type}")

    def _parse_instrument(self, row: dict[str, Any]) -> BrokerInstrument:
        raw_type = str(row.get("type") or "").upper()
        return BrokerInstrument(
            broker_ticker=str(row.get("ticker") or ""),
            name=str(row.get("name") or row.get("shortName") or ""),
            isin=row.get("isin"),
            currency=row.get("currencyCode"),
            # Trading 212 reports its own venue codes, not MICs. Translating
            # them is the mapping layer's job (§5) — guessing here would create
            # false identity matches.
            exchange_mic=None,
            kind=_INSTRUMENT_KIND_MAP.get(raw_type, "unknown"),
            is_currently_available=True,
            min_quantity=_to_decimal(row.get("minTradeQuantity")),
            quantity_step=_to_decimal(row.get("minTradeQuantity")),
            supports_fractional=(_to_decimal(row.get("minTradeQuantity")) or Decimal(1)) < 1,
            raw=row,
        )

    def _parse_position(self, row: dict[str, Any]) -> BrokerPosition:
        return BrokerPosition(
            broker_ticker=str(row.get("ticker") or ""),
            quantity=_to_decimal(row.get("quantity")) or Decimal(0),
            average_price=_to_decimal(row.get("averagePrice")) or Decimal(0),
            current_price=_to_decimal(row.get("currentPrice")),
            unrealised_pnl=_to_decimal(row.get("ppl")),
            initial_fill_at=_parse_dt(row.get("initialFillDate")),
            max_buy=_to_decimal(row.get("maxBuy")),
            max_sell=_to_decimal(row.get("maxSell")),
            raw=row,
        )

    def _parse_order(self, row: dict[str, Any]) -> BrokerOrder:
        quantity = _to_decimal(row.get("quantity")) or Decimal(0)
        # Sign carries the side; magnitude is the quantity.
        side = OrderSide.BUY if quantity >= 0 else OrderSide.SELL
        status_raw = str(row.get("status") or "").upper()
        return BrokerOrder(
            broker_order_id=str(row.get("id") or ""),
            broker_ticker=str(row.get("ticker") or ""),
            side=side,
            order_type=_parse_order_type(row),
            quantity=abs(quantity),
            status=_ORDER_STATUS_MAP.get(status_raw, OrderStatus.UNKNOWN),
            filled_quantity=abs(_to_decimal(row.get("filledQuantity")) or Decimal(0)),
            average_fill_price=_to_decimal(row.get("fillPrice")),
            limit_price=_to_decimal(row.get("limitPrice")),
            stop_price=_to_decimal(row.get("stopPrice")),
            created_at=_parse_dt(row.get("creationTime")),
            updated_at=_parse_dt(row.get("modificationTime")),
            raw=row,
        )


def _parse_order_type(row: dict[str, Any]) -> OrderType:
    raw = str(row.get("type") or "").upper()
    match raw:
        case "LIMIT":
            return OrderType.LIMIT
        case "STOP":
            return OrderType.STOP
        case "STOP_LIMIT":
            return OrderType.STOP_LIMIT
        case _:
            return OrderType.MARKET


def _parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    # Naive timestamps from the broker are documented as UTC; tagging them
    # explicitly keeps every comparison in this codebase timezone-aware.
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


class Trading212DemoBroker(_Trading212Broker):
    """Practice account. Orders are simulated by Trading 212, not by us."""

    kind = BrokerKind.TRADING212_DEMO


class Trading212LiveBroker(_Trading212Broker):
    """Real-money account.

    Reaching this class is not sufficient to trade: the execution layer
    additionally requires LIVE_TRADING_ENABLED, an unexpired arming session and
    a clean risk state before it will call `place_order` (§7, §14).
    """

    kind = BrokerKind.TRADING212_LIVE
