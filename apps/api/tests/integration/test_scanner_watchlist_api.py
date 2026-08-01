"""Watchlist search-and-scan, and how results are listed, over the real HTTP app.

Two behaviours are pinned here.

**Listing.** `/scanner/results` without a run id returns each instrument's most
recent result rather than the most recent *run's* results. A run covers a
rotating slice of the catalogue, so "the latest run" is whatever the sweep
reached last night — and a one-off rescan of a single stock would otherwise
reduce the whole table to one row.

**Refreshing.** Pinning a stock from the search box is followed by an immediate
fetch-and-rescore, because the rotation would not reach it for days. That scan
is marked ad hoc so it records a score without being mistaken for the nightly
ranking the strategy universe is drawn from.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import httpx
import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

import app.api.routes.scanner as scanner_routes
from app.auth.dependencies import AuthContext, get_auth_context, require_csrf
from app.data.mock_provider import MockMarketDataProvider
from app.data.types import Candle as CandleDTO
from app.data.yfinance_provider import YFinanceProvider
from app.main import app
from app.models.enums import (
    InstrumentKind,
    Interval,
    LifecycleState,
    PriceUnit,
    ProviderKind,
)
from app.models.instrument import Exchange, Instrument, MarketDataMapping, WatchlistInstrument
from app.models.scanner import ScannerResult, ScannerRun
from app.models.user import User
from app.scanner.engine import ScannerEngine
from app.services.ingestion import IngestionService

pytestmark = pytest.mark.asyncio


class _FakeYFinance(YFinanceProvider):
    """A yfinance provider that answers from memory.

    Subclasses the real class rather than duck-typing it because the refresh
    endpoint narrows to the concrete provider — the batched download is not part
    of the generic interface.
    """

    def __init__(self, closes: list[float] | None = None) -> None:
        super().__init__(max_retries=1)
        self._closes = closes if closes is not None else [100.0 + i * 0.1 for i in range(300)]
        self.requested: list[str] = []

    async def get_batch_daily_candles(  # type: ignore[override]
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
        *,
        unit_by_symbol: dict[str, PriceUnit] | None = None,
    ) -> dict[str, list[CandleDTO]]:
        self.requested.extend(symbols)
        now = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        return {
            symbol: [
                CandleDTO(
                    symbol=symbol,
                    interval=Interval.D1,
                    timestamp=now - timedelta(days=len(self._closes) - 1 - i),
                    open=Decimal(str(c)),
                    high=Decimal(str(c)) * Decimal("1.01"),
                    low=Decimal(str(c)) * Decimal("0.99"),
                    close=Decimal(str(c)),
                    volume=Decimal("500000"),
                    currency="USD",
                    price_unit=PriceUnit.USD,
                    provider=ProviderKind.YFINANCE,
                    is_closed=True,
                )
                for i, c in enumerate(self._closes)
            ]
            for symbol in symbols
        }

    async def close(self) -> None:
        return None


async def _exchange(db: AsyncSession, mic: str = "XNAS") -> Exchange:
    existing = (await db.execute(select(Exchange).where(Exchange.mic == mic))).scalar_one_or_none()
    if existing is not None:
        return existing
    exchange = Exchange(mic=mic, name=mic, country="US", timezone="America/New_York")
    db.add(exchange)
    await db.flush()
    return exchange


async def _instrument(
    db: AsyncSession, ticker: str, *, mic: str = "XNAS", with_candles: bool = True
) -> Instrument:
    exchange = await _exchange(db, mic)
    instrument = Instrument(
        id=uuid.uuid4(),
        exchange_id=exchange.id,
        exchange_ticker=ticker,
        name=f"{ticker} Inc.",
        kind=InstrumentKind.STOCK,
        currency="USD",
        price_unit=PriceUnit.USD,
        lifecycle_state=LifecycleState.DISCOVERED,
        is_scanner_eligible=True,
    )
    db.add(instrument)
    await db.flush()
    if with_candles:
        db.add(
            MarketDataMapping(
                instrument_id=instrument.id,
                provider=ProviderKind.MOCK,
                provider_symbol=ticker,
                is_signal_source=True,
                confirmed_by_user=True,
            )
        )
        await db.flush()
        await IngestionService(db).ingest_daily(
            instrument, MockMarketDataProvider(), backfill_days=400
        )
    await db.commit()
    return instrument


@pytest_asyncio.fixture
async def authed_client(
    client: httpx.AsyncClient, db: AsyncSession
) -> AsyncIterator[httpx.AsyncClient]:
    """The app client with a real user attached and CSRF waived."""
    user = User(
        id=uuid.uuid4(),
        email=f"scanner-{uuid.uuid4().hex[:8]}@example.com",
        password_hash="x",
        is_admin=True,
        is_active=True,
    )
    db.add(user)
    await db.commit()

    class _Session:
        pass

    app.dependency_overrides[get_auth_context] = lambda: AuthContext(
        user=user,
        session=_Session(),  # type: ignore[arg-type]
    )
    app.dependency_overrides[require_csrf] = lambda: None
    yield client
    app.dependency_overrides.pop(get_auth_context, None)
    app.dependency_overrides.pop(require_csrf, None)


class TestResultsListing:
    async def test_listing_shows_each_instrument_latest_not_the_latest_run(
        self, authed_client: httpx.AsyncClient, db: AsyncSession
    ) -> None:
        """A rotating scan means a stock scored last night is still a current row.

        The bug this pins: defaulting to the latest run made every instrument
        outside that slice vanish from the table, and a single-instrument rescan
        emptied it entirely.
        """
        old = await _instrument(db, "OLDONE")
        fresh = await _instrument(db, "FRESHONE")

        # Two runs, each covering a different slice — exactly what a rotation does.
        await ScannerEngine(db).run([old], selection_reason="sweep one")
        await db.commit()
        await ScannerEngine(db).run([fresh], selection_reason="sweep two")
        await db.commit()

        response = await authed_client.get("/scanner/results?limit=100")
        assert response.status_code == 200
        returned = {row["instrument_id"] for row in response.json()}
        assert str(old.id) in returned, "a stock the latest sweep missed must not disappear"
        assert str(fresh.id) in returned

    async def test_rescanning_replaces_the_earlier_row_rather_than_duplicating_it(
        self, authed_client: httpx.AsyncClient, db: AsyncSession
    ) -> None:
        instrument = await _instrument(db, "TWICE")
        await ScannerEngine(db).run([instrument], selection_reason="first")
        await db.commit()
        await ScannerEngine(db).run([instrument], selection_reason="second")
        await db.commit()

        rows = (await authed_client.get("/scanner/results?limit=100")).json()
        matching = [r for r in rows if r["instrument_id"] == str(instrument.id)]
        assert len(matching) == 1, "one row per instrument, the most recent one"

    async def test_a_named_run_still_returns_exactly_that_run(
        self, authed_client: httpx.AsyncClient, db: AsyncSession
    ) -> None:
        first = await _instrument(db, "AAA")
        second = await _instrument(db, "BBB")
        summary = await ScannerEngine(db).run([first], selection_reason="first")
        await db.commit()
        await ScannerEngine(db).run([second], selection_reason="second")
        await db.commit()

        rows = (await authed_client.get(f"/scanner/results?run_id={summary.run_id}")).json()
        assert [r["instrument_id"] for r in rows] == [str(first.id)]

    async def test_every_row_carries_when_it_was_scored(
        self, authed_client: httpx.AsyncClient, db: AsyncSession
    ) -> None:
        """Mixed-age rows are only honest if each says how old it is."""
        instrument = await _instrument(db, "DATED")
        await ScannerEngine(db).run([instrument], selection_reason="sweep")
        await db.commit()

        rows = (await authed_client.get("/scanner/results?limit=10")).json()
        assert rows and rows[0]["scanned_at"]


class TestRefreshInstrument:
    async def test_refresh_fetches_candles_and_returns_a_fresh_score(
        self,
        authed_client: httpx.AsyncClient,
        db: AsyncSession,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        instrument = await _instrument(db, "PINNED", with_candles=False)
        fake = _FakeYFinance()
        monkeypatch.setattr(scanner_routes, "resolve_provider", lambda kind: fake)

        response = await authed_client.post(f"/scanner/instruments/{instrument.id}/refresh")

        assert response.status_code == 200
        body = response.json()
        assert fake.requested == ["PINNED"], "the venue's symbol must be requested"
        assert body["candles_written"] > 0
        assert body["result"] is not None
        assert body["reason"] is None
        assert body["result"]["instrument_name"] == "PINNED Inc."

    async def test_refresh_does_not_become_the_ranking_run(
        self,
        authed_client: httpx.AsyncClient,
        db: AsyncSession,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The universe sync reads the latest *ranking* run, and must not see this.

        Without the ad-hoc flag, rescanning one stock would make its run the
        latest, and the mean-reversion universe would silently collapse to the
        single name someone happened to look at.
        """
        ranked = await _instrument(db, "RANKED")
        rotation = await ScannerEngine(db).run([ranked], selection_reason="rotation")
        await db.commit()

        pinned = await _instrument(db, "ADHOC", with_candles=False)
        monkeypatch.setattr(scanner_routes, "resolve_provider", lambda kind: _FakeYFinance())
        await authed_client.post(f"/scanner/instruments/{pinned.id}/refresh")

        latest_ranking = (
            await db.execute(
                select(ScannerRun.id)
                .where(ScannerRun.is_ad_hoc.is_(False))
                .order_by(ScannerRun.started_at.desc())
                .limit(1)
            )
        ).scalar_one()
        assert latest_ranking == rotation.run_id

        # …while the score itself is recorded and visible like any other.
        rows = (await authed_client.get("/scanner/results?limit=100")).json()
        assert str(pinned.id) in {r["instrument_id"] for r in rows}

    async def test_an_unmappable_venue_says_so_instead_of_returning_no_score(
        self,
        authed_client: httpx.AsyncClient,
        db: AsyncSession,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A null result with no explanation is indistinguishable from a bug."""
        instrument = await _instrument(db, "OBSCURE", mic="XZZZ", with_candles=False)
        fake = _FakeYFinance()
        monkeypatch.setattr(scanner_routes, "resolve_provider", lambda kind: fake)

        body = (await authed_client.post(f"/scanner/instruments/{instrument.id}/refresh")).json()

        assert body["result"] is None
        assert "venue" in body["reason"]
        assert fake.requested == [], "an unsupported venue must not reach the provider"

    async def test_too_little_history_is_reported_not_silently_unscored(
        self,
        authed_client: httpx.AsyncClient,
        db: AsyncSession,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        instrument = await _instrument(db, "THIN", with_candles=False)
        # Fewer bars than the scanner's floor, so the scan produces no result.
        monkeypatch.setattr(
            scanner_routes, "resolve_provider", lambda kind: _FakeYFinance([100.0, 101.0, 102.0])
        )

        body = (await authed_client.post(f"/scanner/instruments/{instrument.id}/refresh")).json()

        assert body["result"] is None
        assert "history" in body["reason"]

    async def test_refresh_of_an_unknown_instrument_is_a_404(
        self, authed_client: httpx.AsyncClient
    ) -> None:
        response = await authed_client.post(f"/scanner/instruments/{uuid.uuid4()}/refresh")
        assert response.status_code == 404


class TestPinThenRefreshFlow:
    async def test_a_searched_stock_can_be_pinned_and_scored(
        self,
        authed_client: httpx.AsyncClient,
        db: AsyncSession,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The whole search-box flow: find it, pin it, have it scored now."""
        instrument = await _instrument(db, "SEARCHME", with_candles=False)

        found = (await authed_client.get("/instruments?search=SEARCHME")).json()
        assert [i["id"] for i in found["items"]] == [str(instrument.id)]

        pinned = await authed_client.post(
            "/scanner/watchlist", json={"instrument_id": str(instrument.id)}
        )
        assert pinned.status_code == 201
        # Pinned before any market data exists — the pin is recorded regardless.
        assert pinned.json()["is_scannable"] is False

        monkeypatch.setattr(scanner_routes, "resolve_provider", lambda kind: _FakeYFinance())
        refreshed = await authed_client.post(f"/scanner/instruments/{instrument.id}/refresh")
        assert refreshed.json()["result"] is not None

        # The pin persists, and now has the history to be honoured next scan.
        listed = (await authed_client.get("/scanner/watchlist")).json()
        assert [e["instrument_id"] for e in listed] == [str(instrument.id)]
        assert listed[0]["is_scannable"] is True

        assert (
            await db.execute(
                select(WatchlistInstrument).where(
                    WatchlistInstrument.instrument_id == instrument.id
                )
            )
        ).scalar_one_or_none() is not None

        scored = (
            (
                await db.execute(
                    select(ScannerResult).where(ScannerResult.instrument_id == instrument.id)
                )
            )
            .scalars()
            .all()
        )
        assert len(scored) == 1
