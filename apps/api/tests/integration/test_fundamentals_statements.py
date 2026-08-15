"""Cash-flow and balance-sheet ingestion, against real PostgreSQL.

These columns exist ahead of anything consuming them, deliberately: whether a
DCF is worth building at all is a question about free-tier *coverage*, and the
only way to answer it is to accumulate the data and look. So what is tested here
is that the pipeline carries the values end to end without quietly dropping or
rounding them, and that a provider with nothing to say produces nulls rather
than zeros — because a zero free cash flow is a claim about a business, and an
absent one is a claim about our data.
"""

from __future__ import annotations

import uuid
from decimal import Decimal

import pytest
from sqlalchemy import select

from app.data.types import Fundamentals
from app.models.enums import InstrumentKind, PriceUnit, ProviderKind
from app.models.instrument import Exchange, Instrument, MarketDataMapping
from app.models.market_data import FundamentalSnapshot
from app.services.enrichment import EnrichmentService

pytestmark = pytest.mark.asyncio

_FULL = {
    "free_cash_flow": Decimal("107721875456"),
    "operating_cash_flow": Decimal("146723995648"),
    "total_debt": Decimal("84343996416"),
    "total_cash": Decimal("62399000576"),
    "shares_outstanding": Decimal("14594180000"),
    "ebitda": Decimal("167959003136"),
    "enterprise_value": Decimal("4594739445760"),
    "book_value_per_share": Decimal("7.36"),
    # Ratio columns are Numeric(12, 6), so this is stated at the precision the
    # column actually keeps — see `test_a_long_ratio_is_rounded_not_rejected`.
    "return_on_equity": Decimal("1.487510"),
    "current_ratio": Decimal("1.003"),
}


class _StubProvider:
    """Stands in for YFinanceProvider's single `.info` call."""

    def __init__(self, **statement_fields: Decimal | None) -> None:
        self._fields = statement_fields

    async def get_profile(self, symbol: str) -> tuple[str | None, str | None, Fundamentals]:
        from datetime import UTC, datetime

        return (
            "Technology",
            "Consumer Electronics",
            Fundamentals(
                symbol=symbol,
                provider=ProviderKind.YFINANCE,
                as_of=datetime.now(UTC),
                currency="USD",
                trailing_pe=Decimal("30"),
                **self._fields,
            ),
        )


async def _mapped_instrument(db: object, ticker: str) -> Instrument:
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
        exchange_ticker=ticker,
        name=f"{ticker} Inc.",
        kind=InstrumentKind.STOCK,
        currency="USD",
        price_unit=PriceUnit.USD,
    )
    db.add(instrument)  # type: ignore[attr-defined]
    await db.flush()  # type: ignore[attr-defined]
    db.add(  # type: ignore[attr-defined]
        MarketDataMapping(
            instrument_id=instrument.id,
            provider=ProviderKind.YFINANCE,
            provider_symbol=ticker,
            is_signal_source=True,
            is_active=True,
        )
    )
    await db.flush()  # type: ignore[attr-defined]
    return instrument


async def _snapshot(db: object, instrument: Instrument) -> FundamentalSnapshot:
    return (
        (
            await db.execute(  # type: ignore[attr-defined]
                select(FundamentalSnapshot).where(
                    FundamentalSnapshot.instrument_id == instrument.id
                )
            )
        )
        .scalars()
        .one()
    )


class TestIngestion:
    async def test_every_statement_field_is_persisted(self, db: object) -> None:
        instrument = await _mapped_instrument(db, "AAPL")
        result = await EnrichmentService(db).enrich_sectors(  # type: ignore[arg-type]
            _StubProvider(**_FULL),  # type: ignore[arg-type]
            limit=10,
        )
        assert result.attempted == 1
        snapshot = await _snapshot(db, instrument)
        for name, expected in _FULL.items():
            assert getattr(snapshot, name) == expected, name

    async def test_large_amounts_survive_without_loss(self, db: object) -> None:
        """Enterprise values run to the trillions.

        Numeric(18,4) holds fourteen digits before the point; a narrower column
        would have raised, and a float column would have silently rounded — which
        is the failure this project uses Numeric everywhere to avoid.
        """
        instrument = await _mapped_instrument(db, "BIG")
        await EnrichmentService(db).enrich_sectors(  # type: ignore[arg-type]
            _StubProvider(enterprise_value=Decimal("4594739445760")),  # type: ignore[arg-type]
            limit=10,
        )
        snapshot = await _snapshot(db, instrument)
        assert snapshot.enterprise_value == Decimal("4594739445760")

    async def test_a_long_ratio_is_rounded_not_rejected(self, db: object) -> None:
        """yfinance returns ROE to seven decimals; the column keeps six.

        Rounding is the right outcome here — the seventh decimal of a return on
        equity is noise — but it should be a deliberate, tested fact rather than
        something discovered later from a mismatched comparison.
        """
        instrument = await _mapped_instrument(db, "PRECISE")
        await EnrichmentService(db).enrich_sectors(  # type: ignore[arg-type]
            _StubProvider(return_on_equity=Decimal("1.4875101")),  # type: ignore[arg-type]
            limit=10,
        )
        snapshot = await _snapshot(db, instrument)
        assert snapshot.return_on_equity == Decimal("1.487510")

    async def test_absent_fields_are_null_not_zero(self, db: object) -> None:
        """Unknown is not zero.

        A zero free cash flow says the business burns everything it makes; an
        absent one says we do not know. Collapsing them would put a company with
        no coverage at the bottom of any DCF ever built on this table.
        """
        instrument = await _mapped_instrument(db, "SPARSE")
        await EnrichmentService(db).enrich_sectors(  # type: ignore[arg-type]
            _StubProvider(),  # type: ignore[arg-type]
            limit=10,
        )
        snapshot = await _snapshot(db, instrument)
        for name in _FULL:
            assert getattr(snapshot, name) is None, name


class TestCoverageReporting:
    async def test_a_complete_statement_counts_toward_coverage(self, db: object) -> None:
        await _mapped_instrument(db, "COVERED")
        result = await EnrichmentService(db).enrich_sectors(  # type: ignore[arg-type]
            _StubProvider(**_FULL),  # type: ignore[arg-type]
            limit=10,
        )
        assert result.with_statements == 1

    async def test_a_missing_share_count_does_not_count(self, db: object) -> None:
        """Free cash flow without a share count cannot produce a per-share
        value, so it is not a usable DCF input and must not be counted as one."""
        await _mapped_instrument(db, "PARTIAL")
        result = await EnrichmentService(db).enrich_sectors(  # type: ignore[arg-type]
            _StubProvider(free_cash_flow=Decimal("1000")),  # type: ignore[arg-type]
            limit=10,
        )
        assert result.with_statements == 0
        # ...but the field itself is still stored, because a partial reading is
        # still evidence about coverage.
        assert result.attempted == 1
