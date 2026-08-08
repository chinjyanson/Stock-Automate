"""Sector/industry enrichment + sector-ETF setup (§6 extension).

Two jobs that give the scanner its "zoomed-out" sector context:

  * **enrich_sectors** — tag mapped instruments with the sector/industry yfinance
    assigns them (from the same `.info` the fundamentals come from). Batched and
    resumable: `industry` is set on *every* attempt (to the real value, or
    "Unclassified" when Yahoo has nothing), and only un-attempted rows are
    selected, so the sweep terminates and never re-hammers the dead ends.

  * **ensure_sector_etfs** — make the 11 SPDR sector ETFs available as instruments
    with candles, so the scorer can load each sector's proxy series (the same way
    it already loads the SPY benchmark). Idempotent.

Neither is on the hot path: per-symbol `.info` is slow/rate-limited, so this is a
run-once background sweep. Until it runs, the scanner simply produces no sector
signal for untagged stocks (they drop out, never penalised).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.data.yfinance_provider import YFinanceProvider
from app.models.enums import InstrumentKind, LifecycleState, PriceUnit, ProviderKind
from app.models.instrument import Instrument, MarketDataMapping
from app.models.market_data import FundamentalSnapshot
from app.services.ingestion import IngestionService

log = structlog.get_logger(__name__)

#: The 11 SPDR sector ETFs used as GICS sector proxies (symbol → display name).
SECTOR_ETFS: dict[str, str] = {
    "XLK": "Technology Select Sector SPDR Fund",
    "XLF": "Financial Select Sector SPDR Fund",
    "XLV": "Health Care Select Sector SPDR Fund",
    "XLY": "Consumer Discretionary Select Sector SPDR Fund",
    "XLP": "Consumer Staples Select Sector SPDR Fund",
    "XLE": "Energy Select Sector SPDR Fund",
    "XLI": "Industrial Select Sector SPDR Fund",
    "XLB": "Materials Select Sector SPDR Fund",
    "XLRE": "Real Estate Select Sector SPDR Fund",
    "XLU": "Utilities Select Sector SPDR Fund",
    "XLC": "Communication Services Select Sector SPDR Fund",
}

#: Rate-sensitivity proxy for the scanner's risk category. A bond *price* series,
#: deliberately not a yield index — see `scanner.engine.RATES_PROXY_SYMBOL`.
RATES_PROXY_ETFS: dict[str, str] = {"IEF": "iShares 7-10 Year Treasury Bond ETF"}

#: Every ETF seeded as a shared reference series for scoring. None is ever a
#: trade candidate; they exist so the daily refresh keeps them fresh.
REFERENCE_ETFS: dict[str, str] = {**SECTOR_ETFS, **RATES_PROXY_ETFS}

_ETF_BACKFILL_DAYS = 500
_UNCLASSIFIED = "Unclassified"


@dataclass
class EnrichResult:
    attempted: int = 0
    classified: int = 0  # got a real sector
    unclassified: int = 0  # attempted, yfinance had no classification
    with_fundamentals: int = 0  # got at least one fundamental (P/E, P/B, …)
    #: Got the cash-flow / balance-sheet items a DCF would need. Counted
    #: separately from `with_fundamentals` because deciding whether a DCF is
    #: worth building is precisely a question about this number, and it would be
    #: invisible folded into the other one.
    with_statements: int = 0


class EnrichmentService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._ingestion = IngestionService(session)

    async def enrich_sectors(self, provider: YFinanceProvider, limit: int) -> EnrichResult:
        """Tag sector/industry AND snapshot fundamentals for one batch.

        Gated on the absence of a `FundamentalSnapshot` — the "never attempted"
        marker — so the sweep covers the whole mapped universe once and
        terminates (every attempt writes a snapshot, even an empty one, so no
        row is re-selected). One yfinance `.info` call per instrument yields
        both the classification and the fundamentals.
        """
        has_snapshot = (
            select(FundamentalSnapshot.id).where(FundamentalSnapshot.instrument_id == Instrument.id)
        ).exists()
        rows = (
            await self._session.execute(
                select(Instrument, MarketDataMapping.provider_symbol)
                .join(MarketDataMapping, MarketDataMapping.instrument_id == Instrument.id)
                .where(
                    MarketDataMapping.provider == ProviderKind.YFINANCE,
                    MarketDataMapping.is_signal_source.is_(True),
                    MarketDataMapping.is_active.is_(True),
                    ~has_snapshot,  # never attempted for fundamentals
                )
                .order_by(Instrument.id)
                .limit(limit)
            )
        ).all()

        result = EnrichResult(attempted=len(rows))
        for instrument, symbol in rows:
            sector, industry, fundamentals = await provider.get_profile(symbol)
            # Fill classification where missing (idempotent with the sector sweep).
            if sector:
                instrument.sector = sector
                result.classified += 1
            else:
                result.unclassified += 1
            if not instrument.industry:
                instrument.industry = industry or _UNCLASSIFIED

            has_any = any(
                getattr(fundamentals, f) is not None
                for f in ("trailing_pe", "price_to_book", "profit_margin", "dividend_yield")
            )
            if has_any:
                result.with_fundamentals += 1
            # Free cash flow and a share count are the two a DCF cannot proceed
            # without, so they are what "we have statements for this" means.
            if fundamentals.free_cash_flow is not None and (
                fundamentals.shares_outstanding is not None
            ):
                result.with_statements += 1
            # Always write a snapshot (even if sparse) to mark the attempt.
            self._session.add(
                FundamentalSnapshot(
                    instrument_id=instrument.id,
                    provider=ProviderKind.YFINANCE,
                    as_of=datetime.now(UTC).date(),
                    currency=fundamentals.currency,
                    market_cap=fundamentals.market_cap,
                    trailing_pe=fundamentals.trailing_pe,
                    price_to_book=fundamentals.price_to_book,
                    dividend_yield=fundamentals.dividend_yield,
                    revenue_growth=fundamentals.revenue_growth,
                    earnings_growth=fundamentals.earnings_growth,
                    profit_margin=fundamentals.profit_margin,
                    debt_to_equity=fundamentals.debt_to_equity,
                    free_cash_flow=fundamentals.free_cash_flow,
                    operating_cash_flow=fundamentals.operating_cash_flow,
                    total_debt=fundamentals.total_debt,
                    total_cash=fundamentals.total_cash,
                    shares_outstanding=fundamentals.shares_outstanding,
                    ebitda=fundamentals.ebitda,
                    enterprise_value=fundamentals.enterprise_value,
                    book_value_per_share=fundamentals.book_value_per_share,
                    return_on_equity=fundamentals.return_on_equity,
                    current_ratio=fundamentals.current_ratio,
                    retrieved_at=datetime.now(UTC),
                )
            )
        await self._session.flush()
        return result

    async def ensure_sector_etfs(self, provider: YFinanceProvider) -> dict[str, int]:
        """Create + candle-backfill the reference ETFs. Idempotent.

        The 11 SPDR sector proxies plus the rates proxy — every series the
        scanner correlates an instrument against, other than the SPY benchmark.
        """
        created = 0
        candles = 0
        for symbol, name in REFERENCE_ETFS.items():
            instrument = (
                (
                    await self._session.execute(
                        select(Instrument)
                        .join(MarketDataMapping, MarketDataMapping.instrument_id == Instrument.id)
                        .where(
                            MarketDataMapping.provider == ProviderKind.YFINANCE,
                            MarketDataMapping.provider_symbol == symbol,
                        )
                        .limit(1)
                    )
                )
                .scalars()
                .first()
            )

            if instrument is None:
                instrument = Instrument(
                    name=name,
                    kind=InstrumentKind.ETF,
                    currency="USD",
                    price_unit=PriceUnit.USD,
                    exchange_ticker=symbol,
                    lifecycle_state=LifecycleState.DISCOVERED,
                    # Scanner-eligible so the daily refresh keeps it fresh, exactly
                    # like the SPY benchmark; it is never a trade candidate itself.
                    is_scanner_eligible=True,
                    sector="ETF",
                    industry=_UNCLASSIFIED,
                )
                self._session.add(instrument)
                await self._session.flush()
                self._session.add(
                    MarketDataMapping(
                        instrument_id=instrument.id,
                        provider=ProviderKind.YFINANCE,
                        provider_symbol=symbol,
                        price_unit=PriceUnit.USD,
                        is_signal_source=True,
                        resolution_method="sector_etf_seed",
                    )
                )
                await self._session.flush()
                created += 1

            ingest = await self._ingestion.ingest_daily(
                instrument, provider, backfill_days=_ETF_BACKFILL_DAYS
            )
            candles += ingest.candles_written
        await self._session.flush()
        return {"created": created, "candles_written": candles}
