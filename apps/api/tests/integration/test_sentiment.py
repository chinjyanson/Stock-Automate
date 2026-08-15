"""News sentiment, against real PostgreSQL.

The ingest half is one provider call and a parse; what is worth testing is the
*reading* half, because two different consumers read it differently and both
have a failure mode that would be invisible in production:

  * The scanner must get None — not a neutral 50 — for a name with no coverage,
    or ranking drifts toward "how famous is this company".
  * The risk engine must get None for a *stale* reading, or a news feed that
    stopped updating becomes a permanent, silent brake on the whole book.

The provider-versus-lexicon precedence is pinned too. Finnhub's own score is
premium-gated, so the fallback is not a corner case — on a free key it is the
only path that ever runs, and a mistake there would leave a risk control that is
configured, documented, tested and dead.
"""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime, timedelta

import pytest
from sqlalchemy import select

from app.models.enums import InstrumentKind, PriceUnit
from app.models.instrument import Exchange, Instrument
from app.models.sentiment import SentimentSnapshot
from app.services.sentiment import SentimentService

pytestmark = pytest.mark.asyncio


async def _instrument(db: object, ticker: str) -> Instrument:
    exchange = (
        await db.execute(select(Exchange).where(Exchange.mic == "XLON"))  # type: ignore[attr-defined]
    ).scalar_one_or_none()
    if exchange is None:
        exchange = Exchange(mic="XLON", name="London", country="GB", timezone="Europe/London")
        db.add(exchange)  # type: ignore[attr-defined]
        await db.flush()  # type: ignore[attr-defined]
    instrument = Instrument(
        id=uuid.uuid4(),
        exchange_id=exchange.id,
        exchange_ticker=ticker,
        name=f"{ticker} plc",
        kind=InstrumentKind.STOCK,
        currency="GBP",
        price_unit=PriceUnit.GBP,
    )
    db.add(instrument)  # type: ignore[attr-defined]
    await db.flush()  # type: ignore[attr-defined]
    return instrument


def _today() -> date:
    return datetime.now(UTC).date()


async def _record(
    db: object,
    instrument: Instrument,
    *,
    days_ago: int = 0,
    polarity: float | None = -0.5,
    provider_score: float | None = None,
) -> SentimentSnapshot:
    return await SentimentService(db).record(  # type: ignore[arg-type]
        instrument_id=instrument.id,
        as_of=_today() - timedelta(days=days_ago),
        provider_symbol=instrument.exchange_ticker,
        polarity=polarity,
        uncertainty=0.1,
        positive_words=1,
        negative_words=3,
        headline_count=4,
        provider_score=provider_score,
    )


class TestScannerReading:
    async def test_a_stored_reading_is_returned(self, db: object) -> None:
        instrument = await _instrument(db, "AAA")
        await _record(db, instrument, polarity=-0.4)
        score = await SentimentService(db).score_instrument(instrument.id)  # type: ignore[arg-type]
        assert score is not None
        assert score.polarity == pytest.approx(-0.4)
        assert score.headline_count == 4

    async def test_no_coverage_reports_nothing_rather_than_neutral(self, db: object) -> None:
        """None, not 0.0.

        The scanner drops unavailable signals from its category average, so a
        non-observation scored as neutral would dilute the signals that *were*
        observed — and most of a UK catalogue has no news coverage at all.
        """
        instrument = await _instrument(db, "QUIET")
        assert await SentimentService(db).score_instrument(instrument.id) is None  # type: ignore[arg-type]

    async def test_a_row_with_no_tone_words_reports_nothing(self, db: object) -> None:
        """A sweep that ran and found only toneless headlines said nothing."""
        instrument = await _instrument(db, "BLAND")
        await _record(db, instrument, polarity=None)
        assert await SentimentService(db).score_instrument(instrument.id) is None  # type: ignore[arg-type]

    async def test_a_stale_reading_is_ignored(self, db: object) -> None:
        instrument = await _instrument(db, "OLD")
        await _record(db, instrument, days_ago=30)
        assert await SentimentService(db).score_instrument(instrument.id) is None  # type: ignore[arg-type]

    async def test_the_newest_reading_wins(self, db: object) -> None:
        instrument = await _instrument(db, "FRESH")
        await _record(db, instrument, days_ago=2, polarity=-0.9)
        await _record(db, instrument, days_ago=0, polarity=0.6)
        score = await SentimentService(db).score_instrument(instrument.id)  # type: ignore[arg-type]
        assert score is not None
        assert score.polarity == pytest.approx(0.6)


class TestRiskReading:
    async def test_the_provider_score_is_preferred_when_present(self, db: object) -> None:
        """The user's decision: Finnhub gates risk, our lexicon ranks."""
        instrument = await _instrument(db, "BOTH")
        await _record(db, instrument, polarity=-0.2, provider_score=-0.7)
        reading = await SentimentService(db).risk_reading(instrument.id)  # type: ignore[arg-type]
        assert reading is not None
        assert reading.polarity == pytest.approx(-0.7)
        assert reading.source == "provider"

    async def test_it_falls_back_to_the_lexicon(self, db: object) -> None:
        """Finnhub's sentiment endpoint is premium; on a free key this is the
        only path that ever runs. Gating solely on the provider score would
        leave the control permanently disarmed."""
        instrument = await _instrument(db, "LEXONLY")
        await _record(db, instrument, polarity=-0.6, provider_score=None)
        reading = await SentimentService(db).risk_reading(instrument.id)  # type: ignore[arg-type]
        assert reading is not None
        assert reading.polarity == pytest.approx(-0.6)
        assert reading.source == "lexicon"

    async def test_a_stale_reading_yields_no_opinion(self, db: object) -> None:
        """The failure that matters most.

        A feed that stopped updating must not become a permanent brake on the
        book. Old news is not current news.
        """
        instrument = await _instrument(db, "STALE")
        await _record(db, instrument, days_ago=10, polarity=-0.9)
        assert await SentimentService(db).risk_reading(instrument.id) is None  # type: ignore[arg-type]

    async def test_the_staleness_window_is_configurable(self, db: object) -> None:
        instrument = await _instrument(db, "WINDOW")
        await _record(db, instrument, days_ago=5, polarity=-0.9)
        service = SentimentService(db)  # type: ignore[arg-type]
        assert await service.risk_reading(instrument.id, max_age_days=3) is None
        assert await service.risk_reading(instrument.id, max_age_days=7) is not None

    async def test_no_row_yields_no_opinion(self, db: object) -> None:
        instrument = await _instrument(db, "NONE")
        assert await SentimentService(db).risk_reading(instrument.id) is None  # type: ignore[arg-type]


class TestIdempotency:
    async def test_recording_the_same_day_twice_updates_in_place(self, db: object) -> None:
        """A re-run must converge, not accumulate a second opinion for one day."""
        instrument = await _instrument(db, "DUP")
        await _record(db, instrument, polarity=-0.5)
        await _record(db, instrument, polarity=0.3)
        rows = (
            (
                await db.execute(  # type: ignore[attr-defined]
                    select(SentimentSnapshot).where(
                        SentimentSnapshot.instrument_id == instrument.id
                    )
                )
            )
            .scalars()
            .all()
        )
        assert len(rows) == 1
        assert float(rows[0].polarity) == pytest.approx(0.3)
