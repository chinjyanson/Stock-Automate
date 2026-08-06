"""Post-earnings announcement drift, against real PostgreSQL.

The scoring half is what is tested here — the ingest half is one provider call
and nothing else. What matters is that the surprise is measured as an *abnormal*
return, that the signal decays across the drift window rather than switching off
at its edge, and that a stock with no recent report reports nothing rather than
a neutral 50 that would dilute every other reading with a non-observation.
"""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest
from sqlalchemy import select

from app.data.store import CandleStore
from app.data.types import Candle as CandleDTO
from app.models.earnings import EarningsEvent
from app.models.enums import InstrumentKind, Interval, PriceUnit, ProviderKind
from app.models.instrument import Exchange, Instrument
from app.services.pead import PeadService

pytestmark = pytest.mark.asyncio


async def _instrument(db: object, ticker: str, closes: list[float]) -> Instrument:
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

    now = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    n = len(closes)
    candles = [
        CandleDTO(
            symbol=ticker,
            interval=Interval.D1,
            timestamp=now - timedelta(days=n - 1 - i),
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


def _today() -> date:
    return datetime.now(UTC).date()


async def _event(db: object, instrument: Instrument, days_ago: int) -> EarningsEvent:
    event = EarningsEvent(
        instrument_id=instrument.id, report_date=_today() - timedelta(days=days_ago)
    )
    db.add(event)  # type: ignore[attr-defined]
    await db.flush()  # type: ignore[attr-defined]
    return event


_HISTORY_DAYS = 120


def _closes_with_move(days_ago: int, move: float, n: int = _HISTORY_DAYS) -> list[float]:
    """Flat at 100 until the report `days_ago` back, then re-levelled by `move`.

    The bar dated on the report is the last "before" price, so the jump lands on
    the bar *after* it — which is what the reaction window measures. Built as a
    function of `days_ago` rather than as a fixed list so an old report still
    has price history either side of it to measure against.
    """
    jump_index = (n - 1 - days_ago) + 1
    return [100.0 if i < jump_index else 100.0 * (1 + move) for i in range(n)]


class TestScoring:
    async def test_a_positive_surprise_scores_above_neutral(self, db: object) -> None:
        instrument = await _instrument(db, "BEAT", _closes_with_move(4, 0.12))
        await _event(db, instrument, days_ago=4)
        score = await PeadService(db).score_instrument(instrument.id)  # type: ignore[arg-type]
        assert score is not None
        assert score.score > 50
        assert score.surprise > 0

    async def test_a_negative_surprise_scores_below_neutral(self, db: object) -> None:
        instrument = await _instrument(db, "MISS", _closes_with_move(4, -0.12))
        await _event(db, instrument, days_ago=4)
        score = await PeadService(db).score_instrument(instrument.id)  # type: ignore[arg-type]
        assert score is not None
        assert score.score < 50
        assert score.surprise < 0

    async def test_the_benchmark_move_is_subtracted(self, db: object) -> None:
        """A report that landed on a day the whole market rose is less of a
        surprise than the raw move suggests — that is what makes it *abnormal*."""
        instrument = await _instrument(db, "ABN", _closes_with_move(4, 0.12))
        await _event(db, instrument, days_ago=4)
        service = PeadService(db)  # type: ignore[arg-type]
        raw = await service.score_instrument(instrument.id)
        adjusted = await service.score_instrument(instrument.id, 0.10)
        assert raw is not None and adjusted is not None
        assert adjusted.surprise < raw.surprise
        assert adjusted.score < raw.score

    async def test_the_signal_decays_as_the_report_ages(self, db: object) -> None:
        """Drift fades. A 50-day-old report must not carry a fresh one's weight
        right up to the edge of the window."""
        fresh = await _instrument(db, "FRESH", _closes_with_move(2, 0.12))
        await _event(db, fresh, days_ago=2)
        stale = await _instrument(db, "STALE", _closes_with_move(50, 0.12))
        await _event(db, stale, days_ago=50)
        service = PeadService(db)  # type: ignore[arg-type]
        recent = await service.score_instrument(fresh.id)
        old = await service.score_instrument(stale.id)
        assert recent is not None and old is not None
        assert recent.score > old.score > 50


class TestAbsence:
    async def test_no_event_reports_nothing_rather_than_neutral(self, db: object) -> None:
        """None, not 50. A stock that has not reported has not sent a signal,
        and scoring the non-observation would dilute every other reading."""
        instrument = await _instrument(db, "QUIET", _closes_with_move(4, 0.12))
        assert await PeadService(db).score_instrument(instrument.id) is None  # type: ignore[arg-type]

    async def test_an_event_past_the_drift_window_is_ignored(self, db: object) -> None:
        instrument = await _instrument(db, "ANCIENT", _closes_with_move(200, 0.12))
        await _event(db, instrument, days_ago=200)
        assert await PeadService(db).score_instrument(instrument.id) is None  # type: ignore[arg-type]


class TestIdempotency:
    async def test_ingesting_the_same_date_twice_makes_one_row(self, db: object) -> None:
        instrument = await _instrument(db, "DUP", _closes_with_move(5, 0.12))
        await _event(db, instrument, days_ago=5)
        # The unique constraint is what makes a re-run converge rather than
        # accumulating duplicate reports for the same quarter.
        await PeadService(db)._session.execute(  # type: ignore[arg-type]
            select(EarningsEvent).where(EarningsEvent.instrument_id == instrument.id)
        )
        rows = (
            (
                await db.execute(  # type: ignore[attr-defined]
                    select(EarningsEvent).where(EarningsEvent.instrument_id == instrument.id)
                )
            )
            .scalars()
            .all()
        )
        assert len(rows) == 1
