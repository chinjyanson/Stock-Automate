"""Ingest news sentiment and serve it to the two things that read it (§6, §9).

Split like the insider and regime services: one half calls the network and
records, the other half answers questions purely from what was recorded. The
scanner and the risk engine only ever touch the reading half, so scoring an
instrument and sizing a trade both stay store-only.

The two consumers deliberately read *different* things:

  * **The scanner** reads the Loughran-McDonald polarity, as one bounded
    sub-signal inside the risk category. Its job is ranking, it runs over
    thousands of names, and a locally-computed lexicon score is available for
    every name with headlines.
  * **The risk engine** reads `risk_reading`, which prefers Finnhub's own score
    and falls back to the lexicon. Its job is to decline or shrink a specific
    trade, and it names which source it acted on in the audit trail.

Why the fallback exists at all: Finnhub's `/news-sentiment` is premium on
current plans, so on a free key the provider score is always absent. Gating
purely on it would produce a risk control that is configured, documented, tested
— and silently dead in production. Falling back to the lexicon keeps the gate
live, and recording the source keeps it honest about what it used.

Staleness fails **open**, everywhere. A missing or old reading means the gate
does not fire. Absence of news is not bad news, and a feed outage must not
express itself as a portfolio-wide refusal to trade.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.data.finnhub import FinnhubClient
from app.models.sentiment import SentimentSnapshot
from app.sentiment import score_headlines

log = structlog.get_logger(__name__)

#: How many days of headlines make one reading. Long enough that a quiet company
#: still produces something, short enough that last month's news does not blunt
#: this morning's. Seven also spans a weekend, so a Monday sweep is not reading
#: two days of silence.
NEWS_WINDOW_DAYS = 7

#: Beyond this, a stored reading is not a description of current news. The risk
#: gate ignores it entirely rather than acting on it — see the module docstring.
DEFAULT_MAX_AGE_DAYS = 3


@dataclass(frozen=True, slots=True)
class SentimentScore:
    """The scanner-facing reading for one instrument."""

    #: -1..+1, from the local lexicon.
    polarity: float
    uncertainty: float
    headline_count: int
    as_of: date


@dataclass(frozen=True, slots=True)
class RiskSentiment:
    """The risk-engine-facing reading, with its provenance attached."""

    polarity: float
    #: "provider" (Finnhub's own score) or "lexicon" (ours). Recorded in the
    #: audit trail, because "the trade was cut for bad news" is a materially
    #: different claim depending on who decided the news was bad.
    source: str
    as_of: date


def _dec(value: float | None) -> Decimal | None:
    return None if value is None else Decimal(str(round(value, 6)))


class SentimentService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # -- Ingest (network; jobs only) ---------------------------------------

    async def ingest_symbol(
        self,
        client: FinnhubClient,
        symbol: str,
        instrument_id: uuid.UUID,
        *,
        as_of: date | None = None,
    ) -> SentimentSnapshot | None:
        """Fetch, score and upsert one instrument's news tone for today.

        Returns None when the symbol had no headlines at all — no row is
        written, so the absence stays an absence. Writing an empty row would
        make "we looked and found nothing" indistinguishable from "the news was
        neutral", and the risk gate treats those very differently.
        """
        as_of = as_of or datetime.now(UTC).date()
        items = await client.company_news(
            symbol, since=as_of - timedelta(days=NEWS_WINDOW_DAYS), until=as_of
        )
        if not items:
            return None

        reading = score_headlines([item.text for item in items])
        if reading is None:
            return None

        provider = await client.news_sentiment(symbol)
        return await self.record(
            instrument_id=instrument_id,
            as_of=as_of,
            provider_symbol=symbol,
            polarity=reading.polarity,
            uncertainty=reading.uncertainty,
            positive_words=reading.positive_words,
            negative_words=reading.negative_words,
            headline_count=reading.documents,
            provider_score=provider.company_news_score if provider else None,
            provider_bullish_pct=provider.bullish_percent if provider else None,
            provider_buzz=provider.buzz if provider else None,
        )

    async def record(
        self,
        *,
        instrument_id: uuid.UUID,
        as_of: date,
        provider_symbol: str | None,
        polarity: float | None,
        uncertainty: float | None,
        positive_words: int,
        negative_words: int,
        headline_count: int,
        provider_score: float | None = None,
        provider_bullish_pct: float | None = None,
        provider_buzz: float | None = None,
    ) -> SentimentSnapshot:
        """Upsert one instrument-day. Re-running the sweep converges."""
        existing = (
            (
                await self._session.execute(
                    select(SentimentSnapshot).where(
                        SentimentSnapshot.instrument_id == instrument_id,
                        SentimentSnapshot.as_of == as_of,
                    )
                )
            )
            .scalars()
            .first()
        )
        snapshot = existing or SentimentSnapshot(instrument_id=instrument_id, as_of=as_of)
        snapshot.provider_symbol = provider_symbol
        snapshot.polarity = _dec(polarity)
        snapshot.uncertainty = _dec(uncertainty)
        snapshot.positive_words = positive_words
        snapshot.negative_words = negative_words
        snapshot.headline_count = headline_count
        snapshot.provider_score = _dec(provider_score)
        snapshot.provider_bullish_pct = _dec(provider_bullish_pct)
        snapshot.provider_buzz = _dec(provider_buzz)
        snapshot.retrieved_at = datetime.now(UTC)
        if existing is None:
            self._session.add(snapshot)
        await self._session.flush()
        return snapshot

    # -- Read (store-only) --------------------------------------------------

    async def _latest(
        self, instrument_id: uuid.UUID, *, as_of: date, max_age_days: int
    ) -> SentimentSnapshot | None:
        """Newest row for this instrument no older than `max_age_days`."""
        if max_age_days < 0:
            return None
        return (
            (
                await self._session.execute(
                    select(SentimentSnapshot)
                    .where(
                        SentimentSnapshot.instrument_id == instrument_id,
                        SentimentSnapshot.as_of <= as_of,
                        SentimentSnapshot.as_of >= as_of - timedelta(days=max_age_days),
                    )
                    .order_by(SentimentSnapshot.as_of.desc())
                    .limit(1)
                )
            )
            .scalars()
            .first()
        )

    async def score_instrument(
        self,
        instrument_id: uuid.UUID,
        *,
        as_of: date | None = None,
        max_age_days: int = DEFAULT_MAX_AGE_DAYS,
    ) -> SentimentScore | None:
        """Lexicon tone for the scanner, or None when there is nothing to say.

        None rather than a neutral reading, for the reason every optional signal
        here returns None: the scanner drops unavailable signals from its
        category average, so a non-observation scored as neutral would dilute
        the signals that *were* observed.
        """
        as_of = as_of or datetime.now(UTC).date()
        snapshot = await self._latest(instrument_id, as_of=as_of, max_age_days=max_age_days)
        if snapshot is None or snapshot.polarity is None:
            return None
        return SentimentScore(
            polarity=float(snapshot.polarity),
            uncertainty=float(snapshot.uncertainty or 0),
            headline_count=snapshot.headline_count,
            as_of=snapshot.as_of,
        )

    async def risk_reading(
        self,
        instrument_id: uuid.UUID,
        *,
        as_of: date | None = None,
        max_age_days: int = DEFAULT_MAX_AGE_DAYS,
    ) -> RiskSentiment | None:
        """Tone for the risk gate, preferring Finnhub's own score.

        None means "no opinion", and every caller must treat it that way: no
        reading, a reading too old, or a row with neither score present all
        return None and leave the trade unmodified.
        """
        as_of = as_of or datetime.now(UTC).date()
        snapshot = await self._latest(instrument_id, as_of=as_of, max_age_days=max_age_days)
        if snapshot is None:
            return None
        if snapshot.provider_score is not None:
            return RiskSentiment(
                polarity=float(snapshot.provider_score), source="provider", as_of=snapshot.as_of
            )
        if snapshot.polarity is not None:
            return RiskSentiment(
                polarity=float(snapshot.polarity), source="lexicon", as_of=snapshot.as_of
            )
        return None
