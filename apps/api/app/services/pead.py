"""Post-earnings announcement drift (PEAD).

Stocks that surprise the market at earnings keep drifting the same way for
weeks afterwards. It is one of the more durable anomalies in the literature —
documented since Ball and Brown in 1968 and repeatedly since — and the usual
explanation is that investors under-react to the news and finish repricing over
the following month or two rather than at the announcement.

Structured like the insider service, for the same reasons: an ingest half that
talks to a provider and a scoring half that does not, an idempotent upsert so
re-running converges, and `None` rather than a neutral number when there is no
event — a stock that has not reported recently has not sent a signal, and
scoring that as 50 would dilute every other reading with a non-observation.

**What is measured.** Only the report *date* comes from the provider. The
surprise is the stock's abnormal return over the announcement window — its move
minus the benchmark's — computed from candles already in the store. Analyst
estimate data is not reliably free, and the market's own reaction is arguably
the better input anyway: the drift is a continuation of that reaction, not of
the estimate miss.

**Coverage.** `get_earnings_dates` is good for US listings and much thinner
elsewhere, so most of a Trading 212 universe will carry no event. That is
handled the only correct way — the signal reports unavailable and drops out of
its category, rather than marking those instruments down for a gap in the data.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import structlog
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.data.store import CandleStore
from app.indicators.series import candles_to_series
from app.models.earnings import EarningsEvent
from app.models.enums import Interval

log = structlog.get_logger(__name__)

#: How long drift is considered live. The literature puts most of it inside a
#: quarter; 60 days keeps the window inside one reporting cycle so two
#: consecutive reports never both count.
DRIFT_WINDOW_DAYS = 60

#: Bars after the report over which the announcement reaction is measured. One
#: day is the reaction itself; three lets a report that landed after the close,
#: or was digested over a couple of sessions, still register.
REACTION_BARS = 3

#: Abnormal return at which the signal is at full strength. A 10% move against
#: the benchmark on the news is a large surprise; beyond it the signal
#: saturates rather than growing without limit.
SATURATION_MOVE = 0.10

#: Neutral midpoint and the most drift can add or subtract from it.
NEUTRAL = 50.0
MAX_POINTS = 50.0


@dataclass(frozen=True, slots=True)
class PeadScore:
    """A 0-100 reading, plus the evidence for it."""

    score: float
    report_date: date
    surprise: float
    days_since: int
    detail: str


class PeadService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._store = CandleStore(session)

    # -- Ingest -------------------------------------------------------------

    async def ingest_dates(self, symbol: str, instrument_id: uuid.UUID) -> int:
        """Fetch and upsert recent report dates for one instrument. Network.

        One provider call per instrument, and report dates move quarterly, so
        this belongs on a weekly job rather than in the daily scan.
        """
        dates = await self._fetch_dates(symbol)
        if not dates:
            return 0
        rows = [{"instrument_id": instrument_id, "report_date": d} for d in dates]
        statement = pg_insert(EarningsEvent).values(rows)
        await self._session.execute(
            statement.on_conflict_do_nothing(
                index_elements=[EarningsEvent.instrument_id, EarningsEvent.report_date]
            )
        )
        await self._session.flush()
        return len(rows)

    @staticmethod
    async def _fetch_dates(symbol: str) -> list[date]:
        def _fetch() -> list[date]:
            import yfinance as yf

            frame = yf.Ticker(symbol).get_earnings_dates(limit=8)
            if frame is None or frame.empty:
                return []
            today = datetime.now(UTC).date()
            horizon = today - timedelta(days=DRIFT_WINDOW_DAYS * 2)
            out: list[date] = []
            for stamp in frame.index:
                try:
                    when = stamp.date()
                except AttributeError:
                    continue
                # Future-dated rows are scheduled, not reported: nothing has
                # drifted yet, and storing them would let a report score before
                # it happened.
                if horizon <= when <= today:
                    out.append(when)
            return out

        try:
            return await asyncio.to_thread(_fetch)
        except Exception as exc:
            log.warning("pead.fetch_failed", symbol=symbol, error=str(exc))
            return []

    # -- Score --------------------------------------------------------------

    async def score_instrument(
        self,
        instrument_id: uuid.UUID,
        benchmark_returns: float | None = None,
        *,
        as_of: date | None = None,
    ) -> PeadScore | None:
        """Drift score for one instrument, or None when there is no live event.

        `benchmark_returns` is the benchmark's move over the same reaction
        window, so the surprise is measured as *abnormal* return. Without it the
        raw move is used, which on a day the whole market fell would read a
        broad selloff as a bad earnings reaction.
        """
        today = as_of or datetime.now(UTC).date()
        event = (
            (
                await self._session.execute(
                    select(EarningsEvent)
                    .where(
                        EarningsEvent.instrument_id == instrument_id,
                        EarningsEvent.report_date <= today,
                        EarningsEvent.report_date >= today - timedelta(days=DRIFT_WINDOW_DAYS),
                    )
                    .order_by(EarningsEvent.report_date.desc())
                    .limit(1)
                )
            )
            .scalars()
            .first()
        )
        if event is None:
            return None

        surprise = await self._reaction(instrument_id, event, benchmark_returns)
        if surprise is None:
            return None
        event.surprise = Decimal(str(round(surprise, 6)))

        days_since = (today - event.report_date).days
        # Drift decays over the window rather than switching off at its end: the
        # anomaly fades, so a 55-day-old report should not carry the weight of a
        # 5-day-old one right up to the cliff edge.
        decay = max(0.0, 1.0 - days_since / DRIFT_WINDOW_DAYS)
        strength = max(-1.0, min(1.0, surprise / SATURATION_MOVE))
        score = NEUTRAL + MAX_POINTS * strength * decay
        return PeadScore(
            score=round(max(0.0, min(100.0, score)), 2),
            report_date=event.report_date,
            surprise=surprise,
            days_since=days_since,
            detail=(
                f"reported {event.report_date.isoformat()} ({days_since}d ago), "
                f"abnormal reaction {surprise:+.1%}"
            ),
        )

    async def _reaction(
        self,
        instrument_id: uuid.UUID,
        event: EarningsEvent,
        benchmark_returns: float | None,
    ) -> float | None:
        """The stock's abnormal move over the bars following the report."""
        candles = await self._store.get_candles(
            instrument_id, Interval.D1, limit=400, closed_only=True
        )
        if len(candles) < REACTION_BARS + 2:
            return None
        # The last bar at or before the report is the "before" price; the
        # reaction is measured from there over the following bars.
        index = None
        for position, candle in enumerate(candles):
            if candle.timestamp.date() <= event.report_date:
                index = position
            else:
                break
        if index is None or index + 1 >= len(candles):
            return None
        series = candles_to_series(candles)
        end = min(index + REACTION_BARS, series.length - 1)
        before = float(series.close[index])
        after = float(series.close[end])
        if before <= 0:
            return None
        move = after / before - 1.0
        return move if benchmark_returns is None else move - benchmark_returns
