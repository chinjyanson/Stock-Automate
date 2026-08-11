"""Generate and serve Kronos forecasts.

Split like the index-options service: one half runs the model and records, the
other answers questions from what was recorded. The split is load-bearing rather
than tidy here — the generating half imports torch, which is ~250-300MB resident
against a 448MB worker, so the box that evaluates strategies must never reach
it. `latest_for` touches nothing but the table.
"""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.enums import Interval
from app.models.kronos import KronosPrediction

# Safe at module scope: `kronos_client` keeps torch inside the client's
# constructor, so importing the module costs nothing. Only building a client
# pays the ~300MB.
from app.signals.kronos_client import KronosForecast

log = structlog.get_logger(__name__)

#: A forecast older than this describes a market that has moved on. Wider than
#: the options window because a 20-day view decays more slowly than a snapshot
#: of dealer positioning, but still short enough that a stalled job goes quiet
#: rather than serving last month's opinion as today's.
MAX_PREDICTION_AGE_DAYS = 7

#: Bars handed to the model. Its own ceiling is 512 for small and base.
CONTEXT_BARS = 512


def _dec(value: float) -> Decimal:
    return Decimal(str(round(value, 6)))


class KronosService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def latest_for(
        self,
        instrument_ids: list[uuid.UUID],
        *,
        model_name: str,
        horizon_days: int,
        as_of: date | None = None,
    ) -> dict[uuid.UUID, dict[str, float]]:
        """Freshest usable forecast per instrument, as model features.

        Filtered to one model and horizon because the table deliberately allows
        several: a row from a different variant is a different measurement, and
        silently mixing them would make a coefficient fitted on one apply to
        another.

        Instruments with nothing recent are simply absent from the result. The
        caller treats that as missing rather than as an error — a name is never
        rejected for lacking a forecast.
        """
        if not instrument_ids:
            return {}
        cutoff = (as_of or datetime.now(UTC).date()) - timedelta(days=MAX_PREDICTION_AGE_DAYS)
        rows = (
            (
                await self._session.execute(
                    select(KronosPrediction)
                    .where(
                        KronosPrediction.instrument_id.in_(instrument_ids),
                        KronosPrediction.model_name == model_name,
                        KronosPrediction.horizon_days == horizon_days,
                        KronosPrediction.as_of >= cutoff,
                    )
                    .order_by(KronosPrediction.as_of.asc())
                )
            )
            .scalars()
            .all()
        )
        # Ascending, so a later row overwrites an earlier one and the freshest
        # per instrument survives.
        return {row.instrument_id: row.as_features() for row in rows}

    async def record(
        self,
        instrument_id: uuid.UUID,
        as_of: date,
        forecast: KronosForecast,
    ) -> KronosPrediction:
        """Upsert one forecast, keyed by instrument, date, model and horizon."""
        existing = (
            (
                await self._session.execute(
                    select(KronosPrediction).where(
                        KronosPrediction.instrument_id == instrument_id,
                        KronosPrediction.as_of == as_of,
                        KronosPrediction.model_name == forecast.model_name,
                        KronosPrediction.horizon_days == forecast.horizon_days,
                    )
                )
            )
            .scalars()
            .first()
        )
        row = existing or KronosPrediction(
            instrument_id=instrument_id,
            as_of=as_of,
            model_name=forecast.model_name,
            horizon_days=forecast.horizon_days,
        )
        row.predicted_return = _dec(forecast.predicted_return)
        row.path_dispersion = _dec(forecast.path_dispersion)
        row.prob_up = _dec(forecast.prob_up)
        row.predicted_drawdown = _dec(forecast.predicted_drawdown)
        row.context_bars = forecast.context_bars
        row.sample_count = forecast.sample_count
        row.generation_ms = forecast.generation_ms
        row.generated_at = datetime.now(UTC)
        if existing is None:
            self._session.add(row)
        await self._session.flush()
        return row

    async def forecast_universe(
        self, instrument_ids: list[uuid.UUID], *, limit: int | None = None
    ) -> dict[str, int]:
        """Run the model over a universe and record each result.

        **Imports torch, so this cannot run on the deployment box.** It is
        called by a local job; everything that serves strategies uses
        `latest_for` instead.
        """
        from app.config import get_settings
        from app.data.store import CandleStore
        from app.indicators.series import candles_to_series
        from app.signals.kronos_client import KronosClient, is_available, repo_is_present

        if not (is_available() and repo_is_present()):
            log.info("kronos.unavailable")
            return {"attempted": 0, "recorded": 0, "skipped": len(instrument_ids)}

        settings = get_settings()
        client = KronosClient(settings.kronos_model)
        store = CandleStore(self._session)

        attempted = recorded = skipped = 0
        for instrument_id in instrument_ids[: limit or len(instrument_ids)]:
            attempted += 1
            candles = await store.get_candles(
                instrument_id, Interval.D1, limit=CONTEXT_BARS, closed_only=True
            )
            if len(candles) < 64:
                skipped += 1
                continue
            series = candles_to_series(candles)
            forecast = client.forecast(
                open_=series.open,
                high=series.high,
                low=series.low,
                close=series.close,
                volume=series.volume,
                timestamps=[c.timestamp for c in candles],
                horizon_days=settings.kronos_horizon_days,
                sample_count=settings.kronos_sample_count,
            )
            if forecast is None:
                skipped += 1
                continue
            await self.record(instrument_id, candles[-1].timestamp.date(), forecast)
            recorded += 1

        log.info("kronos.swept", attempted=attempted, recorded=recorded, skipped=skipped)
        return {"attempted": attempted, "recorded": recorded, "skipped": skipped}
