"""Serving the crash overlay (§9).

Turns the research in `scripts/backtest_crash_overlay.py` into something that
runs nightly and leaves a record. It imports every feature definition and the
decision rule itself from `signals.crash_features`, so there is exactly one
definition of each and train/serve drift is impossible by construction rather
than by discipline.

## Why the whole history is recomputed every night

The model is refitted from history, so the history *is* the model — there is no
coefficient blob to load, and nothing to go stale against the code that reads
it. Recomputation is safe because **every day's probability depends only on data
before that day**, so replaying yesterday gives yesterday's answer again. That
makes the job idempotent and self-healing: a night the worker did not run leaves
a gap that the next run simply fills.

## Walk-forward, with periodic refits

Fitting one model on all history and scoring every day with it would score the
past with a model that had seen the past — in-sample probabilities, whose
percentile is not comparable to the out-of-sample one the live decision is
ranked against. That mismatch would bias the trigger.

So each day is scored by the most recent model fitted **strictly before** it,
refitted every `refit_every` trading days. Refitting daily would be honest too
and about twenty times slower for a macro model that moves over months; a month
of staleness is immaterial here and the cost is not.

## What it will and will not do

It records a **target exposure** and nothing else. No order is placed, no
position is sized, and the risk engine is not consulted, because that is a
decision about money that a human should take deliberately — and because the
evidence behind this model is two crises, one of which (2008) it handled well
and one of which (COVID) it did not.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import structlog
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.crash_overlay import CrashOverlayReading
from app.models_ml.logistic import Prior, fit
from app.signals import crash_features as cf

log = structlog.get_logger(__name__)

#: Trading days between refits of the walk-forward model. See the module
#: docstring: a month of staleness is immaterial for a macro model and twenty
#: times cheaper than refitting daily.
REFIT_EVERY = 21

#: Postgres binds at most this many parameters in one statement.
PARAMETER_LIMIT = 32_767

#: A feature must be present on at least this share of the fitting window to be
#: used at all. Below it the column is dropped rather than imputed wholesale —
#: `insider_rank` before 2008 is the case this exists for.
MIN_COVERAGE = 0.6


@dataclass(frozen=True)
class OverlayParams:
    """The overlay's tunables, all measured rather than chosen by taste.

    `sell_fraction` is the one real judgement call. Sweeping it, the drawdown
    protection rises monotonically as it fires more often while return falls,
    so there is no optimum to find — only a preference to state. 10% sits where
    2008's drawdown was more than halved (53.3% -> 26.5%) while the return still
    tracked buy-and-hold closely, which is the stated goal. It is deliberately
    *not* the setting that won 2008 outright: that was 15%, and picking it
    because it won the one crisis in the sample is the overfitting this project
    has caught itself doing twice.
    """

    #: What counts as a sharp fall, and over how long.
    fall: float = 0.02
    horizon: int = 1
    #: Warn on this share of the most alarming recent days.
    sell_fraction: float = 0.10
    #: Exposure held while standing aside.
    defensive: float = 0.30
    #: Days a standing warning is honoured before the position returns anyway.
    timeout: int = 20


@dataclass(frozen=True)
class OverlayReading:
    """One day's answer, for callers that do not want the ORM row."""

    as_of: date
    probability: float | None
    trigger: float | None
    is_warning: bool
    target_exposure: float
    days_out: int
    reason: str


class CrashOverlayService:
    """Maintains the daily record and answers what the exposure should be."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def refresh(
        self,
        *,
        since: str = cf.DEFAULT_SINCE,
        insider_path: Path = Path("data/insider_index.csv"),
    ) -> int:
        """Download the feature history and upsert it. Idempotent per date.

        Fetches the whole span every time rather than only the tail. The series
        are small, several are revised after publication (FRED restates), and a
        rolling feature computed from a truncated window is not the same number
        as one computed from the full history — which is exactly the kind of
        difference that would make today's reading incomparable to yesterday's.
        """
        import yfinance as yf

        frame = yf.Ticker(cf.INDEX_SYMBOL).history(period="max", interval="1d")
        frame = frame[frame.index >= since]
        if frame.empty:
            log.warning("crash_overlay.no_index_history", since=since)
            return 0

        close = frame["Close"].to_numpy(dtype=np.float64)
        index = pd.DatetimeIndex(frame.index.tz_localize(None)).normalize()
        daily = np.concatenate([[np.nan], close[1:] / close[:-1] - 1.0])
        signals = cf.build(insider_path, index, close, daily)

        payload: list[dict[str, object]] = [
            {
                "as_of": index[i].date(),
                "index_close": _dec(float(close[i])),
                **{name: _dec(float(signals[name][i])) for name in cf.FEATURES},
            }
            for i in range(close.size)
        ]
        await self._upsert(payload, updates=("index_close", *cf.FEATURES))
        log.info("crash_overlay.refreshed", days=len(payload), since=since)
        return len(payload)

    async def evaluate(self, params: OverlayParams | None = None) -> OverlayReading | None:
        """Score every stored day and record the exposure each implies."""
        params = params or OverlayParams()
        rows = list(
            (
                await self._session.execute(
                    select(CrashOverlayReading).order_by(CrashOverlayReading.as_of)
                )
            )
            .scalars()
            .all()
        )
        if len(rows) < cf.CALIBRATION_MIN:
            log.warning("crash_overlay.too_little_history", days=len(rows))
            return None

        dates = [r.as_of for r in rows]
        close = np.array([float(r.index_close or np.nan) for r in rows])
        signals = {name: np.array([_float(getattr(r, name)) for r in rows]) for name in cf.FEATURES}
        features = self._usable_features(signals)
        probabilities = self._walk_forward(signals, close, features, params)

        # The trigger for day i ranks it against the model's own output *before*
        # i, so it can never be set by the day it is judging.
        triggers = np.full(len(rows), np.nan)
        for i in range(len(rows)):
            window = probabilities[max(0, i - cf.CALIBRATION_WINDOW) : i]
            value = cf.trigger_from(window, params.sell_fraction)
            if value is not None:
                triggers[i] = value

        state = cf.OverlayState()
        payload: list[dict[str, object]] = []
        latest: OverlayReading | None = None
        for i, row in enumerate(rows):
            probability = _or_none(probabilities[i])
            trigger = _or_none(triggers[i])
            state, reason = cf.step(
                state,
                probability=probability,
                trigger=trigger,
                close=float(close[i]) if np.isfinite(close[i]) else 0.0,
                defensive=params.defensive,
                timeout=params.timeout,
            )
            warning = bool(
                probability is not None and trigger is not None and probability >= trigger
            )
            payload.append(
                {
                    "as_of": row.as_of,
                    "probability": _dec(probability),
                    "trigger": _dec(trigger),
                    "is_warning": warning,
                    "target_exposure": _dec(state.exposure),
                    "exit_price": _dec(state.exit_price),
                    "days_out": state.days_out,
                    "reason": reason,
                }
            )
            latest = OverlayReading(
                as_of=row.as_of,
                probability=probability,
                trigger=trigger,
                is_warning=warning,
                target_exposure=state.exposure,
                days_out=state.days_out,
                reason=reason,
            )

        await self._upsert(
            payload,
            updates=(
                "probability",
                "trigger",
                "is_warning",
                "target_exposure",
                "exit_price",
                "days_out",
                "reason",
            ),
        )
        if latest is not None:
            log.info(
                "crash_overlay.evaluated",
                as_of=str(latest.as_of),
                exposure=latest.target_exposure,
                warning=latest.is_warning,
                features=len(features),
                days=len(dates),
            )
        return latest

    async def latest(self) -> CrashOverlayReading | None:
        result = await self._session.execute(
            select(CrashOverlayReading).order_by(CrashOverlayReading.as_of.desc()).limit(1)
        )
        return result.scalar_one_or_none()

    async def count(self) -> int:
        result = await self._session.execute(select(func.count()).select_from(CrashOverlayReading))
        return int(result.scalar_one())

    async def history(self, limit: int = 90) -> list[CrashOverlayReading]:
        result = await self._session.execute(
            select(CrashOverlayReading).order_by(CrashOverlayReading.as_of.desc()).limit(limit)
        )
        return list(result.scalars().all())

    # -- internals -----------------------------------------------------------

    @staticmethod
    def _usable_features(signals: dict[str, np.ndarray]) -> tuple[str, ...]:
        """Drop columns too sparse to fit on, rather than imputing them.

        `insider_rank` is the reason this exists: it is undefined for the first
        two years and absent entirely if the SEC index has never been built. A
        column that is mostly missing contributes noise and a meaningless
        standardisation, so it is better left out than filled in.
        """
        usable = []
        for name in cf.FEATURES:
            column = signals[name]
            if column.size and float(np.mean(np.isfinite(column))) >= MIN_COVERAGE:
                usable.append(name)
        return tuple(usable)

    def _walk_forward(
        self,
        signals: dict[str, np.ndarray],
        close: np.ndarray,
        features: tuple[str, ...],
        params: OverlayParams,
    ) -> np.ndarray:
        """Probability per day, from a model fitted only on that day's past."""
        n = close.size
        out = np.full(n, np.nan)
        start = cf.CALIBRATION_MIN
        model = None

        for i in range(start, n):
            if model is None or (i - start) % REFIT_EVERY == 0:
                # Labels need `horizon` days of future to be known, so the
                # fitting window stops short of the day being scored by exactly
                # that much. This is the look-ahead guard.
                cutoff = i - params.horizon - 1
                usable = [
                    j for j in range(cutoff) if np.isfinite(close[j]) and j + params.horizon < n
                ]
                if len(usable) < cf.CALIBRATION_MIN:
                    continue
                at = np.array(usable)
                model = fit(
                    cf.rows(signals, at, features),
                    np.array(
                        [
                            cf.label_fall(close, j, fall=params.fall, horizon=params.horizon)
                            for j in at
                        ]
                    ),
                    features,
                    priors={f: Prior(0.0, 1.0) for f in features},
                    label_definition=(
                        f"fall of {params.fall:.0%} within {params.horizon} trading day(s)"
                    ),
                )
            if model is None:
                continue
            reading = {f: float(signals[f][i]) for f in features if np.isfinite(signals[f][i])}
            if reading:
                out[i] = model.probability(reading)
        return out

    async def _upsert(self, payload: list[dict[str, object]], *, updates: tuple[str, ...]) -> None:
        """Write in chunks, because Postgres binds a limited number of parameters.

        Sized from the **table's** column count, not the payload dict's. Those
        differ: SQLAlchemy also binds the columns filled by defaults — `id`,
        `created_at`, `updated_at` — which never appear in the payload. Counting
        only the keys undercounts by three per row, which is invisible on a
        thousand rows and blows the limit on a full backfill.
        """
        if not payload:
            return

        # Sent explicitly rather than left to the column defaults. A Core insert
        # takes nothing from the ORM, so a `server_default` the database does
        # not actually have becomes a NOT NULL violation on the first bulk
        # write — and `onupdate=` never fires for ON CONFLICT DO UPDATE, so
        # without this `updated_at` would stay frozen at the backfill date
        # however many times the nightly job rewrote the row.
        now = datetime.now(UTC)
        for row in payload:
            row.setdefault("created_at", now)
            row["updated_at"] = now

        chunk = _chunk_size(len(CrashOverlayReading.__table__.columns))
        for offset in range(0, len(payload), chunk):
            batch = payload[offset : offset + chunk]
            statement = pg_insert(CrashOverlayReading).values(batch)
            await self._session.execute(
                statement.on_conflict_do_update(
                    index_elements=[CrashOverlayReading.as_of],
                    set_={
                        name: getattr(statement.excluded, name) for name in (*updates, "updated_at")
                    },
                )
            )
        await self._session.flush()


def _chunk_size(columns: int) -> int:
    """Rows per statement that keep the bind count inside the limit."""
    return max(1, PARAMETER_LIMIT // max(columns, 1))


def _dec(value: float | None) -> Decimal | None:
    if value is None or not np.isfinite(value):
        return None
    return Decimal(str(round(float(value), 8)))


def _float(value: Decimal | None) -> float:
    return float(value) if value is not None else float("nan")


def _or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None
