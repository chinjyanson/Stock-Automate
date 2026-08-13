"""The crash overlay end to end, against real PostgreSQL.

The network half — fetching VIX, FRED and the rest — is not what is worth
testing; a mocked download proves only that the mock works. What matters is
everything downstream of the data landing:

  * that scoring a day never uses that day's own future, which is the failure
    that makes a backtest beautiful and a strategy worthless;
  * that replaying the history twice gives the same answer, since the nightly
    job recomputes everything and a drifting record would be worse than none;
  * that a thin history produces *no* decision rather than a confident one.

So the fixtures write feature rows directly and let the service do the fitting,
calibration and state machine over them.
"""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

import numpy as np
import pytest
from sqlalchemy import select

from app.models.crash_overlay import CrashOverlayReading
from app.services.crash_overlay import CrashOverlayService, OverlayParams
from app.signals import crash_features as cf

pytestmark = pytest.mark.asyncio

#: Long enough that the trigger can calibrate (needs CALIBRATION_MIN of model
#: output) on top of the history the first fit consumes.
DAYS = cf.CALIBRATION_MIN * 3


def _series(days: int = DAYS, seed: int = 7) -> list[dict[str, object]]:
    """A synthetic history with real structure: falls follow stress.

    The point is not realism but *learnability* — if volatility and spreads
    carried no information about the next day, the model would be fitting noise
    and every assertion below would be about nothing.
    """
    rng = np.random.default_rng(seed)
    stress = np.abs(rng.normal(size=days).cumsum() % 3.0)
    close = [100.0]
    for i in range(1, days):
        shock = rng.normal(scale=0.004 + 0.02 * (stress[i - 1] > 2.0))
        close.append(max(close[-1] * (1.0 + shock), 1.0))

    start = date(2010, 1, 4)
    rows: list[dict[str, object]] = []
    for i in range(days):
        rows.append(
            {
                "as_of": start + timedelta(days=i),
                "index_close": Decimal(str(round(close[i], 6))),
                "vix": Decimal(str(round(12.0 + 8.0 * stress[i], 4))),
                "vix_term_structure": Decimal(str(round(1.1 - 0.1 * stress[i], 6))),
                "credit_spread": Decimal(str(round(0.8 + 0.5 * stress[i], 6))),
                "hyg_tlt": Decimal(str(round(-0.01 * stress[i], 6))),
                "hyg_lqd": Decimal(str(round(-0.005 * stress[i], 6))),
                "skew": Decimal(str(round(120.0 + stress[i], 4))),
                "small_cap_rs": Decimal(str(round(-0.002 * stress[i], 6))),
                "realised_vol": Decimal(str(round(0.10 + 0.08 * stress[i], 6))),
                "vol_of_vol": Decimal(str(round(0.01 + 0.01 * stress[i], 6))),
                "drawdown_from_high": Decimal(str(round(-0.02 * stress[i], 6))),
                "insider_rank": Decimal(str(round(0.5, 6))),
            }
        )
    return rows


async def _seed(db, rows: list[dict[str, object]]) -> None:  # type: ignore[no-untyped-def]
    for row in rows:
        db.add(CrashOverlayReading(**row))
    await db.flush()


class TestEvaluate:
    async def test_records_a_decision_for_every_day_it_can_score(self, db) -> None:  # type: ignore[no-untyped-def]
        await _seed(db, _series())
        reading = await CrashOverlayService(db).evaluate()

        assert reading is not None
        assert 0.0 < reading.target_exposure <= 1.0
        assert reading.reason

        scored = (
            (
                await db.execute(
                    select(CrashOverlayReading).where(CrashOverlayReading.probability.is_not(None))
                )
            )
            .scalars()
            .all()
        )
        assert len(scored) > 0

    async def test_target_exposure_is_only_ever_full_or_defensive(self, db) -> None:  # type: ignore[no-untyped-def]
        params = OverlayParams(defensive=0.25)
        await _seed(db, _series())
        await CrashOverlayService(db).evaluate(params)

        values = {
            float(r.target_exposure)
            for r in (await db.execute(select(CrashOverlayReading))).scalars().all()
            if r.target_exposure is not None
        }
        assert values <= {1.0, 0.25}

    async def test_the_overlay_does_step_aside_at_least_once(self, db) -> None:  # type: ignore[no-untyped-def]
        """A rule that never fires is not conservative, it is broken.

        This is the assertion that would have caught the absolute-threshold bug,
        where a 3.4% base rate meant the model never emitted a probability above
        0.5 and the switch silently never fired.
        """
        await _seed(db, _series())
        await CrashOverlayService(db).evaluate(OverlayParams(sell_fraction=0.10))

        warnings = (
            (
                await db.execute(
                    select(CrashOverlayReading).where(CrashOverlayReading.is_warning.is_(True))
                )
            )
            .scalars()
            .all()
        )
        assert len(warnings) > 0

    async def test_firing_more_often_never_reduces_the_warnings(self, db) -> None:  # type: ignore[no-untyped-def]
        """The trigger is a percentile, so this must hold by construction."""
        await _seed(db, _series())
        service = CrashOverlayService(db)

        await service.evaluate(OverlayParams(sell_fraction=0.05))
        few = await _count_warnings(db)
        await service.evaluate(OverlayParams(sell_fraction=0.25))
        many = await _count_warnings(db)
        assert many >= few


class TestNoLookAhead:
    async def test_the_future_cannot_change_a_past_decision(self, db) -> None:  # type: ignore[no-untyped-def]
        """The load-bearing property of the whole design.

        Score a history, then append more days and score again. Every day's
        probability must be unchanged, because each is produced by a model
        fitted strictly on that day's own past. If appending the future moves a
        past probability, the walk-forward guard has a hole in it and every
        backtest number this project has produced is optimistic.
        """
        rows = _series()
        cut = len(rows) - 60
        await _seed(db, rows[:cut])
        service = CrashOverlayService(db)
        await service.evaluate()
        before = await _probabilities(db)

        await _seed(db, rows[cut:])
        await service.evaluate()
        after = await _probabilities(db)

        shared = set(before) & set(after)
        assert len(shared) > cf.CALIBRATION_MIN
        for day in shared:
            assert before[day] == pytest.approx(after[day], abs=1e-9), day

    async def test_replaying_the_same_history_is_stable(self, db) -> None:  # type: ignore[no-untyped-def]
        """The nightly job recomputes everything; it must not drift."""
        await _seed(db, _series())
        service = CrashOverlayService(db)
        first = await service.evaluate()
        second = await service.evaluate()

        assert first is not None and second is not None
        assert first.as_of == second.as_of
        assert first.target_exposure == second.target_exposure
        assert first.probability == pytest.approx(second.probability, abs=1e-12)


class TestThinHistory:
    async def test_refuses_to_decide_on_too_little_history(self, db) -> None:  # type: ignore[no-untyped-def]
        await _seed(db, _series(days=cf.CALIBRATION_MIN - 5))
        assert await CrashOverlayService(db).evaluate() is None

    async def test_early_days_carry_no_trigger_and_stay_fully_invested(self, db) -> None:  # type: ignore[no-untyped-def]
        await _seed(db, _series())
        await CrashOverlayService(db).evaluate()

        early = (
            (
                await db.execute(
                    select(CrashOverlayReading)
                    .order_by(CrashOverlayReading.as_of)
                    .limit(cf.CALIBRATION_MIN)
                )
            )
            .scalars()
            .all()
        )
        assert all(r.trigger is None for r in early)
        assert all(r.is_warning is False for r in early)
        assert all(r.target_exposure == Decimal("1") for r in early if r.target_exposure)


async def _count_warnings(db) -> int:  # type: ignore[no-untyped-def]
    rows = (
        (
            await db.execute(
                select(CrashOverlayReading).where(CrashOverlayReading.is_warning.is_(True))
            )
        )
        .scalars()
        .all()
    )
    return len(rows)


async def _probabilities(db) -> dict[date, float]:  # type: ignore[no-untyped-def]
    rows = (await db.execute(select(CrashOverlayReading))).scalars().all()
    return {r.as_of: float(r.probability) for r in rows if r.probability is not None}
