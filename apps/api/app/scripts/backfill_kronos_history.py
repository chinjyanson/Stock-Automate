"""Generate Kronos forecasts at historical dates, so the model can train on it.

    python -m app.scripts.backfill_kronos_history --instruments 40 --dates 20

**Why this is needed at all.** The nightly job records one forecast per
instrument per day, going forward. That serves the live strategy perfectly and
is useless for training, because a model fitted today needs to know what Kronos
would have said *on the day of each historical trade* — not what it says now.
Using the latter is a look-ahead leak of the worst kind: today's forecast has
already seen how a trade from two years ago turned out.

So this walks back through stored candles and asks the model what it would have
predicted, feeding it `series.head(i + 1)` and nothing after. That slicing is the
only thing standing between this and a fabricated result, so it is done in one
place and stated here.

**Forecasts are generated at the bars where trades actually open**, found by
replaying first. An even grid of dates would be the obvious choice and would
produce forecasts nothing can use: the fit joins Kronos to a trade by its
decision bar, and an evenly-spaced bar almost never coincides with one.

**The cost is the constraint, and it is severe.** A 32-path forecast takes about
twelve seconds. One per bar per instrument is not remotely affordable — 6,000
training trades would be twenty hours — so this samples `--dates` evaluation
points per instrument, spread across the usable history. That buys a sample big
enough to answer "does Kronos add anything on top of the price features?" while
staying inside an overnight run.

**Committed per instrument, which is the difference between resumable and not.**
`session_scope` commits once, on clean exit. A three-hour run inside one
transaction is three hours of work that a Ctrl-C throws away — which is exactly
what happened the first time this ran: it reported 352 forecasts written and
left zero rows behind. Each instrument is now committed as it completes, so an
interrupted run keeps everything up to the last finished name.

Idempotent on top of that: forecasts are upserted by (instrument, date, model,
horizon), so resuming re-does at most one instrument and a repeat run costs
nothing but time.
"""

from __future__ import annotations

import argparse
import asyncio
import time

import numpy as np

from app.backtest.engine import is_continuous, replay
from app.backtest.entries import EveryBarReader
from app.backtest.service import BacktestService
from app.data.store import CandleStore
from app.db import session_scope
from app.indicators.series import candles_to_series
from app.models.enums import Interval
from app.services.kronos import KronosService

#: Bars of context handed to the model. Its own ceiling is 512 for small/base.
CONTEXT_BARS = 512


async def _run(instruments_wanted: int, dates: int, history: int, samples: int | None) -> None:
    from app.config import get_settings
    from app.signals.kronos_client import KronosClient, is_available, repo_is_present

    if not (is_available() and repo_is_present()):
        print("Kronos is not set up. Run:  uv sync --extra kronos --extra dev")
        print("                     then:  python -m app.scripts.setup_kronos")
        return

    settings = get_settings()
    sample_count = samples if samples is not None else settings.kronos_sample_count
    horizon = settings.kronos_horizon_days

    client = KronosClient(settings.kronos_model)
    print(f"Model:    {client.model_name} on {client.device}")
    print(f"Sampling: {dates} dates x {instruments_wanted} instruments x {sample_count} paths")
    print(f"Horizon:  {horizon} trading days\n")

    reader = EveryBarReader()
    started = time.perf_counter()
    written = 0
    attempted = 0

    async with session_scope() as session:
        service = BacktestService(session)
        store = CandleStore(session)
        kronos = KronosService(session)
        # Ten candidates per wanted instrument, because most do not qualify:
        # Kronos needs 512 bars of context plus the horizon, and roughly one
        # scanner-ranked name in five carries that much history. A 3x multiplier
        # silently caps the run well below what was asked for.
        candidates = instruments_wanted * 10
        universe = await service.top_ranked_instruments(candidates)
        if not universe:
            universe = await service.instruments_with_history(candidates)

        used = 0
        for instrument in universe:
            if used >= instruments_wanted:
                break
            candles = await store.get_candles(
                instrument.id, Interval.D1, limit=history, closed_only=True
            )
            if len(candles) < CONTEXT_BARS + horizon + 10:
                continue
            series = candles_to_series(candles)
            if not is_continuous(series):
                continue
            timestamps = [c.timestamp for c in candles]

            # Already covered on a previous run: skip rather than regenerate.
            # The upsert would make a repeat *correct*, but it would cost the
            # same four minutes an instrument, which makes resuming pointless.
            existing = await kronos.count_for(
                instrument.id, model_name=client.model_name, horizon_days=horizon
            )
            if existing >= dates:
                used += 1
                print(
                    f"  {used:>3}/{instruments_wanted}  {instrument.name[:32]:<32} "
                    f"already has {existing} — skipped"
                )
                continue

            # **Forecast where the trades actually are**, not on an even grid.
            #
            # The obvious approach — sample N dates evenly across the history —
            # produces forecasts that no training row can use. The fit joins
            # Kronos to a trade by its *decision bar*, and the chance that one of
            # 22 evenly-spaced bars coincides with a decision bar out of ~700 is
            # close to zero. Every forecast would cost twelve seconds and match
            # nothing.
            #
            # So the replay runs first, purely to find where positions open, and
            # the forecasts are generated at exactly those bars. Every one is
            # then used by a training row.
            decisions = sorted(
                {t.entry_index - 1 for t in replay(series, reader).trades if t.entry_index > 0}
            )
            usable = [i for i in decisions if CONTEXT_BARS <= i < series.length - horizon - 1]
            if not usable:
                continue
            # Spread across the usable window rather than taking the first N: a
            # run of adjacent entries is very nearly one observation.
            if len(usable) > dates:
                picked = np.linspace(0, len(usable) - 1, num=dates, dtype=int)
                indices = [usable[j] for j in picked]
            else:
                indices = usable

            used += 1
            for i in indices:
                attempted += 1
                window = series.head(i + 1)  # nothing after bar i is visible
                forecast = client.forecast(
                    open_=window.open,
                    high=window.high,
                    low=window.low,
                    close=window.close,
                    volume=window.volume,
                    timestamps=timestamps[: i + 1],
                    horizon_days=horizon,
                    sample_count=sample_count,
                )
                if forecast is None:
                    continue
                await kronos.record(instrument.id, timestamps[i].date(), forecast)
                written += 1

            # Durable before moving on. Without this the whole run is one
            # transaction and an interrupt discards all of it.
            await session.commit()

            elapsed = time.perf_counter() - started
            rate = elapsed / max(attempted, 1)
            remaining = (instruments_wanted - used) * dates * rate
            print(
                f"  {used:>3}/{instruments_wanted}  {instrument.name[:32]:<32} "
                f"{written:>5} written  ~{remaining / 60:.0f} min left"
            )

    elapsed = time.perf_counter() - started
    print(f"\n{written:,} forecasts written in {elapsed / 60:.1f} min")
    print("Now refit with:  python -m app.scripts.fit_stock_model --kronos --save")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate historical Kronos forecasts.")
    parser.add_argument("--instruments", type=int, default=40, help="Instruments to cover.")
    parser.add_argument("--dates", type=int, default=20, help="Evaluation dates each.")
    parser.add_argument("--history", type=int, default=1200, help="Bars per instrument.")
    parser.add_argument(
        "--samples",
        type=int,
        help="Paths per forecast. Defaults to KRONOS_SAMPLE_COUNT (32).",
    )
    args = parser.parse_args()
    asyncio.run(
        _run(
            instruments_wanted=args.instruments,
            dates=args.dates,
            history=args.history,
            samples=args.samples,
        )
    )


if __name__ == "__main__":
    main()
