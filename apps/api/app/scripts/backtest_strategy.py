"""Measure the fitted stock model against stored candles (§8).

    python -m app.scripts.backtest_strategy --sweep probability
    python -m app.scripts.backtest_strategy --sweep exits

The model decides entries now, so the one tunable left worth sweeping is the
probability at which a signal is worth acting on. Everything the old harness
swept — band weights, RSI weights, the entry threshold, the reward:risk gate —
is either fitted or gone.

Reads the candle store and the active model; no provider calls, no writes, so it
is free to run and safe to repeat. Results are in **R multiples**: one R is the
distance from entry to the initial stop, so expectancy is comparable across
instruments and position sizes.

Read the caveats in `app.backtest.engine` before trusting a number. In short:
this measures the *entry* on individual instruments, not the portfolio the risk
engine would have built, and neither PEAD nor insider vetoes can fire
historically — so the live strategy is slightly more selective than this.
"""

from __future__ import annotations

import argparse
import asyncio

from sqlalchemy.ext.asyncio import AsyncSession

from app.backtest.engine import PortfolioResult, ReplayConfig
from app.backtest.entries import EveryBarReader, ModelReader
from app.backtest.service import BacktestService, InstrumentRun
from app.db import session_scope
from app.models.enums import StrategyKind
from app.models.instrument import Instrument
from app.models_ml.logistic import FittedModel
from app.services.strategy_model import StrategyModelService

#: Swept wide on purpose. A winning bucket at either end of a range is the least
#: trustworthy result in it — there is no neighbour beyond it to say whether the
#: edge continues or reverses — so this extends past where anything is expected
#: to be interesting, and a winner at an end is a reason to extend it further
#: rather than to adopt it.
PROBABILITY_GRID = (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85)


def _standard_error_r(pooled: PortfolioResult) -> float | None:
    """Standard error of the mean R, so a bucket can be read with its noise.

    Every comparison in the first run of this harness came back inside its own
    error bars, and reporting expectancy without this invites reading a rounding
    difference as a finding.
    """
    trades = pooled.combined.trades
    n = len(trades)
    if n < 2:
        return None
    mean = sum(t.r_multiple for t in trades) / n
    variance = sum((t.r_multiple - mean) ** 2 for t in trades) / (n - 1)
    return float((variance / n) ** 0.5)


def _print_result(label: str, pooled: PortfolioResult, *, instruments: int) -> None:
    combined = pooled.combined
    if combined.trade_count == 0:
        print(f"  {label:<24} no trades over {instruments} instrument(s)")
        return
    pf = combined.profit_factor
    se = _standard_error_r(pooled)
    # +/- two standard errors. A bucket whose interval spans zero has not
    # demonstrated anything, however good its point estimate looks.
    band = f"+/-{2 * se:.2f}" if se is not None else "  n/a"
    verdict = "" if se is None else ("  *" if abs(combined.expectancy_r) > 2 * se else "")
    # The median sits beside the mean on purpose. The mean is what compounds,
    # but it is also what a single broken series can capture; when the two
    # disagree wildly the mean is describing one trade, not a strategy.
    concentration = combined.largest_trade_share
    alarm = "  !! one trade is " + f"{concentration:.0%} of gross R" if concentration > 0.10 else ""
    print(
        f"  {label:<24} {combined.trade_count:>5} trades  "
        f"win {combined.win_rate:>6.1%}  "
        f"exp {combined.expectancy_r:>+6.2f}R {band}  "
        f"med {combined.median_r:>+5.2f}R  "
        f"maxDD {combined.max_drawdown_r:>6.1f}R  "
        f"PF {('  n/a' if pf is None else f'{pf:>5.2f}')}  "
        f"held {combined.avg_bars_held:>4.1f}d{verdict}{alarm}"
    )


def _print_exits(pooled: PortfolioResult) -> None:
    combined = pooled.combined
    rows = combined.r_by_exit()
    if not rows:
        return
    print("\n  where the money goes, by exit")
    for reason, (count, mean_r) in rows.items():
        share = count / combined.trade_count
        print(f"    {reason:<10} {count:>5} ({share:>5.1%})   mean {mean_r:>+6.2f}R")
    print(
        f"    {'':<10} avg win {combined.avg_win_r:>+5.2f}R   "
        f"avg loss {-combined.avg_loss_r:>+5.2f}R   "
        f"payoff {(combined.avg_win_r / combined.avg_loss_r if combined.avg_loss_r else 0):.2f}"
    )


def _print_top(runs: list[InstrumentRun], limit: int = 5) -> None:
    scored = [r for r in runs if r.result.trade_count]
    if not scored:
        return
    scored.sort(key=lambda r: r.result.total_r, reverse=True)
    print("\n  best / worst instruments by total R")
    for run in scored[:limit]:
        print(
            f"    {run.name[:34]:<34} {run.result.trade_count:>3} trades "
            f"{run.result.total_r:>+7.1f}R"
        )
    if len(scored) > limit:
        print("    ...")
        for run in scored[-limit:]:
            print(
                f"    {run.name[:34]:<34} {run.result.trade_count:>3} trades "
                f"{run.result.total_r:>+7.1f}R"
            )


async def _universe(service: BacktestService, size: int) -> tuple[list[Instrument], str]:
    ranked = await service.top_ranked_instruments(size)
    if ranked:
        return ranked, "scanner top-ranked"
    fallback = await service.instruments_with_history(size)
    return fallback, "stored history (no scan yet — unranked sample)"


async def _load_model(session: AsyncSession) -> FittedModel | None:
    model = await StrategyModelService(session).active(StrategyKind.LOGISTIC_STOCK)
    if model is None:
        print(
            "No active model for logistic_stock. Fit one first:\n"
            "  python -m app.scripts.fit_stock_model --save"
        )
    return model


async def _run(size: int, sweep: str, warmup: int | None) -> None:
    config = ReplayConfig(warmup_bars=warmup)
    base_reader = EveryBarReader()
    effective_warmup = warmup if warmup is not None else base_reader.preferred_bars
    # One eligibility bar for every bucket in the run. Otherwise two buckets
    # could be measured over different instruments, and the one that happened to
    # admit a few extra thinly-covered names would differ for a reason that has
    # nothing to do with the threshold under test.
    min_bars = max(effective_warmup, base_reader.required_bars) + 1

    async with session_scope() as session:
        service = BacktestService(session)
        instruments, source = await _universe(service, size)
        if not instruments:
            print("No instruments to replay. Ingest candles or run a scan first.")
            return
        model = await _load_model(session)
        if model is None:
            return

        print(f"Universe:   {len(instruments)} instrument(s) from {source}")
        print(f"Model:      {model.n_observations} observations, confirm AUC {model.auc:.3f}")
        print(f"Features:   {', '.join(model.feature_names)}")
        print(f"Warmup:     {effective_warmup} bars")
        print(f"Eligible:   >= {min_bars} stored bars, applied identically to every bucket")
        print("Excluded:   series with an unadjusted split (see MAX_DAILY_PRICE_RATIO)")
        print("Marked *:   expectancy is more than two standard errors from zero\n")

        if sweep == "exits":
            print("Exit variants at the shipping probability")
            reader = ModelReader(model=model, threshold=0.55)
            for label, cfg in (
                ("band target", ReplayConfig(warmup_bars=warmup)),
                ("fixed 1.5R", ReplayConfig(warmup_bars=warmup, fixed_target_r=1.5)),
                ("fixed 2.0R", ReplayConfig(warmup_bars=warmup, fixed_target_r=2.0)),
                ("hold 20d", ReplayConfig(warmup_bars=warmup, hold_bars=20)),
                ("trailing stop", ReplayConfig(warmup_bars=warmup, trail_stops=True)),
            ):
                pooled, _, _ = await service.run(instruments, reader, cfg, min_bars=min_bars)
                _print_result(label, pooled, instruments=len(instruments))
            return

        # Default: the one tunable the model leaves to be chosen.
        print("Entry probability sweep, fitted on one half and confirmed on the other")
        for fold in ("fit", "confirm"):
            half = service.split(instruments, fold=fold)
            print(f"\n  {fold.upper()} fold — {len(half)} instruments")
            for threshold in PROBABILITY_GRID:
                reader = ModelReader(model=model, threshold=threshold)
                pooled, _, _ = await service.run(half, reader, config, min_bars=min_bars)
                _print_result(f"P >= {threshold:.2f}", pooled, instruments=len(half))
        print(
            "\n  Two rules for reading this. A threshold that wins on the fit fold and\n"
            "  not on the confirm fold was fitted to noise. And a winner sitting at\n"
            f"  either end of {PROBABILITY_GRID[0]:.2f}-{PROBABILITY_GRID[-1]:.2f} is a\n"
            "  reason to extend the range and re-sweep, not to adopt it — an end\n"
            "  bucket has no neighbour to show whether the edge continues or reverses."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure the fitted stock model.")
    parser.add_argument("--size", type=int, default=400, help="Instruments to replay.")
    parser.add_argument(
        "--sweep",
        choices=("probability", "exits"),
        default="probability",
        help="Which comparison to run.",
    )
    parser.add_argument("--warmup", type=int, help="Override the warmup bar count.")
    args = parser.parse_args()
    asyncio.run(_run(size=args.size, sweep=args.sweep, warmup=args.warmup))


if __name__ == "__main__":
    main()
