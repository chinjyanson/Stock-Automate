"""Measure the mean-reversion strategy against stored candles (§8).

    python -m app.scripts.backtest_strategy
    python -m app.scripts.backtest_strategy --sweep threshold
    python -m app.scripts.backtest_strategy --sweep trend

Every threshold in the strategy was reasoned about; none has been measured. This
answers the questions that were left open when they were chosen:

  * Does the entry have positive expectancy at all?
  * Is `entry_threshold` 0.60 better or worse than the 0.71 the old
    all-or-nothing rules effectively used?
  * Does the 200-day trend gate earn its place, and what does it cost in trades?

Reads the candle store only — no provider calls, no writes — so it is free to run
and safe to repeat. Results are in **R multiples**: one R is the distance from
entry to the initial stop, so expectancy is directly comparable across
instruments and position sizes.

Read the caveats in `app.backtest.engine` before trusting a number. In short:
this measures the *entry rule* on individual instruments, not the portfolio the
risk engine would actually have built, and PEAD vetoes cannot fire historically.
"""

from __future__ import annotations

import argparse
import asyncio

from app.backtest.engine import PortfolioResult, ReplayConfig
from app.backtest.service import BacktestService, InstrumentRun
from app.db import session_scope
from app.models.instrument import Instrument
from app.strategies.mean_reversion import EntryRules


def _print_result(label: str, pooled: PortfolioResult, *, instruments: int) -> None:
    combined = pooled.combined
    if combined.trade_count == 0:
        print(f"  {label:<28} no trades over {instruments} instrument(s)")
        return
    pf = combined.profit_factor
    print(
        f"  {label:<28} {combined.trade_count:>5} trades  "
        f"win {combined.win_rate:>6.1%}  "
        f"exp {combined.expectancy_r:>+6.2f}R  "
        f"total {combined.total_r:>+8.1f}R  "
        f"maxDD {combined.max_drawdown_r:>6.1f}R  "
        f"PF {('  n/a' if pf is None else f'{pf:>5.2f}')}  "
        f"held {combined.avg_bars_held:>5.1f}d"
    )


def _print_exits(pooled: PortfolioResult) -> None:
    breakdown = pooled.combined.exit_breakdown()
    if breakdown:
        parts = ", ".join(f"{k} {v}" for k, v in sorted(breakdown.items()))
        print(f"  {'exits':<28} {parts}")


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


async def _run(size: int, sweep: str | None, warmup: int | None) -> None:
    config = ReplayConfig(warmup_bars=warmup)

    async with session_scope() as session:
        service = BacktestService(session)
        instruments, source = await _universe(service, size)
        if not instruments:
            print("No instruments to replay. Ingest candles or run a scan first.")
            return
        print(f"Universe: {len(instruments)} instrument(s) from {source}")
        print(f"Warmup:   {warmup if warmup is not None else EntryRules().preferred_bars} bars\n")

        if sweep == "threshold":
            # 0.71 is where the old "band break AND RSI <= 35" rules sat on this
            # scale, so it is the honest comparison for the change that replaced
            # them — not an arbitrary point on a grid.
            print("Entry threshold sweep (0.71 = the old all-or-nothing equivalent)")
            for threshold in (0.45, 0.50, 0.55, 0.60, 0.65, 0.71, 0.80):
                pooled, runs = await service.run(
                    instruments, EntryRules(entry_threshold=threshold), config
                )
                _print_result(f"threshold {threshold:.2f}", pooled, instruments=len(instruments))
            return

        if sweep == "trend":
            print("200-day trend gate: does the falling-knife filter earn its place?")
            for label, rules in (
                ("gate off", EntryRules(trend_slope_min=-1.0)),
                ("gate on (default)", EntryRules(trend_slope_min=0.0)),
                ("gate strict", EntryRules(trend_slope_min=0.0005)),
            ):
                pooled, runs = await service.run(instruments, rules, config)
                _print_result(label, pooled, instruments=len(instruments))
            return

        rules = EntryRules()
        pooled, runs = await service.run(instruments, rules, config)
        print("Shipping configuration")
        _print_result("default", pooled, instruments=len(instruments))
        _print_exits(pooled)
        replayed = [r for r in runs if r.result.bars_replayed]
        print(
            f"\n  {len(replayed)} of {len(instruments)} instrument(s) had enough history to replay"
        )
        _print_top(runs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay the mean-reversion strategy.")
    parser.add_argument("--size", type=int, default=40, help="Instruments to replay.")
    parser.add_argument(
        "--sweep",
        choices=["threshold", "trend"],
        help="Compare configurations instead of reporting the shipping one.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=None,
        help=(
            "Bars to skip before trading. Defaults to 260, which is where the "
            "200-day trend gate becomes measurable. Lowering it buys sample size "
            "with a strategy that is not the one that ships."
        ),
    )
    args = parser.parse_args()
    asyncio.run(_run(size=args.size, sweep=args.sweep, warmup=args.warmup))


if __name__ == "__main__":
    main()
