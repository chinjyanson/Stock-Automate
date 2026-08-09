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
from app.backtest.service import BacktestService, InstrumentRun, Skipped
from app.db import session_scope
from app.models.instrument import Instrument
from app.strategies.mean_reversion import EntryRules


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
    baseline = EntryRules()
    effective_warmup = warmup if warmup is not None else baseline.preferred_bars
    # One eligibility bar for every configuration in the run, computed from the
    # baseline rules rather than from each bucket's own. Otherwise two buckets
    # could be measured over different instruments, and the one that happened to
    # admit a few extra thinly-covered names would differ for a reason that has
    # nothing to do with the rule under test.
    min_bars = max(effective_warmup, baseline.required_bars) + 1

    async with session_scope() as session:
        service = BacktestService(session)
        instruments, source = await _universe(service, size)
        if not instruments:
            print("No instruments to replay. Ingest candles or run a scan first.")
            return

        async def measure(
            rules: EntryRules,
        ) -> tuple[PortfolioResult, list[InstrumentRun], Skipped]:
            return await service.run(instruments, rules, config, min_bars=min_bars)

        print(f"Universe:   {len(instruments)} instrument(s) from {source}")
        print(f"Warmup:     {effective_warmup} bars")
        print(f"Eligible:   >= {min_bars} stored bars, applied identically to every bucket")
        print("Excluded:   series with an unadjusted split (see MAX_DAILY_PRICE_RATIO)")
        print("Marked *:   expectancy is more than two standard errors from zero\n")

        if sweep == "threshold":
            # 0.71 is where the old "band break AND RSI <= 35" rules sat on this
            # scale, so it is the honest comparison for the change that replaced
            # them — not an arbitrary point on a grid.
            # Extended past 0.80 on both sides on purpose. The first run of this
            # sweep stopped at 0.80 and 0.80 was the only positive bucket — which
            # is unreadable, because a boundary bucket has no neighbour to show
            # whether the apparent edge continues or reverses. Any bucket that
            # looks good here should be interrogated the same way: if it sits at
            # an end, extend the range rather than believing it.
            print("Entry threshold sweep (0.71 = the old all-or-nothing equivalent)")
            for threshold in (0.35, 0.45, 0.50, 0.55, 0.60, 0.65, 0.71, 0.80, 0.88, 0.94):
                pooled, _, _ = await measure(EntryRules(entry_threshold=threshold))
                _print_result(f"threshold {threshold:.2f}", pooled, instruments=len(instruments))
            print(
                "\n  A bucket at either end of this range is the least trustworthy result\n"
                "  in it: there is no neighbour beyond it to show whether the apparent\n"
                "  edge continues or reverses. Extend the sweep before believing one."
            )
            return

        if sweep == "trend":
            print("200-day trend gate: does the falling-knife filter earn its place?")
            for label, rules in (
                ("gate off", EntryRules(trend_slope_min=-1.0)),
                ("gate on (default)", EntryRules(trend_slope_min=0.0)),
                ("gate strict", EntryRules(trend_slope_min=0.0005)),
            ):
                pooled, _, _ = await measure(rules)
                _print_result(label, pooled, instruments=len(instruments))
            print(
                "\n  Read with care: these trade sets are overlapping, not nested. Only one\n"
                "  position is held at a time, so a looser gate is in a trade more often and\n"
                "  therefore *misses* entries a stricter gate would have taken. The buckets\n"
                "  are not the same trades plus extras."
            )
            return

        pooled, runs, skipped = await measure(baseline)
        print("Shipping configuration")
        _print_result("default", pooled, instruments=len(instruments))
        _print_exits(pooled)
        replayed = [r for r in runs if r.result.bars_replayed]
        bars = sum(r.result.bars_replayed for r in runs)
        print(f"\n  {len(replayed)} of {len(instruments)} instrument(s) replayed, {bars:,} bars")
        print(
            f"  skipped: {skipped.too_short} too short, "
            f"{skipped.discontinuous} discontinuous (unadjusted split), "
            f"{skipped.failed} failed"
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
