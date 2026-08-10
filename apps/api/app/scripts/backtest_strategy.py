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

        if sweep == "norsi":
            # Does removing RSI help? It is the component the forward-return test
            # rated highest (+3.08% at 20d over 11,000 observations), so this is
            # a check on that rather than a foregone conclusion. Weights are
            # relative and renormalise, so zeroing one simply redistributes.
            print("With and without RSI, on both folds")
            entry_variants = [
                ("all three", EntryRules()),
                ("no RSI", EntryRules(weight_rsi=0.0)),
                ("no bands", EntryRules(weight_band=0.0)),
                ("RSI only", EntryRules(weight_band=0.0, weight_discount=0.0)),
                ("bands only", EntryRules(weight_rsi=0.0, weight_discount=0.0)),
            ]
            for fold in ("fit", "confirm"):
                half = service.split(instruments, fold=fold)
                print(f"\n  {fold.upper()} fold — {len(half)} instruments")
                for label, entry_rules in entry_variants:
                    pooled, _, _ = await service.run(half, entry_rules, config, min_bars=min_bars)
                    _print_result(label, pooled, instruments=len(half))
            return

        if sweep == "stopcost":
            # The forward-return test says RSI <= 30 precedes +3.08% at 20 days,
            # yet holding 20 days does not pay. The obvious explanation is that
            # the average includes paths that first fall far enough to trigger any
            # sensible stop — the edge would then be real but unharvestable with a
            # stop in place. Widening the stop toward "none" tests exactly that.
            print("Is the 20-day edge reachable with a stop in the way?")
            for label, mult in (
                ("stop 2x ATR", 2.0),
                ("stop 3x ATR", 3.0),
                ("stop 5x ATR", 5.0),
                ("stop 10x ATR", 10.0),
                ("stop 25x (≈none)", 25.0),
            ):
                for fold in ("fit", "confirm"):
                    half = service.split(instruments, fold=fold)
                    pooled, _, _ = await service.run(
                        half,
                        EntryRules(atr_stop_multiplier=mult),
                        ReplayConfig(
                            warmup_bars=warmup,
                            hold_bars=20,
                            atr_stop_multiplier=mult,
                            trail_stops=False,
                        ),
                        min_bars=min_bars,
                    )
                    _print_result(f"{label} [{fold}]", pooled, instruments=len(half))
            print(
                "\n  R is measured against each run's own stop, so these are not directly\n"
                "  comparable in R — read the win rate and profit factor instead. A wide\n"
                "  stop is not a proposal; it is a diagnostic."
            )
            return

        if sweep == "fixes":
            # The three fixes the forward-return test pointed at, applied one at a
            # time and then together, on both folds. The measured edge builds to
            # ~20 days while the band exit fires at ~7, so "hold" is the headline;
            # "less jumpy" stops noise ejecting the position before the move
            # arrives; "good odds" refuses the sub-1:1 setups.
            variants = [
                ("0. shipping today", ReplayConfig(warmup_bars=warmup), EntryRules()),
                (
                    "1. hold 20d",
                    ReplayConfig(warmup_bars=warmup, hold_bars=20),
                    EntryRules(),
                ),
                (
                    "2. + less jumpy",
                    ReplayConfig(
                        warmup_bars=warmup,
                        hold_bars=20,
                        atr_stop_multiplier=3.0,
                        trail_stops=False,
                    ),
                    EntryRules(atr_stop_multiplier=3.0),
                ),
                (
                    "3. + good odds",
                    ReplayConfig(
                        warmup_bars=warmup,
                        hold_bars=20,
                        atr_stop_multiplier=3.0,
                        trail_stops=False,
                    ),
                    EntryRules(atr_stop_multiplier=3.0, min_reward_risk=1.0),
                ),
                (
                    "4. + choosier entry",
                    ReplayConfig(
                        warmup_bars=warmup,
                        hold_bars=20,
                        atr_stop_multiplier=3.0,
                        trail_stops=False,
                    ),
                    EntryRules(
                        atr_stop_multiplier=3.0,
                        min_reward_risk=1.0,
                        entry_threshold=0.80,
                    ),
                ),
                (
                    "5. hold 40d, all fixes",
                    ReplayConfig(
                        warmup_bars=warmup,
                        hold_bars=40,
                        atr_stop_multiplier=3.0,
                        trail_stops=False,
                    ),
                    EntryRules(
                        atr_stop_multiplier=3.0,
                        min_reward_risk=1.0,
                        entry_threshold=0.80,
                    ),
                ),
            ]
            print("The three fixes, cumulative, on both folds")
            for fold in ("fit", "confirm"):
                half = service.split(instruments, fold=fold)
                print(f"\n  {fold.upper()} fold — {len(half)} instruments")
                for label, cfg, entry_rules in variants:
                    pooled, _, _ = await service.run(half, entry_rules, cfg, min_bars=min_bars)
                    _print_result(label, pooled, instruments=len(half))
            print(
                "\n  A fix that helps on one fold and not the other was fitted to noise.\n"
                "  The stop is still live throughout — a fixed hold waits for the move,\n"
                "  it does not sit through an unlimited loss."
            )
            return

        if sweep == "rr":
            # The one change with a mechanical argument behind it: the median
            # setup risks 1.0 to make 0.90, which at a 50% win rate loses by
            # arithmetic. Fitted on one half of the instruments and confirmed on
            # the other, because selecting the best gate from the same trades it
            # was measured on is how a backtest manufactures an edge.
            print("Reward:risk gate, fitted on one half and confirmed on the other")
            for fold in ("fit", "confirm"):
                half = service.split(instruments, fold=fold)
                print(f"\n  {fold.upper()} fold — {len(half)} instruments")
                for minimum in (0.0, 0.8, 1.0, 1.2, 1.5, 2.0):
                    pooled, _, _ = await service.run(
                        half,
                        EntryRules(min_reward_risk=minimum),
                        config,
                        min_bars=min_bars,
                    )
                    label = "off" if minimum == 0.0 else f">= {minimum:.1f}"
                    _print_result(f"R:R {label}", pooled, instruments=len(half))
            print(
                "\n  A gate that helps on the fit fold and not on the confirm fold was\n"
                "  fitted to noise. Only a change that holds on both is worth adopting."
            )
            return

        if sweep == "target":
            # The band target is a 20-day average that follows price down while a
            # position waits, so it drifts toward the entry: target exits average
            # +0.73R against a structural ~1.3R. A target frozen at entry pays
            # what the setup promised. This is a mechanical fix for a measured
            # defect, not a knob turned until the number improved.
            print("Profit target: moving band vs frozen at entry")
            for label, cfg in (
                ("band (ships)", ReplayConfig(warmup_bars=warmup)),
                ("fixed 1.0R", ReplayConfig(warmup_bars=warmup, fixed_target_r=1.0)),
                ("fixed 1.5R", ReplayConfig(warmup_bars=warmup, fixed_target_r=1.5)),
                ("fixed 2.0R", ReplayConfig(warmup_bars=warmup, fixed_target_r=2.0)),
                ("fixed 3.0R", ReplayConfig(warmup_bars=warmup, fixed_target_r=3.0)),
                (
                    "fixed 1.5R, stop 3x",
                    ReplayConfig(warmup_bars=warmup, fixed_target_r=1.5, atr_stop_multiplier=3.0),
                ),
                (
                    "fixed 1.5R, no trail",
                    ReplayConfig(warmup_bars=warmup, fixed_target_r=1.5, trail_stops=False),
                ),
            ):
                pooled, _, _ = await service.run(instruments, baseline, cfg, min_bars=min_bars)
                _print_result(label, pooled, instruments=len(instruments))
            return

        if sweep == "exits":
            # The entry offers ~1.3:1 at a 50% win rate, which should return
            # about +0.15R. It returns +0.01R, and the implied average win is
            # only 1.04x the average loss — so wins are being truncated. These
            # three knobs already exist in ReplayConfig; nothing new is being
            # invented, which is what makes this the honest first experiment.
            print("Exit mechanics: where is the 0.14R going?")
            exit_variants = [
                ("baseline", ReplayConfig(warmup_bars=warmup)),
                ("no trailing stop", ReplayConfig(warmup_bars=warmup, trail_stops=False)),
                (
                    "stop 1.5x ATR",
                    ReplayConfig(warmup_bars=warmup, atr_stop_multiplier=1.5),
                ),
                (
                    "stop 3.0x ATR",
                    ReplayConfig(warmup_bars=warmup, atr_stop_multiplier=3.0),
                ),
                (
                    "stop 3.0x, no trail",
                    ReplayConfig(warmup_bars=warmup, atr_stop_multiplier=3.0, trail_stops=False),
                ),
                (
                    "give up after 10d",
                    ReplayConfig(warmup_bars=warmup, max_holding_bars=10),
                ),
                (
                    "give up after 20d",
                    ReplayConfig(warmup_bars=warmup, max_holding_bars=20),
                ),
            ]
            for label, variant in exit_variants:
                pooled, _, _ = await service.run(instruments, baseline, variant, min_bars=min_bars)
                _print_result(label, pooled, instruments=len(instruments))
            print(
                "\n  Trailing is the prime suspect: it ratchets on every bar, so a position\n"
                "  that dips before reverting is closed at a fraction of its target. The\n"
                "  live StopService trails identically, so anything found here is real."
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
        trades = pooled.combined.trades
        if trades:
            rr = sorted(t.reward_risk for t in trades)
            n = len(rr)
            print("Reward:risk actually offered at entry (target vs stop distance)")
            for label, idx in (("p10", n // 10), ("median", n // 2), ("p90", 9 * n // 10)):
                print(f"    {label:<8} {rr[min(idx, n - 1)]:.2f}")
            # Split the same trades by their entry R:R. If selecting for a better
            # ratio does not lift expectancy, an R:R filter is not the fix.
            print("\n  outcome by entry reward:risk")
            for lo, hi in ((0.0, 1.0), (1.0, 1.5), (1.5, 2.5), (2.5, 1e9)):
                bucket = [t for t in trades if lo <= t.reward_risk < hi]
                if len(bucket) < 30:
                    continue
                mean = sum(t.r_multiple for t in bucket) / len(bucket)
                wins = sum(1 for t in bucket if t.is_win) / len(bucket)
                var = sum((t.r_multiple - mean) ** 2 for t in bucket) / (len(bucket) - 1)
                se = (var / len(bucket)) ** 0.5
                tag = "  *" if abs(mean) > 2 * se else ""
                hi_label = "+" if hi > 1e8 else f"{hi:.1f}"
                print(
                    f"    R:R {lo:.1f}-{hi_label:<4} {len(bucket):>5} trades  "
                    f"win {wins:>5.1%}  exp {mean:>+6.2f}R +/-{2 * se:.2f}{tag}"
                )
            print()
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
        choices=["threshold", "trend", "exits", "target", "rr", "fixes", "stopcost", "norsi"],
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
