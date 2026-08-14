"""Does the market behave differently around option expiry? (§15)

    python -m app.scripts.expiry_calendar

Dealer gamma cannot be backfilled — an option chain is published for today and
gone tomorrow — so the positioning story cannot be backtested directly. But it
makes a prediction about something we have thirty-five years of and which costs
nothing: **the calendar**.

The claim, in the form its proponents make it. Dealers carry large option
inventories. While those inventories are alive, hedging them pushes against
market moves — sold into strength, bought into weakness — which dampens
volatility and pins price near heavily-traded strikes. On the third Friday of
each month a large share of that inventory expires. The dampening goes with it,
and the days *after* expiry are left unprotected.

If that is true, it is visible in dates alone.

## The prediction, written before the first result was read

**H1 — falls cluster after expiry.** Days +1 to +5 after monthly expiry contain
a higher share of 2% falls than days -5 to -1 before it.

**H2 — volatility does the same.** Daily returns are more spread out in the five
days after than in the five before.

One primary comparison for each, both directional, both fixed here. The
per-offset table further down is a *description*, not twenty-one more tests: if
the primary comparison fails, a lone interesting offset in that table is what
one expects from twenty-one draws, not a discovery.

## Why the obvious statistics would lie here

**Falls arrive in clusters.** October 2008 contributes a dozen of them and they
all sit in two expiry cycles. Treating 9,000 days as 9,000 independent coin
flips would make almost any calendar pattern look overwhelming, because the
effective sample is closer to the number of *crises* than the number of days.
So every interval here comes from a bootstrap that resamples **whole expiry
cycles**, keeping each month's days together.

**The date is confounded.** The third Friday sits near month end, and four of
them are quarterly triple-witching dates with much larger inventories rolling
off. Both are reported separately rather than assumed away.

**The era is confounded too.** Index option volume grew enormously after about
2010 and again with same-day expiries after 2020. If the mechanism is real and
grew, the effect should be stronger recently. If it appears *only* recently and
only faintly, that is equally consistent with having gone looking for it. The
split is printed so the two readings can be told apart.
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from app.signals.crash_features import label_fall

#: Trading days either side of expiry that get an offset label.
WINDOW = 10

#: The primary comparison: this many days before against the same after.
SPAN = 5

#: Resamples behind every interval. Cycles, not days — see the module docstring.
RESAMPLES = 4000


@dataclass(frozen=True, slots=True)
class Season:
    """One calendar position relative to expiry, and what happened there."""

    offset: int
    days: int
    fall_rate: float
    mean_return: float
    volatility: float


def third_friday(year: int, month: int) -> date:
    """The monthly expiry date, before any holiday adjustment."""
    first = date(year, month, 1)
    # weekday(): Monday is 0, Friday is 4.
    first_friday = 1 + (4 - first.weekday()) % 7
    return date(year, month, first_friday + 14)


def expiry_positions(index: pd.DatetimeIndex) -> np.ndarray:
    """Bar numbers of each monthly expiry, moved back over holidays.

    Good Friday lands on the third Friday of April often enough to matter, and
    the market is shut. Taking the last trading day at or before the nominal
    date reproduces what the exchange does — expiry moves to the Thursday —
    without needing a holiday calendar.
    """
    out: list[int] = []
    stamps = index.values.astype("datetime64[D]")
    for year in range(int(index[0].year), int(index[-1].year) + 1):
        for month in range(1, 13):
            nominal = np.datetime64(third_friday(year, month), "D")
            if nominal < stamps[0] or nominal > stamps[-1]:
                continue
            position = int(np.searchsorted(stamps, nominal, side="right")) - 1
            if position >= 0:
                out.append(position)
    return np.array(sorted(set(out)), dtype=int)


def label_offsets(size: int, expiries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each bar: how far it sits from the nearest expiry, and which one.

    Cycles run about 21 trading days and the window is 21 wide, so overlap is
    rare but not impossible; where two expiries both reach a day, the nearer one
    claims it.
    """
    offset = np.full(size, 99, dtype=int)
    cycle = np.full(size, -1, dtype=int)
    closest = np.full(size, 999, dtype=int)
    for number, position in enumerate(expiries):
        for step in range(-WINDOW, WINDOW + 1):
            bar = position + step
            if 0 <= bar < size and abs(step) < closest[bar]:
                closest[bar] = abs(step)
                offset[bar] = step
                cycle[bar] = number
    return offset, cycle


def summarise(offset: np.ndarray, labels: np.ndarray, daily: np.ndarray) -> list[Season]:
    out = []
    for step in range(-WINDOW, WINDOW + 1):
        pick = (offset == step) & np.isfinite(labels) & np.isfinite(daily)
        if not pick.any():
            continue
        out.append(
            Season(
                offset=step,
                days=int(pick.sum()),
                fall_rate=float(labels[pick].mean()),
                mean_return=float(daily[pick].mean()),
                volatility=float(np.std(daily[pick], ddof=1)),
            )
        )
    return out


def cycle_bootstrap(
    values: np.ndarray,
    after: np.ndarray,
    before: np.ndarray,
    cycle: np.ndarray,
    *,
    statistic: str,
    seed: int = 0,
) -> tuple[float, float, float, float]:
    """Resample whole expiry cycles and re-measure the after-minus-before gap.

    Resampling days would treat October 2008 as a dozen independent pieces of
    evidence when it is one. Resampling cycles keeps a crisis month intact, so a
    pattern driven by two or three months is correctly reported as uncertain.
    """

    def measure(rows: np.ndarray) -> float:
        """The gap, over an arbitrary multiset of bars.

        Takes bar numbers rather than masks so that a resample — which repeats
        some cycles and omits others — can be expressed at all.
        """
        taken_after = rows[after[rows]]
        taken_before = rows[before[rows]]
        if taken_after.size < 2 or taken_before.size < 2:
            return float("nan")
        if statistic == "rate":
            return float(values[taken_after].mean() - values[taken_before].mean())
        return float(np.std(values[taken_after], ddof=1) - np.std(values[taken_before], ddof=1))

    numbers = np.unique(cycle[cycle >= 0])
    members = {int(n): np.flatnonzero(cycle == n) for n in numbers}
    point = measure(np.concatenate([members[int(n)] for n in numbers]))

    rng = np.random.default_rng(seed)
    draws = np.empty(RESAMPLES)
    for i in range(RESAMPLES):
        picked = rng.choice(numbers, numbers.size, replace=True)
        draws[i] = measure(np.concatenate([members[int(n)] for n in picked]))

    good = draws[np.isfinite(draws)]
    return (
        point,
        float(np.quantile(good, 0.05)),
        float(np.quantile(good, 0.95)),
        float((good > 0).mean()),
    )


def primary(
    name: str,
    offset: np.ndarray,
    cycle: np.ndarray,
    labels: np.ndarray,
    daily: np.ndarray,
) -> None:
    usable = np.isfinite(labels) & np.isfinite(daily) & (offset != 99)
    after = usable & (offset >= 1) & (offset <= SPAN)
    before = usable & (offset >= -SPAN) & (offset <= -1)

    print(f"\n  {name}")
    print(f"    {int(before.sum()):,} days before expiry, {int(after.sum()):,} after\n")

    gap, low, high, share = cycle_bootstrap(labels, after, before, cycle, statistic="rate")
    verdict = "supported" if low > 0 else ("contradicted" if high < 0 else "not supported")
    print(
        f"    H1  2% falls    before {labels[before].mean():.2%}   after "
        f"{labels[after].mean():.2%}   gap {gap:+.2%}"
    )
    print(
        f"        cycle-resampled 90% range {low:+.2%} to {high:+.2%}, "
        f"positive in {share:.0%}   -> {verdict}"
    )

    gap, low, high, share = cycle_bootstrap(daily, after, before, cycle, statistic="spread")
    verdict = "supported" if low > 0 else ("contradicted" if high < 0 else "not supported")
    print(
        f"    H2  volatility  before {np.std(daily[before], ddof=1):.3%}   after "
        f"{np.std(daily[after], ddof=1):.3%}   gap {gap:+.3%}"
    )
    print(
        f"        cycle-resampled 90% range {low:+.3%} to {high:+.3%}, "
        f"positive in {share:.0%}   -> {verdict}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the option-expiry calendar effect.")
    parser.add_argument("--since", default="1990-01-01")
    parser.add_argument("--until", default=None)
    parser.add_argument("--fall", type=float, default=0.02)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--era-split", default="2012-01-01")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    import yfinance as yf

    frame = yf.Ticker("^GSPC").history(period="max", interval="1d")
    frame = frame[frame.index >= args.since]
    if args.until:
        frame = frame[frame.index < args.until]
    close = frame["Close"].to_numpy(dtype=np.float64)
    index = pd.DatetimeIndex(frame.index.tz_localize(None)).normalize()
    daily = np.concatenate([[np.nan], close[1:] / close[:-1] - 1.0])

    labels = np.full(close.size, np.nan)
    for i in range(close.size - args.horizon):
        labels[i] = label_fall(close, i, fall=args.fall, horizon=args.horizon)

    expiries = expiry_positions(index)
    offset, cycle = label_offsets(close.size, expiries)

    known = np.isfinite(labels)
    print(f"\n  {close.size:,} trading days, {index[0].date()} to {index[-1].date()}")
    print(
        f"  {expiries.size:,} monthly expiries, "
        f"{int(labels[known].sum())} days followed by a {args.fall:.0%} fall "
        f"({labels[known].mean():.2%} of all days)"
    )
    print("\n  The prediction, fixed before looking: more falls and more volatility in")
    print(f"  the {SPAN} days after expiry than the {SPAN} days before.")

    primary("Everything", offset, cycle, labels, daily)

    recent = np.asarray(index >= pd.Timestamp(args.era_split))
    for name, pick in (
        (f"Before {args.era_split[:4]}", ~recent),
        (f"{args.era_split[:4]} onward", recent),
    ):
        masked = np.where(pick, offset, 99)
        primary(name, masked, np.where(pick, cycle, -1), labels, daily)

    # Triple witching: March, June, September, December, when far more
    # inventory rolls off at once. If the mechanism is real the effect should be
    # larger here than in an ordinary month.
    quarterly = np.zeros(close.size, dtype=bool)
    for position in expiries:
        if index[position].month in (3, 6, 9, 12):
            lo, hi = max(0, position - WINDOW), min(close.size, position + WINDOW + 1)
            quarterly[lo:hi] = True
    primary(
        "Quarterly expiries only",
        np.where(quarterly, offset, 99),
        np.where(quarterly, cycle, -1),
        labels,
        daily,
    )

    print("\n\n  Day by day, for description only — not twenty-one more tests.\n")
    print(f"  {'offset':>7} {'days':>6} {'2% falls':>10} {'mean return':>13} {'volatility':>12}")
    print("  " + "-" * 54)
    for season in summarise(offset, labels, daily):
        marker = "  <- expiry" if season.offset == 0 else ""
        print(
            f"  {season.offset:>+7} {season.days:>6} {season.fall_rate:>10.2%} "
            f"{season.mean_return:>13.3%} {season.volatility:>12.3%}{marker}"
        )
    print()


if __name__ == "__main__":
    main()
