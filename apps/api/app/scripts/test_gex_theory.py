"""The dealer-gamma test, declared now and answered later. (§16)

    python -m app.scripts.test_gex_theory

Dealer gamma cannot be backfilled. An option chain is published for today and
per-strike open interest is gone once the day passes, so unlike everything else
in this system there is no history to test against — only a history to
accumulate. `worker.jobs.index_options` records one row a night; this script is
what eventually reads them.

It is written **now, while there is no data**, and that is the point. Every
hypothesis, every threshold and every stopping rule below was fixed before a
single row existed to look at. A test specified after seeing the data is not a
test, and this project has already been caught twice by selection windows too
small to choose on — once in the Kronos probe, once in the feature ablation.
Writing the analysis first is the only defence that actually works.

## The hypotheses, in the order they should be believed

**H1 — the core claim.** When dealers are short gamma (`gamma_tilt` negative)
they hedge *with* the market, amplifying moves. So more negative tilt should be
followed by more 2% falls. Scored as AUC of `-gamma_tilt` against the
production label.

**H2 — the mechanism, not the outcome.** Short gamma should show up as
amplified movement whether or not it crosses 2%. Scored as the correlation
between `-gamma_tilt` and the next day's absolute return. H2 is the more
sensitive test: it uses every day rather than only the rare ones, so it will
have an answer long before H1 does.

**H3 — does it add anything?** The detector already scores 0.798 held out, and
the ablation showed that is essentially VIX. Gamma tilt is one of the few
candidate signals that is *not* a volatility measure, which is the whole reason
it is interesting. Scored as the paired AUC gap of (detector + tilt) against the
detector alone — the same `auc_gap` used everywhere else here, because two
models judged on the same rare days share most of their noise.

**H4 — charm, on the same three tests.** Weaker prior: charm is a second-order
Greek and its aggregate effect is more theory than observation.

## The stopping rule, which matters more than the hypotheses

The script refuses to call a result before it can. With a 3.5% base rate, a
year of recording holds about nine falls, and nine events cannot distinguish a
real detector from a coin toss — the interval on an AUC computed from nine
positives spans most of the unit interval.

So every run prints **how much more data is needed**, from the Hanley-McNeil
standard error, and declines to render a verdict until there is enough. That
number is unwelcome and it is the honest one: this is a multi-year programme,
and a script that produced an encouraging figure after three months would be
worse than useless.

**And the kill rule, fixed here.** If, once there is enough data, H1 and H2 both
fail, the feature comes out. Written down now because a belief with no
disconfirming condition is not a hypothesis, and because the moment to decide
what would change your mind is before you have anything invested in the answer.
"""

from __future__ import annotations

import argparse
import asyncio
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import select

from app.db import session_scope
from app.models.index_options import IndexOptionsSnapshot
from app.models_ml.logistic import auc, auc_gap
from app.signals.crash_features import label_fall

#: AUC we would consider worth having. Below this the feature cannot pay for a
#: data dependency, so it is the effect the power calculation is sized on.
WORTH_HAVING = 0.60

#: Minimum events before any verdict is printed, whatever the power says.
#: Two dozen is not many; it is the floor below which the interval is wider
#: than the entire range of interesting answers.
MIN_EVENTS = 25

#: Long-run share of days followed by a 2% fall, from 1990-2026. Used only to
#: project how long the wait will be before any of our own days have accrued —
#: an estimate from a handful of recorded days would be worse than the history.
ASSUMED_BASE_RATE = 0.035


@dataclass(frozen=True, slots=True)
class Readiness:
    days: int
    events: int
    needed_events: int
    ready: bool

    @property
    def needed_days(self) -> int:
        """Trading days still to record before a verdict is possible.

        Falls back to the long-run base rate until enough of our own days have
        accrued to estimate one. Projecting from three recorded days would
        produce either zero or infinity, and both would read as information.
        """
        if self.ready:
            return 0
        rate = (
            self.events / self.days if self.days >= 200 and self.events > 0 else ASSUMED_BASE_RATE
        )
        return max(0, math.ceil((self.needed_events - self.events) / max(rate, 1e-9)))


def auc_standard_error(area: float, n_pos: int, n_neg: int) -> float:
    """Hanley and McNeil's standard error for an AUC.

    Used rather than a bootstrap because this has to answer "how much more data
    do I need" for a sample size we do not have yet, which a bootstrap of the
    data we do have cannot do.
    """
    if n_pos < 1 or n_neg < 1:
        return float("inf")
    q1 = area / (2.0 - area)
    q2 = 2.0 * area * area / (1.0 + area)
    variance = (
        area * (1.0 - area) + (n_pos - 1) * (q1 - area * area) + (n_neg - 1) * (q2 - area * area)
    ) / (n_pos * n_neg)
    return math.sqrt(max(variance, 0.0))


def events_needed(base_rate: float, *, area: float = WORTH_HAVING) -> int:
    """Positives required before an AUC of `area` clears 0.5 at 90% confidence."""
    if base_rate <= 0.0:
        return MIN_EVENTS
    for n_pos in range(5, 5000):
        n_neg = int(n_pos * (1.0 - base_rate) / base_rate)
        if 1.645 * auc_standard_error(area, n_pos, n_neg) < (area - 0.5):
            return max(n_pos, MIN_EVENTS)
    return MIN_EVENTS


async def load_snapshots() -> pd.DataFrame:
    async with session_scope() as session:
        rows = (
            (
                await session.execute(
                    select(IndexOptionsSnapshot).order_by(IndexOptionsSnapshot.as_of)
                )
            )
            .scalars()
            .all()
        )
    return pd.DataFrame(
        [
            {
                "as_of": pd.Timestamp(row.as_of),
                "symbol": row.symbol,
                "gamma_tilt": None if row.gamma_tilt is None else float(row.gamma_tilt),
                "charm_tilt": None if row.charm_tilt is None else float(row.charm_tilt),
                "skew_25delta": None if row.skew_25delta is None else float(row.skew_25delta),
                "atm_iv": None if row.atm_iv is None else float(row.atm_iv),
                "contracts_used": row.contracts_used or 0,
            }
            for row in rows
        ]
    )


def market(since: str) -> pd.DataFrame:
    import yfinance as yf

    frame = yf.Ticker("^GSPC").history(period="max", interval="1d")
    frame = frame[frame.index >= since]
    close = frame["Close"].to_numpy(dtype=np.float64)
    index = pd.DatetimeIndex(frame.index.tz_localize(None)).normalize()
    labels = np.full(close.size, np.nan)
    for i in range(close.size - 1):
        labels[i] = label_fall(close, i, fall=0.02, horizon=1)
    forward = np.full(close.size, np.nan)
    forward[:-1] = close[1:] / close[:-1] - 1.0
    return pd.DataFrame({"as_of": index, "label": labels, "forward": forward})


def report_readiness(frame: pd.DataFrame, detector: Path | None) -> Readiness:
    usable = frame.dropna(subset=["gamma_tilt", "label"])
    days = int(usable.shape[0])
    events = int(usable["label"].sum()) if days else 0
    base = events / days if days else 0.035
    needed = events_needed(base if base > 0 else 0.035)
    state = Readiness(days=days, events=events, needed_events=needed, ready=events >= needed)

    recorded = int(frame.shape[0])
    print(f"\n  {recorded:,} rows recorded; {days:,} of them have a known answer")
    if recorded and not days:
        print("  (the most recent row is always pending — its label depends on a")
        print("   close that has not happened yet)")
    print(f"  {events} of those were followed by a 2% fall")
    print(f"\n  To distinguish an AUC of {WORTH_HAVING:.2f} from a coin toss at 90%")
    print(f"  confidence takes about {needed} such events.")
    if state.ready:
        print("  There are enough. Verdicts below.\n")
    else:
        years = state.needed_days / 252.0
        print(
            f"  There are {events}. About {state.needed_days:,} more trading days "
            f"({years:.1f} years) to go.\n"
        )
        print("  No verdict is printed below that line, deliberately. A number")
        print("  computed from this much data would look like an answer.\n")
    return state


def evaluate(frame: pd.DataFrame, state: Readiness) -> None:
    usable = frame.dropna(subset=["gamma_tilt", "label", "forward"])
    if usable.shape[0] < 30:
        print("  Too few rows even to describe. Come back once the recorder has run.\n")
        return

    labels = usable["label"].to_numpy(dtype=float)
    tilt = usable["gamma_tilt"].to_numpy(dtype=float)
    moves = np.abs(usable["forward"].to_numpy(dtype=float))

    print(f"  {'hypothesis':<44} {'reading':>10} {'verdict':>22}")
    print("  " + "-" * 78)

    def line(name: str, value: float, ready: bool, positive_is_support: bool = True) -> None:
        if not ready:
            verdict = "not enough data yet"
        elif (value > 0.5) if positive_is_support else (value < 0.5):
            verdict = "supported"
        else:
            verdict = "not supported"
        print(f"  {name:<44} {value:>10.3f} {verdict:>22}")

    line("H1  short gamma -> more 2% falls", auc(labels, -tilt), state.ready)

    # H2 uses every day, not only the rare ones, so it earns an answer sooner.
    # Spearman rather than Pearson: the claim is about ordering, and a single
    # crash day would otherwise dominate a linear correlation.
    rank_tilt = pd.Series(-tilt).rank().to_numpy()
    rank_move = pd.Series(moves).rank().to_numpy()
    correlation = float(np.corrcoef(rank_tilt, rank_move)[0, 1])
    enough_for_h2 = usable.shape[0] >= 250
    if not enough_for_h2:
        h2_verdict = "not enough data yet"
    else:
        h2_verdict = "supported" if correlation > 0 else "not supported"
    print(
        f"  {'H2  short gamma -> bigger moves next day':<44} {correlation:>10.3f} {h2_verdict:>22}"
    )

    if "charm_tilt" in usable and usable["charm_tilt"].notna().any():
        charm = usable["charm_tilt"].fillna(0.0).to_numpy(dtype=float)
        line("H4  charm tilt -> more 2% falls", auc(labels, -charm), state.ready)

    if "probability" in usable.columns and usable["probability"].notna().sum() > 30:
        both = usable.dropna(subset=["probability"])
        base_scores = both["probability"].to_numpy(dtype=float)
        blended = (
            pd.Series(base_scores).rank().to_numpy()
            + pd.Series(-both["gamma_tilt"].to_numpy(dtype=float)).rank().to_numpy()
        ) / 2.0
        gap = auc_gap(both["label"].to_numpy(dtype=float), blended, base_scores)
        verdict = (
            "not enough data yet"
            if not state.ready
            else ("supported" if gap.low > 0 else "not supported")
        )
        print(
            f"  {'H3  tilt adds to the shipped detector':<44} {gap.difference:>+10.3f} "
            f"{verdict:>22}"
        )
    else:
        print(
            f"  {'H3  tilt adds to the shipped detector':<44} {'--':>10} "
            f"{'needs detector scores':>22}"
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the dealer-gamma hypotheses.")
    parser.add_argument("--since", default="2026-01-01")
    parser.add_argument(
        "--detector",
        type=Path,
        default=None,
        help="optional CSV of date,probability from dump_crash_probabilities, for H3",
    )
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    snapshots = asyncio.run(load_snapshots())
    if snapshots.empty:
        print("\n  Nothing recorded yet. The series starts the first night")
        print("  `worker.jobs.index_options` runs — there is no backfill.\n")
        return

    frame = snapshots.merge(market(args.since), on="as_of", how="left")
    if args.detector is not None and args.detector.exists():
        scores = pd.read_csv(args.detector, parse_dates=["date"]).rename(columns={"date": "as_of"})
        frame = frame.merge(scores[["as_of", "probability"]], on="as_of", how="left")

    span = f"{frame['as_of'].min().date()} to {frame['as_of'].max().date()}"
    proxies = ", ".join(sorted(frame["symbol"].unique()))
    print(f"\n  Dealer-gamma test — recorded {span}, from {proxies}")

    state = report_readiness(frame, args.detector)
    evaluate(frame, state)


if __name__ == "__main__":
    main()
