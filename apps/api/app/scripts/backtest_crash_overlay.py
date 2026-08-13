"""Stay invested; step aside for crashes; buy back into the fall (§9).

    python -m app.scripts.backtest_crash_overlay --capital 5000

Every strategy tested so far has been symmetric: decide each period whether to
be in or out, which forces an opinion on rises as well as falls. Direction has
failed at every horizon tried, so this stops asking.

The design is asymmetric by construction, and rests on one assumption stated
plainly: **the index rises over time, so being invested is the default and
selling must justify itself.** Two questions, neither of which is "will it go
up":

  * **Is a sharp fall imminent?** If so, step aside. Deliberately reluctant —
    the probability threshold is high, because a false alarm costs real return
    in a market that mostly rises.
  * **Has it fallen far enough to buy back?** Mechanical, not predictive: as the
    price drops below the level it was sold at, capital returns in steps. No
    forecast of the bottom is required, only a ladder.

A timeout returns the position if the feared fall never arrives, so a wrong
warning costs days rather than years.

**On measuring the fall detector.** A sharp one-day fall is rare — a few percent
of days — so an AUC can look respectable while the trigger is useless in
practice. What matters for a trigger is what happens *when it fires*: how often
a warning is followed by the fall, and how much of the market's fall the warnings
actually cover. Both are reported, because the AUC alone would flatter it.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from app.models_ml.logistic import FittedModel, Prior, auc, fit
from app.signals.crash_features import (
    CALIBRATION_MIN,
    CALIBRATION_WINDOW,
    FEATURES,
    INSIDER_MIN_HISTORY,
    TRADING_DAYS,
)
from app.signals.crash_features import build as _build
from app.signals.crash_features import label_fall as _label_fall
from app.signals.crash_features import rows as _rows

BORROW = 0.05


def _drawdown(curve: np.ndarray) -> float:
    peak = np.maximum.accumulate(curve)
    return float(np.max((peak - curve) / peak)) if curve.size else 0.0


def _simulate(
    daily: np.ndarray,
    close: np.ndarray,
    signals: dict[str, np.ndarray],
    model: FittedModel | None,
    start: int,
    *,
    capital: float,
    mode: str,
    features: tuple[str, ...],
    triggers: np.ndarray,
    defensive: float,
    ladder: float,
    rebound: float,
    reentry: str,
    timeout: int,
    cost: float,
) -> tuple[np.ndarray, int, float]:
    """Invested by default; step aside on a warning; ladder back in as it falls."""
    equity = capital
    exposure = 1.0
    curve: list[float] = []
    alarms = 0
    exit_price: float | None = None
    low_since = float("inf")
    days_out = 0
    held: list[float] = []

    for i in range(start, daily.size):
        prior = i - 1
        if mode == "hold":
            wanted = 1.0
        else:
            reading = {
                f: float(signals[f][prior]) for f in features if np.isfinite(signals[f][prior])
            }
            probability = model.probability(reading) if (model is not None and reading) else 0.0

            if exit_price is None:
                # Fully invested: the only question is whether to step aside.
                if probability >= triggers[prior]:
                    exit_price = float(close[prior])
                    low_since = float(close[prior])
                    days_out = 0
                    alarms += 1
                    wanted = defensive
                else:
                    wanted = 1.0
            else:
                days_out += 1
                here = float(close[prior])
                low_since = min(low_since, here)
                fallen = 1.0 - here / exit_price
                # Ladder back in proportionally to how far it has fallen, so
                # capital returns *into* the decline rather than waiting for a
                # bottom nobody can identify.
                recovered = float(np.clip(fallen / ladder, 0.0, 1.0))
                still_warning = probability >= triggers[prior]

                if reentry == "price":
                    wanted = defensive + (1.0 - defensive) * recovered

                    # **Buy back into the bounce.** The ladder only reacts to
                    # further falls — `fallen` clips at zero — so a market that
                    # rallies straight off the alarm leaves the position pinned
                    # at `defensive` until the timeout, standing outside the
                    # rebound. That is the wrong way round: the worst days and
                    # the best days are neighbours, so the recovery is precisely
                    # what must not be missed. A rise of `rebound` off the lowest
                    # close since the alarm returns the position in full.
                    bounced = rebound > 0.0 and here / low_since - 1.0 >= rebound
                    if recovered >= 1.0 or bounced or days_out >= timeout:
                        exit_price = None
                        wanted = 1.0

                # **Model-driven re-entry.** The two rules below re-read the
                # detector every day instead of waiting on price, which is what
                # makes them fast: the position returns the day *after* the
                # warning clears rather than after a fixed timeout. The detector
                # is already being computed daily; not consulting it while out
                # was the waste.
                elif not still_warning:
                    # All clear. Whatever the price has done, the reason for
                    # standing aside has gone, so step fully back in.
                    exit_price = None
                    wanted = 1.0
                elif reentry == "on-warning":
                    # Still warning, and this rule reads that as a reason to
                    # accumulate: another fall means a lower price to buy, so
                    # capital returns in proportion to how far it has already
                    # fallen. Deliberately buying into a predicted decline —
                    # which is either the ladder's logic taken seriously or a
                    # way to lose money faster, and only measurement decides.
                    wanted = defensive + (1.0 - defensive) * recovered
                else:  # "all-clear" — stay aside while the warning stands
                    wanted = defensive

                if reentry != "price" and days_out >= timeout:
                    exit_price = None
                    wanted = 1.0

        if abs(wanted - exposure) > 0.02:
            equity -= equity * abs(wanted - exposure) * cost / 2.0
            exposure = wanted

        equity *= 1.0 + exposure * daily[i]
        curve.append(equity)
        held.append(exposure)

    return np.asarray(curve), alarms, float(np.mean(held)) if held else 0.0


def _run(
    path: Path,
    capital: float,
    since: str,
    split: float,
    split_date: str | None,
    horizon: int,
    fall: float,
    fractions: list[float],
    defensive: float,
    ladder: float,
    rebound: float,
    reentry: str,
    timeout: int,
    cost: float,
) -> None:
    warnings.filterwarnings("ignore")
    import yfinance as yf

    frame = yf.Ticker("^GSPC").history(period="max", interval="1d")
    frame = frame[frame.index >= since]
    close = frame["Close"].to_numpy(dtype=np.float64)
    index = pd.DatetimeIndex(frame.index.tz_localize(None)).normalize()
    daily = np.concatenate([[np.nan], close[1:] / close[:-1] - 1.0])
    n = close.size

    signals = _build(path, index, close, daily)

    # **The insider series begins in 2006.** Reaching further back means giving
    # it up — and for a detector limited by how few crashes it has ever seen,
    # sixteen extra years of history buys far more than one feature does. The
    # trade is stated here rather than hidden, because dropping a feature
    # silently would make two runs incomparable for no visible reason.
    features = FEATURES
    if since < "2006-01-01":
        features = tuple(f for f in FEATURES if f != "insider_rank")
        print(f"Features:    {len(features)} — insider_rank dropped, it starts in 2006")

    begin_at = INSIDER_MIN_HISTORY + 1 if "insider_rank" in features else CALIBRATION_MIN

    def label_fall(i: int) -> float:
        return _label_fall(close, i, fall=fall, horizon=horizon)

    # A crisis has to be *held out*, not merely present. Splitting by fraction
    # lands the cut wherever the data happens to end; splitting by date puts it
    # deliberately before a known event, so what follows is a real forecast of
    # that event rather than a recollection of it.
    at_date = int(index.searchsorted(pd.Timestamp(split_date))) if split_date else 0
    cut = at_date or int(n * split)
    begin = begin_at
    train = np.array([i for i in range(begin, cut - horizon - 1) if np.isfinite(daily[i])])
    model = fit(
        _rows(signals, train, features),
        np.array([label_fall(i) for i in train]),
        features,
        priors={f: Prior(0.0, 1.0) for f in FEATURES},
        label_definition=f"fall of {fall:.0%} within {horizon} days",
    )

    # **The threshold has to be calibrated, and it has to keep recalibrating.**
    # A 3.4% base rate means the model almost never emits a probability above
    # 0.5, so an absolute cutoff never fires — that is not caution, it is a
    # broken switch. But a percentile of the *training* output is barely better:
    # training spans 2008, so its 95th percentile is a bar the calmer years that
    # follow almost never clear, and the switch stays off through a decade it
    # was supposed to be watching.
    #
    # So the trigger is a percentile of every probability the model has emitted
    # *before today* — expanding, never forward-looking. "Fire on the most
    # alarming 5% of days" then means the same thing in 2017 as in 2008, because
    # it is always measured against the regime actually in force.
    every = np.array([i for i in range(begin, n) if np.isfinite(daily[i])])
    p_all = np.full(n, np.nan)
    p_all[every] = [
        model.probability(
            {f: float(v) for f, v in zip(features, row, strict=True) if np.isfinite(v)}
        )
        for row in _rows(signals, every, features)
    ]

    def make_triggers(fraction: float) -> np.ndarray:
        out = np.full(n, np.inf)
        for i in range(begin + CALIBRATION_MIN, n):
            past = p_all[max(begin, i - CALIBRATION_WINDOW) : i]
            past = past[np.isfinite(past)]
            if past.size >= CALIBRATION_MIN:
                out[i] = float(np.quantile(past, 1.0 - fraction))
        return out

    held = np.array(range(cut, n - horizon - 1))
    y = np.array([label_fall(i) for i in held])
    p = p_all[held]

    print(f"Traded:      {index[cut].date()} to {index[-1].date()}")
    print(f"Warning of:  a {fall:.0%} fall within {horizon} trading day(s)")
    print(f"Detector:    out-of-sample AUC {auc(y, p):.4f}, base rate {y.mean():.1%}\n")

    curve, _, _ = _simulate(
        daily,
        close,
        signals,
        None,
        cut,
        capital=capital,
        mode="hold",
        features=features,
        triggers=np.full(n, np.inf),
        defensive=defensive,
        ladder=ladder,
        rebound=rebound,
        reentry=reentry,
        timeout=timeout,
        cost=cost,
    )
    hold_final, hold_dd = float(curve[-1]), _drawdown(curve)
    hold_curve = curve

    print(
        f"  {'rule':<26} {'fires':>7} {'precis':>7} {'recall':>7} {'final':>9} "
        f"{'return':>8} {'drawdn':>8} {'ret/dd':>7} {'track':>7} {'held':>6}"
    )
    print(
        f"  {'buy and hold':<26} {'-':>7} {'-':>7} {'-':>7} {hold_final:>9,.0f} "
        f"{hold_final / capital - 1:>7.1%} {hold_dd:>7.1%} "
        f"{(hold_final / capital - 1) / hold_dd:>7.2f} {'-':>7} {'100%':>6}"
    )

    for fraction in fractions:
        triggers = make_triggers(fraction)
        fired = p >= triggers[held]
        curve, _alarms, average = _simulate(
            daily,
            close,
            signals,
            model,
            cut,
            capital=capital,
            mode="overlay",
            features=features,
            triggers=triggers,
            defensive=defensive,
            ladder=ladder,
            rebound=rebound,
            reentry=reentry,
            timeout=timeout,
            cost=cost,
        )
        final = float(curve[-1])
        drawdown = _drawdown(curve)
        # Tracking error against buy and hold, annualised. The goal here is not
        # to beat the benchmark but to sit close to it and lose less in a crash,
        # so how far the ride differs day to day is the thing to report.
        pair = min(curve.size, hold_curve.size)
        diff = np.diff(np.log(curve[:pair])) - np.diff(np.log(hold_curve[:pair]))
        tracking = float(np.std(diff, ddof=1)) * np.sqrt(TRADING_DAYS) if diff.size > 1 else 0.0
        # **The control, on the same row.** Owning less lowers drawdown by
        # itself, so the overlay's drawdown is only interesting beside a
        # portfolio held flat at the same average exposure all along. If the two
        # match, the model timed nothing; if the overlay is far lower, it stepped
        # aside at moments that actually mattered.
        flat_curve = capital * np.cumprod(1.0 + average * np.nan_to_num(daily[cut:]))
        flat_dd = _drawdown(flat_curve)
        flat_ret = float(flat_curve[-1]) / capital - 1.0
        precision = y[fired].mean() if fired.any() else float("nan")
        recall = fired[y == 1].mean() if fired.any() else 0.0
        print(
            f"  {f'{reentry}, top {fraction:.2%}':<26} {fired.mean():>7.1%} "
            f"{precision:>7.1%} {recall:>7.1%} {final:>9,.0f} {final / capital - 1:>7.1%} "
            f"{drawdown:>7.1%} {(final / capital - 1) / drawdown if drawdown else 0:>7.2f} "
            f"{tracking:>7.1%} {average:>6.0%}"
        )
        print(
            f"  {'    ^ flat at same exposure':<26} {'':>7} {'':>7} {'':>7} "
            f"{capital * (1 + flat_ret):>9,.0f} {flat_ret:>7.1%} {flat_dd:>7.1%} "
            f"{flat_ret / flat_dd if flat_dd else 0:>7.2f} {'':>7} {average:>6.0%}"
        )

    print(
        "\n  'precis' is how often a warning was actually followed by the fall;\n"
        "  'recall' is how much of the market's falling the warnings covered.\n"
        "  A rule is only worth reading if ret/dd beats buy and hold AND it fired\n"
        "  enough times to have been tested rather than merely sampled."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Asymmetric crash overlay.")
    parser.add_argument("--path", type=Path, default=Path("data/insider_index.csv"))
    parser.add_argument("--capital", type=float, default=5000.0)
    parser.add_argument(
        "--since",
        default="2006-01-01",
        help="History start. Before 2006 the insider feature is dropped automatically.",
    )
    parser.add_argument("--split", type=float, default=0.5)
    parser.add_argument(
        "--split-date",
        default=None,
        help="Train up to this date and trade everything after. Overrides --split.",
    )
    parser.add_argument("--horizon", type=int, default=1, help="Days ahead the warning covers.")
    parser.add_argument("--fall", type=float, default=0.02, help="What counts as a sharp fall.")
    parser.add_argument(
        "--sell-fraction",
        type=float,
        nargs="+",
        default=[0.02, 0.05, 0.10, 0.15, 0.20, 0.30],
        help="Fire on this fraction of the most alarming recent days. Swept.",
    )
    parser.add_argument("--defensive", type=float, default=0.3, help="Exposure kept when out.")
    parser.add_argument("--ladder", type=float, default=0.10, help="Fall over which to buy back.")
    parser.add_argument(
        "--reentry",
        choices=("price", "all-clear", "on-warning"),
        default="price",
        help=(
            "How to come back. 'price' ladders on further falls plus --rebound; "
            "'all-clear' returns in full the day the warning clears; 'on-warning' "
            "also accumulates while the warning still stands."
        ),
    )
    parser.add_argument(
        "--rebound",
        type=float,
        default=0.0,
        help="Rise off the post-alarm low that buys back in full. 0 disables.",
    )
    parser.add_argument("--timeout", type=int, default=20, help="Days before returning anyway.")
    parser.add_argument("--cost", type=float, default=0.0005)
    args = parser.parse_args()
    _run(
        args.path,
        args.capital,
        args.since,
        args.split,
        args.split_date,
        args.horizon,
        args.fall,
        args.sell_fraction,
        args.defensive,
        args.ladder,
        args.rebound,
        args.reentry,
        args.timeout,
        args.cost,
    )


if __name__ == "__main__":
    main()
