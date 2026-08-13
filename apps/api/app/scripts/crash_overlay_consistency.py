"""How often does the crash overlay actually beat buy and hold? (§9)

    python -m app.scripts.crash_overlay_consistency

A single train/test split produced a result that looked decisive — the overlay
beating buy and hold on return *and* drawdown — and then dissolved when the split
moved. The reason was visible in the numbers: the drawdown came out identically
16.3% across five adjacent settings, which is the signature of one event being
clipped the same way each time rather than a rule working repeatedly. That event
was COVID. Tested on a period that excluded it, the same setting underperformed.

So this asks the question the single split cannot: **across many separate years,
how often does it win, and by how much?** One good year is not a strategy.

**Rolling origin, disjoint test windows.** Fit on everything up to a cut, trade
the year that follows, step the cut forward by a whole year and repeat. Because
the step equals the test length, no two windows share a trading day — the win
rate is over genuinely separate years rather than the same crisis counted
fifteen times. Each fit sees only its own past.

Three outcomes are tracked separately, because they are different claims:

  * **return** — did it make more money? The hard test, and the one the single
    split flattered.
  * **drawdown** — did it fall less far? This is what the overlay is actually
    for, and what survived the split moving.
  * **ret/dd** — return per unit of drawdown, which is the honest summary when a
    rule trades return away for safety.

A rule that wins on drawdown and loses on return is not thereby useless; it is a
different instrument, and whether that trade is worth making is a decision about
risk, not a fact about the data. What would make it useless is winning no more
often than a coin.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from app.models_ml.logistic import Prior, fit
from app.scripts.backtest_crash_overlay import (
    CALIBRATION_MIN,
    CALIBRATION_WINDOW,
    FEATURES,
    INSIDER_MIN_HISTORY,
    _build,
    _drawdown,
    _rows,
    _simulate,
)

TRADING_YEAR = 252


def _run(
    path: Path,
    capital: float,
    since: str,
    horizon: int,
    fall: float,
    fractions: list[float],
    test_length: int,
    step: int,
    defensive: float,
    ladder: float,
    rebound: float,
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

    # Reaching before 2006 means giving up the insider feature, which does not
    # exist that far back. For a detector limited by how few crashes it has ever
    # seen, the extra history is worth more than the feature.
    features = FEATURES
    begin = INSIDER_MIN_HISTORY + 1
    if since < "2006-01-01":
        features = tuple(f for f in FEATURES if f != "insider_rank")
        begin = CALIBRATION_MIN
        print(f"Features:    {len(features)} — insider_rank dropped, it starts in 2006")

    def label_fall(i: int) -> float:
        ahead = close[i : i + 1 + horizon]
        return 1.0 if float(np.min(ahead) / close[i] - 1.0) <= -fall else 0.0

    # The first cut must leave enough history to both fit the model and fill the
    # rolling calibration window; otherwise the trigger is undefined for the
    # first year and the window silently tests nothing.
    first = begin + CALIBRATION_MIN + TRADING_YEAR
    cuts = list(range(first, n - test_length, step))
    overlap = "disjoint" if step >= test_length else f"overlapping by {test_length - step}d"

    print(f"Windows:     {len(cuts)} of {test_length} trading days each, {overlap}")
    print(f"Warning of:  a {fall:.0%} fall within {horizon} trading day(s)")
    print("Each fit:    trained on its own past only, trigger recalibrated rolling\n")

    for fraction in fractions:
        rows: list[tuple[float, ...]] = []
        for cut in cuts:
            stop = cut + test_length
            train = np.array([i for i in range(begin, cut - horizon - 1) if np.isfinite(daily[i])])
            if train.size < CALIBRATION_MIN:
                continue
            model = fit(
                _rows(signals, train, features),
                np.array([label_fall(i) for i in train]),
                features,
                priors={f: Prior(0.0, 1.0) for f in FEATURES},
                label_definition=f"fall of {fall:.0%} within {horizon} days",
            )

            # Probabilities for every day up to the end of this window, so the
            # rolling trigger can be built from the model's own past output.
            every = np.array([i for i in range(begin, stop) if np.isfinite(daily[i])])
            p_all = np.full(n, np.nan)
            p_all[every] = [
                model.probability(
                    {f: float(v) for f, v in zip(features, row, strict=True) if np.isfinite(v)}
                )
                for row in _rows(signals, every, features)
            ]
            triggers = np.full(n, np.inf)
            for i in range(cut, stop):
                past = p_all[max(begin, i - CALIBRATION_WINDOW) : i]
                past = past[np.isfinite(past)]
                if past.size >= CALIBRATION_MIN:
                    triggers[i] = float(np.quantile(past, 1.0 - fraction))

            results = {}
            for mode, used in (("hold", None), ("overlay", model)):
                curve, alarms, average = _simulate(
                    daily[:stop],
                    close[:stop],
                    signals,
                    used,
                    cut,
                    capital=capital,
                    mode=mode,
                    features=features,
                    triggers=triggers,
                    defensive=defensive,
                    ladder=ladder,
                    rebound=rebound,
                    timeout=timeout,
                    cost=cost,
                )
                results[mode] = (
                    float(curve[-1]) / capital - 1.0,
                    _drawdown(curve),
                    alarms,
                    average,
                )

            hold_r, hold_d, _, _ = results["hold"]
            over_r, over_d, alarms, over_held = results["overlay"]

            # **The control that matters.** The overlay spends this year only
            # ~85% invested, and *any* portfolio held below 100% has a smaller
            # drawdown than one held at 100% — no forecasting required. So the
            # question is not "did drawdown fall", it is "did drawdown fall by
            # more than simply owning less would have". This holds the overlay's
            # own average exposure, constant, all year: same amount of market
            # risk, none of the timing. If the overlay cannot beat it, the model
            # is contributing nothing that a smaller position would not.
            #
            # Note the control is handed an advantage: it is set to the exposure
            # the overlay turned out to average, which is not knowable in
            # advance. That makes it a *generous* benchmark, so the overlay
            # merely tying it is the weaker of the two possible readings — but
            # the advantage is one number against a whole year of timing
            # decisions, and the overlay does not come close to overcoming it.
            flat = over_held
            window = daily[cut:stop]
            flat_curve = capital * np.cumprod(1.0 + flat * np.nan_to_num(window))
            flat_r = float(flat_curve[-1]) / capital - 1.0
            flat_d = _drawdown(flat_curve)
            rows.append(
                (
                    hold_r,
                    over_r,
                    hold_d,
                    over_d,
                    hold_r / hold_d if hold_d else 0.0,
                    over_r / over_d if over_d else 0.0,
                    flat_r,
                    flat_d,
                    flat_r / flat_d if flat_d else 0.0,
                    alarms,
                )
            )

        if not rows:
            continue
        table = np.array([r[:9] for r in rows])
        alarms_total = sum(int(r[9]) for r in rows)
        wins_r = float(np.mean(table[:, 1] > table[:, 0]))
        wins_d = float(np.mean(table[:, 3] < table[:, 2]))
        wins_q = float(np.mean(table[:, 5] > table[:, 4]))
        beats_flat_r = float(np.mean(table[:, 1] > table[:, 6]))
        beats_flat_d = float(np.mean(table[:, 3] < table[:, 7]))
        beats_flat_q = float(np.mean(table[:, 5] > table[:, 8]))

        print(f"  Firing on the top {fraction:.0%} of alarming days — {alarms_total} alarms total")
        print(
            f"    {'measure':<22} {'buy+hold':>10} {'flat':>8} {'overlay':>8} "
            f"{'vs b+h':>8} {'vs flat':>8}"
        )
        print(
            f"    {'median year return':<22} {np.median(table[:, 0]):>10.1%} "
            f"{np.median(table[:, 6]):>8.1%} {np.median(table[:, 1]):>8.1%} "
            f"{wins_r:>8.0%} {beats_flat_r:>8.0%}"
        )
        print(
            f"    {'median drawdown':<22} {np.median(table[:, 2]):>10.1%} "
            f"{np.median(table[:, 7]):>8.1%} {np.median(table[:, 3]):>8.1%} "
            f"{wins_d:>8.0%} {beats_flat_d:>8.0%}"
        )
        print(
            f"    {'median ret/dd':<22} {np.median(table[:, 4]):>10.2f} "
            f"{np.median(table[:, 8]):>8.2f} {np.median(table[:, 5]):>8.2f} "
            f"{wins_q:>8.0%} {beats_flat_q:>8.0%}\n"
        )

    print(
        "  'flat' holds the overlay's own average exposure, constant, all year — the\n"
        "  same market risk with none of the timing. 'vs b+h' and 'vs flat' are the\n"
        "  share of separate years the overlay beat each. 50% is a coin, and beating\n"
        "  buy and hold while losing to flat means the model earned nothing that\n"
        "  simply owning less would not have earned by itself."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Consistency of the crash overlay.")
    parser.add_argument("--path", type=Path, default=Path("data/insider_index.csv"))
    parser.add_argument("--capital", type=float, default=5000.0)
    parser.add_argument("--since", default="2006-01-01")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--fall", type=float, default=0.02)
    parser.add_argument("--sell-fraction", type=float, nargs="+", default=[0.05, 0.10, 0.15, 0.20])
    parser.add_argument("--test-length", type=int, default=TRADING_YEAR)
    parser.add_argument(
        "--step",
        type=int,
        default=TRADING_YEAR,
        help="Days between window starts. Equal to --test-length means disjoint windows.",
    )
    parser.add_argument("--defensive", type=float, default=0.3)
    parser.add_argument("--ladder", type=float, default=0.10)
    parser.add_argument("--rebound", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--cost", type=float, default=0.0005)
    args = parser.parse_args()
    _run(
        args.path,
        args.capital,
        args.since,
        args.horizon,
        args.fall,
        args.sell_fraction,
        args.test_length,
        args.step,
        args.defensive,
        args.ladder,
        args.rebound,
        args.timeout,
        args.cost,
    )


if __name__ == "__main__":
    main()
