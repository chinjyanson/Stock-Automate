"""Backtesting the scanner (§6).

The scanner has never been measured. Every backtest in this project so far has
tested a *strategy* — an entry rule, an exit, a timing overlay — and the one
layer that decides *which companies to own at all* has only ever been argued
about. This package answers the question the others could not: does ranking by
`scanner.scoring.combine_score` produce a portfolio that beats simply owning the
index?

Two rules make the answer trustworthy, both borrowed from the harnesses that
came before:

  * **The same function scores here as scores in production.** `replay` calls
    `scoring.score_series`, on `PriceSeries.head`-style truncations, so the
    backtest and the nightly scan cannot drift apart.
  * **Every benchmark is stated, including the uncomfortable one.** Beating SPY
    is not enough for a stock picker: an equal-weighted holding of the *same
    universe* is the control that separates picking from the size and sector
    tilt that comes free with not being cap-weighted.
"""
