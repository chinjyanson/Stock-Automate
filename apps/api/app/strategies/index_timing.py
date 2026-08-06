"""Timed exposure to an S&P 500 tracker, from index-level signals (§8).

A separate pipeline from mean reversion, and deliberately so. Mean reversion
asks "has *this company* been pushed unusually cheap relative to its own recent
range?" — a question that only means anything for a single business. An index
fund has no such thing: VUAG.L falling below its lower Bollinger band is not a
dislocation to be bought, it is the S&P 500 going down. Running the same logic
over both would be a category error dressed as diversification.

So this strategy asks the question that *is* meaningful for an index: what is
the market itself pricing, and how is it positioned? Three inputs, none of them
derived from the tracker's own price:

  * **Dealer gamma (GEX).** Positive means option dealers are long gamma and
    must hedge against moves — selling rallies, buying dips — which suppresses
    volatility and makes declines get absorbed. Negative means they hedge with
    moves and amplify them. This is the primary read: it describes the market's
    mechanical behaviour rather than anyone's opinion of it.
  * **25-delta skew.** How much more downside protection costs than upside.
    A steepening is the market paying up for a crash it did not previously fear.
  * **Market regime.** The same posture multiplier the risk engine uses, so this
    strategy and position sizing cannot disagree about whether it is a bad tape.

All three are measured on SPX options and index proxies, never on the traded
instrument, and all three are read from stored daily snapshots rather than
fetched — so evaluating a signal touches no network.

Long-only and long-flat: it holds the tracker or it holds nothing. There is no
shorting and no leverage, so the worst case is being in cash through a rally.

**What the signals cannot do.** GEX rests on an assumed dealer position (see
`app.signals.spx_options`) — it is an inference about positioning nobody outside
the dealers can observe, and its sign depends entirely on that assumption. It is
treated here as one input among three, gated by a regime check, rather than as a
fact to trade on alone.
"""

from __future__ import annotations

from app.models.enums import Interval, OrderSide, StrategyKind
from app.strategies.base import Strategy, StrategyContext, StrategySignal

#: Below this risk multiplier the market regime is bad enough that no options
#: reading justifies holding the index. Matches the "defensive" posture band in
#: `app.services.market_regime`.
DEFAULT_REGIME_FLOOR = 0.70

#: Skew above this is the market paying hard for crash protection. Expressed in
#: implied-volatility points (0.08 = the 25-delta put prices 8 vol points over
#: the matching call), which is elevated but not extreme by historical standards.
DEFAULT_SKEW_EXIT = 0.08


class IndexTimingStrategy(Strategy):
    kind = StrategyKind.INDEX_TIMING
    interval = Interval.D1

    async def evaluate(self, ctx: StrategyContext) -> list[StrategySignal]:
        regime_floor = float(self.param("regime_floor", DEFAULT_REGIME_FLOOR))
        skew_exit = float(self.param("skew_exit", DEFAULT_SKEW_EXIT))
        # Dealer gamma is reported in billions per 1% move; zero is the
        # long/short boundary, and a band around it avoids flipping the position
        # on a reading that is really just noise either side of neutral.
        gex_enter = float(self.param("gex_enter", 0.0))
        gex_exit = float(self.param("gex_exit", -0.5))

        conditions = ctx.index_conditions
        regime = conditions.regime_factor

        signals: list[StrategySignal] = []
        for instrument in ctx.instruments:
            held = ctx.held_quantity(instrument.id)

            if not conditions.options_available:
                # No usable options reading. Hold what is held and buy nothing:
                # this strategy's entire thesis is the options surface, so
                # without it there is no opinion to act on. Not an exit —
                # a missing measurement is not a sell signal.
                continue

            gex = conditions.gamma_exposure
            skew = conditions.skew_25delta
            metrics = {
                "regime_factor": regime,
                "gamma_exposure": gex if gex is not None else 0.0,
                "skew_25delta": skew if skew is not None else 0.0,
                "contracts_used": float(conditions.contracts_used),
            }

            regime_bad = regime < regime_floor
            skew_bad = skew is not None and skew >= skew_exit
            gex_bad = gex is not None and gex <= gex_exit

            if held > 0:
                # Exit on any one of the three turning hostile. Exposure to a
                # whole index is not a thesis about a company that deserves
                # patience; when the conditions that justified holding it stop
                # holding, the reason to be there has gone.
                reasons = []
                if regime_bad:
                    reasons.append(f"regime {regime:.2f} below {regime_floor:.2f}")
                if skew_bad and skew is not None:
                    reasons.append(f"25-delta skew {skew:.3f} at or above {skew_exit:.3f}")
                if gex_bad and gex is not None:
                    reasons.append(f"dealer gamma {gex:+.2f}bn at or below {gex_exit:+.2f}bn")
                if reasons:
                    signals.append(
                        StrategySignal(
                            instrument_id=instrument.id,
                            side=OrderSide.SELL,
                            conviction=1.0,
                            reason="Index exit: " + "; ".join(reasons),
                            target_quantity=held,
                            metrics=metrics,
                        )
                    )
                continue

            # Entry needs the regime to permit it *and* dealers to be long
            # gamma. Skew is a veto rather than a trigger: cheap protection is
            # not a reason to buy, but expensive protection is a reason not to.
            if gex is None or regime_bad or skew_bad or gex <= gex_enter:
                continue
            signals.append(
                StrategySignal(
                    instrument_id=instrument.id,
                    side=OrderSide.BUY,
                    conviction=1.0,
                    reason=(
                        f"Index entry: dealer gamma {gex:+.2f}bn (above "
                        f"{gex_enter:+.2f}bn, volatility suppressed), regime "
                        f"{regime:.2f}"
                        + (f", 25-delta skew {skew:.3f}" if skew is not None else "")
                    ),
                    metrics=metrics,
                )
            )
        return signals
