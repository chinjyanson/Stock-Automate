# Risk model

> **Status: implemented for the internal paper venue (Phase 3).** `app.risk.engine`
> sizes and gates every order, `app.risk.halts` records halts as state,
> `app.risk.execution` wires approval → risk → paper fill → broker-side stop, and
> `app.risk.stops` trails stops upward, applies time stops, and flattens on an
> emergency exit. `app.services.reconciliation` halts on divergence and clears
> when clean; `app.services.eod` persists the end-of-day account summary. What is
> **not** built here is live execution — `/live/arm` still refuses without live
> credentials and a clean reconciliation, and paper is the only executing venue
> (live is Phase 6). The correlation filter uses a gross-exposure approximation
> for portfolio S&P weight, noted in `app.risk.engine`, to refine when strategies
> track per-position correlation (Phase 4).

## The central rule

**No strategy may submit an order.** Every proposed trade passes through the
risk engine, which can reduce it, reject it, or halt trading entirely. A
strategy proposes; the risk engine disposes.

```
Strategy ──proposal──► Risk engine ──approved size──► Trade intent ──► Broker
                            │
                            └──► rejected, with a recorded reason
```

## Position sizing

Volatility-adjusted, so a position's *risk* is constant rather than its size:

```
risk_budget   = account_equity × risk_per_trade_pct
stop_distance = max(ATR × stop_multiplier, minimum_stop_distance)
raw_quantity  = risk_budget / stop_distance
```

A volatile instrument therefore gets a smaller position for the same risk. Sizing
by fixed cash instead would take wildly different risk per trade without saying
so.

`raw_quantity` is then capped by **every** one of:

- Maximum position percentage
- Maximum strategy allocation
- Maximum instrument allocation
- Available cash
- Maximum total open risk
- Broker quantity restrictions (min size, step, fractional support)
- Liquidity restrictions
- Correlation adjustment
- User-configured monetary cap

The **smallest** cap wins. Quantities round *down* to the instrument's step —
rounding up would breach the cap that produced the number.

## Correlation

Rolling daily-return correlations over configurable windows (60 and 120 trading
days), against a configured benchmark (default SPY).

```
if portfolio_sp500_exposure > configured_limit
and candidate_sp500_correlation > configured_threshold:
    reduce_position_size_or_reject()
```

**Correlation must influence sizing, not merely warn.** A warning that does not
change the order is decoration. Six "diversified" positions that are all really
one S&P bet is the failure this prevents, and the adjustment is recorded in the
decision explanation.

## Controls

| Control | Purpose |
|---|---|
| Risk per trade | Bounds a single loss |
| Max position size | Bounds concentration |
| Max strategy allocation | Bounds one strategy being wrong |
| Max portfolio exposure | Bounds total market exposure |
| Max total open risk | Bounds simultaneous stop-outs |
| Max daily realised loss | Stops the day |
| Max daily realised + unrealised | Stops before it is realised |
| Max portfolio drawdown | Kill switch |
| Max open positions | Bounds complexity |
| Max trades per day | Bounds runaway logic |
| Cooldown after consecutive losses | Bounds a broken regime |
| Stale-data block | Fail closed |
| Provider-failure block | Fail closed |
| Broker-reconciliation block | Fail closed |
| Emergency kill switch | Human override |
| Per-instrument / per-strategy suspension | Surgical override |
| Global live suspension | Blunt override |

Halts are **states, not exceptions**: recorded, visible, and requiring explicit
clearing. A halt that clears itself on restart is not a halt.

## Stops

- Initial ATR stop, placed broker-side after entry confirmation where possible
- Trailing ATR stop
- Strategy exit
- Time stop
- Emergency market exit

A **broker-side** stop survives our process dying. The local synthetic-stop
monitor is a secondary safeguard only — relying on an application process
staying online to protect a position is not a stop, it is a hope.

## Fail-closed defaults

When anything is uncertain — data, broker state, risk state — the answer is
**no trade**. Rejecting a good trade costs an opportunity; accepting a trade on
bad data costs money.

## Configuration

All thresholds live in `RiskConfiguration`, versioned and audited. No risk limit
is a code constant. Changing one is an audited event, because "who loosened the
drawdown limit, and when" is a question that gets asked after the loss.
