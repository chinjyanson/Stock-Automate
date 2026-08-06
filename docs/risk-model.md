# Risk model

> **Status: implemented for the internal paper venue (Phase 3).** `app.risk.engine`
> sizes and gates every order, `app.risk.halts` records halts as state,
> `app.risk.execution` wires approval → risk → paper fill → broker-side stop, and
> `app.risk.stops` trails stops upward, applies time stops, and flattens on an
> emergency exit. `app.services.reconciliation` halts on divergence and clears
> when clean; `app.services.eod` persists the end-of-day account summary. Live
> execution exists too: a single Paper/Live toggle selects the venue, and live is
> gated by `LIVE_TRADING_ENABLED` plus live credentials, bounded by the
> persistent `max_live_capital` / `max_daily_loss` on `RiskConfiguration`, and
> reverted to paper on a daily-loss breach. The correlation filter measures
> portfolio weight position-by-position (each holding's own correlation to the
> reference), having replaced the earlier gross-exposure approximation, and runs
> against two references: the benchmark and a rates proxy. `app.risk.stress`
> adds the whole-book check — a historical bootstrap of the portfolio's 20-day
> tail loss, enforcing `max_portfolio_drawdown_pct`.

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
- Correlation adjustment (benchmark and rates)
- Whole-book stress drawdown
- Total portfolio exposure
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

The same gate runs a second time against a rates proxy (a bond *price* series,
default IEF — never a yield index, whose inverse sign would silently reverse
every correlation), bounded by `max_portfolio_rate_sensitive_pct`. There it
compares on **magnitude**: a book that moves hard against yields is as much a
rates bet as one that moves with them. When both gates fire the size is cut
once, not twice — being concentrated in two ways is not twice as bad as being
concentrated in one — though both reasons are recorded.

## Whole-book stress

Every cap above sizes a position against itself. `app.risk.stress` asks the
portfolio-level question instead: with this candidate added, what does a bad
month look like?

```
resample historical daily returns by date, with replacement, over 20 days
→ 95th-percentile loss of the whole book
→ if it exceeds max_portfolio_drawdown_pct, scale the candidate down
```

Resampling **dates** rather than each holding independently is the point: it
preserves the correlation between holdings exactly, so a book of six positions
that fall together is measured as one bet rather than six. There is no
covariance matrix to estimate and no assumption that returns are normal, which
in the tail this measures they observably are not.

A book already past the limit takes no new position at all — shrinking the
candidate cannot repair it. A book with too little history is reported as
*unknown*, and an unknown never blocks a trade.

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
