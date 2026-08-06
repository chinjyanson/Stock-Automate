"""What the S&P index options market is pricing, and how dealers are positioned.

Three readings, all from one option chain, all about the index rather than any
single company:

  * **Dealer gamma exposure (GEX).** Option market makers hedge their books
    continuously. A dealer who is net *long* gamma must sell into rallies and
    buy into dips to stay delta-neutral, which dampens volatility and makes
    dips get bought. A dealer who is net *short* gamma must do the opposite —
    buy strength, sell weakness — which amplifies moves. GEX estimates which
    regime the aggregate book is in.

  * **25-delta skew.** How much more the market pays for downside protection
    than for upside participation. Steep skew is the market pricing a crash.

  * **At-the-money implied volatility.** The market's own forward expectation
    of how much the index will move, as opposed to how much it has moved.

**On the ticker.** These are measured on `^SPX` (falling back to `SPY`) because
that is where the data exists — UK-listed S&P trackers such as VUAG.L, VUSA.L
and CSPX.L have no listed options at all, which was verified rather than
assumed. The readings describe the S&P 500 itself, so they apply just as well
to a UK-domiciled tracker of that index as to the US-listed one. The signal is
measured in one wrapper and acted on in another, deliberately.

**On the sign convention, which is a convention and not a fact.** GEX requires
an assumption about which side of the open interest dealers are on. The one
used here — and in essentially all public GEX work — is that dealers are long
call open interest and short put open interest, on the reasoning that
customers predominantly buy puts for protection and sell calls for yield.
It is an approximation of a thing nobody outside the dealers can observe. The
*sign* of the output depends on it entirely, so it is named, isolated in one
constant, and reported alongside the number rather than buried.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, date, datetime

import structlog

from app.signals import options_math as om

log = structlog.get_logger(__name__)

#: Index proxies, in preference order. `^SPX` is the index itself and carries
#: the deepest, most strike-dense chain; SPY is a liquid stand-in when it fails.
INDEX_PROXIES = ("^SPX", "SPY")

#: Contract multiplier. Both SPX and SPY options are 100 units per contract.
CONTRACT_MULTIPLIER = 100.0

#: Dealers are assumed long calls and short puts. See the module docstring —
#: this single constant decides the sign of every GEX reading produced.
DEALER_CALL_SIGN, DEALER_PUT_SIGN = +1.0, -1.0

#: Target horizon for the reading. ~30 days is the standard window for a vol
#: surface: long enough to be past expiry-week distortion, short enough to
#: describe conditions now rather than next quarter.
TARGET_EXPIRY_DAYS = 30

#: Expiries inside this are dominated by pinning and gamma effects that say more
#: about the expiry than about the market, so they are skipped.
MIN_EXPIRY_DAYS = 7

#: The wing the skew is measured at. 25-delta is the market convention: far
#: enough out to be genuine protection, liquid enough to carry a real quote.
SKEW_DELTA = 0.25

#: Implied volatilities outside this band are broken marks, not readings.
_IV_SANE_LOW, _IV_SANE_HIGH = 0.01, 4.0

#: Below this open interest a strike carries no real positioning and its quote
#: is usually stale; including it adds noise to GEX and nothing else.
MIN_OPEN_INTEREST = 1.0

_TRADING_DAYS_PER_YEAR = 365.0


@dataclass(frozen=True, slots=True)
class OptionQuote:
    strike: float
    last_price: float
    open_interest: float
    is_call: bool
    #: The provider's own implied volatility, where it published one.
    provider_iv: float | None = None


@dataclass(frozen=True, slots=True)
class IndexOptionsReading:
    as_of: date
    symbol: str
    spot: float
    expiry_days: int
    #: Net dealer gamma, in billions of currency units per 1% index move.
    #: Positive = dealers long gamma = volatility dampened. Sign depends on
    #: DEALER_CALL_SIGN / DEALER_PUT_SIGN.
    gamma_exposure: float | None = None
    #: 25-delta put IV minus 25-delta call IV. Positive = downside is dearer.
    skew_25delta: float | None = None
    #: Implied volatility at the strike nearest spot.
    atm_iv: float | None = None
    contracts_used: int = 0


def _usable_iv(quote: OptionQuote, spot: float, t: float) -> float | None:
    """Best available implied volatility for one quote.

    Prefers the provider's own figure, which is derived from a live bid/ask and
    so survives a strike that has not traded today. Falls back to solving from
    the last traded price when the provider published nothing usable — a stale
    trade is a worse input than a live mid, but a far better one than dropping
    the strike and silently narrowing the surface.
    """
    if quote.provider_iv is not None and _IV_SANE_LOW < quote.provider_iv < _IV_SANE_HIGH:
        return quote.provider_iv
    solved = om.implied_vol(quote.last_price, spot, quote.strike, t, is_call=quote.is_call)
    if solved is not None and _IV_SANE_LOW < solved < _IV_SANE_HIGH:
        return solved
    return None


def compute_reading(
    *,
    as_of: date,
    symbol: str,
    spot: float,
    quotes: list[OptionQuote],
    expiry_days: int,
) -> IndexOptionsReading | None:
    """Reduce one expiry's chain to the three index readings. Pure.

    Returns None when the chain is unusable. A partial reading is fine and
    expected — GEX needs open interest, the skew needs both wings quoted, and a
    chain can support one without the other; each field is independently None.
    """
    if spot <= 0 or expiry_days <= 0 or not quotes:
        return None
    t = expiry_days / _TRADING_DAYS_PER_YEAR

    gamma_sum = 0.0
    used = 0
    atm_iv: float | None = None
    atm_distance = float("inf")
    # (distance from the target delta, iv) for each wing.
    best_call: tuple[float, float] | None = None
    best_put: tuple[float, float] | None = None

    for quote in quotes:
        if quote.strike <= 0:
            continue
        iv = _usable_iv(quote, spot, t)
        if iv is None:
            continue

        distance = abs(quote.strike - spot)
        if distance < atm_distance:
            atm_distance, atm_iv = distance, iv

        if quote.open_interest >= MIN_OPEN_INTEREST:
            g = om.gamma(spot, quote.strike, t, iv)
            if g is not None:
                sign = DEALER_CALL_SIGN if quote.is_call else DEALER_PUT_SIGN
                gamma_sum += sign * g * quote.open_interest
                used += 1

        d = om.delta(spot, quote.strike, t, iv, is_call=quote.is_call)
        if d is not None:
            gap = abs(abs(d) - SKEW_DELTA)
            if quote.is_call:
                if best_call is None or gap < best_call[0]:
                    best_call = (gap, iv)
            elif best_put is None or gap < best_put[0]:
                best_put = (gap, iv)

    # Gamma is per one currency unit of move. Scaling by spot^2 * 1% converts it
    # to "currency of dealer hedging flow per 1% index move", the form GEX is
    # conventionally quoted in, then /1e9 to report in billions.
    gamma_exposure = gamma_sum * CONTRACT_MULTIPLIER * spot * spot * 0.01 / 1e9 if used else None
    skew = best_put[1] - best_call[1] if best_put and best_call else None

    return IndexOptionsReading(
        as_of=as_of,
        symbol=symbol,
        spot=spot,
        expiry_days=expiry_days,
        gamma_exposure=gamma_exposure,
        skew_25delta=skew,
        atm_iv=atm_iv,
        contracts_used=used,
    )


def _pick_expiry(expiries: list[str], today: date) -> tuple[str, int] | None:
    """The listed expiry closest to the target horizon, past the pinning window."""
    candidates: list[tuple[int, str, int]] = []
    for raw in expiries:
        try:
            days = (date.fromisoformat(raw) - today).days
        except ValueError:
            continue
        if days >= MIN_EXPIRY_DAYS:
            candidates.append((abs(days - TARGET_EXPIRY_DAYS), raw, days))
    if not candidates:
        return None
    _, chosen, days = min(candidates)
    return chosen, days


async def fetch_reading() -> IndexOptionsReading | None:
    """Read a live index option chain. Network — call from a scheduled job only.

    Tries each proxy in turn and returns the first that yields a chain, so a
    single ticker failing degrades the reading rather than losing the day.
    """

    def _fetch() -> IndexOptionsReading | None:
        import yfinance as yf

        today = datetime.now(UTC).date()
        for symbol in INDEX_PROXIES:
            try:
                ticker = yf.Ticker(symbol)
                expiries = list(ticker.options or [])
                picked = _pick_expiry(expiries, today)
                if picked is None:
                    continue
                expiry, days = picked
                chain = ticker.option_chain(expiry)
                history = ticker.history(period="5d", interval="1d")
                if history.empty:
                    continue
                spot = float(history["Close"].iloc[-1])

                quotes: list[OptionQuote] = []
                for frame, is_call in ((chain.calls, True), (chain.puts, False)):
                    for row in frame.itertuples():
                        quotes.append(
                            OptionQuote(
                                strike=float(getattr(row, "strike", 0.0) or 0.0),
                                last_price=float(getattr(row, "lastPrice", 0.0) or 0.0),
                                open_interest=float(getattr(row, "openInterest", 0.0) or 0.0),
                                is_call=is_call,
                                provider_iv=_optional_float(
                                    getattr(row, "impliedVolatility", None)
                                ),
                            )
                        )
                reading = compute_reading(
                    as_of=today,
                    symbol=symbol,
                    spot=spot,
                    quotes=quotes,
                    expiry_days=days,
                )
                if reading is not None:
                    return reading
            except Exception as exc:
                log.warning("spx_options.proxy_failed", symbol=symbol, error=str(exc))
        return None

    try:
        return await asyncio.to_thread(_fetch)
    except Exception as exc:
        log.warning("spx_options.fetch_failed", error=str(exc))
        return None


def _optional_float(value: object) -> float | None:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return None if out != out else out  # NaN is not a reading
