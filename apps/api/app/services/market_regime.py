"""Market-wide risk posture: measure it, and turn it into one number (§9).

Seven inputs, split by how fast they move — which decides what each is allowed to
do:

  **Fast — these scale risk.**
    * `VIX`, live from yfinance. The market's own priced expectation of turbulence.
    * `Breadth`, the fraction of instruments above their own 200-day average.
      Computed from the local candle store, so it costs one query and no
      provider call. Breadth rolls over before an index does.
    * `Index regime`, the 50/200 crossover on an S&P and a world proxy.
    * `Realised volatility`, the annualised 20-day move of the S&P proxy. VIX is
      what the market *expects*; this is what happened. A slow, orderly decline
      can leave VIX untroubled while the tape plainly deteriorates.
    * `Credit spread`, the high-yield option-adjusted spread from FRED. Credit
      reprices ahead of equity, and unlike the yield curve below it moves on a
      timescale a position size can actually respond to.

  **Context — recorded and alerted on, never gating.**
    * `Yield curve` (10y-2y, 10y-3m) from FRED. Genuine recession signal, but
      with a 6-24 month lead — too slow to size a position from.
    * `Buffett indicator`, corporate equities over GDP. Quarterly, sourced from
      the Fed's Z.1 release, and therefore months stale on arrival.

The split is the important design decision. Valuation measures have been at
extremes for years; a bot that trades less whenever they are elevated is a bot
that does not trade. They belong in an alert, not in a gate.

Equally, the factor is *floored*: the worst regime shrinks positions, it never
halts them. A risk overlay that can reach zero stops being a risk control and
becomes a market-timing bet.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal

import httpx
import structlog
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.data.store import CandleStore
from app.indicators import functions as ind
from app.indicators.series import PriceSeries, candles_to_series
from app.models.enums import Interval
from app.models.instrument import Exchange, Instrument
from app.models.market_regime import MarketRegimeSnapshot

log = structlog.get_logger(__name__)

_FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"

#: Proxies for the two indices tracked. First match with enough history wins, so
#: a venue outage or a missing listing degrades rather than breaks the reading.
_SP500_PROXIES = (("SPY", "ARCX"), ("VUAG", "XLON"), ("VUSA", "XAMS"))
_WORLD_PROXIES = (("IWDA", "XLON"), ("IWDA", "XAMS"), ("VWRL", "XAMS"), ("ACWI", "XSWX"))

#: Floor on the risk multiplier. At 0.55 the worst regime roughly halves position
#: size while leaving the strategy fully operational — the explicit intent is to
#: keep trading through a downturn, not to sit it out.
MIN_RISK_FACTOR = 0.55

#: Deductions from a starting factor of 1.0. Deliberately modest: **four**
#: simultaneous full risk-off signals reach the floor, any one alone barely
#: registers.
#:
#: Four, not the original three, because there are now six signals rather than
#: four. Adding credit spreads and realised volatility at the old cut sizes
#: would have made the floor markedly easier to reach — two full breaches plus
#: two mild ones would have got there — which is a quietly more defensive system
#: than the one that was tuned and tested. The cuts were shrunk to hold the
#: overall posture roughly where it was while giving the new inputs a real vote.
#: A consequence worth knowing: the original four signals alone can no longer
#: floor the factor (3 full + 1 mild = 0.42, leaving 0.58). That is intended —
#: four of six confirming is the bar now.
_VIX_ELEVATED, _VIX_HIGH = 20.0, 30.0
_BREADTH_WEAK, _BREADTH_POOR = 0.45, 0.30
#: ICE BofA US high-yield option-adjusted spread, in percentage points. Credit
#: reprices before equity does, and unlike the yield curve it moves on a
#: timescale a position size can respond to.
_CREDIT_WIDE, _CREDIT_STRESSED = 5.5, 8.0
#: Annualised 20-day realised volatility of the S&P proxy. VIX is the market's
#: *expectation*; this is what actually happened. They diverge exactly when it
#: matters — a calm VIX during a grinding selloff is the case this catches.
_RVOL_ELEVATED, _RVOL_HIGH = 0.20, 0.30
#: Window for the realised-volatility read.
_RVOL_WINDOW = 20
_CUT_MILD, _CUT_FULL = 0.06, 0.12


@dataclass(slots=True)
class RegimeReading:
    as_of: date
    vix: float | None = None
    breadth: float | None = None
    breadth_sample: int = 0
    sp500_uptrend: bool | None = None
    world_uptrend: bool | None = None
    sp500_vs_200dma: float | None = None
    credit_spread: float | None = None
    sp500_realised_vol: float | None = None
    yield_curve_10y2y: float | None = None
    yield_curve_10y3m: float | None = None
    buffett_indicator: float | None = None
    risk_factor: float = 1.0
    posture: str = "normal"
    reasons: list[str] = field(default_factory=list)


class MarketRegimeService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def measure(self) -> RegimeReading:
        """Read every input, then reduce the fast ones to a single multiplier."""
        reading = RegimeReading(as_of=datetime.now(UTC).date())

        reading.vix = await self._vix()
        reading.breadth, reading.breadth_sample = await self._breadth()
        # The S&P proxy series is reused for realised volatility rather than
        # loaded twice — it is the same candles answering a second question.
        reading.sp500_uptrend, reading.sp500_vs_200dma, sp500_series = await self._regime(
            _SP500_PROXIES
        )
        reading.world_uptrend, _, _ = await self._regime(_WORLD_PROXIES)
        if sp500_series is not None:
            reading.sp500_realised_vol = ind.annualised_volatility(sp500_series.close, _RVOL_WINDOW)
        reading.credit_spread = await self._fred("BAMLH0A0HYM2")
        reading.yield_curve_10y2y = await self._fred("T10Y2Y")
        reading.yield_curve_10y3m = await self._fred("T10Y3M")
        reading.buffett_indicator = await self._buffett()

        self._score(reading)
        return reading

    # -- The verdict --------------------------------------------------------

    @staticmethod
    def _score(reading: RegimeReading) -> None:
        """Reduce the fast signals to a risk multiplier.

        VIX, breadth, index trend, realised volatility and credit spreads
        participate. The yield curve and the Buffett indicator are read and
        alerted on but excluded on purpose: both sit at extremes for years at a
        time, and a multiplier driven by them would be permanently depressed
        rather than responsive.

        A missing input contributes nothing rather than a penalty — a failed
        provider call must not look like a risk-off signal.
        """
        factor = 1.0
        reasons: list[str] = []

        if reading.vix is not None:
            if reading.vix >= _VIX_HIGH:
                factor -= _CUT_FULL
                reasons.append(f"VIX {reading.vix:.0f} (high)")
            elif reading.vix >= _VIX_ELEVATED:
                factor -= _CUT_MILD
                reasons.append(f"VIX {reading.vix:.0f} (elevated)")

        if reading.breadth is not None:
            if reading.breadth <= _BREADTH_POOR:
                factor -= _CUT_FULL
                reasons.append(f"breadth {reading.breadth:.0%} (poor)")
            elif reading.breadth <= _BREADTH_WEAK:
                factor -= _CUT_MILD
                reasons.append(f"breadth {reading.breadth:.0%} (weak)")

        if reading.sp500_uptrend is False:
            factor -= _CUT_FULL
            reasons.append("S&P below its 200-day average")
        if reading.world_uptrend is False:
            factor -= _CUT_MILD
            reasons.append("world index below its 200-day average")

        if reading.credit_spread is not None:
            if reading.credit_spread >= _CREDIT_STRESSED:
                factor -= _CUT_FULL
                reasons.append(f"high-yield spread {reading.credit_spread:.1f}% (stressed)")
            elif reading.credit_spread >= _CREDIT_WIDE:
                factor -= _CUT_MILD
                reasons.append(f"high-yield spread {reading.credit_spread:.1f}% (wide)")

        if reading.sp500_realised_vol is not None:
            if reading.sp500_realised_vol >= _RVOL_HIGH:
                factor -= _CUT_FULL
                reasons.append(f"realised volatility {reading.sp500_realised_vol:.0%} (high)")
            elif reading.sp500_realised_vol >= _RVOL_ELEVATED:
                factor -= _CUT_MILD
                reasons.append(f"realised volatility {reading.sp500_realised_vol:.0%} (elevated)")

        reading.risk_factor = max(MIN_RISK_FACTOR, round(factor, 4))
        reading.reasons = reasons
        if reading.risk_factor >= 0.99:
            reading.posture = "normal"
        elif reading.risk_factor >= 0.85:
            reading.posture = "cautious"
        elif reading.risk_factor >= 0.70:
            reading.posture = "defensive"
        else:
            reading.posture = "risk-off"

    # -- Inputs -------------------------------------------------------------

    async def _breadth(self) -> tuple[float | None, int]:
        """Fraction of instruments above their own 200-day average.

        One SQL pass over the local store rather than 15,000 indicator calls:
        the window function ranks each instrument's closes by recency, and the
        aggregate compares its latest close with the mean of its last 200.
        Instruments without a full 200 bars are excluded — a partial average is
        a different statistic, not a rough one.
        """
        rows = (
            await self._session.execute(
                text("""
                WITH ranked AS (
                    SELECT c.instrument_id, c.close,
                           row_number() OVER (
                               PARTITION BY c.instrument_id ORDER BY c.timestamp DESC
                           ) AS rn
                    FROM candles c
                    JOIN instruments i ON i.id = c.instrument_id
                    WHERE c.interval = '1d' AND c.is_closed
                      AND i.is_scanner_eligible AND i.suspended_at IS NULL
                ),
                per_instrument AS (
                    SELECT instrument_id,
                           max(close) FILTER (WHERE rn = 1) AS last_close,
                           avg(close) AS ma200,
                           count(*) AS bars
                    FROM ranked WHERE rn <= 200 GROUP BY instrument_id
                )
                SELECT count(*) FILTER (WHERE last_close > ma200) AS above,
                       count(*) AS total
                FROM per_instrument WHERE bars = 200
                """)
            )
        ).one()
        above, total = int(rows[0] or 0), int(rows[1] or 0)
        if total == 0:
            return None, 0
        return above / total, total

    async def _regime(
        self, proxies: tuple[tuple[str, str], ...]
    ) -> tuple[bool | None, float | None, PriceSeries | None]:
        """50/200 crossover state for the first proxy with enough history.

        Returns the series it used alongside the verdict, so a caller wanting a
        second statistic from the same candles does not reload them.
        """
        store = CandleStore(self._session)
        for ticker, mic in proxies:
            instrument = (
                (
                    await self._session.execute(
                        select(Instrument)
                        .join(Exchange, Exchange.id == Instrument.exchange_id)
                        .where(Instrument.exchange_ticker == ticker, Exchange.mic == mic)
                    )
                )
                .scalars()
                .first()
            )
            if instrument is None:
                continue
            candles = await store.get_candles(
                instrument.id, Interval.D1, limit=400, closed_only=True
            )
            if len(candles) < 200:
                continue
            series = candles_to_series(candles)
            ma50 = ind.simple_moving_average(series.close, 50)
            ma200 = ind.simple_moving_average(series.close, 200)
            if ma50 is None or ma200 is None or ma200 <= 0:
                continue
            last = float(series.close[-1])
            return ma50 > ma200, last / ma200 - 1.0, series
        return None, None, None

    @staticmethod
    async def _vix() -> float | None:
        def _fetch() -> float | None:
            import yfinance as yf

            frame = yf.Ticker("^VIX").history(period="5d", interval="1d")
            return None if frame.empty else float(frame["Close"].iloc[-1])

        try:
            return await asyncio.to_thread(_fetch)
        except Exception as exc:
            log.warning("regime.vix_failed", error=str(exc))
            return None

    @staticmethod
    async def _fred(series: str) -> float | None:
        """Latest non-missing value of a FRED series. No API key required."""
        try:
            async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
                response = await client.get(_FRED_CSV.format(series=series))
                response.raise_for_status()
            for line in reversed(response.text.strip().split("\n")[1:]):
                parts = line.split(",")
                # FRED marks missing observations with "."; skip rather than
                # carry a hole forward as a zero.
                if len(parts) == 2 and parts[1] not in (".", ""):
                    return float(parts[1])
        except Exception as exc:
            log.warning("regime.fred_failed", series=series, error=str(exc))
        return None

    async def _buffett(self) -> float | None:
        """Corporate equities over GDP, as a percentage.

        Both series are quarterly and the equities figure trails GDP by a
        quarter or more, so this is a description of valuation months ago. That
        is exactly why it is excluded from the risk factor.
        """
        equities = await self._fred("NCBEILQ027S")  # $ millions
        gdp = await self._fred("GDP")  # $ billions
        if equities is None or gdp is None or gdp <= 0:
            return None
        return (equities / 1000.0) / gdp * 100.0

    # -- Persistence --------------------------------------------------------

    async def record(self, reading: RegimeReading) -> MarketRegimeSnapshot:
        """Upsert today's snapshot, so re-running the job converges."""
        existing = (
            (
                await self._session.execute(
                    select(MarketRegimeSnapshot).where(MarketRegimeSnapshot.as_of == reading.as_of)
                )
            )
            .scalars()
            .first()
        )
        snapshot = existing or MarketRegimeSnapshot(as_of=reading.as_of)
        snapshot.vix = _dec(reading.vix)
        snapshot.breadth_above_200dma = _dec(reading.breadth)
        snapshot.breadth_sample = reading.breadth_sample
        snapshot.sp500_uptrend = reading.sp500_uptrend
        snapshot.world_uptrend = reading.world_uptrend
        snapshot.sp500_vs_200dma = _dec(reading.sp500_vs_200dma)
        snapshot.credit_spread = _dec(reading.credit_spread)
        snapshot.sp500_realised_vol = _dec(reading.sp500_realised_vol)
        snapshot.yield_curve_10y2y = _dec(reading.yield_curve_10y2y)
        snapshot.yield_curve_10y3m = _dec(reading.yield_curve_10y3m)
        snapshot.buffett_indicator = _dec(reading.buffett_indicator)
        snapshot.risk_factor = Decimal(str(reading.risk_factor))
        snapshot.posture = reading.posture
        snapshot.detail = "; ".join(reading.reasons) or "no risk-off signals"
        if existing is None:
            self._session.add(snapshot)
        await self._session.flush()
        return snapshot

    async def previous(self, before: date) -> MarketRegimeSnapshot | None:
        """The most recent snapshot before `before` — what alerts compare against."""
        return (
            (
                await self._session.execute(
                    select(MarketRegimeSnapshot)
                    .where(MarketRegimeSnapshot.as_of < before)
                    .order_by(MarketRegimeSnapshot.as_of.desc())
                    .limit(1)
                )
            )
            .scalars()
            .first()
        )

    async def current_risk_factor(self) -> float:
        """Today's multiplier, or 1.0 when nothing has been measured yet.

        Defaults to *no adjustment* rather than a cautious one: an absent
        measurement is missing information, and inventing a penalty from it
        would quietly shrink every position for a reason nobody chose.
        """
        snapshot = (
            (
                await self._session.execute(
                    select(MarketRegimeSnapshot)
                    .order_by(MarketRegimeSnapshot.as_of.desc())
                    .limit(1)
                )
            )
            .scalars()
            .first()
        )
        if snapshot is None or snapshot.risk_factor is None:
            return 1.0
        return max(MIN_RISK_FACTOR, min(1.0, float(snapshot.risk_factor)))


def _dec(value: float | None) -> Decimal | None:
    return None if value is None else Decimal(str(round(value, 6)))
