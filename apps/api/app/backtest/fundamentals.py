"""Fundamentals as they were known on the day, from SEC filings.

The scanner is fundamentals-first: `value` alone carries 30 of the 100 points
and the business third of `quality` carries more. Every backtest of it so far
has left all of that out, because the only fundamentals the system has are a
yfinance snapshot of *today* — and scoring a 2015 decision with 2026 earnings is
not a backtest, it is reading the answer.

This module supplies the missing half honestly. SEC XBRL company facts carry,
for every reported figure, the date the filing became public. Ask for the
figures filed on or before a decision date and you get what the market actually
knew, restatements and all.

## Why this is not merely "more data"

Using today's numbers would not just flatter the result, it would make it
uninterpretable, because the two halves pull in **opposite** directions:

  * `value` measures cheapness against *today's price*. A company that has
    since collapsed looks cheap now, so today's data makes losers score high.
  * `quality` measures margins and returns *today*. A company that has since
    thrived looks sound now, so today's data makes winners score high.

The errors do not share a sign, so the result could not even be read as "true
value is somewhat lower than this". Point-in-time data is the only version of
this test worth running.

## What is deliberately crude

**Annual figures only.** Facts are taken from filings covering about a year,
which means a decision in November scores on a fiscal year that ended up to
fifteen months earlier. Quarterly assembly would be fresher and needs
trailing-twelve-month arithmetic across fiscal calendars that do not align
between companies; the staleness here is a known, one-directional handicap —
the score is *behind* the market, never ahead of it — which is the safe
direction for a backtest to be wrong in.

**Restatements resolve to what was visible.** Among the facts filed on or before
the decision date, the one filed *most recently* wins for each period. If a
figure was restated in March, a decision in April sees the restated number,
because that is what a reader had. Apple's FY2023 earnings, for instance, appear
again in the FY2025 filing with a 2025 filing date; before that date, the
original stands.

**US filers only.** EDGAR covers SEC registrants, so this cannot be extended to
the European half of the broker catalogue. Any point-in-time fundamental test
here is a test on US names.
"""

from __future__ import annotations

import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

#: SEC asks for a contactable agent and throttles hard without one.
USER_AGENT = "Stock-Automate research chinjyanson2003@gmail.com"

TICKER_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

#: The concepts each metric may appear under, in preference order. XBRL lets a
#: filer choose, and they do: Visa reports no `EarningsPerShareDiluted`, Goldman
#: no `Revenues`. A single tag per metric silently loses those companies.
TAGS: dict[str, tuple[str, ...]] = {
    "eps": (
        "EarningsPerShareDiluted",
        "EarningsPerShareBasicAndDiluted",
        "EarningsPerShareBasic",
        "IncomeLossFromContinuingOperationsPerDilutedShare",
    ),
    "revenue": (
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet",
        "RevenuesNetOfInterestExpense",
    ),
    "net_income": ("NetIncomeLoss", "ProfitLoss"),
    "equity": (
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ),
    "shares": (
        "WeightedAverageNumberOfDilutedSharesOutstanding",
        "CommonStockSharesOutstanding",
        "WeightedAverageNumberOfSharesOutstandingBasic",
    ),
    "debt": (
        "LongTermDebtNoncurrent",
        "LongTermDebt",
        "DebtLongtermAndShorttermCombinedAmount",
    ),
    "dividend": (
        "CommonStockDividendsPerShareDeclared",
        "CommonStockDividendsPerShareCashPaid",
    ),
}

#: A duration fact counts as annual within this many days of a year. Fiscal
#: years wander by a week; 52-53 week retail calendars wander by more.
ANNUAL_LOW, ANNUAL_HIGH = 340, 400


@dataclass(frozen=True, slots=True)
class Fact:
    """One reported number, and the day it became public."""

    end: str
    filed: str
    value: float


@dataclass(frozen=True, slots=True)
class Company:
    ticker: str
    cik: int
    #: metric -> facts, ascending by period end.
    series: dict[str, tuple[Fact, ...]]


def _get(url: str) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    return json.loads(urllib.request.urlopen(request, timeout=60).read())


def cik_map() -> dict[str, int]:
    """Ticker to CIK. EDGAR is keyed by company number, never by symbol."""
    return {entry["ticker"]: int(entry["cik_str"]) for entry in _get(TICKER_URL).values()}


def _extract(gaap: dict[str, Any], names: tuple[str, ...]) -> tuple[Fact, ...]:
    """Annual facts for a metric, merged across every concept that carries it.

    Merged rather than "the first concept present", which was the first version
    and was wrong in a way that produced plausible numbers. Apple reports under
    both `Revenues` (eleven facts, abandoned years ago) and
    `RevenueFromContractWithCustomerExcludingAssessedTax` (a hundred and
    seventeen, current). Taking the first match picked the dead tag, and every
    revenue-growth reading after 2018 silently compared the same two ancient
    years to each other.

    The same shape of problem arrives from the other direction when a filer
    *switches* concepts mid-history — ASC 606 moved most of the market onto new
    revenue tags in 2018 — so "whichever tag has the most facts" would lose
    everything before the switch. Merging keeps both eras; where two concepts
    report the same period, the earlier entry in `names` wins.
    """
    # Local, because `fetch` runs this on five threads at once and a module
    # level scratch dictionary would let one company's periods overwrite
    # another's.
    chosen: dict[tuple[str, str], tuple[int, Fact]] = {}
    for rank, name in enumerate(names):
        concept = gaap.get(name)
        if not concept:
            continue
        for entries in concept.get("units", {}).values():
            for entry in entries:
                end, filed = entry.get("end"), entry.get("filed")
                value = entry.get("val")
                if not end or not filed or value is None:
                    continue
                start = entry.get("start")
                if start is not None:
                    span = (_days(end) - _days(start)).days
                    if not ANNUAL_LOW <= span <= ANNUAL_HIGH:
                        continue
                key = (end, filed)
                held = chosen.get(key)
                if held is None or rank < held[0]:
                    chosen[key] = (rank, Fact(end=end, filed=filed, value=float(value)))
    return tuple(sorted((f for _, f in chosen.values()), key=lambda f: (f.end, f.filed)))


def _days(text: str) -> Any:
    from datetime import date

    year, month, day = (int(p) for p in text.split("-"))
    return date(year, month, day)


def fetch(tickers: tuple[str, ...], cache: Path, *, refresh: bool = False) -> dict[str, Company]:
    """Download and distil company facts, keeping only what the score reads.

    A full `companyfacts` document runs to several megabytes and five hundred
    concepts; seven of them are wanted. Distilling on the way in turns two
    gigabytes of download into a cache of a few megabytes that later runs open
    instantly.
    """
    if cache.exists() and not refresh:
        stored = json.loads(cache.read_text())
        cached: dict[str, Company] = {
            ticker: Company(
                ticker=ticker,
                cik=int(body["cik"]),
                series={
                    metric: tuple(Fact(**fact) for fact in facts)
                    for metric, facts in body["series"].items()
                },
            )
            for ticker, body in stored.items()
        }
        # Only reuse the cache if it covers everything asked for; a partial
        # cache silently scoring some names without fundamentals is exactly the
        # kind of quiet degradation this whole module exists to avoid.
        if set(cached) >= set(tickers):
            return cached

    numbers = cik_map()

    def one(ticker: str) -> tuple[str, Company | None]:
        cik = numbers.get(ticker) or numbers.get(ticker.replace(".", "-"))
        if cik is None:
            return ticker, None
        try:
            facts = _get(FACTS_URL.format(cik=cik))
        except Exception:
            return ticker, None
        gaap = facts.get("facts", {}).get("us-gaap", {})
        series = {metric: _extract(gaap, names) for metric, names in TAGS.items()}
        return ticker, Company(ticker=ticker, cik=cik, series=series)

    # Five at a time: SEC publishes a ten-per-second ceiling and this stays
    # comfortably under it while still finishing five hundred names in minutes.
    companies: dict[str, Company] = {}
    with ThreadPoolExecutor(max_workers=5) as pool:
        for index, (ticker, company) in enumerate(pool.map(one, tickers), 1):
            if company is not None:
                companies[ticker] = company
            if index % 50 == 0 or index == len(tickers):
                print(f"    fetched {index:,}/{len(tickers):,}", flush=True)

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(
        json.dumps(
            {
                ticker: {
                    "cik": company.cik,
                    "series": {
                        # Written out rather than `vars()`: these are slotted
                        # dataclasses and have no instance dictionary.
                        metric: [{"end": f.end, "filed": f.filed, "value": f.value} for f in facts]
                        for metric, facts in company.series.items()
                    },
                }
                for ticker, company in companies.items()
            }
        )
    )
    return companies


def splits(
    tickers: tuple[str, ...], cache: Path, *, refresh: bool = False
) -> dict[str, list[tuple[str, float]]]:
    """Every share split per ticker, as `(date, ratio)` pairs.

    Needed to reconcile as-filed per-share figures with an adjusted price
    series; see `reading`. Fetched from yfinance rather than EDGAR because the
    split is a market event rather than a reported figure, and because
    `Ticker.splits` is one small request per name.
    """
    if cache.exists() and not refresh:
        stored: dict[str, list[tuple[str, float]]] = {
            ticker: [(day, float(ratio)) for day, ratio in events]
            for ticker, events in json.loads(cache.read_text()).items()
        }
        if set(stored) >= set(tickers):
            return stored

    import yfinance as yf

    out: dict[str, list[tuple[str, float]]] = {}
    for index, ticker in enumerate(tickers, 1):
        try:
            series = yf.Ticker(ticker).splits
            out[ticker] = [
                (str(stamp.date()), float(ratio)) for stamp, ratio in series.items() if ratio > 0
            ]
        except Exception:
            out[ticker] = []
        if index % 100 == 0 or index == len(tickers):
            print(f"    splits {index:,}/{len(tickers):,}", flush=True)

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out))
    return out


def split_factor(events: list[tuple[str, float]], on: str) -> float:
    """How many of today's shares one share held on `on` has become.

    The product of every split *after* the date, which is exactly the factor a
    price series has already been divided by — so dividing a filed per-share
    figure by it puts the two on one basis.
    """
    factor = 1.0
    for day, ratio in events:
        if day > on:
            factor *= ratio
    return factor


def _latest(facts: tuple[Fact, ...], on: str, *, skip: int = 0) -> Fact | None:
    """The newest period visible on `on`, or the one `skip` years before it.

    Two filters, and the order matters. First discard anything filed after the
    decision — that is the point-in-time rule. Then, for each period that
    remains, keep the most recently filed version of it, which is what a reader
    on that day would have seen if it had been restated in the meantime.
    """
    visible = [f for f in facts if f.filed <= on]
    if not visible:
        return None
    newest: dict[str, Fact] = {}
    for fact in visible:
        held = newest.get(fact.end)
        if held is None or fact.filed > held.filed:
            newest[fact.end] = fact
    ordered = sorted(newest.values(), key=lambda f: f.end, reverse=True)
    return ordered[skip] if len(ordered) > skip else None


def _ratio(top: float | None, bottom: float | None) -> Decimal | None:
    if top is None or bottom is None or bottom == 0:
        return None
    return Decimal(str(top / bottom))


def reading(
    company: Company,
    on: str,
    price: float,
    *,
    split_factor: float = 1.0,
) -> dict[str, Decimal | None]:
    """The seven figures `score_series` reads, as of `on`.

    `price` is the close on that day, so the ratios are struck against what the
    market was charging then rather than what it charges now. A key that cannot
    be computed is returned as None, which the scanner treats as "unavailable"
    and renormalises around — the same thing it does in production.

    ## `split_factor`, and why leaving it out produced believable nonsense

    EDGAR reports per-share figures **as filed**. A price series is
    split-adjusted. Apple split four-for-one in 2020, so its 2015 filing says
    $9.22 of earnings per share against an adjusted January-2016 price of about
    $26 — a price/earnings ratio of 2.9, for a company that actually traded near
    11. Cheap by a factor of four, in the group that carries the most weight in
    the whole score, for every company in the years before a split.

    `split_factor` is the cumulative number of shares that one share held on
    `on` has become by the end of the price series. Dividing the filed per-share
    figures by it puts them on the same basis as the adjusted price. Ratios with
    no per-share term in them — margin, growth in totals, debt to equity — are
    untouched by any of this, which is why they are computed from totals
    wherever there is a choice.
    """

    def value(metric: str, *, skip: int = 0) -> float | None:
        fact = _latest(company.series.get(metric, ()), on, skip=skip)
        return None if fact is None else fact.value

    equity, shares = value("equity"), value("shares")
    revenue, income, debt = value("revenue"), value("net_income"), value("debt")
    income_prior, revenue_prior = value("net_income", skip=1), value("revenue", skip=1)

    factor = split_factor if split_factor > 0 else 1.0
    eps = value("eps")
    eps = None if eps is None else eps / factor
    dividend = value("dividend")
    dividend = None if dividend is None else dividend / factor
    book_per_share = equity / (shares * factor) if equity is not None and shares else None

    return {
        # Negative earnings make a P/E meaningless rather than merely bad, and
        # the scanner reads a low ratio as cheap — so a loss-maker with a
        # negative ratio would rank as the cheapest name in the universe.
        "trailing_pe": _ratio(price, eps) if eps and eps > 0 else None,
        "price_to_book": _ratio(price, book_per_share) if book_per_share else None,
        # Growth from *totals* rather than per-share figures, so a split in the
        # middle of the two years being compared cannot masquerade as a
        # collapse in earnings.
        "earnings_growth": (
            _ratio(income - income_prior, abs(income_prior))
            if income is not None and income_prior not in (None, 0)
            else None
        ),
        "revenue_growth": (
            _ratio(revenue - revenue_prior, abs(revenue_prior))
            if revenue is not None and revenue_prior not in (None, 0)
            else None
        ),
        "profit_margin": _ratio(income, revenue),
        "debt_to_equity": _ratio(debt, equity),
        "dividend_yield": _ratio(dividend, price),
    }
