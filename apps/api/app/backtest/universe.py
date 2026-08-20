"""The price history a scanner backtest needs, fetched once and cached.

The local candle store cannot answer this question. It holds 3.4M bars, but
9,834 of its 12,800 instruments carry 250-499 of them and only **156 reach four
years** — the Trading 212 catalogue is largely young listings, and no amount of
re-fetching deepens a series that does not exist. A multi-year ranking test
therefore needs a different universe, and this module fetches one.

## What is being traded away

The universe here is **today's** S&P 500. That is survivorship bias, stated
plainly rather than buried: companies that failed out of the index are absent,
so every strategy measured on it — including buy and hold — looks better than it
would have. The mitigation is the comparison itself. The headline control is an
equal-weighted holding of *the same 503 names*, which carries exactly the same
bias, so the difference between "rank them" and "own all of them" is close to
unbiased even though both levels are flattered. Only the SPY comparison is
distorted, and it is reported as context, not as the verdict.
"""

from __future__ import annotations

import re
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CONSTITUENTS_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

#: GICS sector -> the ETF the scanner's `sector` group reads.
#:
#: Two of these are not the obvious SPDR. XLC (Communication Services) and XLRE
#: (Real Estate) were only listed in 2018 and 2015, and a sector proxy that
#: begins mid-backtest would silently drop those names' sector group for the
#: first half of the run — a change in *what is being scored*, dressed up as a
#: change in score. VOX and VNQ reach back to 2004 and keep the scoring
#: consistent across the whole period.
SECTOR_ETFS: dict[str, str] = {
    "Information Technology": "XLK",
    "Health Care": "XLV",
    "Financials": "XLF",
    "Consumer Discretionary": "XLY",
    "Communication Services": "VOX",
    "Industrials": "XLI",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Utilities": "XLU",
    "Real Estate": "VNQ",
    "Materials": "XLB",
}

#: The market proxy, and the bond *price* proxy for the rate-sensitivity signal.
#: TLT rather than ^TNX deliberately: `scoring._score_risk` documents that a
#: yield series would silently invert the correlation's sign.
BENCHMARK = "SPY"
RATES = "TLT"

#: The real equal-weighted S&P 500. Not a benchmark for the strategy — it is the
#: yardstick for this harness's *own* survivorship bias. It equal-weights whoever
#: was in the index at the time; the backtest's control equal-weights whoever is
#: in it today. Their difference is the bias, in percent per year.
EQUAL_WEIGHT = "RSP"

FIELDS = ("open", "high", "low", "close", "adjusted_close", "volume")

#: Tickers per download call. Large enough that the per-request overhead is
#: irrelevant, small enough that a failure loses seconds rather than an hour.
CHUNK = 500

#: Attempts per chunk, and the base seconds between calls. yfinance answers a
#: rate limit with an *empty frame* rather than an exception, so patience is
#: the only available remedy and silence is not evidence of success.
RETRIES = 4
PAUSE = 2.0

#: Below this share of symbols carrying any price at all, the download is
#: treated as failed rather than sparse. Real coverage of a broker catalogue
#: runs far above it; a rate-limited run lands far below.
MIN_COVERAGE = 0.5


@dataclass(frozen=True, slots=True)
class Panel:
    """Aligned OHLCV for every symbol, on one shared trading calendar.

    Each field is a `(days, symbols)` array with NaN wherever a symbol had not
    listed yet. One shared date axis is what makes a point-in-time cut cheap: a
    single `searchsorted` locates the same bar for every name at once.
    """

    dates: np.ndarray  # datetime64[D], ascending
    symbols: tuple[str, ...]
    sectors: dict[str, str]
    fields: dict[str, np.ndarray]

    def column(self, symbol: str) -> int:
        return self.symbols.index(symbol)


def constituents(cache: Path) -> pd.DataFrame:
    """S&P 500 symbols and their GICS sector, cached to CSV after one fetch."""
    if cache.exists():
        return pd.read_csv(cache)

    request = urllib.request.Request(CONSTITUENTS_URL, headers={"User-Agent": "Mozilla/5.0"})
    html = urllib.request.urlopen(request, timeout=30).read().decode()
    anchor = html.find('id="constituents"')
    if anchor < 0:
        raise RuntimeError("constituents table not found; the page layout changed")

    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html[anchor : anchor + 500_000], re.S)
    records: list[dict[str, str]] = []
    for row in rows:
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.S)
        if len(cells) < 5:
            continue
        text = [re.sub(r"<[^>]+>", " ", c).replace("&amp;", "&").strip() for c in cells]
        if text[0] == "Symbol":
            continue
        records.append({"symbol": text[0].replace(".", "-"), "sector": text[2]})

    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        raise RuntimeError("constituents table parsed to nothing; the page layout changed")
    cache.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cache, index=False)
    return frame


def load(cache: Path, since: str, *, refresh: bool = False) -> Panel:
    """Fetch (or reload) the whole panel: constituents, sector ETFs, SPY, TLT."""
    members = constituents(cache.parent / "sp500_constituents.csv")
    sectors = dict(zip(members["symbol"], members["sector"], strict=True))
    return build(tuple(sorted(set(members["symbol"]))), sectors, cache, since, refresh=refresh)


def build(
    names: tuple[str, ...],
    sectors: dict[str, str],
    cache: Path,
    since: str,
    *,
    refresh: bool = False,
) -> Panel:
    """Download `names` plus the proxies every score needs, and cache the lot.

    Split out from `load` so that a universe other than the S&P 500 — the whole
    broker catalogue, say — gets the identical download, alignment and caching
    rather than a second implementation of them that drifts.

    `sectors` maps a symbol to a **GICS** sector name, the keys of
    `SECTOR_ETFS`. A symbol missing from it is scored without its sector group,
    which `combine_score` handles by renormalising — the same thing production
    does for a name whose sector is unknown.

    Cached as a single compressed `.npz`, because the alternative on this box is
    a ~300MB CSV — pyarrow is not installed and this is not worth a dependency.
    """
    cache.parent.mkdir(parents=True, exist_ok=True)
    extras = sorted({*SECTOR_ETFS.values(), BENCHMARK, RATES, EQUAL_WEIGHT})
    symbols = tuple(list(names) + [e for e in extras if e not in names])

    if cache.exists() and not refresh:
        stored = np.load(cache, allow_pickle=False)
        cached_symbols = tuple(str(s) for s in stored["symbols"])
        if cached_symbols == symbols:
            return Panel(
                dates=stored["dates"],
                symbols=cached_symbols,
                sectors=sectors,
                fields={f: stored[f] for f in FIELDS},
            )

    import yfinance as yf

    wanted = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "adjusted_close": "Adj Close",
        "volume": "Volume",
    }

    # Chunked, because this is now asked for the whole broker catalogue and not
    # just an index. One call for fifteen thousand tickers builds a multi-
    # gigabyte frame in a single response and gives no way to tell a slow
    # download from a hung one; five hundred at a time costs nothing and
    # reports progress. The chunks are re-aligned on a union calendar below,
    # so a batch whose members all happen to be closed on some holiday cannot
    # shift another batch's rows.
    pieces: list[pd.DataFrame] = []
    failed: list[str] = []
    for start in range(0, len(symbols), CHUNK):
        batch = list(symbols[start : start + CHUNK])
        part = None
        for attempt in range(RETRIES):
            part = yf.download(
                batch,
                start=since,
                interval="1d",
                auto_adjust=False,
                actions=False,
                progress=False,
                group_by="column",
                threads=True,
            )
            if part is not None and not part.empty:
                break
            # An empty frame here is almost always the rate limiter, which
            # yfinance reports by returning nothing rather than raising. Backing
            # off and retrying is the difference between a complete panel and a
            # panel that is four-fifths NaN.
            time.sleep(PAUSE * (2**attempt))
        if part is not None and not part.empty:
            pieces.append(part)
        else:
            failed.extend(batch)
        done = min(start + CHUNK, len(symbols))
        print(f"    downloaded {done:,}/{len(symbols):,} ({len(failed):,} unfetched)", flush=True)
        time.sleep(PAUSE)

    if not pieces:
        raise RuntimeError("yfinance returned nothing for the whole universe")
    raw = pd.concat(pieces, axis=1) if len(pieces) > 1 else pieces[0]

    dates = pd.DatetimeIndex(raw.index).tz_localize(None).normalize().to_numpy("datetime64[D]")
    fields: dict[str, np.ndarray] = {}
    for name, column in wanted.items():
        if column not in raw.columns.get_level_values(0):
            raise RuntimeError(f"yfinance did not return {column!r}; check auto_adjust")
        # Wrapped rather than indexed straight, because after concatenating the
        # chunks the static type of `raw[column]` widens to Series-or-frame.
        frame = pd.DataFrame(raw[column]).reindex(columns=list(symbols))
        fields[name] = frame.to_numpy(dtype=np.float64)

    # `reindex` fills a symbol the download never returned with a column of
    # NaN, which is indistinguishable downstream from a symbol that had not
    # listed yet — and a whole panel of those reads as "nothing was scoreable"
    # rather than "the download failed". It happened: thirty-one chunks fired
    # back to back tripped the rate limiter, four fifths came back empty, and
    # the run produced a confident, wrong, empty answer. Fail here instead.
    covered = np.isfinite(fields["adjusted_close"]).any(axis=0)
    share = float(covered.mean())
    if share < MIN_COVERAGE:
        raise RuntimeError(
            f"only {covered.sum():,} of {len(symbols):,} symbols ({share:.0%}) came back with "
            f"any prices, below the {MIN_COVERAGE:.0%} floor — most likely rate limiting. "
            f"Re-run with --refresh; the partial download has not been cached."
        )
    print(f"    {covered.sum():,} of {len(symbols):,} symbols carry prices ({share:.0%})")

    arrays: dict[str, np.ndarray] = {"dates": dates, "symbols": np.array(symbols), **fields}
    np.savez_compressed(cache, **arrays)  # type: ignore[arg-type]  # numpy stubs say bool
    return Panel(dates=dates, symbols=symbols, sectors=sectors, fields=fields)
