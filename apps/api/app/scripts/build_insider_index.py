"""Aggregate insider buying across the whole market, as a market-timing signal.

    python -m app.scripts.build_insider_index --from 2006 --to 2026

The insider data already in this database covers 175 micro-caps over three
months, which is the wrong companies and far too short a history to time an
index with. This builds the signal properly from the SEC's own structured
quarterly datasets, which carry every Form 3/4/5 filed since 2006 — roughly 14MB
a quarter, free, and fully backfillable. That last property is what separates
this from dealer gamma: the filings are a permanent public record rather than a
snapshot that evaporates.

**What is computed.** For each day, the ratio of open-market *purchases* to
total open-market transactions across all filers:

    buy share = P / (P + S)

using transaction code `P` (open-market purchase) and `S` (open-market sale) and
nothing else. Option exercises, gifts, grants and 10b5-1 dispositions are
excluded deliberately — an executive exercising options on a schedule is not
expressing a view, and including those swamps the signal with noise that has no
information in it.

**Why the aggregate rather than per company.** A single insider buying their own
stock says something about that stock. *Insiders in aggregate* buying says
something about the market, and the academic work on this (Lakonishok and Lee;
Jeng, Metrick and Zeckhauser) finds the aggregate is the more reliable of the
two for predicting market direction. It is also the only version that can be
tested here, since the constituents of an index change over time and historical
membership is not free.

Two versions are produced: all filers, and officers and directors only. The
second is narrower and slower to move but is the one with a real information
claim behind it — a director buys because they think the shares are cheap, where
a ten-percent holder may be doing something else entirely.

Writes a CSV so the download is paid for once. `rank_macro_signals` and the
model scripts read it from there.
"""

from __future__ import annotations

import argparse
import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

BASE = "https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets"

#: The SEC asks for a contact address in the User-Agent and rate-limits without
#: one. This is a courtesy requirement, not authentication.
HEADERS = {"User-Agent": "Stock-Automate research chinjyanson2003@gmail.com"}

#: Open-market purchase and sale. Everything else — option exercises (M),
#: grants (A), gifts (G), tax withholding (F) — is excluded: they are scheduled
#: or mechanical, and carry no view about price.
BUY_CODE = "P"
SELL_CODE = "S"

#: Relationship flags in REPORTINGOWNER, as a substring test on a field that is
#: formatted inconsistently across years.
INSIDER_ROLES = ("Officer", "Director")


def _quarter(year: int, quarter: int) -> pd.DataFrame | None:
    """One quarter's open-market transactions, or None if unavailable."""
    url = f"{BASE}/{year}q{quarter}_form345.zip"
    try:
        request = urllib.request.Request(url, headers=HEADERS)
        payload = urllib.request.urlopen(request, timeout=120).read()
    except Exception:
        return None

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = set(archive.namelist())
        if not {"NONDERIV_TRANS.tsv", "SUBMISSION.tsv", "REPORTINGOWNER.tsv"} <= names:
            return None
        transactions = pd.read_csv(
            archive.open("NONDERIV_TRANS.tsv"),
            sep="\t",
            usecols=["ACCESSION_NUMBER", "TRANS_DATE", "TRANS_CODE", "TRANS_SHARES"],
            low_memory=False,
        )
        owners = pd.read_csv(
            archive.open("REPORTINGOWNER.tsv"),
            sep="\t",
            usecols=["ACCESSION_NUMBER", "RPTOWNER_RELATIONSHIP"],
            low_memory=False,
        )

    transactions = transactions[transactions["TRANS_CODE"].isin([BUY_CODE, SELL_CODE])]
    if transactions.empty:
        return None

    # One filing can list several owners; keep one relationship per filing.
    owners = owners.drop_duplicates("ACCESSION_NUMBER")
    merged = transactions.merge(owners, on="ACCESSION_NUMBER", how="left")
    relationship = merged["RPTOWNER_RELATIONSHIP"].fillna("")
    merged["is_insider"] = relationship.str.contains("|".join(INSIDER_ROLES), case=False)

    # `DD-MMM-YYYY`, stated explicitly. Left to infer, pandas resolves some rows
    # day-first and others year-first and produces transaction dates in 1983 and
    # 2033 — neither of which the dataset contains, and both of which sail
    # through silently because a bad date is still a date.
    merged["date"] = pd.to_datetime(merged["TRANS_DATE"], errors="coerce", format="%d-%b-%Y")
    merged = merged.dropna(subset=["date"])
    # A filing may legitimately report a transaction from an earlier quarter, so
    # dates outside the file's own quarter are expected; dates outside the
    # dataset's lifetime are not.
    merged = merged[(merged["date"] >= "2003-01-01") & (merged["date"] <= pd.Timestamp.today())]
    merged["is_buy"] = merged["TRANS_CODE"] == BUY_CODE
    return merged[["date", "is_buy", "is_insider"]]


def _aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    """Daily buy share, for all filers and for officers/directors only."""
    grouped = frame.groupby("date")
    out = pd.DataFrame(
        {
            "buys": grouped["is_buy"].sum(),
            "total": grouped["is_buy"].count(),
        }
    )
    insiders = frame[frame["is_insider"]].groupby("date")
    out["officer_buys"] = insiders["is_buy"].sum()
    out["officer_total"] = insiders["is_buy"].count()
    out["buy_share"] = out["buys"] / out["total"].replace(0, pd.NA)
    out["officer_buy_share"] = out["officer_buys"] / out["officer_total"].replace(0, pd.NA)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an aggregate insider-buying index.")
    parser.add_argument("--from", dest="start", type=int, default=2006)
    parser.add_argument("--to", dest="end", type=int, default=2026)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/insider_index.csv"),
        help="Where to write the daily series.",
    )
    args = parser.parse_args()

    collected: list[pd.DataFrame] = []
    for year in range(args.start, args.end + 1):
        for quarter in (1, 2, 3, 4):
            frame = _quarter(year, quarter)
            if frame is None:
                continue
            collected.append(frame)
            print(f"  {year}q{quarter}  {len(frame):>8,} open-market transactions")

    if not collected:
        print("Nothing downloaded.")
        return

    everything = pd.concat(collected, ignore_index=True)
    daily = _aggregate(everything)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(args.out)

    print(f"\n{len(everything):,} transactions -> {len(daily):,} days")
    print(f"  median daily buy share            {daily['buy_share'].median():.1%}")
    print(f"  median officer/director buy share {daily['officer_buy_share'].median():.1%}")
    print(f"  written to {args.out}")


if __name__ == "__main__":
    main()
