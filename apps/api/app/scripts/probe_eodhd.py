"""Measure EODHD fundamentals: coverage % and the real free-tier rate limit.

    python -m app.scripts.probe_eodhd --limit 15

Samples mapped instruments across venues, asks EODHD for each one's fundamentals,
and reports how many calls succeeded before any quota error (the real limit) plus
how often each field was present (coverage) — the two numbers needed to decide
whether EODHD is worth wiring into scoring. Kept small by default: EODHD
fundamentals calls are credit-expensive on the free tier.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter

from sqlalchemy import text

from app.config import get_settings
from app.data.eodhd_provider import EODHDProvider, eodhd_symbol
from app.data.types import (
    ProviderQuotaExceededError,
    ProviderSymbolNotFoundError,
)
from app.db import session_scope

_FIELDS = ("market_cap", "trailing_pe", "price_to_book", "dividend_yield", "profit_margin")


async def _run(limit: int) -> None:
    settings = get_settings()
    key = settings.eodhd_api_key
    if key is None or not key.get_secret_value():
        print("EODHD_API_KEY is not set — nothing to probe.")
        return

    # Spread the sample across venues rather than clustering on one.
    async with session_scope() as s:
        rows = (
            await s.execute(
                text(
                    "select e.mic, i.exchange_ticker from instruments i "
                    "join market_data_mappings m on m.instrument_id = i.id "
                    "join exchanges e on e.id = i.exchange_id "
                    "where m.provider='yfinance' and m.is_signal_source "
                    "and i.exchange_ticker is not null "
                    "order by random() limit :n"
                ),
                {"n": limit},
            )
        ).all()

    provider = EODHDProvider(api_key=key.get_secret_value(), base_url=settings.eodhd_base_url)
    succeeded = 0
    not_found = 0
    unsupported = 0
    field_hits: Counter[str] = Counter()
    per_venue: Counter[str] = Counter()
    per_venue_ok: Counter[str] = Counter()
    quota_hit_after: int | None = None

    try:
        for i, (mic, ticker) in enumerate(rows, start=1):
            sym = eodhd_symbol(ticker, mic)
            per_venue[mic] += 1
            if sym is None:
                unsupported += 1
                print(f"{i:3} {mic:6} {ticker:12} — unsupported venue")
                continue
            try:
                f = await provider.get_basic_fundamentals(sym)
            except ProviderQuotaExceededError as exc:
                quota_hit_after = i - 1
                print(f"\n>>> QUOTA/RATE LIMIT hit after {quota_hit_after} calls: {exc}")
                break
            except ProviderSymbolNotFoundError:
                not_found += 1
                print(f"{i:3} {mic:6} {sym:14} — not found")
                continue
            except Exception as exc:
                print(f"{i:3} {mic:6} {sym:14} — error: {exc}")
                continue

            present = [name for name in _FIELDS if getattr(f, name) is not None]
            for name in present:
                field_hits[name] += 1
            if present:
                succeeded += 1
                per_venue_ok[mic] += 1
            print(f"{i:3} {mic:6} {sym:14} — {len(present)}/{len(_FIELDS)} fields: {present}")
    finally:
        await provider.close()

    attempted = quota_hit_after if quota_hit_after is not None else len(rows)
    print("\n==================== SUMMARY ====================")
    rate_limit = (
        f"hit after {quota_hit_after} calls"
        if quota_hit_after is not None
        else f"not hit in {len(rows)} calls"
    )
    print(f"rate limit:  {rate_limit}")
    print(f"attempted:   {attempted}")
    print(
        f"had data:    {succeeded}  ({100 * succeeded / attempted:.0f}%)"
        if attempted
        else "attempted: 0"
    )
    print(f"not found:   {not_found}   unsupported venue: {unsupported}")
    print("field coverage (of attempts):")
    for name in _FIELDS:
        hits = field_hits[name]
        print(f"    {name:16} {hits:3}  ({100 * hits / attempted:.0f}%)" if attempted else name)
    print("per-venue coverage:")
    for mic in sorted(per_venue):
        tot = per_venue[mic]
        print(f"    {mic:6} {per_venue_ok[mic]}/{tot}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe EODHD fundamentals coverage + rate limit.")
    parser.add_argument("--limit", type=int, default=15, help="Symbols to sample (keep small).")
    args = parser.parse_args()
    asyncio.run(_run(args.limit))


if __name__ == "__main__":
    main()
