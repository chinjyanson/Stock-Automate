"""SEC EDGAR Form 4 reader (§4).

Free, unauthenticated, and near real-time — the newest filing on the feed is
typically minutes old. That is the whole reason this exists rather than a
scraped aggregator: the public secform4 page is delayed roughly six months,
which is long enough to make an insider signal untradeable.

Two obligations the SEC places on automated readers, both honoured here:

  * A descriptive `User-Agent` carrying a contact address. Requests without one
    are refused, and the address is how they reach you before blocking.
  * At most 10 requests/second. This client is deliberately sequential with a
    small delay — a Form 4 sweep is not latency-critical, and being throttled
    off the feed costs far more than the seconds saved.

The parser reads the ownership XML rather than the rendered HTML. The XML is a
stable, documented schema; the HTML is a stylesheet's output and changes without
notice.
"""

from __future__ import annotations

import asyncio
import html
import re
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation

import httpx
import structlog

log = structlog.get_logger(__name__)

_BASE = "https://www.sec.gov"
_BROWSE = f"{_BASE}/cgi-bin/browse-edgar"

#: SEC's published ceiling is 10 req/s; this leaves generous headroom because
#: nothing here is time-critical to the second.
_REQUEST_DELAY_SECONDS = 0.15

#: Chief-level titles, plus President. Vice presidents are excluded — an SVP is
#: not a chief officer, and treating them alike would flatten the strongest tier
#: of this signal into the noisiest.
_C_SUITE = re.compile(
    r"chief[a-z &.,/()-]*officer|\b(ceo|cfo|cto|coo|cio|cmo|clo|cao|chro|ciso|cro|cco|cdo)\b",
    re.IGNORECASE,
)
_PRESIDENT = re.compile(r"\bpresident\b", re.IGNORECASE)
_VICE_PRESIDENT = re.compile(r"vice[ -]president|\b[esv]vp\b", re.IGNORECASE)


@dataclass(slots=True)
class InsiderFiling:
    """One transaction line parsed from a Form 4."""

    accession_number: str
    line_index: int
    issuer_ticker: str | None
    issuer_name: str | None
    insider_name: str | None
    officer_title: str | None
    is_officer: bool
    is_director: bool
    is_ten_percent_owner: bool
    is_c_suite: bool
    transaction_code: str | None
    acquired_disposed: str | None
    transaction_date: date | None
    filed_at: datetime | None
    shares: Decimal | None
    price_per_share: Decimal | None
    shares_owned_after: Decimal | None
    is_10b5_1_plan: bool
    filing_url: str
    value_usd: Decimal | None = field(default=None)

    def __post_init__(self) -> None:
        if self.shares is not None and self.price_per_share is not None:
            self.value_usd = (self.shares * self.price_per_share).quantize(Decimal("0.01"))


def is_c_suite_title(title: str | None) -> bool:
    """Does this free-text title denote a chief-level officer or President?"""
    if not title:
        return False
    if _C_SUITE.search(title):
        return True
    return bool(_PRESIDENT.search(title) and not _VICE_PRESIDENT.search(title))


def _text(tag: str, blob: str) -> str | None:
    """First value of `tag`, unwrapping Form 4's `<value>` envelope if present."""
    match = re.search(rf"<{tag}>\s*(?:<value>)?\s*([^<]*)", blob)
    if match is None:
        return None
    # EDGAR embeds entities in free-text fields ("EVP &amp; COO"), which would
    # otherwise reach the title matcher and the UI verbatim.
    value = html.unescape(match.group(1)).strip()
    return value or None


def _decimal(tag: str, blob: str) -> Decimal | None:
    raw = _text(tag, blob)
    if raw is None:
        return None
    try:
        return Decimal(raw)
    except (InvalidOperation, ValueError):
        return None


def _bool(tag: str, blob: str) -> bool:
    raw = _text(tag, blob)
    return raw is not None and raw.strip().lower() in {"1", "true"}


def _date(tag: str, blob: str) -> date | None:
    raw = _text(tag, blob)
    if raw is None:
        return None
    try:
        return datetime.strptime(raw[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


class EDGARClient:
    """Reads recent Form 4 filings. Sequential and polite by construction."""

    def __init__(self, *, contact_email: str, timeout_seconds: float = 20.0) -> None:
        if not contact_email:
            raise ValueError(
                "SEC EDGAR requires a contact address in the User-Agent; set EDGAR_CONTACT_EMAIL."
            )
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_seconds),
            headers={"User-Agent": f"Stock-Automate research {contact_email}"},
            follow_redirects=True,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def recent_form4_index_urls(self, limit: int = 100) -> list[str]:
        """Index-page URLs for the most recent Form 4 filings, newest first."""
        response = await self._client.get(
            _BROWSE,
            params={
                "action": "getcurrent",
                "type": "4",
                "owner": "include",
                "count": str(min(limit, 100)),
                "output": "atom",
            },
        )
        response.raise_for_status()
        urls = re.findall(r'<link rel="alternate" type="text/html" href="([^"]+)"', response.text)
        # De-duplicate on the *accession*, not the URL. EDGAR indexes one filing
        # under every CIK it involves — the issuer and each reporting owner — so
        # a single Form 4 appears at several distinct paths that all resolve to
        # the same document. Deduping on the URL leaves those in, and a filing
        # fetched three times is three times the rate-limited requests and a
        # signal counted three times downstream.
        seen: set[str] = set()
        unique: list[str] = []
        for url in urls:
            if "-index.htm" not in url:
                continue
            accession = _accession_from_url(url)
            if accession in seen:
                continue
            seen.add(accession)
            unique.append(url)
        return unique[:limit]

    async def fetch_filing(self, index_url: str) -> list[InsiderFiling]:
        """Parse every transaction line in one Form 4."""
        await asyncio.sleep(_REQUEST_DELAY_SECONDS)
        index = await self._client.get(index_url)
        index.raise_for_status()

        # The rendered document is xslF345…/form4.xml; the raw one is the same
        # name without the stylesheet prefix. Only the raw XML is parseable.
        candidates = [
            u
            for u in re.findall(r'href="(/Archives/[^"]+\.xml)"', index.text)
            if "xslF345" not in u
        ]
        if not candidates:
            log.debug("edgar.no_ownership_xml", index_url=index_url)
            return []

        await asyncio.sleep(_REQUEST_DELAY_SECONDS)
        document = await self._client.get(_BASE + candidates[0])
        document.raise_for_status()
        return self.parse_form4(document.text, filing_url=_BASE + candidates[0])

    @staticmethod
    def parse_form4(document: str, *, filing_url: str) -> list[InsiderFiling]:
        """Extract transaction lines from Form 4 ownership XML.

        Static and pure so it can be tested against a fixture without touching
        the network — the parsing is where the subtle mistakes live, not the
        HTTP.
        """
        accession = _text("accessionNumber", document) or _accession_from_url(filing_url)
        issuer_ticker = _text("issuerTradingSymbol", document)
        issuer_name = _text("issuerName", document)
        insider_name = _text("rptOwnerName", document)
        officer_title = _text("officerTitle", document)
        is_officer = _bool("isOfficer", document)
        is_director = _bool("isDirector", document)
        is_ten_pct = _bool("isTenPercentOwner", document)

        # A 10b5-1 plan is usually declared in a footnote rather than a field.
        plan = bool(
            re.search(r"10b5-1", document)
            or re.search(r"<aff10b5One>\s*(?:<value>)?\s*(1|true)", document, re.IGNORECASE)
        )
        filed_at = _parse_filed_at(document)

        filings: list[InsiderFiling] = []
        blocks = re.findall(
            r"<nonDerivativeTransaction>(.*?)</nonDerivativeTransaction>", document, re.S
        )
        for index, block in enumerate(blocks):
            code = _text("transactionCode", block)
            filings.append(
                InsiderFiling(
                    accession_number=accession,
                    line_index=index,
                    issuer_ticker=issuer_ticker.upper() if issuer_ticker else None,
                    issuer_name=issuer_name,
                    insider_name=insider_name,
                    officer_title=officer_title,
                    is_officer=is_officer,
                    is_director=is_director,
                    is_ten_percent_owner=is_ten_pct,
                    is_c_suite=is_c_suite_title(officer_title),
                    transaction_code=code.upper() if code else None,
                    acquired_disposed=_text("transactionAcquiredDisposedCode", block),
                    transaction_date=_date("transactionDate", block),
                    filed_at=filed_at,
                    shares=_decimal("transactionShares", block),
                    price_per_share=_decimal("transactionPricePerShare", block),
                    shares_owned_after=_decimal("sharesOwnedFollowingTransaction", block),
                    is_10b5_1_plan=plan,
                    filing_url=filing_url,
                )
            )
        return filings


def _accession_from_url(url: str) -> str:
    match = re.search(r"(\d{10}-?\d{2}-?\d{6})", url)
    if match is None:
        return url[-32:]
    raw = match.group(1).replace("-", "")
    return f"{raw[:10]}-{raw[10:12]}-{raw[12:]}"


def _parse_filed_at(document: str) -> datetime | None:
    """Signature date is the closest in-document proxy for publication."""
    raw = _text("periodOfReport", document) or _text("signatureDate", document)
    if raw is None:
        return None
    try:
        return datetime.strptime(raw[:10], "%Y-%m-%d").replace(tzinfo=UTC)
    except ValueError:
        return None
