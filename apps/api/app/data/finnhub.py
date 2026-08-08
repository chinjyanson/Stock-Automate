"""Finnhub news reader, for the sentiment signal (§4, §6).

Deliberately *not* wired into `resolve_provider`. That factory's contract is
`MarketDataProvider` — candles, quotes, fundamentals — and Finnhub is used here
for none of those. Forcing it in would mean implementing three abstract methods
that raise, and the next person reading the chain would reasonably believe
Finnhub could serve candles. `EDGARClient` is the precedent: a source that
answers one narrow question lives beside the factory, not inside it.

Two endpoints, and they are not equally available:

  * ``/company-news`` — headlines. Free tier, and the input the Loughran-McDonald
    scorer actually reads. Coverage is strong for US listings and thin
    elsewhere, which is the usual shape of a free feed.
  * ``/news-sentiment`` — Finnhub's own proprietary score. **Premium on current
    plans**, and a free key gets 403. It is requested best-effort and its
    absence is recorded as absence, never as neutrality. Callers must degrade to
    the lexicon reading rather than treating a missing score as "no bad news" —
    see `SentimentService.risk_reading`, where that distinction is what stops a
    403 from silently disarming a risk gate.

Rate limits are the caller's business, not this client's: the job that drives it
holds a `ProviderBudget` and spends a credit per call. Putting the limiter here
would hide the spend from the ledger that exists to make it visible.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime

import httpx
import structlog

log = structlog.get_logger(__name__)

DEFAULT_BASE_URL = "https://finnhub.io/api/v1"

#: Cap on headlines read per symbol per pass. Finnhub returns a whole window of
#: articles and the tail is syndication noise — the same wire story rewritten by
#: six outlets, which would count its tone six times.
MAX_HEADLINES = 40


@dataclass(frozen=True, slots=True)
class NewsItem:
    headline: str
    summary: str
    source: str | None
    published_at: datetime | None
    url: str | None

    @property
    def text(self) -> str:
        """Headline plus summary — the text the scorer reads.

        Both, because a headline alone is often only a few tone words and the
        summary is where the qualification lives ("beats estimates, cuts
        guidance"). Summaries are frequently empty, in which case this is just
        the headline.
        """
        return f"{self.headline} {self.summary}".strip()


@dataclass(frozen=True, slots=True)
class ProviderSentiment:
    """Finnhub's own scoring of a company's news. Premium-gated."""

    #: Their composite, roughly -1..+1 in practice though undocumented as bounded.
    company_news_score: float | None
    #: Share of articles they read as bullish, 0..1.
    bullish_percent: float | None
    #: Article count over their window, relative to the company's own average.
    buzz: float | None


class FinnhubPremiumRequiredError(Exception):
    """The endpoint exists but this key's plan does not include it."""


class FinnhubClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout_seconds: float = 20.0,
    ) -> None:
        if not api_key:
            raise ValueError("Finnhub requires an API key; set FINNHUB_API_KEY.")
        self._key = api_key
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout_seconds),
            follow_redirects=True,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> FinnhubClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def company_news(self, symbol: str, *, since: date, until: date) -> list[NewsItem]:
        """Headlines for one symbol over a date window, newest first.

        An empty list means "nothing published", which for a quiet small-cap is
        the ordinary case and not an error.
        """
        response = await self._client.get(
            "/company-news",
            params={
                "symbol": symbol,
                "from": since.isoformat(),
                "to": until.isoformat(),
                "token": self._key,
            },
        )
        if response.status_code in (401, 403):
            raise FinnhubPremiumRequiredError(
                f"Finnhub refused /company-news for {symbol} (HTTP {response.status_code}); "
                "the key is missing, invalid, or not entitled to this endpoint."
            )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            return []

        items: list[NewsItem] = []
        for row in payload[:MAX_HEADLINES]:
            if not isinstance(row, dict):
                continue
            headline = str(row.get("headline") or "").strip()
            if not headline:
                continue
            items.append(
                NewsItem(
                    headline=headline,
                    summary=str(row.get("summary") or "").strip(),
                    source=str(row.get("source")) if row.get("source") else None,
                    published_at=_epoch(row.get("datetime")),
                    url=str(row.get("url")) if row.get("url") else None,
                )
            )
        return items

    async def news_sentiment(self, symbol: str) -> ProviderSentiment | None:
        """Finnhub's own sentiment score, or None when the plan excludes it.

        None rather than raising: this is the optional half. The lexicon reading
        stands on its own, and a premium wall must not fail an ingest that
        otherwise succeeded.
        """
        try:
            response = await self._client.get(
                "/news-sentiment", params={"symbol": symbol, "token": self._key}
            )
        except httpx.HTTPError as exc:
            log.warning("finnhub.sentiment_unreachable", symbol=symbol, error=str(exc))
            return None
        if response.status_code in (401, 403):
            log.info("finnhub.sentiment_premium_gated", symbol=symbol)
            return None
        if response.status_code >= 400:
            log.warning("finnhub.sentiment_failed", symbol=symbol, status=response.status_code)
            return None

        payload = response.json()
        if not isinstance(payload, dict):
            return None
        sentiment = payload.get("sentiment")
        buzz = payload.get("buzz")
        return ProviderSentiment(
            company_news_score=_float(payload.get("companyNewsScore")),
            bullish_percent=_float(sentiment.get("bullishPercent"))
            if isinstance(sentiment, dict)
            else None,
            buzz=_float(buzz.get("buzz")) if isinstance(buzz, dict) else None,
        )

    async def health_check(self) -> bool:
        try:
            today = datetime.now(UTC).date()
            await self.company_news("AAPL", since=today, until=today)
        except Exception:
            return False
        return True


def _epoch(value: object) -> datetime | None:
    try:
        return datetime.fromtimestamp(float(value), tz=UTC)  # type: ignore[arg-type]
    except (TypeError, ValueError, OSError, OverflowError):
        return None


def _float(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
