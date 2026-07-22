"""Probe configured credentials and report whether they actually work.

    pnpm check:keys

Answers the question "is my setup real?" without starting the app or placing an
order. Every check is a GET.

Safety rules this script follows:

  * **It never probes the live endpoint with the live key by default.** A live
    key is a real-money credential; the read is harmless, but silently touching
    a live account because someone ran a diagnostic is not a behaviour worth
    having. Pass --include-live to opt in.
  * **It never prints a secret**, only a length and a last-4 fingerprint.
  * **It cross-probes deliberately.** A key rejected by demo *and* accepted by
    live is the single most common Trading 212 misconfiguration, and it is
    invisible from the demo result alone.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import sys
from dataclasses import dataclass
from enum import StrEnum

import httpx

from app.config import Settings, get_settings


class Status(StrEnum):
    OK = "ok"
    NOT_CONFIGURED = "not configured"
    REJECTED = "rejected"
    UNREACHABLE = "unreachable"
    RATE_LIMITED = "rate limited"
    SKIPPED = "skipped"


_MARK = {
    Status.OK: "PASS",
    Status.NOT_CONFIGURED: "----",
    Status.REJECTED: "FAIL",
    Status.UNREACHABLE: "FAIL",
    Status.RATE_LIMITED: "WARN",
    Status.SKIPPED: "SKIP",
}


@dataclass
class CheckResult:
    name: str
    status: Status
    detail: str
    remedy: str | None = None

    @property
    def is_failure(self) -> bool:
        return self.status in {Status.REJECTED, Status.UNREACHABLE}


def _fingerprint(secret: str | None) -> str:
    """Enough to identify a key, not enough to use one."""
    if not secret:
        return "unset"
    return f"{len(secret)} chars, ends …{secret[-4:]}"


async def _probe_trading212(
    name: str,
    api_key: str | None,
    api_secret: str | None,
    base_url: str,
    *,
    timeout_seconds: float,
) -> CheckResult:
    """GET the account endpoint with Basic auth. Read-only; places nothing."""
    if not api_key and not api_secret:
        return CheckResult(name=name, status=Status.NOT_CONFIGURED, detail="no key or secret set")
    if not api_key or not api_secret:
        # A half-configured credential can never authenticate, and saying which
        # half is missing is more useful than a bare 401 later.
        missing = "secret" if api_key else "key"
        return CheckResult(
            name=name,
            status=Status.REJECTED,
            detail=f"the API {missing} is missing — Basic auth needs both halves",
            remedy=(
                "The secret is shown only once at generation. If it was not saved, "
                "regenerate the key in Trading 212 to get a fresh key and secret."
            ),
        )

    url = f"{base_url.rstrip('/')}/equity/account/info"
    token = base64.b64encode(f"{api_key}:{api_secret}".encode()).decode("ascii")
    try:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            # Basic auth in a header, never a query string.
            response = await client.get(url, headers={"Authorization": f"Basic {token}"})
    except httpx.TimeoutException:
        return CheckResult(
            name=name,
            status=Status.UNREACHABLE,
            detail=f"no response within {timeout_seconds:.0f}s",
            remedy="Check network access to Trading 212.",
        )
    except httpx.HTTPError as exc:
        return CheckResult(
            name=name,
            status=Status.UNREACHABLE,
            detail=f"transport error: {type(exc).__name__}",
            remedy="Check network access to Trading 212.",
        )

    match response.status_code:
        case 200:
            account_id = "unknown"
            currency = "?"
            try:
                body = response.json()
                account_id = str(body.get("id", "unknown"))
                currency = str(body.get("currencyCode", "?"))
            except ValueError:
                pass
            masked = f"****{account_id[-4:]}" if len(account_id) > 4 else "****"
            return CheckResult(
                name=name,
                status=Status.OK,
                detail=f"accepted — account {masked}, currency {currency}",
            )
        case 401 | 403:
            return CheckResult(
                name=name,
                status=Status.REJECTED,
                detail=f"HTTP {response.status_code} — key not accepted",
                remedy=(
                    "Regenerate the key in Trading 212 (Settings -> API (Beta)) on the "
                    "matching account type. A key made on a Live account is rejected by "
                    "the demo endpoint, and vice versa."
                ),
            )
        case 429:
            return CheckResult(
                name=name,
                status=Status.RATE_LIMITED,
                detail="HTTP 429 — rate limited, so the key is probably valid",
                remedy="Wait a minute and re-run.",
            )
        case _:
            return CheckResult(
                name=name,
                status=Status.UNREACHABLE,
                detail=f"unexpected HTTP {response.status_code}",
            )


async def _probe_yfinance() -> CheckResult:
    """yfinance needs no key, but it can still be blocked or throttled."""
    try:
        import yfinance as yf

        def _fetch() -> bool:
            ticker = yf.Ticker("SPY")
            frame = ticker.history(period="1d", interval="1d", raise_errors=False)
            return frame is not None and not frame.empty

        ok = await asyncio.to_thread(_fetch)
    except Exception as exc:
        return CheckResult(
            name="yfinance",
            status=Status.UNREACHABLE,
            detail=f"{type(exc).__name__}: {str(exc)[:60]}",
            remedy="yfinance is unofficial and throttles by IP. Retry later.",
        )

    if not ok:
        return CheckResult(
            name="yfinance",
            status=Status.UNREACHABLE,
            detail="returned no data for SPY",
            remedy="Likely throttling. Retry later.",
        )
    return CheckResult(name="yfinance", status=Status.OK, detail="accepted (no key required)")


async def _probe_twelve_data(
    api_key: str | None, base_url: str, *, timeout_seconds: float
) -> CheckResult:
    if not api_key:
        return CheckResult(
            name="twelve_data", status=Status.NOT_CONFIGURED, detail="no key set (Phase 4)"
        )
    try:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            response = await client.get(
                f"{base_url.rstrip('/')}/quote", params={"symbol": "SPY", "apikey": api_key}
            )
    except httpx.HTTPError as exc:
        return CheckResult(name="twelve_data", status=Status.UNREACHABLE, detail=type(exc).__name__)

    # Twelve Data signals auth failure in a 200 body, not the status code.
    try:
        body = response.json()
    except ValueError:
        return CheckResult(name="twelve_data", status=Status.UNREACHABLE, detail="non-JSON reply")

    if isinstance(body, dict) and body.get("status") == "error":
        return CheckResult(
            name="twelve_data",
            status=Status.REJECTED,
            detail=str(body.get("message", "rejected"))[:80],
            remedy="Check the key at twelvedata.com.",
        )
    return CheckResult(name="twelve_data", status=Status.OK, detail="accepted")


async def _probe_eodhd(
    api_key: str | None, base_url: str, *, timeout_seconds: float
) -> CheckResult:
    if not api_key:
        return CheckResult(
            name="eodhd", status=Status.NOT_CONFIGURED, detail="no key set (Phase 2)"
        )
    try:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            response = await client.get(
                f"{base_url.rstrip('/')}/eod/SPY.US",
                params={"api_token": api_key, "fmt": "json", "period": "d", "order": "d"},
            )
    except httpx.HTTPError as exc:
        return CheckResult(name="eodhd", status=Status.UNREACHABLE, detail=type(exc).__name__)

    if response.status_code in (401, 403):
        return CheckResult(
            name="eodhd",
            status=Status.REJECTED,
            detail=f"HTTP {response.status_code}",
            remedy="Check the key at eodhd.com.",
        )
    if response.status_code != 200:
        return CheckResult(
            name="eodhd", status=Status.UNREACHABLE, detail=f"HTTP {response.status_code}"
        )
    return CheckResult(name="eodhd", status=Status.OK, detail="accepted")


def _unwrap(secret: object) -> str | None:
    value = secret.get_secret_value() if secret is not None else None  # type: ignore[attr-defined]
    return value or None


async def run_checks(settings: Settings, *, include_live: bool) -> list[CheckResult]:
    demo_key = _unwrap(settings.trading212_demo_api_key)
    demo_secret = _unwrap(settings.trading212_demo_api_secret)
    live_key = _unwrap(settings.trading212_live_api_key)
    live_secret = _unwrap(settings.trading212_live_api_secret)
    timeout = settings.trading212_timeout_seconds

    results = [
        await _probe_trading212(
            "trading212 demo -> demo endpoint",
            demo_key,
            demo_secret,
            settings.trading212_demo_base_url,
            timeout_seconds=timeout,
        )
    ]

    # The diagnostic cross-probe. If the demo credential was rejected by demo,
    # the most likely explanation is that it was generated on a Live account —
    # which only a probe against the live endpoint can distinguish from a simply
    # dead credential. Read-only, and it uses the *demo* credential, so it cannot
    # touch a live account it has no rights to. Skipped when only half the
    # credential is present, since that failure is already fully explained.
    if demo_key and demo_secret and results[0].status is Status.REJECTED:
        cross = await _probe_trading212(
            "  ↳ same demo credential -> live endpoint (diagnostic)",
            demo_key,
            demo_secret,
            settings.trading212_live_base_url,
            timeout_seconds=timeout,
        )
        if cross.status is Status.OK:
            cross.detail = "ACCEPTED by live — this is a LIVE credential in the DEMO slot"
            cross.remedy = (
                "Move it to the TRADING212_LIVE_* variables and generate a separate "
                "credential on a Practice account for the demo slot."
            )
        else:
            cross.detail = "also rejected by live — the credential is dead, not misplaced"
            cross.remedy = "Regenerate it in Trading 212 (you will get a new key and secret)."
        results.append(cross)

    if include_live:
        results.append(
            await _probe_trading212(
                "trading212 live -> live endpoint",
                live_key,
                live_secret,
                settings.trading212_live_base_url,
                timeout_seconds=timeout,
            )
        )
    elif live_key or live_secret:
        results.append(
            CheckResult(
                name="trading212 live -> live endpoint",
                status=Status.SKIPPED,
                detail="a live credential is configured but was not probed",
                remedy="Pass --include-live to check it (read-only).",
            )
        )

    results.append(await _probe_yfinance())
    results.append(
        await _probe_twelve_data(
            settings.twelve_data_api_key.get_secret_value()
            if settings.twelve_data_api_key
            else None,
            settings.twelve_data_base_url,
            timeout_seconds=10.0,
        )
    )
    results.append(
        await _probe_eodhd(
            settings.eodhd_api_key.get_secret_value() if settings.eodhd_api_key else None,
            settings.eodhd_base_url,
            timeout_seconds=10.0,
        )
    )
    return results


def _render(results: list[CheckResult], settings: Settings) -> None:
    print()
    print("Credential check")
    print("=" * 72)

    print(f"  demo key    : {_fingerprint(_unwrap(settings.trading212_demo_api_key))}")
    print(f"  demo secret : {_fingerprint(_unwrap(settings.trading212_demo_api_secret))}")
    print(f"  live key    : {_fingerprint(_unwrap(settings.trading212_live_api_key))}")
    print(f"  live secret : {_fingerprint(_unwrap(settings.trading212_live_api_secret))}")
    print(f"  LIVE_TRADING_ENABLED = {settings.live_trading_enabled}")
    print("=" * 72)
    print()

    for result in results:
        print(f"  [{_MARK[result.status]}] {result.name}")
        print(f"         {result.detail}")
        if result.remedy:
            print(f"         -> {result.remedy}")
        print()

    failures = [r for r in results if r.is_failure]
    print("=" * 72)
    if failures:
        print(f"  {len(failures)} check(s) failed.")
    else:
        print("  No failures. Unconfigured providers are expected for later phases.")
    print()

    if _unwrap(settings.trading212_live_api_key) and not settings.live_trading_enabled:
        print(
            "  Note: a live credential is present but LIVE_TRADING_ENABLED=false, so it\n"
            "  is inert until live trading is enabled and armed (Phase 5).\n"
        )


async def main_async(include_live: bool) -> int:
    settings = get_settings()
    results = await run_checks(settings, include_live=include_live)
    _render(results, settings)
    # Non-zero on failure so this is usable as a gate in a script or CI step.
    return 1 if any(r.is_failure for r in results) else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe configured credentials.")
    parser.add_argument(
        "--include-live",
        action="store_true",
        help=(
            "Also probe the live key against the live endpoint. Read-only, but it "
            "touches a real-money account, so it is opt-in."
        ),
    )
    args = parser.parse_args()
    return asyncio.run(main_async(include_live=args.include_live))


if __name__ == "__main__":
    sys.exit(main())
