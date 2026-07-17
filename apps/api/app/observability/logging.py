"""Structured JSON logging (§18).

Two non-obvious requirements shape this module:

  * §17 forbids logging full authentication headers or secret values. A
    redacting processor runs on every event, so a careless `log.info(...,
    api_key=key)` is scrubbed rather than persisted. Discipline at call sites is
    necessary but not sufficient.

  * §18 requires request/trade-intent/strategy-decision correlation. Those are
    bound to a contextvar, so they attach to every log line in a request
    without being threaded through call signatures.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog
from structlog.contextvars import bind_contextvars, clear_contextvars, merge_contextvars
from structlog.types import EventDict, Processor

#: Substrings that mark a key as secret-bearing. Substring rather than exact
#: match so `trading212_live_api_key` and `x_api_key` are both caught.
_SECRET_KEY_MARKERS = (
    "password",
    "secret",
    "token",
    "api_key",
    "apikey",
    "authorization",
    "auth_header",
    "private_key",
    "credential",
    "cookie",
)

_REDACTED = "[redacted]"


def _redact_secrets(_logger: Any, _method: str, event_dict: EventDict) -> EventDict:
    """Scrub secret-looking keys from every log event.

    Deliberately blunt. A false positive costs one unhelpful log line; a false
    negative writes a live trading key to disk.
    """
    for key in list(event_dict.keys()):
        lowered = key.lower()
        if any(marker in lowered for marker in _SECRET_KEY_MARKERS):
            # Keep the fact that the field was present — its absence would be
            # its own kind of misleading.
            event_dict[key] = _REDACTED
    return event_dict


def _add_service_context(_logger: Any, _method: str, event_dict: EventDict) -> EventDict:
    event_dict.setdefault("service", "trading-platform-api")
    return event_dict


def configure_logging(*, log_level: str = "INFO", json_output: bool = True) -> None:
    """Install the logging configuration. Idempotent.

    JSON in deployment (machine-parseable, §18); a console renderer in
    development, where a human is reading.
    """
    processors: list[Processor] = [
        merge_contextvars,
        structlog.stdlib.add_log_level,
        # `add_logger_name` is deliberately absent: it reads `logger.name`,
        # which PrintLogger does not have. Event names are namespaced instead
        # ("sync.completed", "yfinance.retry"), which carries the same
        # information without coupling to a stdlib logger.
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
        _add_service_context,
        # Redaction runs last, so nothing a prior processor added can slip past.
        _redact_secrets,
    ]

    if json_output:
        processors.extend(
            [
                structlog.processors.format_exc_info,
                structlog.processors.JSONRenderer(),
            ]
        )
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelNamesMapping()[log_level.upper()]
        ),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )

    # Route stdlib logging (uvicorn, sqlalchemy) through the same handler so
    # output is uniformly parseable.
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.getLevelNamesMapping()[log_level.upper()],
        force=True,
    )
    # SQLAlchemy's engine logger is noisy at INFO and duplicates our own.
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)


def bind_request_context(
    *,
    request_id: str | None = None,
    trade_intent_id: str | None = None,
    strategy_decision_id: str | None = None,
    user_id: str | None = None,
) -> None:
    """Attach correlation ids to every subsequent log line in this context."""
    bindings: dict[str, str] = {}
    if request_id:
        bindings["request_id"] = request_id
    if trade_intent_id:
        bindings["trade_intent_id"] = trade_intent_id
    if strategy_decision_id:
        bindings["strategy_decision_id"] = strategy_decision_id
    if user_id:
        bindings["user_id"] = user_id
    if bindings:
        bind_contextvars(**bindings)


def clear_request_context() -> None:
    clear_contextvars()
