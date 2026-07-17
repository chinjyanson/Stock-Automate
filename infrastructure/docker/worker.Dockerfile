# syntax=docker/dockerfile:1

FROM ghcr.io/astral-sh/uv:0.11-python3.13-bookworm-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /app/apps/worker

COPY apps/worker/pyproject.toml apps/worker/uv.lock* ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev 2>/dev/null \
    || uv sync --no-install-project --no-dev

# The worker imports the API's domain layer — brokers, providers, services — so
# that a scheduled job and an endpoint run the *same* code. Duplicating the
# logic is how a job and an endpoint quietly diverge.
COPY apps/api/ /app/apps/api/
COPY apps/worker/ ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev 2>/dev/null || uv sync --no-dev

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app/apps/api:/app/apps/worker"

RUN groupadd --system --gid 1001 appuser \
    && useradd --system --uid 1001 --gid appuser appuser \
    && chown -R appuser:appuser /app /opt/venv
USER appuser

# Beat is colocated with the worker for local development only. In deployment
# beat must run as exactly one process — two schedulers means every job fires
# twice, and §16 requires duplicate execution be impossible where it could
# place duplicate orders.
CMD ["celery", "-A", "worker.app", "worker", "--beat", "--loglevel=info", "--concurrency=2"]
