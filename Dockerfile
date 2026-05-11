# Multi-stage build using uv for fast, reproducible installs.
# Stage 1 builds the venv; stage 2 carries only what's needed at runtime.

FROM python:3.12-slim-bookworm AS builder

# uv handles the install — faster than pip and respects uv.lock.
COPY --from=ghcr.io/astral-sh/uv:0.4.30 /uv /usr/local/bin/uv

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /app

# Copy lockfile + project metadata first so the dep layer caches across
# source-only changes. Installing without --frozen would re-resolve.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# Now copy the source and install the project itself (link, not full re-resolve).
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev


# ---------------------------------------------------------------- runtime
FROM python:3.12-slim-bookworm AS runtime

# Minimal runtime deps:
#   * tzdata so APScheduler crontabs in non-UTC zones work.
#   * ca-certificates for outbound HTTPS (yfinance, Telegram, etc.).
# Keep this list short — every package is attack surface.
RUN apt-get update && apt-get install -y --no-install-recommends \
        tzdata \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Non-root user. UID/GID 10001 is high enough to never collide with host users
# when volumes are bind-mounted from a regular user account.
RUN groupadd --system --gid 10001 trader \
    && useradd  --system --uid 10001 --gid trader --create-home --shell /usr/sbin/nologin trader

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TRADING_ENV=research

# Carry the resolved venv across.
COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=trader:trader src ./src
COPY --chown=trader:trader config ./config
COPY --chown=trader:trader docker ./docker
COPY --chown=trader:trader pyproject.toml ./

# Volumes are mounted by docker-compose. We pre-create them so an unmounted
# run doesn't fail-open in confusing ways.
RUN mkdir -p /app/data /app/logs /app/state \
    && chown -R trader:trader /app/data /app/logs /app/state

USER trader

# The container is healthy iff the heartbeat file is fresh (<5 min). The
# runner writes it once per cycle; if cycles stop, this fails. The check
# itself never raises — exit 0/1 only.
HEALTHCHECK --interval=60s --timeout=10s --start-period=120s --retries=3 \
    CMD ["python", "/app/docker/healthcheck.py", "/app/state/heartbeat.json", "300"]

# Default to status — operators choose the actual subcommand at `docker run`
# or in docker-compose. Refusing to assume "live" is intentional.
ENTRYPOINT ["trading"]
CMD ["status"]
