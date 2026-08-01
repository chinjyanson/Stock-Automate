# syntax=docker/dockerfile:1
#
# The Next.js frontend.
#
# One deliberate asymmetry with the API and worker images: this one is built
# *per environment*, because `NEXT_PUBLIC_API_URL` is compiled into the browser
# bundle rather than read at runtime. That is a Next.js constraint, not a choice
# — anything the browser reads has to exist at build time.
#
# So dev and prod get separate web images from the same commit, while the API
# and worker images are built once and promoted. That split is acceptable
# precisely because this container holds no trading logic: it renders what the
# API tells it. The code whose behaviour must be identical between environments
# is the code that talks to a broker, and that is promoted as one artifact.

FROM node:22-bookworm-slim AS base
ENV PNPM_HOME=/pnpm \
    PATH="/pnpm:$PATH" \
    NEXT_TELEMETRY_DISABLED=1
RUN corepack enable

# -- Dependencies ------------------------------------------------------------
# Only the manifests, so this layer survives every source-only change.
FROM base AS deps
WORKDIR /app
COPY package.json pnpm-lock.yaml pnpm-workspace.yaml ./
COPY apps/web/package.json apps/web/
RUN --mount=type=cache,id=pnpm,target=/pnpm/store \
    pnpm install --frozen-lockfile --filter web...

# -- Build -------------------------------------------------------------------
FROM base AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY --from=deps /app/apps/web/node_modules ./apps/web/node_modules
COPY package.json pnpm-lock.yaml pnpm-workspace.yaml ./
COPY apps/web ./apps/web

# The API origin the *browser* will call. Must be reachable from the user's
# machine, not from inside the cluster — so never a service-discovery name.
ARG NEXT_PUBLIC_API_URL=http://localhost:8000
ENV NEXT_PUBLIC_API_URL=${NEXT_PUBLIC_API_URL}
# Fail loudly rather than shipping a bundle that silently points at localhost:
# a frontend that cannot reach its API looks identical to one that is down, and
# the mistake would only surface in a browser, after deployment.
RUN test "${NEXT_PUBLIC_API_URL}" != "http://localhost:8000" \
    || echo "WARNING: building with the default localhost API URL" >&2
RUN --mount=type=cache,id=pnpm,target=/pnpm/store \
    cd apps/web && pnpm exec next build

# -- Runtime -----------------------------------------------------------------
# Deliberately NOT `FROM base`: the runtime runs `node server.js` and nothing
# else, so it has no use for pnpm or corepack. Leaving a package manager in a
# production image is free weight and one more thing that can execute.
FROM node:22-bookworm-slim AS runner
WORKDIR /app
ENV NEXT_TELEMETRY_DISABLED=1
ENV NODE_ENV=production \
    PORT=3000 \
    HOSTNAME=0.0.0.0

# Non-root, and numeric so a Kubernetes/ECS runAsNonRoot check can verify it
# without resolving a name inside the image.
RUN groupadd --system --gid 1001 nodejs \
    && useradd --system --uid 1001 --gid nodejs nextjs

# `standalone` already contains the traced node_modules and a server.js; the
# static assets and public/ are the two things it deliberately leaves out.
COPY --from=builder --chown=nextjs:nodejs /app/apps/web/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/apps/web/.next/static ./apps/web/.next/static
COPY --from=builder --chown=nextjs:nodejs /app/apps/web/public ./apps/web/public

USER nextjs
EXPOSE 3000

# No shell: signals reach the server directly, so a rolling deploy terminates
# cleanly instead of waiting out a kill timeout.
CMD ["node", "apps/web/server.js"]
