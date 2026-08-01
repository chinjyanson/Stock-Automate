import path from "node:path";
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  // Emit a self-contained server bundle with only the traced dependencies, so
  // the runtime image carries neither the pnpm store nor the build toolchain.
  // Without this the deployed image is roughly an order of magnitude larger and
  // ships a compiler to a machine that is one hop from a broker credential.
  output: "standalone",
  // The workspace root, not apps/web — dependency tracing has to see the pnpm
  // symlink farm above this package or the standalone bundle omits half of it.
  outputFileTracingRoot: path.join(__dirname, "../.."),
  // `next build` and `next dev` write incompatible manifests into the same
  // directory, so building while a dev server is up corrupts it — the dev
  // server then fails on modules missing from the production client manifest.
  // Setting NEXT_DIST_DIR sends a verification build somewhere else instead.
  ...(process.env.NEXT_DIST_DIR ? { distDir: process.env.NEXT_DIST_DIR } : {}),
  // The API is a separate service (FastAPI); Next never talks to the database
  // and never holds a broker credential (§17).
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000",
  },
  async headers() {
    return [
      {
        source: "/:path*",
        headers: [
          { key: "X-Content-Type-Options", value: "nosniff" },
          { key: "X-Frame-Options", value: "DENY" },
          { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
          // This app can arm real-money trading; it must never be framed.
          { key: "Permissions-Policy", value: "geolocation=(), microphone=(), camera=()" },
        ],
      },
    ];
  },
};

export default nextConfig;
