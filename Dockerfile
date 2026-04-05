# ── Python dependency stage ──────────────────────────────────────
FROM python:3.13-slim AS python-deps

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && rm -rf /var/lib/apt/lists/*

COPY ml/requirements.txt ml/requirements.txt
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r ml/requirements.txt

# ── Node build stage ────────────────────────────────────────────
FROM node:20-slim AS node-build

# better-sqlite3 needs build tools for native compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 make g++ && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm ci --production=false
COPY . .
RUN npm run build

# ── Runtime stage ───────────────────────────────────────────────
FROM python:3.13-slim

# Install Node.js 20 + curl (for health check in start-prod.sh)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl gnupg && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python packages from build stage
COPY --from=python-deps /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=python-deps /usr/local/bin /usr/local/bin

# Node production dependencies + built frontend
COPY --from=node-build /app/node_modules ./node_modules
COPY --from=node-build /app/dist ./dist

# Application code
COPY . .

# Ensure data directory exists (Railway Volume mounts here)
RUN mkdir -p /data/models

# Production startup
RUN chmod +x start-prod.sh
CMD ["./start-prod.sh"]
