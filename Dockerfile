# ---------------------------------------------------------------------------
# Market Regime Detection Dashboard — Docker image
# Multi-arch base: works natively on Apple Silicon (arm64) and on linux/amd64.
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS base

# Faster, quieter Python; no .pyc clutter; stdout/stderr unbuffered for `docker logs`.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHERUSAGESTATS=false

# Build-time deps for hmmlearn / scikit-learn / numpy native extensions.
# We keep them in the runtime image because they're tiny and avoid surprises
# if pip ever decides to rebuild from source on a slim base.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (cached layer — only rebuilds when requirements.txt changes).
COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Then copy the application code.
COPY . .

# Pre-create directories that named volumes will mount onto. If we don't,
# Docker creates them owned by root, and the non-root streamlit user can't
# write to them (named volumes inherit ownership from the image's directory
# when first created).
RUN mkdir -p /app/cache_5m /app/predictions /app/twitter_sentiment /app/twitter_sentiment/snapshots /app/claude_calls /app/tv_ideas /app/tv_ideas/snapshots /app/influencer_track_record /app/tv_bridge_queue

# Create a non-root user and give it ownership of the working directory.
RUN useradd --create-home --shell /bin/bash --uid 1000 streamlit \
    && chown -R streamlit:streamlit /app
USER streamlit

EXPOSE 8501

# Streamlit's own readiness endpoint — used by docker / compose health checks.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl --fail --silent http://localhost:8501/_stcore/health || exit 1

# Bind to 0.0.0.0 so the port is reachable from outside the container.
CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
