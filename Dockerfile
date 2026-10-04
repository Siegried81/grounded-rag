# Container image for the grounded RAG assistant (Streamlit UI + CLI).
#
# Why it is built this way:
# - python:3.12-slim: small Debian base with wheels available for every
#   dependency in requirements.txt (numpy, pypdf...), so no compiler is needed.
# - requirements.txt is copied and installed BEFORE the source code, so editing
#   code only rebuilds the last, cheap layer instead of reinstalling packages.
# - The app runs as an unprivileged user: a compromised dependency or a prompt
#   gone wrong cannot write outside the folders that user owns.
# - Secrets (.env), the local venv, built indexes, caches and logs are excluded
#   by .dockerignore; indexes and logs live in volumes (see docker-compose.yml),
#   so the image stays reproducible and never contains an API key.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_HEADLESS=true

WORKDIR /app

# Dependency layer: cached until requirements.txt changes.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Non-root user. index/ and logs/ are created and owned by it so the named
# volumes mounted there inherit that ownership on first use.
RUN useradd --create-home --uid 10001 app \
    && mkdir -p /app/index /app/logs /app/data \
    && chown -R app:app /app/index /app/logs

# Source layer: changes often, so it comes last.
COPY --chown=app:app . .

USER app

EXPOSE 8501

# Streamlit exposes a lightweight liveness endpoint; python is used instead of
# curl because the slim image does not ship curl.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4).status == 200 else 1)"]

# Exec form: Streamlit is PID 1 and receives SIGTERM directly on `docker stop`.
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
