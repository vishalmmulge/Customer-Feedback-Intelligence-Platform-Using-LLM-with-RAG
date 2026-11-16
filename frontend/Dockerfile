# ---------------------------------------------------------------
# Frontend Dockerfile — Streamlit dashboard
# ---------------------------------------------------------------
FROM python:3.15-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY frontend ./frontend
COPY .env.example ./

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Inside Docker, the backend is reachable at http://backend:8000 (Compose
# service name), never localhost.
ENV BACKEND_URL=http://backend:8000 \
    LOG_LEVEL=INFO

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=5)" || exit 1

CMD ["streamlit", "run", "frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
