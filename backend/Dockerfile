# ---------------------------------------------------------------
# Backend Dockerfile — FastAPI application
# ---------------------------------------------------------------
FROM python:3.15-slim

WORKDIR /app

# System dependencies (kept minimal). gcc/libffi are not needed for our
# pure-Python wheel dependencies; we install only what is required.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (better layer caching).
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code.
COPY backend ./backend
COPY data ./data
COPY .env.example ./

# Non-root user for security.
RUN useradd -m -u 1000 appuser

# Create runtime directories with proper permissions for non-root user.
RUN mkdir -p /data /app/backend/index /app/data/uploads \
    && chown -R appuser:appuser /data /app/backend/index /app/data/uploads \
    && chown -R appuser:appuser /app

USER appuser

# Inter-container networking: backend reaches Ollama at http://ollama:11434.
# Default OLLAMA_BASE_URL is overridden by docker-compose.
ENV OLLAMA_BASE_URL=http://ollama:11434 \
    DATABASE_PATH=/data/feedback.db \
    LOG_LEVEL=INFO

EXPOSE 8000

# Health check: the container is ready when /health responds.
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5)" || exit 1

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
