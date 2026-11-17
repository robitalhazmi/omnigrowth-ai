FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    MALLOC_TRIM_THRESHOLD_=65536 \
    DUCKDB_MEMORY_LIMIT=192MB \
    DUCKDB_THREADS=1

WORKDIR /app

COPY requirements-serve.txt ./
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY src/ ./src/
COPY app/ ./app/
COPY data/processed/ ./data/processed/
COPY data/fallback/ ./data/fallback/

RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)"]

CMD ["streamlit", "run", "app/main.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.fileWatcherType=none", \
     "--server.maxUploadSize=5", \
     "--browser.gatherUsageStats=false"]
