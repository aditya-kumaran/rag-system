FROM python:3.11-slim AS base

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir poetry

COPY pyproject.toml ./
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root 2>/dev/null || true

COPY . .
RUN poetry install --no-interaction --no-ansi

EXPOSE 8000 7860

ENV PYTHONPATH=/app
ENV DATA_DIR=/app/data

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "rag_system.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
