FROM python:3.12-slim

WORKDIR /app

# Install dependencies first to leverage Docker cache.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code, migration scripts, and frontend demo.
COPY app/ ./app/
COPY static/ ./static/
COPY alembic/ ./alembic/
COPY alembic.ini .

# Ensure data directory exists for SQLite persistence.
RUN mkdir -p /app/data

# Default environment (overridden by docker-compose or -e flags).
ENV SUMMARIZER_MODE=mock \
    DATABASE_URL=sqlite+aiosqlite:///data/context.db \
    REDIS_URL=redis://redis:6379/0

EXPOSE 8000

# Run migrations (if applicable) then start the service.
CMD ["sh", "-c", "alembic upgrade head 2>/dev/null; uvicorn app.main:app --host 0.0.0.0 --port 8000"]
