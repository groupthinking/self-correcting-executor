#!/usr/bin/env bash
set -e

# Wait for database to be ready if env variables provided
if [[ -n "$POSTGRES_USER" ]]; then
  until pg_isready -h "${POSTGRES_HOST:-db}" -p "${POSTGRES_PORT:-5432}" -U "$POSTGRES_USER"; do
    echo "Waiting for Postgres..."
    sleep 2
  done
fi

# Start FastAPI server
exec uvicorn mcp.main:app --host 0.0.0.0 --port 8080 --reload 