#!/usr/bin/env bash
set -e

echo "Starting MCP server..."

# Wait for database to be ready if env variables provided
if [[ -n "$POSTGRES_USER" ]]; then
  echo "Waiting for database..."
  for i in {1..30}; do
    if python3 -c "import psycopg2; psycopg2.connect('postgresql://$POSTGRES_USER:$POSTGRES_PASSWORD@${POSTGRES_HOST:-db}:${POSTGRES_PORT:-5432}/$POSTGRES_DB')" 2>/dev/null; then
      echo "Database is ready!"
      break
    fi
    echo "Database not ready yet, waiting... ($i/30)"
    sleep 2
  done
fi

# Start FastAPI server
echo "Starting FastAPI server on port 8080..."
exec uvicorn mcp.main:app --host 0.0.0.0 --port 8080 --reload 