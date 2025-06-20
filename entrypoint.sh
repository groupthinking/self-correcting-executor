#!/usr/bin/env bash
set -e

# Determine startup mode
MODE=${1:-"standard"}

echo "🚀 Starting MCP Self-Correcting Executor - Mode: $MODE"

# Verify D-Wave Ocean SDK if in quantum mode
if [[ "$MODE" == "quantum-dev" || "$MODE" == "quantum" ]]; then
    echo "⚛️  Verifying D-Wave Ocean SDK..."
    python -c "import dwave.ocean; print('✅ D-Wave Ocean SDK ready')" || {
        echo "❌ D-Wave Ocean SDK not available"
        exit 1
    }
    
    # Check D-Wave configuration
    if [[ -f "/app/.dwave/dwave.conf" ]]; then
        echo "✅ D-Wave configuration found"
    else
        echo "⚠️  D-Wave configuration not found - run 'make setup-dwave' first"
    fi
fi

# Wait for database to be ready if env variables provided
if [[ -n "$POSTGRES_USER" ]]; then
    echo "📊 Waiting for database..."
    for i in {1..30}; do
        if python3 -c "import psycopg2; psycopg2.connect('postgresql://$POSTGRES_USER:$POSTGRES_PASSWORD@${POSTGRES_HOST:-postgres}:${POSTGRES_PORT:-5432}/$POSTGRES_DB')" 2>/dev/null; then
            echo "✅ Database is ready!"
            break
        fi
        echo "⏳ Database not ready yet, waiting... ($i/30)"
        sleep 2
    done
fi

# Wait for Redis if available
if [[ -n "$REDIS_HOST" ]]; then
    echo "🔴 Waiting for Redis..."
    for i in {1..15}; do
        if python3 -c "import redis; r=redis.Redis(host='${REDIS_HOST:-redis}', port=${REDIS_PORT:-6379}); r.ping()" 2>/dev/null; then
            echo "✅ Redis is ready!"
            break
        fi
        echo "⏳ Redis not ready yet, waiting... ($i/15)"
        sleep 1
    done
fi

# Start based on mode
case "$MODE" in
    "quantum-dev"|"quantum")
        echo "⚛️  Starting Quantum Development Server..."
        echo "🌐 Backend: http://localhost:8000"
        echo "🔧 MCP Server: http://localhost:8001"
        exec python main.py --quantum-enabled --development
        ;;
    "mcp-server")
        echo "🔧 Starting MCP Server..."
        exec python mcp_server/main.py
        ;;
    "production")
        echo "🚀 Starting Production Server..."
        exec uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
        ;;
    "test")
        echo "🧪 Running Tests..."
        exec python -m pytest tests/ -v
        ;;
    *)
        echo "📡 Starting Standard Development Server..."
        echo "🌐 Backend: http://localhost:8000"
        exec python main.py --development
        ;;
esac 