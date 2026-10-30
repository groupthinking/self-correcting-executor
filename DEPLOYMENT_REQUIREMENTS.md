# Deployment Requirements - Specific Setup Guide

## 1. Required Environment Variables

Create a `.env` file with these specific variables:

```bash
# MCP Server Configuration
MCP_SERVER_URL=http://localhost:8090  # or stdio:// for local MCP
MCP_SERVER_HOST=0.0.0.0
MCP_SERVER_PORT=8090
MCP_TRANSPORT=stdio  # Use stdio for Cursor integration

# D-Wave Quantum Computing (FREE OPTIONS AVAILABLE)
DWAVE_API_TOKEN=DEV-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx  # Get free from D-Wave Leap
DWAVE_API_URL=https://cloud.dwavesys.com/sapi/v2
DWAVE_SOLVER_NAME=  # Leave empty to auto-select
QUANTUM_NUM_READS=100  # Reduce for free tier
REQUIRE_QPU=false  # Set to false for development/free tier

# Database (LOCAL DEVELOPMENT OPTIONS)
DATABASE_URL=postgresql://postgres:password@localhost:5432/mcp_db
# OR use SQLite for development:
DATABASE_URL=sqlite:///./mcp_local.db

# Redis (OPTIONAL for development)
REDIS_URL=redis://localhost:6379
# OR skip Redis:
REDIS_URL=memory://  # Use in-memory cache

# Security Keys (Generate these!)
JWT_SECRET_KEY=your-generated-jwt-secret-min-32-chars
API_SECRET_KEY=your-generated-api-secret-min-32-chars
ENCRYPTION_KEY=your-generated-256-bit-encryption-key

# Development Settings
ENVIRONMENT=development
AUTH_ENABLED=false  # Disable for local development
ENABLE_SANDBOXING=false  # Disable for local development
DATA_DIR=./data  # Local data directory
```

## 2. MCP Server to Deploy

The MCP server we created (`mcp_server/real_mcp_server.py`) needs to be run as a **Cursor MCP Server**. Here's how:

### Option A: Run as Cursor MCP Server (Recommended)
1. Update your Cursor settings.json:
```json
{
  "mcpServers": {
    "real-mcp-server": {
      "command": "python3",
      "args": ["/workspace/mcp_server/real_mcp_server.py"],
      "env": {
        "PYTHONPATH": "/workspace"
      }
    }
  }
}
```

### Option B: Run as standalone HTTP server
```bash
# Modify real_mcp_server.py to use HTTP transport instead of stdio
python3 mcp_server/real_mcp_server.py
```

## 3. PostgreSQL/Redis Setup Options

### RECOMMENDED: Use Local/Lightweight Options First

#### Option 1: SQLite (Easiest for Development)
```bash
# No installation needed! Just change DATABASE_URL to:
DATABASE_URL=sqlite:///./mcp_local.db
```

#### Option 2: PostgreSQL via Docker
```bash
# Run PostgreSQL in Docker (no installation needed)
docker run -d \
  --name mcp-postgres \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=mcp_db \
  -p 5432:5432 \
  postgres:15-alpine
```

#### Option 3: MCP Database Server
There's an MCP PostgreSQL server you can use:
```bash
# Install MCP PostgreSQL server
npm install -g @modelcontextprotocol/server-postgres

# Add to Cursor settings.json:
{
  "mcpServers": {
    "postgres": {
      "command": "mcp-server-postgres",
      "args": ["postgresql://localhost/mcp_db"]
    }
  }
}
```

#### For Redis:
- **Development**: Skip it! Use `REDIS_URL=memory://`
- **Production**: Use Docker: `docker run -d -p 6379:6379 redis:alpine`

## 4. D-Wave Quantum API - FREE Access

### ✅ FREE D-Wave Access Available!

1. **Sign up for D-Wave Leap (FREE)**:
   - Go to: https://cloud.dwavesys.com/leap/signup/
   - Get **1 minute of FREE QPU time per month**
   - Perfect for development and testing

2. **Get your API Token**:
   ```bash
   # After signing up, get token from:
   # https://cloud.dwavesys.com/leap/
   # Click on "API Token" in the dashboard
   ```

3. **For Development Without QPU**:
   ```python
   # Set in .env:
   REQUIRE_QPU=false
   
   # This allows using D-Wave's hybrid solvers and simulators
   # Still uses real D-Wave SDK, just not quantum hardware
   ```

4. **Alternative: Use D-Wave Ocean SDK Simulators**:
   ```bash
   # Install Ocean SDK
   pip install dwave-ocean-sdk
   
   # Use local simulators for development
   # (We already removed SimulatedAnnealingSampler, but you can use hybrid solvers)
   ```

## 5. Quick Start Commands

### Step 1: Install Python Dependencies
```bash
pip install dwave-ocean-sdk mcp psycopg2-binary aiohttp pyyaml
```

### Step 2: Generate Security Keys
```bash
python3 scripts/generate-secure-keys.py > .env
```

### Step 3: Set Up Free D-Wave Account
1. Sign up at https://cloud.dwavesys.com/leap/signup/
2. Get API token from dashboard
3. Add to .env: `DWAVE_API_TOKEN=DEV-xxxxx`

### Step 4: Use Local Database
```bash
# Just use SQLite for development:
echo "DATABASE_URL=sqlite:///./mcp_local.db" >> .env
```

### Step 5: Run MCP Server
```bash
# For Cursor integration:
# Add to Cursor settings as shown above

# For standalone testing:
python3 mcp_server/real_mcp_server.py
```

## 6. Minimal Development Setup

For the absolute minimum to get started:

```bash
# 1. Create minimal .env
cat > .env << EOF
MCP_TRANSPORT=stdio
DATABASE_URL=sqlite:///./mcp_local.db
REDIS_URL=memory://
DWAVE_API_TOKEN=skip  # Add real token later
REQUIRE_QPU=false
JWT_SECRET_KEY=dev-secret-key-change-in-production
API_SECRET_KEY=dev-api-key-change-in-production
ENCRYPTION_KEY=dev-encryption-key-change-in-production
AUTH_ENABLED=false
ENVIRONMENT=development
EOF

# 2. Install minimal dependencies
pip install mcp pyyaml

# 3. Create data directory
mkdir -p data

# 4. Test MCP server
python3 mcp_server/real_mcp_server.py
```

## Summary

- **PostgreSQL**: Use SQLite for dev, Docker for local PostgreSQL, or real PostgreSQL for production
- **Redis**: Skip for development (use memory://), use Docker for local Redis
- **D-Wave**: FREE tier available with 1 min/month QPU time
- **MCP Server**: Run as Cursor MCP server (recommended) or standalone HTTP server

No need to pay for anything during development! All services have free/local options.