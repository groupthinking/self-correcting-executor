#!/bin/bash
# Quick Development Setup Script
# Sets up minimal environment for MCP development without external services

echo "🚀 Quick MCP Development Setup"
echo "=============================="

# Step 1: Create minimal .env file
echo "📝 Creating development .env file..."
cat > .env << EOF
# MCP Server Configuration
MCP_TRANSPORT=stdio
MCP_SERVER_URL=stdio://
MCP_SERVER_HOST=localhost
MCP_SERVER_PORT=8090

# Database (SQLite for development)
DATABASE_URL=sqlite:///./mcp_local.db

# Cache (In-memory for development)
REDIS_URL=memory://

# D-Wave (Development mode - no QPU required)
DWAVE_API_TOKEN=development-mode
REQUIRE_QPU=false
QUANTUM_NUM_READS=10

# Security (Development keys - DO NOT USE IN PRODUCTION)
JWT_SECRET_KEY=dev-jwt-secret-key-minimum-32-characters-long
API_SECRET_KEY=dev-api-secret-key-minimum-32-characters-long
ENCRYPTION_KEY=dev-encryption-key-for-local-development-only
AUTH_ENABLED=false
ENABLE_SANDBOXING=false

# Environment
ENVIRONMENT=development
DATA_DIR=./data
CACHE_DIR=./cache
LOG_LEVEL=INFO
EOF

echo "✅ Created .env file for development"

# Step 2: Create required directories
echo "📁 Creating required directories..."
mkdir -p data cache logs middleware

# Step 3: Create sample data file
echo "📄 Creating sample data..."
cat > data/sample.json << EOF
{
  "items": [
    {"id": 1, "name": "Test Item 1", "value": 100},
    {"id": 2, "name": "Test Item 2", "value": 200},
    {"id": 3, "name": "Test Item 3", "value": 300}
  ],
  "metadata": {
    "created": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "source": "development"
  }
}
EOF

# Step 4: Install minimal Python dependencies
echo "📦 Installing Python dependencies..."
pip install pyyaml 2>/dev/null || echo "⚠️  PyYAML already installed or pip not available"

# Step 5: Create Cursor MCP configuration
echo "⚙️  Creating Cursor MCP configuration..."
mkdir -p .cursor
cat > .cursor/mcp_config.json << EOF
{
  "mcpServers": {
    "real-mcp-server": {
      "command": "python3",
      "args": ["${PWD}/mcp_server/real_mcp_server.py"],
      "env": {
        "PYTHONPATH": "${PWD}"
      }
    }
  }
}
EOF

# Step 6: Test imports
echo "🧪 Testing Python imports..."
python3 -c "
try:
    from config.mcp_config import MCPConfig
    print('✅ MCPConfig imports successfully')
except Exception as e:
    print(f'❌ Import error: {e}')

try:
    from pathlib import Path
    if Path('.env').exists():
        print('✅ .env file exists')
    if Path('data/sample.json').exists():
        print('✅ Sample data created')
except Exception as e:
    print(f'❌ Path check error: {e}')
"

# Step 7: Create a test script
echo "🔧 Creating test script..."
cat > test_mcp_setup.py << 'EOF'
#!/usr/bin/env python3
"""Test MCP Setup"""

import os
import sys
from pathlib import Path

def test_setup():
    """Test the development setup"""
    
    print("\n🔍 Testing MCP Development Setup\n")
    
    # Check .env file
    if Path('.env').exists():
        print("✅ .env file exists")
        # Load and check key variables
        with open('.env', 'r') as f:
            env_content = f.read()
            required_vars = ['DATABASE_URL', 'MCP_TRANSPORT', 'JWT_SECRET_KEY']
            for var in required_vars:
                if var in env_content:
                    print(f"✅ {var} is configured")
                else:
                    print(f"❌ {var} is missing")
    else:
        print("❌ .env file not found")
    
    # Check directories
    for dir_name in ['data', 'cache', 'logs']:
        if Path(dir_name).exists():
            print(f"✅ {dir_name}/ directory exists")
        else:
            print(f"❌ {dir_name}/ directory missing")
    
    # Check sample data
    if Path('data/sample.json').exists():
        print("✅ Sample data exists")
    
    # Test imports
    try:
        from config.mcp_config import MCPConfig
        config = MCPConfig()
        print("✅ Can import MCPConfig")
        
        # Test no mocks
        try:
            config.check_no_mocks()
            print("✅ No mock endpoints configured")
        except ValueError as e:
            print(f"❌ Mock endpoint found: {e}")
    except ImportError as e:
        print(f"❌ Cannot import MCPConfig: {e}")
    
    print("\n📊 Setup Summary:")
    print("- Use 'source .env' to load environment variables")
    print("- Run 'python3 mcp_server/real_mcp_server.py' to start MCP server")
    print("- D-Wave: Sign up at https://cloud.dwavesys.com/leap/ for free API token")
    print("\n✨ Development environment ready!")

if __name__ == "__main__":
    test_setup()
EOF

chmod +x test_mcp_setup.py

# Final summary
echo ""
echo "✅ Quick Development Setup Complete!"
echo "===================================="
echo ""
echo "Next steps:"
echo "1. Load environment: source .env"
echo "2. Test setup: python3 test_mcp_setup.py"
echo "3. Start MCP server: python3 mcp_server/real_mcp_server.py"
echo ""
echo "Optional:"
echo "- Get free D-Wave token: https://cloud.dwavesys.com/leap/signup/"
echo "- Install full dependencies: pip install dwave-ocean-sdk mcp aiohttp"
echo ""
echo "📍 Current setup uses:"
echo "   - SQLite (no PostgreSQL needed)"
echo "   - In-memory cache (no Redis needed)"
echo "   - Development mode (no D-Wave QPU needed)"
echo ""