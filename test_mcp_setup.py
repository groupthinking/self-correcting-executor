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
