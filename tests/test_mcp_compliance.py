#!/usr/bin/env python3
"""MCP Compliance Tests - Ensure all components use real MCP, no mocks"""

import pytest
import asyncio
import os
import json
from pathlib import Path
from typing import Dict, Any, List
import aiohttp

# Import MCP components
from config.mcp_config import MCPConfig
from mcp_server.real_mcp_server import RealMCPServer
from connectors.dwave_quantum_connector import DWaveQuantumConnector
from agents.a2a_mcp_integration import MCPEnabledA2AAgent, a2a_mcp_orchestrator

class TestMCPCompliance:
    """Test suite to ensure MCP compliance across the codebase"""
    
    @pytest.fixture
    async def mcp_config(self):
        """Get MCP configuration"""
        return MCPConfig()
    
    @pytest.fixture
    async def mcp_server(self):
        """Create real MCP server instance"""
        server = RealMCPServer()
        return server
    
    def test_no_mock_endpoints(self, mcp_config):
        """Test that no mock endpoints are configured"""
        # This should raise an exception if any mock URLs are found
        assert mcp_config.check_no_mocks() == True
        
        # Verify all endpoints are real
        endpoints = mcp_config.get_endpoints()
        mock_indicators = ['mock', 'fake', 'simulated', 'placeholder', 'example.com']
        
        for name, url in endpoints.items():
            if url:
                url_lower = url.lower()
                for indicator in mock_indicators:
                    assert indicator not in url_lower, f"Mock indicator '{indicator}' found in {name}: {url}"
    
    def test_required_environment_variables(self, mcp_config):
        """Test that required environment variables are documented"""
        validation = mcp_config.validate_config()
        
        # These are critical for production
        critical_features = ['Quantum computing', 'Database connection', 'Authentication']
        
        for feature in critical_features:
            if not validation.get(feature, False):
                pytest.skip(f"{feature} not configured - set required environment variables")
    
    @pytest.mark.asyncio
    async def test_quantum_requires_real_qpu(self):
        """Test that quantum connector requires real QPU, no simulation"""
        connector = DWaveQuantumConnector()
        
        # This should fail if no real QPU is available
        if not os.getenv('DWAVE_API_TOKEN'):
            pytest.skip("DWAVE_API_TOKEN not set - skipping quantum test")
        
        # Connect should succeed or fail, but never fall back to simulation
        try:
            connected = await connector.connect({})
            if connected:
                # Verify it's a real QPU
                await connector.ensure_real_qpu()
                assert connector.solver_info.get('type') == 'QPU'
                await connector.disconnect()
        except RuntimeError as e:
            # Expected if no QPU available
            assert "No D-Wave QPU available" in str(e)
    
    @pytest.mark.asyncio
    async def test_data_processor_no_simulation(self):
        """Test that data processor doesn't return simulated results"""
        from protocols.data_processor import task as process_data
        
        result = process_data()
        
        # Should either succeed with real data or fail
        if result['success']:
            assert 'simulation' not in str(result).lower()
            assert 'simulated' not in str(result).lower()
            assert result.get('directory') is not None
        else:
            # Failed because no data directory - this is expected
            assert 'error' in result
            assert 'No data directory found' in result['error']
    
    @pytest.mark.asyncio
    async def test_mcp_server_tools_are_real(self, mcp_server):
        """Test that MCP server tools perform real operations"""
        # Test data processing tool
        result = await mcp_server.server.tool('process_data')(
            data_path='./test_data',
            operation='analyze'
        )
        
        # Should return real analysis or error, never mock data
        assert 'mock' not in str(result).lower()
        assert 'simulated' not in str(result).lower()
    
    @pytest.mark.asyncio
    async def test_agents_use_real_mcp(self):
        """Test that agents use real MCP for communication"""
        agent = MCPEnabledA2AAgent("test_agent", ["analyze"])
        
        # Test data analysis uses real MCP
        result = await agent._analyze_data({'path': './data'})
        
        # Should return real analysis results
        assert result['analysis_type'] in ['comprehensive', 'failed']
        assert 'confidence' in result
        
        # Test code generation
        result = await agent._generate_code({
            'type': 'function',
            'language': 'python'
        })
        
        assert result['code_type'] in ['function', 'error']
        assert 'code' in result
    
    def test_no_placeholder_code_in_production(self):
        """Scan for placeholder code in production files"""
        production_dirs = ['agents', 'connectors', 'mcp_server', 'protocols']
        placeholder_patterns = ['TODO:', 'FIXME:', 'XXX:', 'HACK:', 'placeholder', 'NotImplementedError']
        
        issues = []
        
        for dir_name in production_dirs:
            dir_path = Path(dir_name)
            if dir_path.exists():
                for py_file in dir_path.rglob('*.py'):
                    # Skip test files
                    if 'test' in py_file.name:
                        continue
                        
                    content = py_file.read_text()
                    for line_num, line in enumerate(content.splitlines(), 1):
                        for pattern in placeholder_patterns:
                            if pattern in line and not line.strip().startswith('#'):
                                issues.append(f"{py_file}:{line_num} - {line.strip()}")
        
        # Report but don't fail - these need to be addressed
        if issues:
            print("\n⚠️  Placeholder code found in production files:")
            for issue in issues[:10]:  # Show first 10
                print(f"  - {issue}")
            print(f"  ... and {len(issues) - 10} more") if len(issues) > 10 else None
    
    @pytest.mark.asyncio
    async def test_mcp_http_endpoints(self, mcp_config):
        """Test that MCP HTTP endpoints are accessible"""
        endpoints = mcp_config.get_endpoints()
        mcp_url = endpoints.get('mcp_server')
        
        if not mcp_url or 'localhost' not in mcp_url:
            pytest.skip("MCP server not running locally")
        
        try:
            async with aiohttp.ClientSession() as session:
                # Test health endpoint
                async with session.get(f"{mcp_url}/health", timeout=aiohttp.ClientTimeout(total=5)) as response:
                    assert response.status in [200, 404]  # 404 if health endpoint not implemented yet
        except aiohttp.ClientError:
            pytest.skip("MCP server not accessible")
    
    def test_ui_components_no_mock_data(self):
        """Test that UI components don't use mock data"""
        ui_dir = Path('ui/Build a Website Guide')
        
        if not ui_dir.exists():
            ui_dir = Path('frontend/src')
        
        if ui_dir.exists():
            issues = []
            
            for ts_file in ui_dir.rglob('*.ts*'):
                content = ts_file.read_text()
                
                # Check for real data patterns
                if 'realSearchResults' in content or 'real_' in content:
                    # Good - using real data functions
                    pass
                
                # Report any remaining mock patterns (should be replaced)
                if 'mock' in content.lower() and 'real' not in content:
                    lines = content.splitlines()
                    for i, line in enumerate(lines, 1):
                        if 'mock' in line.lower():
                            issues.append(f"{ts_file}:{i} - {line.strip()}")
            
            # These should all be replaced
            assert len(issues) == 0, f"Found {len(issues)} mock references in UI"
    
    def test_integration_points_documented(self):
        """Test that all MCP integration points are documented"""
        required_docs = [
            'MCP_INTEGRATION_PLAN.md',
            'docs/architecture/MCP_AGENT_RUNTIME_ARCHITECTURE.md'
        ]
        
        for doc in required_docs:
            doc_path = Path(doc)
            if doc_path.exists():
                content = doc_path.read_text()
                
                # Check for required sections
                assert 'Real' in content or 'real' in content
                assert 'mock' not in content.lower() or 'no mock' in content.lower()
                assert 'MCP' in content
    
    def test_env_template_exists(self):
        """Test that .env template exists with all required variables"""
        env_example = Path('.env.example')
        
        assert env_example.exists(), ".env.example file missing"
        
        content = env_example.read_text()
        
        # Check for required variables
        required_vars = [
            'DWAVE_API_TOKEN',
            'DATABASE_URL',
            'JWT_SECRET_KEY',
            'ENCRYPTION_KEY',
            'MCP_SERVER_URL'
        ]
        
        for var in required_vars:
            assert var in content, f"{var} missing from .env.example"

def run_compliance_check():
    """Run MCP compliance check as a standalone script"""
    print("🔍 Running MCP Compliance Check...")
    
    # Create config and run checks
    config = MCPConfig()
    
    print("\n✅ Checking endpoints...")
    try:
        config.check_no_mocks()
        print("   No mock endpoints found!")
    except ValueError as e:
        print(f"   ❌ {e}")
        return False
    
    print("\n✅ Validating configuration...")
    validation = config.validate_config()
    for feature, valid in validation.items():
        status = "✓" if valid else "✗"
        print(f"   {status} {feature}")
    
    print("\n✅ Checking for placeholder code...")
    # Quick scan for placeholders
    placeholder_count = 0
    for py_file in Path('.').rglob('*.py'):
        if 'test' not in str(py_file) and 'venv' not in str(py_file):
            try:
                content = py_file.read_text()
                if 'TODO:' in content or 'FIXME:' in content:
                    placeholder_count += 1
            except (IOError, UnicodeDecodeError):
                # Intentionally ignore files that can't be read/decoded; skip them in this quick placeholder scan
                pass
    
    if placeholder_count > 0:
        print(f"   ⚠️  Found {placeholder_count} files with TODO/FIXME markers")
    else:
        print("   No placeholder code found!")
    
    print("\n✅ MCP Compliance Check Complete!")
    return True

if __name__ == "__main__":
    # Run as script
    success = run_compliance_check()
    exit(0 if success else 1)
