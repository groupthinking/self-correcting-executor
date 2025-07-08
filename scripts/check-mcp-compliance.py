#!/usr/bin/env python3
"""MCP Compliance Check - Ensure all components use real MCP, no mocks"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

def check_no_mock_endpoints() -> Tuple[bool, List[str]]:
    """Check configuration for mock endpoints"""
    issues = []
    
    try:
        from config.mcp_config import MCPConfig
        config = MCPConfig()
        
        # Check for mock indicators
        endpoints = config.get_endpoints()
        mock_indicators = ['mock', 'fake', 'simulated', 'placeholder', 'example.com', 'test']
        
        for name, url in endpoints.items():
            if url:
                url_lower = url.lower()
                for indicator in mock_indicators:
                    if indicator in url_lower and 'localhost' not in url_lower:
                        issues.append(f"Mock indicator '{indicator}' found in {name}: {url}")
        
        # Run built-in check
        try:
            config.check_no_mocks()
        except ValueError as e:
            issues.append(str(e))
            
    except ImportError as e:
        issues.append(f"Failed to import MCPConfig: {e}")
    
    return len(issues) == 0, issues

def check_mock_implementations_in_code() -> Tuple[bool, List[str]]:
    """Scan codebase for mock implementations"""
    issues = []
    mock_patterns = ['mock-gcp-api', 'mockSearchResults', 'mock_', 'SimulatedAnnealingSampler']
    
    # Directories to check
    check_dirs = ['agents', 'connectors', 'protocols', 'mcp_server', 'ui', 'frontend']
    
    for dir_name in check_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            # Check Python files
            for py_file in dir_path.rglob('*.py'):
                if 'test' in py_file.name or '__pycache__' in str(py_file):
                    continue
                    
                try:
                    content = py_file.read_text()
                    for pattern in mock_patterns:
                        if pattern in content:
                            # Count occurrences
                            count = content.count(pattern)
                            issues.append(f"{py_file}: {count} occurrences of '{pattern}'")
                except Exception as e:
                    pass
            
            # Check TypeScript files
            for ts_file in dir_path.rglob('*.ts*'):
                if 'node_modules' in str(ts_file):
                    continue
                    
                try:
                    content = ts_file.read_text()
                    for pattern in mock_patterns:
                        if pattern in content:
                            count = content.count(pattern)
                            issues.append(f"{ts_file}: {count} occurrences of '{pattern}'")
                except Exception as e:
                    pass
    
    return len(issues) == 0, issues

def check_placeholder_code() -> Tuple[bool, List[str]]:
    """Check for TODO/FIXME/placeholder code"""
    issues = []
    placeholder_patterns = ['TODO:', 'FIXME:', 'XXX:', 'HACK:', 'NotImplementedError']
    
    production_dirs = ['agents', 'connectors', 'mcp_server', 'protocols']
    
    for dir_name in production_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            for py_file in dir_path.rglob('*.py'):
                if 'test' in py_file.name:
                    continue
                    
                try:
                    content = py_file.read_text()
                    lines = content.splitlines()
                    for i, line in enumerate(lines, 1):
                        for pattern in placeholder_patterns:
                            if pattern in line and not line.strip().startswith('#'):
                                issues.append(f"{py_file}:{i} - {pattern}")
                except Exception:
                    pass
    
    # Just count, don't list all
    if issues:
        return False, [f"Found {len(issues)} placeholder code instances"]
    return True, []

def check_environment_setup() -> Tuple[bool, List[str]]:
    """Check if environment is properly configured"""
    issues = []
    
    # Check .env.example exists
    if not Path('.env.example').exists():
        issues.append(".env.example file is missing")
    
    # Check critical environment variables
    critical_vars = [
        ('DWAVE_API_TOKEN', 'Quantum computing'),
        ('DATABASE_URL', 'Database connection'),
        ('JWT_SECRET_KEY', 'Authentication'),
        ('MCP_SERVER_URL', 'MCP Server')
    ]
    
    missing_vars = []
    for var, feature in critical_vars:
        if not os.getenv(var):
            missing_vars.append(f"{var} ({feature})")
    
    if missing_vars:
        issues.append(f"Missing environment variables: {', '.join(missing_vars)}")
    
    return len(issues) == 0, issues

def check_quantum_no_simulation() -> Tuple[bool, List[str]]:
    """Check that quantum connector doesn't use simulation"""
    issues = []
    
    quantum_file = Path('connectors/dwave_quantum_connector.py')
    if quantum_file.exists():
        content = quantum_file.read_text()
        
        # Should NOT import SimulatedAnnealingSampler
        if 'SimulatedAnnealingSampler' in content and 'removed - real QPU only' not in content:
            issues.append("Quantum connector still imports SimulatedAnnealingSampler")
        
        # Should require real QPU
        if 'Real QPU required' not in content:
            issues.append("Quantum connector doesn't enforce real QPU requirement")
    else:
        issues.append("Quantum connector file not found")
    
    return len(issues) == 0, issues

def check_data_processor_real() -> Tuple[bool, List[str]]:
    """Check that data processor uses real data"""
    issues = []
    
    data_file = Path('protocols/data_processor.py')
    if data_file.exists():
        content = data_file.read_text()
        
        # Should NOT return simulated results
        if 'simulated results' in content.lower() and 'NO SIMULATIONS' not in content:
            issues.append("Data processor still returns simulated results")
    else:
        issues.append("Data processor file not found")
    
    return len(issues) == 0, issues

def run_all_checks() -> bool:
    """Run all compliance checks"""
    print("🔍 MCP Compliance Check\n")
    print("=" * 50)
    
    all_passed = True
    
    # Check 1: No mock endpoints
    print("\n1️⃣  Checking endpoints configuration...")
    passed, issues = check_no_mock_endpoints()
    if passed:
        print("   ✅ No mock endpoints found")
    else:
        print("   ❌ Mock endpoints detected:")
        for issue in issues[:5]:
            print(f"      - {issue}")
        all_passed = False
    
    # Check 2: No mock implementations
    print("\n2️⃣  Checking for mock implementations in code...")
    passed, issues = check_mock_implementations_in_code()
    if passed:
        print("   ✅ No mock implementations found")
    else:
        print("   ❌ Mock implementations found:")
        for issue in issues[:5]:
            print(f"      - {issue}")
        if len(issues) > 5:
            print(f"      ... and {len(issues) - 5} more")
        all_passed = False
    
    # Check 3: Placeholder code
    print("\n3️⃣  Checking for placeholder code...")
    passed, issues = check_placeholder_code()
    if passed:
        print("   ✅ No placeholder code found")
    else:
        print("   ⚠️  Placeholder code found:")
        for issue in issues:
            print(f"      - {issue}")
    
    # Check 4: Environment setup
    print("\n4️⃣  Checking environment configuration...")
    passed, issues = check_environment_setup()
    if passed:
        print("   ✅ Environment properly configured")
    else:
        print("   ⚠️  Environment issues:")
        for issue in issues:
            print(f"      - {issue}")
    
    # Check 5: Quantum no simulation
    print("\n5️⃣  Checking quantum connector...")
    passed, issues = check_quantum_no_simulation()
    if passed:
        print("   ✅ Quantum connector requires real QPU")
    else:
        print("   ❌ Quantum connector issues:")
        for issue in issues:
            print(f"      - {issue}")
        all_passed = False
    
    # Check 6: Data processor
    print("\n6️⃣  Checking data processor...")
    passed, issues = check_data_processor_real()
    if passed:
        print("   ✅ Data processor uses real data")
    else:
        print("   ❌ Data processor issues:")
        for issue in issues:
            print(f"      - {issue}")
        all_passed = False
    
    # Summary
    print("\n" + "=" * 50)
    if all_passed:
        print("✅ MCP COMPLIANCE CHECK PASSED!")
        print("All components use real implementations.")
    else:
        print("❌ MCP COMPLIANCE CHECK FAILED!")
        print("Mock implementations still exist in the codebase.")
        print("\nRun the following to fix:")
        print("1. python3 scripts/replace-all-mocks.py")
        print("2. Review and fix remaining issues manually")
        print("3. Set up required environment variables")
    
    return all_passed

if __name__ == "__main__":
    success = run_all_checks()
    sys.exit(0 if success else 1)
