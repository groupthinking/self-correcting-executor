#!/usr/bin/env python3
"""
Comprehensive test runner for utils helpers
Runs all tests including performance and stress tests
"""

import subprocess
import sys
import time

def run_tests():
    """Run comprehensive test suite"""
    print("Running comprehensive tests for utils/helpers.py...")
    print("=" * 60)
    
    # Run regular tests
    print("\n1. Running standard tests...")
    result = subprocess.run([
        sys.executable, "-m", "pytest", 
        "test_utils_helpers.py", 
        "-v", "--tb=short",
        "-m", "not slow"
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✓ Standard tests passed")
    else:
        print("✗ Standard tests failed")
        print(result.stdout)
        print(result.stderr)
    
    # Run slow/performance tests
    print("\n2. Running performance and stress tests...")
    result_slow = subprocess.run([
        sys.executable, "-m", "pytest", 
        "test_utils_helpers.py", 
        "-v", "--tb=short",
        "-m", "slow"
    ], capture_output=True, text=True)
    
    if result_slow.returncode == 0:
        print("✓ Performance tests passed")
    else:
        print("✗ Performance tests failed")
        print(result_slow.stdout)
        print(result_slow.stderr)
    
    # Summary
    print("\n" + "=" * 60)
    if result.returncode == 0 and result_slow.returncode == 0:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(run_tests())