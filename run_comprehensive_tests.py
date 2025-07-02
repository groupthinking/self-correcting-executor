#!/usr/bin/env python3
"""
Comprehensive test runner for utils helpers
Runs all tests including performance and stress tests
"""

import subprocess
import sys
import time

def run_tests():
    """
    Runs both standard and performance test suites for the `utils/helpers.py` module using pytest.
    
    This function executes two categories of tests: standard tests (excluding those marked as slow) and performance/stress tests (marked as slow). It prints the results of each test run, including output and errors if any tests fail, and provides a summary at the end.
    
    Returns:
        int: 0 if all tests pass, 1 if any test fails.
    
    Example:
        status = run_tests()
        if status == 0:
            print("All tests succeeded.")
        else:
            print("Some tests failed.")
    """
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