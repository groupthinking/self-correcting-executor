"""
Shared pytest configuration and fixtures for test_helpers tests.
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch
from pathlib import Path
import tempfile
import shutil

# Configure pytest
def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "benchmark: marks tests as benchmarks")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "performance: marks tests as performance tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")

def pytest_collection_modifyitems(config, items):
    """Modify test collection based on command line options."""
    if config.getoption("--runslow", default=False):
        return  # Don't skip slow tests if --runslow is specified
    
    skip_slow = pytest.mark.skip(reason="slow test skipped, use --runslow to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)

def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--runslow", 
        action="store_true", 
        default=False, 
        help="run slow tests"
    )
    parser.addoption(
        "--runperformance", 
        action="store_true", 
        default=False, 
        help="run performance tests"
    )

# Global fixtures
@pytest.fixture(scope="session")
def test_session_data():
    """Session-scoped fixture for test data."""
    return {
        'session_id': 'test_session_123',
        'start_time': '2024-01-01T00:00:00Z'
    }

@pytest.fixture(scope="function")
def clean_environment():
    """Fixture that ensures a clean environment for each test."""
    original_env = os.environ.copy()
    yield
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)

@pytest.fixture
def mock_file_system():
    """Fixture for mocking file system operations."""
    with patch('builtins.open') as mock_open, \
         patch('os.path.exists') as mock_exists, \
         patch('os.makedirs') as mock_makedirs:
        
        mock_exists.return_value = True
        yield {
            'open': mock_open,
            'exists': mock_exists,
            'makedirs': mock_makedirs
        }

@pytest.fixture
def performance_monitor():
    """Fixture for monitoring test performance."""
    import time
    start_time = time.perf_counter()
    yield
    end_time = time.perf_counter()
    duration = end_time - start_time
    if duration > 1.0:  # Log slow tests
        print(f"\n⚠️  Slow test detected: {duration:.3f}s")