"""
Pytest configuration and shared fixtures for script improvement tests.
"""

import pytest
import tempfile
import os
import textwrap
from pathlib import Path

@pytest.fixture(scope="session")
def project_root():
    """Get the project root directory."""
    return Path(__file__).parent.absolute()

@pytest.fixture(scope="session") 
def temp_workspace():
    """Create a temporary workspace for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)

@pytest.fixture
def sample_scripts():
    """Provide a collection of sample Python scripts for testing."""
    return {
        "minimal": "def main(): pass",
        
        "basic": textwrap.dedent("""
            def greet(name):
                print(f"Hello, {name}!")
            
            def main():
                greet("World")
            
            if __name__ == "__main__":
                main()
        """).strip(),
        
        "with_imports": textwrap.dedent("""
            import os
            import sys
            from datetime import datetime
            
            def get_timestamp():
                return datetime.now().isoformat()
            
            def main():
                print(f"Current time: {get_timestamp()}")
                print(f"Python version: {sys.version}")
        """).strip(),
        
        "with_classes": textwrap.dedent("""
            class Calculator:
                def add(self, a, b):
                    return a + b
                
                def multiply(self, a, b):
                    return a * b
            
            def main():
                calc = Calculator()
                result = calc.add(2, 3)
                print(f"Result: {result}")
        """).strip(),
        
        "with_error_handling": textwrap.dedent("""
            import logging
            
            def divide(a, b):
                try:
                    return a / b
                except ZeroDivisionError:
                    logging.error("Division by zero")
                    return None
            
            def main():
                result = divide(10, 2)
                print(f"Result: {result}")
        """).strip(),
        
        "complex": textwrap.dedent("""
            import json
            import requests
            from typing import List, Dict, Optional
            
            class DataProcessor:
                def __init__(self, api_url: str):
                    self.api_url = api_url
                    self.data = []
                
                def fetch_data(self) -> Optional[List[Dict]]:
                    try:
                        response = requests.get(self.api_url)
                        response.raise_for_status()
                        return response.json()
                    except requests.RequestException as e:
                        print(f"Error fetching data: {e}")
                        return None
                
                def process_data(self, raw_data: List[Dict]) -> List[Dict]:
                    processed = []
                    for item in raw_data:
                        if 'id' in item and 'name' in item:
                            processed.append({
                                'id': item['id'],
                                'name': item['name'].upper(),
                                'processed_at': '2023-01-01T00:00:00'
                            })
                    return processed
                
                def save_data(self, data: List[Dict], filename: str) -> bool:
                    try:
                        with open(filename, 'w') as f:
                            json.dump(data, f, indent=2)
                        return True
                    except IOError as e:
                        print(f"Error saving data: {e}")
                        return False
            
            def main():
                processor = DataProcessor("https://api.example.com/data")
                raw_data = processor.fetch_data()
                
                if raw_data:
                    processed_data = processor.process_data(raw_data)
                    success = processor.save_data(processed_data, "output.json")
                    
                    if success:
                        print(f"Successfully processed {len(processed_data)} items")
                    else:
                        print("Failed to save processed data")
                else:
                    print("Failed to fetch data")
            
            if __name__ == "__main__":
                main()
        """).strip()
    }

@pytest.fixture
def script_files(temp_workspace, sample_scripts):
    """Create temporary script files for testing."""
    script_files = {}
    
    for name, content in sample_scripts.items():
        file_path = temp_workspace / f"{name}_script.py"
        file_path.write_text(content)
        script_files[name] = file_path
    
    return script_files

# Configure pytest
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"  
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as a performance test"
    )

def pytest_collection_modifyitems(config, items):
    """Modify test collection to add default markers."""
    for item in items:
        # Add 'unit' marker to tests that don't have integration/slow markers
        if not any(marker.name in ['integration', 'slow', 'performance'] 
                  for marker in item.iter_markers()):
            item.add_marker(pytest.mark.unit)

@pytest.fixture(autouse=True)
def reset_logging():
    """Reset logging configuration between tests."""
    import logging
    # Clear any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Reset logging level
    logging.root.setLevel(logging.WARNING)
    
    yield
    
    # Clean up after test
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)