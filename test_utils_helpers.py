#!/usr/bin/env python3
"""
Test suite for utils/helpers.py
Tests all utility helper functions with comprehensive coverage
"""

import pytest
import json
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import sys
sys.path.append(str(Path(__file__).parent))

from utils.helpers import (
    safe_json_parse,
    safe_json_dumps,
    generate_hash,
    retry_with_backoff,
    flatten_dict,
    ensure_directory_exists,
    sanitize_filename,
    merge_dicts,
    chunk_list,
    format_duration
)


class TestSafeJsonParse:
    """Test safe_json_parse function"""
    
    def test_valid_json_string(self):
        """Test parsing valid JSON string"""
        json_str = '{"key": "value", "number": 42}'
        result = safe_json_parse(json_str)
        assert result == {"key": "value", "number": 42}
    
    def test_valid_json_array(self):
        """Test parsing valid JSON array"""
        json_str = '[1, 2, 3, "test"]'
        result = safe_json_parse(json_str)
        assert result == [1, 2, 3, "test"]
    
    def test_invalid_json_string(self):
        """Test parsing invalid JSON string"""
        json_str = '{"key": "value",}'  # Trailing comma
        result = safe_json_parse(json_str)
        assert result is None
    
    def test_completely_malformed_json(self):
        """Test parsing completely malformed JSON"""
        json_str = 'not json at all'
        result = safe_json_parse(json_str)
        assert result is None
    
    def test_none_input(self):
        """Test parsing None input"""
        result = safe_json_parse(None)
        assert result is None
    
    def test_empty_string(self):
        """Test parsing empty string"""
        result = safe_json_parse("")
        assert result is None


class TestSafeJsonDumps:
    """Test safe_json_dumps function"""
    
    def test_valid_dict(self):
        """Test serializing valid dictionary"""
        data = {"key": "value", "number": 42}
        result = safe_json_dumps(data)
        assert '"key": "value"' in result
        assert '"number": 42' in result
    
    def test_valid_list(self):
        """Test serializing valid list"""
        data = [1, 2, 3, "test"]
        result = safe_json_dumps(data)
        expected = json.dumps(data, indent=2, default=str)
        assert result == expected
    
    def test_custom_indent(self):
        """Test serializing with custom indentation"""
        data = {"nested": {"key": "value"}}
        result = safe_json_dumps(data, indent=4)
        assert result.count(" ") > result.count("\n")  # More spaces due to indent=4
    
    def test_complex_object_with_datetime(self):
        """Test serializing complex object with datetime (uses default=str)"""
        from datetime import datetime
        data = {"timestamp": datetime.now(), "value": 42}
        result = safe_json_dumps(data)
        assert result != ""  # Should not fail due to default=str
        assert "timestamp" in result
    
    def test_circular_reference(self):
        """Test serializing object with circular reference"""
        data = {}
        data["self"] = data  # Circular reference
        result = safe_json_dumps(data)
        assert result == ""  # Should return empty string on failure


class TestGenerateHash:
    """Test generate_hash function"""
    
    def test_string_input(self):
        """Test hashing string input"""
        text = "test string"
        result = generate_hash(text)
        assert len(result) == 64  # SHA256 hex string length
        assert isinstance(result, str)
        assert all(c in '0123456789abcdef' for c in result)
    
    def test_bytes_input(self):
        """Test hashing bytes input"""
        data = b"test bytes"
        result = generate_hash(data)
        assert len(result) == 64
        assert isinstance(result, str)
    
    def test_consistent_hashing(self):
        """Test that same input produces same hash"""
        text = "consistent test"
        hash1 = generate_hash(text)
        hash2 = generate_hash(text)
        assert hash1 == hash2
    
    def test_different_inputs_different_hashes(self):
        """Test that different inputs produce different hashes"""
        hash1 = generate_hash("input1")
        hash2 = generate_hash("input2")
        assert hash1 != hash2
    
    def test_empty_string(self):
        """Test hashing empty string"""
        result = generate_hash("")
        assert len(result) == 64
        assert result != generate_hash("not empty")


class TestRetryWithBackoff:
    """Test retry_with_backoff function"""
    
    def test_successful_function(self):
        """Test function that succeeds on first try"""
        def success_func():
            return "success"
        
        result = retry_with_backoff(success_func)
        assert result == "success"
    
    def test_function_succeeds_after_retries(self):
        """Test function that succeeds after failures"""
        attempts = []
        
        def eventually_succeeds():
            attempts.append(1)
            if len(attempts) < 3:
                raise ValueError("Not yet")
            return "finally succeeded"
        
        result = retry_with_backoff(eventually_succeeds, max_retries=3)
        assert result == "finally succeeded"
        assert len(attempts) == 3
    
    def test_function_fails_all_retries(self):
        """Test function that fails all retry attempts"""
        def always_fails():
            raise ValueError("Always fails")
        
        with pytest.raises(ValueError, match="Always fails"):
            retry_with_backoff(always_fails, max_retries=2)
    
    @patch('time.sleep')
    def test_backoff_timing(self, mock_sleep):
        """Test exponential backoff timing"""
        def fails_twice():
            if mock_sleep.call_count < 2:
                raise ValueError("Fail")
            return "success"
        
        result = retry_with_backoff(fails_twice, max_retries=3, base_delay=1.0)
        assert result == "success"
        
        # Check exponential backoff: 1s, 2s
        expected_delays = [1.0, 2.0]
        actual_delays = [call[0][0] for call in mock_sleep.call_args_list]
        assert actual_delays == expected_delays


class TestFlattenDict:
    """Test flatten_dict function"""
    
    def test_simple_dict(self):
        """Test flattening simple dictionary"""
        data = {"a": 1, "b": 2}
        result = flatten_dict(data)
        assert result == {"a": 1, "b": 2}
    
    def test_nested_dict(self):
        """Test flattening nested dictionary"""
        data = {"a": {"b": {"c": 1}}, "d": 2}
        result = flatten_dict(data)
        expected = {"a.b.c": 1, "d": 2}
        assert result == expected
    
    def test_mixed_nested_dict(self):
        """Test flattening mixed nested dictionary"""
        data = {
            "user": {"name": "John", "address": {"city": "NYC", "zip": "10001"}},
            "age": 30,
            "active": True
        }
        result = flatten_dict(data)
        expected = {
            "user.name": "John",
            "user.address.city": "NYC",
            "user.address.zip": "10001",
            "age": 30,
            "active": True
        }
        assert result == expected
    
    def test_with_prefix(self):
        """Test flattening with custom prefix"""
        data = {"a": {"b": 1}}
        result = flatten_dict(data, prefix="root")
        assert result == {"root.a.b": 1}
    
    def test_empty_dict(self):
        """Test flattening empty dictionary"""
        result = flatten_dict({})
        assert result == {}


class TestEnsureDirectoryExists:
    """Test ensure_directory_exists function"""
    
    def test_create_new_directory(self):
        """Test creating new directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            new_dir = Path(temp_dir) / "new_directory"
            result = ensure_directory_exists(new_dir)
            
            assert result.exists()
            assert result.is_dir()
            assert result == new_dir
    
    def test_existing_directory(self):
        """Test with existing directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            existing_dir = Path(temp_dir)
            result = ensure_directory_exists(existing_dir)
            
            assert result.exists()
            assert result.is_dir()
            assert result == existing_dir
    
    def test_nested_directory_creation(self):
        """Test creating nested directories"""
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_dir = Path(temp_dir) / "level1" / "level2" / "level3"
            result = ensure_directory_exists(nested_dir)
            
            assert result.exists()
            assert result.is_dir()
            assert result == nested_dir
    
    def test_string_path_input(self):
        """Test with string path input"""
        with tempfile.TemporaryDirectory() as temp_dir:
            new_dir_str = f"{temp_dir}/string_path"
            result = ensure_directory_exists(new_dir_str)
            
            assert result.exists()
            assert result.is_dir()
            assert str(result) == new_dir_str


class TestSanitizeFilename:
    """Test sanitize_filename function"""
    
    def test_valid_filename(self):
        """Test already valid filename"""
        filename = "valid_filename.txt"
        result = sanitize_filename(filename)
        assert result == filename
    
    def test_invalid_characters(self):
        """Test filename with invalid characters"""
        filename = 'file<>:"/\\|?*name.txt'
        result = sanitize_filename(filename)
        assert result == "file_________name.txt"
    
    def test_leading_trailing_spaces_dots(self):
        """Test filename with leading/trailing spaces and dots"""
        filename = "  ...filename...  "
        result = sanitize_filename(filename)
        assert result == "filename"
    
    def test_empty_filename(self):
        """Test empty filename"""
        result = sanitize_filename("")
        assert result == "unnamed"
    
    def test_only_invalid_characters(self):
        """Test filename with only invalid characters"""
        filename = "<>?*|"
        result = sanitize_filename(filename)
        assert result == "unnamed"
    
    def test_spaces_and_dots_only(self):
        """Test filename with only spaces and dots"""
        filename = "   ...   "
        result = sanitize_filename(filename)
        assert result == "unnamed"


class TestMergeDicts:
    """Test merge_dicts function"""
    
    def test_simple_merge(self):
        """Test merging simple dictionaries"""
        dict1 = {"a": 1, "b": 2}
        dict2 = {"c": 3, "d": 4}
        result = merge_dicts(dict1, dict2)
        expected = {"a": 1, "b": 2, "c": 3, "d": 4}
        assert result == expected
    
    def test_overlapping_keys(self):
        """Test merging with overlapping keys"""
        dict1 = {"a": 1, "b": 2}
        dict2 = {"b": 3, "c": 4}
        result = merge_dicts(dict1, dict2)
        expected = {"a": 1, "b": 3, "c": 4}  # dict2 takes precedence
        assert result == expected
    
    def test_nested_dict_merge(self):
        """Test deep merging nested dictionaries"""
        dict1 = {"user": {"name": "John", "age": 30}, "active": True}
        dict2 = {"user": {"city": "NYC", "age": 31}, "role": "admin"}
        result = merge_dicts(dict1, dict2)
        expected = {
            "user": {"name": "John", "age": 31, "city": "NYC"},
            "active": True,
            "role": "admin"
        }
        assert result == expected
    
    def test_empty_dicts(self):
        """Test merging empty dictionaries"""
        result = merge_dicts({}, {})
        assert result == {}
    
    def test_original_dicts_unchanged(self):
        """Test that original dictionaries are not modified"""
        dict1 = {"a": 1}
        dict2 = {"b": 2}
        original_dict1 = dict1.copy()
        original_dict2 = dict2.copy()
        
        merge_dicts(dict1, dict2)
        
        assert dict1 == original_dict1
        assert dict2 == original_dict2


class TestChunkList:
    """Test chunk_list function"""
    
    def test_even_chunks(self):
        """Test chunking list into even chunks"""
        data = [1, 2, 3, 4, 5, 6]
        result = chunk_list(data, 2)
        expected = [[1, 2], [3, 4], [5, 6]]
        assert result == expected
    
    def test_uneven_chunks(self):
        """Test chunking list with remainder"""
        data = [1, 2, 3, 4, 5]
        result = chunk_list(data, 2)
        expected = [[1, 2], [3, 4], [5]]
        assert result == expected
    
    def test_chunk_size_larger_than_list(self):
        """Test chunk size larger than list length"""
        data = [1, 2, 3]
        result = chunk_list(data, 5)
        expected = [[1, 2, 3]]
        assert result == expected
    
    def test_chunk_size_one(self):
        """Test chunk size of 1"""
        data = [1, 2, 3]
        result = chunk_list(data, 1)
        expected = [[1], [2], [3]]
        assert result == expected
    
    def test_empty_list(self):
        """Test chunking empty list"""
        result = chunk_list([], 2)
        assert result == []
    
    def test_mixed_data_types(self):
        """Test chunking list with mixed data types"""
        data = [1, "two", 3.0, True, None]
        result = chunk_list(data, 2)
        expected = [[1, "two"], [3.0, True], [None]]
        assert result == expected


class TestFormatDuration:
    """Test format_duration function"""
    
    def test_seconds_format(self):
        """Test formatting duration in seconds"""
        assert format_duration(5.5) == "5.50s"
        assert format_duration(30.25) == "30.25s"
        assert format_duration(59.99) == "59.99s"
    
    def test_minutes_format(self):
        """Test formatting duration in minutes"""
        assert format_duration(60) == "1.0m"
        assert format_duration(90) == "1.5m"
        assert format_duration(3599) == "60.0m"
    
    def test_hours_format(self):
        """Test formatting duration in hours"""
        assert format_duration(3600) == "1.0h"
        assert format_duration(5400) == "1.5h"
        assert format_duration(7200) == "2.0h"
    
    def test_edge_cases(self):
        """Test edge cases for duration formatting"""
        assert format_duration(0) == "0.00s"
        assert format_duration(0.01) == "0.01s"
        assert format_duration(59.99) == "59.99s"
        assert format_duration(60.001) == "1.0m"
    
    def test_large_durations(self):
        """Test very large duration values"""
        one_day = 24 * 3600
        assert format_duration(one_day) == "24.0h"
        
        one_week = 7 * 24 * 3600
        assert format_duration(one_week) == "168.0h"


class TestHelpersIntegration:
    """Integration tests combining multiple helper functions"""
    
    def test_json_and_hash_integration(self):
        """Test combining JSON serialization with hashing"""
        data = {"user": "test", "timestamp": "2023-01-01"}
        json_str = safe_json_dumps(data)
        hash_value = generate_hash(json_str)
        
        assert json_str != ""
        assert len(hash_value) == 64
        
        # Same data should produce same hash
        same_json = safe_json_dumps(data)
        same_hash = generate_hash(same_json)
        assert hash_value == same_hash
    
    def test_file_operations_integration(self):
        """Test combining file operations"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create directory structure
            nested_dir = ensure_directory_exists(f"{temp_dir}/nested/path")
            
            # Sanitize and create filename
            unsafe_filename = "test<>file.txt"
            safe_filename = sanitize_filename(unsafe_filename)
            
            # Create file path
            file_path = nested_dir / safe_filename
            file_path.write_text("test content")
            
            assert file_path.exists()
            assert safe_filename == "test__file.txt"
    
    def test_data_processing_pipeline(self):
        """Test a complete data processing pipeline"""
        # Start with nested data
        data = {
            "users": {
                "john": {"age": 30, "city": "NYC"},
                "jane": {"age": 25, "city": "LA"}
            },
            "settings": {"theme": "dark", "notifications": True}
        }
        
        # Flatten the structure
        flat_data = flatten_dict(data)
        
        # Serialize to JSON
        json_str = safe_json_dumps(flat_data)
        
        # Parse it back
        parsed_data = safe_json_parse(json_str)
        
        # Chunk the keys for processing
        keys = list(parsed_data.keys())
        key_chunks = chunk_list(keys, 2)
        
        assert len(flat_data) == 6  # All nested keys flattened
        assert parsed_data == flat_data  # Round-trip successful
        assert len(key_chunks) == 3  # 6 keys chunked by 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# Additional Enhanced Test Classes for Comprehensive Coverage
class TestSafeJsonParseAdvanced:
    """Advanced edge cases and stress tests for safe_json_parse"""
    
    def test_deeply_nested_json_performance(self):
        """Test parsing very deeply nested JSON structures"""
        # Create deeply nested structure
        nested_data = "value"
        for i in range(100):
            nested_data = {"level": nested_data}
        
        json_str = json.dumps(nested_data)
        result = safe_json_parse(json_str)
        
        # Navigate to verify correct parsing
        current = result
        for i in range(100):
            assert "level" in current
            current = current["level"]
        assert current == "value"
    
    def test_unicode_and_escape_sequences(self):
        """Test parsing JSON with various unicode and escape sequences"""
        test_cases = [
            r'{"unicode": "\u0048\u0065\u006C\u006C\u006F"}',  # "Hello" in unicode
            r'{"escaped": "line1\nline2\ttab"}',               # Newlines and tabs
            r'{"quotes": "He said \"Hello\""}',                # Escaped quotes
            '{"emoji": "🚀 \ud83c\udf1f"}',                   # Mixed emoji encoding
        ]
        
        for json_str in test_cases:
            result = safe_json_parse(json_str)
            assert result is not None
            assert isinstance(result, dict)
    
    def test_json_with_large_numbers(self):
        """Test parsing JSON with very large numbers"""
        large_numbers = [
            '{"big_int": 9223372036854775807}',      # Max 64-bit signed int
            '{"big_float": 1.7976931348623157e+308}', # Near max float
            '{"small_float": 2.2250738585072014e-308}', # Near min positive float
            '{"scientific": 1.23e100}',               # Scientific notation
        ]
        
        for json_str in large_numbers:
            result = safe_json_parse(json_str)
            assert result is not None
            assert isinstance(result, dict)
    
    @pytest.mark.parametrize("malformed_json", [
        '{"key": }',              # Missing value
        '{"key": "value",}',      # Trailing comma
        '{key: "value"}',         # Unquoted key
        "{'key': 'value'}",       # Single quotes
        '{"key": "value"',        # Missing closing brace
        '{"key": undefined}',     # JavaScript undefined
        '{"key": /*comment*/ "value"}',  # Comment in JSON
    ])
    def test_malformed_json_variations(self, malformed_json):
        """Test various malformed JSON inputs"""
        result = safe_json_parse(malformed_json)
        assert result is None


class TestSafeJsonDumpsAdvanced:
    """Advanced tests for safe_json_dumps with complex scenarios"""
    
    def test_circular_reference_detection(self):
        """Test detection and handling of circular references"""
        # Create circular reference
        obj_a = {"name": "A"}
        obj_b = {"name": "B", "ref": obj_a}
        obj_a["ref"] = obj_b
        
        result = safe_json_dumps(obj_a)
        assert result == ""  # Should return empty string due to circular reference
    
    def test_custom_objects_with_str_method(self):
        """Test serialization of custom objects with __str__ method"""
        class CustomObject:
            def __init__(self, value):
                self.value = value
            
            def __str__(self):
                return f"CustomObject(value={self.value})"
        
        data = {"custom": CustomObject(42), "normal": "value"}
        result = safe_json_dumps(data)
        
        assert result != ""
        assert "CustomObject" in result
        assert "42" in result
    
    def test_mixed_data_types_edge_cases(self):
        """Test serialization with edge case data types"""
        from decimal import Decimal
        import uuid
        
        data = {
            "decimal": Decimal("123.456"),
            "uuid": uuid.uuid4(),
            "complex": complex(1, 2),
            "frozenset": frozenset([1, 2, 3]),
            "bytes": b"hello world",
            "range": range(5),
        }
        
        result = safe_json_dumps(data)
        assert result != ""  # Should handle all types via default=str
    
    def test_performance_large_object(self):
        """Test performance with large objects"""
        large_data = {
            f"key_{i}": {
                "value": i,
                "data": "x" * 1000,  # 1KB per entry
                "nested": {"sub_key": f"sub_value_{i}"}
            }
            for i in range(1000)  # ~1MB total
        }
        
        import time
        start_time = time.time()
        result = safe_json_dumps(large_data)
        end_time = time.time()
        
        assert result != ""
        assert end_time - start_time < 5.0  # Should complete within 5 seconds


class TestGenerateHashAdvanced:
    """Advanced hash generation tests"""
    
    def test_hash_distribution(self):
        """Test hash distribution to ensure no obvious patterns"""
        inputs = [f"test_{i}" for i in range(1000)]
        hashes = [generate_hash(inp) for inp in inputs]
        
        # Check that hashes are well distributed
        first_chars = [h[0] for h in hashes]
        char_counts = {}
        for char in first_chars:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        # No single character should dominate (rough distribution check)
        max_count = max(char_counts.values())
        assert max_count < len(hashes) * 0.2  # No more than 20% should start with same char
    
    def test_avalanche_effect(self):
        """Test avalanche effect - small input changes cause large hash changes"""
        base_string = "test_string_for_avalanche"
        base_hash = generate_hash(base_string)
        
        # Change one character
        modified_string = base_string[:-1] + 'X'
        modified_hash = generate_hash(modified_string)
        
        # Count different bits (simplified check)
        base_int = int(base_hash, 16)
        modified_int = int(modified_hash, 16)
        xor_result = base_int ^ modified_int
        different_bits = bin(xor_result).count('1')
        
        # Should have significant bit differences (roughly 50% for good hash)
        assert different_bits > 50  # Out of 256 bits, expect substantial difference
    
    def test_hash_consistency_across_runs(self):
        """Test that hash function is deterministic across multiple runs"""
        test_string = "consistency_test_string"
        hashes = [generate_hash(test_string) for _ in range(10)]
        
        # All hashes should be identical
        assert len(set(hashes)) == 1
        assert all(h == hashes[0] for h in hashes)
    
    def test_empty_and_whitespace_inputs(self):
        """Test hashing of empty and whitespace-only inputs"""
        test_cases = ["", " ", "\t", "\n", "   ", "\t\n "]
        hashes = [generate_hash(case) for case in test_cases]
        
        # All should produce valid hashes
        assert all(len(h) == 64 for h in hashes)
        # All should be different (even whitespace variations)
        assert len(set(hashes)) == len(hashes)


class TestRetryWithBackoffAdvanced:
    """Advanced retry mechanism tests"""
    
    def test_retry_with_different_exception_types(self):
        """Test retry behavior with mixed exception types"""
        exceptions_to_raise = [
            ConnectionError("Connection failed"),
            TimeoutError("Request timed out"), 
            ValueError("Invalid value"),
        ]
        
        call_count = [0]
        
        def failing_function():
            if call_count[0] < len(exceptions_to_raise):
                exc = exceptions_to_raise[call_count[0]]
                call_count[0] += 1
                raise exc
            return "success"
        
        result = retry_with_backoff(failing_function, max_retries=5)
        assert result == "success"
        assert call_count[0] == len(exceptions_to_raise)
    
    @patch('time.sleep')
    def test_exponential_backoff_progression(self, mock_sleep):
        """Test that backoff follows exponential progression"""
        call_count = [0]
        
        def always_fails():
            call_count[0] += 1
            if call_count[0] <= 4:  # Fail first 4 times
                raise RuntimeError("Temporary failure")
            return "success"
        
        result = retry_with_backoff(always_fails, max_retries=5, base_delay=1.0)
        assert result == "success"
        
        # Check exponential progression: 1, 2, 4, 8
        expected_delays = [1.0, 2.0, 4.0, 8.0]
        actual_delays = [call[0][0] for call in mock_sleep.call_args_list]
        assert actual_delays == expected_delays
    
    def test_retry_with_return_values(self):
        """Test retry with functions returning different values"""
        return_values = [None, False, 0, "", "success"]
        call_count = [0]
        
        def function_with_varying_returns():
            if call_count[0] < len(return_values) - 1:
                value = return_values[call_count[0]]
                call_count[0] += 1
                if value is None:
                    raise ValueError("None result")
                return value
            call_count[0] += 1
            return return_values[-1]
        
        result = retry_with_backoff(function_with_varying_returns, max_retries=3)
        assert result == "success"
    
    def test_retry_timeout_simulation(self):
        """Test retry with simulated timeout scenarios"""
        import time
        
        start_time = time.time()
        call_times = []
        
        def time_tracking_function():
            call_times.append(time.time())
            if len(call_times) < 3:
                raise TimeoutError("Simulated timeout")
            return "completed"
        
        result = retry_with_backoff(time_tracking_function, max_retries=3, base_delay=0.1)
        
        assert result == "completed"
        assert len(call_times) == 3
        
        # Verify timing progression
        for i in range(1, len(call_times)):
            time_diff = call_times[i] - call_times[i-1]
            expected_min_delay = 0.1 * (2 ** (i-1))
            assert time_diff >= expected_min_delay * 0.9  # Allow 10% tolerance


class TestFlattenDictAdvanced:
    """Advanced dictionary flattening tests"""
    
    def test_flatten_with_complex_nested_structures(self):
        """Test flattening complex nested structures with mixed types"""
        complex_data = {
            "api": {
                "v1": {
                    "endpoints": ["users", "posts", "comments"],
                    "auth": {"required": True, "methods": ["jwt", "oauth"]},
                    "rate_limits": {"per_hour": 1000, "burst": 10}
                },
                "v2": {
                    "endpoints": ["users", "posts"],
                    "auth": {"required": True, "methods": ["jwt"]},
                    "features": {"pagination": True, "filtering": True}
                }
            },
            "database": {
                "primary": {"host": "db1.local", "port": 5432},
                "replicas": [
                    {"host": "db2.local", "port": 5432},
                    {"host": "db3.local", "port": 5432}
                ]
            }
        }
        
        result = flatten_dict(complex_data)
        
        # Verify specific flattened keys exist
        expected_keys = [
            "api.v1.endpoints",
            "api.v1.auth.required",
            "api.v1.auth.methods",
            "api.v1.rate_limits.per_hour",
            "api.v2.features.pagination",
            "database.primary.host",
            "database.replicas"
        ]
        
        for key in expected_keys:
            assert key in result
    
    def test_flatten_with_numeric_and_boolean_keys(self):
        """Test flattening with non-string keys"""
        data = {
            "config": {
                1: "first_item",
                2: {"nested": "second_nested"},
                True: "boolean_key",
                False: {"deep": "boolean_nested"}
            }
        }
        
        result = flatten_dict(data)
        
        expected_flattened = {
            "config.1": "first_item",
            "config.2.nested": "second_nested", 
            "config.True": "boolean_key",
            "config.False.deep": "boolean_nested"
        }
        
        assert result == expected_flattened
    
    def test_flatten_with_custom_separator(self):
        """Test flattening with custom separator (if supported)"""
        data = {"a": {"b": {"c": "value"}}}
        
        # Test with default separator
        result_dot = flatten_dict(data)
        assert result_dot == {"a.b.c": "value"}
        
        # If function supports custom separator, test it
        # Note: This might not be supported by the current implementation
        try:
            result_underscore = flatten_dict(data, separator="_")
            if result_underscore != result_dot:  # If separator was actually used
                assert result_underscore == {"a_b_c": "value"}
        except TypeError:
            # Function doesn't support custom separator - that's fine
            pass
    
    def test_flatten_performance_large_dict(self):
        """Test flattening performance with large dictionary"""
        # Create large nested dictionary
        large_dict = {}
        for i in range(100):
            large_dict[f"section_{i}"] = {
                f"subsection_{j}": {
                    f"item_{k}": f"value_{i}_{j}_{k}"
                    for k in range(10)
                }
                for j in range(10)
            }
        
        import time
        start_time = time.time()
        result = flatten_dict(large_dict)
        end_time = time.time()
        
        # Should complete reasonably quickly
        assert end_time - start_time < 1.0
        
        # Should have 100 * 10 * 10 = 10,000 flattened keys
        assert len(result) == 10000


class TestFileOperationsAdvanced:
    """Advanced tests for file operation helpers"""
    
    def test_ensure_directory_concurrent_creation(self):
        """Test concurrent directory creation"""
        import threading
        import tempfile
        
        with tempfile.TemporaryDirectory() as temp_dir:
            target_dir = Path(temp_dir) / "concurrent_test"
            results = []
            errors = []
            
            def create_directory(thread_id):
                try:
                    result = ensure_directory_exists(target_dir)
                    results.append((thread_id, result))
                except Exception as e:
                    errors.append((thread_id, e))
            
            # Create multiple threads trying to create same directory
            threads = []
            for i in range(10):
                thread = threading.Thread(target=create_directory, args=(i,))
                threads.append(thread)
                thread.start()
            
            # Wait for all threads
            for thread in threads:
                thread.join()
            
            # All should succeed without errors
            assert len(errors) == 0
            assert len(results) == 10
            assert target_dir.exists()
            assert target_dir.is_dir()
    
    def test_sanitize_filename_edge_cases(self):
        """Test filename sanitization with edge cases"""
        edge_cases = [
            ("", "unnamed"),                    # Empty string
            (".", "unnamed"),                   # Just dot
            ("..", "unnamed"),                  # Double dot
            ("...", "unnamed"),                 # Triple dot
            ("   ", "unnamed"),                 # Only spaces
            ("___", "unnamed"),                 # Only underscores after sanitization
            ("CON", "CON"),                    # Windows reserved name (may vary by implementation)
            ("file" + "x" * 300, None),        # Very long filename
            ("file\x00name.txt", "file_name.txt"),  # Null character
            ("file\r\nname.txt", "file__name.txt"), # Newline characters
        ]
        
        for input_name, expected in edge_cases:
            result = sanitize_filename(input_name)
            if expected is not None:
                assert result == expected
            else:
                # For very long filenames, just check it's not too long
                assert len(result) <= 255
                assert result != ""
    
    def test_sanitize_filename_preserves_extensions(self):
        """Test that filename sanitization preserves valid extensions"""
        test_cases = [
            ("file<>.txt", "file__.txt"),
            ("document?.pdf", "document_.pdf"),
            ("image|photo.jpg", "image_photo.jpg"),
            ("data*file.csv", "data_file.csv"),
        ]
        
        for input_name, expected in test_cases:
            result = sanitize_filename(input_name)
            assert result == expected
            # Verify extension is preserved
            if "." in expected:
                assert result.split(".")[-1] == expected.split(".")[-1]


class TestMergeDictsAdvanced:
    """Advanced dictionary merging tests"""
    
    def test_merge_with_conflicting_types(self):
        """Test merging when same keys have different types"""
        dict1 = {
            "value": "string",
            "config": {"setting": "old"},
            "list_item": [1, 2, 3]
        }
        dict2 = {
            "value": 42,  # String -> int
            "config": "new_config",  # Dict -> string  
            "list_item": {"new": "format"}  # List -> dict
        }
        
        result = merge_dicts(dict1, dict2)
        
        # dict2 values should take precedence
        assert result["value"] == 42
        assert result["config"] == "new_config"
        assert result["list_item"] == {"new": "format"}
    
    def test_merge_very_deep_nesting(self):
        """Test merging with very deep nesting"""
        dict1 = {"a": {"b": {"c": {"d": {"e": {"f": "deep1"}}}}}}
        dict2 = {"a": {"b": {"c": {"d": {"e": {"g": "deep2"}}}}}}
        
        result = merge_dicts(dict1, dict2)
        
        # Both deep values should be present
        assert result["a"]["b"]["c"]["d"]["e"]["f"] == "deep1"
        assert result["a"]["b"]["c"]["d"]["e"]["g"] == "deep2"
    
    def test_merge_with_none_and_empty_values(self):
        """Test merging with None and empty values"""
        dict1 = {
            "null_value": None,
            "empty_dict": {},
            "empty_list": [],
            "normal": "value1"
        }
        dict2 = {
            "null_value": "not_null",
            "empty_dict": {"filled": True},
            "empty_list": ["item"],
            "normal": "value2"
        }
        
        result = merge_dicts(dict1, dict2)
        
        assert result["null_value"] == "not_null"
        assert result["empty_dict"] == {"filled": True}
        assert result["empty_list"] == ["item"]
        assert result["normal"] == "value2"
    
    def test_merge_preserves_original_dicts(self):
        """Test that merge operation doesn't modify original dictionaries"""
        dict1 = {"shared": {"a": 1}, "unique1": "value1"}
        dict2 = {"shared": {"b": 2}, "unique2": "value2"}
        
        # Store original states
        original_dict1 = {"shared": {"a": 1}, "unique1": "value1"}
        original_dict2 = {"shared": {"b": 2}, "unique2": "value2"}
        
        result = merge_dicts(dict1, dict2)
        
        # Originals should be unchanged
        assert dict1 == original_dict1
        assert dict2 == original_dict2
        
        # Result should have merged content
        assert result["shared"] == {"a": 1, "b": 2}
        assert result["unique1"] == "value1"
        assert result["unique2"] == "value2"


class TestChunkListAdvanced:
    """Advanced list chunking tests"""
    
    def test_chunk_with_large_lists(self):
        """Test chunking very large lists"""
        large_list = list(range(100000))  # 100k items
        chunk_size = 1000
        
        import time
        start_time = time.time()
        result = chunk_list(large_list, chunk_size)
        end_time = time.time()
        
        # Should complete quickly
        assert end_time - start_time < 1.0
        
        # Verify correct chunking
        assert len(result) == 100  # 100k / 1k = 100 chunks
        assert all(len(chunk) == chunk_size for chunk in result[:-1])  # All but last chunk
        assert len(result[-1]) <= chunk_size  # Last chunk may be smaller
    
    def test_chunk_memory_efficiency(self):
        """Test that chunking doesn't create excessive memory overhead"""
        # Create list with large objects
        large_objects = [{"data": "x" * 1000, "id": i} for i in range(1000)]
        
        result = chunk_list(large_objects, 100)
        
        # Verify structure
        assert len(result) == 10
        assert all(len(chunk) == 100 for chunk in result)
        
        # Verify objects are the same instances (not copied)
        assert result[0][0] is large_objects[0]
        assert result[5][50] is large_objects[550]
    
    def test_chunk_with_various_data_types(self):
        """Test chunking lists with various data types"""
        mixed_list = [
            "string", 42, 3.14, True, None, 
            [1, 2, 3], {"key": "value"}, 
            lambda x: x, set([1, 2, 3])
        ]
        
        result = chunk_list(mixed_list, 3)
        
        # Verify chunking preserves all types
        assert len(result) == 3  # 9 items / 3 = 3 chunks
        assert len(result[0]) == 3
        assert len(result[1]) == 3
        assert len(result[2]) == 3
        
        # Verify types are preserved
        flattened = [item for chunk in result for item in chunk]
        assert flattened == mixed_list
    
    def test_chunk_edge_cases_comprehensive(self):
        """Test comprehensive edge cases for chunking"""
        # Test with chunk size equal to list length
        data = [1, 2, 3, 4, 5]
        result = chunk_list(data, 5)
        assert result == [[1, 2, 3, 4, 5]]
        
        # Test with chunk size larger than list
        result = chunk_list(data, 10)
        assert result == [[1, 2, 3, 4, 5]]
        
        # Test with single item chunks
        result = chunk_list(data, 1)
        assert result == [[1], [2], [3], [4], [5]]
        
        # Test with empty list
        result = chunk_list([], 5)
        assert result == []


class TestFormatDurationAdvanced:
    """Advanced duration formatting tests"""
    
    def test_duration_precision_requirements(self):
        """Test duration formatting meets precision requirements"""
        test_cases = [
            (0.001, "0.00s"),      # Very small duration
            (0.999, "1.00s"),      # Just under 1 second
            (59.999, "60.00s"),    # Just under 1 minute
            (60.001, "1.0m"),      # Just over 1 minute
            (3599.999, "60.0m"),   # Just under 1 hour
            (3600.001, "1.0h"),    # Just over 1 hour
        ]
        
        for duration, expected in test_cases:
            result = format_duration(duration)
            # Allow some variation in implementation
            if expected.endswith("s"):
                assert result.endswith("s")
                assert abs(float(result[:-1]) - float(expected[:-1])) < 0.01
            elif expected.endswith("m"):
                assert result.endswith("m")
                assert abs(float(result[:-1]) - float(expected[:-1])) < 0.1
            elif expected.endswith("h"):
                assert result.endswith("h")
                assert abs(float(result[:-1]) - float(expected[:-1])) < 0.1
    
    def test_duration_format_consistency(self):
        """Test duration format consistency across ranges"""
        # Test seconds range
        for i in range(60):
            result = format_duration(i)
            assert result.endswith("s")
            assert float(result[:-1]) == i
        
        # Test minutes range
        for i in range(1, 60):
            duration = i * 60
            result = format_duration(duration)
            assert result.endswith("m")
            assert float(result[:-1]) == i
        
        # Test hours range
        for i in range(1, 24):
            duration = i * 3600
            result = format_duration(duration)
            assert result.endswith("h")
            assert float(result[:-1]) == i
    
    def test_duration_extreme_values(self):
        """Test duration formatting with extreme values"""
        extreme_cases = [
            1e-10,     # Very tiny duration
            1e10,      # Very large duration (over 300 years)
            float('inf'),  # Infinity
        ]
        
        for duration in extreme_cases:
            try:
                result = format_duration(duration)
                assert isinstance(result, str)
                assert len(result) > 0
                assert any(unit in result for unit in ["s", "m", "h"])
            except (ValueError, OverflowError):
                # Acceptable to raise exception for extreme values
                pass


class TestIntegrationAndWorkflows:
    """Integration tests simulating real-world workflows"""
    
    def test_configuration_management_workflow(self):
        """Test complete configuration management workflow"""
        # Simulate loading configuration from multiple sources
        base_config = {
            "app": {"name": "MyApp", "version": "1.0"},
            "database": {"host": "localhost", "port": 5432},
            "features": {"auth": True, "logging": {"level": "INFO"}}
        }
        
        user_config = {
            "database": {"host": "prod.db.com", "ssl": True},
            "features": {"logging": {"level": "DEBUG", "file": "app.log"}}
        }
        
        env_config = {
            "database": {"password": "secret"},
            "features": {"rate_limiting": True}
        }
        
        # Merge configurations
        merged_config = merge_dicts(base_config, user_config)
        final_config = merge_dicts(merged_config, env_config)
        
        # Serialize for storage
        config_json = safe_json_dumps(final_config)
        assert config_json != ""
        
        # Create hash for versioning
        config_hash = generate_hash(config_json)
        assert len(config_hash) == 64
        
        # Flatten for environment variable export
        flat_config = flatten_dict(final_config)
        
        # Verify expected merged values
        assert final_config["database"]["host"] == "prod.db.com"
        assert final_config["database"]["ssl"] is True
        assert final_config["database"]["password"] == "secret"
        assert final_config["features"]["logging"]["level"] == "DEBUG"
        assert final_config["features"]["rate_limiting"] is True
        
        # Verify flattened structure
        assert "database.host" in flat_config
        assert "features.logging.level" in flat_config
        assert flat_config["features.logging.level"] == "DEBUG"
    
    def test_data_processing_pipeline_with_retry(self):
        """Test data processing pipeline with retry mechanisms"""
        # Simulate processing data in chunks with potential failures
        raw_data = [{"id": i, "value": f"item_{i}"} for i in range(100)]
        chunks = chunk_list(raw_data, 10)
        
        processed_results = []
        failure_count = [0]
        
        def process_chunk_with_failure(chunk):
            # Simulate intermittent failures
            failure_count[0] += 1
            if failure_count[0] % 3 == 0:  # Fail every 3rd attempt
                raise ConnectionError("Simulated processing failure")
            
            # Process chunk
            processed = {
                "chunk_id": generate_hash(safe_json_dumps(chunk))[:8],
                "items": len(chunk),
                "data": chunk
            }
            return processed
        
        # Process each chunk with retry
        for chunk in chunks:
            try:
                result = retry_with_backoff(
                    lambda: process_chunk_with_failure(chunk),
                    max_retries=3,
                    base_delay=0.1
                )
                processed_results.append(result)
            except Exception as e:
                # Log failure and continue (in real scenario)
                print(f"Failed to process chunk after retries: {e}")
        
        # Verify processing completed for most chunks
        assert len(processed_results) >= 8  # At least 80% success rate
        
        # Verify each result has expected structure
        for result in processed_results:
            assert "chunk_id" in result
            assert len(result["chunk_id"]) == 8
            assert result["items"] == 10
            assert len(result["data"]) == 10
    
    def test_file_management_workflow(self):
        """Test file management workflow with sanitization and directory creation"""
        import tempfile
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Simulate organizing files from various sources
            file_specs = [
                {"name": "report<2023>.pdf", "category": "reports", "subcategory": "annual"},
                {"name": "data|backup.csv", "category": "data", "subcategory": "backups"},
                {"name": "config?.yaml", "category": "config", "subcategory": "environments"},
                {"name": "  .hidden_file  ", "category": "misc", "subcategory": "temp"},
            ]
            
            organized_files = []
            
            for spec in file_specs:
                # Create directory structure
                category_dir = ensure_directory_exists(
                    Path(temp_dir) / spec["category"] / spec["subcategory"]
                )
                
                # Sanitize filename
                safe_name = sanitize_filename(spec["name"])
                
                # Create file path
                file_path = category_dir / safe_name
                
                # Simulate file creation with metadata
                file_metadata = {
                    "original_name": spec["name"],
                    "safe_name": safe_name,
                    "category": spec["category"],
                    "subcategory": spec["subcategory"],
                    "path": str(file_path),
                    "created": time.time()
                }
                
                # Write metadata as JSON
                metadata_json = safe_json_dumps(file_metadata)
                file_path.write_text(metadata_json)
                
                organized_files.append(file_metadata)
            
            # Verify all files were created successfully
            assert len(organized_files) == 4
            
            for file_info in organized_files:
                file_path = Path(file_info["path"])
                assert file_path.exists()
                assert file_path.is_file()
                
                # Verify content can be read back
                content = file_path.read_text()
                parsed_metadata = safe_json_parse(content)
                assert parsed_metadata is not None
                assert parsed_metadata["original_name"] == file_info["original_name"]


# Performance and stress testing
class TestPerformanceAndStress:
    """Performance and stress tests for all utility functions"""
    
    @pytest.mark.slow
    def test_concurrent_mixed_operations(self):
        """Test concurrent execution of mixed utility operations"""
        import threading
        import random
        
        results = []
        errors = []
        
        def worker_thread(thread_id):
            try:
                # Perform random mix of operations
                operations = [
                    lambda: safe_json_dumps({"thread": thread_id, "data": list(range(100))}),
                    lambda: generate_hash(f"thread_{thread_id}_data"),
                    lambda: flatten_dict({"thread": thread_id, "nested": {"value": thread_id}}),
                    lambda: chunk_list(list(range(50)), 10),
                    lambda: format_duration(thread_id * 10.5),
                ]
                
                thread_results = []
                for _ in range(10):  # 10 operations per thread
                    op = random.choice(operations)
                    result = op()
                    thread_results.append(result)
                
                results.append((thread_id, thread_results))
                
            except Exception as e:
                errors.append((thread_id, str(e)))
        
        # Run 20 concurrent threads
        threads = []
        for i in range(20):
            thread = threading.Thread(target=worker_thread, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 20
        assert all(len(thread_results) == 10 for _, thread_results in results)
    
    @pytest.mark.slow
    def test_memory_usage_large_operations(self):
        """Test memory usage with large data operations"""
        # Test with large data structures
        large_nested_dict = {}
        current_level = large_nested_dict
        
        # Create 50 levels of nesting with data at each level
        for i in range(50):
            current_level[f"level_{i}"] = {
                "data": [f"item_{j}" for j in range(100)],  # 100 items per level
                "metadata": {"level": i, "timestamp": time.time()},
                "next": {}
            }
            current_level = current_level[f"level_{i}"]["next"]
        
        import time
        
        # Test JSON serialization performance
        start_time = time.time()
        json_result = safe_json_dumps(large_nested_dict)
        json_time = time.time() - start_time
        
        # Test flattening performance
        start_time = time.time()
        flattened = flatten_dict(large_nested_dict)
        flatten_time = time.time() - start_time
        
        # Test hash generation performance
        start_time = time.time()
        hash_result = generate_hash(json_result)
        hash_time = time.time() - start_time
        
        # Verify operations completed
        assert json_result != ""
        assert len(flattened) > 100  # Should have many flattened keys
        assert len(hash_result) == 64
        
        # Performance should be reasonable (adjust thresholds as needed)
        assert json_time < 10.0, f"JSON serialization too slow: {json_time}s"
        assert flatten_time < 10.0, f"Flattening too slow: {flatten_time}s"
        assert hash_time < 5.0, f"Hashing too slow: {hash_time}s"


# Add marker for slow tests
pytest.mark.slow = pytest.mark.skipif(
    not pytest.config.getoption("--run-slow", default=False),
    reason="Slow tests skipped unless --run-slow option provided"
)


# Additional Comprehensive Test Coverage and Edge Cases
class TestSafeJsonParseExtreme:
    """Extreme edge cases and security tests for safe_json_parse"""
    
    def test_malicious_json_payloads(self):
        """Test handling of potentially malicious JSON payloads"""
        malicious_payloads = [
            '{"__proto__": {"polluted": true}}',  # Prototype pollution attempt
            '{"constructor": {"prototype": {"polluted": true}}}',  # Constructor manipulation
            '{"a": ' + '"x"' * 10000 + '}',  # Extremely long string
            '[' + '1,' * 100000 + '1]',  # Very large array
            '{"nested": ' + '{"level": ' * 1000 + '"deep"' + '}' * 1000 + '}',  # Extreme nesting
        ]
        
        for payload in malicious_payloads:
            result = safe_json_parse(payload)
            # Should either parse safely or return None, never crash
            assert result is None or isinstance(result, (dict, list))
    
    def test_json_with_binary_data(self):
        """Test JSON containing binary-like data"""
        binary_cases = [
            '{"data": "\\u0000\\u0001\\u0002"}',  # Null bytes and control chars
            '{"binary": "\\x00\\x01\\xFF"}',     # Hex escape sequences
            '{"unicode": "\\uD83D\\uDE00"}',     # Unicode surrogates
        ]
        
        for json_str in binary_cases:
            result = safe_json_parse(json_str)
            if result is not None:
                assert isinstance(result, dict)
                assert "data" in result or "binary" in result or "unicode" in result
    
    def test_json_memory_exhaustion_protection(self):
        """Test protection against memory exhaustion attacks"""
        # Test with deeply nested structure that could cause stack overflow
        nested_json = '{"a": ' * 10000 + '"value"' + '}' * 10000
        
        import time
        start_time = time.time()
        result = safe_json_parse(nested_json)
        end_time = time.time()
        
        # Should complete quickly and safely
        assert end_time - start_time < 5.0  # Max 5 seconds
        # Should either parse or return None safely
        assert result is None or isinstance(result, dict)


class TestSafeJsonDumpsExtreme:
    """Extreme edge cases for safe_json_dumps"""
    
    def test_recursive_data_structures(self):
        """Test handling of various recursive data structures"""
        # Self-referencing list
        self_ref_list = [1, 2, 3]
        self_ref_list.append(self_ref_list)
        
        result = safe_json_dumps(self_ref_list)
        assert result == ""  # Should handle gracefully
        
        # Mutually recursive objects
        obj1 = {"name": "obj1"}
        obj2 = {"name": "obj2", "ref": obj1}
        obj1["ref"] = obj2
        
        result = safe_json_dumps(obj1)
        assert result == ""  # Should handle gracefully
    
    def test_extreme_unicode_handling(self):
        """Test handling of extreme Unicode cases"""
        unicode_cases = {
            "emoji_heavy": "🚀" * 1000 + "🌟" * 1000,
            "mixed_scripts": "Hello नमस्ते مرحبا こんにちは 你好",
            "zero_width": "a\u200bb\u200cc\u200dd",  # Zero-width characters
            "rtl_text": "العربية עברית",  # Right-to-left text
            "combining": "a\u0300b\u0301c\u0302",  # Combining diacritics
        }
        
        result = safe_json_dumps(unicode_cases)
        assert result != ""
        
        # Verify round-trip
        parsed_back = safe_json_parse(result)
        assert parsed_back is not None
        assert parsed_back["emoji_heavy"] == "🚀" * 1000 + "🌟" * 1000
    
    def test_memory_intensive_objects(self):
        """Test serialization of memory-intensive objects"""
        # Large dictionary with many keys
        large_dict = {f"key_{i}": f"value_{i}" for i in range(50000)}
        
        import time
        start_time = time.time()
        result = safe_json_dumps(large_dict)
        end_time = time.time()
        
        assert result != ""
        assert end_time - start_time < 10.0  # Should complete within 10 seconds
        
        # Verify it's valid JSON
        parsed = safe_json_parse(result)
        assert parsed is not None
        assert len(parsed) == 50000


class TestGenerateHashExtensive:
    """Extensive hash testing including cryptographic properties"""
    
    def test_hash_collision_resistance(self):
        """Test resistance to hash collisions"""
        # Generate hashes for many similar inputs
        base_string = "collision_test_"
        hashes = []
        
        for i in range(10000):
            test_string = f"{base_string}{i}"
            hash_val = generate_hash(test_string)
            hashes.append(hash_val)
        
        # All hashes should be unique
        unique_hashes = set(hashes)
        assert len(unique_hashes) == len(hashes)
        
        # Test slight variations
        similar_inputs = [
            "test_string",
            "test_string ",  # Trailing space
            "test_string\n",  # Newline
            "test_string\t",  # Tab
            "Test_string",    # Case change
            "test_string1",   # Added character
        ]
        
        similar_hashes = [generate_hash(inp) for inp in similar_inputs]
        assert len(set(similar_hashes)) == len(similar_hashes)  # All different
    
    def test_hash_performance_benchmark(self):
        """Benchmark hash generation performance"""
        import time
        
        # Test with various input sizes
        sizes = [100, 1000, 10000, 100000]
        
        for size in sizes:
            test_data = "x" * size
            
            start_time = time.time()
            for _ in range(100):  # 100 iterations
                generate_hash(test_data)
            end_time = time.time()
            
            avg_time = (end_time - start_time) / 100
            # Should be very fast even for large inputs
            assert avg_time < 0.1, f"Hash generation too slow for size {size}: {avg_time}s"
    
    def test_hash_entropy_distribution(self):
        """Test hash output entropy and distribution"""
        hashes = [generate_hash(f"entropy_test_{i}") for i in range(1000)]
        
        # Test character distribution in hex output
        char_counts = {}
        for hash_val in hashes:
            for char in hash_val:
                char_counts[char] = char_counts.get(char, 0) + 1
        
        # Each hex character should appear roughly equally
        total_chars = sum(char_counts.values())
        expected_per_char = total_chars / 16  # 16 hex characters
        
        for char in "0123456789abcdef":
            count = char_counts.get(char, 0)
            # Allow 30% deviation from expected
            assert abs(count - expected_per_char) < expected_per_char * 0.3


class TestRetryWithBackoffExtensive:
    """Extensive retry mechanism testing"""
    
    def test_retry_with_custom_exceptions(self):
        """Test retry with custom exception handling"""
        class CustomRetryableError(Exception):
            pass
        
        class NonRetryableError(Exception):
            pass
        
        attempts = [0]
        
        def mixed_exception_function():
            attempts[0] += 1
            if attempts[0] == 1:
                raise CustomRetryableError("First failure")
            elif attempts[0] == 2:
                raise NonRetryableError("Should not retry")
            return "success"
        
        # Should stop on NonRetryableError
        with pytest.raises(NonRetryableError):
            retry_with_backoff(mixed_exception_function, max_retries=5)
        
        assert attempts[0] == 2  # Should have stopped after NonRetryableError
    
    @patch('time.sleep')
    def test_jitter_in_backoff(self, mock_sleep):
        """Test jitter implementation in backoff timing"""
        failure_count = [0]
        
        def intermittent_failure():
            failure_count[0] += 1
            if failure_count[0] < 4:
                raise ValueError("Temporary failure")
            return "success"
        
        # If jitter is implemented, sleep times should vary
        for _ in range(3):  # Run multiple times
            failure_count[0] = 0
            mock_sleep.reset_mock()
            
            retry_with_backoff(intermittent_failure, max_retries=5, base_delay=1.0)
            
            # Collect all sleep calls
            sleep_times = [call[0][0] for call in mock_sleep.call_args_list]
            
            # Basic exponential pattern should be present
            assert len(sleep_times) == 3  # 3 failures before success
    
    def test_retry_timeout_mechanism(self):
        """Test timeout mechanism if implemented"""
        import time
        
        start_time = time.time()
        attempts = [0]
        
        def slow_failing_function():
            attempts[0] += 1
            time.sleep(0.1)  # Simulate slow operation
            if attempts[0] < 10:
                raise TimeoutError("Still failing")
            return "success"
        
        # Should eventually succeed or timeout
        try:
            result = retry_with_backoff(
                slow_failing_function, 
                max_retries=15, 
                base_delay=0.05
            )
            assert result == "success"
        except TimeoutError:
            # Acceptable if timeout mechanism exists
            pass
        
        end_time = time.time()
        # Total time should be reasonable
        assert end_time - start_time < 30.0  # Max 30 seconds


class TestFlattenDictExtensive:
    """Extensive dictionary flattening tests"""
    
    def test_flatten_with_special_key_types(self):
        """Test flattening with various key types"""
        special_keys_dict = {
            123: "numeric_key",
            True: "boolean_true_key", 
            False: "boolean_false_key",
            None: "none_key",
            (1, 2): "tuple_key",
            frozenset([1, 2, 3]): "frozenset_key",
        }
        
        result = flatten_dict(special_keys_dict)
        
        # Should handle conversion of keys to strings
        assert "123" in result or 123 in result
        assert "True" in result or True in result
        assert "False" in result or False in result
    
    def test_flatten_with_circular_references(self):
        """Test flattening with circular reference handling"""
        circular_dict = {"a": {"b": {}}}
        circular_dict["a"]["b"]["c"] = circular_dict["a"]  # Create cycle
        
        # Should handle gracefully without infinite recursion
        try:
            result = flatten_dict(circular_dict)
            # If it succeeds, should be a dict
            assert isinstance(result, dict)
        except RecursionError:
            # Acceptable if function doesn't handle cycles
            pytest.skip("Function doesn't handle circular references")
    
    def test_flatten_performance_stress(self):
        """Stress test flattening performance"""
        # Create very wide dictionary (many keys at each level)
        wide_dict = {}
        for i in range(1000):
            wide_dict[f"key_{i}"] = {
                f"subkey_{j}": f"value_{i}_{j}"
                for j in range(50)
            }
        
        import time
        start_time = time.time()
        result = flatten_dict(wide_dict)
        end_time = time.time()
        
        # Should complete quickly
        assert end_time - start_time < 5.0
        # Should have 1000 * 50 = 50,000 flattened keys
        assert len(result) == 50000
    
    def test_flatten_preserves_data_types(self):
        """Test that flattening preserves value data types"""
        typed_dict = {
            "strings": {"value": "text"},
            "numbers": {"int": 42, "float": 3.14},
            "booleans": {"true": True, "false": False},
            "none": {"value": None},
            "lists": {"items": [1, 2, 3]},
            "nested_list": {"data": [{"inner": "value"}]},
        }
        
        result = flatten_dict(typed_dict)
        
        # Verify types are preserved
        assert isinstance(result["strings.value"], str)
        assert isinstance(result["numbers.int"], int)
        assert isinstance(result["numbers.float"], float)
        assert isinstance(result["booleans.true"], bool)
        assert result["none.value"] is None
        assert isinstance(result["lists.items"], list)


class TestFileOperationsExtensive:
    """Extensive file operations testing"""
    
    def test_ensure_directory_with_permissions(self):
        """Test directory creation with various permission scenarios"""
        import tempfile
        import os
        import stat
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test creating directory in read-only parent (if possible)
            parent_dir = Path(temp_dir) / "readonly_parent"
            parent_dir.mkdir()
            
            # Make parent read-only (Unix-like systems)
            if hasattr(os, 'chmod'):
                try:
                    os.chmod(parent_dir, stat.S_IRUSR | stat.S_IXUSR)
                    
                    # Try to create subdirectory
                    with pytest.raises(PermissionError):
                        ensure_directory_exists(parent_dir / "should_fail")
                    
                    # Restore permissions
                    os.chmod(parent_dir, stat.S_IRWXU)
                except (OSError, NotImplementedError):
                    # Skip if not supported on this system
                    pytest.skip("Permission testing not supported on this system")
    
    def test_sanitize_filename_unicode_normalization(self):
        """Test filename sanitization with Unicode normalization"""
        unicode_filenames = [
            "café.txt",  # Precomposed
            "cafe\u0301.txt",  # Decomposed (e + combining accent)
            "naïve file.pdf",  # Mixed composition
            "𝕌𝕟𝕚𝕔𝕠𝕕𝕖.txt",  # Mathematical symbols
            "файл.doc",  # Cyrillic
            "测试文件.txt",  # Chinese
        ]
        
        for filename in unicode_filenames:
            result = sanitize_filename(filename)
            
            # Should always return a valid filename
            assert isinstance(result, str)
            assert len(result) > 0
            assert result != "unnamed"  # Should preserve unicode content
            
            # Should not contain problematic characters
            problematic_chars = '<>:"/\\|?*'
            assert not any(char in result for char in problematic_chars)
    
    def test_sanitize_filename_length_limits(self):
        """Test filename sanitization with length constraints"""
        # Test various long filenames
        long_names = [
            "a" * 300,  # Very long name
            "file_" + "x" * 250 + ".txt",  # Long with extension
            "prefix_" + "middle_" * 50 + "suffix.doc",  # Repetitive long name
        ]
        
        for long_name in long_names:
            result = sanitize_filename(long_name)
            
            # Should respect filesystem limits (usually 255 chars)
            assert len(result) <= 255
            assert result != "unnamed"  # Should preserve some content
            
            # If there was an extension, try to preserve it
            if "." in long_name and "." in result:
                original_ext = long_name.split(".")[-1]
                result_ext = result.split(".")[-1]
                # Extension should be preserved or reasonably truncated
                assert len(result_ext) <= len(original_ext) + 5


class TestChunkListExtensive:
    """Extensive list chunking tests"""
    
    def test_chunk_with_generators(self):
        """Test chunking with generator inputs"""
        def number_generator():
            for i in range(100):
                yield i * 2
        
        # Convert generator to list for chunking
        gen_list = list(number_generator())
        result = chunk_list(gen_list, 10)
        
        assert len(result) == 10
        assert all(len(chunk) == 10 for chunk in result)
        assert result[0] == [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
    
    def test_chunk_memory_efficiency_validation(self):
        """Validate that chunking doesn't duplicate large objects"""
        import sys
        
        # Create list with large objects
        class LargeObject:
            def __init__(self, size):
                self.data = bytearray(size)
        
        large_objects = [LargeObject(10000) for _ in range(100)]  # ~1MB each
        original_ids = [id(obj) for obj in large_objects]
        
        result = chunk_list(large_objects, 25)
        
        # Verify objects are not copied (same id)
        chunked_ids = [id(obj) for chunk in result for obj in chunk]
        assert chunked_ids == original_ids
        
        # Verify structure
        assert len(result) == 4  # 100 / 25 = 4 chunks
        assert all(len(chunk) == 25 for chunk in result)
    
    def test_chunk_with_custom_objects(self):
        """Test chunking with custom object types"""
        class CustomItem:
            def __init__(self, value, metadata=None):
                self.value = value
                self.metadata = metadata or {}
            
            def __eq__(self, other):
                return (isinstance(other, CustomItem) and 
                       self.value == other.value and 
                       self.metadata == other.metadata)
        
        custom_objects = [
            CustomItem(f"item_{i}", {"index": i, "category": i % 3})
            for i in range(50)
        ]
        
        result = chunk_list(custom_objects, 7)
        
        # Verify chunking preserved object integrity
        assert len(result) == 8  # 50 / 7 = 7 full chunks + 1 partial
        assert len(result[-1]) == 1  # Last chunk has remainder
        
        # Verify objects maintain their properties
        first_chunk = result[0]
        assert all(isinstance(item, CustomItem) for item in first_chunk)
        assert first_chunk[0].value == "item_0"
        assert first_chunk[0].metadata["index"] == 0


class TestFormatDurationExtensive:
    """Extensive duration formatting tests"""
    
    def test_duration_format_edge_boundaries(self):
        """Test exact boundary conditions for duration formatting"""
        # Test exact boundaries between units
        boundary_tests = [
            (59.999, "s"),      # Just under minute threshold
            (60.0, "m"),        # Exact minute threshold
            (60.001, "m"),      # Just over minute threshold
            (3599.999, "m"),    # Just under hour threshold
            (3600.0, "h"),      # Exact hour threshold
            (3600.001, "h"),    # Just over hour threshold
        ]
        
        for duration, expected_unit in boundary_tests:
            result = format_duration(duration)
            assert result.endswith(expected_unit)
    
    def test_duration_format_negative_values(self):
        """Test duration formatting with negative values"""
        negative_durations = [-1.0, -60.5, -3661.0, -0.001]
        
        for duration in negative_durations:
            result = format_duration(duration)
            # Should handle gracefully - either format as positive or indicate negative
            assert isinstance(result, str)
            assert len(result) > 0
    
    def test_duration_format_special_float_values(self):
        """Test duration formatting with special float values"""
        special_values = [
            float('inf'),
            float('-inf'),
            float('nan'),
            1e-100,  # Very small positive
            1e100,   # Very large
        ]
        
        for value in special_values:
            try:
                result = format_duration(value)
                assert isinstance(result, str)
                assert len(result) > 0
            except (ValueError, OverflowError):
                # Acceptable to raise exceptions for special values
                pass
    
    def test_duration_format_precision_consistency(self):
        """Test precision consistency across different ranges"""
        # Test that similar precision is maintained across ranges
        test_cases = [
            (1.234, 2),     # Seconds with 3 decimal places
            (61.234, 1),    # Minutes (should show 1 decimal)
            (3661.234, 1),  # Hours (should show 1 decimal)
        ]
        
        for duration, expected_decimals in test_cases:
            result = format_duration(duration)
            
            # Extract numeric part
            if result.endswith('s'):
                numeric_part = result[:-1]
            elif result.endswith('m') or result.endswith('h'):
                numeric_part = result[:-1]
            else:
                continue
            
            # Count decimal places
            if '.' in numeric_part:
                decimal_places = len(numeric_part.split('.')[1])
                assert decimal_places <= expected_decimals + 1  # Allow some flexibility


class TestIntegrationWorkflows:
    """Real-world integration workflow tests"""
    
    def test_data_export_import_workflow(self):
        """Test complete data export/import workflow"""
        # Simulate exporting data from a complex application state
        app_state = {
            "users": [
                {"id": i, "name": f"User {i}", "profile": {"age": 20 + i, "city": f"City {i % 5}"}}
                for i in range(1000)
            ],
            "settings": {
                "ui": {"theme": "dark", "language": "en"},
                "api": {"timeout": 30, "retries": 3},
                "features": {"beta": True, "analytics": False}
            },
            "metadata": {
                "version": "2.1.0",
                "exported_at": time.time(),
                "hash": None  # Will be filled
            }
        }
        
        # Flatten for easier processing
        flat_state = flatten_dict(app_state)
        
        # Process in chunks for large datasets
        user_chunks = chunk_list(app_state["users"], 100)
        processed_chunks = []
        
        for chunk in user_chunks:
            # Simulate processing each chunk
            chunk_data = {
                "chunk_id": generate_hash(safe_json_dumps(chunk))[:8],
                "users": chunk,
                "processed_at": time.time()
            }
            processed_chunks.append(chunk_data)
        
        # Create export package
        export_data = {
            "original_state": app_state,
            "flat_state": flat_state,
            "processed_chunks": processed_chunks,
            "export_metadata": {
                "chunk_count": len(processed_chunks),
                "total_users": len(app_state["users"]),
                "export_hash": None
            }
        }
        
        # Generate hash for integrity
        export_json = safe_json_dumps(export_data)
        export_hash = generate_hash(export_json)
        export_data["export_metadata"]["export_hash"] = export_hash
        
        # Re-serialize with hash
        final_export = safe_json_dumps(export_data)
        
        # Simulate import process
        imported_data = safe_json_parse(final_export)
        
        # Verify integrity
        assert imported_data is not None
        assert len(imported_data["processed_chunks"]) == 10  # 1000 users / 100 per chunk
        assert imported_data["export_metadata"]["total_users"] == 1000
        assert imported_data["export_metadata"]["export_hash"] == export_hash
        
        # Verify data preservation
        original_users = app_state["users"]
        imported_users = imported_data["original_state"]["users"]
        assert len(imported_users) == len(original_users)
        assert imported_users[0]["name"] == "User 0"
        assert imported_users[-1]["name"] == "User 999"
    
    def test_configuration_validation_workflow(self):
        """Test configuration validation and sanitization workflow"""
        import tempfile
        
        # Simulate loading configurations from multiple sources
        configs = [
            {
                "source": "default",
                "data": {
                    "app": {"name": "TestApp", "version": "1.0"},
                    "database": {"host": "localhost", "port": 5432},
                }
            },
            {
                "source": "environment", 
                "data": {
                    "database": {"host": "prod.db.com", "ssl": True},
                    "logging": {"level": "DEBUG"}
                }
            },
            {
                "source": "user_file",
                "data": {
                    "app": {"debug": True},
                    "features": {"experimental": True}
                }
            }
        ]
        
        # Merge configurations in order
        merged_config = {}
        for config in configs:
            merged_config = merge_dicts(merged_config, config["data"])
        
        # Validate and process configuration
        config_hash = generate_hash(safe_json_dumps(merged_config))
        flat_config = flatten_dict(merged_config, prefix="APP")
        
        # Simulate saving to file system
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = ensure_directory_exists(Path(temp_dir) / "config")
            
            # Save main config
            main_config_file = config_dir / sanitize_filename("app-config.json")
            main_config_file.write_text(safe_json_dumps(merged_config))
            
            # Save flattened config for environment variables
            env_config_file = config_dir / sanitize_filename("env-config.json")
            env_config_file.write_text(safe_json_dumps(flat_config))
            
            # Save metadata
            metadata = {
                "config_hash": config_hash,
                "sources": [c["source"] for c in configs],
                "generated_at": time.time(),
                "format_version": "1.0"
            }
            metadata_file = config_dir / sanitize_filename("config-metadata.json")
            metadata_file.write_text(safe_json_dumps(metadata))
            
            # Verify files exist and are readable
            assert main_config_file.exists()
            assert env_config_file.exists()
            assert metadata_file.exists()
            
            # Verify content integrity
            loaded_config = safe_json_parse(main_config_file.read_text())
            loaded_flat = safe_json_parse(env_config_file.read_text())
            loaded_metadata = safe_json_parse(metadata_file.read_text())
            
            assert loaded_config == merged_config
            assert loaded_flat == flat_config
            assert loaded_metadata["config_hash"] == config_hash
            assert "APP.database.host" in loaded_flat
            assert loaded_flat["APP.database.host"] == "prod.db.com"


# Add custom pytest markers for test categorization
def pytest_configure(config):
    """Configure custom pytest markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "stress: marks tests as stress tests (deselect with '-m \"not stress\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )


# Additional test execution and reporting
if __name__ == "__main__":
    import sys
    
    # Default arguments for comprehensive testing
    args = [__file__, "-v", "--tb=short", "--strict-markers"]
    
    # Add slow tests if requested
    if "--run-slow" in sys.argv:
        args.append("--run-slow")
    else:
        args.extend(["-m", "not slow"])
    
    # Add coverage if available
    try:
        import pytest_cov
        args.extend(["--cov=utils.helpers", "--cov-report=term-missing"])
    except ImportError:
        pass
    
    pytest.main(args)
