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