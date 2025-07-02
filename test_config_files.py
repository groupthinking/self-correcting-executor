import pytest
import json
import yaml
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock
import configparser
from io import StringIO


class TestConfigFileValidation:
    """Test suite for validating configuration files."""
    
    @pytest.fixture
    def temp_config_dir(self):
        """Create a temporary directory for test config files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def sample_json_config(self):
        """Sample JSON configuration for testing."""
        return {
            "database": {
                "host": "localhost",
                "port": 5432,
                "name": "testdb"
            },
            "api": {
                "base_url": "https://api.example.com",
                "timeout": 30,
                "retries": 3
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        }
    
    @pytest.fixture
    def sample_yaml_config(self):
        """Sample YAML configuration for testing."""
        return """
        database:
          host: localhost
          port: 5432
          name: testdb
        api:
          base_url: https://api.example.com
          timeout: 30
          retries: 3
        logging:
          level: INFO
          format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        """
    
    @pytest.fixture
    def sample_ini_config(self):
        """Sample INI configuration for testing."""
        return """
        [database]
        host = localhost
        port = 5432
        name = testdb
        
        [api]
        base_url = https://api.example.com
        timeout = 30
        retries = 3
        
        [logging]
        level = INFO
        format = %%(asctime)s - %%(name)s - %%(levelname)s - %%(message)s
        """


class TestJSONConfigFiles:
    """Test JSON configuration file handling."""
    
    def test_valid_json_config_loading(self, temp_config_dir, sample_json_config):
        """Test loading a valid JSON configuration file."""
        config_file = temp_config_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(sample_json_config, f)
        
        with open(config_file, 'r') as f:
            loaded_config = json.load(f)
        
        assert loaded_config == sample_json_config
        assert loaded_config["database"]["host"] == "localhost"
        assert loaded_config["database"]["port"] == 5432
        assert loaded_config["api"]["timeout"] == 30
    
    def test_invalid_json_config_syntax(self, temp_config_dir):
        """Test handling of invalid JSON syntax."""
        config_file = temp_config_dir / "invalid.json"
        with open(config_file, 'w') as f:
            f.write('{"key": value}')  # Missing quotes around value
        
        with pytest.raises(json.JSONDecodeError):
            with open(config_file, 'r') as f:
                json.load(f)
    
    def test_empty_json_config(self, temp_config_dir):
        """Test handling of empty JSON configuration."""
        config_file = temp_config_dir / "empty.json"
        with open(config_file, 'w') as f:
            f.write('{}')
        
        with open(config_file, 'r') as f:
            loaded_config = json.load(f)
        
        assert loaded_config == {}
    
    def test_json_config_schema_validation(self, temp_config_dir):
        """Test JSON configuration schema validation."""
        # Test missing required keys
        incomplete_config = {"database": {"host": "localhost"}}
        config_file = temp_config_dir / "incomplete.json"
        
        with open(config_file, 'w') as f:
            json.dump(incomplete_config, f)
        
        with open(config_file, 'r') as f:
            loaded_config = json.load(f)
        
        # Validate required keys are present
        assert "database" in loaded_config
        assert "host" in loaded_config["database"]
        
        # Check for missing keys
        with pytest.raises(KeyError):
            _ = loaded_config["api"]["base_url"]
    
    def test_json_config_data_types(self, temp_config_dir):
        """Test JSON configuration data type validation."""
        config_with_types = {
            "string_value": "test",
            "integer_value": 42,
            "float_value": 3.14,
            "boolean_value": True,
            "list_value": [1, 2, 3],
            "null_value": None
        }
        
        config_file = temp_config_dir / "types.json"
        with open(config_file, 'w') as f:
            json.dump(config_with_types, f)
        
        with open(config_file, 'r') as f:
            loaded_config = json.load(f)
        
        assert isinstance(loaded_config["string_value"], str)
        assert isinstance(loaded_config["integer_value"], int)
        assert isinstance(loaded_config["float_value"], float)
        assert isinstance(loaded_config["boolean_value"], bool)
        assert isinstance(loaded_config["list_value"], list)
        assert loaded_config["null_value"] is None


class TestYAMLConfigFiles:
    """Test YAML configuration file handling."""
    
    def test_valid_yaml_config_loading(self, temp_config_dir, sample_yaml_config):
        """Test loading a valid YAML configuration file."""
        config_file = temp_config_dir / "config.yaml"
        with open(config_file, 'w') as f:
            f.write(sample_yaml_config)
        
        with open(config_file, 'r') as f:
            loaded_config = yaml.safe_load(f)
        
        assert loaded_config["database"]["host"] == "localhost"
        assert loaded_config["database"]["port"] == 5432
        assert loaded_config["api"]["timeout"] == 30
    
    def test_invalid_yaml_syntax(self, temp_config_dir):
        """Test handling of invalid YAML syntax."""
        config_file = temp_config_dir / "invalid.yaml"
        with open(config_file, 'w') as f:
            f.write('key: value\n  invalid_indent: value')
        
        with pytest.raises(yaml.YAMLError):
            with open(config_file, 'r') as f:
                yaml.safe_load(f)
    
    def test_yaml_config_with_references(self, temp_config_dir):
        """Test YAML configuration with references and anchors."""
        yaml_with_refs = """
        defaults: &defaults
          timeout: 30
          retries: 3
        
        production:
          <<: *defaults
          host: prod.example.com
        
        development:
          <<: *defaults
          host: dev.example.com
        """
        
        config_file = temp_config_dir / "refs.yaml"
        with open(config_file, 'w') as f:
            f.write(yaml_with_refs)
        
        with open(config_file, 'r') as f:
            loaded_config = yaml.safe_load(f)
        
        assert loaded_config["production"]["timeout"] == 30
        assert loaded_config["development"]["timeout"] == 30
        assert loaded_config["production"]["host"] == "prod.example.com"
        assert loaded_config["development"]["host"] == "dev.example.com"
    
    def test_empty_yaml_config(self, temp_config_dir):
        """Test handling of empty YAML configuration."""
        config_file = temp_config_dir / "empty.yaml"
        with open(config_file, 'w') as f:
            f.write('')
        
        with open(config_file, 'r') as f:
            loaded_config = yaml.safe_load(f)
        
        assert loaded_config is None


class TestINIConfigFiles:
    """Test INI configuration file handling."""
    
    def test_valid_ini_config_loading(self, temp_config_dir, sample_ini_config):
        """Test loading a valid INI configuration file."""
        config_file = temp_config_dir / "config.ini"
        with open(config_file, 'w') as f:
            f.write(sample_ini_config)
        
        config = configparser.ConfigParser()
        config.read(config_file)
        
        assert config.get('database', 'host') == "localhost"
        assert config.getint('database', 'port') == 5432
        assert config.get('api', 'base_url') == "https://api.example.com"
    
    def test_ini_config_missing_section(self, temp_config_dir):
        """Test handling of missing section in INI file."""
        config_file = temp_config_dir / "missing_section.ini"
        with open(config_file, 'w') as f:
            f.write("[database]\nhost = localhost\n")
        
        config = configparser.ConfigParser()
        config.read(config_file)
        
        assert config.has_section('database')
        assert not config.has_section('api')
        
        with pytest.raises(configparser.NoSectionError):
            config.get('api', 'base_url')
    
    def test_ini_config_missing_option(self, temp_config_dir):
        """Test handling of missing option in INI file."""
        config_file = temp_config_dir / "missing_option.ini"
        with open(config_file, 'w') as f:
            f.write("[database]\nhost = localhost\n")
        
        config = configparser.ConfigParser()
        config.read(config_file)
        
        assert config.has_option('database', 'host')
        assert not config.has_option('database', 'port')
        
        with pytest.raises(configparser.NoOptionError):
            config.get('database', 'port')
    
    def test_ini_config_interpolation(self, temp_config_dir):
        """Test INI configuration value interpolation."""
        ini_with_interpolation = """
        [paths]
        home_dir = /home/user
        config_dir = %(home_dir)s/config
        log_dir = %(home_dir)s/logs
        """
        
        config_file = temp_config_dir / "interpolation.ini"
        with open(config_file, 'w') as f:
            f.write(ini_with_interpolation)
        
        config = configparser.ConfigParser()
        config.read(config_file)
        
        assert config.get('paths', 'config_dir') == "/home/user/config"
        assert config.get('paths', 'log_dir') == "/home/user/logs"


class TestConfigFileErrors:
    """Test error handling for configuration files."""
    
    def test_file_not_found_error(self):
        """Test handling of non-existent configuration files."""
        non_existent_file = "/path/to/non/existent/config.json"
        
        with pytest.raises(FileNotFoundError):
            with open(non_existent_file, 'r') as f:
                json.load(f)
    
    def test_permission_denied_error(self, temp_config_dir):
        """Test handling of permission denied errors."""
        config_file = temp_config_dir / "restricted.json"
        with open(config_file, 'w') as f:
            json.dump({"key": "value"}, f)
        
        # Make file unreadable
        os.chmod(config_file, 0o000)
        
        try:
            with pytest.raises(PermissionError):
                with open(config_file, 'r') as f:
                    json.load(f)
        finally:
            # Restore permissions for cleanup
            os.chmod(config_file, 0o644)
    
    @patch('builtins.open', side_effect=IOError("Simulated IO error"))
    def test_io_error_handling(self, mock_open):
        """Test handling of IO errors during file operations."""
        with pytest.raises(IOError):
            with open("any_file.json", 'r') as f:
                json.load(f)


class TestConfigFileIntegration:
    """Integration tests for configuration file operations."""
    
    def test_config_file_backup_and_restore(self, temp_config_dir, sample_json_config):
        """Test creating backups and restoring configuration files."""
        config_file = temp_config_dir / "config.json"
        backup_file = temp_config_dir / "config.json.backup"
        
        # Create original config
        with open(config_file, 'w') as f:
            json.dump(sample_json_config, f)
        
        # Create backup
        with open(config_file, 'r') as src, open(backup_file, 'w') as dst:
            dst.write(src.read())
        
        # Modify original
        modified_config = sample_json_config.copy()
        modified_config["database"]["host"] = "modified.example.com"
        
        with open(config_file, 'w') as f:
            json.dump(modified_config, f)
        
        # Restore from backup
        with open(backup_file, 'r') as src, open(config_file, 'w') as dst:
            dst.write(src.read())
        
        # Verify restoration
        with open(config_file, 'r') as f:
            restored_config = json.load(f)
        
        assert restored_config["database"]["host"] == "localhost"
    
    def test_config_file_merging(self, temp_config_dir):
        """Test merging multiple configuration files."""
        base_config = {"database": {"host": "localhost", "port": 5432}}
        override_config = {"database": {"host": "override.example.com"}, "api": {"timeout": 60}}
        
        base_file = temp_config_dir / "base.json"
        override_file = temp_config_dir / "override.json"
        
        with open(base_file, 'w') as f:
            json.dump(base_config, f)
        
        with open(override_file, 'w') as f:
            json.dump(override_config, f)
        
        # Load and merge configs
        with open(base_file, 'r') as f:
            merged_config = json.load(f)
        
        with open(override_file, 'r') as f:
            override_data = json.load(f)
        
        # Simple merge logic for testing
        for key, value in override_data.items():
            if key in merged_config and isinstance(merged_config[key], dict) and isinstance(value, dict):
                merged_config[key].update(value)
            else:
                merged_config[key] = value
        
        assert merged_config["database"]["host"] == "override.example.com"
        assert merged_config["database"]["port"] == 5432
        assert merged_config["api"]["timeout"] == 60


class TestConfigFilePerformance:
    """Performance tests for configuration file operations."""
    
    def test_large_json_config_loading(self, temp_config_dir):
        """Test loading large JSON configuration files."""
        large_config = {"items": [{"id": i, "value": f"item_{i}"} for i in range(1000)]}
        
        config_file = temp_config_dir / "large.json"
        with open(config_file, 'w') as f:
            json.dump(large_config, f)
        
        import time
        start_time = time.time()
        
        with open(config_file, 'r') as f:
            loaded_config = json.load(f)
        
        load_time = time.time() - start_time
        
        assert len(loaded_config["items"]) == 1000
        assert load_time < 1.0  # Should load within 1 second
    
    def test_config_file_caching(self, temp_config_dir, sample_json_config):
        """Test configuration file caching mechanisms."""
        config_file = temp_config_dir / "cached.json"
        with open(config_file, 'w') as f:
            json.dump(sample_json_config, f)
        
        # Simulate caching by loading multiple times
        configs = []
        for _ in range(3):
            with open(config_file, 'r') as f:
                configs.append(json.load(f))
        
        # All configs should be identical
        assert all(config == sample_json_config for config in configs)


class TestConfigFileValidationRules:
    """Test validation rules for configuration files."""
    
    @pytest.mark.parametrize("port", [80, 443, 8080, 3000])
    def test_valid_port_numbers(self, port):
        """Test validation of valid port numbers."""
        assert 1 <= port <= 65535
    
    @pytest.mark.parametrize("port", [-1, 0, 65536, 100000])
    def test_invalid_port_numbers(self, port):
        """Test validation of invalid port numbers."""
        assert not (1 <= port <= 65535)
    
    @pytest.mark.parametrize("url", [
        "http://example.com",
        "https://api.example.com",
        "https://api.example.com:8080/v1"
    ])
    def test_valid_urls(self, url):
        """Test validation of valid URLs."""
        import re
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        
        assert url_pattern.match(url) is not None
    
    @pytest.mark.parametrize("url", [
        "not-a-url",
        "ftp://example.com",
        "http://",
        "https://"
    ])
    def test_invalid_urls(self, url):
        """Test validation of invalid URLs."""
        import re
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        
        assert url_pattern.match(url) is None


@pytest.mark.slow
class TestConfigFileStress:
    """Stress tests for configuration file operations."""
    
    def test_concurrent_config_access(self, temp_config_dir, sample_json_config):
        """Test concurrent access to configuration files."""
        import threading
        import time
        
        config_file = temp_config_dir / "concurrent.json"
        with open(config_file, 'w') as f:
            json.dump(sample_json_config, f)
        
        results = []
        errors = []
        
        def read_config():
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    results.append(config)
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=read_config) for _ in range(10)]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        assert len(errors) == 0
        assert len(results) == 10
        assert all(result == sample_json_config for result in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

class TestConfigFileSecurity:
    """Security tests for configuration files."""
    
    def test_yaml_bomb_protection(self, temp_config_dir):
        """Test protection against YAML bomb attacks."""
        yaml_bomb = """
        a: &anchor [*anchor, *anchor, *anchor, *anchor, *anchor, *anchor, *anchor]
        """
        
        config_file = temp_config_dir / "bomb.yaml"
        with open(config_file, 'w') as f:
            f.write(yaml_bomb)
        
        # This should either fail gracefully or have reasonable limits
        with pytest.raises((yaml.YAMLError, RecursionError, MemoryError)):
            with open(config_file, 'r') as f:
                yaml.safe_load(f)
    
    def test_json_injection_prevention(self, temp_config_dir):
        """Test prevention of JSON injection attacks."""
        malicious_json = '{"__proto__": {"polluted": "true"}, "key": "value"}'
        
        config_file = temp_config_dir / "malicious.json"
        with open(config_file, 'w') as f:
            f.write(malicious_json)
        
        with open(config_file, 'r') as f:
            loaded_config = json.load(f)
        
        # Ensure prototype pollution doesn't occur
        assert "__proto__" in loaded_config  # It's just a regular key
        assert loaded_config["key"] == "value"
    
    def test_path_traversal_prevention(self, temp_config_dir):
        """Test prevention of path traversal in file paths."""
        malicious_config = {
            "log_file": "../../../etc/passwd",
            "data_dir": "../../../../sensitive/data"
        }
        
        config_file = temp_config_dir / "traversal.json"
        with open(config_file, 'w') as f:
            json.dump(malicious_config, f)
        
        with open(config_file, 'r') as f:
            loaded_config = json.load(f)
        
        # Configuration loading should work, but path validation should be done by the application
        assert "../" in loaded_config["log_file"]
        assert loaded_config["data_dir"].count("../") == 4
    
    @pytest.mark.parametrize("encoding", ["utf-8", "utf-16", "latin1"])
    def test_encoding_handling(self, temp_config_dir, encoding):
        """Test handling of different file encodings."""
        config_data = {"message": "Hello, 世界! 🌍"}
        
        config_file = temp_config_dir / f"encoded_{encoding}.json"
        
        with open(config_file, 'w', encoding=encoding) as f:
            json.dump(config_data, f, ensure_ascii=False)
        
        with open(config_file, 'r', encoding=encoding) as f:
            loaded_config = json.load(f)
        
        assert loaded_config["message"] == "Hello, 世界! 🌍"


class TestConfigFileEdgeCases:
    """Edge case tests for configuration files."""
    
    def test_deeply_nested_json_config(self, temp_config_dir):
        """Test handling of deeply nested JSON configurations."""
        # Create a deeply nested structure
        deep_config = {"level": 1}
        current = deep_config
        for i in range(2, 50):  # 49 levels deep
            current["nested"] = {"level": i}
            current = current["nested"]
        
        config_file = temp_config_dir / "deep.json"
        with open(config_file, 'w') as f:
            json.dump(deep_config, f)
        
        with open(config_file, 'r') as f:
            loaded_config = json.load(f)
        
        # Navigate to the deepest level
        current = loaded_config
        for _ in range(48):
            current = current["nested"]
        
        assert current["level"] == 49
    
    def test_unicode_keys_and_values(self, temp_config_dir):
        """Test handling of Unicode characters in keys and values."""
        unicode_config = {
            "🔑_key": "🌟_value",
            "中文键": "中文值",
            "עברית": "ערך בעברית",
            "русский": "русское значение",
            "emoji_🎉": "celebration_🎊"
        }
        
        config_file = temp_config_dir / "unicode.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(unicode_config, f, ensure_ascii=False)
        
        with open(config_file, 'r', encoding='utf-8') as f:
            loaded_config = json.load(f)
        
        assert loaded_config["🔑_key"] == "🌟_value"
        assert loaded_config["中文键"] == "中文值"
        assert loaded_config["emoji_🎉"] == "celebration_🎊"
    
    def test_extremely_long_strings(self, temp_config_dir):
        """Test handling of extremely long string values."""
        long_string = "x" * 100000  # 100KB string
        config_with_long_string = {
            "short_key": "short_value",
            "long_key": long_string
        }
        
        config_file = temp_config_dir / "long_strings.json"
        with open(config_file, 'w') as f:
            json.dump(config_with_long_string, f)
        
        with open(config_file, 'r') as f:
            loaded_config = json.load(f)
        
        assert len(loaded_config["long_key"]) == 100000
        assert loaded_config["short_key"] == "short_value"
    
    def test_numeric_precision(self, temp_config_dir):
        """Test handling of numeric precision in configurations."""
        precision_config = {
            "small_float": 0.000000000001,
            "large_float": 1234567890.123456789,
            "scientific": 1.23e-10,
            "large_int": 9007199254740991,  # MAX_SAFE_INTEGER in JavaScript
            "negative": -9007199254740991
        }
        
        config_file = temp_config_dir / "precision.json"
        with open(config_file, 'w') as f:
            json.dump(precision_config, f)
        
        with open(config_file, 'r') as f:
            loaded_config = json.load(f)
        
        assert abs(loaded_config["small_float"] - 0.000000000001) < 1e-15
        assert loaded_config["large_int"] == 9007199254740991
        assert loaded_config["scientific"] == 1.23e-10
    
    def test_special_characters_in_strings(self, temp_config_dir):
        """Test handling of special characters and escape sequences."""
        special_config = {
            "newlines": "line1\nline2\nline3",
            "tabs": "col1\tcol2\tcol3",
            "quotes": 'He said "Hello" and she replied \'Hi\'',
            "backslashes": "C:\\Users\\Name\\Documents",
            "null_char": "before\x00after",
            "control_chars": "\x01\x02\x03\x04\x05",
            "unicode_escapes": "\u03B1\u03B2\u03B3"  # Greek letters
        }
        
        config_file = temp_config_dir / "special_chars.json"
        with open(config_file, 'w') as f:
            json.dump(special_config, f)
        
        with open(config_file, 'r') as f:
            loaded_config = json.load(f)
        
        assert loaded_config["newlines"].count('\n') == 2
        assert loaded_config["tabs"].count('\t') == 2
        assert "Hello" in loaded_config["quotes"]
        assert loaded_config["unicode_escapes"] == "αβγ"


class TestConfigFileFormatConversion:
    """Tests for converting between different configuration formats."""
    
    def test_json_to_yaml_conversion(self, temp_config_dir, sample_json_config):
        """Test converting JSON configuration to YAML format."""
        # Save as JSON first
        json_file = temp_config_dir / "config.json"
        with open(json_file, 'w') as f:
            json.dump(sample_json_config, f)
        
        # Load JSON and save as YAML
        with open(json_file, 'r') as f:
            config_data = json.load(f)
        
        yaml_file = temp_config_dir / "config.yaml"
        with open(yaml_file, 'w') as f:
            yaml.dump(config_data, f)
        
        # Load YAML and verify it matches original JSON
        with open(yaml_file, 'r') as f:
            yaml_data = yaml.safe_load(f)
        
        assert yaml_data == sample_json_config
    
    def test_yaml_to_json_conversion(self, temp_config_dir, sample_yaml_config):
        """Test converting YAML configuration to JSON format."""
        # Save YAML
        yaml_file = temp_config_dir / "config.yaml"
        with open(yaml_file, 'w') as f:
            f.write(sample_yaml_config)
        
        # Load YAML and save as JSON
        with open(yaml_file, 'r') as f:
            yaml_data = yaml.safe_load(f)
        
        json_file = temp_config_dir / "config.json"
        with open(json_file, 'w') as f:
            json.dump(yaml_data, f)
        
        # Load JSON and verify conversion
        with open(json_file, 'r') as f:
            json_data = json.load(f)
        
        assert json_data["database"]["host"] == "localhost"
        assert json_data["api"]["timeout"] == 30
    
    def test_ini_to_dict_conversion(self, temp_config_dir, sample_ini_config):
        """Test converting INI configuration to dictionary format."""
        # Save INI
        ini_file = temp_config_dir / "config.ini"
        with open(ini_file, 'w') as f:
            f.write(sample_ini_config)
        
        # Load INI and convert to dict
        config = configparser.ConfigParser()
        config.read(ini_file)
        
        config_dict = {}
        for section_name in config.sections():
            config_dict[section_name] = dict(config.items(section_name))
        
        assert config_dict["database"]["host"] == "localhost"
        assert config_dict["database"]["port"] == "5432"  # INI values are strings
        assert config_dict["api"]["base_url"] == "https://api.example.com"


class TestConfigFileTemplating:
    """Tests for configuration file templating and variable substitution."""
    
    def test_environment_variable_substitution(self, temp_config_dir):
        """Test substitution of environment variables in configurations."""
        import os
        
        # Set test environment variables
        os.environ["TEST_HOST"] = "test.example.com"
        os.environ["TEST_PORT"] = "8080"
        
        try:
            template_config = {
                "database": {
                    "host": "${TEST_HOST}",
                    "port": "${TEST_PORT}"
                }
            }
            
            config_file = temp_config_dir / "template.json"
            with open(config_file, 'w') as f:
                json.dump(template_config, f)
            
            # Load and substitute variables
            with open(config_file, 'r') as f:
                config_str = f.read()
            
            # Simple substitution for testing
            import re
            def substitute_env_vars(text):
                def replacer(match):
                    var_name = match.group(1)
                    return os.environ.get(var_name, match.group(0))
                return re.sub(r'\$\{([^}]+)\}', replacer, text)
            
            substituted_config = substitute_env_vars(config_str)
            loaded_config = json.loads(substituted_config)
            
            assert loaded_config["database"]["host"] == "test.example.com"
            assert loaded_config["database"]["port"] == "8080"
            
        finally:
            # Clean up environment variables
            os.environ.pop("TEST_HOST", None)
            os.environ.pop("TEST_PORT", None)
    
    def test_nested_template_substitution(self, temp_config_dir):
        """Test nested template variable substitution."""
        template_config = {
            "base_url": "https://api.example.com",
            "endpoints": {
                "users": "${base_url}/users",
                "orders": "${base_url}/orders",
                "nested": {
                    "deep": "${base_url}/deep/path"
                }
            }
        }
        
        config_file = temp_config_dir / "nested_template.json"
        with open(config_file, 'w') as f:
            json.dump(template_config, f)
        
        # Simple nested substitution logic for testing
        def substitute_internal_vars(config_dict):
            import copy
            result = copy.deepcopy(config_dict)
            
            def substitute_value(value, context):
                if isinstance(value, str) and "${" in value:
                    for key, val in context.items():
                        if isinstance(val, str):
                            value = value.replace(f"${{{key}}}", val)
                return value
            
            # First pass: substitute simple values
            for key, value in result.items():
                if isinstance(value, str):
                    result[key] = substitute_value(value, result)
                elif isinstance(value, dict):
                    for nested_key, nested_value in value.items():
                        if isinstance(nested_value, str):
                            value[nested_key] = substitute_value(nested_value, result)
                        elif isinstance(nested_value, dict):
                            for deep_key, deep_value in nested_value.items():
                                if isinstance(deep_value, str):
                                    nested_value[deep_key] = substitute_value(deep_value, result)
            
            return result
        
        with open(config_file, 'r') as f:
            loaded_config = json.load(f)
        
        substituted = substitute_internal_vars(loaded_config)
        
        assert substituted["endpoints"]["users"] == "https://api.example.com/users"
        assert substituted["endpoints"]["nested"]["deep"] == "https://api.example.com/deep/path"


class TestConfigFileAtomicity:
    """Tests for atomic configuration file operations."""
    
    def test_atomic_config_update(self, temp_config_dir, sample_json_config):
        """Test atomic updates to configuration files."""
        config_file = temp_config_dir / "atomic.json"
        temp_file = temp_config_dir / "atomic.json.tmp"
        
        # Initial config
        with open(config_file, 'w') as f:
            json.dump(sample_json_config, f)
        
        # Atomic update simulation
        updated_config = sample_json_config.copy()
        updated_config["database"]["host"] = "updated.example.com"
        
        # Write to temporary file first
        with open(temp_file, 'w') as f:
            json.dump(updated_config, f)
        
        # Atomic move
        import shutil
        shutil.move(str(temp_file), str(config_file))
        
        # Verify update
        with open(config_file, 'r') as f:
            final_config = json.load(f)
        
        assert final_config["database"]["host"] == "updated.example.com"
        assert not temp_file.exists()
    
    def test_config_rollback_on_error(self, temp_config_dir, sample_json_config):
        """Test configuration rollback on update errors."""
        import shutil
        
        config_file = temp_config_dir / "rollback.json"
        backup_file = temp_config_dir / "rollback.json.backup"
        
        # Initial config
        with open(config_file, 'w') as f:
            json.dump(sample_json_config, f)
        
        # Create backup
        shutil.copy2(str(config_file), str(backup_file))
        
        # Simulate failed update (invalid JSON)
        try:
            with open(config_file, 'w') as f:
                f.write('{"invalid": json}')  # Invalid JSON
            
            # Try to load - should fail
            with open(config_file, 'r') as f:
                json.load(f)
                
        except json.JSONDecodeError:
            # Rollback on error
            shutil.copy2(str(backup_file), str(config_file))
        
        # Verify rollback worked
        with open(config_file, 'r') as f:
            restored_config = json.load(f)
        
        assert restored_config == sample_json_config


class TestConfigFileVersioning:
    """Tests for configuration file versioning and compatibility."""
    
    def test_config_version_detection(self, temp_config_dir):
        """Test detection of configuration file versions."""
        v1_config = {
            "version": "1.0",
            "database": {
                "host": "localhost",
                "port": 5432
            }
        }
        
        v2_config = {
            "version": "2.0",
            "database": {
                "connection_string": "postgresql://localhost:5432/db",
                "pool_size": 10
            }
        }
        
        v1_file = temp_config_dir / "config_v1.json"
        v2_file = temp_config_dir / "config_v2.json"
        
        with open(v1_file, 'w') as f:
            json.dump(v1_config, f)
        
        with open(v2_file, 'w') as f:
            json.dump(v2_config, f)
        
        # Test version detection
        with open(v1_file, 'r') as f:
            config1 = json.load(f)
        
        with open(v2_file, 'r') as f:
            config2 = json.load(f)
        
        assert config1["version"] == "1.0"
        assert config2["version"] == "2.0"
        assert "connection_string" not in config1["database"]
        assert "connection_string" in config2["database"]
    
    def test_config_migration_compatibility(self, temp_config_dir):
        """Test configuration migration between versions."""
        old_config = {
            "version": "1.0",
            "db_host": "localhost",
            "db_port": 5432,
            "db_name": "myapp"
        }
        
        config_file = temp_config_dir / "migration.json"
        with open(config_file, 'w') as f:
            json.dump(old_config, f)
        
        # Migration logic
        def migrate_config(config):
            if config.get("version") == "1.0":
                # Migrate to v2.0 format
                new_config = {
                    "version": "2.0",
                    "database": {
                        "host": config.get("db_host"),
                        "port": config.get("db_port"),
                        "name": config.get("db_name")
                    }
                }
                return new_config
            return config
        
        # Load and migrate
        with open(config_file, 'r') as f:
            loaded_config = json.load(f)
        
        migrated_config = migrate_config(loaded_config)
        
        assert migrated_config["version"] == "2.0"
        assert migrated_config["database"]["host"] == "localhost"
        assert migrated_config["database"]["port"] == 5432


class TestConfigFileMemoryUsage:
    """Tests for configuration file memory usage and efficiency."""
    
    def test_memory_efficient_large_config(self, temp_config_dir):
        """Test memory efficiency with large configuration files."""
        # Create a large configuration
        large_config = {
            f"section_{i}": {
                f"key_{j}": f"value_{i}_{j}" 
                for j in range(100)
            } for i in range(100)
        }
        
        config_file = temp_config_dir / "large_memory.json"
        with open(config_file, 'w') as f:
            json.dump(large_config, f)
        
        # Measure memory usage
        import tracemalloc
        tracemalloc.start()
        
        with open(config_file, 'r') as f:
            loaded_config = json.load(f)
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Verify loading worked and memory usage is reasonable
        assert len(loaded_config) == 100
        assert len(loaded_config["section_0"]) == 100
        assert peak < 50 * 1024 * 1024  # Less than 50MB peak memory
    
    def test_streaming_large_config(self, temp_config_dir):
        """Test streaming processing of large configuration files."""
        # Create a configuration with large arrays
        streaming_config = {
            "metadata": {"version": "1.0"},
            "items": [{"id": i, "data": f"item_{i}"} for i in range(1000)]
        }
        
        config_file = temp_config_dir / "streaming.json"
        with open(config_file, 'w') as f:
            json.dump(streaming_config, f)
        
        # Test that we can at least load it normally
        with open(config_file, 'r') as f:
            loaded_config = json.load(f)
        
        assert loaded_config["metadata"]["version"] == "1.0"
        assert len(loaded_config["items"]) == 1000


class TestConfigFileValidationEnhanced:
    """Enhanced validation tests for configuration files."""
    
    def test_recursive_validation(self, temp_config_dir):
        """Test recursive validation of nested configuration structures."""
        nested_config = {
            "level1": {
                "level2": {
                    "level3": {
                        "required_field": "value",
                        "optional_field": None
                    }
                }
            }
        }
        
        config_file = temp_config_dir / "nested_validation.json"
        with open(config_file, 'w') as f:
            json.dump(nested_config, f)
        
        def validate_nested(config, path=""):
            """Recursive validation function."""
            errors = []
            
            if isinstance(config, dict):
                for key, value in config.items():
                    current_path = f"{path}.{key}" if path else key
                    
                    if key == "required_field" and value is None:
                        errors.append(f"Required field {current_path} is null")
                    
                    if isinstance(value, dict):
                        errors.extend(validate_nested(value, current_path))
            
            return errors
        
        with open(config_file, 'r') as f:
            loaded_config = json.load(f)
        
        validation_errors = validate_nested(loaded_config)
        
        # Should pass validation since required_field has a value
        assert len(validation_errors) == 0
        assert loaded_config["level1"]["level2"]["level3"]["required_field"] == "value"
    
    @pytest.mark.parametrize("config_data,expected_valid", [
        ({"timeout": 30, "retries": 3}, True),
        ({"timeout": -1, "retries": 3}, False),
        ({"timeout": 30, "retries": -1}, False),
        ({"timeout": "30", "retries": 3}, False),  # Wrong type
        ({"timeout": 30}, False),  # Missing required field
    ])
    def test_parametrized_config_validation(self, temp_config_dir, config_data, expected_valid):
        """Test parametrized configuration validation scenarios."""
        config_file = temp_config_dir / "param_validation.json"
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        def validate_config(config):
            """Simple validation function."""
            try:
                # Check required fields
                if "timeout" not in config or "retries" not in config:
                    return False
                
                # Check types
                if not isinstance(config["timeout"], int) or not isinstance(config["retries"], int):
                    return False
                
                # Check ranges
                if config["timeout"] <= 0 or config["retries"] < 0:
                    return False
                
                return True
            except (KeyError, TypeError):
                return False
        
        with open(config_file, 'r') as f:
            loaded_config = json.load(f)
        
        is_valid = validate_config(loaded_config)
        assert is_valid == expected_valid


class TestConfigFileRobustness:
    """Robustness tests for configuration file handling."""
    
    def test_partial_file_corruption_recovery(self, temp_config_dir, sample_json_config):
        """Test recovery from partial file corruption."""
        config_file = temp_config_dir / "corrupted.json"
        
        # Write valid config first
        with open(config_file, 'w') as f:
            json.dump(sample_json_config, f)
        
        # Simulate partial corruption by truncating the file
        with open(config_file, 'r+') as f:
            content = f.read()
            f.seek(0)
            f.write(content[:-10])  # Remove last 10 characters
            f.truncate()
        
        # Should fail to load
        with pytest.raises(json.JSONDecodeError):
            with open(config_file, 'r') as f:
                json.load(f)
    
    def test_config_with_comments_handling(self, temp_config_dir):
        """Test handling of configurations with comments (JSON5-like)."""
        # Standard JSON doesn't support comments, but test handling
        json_with_comments = """{
    // This is a comment
    "database": {
        "host": "localhost", // Another comment
        "port": 5432
    },
    /* Multi-line
       comment */
    "api": {
        "timeout": 30
    }
}"""
        
        config_file = temp_config_dir / "with_comments.json"
        with open(config_file, 'w') as f:
            f.write(json_with_comments)
        
        # Standard JSON parser should fail with comments
        with pytest.raises(json.JSONDecodeError):
            with open(config_file, 'r') as f:
                json.load(f)
        
        # Test comment removal for basic cases
        def remove_json_comments(text):
            """Simple comment removal - not production ready."""
            import re
            # Remove single-line comments
            text = re.sub(r'//.*$', '', text, flags=re.MULTILINE)
            # Remove multi-line comments
            text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
            return text
        
        cleaned_json = remove_json_comments(json_with_comments)
        cleaned_config = json.loads(cleaned_json)
        
        assert cleaned_config["database"]["host"] == "localhost"
        assert cleaned_config["api"]["timeout"] == 30
    
    def test_config_file_locking(self, temp_config_dir, sample_json_config):
        """Test file locking during configuration updates."""
        import threading
        import time
        
        config_file = temp_config_dir / "locked.json"
        with open(config_file, 'w') as f:
            json.dump(sample_json_config, f)
        
        lock_acquired = threading.Event()
        lock_released = threading.Event()
        
        def lock_and_hold():
            try:
                with open(config_file, 'r+') as f:
                    lock_acquired.set()
                    # Hold file handle briefly
                    time.sleep(0.1)
                    lock_released.set()
            except (OSError, IOError):
                # Handle any file access issues
                lock_released.set()
        
        # Start locking thread
        lock_thread = threading.Thread(target=lock_and_hold)
        lock_thread.start()
        
        # Wait for lock to be acquired
        if lock_acquired.wait(timeout=1.0):
            # Try to access file while potentially locked
            try:
                with open(config_file, 'r') as f:
                    # This should still work for reading
                    loaded_config = json.load(f)
                    assert loaded_config == sample_json_config
            except (OSError, IOError):
                # Expected if exclusive lock prevents reading
                pass
        
        lock_thread.join()
        assert lock_released.is_set()


class TestConfigFileAdvancedFeatures:
    """Tests for advanced configuration file features."""
    
    def test_config_schema_validation(self, temp_config_dir):
        """Test configuration validation against a schema."""
        # Define a simple schema
        config_schema = {
            "type": "object",
            "required": ["database", "api"],
            "properties": {
                "database": {
                    "type": "object",
                    "required": ["host", "port"],
                    "properties": {
                        "host": {"type": "string"},
                        "port": {"type": "integer", "minimum": 1, "maximum": 65535}
                    }
                },
                "api": {
                    "type": "object",
                    "required": ["timeout"],
                    "properties": {
                        "timeout": {"type": "integer", "minimum": 1}
                    }
                }
            }
        }
        
        valid_config = {
            "database": {"host": "localhost", "port": 5432},
            "api": {"timeout": 30}
        }
        
        invalid_config = {
            "database": {"host": "localhost", "port": "invalid"},  # Wrong type
            "api": {"timeout": -1}  # Invalid value
        }
        
        def validate_against_schema(config, schema):
            """Simple schema validation - in practice use jsonschema library."""
            def validate_type(value, expected_type):
                if expected_type == "object":
                    return isinstance(value, dict)
                elif expected_type == "string":
                    return isinstance(value, str)
                elif expected_type == "integer":
                    return isinstance(value, int)
                return True
            
            def validate_object(obj, schema_obj):
                if not isinstance(obj, dict):
                    return False
                
                # Check required fields
                for required_field in schema_obj.get("required", []):
                    if required_field not in obj:
                        return False
                
                # Check properties
                for prop, prop_schema in schema_obj.get("properties", {}).items():
                    if prop in obj:
                        if not validate_type(obj[prop], prop_schema.get("type")):
                            return False
                        
                        # Check nested objects
                        if prop_schema.get("type") == "object":
                            if not validate_object(obj[prop], prop_schema):
                                return False
                        
                        # Check integer constraints
                        if prop_schema.get("type") == "integer":
                            value = obj[prop]
                            if isinstance(value, int):
                                min_val = prop_schema.get("minimum")
                                max_val = prop_schema.get("maximum")
                                if min_val is not None and value < min_val:
                                    return False
                                if max_val is not None and value > max_val:
                                    return False
                
                return True
            
            return validate_object(config, schema)
        
        assert validate_against_schema(valid_config, config_schema) == True
        assert validate_against_schema(invalid_config, config_schema) == False
    
    def test_config_profile_management(self, temp_config_dir):
        """Test management of different configuration profiles."""
        profiles = {
            "development": {
                "database": {"host": "localhost", "debug": True},
                "api": {"base_url": "http://localhost:8000"}
            },
            "staging": {
                "database": {"host": "staging.db.com", "debug": False},
                "api": {"base_url": "https://staging-api.example.com"}
            },
            "production": {
                "database": {"host": "prod.db.com", "debug": False},
                "api": {"base_url": "https://api.example.com"}
            }
        }
        
        profiles_file = temp_config_dir / "profiles.json"
        with open(profiles_file, 'w') as f:
            json.dump(profiles, f)
        
        def get_profile_config(profile_name):
            with open(profiles_file, 'r') as f:
                all_profiles = json.load(f)
            return all_profiles.get(profile_name)
        
        dev_config = get_profile_config("development")
        prod_config = get_profile_config("production")
        
        assert dev_config["database"]["debug"] == True
        assert prod_config["database"]["debug"] == False
        assert dev_config["api"]["base_url"].startswith("http://")
        assert prod_config["api"]["base_url"].startswith("https://")
    
    def test_config_inheritance(self, temp_config_dir):
        """Test configuration inheritance from base configurations."""
        base_config = {
            "database": {"port": 5432, "timeout": 30},
            "logging": {"level": "INFO"}
        }
        
        override_config = {
            "database": {"host": "override.com"},
            "logging": {"level": "DEBUG"},  # Override
            "api": {"timeout": 60}  # New section
        }
        
        base_file = temp_config_dir / "base.json"
        override_file = temp_config_dir / "override.json"
        
        with open(base_file, 'w') as f:
            json.dump(base_config, f)
        
        with open(override_file, 'w') as f:
            json.dump(override_config, f)
        
        def merge_configs(base_config, override_config):
            """Deep merge two configuration dictionaries."""
            import copy
            result = copy.deepcopy(base_config)
            
            def deep_merge(base_dict, override_dict):
                for key, value in override_dict.items():
                    if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                        deep_merge(base_dict[key], value)
                    else:
                        base_dict[key] = value
            
            deep_merge(result, override_config)
            return result
        
        with open(base_file, 'r') as f:
            base_data = json.load(f)
        
        with open(override_file, 'r') as f:
            override_data = json.load(f)
        
        merged_config = merge_configs(base_data, override_data)
        
        # Base values should be preserved
        assert merged_config["database"]["port"] == 5432
        assert merged_config["database"]["timeout"] == 30
        
        # Override values should take precedence
        assert merged_config["database"]["host"] == "override.com"
        assert merged_config["logging"]["level"] == "DEBUG"
        
        # New sections should be added
        assert merged_config["api"]["timeout"] == 60


# Add pytest marks for different test categories
pytest.mark.security = pytest.mark.mark("security")
pytest.mark.edge_cases = pytest.mark.mark("edge_cases")
pytest.mark.performance = pytest.mark.mark("performance")
pytest.mark.advanced = pytest.mark.mark("advanced")


if __name__ == "__main__":
    # Run with various markers to categorize tests
    pytest.main([__file__, "-v", "--tb=short"])