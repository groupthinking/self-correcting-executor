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