"""
Comprehensive pytest tests for scripts auto improvement functionality.
Testing Framework: pytest with fixtures, mocks, and parametrized tests.

This module tests the automatic improvement of Python scripts including:
- Adding error handling
- Adding logging
- Adding docstrings  
- Improving code structure
- Adding type hints
- Code formatting improvements
"""

import pytest
import tempfile
import os
import sys
import ast
import textwrap
from unittest.mock import patch, mock_open, MagicMock, call
from io import StringIO
from pathlib import Path


class ScriptImprover:
    """
    Main class for improving Python scripts with various enhancements.
    This is the class being tested - normally would be imported from another module.
    """
    
    def __init__(self, config=None):
        self.config = config or {
            'add_logging': True,
            'add_error_handling': True,
            'add_docstrings': True,
            'add_type_hints': True,
            'format_code': True
        }
    
    def improve_script(self, script_content):
        """Main method to improve a Python script."""
        if not script_content.strip():
            return script_content
        
        improved = script_content
        
        if self.config.get('add_logging', True):
            improved = self.add_logging(improved)
        
        if self.config.get('add_error_handling', True):
            improved = self.add_error_handling(improved)
        
        if self.config.get('add_docstrings', True):
            improved = self.add_docstrings(improved)
        
        return improved
    
    def add_logging(self, script_content):
        """Add logging configuration to script."""
        if 'import logging' in script_content:
            return script_content
        
        lines = script_content.split('\n')
        
        # Find where to insert logging imports
        insert_index = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('import ') or line.strip().startswith('from '):
                insert_index = i + 1
            elif line.strip() and not line.strip().startswith('#'):
                break
        
        # Insert logging setup
        logging_setup = [
            'import logging',
            'logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")',
            ''
        ]
        
        for i, setup_line in enumerate(logging_setup):
            lines.insert(insert_index + i, setup_line)
        
        return '\n'.join(lines)
    
    def add_error_handling(self, script_content):
        """Add error handling to main functions."""
        if 'try:' in script_content and 'except' in script_content:
            return script_content  # Already has error handling
        
        lines = script_content.split('\n')
        
        # Find main function
        for i, line in enumerate(lines):
            if 'def main(' in line or 'def main():' in line:
                # Find the end of the function
                indent_level = len(line) - len(line.lstrip())
                function_end = len(lines)
                
                for j in range(i + 1, len(lines)):
                    if lines[j].strip() and len(lines[j]) - len(lines[j].lstrip()) <= indent_level and not lines[j].startswith(' '):
                        function_end = j
                        break
                
                # Wrap function body in try-except
                function_body_start = i + 1
                while function_body_start < len(lines) and not lines[function_body_start].strip():
                    function_body_start += 1
                
                if function_body_start < function_end:
                    # Add try block
                    lines.insert(function_body_start, '    try:')
                    
                    # Indent existing function body
                    for k in range(function_body_start + 1, function_end + 1):
                        if k < len(lines) and lines[k].strip():
                            lines[k] = '    ' + lines[k]
                    
                    # Add except block
                    lines.insert(function_end + 1, '    except Exception as e:')
                    lines.insert(function_end + 2, '        logging.error(f"Error in main function: {e}")')
                    lines.insert(function_end + 3, '        raise')
                
                break
        
        return '\n'.join(lines)
    
    def add_docstrings(self, script_content):
        """Add docstrings to functions that don't have them."""
        try:
            tree = ast.parse(script_content)
        except SyntaxError:
            return script_content  # Return original if syntax error
        
        lines = script_content.split('\n')
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check if function already has docstring
                if (node.body and isinstance(node.body[0], ast.Expr) and 
                    isinstance(node.body[0].value, ast.Str)):
                    continue  # Already has docstring
                
                # Add docstring after function definition
                func_line = node.lineno - 1  # Convert to 0-based index
                indent = '    ' * (node.col_offset // 4 + 1)
                docstring = f'{indent}"""Function docstring for {node.name}."""'
                
                if func_line + 1 < len(lines):
                    lines.insert(func_line + 1, docstring)
        
        return '\n'.join(lines)


# Test fixtures
@pytest.fixture
def script_improver():
    """Fixture providing a ScriptImprover instance."""
    return ScriptImprover()


@pytest.fixture
def script_improver_minimal():
    """Fixture providing a ScriptImprover with minimal configuration."""
    return ScriptImprover({
        'add_logging': False,
        'add_error_handling': False,
        'add_docstrings': True,
        'add_type_hints': False,
        'format_code': False
    })


@pytest.fixture
def simple_script():
    """Fixture providing a simple Python script."""
    return textwrap.dedent("""
        def main():
            print("Hello World")
        
        if __name__ == "__main__":
            main()
    """).strip()


@pytest.fixture
def complex_script():
    """Fixture providing a more complex Python script."""
    return textwrap.dedent("""
        import os
        import sys
        from datetime import datetime
        
        def process_data(data):
            result = []
            for item in data:
                result.append(item.upper())
            return result
        
        def save_to_file(data, filename):
            with open(filename, 'w') as f:
                f.write(str(data))
        
        def main():
            data = ["hello", "world", "python"]
            processed = process_data(data)
            save_to_file(processed, "output.txt")
            print("Processing complete")
        
        if __name__ == "__main__":
            main()
    """).strip()


@pytest.fixture
def script_with_existing_improvements():
    """Fixture providing a script that already has some improvements."""
    return textwrap.dedent("""
        import logging
        import sys
        
        logging.basicConfig(level=logging.INFO)
        
        def main():
            \"\"\"Main function with existing docstring.\"\"\"
            try:
                print("Hello World")
            except Exception as e:
                logging.error(f"Error: {e}")
                raise
        
        if __name__ == "__main__":
            main()
    """).strip()


class TestScriptImprover:
    """Test suite for the ScriptImprover class."""
    
    def test_init_default_config(self):
        """Test ScriptImprover initialization with default config."""
        improver = ScriptImprover()
        
        assert improver.config['add_logging'] is True
        assert improver.config['add_error_handling'] is True
        assert improver.config['add_docstrings'] is True
    
    def test_init_custom_config(self):
        """Test ScriptImprover initialization with custom config."""
        config = {'add_logging': False, 'add_error_handling': True}
        improver = ScriptImprover(config)
        
        assert improver.config == config
    
    def test_improve_script_empty_string(self, script_improver):
        """Test improvement of empty script."""
        result = script_improver.improve_script("")
        assert result == ""
    
    def test_improve_script_whitespace_only(self, script_improver):
        """Test improvement of script with only whitespace."""
        result = script_improver.improve_script("   \n\t\n   ")
        assert result == "   \n\t\n   "
    
    def test_improve_script_basic(self, script_improver, simple_script):
        """Test basic script improvement."""
        result = script_improver.improve_script(simple_script)
        
        # Should add logging
        assert 'import logging' in result
        assert 'logging.basicConfig' in result
        
        # Should add error handling
        assert 'try:' in result
        assert 'except Exception as e:' in result
        
        # Should add docstrings
        assert 'Function docstring for main' in result
    
    def test_improve_script_preserves_structure(self, script_improver, complex_script):
        """Test that script improvement preserves original structure."""
        result = script_improver.improve_script(complex_script)
        
        # Should preserve original imports
        assert 'import os' in result
        assert 'import sys' in result
        assert 'from datetime import datetime' in result
        
        # Should preserve original functions
        assert 'def process_data' in result
        assert 'def save_to_file' in result
        assert 'def main' in result
        
        # Should preserve original logic
        assert 'for item in data:' in result
        assert 'result.append(item.upper())' in result


class TestAddLogging:
    """Test suite for the add_logging method."""
    
    def test_add_logging_to_script_without_imports(self, script_improver):
        """Test adding logging to script without any imports."""
        script = textwrap.dedent("""
            def main():
                print("Hello")
        """).strip()
        
        result = script_improver.add_logging(script)
        
        assert 'import logging' in result
        assert 'logging.basicConfig' in result
        lines = result.split('\n')
        assert lines[0] == 'import logging'
    
    def test_add_logging_to_script_with_existing_imports(self, script_improver):
        """Test adding logging to script with existing imports."""
        script = textwrap.dedent("""
            import os
            import sys
            
            def main():
                print("Hello")
        """).strip()
        
        result = script_improver.add_logging(script)
        
        assert 'import logging' in result
        assert 'import os' in result
        assert 'import sys' in result
        
        # Logging should be added after existing imports
        lines = result.split('\n')
        import_indices = [i for i, line in enumerate(lines) if 'import' in line]
        logging_index = next(i for i, line in enumerate(lines) if 'import logging' in line)
        
        # Logging import should be within the import section
        assert logging_index in import_indices
    
    def test_add_logging_already_exists(self, script_improver):
        """Test adding logging when it already exists."""
        script = textwrap.dedent("""
            import logging
            import os
            
            def main():
                print("Hello")
        """).strip()
        
        result = script_improver.add_logging(script)
        
        # Should not duplicate logging import
        assert result.count('import logging') == 1
        assert result == script
    
    def test_add_logging_with_from_imports(self, script_improver):
        """Test adding logging to script with from imports."""
        script = textwrap.dedent("""
            from datetime import datetime
            from os.path import join
            
            def main():
                print("Hello")
        """).strip()
        
        result = script_improver.add_logging(script)
        
        assert 'import logging' in result
        assert 'from datetime import datetime' in result
        assert 'from os.path import join' in result


class TestAddErrorHandling:
    """Test suite for the add_error_handling method."""
    
    def test_add_error_handling_to_main_function(self, script_improver):
        """Test adding error handling to main function."""
        script = textwrap.dedent("""
            def main():
                print("Hello World")
                return True
        """).strip()
        
        result = script_improver.add_error_handling(script)
        
        assert 'try:' in result
        assert 'except Exception as e:' in result
        assert 'logging.error' in result
        assert 'raise' in result
    
    def test_add_error_handling_already_exists(self, script_improver):
        """Test adding error handling when it already exists."""
        script = textwrap.dedent("""
            def main():
                try:
                    print("Hello World")
                except Exception as e:
                    print(f"Error: {e}")
        """).strip()
        
        result = script_improver.add_error_handling(script)
        
        # Should not add additional error handling
        assert result == script
    
    def test_add_error_handling_no_main_function(self, script_improver):
        """Test adding error handling when no main function exists."""
        script = textwrap.dedent("""
            def helper():
                print("Helper")
            
            def process():
                print("Process")
        """).strip()
        
        result = script_improver.add_error_handling(script)
        
        # Should not modify script if no main function
        assert result == script
    
    def test_add_error_handling_preserves_indentation(self, script_improver):
        """Test that error handling preserves proper indentation."""
        script = textwrap.dedent("""
            def main():
                x = 1
                y = 2
                print(x + y)
        """).strip()
        
        result = script_improver.add_error_handling(script)
        
        lines = result.split('\n')
        
        # Check that the original code is properly indented within the try block
        for line in lines:
            if 'x = 1' in line or 'y = 2' in line or 'print(x + y)' in line:
                # Should have 8 spaces (4 for function + 4 for try block)
                assert line.startswith('        ')


class TestAddDocstrings:
    """Test suite for the add_docstrings method."""
    
    def test_add_docstrings_to_functions(self, script_improver):
        """Test adding docstrings to functions without them."""
        script = textwrap.dedent("""
            def main():
                print("Hello")
            
            def helper(data):
                return data.upper()
        """).strip()
        
        result = script_improver.add_docstrings(script)
        
        assert 'Function docstring for main' in result
        assert 'Function docstring for helper' in result
    
    def test_add_docstrings_preserves_existing(self, script_improver):
        """Test that existing docstrings are preserved."""
        script = textwrap.dedent("""
            def main():
                \"\"\"Existing docstring.\"\"\"
                print("Hello")
            
            def helper():
                print("Helper")
        """).strip()
        
        result = script_improver.add_docstrings(script)
        
        # Should preserve existing docstring
        assert 'Existing docstring' in result
        
        # Should add docstring to function without one
        assert 'Function docstring for helper' in result
        
        # Should not duplicate docstring for main
        assert result.count('Function docstring for main') == 0
    
    def test_add_docstrings_syntax_error(self, script_improver):
        """Test adding docstrings to script with syntax error."""
        script = "def main(\n    print('hello')"  # Malformed function
        
        result = script_improver.add_docstrings(script)
        
        # Should return original script if syntax error
        assert result == script
    
    def test_add_docstrings_no_functions(self, script_improver):
        """Test adding docstrings to script without functions."""
        script = textwrap.dedent("""
            import os
            print("Hello World")
            x = 1 + 2
        """).strip()
        
        result = script_improver.add_docstrings(script)
        
        # Should not modify script
        assert result == script
    
    def test_add_docstrings_class_methods(self, script_improver):
        """Test adding docstrings to class methods."""
        script = textwrap.dedent("""
            class MyClass:
                def method1(self):
                    pass
                
                def method2(self, data):
                    return data
        """).strip()
        
        result = script_improver.add_docstrings(script)
        
        assert 'Function docstring for method1' in result
        assert 'Function docstring for method2' in result


class TestParametrizedScenarios:
    """Parametrized tests for various scenarios."""
    
    @pytest.mark.parametrize("script_content,expected_improvements", [
        # Basic script should get all improvements
        ("def main(): pass", ["import logging", "try:", "Function docstring"]),
        
        # Script with logging should not get duplicate logging
        ("import logging\ndef main(): pass", ["try:", "Function docstring"]),
        
        # Script with error handling should not get duplicate error handling
        ("def main():\n    try:\n        pass\n    except:\n        pass", ["import logging", "Function docstring"]),
        
        # Script with docstring should not get duplicate docstring
        ('def main():\n    """Existing docstring."""\n    pass', ["import logging", "try:"]),
        
        # Empty script should remain empty
        ("", []),
    ])
    def test_improve_script_scenarios(self, script_content, expected_improvements):
        """Test various script improvement scenarios."""
        improver = ScriptImprover()
        result = improver.improve_script(script_content)
        
        if not script_content.strip():
            assert result == script_content
        else:
            for improvement in expected_improvements:
                assert improvement in result
    
    @pytest.mark.parametrize("config,script,expected_features", [
        # Only logging enabled
        ({"add_logging": True, "add_error_handling": False, "add_docstrings": False}, 
         "def main(): pass", ["import logging"]),
        
        # Only error handling enabled
        ({"add_logging": False, "add_error_handling": True, "add_docstrings": False}, 
         "def main(): pass", ["try:", "except"]),
        
        # Only docstrings enabled
        ({"add_logging": False, "add_error_handling": False, "add_docstrings": True}, 
         "def main(): pass", ["Function docstring"]),
        
        # All disabled
        ({"add_logging": False, "add_error_handling": False, "add_docstrings": False}, 
         "def main(): pass", []),
    ])
    def test_selective_improvements(self, config, script, expected_features):
        """Test selective application of improvements based on configuration."""
        improver = ScriptImprover(config)
        result = improver.improve_script(script)
        
        for feature in expected_features:
            assert feature in result
        
        # Test that disabled features are not added
        if not config.get("add_logging", False):
            assert "import logging" not in result or "import logging" in script
        if not config.get("add_error_handling", False):
            assert ("try:" not in result or "try:" in script) and ("except" not in result or "except" in script)
        if not config.get("add_docstrings", False):
            assert "Function docstring" not in result


class TestEdgeCases:
    """Test suite for edge cases and error conditions."""
    
    def test_very_long_script(self, script_improver):
        """Test improvement of very long script."""
        # Generate a script with many functions
        functions = [f"def function_{i}():\n    pass\n" for i in range(100)]
        long_script = "\n".join(functions)
        
        result = script_improver.improve_script(long_script)
        
        # Should handle long scripts without crashing
        assert isinstance(result, str)
        assert len(result) >= len(long_script)
        
        # Should add logging
        assert "import logging" in result
    
    def test_unicode_characters(self, script_improver):
        """Test handling of scripts with unicode characters."""
        script = textwrap.dedent("""
            def main():
                print("Hello 世界")
                print("Olá mundo")
                print("Привет мир")
        """).strip()
        
        result = script_improver.improve_script(script)
        
        # Should preserve unicode characters
        assert "世界" in result
        assert "Olá mundo" in result
        assert "Привет мир" in result
        
        # Should still add improvements
        assert "import logging" in result
    
    def test_special_string_characters(self, script_improver):
        """Test handling of special characters in strings."""
        script = textwrap.dedent("""
            def main():
                print("String with 'quotes'")
                print('String with "double quotes"')
                print("String with \\n newline")
                print("String with \\t tab")
                print(f"F-string with {variable}")
        """).strip()
        
        result = script_improver.improve_script(script)
        
        # Should preserve special characters
        assert "String with 'quotes'" in result
        assert 'String with "double quotes"' in result
        assert "String with \\n newline" in result
        assert "String with \\t tab" in result
        
        # Should still add improvements
        assert "import logging" in result
    
    def test_complex_indentation(self, script_improver):
        """Test handling of complex indentation scenarios."""
        script = textwrap.dedent("""
            class MyClass:
                def __init__(self):
                    self.data = []
                
                def method(self):
                    if True:
                        for i in range(10):
                            if i % 2 == 0:
                                self.data.append(i)
            
            def main():
                obj = MyClass()
                obj.method()
        """).strip()
        
        result = script_improver.improve_script(script)
        
        # Should preserve complex indentation
        assert "class MyClass:" in result
        assert "    def __init__(self):" in result
        assert "        self.data = []" in result
        assert "            if i % 2 == 0:" in result
        
        # Should add improvements
        assert "import logging" in result
        assert "Function docstring" in result
    
    def test_malformed_python_code(self, script_improver):
        """Test handling of malformed Python code."""
        malformed_scripts = [
            "def main(\n    print('hello')",  # Missing closing parenthesis
            "if True\n    print('hello')",   # Missing colon
            "def main():\nprint('hello')",   # Wrong indentation
        ]
        
        for script in malformed_scripts:
            result = script_improver.improve_script(script)
            
            # Should not crash and should return some result
            assert isinstance(result, str)
            
            # May or may not add improvements depending on what can be parsed
            # But should not raise exceptions


class TestIntegration:
    """Integration tests for complete workflow."""
    
    def test_real_world_script_improvement(self, script_improver):
        """Test improvement of a realistic script."""
        script = textwrap.dedent("""
            import requests
            import json
            from datetime import datetime
            
            def fetch_data(url):
                response = requests.get(url)
                return response.json()
            
            def process_data(data):
                processed = []
                for item in data:
                    if 'name' in item:
                        processed.append({
                            'name': item['name'].upper(),
                            'timestamp': datetime.now().isoformat()
                        })
                return processed
            
            def save_data(data, filename):
                with open(filename, 'w') as f:
                    json.dump(data, f, indent=2)
            
            def main():
                url = "https://api.example.com/users"
                raw_data = fetch_data(url)
                processed_data = process_data(raw_data)
                save_data(processed_data, "output.json")
                print(f"Processed {len(processed_data)} items")
            
            if __name__ == "__main__":
                main()
        """).strip()
        
        result = script_improver.improve_script(script)
        
        # Should preserve all original functionality
        assert "import requests" in result
        assert "import json" in result
        assert "from datetime import datetime" in result
        assert "def fetch_data(url):" in result
        assert "def process_data(data):" in result
        assert "def save_data(data, filename):" in result
        assert "def main():" in result
        assert 'if __name__ == "__main__":' in result
        
        # Should add all improvements
        assert "import logging" in result
        assert "try:" in result
        assert "except Exception as e:" in result
        assert "Function docstring for fetch_data" in result
        assert "Function docstring for process_data" in result
        assert "Function docstring for save_data" in result
        assert "Function docstring for main" in result
    
    @pytest.mark.slow
    def test_performance_large_script(self, script_improver):
        """Test performance on large scripts."""
        import time
        
        # Generate a large script
        functions = []
        for i in range(500):
            func = textwrap.dedent(f"""
                def function_{i}(param_{i}):
                    result = param_{i} * {i}
                    if result > 100:
                        return result
                    else:
                        return 0
            """).strip()
            functions.append(func)
        
        large_script = "\n\n".join(functions)
        large_script += "\n\ndef main():\n    print('Main function')\n"
        
        start_time = time.time()
        result = script_improver.improve_script(large_script)
        end_time = time.time()
        
        # Should complete within reasonable time (5 seconds)
        assert end_time - start_time < 5.0
        
        # Should still produce valid improvements
        assert "import logging" in result
        assert len(result) > len(large_script)


class TestMockingAndFileOperations:
    """Test suite for mocking external dependencies."""
    
    @patch('builtins.open', new_callable=mock_open, read_data="def main(): pass")
    def test_improve_script_from_file(self, mock_file):
        """Test improving script read from file."""
        improver = ScriptImprover()
        
        # Simulate reading from file
        with open('test_script.py', 'r') as f:
            content = f.read()
        
        result = improver.improve_script(content)
        
        # Should have called open
        mock_file.assert_called_once_with('test_script.py', 'r')
        
        # Should add improvements
        assert "import logging" in result
        assert "try:" in result
    
    @patch('logging.basicConfig')
    def test_logging_configuration_called(self, mock_logging_config):
        """Test that logging configuration is properly set up."""
        script = "def main(): pass"
        improver = ScriptImprover()
        
        # Improve script (which should add logging)
        result = improver.improve_script(script)
        
        # Verify logging import was added
        assert "import logging" in result
        assert "logging.basicConfig" in result
    
    def test_with_temp_files(self):
        """Test script improvement with temporary files."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            test_script = textwrap.dedent("""
                def main():
                    print("Test script")
                
                if __name__ == "__main__":
                    main()
            """).strip()
            
            f.write(test_script)
            temp_filename = f.name
        
        try:
            # Read the temp file
            with open(temp_filename, 'r') as f:
                content = f.read()
            
            # Improve the script
            improver = ScriptImprover()
            improved = improver.improve_script(content)
            
            # Write improved version back
            with open(temp_filename, 'w') as f:
                f.write(improved)
            
            # Verify improvements were applied
            with open(temp_filename, 'r') as f:
                final_content = f.read()
            
            assert "import logging" in final_content
            assert "Function docstring for main" in final_content
            
        finally:
            # Clean up
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)


# Pytest markers and configuration
pytestmark = pytest.mark.unit


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])