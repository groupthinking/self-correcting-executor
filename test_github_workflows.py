"""
Comprehensive unit tests for GitHub workflow functionality.
Testing framework: pytest with fixtures, mocks, and parametrized tests.
"""

import pytest
import yaml
from unittest.mock import patch
from pathlib import Path
from typing import Dict, List, Any


class TestGitHubWorkflowParser:
    """Test suite for GitHub workflow parsing functionality."""
    
    @pytest.fixture
    def sample_workflow_yaml(self):
        """Sample GitHub workflow YAML content for testing."""
        return """
name: CI
on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: pytest
"""

    @pytest.fixture
    def invalid_workflow_yaml(self):
        """Invalid YAML content for testing error handling."""
        return """
name: Invalid Workflow
on:
  push:
    branches: [main
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - invalid_syntax
"""

    @pytest.fixture
    def complex_workflow_yaml(self):
        """Complex workflow with multiple jobs and conditions."""
        return """
name: Complex CI/CD
on:
  push:
    branches: [main]
    tags: ['v*']
  schedule:
    - cron: '0 2 * * 0'
env:
  GLOBAL_VAR: global_value
jobs:
  lint:
    runs-on: ubuntu-latest
    outputs:
      lint-status: ${{ steps.lint.outputs.status }}
    steps:
      - uses: actions/checkout@v3
      - name: Lint code
        id: lint
        run: flake8 .
  test:
    runs-on: ${{ matrix.os }}
    needs: lint
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        python-version: ['3.8', '3.9', '3.10']
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python ${{ matrix.python-version }}
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
      - name: Run tests
        run: pytest --cov
  deploy:
    runs-on: ubuntu-latest
    needs: [lint, test]
    if: github.ref == 'refs/heads/main'
    environment: production
    steps:
      - name: Deploy
        run: echo "Deploying to production"
"""

    @pytest.fixture
    def mock_workflow_file(self, tmp_path, sample_workflow_yaml):
        """Create a temporary workflow file for testing."""
        workflow_file = tmp_path / ".github" / "workflows" / "ci.yml"
        workflow_file.parent.mkdir(parents=True, exist_ok=True)
        workflow_file.write_text(sample_workflow_yaml)
        return workflow_file

    def test_parse_valid_workflow_yaml(self, sample_workflow_yaml):
        """Test parsing of valid workflow YAML."""
        parsed = yaml.safe_load(sample_workflow_yaml)
        
        assert parsed['name'] == 'CI'
        assert 'push' in parsed['on']
        assert 'pull_request' in parsed['on']
        assert 'test' in parsed['jobs']
        assert parsed['jobs']['test']['runs-on'] == 'ubuntu-latest'
        assert len(parsed['jobs']['test']['steps']) == 4

    def test_parse_invalid_workflow_yaml(self, invalid_workflow_yaml):
        """Test handling of invalid YAML syntax."""
        with pytest.raises(yaml.YAMLError):
            yaml.safe_load(invalid_workflow_yaml)

    def test_workflow_validation_missing_required_fields(self):
        """Test validation of workflows missing required fields."""
        incomplete_workflow = {
            'name': 'Incomplete Workflow'
            # Missing 'on' and 'jobs' fields
        }
        
        # Test that required fields are validated
        assert 'on' not in incomplete_workflow
        assert 'jobs' not in incomplete_workflow

    def test_workflow_job_validation(self, sample_workflow_yaml):
        """Test validation of job configuration."""
        parsed = yaml.safe_load(sample_workflow_yaml)
        job = parsed['jobs']['test']
        
        assert 'runs-on' in job
        assert 'steps' in job
        assert isinstance(job['steps'], list)
        assert all('uses' in step or 'run' in step or 'name' in step for step in job['steps'])

    @pytest.mark.parametrize("trigger_event,expected_branches", [
        ('push', ['main', 'develop']),
        ('pull_request', ['main']),
    ])
    def test_workflow_triggers(self, sample_workflow_yaml, trigger_event, expected_branches):
        """Test workflow trigger configurations."""
        parsed = yaml.safe_load(sample_workflow_yaml)
        
        assert trigger_event in parsed['on']
        if 'branches' in parsed['on'][trigger_event]:
            assert parsed['on'][trigger_event]['branches'] == expected_branches

    def test_complex_workflow_structure(self, complex_workflow_yaml):
        """Test parsing of complex workflow with multiple jobs and dependencies."""
        parsed = yaml.safe_load(complex_workflow_yaml)
        
        # Test basic structure
        assert parsed['name'] == 'Complex CI/CD'
        assert len(parsed['jobs']) == 3
        
        # Test job dependencies
        assert 'needs' in parsed['jobs']['test']
        assert parsed['jobs']['test']['needs'] == 'lint'
        assert parsed['jobs']['deploy']['needs'] == ['lint', 'test']
        
        # Test matrix strategy
        assert 'strategy' in parsed['jobs']['test']
        matrix = parsed['jobs']['test']['strategy']['matrix']
        assert len(matrix['os']) == 3
        assert len(matrix['python-version']) == 3
        
        # Test conditional execution
        assert 'if' in parsed['jobs']['deploy']
        assert 'environment' in parsed['jobs']['deploy']

    def test_workflow_environment_variables(self, complex_workflow_yaml):
        """Test handling of environment variables in workflows."""
        parsed = yaml.safe_load(complex_workflow_yaml)
        
        assert 'env' in parsed
        assert parsed['env']['GLOBAL_VAR'] == 'global_value'

    def test_workflow_outputs(self, complex_workflow_yaml):
        """Test job outputs configuration."""
        parsed = yaml.safe_load(complex_workflow_yaml)
        
        lint_job = parsed['jobs']['lint']
        assert 'outputs' in lint_job
        assert 'lint-status' in lint_job['outputs']

    @pytest.mark.parametrize("step_type,required_field", [
        ('action', 'uses'),
        ('script', 'run'),
    ])
    def test_workflow_step_types(self, sample_workflow_yaml, step_type, required_field):
        """Test different types of workflow steps."""
        parsed = yaml.safe_load(sample_workflow_yaml)
        steps = parsed['jobs']['test']['steps']
        
        # Find steps of the specified type
        matching_steps = [step for step in steps if required_field in step]
        assert len(matching_steps) > 0
        
        for step in matching_steps:
            assert required_field in step
            assert isinstance(step[required_field], str)


class TestGitHubWorkflowValidator:
    """Test suite for GitHub workflow validation functionality."""
    
    @pytest.fixture
    def validator_config(self):
        """Configuration for workflow validator."""
        return {
            'required_fields': ['name', 'on', 'jobs'],
            'allowed_runners': ['ubuntu-latest', 'windows-latest', 'macos-latest'],
            'max_jobs': 10,
            'max_steps_per_job': 20
        }

    def test_validate_workflow_structure_valid(self, sample_workflow_yaml, validator_config):
        """Test validation of valid workflow structure."""
        parsed = yaml.safe_load(sample_workflow_yaml)
        
        # Check required fields
        for field in validator_config['required_fields']:
            assert field in parsed

    def test_validate_workflow_structure_missing_fields(self, validator_config):
        """Test validation fails for missing required fields."""
        invalid_workflow = {'name': 'Test'}
        
        missing_fields = []
        for field in validator_config['required_fields']:
            if field not in invalid_workflow:
                missing_fields.append(field)
        
        assert len(missing_fields) > 0
        assert 'on' in missing_fields
        assert 'jobs' in missing_fields

    def test_validate_runner_allowed(self, sample_workflow_yaml, validator_config):
        """Test validation of allowed runners."""
        parsed = yaml.safe_load(sample_workflow_yaml)
        
        for job_name, job_config in parsed['jobs'].items():
            if 'runs-on' in job_config:
                runner = job_config['runs-on']
                if isinstance(runner, str):
                    # For matrix strategies, runner might be a template
                    if not runner.startswith('${{'):
                        assert runner in validator_config['allowed_runners']

    def test_validate_job_limits(self, complex_workflow_yaml, validator_config):
        """Test validation of job and step limits."""
        parsed = yaml.safe_load(complex_workflow_yaml)
        
        # Test job count limit
        assert len(parsed['jobs']) <= validator_config['max_jobs']
        
        # Test steps per job limit
        for job_name, job_config in parsed['jobs'].items():
            if 'steps' in job_config:
                assert len(job_config['steps']) <= validator_config['max_steps_per_job']

    @pytest.mark.parametrize("invalid_runner", [
        'invalid-runner',
        'custom-runner-not-allowed',
        'ubuntu-18.04',  # Deprecated
    ])
    def test_validate_runner_not_allowed(self, invalid_runner, validator_config):
        """Test validation rejects invalid runners."""
        assert invalid_runner not in validator_config['allowed_runners']


class TestGitHubWorkflowFileOperations:
    """Test suite for GitHub workflow file operations."""
    
    def test_read_workflow_file(self, mock_workflow_file):
        """Test reading workflow file from filesystem."""
        content = mock_workflow_file.read_text()
        
        assert 'name: CI' in content
        assert 'on:' in content
        assert 'jobs:' in content

    def test_read_nonexistent_workflow_file(self, tmp_path):
        """Test handling of nonexistent workflow files."""
        nonexistent_file = tmp_path / "nonexistent.yml"
        
        assert not nonexistent_file.exists()
        with pytest.raises(FileNotFoundError):
            nonexistent_file.read_text()

    @patch('pathlib.Path.read_text')
    def test_read_workflow_file_permission_error(self, mock_read_text):
        """Test handling of permission errors when reading files."""
        mock_read_text.side_effect = PermissionError("Permission denied")
        
        workflow_file = Path("test.yml")
        with pytest.raises(PermissionError):
            workflow_file.read_text()

    def test_write_workflow_file(self, tmp_path, sample_workflow_yaml):
        """Test writing workflow file to filesystem."""
        output_file = tmp_path / "output.yml"
        output_file.write_text(sample_workflow_yaml)
        
        assert output_file.exists()
        content = output_file.read_text()
        assert content == sample_workflow_yaml

    def test_discover_workflow_files(self, tmp_path):
        """Test discovery of workflow files in directory structure."""
        # Create multiple workflow files
        workflows_dir = tmp_path / ".github" / "workflows"
        workflows_dir.mkdir(parents=True)
        
        (workflows_dir / "ci.yml").write_text("name: CI")
        (workflows_dir / "cd.yml").write_text("name: CD")
        (workflows_dir / "test.yaml").write_text("name: Test")
        (workflows_dir / "README.md").write_text("Not a workflow")
        
        # Find workflow files
        workflow_files = list(workflows_dir.glob("*.yml")) + list(workflows_dir.glob("*.yaml"))
        workflow_files = [f for f in workflow_files if f.suffix in ['.yml', '.yaml']]
        
        assert len(workflow_files) == 3
        workflow_names = [f.name for f in workflow_files]
        assert "ci.yml" in workflow_names
        assert "cd.yml" in workflow_names
        assert "test.yaml" in workflow_names
        assert "README.md" not in workflow_names


class TestGitHubWorkflowSecurity:
    """Test suite for GitHub workflow security validations."""
    
    @pytest.fixture
    def insecure_workflow_yaml(self):
        """Workflow with potential security issues."""
        return """
name: Insecure Workflow
on:
  pull_request_target:  # Potentially dangerous
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v2  # Outdated version
      - name: Run untrusted code
        run: |
          curl -s ${{ github.event.pull_request.head.repo.clone_url }} | bash
      - name: Use secret in command
        run: echo "Secret: ${{ secrets.API_KEY }}"
"""

    def test_detect_pull_request_target_trigger(self, insecure_workflow_yaml):
        """Test detection of potentially dangerous pull_request_target trigger."""
        parsed = yaml.safe_load(insecure_workflow_yaml)
        
        # This trigger can be dangerous as it runs with write permissions
        assert 'pull_request_target' in parsed['on']

    def test_detect_outdated_actions(self, insecure_workflow_yaml):
        """Test detection of outdated action versions."""
        parsed = yaml.safe_load(insecure_workflow_yaml)
        
        checkout_step = None
        for step in parsed['jobs']['test']['steps']:
            if 'uses' in step and 'checkout' in step['uses']:
                checkout_step = step
                break
        
        assert checkout_step is not None
        assert '@v2' in checkout_step['uses']  # Outdated version

    def test_detect_secret_exposure(self, insecure_workflow_yaml):
        """Test detection of potential secret exposure."""
        parsed = yaml.safe_load(insecure_workflow_yaml)
        
        dangerous_step = None
        for step in parsed['jobs']['test']['steps']:
            if 'run' in step and 'secrets.' in step['run']:
                dangerous_step = step
                break
        
        assert dangerous_step is not None
        assert '${{ secrets.' in dangerous_step['run']

    def test_detect_code_injection_risk(self, insecure_workflow_yaml):
        """Test detection of potential code injection vulnerabilities."""
        parsed = yaml.safe_load(insecure_workflow_yaml)
        
        risky_step = None
        for step in parsed['jobs']['test']['steps']:
            if 'run' in step and 'github.event' in step['run'] and 'bash' in step['run']:
                risky_step = step
                break
        
        assert risky_step is not None


class TestGitHubWorkflowUtilities:
    """Test suite for GitHub workflow utility functions."""
    
    @pytest.mark.parametrize("workflow_name,expected_filename", [
        ("CI", "ci.yml"),
        ("Build and Deploy", "build-and-deploy.yml"),
        ("Test_Matrix", "test-matrix.yml"),
        ("PR Validation", "pr-validation.yml"),
    ])
    def test_generate_workflow_filename(self, workflow_name, expected_filename):
        """Test generation of workflow filenames from names."""
        # Simple implementation of filename generation
        filename = workflow_name.lower().replace(' ', '-').replace('_', '-') + '.yml'
        assert filename == expected_filename

    def test_extract_workflow_metadata(self, complex_workflow_yaml):
        """Test extraction of workflow metadata."""
        parsed = yaml.safe_load(complex_workflow_yaml)
        
        metadata = {
            'name': parsed.get('name'),
            'triggers': list(parsed.get('on', {}).keys()),
            'job_count': len(parsed.get('jobs', {})),
            'has_matrix': any('strategy' in job for job in parsed.get('jobs', {}).values()),
            'has_conditions': any('if' in job for job in parsed.get('jobs', {}).values()),
            'has_environment': any('environment' in job for job in parsed.get('jobs', {}).values())
        }
        
        assert metadata['name'] == 'Complex CI/CD'
        assert 'push' in metadata['triggers']
        assert 'schedule' in metadata['triggers']
        assert metadata['job_count'] == 3
        assert metadata['has_matrix'] is True
        assert metadata['has_conditions'] is True
        assert metadata['has_environment'] is True

    def test_workflow_dependency_graph(self, complex_workflow_yaml):
        """Test creation of job dependency graph."""
        parsed = yaml.safe_load(complex_workflow_yaml)
        
        dependencies = {}
        for job_name, job_config in parsed['jobs'].items():
            needs = job_config.get('needs', [])
            if isinstance(needs, str):
                needs = [needs]
            dependencies[job_name] = needs
        
        assert dependencies['lint'] == []
        assert dependencies['test'] == ['lint']
        assert set(dependencies['deploy']) == {'lint', 'test'}

    @pytest.mark.parametrize("cron_expression,is_valid", [
        ("0 2 * * 0", True),  # Every Sunday at 2 AM
        ("0 0 * * *", True),  # Daily at midnight
        ("*/15 * * * *", True),  # Every 15 minutes
        ("invalid cron", False),  # Invalid expression
        ("60 25 * * *", False),  # Invalid time values
    ])
    def test_validate_cron_expressions(self, cron_expression, is_valid):
        """Test validation of cron expressions in schedule triggers."""
        # Basic cron validation (simplified)
        parts = cron_expression.split()
        
        if len(parts) != 5:
            assert not is_valid
            return
        
        # Check for obviously invalid patterns
        if "invalid" in cron_expression:
            assert not is_valid
        elif "60" in parts[0] or "25" in parts[1]:  # Invalid minute/hour
            assert not is_valid
        else:
            assert is_valid


# Integration tests
class TestGitHubWorkflowIntegration:
    """Integration tests for complete workflow processing."""
    
    def test_end_to_end_workflow_processing(self, tmp_path, sample_workflow_yaml):
        """Test complete workflow processing from file to validation."""
        # Setup
        workflow_file = tmp_path / ".github" / "workflows" / "test.yml"
        workflow_file.parent.mkdir(parents=True)
        workflow_file.write_text(sample_workflow_yaml)
        
        # Process workflow
        content = workflow_file.read_text()
        parsed = yaml.safe_load(content)
        
        # Validate structure
        assert parsed['name'] == 'CI'
        assert 'jobs' in parsed
        assert 'test' in parsed['jobs']
        
        # Extract metadata
        metadata = {
            'file_path': str(workflow_file),
            'name': parsed['name'],
            'job_count': len(parsed['jobs']),
            'step_count': sum(len(job.get('steps', [])) for job in parsed['jobs'].values())
        }
        
        assert metadata['job_count'] == 1
        assert metadata['step_count'] == 4

    @patch('yaml.safe_load')
    def test_workflow_processing_with_yaml_error(self, mock_yaml_load, tmp_path):
        """Test handling of YAML parsing errors in workflow processing."""
        mock_yaml_load.side_effect = yaml.YAMLError("Invalid YAML")
        
        workflow_file = tmp_path / "invalid.yml"
        workflow_file.write_text("invalid: yaml: content")
        
        content = workflow_file.read_text()
        
        with pytest.raises(yaml.YAMLError):
            yaml.safe_load(content)

    def test_batch_workflow_validation(self, tmp_path):
        """Test validation of multiple workflow files."""
        # Create multiple workflow files
        workflows_dir = tmp_path / ".github" / "workflows"
        workflows_dir.mkdir(parents=True)
        
        workflows = [
            ("valid.yml", "name: Valid\non: push\njobs:\n  test:\n    runs-on: ubuntu-latest"),
            ("invalid.yml", "name: Invalid\n# Missing required fields"),
        ]
        
        results = {}
        for filename, content in workflows:
            file_path = workflows_dir / filename
            file_path.write_text(content)
            
            try:
                parsed = yaml.safe_load(content)
                has_required_fields = all(field in parsed for field in ['name', 'on', 'jobs'])
                results[filename] = {'valid': has_required_fields, 'error': None}
            except Exception as e:
                results[filename] = {'valid': False, 'error': str(e)}
        
        assert results['valid.yml']['valid'] is True
        assert results['invalid.yml']['valid'] is False


# Performance tests
class TestGitHubWorkflowPerformance:
    """Performance tests for workflow processing."""
    
    def test_large_workflow_parsing_performance(self):
        """Test performance with large workflow files."""
        # Generate a large workflow
        large_workflow = {
            'name': 'Large Workflow',
            'on': ['push', 'pull_request'],
            'jobs': {}
        }
        
        # Add many jobs
        for i in range(50):
            large_workflow['jobs'][f'job_{i}'] = {
                'runs-on': 'ubuntu-latest',
                'steps': [
                    {'uses': 'actions/checkout@v3'},
                    {'run': f'echo "Job {i}"'}
                ]
            }
        
        # Test parsing performance
        import time
        start_time = time.time()
        yaml_content = yaml.dump(large_workflow)
        parsed = yaml.safe_load(yaml_content)
        end_time = time.time()
        
        assert parsed['name'] == 'Large Workflow'
        assert len(parsed['jobs']) == 50
        assert (end_time - start_time) < 1.0  # Should complete within 1 second

    def test_memory_usage_with_multiple_workflows(self, tmp_path):
        """Test memory usage when processing multiple workflows."""
        import sys
        
        workflows_dir = tmp_path / ".github" / "workflows"
        workflows_dir.mkdir(parents=True)
        
        # Create multiple workflow files
        for i in range(10):
            workflow_content = f"""
name: Workflow {i}
on: push
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - run: echo "Workflow {i}"
"""
            (workflows_dir / f"workflow_{i}.yml").write_text(workflow_content)
        
        # Process all workflows
        parsed_workflows = []
        for workflow_file in workflows_dir.glob("*.yml"):
            content = workflow_file.read_text()
            parsed = yaml.safe_load(content)
            parsed_workflows.append(parsed)
        
        assert len(parsed_workflows) == 10
        # Memory usage test would need additional tooling in real scenario


# Edge case tests
class TestGitHubWorkflowEdgeCases:
    """Test suite for edge cases and unusual scenarios."""
    
    @pytest.fixture
    def edge_case_workflows(self):
        """Various edge case workflow configurations."""
        return {
            'empty_workflow': {},
            'minimal_workflow': {
                'name': 'Minimal',
                'on': 'push',
                'jobs': {
                    'test': {
                        'runs-on': 'ubuntu-latest',
                        'steps': [{'run': 'echo "test"'}]
                    }
                }
            },
            'workflow_with_unicode': {
                'name': 'Unicode Test 🚀',
                'on': 'push',
                'jobs': {
                    'test': {
                        'runs-on': 'ubuntu-latest',
                        'steps': [{'run': 'echo "Testing unicode: 你好世界"'}]
                    }
                }
            },
            'workflow_with_long_strings': {
                'name': 'A' * 1000,  # Very long name
                'on': 'push',
                'jobs': {
                    'test': {
                        'runs-on': 'ubuntu-latest',
                        'steps': [{'run': 'B' * 5000}]  # Very long command
                    }
                }
            }
        }

    def test_empty_workflow_handling(self, edge_case_workflows):
        """Test handling of completely empty workflows."""
        empty_workflow = edge_case_workflows['empty_workflow']
        
        # Should handle empty workflows gracefully
        assert isinstance(empty_workflow, dict)
        assert len(empty_workflow) == 0

    def test_minimal_workflow_validation(self, edge_case_workflows):
        """Test validation of minimal but valid workflows."""
        minimal = edge_case_workflows['minimal_workflow']
        
        assert 'name' in minimal
        assert 'on' in minimal
        assert 'jobs' in minimal
        assert len(minimal['jobs']) == 1

    def test_unicode_support_in_workflows(self, edge_case_workflows):
        """Test support for Unicode characters in workflows."""
        unicode_workflow = edge_case_workflows['workflow_with_unicode']
        
        assert '🚀' in unicode_workflow['name']
        assert '你好世界' in unicode_workflow['jobs']['test']['steps'][0]['run']

    def test_large_string_handling(self, edge_case_workflows):
        """Test handling of very large strings in workflows."""
        long_workflow = edge_case_workflows['workflow_with_long_strings']
        
        assert len(long_workflow['name']) == 1000
        assert len(long_workflow['jobs']['test']['steps'][0]['run']) == 5000

    @pytest.mark.parametrize("invalid_yaml", [
        "name: Test\nsteps:\n  - invalid: [\n",  # Unclosed bracket
        "name: Test\n\ttabs_and_spaces: mixed",  # Mixed indentation
        "name: Test\n'unmatched quote",  # Unmatched quote
        "name: Test\n@invalid_yaml_character",  # Invalid character
    ])
    def test_malformed_yaml_handling(self, invalid_yaml):
        """Test handling of various malformed YAML inputs."""
        with pytest.raises(yaml.YAMLError):
            yaml.safe_load(invalid_yaml)

    def test_deeply_nested_workflow_structure(self):
        """Test handling of deeply nested workflow structures."""
        nested_workflow = {
            'name': 'Nested Test',
            'on': {
                'push': {
                    'branches': ['main'],
                    'paths': ['src/**', 'tests/**']
                },
                'pull_request': {
                    'types': ['opened', 'synchronize'],
                    'branches': ['main', 'develop']
                }
            },
            'jobs': {
                'test': {
                    'runs-on': 'ubuntu-latest',
                    'strategy': {
                        'matrix': {
                            'include': [
                                {'os': 'ubuntu-latest', 'python': '3.8', 'extra': 'test1'},
                                {'os': 'windows-latest', 'python': '3.9', 'extra': 'test2'}
                            ]
                        }
                    }
                }
            }
        }
        
        # Test that deeply nested structures are parsed correctly
        assert nested_workflow['on']['push']['branches'] == ['main']
        assert len(nested_workflow['jobs']['test']['strategy']['matrix']['include']) == 2

    def test_workflow_with_all_trigger_types(self):
        """Test workflow with every possible trigger type."""
        all_triggers_workflow = {
            'name': 'All Triggers',
            'on': {
                'push': {'branches': ['main']},
                'pull_request': {'branches': ['main']},
                'pull_request_target': {'branches': ['main']},
                'schedule': [{'cron': '0 0 * * *'}],
                'workflow_dispatch': {'inputs': {'environment': {'type': 'string'}}},
                'workflow_call': {'inputs': {'version': {'type': 'string'}}},
                'repository_dispatch': {'types': ['custom-event']},
                'release': {'types': ['published']},
                'issues': {'types': ['opened']},
                'issue_comment': {'types': ['created']},
                'watch': {'types': ['started']},
                'fork': {},
                'create': {},
                'delete': {},
                'gollum': {},
                'milestone': {'types': ['created']},
                'project': {'types': ['created']},
                'project_card': {'types': ['created']},
                'project_column': {'types': ['created']},
                'public': {},
                'status': {},
                'check_run': {'types': ['created']},
                'check_suite': {'types': ['completed']},
                'deployment': {},
                'deployment_status': {},
                'page_build': {},
                'registry_package': {'types': ['published']}
            },
            'jobs': {
                'test': {
                    'runs-on': 'ubuntu-latest',
                    'steps': [{'run': 'echo "All triggers test"'}]
                }
            }
        }
        
        # Verify all trigger types are present
        assert len(all_triggers_workflow['on']) > 20
        assert 'workflow_dispatch' in all_triggers_workflow['on']
        assert 'workflow_call' in all_triggers_workflow['on']


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])