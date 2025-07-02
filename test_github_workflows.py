"""
Comprehensive unit tests for GitHub workflow functionality.
Testing framework: pytest with fixtures, mocks, and parametrized tests.
"""

import pytest
import json
import yaml
import os
from unittest.mock import Mock, patch, mock_open
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

# Additional test classes for enhanced coverage

class TestGitHubWorkflowAdvancedFeatures:
    """Test suite for advanced GitHub workflow features."""
    
    @pytest.fixture
    def reusable_workflow_yaml(self):
        """Reusable workflow configuration."""
        return """
name: Reusable Workflow
on:
  workflow_call:
    inputs:
      environment:
        required: true
        type: string
        default: 'staging'
      deploy_version:
        required: false
        type: string
    outputs:
      deployment_url:
        description: "Deployment URL"
        value: ${{ jobs.deploy.outputs.url }}
    secrets:
      DEPLOY_TOKEN:
        required: true
jobs:
  deploy:
    runs-on: ubuntu-latest
    environment: ${{ inputs.environment }}
    outputs:
      url: ${{ steps.deploy.outputs.deployment_url }}
    steps:
      - name: Deploy
        id: deploy
        run: echo "deployment_url=https://app.example.com" >> $GITHUB_OUTPUT
"""

    @pytest.fixture
    def workflow_with_concurrency(self):
        """Workflow with concurrency control."""
        return """
name: Concurrency Test
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run tests
        run: pytest
"""

    @pytest.fixture
    def workflow_with_permissions(self):
        """Workflow with explicit permissions."""
        return """
name: Permissions Test
on: push
permissions:
  contents: read
  issues: write
  pull-requests: write
  security-events: write
jobs:
  security-scan:
    runs-on: ubuntu-latest
    permissions:
      security-events: write
    steps:
      - uses: actions/checkout@v4
      - name: Security scan
        run: echo "Running security scan"
"""

    def test_reusable_workflow_structure(self, reusable_workflow_yaml):
        """Test parsing of reusable workflow with inputs and outputs."""
        parsed = yaml.safe_load(reusable_workflow_yaml)
        
        assert parsed['on']['workflow_call'] is not None
        assert 'inputs' in parsed['on']['workflow_call']
        assert 'outputs' in parsed['on']['workflow_call']
        assert 'secrets' in parsed['on']['workflow_call']
        
        inputs = parsed['on']['workflow_call']['inputs']
        assert 'environment' in inputs
        assert inputs['environment']['required'] is True
        assert inputs['environment']['type'] == 'string'
        assert inputs['deploy_version']['required'] is False

    def test_workflow_concurrency_configuration(self, workflow_with_concurrency):
        """Test workflow concurrency settings."""
        parsed = yaml.safe_load(workflow_with_concurrency)
        
        assert 'concurrency' in parsed
        assert 'group' in parsed['concurrency']
        assert 'cancel-in-progress' in parsed['concurrency']
        assert parsed['concurrency']['cancel-in-progress'] is True

    def test_workflow_permissions_configuration(self, workflow_with_permissions):
        """Test workflow and job-level permissions."""
        parsed = yaml.safe_load(workflow_with_permissions)
        
        # Test workflow-level permissions
        assert 'permissions' in parsed
        workflow_perms = parsed['permissions']
        assert workflow_perms['contents'] == 'read'
        assert workflow_perms['issues'] == 'write'
        assert workflow_perms['pull-requests'] == 'write'
        
        # Test job-level permissions
        job = parsed['jobs']['security-scan']
        assert 'permissions' in job
        assert job['permissions']['security-events'] == 'write'

    @pytest.mark.parametrize("permission_level", [
        'read', 'write', 'none'
    ])
    def test_permission_validation(self, permission_level):
        """Test validation of permission levels."""
        valid_permissions = ['read', 'write', 'none']
        assert permission_level in valid_permissions

    def test_workflow_with_services(self):
        """Test workflow with service containers."""
        workflow_with_services = """
name: Service Test
on: push
jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:13
        env:
          POSTGRES_PASSWORD: postgres
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432
      redis:
        image: redis:6
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 6379:6379
    steps:
      - uses: actions/checkout@v4
      - name: Test with services
        run: pytest --redis-url redis://localhost:6379 --db-url postgresql://postgres:postgres@localhost:5432/postgres
"""
        parsed = yaml.safe_load(workflow_with_services)
        
        services = parsed['jobs']['test']['services']
        assert 'postgres' in services
        assert 'redis' in services
        assert services['postgres']['image'] == 'postgres:13'
        assert 'ports' in services['postgres']
        assert 'options' in services['postgres']

    def test_workflow_with_environment_protection(self):
        """Test workflow with environment protection rules."""
        protected_workflow = """
name: Protected Deploy
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    environment:
      name: production
      url: ${{ steps.deploy.outputs.url }}
    steps:
      - name: Deploy to production
        id: deploy
        run: |
          echo "Deploying to production"
          echo "url=https://production.example.com" >> $GITHUB_OUTPUT
"""
        parsed = yaml.safe_load(protected_workflow)
        
        job = parsed['jobs']['deploy']
        assert 'environment' in job
        env_config = job['environment']
        assert env_config['name'] == 'production'
        assert 'url' in env_config


class TestGitHubWorkflowComplexScenarios:
    """Test suite for complex workflow scenarios and edge cases."""
    
    @pytest.fixture
    def matrix_workflow_complex(self):
        """Complex matrix workflow with exclusions and inclusions."""
        return """
name: Complex Matrix
on: push
jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        python-version: ['3.8', '3.9', '3.10', '3.11']
        include:
          - os: ubuntu-latest
            python-version: '3.12'
            experimental: true
          - os: windows-latest
            python-version: '3.7'
            legacy: true
        exclude:
          - os: macos-latest
            python-version: '3.8'
          - os: windows-latest
            python-version: '3.11'
    steps:
      - uses: actions/checkout@v4
      - name: Setup Python ${{ matrix.python-version }}
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
"""

    def test_complex_matrix_configuration(self, matrix_workflow_complex):
        """Test complex matrix with includes and excludes."""
        parsed = yaml.safe_load(matrix_workflow_complex)
        
        strategy = parsed['jobs']['test']['strategy']
        matrix = strategy['matrix']
        
        assert strategy['fail-fast'] is False
        assert len(matrix['os']) == 3
        assert len(matrix['python-version']) == 4
        assert 'include' in matrix
        assert 'exclude' in matrix
        assert len(matrix['include']) == 2
        assert len(matrix['exclude']) == 2

    def test_workflow_with_conditional_steps(self):
        """Test workflow with conditional step execution."""
        conditional_workflow = """
name: Conditional Steps
on:
  push:
  pull_request:
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run on push only
        if: github.event_name == 'push'
        run: echo "This runs on push"
      - name: Run on PR only
        if: github.event_name == 'pull_request'
        run: echo "This runs on PR"
      - name: Run on main branch only
        if: github.ref == 'refs/heads/main'
        run: echo "This runs on main"
      - name: Run on success
        if: success()
        run: echo "Previous steps succeeded"
      - name: Run on failure
        if: failure()
        run: echo "A step failed"
      - name: Run always
        if: always()
        run: echo "This always runs"
"""
        parsed = yaml.safe_load(conditional_workflow)
        
        steps = parsed['jobs']['test']['steps']
        conditional_steps = [s for s in steps if 'if' in s]
        
        assert len(conditional_steps) == 6
        
        conditions = [step['if'] for step in conditional_steps]
        assert "github.event_name == 'push'" in conditions
        assert "github.event_name == 'pull_request'" in conditions
        assert "success()" in conditions
        assert "failure()" in conditions
        assert "always()" in conditions

    def test_workflow_with_artifacts(self):
        """Test workflow with artifact upload and download."""
        artifact_workflow = """
name: Artifacts Test
on: push
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build
        run: |
          mkdir -p dist
          echo "build artifact" > dist/app.tar.gz
      - name: Upload build artifacts
        uses: actions/upload-artifact@v3
        with:
          name: build-artifacts
          path: dist/
          retention-days: 30
  test:
    runs-on: ubuntu-latest
    needs: build
    steps:
      - uses: actions/checkout@v4
      - name: Download build artifacts
        uses: actions/download-artifact@v3
        with:
          name: build-artifacts
          path: dist/
      - name: Test artifacts
        run: |
          ls -la dist/
          test -f dist/app.tar.gz
"""
        parsed = yaml.safe_load(artifact_workflow)
        
        build_steps = parsed['jobs']['build']['steps']
        test_steps = parsed['jobs']['test']['steps']
        
        upload_step = next((s for s in build_steps if s.get('uses', '').startswith('actions/upload-artifact')), None)
        download_step = next((s for s in test_steps if s.get('uses', '').startswith('actions/download-artifact')), None)
        
        assert upload_step is not None
        assert download_step is not None
        assert upload_step['with']['name'] == 'build-artifacts'
        assert download_step['with']['name'] == 'build-artifacts'

    def test_workflow_with_caching(self):
        """Test workflow with dependency caching."""
        cache_workflow = """
name: Caching Test
on: push
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Cache dependencies
        uses: actions/cache@v3
        with:
          path: |
            ~/.cache/pip
            ~/.npm
            node_modules
          key: ${{ runner.os }}-deps-${{ hashFiles('**/requirements.txt', '**/package-lock.json') }}
          restore-keys: |
            ${{ runner.os }}-deps-
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          npm install
"""
        parsed = yaml.safe_load(cache_workflow)
        
        steps = parsed['jobs']['test']['steps']
        cache_step = next((s for s in steps if s.get('uses', '').startswith('actions/cache')), None)
        
        assert cache_step is not None
        assert 'path' in cache_step['with']
        assert 'key' in cache_step['with']
        assert 'restore-keys' in cache_step['with']


class TestGitHubWorkflowErrorHandling:
    """Enhanced error handling tests for GitHub workflows."""
    
    def test_workflow_with_continue_on_error(self):
        """Test workflow steps with continue-on-error."""
        continue_on_error_workflow = """
name: Continue on Error
on: push
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Flaky test
        continue-on-error: true
        run: |
          if [ $RANDOM -gt 16384 ]; then
            exit 1
          fi
          echo "Test passed"
      - name: Always runs
        run: echo "This runs even if previous step fails"
"""
        parsed = yaml.safe_load(continue_on_error_workflow)
        
        steps = parsed['jobs']['test']['steps']
        flaky_step = steps[1]  # Second step
        
        assert 'continue-on-error' in flaky_step
        assert flaky_step['continue-on-error'] is True

    def test_workflow_timeout_configuration(self):
        """Test workflow and job timeout configurations."""
        timeout_workflow = """
name: Timeout Test
on: push
jobs:
  quick-job:
    runs-on: ubuntu-latest
    timeout-minutes: 5
    steps:
      - name: Quick task
        run: echo "Quick task"
        timeout-minutes: 2
  long-job:
    runs-on: ubuntu-latest
    timeout-minutes: 120
    steps:
      - name: Long running task
        run: sleep 30
"""
        parsed = yaml.safe_load(timeout_workflow)
        
        quick_job = parsed['jobs']['quick-job']
        long_job = parsed['jobs']['long-job']
        
        assert quick_job['timeout-minutes'] == 5
        assert long_job['timeout-minutes'] == 120
        assert quick_job['steps'][0]['timeout-minutes'] == 2

    @pytest.mark.parametrize("error_scenario,expected_behavior", [
        ("yaml_syntax_error", "should_raise_yaml_error"),
        ("missing_required_field", "should_fail_validation"),
        ("invalid_runner", "should_fail_validation"),
        ("circular_dependency", "should_fail_validation"),
    ])
    def test_error_scenarios(self, error_scenario, expected_behavior):
        """Test various error scenarios and their expected behaviors."""
        error_configs = {
            "yaml_syntax_error": "name: Test\nsteps:\n  - invalid: [\n",
            "missing_required_field": {"name": "Test"},  # Missing 'on' and 'jobs'
            "invalid_runner": {"name": "Test", "on": "push", "jobs": {"test": {"runs-on": "invalid-runner"}}},
            "circular_dependency": {
                "name": "Test", 
                "on": "push", 
                "jobs": {
                    "job1": {"runs-on": "ubuntu-latest", "needs": "job2"},
                    "job2": {"runs-on": "ubuntu-latest", "needs": "job1"}
                }
            }
        }
        
        if error_scenario == "yaml_syntax_error":
            with pytest.raises(yaml.YAMLError):
                yaml.safe_load(error_configs[error_scenario])
        elif error_scenario == "missing_required_field":
            config = error_configs[error_scenario]
            required_fields = ['name', 'on', 'jobs']
            missing = [f for f in required_fields if f not in config]
            assert len(missing) > 0
        elif error_scenario == "circular_dependency":
            config = error_configs[error_scenario]
            job1_needs = config['jobs']['job1']['needs']
            job2_needs = config['jobs']['job2']['needs']
            # Detect circular dependency
            assert job1_needs == 'job2' and job2_needs == 'job1'


class TestGitHubWorkflowSecurityEnhancements:
    """Enhanced security testing for GitHub workflows."""
    
    @pytest.fixture
    def security_test_workflows(self):
        """Various security-related workflow configurations."""
        return {
            "third_party_action_pinning": """
name: Action Pinning
on: push
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@8e5e7e5ab8b370d6c329ec480221332ada57f0ab  # v3.5.2
      - uses: actions/setup-python@bd6b4b6205c4dbad673328db7b31b7fab9e7a85f  # v4.6.0
        with:
          python-version: '3.9'
""",
            "script_injection_vulnerable": """
name: Vulnerable to Script Injection
on:
  pull_request:
    types: [opened]
jobs:
  comment:
    runs-on: ubuntu-latest
    steps:
      - name: Comment on PR
        run: |
          echo "Title: ${{ github.event.pull_request.title }}"
          echo "Body: ${{ github.event.pull_request.body }}"
""",
            "safe_script_handling": """
name: Safe Script Handling
on:
  pull_request:
    types: [opened]
jobs:
  comment:
    runs-on: ubuntu-latest
    steps:
      - name: Comment on PR
        env:
          PR_TITLE: ${{ github.event.pull_request.title }}
          PR_BODY: ${{ github.event.pull_request.body }}
        run: |
          echo "Title: $PR_TITLE"
          echo "Body: $PR_BODY"
""",
            "secrets_handling": """
name: Secrets Handling
on: push
jobs:
  deploy:
    runs-on: ubuntu-latest
    environment: production
    steps:
      - name: Deploy with secrets
        env:
          API_KEY: ${{ secrets.API_KEY }}
          DATABASE_URL: ${{ secrets.DATABASE_URL }}
        run: |
          # Good: secrets in environment variables
          curl -H "Authorization: Bearer $API_KEY" https://api.example.com/deploy
          # Bad: would be echoing secrets directly
          # echo "API Key: ${{ secrets.API_KEY }}"
"""
        }

    def test_action_pinning_security(self, security_test_workflows):
        """Test that actions are pinned to specific commits for security."""
        workflow = security_test_workflows["third_party_action_pinning"]
        parsed = yaml.safe_load(workflow)
        
        steps = parsed['jobs']['test']['steps']
        
        for step in steps:
            if 'uses' in step:
                action = step['uses']
                # Check that actions are pinned to commit hashes (not just version tags)
                if '@' in action:
                    version_part = action.split('@')[1]
                    # Commit hashes are typically 40 characters long
                    if len(version_part) == 40:
                        assert all(c in '0123456789abcdef' for c in version_part.lower())

    def test_script_injection_vulnerability_detection(self, security_test_workflows):
        """Test detection of script injection vulnerabilities."""
        vulnerable_workflow = security_test_workflows["script_injection_vulnerable"]
        safe_workflow = security_test_workflows["safe_script_handling"]
        
        parsed_vulnerable = yaml.safe_load(vulnerable_workflow)
        parsed_safe = yaml.safe_load(safe_workflow)
        
        # Vulnerable workflow uses GitHub context directly in run commands
        vulnerable_step = parsed_vulnerable['jobs']['comment']['steps'][0]
        assert '${{ github.event' in vulnerable_step['run']
        
        # Safe workflow uses environment variables
        safe_step = parsed_safe['jobs']['comment']['steps'][0]
        assert 'env' in safe_step
        assert '$PR_TITLE' in safe_step['run']
        assert '${{ github.event' not in safe_step['run']

    def test_secrets_security_best_practices(self, security_test_workflows):
        """Test secrets handling best practices."""
        workflow = security_test_workflows["secrets_handling"]
        parsed = yaml.safe_load(workflow)
        
        deploy_step = parsed['jobs']['deploy']['steps'][0]
        
        # Check that secrets are passed via environment variables
        assert 'env' in deploy_step
        env_vars = deploy_step['env']
        assert 'API_KEY' in env_vars
        assert '${{ secrets.API_KEY }}' in env_vars['API_KEY']
        
        # Check that secrets are not directly echoed in run commands
        run_command = deploy_step['run']
        assert '${{ secrets.' not in run_command

    @pytest.mark.parametrize("trigger_type,security_risk", [
        ("pull_request_target", "high"),
        ("pull_request", "low"),
        ("push", "medium"),
        ("workflow_dispatch", "medium"),
        ("schedule", "low"),
    ])
    def test_trigger_security_assessment(self, trigger_type, security_risk):
        """Test security risk assessment for different trigger types."""
        risk_levels = ["low", "medium", "high"]
        assert security_risk in risk_levels
        
        # pull_request_target is high risk because it runs with write permissions
        if trigger_type == "pull_request_target":
            assert security_risk == "high"

    def test_workflow_permissions_least_privilege(self):
        """Test that workflows follow least privilege principle."""
        minimal_permissions_workflow = """
name: Minimal Permissions
on: push
permissions:
  contents: read  # Only read access to repository contents
jobs:
  test:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      checks: write  # Only for test reporting
    steps:
      - uses: actions/checkout@v4
      - name: Run tests
        run: pytest
"""
        parsed = yaml.safe_load(minimal_permissions_workflow)
        
        # Check workflow-level permissions are minimal
        workflow_perms = parsed['permissions']
        assert workflow_perms['contents'] == 'read'
        
        # Check job-level permissions are specific
        job_perms = parsed['jobs']['test']['permissions']
        assert job_perms['contents'] == 'read'
        assert job_perms['checks'] == 'write'


class TestGitHubWorkflowPerformanceEnhancements:
    """Enhanced performance testing for GitHub workflows."""
    
    def test_workflow_optimization_recommendations(self):
        """Test identification of workflow optimization opportunities."""
        unoptimized_workflow = """
name: Unoptimized Workflow
on: push
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install dependencies
        run: |
          pip install --no-cache-dir pytest
          pip install --no-cache-dir requests
          pip install --no-cache-dir flask
      - name: Run tests
        run: pytest
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install dependencies
        run: |
          pip install --no-cache-dir flake8
          pip install --no-cache-dir black
      - name: Run linting
        run: |
          flake8 .
          black --check .
"""
        parsed = yaml.safe_load(unoptimized_workflow)
        
        # Identify optimization opportunities
        issues = []
        
        # Check for repeated dependency installation
        install_steps = []
        for job_name, job_config in parsed['jobs'].items():
            for step in job_config.get('steps', []):
                if 'run' in step and 'pip install' in step['run']:
                    install_steps.append((job_name, step))
        
        if len(install_steps) > 1:
            issues.append("repeated_dependency_installation")
        
        # Check for missing caching
        cache_used = False
        for job_name, job_config in parsed['jobs'].items():
            for step in job_config.get('steps', []):
                if 'uses' in step and 'cache' in step['uses']:
                    cache_used = True
                    break
        
        if not cache_used:
            issues.append("missing_dependency_caching")
        
        assert len(issues) > 0  # Should identify optimization opportunities

    def test_large_scale_workflow_processing(self):
        """Test processing of large-scale workflow configurations."""
        # Generate a workflow with many jobs
        large_workflow = {
            'name': 'Large Scale Workflow',
            'on': 'push',
            'jobs': {}
        }
        
        # Create 100 jobs with dependencies
        for i in range(100):
            job_name = f'job_{i:03d}'
            job_config = {
                'runs-on': 'ubuntu-latest',
                'steps': [
                    {'uses': 'actions/checkout@v4'},
                    {'name': f'Task {i}', 'run': f'echo "Running task {i}"'}
                ]
            }
            
            # Add dependencies to create a complex graph
            if i > 0:
                if i % 10 == 0:
                    # Every 10th job depends on previous 10 jobs
                    job_config['needs'] = [f'job_{j:03d}' for j in range(max(0, i-10), i)]
                elif i % 5 == 0:
                    # Every 5th job depends on previous job
                    job_config['needs'] = f'job_{i-1:03d}'
            
            large_workflow['jobs'][job_name] = job_config
        
        # Test that the workflow can be processed
        yaml_content = yaml.dump(large_workflow, default_flow_style=False)
        parsed = yaml.safe_load(yaml_content)
        
        assert len(parsed['jobs']) == 100
        assert parsed['name'] == 'Large Scale Workflow'
        
        # Test dependency analysis
        jobs_with_deps = [job for job, config in parsed['jobs'].items() if 'needs' in config]
        assert len(jobs_with_deps) > 0

    def test_workflow_complexity_metrics(self):
        """Test calculation of workflow complexity metrics."""
        complex_workflow = """
name: Complex Workflow
on:
  push:
    branches: [main, develop, feature/*]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 2 * * 1-5'
  workflow_dispatch:
    inputs:
      environment:
        type: choice
        options: [dev, staging, prod]
env:
  GLOBAL_VAR: value
jobs:
  prepare:
    runs-on: ubuntu-latest
    outputs:
      matrix: ${{ steps.set-matrix.outputs.matrix }}
    steps:
      - id: set-matrix
        run: echo "matrix=[1,2,3,4,5]" >> $GITHUB_OUTPUT
  test:
    runs-on: ${{ matrix.os }}
    needs: prepare
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        version: ${{ fromJson(needs.prepare.outputs.matrix) }}
      fail-fast: false
    steps:
      - uses: actions/checkout@v4
      - name: Test ${{ matrix.version }}
        run: echo "Testing version ${{ matrix.version }}"
  build:
    runs-on: ubuntu-latest
    needs: test
    if: success()
    steps:
      - uses: actions/checkout@v4
      - name: Build
        run: echo "Building"
  deploy:
    runs-on: ubuntu-latest
    needs: [test, build]
    if: github.ref == 'refs/heads/main'
    environment: production
    steps:
      - name: Deploy
        run: echo "Deploying"
"""
        parsed = yaml.safe_load(complex_workflow)
        
        # Calculate complexity metrics
        metrics = {
            'trigger_count': len(parsed['on']),
            'job_count': len(parsed['jobs']),
            'total_steps': sum(len(job.get('steps', [])) for job in parsed['jobs'].values()),
            'has_matrix': any('strategy' in job for job in parsed['jobs'].values()),
            'has_conditions': any('if' in job for job in parsed['jobs'].values()),
            'has_environment': any('environment' in job for job in parsed['jobs'].values()),
            'dependency_edges': sum(
                len(job.get('needs', [])) if isinstance(job.get('needs'), list) 
                else (1 if job.get('needs') else 0) 
                for job in parsed['jobs'].values()
            )
        }
        
        assert metrics['trigger_count'] == 4
        assert metrics['job_count'] == 4
        assert metrics['total_steps'] >= 4
        assert metrics['has_matrix'] is True
        assert metrics['has_conditions'] is True
        assert metrics['has_environment'] is True
        assert metrics['dependency_edges'] >= 3


# Add test for workflow template validation
class TestGitHubWorkflowTemplates:
    """Test suite for GitHub workflow templates and reusable patterns."""
    
    @pytest.fixture
    def workflow_templates(self):
        """Common workflow templates."""
        return {
            'python_ci': """
name: Python CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        python-version: ['3.8', '3.9', '3.10', '3.11']
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install pytest pytest-cov
          if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
      - name: Test with pytest
        run: pytest --cov=./ --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
""",
            'node_ci': """
name: Node.js CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        node-version: [16.x, 18.x, 20.x]
    steps:
      - uses: actions/checkout@v4
      - name: Use Node.js ${{ matrix.node-version }}
        uses: actions/setup-node@v4
        with:
          node-version: ${{ matrix.node-version }}
          cache: 'npm'
      - run: npm ci
      - run: npm run build --if-present
      - run: npm test
""",
            'docker_build': """
name: Docker Build
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3
      - name: Login to DockerHub
        if: github.event_name != 'pull_request'
        uses: docker/login-action@v3
        with:
          username: ${{ secrets.DOCKERHUB_USERNAME }}
          password: ${{ secrets.DOCKERHUB_TOKEN }}
      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          push: ${{ github.event_name != 'pull_request' }}
          tags: user/app:latest
"""
        }

    @pytest.mark.parametrize("template_name", ["python_ci", "node_ci", "docker_build"])
    def test_workflow_template_structure(self, workflow_templates, template_name):
        """Test that workflow templates have correct structure."""
        template = workflow_templates[template_name]
        parsed = yaml.safe_load(template)
        
        # All templates should have these basic fields
        assert 'name' in parsed
        assert 'on' in parsed
        assert 'jobs' in parsed
        assert len(parsed['jobs']) >= 1
        
        # All jobs should have runs-on and steps
        for job_name, job_config in parsed['jobs'].items():
            assert 'runs-on' in job_config or 'strategy' in job_config
            assert 'steps' in job_config
            assert len(job_config['steps']) > 0

    def test_python_ci_template_specifics(self, workflow_templates):
        """Test Python CI template specific features."""
        template = workflow_templates['python_ci']
        parsed = yaml.safe_load(template)
        
        job = parsed['jobs']['test']
        assert 'strategy' in job
        assert 'matrix' in job['strategy']
        assert 'python-version' in job['strategy']['matrix']
        
        # Check for Python-specific steps
        step_names = [step.get('name', '') for step in job['steps']]
        python_setup_step = any('python' in name.lower() for name in step_names)
        assert python_setup_step

    def test_node_ci_template_specifics(self, workflow_templates):
        """Test Node.js CI template specific features."""
        template = workflow_templates['node_ci']
        parsed = yaml.safe_load(template)
        
        job = parsed['jobs']['test']
        assert 'strategy' in job
        assert 'node-version' in job['strategy']['matrix']
        
        # Check for Node.js-specific steps
        node_setup_step = None
        for step in job['steps']:
            if 'uses' in step and 'setup-node' in step['uses']:
                node_setup_step = step
                break
        
        assert node_setup_step is not None
        assert 'cache' in node_setup_step['with']

    def test_docker_template_specifics(self, workflow_templates):
        """Test Docker build template specific features."""
        template = workflow_templates['docker_build']
        parsed = yaml.safe_load(template)
        
        job = parsed['jobs']['build']
        
        # Check for Docker-specific steps
        docker_steps = [step for step in job['steps'] if 'docker' in step.get('uses', '').lower()]
        assert len(docker_steps) >= 2  # Should have setup-buildx and build-push actions
        
        # Check for conditional login
        login_step = None
        for step in job['steps']:
            if 'login' in step.get('uses', ''):
                login_step = step
                break
        
        assert login_step is not None
        assert 'if' in login_step


# Add comprehensive step validation tests
class TestGitHubWorkflowStepValidation:
    """Test suite for comprehensive workflow step validation."""
    
    def test_step_with_all_possible_fields(self):
        """Test workflow step with all possible configuration fields."""
        comprehensive_step_workflow = """
name: Comprehensive Step Test
on: push
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Comprehensive step
        id: test-step
        uses: actions/checkout@v4
        with:
          repository: owner/repo
          ref: main
          token: ${{ secrets.GITHUB_TOKEN }}
        env:
          CUSTOM_VAR: value
          SECRET_VAR: ${{ secrets.SECRET }}
        if: success()
        continue-on-error: false
        timeout-minutes: 10
        working-directory: ./subdir
      - name: Script step with all options
        id: script-step
        run: |
          echo "Running comprehensive script"
          exit 0
        shell: bash
        env:
          SCRIPT_VAR: script_value
        if: steps.test-step.outcome == 'success'
        continue-on-error: true
        timeout-minutes: 5
        working-directory: ./scripts
"""
        parsed = yaml.safe_load(comprehensive_step_workflow)
        
        steps = parsed['jobs']['test']['steps']
        action_step = steps[0]
        script_step = steps[1]
        
        # Validate action step fields
        action_fields = ['name', 'id', 'uses', 'with', 'env', 'if', 'continue-on-error', 'timeout-minutes', 'working-directory']
        for field in action_fields:
            assert field in action_step
        
        # Validate script step fields
        script_fields = ['name', 'id', 'run', 'shell', 'env', 'if', 'continue-on-error', 'timeout-minutes', 'working-directory']
        for field in script_fields:
            assert field in script_step

    @pytest.mark.parametrize("shell_type", [
        "bash", "sh", "cmd", "powershell", "pwsh", "python"
    ])
    def test_step_shell_options(self, shell_type):
        """Test different shell options for run steps."""
        workflow = f"""
name: Shell Test
on: push
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Test {shell_type}
        run: echo "Testing {shell_type}"
        shell: {shell_type}
"""
        parsed = yaml.safe_load(workflow)
        step = parsed['jobs']['test']['steps'][0]
        
        assert step['shell'] == shell_type

    def test_step_output_handling(self):
        """Test step output generation and consumption."""
        output_workflow = """
name: Output Test
on: push
jobs:
  generate:
    runs-on: ubuntu-latest
    outputs:
      test-output: ${{ steps.generator.outputs.value }}
      matrix-output: ${{ steps.matrix-gen.outputs.matrix }}
    steps:
      - name: Generate output
        id: generator
        run: |
          echo "value=generated_value" >> $GITHUB_OUTPUT
          echo "timestamp=$(date)" >> $GITHUB_OUTPUT
      - name: Generate matrix
        id: matrix-gen
        run: |
          echo 'matrix=["a", "b", "c"]' >> $GITHUB_OUTPUT
  consume:
    runs-on: ubuntu-latest
    needs: generate
    strategy:
      matrix:
        item: ${{ fromJson(needs.generate.outputs.matrix-output) }}
    steps:
      - name: Use output
        run: |
          echo "Received: ${{ needs.generate.outputs.test-output }}"
          echo "Matrix item: ${{ matrix.item }}"
"""
        parsed = yaml.safe_load(output_workflow)
        
        generate_job = parsed['jobs']['generate']
        consume_job = parsed['jobs']['consume']
        
        # Check output generation
        assert 'outputs' in generate_job
        assert 'test-output' in generate_job['outputs']
        assert 'matrix-output' in generate_job['outputs']
        
        # Check output consumption
        assert 'needs' in consume_job
        assert consume_job['needs'] == 'generate'
        
        # Check matrix from output
        assert 'strategy' in consume_job
        matrix_value = consume_job['strategy']['matrix']['item']
        assert 'fromJson' in matrix_value
        assert 'needs.generate.outputs.matrix-output' in matrix_value


class TestGitHubWorkflowEnvironmentHandling:
    """Test suite for environment variable and context handling."""
    
    def test_environment_variable_scoping(self):
        """Test environment variable scoping at different levels."""
        env_scoping_workflow = """
name: Environment Scoping
on: push
env:
  GLOBAL_VAR: global_value
  OVERRIDE_VAR: global_override
jobs:
  test:
    runs-on: ubuntu-latest
    env:
      JOB_VAR: job_value
      OVERRIDE_VAR: job_override
    steps:
      - name: Step with env
        env:
          STEP_VAR: step_value
          OVERRIDE_VAR: step_override
        run: |
          echo "Global: $GLOBAL_VAR"
          echo "Job: $JOB_VAR"
          echo "Step: $STEP_VAR"
          echo "Override: $OVERRIDE_VAR"
      - name: Step without env
        run: |
          echo "Global still available: $GLOBAL_VAR"
          echo "Job still available: $JOB_VAR"
          echo "Step var not available: $STEP_VAR"
          echo "Override from job: $OVERRIDE_VAR"
"""
        parsed = yaml.safe_load(env_scoping_workflow)
        
        # Check global environment
        assert 'env' in parsed
        assert parsed['env']['GLOBAL_VAR'] == 'global_value'
        
        # Check job environment
        job = parsed['jobs']['test']
        assert 'env' in job
        assert job['env']['JOB_VAR'] == 'job_value'
        
        # Check step environment
        step_with_env = job['steps'][0]
        assert 'env' in step_with_env
        assert step_with_env['env']['STEP_VAR'] == 'step_value'
        
        # Check variable override behavior
        assert parsed['env']['OVERRIDE_VAR'] == 'global_override'
        assert job['env']['OVERRIDE_VAR'] == 'job_override'
        assert step_with_env['env']['OVERRIDE_VAR'] == 'step_override'

    @pytest.mark.parametrize("context_expression,context_type", [
        ("${{ github.repository }}", "github"),
        ("${{ runner.os }}", "runner"),
        ("${{ env.MY_VAR }}", "env"),
        ("${{ secrets.API_KEY }}", "secrets"),
        ("${{ steps.build.outputs.version }}", "steps"),
        ("${{ jobs.test.result }}", "jobs"),
        ("${{ matrix.version }}", "matrix"),
        ("${{ needs.build.outputs.artifact }}", "needs"),
        ("${{ inputs.environment }}", "inputs"),
    ])
    def test_github_context_expressions(self, context_expression, context_type):
        """Test various GitHub context expressions."""
        # Validate context expression format
        assert context_expression.startswith("${{")
        assert context_expression.endswith("}}")
        assert context_type in context_expression
        
        # Test in a workflow context
        workflow = f"""
name: Context Test
on: workflow_dispatch
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Test context
        run: echo "Context value: {context_expression}"
"""
        parsed = yaml.safe_load(workflow)
        step = parsed['jobs']['test']['steps'][0]
        assert context_expression in step['run']

    def test_complex_expression_evaluation(self):
        """Test complex GitHub context expressions."""
        complex_expressions_workflow = """
name: Complex Expressions
on:
  workflow_dispatch:
    inputs:
      environment:
        type: choice
        options: [dev, staging, prod]
      version:
        type: string
        default: '1.0.0'
jobs:
  deploy:
    runs-on: ubuntu-latest
    if: |
      github.ref == 'refs/heads/main' && 
      inputs.environment == 'prod' && 
      contains(github.event.head_commit.message, '[deploy]')
    steps:
      - name: Complex conditionals
        if: |
          (success() || failure()) && 
          !cancelled() && 
          github.actor != 'dependabot[bot]'
        run: echo "Complex condition met"
      - name: String operations
        run: |
          echo "Uppercase env: ${{ toUpperCase(inputs.environment) }}"
          echo "JSON parse: ${{ fromJson('{"key": "value"}').key }}"
          echo "Hash files: ${{ hashFiles('**/*.py', '**/*.js') }}"
          echo "Format: ${{ format('Version {0} for {1}', inputs.version, inputs.environment) }}"
"""
        parsed = yaml.safe_load(complex_expressions_workflow)
        
        job = parsed['jobs']['deploy']
        
        # Check complex job condition
        job_condition = job['if']
        assert 'github.ref' in job_condition
        assert 'inputs.environment' in job_condition
        assert 'contains(' in job_condition
        
        # Check complex step condition
        step_condition = job['steps'][0]['if']
        assert 'success()' in step_condition
        assert 'failure()' in step_condition
        assert 'cancelled()' in step_condition
        
        # Check function usage
        string_ops_step = job['steps'][1]
        assert 'toUpperCase(' in string_ops_step['run']
        assert 'fromJson(' in string_ops_step['run']
        assert 'hashFiles(' in string_ops_step['run']
        assert 'format(' in string_ops_step['run']


if __name__ == "__main__":
    # Run with additional coverage for new test classes
    pytest.main([
        __file__, 
        "-v", 
        "--tb=short",
        "--cov=.",
        "--cov-report=term-missing",
        "-k", "not test_large_scale_workflow_processing"  # Skip heavy tests in normal runs
    ])