"""
Comprehensive unit tests for dependabot configuration validation.
Testing framework: pytest
"""

import os
from pathlib import Path
from typing import Any, Dict, List

import pytest
import yaml


class TestDependabotConfig:
    """Test suite for dependabot configuration validation."""

    @pytest.fixture
    def dependabot_config_path(self) -> Path:
        """Fixture to provide the path to the dependabot configuration file."""
        return Path(".github/dependabot.yml")

    @pytest.fixture
    def dependabot_config(self, dependabot_config_path: Path) -> Dict[str, Any]:
        """Fixture to load and parse the dependabot configuration."""
        if not dependabot_config_path.exists():
            pytest.skip("Dependabot configuration file not found")

        with dependabot_config_path.open("r") as f:
            return yaml.safe_load(f)

    def test_config_file_exists(self, dependabot_config_path: Path) -> None:
        """Test that the dependabot configuration file exists."""
        assert dependabot_config_path.exists(), "Dependabot configuration file should exist"

    def test_config_file_is_valid_yaml(self, dependabot_config_path: Path) -> None:
        """Test that the dependabot configuration file is valid YAML."""
        try:
            with dependabot_config_path.open("r") as f:
                yaml.safe_load(f)
        except yaml.YAMLError as e:
            pytest.fail(f"Invalid YAML syntax: {e}")

    def test_config_has_required_version_key(self, dependabot_config: Dict[str, Any]) -> None:
        """Test that the configuration has the required 'version' key."""
        assert "version" in dependabot_config, "Configuration must have 'version' key"

    def test_config_version_is_supported(self, dependabot_config: Dict[str, Any]) -> None:
        """Test that the configuration uses a supported version."""
        supported_versions = [2]
        version = dependabot_config.get("version")
        assert version in supported_versions, (
            f"Version {version} is not supported. Supported versions: {supported_versions}"
        )

    def test_config_has_updates_key(self, dependabot_config: Dict[str, Any]) -> None:
        """Test that the configuration has the 'updates' key."""
        assert "updates" in dependabot_config, "Configuration must have 'updates' key"

    def test_updates_is_list(self, dependabot_config: Dict[str, Any]) -> None:
        """Test that the 'updates' value is a list."""
        updates = dependabot_config.get("updates", [])
        assert isinstance(updates, list), "The 'updates' value must be a list"

    def test_updates_list_not_empty(self, dependabot_config: Dict[str, Any]) -> None:
        """Test that the updates list is not empty."""
        updates = dependabot_config.get("updates", [])
        assert updates, "The 'updates' list should not be empty"

    def test_each_update_has_required_keys(self, dependabot_config: Dict[str, Any]) -> None:
        """Test that each update entry has required keys."""
        required_keys = ["package-ecosystem", "directory", "schedule"]
        updates = dependabot_config.get("updates", [])

        for i, update in enumerate(updates):
            assert isinstance(update, dict), f"Update {i} must be a dictionary"
            for key in required_keys:
                assert key in update, f"Update {i} missing required key: {key}"

    def test_package_ecosystem_values_are_valid(self, dependabot_config: Dict[str, Any]) -> None:
        """Test that package-ecosystem values are valid."""
        valid_ecosystems = [
            "bundler",
            "cargo",
            "composer",
            "docker",
            "elm",
            "gitsubmodule",
            "github-actions",
            "gomod",
            "gradle",
            "maven",
            "mix",
            "npm",
            "nuget",
            "pip",
            "swift",
            "terraform",
        ]

        updates = dependabot_config.get("updates", [])
        for i, update in enumerate(updates):
            ecosystem = update.get("package-ecosystem")
            assert ecosystem in valid_ecosystems, (
                f"Update {i} has invalid package-ecosystem: {ecosystem}"
            )

    def test_directory_values_are_valid_paths(self, dependabot_config: Dict[str, Any]) -> None:
        """Test that directory values are valid paths."""
        updates = dependabot_config.get("updates", [])
        for i, update in enumerate(updates):
            directory = update.get("directory")
            assert isinstance(directory, str), f"Update {i} directory must be a string"
            assert directory.startswith(
                "/"
            ), f"Update {i} directory must start with '/'"

    def test_schedule_has_required_interval(self, dependabot_config: Dict[str, Any]) -> None:
        """Test that each schedule has a required interval."""
        updates = dependabot_config.get("updates", [])
        for i, update in enumerate(updates):
            schedule = update.get("schedule", {})
            assert isinstance(
                schedule, dict
            ), f"Update {i} schedule must be a dictionary"
            assert "interval" in schedule, f"Update {i} schedule missing 'interval' key"

    def test_schedule_interval_values_are_valid(self, dependabot_config: Dict[str, Any]) -> None:
        """Test that schedule interval values are valid."""
        valid_intervals = ["daily", "weekly", "monthly"]
        updates = dependabot_config.get("updates", [])

        for i, update in enumerate(updates):
            interval = update.get("schedule", {}).get("interval")
            assert interval in valid_intervals, (
                f"Update {i} has invalid schedule interval: {interval}"
            )

    def test_schedule_day_is_valid_when_present(self, dependabot_config: Dict[str, Any]) -> None:
        """Test that schedule day values are valid when present."""
        valid_days = [
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday",
        ]
        updates = dependabot_config.get("updates", [])

        for i, update in enumerate(updates):
            schedule = update.get("schedule", {})
            if "day" in schedule:
                day = schedule["day"]
                assert day in valid_days, f"Update {i} has invalid schedule day: {day}"

    def test_schedule_time_format_is_valid_when_present(self, dependabot_config: Dict[str, Any]) -> None:
        """Test that schedule time values are in valid format when present."""
        import re

        time_pattern = re.compile(r"^([01]?[0-9]|2[0-3]):[0-5][0-9]$")
        updates = dependabot_config.get("updates", [])

        for i, update in enumerate(updates):
            schedule = update.get("schedule", {})
            if "time" in schedule:
                time = schedule["time"]
                assert isinstance(time, str), (
                    f"Update {i} schedule time must be a string"
                )
                assert time_pattern.match(time), (
                    f"Update {i} has invalid time format: {time}"
                )

    def test_schedule_timezone_is_valid_when_present(self, dependabot_config: Dict[str, Any]) -> None:
        """Test that schedule timezone values are valid when present."""
        updates = dependabot_config.get("updates", [])

        for i, update in enumerate(updates):
            timezone = update.get("schedule", {}).get("timezone")
            if timezone is not None:
                assert isinstance(timezone, str), (
                    f"Update {i} schedule timezone must be a string"
                )
                assert timezone, (
                    f"Update {i} schedule timezone cannot be empty"
                )

    def test_open_pull_requests_limit_is_valid_when_present(self, dependabot_config: Dict[str, Any]) -> None:
        """Test that open-pull-requests-limit values are valid when present."""
        updates = dependabot_config.get("updates", [])

        for i, update in enumerate(updates):
            if "open-pull-requests-limit" in update:
                limit = update["open-pull-requests-limit"]
                assert isinstance(limit, int), (
                    f"Update {i} open-pull-requests-limit must be an integer"
                )
                assert 0 <= limit <= 20, (
                    f"Update {i} open-pull-requests-limit must be between 0 and 20"
                )

    def test_target_branch_is_valid_when_present(self, dependabot_config: Dict[str, Any]) -> None:
        """Test that target-branch values are valid when present."""
        updates = dependabot_config.get("updates", [])

        for i, update in enumerate(updates):
            if "target-branch" in update:
                branch = update["target-branch"]
                assert isinstance(branch, str), (
                    f"Update {i} target-branch must be a string"
                )
                assert branch, f"Update {i} target-branch cannot be empty"

    def test_reviewers_format_is_valid_when_present(self, dependabot_config: Dict[str, Any]) -> None:
        """Test that reviewers format is valid when present."""
        updates = dependabot_config.get("updates", [])

        for i, update in enumerate(updates):
            reviewers = update.get("reviewers")
            if reviewers is not None:
                assert isinstance(reviewers, list), (
                    f"Update {i} reviewers must be a list"
                )
                for reviewer in reviewers:
                    assert isinstance(reviewer, str), (
                        f"Update {i} reviewer must be a string"
                    )

    def test_assignees_format_is_valid_when_present(self, dependabot_config: Dict[str, Any]) -> None:
        """Test that assignees format is valid when present."""
        updates = dependabot_config.get("updates", [])

        for i, update in enumerate(updates):
            assignees = update.get("assignees")
            if assignees is not None:
                assert isinstance(assignees, list), (
                    f"Update {i} assignees must be a list"
                )
                for assignee in assignees:
                    assert isinstance(assignee, str), (
                        f"Update {i} assignee must be a string"
                    )

    def test_labels_format_is_valid_when_present(self, dependabot_config: Dict[str, Any]) -> None:
        """Test that labels format is valid when present."""
        updates = dependabot_config.get("updates", [])

        for i, update in enumerate(updates):
            labels = update.get("labels")
            if labels is not None:
                assert isinstance(labels, list), (
                    f"Update {i} labels must be a list"
                )
                for label in labels:
                    assert isinstance(label, str), (
                        f"Update {i} label must be a string"
                    )

    def test_commit_message_format_is_valid_when_present(self, dependabot_config: Dict[str, Any]) -> None:
        """Test that commit-message format is valid when present."""
        updates = dependabot_config.get("updates", [])

        for i, update in enumerate(updates):
            commit_message = update.get("commit-message")
            if commit_message is not None:
                assert isinstance(commit_message, dict), (
                    f"Update {i} commit-message must be a dictionary"
                )

                prefix = commit_message.get("prefix")
                if prefix is not None:
                    assert isinstance(prefix, str), (
                        f"Update {i} commit-message prefix must be a string"
                    )

                include = commit_message.get("include")
                if include is not None:
                    assert include == "scope", (
                        f"Update {i} commit-message include must be 'scope'"
                    )

    def test_rebase_strategy_is_valid_when_present(self, dependabot_config: Dict[str, Any]) -> None:
        """Test that rebase-strategy values are valid when present."""
        valid_strategies = ["auto", "disabled"]
        updates = dependabot_config.get("updates", [])

        for i, update in enumerate(updates):
            strategy = update.get("rebase-strategy")
            if strategy is not None:
                assert strategy in valid_strategies, (
                    f"Update {i} has invalid rebase-strategy: {strategy}"
                )

    def test_allow_format_is_valid_when_present(self, dependabot_config: Dict[str, Any]) -> None:
        """Test that allow format is valid when present."""
        valid_types = ["direct", "indirect", "all"]
        updates = dependabot_config.get("updates", [])

        for i, update in enumerate(updates):
            allow = update.get("allow")
            if allow is not None:
                assert isinstance(allow, list), f"Update {i} allow must be a list"

                for allow_item in allow:
                    assert isinstance(allow_item, dict), (
                        f"Update {i} allow item must be a dictionary"
                    )
                    dep_type = allow_item.get("dependency-type")
                    assert dep_type in valid_types, (
                        f"Update {i} allow item has invalid dependency-type: {dep_type}"
                    )

    def test_ignore_format_is_valid_when_present(self, dependabot_config: Dict[str, Any]) -> None:
        """Test that ignore format is valid when present."""
        updates = dependabot_config.get("updates", [])

        for i, update in enumerate(updates):
            ignore = update.get("ignore")
            if ignore is not None:
                assert isinstance(ignore, list), (
                    f"Update {i} ignore must be a list"
                )

                for ignore_item in ignore:
                    assert isinstance(ignore_item, dict), (
                        f"Update {i} ignore item must be a dictionary"
                    )
                    assert "dependency-name" in ignore_item, (
                        f"Update {i} ignore item must have 'dependency-name'"
                    )

    def test_no_duplicate_package_ecosystem_directory_combinations(
        self, dependabot_config: Dict[str, Any]
    ) -> None:
        """Test that there are no duplicate package-ecosystem and directory combinations."""
        updates = dependabot_config.get("updates", [])
        combinations: List[Any] = []

        for update in updates:
            combo = (
                update.get("package-ecosystem"),
                update.get("directory"),
            )
            assert combo not in combinations, f"Duplicate combination found: {combo}"
            combinations.append(combo)

    def test_config_follows_best_practices(self, dependabot_config: Dict[str, Any]) -> None:
        """Test that the configuration follows best practices."""
        updates = dependabot_config.get("updates", [])

        for i, update in enumerate(updates):
            limit = update.get("open-pull-requests-limit")
            if limit is not None:
                assert limit <= 10, (
                    f"Update {i} open-pull-requests-limit should be <= 10 for better "
                    "management"
                )

    def test_config_has_security_updates_enabled(self, dependabot_config: Dict[str, Any]) -> None:
        """Test that security updates are not explicitly disabled."""
        updates = dependabot_config.get("updates", [])

        for i, update in enumerate(updates):
            limit = update.get("open-pull-requests-limit")
            if limit is not None:
                assert limit > 0, (
                    f"Update {i} should allow at least one PR for security updates"
                )

    @pytest.mark.parametrize(
        "ecosystem,expected_files",
        [
            ("npm", ["package.json"]),
            (
                "pip",
                [
                    "requirements.txt",
                    "setup.py",
                    "pyproject.toml",
                ],
            ),
            ("docker", ["Dockerfile"]),
            ("github-actions", [".github/workflows"]),
        ],
    )
    def test_ecosystem_has_corresponding_files(
        self,
        dependabot_config: Dict[str, Any],
        ecosystem: str,
        expected_files: List[str],
    ) -> None:
        """Test that ecosystems have corresponding files in the repository."""
        updates = dependabot_config.get("updates", [])
        ecosystems_in_config = [
            update.get("package-ecosystem") for update in updates
        ]

        if ecosystem in ecosystems_in_config:
            for expected_file in expected_files:
                file_path = Path(expected_file)
                assert file_path.exists() or any(
                    Path(".").glob(f"**/{expected_file}")
                ), f"Expected file {expected_file} not found for ecosystem {ecosystem}"

    def test_config_file_permissions(self, dependabot_config_path: Path) -> None:
        """Test that the config file has appropriate permissions."""
        import stat

        file_stat = os.stat(dependabot_config_path)
        mode = file_stat.st_mode

        # Check that file is readable
        assert mode & stat.S_IRUSR, "Config file should be readable by owner"
        assert mode & stat.S_IRGRP, "Config file should be readable by group"
        assert mode & stat.S_IROTH, "Config file should be readable by others"

    def test_yaml_structure_depth_is_reasonable(self, dependabot_config: Dict[str, Any]) -> None:
        """Test that the YAML structure doesn't have excessive nesting."""
        def get_max_depth(obj: Any, current_depth: int = 0) -> int:
            if isinstance(obj, dict):
                return max(
                    [get_max_depth(v, current_depth + 1) for v in obj.values()]
                    + [current_depth]
                )
            if isinstance(obj, list):
                return max(
                    [get_max_depth(item, current_depth + 1) for item in obj]
                    + [current_depth]
                )
            return current_depth

        max_depth = get_max_depth(dependabot_config)
        assert max_depth <= 5, f"YAML structure is too deeply nested: {max_depth} levels"

    def test_config_contains_no_sensitive_information(self, dependabot_config_path: Path) -> None:
        """Test that the config file doesn't contain sensitive information."""
        sensitive_patterns = [
            "password",
            "secret",
            "token",
            "key",
            "credential",
            "auth",
            "login",
            "passwd",
            "pass",
        ]

        content = dependabot_config_path.read_text().lower()
        for pattern in sensitive_patterns:
            assert (
                pattern not in content
            ), f"Potential sensitive information found: {pattern}"