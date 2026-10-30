"""Regression checks for the actual Dependabot and CI configuration.

The repository has no dependabot_test.py runtime module. Validate the shipped
configuration and the CI dependency set instead of importing a nonexistent helper.
"""

from pathlib import Path

import pytest
import yaml
from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]


def read_yaml(path):
    return yaml.safe_load((ROOT / path).read_text())


def test_dependabot_configuration_is_valid_and_complete():
    config = read_yaml(".github/dependabot.yml")
    assert config["version"] == 2
    updates = config["updates"]
    assert len(updates) == 3
    assert {(item["package-ecosystem"], item["directory"]) for item in updates} == {
        ("pip", "/"),
        ("npm", "/frontend"),
        ("docker", "/"),
    }
    assert all(item["schedule"]["interval"] == "weekly" for item in updates)


@pytest.mark.parametrize(
    "ecosystem,manifest",
    [("pip", "requirements.txt"), ("npm", "package.json"), ("docker", "Dockerfile")],
)
def test_each_dependabot_update_has_a_real_manifest(ecosystem, manifest):
    update = next(
        item
        for item in read_yaml(".github/dependabot.yml")["updates"]
        if item["package-ecosystem"] == ecosystem
    )
    assert (ROOT / update["directory"].lstrip("/") / manifest).is_file()


@pytest.mark.parametrize(
    "package", ["mcp", "numpy", "psycopg2-binary", "PyYAML", "packaging"]
)
def test_ci_installs_packages_imported_by_the_suite(package):
    requirements = {
        Requirement(line).name.lower()
        for line in (ROOT / "requirements-ci.txt").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert package.lower() in requirements


def test_ci_propagates_test_failures():
    workflow = read_yaml(".github/workflows/python-ci.yml")
    job = workflow["jobs"]["lint-test"]
    assert job.get("continue-on-error", False) is False
    assert job["strategy"]["matrix"]["python-version"] == ["3.10", "3.11"]
    test_step = next(
        step for step in job["steps"] if step.get("name") == "Run tests with coverage"
    )
    assert test_step["run"].strip() == "pytest --cov=."
    assert test_step.get("continue-on-error", False) is False
