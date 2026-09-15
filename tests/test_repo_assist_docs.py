from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_repo_map_contains_requested_sections():
    repo_map = REPO_ROOT / "docs" / "REPO_MAP.md"

    assert repo_map.exists()
    content = repo_map.read_text()
    assert "## Repo tree (focused map)" in content
    assert "## Dependency snapshot" in content
    assert "## Mermaid diagram" in content
    assert "## Agentic workflow checklist for this repo" in content
    assert "## MCP baseline: is this repo using MCP?" in content
    assert "## Live verification steps" in content


def test_repo_skills_are_present_and_repo_specific():
    skills = {
        "github-actions-failure-debugging": (
            REPO_ROOT
            / ".github"
            / "skills"
            / "github-actions-failure-debugging"
            / "SKILL.md"
        ),
        "codebase-summary": (
            REPO_ROOT / ".github" / "skills" / "codebase-summary" / "SKILL.md"
        ),
    }

    for name, skill_path in skills.items():
        assert skill_path.exists(), f"Missing skill: {name}"
        content = skill_path.read_text()
        assert f"name: {name}" in content
        assert "self-correcting-executor" in content
