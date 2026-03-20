import asyncio

import pytest

from orchestrator import OrchestrationEngine
from workflows.registry import list_templates, materialize_workflow


def test_templates_cover_all_sources():
    templates = list_templates()
    sources = {template.source_repo for template in templates}
    assert {"WRKFLW", "workflows", "self-correcting-executor"}.issubset(sources)


def test_materialize_adds_self_correction():
    workflow = materialize_workflow(
        "Build a TypeScript API pipeline",
        preferred_template="typescript_service_pipeline",
    )
    assert workflow is not None
    assert workflow["self_correction"]["policy"]["max_attempts"] >= 2
    assert all("retry_policy" in step for step in workflow["steps"])


@pytest.mark.asyncio
async def test_orchestrator_prefers_template_when_requested():
    engine = OrchestrationEngine()
    analyzed = await engine.analyze_intent("Create a construction daily log workflow")
    components = await engine.discover_components(analyzed, [])

    workflow = await engine.generate_workflow(
        analyzed, components, {"workflow_template": "construction_daily_log"}
    )

    assert workflow.get("template") == "construction_daily_log"
    assert workflow["self_correction"]["policy"]["max_attempts"] >= 2
