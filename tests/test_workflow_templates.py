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


def test_external_templates_loaded_from_data_dir():
    workflow = materialize_workflow(
        "construction daily log using external data",
        preferred_template="construction_daily_log_external",
    )
    assert workflow is not None
    assert workflow.get("template") == "construction_daily_log_external"
    assert workflow["self_correction"]["policy"]["fallback_protocol"] == "log_analyzer"


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


def test_materialized_steps_and_runs_do_not_share_nested_state():
    from workflows.templates import get_template

    template = get_template("construction_daily_log")
    first = template.materialize("first")
    second = template.materialize("second")
    first["steps"][0]["retry_policy"]["max_attempts"] = 7
    first["steps"][0]["inputs"]["source"] = "changed"
    first["metadata"]["patterns"].append("changed")
    assert first["steps"][1]["retry_policy"]["max_attempts"] == 2
    assert second["steps"][0]["retry_policy"]["max_attempts"] == 2
    assert second["steps"][0]["inputs"]["source"] == "field_capture"
    assert "changed" not in second["metadata"]["patterns"]
    assert template.steps[0]["inputs"]["source"] == "field_capture"


@pytest.mark.parametrize("correction", [None, {"enabled": False}, {}])
def test_retry_defaults_require_explicit_enablement(correction):
    from workflows.self_correcting import apply_self_correction

    workflow = {"steps": [{"type": "protocol", "name": "test", "inputs": {}}]}
    if correction is not None:
        workflow["self_correction"] = correction
    assert "retry_policy" not in apply_self_correction(workflow)["steps"][0]


def test_wrkflw_repository_name_matches_intent():
    from workflows.registry import match_template_for_intent

    assert match_template_for_intent("Import WRKFLW").name == "construction_daily_log"


@pytest.mark.parametrize("attempts", [0, -1, None, "bad", False])
@pytest.mark.asyncio
async def test_invalid_retry_count_executes_once_instead_of_skipping(
    attempts, monkeypatch
):
    engine = OrchestrationEngine()
    calls = []

    async def execute(name, inputs):
        calls.append(name)
        return {"success": True}

    monkeypatch.setattr(engine, "execute_protocol", execute)
    workflow = {
        "id": "wf_invalid",
        "steps": [
            {
                "type": "protocol",
                "name": "test",
                "inputs": {},
                "retry_policy": {"max_attempts": attempts},
            }
        ],
    }
    result = await engine.execute_workflow(workflow)
    assert result["status"] == "completed"
    assert calls == ["test"]
    assert result["steps_completed"][0]["attempt"] == 1


@pytest.mark.asyncio
async def test_disabled_self_correction_prevents_explicit_retries(monkeypatch):
    engine = OrchestrationEngine()
    calls = []

    async def fail(name, inputs):
        calls.append(name)
        raise RuntimeError("unavailable")

    monkeypatch.setattr(engine, "execute_protocol", fail)
    workflow = {
        "id": "wf_disabled",
        "self_correction": {"enabled": False},
        "steps": [
            {
                "type": "protocol",
                "name": "test",
                "inputs": {},
                "retry_policy": {"max_attempts": 3},
            }
        ],
    }
    result = await engine.execute_workflow(workflow)
    assert result["status"] == "failed"
    assert calls == ["test"]
    assert len(engine.knowledge_graph.graph["wf_disabled"]["failures"]) == 1


@pytest.mark.asyncio
async def test_failed_result_retries_and_preserves_failure_history(monkeypatch):
    engine = OrchestrationEngine()
    calls = []

    async def execute(name, inputs):
        calls.append(name)
        return {"success": len(calls) == 2, "error": "temporary"}

    monkeypatch.setattr(engine, "execute_protocol", execute)
    workflow = {
        "id": "wf_history",
        "steps": [
            {
                "type": "protocol",
                "name": "test",
                "inputs": {},
                "retry_policy": {"max_attempts": 2},
                "outputs": ["result"],
            }
        ],
    }
    result = await engine.execute_workflow(workflow)
    assert result["status"] == "completed"
    assert calls == ["test", "test"]
    assert result["steps_completed"][0]["attempt"] == 2
    assert await engine.knowledge_graph.find_similar_executions("daily") == []
    node = engine.knowledge_graph.graph["wf_history"]
    node["relationships"].append("previous")
    node["insights"].append("retry helped")
    await engine.knowledge_graph.store_execution(
        {"workflow_id": "wf_history", "intent": "daily log"}
    )
    assert node["failures"][0]["error"] == "temporary"
    assert node["relationships"] == ["previous"]
    assert node["insights"] == ["retry helped"]
    assert (await engine.knowledge_graph.find_similar_executions("daily"))[0][
        "workflow_id"
    ] == "wf_history"


@pytest.mark.parametrize("suffix", ["yaml", "json"])
@pytest.mark.asyncio
async def test_loaded_template_executes_real_data_processor(
    suffix, tmp_path, monkeypatch
):
    import json
    import yaml
    from workflows.templates import (
        WORKFLOW_TEMPLATES,
        _load_external_templates_from_data_dir,
    )

    catalog_dir = tmp_path / "catalog"
    catalog_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "records.json").write_text(json.dumps([{"value": 1}, {"value": 2}]))
    definition = {
        "name": f"external_{suffix}",
        "tags": [f"external_{suffix}"],
        "steps": [{"type": "protocol", "name": "data_processor", "outputs": ["data"]}],
        "resilience": {"max_attempts": 0},
    }
    serialized = (
        yaml.safe_dump(definition) if suffix == "yaml" else json.dumps(definition)
    )
    (catalog_dir / f"fixture.{suffix}").write_text(serialized)
    (catalog_dir / "malformed.json").write_text(
        '{"name":"broken","steps":"wrong type"}'
    )
    loaded = _load_external_templates_from_data_dir(catalog_dir)
    assert set(loaded) == {f"external_{suffix}"}
    monkeypatch.setitem(
        WORKFLOW_TEMPLATES, f"external_{suffix}", loaded[f"external_{suffix}"]
    )
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    engine = OrchestrationEngine()
    result = await engine.execute_intent(
        f"external_{suffix}",
        options={"workflow_template": f"external_{suffix}"},
    )
    assert result["status"] == "completed"
    assert result["outputs"]["data"]["directory"] == str(data_dir)
    assert result["outputs"]["data"]["total_records"] == 2
    assert result["steps_completed"][0]["attempt"] == 1
