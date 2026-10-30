"""Real workflow/dataflow checks. Only the paid HTTP transport is substituted."""

import ast
from copy import deepcopy
import importlib
import json
import sys
import types

import pytest

from analyzers.pattern_detector import PatternDetector
from orchestrator import OrchestrationEngine
from workflows.runtime import bind_outputs, resolve_inputs

llm_module = importlib.import_module("connectors.llm_connector")


@pytest.fixture
def field_capture():
    return {
        "project_id": "project-17",
        "report_date": "2026-10-29",
        "daily_logs": [
            {
                "id": "log-1",
                "project_id": "project-17",
                "date": "2026-10-29",
                "notes": "Concrete delivery delayed; crew completed formwork.",
                "attachments": [
                    {
                        "id": "photo-1",
                        "url": "https://example.invalid/formwork.jpg",
                        "caption": "Formwork completed",
                    }
                ],
            }
        ],
        "tasks": [
            {
                "id": "task-1",
                "project_id": "project-17",
                "title": "Inspect formwork",
                "status": "In Progress",
                "due_date": "2026-10-30",
            }
        ],
        "rfis": [
            {
                "id": "rfi-1",
                "project_id": "project-17",
                "status": "Pending Approval",
                "description": "Confirm slab specification",
            }
        ],
    }


@pytest.fixture
def provider_transport(monkeypatch):
    calls = []
    state = {"status": 200, "source_id": "daily_logs/log-1", "finish_reason": "stop"}

    class Response:
        @property
        def status(self):
            return state["status"]

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def json(self):
            summary = {
                name: []
                for name in (
                    "overview",
                    "issues",
                    "photos",
                    "tasks_due",
                    "risk_flags",
                    "recommendations",
                )
            }
            summary["overview"] = [
                {
                    "text": "Crew completed formwork; concrete delivery was delayed.",
                    "source_ids": [state["source_id"]],
                }
            ]
            return {
                "model": "test-provider-response-model",
                "choices": [
                    {
                        "finish_reason": state["finish_reason"],
                        "message": {"content": json.dumps(summary)},
                    }
                ],
            }

    class Session:
        def __init__(self, **kwargs):
            self.options = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def post(self, url, **kwargs):
            calls.append({"url": url, **kwargs, **self.options})
            return Response()

    monkeypatch.setattr(llm_module.aiohttp, "ClientSession", Session)
    monkeypatch.setattr(
        llm_module.llm_connector, "openai_key", "test-only-not-a-real-key"
    )
    return calls, state


@pytest.mark.parametrize(
    "template,steps",
    [("construction_daily_log", 4), ("construction_daily_log_external", 3)],
)
@pytest.mark.asyncio
async def test_actual_construction_template_propagates_records_into_real_analyzer_and_summary(
    template, steps, field_capture, provider_transport, monkeypatch
):
    observed = []
    original = PatternDetector.analyze_supplied_data

    def inspect_data(self, data):
        observed.append(deepcopy(data))
        return original(self, data)

    monkeypatch.setattr(PatternDetector, "analyze_supplied_data", inspect_data)
    untouched = deepcopy(field_capture)
    engine = OrchestrationEngine()
    result = await engine.execute_intent(
        "Generate a construction daily log workflow",
        options={
            "workflow_template": template,
            "inputs": {"field_capture": field_capture},
        },
    )
    assert result["status"] == "completed", result
    assert len(result["steps_completed"]) == steps
    assert field_capture == untouched
    assert (
        observed[0]["records"]["daily_logs"][0]["notes"]
        == field_capture["daily_logs"][0]["notes"]
    )
    assert result["outputs"]["patterns"]["record_counts"] == {
        "daily_logs": 1,
        "tasks": 1,
        "rfis": 1,
    }
    assert result["outputs"]["patterns"]["status_counts"]["rfis"] == {
        "Pending Approval": 1
    }
    draft = result["outputs"]["draft_summary"]
    assert draft["status"] == "draft" and draft["requires_pm_review"] is True
    assert draft["project_id"] == "project-17"
    assert draft["retrieval"] == "caller_supplied_project_snapshot"
    assert draft["sources"][0]["record"] == field_capture["daily_logs"][0]
    assert draft["content"]["overview"][0]["source_ids"] == ["daily_logs/log-1"]
    calls, _ = provider_transport
    assert len(calls) == 1
    sent_context = json.loads(calls[0]["json"]["messages"][1]["content"])
    assert sent_context["sources"] == result["outputs"]["insights"][0]["sources"]
    assert sent_context["project_id"] == field_capture["project_id"]
    assert calls[0]["url"] == "https://api.openai.com/v1/chat/completions"
    assert calls[0]["headers"]["Authorization"] == "Bearer test-only-not-a-real-key"
    assert calls[0]["timeout"].total == 60
    assert calls[0]["allow_redirects"] is False
    assert calls[0]["json"]["response_format"]["json_schema"]["strict"] is True
    if template == "construction_daily_log":
        code = result["outputs"]["generated_code"]
        assert isinstance(code, str) and isinstance(
            result["outputs"]["instructions"], str
        )
        ast.parse(code)
        from datetime import datetime
        from fastapi import FastAPI, HTTPException
        from fastapi.testclient import TestClient

        app = FastAPI()
        namespace = {"app": app, "HTTPException": HTTPException, "datetime": datetime}
        exec(compile(code, "generated_rfi_scaffold.py", "exec"), namespace)
        response = TestClient(app).post(
            "/api/v1/rfi", params={"project_id": 17, "status": "Draft"}
        )
        assert response.status_code == 200
        assert response.json()["result"]["data"] == {
            "project_id": 17,
            "status": "Draft",
        }
        # This is an executable scaffold, not implemented RFI approval/persistence.


@pytest.mark.parametrize(
    "failure", ["missing-key", "foreign-citation", "incomplete", "http-error"]
)
@pytest.mark.asyncio
async def test_summary_failures_do_not_return_fake_complete_and_keep_upstream_evidence(
    failure, field_capture, provider_transport, monkeypatch
):
    calls, state = provider_transport
    if failure == "missing-key":
        monkeypatch.setattr(llm_module.llm_connector, "openai_key", "")
    elif failure == "foreign-citation":
        state["source_id"] = "daily_logs/invented-record"
    elif failure == "incomplete":
        state["finish_reason"] = "length"
    else:
        state["status"] = 429
    engine = OrchestrationEngine()
    result = await engine.execute_intent(
        "construction daily log",
        options={
            "workflow_template": "construction_daily_log_external",
            "inputs": {"field_capture": field_capture},
        },
    )
    assert result["status"] == "failed"
    assert result["steps_completed"][-1]["step"] == "log_analyzer"
    assert result["steps_completed"][-1]["attempt"] == 2
    assert "draft_summary" not in result["outputs"]
    assert result["outputs"]["normalized_data"]["sources"][0]["record_id"] == "log-1"
    assert len(engine.knowledge_graph.graph[result["workflow_id"]]["failures"]) == 2
    assert len(calls) == (0 if failure == "missing-key" else 2)


@pytest.mark.asyncio
async def test_missing_capture_reference_fails_before_any_provider_call(
    provider_transport,
):
    result = await OrchestrationEngine().execute_intent(
        "construction daily log",
        options={"workflow_template": "construction_daily_log_external"},
    )
    assert result["status"] == "failed"
    assert (
        "Unresolved workflow reference: field_capture"
        in result["steps_completed"][0]["error"]
    )
    assert provider_transport[0] == []


@pytest.mark.asyncio
async def test_cross_project_records_fail_before_sending(
    field_capture, provider_transport
):
    field_capture["daily_logs"][0]["project_id"] = "another-project"
    result = await OrchestrationEngine().execute_intent(
        "construction daily log",
        options={
            "workflow_template": "construction_daily_log_external",
            "inputs": {"field_capture": field_capture},
        },
    )
    assert result["status"] == "failed"
    assert "another project" in result["steps_completed"][0]["error"]
    assert provider_transport[0] == []


def test_nested_references_named_outputs_and_independent_values():
    inputs = {"source": {"notes": ["original"]}}
    outputs = {"normalized": {"records": [1, 2]}}
    resolved = resolve_inputs(
        {
            "data": [{"$ref": "outputs.normalized"}],
            "source": "source",
            "literal": {"$literal": "normalized"},
        },
        inputs,
        outputs,
        {"normalized", "future"},
    )
    resolved["source"]["notes"].append("changed")
    assert inputs["source"]["notes"] == ["original"]
    assert resolved["data"] == [{"records": [1, 2]}]
    assert resolved["literal"] == "normalized"
    assert bind_outputs(
        ["patterns", "insights"], {"patterns": [1], "insights": [2]}
    ) == {"patterns": [1], "insights": [2]}
    with pytest.raises(ValueError, match="Unresolved"):
        resolve_inputs("future", inputs, outputs, {"future"})
    with pytest.raises(ValueError, match="named outputs"):
        bind_outputs(["patterns", "insights"], {"patterns": []})


@pytest.mark.asyncio
async def test_protocols_receive_exact_kwargs_and_async_results_are_awaited(
    monkeypatch,
):
    calls = []

    async def task(data, label):
        calls.append((data, label))
        return {"success": True, "value": data}

    module = types.ModuleType("protocols.test_async_input")
    module.task = task
    monkeypatch.setitem(sys.modules, "protocols.test_async_input", module)
    engine = OrchestrationEngine()
    assert await engine.execute_protocol(
        "test_async_input", {"data": [1, 2], "label": "actual"}
    ) == {"success": True, "value": [1, 2]}
    assert calls == [([1, 2], "actual")]
    with pytest.raises(TypeError):
        await engine.execute_protocol("test_async_input", {"unused": "not discarded"})
    with pytest.raises(ValueError, match="Unknown analyzer"):
        await engine.execute_analyzer("missing", {})
    with pytest.raises(ImportError, match="could not be loaded"):
        await engine.execute_protocol("definitely_missing_no_generated_stub", {})


@pytest.mark.asyncio
async def test_database_analyzer_failure_is_not_reported_successful(monkeypatch):
    async def unavailable(self):
        return {"error": "database unavailable"}

    monkeypatch.setattr(PatternDetector, "detect_patterns", unavailable)
    result = await OrchestrationEngine().execute_analyzer("pattern_detector", {})
    assert result["success"] is False
    assert "database unavailable" in result["error"]


@pytest.mark.asyncio
async def test_explicit_protocol_path_is_used_and_invalid_path_never_falls_back(
    tmp_path,
):
    directory = tmp_path / "selected"
    directory.mkdir()
    (directory / "items.json").write_text('[{"id":1},{"id":2},{"id":3}]')
    engine = OrchestrationEngine()
    result = await engine.execute_protocol(
        "data_processor", {"data_path": str(directory)}
    )
    assert result["directory"] == str(directory)
    assert result["total_records"] == 3
    with pytest.raises(ValueError, match="does not exist"):
        await engine.execute_protocol(
            "data_processor", {"data_path": str(tmp_path / "missing")}
        )


@pytest.mark.asyncio
async def test_retry_resolves_fresh_input_copy_and_preserves_published_outputs(
    monkeypatch,
):
    engine = OrchestrationEngine()
    attempts = []

    async def protocol(name, inputs):
        attempts.append(deepcopy(inputs))
        inputs["data"].append("mutated")
        if len(attempts) == 1:
            raise RuntimeError("retry")
        return {"result": inputs["data"]}

    monkeypatch.setattr(engine, "execute_protocol", protocol)
    workflow = {
        "id": "wf_retry_data",
        "inputs": {"records": [1]},
        "steps": [
            {
                "type": "protocol",
                "name": "test",
                "inputs": {"data": {"$ref": "inputs.records"}},
                "outputs": ["result"],
                "retry_policy": {"max_attempts": 2},
            }
        ],
    }
    result = await engine.execute_workflow(workflow)
    assert result["status"] == "completed"
    assert attempts == [{"data": [1]}, {"data": [1]}]
    assert workflow["inputs"]["records"] == [1]
    assert result["outputs"]["result"] == [1, "mutated"]


@pytest.mark.asyncio
async def test_forged_source_envelope_is_rejected_before_http(
    field_capture, provider_transport
):
    from workflows.field_capture import normalize_field_capture

    analyzed = await PatternDetector().analyze(normalize_field_capture(field_capture))
    analyzed["insights"][0]["sources"][0]["source_id"] = "invented-provenance"
    with pytest.raises(ValueError, match="provenance"):
        await llm_module.llm_connector.summarize_project_records(analyzed["insights"])
    assert provider_transport[0] == []
