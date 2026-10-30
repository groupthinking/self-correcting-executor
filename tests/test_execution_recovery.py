"""Independent review reproductions: partial effects must not be replayed."""

import asyncio
from copy import deepcopy

import pytest

from workflows.build_plan import BuildPlanExecutor, ProtocolBindings, WorkflowStore
from workflows.correction import Repair, run_correcting
from test_build_plan import plan


@pytest.mark.asyncio
async def test_successful_operation_is_never_replayed_when_observer_fails():
    effects = []

    def operation(inputs):
        effects.append("committed")
        return {"receipt": "actual-1"}

    def observer(entry):
        raise OSError("receipt store unavailable")

    result = await run_correcting(operation, {}, attempts=3, observe=observer)
    assert effects == ["committed"]
    assert result["status"] == "failed" and result["execution_status"] == "completed"
    assert result["output"] == {"receipt": "actual-1"}
    assert (
        len(result["attempts"]) == 1 and result["attempts"][0]["status"] == "completed"
    )


@pytest.mark.parametrize("kind", ["SequentialAgent", "ParallelAgent", "LoopAgent"])
@pytest.mark.asyncio
async def test_composite_retry_checkpoints_completed_children_and_loop_iterations(
    tmp_path, kind
):
    store = WorkflowStore(tmp_path / "runs.sqlite3")
    effects, checks, order = [], [], []

    async def effect(inputs, context):
        if kind == "ParallelAgent":
            await asyncio.sleep(0.02)
        value = inputs["value"] + 1
        effects.append(value)
        order.append("effect")
        return {"value": value}

    async def check(inputs, context):
        checks.append(inputs["value"])
        order.append("check")
        target = 2 if kind == "LoopAgent" else (0 if kind == "ParallelAgent" else 1)
        if inputs["value"] == target and checks.count(target) == 1:
            raise RuntimeError("Failed validation once")
        return inputs

    agents = [
        {"name": "root", "kind": kind, "description": ""},
        {"name": "effect", "kind": "CustomAgent", "description": ""},
        {"name": "check", "kind": "CustomAgent", "description": ""},
    ]
    executor = BuildPlanExecutor(
        store,
        handlers={"effect": effect, "check": check},
        composites={
            "root": {
                "children": ["effect", "check"],
                "max_iterations": 3,
                "until": lambda v, i: v["value"] == 2,
            }
        },
    )
    result = await executor.execute(executor.compile(plan(agents)), {"value": 0})
    assert result["status"] == "completed", result
    assert effects == ([1, 2] if kind == "LoopAgent" else [1])
    assert len(checks) == (3 if kind == "LoopAgent" else 2)
    if kind == "ParallelAgent":
        assert order == [
            "check",
            "effect",
            "check",
        ]  # Retry waited for the owned sibling.
        await asyncio.sleep(0.03)
        assert effects == [1]
    assert [a["status"] for a in result["attempts"]["t1"]] == ["failed", "completed"]
    store.close()


@pytest.mark.asyncio
async def test_changed_retry_input_cannot_mix_with_completed_child_checkpoint(tmp_path):
    store = WorkflowStore(tmp_path / "runs.sqlite3")
    effects = []

    def done(inputs, context):
        effects.append(deepcopy(inputs))
        return inputs

    def fail(inputs, context):
        raise ValueError("change requested")

    agents = [
        {"name": "seq", "kind": "SequentialAgent", "description": ""},
        {"name": "done", "kind": "CustomAgent", "description": ""},
        {"name": "fail", "kind": "CustomAgent", "description": ""},
    ]
    executor = BuildPlanExecutor(
        store,
        handlers={"done": done, "fail": fail},
        composites={"seq": {"children": ["done", "fail"]}},
        repairs={"t1": lambda e, i, a: Repair({"value": 2}, 1, 2, "Changed input")},
    )
    result = await executor.execute(executor.compile(plan(agents)), {"value": 1})
    assert result["status"] == "failed" and "input changed" in result["error"]
    assert effects == [{"value": 1}]
    store.close()


@pytest.mark.asyncio
async def test_successful_mcp_effect_is_not_replayed_after_receipt_store_failure(
    tmp_path,
):
    store = WorkflowStore(tmp_path / "runs.sqlite3")
    calls = []
    original = store.event

    def event(run_id, entry):
        if entry.get("type") == "tool_call" and entry.get("status") == "completed":
            raise OSError("Controlled receipt storage outage")
        return original(run_id, entry)

    store.event = event

    class Client:
        async def list_tools(self):
            return [{"name": "effect"}]

        async def call_tool(self, name, args):
            calls.append(name)
            return {"isError": False, "content": [{"text": "saved"}]}

    async def worker(inputs, context):
        return await context["call_tool"]("mcp", "effect", {})

    executor = BuildPlanExecutor(
        store, handlers={"worker": worker}, protocols=ProtocolBindings(mcp=Client())
    )
    result = await executor.execute(executor.compile(plan(mcp_tools=["effect"])))
    assert result["status"] == "failed" and "no replay" in result["error"]
    assert calls == ["effect"]
    store.close()


@pytest.mark.asyncio
async def test_mcp_json_error_cannot_complete_a_graph(tmp_path):
    store = WorkflowStore(tmp_path / "runs.sqlite3")

    class Client:
        async def list_tools(self):
            return [{"name": "failed"}]

        async def call_tool(self, name, args):
            return {"isError": True, "content": [{"text": "denied"}]}

    async def worker(inputs, context):
        return await context["call_tool"]("mcp", "failed", {})

    executor = BuildPlanExecutor(
        store, handlers={"worker": worker}, protocols=ProtocolBindings(mcp=Client())
    )
    result = await executor.execute(executor.compile(plan(mcp_tools=["failed"])))
    assert result["status"] == "failed" and not result["outputs"]
    assert not any(
        e["type"] == "tool_call" and e["status"] == "completed"
        for e in store.events("run-1")
    )
    store.close()


@pytest.mark.asyncio
async def test_native_output_binding_failure_keeps_receipt_without_replaying_dispatch(
    monkeypatch,
):
    from orchestrator import OrchestrationEngine

    engine = OrchestrationEngine()
    effects = []

    async def effect(name, inputs):
        effects.append("committed")
        return {"success": True, "receipt": "r1"}

    monkeypatch.setattr(engine, "execute_protocol", effect)
    result = await engine.execute_workflow(
        {
            "id": "bind-failure",
            "steps": [
                {
                    "type": "protocol",
                    "name": "effect",
                    "inputs": {},
                    "outputs": ["receipt", "missing"],
                    "retry_policy": {"max_attempts": 2},
                }
            ],
        }
    )
    assert result["status"] == "failed" and effects == ["committed"]
    assert result["steps_completed"][0]["output"] == {"success": True, "receipt": "r1"}
    assert "no replay" in result["steps_completed"][0]["error"]
    assert len(result["attempt_history"][0]["attempts"]) == 1
