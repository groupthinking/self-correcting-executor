"""Execute imported graph semantics and actual correction, not success placeholders."""

import asyncio
from copy import deepcopy
import json
import types
import sys

import pytest
from mcp.server.fastmcp import FastMCP

from orchestrator import OrchestrationEngine
from workflows.build_plan import (
    BuildPlan,
    BuildPlanExecutor,
    ProtocolBindings,
    WorkflowStore,
)
from workflows.correction import Repair, run_correcting
from workflows.protocol_adapters import LocalMCP


def plan(agents=None, tasks=None, **values):
    return {
        "id": "plan-1",
        "workflow_run_id": "run-1",
        "exec_summary_id": "summary-1",
        "summary_title": "Source-grounded plan",
        "summary_goal": "Transform supplied inputs",
        "agents": agents
        or [{"name": "worker", "kind": "CustomAgent", "description": "Transform"}],
        "tasks": tasks or [{"id": "t1", "name": "Transform", "depends_on": []}],
        "mcp_tools": [],
        "a2a_required": False,
        "ap2_required": False,
        **values,
    }


@pytest.fixture
def store(tmp_path):
    value = WorkflowStore(tmp_path / "workflows.sqlite3")
    yield value
    value.close()


@pytest.mark.asyncio
async def test_unsorted_dependency_graph_uses_real_outputs_and_persists(store):
    value = plan(
        tasks=[
            {"id": "t2", "name": "Double", "depends_on": ["t1"]},
            {"id": "t1", "name": "Read", "depends_on": []},
        ]
    )
    original = deepcopy(value)
    calls = []

    def handler(inputs, context):
        calls.append(context["task"]["id"])
        if context["task"]["id"] == "t1":
            return {"value": inputs["value"] + 1}
        return {"value": context["dependencies"]["t1"]["value"] * 2}

    engine = BuildPlanExecutor(store, handlers={"worker": handler})
    result = await engine.execute(engine.compile(value), {"value": 3})
    assert result["status"] == "completed"
    assert result["outputs"]["t2"] == {"value": 8}
    assert calls == ["t1", "t2"] and value == original
    assert store.get("results", "run-1") == result
    assert store.get("runs", "run-1")["status"] == "completed"
    assert len([e for e in store.events("run-1") if e["type"] == "trajectory"]) == 2


@pytest.mark.parametrize(
    "tasks",
    [
        [{"id": "one", "name": "missing", "depends_on": ["absent"]}],
        [{"id": "one", "name": "cycle", "depends_on": ["one"]}],
        [{"id": "one", "name": "duplicate", "depends_on": []}] * 2,
    ],
)
def test_invalid_graph_rejected_before_execution(tasks):
    with pytest.raises(ValueError):
        BuildPlan.model_validate(plan(tasks=tasks))


@pytest.mark.asyncio
async def test_all_five_agent_kinds_execute_real_children_and_bound_loop(store):
    kinds = ["SequentialAgent", "ParallelAgent", "LoopAgent", "LlmAgent", "CustomAgent"]
    agents = [{"name": kind, "kind": kind, "description": kind} for kind in kinds]
    tasks = [
        {"id": str(i), "name": kind, "depends_on": []} for i, kind in enumerate(kinds)
    ]
    started = []

    async def increment(inputs, context):
        started.append(inputs["value"])
        await asyncio.sleep(0)
        return {"value": inputs["value"] + 1}

    async def reason(inputs, context):
        return {"value": inputs["value"] * 2}

    engine = BuildPlanExecutor(
        store,
        handlers={"CustomAgent": increment, "LlmAgent": reason},
        composites={
            "SequentialAgent": {"children": ["CustomAgent", "LlmAgent"]},
            "ParallelAgent": {"children": ["CustomAgent", "LlmAgent"]},
            "LoopAgent": {
                "children": ["CustomAgent"],
                "max_iterations": 3,
                "until": lambda value, iteration: value["value"] == 3,
            },
        },
    )
    result = await engine.execute(engine.compile(plan(agents, tasks)), {"value": 1})
    assert result["status"] == "completed", result
    assert result["outputs"] == {
        "0": {"value": 4},
        "1": {"CustomAgent": {"value": 2}, "LlmAgent": {"value": 2}},
        "2": {"value": 3},
        "3": {"value": 2},
        "4": {"value": 2},
    }
    engine.composites["LoopAgent"]["until"] = lambda value, iteration: False
    result = await engine.execute(engine.compile(plan(agents, tasks)), {"value": 1})
    assert result["status"] == "failed" and "stop condition" in result["error"]
    assert result["taskGraph"][3]["status"] == "pending"


@pytest.mark.asyncio
async def test_parallel_children_actually_overlap_and_do_not_share_mutable_input(store):
    reached = set()
    both = asyncio.Event()

    async def handler(inputs, context, name):
        inputs["items"].append(name)
        reached.add(name)
        if len(reached) == 2:
            both.set()
        await asyncio.wait_for(both.wait(), 1)
        return inputs

    agents = [
        {"name": "parallel", "kind": "ParallelAgent", "description": "Concurrent"},
        {"name": "a", "kind": "CustomAgent", "description": "a"},
        {"name": "b", "kind": "CustomAgent", "description": "b"},
    ]
    engine = BuildPlanExecutor(
        store,
        handlers={
            "a": lambda inputs, context: handler(inputs, context, "a"),
            "b": lambda inputs, context: handler(inputs, context, "b"),
        },
        composites={"parallel": {"children": ["a", "b"]}},
    )
    original = {"items": []}
    result = await engine.execute(engine.compile(plan(agents)), original)
    assert result["outputs"]["t1"] == {"a": {"items": ["a"]}, "b": {"items": ["b"]}}
    assert original == {"items": []}


@pytest.mark.asyncio
async def test_real_file_failure_repair_retry_and_restart_readback(tmp_path):
    file = tmp_path / "source.json"
    file.write_text('{"value":')
    database = tmp_path / "attempts.sqlite3"
    store = WorkflowStore(database)

    def read(inputs, context):
        return json.loads(file.read_text())

    def repair(error, inputs, attempt):
        before = file.read_text()
        # Explicitly supplied known-good source; no random rewrite or inferred business value.
        file.write_text(inputs["approved_source"])
        return Repair(
            inputs, before, file.read_text(), "Restore caller-supplied valid JSON"
        )

    engine = BuildPlanExecutor(store, handlers={"worker": read}, repairs={"t1": repair})
    result = await engine.execute(
        engine.compile(plan()), {"approved_source": '{"value":42}'}
    )
    assert result["status"] == "completed" and result["outputs"]["t1"] == {"value": 42}
    assert [a["status"] for a in result["attempts"]["t1"]] == ["failed", "completed"]
    assert result["attempts"]["t1"][0]["correction"]["before"] == '{"value":'
    store.close()
    reopened = WorkflowStore(database)
    assert reopened.get("results", "run-1") == result
    assert any(
        e["status"] == "failed"
        for e in reopened.events("run-1")
        if e["type"] == "trajectory"
    )
    reopened.close()


@pytest.mark.asyncio
async def test_native_catalog_dispatch_uses_same_concrete_repair_and_optout(
    tmp_path, monkeypatch
):
    path = tmp_path / "input.json"
    path.write_text("broken")
    module = types.ModuleType("protocols.local_repair_test")
    module.task = lambda path: json.loads(__import__("pathlib").Path(path).read_text())
    monkeypatch.setitem(sys.modules, "protocols.local_repair_test", module)
    engine = OrchestrationEngine()

    def repair(error, inputs, attempt):
        before = path.read_text()
        path.write_text('{"value":7}')
        return Repair(
            inputs, before, path.read_text(), "Restore explicitly supplied fixture"
        )

    engine.repairs["restore"] = repair
    workflow = {
        "id": "repair-native",
        "inputs": {},
        "self_correction": {"enabled": True},
        "steps": [
            {
                "name": "local_repair_test",
                "type": "protocol",
                "inputs": {"path": str(path)},
                "outputs": ["data"],
                "retry_policy": {"max_attempts": 2, "repair": "restore"},
            }
        ],
    }
    result = await engine.execute_workflow(workflow)
    assert result["outputs"]["data"] == {"value": 7}
    assert result["attempt_history"][0]["attempts"][0]["status"] == "failed"
    path.write_text("broken")
    workflow["self_correction"]["enabled"] = False
    result = await engine.execute_workflow(workflow)
    assert result["status"] == "failed" and path.read_text() == "broken"


@pytest.mark.asyncio
async def test_noop_and_unrepairable_operations_stay_failed():
    def fail(inputs):
        raise ValueError("Still broken")

    result = await run_correcting(
        fail,
        {},
        attempts=2,
        repair=lambda error, inputs, attempt: Repair(
            inputs, "same", "same", "No change"
        ),
    )
    assert result["status"] == "failed" and len(result["attempts"]) == 1
    assert "concrete change" in result["attempts"][0]["repair_error"]
    result = await run_correcting(
        fail,
        {},
        attempts=2,
        repair=lambda error, inputs, attempt: Repair(
            inputs, 1, 2, "Unsuccessful repair"
        ),
    )
    assert result["status"] == "failed" and len(result["attempts"]) == 2


@pytest.mark.asyncio
async def test_real_fastmcp_adapter_calls_registered_tool_and_records_trajectory(store):
    server = FastMCP("BuildPlan test")

    @server.tool()
    def add(left: int, right: int) -> int:
        return left + right

    async def handler(inputs, context):
        return await context["call_tool"](
            "mcp", "add", {"left": inputs["value"], "right": 2}
        )

    executor = BuildPlanExecutor(
        store,
        handlers={"worker": handler},
        protocols=ProtocolBindings(mcp=LocalMCP(server)),
    )
    result = await executor.execute(
        executor.compile(plan(mcp_tools=["add"])), {"value": 5}
    )
    assert result["status"] == "completed", result
    assert result["outputs"]["t1"][0]["text"] == "7"
    calls = [e for e in store.events("run-1") if e["type"] == "tool_call"]
    assert [e["status"] for e in calls] == ["started", "completed"]
    with pytest.raises(ValueError, match="not declared"):
        await executor.protocols.call(
            BuildPlan.model_validate(plan()), "mcp", "add", {}
        )


@pytest.mark.asyncio
async def test_protocol_flags_never_create_demo_mandate_or_fake_success(store):
    for values, message in [
        ({"mcp_tools": ["missing"]}, "MCP"),
        ({"a2a_required": True}, "A2A"),
        ({"ap2_required": True}, "AP2"),
    ]:
        executor = BuildPlanExecutor(
            store, handlers={"worker": lambda i, c: {"ok": True}}
        )
        result = await executor.execute(executor.compile(plan(**values)))
        assert result["status"] == "failed" and message in result["error"]
        assert not result["outputs"]


@pytest.mark.asyncio
async def test_ap2_contract_uses_only_supplied_mandate_and_exact_amount(store):
    calls = []

    class Provider:
        async def validate_mandate(self, mandate):
            return mandate["mandateId"] == "approved-fixture"

        async def process_payment(self, mandate, params):
            calls.append((deepcopy(mandate), deepcopy(params)))
            return {
                "transactionId": "controlled-transport-receipt",
                "status": "authorized",
            }

    mandate = {
        "mandateId": "approved-fixture",
        "vendorId": "vendor",
        "maxAmount": 10,
        "currency": "USD",
    }
    bindings = ProtocolBindings(ap2=Provider(), mandate=mandate)
    value = BuildPlan.model_validate(plan(ap2_required=True))
    await bindings.prepare(value)
    result = await bindings.call(
        value,
        "ap2",
        "vendor",
        {"amount": 2, "currency": "USD", "idempotencyKey": "authorized-operation-1"},
    )
    assert result["status"] == "authorized" and len(calls) == 1
    with pytest.raises(ValueError, match="mandate"):
        await bindings.call(value, "ap2", "vendor", {"amount": 20, "currency": "USD"})
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_builtin_snapshot_repair_is_real_atomic_and_preserves_valid_source(
    tmp_path,
):
    from workflows.correction import restore_json_snapshot

    path = tmp_path / "source.json"
    path.write_text('{"broken":')
    path.chmod(0o600)
    repair = restore_json_snapshot(path, '{"restored":true}')
    result = await run_correcting(
        lambda inputs: json.loads(path.read_text()), {}, attempts=2, repair=repair
    )
    assert result["status"] == "completed" and result["output"] == {"restored": True}
    assert path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(ValueError, match="already-valid"):
        repair(ValueError("other failure"), {}, 1)
    assert json.loads(path.read_text()) == {"restored": True}


@pytest.mark.parametrize(
    "kind", ["SequentialAgent", "ParallelAgent", "LoopAgent", "LlmAgent", "CustomAgent"]
)
@pytest.mark.asyncio
async def test_each_imported_kind_is_wrapped_by_actual_correction(
    kind, tmp_path, store
):
    from workflows.correction import restore_json_snapshot

    source = tmp_path / "source.json"
    source.write_text("broken")
    agents = [{"name": "selected", "kind": kind, "description": "Read source"}]
    handlers = {"selected": lambda inputs, context: json.loads(source.read_text())}
    controls = {}
    if kind in {"SequentialAgent", "ParallelAgent", "LoopAgent"}:
        agents.append(
            {"name": "reader", "kind": "CustomAgent", "description": "Read source"}
        )
        handlers["reader"] = handlers.pop("selected")
        controls["selected"] = {
            "children": ["reader"],
            "max_iterations": 2,
            "until": lambda v, i: v["ok"] is True,
        }
    executor = BuildPlanExecutor(
        store,
        handlers=handlers,
        composites=controls,
        repairs={"t1": restore_json_snapshot(source, '{"ok":true}')},
    )
    result = await executor.execute(executor.compile(plan(agents)))
    assert result["status"] == "completed", result
    assert [a["status"] for a in result["attempts"]["t1"]] == ["failed", "completed"]
    assert result["attempts"]["t1"][0]["correction"]["after"] == '{"ok":true}'


@pytest.mark.asyncio
async def test_real_registered_a2a_agent_dispatch_and_failed_child_propagation(store):
    from agents.a2a_mcp_integration import A2AMCPOrchestrator
    from workflows.protocol_adapters import LocalA2A

    class ActualLocalAgent:
        agent_id = "local-code-agent"

        async def process_intent(self, intent):
            return {"sum": sum(intent["values"])}

    registry = A2AMCPOrchestrator()
    registry.register_agent(ActualLocalAgent())
    bindings = ProtocolBindings(
        a2a=LocalA2A(registry, {"local-code-agent": ["code_generation"]})
    )

    async def worker(inputs, context):
        return await context["call_tool"](
            "a2a",
            "local-code-agent",
            {"taskId": "a2a-1", "parameters": {"values": [2, 3]}},
        )

    executor = BuildPlanExecutor(store, handlers={"worker": worker}, protocols=bindings)
    result = await executor.execute(executor.compile(plan(a2a_required=True)))
    assert result["outputs"]["t1"]["result"] == {"sum": 5}
    agents = [
        {"name": "parallel", "kind": "ParallelAgent", "description": ""},
        {"name": "bad", "kind": "CustomAgent", "description": ""},
    ]
    executor = BuildPlanExecutor(
        store,
        handlers={"bad": lambda i, c: {"success": False, "error": "actual failure"}},
        composites={"parallel": {"children": ["bad"]}},
    )
    result = await executor.execute(executor.compile(plan(agents)))
    assert result["status"] == "failed" and "actual failure" in result["error"]


@pytest.mark.asyncio
async def test_duplicate_parallel_bindings_fail_before_any_child_side_effect(store):
    called = []
    agents = [
        {"name": "parallel", "kind": "ParallelAgent", "description": ""},
        {"name": "leaf", "kind": "CustomAgent", "description": ""},
    ]
    executor = BuildPlanExecutor(
        store,
        handlers={"leaf": lambda i, c: called.append("effect")},
        composites={"parallel": {"children": ["leaf", "leaf"]}},
    )
    result = await executor.execute(executor.compile(plan(agents)))
    assert result["status"] == "failed" and "unique names" in result["error"]
    assert called == []
