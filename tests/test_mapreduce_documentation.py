"""Execute the guide's local examples and its documented dispatch/timeout limits."""

import asyncio
import inspect
import json
import sqlite3
import threading
from pathlib import Path

import pytest

from examples.mapreduce_local import demonstrate, make_orchestrator
import orchestrator_mapreduce as module
from protocols import api_health_checker, data_processor, loader


@pytest.fixture(autouse=True)
def isolated_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    async def no_provider(*args, **kwargs):
        pytest.fail("The local guide checks must not call a provider")

    async def no_slack(*args, **kwargs):
        pytest.fail("The local guide checks must not attempt Slack")

    monkeypatch.setattr(module.SemanticGate, "_llm_evaluate", no_provider)
    monkeypatch.setattr(module.SlackEscalation, "escalate", no_slack)


@pytest.mark.asyncio
async def test_all_five_local_patterns_process_actual_files_and_persist(tmp_path):
    jobs = await demonstrate(tmp_path / "fixtures")
    assert set(jobs) == {
        "minimal",
        "fan_out",
        "heterogeneous",
        "custom",
        "step_by_step",
    }
    assert all(job.verification_passed for job in jobs.values())
    assert all(job.status == module.TaskStatus.COMPLETED for job in jobs.values())
    assert jobs["minimal"].subtasks[0].result["total_records"] == 1
    assert [
        subtask.result["total_records"] for subtask in jobs["fan_out"].subtasks
    ] == [1, 2]
    mixed = jobs["heterogeneous"].subtasks
    assert mixed[0].result["action"] == "data_processing"
    assert mixed[1].result["total_lines_analyzed"] == 1
    assert jobs["custom"].reduced_result["categories"] == {
        "data_processing": 1,
        "log_analysis": 1,
    }
    directories = ["minimal", "fan-out", "mixed", "custom", "steps"]
    for job, directory in zip(jobs.values(), directories):
        saved = tmp_path / "fixtures" / directory
        with sqlite3.connect(saved / "state.sqlite3") as conn:
            row = conn.execute(
                "SELECT status,verification_passed,subtask_count FROM jobs WHERE id=?",
                (job.id,),
            ).fetchone()
        assert row == ("completed", 1, len(job.subtasks))
        assert f"Job: {job.id}" in (saved / "STATE.md").read_text()


@pytest.mark.asyncio
async def test_signature_dispatch_defaults_full_kwargs_and_tracking(
    tmp_path, monkeypatch
):
    def zero():
        return {"success": True, "source": "zero"}

    def required(value, scale=2):
        return {"success": True, "value": value * scale}

    def kwargs(**inputs):
        return {"success": True, "received": inputs}

    monkeypatch.setattr(
        loader,
        "load_protocol",
        lambda name: {
            "task": {"zero": zero, "required": required, "kwargs": kwargs}[name]
        },
    )
    orchestrator = make_orchestrator(tmp_path / "state")
    job = await orchestrator.plan(
        "Dispatch supplied arguments",
        [
            {"protocol": "zero", "inputs": {"ignored": "yes"}},
            {"protocol": "required", "inputs": {"value": 3}},
            {"protocol": "required", "inputs": {"value": 3, "unexpected": True}},
            {"protocol": "required", "inputs": {}},
            {"protocol": "kwargs", "inputs": {"value": 4, "extra": True}},
        ],
    )
    await orchestrator.map_execute(job)
    assert job.subtasks[0].result == {"success": True, "source": "zero"}
    assert job.subtasks[1].result["value"] == 6
    assert job.subtasks[2].status == module.TaskStatus.FAILED
    assert "unexpected" in job.subtasks[2].error
    assert job.subtasks[3].status == module.TaskStatus.FAILED
    assert "value" in job.subtasks[3].error
    assert job.subtasks[4].result["received"] == {"value": 4, "extra": True}
    tracked = [
        json.loads(line)
        for line in Path("memory/required.json").read_text().splitlines()
    ]
    assert len(tracked) == 3
    assert sum(result["success"] is True for result in tracked) == 1


@pytest.mark.asyncio
async def test_timeout_does_not_track_or_terminate_running_worker(
    tmp_path, monkeypatch
):
    started, release, finished = threading.Event(), threading.Event(), threading.Event()

    def blocking():
        started.set()
        release.wait(2)
        finished.set()
        return {"success": True}

    monkeypatch.setattr(loader, "load_protocol", lambda name: {"task": blocking})
    orchestrator = make_orchestrator(tmp_path / "state", subtask_timeout=0.03)
    job = await orchestrator.plan("Observe timeout", [{"protocol": "blocking"}])
    try:
        await orchestrator.map_execute(job)
        assert started.is_set()
        assert job.subtasks[0].status == module.TaskStatus.FAILED
        assert "timed out" in job.subtasks[0].error
        assert not finished.is_set()
        assert not Path("memory/blocking.json").exists()
    finally:
        release.set()
        assert await asyncio.to_thread(finished.wait, 2)
    assert job.subtasks[0].status == module.TaskStatus.FAILED
    assert job.subtasks[0].result is None
    assert not Path("memory/blocking.json").exists()


@pytest.mark.asyncio
async def test_normal_failure_return_is_completed_and_mechanical_gate_passes(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        loader, "load_protocol", lambda name: {"task": lambda: {"success": False}}
    )
    orchestrator = make_orchestrator(tmp_path / "state")
    job = await orchestrator.plan(
        "Observe return status", [{"protocol": "failure_return"}]
    )
    await orchestrator.map_execute(job)
    assert job.subtasks[0].status == module.TaskStatus.COMPLETED
    assert (await orchestrator._default_gate(job))["passed"] is True
    assert (await orchestrator.verification_gate(job))["passed"] is False


@pytest.mark.asyncio
async def test_retry_dispatch_is_not_outcome_tracked(tmp_path, monkeypatch):
    def task(**kwargs):
        return {"success": True, "received": kwargs}

    monkeypatch.setattr(loader, "load_protocol", lambda name: {"task": task})
    orchestrator = make_orchestrator(tmp_path / "state")
    subtask = module.SubTask(
        "retry", "retry_fixture", {}, status=module.TaskStatus.FAILED, attempts=1
    )
    await orchestrator.mutator.mutate("retry_fixture", subtask, "fixture failure")
    await orchestrator._retry_subtask(subtask)
    assert subtask.status == module.TaskStatus.COMPLETED
    assert subtask.result["received"]["_retry"] is True
    assert subtask.attempts == 2
    assert not Path("memory/retry_fixture.json").exists()


@pytest.mark.asyncio
async def test_async_protocol_result_is_not_awaited(tmp_path, monkeypatch):
    invoked = []

    async def task():
        invoked.append(True)
        return {"success": True}

    monkeypatch.setattr(loader, "load_protocol", lambda name: {"task": task})
    orchestrator = make_orchestrator(tmp_path / "state")
    result = await orchestrator._run_protocol_isolated("async_fixture", {})
    try:
        assert inspect.iscoroutine(result)
        assert invoked == []
    finally:
        result.close()


def test_current_builtin_signatures_and_missing_data_directory():
    assert list(inspect.signature(api_health_checker.task).parameters) == [
        "service",
        "url",
    ]
    assert list(inspect.signature(data_processor.task).parameters) == [
        "data_path",
        "source",
    ]
    with pytest.raises(ValueError, match="does not exist"):
        data_processor.task(data_path="/does-not-exist-mapreduce-guide")


@pytest.mark.asyncio
async def test_semantic_provider_failure_returns_documented_mechanical_fallback(
    monkeypatch,
):
    calls = []

    async def unavailable(self, job):
        calls.append(job.id)
        raise RuntimeError("controlled unavailable provider")

    monkeypatch.setattr(module.SemanticGate, "_llm_evaluate", unavailable)
    job = module.OrchestratedJob(
        "fallback",
        "Observe fallback",
        subtasks=[
            module.SubTask("one", "fixture", {}, status=module.TaskStatus.COMPLETED)
        ],
    )
    result = await module.SemanticGate().verify(job)
    assert result == {
        "passed": True,
        "layer": "mechanical_fallback",
        "success_rate": 1.0,
    }
    assert calls == [job.id]
