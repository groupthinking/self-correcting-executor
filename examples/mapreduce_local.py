"""Run the documented Map/Reduce patterns against real, temporary local files.

Run from the repository root: python -m examples.mapreduce_local
No provider, service, source mutation or Slack operation is needed.
"""

import asyncio
import json
import os
from pathlib import Path
import tempfile

from orchestrator_mapreduce import HierarchicalOrchestrator, TaskStatus


async def strict_gate(job):
    failures = [
        subtask
        for subtask in job.subtasks
        if subtask.status != TaskStatus.COMPLETED
        or not isinstance(subtask.result, dict)
        or subtask.result.get("success") is not True
    ]
    return {
        "passed": bool(job.subtasks) and not failures,
        "reason": f"{len(failures)} unsuccessful local subtasks",
    }


async def category_reducer(results, failures):
    categories = {}
    for result in results:
        action = result.get("action", "unknown")
        categories[action] = categories.get(action, 0) + 1
    return {"categories": categories, "failures": len(failures), "results": results}


class LocalEscalation:
    async def escalate(self, job, reason):
        raise RuntimeError(f"Local demo failed: {job.id}: {reason}")


def make_orchestrator(output_dir, **kwargs):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    orchestrator = HierarchicalOrchestrator(
        verification_gate=strict_gate,
        state_file=str(output_dir / "STATE.md"),
        db_path=str(output_dir / "state.sqlite3"),
        **kwargs,
    )
    # This local demonstration fails visibly instead of attempting Slack.
    orchestrator.escalation = LocalEscalation()
    return orchestrator


def data_tasks(data_paths):
    return [
        {"protocol": "data_processor", "inputs": {"data_path": str(path)}}
        for path in data_paths
    ]


async def minimal(data_path, output_dir):
    return await make_orchestrator(output_dir).run(
        "Count records in the supplied local directory", data_tasks([data_path])
    )


async def fan_out(data_paths, output_dir):
    return await make_orchestrator(output_dir, max_concurrency=2).run(
        "Count records separately in each supplied directory", data_tasks(data_paths)
    )


async def heterogeneous(data_path, output_dir):
    # The existing log_analyzer reads LOG_DIR; it does not accept run_id/keyword.
    return await make_orchestrator(output_dir).run(
        "Count local data records and summarize the configured local logs",
        data_tasks([data_path]) + [{"protocol": "log_analyzer", "inputs": {}}],
    )


async def custom_gate_and_reducer(data_path, output_dir):
    return await make_orchestrator(output_dir, reducer=category_reducer).run(
        "Group actual local data and log results by their action",
        data_tasks([data_path]) + [{"protocol": "log_analyzer", "inputs": {}}],
    )


async def step_by_step(data_path, output_dir):
    orchestrator = make_orchestrator(output_dir)
    job = await orchestrator.plan(
        "Inspect then count local records", data_tasks([data_path])
    )
    assert all(subtask.status == TaskStatus.PENDING for subtask in job.subtasks)
    job = await orchestrator.map_execute(job)
    assert all(subtask.status == TaskStatus.COMPLETED for subtask in job.subtasks)
    return await orchestrator.reduce_verify(job)


async def demonstrate(directory):
    root = Path(directory)
    data_paths = [root / "users", root / "orders"]
    for path, rows in zip(data_paths, [[{"id": 1}], [{"id": 2}, {"id": 3}]]):
        path.mkdir(parents=True)
        (path / "records.json").write_text(json.dumps(rows))
    log_dir = root / "input-logs"
    log_dir.mkdir()
    (log_dir / "service.log").write_text("[2026-10-29 09:00] INFO fixture loaded\n")
    previous = os.environ.get("LOG_DIR")
    os.environ["LOG_DIR"] = str(log_dir)
    try:
        return {
            "minimal": await minimal(data_paths[0], root / "minimal"),
            "fan_out": await fan_out(data_paths, root / "fan-out"),
            "heterogeneous": await heterogeneous(data_paths[0], root / "mixed"),
            "custom": await custom_gate_and_reducer(data_paths[0], root / "custom"),
            "step_by_step": await step_by_step(data_paths[0], root / "steps"),
        }
    finally:
        if previous is None:
            os.environ.pop("LOG_DIR", None)
        else:
            os.environ["LOG_DIR"] = previous


def main():
    previous = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="mapreduce-demo-") as directory:
        try:
            # Existing tracker/logger write relative paths; contain these too.
            os.chdir(directory)
            jobs = asyncio.run(demonstrate(Path(directory) / "fixtures"))
            print(
                json.dumps(
                    {name: job.reduced_result for name, job in jobs.items()}, indent=2
                )
            )
        finally:
            os.chdir(previous)


if __name__ == "__main__":
    main()
