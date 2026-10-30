"""Run a BuildPlan example or a configured source-to-plan workflow.

No external services are contacted by --demo. It executes a real two-task graph,
stores every trajectory and emits a usable TypeScript library artifact bundle.
A URL run requires a trusted --bindings module exposing configure(store) with
ImportedWorkflows constructor arguments (ingestor, planner, executor as needed).
"""

import argparse
import asyncio
import importlib
import json
from pathlib import Path

from orchestrator import OrchestrationEngine
from workflows.build_plan import BuildPlan, BuildPlanExecutor, WorkflowStore
from workflows.correction import invoke
from workflows.imported import ImportedWorkflows


async def run(args):
    store = WorkflowStore(args.database)
    try:
        inputs = (
            json.loads(Path(args.inputs).read_text())
            if args.inputs
            else {"values": [2, 3, 5]}
        )
        if args.demo:
            plan = {
                "id": "local-example-plan",
                "workflow_run_id": "local-example-run",
                "exec_summary_id": "local-example-summary",
                "summary_title": "Local graph example",
                "summary_goal": "Sum supplied values and double the observed result",
                "agents": [
                    {
                        "name": "calculator",
                        "kind": "CustomAgent",
                        "description": "Compute the selected task",
                    }
                ],
                "tasks": [
                    {"id": "double", "name": "Double", "depends_on": ["sum"]},
                    {"id": "sum", "name": "Sum", "depends_on": []},
                ],
                "mcp_tools": [],
                "a2a_required": False,
                "ap2_required": False,
                "metadata": {"source": "explicit local example, not AI generation"},
            }

            def calculate(value, context):
                if context["task"]["id"] == "sum":
                    return {"value": sum(value["values"])}
                return {"value": context["dependencies"]["sum"]["value"] * 2}

            executor = BuildPlanExecutor(store, handlers={"calculator": calculate})
            service = ImportedWorkflows(store, executor=executor)
            # This is a caller-defined plan example, not a claimed LLM summary.
            plan = BuildPlan.model_validate(plan).model_dump()
            store.put("plans", plan["id"], plan)
            compiled = await service.compile(plan)
            result = await service.run(compiled, inputs)
            if result["status"] != "completed":
                return result
            artifact = await service.scaffold(plan)
            return {
                "execution": result,
                "artifacts_saved": len(artifact["artifacts"]),
                "deployment_status": artifact["deployment_status"],
            }
        if not args.bindings:
            raise ValueError("A URL run requires --bindings with actual host services")
        module = importlib.import_module(args.bindings)
        configured = await invoke(module.configure, store)
        service = ImportedWorkflows(store, **configured)
        engine = OrchestrationEngine()
        engine.agents["imported_workflows"] = service
        return await engine.execute_intent(
            "Source build plan",
            options={
                "workflow_template": "build_plan_pipeline",
                "inputs": {"source_url": args.url, "task_inputs": inputs},
            },
        )
    finally:
        store.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--demo", action="store_true")
    choice.add_argument("--url")
    parser.add_argument("--bindings")
    parser.add_argument("--inputs")
    args = parser.parse_args()
    try:
        result = asyncio.run(run(args))
        print(json.dumps(result, indent=2))
        if result.get("status") == "failed":
            raise SystemExit(1)
    except (ValueError, RuntimeError, ImportError, KeyError) as error:
        raise SystemExit(f"Workflow failed: {error}") from error


if __name__ == "__main__":
    main()
