"""Callable canonical equivalents of the six archived workflows API routes.

The host supplies storage, input transport, provider configuration and execution
bindings. This adapter keeps those dependencies out of serialized templates.
"""

from copy import deepcopy
from uuid import uuid4

from .build_plan import BuildPlanExecutor, BuildPlan, ExecSummary, now
from .plan_scaffold import generate_plan_scaffold
from .planning import WorkflowPlanner


class ImportedWorkflows:
    def __init__(self, store, *, ingestor=None, planner=None, executor=None):
        self.store = store
        self.ingestor = ingestor
        self.planner = planner or WorkflowPlanner()
        self.executor = executor or BuildPlanExecutor(store, llm=self.planner.agent)

    async def execute(self, arguments):
        values = deepcopy(arguments)
        operation = values.pop("operation")
        allowed = {"ingest", "summarize", "plan", "compile", "run", "scaffold"}
        if operation not in allowed:
            raise ValueError(f"Unknown imported workflow operation: {operation}")
        return await getattr(self, operation)(**values)

    async def ingest(self, url, source_type=None, language=None, run_id=None):
        if self.ingestor is None:
            raise ValueError("Source intake requires a configured Ingestor")
        run_id = run_id or str(uuid4())
        self.store.status(run_id, "pending", source_url=url)
        self.store.status(run_id, "ingesting")
        try:
            source = await self.ingestor.ingest(
                url, source_type=source_type, language=language
            )
            source.update(id=str(uuid4()), workflow_run_id=run_id, created_at=now())
            self.store.put("sources", source["id"], source)
            self.store.status(
                run_id,
                "ingesting",
                source_type=source["sourceType"],
                source_id=source["id"],
            )
            return source
        except Exception as error:
            self.store.status(run_id, "failed", error_message=str(error))
            raise

    async def summarize(self, source):
        saved = self.store.get("sources", source["id"])
        if saved != source:
            raise ValueError("Summary source differs from stored intake")
        run_id = source["workflow_run_id"]
        self.store.status(run_id, "planning")
        try:
            value = await self.planner.summarize(source["content"])
            summary = ExecSummary.model_validate(value).model_dump()
            summary.update(
                id=str(uuid4()),
                workflow_run_id=run_id,
                created_at=now(),
                raw_content=source["content"],
                metadata={
                    "source_id": source["id"],
                    "content_sha256": source["metadata"]["content_sha256"],
                },
            )
            return self.store.put("summaries", summary["id"], summary)
        except Exception as error:
            self.store.status(run_id, "failed", error_message=str(error))
            raise

    async def plan(self, summary):
        if self.store.get("summaries", summary["id"]) != summary:
            raise ValueError("Plan summary differs from stored summary")
        run_id = summary["workflow_run_id"]
        try:
            value = await self.planner.plan(summary)
            value.update(
                id=str(uuid4()), workflow_run_id=run_id, exec_summary_id=summary["id"]
            )
            plan = BuildPlan.model_validate(value).model_dump()
            self.store.status(run_id, "compiling", build_plan_id=plan["id"])
            return self.store.put("plans", plan["id"], plan)
        except Exception as error:
            self.store.status(run_id, "failed", error_message=str(error))
            raise

    async def compile(self, plan):
        if self.store.get("plans", plan["id"]) != plan:
            raise ValueError("Compile plan differs from stored plan")
        graph = self.executor.compile(plan)
        return self.store.put("compiled", plan["id"], graph)

    async def run(self, compiled, inputs=None):
        if self.store.get("compiled", compiled["workflowId"]) != compiled:
            raise ValueError("Compiled graph differs from stored graph")
        return await self.executor.execute(compiled, inputs)

    async def scaffold(self, plan):
        if self.store.get("plans", plan["id"]) != plan:
            raise ValueError("Artifact plan differs from stored plan")
        self.store.status(
            plan["workflow_run_id"],
            "deploying",
            deployment_kind="scaffold_generation_only",
        )
        try:
            bundle = generate_plan_scaffold(plan)
            artifacts = [
                self.store.artifact(
                    plan["workflow_run_id"],
                    plan["id"],
                    "code_scaffold",
                    file_path=item["path"],
                    content=item["content"],
                    metadata={
                        "language": item["language"],
                        "deployment_status": "scaffold_only",
                    },
                )
                for item in bundle["files"]
            ]
            self.store.status(
                plan["workflow_run_id"], "completed", deployment_status="scaffold_only"
            )
            return {**bundle, "artifacts": artifacts}
        except Exception as error:
            self.store.status(
                plan["workflow_run_id"], "failed", error_message=str(error)
            )
            raise
