"""Executable port of workflows/lib/types/workflow.ts and workflow-compiler.ts.

Round-robin task/agent assignment is retained. Unlike the archived compiler,
dependency order is validated and execution invokes handlers instead of sleeping.
Composite membership and loop termination were absent upstream; callers supply
those runtime bindings explicitly, without changing the imported plan schema.
"""

import asyncio
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sqlite3
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator

from .correction import NonRetryableError, invoke, run_correcting

AgentKind = Literal[
    "SequentialAgent", "ParallelAgent", "LoopAgent", "LlmAgent", "CustomAgent"
]
WorkflowStatus = Literal[
    "pending", "ingesting", "planning", "compiling", "deploying", "completed", "failed"
]
ArtifactType = Literal["code_scaffold", "full_app", "docker_image", "deployment_config"]


def now():
    return datetime.now(timezone.utc).isoformat()


class SourceModel(BaseModel):
    model_config = ConfigDict(extra="allow")


class ExecSummary(SourceModel):
    title: str = Field(min_length=1)
    goal: str = Field(min_length=1)
    key_requirements: list[str]
    constraints: list[str]
    assumptions: list[str]
    success_metrics: list[str]


class AgentSpec(SourceModel):
    name: str = Field(min_length=1)
    kind: AgentKind
    description: str


class Task(SourceModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    depends_on: list[str]


class BuildPlan(SourceModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    workflow_run_id: str = Field(default_factory=lambda: str(uuid4()))
    exec_summary_id: str | None = None
    created_at: str = Field(default_factory=now)
    summary_title: str = Field(min_length=1)
    summary_goal: str = Field(min_length=1)
    agents: list[AgentSpec] = Field(min_length=1)
    tasks: list[Task] = Field(min_length=1)
    mcp_tools: list[str]
    a2a_required: StrictBool
    ap2_required: StrictBool

    @model_validator(mode="after")
    def valid_graph(self):
        names = [agent.name for agent in self.agents]
        ids = [task.id for task in self.tasks]
        if len(names) != len(set(names)) or len(ids) != len(set(ids)):
            raise ValueError("Agent names and task IDs must be unique")
        if len(self.mcp_tools) != len(set(self.mcp_tools)):
            raise ValueError("MCP tool names must be unique")
        if any(dep not in ids for task in self.tasks for dep in task.depends_on):
            raise ValueError("Task dependency is missing")
        pending = {task.id: set(task.depends_on) for task in self.tasks}
        done = set()
        while pending:
            ready = {key for key, deps in pending.items() if deps <= done}
            if not ready:
                raise ValueError("Task graph contains a cycle")
            done.update(ready)
            pending = {key: deps for key, deps in pending.items() if key not in ready}
        return self


class WorkflowStore:
    """Durable local execution/status/trajectory/artifact store, explicit location.

    This is the canonical engine's local store, not a claim to provide the old
    application's hosted Supabase UI. Reopening the file retains all attempts.
    """

    def __init__(self, path):
        if str(path) != ":memory:":
            # The local outbox may contain invitation links. Create the database
            # privately from the outset, rather than exposing a permissive window.
            fd = (
                os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                if not Path(path).exists()
                else None
            )
            if fd is not None:
                os.close(fd)
        self.db = sqlite3.connect(str(path))
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS objects "
            "(kind TEXT, id TEXT, body TEXT NOT NULL, PRIMARY KEY(kind,id))"
        )
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS events "
            "(seq INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, body TEXT NOT NULL)"
        )
        self.db.commit()

    def put(self, kind, key, value):
        serialized = json.dumps(value, allow_nan=False)
        self.db.execute(
            "INSERT INTO objects VALUES(?,?,?) ON CONFLICT(kind,id) "
            "DO UPDATE SET body=excluded.body",
            (kind, key, serialized),
        )
        self.db.commit()
        return self.get(kind, key)

    def get(self, kind, key):
        row = self.db.execute(
            "SELECT body FROM objects WHERE kind=? AND id=?", (kind, key)
        ).fetchone()
        if row is None:
            raise KeyError(f"Missing {kind}: {key}")
        return json.loads(row[0])

    def all(self, kind):
        return [
            json.loads(row[0])
            for row in self.db.execute(
                "SELECT body FROM objects WHERE kind=? ORDER BY id", (kind,)
            )
        ]

    def event(self, run_id, value):
        self.db.execute(
            "INSERT INTO events(run_id,body) VALUES(?,?)",
            (run_id, json.dumps({"created_at": now(), **value}, allow_nan=False)),
        )
        self.db.commit()

    def events(self, run_id):
        return [
            json.loads(row[0])
            for row in self.db.execute(
                "SELECT body FROM events WHERE run_id=? ORDER BY seq", (run_id,)
            )
        ]

    def status(self, run_id, status: WorkflowStatus, **metadata):
        if status not in WorkflowStatus.__args__:
            raise ValueError("Unknown workflow status")
        try:
            run = self.get("runs", run_id)
        except KeyError:
            run = {"id": run_id, "created_at": now()}
        run.update(status=status, updated_at=now(), **metadata)
        self.put("runs", run_id, run)
        self.event(run_id, {"type": "workflow_status", "status": status})

    def artifact(self, run_id, plan_id, artifact_type: ArtifactType, **values):
        if artifact_type not in ArtifactType.__args__:
            raise ValueError("Unknown artifact type")
        artifact = dict(
            id=str(uuid4()),
            workflow_run_id=run_id,
            build_plan_id=plan_id,
            artifact_type=artifact_type,
            created_at=now(),
            **values,
        )
        return self.put("artifacts", artifact["id"], artifact)

    def close(self):
        self.db.close()


class ProtocolBindings:
    """Host-owned adapters. A protocol flag does not authorize a payment or tool.

    MCP expects list_tools/call_tool, A2A discover_agents/dispatch_task, AP2
    validate_mandate/process_payment. No provider, mandate or spend is invented.
    Tool names and payment parameters must be supplied by the application.
    """

    def __init__(self, *, mcp=None, a2a=None, ap2=None, mandate=None):
        self.mcp, self.a2a, self.ap2, self.mandate = mcp, a2a, ap2, mandate

    async def call(self, plan, protocol, name, parameters):
        if protocol == "mcp":
            if self.mcp is None or name not in plan.mcp_tools:
                raise ValueError("MCP call is not declared by this plan")
            result = await invoke(self.mcp.call_tool, name, deepcopy(parameters))
            if getattr(result, "isError", False) or (
                isinstance(result, dict) and result.get("isError") is True
            ):
                raise RuntimeError("MCP tool reported failure")
            return (
                result.model_dump(mode="json")
                if hasattr(result, "model_dump")
                else result
            )
        if protocol == "a2a":
            if not plan.a2a_required or self.a2a is None:
                raise ValueError("A2A is not enabled for this plan")
            result = await invoke(self.a2a.dispatch_task, name, deepcopy(parameters))
            if not isinstance(result, dict) or result.get("status") != "completed":
                raise RuntimeError("A2A task did not complete")
            return result
        if protocol == "ap2":
            if not plan.ap2_required or self.ap2 is None or not self.mandate:
                raise ValueError("AP2 is not enabled and authorized for this plan")
            amount = parameters.get("amount")
            maximum = self.mandate.get("maxAmount")
            if (
                type(maximum) not in {int, float}
                or not math.isfinite(maximum)
                or maximum <= 0
            ):
                raise ValueError("AP2 mandate requires a finite positive maximum")
            if (
                type(amount) not in {int, float}
                or not math.isfinite(amount)
                or amount <= 0
                or amount > self.mandate["maxAmount"]
                or parameters.get("currency") != self.mandate["currency"]
                or name != self.mandate["vendorId"]
                or not isinstance(parameters.get("idempotencyKey"), str)
                or not parameters["idempotencyKey"]
            ):
                raise ValueError("Payment does not match the caller's mandate")
            if await invoke(self.ap2.validate_mandate, self.mandate) is not True:
                raise ValueError("AP2 mandate is no longer valid")
            result = await invoke(
                self.ap2.process_payment, deepcopy(self.mandate), deepcopy(parameters)
            )
            if not isinstance(result, dict) or result.get("status") not in {
                "authorized",
                "completed",
            }:
                raise RuntimeError("AP2 provider did not authorize payment")
            return result
        raise ValueError("Unknown protocol")

    async def prepare(self, plan):
        if plan.mcp_tools:
            if self.mcp is None:
                raise ValueError("Plan requires configured MCP client")
            tools = await invoke(self.mcp.list_tools)
            names = {t.name if hasattr(t, "name") else t["name"] for t in tools}
            missing = set(plan.mcp_tools) - names
            if missing:
                raise ValueError(f"MCP tools unavailable: {sorted(missing)}")
        if plan.a2a_required:
            if self.a2a is None:
                raise ValueError("Plan requires configured A2A registry")
            agents = await invoke(self.a2a.discover_agents, "code_generation")
            if not agents:
                raise ValueError("No A2A agents discovered")
        if plan.ap2_required:
            if self.ap2 is None or self.mandate is None:
                raise ValueError("AP2 requires a caller-authorized mandate and client")
            valid = await invoke(self.ap2.validate_mandate, self.mandate)
            if valid is not True:
                raise ValueError("AP2 mandate invalid")


class BuildPlanExecutor:
    def __init__(
        self,
        store,
        *,
        handlers=None,
        composites=None,
        protocols=None,
        repairs=None,
        max_attempts=2,
        self_correction=True,
        llm=None,
    ):
        self.store = store
        self.handlers = handlers or {}
        self.composites = composites or {}
        self.protocols = protocols or ProtocolBindings()
        self.repairs = repairs or {}
        self.max_attempts = max_attempts if self_correction else 1
        self.llm = llm

    def compile(self, value):
        plan = BuildPlan.model_validate(deepcopy(value))
        agents = [
            dict(
                id=f"agent_{i}",
                **agent.model_dump(),
                inputs=[],
                outputs=[],
                tools=list(plan.mcp_tools),
            )
            for i, agent in enumerate(plan.agents)
        ]
        tasks = [
            dict(
                id=task.id,
                name=task.name,
                agentId=agents[i % len(agents)]["id"],
                dependencies=task.depends_on,
                status="pending",
            )
            for i, task in enumerate(plan.tasks)
        ]
        return {
            "workflowId": plan.id,
            "agentGraph": agents,
            "taskGraph": tasks,
            "plan": plan.model_dump(),
            "metadata": {"compiledAt": now()},
        }

    async def _agent(self, agent, inputs, context, checkpoints, path=(), stack=()):
        name, kind = agent["name"], agent["kind"]
        key = (*path, name)
        if key in checkpoints:
            saved = checkpoints[key]
            if saved["input"] != inputs:
                raise NonRetryableError(
                    "Retry input changed after a completed child; reconcile before replay"
                )
            return deepcopy(saved["output"])
        if name in stack:
            raise ValueError("Composite agent cycle")
        if kind in {"CustomAgent", "LlmAgent"}:
            handler = self.handlers.get(name)
            if handler is not None:
                result = await invoke(handler, deepcopy(inputs), context)
            elif kind == "LlmAgent" and self.llm is not None:
                result = await invoke(self.llm, agent, deepcopy(inputs), context)
            else:
                raise ValueError(f"No executable binding for {kind}: {name}")
            if isinstance(result, dict) and (
                result.get("success") is False
                or result.get("status") in {"failed", "error"}
            ):
                raise RuntimeError(
                    result.get("error", f"Agent {name} reported failure")
                )
        else:
            config = self.composites.get(name)
            if not config or not config.get("children"):
                raise ValueError(f"{kind} requires explicit child bindings: {name}")
            by_name = {a["name"]: a for a in context["agents"]}
            try:
                children = [by_name[child] for child in config["children"]]
            except KeyError as error:
                raise ValueError("Composite child is absent from plan") from error

            async def child_run(child, value, index, iteration=0):
                return await self._agent(
                    child,
                    value,
                    context,
                    checkpoints,
                    (*key, iteration, index),
                    (*stack, name),
                )

            if kind == "ParallelAgent":
                if len(config["children"]) != len(set(config["children"])):
                    raise ValueError("Parallel children must have unique names")
                # Await every owned child before retrying any failed child. A
                # successful sibling is checkpointed and is never replayed.
                values = await asyncio.gather(
                    *(
                        child_run(c, deepcopy(inputs), i)
                        for i, c in enumerate(children)
                    ),
                    return_exceptions=True,
                )
                errors = [v for v in values if isinstance(v, BaseException)]
                if errors:
                    raise next(
                        (e for e in errors if getattr(e, "retryable", True) is False),
                        errors[0],
                    )
                result = {c["name"]: value for c, value in zip(children, values)}
            else:

                async def sequence(value, iteration=0):
                    for index, child in enumerate(children):
                        value = await child_run(child, value, index, iteration)
                    return value

                if kind == "SequentialAgent":
                    result = await sequence(deepcopy(inputs))
                else:
                    limit, until = config.get("max_iterations"), config.get("until")
                    if (
                        type(limit) is not int
                        or not 1 <= limit <= 1000
                        or not callable(until)
                    ):
                        raise ValueError(
                            "LoopAgent requires bounded max_iterations and until"
                        )
                    result = deepcopy(inputs)
                    for iteration in range(1, limit + 1):
                        result = await sequence(result, iteration)
                        if await invoke(until, deepcopy(result), iteration) is True:
                            break
                    else:
                        raise RuntimeError(
                            f"LoopAgent {name} did not reach its stop condition"
                        )
        try:
            checkpoints[key] = {"input": deepcopy(inputs), "output": deepcopy(result)}
        except Exception as error:
            raise NonRetryableError(
                "Completed child could not be checkpointed; no replay"
            ) from error
        return result

    async def execute(self, compiled, inputs=None):
        # Revalidate the source; don't trust a serialized compiled graph.
        graph = self.compile(compiled["plan"])
        plan = BuildPlan.model_validate(graph["plan"])
        run = plan.workflow_run_id
        self.store.status(run, "compiling", build_plan_id=plan.id)
        outputs, attempts = {}, {}
        self.store.put("plans", plan.id, plan.model_dump())
        try:
            await self.protocols.prepare(plan)
            pending = list(graph["taskGraph"])
            agents = {agent["id"]: agent for agent in graph["agentGraph"]}
            while pending:
                ready = [
                    task
                    for task in pending
                    if all(d in outputs for d in task["dependencies"])
                ]
                if not ready:
                    raise ValueError("Task graph could not make progress")
                for task in ready:
                    agent = agents[task["agentId"]]
                    context = {
                        "task": deepcopy(task),
                        "plan": plan.model_dump(),
                        "agents": graph["agentGraph"],
                        "dependencies": {
                            d: deepcopy(outputs[d]) for d in task["dependencies"]
                        },
                        "protocols": self.protocols,
                    }

                    async def call_tool(protocol, name, parameters):
                        call_id = str(uuid4())
                        event = {
                            "type": "tool_call",
                            "id": call_id,
                            "task_id": task["id"],
                            "protocol": protocol,
                            "tool_name": name,
                            "parameters": deepcopy(parameters),
                        }
                        self.store.event(run, {**event, "status": "started"})
                        try:
                            value = await self.protocols.call(
                                plan, protocol, name, parameters
                            )
                            try:
                                self.store.event(
                                    run,
                                    {**event, "status": "completed", "result": value},
                                )
                            except Exception as observation_error:
                                raise NonRetryableError(
                                    "Tool returned successfully but receipt persistence failed; no replay"
                                ) from observation_error
                            return value
                        except NonRetryableError:
                            raise
                        except Exception as error:
                            self.store.event(
                                run,
                                {
                                    **event,
                                    "status": "failed",
                                    "error_message": str(error),
                                },
                            )
                            raise

                    checkpoints = {}
                    context["call_tool"] = call_tool
                    task["status"] = "running"

                    async def operation(value):
                        return await self._agent(agent, value, context, checkpoints)

                    def observe(entry):
                        self.store.event(
                            run,
                            {
                                "type": "trajectory",
                                "task_id": task["id"],
                                "agent_name": agent["name"],
                                "agent_kind": agent["kind"],
                                **entry,
                            },
                        )

                    result = await run_correcting(
                        operation,
                        inputs or {},
                        attempts=self.max_attempts,
                        repair=self.repairs.get(task["id"]),
                        observe=observe,
                    )
                    attempts[task["id"]] = result["attempts"]
                    task["status"] = result["status"]
                    if result["status"] == "failed":
                        raise RuntimeError(result["error"])
                    outputs[task["id"]] = result["output"]
                    pending.remove(task)
            self.store.status(run, "completed")
            result = {
                "status": "completed",
                "outputs": outputs,
                "attempts": attempts,
                "taskGraph": graph["taskGraph"],
            }
        except Exception as error:
            self.store.status(run, "failed", error_message=str(error))
            result = {
                "status": "failed",
                "error": str(error),
                "outputs": outputs,
                "attempts": attempts,
                "taskGraph": graph["taskGraph"],
            }
        self.store.put("results", run, result)
        return result
