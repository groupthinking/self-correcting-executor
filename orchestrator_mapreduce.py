"""
Hierarchical Orchestration Engine with Map/Reduce
=================================================
Drop-in module for the self-correcting-executor project.

Extends the existing OrchestrationEngine with parallel fan-out,
verification gateway, and self-correcting mutation on failure.

Usage:
    from orchestrator_mapreduce import HierarchicalOrchestrator

    orchestrator = HierarchicalOrchestrator(max_concurrency=10)
    job = await orchestrator.run(
        intent="Validate all API endpoints after deployment",
        task_list=[
            {"protocol": "api_health_checker", "inputs": {"endpoint": "/users"}},
            {"protocol": "api_health_checker", "inputs": {"endpoint": "/orders"}},
        ],
    )
"""

import asyncio
import uuid
import time
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from utils.logger import log
from agents.mutator import mutate_protocol
from utils.tracker import track_outcome, get_protocol_stats


# ─── Data Models ───────────────────────────────────────────────


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class SubTask:
    """A single unit of work dispatched to a worker."""

    id: str
    protocol: str
    inputs: Dict[str, Any]
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    attempts: int = 0
    max_attempts: int = 3
    started_at: Optional[float] = None
    completed_at: Optional[float] = None


@dataclass
class OrchestratedJob:
    """A top-level job decomposed into parallel subtasks."""

    id: str
    intent: str
    subtasks: List[SubTask] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    reduced_result: Optional[Dict[str, Any]] = None
    verification_passed: bool = False
    created_at: float = field(default_factory=time.time)


# ─── The Orchestrator ──────────────────────────────────────────


class HierarchicalOrchestrator:
    """
    Three-phase orchestration:
      1. PLAN  — Decompose intent into subtasks
      2. MAP   — Execute subtasks in parallel (isolated state)
      3. REDUCE — Collect, verify, and self-correct
    """

    def __init__(
        self,
        max_concurrency: int = 10,
        verification_gate: Optional[Callable] = None,
        reducer: Optional[Callable] = None,
        state_file: str = "STATE.md",
    ):
        self.max_concurrency = max_concurrency
        self.verification_gate = verification_gate or self._default_gate
        self.reducer = reducer or self._default_reducer
        self.jobs: Dict[str, OrchestratedJob] = {}
        self.state_file = state_file

    # ─── Phase 1: PLAN ─────────────────────────────────────────

    async def plan(
        self, intent: str, task_list: List[Dict[str, Any]]
    ) -> OrchestratedJob:
        """
        Decompose a high-level intent into parallel subtasks.

        Args:
            intent: Human-readable description of the job
            task_list: List of dicts with "protocol" and "inputs" keys

        Returns:
            OrchestratedJob with subtasks ready for execution
        """
        job_id = f"job_{int(time.time())}_{uuid.uuid4().hex[:8]}"

        subtasks = [
            SubTask(
                id=f"{job_id}_sub_{i}",
                protocol=task["protocol"],
                inputs=task.get("inputs", {}),
            )
            for i, task in enumerate(task_list)
        ]

        job = OrchestratedJob(
            id=job_id,
            intent=intent,
            subtasks=subtasks,
        )
        self.jobs[job_id] = job

        log(f"📋 PLAN: Decomposed '{intent}' into {len(subtasks)} parallel subtasks")
        return job

    # ─── Phase 2: MAP (Fan-Out) ────────────────────────────────

    async def map_execute(self, job: OrchestratedJob) -> OrchestratedJob:
        """
        Execute all subtasks in parallel with bounded concurrency.
        Each subtask runs in isolation — no shared mutable state.
        """
        job.status = TaskStatus.RUNNING
        semaphore = asyncio.Semaphore(self.max_concurrency)

        log(
            f"🚀 MAP: Dispatching {len(job.subtasks)} workers "
            f"(concurrency={self.max_concurrency})"
        )

        async def _execute_one(subtask: SubTask):
            async with semaphore:
                subtask.status = TaskStatus.RUNNING
                subtask.started_at = time.time()
                subtask.attempts += 1

                try:
                    result = await self._run_protocol_isolated(
                        subtask.protocol, subtask.inputs
                    )
                    subtask.result = result
                    subtask.status = TaskStatus.COMPLETED
                    track_outcome(subtask.protocol, result)

                except Exception as e:
                    subtask.error = str(e)
                    subtask.status = TaskStatus.FAILED
                    track_outcome(
                        subtask.protocol, {"success": False, "error": str(e)}
                    )

                subtask.completed_at = time.time()

        # Fan-out: all subtasks run concurrently
        await asyncio.gather(
            *[_execute_one(st) for st in job.subtasks],
            return_exceptions=True,
        )

        completed = sum(
            1 for st in job.subtasks if st.status == TaskStatus.COMPLETED
        )
        failed = sum(1 for st in job.subtasks if st.status == TaskStatus.FAILED)
        log(f"📊 MAP complete: {completed} succeeded, {failed} failed")

        return job

    # ─── Phase 3: REDUCE (Fan-In + Verify + Mutate) ────────────

    async def reduce_verify(self, job: OrchestratedJob) -> OrchestratedJob:
        """
        Collect results, run through Verification Gateway,
        and trigger self-correction (mutation) on failure.
        """
        # Step 1: Reduce — aggregate results
        results = [st.result for st in job.subtasks if st.result]
        failures = [st for st in job.subtasks if st.status == TaskStatus.FAILED]

        job.reduced_result = await self.reducer(results, failures)

        # Step 2: Verify — run through the Verification Gateway
        verification = await self.verification_gate(job)

        if verification["passed"]:
            job.verification_passed = True
            job.status = TaskStatus.COMPLETED
            log(f"✅ REDUCE: Verification PASSED for job '{job.intent}'")
        else:
            job.verification_passed = False
            log(f"❌ REDUCE: Verification FAILED — {verification['reason']}")

            # Step 3: Self-Correct — retry failed subtasks with mutation
            retryable = [
                st for st in failures if st.attempts < st.max_attempts
            ]

            if retryable:
                log(
                    f"🔄 MUTATE: Retrying {len(retryable)} failed subtasks "
                    f"after mutation"
                )
                for st in retryable:
                    mutated = mutate_protocol(st.protocol)
                    if mutated:
                        log(f"   → Protocol '{st.protocol}' mutated before retry")
                    st.status = TaskStatus.RETRYING

                # Re-run failed subtasks
                await asyncio.gather(
                    *[self._retry_subtask(st) for st in retryable],
                    return_exceptions=True,
                )

                # Re-verify after retry
                job.reduced_result = await self.reducer(
                    [st.result for st in job.subtasks if st.result],
                    [
                        st
                        for st in job.subtasks
                        if st.status == TaskStatus.FAILED
                    ],
                )
                re_verification = await self.verification_gate(job)
                job.verification_passed = re_verification["passed"]
                job.status = (
                    TaskStatus.COMPLETED
                    if re_verification["passed"]
                    else TaskStatus.FAILED
                )
            else:
                # All retries exhausted — escalate to human
                job.status = TaskStatus.FAILED
                await self._escalate_to_human(job, verification["reason"])

        # Step 4: Persist state
        await self._update_state_file(job)
        return job

    # ─── Full Orchestration Cycle ──────────────────────────────

    async def run(
        self, intent: str, task_list: List[Dict[str, Any]]
    ) -> OrchestratedJob:
        """
        Execute the full Plan → Map → Reduce cycle.
        This is the single entry point.
        """
        job = await self.plan(intent, task_list)
        job = await self.map_execute(job)
        job = await self.reduce_verify(job)
        return job

    # ─── Internal Helpers ──────────────────────────────────────

    async def _run_protocol_isolated(
        self, protocol_name: str, inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a protocol in isolation.
        Uses the existing protocol loader but wraps it in async.
        """
        from protocols.loader import load_protocol

        protocol = load_protocol(protocol_name)
        if not protocol:
            raise Exception(f"Protocol '{protocol_name}' not found")

        # Run in executor to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, protocol["task"])
        return result

    async def _retry_subtask(self, subtask: SubTask):
        """Retry a single failed subtask after mutation."""
        subtask.attempts += 1
        subtask.started_at = time.time()
        try:
            result = await self._run_protocol_isolated(
                subtask.protocol, subtask.inputs
            )
            subtask.result = result
            subtask.status = TaskStatus.COMPLETED
        except Exception as e:
            subtask.error = str(e)
            subtask.status = TaskStatus.FAILED
        subtask.completed_at = time.time()

    async def _escalate_to_human(self, job: OrchestratedJob, reason: str):
        """
        Graceful degradation: when all retries are exhausted,
        escalate to a human via MCP connector (Slack, email, etc.)
        """
        log(f"🚨 ESCALATION: Job '{job.intent}' failed after all retries.")
        log(f"   Reason: {reason}")
        log(
            f"   Failed subtasks: "
            f"{[st.id for st in job.subtasks if st.status == TaskStatus.FAILED]}"
        )

        # In production, uncomment to use your MCP Slack connector:
        # import subprocess
        # subprocess.run([
        #     "manus-mcp-cli", "tool", "call", "slack_post_message",
        #     "--server", "slack",
        #     "--input", json.dumps({
        #         "channel": "#agent-alerts",
        #         "text": f"🚨 Loop failed: {job.intent}\nReason: {reason}\nManual review required."
        #     })
        # ])

    async def _update_state_file(self, job: OrchestratedJob):
        """Persist job state to STATE.md for cross-run continuity."""
        state_entry = (
            f"\n## Job: {job.id}\n"
            f"- **Intent:** {job.intent}\n"
            f"- **Status:** {job.status.value}\n"
            f"- **Verification:** "
            f"{'PASSED' if job.verification_passed else 'FAILED'}\n"
            f"- **Subtasks:** {len(job.subtasks)} total, "
            f"{sum(1 for st in job.subtasks if st.status == TaskStatus.COMPLETED)} completed\n"
            f"- **Timestamp:** {datetime.utcnow().isoformat()}\n"
        )
        with open(self.state_file, "a") as f:
            f.write(state_entry)

    # ─── Default Gate & Reducer ────────────────────────────────

    async def _default_gate(self, job: OrchestratedJob) -> Dict[str, Any]:
        """
        Default Verification Gateway: passes if >80% of subtasks succeeded.
        Replace this with your actual verification logic.
        """
        total = len(job.subtasks)
        succeeded = sum(
            1 for st in job.subtasks if st.status == TaskStatus.COMPLETED
        )
        rate = succeeded / total if total > 0 else 0

        if rate >= 0.8:
            return {"passed": True, "success_rate": rate}
        else:
            return {
                "passed": False,
                "success_rate": rate,
                "reason": f"Success rate {rate:.0%} below 80% threshold",
            }

    async def _default_reducer(
        self, results: List[Dict], failures: List[SubTask]
    ) -> Dict[str, Any]:
        """
        Default reducer: aggregate results into a summary.
        Replace this with domain-specific aggregation logic.
        """
        total = len(results) + len(failures)
        return {
            "total_results": len(results),
            "total_failures": len(failures),
            "success_rate": len(results) / total if total > 0 else 0,
            "results": results,
        }
