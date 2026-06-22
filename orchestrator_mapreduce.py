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

__all__ = ["HierarchicalOrchestrator", "OrchestratedJob", "SubTask", "TaskStatus"]

import asyncio
import uuid
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone

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
        state_file: Optional[str] = None,
        subtask_timeout: float = 300.0,
        max_job_history: int = 1000,
    ):
        self.max_concurrency = max_concurrency
        self.verification_gate = verification_gate or self._default_gate
        self.reducer = reducer or self._default_reducer
        self.jobs: Dict[str, OrchestratedJob] = {}
        self.state_file = state_file or str(
            Path(__file__).parent / "STATE.md"
        )
        self.subtask_timeout = subtask_timeout
        self.max_job_history = max_job_history

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
        self._prune_job_history()

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
        loop = asyncio.get_running_loop()

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
                    # Apply timeout to prevent hanging protocols
                    result = await asyncio.wait_for(
                        self._run_protocol_isolated(
                            subtask.protocol, subtask.inputs
                        ),
                        timeout=self.subtask_timeout,
                    )
                    subtask.result = result
                    subtask.status = TaskStatus.COMPLETED

                    # Track outcome without blocking the event loop
                    try:
                        await loop.run_in_executor(
                            None, track_outcome, subtask.protocol, result
                        )
                    except Exception:
                        pass  # Tracking failure shouldn't fail the subtask

                except asyncio.TimeoutError:
                    subtask.error = (
                        f"Protocol '{subtask.protocol}' timed out "
                        f"after {self.subtask_timeout}s"
                    )
                    subtask.status = TaskStatus.FAILED

                except Exception as e:
                    subtask.error = str(e)
                    subtask.status = TaskStatus.FAILED
                    try:
                        await loop.run_in_executor(
                            None,
                            track_outcome,
                            subtask.protocol,
                            {"success": False, "error": str(e)},
                        )
                    except Exception:
                        pass

                finally:
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
        Retries up to max_attempts per subtask.
        """
        loop = asyncio.get_running_loop()

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

            # Step 3: Self-Correct — retry loop until max_attempts exhausted
            while True:
                retryable = [
                    st
                    for st in job.subtasks
                    if st.status == TaskStatus.FAILED
                    and st.attempts < st.max_attempts
                ]

                if not retryable:
                    break

                log(
                    f"🔄 MUTATE: Retrying {len(retryable)} failed subtasks "
                    f"(attempt {retryable[0].attempts + 1}/{retryable[0].max_attempts})"
                )

                # Deduplicate protocols before mutating to avoid race conditions
                unique_protocols = set(st.protocol for st in retryable)
                for proto in unique_protocols:
                    mutated = await loop.run_in_executor(
                        None, mutate_protocol, proto
                    )
                    if mutated:
                        log(f"   → Protocol '{proto}' mutated before retry")

                for st in retryable:
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

                if re_verification["passed"]:
                    job.verification_passed = True
                    job.status = TaskStatus.COMPLETED
                    log(
                        f"✅ REDUCE: Verification PASSED after retry "
                        f"for job '{job.intent}'"
                    )
                    break

            # If we exited the loop without passing — escalate
            if not job.verification_passed:
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
        Passes inputs to the protocol task function.
        """
        from protocols.loader import load_protocol

        protocol = load_protocol(protocol_name)
        if not protocol:
            raise Exception(f"Protocol '{protocol_name}' not found")

        task_fn = protocol["task"]
        loop = asyncio.get_running_loop()

        # Pass inputs to the protocol task function.
        # Protocols that accept kwargs will receive them;
        # legacy protocols with no parameters still work via fallback.
        def _execute_with_inputs():
            import inspect

            sig = inspect.signature(task_fn)
            if sig.parameters:
                # Protocol accepts arguments — pass inputs as kwargs
                return task_fn(**inputs)
            else:
                # Legacy protocol with no parameters — call without args
                return task_fn()

        result = await loop.run_in_executor(None, _execute_with_inputs)
        return result

    async def _retry_subtask(self, subtask: SubTask):
        """Retry a single failed subtask after mutation."""
        subtask.attempts += 1
        subtask.started_at = time.time()
        try:
            result = await asyncio.wait_for(
                self._run_protocol_isolated(
                    subtask.protocol, subtask.inputs
                ),
                timeout=self.subtask_timeout,
            )
            subtask.result = result
            subtask.status = TaskStatus.COMPLETED
        except asyncio.TimeoutError:
            subtask.error = (
                f"Protocol '{subtask.protocol}' timed out "
                f"after {self.subtask_timeout}s on retry"
            )
            subtask.status = TaskStatus.FAILED
        except Exception as e:
            subtask.error = str(e)
            subtask.status = TaskStatus.FAILED
        finally:
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
        # import subprocess, json
        # subprocess.run([
        #     "manus-mcp-cli", "tool", "call", "slack_post_message",
        #     "--server", "slack",
        #     "--input", json.dumps({
        #         "channel": "#agent-alerts",
        #         "text": f"Loop failed: {job.intent}\nReason: {reason}\nManual review required."
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
            f"- **Timestamp:** {datetime.now(timezone.utc).isoformat()}\n"
        )
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._write_state_sync, state_entry)

    def _write_state_sync(self, entry: str):
        """Synchronous state file append (called via run_in_executor)."""
        with open(self.state_file, "a") as f:
            f.write(entry)

    def _prune_job_history(self):
        """Prevent unbounded memory growth from stored jobs."""
        if len(self.jobs) > self.max_job_history:
            # Remove oldest jobs beyond the limit
            sorted_jobs = sorted(
                self.jobs.items(), key=lambda x: x[1].created_at
            )
            excess = len(self.jobs) - self.max_job_history
            for job_id, _ in sorted_jobs[:excess]:
                del self.jobs[job_id]

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
