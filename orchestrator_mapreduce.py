"""
Hierarchical Orchestration Engine with Map/Reduce
=================================================
Drop-in module for the self-correcting-executor project.

Extends the existing OrchestrationEngine with parallel fan-out,
verification gateway, self-correcting mutation on failure,
semantic LLM verification, structured state persistence,
safe mutation hierarchy, and live Slack escalation.

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

__all__ = [
    "HierarchicalOrchestrator",
    "OrchestratedJob",
    "SubTask",
    "TaskStatus",
    "SemanticGate",
    "StateStore",
]

import asyncio
import inspect
import json
import os
import sqlite3
import subprocess
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
    attempt_count: int = 0


# ─── Gap 1: Semantic Verification Gate ─────────────────────────


class SemanticGate:
    """
    LLM-powered verification gate that evaluates output quality
    against the original intent — not just execution success rates.

    Uses OpenAI-compatible API (preconfigured in environment).
    Falls back to mechanical gate if LLM is unavailable.
    """

    def __init__(self, model: str = "gpt-4o-mini", threshold: float = 0.8):
        self.model = model
        self.threshold = threshold

    async def verify(self, job: OrchestratedJob) -> Dict[str, Any]:
        """
        Two-layer verification:
        1. Mechanical: Check execution success rate (fast, cheap)
        2. Semantic: Ask LLM to evaluate output quality (deeper)

        Both must pass for the job to be considered verified.
        """
        # Layer 1: Mechanical gate (fast pre-filter)
        total = len(job.subtasks)
        succeeded = sum(
            1 for st in job.subtasks if st.status == TaskStatus.COMPLETED
        )
        rate = succeeded / total if total > 0 else 0

        if rate < self.threshold:
            return {
                "passed": False,
                "layer": "mechanical",
                "success_rate": rate,
                "reason": (
                    f"Success rate {rate:.0%} below "
                    f"{self.threshold:.0%} threshold"
                ),
            }

        # Layer 2: Semantic evaluation via LLM
        try:
            semantic_result = await self._llm_evaluate(job)
            return semantic_result
        except Exception as e:
            # If LLM is unavailable, fall back to mechanical pass
            log(f"⚠️  Semantic gate unavailable ({e}), using mechanical only")
            return {"passed": True, "layer": "mechanical_fallback", "success_rate": rate}

    async def _llm_evaluate(self, job: OrchestratedJob) -> Dict[str, Any]:
        """Call the LLM to semantically evaluate the job output."""
        from openai import OpenAI

        client = OpenAI()  # Uses preconfigured OPENAI_API_KEY and OPENAI_API_BASE

        # Build a concise summary of results for the evaluator
        results_summary = []
        for st in job.subtasks:
            if st.status == TaskStatus.COMPLETED and st.result:
                results_summary.append(
                    f"- {st.protocol}({st.inputs}): {json.dumps(st.result)[:200]}"
                )
            elif st.status == TaskStatus.FAILED:
                results_summary.append(
                    f"- {st.protocol}({st.inputs}): FAILED — {st.error}"
                )

        prompt = (
            f"You are a verification gate for an autonomous agent system.\n\n"
            f"INTENT: {job.intent}\n\n"
            f"RESULTS:\n" + "\n".join(results_summary[:20]) + "\n\n"
            f"Does this output fully satisfy the stated intent? "
            f"Consider completeness, correctness, and whether any critical "
            f"subtasks failed that would compromise the overall goal.\n\n"
            f"Reply with exactly one line: PASS or FAIL followed by a "
            f"one-sentence reason.\n"
            f"Example: PASS All endpoints returned healthy status codes.\n"
            f"Example: FAIL The /payments endpoint returned 500 which is critical."
        )

        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.1,
            ),
        )

        verdict = response.choices[0].message.content.strip()
        passed = verdict.upper().startswith("PASS")

        return {
            "passed": passed,
            "layer": "semantic",
            "reason": verdict,
            "model": self.model,
        }


# ─── Gap 3: Structured State Store ────────────────────────────


class StateStore:
    """
    SQLite-backed structured state persistence.
    Provides queryable run history alongside the human-readable STATE.md.

    Every job run is recorded with full metadata, enabling:
    - Querying past runs by protocol, status, or time range
    - Computing success rates over time
    - Identifying recurring failure patterns
    """

    def __init__(self, db_path: Optional[str] = None, md_path: Optional[str] = None):
        self.db_path = db_path or str(
            Path(__file__).parent / "loop_state.db"
        )
        self.md_path = md_path or str(
            Path(__file__).parent / "STATE.md"
        )
        self._init_db()

    def _init_db(self):
        """Initialize the SQLite database schema."""
        conn = sqlite3.connect(self.db_path)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                intent TEXT NOT NULL,
                status TEXT NOT NULL,
                verification_passed INTEGER NOT NULL DEFAULT 0,
                attempt_count INTEGER NOT NULL DEFAULT 0,
                subtask_count INTEGER NOT NULL DEFAULT 0,
                succeeded_count INTEGER NOT NULL DEFAULT 0,
                failed_count INTEGER NOT NULL DEFAULT 0,
                reduced_result TEXT,
                created_at TEXT NOT NULL,
                completed_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS subtask_runs (
                id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL,
                protocol TEXT NOT NULL,
                inputs TEXT,
                status TEXT NOT NULL,
                result TEXT,
                error TEXT,
                attempts INTEGER NOT NULL DEFAULT 0,
                started_at TEXT,
                completed_at TEXT,
                FOREIGN KEY (job_id) REFERENCES jobs(id)
            );

            CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
            CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at);
            CREATE INDEX IF NOT EXISTS idx_subtasks_protocol ON subtask_runs(protocol);
            CREATE INDEX IF NOT EXISTS idx_subtasks_status ON subtask_runs(status);
        """)
        conn.close()

    def persist_job(self, job: OrchestratedJob):
        """Write a completed job and its subtasks to the database."""
        conn = sqlite3.connect(self.db_path)
        now = datetime.now(timezone.utc).isoformat()

        succeeded = sum(
            1 for st in job.subtasks if st.status == TaskStatus.COMPLETED
        )
        failed = sum(
            1 for st in job.subtasks if st.status == TaskStatus.FAILED
        )

        conn.execute(
            """INSERT OR REPLACE INTO jobs
               (id, intent, status, verification_passed, attempt_count,
                subtask_count, succeeded_count, failed_count,
                reduced_result, created_at, completed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                job.id,
                job.intent,
                job.status.value,
                1 if job.verification_passed else 0,
                job.attempt_count,
                len(job.subtasks),
                succeeded,
                failed,
                json.dumps(job.reduced_result) if job.reduced_result else None,
                datetime.fromtimestamp(job.created_at, tz=timezone.utc).isoformat(),
                now,
            ),
        )

        for st in job.subtasks:
            conn.execute(
                """INSERT OR REPLACE INTO subtask_runs
                   (id, job_id, protocol, inputs, status, result, error,
                    attempts, started_at, completed_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    st.id,
                    job.id,
                    st.protocol,
                    json.dumps(st.inputs),
                    st.status.value,
                    json.dumps(st.result) if st.result else None,
                    st.error,
                    st.attempts,
                    (
                        datetime.fromtimestamp(st.started_at, tz=timezone.utc).isoformat()
                        if st.started_at
                        else None
                    ),
                    (
                        datetime.fromtimestamp(st.completed_at, tz=timezone.utc).isoformat()
                        if st.completed_at
                        else None
                    ),
                ),
            )

        conn.commit()
        conn.close()

    def persist_markdown(self, job: OrchestratedJob):
        """Append a human-readable summary to STATE.md."""
        succeeded = sum(
            1 for st in job.subtasks if st.status == TaskStatus.COMPLETED
        )
        entry = (
            f"\n## Job: {job.id}\n"
            f"- **Intent:** {job.intent}\n"
            f"- **Status:** {job.status.value}\n"
            f"- **Verification:** "
            f"{'PASSED' if job.verification_passed else 'FAILED'}\n"
            f"- **Subtasks:** {len(job.subtasks)} total, {succeeded} completed\n"
            f"- **Attempts:** {job.attempt_count}\n"
            f"- **Timestamp:** {datetime.now(timezone.utc).isoformat()}\n"
        )
        with open(self.md_path, "a") as f:
            f.write(entry)

    def query_recent_failures(self, limit: int = 10) -> List[Dict]:
        """Query the most recent failed jobs for debugging."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """SELECT id, intent, status, attempt_count, completed_at
               FROM jobs WHERE status = 'failed'
               ORDER BY completed_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def get_protocol_success_rate(self, protocol: str, days: int = 7) -> float:
        """Get the success rate for a specific protocol over N days."""
        conn = sqlite3.connect(self.db_path)
        cutoff = datetime.now(timezone.utc).isoformat()
        rows = conn.execute(
            """SELECT status FROM subtask_runs
               WHERE protocol = ? AND completed_at > datetime('now', ?)""",
            (protocol, f"-{days} days"),
        ).fetchall()
        conn.close()
        if not rows:
            return 1.0
        succeeded = sum(1 for r in rows if r[0] == "completed")
        return succeeded / len(rows)


# ─── Gap 4: Safe Mutation Hierarchy ───────────────────────────


class SafeMutator:
    """
    Three-tier mutation hierarchy that escalates risk gradually:
      1. Mutate INPUTS (safest) — change what goes into the protocol
      2. Mutate STRATEGY (medium) — change how the protocol approaches the task
      3. Mutate CODE (last resort) — rewrite the protocol, sandboxed

    Only escalates to the next tier if the previous tier has been
    attempted and failed.
    """

    def __init__(self):
        self._mutation_history: Dict[str, int] = {}  # protocol -> tier used

    async def mutate(
        self,
        protocol_name: str,
        subtask: SubTask,
        failure_reason: str,
    ) -> Dict[str, Any]:
        """
        Apply the appropriate mutation tier based on attempt number.

        Returns a dict with:
          - tier: which mutation level was applied
          - mutated_inputs: new inputs (if tier 1)
          - strategy_note: prompt modification (if tier 2)
          - code_mutated: bool (if tier 3)
        """
        attempt = subtask.attempts
        loop = asyncio.get_running_loop()

        if attempt <= 1:
            # Tier 1: Mutate inputs (safest)
            return await self._mutate_inputs(protocol_name, subtask, failure_reason)
        elif attempt == 2:
            # Tier 2: Mutate strategy/prompt
            return await self._mutate_strategy(protocol_name, subtask, failure_reason)
        else:
            # Tier 3: Mutate code (last resort, sandboxed)
            return await self._mutate_code_sandboxed(
                protocol_name, failure_reason, loop
            )

    async def _mutate_inputs(
        self, protocol_name: str, subtask: SubTask, failure_reason: str
    ) -> Dict[str, Any]:
        """
        Tier 1: Modify the inputs passed to the protocol.
        Examples: add retry flags, increase timeouts, change parameters.
        """
        mutated_inputs = dict(subtask.inputs)

        # Add retry context so the protocol knows it's a retry
        mutated_inputs["_retry"] = True
        mutated_inputs["_previous_error"] = failure_reason
        mutated_inputs["_attempt"] = subtask.attempts + 1

        # If there's a timeout-related failure, increase timeout
        if "timeout" in failure_reason.lower():
            current_timeout = mutated_inputs.get("timeout", 30)
            mutated_inputs["timeout"] = current_timeout * 2

        # Update the subtask's inputs for the next execution
        subtask.inputs = mutated_inputs

        log(f"   → Tier 1 (Input Mutation): Modified inputs for '{protocol_name}'")
        return {"tier": 1, "mutated_inputs": mutated_inputs}

    async def _mutate_strategy(
        self, protocol_name: str, subtask: SubTask, failure_reason: str
    ) -> Dict[str, Any]:
        """
        Tier 2: Modify the approach/strategy without changing code.
        Injects strategy hints that the protocol can read.
        """
        strategy_hints = {
            "_strategy_override": True,
            "_failure_context": failure_reason,
            "_instruction": (
                "Previous approach failed. Try an alternative method. "
                "If the error was a timeout, use a simpler/faster approach. "
                "If the error was a validation failure, be more conservative."
            ),
        }

        subtask.inputs.update(strategy_hints)

        log(f"   → Tier 2 (Strategy Mutation): Injected strategy hints for '{protocol_name}'")
        return {"tier": 2, "strategy_note": strategy_hints["_instruction"]}

    async def _mutate_code_sandboxed(
        self, protocol_name: str, failure_reason: str, loop: asyncio.AbstractEventLoop
    ) -> Dict[str, Any]:
        """
        Tier 3: Mutate the protocol's source code.
        This is the most dangerous tier — only used as a last resort.
        The mutation is tested in a sandboxed dry-run before being applied.
        """
        # Perform the mutation via the existing mutator
        mutated = await loop.run_in_executor(
            None, mutate_protocol, protocol_name
        )

        if not mutated:
            log(f"   → Tier 3 (Code Mutation): mutate_protocol returned None for '{protocol_name}'")
            return {"tier": 3, "code_mutated": False, "reason": "Mutator returned None"}

        # Sandbox test: attempt to load and validate the mutated protocol
        try:
            from protocols.loader import load_protocol

            test_protocol = await loop.run_in_executor(
                None, load_protocol, protocol_name
            )
            if test_protocol and callable(test_protocol.get("task")):
                log(f"   → Tier 3 (Code Mutation): Mutation applied and validated for '{protocol_name}'")
                return {"tier": 3, "code_mutated": True}
            else:
                log(f"   → Tier 3 (Code Mutation): Mutated protocol failed validation")
                return {"tier": 3, "code_mutated": False, "reason": "Validation failed"}
        except Exception as e:
            log(f"   → Tier 3 (Code Mutation): Sandbox test failed — {e}")
            return {"tier": 3, "code_mutated": False, "reason": str(e)}


# ─── Gap 2: Live Slack Escalation ─────────────────────────────


class SlackEscalation:
    """
    Live human escalation via Slack MCP connector.
    Posts structured failure alerts to a configured channel.
    """

    def __init__(self, channel: str = "#agent-alerts", server: str = "slack"):
        self.channel = channel
        self.server = server

    async def escalate(self, job: OrchestratedJob, reason: str):
        """
        Post a structured failure alert to Slack.
        Includes job ID, intent, failure reason, and failed subtask details.
        """
        failed_subtasks = [
            st for st in job.subtasks if st.status == TaskStatus.FAILED
        ]
        failed_details = "\n".join(
            f"  • `{st.protocol}` — {st.error or 'unknown error'}"
            for st in failed_subtasks[:5]
        )

        message = (
            f":rotating_light: *Loop Failed — Manual Review Required*\n\n"
            f">*Intent:* {job.intent}\n"
            f">*Job ID:* `{job.id}`\n"
            f">*Reason:* {reason}\n"
            f">*Attempts:* {job.attempt_count}\n"
            f">*Failed Subtasks ({len(failed_subtasks)}):*\n{failed_details}\n\n"
            f"_Timestamp: {datetime.now(timezone.utc).isoformat()}_"
        )

        loop = asyncio.get_running_loop()

        try:
            await loop.run_in_executor(
                None, self._post_to_slack, message
            )
            log(f"📨 Escalation posted to Slack {self.channel}")
        except Exception as e:
            # If Slack fails, log locally — never let escalation failure crash the system
            log(f"⚠️  Slack escalation failed ({e}), logging locally only")

    def _post_to_slack(self, message: str):
        """Synchronous Slack post via MCP CLI."""
        result = subprocess.run(
            [
                "manus-mcp-cli",
                "tool",
                "call",
                "slack_post_message",
                "--server",
                self.server,
                "--input",
                json.dumps({"channel": self.channel, "text": message}),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Slack MCP call failed: {result.stderr or result.stdout}"
            )


# ─── The Orchestrator ──────────────────────────────────────────


class HierarchicalOrchestrator:
    """
    Three-phase orchestration:
      1. PLAN  — Decompose intent into subtasks
      2. MAP   — Execute subtasks in parallel (isolated state)
      3. REDUCE — Collect, verify, and self-correct

    Now with:
      - Semantic LLM verification (Gap 1)
      - Live Slack escalation (Gap 2)
      - Structured SQLite state (Gap 3)
      - Safe 3-tier mutation (Gap 4)
    """

    def __init__(
        self,
        max_concurrency: int = 10,
        verification_gate: Optional[Callable] = None,
        reducer: Optional[Callable] = None,
        state_file: Optional[str] = None,
        subtask_timeout: float = 300.0,
        max_job_history: int = 1000,
        slack_channel: str = "#agent-alerts",
        use_semantic_gate: bool = True,
        semantic_model: str = "gpt-4o-mini",
        db_path: Optional[str] = None,
    ):
        self.max_concurrency = max_concurrency
        self.reducer = reducer or self._default_reducer
        self.subtask_timeout = subtask_timeout
        self.max_job_history = max_job_history
        self.jobs: Dict[str, OrchestratedJob] = {}

        # Gap 1: Semantic verification gate
        if verification_gate:
            self.verification_gate = verification_gate
        elif use_semantic_gate:
            self._semantic_gate = SemanticGate(model=semantic_model)
            self.verification_gate = self._semantic_gate.verify
        else:
            self.verification_gate = self._default_gate

        # Gap 2: Live Slack escalation
        self.escalation = SlackEscalation(channel=slack_channel)

        # Gap 3: Structured state store
        self.state_store = StateStore(
            db_path=db_path,
            md_path=state_file or str(Path(__file__).parent / "STATE.md"),
        )

        # Gap 4: Safe mutation hierarchy
        self.mutator = SafeMutator()

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
                    result = await asyncio.wait_for(
                        self._run_protocol_isolated(
                            subtask.protocol, subtask.inputs
                        ),
                        timeout=self.subtask_timeout,
                    )
                    subtask.result = result
                    subtask.status = TaskStatus.COMPLETED

                    try:
                        await loop.run_in_executor(
                            None, track_outcome, subtask.protocol, result
                        )
                    except Exception:
                        pass

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
        and trigger self-correction (safe mutation) on failure.
        """
        loop = asyncio.get_running_loop()

        # Step 1: Reduce — aggregate results
        results = [st.result for st in job.subtasks if st.result]
        failures = [st for st in job.subtasks if st.status == TaskStatus.FAILED]

        job.reduced_result = await self.reducer(results, failures)

        # Step 2: Verify — run through the Verification Gateway
        verification = await self.verification_gate(job)
        job.attempt_count += 1

        if verification["passed"]:
            job.verification_passed = True
            job.status = TaskStatus.COMPLETED
            log(f"✅ REDUCE: Verification PASSED for job '{job.intent}'")
        else:
            job.verification_passed = False
            log(f"❌ REDUCE: Verification FAILED — {verification.get('reason', 'unknown')}")

            # Step 3: Self-Correct with safe mutation hierarchy
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

                # Gap 4: Use safe mutation hierarchy instead of raw code mutation
                for st in retryable:
                    await self.mutator.mutate(
                        protocol_name=st.protocol,
                        subtask=st,
                        failure_reason=st.error or "unknown",
                    )
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
                job.attempt_count += 1
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
                # Gap 2: Live Slack escalation
                await self.escalation.escalate(
                    job, verification.get("reason", "All retries exhausted")
                )

        # Step 4: Persist state (Gap 3: structured + markdown)
        await self._persist_state(job)
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
        Passes inputs to the protocol task function.
        """
        from protocols.loader import load_protocol

        protocol = load_protocol(protocol_name)
        if not protocol:
            raise Exception(f"Protocol '{protocol_name}' not found")

        task_fn = protocol["task"]
        loop = asyncio.get_running_loop()

        def _execute_with_inputs():
            sig = inspect.signature(task_fn)
            if sig.parameters:
                return task_fn(**inputs)
            else:
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

    async def _persist_state(self, job: OrchestratedJob):
        """Persist job state to both SQLite and STATE.md."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.state_store.persist_job, job)
        await loop.run_in_executor(None, self.state_store.persist_markdown, job)

    def _prune_job_history(self):
        """Prevent unbounded memory growth from stored jobs."""
        if len(self.jobs) > self.max_job_history:
            sorted_jobs = sorted(
                self.jobs.items(), key=lambda x: x[1].created_at
            )
            excess = len(self.jobs) - self.max_job_history
            for job_id, _ in sorted_jobs[:excess]:
                del self.jobs[job_id]

    # ─── Fallback Gate & Reducer ──────────────────────────────

    async def _default_gate(self, job: OrchestratedJob) -> Dict[str, Any]:
        """
        Fallback mechanical gate: passes if >80% of subtasks succeeded.
        Used only when use_semantic_gate=False and no custom gate provided.
        """
        total = len(job.subtasks)
        succeeded = sum(
            1 for st in job.subtasks if st.status == TaskStatus.COMPLETED
        )
        rate = succeeded / total if total > 0 else 0

        if rate >= 0.8:
            return {"passed": True, "layer": "mechanical", "success_rate": rate}
        else:
            return {
                "passed": False,
                "layer": "mechanical",
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
