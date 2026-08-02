# `orchestrator_mapreduce` — Module Documentation

**Module:** `orchestrator_mapreduce.py`
**Version:** 1.0.0
**Location:** Repository root (`self-correcting-executor/orchestrator_mapreduce.py`)
**Python:** 3.10+
**Dependencies:** `utils.logger`, `utils.tracker`, `agents.mutator`, `protocols.loader`

---

## Overview

The `orchestrator_mapreduce` module introduces a formal **Plan → Map → Reduce** orchestration layer to the self-correcting executor. It enables parallel execution of protocol-based subtasks with bounded concurrency, a configurable Verification Gateway, automatic self-correction via protocol mutation on failure, and graceful human escalation when all retries are exhausted.

The module is additive — it imports from existing project modules without modifying them.

---

## Architecture

```
                    ┌──────────────────────────────┐
                    │     orchestrator.run()        │
                    │     (Single Entry Point)      │
                    └──────────────┬───────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                    ▼
     ┌────────────────┐  ┌────────────────┐  ┌────────────────┐
     │   1. PLAN      │  │   2. MAP       │  │   3. REDUCE    │
     │                │  │                │  │                │
     │ Decompose      │  │ Fan-out with   │  │ Aggregate +    │
     │ intent into    │→ │ bounded        │→ │ Verify +       │
     │ N subtasks     │  │ concurrency    │  │ Mutate/Retry   │
     └────────────────┘  └────────────────┘  └────────────────┘
                                                      │
                                              ┌───────┴───────┐
                                              ▼               ▼
                                        ┌──────────┐   ┌──────────┐
                                        │  PASS    │   │  FAIL    │
                                        │  → Done  │   │  → Retry │
                                        └──────────┘   │  → Mutate│
                                                       │  → Escalate
                                                       └──────────┘
```

---

## Public API

### Exports

```python
__all__ = ["HierarchicalOrchestrator", "OrchestratedJob", "SubTask", "TaskStatus"]
```

---

## Classes

### `TaskStatus`

An enumeration of possible states for subtasks and jobs.

| Value | Description |
| :--- | :--- |
| `PENDING` | Created but not yet started |
| `RUNNING` | Currently executing |
| `COMPLETED` | Finished successfully |
| `FAILED` | Finished with an error or timeout |
| `RETRYING` | Previously failed, now being re-executed after mutation |

---

### `SubTask`

A dataclass representing a single unit of work dispatched to a worker.

| Field | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `id` | `str` | — | Unique identifier (auto-generated from job ID) |
| `protocol` | `str` | — | Name of the protocol to execute (matches `protocols/<name>.py`) |
| `inputs` | `Dict[str, Any]` | — | Keyword arguments passed to the protocol's `task()` function |
| `status` | `TaskStatus` | `PENDING` | Current execution state |
| `result` | `Optional[Dict]` | `None` | Return value from the protocol on success |
| `error` | `Optional[str]` | `None` | Error message on failure |
| `attempts` | `int` | `0` | Number of execution attempts made |
| `max_attempts` | `int` | `3` | Maximum allowed attempts before giving up |
| `started_at` | `Optional[float]` | `None` | Unix timestamp when execution began |
| `completed_at` | `Optional[float]` | `None` | Unix timestamp when execution ended |

---

### `OrchestratedJob`

A dataclass representing a top-level job composed of parallel subtasks.

| Field | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `id` | `str` | — | Unique identifier (`job_{timestamp}_{uuid8}`) |
| `intent` | `str` | — | Human-readable description of the job's purpose |
| `subtasks` | `List[SubTask]` | `[]` | All subtasks belonging to this job |
| `status` | `TaskStatus` | `PENDING` | Overall job status |
| `reduced_result` | `Optional[Dict]` | `None` | Aggregated output from the reducer |
| `verification_passed` | `bool` | `False` | Whether the Verification Gateway passed |
| `created_at` | `float` | `time.time()` | Unix timestamp of job creation |

---

### `HierarchicalOrchestrator`

The main orchestration engine. Manages the full Plan → Map → Reduce lifecycle.

#### Constructor

```python
HierarchicalOrchestrator(
    max_concurrency: int = 10,
    verification_gate: Optional[Callable] = None,
    reducer: Optional[Callable] = None,
    state_file: Optional[str] = None,
    subtask_timeout: float = 300.0,
    max_job_history: int = 1000,
)
```

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `max_concurrency` | `int` | `10` | Maximum number of subtasks executing simultaneously (semaphore size) |
| `verification_gate` | `Callable` | `_default_gate` | Async function that evaluates whether a job passes verification |
| `reducer` | `Callable` | `_default_reducer` | Async function that aggregates subtask results into a single output |
| `state_file` | `str` | `<module_dir>/STATE.md` | Path to the Markdown file where job state is persisted |
| `subtask_timeout` | `float` | `300.0` | Maximum seconds a single subtask may run before being killed |
| `max_job_history` | `int` | `1000` | Maximum number of completed jobs retained in memory |

---

## Methods

### `run(intent, task_list) → OrchestratedJob`

The single entry point. Executes the full Plan → Map → Reduce cycle.

```python
job = await orchestrator.run(
    intent="Validate all API endpoints after deployment",
    task_list=[
        {"protocol": "api_health_checker", "inputs": {"endpoint": "/users"}},
        {"protocol": "api_health_checker", "inputs": {"endpoint": "/orders"}},
    ],
)
```

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `intent` | `str` | Human-readable description of what this job accomplishes |
| `task_list` | `List[Dict]` | List of task definitions, each with `"protocol"` (required) and `"inputs"` (optional) keys |

**Returns:** `OrchestratedJob` with final status, reduced results, and verification outcome.

---

### `plan(intent, task_list) → OrchestratedJob`

Phase 1. Decomposes the intent into subtasks without executing them. Useful when you need to inspect or modify the job before execution.

```python
job = await orchestrator.plan("Check all endpoints", task_list)
# Inspect or modify job.subtasks here
job = await orchestrator.map_execute(job)
job = await orchestrator.reduce_verify(job)
```

---

### `map_execute(job) → OrchestratedJob`

Phase 2. Executes all subtasks in parallel with bounded concurrency. Each subtask runs in isolation — no shared mutable state between workers.

**Behavior:**
- Acquires a semaphore slot before executing each subtask
- Applies `subtask_timeout` via `asyncio.wait_for()`
- Tracks outcomes via `track_outcome()` (non-blocking, via `run_in_executor`)
- Sets `subtask.status` to `COMPLETED` or `FAILED`
- Never raises — failures are captured on individual subtasks

---

### `reduce_verify(job) → OrchestratedJob`

Phase 3. Aggregates results, runs the Verification Gateway, and triggers self-correction on failure.

**Behavior:**
1. Calls the `reducer` to aggregate all results
2. Calls the `verification_gate` to evaluate pass/fail
3. If failed: enters retry loop
   - Identifies subtasks with `attempts < max_attempts`
   - Deduplicates protocols and calls `mutate_protocol()` on each (non-blocking)
   - Re-executes failed subtasks in parallel
   - Re-verifies after each retry round
   - Loops until verification passes or all retries exhausted
4. If all retries exhausted: calls `_escalate_to_human()`
5. Persists final state to `STATE.md`

---

## Customization Points

### Custom Verification Gate

A verification gate is an async function that receives an `OrchestratedJob` and returns a dict:

```python
async def my_gate(job: OrchestratedJob) -> Dict[str, Any]:
    # Your logic here
    if all_good:
        return {"passed": True}
    else:
        return {"passed": False, "reason": "Explanation of failure"}
```

**Contract:**
- Must return a dict with `"passed": bool`
- If `passed` is `False`, must include `"reason": str`

**Example — All-or-nothing gate:**

```python
async def strict_gate(job):
    failed = [st for st in job.subtasks if st.status == TaskStatus.FAILED]
    if not failed:
        return {"passed": True}
    return {
        "passed": False,
        "reason": f"{len(failed)} subtasks still failing: {[st.id for st in failed]}",
    }

orchestrator = HierarchicalOrchestrator(verification_gate=strict_gate)
```

---

### Custom Reducer

A reducer is an async function that receives the list of successful results and the list of failed subtasks, and returns an aggregated dict:

```python
async def my_reducer(
    results: List[Dict[str, Any]],
    failures: List[SubTask],
) -> Dict[str, Any]:
    # Aggregate results into a meaningful summary
    return {"summary": "..."}
```

**Example — Group by category:**

```python
async def category_reducer(results, failures):
    by_category = {}
    for r in results:
        cat = r.get("category", "uncategorized")
        by_category.setdefault(cat, []).append(r)
    return {
        "categories": {k: len(v) for k, v in by_category.items()},
        "total": len(results),
        "failures": len(failures),
    }

orchestrator = HierarchicalOrchestrator(reducer=category_reducer)
```

---

## Protocol Compatibility

The module uses `inspect.signature()` to determine how to call each protocol's `task()` function:

| Protocol Signature | Behavior |
| :--- | :--- |
| `def task():` | Called with no arguments (legacy compatibility) |
| `def task(**kwargs):` | Receives the full `inputs` dict as keyword arguments |
| `def task(endpoint, timeout=30):` | Receives matching keys from `inputs` as named arguments |

**Example — New-style protocol (`protocols/api_health_checker.py`):**

```python
import requests

def task(endpoint: str, timeout: int = 10) -> dict:
    """Check if an API endpoint is healthy."""
    try:
        resp = requests.get(f"https://api.example.com{endpoint}", timeout=timeout)
        return {"success": resp.status_code == 200, "status_code": resp.status_code}
    except Exception as e:
        return {"success": False, "error": str(e)}
```

**Example — Legacy protocol (still works unchanged):**

```python
from random import randint

def task():
    """Legacy protocol with no parameters."""
    return {"success": bool(randint(0, 1)), "action": "default_execution"}
```

---

## State Persistence

After every job completes (pass or fail), the orchestrator appends a summary to `STATE.md`:

```markdown
## Job: job_1719082800_a1b2c3d4
- **Intent:** Validate all API endpoints after deployment
- **Status:** completed
- **Verification:** PASSED
- **Subtasks:** 47 total, 45 completed
- **Timestamp:** 2026-06-22T19:00:00.000000+00:00
```

This file serves as a persistent audit log across runs. It is append-only and can be read by other tools, skills, or agents for cross-run continuity.

---

## Error Handling and Escalation

The module implements a three-tier failure strategy:

| Tier | Condition | Action |
| :--- | :--- | :--- |
| **Retry** | Subtask failed, `attempts < max_attempts` | Mutate protocol, re-execute |
| **Escalate** | All retries exhausted, verification still failing | Log escalation, notify human (via MCP/Slack when configured) |
| **Timeout** | Subtask exceeds `subtask_timeout` seconds | Kill subtask, mark as FAILED, enter retry tier |

To enable Slack escalation in production, uncomment the MCP connector call in `_escalate_to_human()` and configure your Slack channel.

---

## Memory Management

The orchestrator prevents unbounded memory growth through the `max_job_history` parameter. When the number of stored jobs exceeds this limit, the oldest jobs are pruned automatically during the `plan()` phase.

---

## Usage Examples

### Minimal Example

```python
import asyncio
from orchestrator_mapreduce import HierarchicalOrchestrator

async def main():
    orchestrator = HierarchicalOrchestrator(max_concurrency=5)
    
    job = await orchestrator.run(
        intent="Run smoke tests",
        task_list=[
            {"protocol": "smoke_test", "inputs": {"service": "auth"}},
            {"protocol": "smoke_test", "inputs": {"service": "payments"}},
            {"protocol": "smoke_test", "inputs": {"service": "notifications"}},
        ],
    )
    
    print(f"Status: {job.status.value}")
    print(f"Passed: {job.verification_passed}")
    print(f"Results: {job.reduced_result}")

asyncio.run(main())
```

### Fan-Out Same Protocol Across Many Inputs

```python
async def validate_endpoints():
    orchestrator = HierarchicalOrchestrator(
        max_concurrency=20,
        subtask_timeout=30.0,
    )

    endpoints = ["/users", "/orders", "/payments", "/auth", "/products"]
    task_list = [
        {"protocol": "api_health_checker", "inputs": {"endpoint": ep}}
        for ep in endpoints
    ]

    return await orchestrator.run(
        intent="Post-deployment endpoint validation",
        task_list=task_list,
    )
```

### Fan-Out Multiple Protocols (Heterogeneous)

```python
async def research_pipeline():
    orchestrator = HierarchicalOrchestrator(max_concurrency=4)

    keywords = ["ai-agents", "mcp-protocol", "loop-engineering"]

    task_list = [
        {"protocol": "keyword_research", "inputs": {"keyword": kw}}
        for kw in keywords
    ] + [
        {"protocol": "competitor_scan", "inputs": {"keyword": kw}}
        for kw in keywords
    ]

    return await orchestrator.run(
        intent="Research and scan competitors for target keywords",
        task_list=task_list,
    )
```

### Custom Gate + Custom Reducer

```python
async def ci_triage():
    async def triage_gate(job):
        results = [st.result for st in job.subtasks if st.result]
        unknowns = [r for r in results if r.get("classification") == "unknown"]
        if not unknowns:
            return {"passed": True}
        return {
            "passed": False,
            "reason": f"{len(unknowns)} failures could not be classified",
        }

    async def triage_reducer(results, failures):
        categories = {}
        for r in results:
            cat = r.get("classification", "unknown")
            categories.setdefault(cat, []).append(r)
        return {
            "total_classified": len(results),
            "categories": {k: len(v) for k, v in categories.items()},
        }

    orchestrator = HierarchicalOrchestrator(
        max_concurrency=5,
        verification_gate=triage_gate,
        reducer=triage_reducer,
    )

    return await orchestrator.run(
        intent="Classify overnight CI failures",
        task_list=[
            {"protocol": "log_analyzer", "inputs": {"run_id": rid}}
            for rid in failed_run_ids
        ],
    )
```

### Step-by-Step Execution (Plan → Inspect → Execute)

```python
async def controlled_execution():
    orchestrator = HierarchicalOrchestrator()

    # Phase 1: Plan only
    job = await orchestrator.plan(
        intent="Validate all endpoints",
        task_list=task_list,
    )
    
    # Inspect the plan
    print(f"Will execute {len(job.subtasks)} subtasks")
    for st in job.subtasks:
        print(f"  - {st.protocol}({st.inputs})")

    # Phase 2: Execute
    job = await orchestrator.map_execute(job)

    # Inspect intermediate results
    completed = [st for st in job.subtasks if st.status == TaskStatus.COMPLETED]
    print(f"{len(completed)} completed before verification")

    # Phase 3: Verify and self-correct
    job = await orchestrator.reduce_verify(job)
    return job
```

---

## Integration with Existing Modules

| Module | How It's Used |
| :--- | :--- |
| `protocols.loader.load_protocol(name)` | Dynamically loads protocol modules by name |
| `agents.mutator.mutate_protocol(name)` | Rewrites protocol source when failure rate is high |
| `utils.tracker.track_outcome(name, result)` | Appends execution outcomes to `memory/<name>.json` |
| `utils.logger.log(message)` | Structured logging throughout the orchestration lifecycle |

No modifications to these modules are required. The orchestrator wraps all synchronous calls in `asyncio.get_running_loop().run_in_executor()` to maintain non-blocking behavior.

---

## Configuration Reference

| Parameter | Env Variable | Default | Recommended Range |
| :--- | :--- | :--- | :--- |
| `max_concurrency` | — | `10` | 1–100 (depends on protocol I/O characteristics) |
| `subtask_timeout` | — | `300.0` | 5–600 seconds |
| `max_job_history` | — | `1000` | 100–10000 |
| `state_file` | — | `<module_dir>/STATE.md` | Any writable path |

**Tuning guidance:**
- For CPU-bound protocols: set `max_concurrency` to CPU core count
- For I/O-bound protocols (API calls, file reads): set `max_concurrency` to 20–50
- For long-running protocols: increase `subtask_timeout` accordingly
- For high-throughput systems: increase `max_job_history` or implement external persistence

---

## Lifecycle Diagram

```
User Intent
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ PLAN                                                     │
│  • Generate job ID                                       │
│  • Create SubTask for each item in task_list             │
│  • Prune old jobs if history exceeds max_job_history     │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│ MAP                                                      │
│  • Acquire semaphore (bounded concurrency)               │
│  • For each subtask (parallel):                          │
│    ├─ Load protocol via protocols.loader                 │
│    ├─ Inspect signature → pass inputs or call bare       │
│    ├─ Execute in thread pool (run_in_executor)           │
│    ├─ Apply timeout (asyncio.wait_for)                   │
│    ├─ Track outcome (non-blocking)                       │
│    └─ Set status: COMPLETED or FAILED                    │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│ REDUCE                                                   │
│  • Call reducer(results, failures) → aggregated output   │
│  • Call verification_gate(job) → pass/fail               │
│  •                                                       │
│  • If PASSED → mark job COMPLETED                        │
│  •                                                       │
│  • If FAILED → enter retry loop:                         │
│    ├─ Find retryable subtasks (attempts < max_attempts)  │
│    ├─ Deduplicate protocols                              │
│    ├─ mutate_protocol() on each (non-blocking)           │
│    ├─ Re-execute failed subtasks in parallel             │
│    ├─ Re-reduce and re-verify                            │
│    └─ Loop until PASSED or no retryable subtasks remain  │
│  •                                                       │
│  • If exhausted → escalate_to_human()                    │
│  • Persist state to STATE.md                             │
└─────────────────────────────────────────────────────────┘
```

---

## Changelog

| Version | Date | Changes |
| :--- | :--- | :--- |
| 1.0.0 | 2026-06-22 | Initial release with all 8 review fixes applied |
