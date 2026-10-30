# `orchestrator_mapreduce` — Module Documentation

**Module:** `orchestrator_mapreduce.py`
**Version:** 1.0.0
**Location:** Repository root (`self-correcting-executor/orchestrator_mapreduce.py`)
**Python:** 3.10+
**Dependencies:** `utils.logger`, `utils.tracker`, `agents.mutator`, `protocols.loader`

---

## Overview

The `orchestrator_mapreduce` module introduces a formal **Plan → Map → Reduce** orchestration layer to the self-correcting executor. It enables parallel execution of protocol-based subtasks with bounded concurrency, a configurable Verification Gateway, automatic self-correction via protocol mutation on failure, and graceful human escalation when all retries are exhausted.

This guide describes the current `HierarchicalOrchestrator` in this module, not the separate `OrchestrationEngine` or BuildPlan runtime. The class invokes synchronous protocol functions in a shared thread pool. It does not provide process isolation, hard cancellation, or an authorization boundary. Read the return-value, retry, and timeout limits below before using it for side effects.

---

## Architecture

```text
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
__all__ = [
    "HierarchicalOrchestrator", "OrchestratedJob", "SubTask", "TaskStatus",
    "SemanticGate", "StateStore",
]
```

---

## Classes

### `TaskStatus`

An enumeration of possible states for subtasks and jobs.

| Value | Description |
| :--- | :--- |
| `PENDING` | Created but not yet started |
| `RUNNING` | Currently executing |
| `COMPLETED` | Protocol returned without raising, or job passed its configured gate; not proof of business success |
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
| `completed_at` | `Optional[float]` | `None` | Time the await finished or timed out; a timed-out thread may still be running |

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
| `attempt_count` | `int` | `0` | Number of verification rounds |

---

### `HierarchicalOrchestrator`

The main orchestration engine. Manages the full Plan → Map → Reduce lifecycle.

#### Constructor

```python
HierarchicalOrchestrator(
    max_concurrency=10,
    verification_gate=None,
    reducer=None,
    state_file=None,
    subtask_timeout=300.0,
    max_job_history=1000,
    slack_channel="#agent-alerts",
    use_semantic_gate=True,
    semantic_model="gpt-4o-mini",
    db_path=None,
)
```

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `max_concurrency` | `int` | `10` | Initial map-phase semaphore size; does not cap retry fan-out or already-timed-out threads |
| `verification_gate` | `Callable` | `SemanticGate.verify` | Custom async gate takes precedence; otherwise semantic verification is enabled by default |
| `reducer` | `Callable` | `_default_reducer` | Async function that aggregates subtask results into a single output |
| `state_file` | `str` | `<module_dir>/STATE.md` | Path to the Markdown file where job state is persisted |
| `subtask_timeout` | `float` | `300.0` | Maximum seconds to await a subtask before marking it failed; a running protocol thread may continue |
| `max_job_history` | `int` | `1000` | Maximum stored jobs of any status; oldest entries are pruned during planning |
| `slack_channel` | `str` | `#agent-alerts` | Channel passed to the Slack MCP CLI on exhausted verification failure |
| `use_semantic_gate` | `bool` | `True` | Use `SemanticGate` unless a custom gate is supplied; `False` selects mechanical verification |
| `semantic_model` | `str` | `gpt-4o-mini` | Model for the optional OpenAI client |
| `db_path` | `str` | `<module_dir>/loop_state.db` | SQLite persistence path |

---

## Methods

### `run(intent, task_list) → OrchestratedJob`

The single entry point. Executes the full Plan → Map → Reduce cycle.

```python
job = await orchestrator.run(
    intent="Process multiple data directories",
    task_list=[
        {"protocol": "data_processor", "inputs": {"data_path": "/data/users"}},
        {"protocol": "data_processor", "inputs": {"data_path": "/data/orders"}},
    ],
)
```

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `intent` | `str` | Human-readable description of what this job accomplishes |
| `task_list` | `List[Dict]` | List of task definitions, each with `"protocol"` (required) and `"inputs"` (optional) keys |

**Returns:** `OrchestratedJob` with final status, reduced results, and verification outcome.

**Example paths:** The `/data/users` and `/data/orders` directories above must already exist and contain the intended files. `data_processor` rejects an invalid explicit path on current master. The self-contained local example below creates real temporary fixtures.

**Inputs:** Zero-argument `task()` functions ignore `inputs`. Parameterized tasks receive the full dict via `task(**inputs)` without key filtering, so callers must supply compatible keys or use a protocol accepting `**kwargs`.

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

Phase 2. Executes all subtasks using a semaphore and the event loop's shared thread pool. Inputs and protocol module state are not cloned or isolated. The loader runs on the event-loop thread before the task is submitted; blocking import work can delay the loop and timeout delivery.

**Behavior:**
- Acquires a semaphore slot before executing each subtask
- Applies `subtask_timeout` via `asyncio.wait_for()`
- Tracks successful returns and non-timeout exceptions via `track_outcome()` (non-blocking, via `run_in_executor`); timeout failures are not sent to this tracker
- Sets `subtask.status` to `COMPLETED` for any normal return, including `{"success": False}`; exceptions and timeouts set `FAILED`
- Captures ordinary per-subtask exceptions; caller cancellation and setup/configuration errors are not a guarantee of successful completion
- Does not await coroutine results returned by async protocols. Use synchronous `task()` functions here; the separate `OrchestrationEngine` has a different dispatch contract

---

### `reduce_verify(job) → OrchestratedJob`

Phase 3. Aggregates results, runs the Verification Gateway, and triggers self-correction on failure.

**Behavior:**
1. Calls the `reducer` to aggregate all results
2. Calls the `verification_gate` to evaluate pass/fail
3. If failed: enters retry loop
   - Identifies `FAILED` subtasks with `attempts < max_attempts`
   - Calls `SafeMutator.mutate()` for each retryable subtask, without deduplicating protocol names
   - Re-executes those subtasks in parallel, without the initial map semaphore or `track_outcome()` calls
   - Re-verifies after each retry round
   - Loops until verification passes or all retries exhausted
4. If verification still fails with no retryable subtasks: calls `SlackEscalation.escalate()`
5. Persists the final snapshot to SQLite and appends a summary to `STATE.md`

Only subtasks whose status is `FAILED` are retried. A normal return with `success: false` is still `COMPLETED`, so a rejecting custom gate does not make it retryable. Likewise, a semantic rejection with no failed subtasks proceeds to escalation. Reducers receive truthy stored results and the failed-subtask list; the default reducer counts results, not their `success` fields. Errors from a custom reducer/gate or persistence can propagate and prevent the normal final-state write.

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
    failed = [
        st for st in job.subtasks
        if st.status != TaskStatus.COMPLETED
        or not isinstance(st.result, dict)
        or st.result.get("success") is not True
    ]
    if job.subtasks and not failed:
        return {"passed": True}
    return {
        "passed": False,
        "reason": f"{len(failed)} subtasks still failing: {[st.id for st in failed]}",
    }

orchestrator = HierarchicalOrchestrator(verification_gate=strict_gate)
```

---

### Custom Reducer

A reducer is an async function that receives truthy stored results and the list of subtasks marked `FAILED`, and returns an aggregated dict. A stored result is not necessarily a successful domain outcome; inspect its fields as needed:

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
| `def task():` | Called with no arguments; `inputs` is ignored (legacy compatibility) |
| `def task(**kwargs):` | Receives the full `inputs` dict as keyword arguments |
| `def task(endpoint, timeout=30):` | Receives the full `inputs` dict as keyword arguments; extra keys or missing required keys raise `TypeError` and fail the subtask |

**Current built-in signatures:** `data_processor.task(data_path=None, source=None)` accepts an existing directory or the field-capture source contract. `api_health_checker.task(service=None, url=None)` now requires an artifact bundle and an explicit full health URL; it no longer uses the legacy fixed endpoint list. `log_analyzer.task(source=None)` is synchronous only when `source` is omitted. Its source-based branch returns a coroutine and belongs in `OrchestrationEngine`, not this class.

**Illustrative new protocol, `protocols/example_endpoint_check.py` (not a shipped module):**

Create and review this module before naming `example_endpoint_check` in a task list. Replace `https://api.example.com` with an authorized service. Do not pass `endpoint` to the existing `api_health_checker` expecting this different contract:

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
def task():
    """Illustrative zero-argument fixture, not a real service check."""
    return {"success": True, "action": "fixture_execution"}
```

---

## Verification and Persistence Helpers

`SemanticGate(model="gpt-4o-mini", threshold=0.8)` first checks the fraction of subtasks with `COMPLETED` status. If it meets the threshold, it asks an OpenAI-compatible chat model to evaluate up to 20 result summaries (each result is truncated). It accepts a response beginning with `PASS`. If the client/import/provider call fails, it returns a **mechanical fallback pass**; semantic verification is not guaranteed. The default mechanical gate also passes at **at least 80%**, even if some subtasks failed or returned `success: false`. Supply a domain-specific gate when those outcomes must fail.

The LLM client is constructed as `OpenAI()` and therefore uses that installed SDK's configuration. The optional `openai` dependency and credentials are required for live semantic evaluation. Set `use_semantic_gate=False` or supply a custom gate to avoid this provider call. Local examples below use a custom gate.

`StateStore(db_path=None, md_path=None)` creates SQLite `jobs` and `subtask_runs` tables. `persist_job(job)` replaces each job/subtask row by ID; it is a **final snapshot**, not per-attempt history. `persist_markdown(job)` appends the human-readable summary. `query_recent_failures(limit=10)` and `get_protocol_success_rate(protocol, days=7)` query stored final rows; the latter returns `1.0` when no rows exist, which is not measured success. These methods perform synchronous SQLite/file I/O when called directly.

## State Persistence

On the normal return path after verification (pass or fail), the orchestrator appends a summary to `STATE.md`:

```markdown
## Job: job_1719082800_a1b2c3d4
- **Intent:** Validate all API endpoints after deployment
- **Status:** completed
- **Verification:** PASSED
- **Subtasks:** 47 total, 45 completed
- **Timestamp:** 2026-06-22T19:00:00.000000+00:00
```

This file records final summaries across runs, not every failed attempt or worker side effect. It is append-only and can be read by other tools, skills, or agents for cross-run continuity.

---

## Error Handling and Escalation

The module implements a three-tier failure strategy:

| Tier | Condition | Action |
| :--- | :--- | :--- |
| **Retry** | Subtask status is `FAILED`, `attempts < max_attempts` | Apply the attempt-based mutation tier and re-execute |
| **Escalate** | All retries exhausted, verification still failing | Log escalation, notify human (via MCP/Slack when configured) |
| **Timeout** | Awaiting the subtask exceeds `subtask_timeout` seconds | Mark as FAILED and enter retry tier; the underlying protocol thread may continue |

`asyncio.wait_for()` bounds the wait for the executor result; it does not terminate an already-running thread. A timed-out protocol may still produce side effects, and a retry may overlap that work. Set timeouts inside blocking operations and make retryable protocols idempotent.

`SafeMutator` adds `_retry`, `_previous_error`, and `_attempt` on the first retry; on a timeout-like error it also adds/doubles `timeout`. The second retry adds strategy-hint keys. Protocols without compatible parameters or `**kwargs` will reject these added keys; hints are not proof of a correction. With the default three total attempts, code mutation is never reached. Raising a subtask's `max_attempts` beyond three permits the code tier, which invokes the existing mutator and checks that the loaded module exposes a callable task. Despite its method name, this is **not a process sandbox, transactional rollback, or successful dry-run**; the legacy mutator can rewrite source files and Python imports can remain cached.

Escalation is already active: `SlackEscalation` runs `manus-mcp-cli tool call slack_post_message --server slack --input ...` with the configured channel and a 30-second subprocess timeout. It catches CLI errors and logs locally; logging is not delivery proof. There is no `_escalate_to_human()` method to uncomment. A failed full `run()` can therefore attempt an external Slack call. The local demonstration uses successful controlled inputs; tests replace escalation with a failing assertion to detect any unexpected external attempt.

`asyncio.run()` normally waits for its default executor to finish during shutdown. A protocol that times out from the orchestrator's perspective may therefore still delay process exit. The timeout also excludes tracking, reduction, verification and persistence; this is not a deadline for the whole job.

---

## Memory Management

The orchestrator prevents unbounded memory growth through the `max_job_history` parameter. When the number of stored jobs exceeds this limit, the oldest jobs are pruned automatically during the `plan()` phase.

---

## Usage Examples

All five patterns below are implemented in `examples/mapreduce_local.py` using the actual `data_processor` and synchronous `log_analyzer` protocols. Run the complete demonstration from the repository root:

```sh
python -m examples.mapreduce_local
```

The runner creates JSON files with one and two records and one log line in a temporary directory. It uses a strict gate that requires every subtask to return `success: true`, writes real SQLite/Markdown/tracker state, then removes the fixtures. It replaces Slack escalation with a visible local exception and never invokes the LLM gate. The example is local I/O verification, not a production health check.

For the callable recipes below, `data_path`/`data_paths` must be existing directories containing the intended files and `output_dir` must be writable. `LOG_DIR` must name an existing directory for the two log-analysis recipes. The standalone runner supplies all these inputs. Helpers are ordinary application examples, not additional orchestrator API methods.

### Minimal Example

```python
from examples.mapreduce_local import minimal

job = await minimal(data_path, output_dir)
print(job.status.value, job.verification_passed, job.reduced_result)
```

`minimal` supplies one `data_processor` task with `inputs={"data_path": ...}`. It does not use an absent `smoke_test` protocol: the loader's fallback would create a random placeholder, not run a real smoke test.

### Fan-Out Same Protocol Across Many Inputs

```python
from examples.mapreduce_local import fan_out

job = await fan_out(data_paths, output_dir)
```

`fan_out` creates one `data_processor` task per existing directory and uses `max_concurrency=2`. The supplied directory paths are actually consumed; they are not ignored zero-argument inputs.

### Fan-Out Multiple Protocols (Heterogeneous)

```python
from examples.mapreduce_local import heterogeneous

job = await heterogeneous(data_path, output_dir)
```

This runs `data_processor(data_path=...)` alongside `log_analyzer()` configured by `LOG_DIR`. To adapt the pattern to keyword research and competitor scans, first implement and review those protocol modules with the intended `keyword` contract. `keyword_research` and `competitor_scan` are not shipped protocols; naming them alone does not provide research functionality.

### Custom Gate + Custom Reducer

```python
from examples.mapreduce_local import custom_gate_and_reducer

job = await custom_gate_and_reducer(data_path, output_dir)
print(job.reduced_result["categories"])
```

The example's strict gate checks both execution status and `result["success"]`. Its reducer groups actual results by `action`, producing one `data_processing` and one `log_analysis` result for the fixture. A CI-classification application can supply its own classifier and classification-aware gate/reducer; the existing `log_analyzer` does not accept `run_id` or produce a `classification` field.

### Step-by-Step Execution (Plan → Inspect → Execute)

```python
from examples.mapreduce_local import step_by_step

job = await step_by_step(data_path, output_dir)
```

`step_by_step` calls `plan()`, checks the pending plan, calls `map_execute()`, inspects returned statuses, then calls `reduce_verify()`. Inspecting `COMPLETED` alone is insufficient for business verification, so it uses the same result-aware custom gate.

---

## Integration with Existing Modules

| Module | How It's Used |
| :--- | :--- |
| `protocols.loader.load_protocol(name)` | Dynamically loads protocol modules by name |
| `agents.mutator.mutate_protocol(name)` | Rewrites protocol source when failure rate is high |
| `utils.tracker.track_outcome(name, result)` | Appends execution outcomes to `memory/<name>.json` |
| `utils.logger.log(message)` | Structured logging throughout the orchestration lifecycle |

Task execution, initial outcome tracking and final persistence use `run_in_executor()`. Protocol loading/imports and constructor database initialization remain synchronous; the module does not make every synchronous dependency non-blocking. The loader defaults to creating a random placeholder for missing protocol modules, so use only reviewed, existing protocol names. That placeholder is not the requested protocol behavior.

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

```text
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
│    ├─ Track returns / non-timeout exceptions             │
│    ├─ Timeout: fail await; no tracker; thread may run    │
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
│    ├─ SafeMutator per failed subtask (no deduplication)  │
│    ├─ Retry in parallel (no semaphore / outcome tracker) │
│    ├─ Re-reduce and re-verify                            │
│    └─ Loop until PASSED or no retryable subtasks remain  │
│  •                                                       │
│  • If no retries remain → SlackEscalation.escalate()     │
│  • Persist final SQLite snapshot + STATE.md summary      │
└─────────────────────────────────────────────────────────┘
```

---

## Changelog

| Version | Date | Changes |
| :--- | :--- | :--- |
| 1.0.0 | 2026-06-22 | Initial module guide |
| Documentation correction | 2026-10-29 | Reconciled invocation, examples, timeout/tracking, current signatures, verification and persistence guarantees against current master |
