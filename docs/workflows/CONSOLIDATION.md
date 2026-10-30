# Workflow consolidation and execution

This page describes the original four catalog paths from self-correcting-executor, WRKFLW and workflows. The complete imported BuildPlan and construction lifecycle pattern mapping, host contracts, correction engine and runnable examples are in [Imported patterns](IMPORTED_PATTERNS.md). It loads `workflows/data/*.yaml|json` at module import. Restart after changing definitions. Autoloading a definition does not establish that every component or external service in it is operational.

## Inputs and outputs

Pass run-specific data through `options={"inputs": {...}}`. A template's `required_inputs` declares external reference names. Existing string values matching a supplied input or declared output resolve to that data. Nested dictionaries/lists are supported. Use `{"$ref": "inputs.field_capture"}` or `{"$ref": "outputs.normalized_data"}` for explicit references, and `{"$literal": "normalized_data"}` when the same text is intended literally. Missing references fail before dispatch.

Each step receives a fresh copy, including on retries. Protocols receive their exact keyword inputs; unsupported keywords fail rather than being discarded. Both synchronous tasks and returned coroutines are awaited. Unknown protocol imports fail without generating a default protocol file. Analyzer steps call the real PatternDetector. Named outputs select matching result fields; a single unmatched name aliases the complete result. Multiple missing output fields fail instead of assigning the same whole result to every name.

`resilience.enabled: false` disables automatic retries. Invalid, zero or negative retry counts execute once. Failed attempts remain in the knowledge graph; later execution metadata does not overwrite failure history or relationships. An upstream failure stops subsequent steps.

## Construction daily log

The selected meaning comes from [the WRKFLW PRD](../../ported/wrkflw/PRD.md): project daily logs, tasks, RFIs, attachment metadata, cited summary sections and a PM-review draft. Supply a project-scoped snapshot retrieved by your caller. Source IDs are the actual collection and record IDs, not fabricated embedding or retrieval IDs.

```python
import asyncio
import json
from orchestrator import OrchestrationEngine

# A real caller-supplied snapshot, not a built-in demonstration dataset.
with open("field-capture.json", encoding="utf-8") as handle:
    field_capture = json.load(handle)

result = asyncio.run(OrchestrationEngine().execute_intent(
    "Generate a construction daily log workflow",
    options={
        "workflow_template": "construction_daily_log",
        "inputs": {"field_capture": field_capture},
    },
))
if result["status"] != "completed":
    raise RuntimeError(result["steps_completed"][-1]["error"])
draft = result["outputs"]["draft_summary"]
```

The snapshot schema is a mapping with `project_id`, ISO `report_date`, and `daily_logs`, `tasks`, `rfis` arrays. Each supplied record needs a unique `id` within its collection. Record `project_id`/`projectId`, when present, must match the selected project. At least one record is required. Preserve log notes, status, dates and attachment metadata in those records; the normalizer copies them without inventing missing information.

The data processor validates the snapshot. PatternDetector computes observed counts and supplied statuses; it does not invent risk scores or business thresholds. The summary protocol then uses the existing OpenAI connector with a construction-specific prompt and JSON schema. Every generated statement must cite a supplied source ID. Invalid provenance, unknown citations, malformed sections, incomplete output, missing keys and provider errors fail explicitly. A citation passing structural checks does not prove the text accurately follows its source; PM review remains required.

Set `OPENAI_API_KEY` before starting the process. `WORKFLOW_SUMMARY_MODEL` optionally selects a compatible model; default `gpt-6-luna` supports the requested API/schema according to [OpenAI's model documentation](https://developers.openai.com/api/docs/models/gpt-6-luna). Model access, quota and live response quality remain account-specific. Requests time out after 60 seconds and do not redirect. Failed provider calls may be retried by the workflow policy; a timeout does not prove the provider did not process or bill a request.

The returned `draft_summary` retains project/date, cited content, the supplied source records, requested/returned model metadata and prompt version. It is marked `draft` and `requires_pm_review`. This path implements generation from the supplied project snapshot. It does **not** fetch a project's database, embed/search a vector store, inspect images, persist a project note, notify a PM or publish a PDF. The additional `ConstructionLifecycle` adapter represents those retrieval, persistence, notification, review and PDF steps with configured services; see [Imported patterns](IMPORTED_PATTERNS.md). Their live provider acceptance remains unverified; no credential alone establishes it.

The built-in `construction_daily_log` keeps its fourth code-generation step and returns an executable FastAPI scaffold plus integration instructions. That artifact does not implement RFI approval, audit storage or deployment. The external `construction_daily_log_external` retains its three daily-summary steps.

## TypeScript service pipeline

The artifact shape follows the archived [workflows code generator](https://github.com/groupthinking/workflows/blob/main/lib/services/code-generator.ts) and [ported scaffold route](../../ported/workflows-api/deploy-scaffold.ts): files with path, content and language, a package manifest and strict TypeScript configuration. The selected catalog pattern generates a health + CRUD service scaffold; it does not reproduce the archived application's full build-plan product.

Install the pinned compiler once with `npm ci --ignore-scripts --prefix tools/typescript-validation` and use Node 22+. The validator compiles the actual TypeScript in a temporary directory with fixed strict flags, parses JSON artifacts and rejects unsafe or duplicate paths. It never runs generated code, npm scripts or package installation. Missing tooling, malformed files and type errors fail explicitly; `TYPESCRIPT_COMPILER` may point to an already installed trusted compiler file.

```python
import os

result = asyncio.run(OrchestrationEngine().execute_intent(
    "Generate TypeScript API",
    options={
        "workflow_template": "typescript_service_pipeline",
        "inputs": {
            # Full health URL for a service you have already made available.
            "service_health_url": os.environ["SERVICE_HEALTH_URL"],
            "cache_config": {
                "key": os.environ["SERVICE_METADATA_CACHE_KEY"],
                "ttl_seconds": int(os.environ["SERVICE_METADATA_CACHE_TTL"]),
            },
        },
    },
))
```

The generated files expose `createService(store)`, a standard Request/Response handler with `GET /health`, list/create at `/api/v1/service` and get/update/delete at `/api/v1/service/:id`. The input schema is a string `id` and object `payload`. Supply the typed `Store` methods and an HTTP adapter. CRUD calls those methods and reports duplicate/missing IDs, invalid requests and storage errors; it does not fake persistence. The source does not choose authentication, authorization, a database, or project-specific business rules. The generated README describes how to compile and integrate the scaffold.

`service_health_url` is required. The health protocol uses that exact URL, with a five-second timeout and no redirects, and requires HTTP 200. It does not probe public examples as a substitute. The report records the generated artifact hash and explicitly says `deployment_verified: false`: probing a supplied URL does not prove that URL runs the generated artifact. The generated health handler reports availability, not storage readiness.

Set `REDIS_URL` (or explicit `REDIS_HOST`, optional `REDIS_PORT`/`REDIS_PASSWORD`) before execution. A per-run `cache_config.redis_url` can override it; use `rediss://` when your Redis service requires TLS. `cache_config.key` and a positive integer `ttl_seconds` are required. The step overwrites that selected key with the actual artifact schema and successful matching health report, applies the specified expiry, and verifies an exact readback. Connection details and credentials are not included in its output. Missing configuration, wrong artifact hashes, unhealthy reports or mismatched readback fail. Retries repeat the same key/value write and expiry. This is a cache write, not a deployment: returned deployment artifacts explicitly say `not_deployed_by_workflow`.

## Core workflow and limits

`self_correcting_core` accepts `user_data` as an existing data directory or the supported project snapshot. It performs actual data processing and analysis, then `system_monitor(scope="orchestration")` measures the host running orchestration with psutil. It preserves the monitor's existing resource thresholds and reports unhealthy hosts as failures. It does not claim process-specific or remote fleet health. No-argument calls retain host monitoring; unsupported scopes fail.

Legacy no-argument data-processor/log-analyzer callers retain their file-based behavior. Explicit paths no longer fall back to unrelated directories when invalid. No-argument calls to the old randomized file validator and demonstration API/cache protocols now fail for missing artifact/configuration instead of reporting invented validation or checking unrelated services.

The external source repositories remain archived. This repository is the consolidation destination, but the catalog is not a claim of full WRKFLW product parity or deployment readiness. Real service URLs, chosen persistence and cache settings, model credentials and production evaluation belong to the caller.

## Validation boundary

`tests/test_workflow_runtime.py` executes both actual construction templates using controlled project records, actual protocol/analyzer/code-generator implementations and a test-only replacement for the paid HTTP transport. It checks propagated data, summary requests/provenance, generated scaffold execution, missing inputs, unknown components, invalid citations, incomplete/provider-error responses, retry state and missing credentials. These tests do not prove a live paid-provider request, database retrieval, project-note persistence or production RFI behavior.

`tests/test_catalog_services.py` executes the complete TypeScript pipeline with the real generator/compiler, a Node HTTP adapter around the generated handlers and the actual redis-py TCP client against a controlled RESP fixture. It exercises CRUD persistence through a test-only Store, health failures/redirects, deterministic type-error/path rejection, cache readback/configuration failures and the core pipeline with controlled psutil measurements. The RESP fixture verifies client command/value/TTL behavior; it is not a live production Redis service. Python CI installs the pinned compiler and runs these tests on Python 3.10 and 3.11.
