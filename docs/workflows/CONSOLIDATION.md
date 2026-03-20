# Workflow Consolidation

This repository now hosts all workflow patterns from:
- **self-correcting-executor** — auto-fix orchestration engine (core runtime)
- **WRKFLW** — construction management workflows (daily logs, RFIs, AI summaries)
- **workflows** — TypeScript workflow definitions (service scaffolds + CI/deploy)

## What changed
- Added `workflows` package with declarative templates for each source repo.
- Templates are wrapped with self-correction metadata so the orchestrator retries and records failures in the knowledge graph.
- `orchestrator.OrchestrationEngine` now prefers these templates when an intent matches their tags or when `workflow_template` is provided.
- External data-backed templates can be added under `workflows/data/*.yaml|json`; these are auto-loaded into the catalog (example: `construction_daily_log_external` ported from WRKFLW PRD).

## How to use
```python
from orchestrator import OrchestrationEngine

engine = OrchestrationEngine()
result = asyncio.run(
    engine.execute_intent(
        "Generate a construction daily log workflow",
        options={"workflow_template": "construction_daily_log"},
    )
)
```

See `workflows/templates.py` for the consolidated catalog. External source repos (`WRKFLW`, `workflows`) are now archived; this repository is the canonical home for workflow execution and self-healing.
