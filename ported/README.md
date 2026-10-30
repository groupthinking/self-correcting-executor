# Ported Workflow Code

## From WRKFLW
TypeScript workflow engine with GraphQL schema and OpenAPI spec.
- `wrkflw/index.ts` — Main backend entry point
- `wrkflw/schema.graphql` — GraphQL workflow schema
- `wrkflw/openapi.yaml` — OpenAPI spec for workflow endpoints
- `wrkflw/PRD.md` — Product requirements document

## From workflows
Next.js API routes for workflow execution pipeline.
- `workflows-api/workflow-execute.ts` — Execute compiled workflows
- `workflows-api/workflow-compile.ts` — Compile workflow definitions
- `workflows-api/plan.ts` — Planning/orchestration endpoint
- `workflows-api/ingest.ts` — Content ingestion endpoint
- `workflows-api/summarize.ts` — Summarization endpoint
- `workflows-api/deploy-scaffold.ts` — Deploy scaffolding

Both repos archived — this is now the canonical workflow execution engine.

## Canonical execution

These files preserve the original source snapshots, including their historical Next.js aliases. Their runnable canonical API equivalents are `workflows/imported.py`; supporting intake, planning, graph, protocol and scaffold modules live in `workflows/`. The archived clients included simulated outcomes, which the canonical engine replaces with explicit real host adapters and failure paths. See [complete mapping and examples](../docs/workflows/IMPORTED_PATTERNS.md). Source snapshots are not a separately deployable Next.js application.
