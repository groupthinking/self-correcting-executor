# Ported Workflow Code

## From WRKFLW
TypeScript workflow engine with GraphQL schema and OpenAPI spec.
- `wrkflw/index.ts` — Main backend entry point
- `wrkflw/schema.graphql` — GraphQL workflow schema
- `wrkflw/openapi.yaml` — OpenAPI spec for workflow endpoints
- `wrkflw/PRD.md` — Product requirements document
- `wrkflw/seed-data.json` — Sample workflow data

## From workflows
Next.js API routes for workflow execution pipeline.
- `workflows-api/workflow-execute.ts` — Execute compiled workflows
- `workflows-api/workflow-compile.ts` — Compile workflow definitions
- `workflows-api/plan.ts` — Planning/orchestration endpoint
- `workflows-api/ingest.ts` — Content ingestion endpoint
- `workflows-api/summarize.ts` — Summarization endpoint
- `workflows-api/deploy-scaffold.ts` — Deploy scaffolding

Both repos archived — this is now the canonical workflow execution engine.
