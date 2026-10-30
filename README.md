

- ✅ **Dockerfile** – Multi-stage, optimized build
- ✅ **docker-compose.yml** – Complete service orchestration
- ✅ **devcontainer.json** – VS Code / Cursor integration
- ✅ **entrypoint.sh** – MCP-compliant startup script
- ✅ **requirements.txt** – Pinned Python dependencies
- ✅ **Makefile** – Developer experience shortcuts
- ✅ **README.md** – Comprehensive documentation
- ✅ **mcp/main.py** – FastAPI MCP server implementation
- ✅ **Workflow catalog** – `workflows/templates.py` consolidates WRKFLW + workflows patterns with self-correction

### Supporting infrastructure

- ✅ PostgreSQL database schema (containerised)
- ✅ Redis configuration optimised for MCP workloads
- ✅ Basic health endpoint at `/health`

## Quick start

```bash
git clone <repo>
cd <repo>
make up   # or docker-compose up -d
make logs # follow logs
make health
```

Open http://localhost:8080/health to verify the API is running.

WRKFLW and workflows repositories are archived. Their BuildPlan/agent/DAG, source intake, construction lifecycle, and artifact patterns are mapped into the canonical engine alongside the original templates. See [source coverage and runnable examples](docs/workflows/IMPORTED_PATTERNS.md) for the exact contracts and required host services.
You can extend the catalog by dropping yaml/json workflow definitions into `workflows/data/`; they are auto-loaded when the catalog module is imported.

## Development

The workspace includes a **DevContainer** definition – simply open the folder in VS Code or Cursor and choose *Reopen in Container* to get an IDE connected to the running services.

## Workflow consolidation runtime status

The catalog auto-loads external definitions, but loader success alone is not runtime readiness. The construction flow now accepts explicit project records and calls the real analyzer and source-grounded summary connector. The TypeScript pipeline generates and compiles actual handlers, checks a supplied service URL and caches schema/report metadata using explicit Redis settings. Live provider access and integration requirements are described in [workflow execution and limits](docs/workflows/CONSOLIDATION.md). The additional imported lifecycle adapters execute RFI state transitions and configured vector-retrieval contracts; live provider acceptance, web/mobile UI parity, and external deployment remain separate. See [imported patterns](docs/workflows/IMPORTED_PATTERNS.md).

## Repo assist

For a repo-specific map of the current tree, dependency snapshot, mermaid
architecture diagram, MCP capability-verification notes, and agent workflow
checklist, see [`docs/REPO_MAP.md`](docs/REPO_MAP.md).

