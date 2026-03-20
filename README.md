# Unified MCP DevContainer & Runtime (UMDR)

This repository contains a canonical Docker-based development environment for MCP services.

## Deliverables

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

WRKFLW and workflows repositories are archived; their workflow definitions now live in `workflows/templates.py` alongside the self-correcting engine.

## Development

The workspace includes a **DevContainer** definition – simply open the folder in VS Code or Cursor and choose *Reopen in Container* to get an IDE connected to the running services. 
