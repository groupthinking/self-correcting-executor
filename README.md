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

## Development

The workspace includes a **DevContainer** definition – simply open the folder in VS Code or Cursor and choose *Reopen in Container* to get an IDE connected to the running services.

## Agentic Workflows

This repository includes agentic workflow definitions designed to integrate with AI agents to automate development tasks. Note that the current Repo Ask workflow returns a static/template response and does not yet invoke an AI agent engine. See [docs/workflows/README.md](docs/workflows/README.md) for available workflows.

### Repo Ask

A repository research assistant workflow (currently implemented as a template-based response workflow). Trigger it by adding a comment to any issue or PR:

```
/repo-ask How does the authentication system work?
```

See [Repo Ask documentation](docs/workflows/repo-ask.md) for more details.
