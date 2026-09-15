---
name: codebase-summary
description: Generate a repo-specific map of self-correcting-executor, including the tree, key dependencies, and MCP verification steps. Use this when asked to summarize the architecture or prepare an agent run for this repository.
---

When asked to summarize or orient an agent in this repository, use the current
repo structure instead of generic MCP examples.

1. Start from `/home/runner/work/self-correcting-executor/self-correcting-executor`.
2. Build the summary around these directories:
   - `agents/`
   - `connectors/`
   - `mcp_server/`
   - `protocols/`
   - `frontend/`
   - `tests/`
3. Call out the primary dependency sources:
   - `requirements.txt`
   - `requirements-ci.txt`
   - `requirements-test.txt`
   - `frontend/package.json`
4. When MCP is relevant, explicitly distinguish:
   - **host**: external Copilot/CLI/operator
   - **client**: external MCP-aware caller
   - **server**: the repo's implementations in `mcp_server/` and
     `quantum_mcp_server/`
5. Use the live baseline verification sequence before claiming MCP is working:

   ```python
   import asyncio
   from mcp_server.main import MCPServer

   async def main():
       server = MCPServer()
       await server.handle_request(
           {
               "jsonrpc": "2.0",
               "id": 1,
               "method": "initialize",
               "params": {"clientInfo": {"name": "summary-skill", "version": "1.0"}},
           }
       )
       await server.handle_request(
           {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
       )

   asyncio.run(main())
   ```

6. Point the reader to `docs/REPO_MAP.md` for the maintained tree, mermaid
   diagram, dependency snapshot, workflow checklist, and CI/MCP notes.
