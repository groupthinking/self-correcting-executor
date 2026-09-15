---
name: github-actions-failure-debugging
description: Guide for debugging failing GitHub Actions workflows in self-correcting-executor. Use this when asked to investigate Python CI, Frontend CI, or PR workflow failures in this repository.
---

Use GitHub MCP tools first, then reproduce locally with the repo's existing
commands.

1. List recent workflow runs for `groupthinking/self-correcting-executor`.
   Focus on:
   - `.github/workflows/python-ci.yml`
   - `.github/workflows/frontend-ci.yml`
2. Get job logs for the failed run or failed jobs.
3. Interpret the result against this repo's workflow behavior:
   - `python-ci.yml` runs `black --check .`
   - `python-ci.yml` currently masks `flake8` and `pytest` failures with
     `|| echo ...` and `continue-on-error`, so a green workflow does not always
     mean imports/tests are healthy
   - `frontend-ci.yml` runs `npm ci || npm install`, then lint/test/build from
     `/frontend`
4. Reproduce only the relevant local command:

   ```bash
   cd /home/runner/work/self-correcting-executor/self-correcting-executor
   python -m pytest tests/ -q
   cd /home/runner/work/self-correcting-executor/self-correcting-executor/frontend
   npm run lint
   npm run build
   ```

5. If the failure touches MCP behavior, also run the repo's live MCP baseline:

   ```bash
   cd /home/runner/work/self-correcting-executor/self-correcting-executor
   python -m pytest tests/test_repo_assist_docs.py -q
   python - <<'PY'
   import asyncio
   from mcp_server.main import MCPServer

   async def main():
       server = MCPServer()
       response = await server.handle_request(
           {
               "jsonrpc": "2.0",
               "id": 1,
               "method": "initialize",
               "params": {"clientInfo": {"name": "ci-debug", "version": "1.0"}},
           }
       )
       print(response["result"]["serverInfo"])

   asyncio.run(main())
   PY
   ```

6. Keep fixes surgical. Do not "clean up" unrelated files just because the
   workflow surfaced pre-existing noise.
