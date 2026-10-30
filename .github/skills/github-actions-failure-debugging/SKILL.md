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
   - `python-ci.yml` runs pytest as a blocking step. Its existing `flake8`
     step remains advisory via `|| echo ...` and `continue-on-error`, so a
     green workflow does not establish a lint-clean repository.
   - `frontend-ci.yml` runs `npm ci || npm install`, then lint/test/build from
     `/frontend`
4. Reproduce only the relevant local command:

   ```bash
   cd "$(git rev-parse --show-toplevel)"
   python -m pytest tests/ -q
   cd "$(git rev-parse --show-toplevel)/frontend"
   npm run lint
   npm run build
   ```

5. If the failure touches MCP behavior, also run the repo's in-process MCP handler checks:

   ```bash
   cd "$(git rev-parse --show-toplevel)"
   python -m scripts.verify_mcp_baseline
   python -m pytest tests/test_mcp_baseline_verifier.py -q
   ```

   These checks exercise the custom handler in-process; they do not verify
   an external MCP client/transport connection.

6. Keep fixes surgical. Do not "clean up" unrelated files just because the
   workflow surfaced pre-existing noise.
