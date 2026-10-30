---
name: codebase-summary
description: Generate a repo-specific map of self-correcting-executor, including the tree, key dependencies, and MCP verification steps. Use this when asked to summarize the architecture or prepare an agent run for this repository.
---

When asked to summarize or orient an agent in this repository, use the current
repo structure instead of generic MCP examples.

1. Start from the current repository root (`git rev-parse --show-toplevel`).
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
5. Use the in-process handler checks before reporting that baseline as verified:

   ```bash
   cd "$(git rev-parse --show-toplevel)"
   python -m scripts.verify_mcp_baseline
   python -m pytest tests/test_mcp_baseline_verifier.py -q
   ```

   These checks exercise the custom handler in-process; they do not verify
   an external MCP client/transport connection.

6. Point the reader to `docs/REPO_MAP.md` for the maintained tree, mermaid
   diagram, dependency snapshot, workflow checklist, and CI/MCP notes.
