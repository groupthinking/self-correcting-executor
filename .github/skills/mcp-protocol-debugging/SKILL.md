---
name: mcp-protocol-debugging
description: Guide for exercising and debugging this repo's MCP JSON-RPC server (mcp_server/main.py), including capability negotiation (initialize/tools/list), tool calls, and error handling. Use when asked to verify, debug, or extend MCP support in self-correcting-executor.
---

This repo ships a canonical JSON-RPC MCP server at `mcp_server/main.py`
(`MCPServer` class), independent from the FastAPI-based
`mcp_server/real_mcp_server.py`. Use this process to verify or debug MCP
behavior:

1. **Confirm the baseline capability negotiation works** by driving the
   server directly in Python (no transport needed for debugging):

   ```python
   import asyncio
   from mcp_server.main import MCPServer

   async def main():
       server = MCPServer()
       # 1. initialize — must return serverInfo + capabilities.tools/resources
       init = await server.handle_request({
           "jsonrpc": "2.0", "id": 1, "method": "initialize",
           "params": {"clientInfo": {"name": "debug-client", "version": "1.0"}},
       })
       print(init)

       # 2. tools/list — must match the tools advertised in `initialize`
       print(await server.handle_request(
           {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
       ))

       # 3. tools/call — must execute real logic (code_analyzer, protocol_validator,
       #    self_corrector), never mocked/simulated results
       print(await server.handle_request({
           "jsonrpc": "2.0", "id": 3, "method": "tools/call",
           "params": {"name": "code_analyzer", "arguments": {"code": "x = 1"}},
       }))

   asyncio.run(main())
   ```

2. **Run the automated baseline test** that pins this behavior down:

   ```bash
   python -m pytest tests/test_mcp_baseline.py -v
   ```

3. **When adding a new MCP tool or resource**, update all three of:
   - `MCPServer._setup_tools` / `_setup_resources` (declares the capability)
   - the corresponding `_execute_<tool_name>` method (real implementation,
     never a stub/mock — this repo enforces "no mocks in production", see
     `tests/test_mcp_compliance.py::test_no_placeholder_code_in_production`)
   - `tests/test_mcp_baseline.py` and/or `tests/test_mcp_compliance.py` to
     cover the new capability.

4. **Unknown or malformed methods must degrade gracefully**: `handle_request`
   should always return a well-formed JSON-RPC response with an
   `{"error": {"code": ..., "message": ...}}` field rather than raising
   an unhandled exception — verify this whenever you touch `_get_handler`.

5. For the FastAPI-based server (`mcp_server/real_mcp_server.py`) or the
   quantum-specific server (`quantum_mcp_server/quantum_mcp.py`), apply the
   same principle: prove `initialize`/capability discovery works live before
   relying on more advanced features (subscriptions, resource reads, etc.).
