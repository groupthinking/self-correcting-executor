---
name: mcp-protocol-debugging
description: Guide for exercising and debugging this repo's MCP JSON-RPC server (mcp_server/main.py), including capability negotiation (initialize/tools/list), tool calls, and error handling. Use when asked to verify, debug, or extend MCP support in self-correcting-executor.
---

This repo ships a canonical JSON-RPC MCP server at `mcp_server/main.py`
(`MCPServer` class), independent from the FastAPI-based
`mcp_server/real_mcp_server.py`. Use this process to verify or debug MCP
behavior. Direct handler calls do not prove transport compatibility or MCP
specification compliance:

1. **Confirm the baseline capability negotiation works** by driving the
   server directly in Python (no transport needed for debugging):

   ```python
   import asyncio
   import json
   from mcp_server.main import MCPServer

   async def main():
       server = MCPServer()
       # 1. initialize — must return serverInfo + capabilities.tools/resources
       init = await server.handle_request({
           "jsonrpc": "2.0", "id": 1, "method": "initialize",
           "params": {"clientInfo": {"name": "debug-client", "version": "1.0"}},
       })
       assert "error" not in init
       advertised = {tool["name"] for tool in init["result"]["capabilities"]["tools"]}
       assert "code_analyzer" in advertised

       # 2. tools/list — must match the tools advertised in `initialize`
       listed = await server.handle_request(
           {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
       )
       assert "error" not in listed
       assert {tool["name"] for tool in listed["result"]["tools"]} == advertised

       # 3. tools/call — must execute real logic (code_analyzer, protocol_validator,
       #    self_corrector), never mocked/simulated results
       called = await server.handle_request({
           "jsonrpc": "2.0", "id": 3, "method": "tools/call",
           "params": {"name": "code_analyzer", "arguments": {"code": "x = 1"}},
       })
       assert "error" not in called
       assert called["result"]["tool"] == "code_analyzer"
       analysis = json.loads(called["result"]["result"]["content"][0]["text"])
       assert analysis == {"lines_of_code": 1, "functions": 0, "complexity": 1, "imports": []}

   asyncio.run(main())
   ```

2. **Run the automated baseline test** that pins this behavior down:

   ```bash
   python -m pytest tests/test_mcp_baseline.py -v
   ```

3. **When adding a new MCP tool or resource**, update all three of:
   - `MCPServer._setup_tools` / `_setup_resources` (declares the capability)
   - the corresponding `_execute_<tool_name>` method (real implementation,
     never a stub/mock; inspect production behavior directly because
     `TestMCPCompliance.test_no_placeholder_code_in_production` reports
     placeholder patterns but does not fail the test)
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
