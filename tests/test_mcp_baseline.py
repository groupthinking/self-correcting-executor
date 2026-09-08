#!/usr/bin/env python3
"""Baseline MCP capability negotiation tests.

These tests prove that the repository's JSON-RPC based MCP server
(`mcp_server/main.py`) implements the minimum handshake required for a host
to negotiate capabilities with the server: `initialize`, `tools/list`, and
`tools/call`. This is the smallest possible verification that MCP is
functional for this repo, per the Model Context Protocol specification
(https://modelcontextprotocol.io/specification).
"""

import pytest

from mcp_server.main import MCPServer, MCP_VERSION


@pytest.fixture
def mcp_server() -> MCPServer:
    """Create a fresh MCP server instance for each test."""
    return MCPServer()


@pytest.mark.asyncio
async def test_initialize_declares_capabilities(mcp_server: MCPServer):
    """The 'initialize' handshake must advertise server info and capabilities."""
    response = await mcp_server.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"clientInfo": {"name": "test-client", "version": "1.0"}},
        }
    )

    assert response["jsonrpc"] == "2.0"
    assert response["id"] == 1
    assert "error" not in response

    result = response["result"]
    assert result["serverInfo"]["mcpVersion"] == MCP_VERSION
    assert "tools" in result["capabilities"]
    assert "resources" in result["capabilities"]
    tool_names = {tool["name"] for tool in result["capabilities"]["tools"]}
    assert {"code_analyzer", "protocol_validator", "self_corrector"} <= tool_names


@pytest.mark.asyncio
async def test_tools_list_matches_advertised_capabilities(mcp_server: MCPServer):
    """'tools/list' must be callable independently of 'initialize'."""
    response = await mcp_server.handle_request(
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
    )

    assert "error" not in response
    tool_names = {tool["name"] for tool in response["result"]["tools"]}
    assert {"code_analyzer", "protocol_validator", "self_corrector"} <= tool_names


@pytest.mark.asyncio
async def test_tools_call_executes_a_real_tool(mcp_server: MCPServer):
    """'tools/call' must invoke real tool logic, proving negotiation works end-to-end."""
    response = await mcp_server.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "code_analyzer",
                "arguments": {"code": "def f():\n    return 1\n"},
            },
        }
    )

    assert "error" not in response
    assert response["result"]["tool"] == "code_analyzer"


@pytest.mark.asyncio
async def test_unknown_method_returns_json_rpc_error(mcp_server: MCPServer):
    """Unsupported methods must return a well-formed JSON-RPC error, not crash."""
    response = await mcp_server.handle_request(
        {"jsonrpc": "2.0", "id": 4, "method": "not/a/real/method", "params": {}}
    )

    assert response["jsonrpc"] == "2.0"
    assert response["id"] == 4
    assert "error" in response
    assert response["error"]["code"] == -32000
