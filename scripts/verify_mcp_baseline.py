"""Verify the custom MCP handler in-process; no network transport is tested."""

import asyncio
import json

from mcp_server.main import MCPServer, MCP_VERSION


async def verify_baseline(server=None):
    """Raise on RPC errors, missing capabilities, or incorrect analysis."""
    server = server if server is not None else MCPServer()

    async def request(request_id, method, params):
        response = await server.handle_request(
            {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        )
        if response.get("error") is not None or "result" not in response:
            raise RuntimeError(f"{method} failed: {response}")
        if response.get("jsonrpc") != "2.0" or response.get("id") != request_id:
            raise RuntimeError(f"{method} returned an invalid response: {response}")
        return response["result"]

    initialized = await request(
        1, "initialize", {"clientInfo": {"name": "repo-assist", "version": "1.0"}}
    )
    if initialized["serverInfo"]["mcpVersion"] != MCP_VERSION:
        raise RuntimeError("initialize returned an unexpected protocol version")
    expected_tools = {"code_analyzer", "protocol_validator", "self_corrector"}
    advertised_tools = {tool["name"] for tool in initialized["capabilities"]["tools"]}
    if not expected_tools <= advertised_tools:
        raise RuntimeError("initialize did not advertise the expected tools")
    if not initialized["capabilities"]["resources"]:
        raise RuntimeError("initialize did not advertise resources")
    listed = await request(2, "tools/list", {})
    if {tool["name"] for tool in listed["tools"]} != advertised_tools:
        raise RuntimeError("tools/list differs from advertised capabilities")
    called = await request(
        3,
        "tools/call",
        {
            "name": "code_analyzer",
            "arguments": {"code": "def verify():\n    return 1\n"},
        },
    )
    if called["tool"] != "code_analyzer":
        raise RuntimeError("tools/call returned the wrong tool")
    analysis = json.loads(called["result"]["content"][0]["text"])
    if analysis != {
        "lines_of_code": 2,
        "functions": 1,
        "complexity": 1,
        "imports": [],
    }:
        raise RuntimeError(f"code_analyzer returned unexpected analysis: {analysis}")
    return {"serverInfo": initialized["serverInfo"], "analysis": analysis}


if __name__ == "__main__":
    print(json.dumps(asyncio.run(verify_baseline()), indent=2))
