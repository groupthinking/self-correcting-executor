"""Exercise the verifier's success and failure paths without external services."""

import asyncio

import pytest

from mcp_server.main import MCPServer
from scripts.verify_mcp_baseline import verify_baseline


def test_verifier_checks_actual_handler():
    result = asyncio.run(verify_baseline())
    assert result["analysis"]["functions"] == 1


@pytest.mark.parametrize("method", ["initialize", "tools/list", "tools/call"])
def test_verifier_rejects_rpc_error_at_each_step(method):
    class FailingHandler(MCPServer):
        async def handle_request(self, request):
            if request["method"] == method:
                return {
                    "jsonrpc": "2.0",
                    "id": request["id"],
                    "error": {"code": -32000},
                }
            return await super().handle_request(request)

    with pytest.raises(RuntimeError, match=method + " failed"):
        asyncio.run(verify_baseline(FailingHandler()))


def test_verifier_rejects_wrong_analysis_even_when_rpc_succeeds():
    class IncorrectAnalysis(MCPServer):
        async def handle_request(self, request):
            response = await super().handle_request(request)
            if request["method"] == "tools/call":
                response["result"]["result"]["content"][0]["text"] = "{}"
            return response

    with pytest.raises(RuntimeError, match="unexpected analysis"):
        asyncio.run(verify_baseline(IncorrectAnalysis()))


def test_verifier_rejects_missing_advertised_tool():
    class MissingTool(MCPServer):
        async def handle_request(self, request):
            response = await super().handle_request(request)
            if request["method"] == "initialize":
                response["result"]["capabilities"]["tools"] = []
            return response

    with pytest.raises(RuntimeError, match="expected tools"):
        asyncio.run(verify_baseline(MissingTool()))
