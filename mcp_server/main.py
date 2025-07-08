#!/usr/bin/env python3
# pylint: disable=line-too-long, too-many-lines
"""
Proper MCP Server Implementation
===============================

This implements a Model Context Protocol (MCP) server that follows the official
MCP specification for JSON-RPC communication with AI clients.
"""

import asyncio
import json
import logging
import sys
import ast
import re
from typing import Dict, List, Any, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

# MCP Protocol Constants
MCP_VERSION = "2024-11-05"
PROTOCOL_VERSION = "2024-11-05"


class MCPTool:
    """Represents an MCP tool that can be called by clients."""

    def __init__(self, name: str, description: str, input_schema: Dict[str, Any]):
        self.name = name
        self.description = description
        self.input_schema = input_schema

    def to_dict(self) -> Dict[str, Any]:
        """Converts tool to a dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }


class MCPResource:
    """Represents an MCP resource that can be accessed by clients."""

    def __init__(self, uri: str, name: str, description: str, mime_type: str):
        self.uri = uri
        self.name = name
        self.description = description
        self.mime_type = mime_type

    def to_dict(self) -> Dict[str, Any]:
        """Converts resource to a dictionary."""
        return {
            "uri": self.uri,
            "name": self.name,
            "description": self.description,
            "mimeType": self.mime_type,
        }


class MCPServer:
    """Main MCP server implementation."""

    def __init__(self):
        self.tools: List[MCPTool] = []
        self.resources: List[MCPResource] = []
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self._setup_tools()
        self._setup_resources()

    def _setup_tools(self):
        """Initialize available tools."""

        # Code Analyzer Tool
        self.tools.append(
            MCPTool(
                name="code_analyzer",
                description="Analyzes code structure, complexity, and provides insights",
                input_schema={
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "The code to analyze",
                        },
                        "language": {
                            "type": "string",
                            "description": "Programming language",
                            "default": "python",
                        },
                    },
                    "required": ["code"],
                },
            )
        )

        # Protocol Validator Tool
        self.tools.append(
            MCPTool(
                name="protocol_validator",
                description="Validates MCP protocol compliance and JSON-RPC messages",
                input_schema={
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "JSON-RPC message to validate",
                        },
                        "protocol_version": {
                            "type": "string",
                            "description": "Expected protocol version",
                            "default": MCP_VERSION,
                        },
                    },
                    "required": ["message"],
                },
            )
        )

        # Self Corrector Tool
        self.tools.append(
            MCPTool(
                name="self_corrector",
                description="Identifies and suggests corrections for code issues",
                input_schema={
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Code to analyze for issues",
                        },
                        "language": {
                            "type": "string",
                            "description": "Programming language",
                            "default": "python",
                        },
                        "strict_mode": {
                            "type": "boolean",
                            "description": "Enable strict validation",
                            "default": False,
                        },
                    },
                    "required": ["code"],
                },
            )
        )

    def _setup_resources(self):
        """Initialize available resources."""

        self.resources.append(
            MCPResource(
                uri="file:///mcp/protocol-spec.md",
                name="MCP Protocol Specification",
                description="Official Model Context Protocol specification",
                mime_type="text/markdown",
            )
        )

        self.resources.append(
            MCPResource(
                uri="file:///mcp/tools.json",
                name="Available Tools",
                description="List of all available MCP tools",
                mime_type="application/json",
            )
        )

    async def handle_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming JSON-RPC requests."""

        request_id = request_data.get("id")
        method = request_data.get("method")
        params = request_data.get("params", {})

        LOGGER.info("Handling request: %s (ID: %s)", method, request_id)

        try:
            handler = self._get_handler(method)
            result = await handler(params)

            return {"jsonrpc": "2.0", "id": request_id, "result": result}

        except Exception as e:  # pylint: disable=broad-except-clause
            LOGGER.error("Error handling request: %s", e)

            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32000, "message": str(e)},
            }

    def _get_handler(self, method: Optional[str]):
        """Get the appropriate handler for a given method."""
        handlers = {
            "initialize": self._handle_initialize,
            "tools/list": self._handle_tools_list,
            "tools/call": self._handle_tools_call,
            "resources/list": self._handle_resources_list,
            "resources/read": self._handle_resources_read,
            "notifications/list": self._handle_notifications_list,
            "notifications/subscribe": self._handle_notifications_subscribe,
        }
        if method not in handlers:
            raise Exception(f"Unknown method: {method}")
        return handlers[method]

    async def _handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle the 'initialize' request."""
        client_info = params.get("clientInfo", {})
        LOGGER.info("Initializing session for client: %s", client_info.get("name"))

        return {
            "serverInfo": {
                "name": "Self-Correcting Executor MCP Server",
                "version": "1.0.0",
                "mcpVersion": MCP_VERSION,
            },
            "capabilities": {
                "tools": [tool.to_dict() for tool in self.tools],
                "resources": [resource.to_dict() for resource in self.resources],
            },
        }

    async def _handle_tools_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle the 'tools/list' request."""
        _ = params  # Unused
        return {"tools": [tool.to_dict() for tool in self.tools]}

    async def _handle_tools_call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle the 'tools/call' request."""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        if tool_name == "code_analyzer":
            result = await self._execute_code_analyzer(arguments)
        elif tool_name == "protocol_validator":
            result = await self._execute_protocol_validator(arguments)
        elif tool_name == "self_corrector":
            result = await self._execute_self_corrector(arguments)
        else:
            raise Exception(f"Unknown tool: {tool_name}")

        return {
            "tool": tool_name,
            "result": result,
            "timestamp": datetime.utcnow().isoformat(),
        }

    async def _handle_resources_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle the 'resources/list' request."""
        _ = params  # Unused
        return {"resources": [resource.to_dict() for resource in self.resources]}

    async def _handle_resources_read(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle the 'resources/read' request."""
        uri = params.get("uri")
        content = ""
        if uri == "file:///mcp/protocol-spec.md":
            content = f"# MCP Protocol Specification\n\nVersion: {PROTOCOL_VERSION}"
        elif uri == "file:///mcp/tools.json":
            content = json.dumps(
                {"tools": [tool.to_dict() for tool in self.tools]}, indent=2
            )
        else:
            raise Exception(f"Unknown resource URI: {uri}")

        return {
            "uri": uri,
            "content": [
                {
                    "mimeType": "text/plain",
                    "text": content,
                    "language": "en",
                }
            ],
        }

    async def _handle_notifications_list(
        self, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle the 'notifications/list' request."""
        _ = params  # Unused
        return {"notifications": ["server/statusUpdate", "agent/taskComplete"]}

    async def _handle_notifications_subscribe(
        self, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle the 'notifications/subscribe' request."""
        _ = params  # Unused
        return {"status": "subscribed"}

    async def _execute_code_analyzer(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the 'code_analyzer' tool."""
        code = arguments.get("code")
        if not code:
            raise ValueError("'code' argument is required for code_analyzer")

        try:
            tree = ast.parse(code)
            lines_of_code = len(code.splitlines())
            num_functions = sum(
                1 for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
            )
            complexity = self._calculate_complexity(tree)

            result = {
                "lines_of_code": lines_of_code,
                "functions": num_functions,
                "complexity": complexity,
                "imports": [
                    node.names[0].name
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Import)
                ],
            }
            return {
                "content": [
                    {
                        "mimeType": "application/json",
                        "text": json.dumps(result, indent=2),
                    }
                ]
            }
        except SyntaxError as e:
            raise ValueError(f"Invalid syntax in provided code: {e}") from e

    def _calculate_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity of the code."""
        complexity = 1
        for node in ast.walk(tree):
            if isinstance(
                node,
                (
                    ast.If,
                    ast.For,
                    ast.While,
                    ast.And,
                    ast.Or,
                    ast.Try,
                    ast.ExceptHandler,
                    ast.With,
                ),
            ):
                complexity += 1
        return complexity

    async def _execute_protocol_validator(
        self, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute the 'protocol_validator' tool."""
        message_str = arguments.get("message")
        if not message_str:
            raise ValueError("'message' argument is required for protocol_validator")

        issues = []
        try:
            message = json.loads(message_str)
            if not isinstance(message, dict):
                issues.append("Message is not a JSON object")
                raise TypeError()

            if message.get("jsonrpc") != "2.0":
                issues.append("Invalid 'jsonrpc' version")
            if (
                "method" not in message
                and "result" not in message
                and "error" not in message
            ):
                issues.append("Missing 'method', 'result', or 'error' field")
            if "id" not in message:
                issues.append("Missing 'id' field for request/response")

        except json.JSONDecodeError:
            issues.append("Invalid JSON format")
        except TypeError:
            pass  # Already handled by check

        result = {"valid": not issues, "issues": issues}
        return {
            "content": [
                {
                    "mimeType": "application/json",
                    "text": json.dumps(result, indent=2),
                }
            ]
        }

    async def _execute_self_corrector(
        self, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute the 'self_corrector' tool."""
        code = arguments.get("code")
        if not code:
            raise ValueError("'code' argument is required for self_corrector")

        suggestions = []
        try:
            ast.parse(code)
            # Check for common anti-patterns
            if "time.sleep" in code:
                suggestions.append(
                    "Found 'time.sleep'. Consider using 'asyncio.sleep' in async code."
                )
            if re.search(r"except\s*:", code):
                suggestions.append("Found broad 'except:'. Specify the exception type.")

        except SyntaxError as e:
            suggestions.append(f"Syntax Error: {e}")

        result = {"issues_found": len(suggestions), "suggestions": suggestions}
        return {
            "content": [
                {
                    "mimeType": "application/json",
                    "text": json.dumps(result, indent=2),
                }
            ]
        }

    def _has_import(self, module_name: str, tree: ast.AST) -> bool:
        """Check if a module is imported."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Import) and any(
                alias.name == module_name for alias in node.names
            ):
                return True
            if isinstance(node, ast.ImportFrom) and node.module == module_name:
                return True
        return False


async def handle_stdin_stdout():
    """Handle JSON-RPC communication over stdin/stdout."""
    server = MCPServer()
    LOGGER.info("MCP Server listening on stdin/stdout...")

    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await asyncio.get_event_loop().connect_read_pipe(lambda: protocol, sys.stdin)

    writer = None
    if sys.platform != "win32":
        (
            w_transport,
            w_protocol,
        ) = await asyncio.get_event_loop().connect_write_pipe(
            asyncio.Protocol, sys.stdout
        )
        writer = asyncio.StreamWriter(
            w_transport, w_protocol, None, asyncio.get_event_loop()
        )

    while not reader.at_eof():
        line = await reader.readline()
        if not line:
            break

        try:
            request = json.loads(line)
            response = await server.handle_request(request)
            response_str = json.dumps(response) + "\n"

            if writer:
                writer.write(response_str.encode())
                await writer.drain()
            else:  # Fallback for Windows
                print(response_str, flush=True)

        except json.JSONDecodeError:
            error_response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": "Parse error"},
            }
            print(json.dumps(error_response), flush=True)


if __name__ == "__main__":
    try:
        asyncio.run(handle_stdin_stdout())
    except KeyboardInterrupt:
        LOGGER.info("MCP Server deactivated.")
