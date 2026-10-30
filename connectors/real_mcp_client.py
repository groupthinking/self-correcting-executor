#!/usr/bin/env python3
"""
Real MCP Client Implementation
==============================

This implements a real MCP client that communicates with MCP servers using
the JSON-RPC protocol over stdio transport. This replaces all simulated
MCP processing with actual MCP server communication.
"""

import asyncio
import json
import logging
import subprocess
import time
import uuid
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class MCPClient:
    """Real MCP client for stdio communication with MCP servers"""

    def __init__(self, server_path: Optional[str] = None):
        self.server_path = server_path or str(
            Path(__file__).parent.parent / "mcp_server" / "main.py"
        )
        self.process: Optional[subprocess.Popen] = None
        self.connected = False
        self.server_info: dict[str, Any] = {}
        self.tools: list[dict[str, Any]] = []
        self.resources: list[dict[str, Any]] = []
        self._message_id_counter = 0
        self._pending_requests: Dict[int, asyncio.Future] = {}
        self._reader_task: Optional[asyncio.Task] = None

    async def connect(self, max_connection_attempts: int = 3) -> bool:
        """Connect to MCP server via stdio with retry logic"""
        for attempt in range(max_connection_attempts):
            try:
                logger.info(f"Starting MCP server (attempt {attempt + 1}): {self.server_path}")
                
                # Start MCP server process
                self.process = subprocess.Popen(
                    ["python3", self.server_path],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    text=True,
                    bufsize=0  # Unbuffered for real-time communication
                )
                
                # Wait a moment for server to start
                await asyncio.sleep(0.1)
                
                # Check if process started successfully
                if self.process.poll() is not None:
                    stdout, stderr = self.process.communicate()
                    logger.error(f"MCP server failed to start: {stderr}")
                    if attempt < max_connection_attempts - 1:
                        await asyncio.sleep(1.0)  # Wait before retry
                        continue
                    return False
                
                # Initialize connection
                result = await self._send_request("initialize", {
                    "clientInfo": {
                        "name": "Self-Correcting Executor", 
                        "version": "1.0.0"
                    },
                    "protocolVersion": "2024-11-05"
                })
                
                if result and "serverInfo" in result:
                    self.server_info = result["serverInfo"]
                    self.connected = True
                    
                    # Get available tools
                    tools_result = await self._send_request("tools/list", {})
                    if tools_result and "tools" in tools_result:
                        self.tools = tools_result["tools"]
                    
                    # Get available resources
                    resources_result = await self._send_request("resources/list", {})
                    if resources_result and "resources" in resources_result:
                        self.resources = resources_result["resources"]
                    
                    logger.info(f"Connected to MCP server: {self.server_info.get('name')}")
                    logger.info(f"Available tools: {len(self.tools)}")
                    return True
                else:
                    logger.error("Failed to initialize MCP connection")
                    await self.disconnect()
                    if attempt < max_connection_attempts - 1:
                        await asyncio.sleep(1.0)
                        continue
                
            except Exception as e:
                logger.error(f"Connection attempt {attempt + 1} failed: {e}")
                await self.disconnect()
                if attempt < max_connection_attempts - 1:
                    await asyncio.sleep(1.0)
                    continue
        
        logger.error(f"All {max_connection_attempts} connection attempts failed")
        return False

    async def disconnect(self):
        """Disconnect from MCP server"""
        if self.process:
            try:
                self.process.terminate()
                await asyncio.wait_for(
                    asyncio.create_task(self._wait_for_process()), 
                    timeout=5.0
                )
            except asyncio.TimeoutError:
                logger.warning("MCP server didn't terminate gracefully, killing")
                self.process.kill()
            finally:
                self.process = None
        
        self.connected = False
        self._pending_requests.clear()
        logger.info("Disconnected from MCP server")

    async def _wait_for_process(self):
        """Wait for process to terminate"""
        if self.process:
            while self.process.poll() is None:
                await asyncio.sleep(0.1)

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool on the MCP server"""
        if not self.connected:
            raise ConnectionError("Not connected to MCP server")
        
        start_time = time.time()
        
        try:
            result = await self._send_request("tools/call", {
                "name": tool_name,
                "arguments": arguments
            })
            
            if result:
                # Calculate real latency
                latency_ms = (time.time() - start_time) * 1000
                
                return {
                    "tool": tool_name,
                    "result": result.get("result", {}),
                    "latency_ms": latency_ms,
                    "timestamp": result.get("timestamp"),
                    "status": "success"
                }
            else:
                return {
                    "tool": tool_name,
                    "error": "No result received",
                    "status": "error"
                }
                
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error(f"Tool call failed: {e}")
            return {
                "tool": tool_name,
                "error": str(e),
                "latency_ms": latency_ms,
                "status": "error"
            }

    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools"""
        if not self.connected:
            await self.connect()
        
        return self.tools.copy()

    async def list_resources(self) -> List[Dict[str, Any]]:
        """List available resources"""
        if not self.connected:
            await self.connect()
        
        return self.resources.copy()

    async def read_resource(self, uri: str) -> Dict[str, Any]:
        """Read a resource from the MCP server"""
        if not self.connected:
            raise ConnectionError("Not connected to MCP server")
        
        try:
            result = await self._send_request("resources/read", {"uri": uri})
            return result or {}
        except Exception as e:
            logger.error(f"Resource read failed: {e}")
            return {"error": str(e)}

    async def _send_request(self, method: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Send JSON-RPC request to MCP server"""
        if not self.process or not self.process.stdin:
            raise ConnectionError("MCP server process not available")
        
        # Generate unique message ID
        message_id = self._get_next_message_id()
        
        # Create JSON-RPC request
        request = {
            "jsonrpc": "2.0",
            "id": message_id,
            "method": method,
            "params": params
        }
        
        # Create future for response
        future: asyncio.Future[dict[str, Any]] = asyncio.Future()
        self._pending_requests[message_id] = future
        
        try:
            # Send request
            request_json = json.dumps(request) + "\n"
            self.process.stdin.write(request_json)
            self.process.stdin.flush()
            
            # Start response reader if not already running
            if self._reader_task is None or self._reader_task.done():
                self._reader_task = asyncio.create_task(self._read_responses())
            
            # Wait for response with timeout
            result = await asyncio.wait_for(future, timeout=30.0)
            return result
            
        except asyncio.TimeoutError:
            logger.error(f"Request timeout for method: {method}")
            self._pending_requests.pop(message_id, None)
            return None
        except Exception as e:
            logger.error(f"Request failed: {e}")
            self._pending_requests.pop(message_id, None)
            return None

    async def _read_responses(self):
        """Read responses from MCP server stdout"""
        if not self.process or not self.process.stdout:
            return
        
        try:
            while self.process and self.process.poll() is None:
                line = await asyncio.create_task(
                    self._read_line_async(self.process.stdout)
                )
                
                if not line:
                    break
                
                try:
                    response = json.loads(line.strip())
                    message_id = response.get("id")
                    
                    if message_id in self._pending_requests:
                        future = self._pending_requests.pop(message_id)
                        
                        if "error" in response:
                            future.set_exception(
                                Exception(f"MCP Error: {response['error']}")
                            )
                        else:
                            future.set_result(response.get("result"))
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON response: {e}")
                except Exception as e:
                    logger.error(f"Error processing response: {e}")
                    
        except Exception as e:
            logger.error(f"Response reader error: {e}")

    async def _read_line_async(self, stream) -> str:
        """Read a line asynchronously from stream"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, stream.readline)

    def _get_next_message_id(self) -> int:
        """Get next message ID"""
        self._message_id_counter += 1
        return self._message_id_counter

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on MCP connection"""
        if not self.connected:
            return {
                "status": "disconnected",
                "server_info": None,
                "tools_available": 0,
                "resources_available": 0
            }
        
        # Test with a simple tool call
        try:
            start_time = time.time()
            tools = await self.list_tools()
            latency = (time.time() - start_time) * 1000
            
            return {
                "status": "connected",
                "server_info": self.server_info,
                "tools_available": len(tools),
                "resources_available": len(self.resources),
                "latency_ms": latency,
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }


class MCPClientPool:
    """Pool of MCP clients for high-throughput scenarios"""
    
    def __init__(self, pool_size: int = 3):
        self.pool_size = pool_size
        self.clients: List[MCPClient] = []
        self.available_clients: asyncio.Queue = asyncio.Queue()
        self.stats = {
            "total_requests": 0,
            "active_connections": 0,
            "avg_latency_ms": 0.0,
            "error_rate": 0.0
        }
        
    async def initialize(self):
        """Initialize the client pool"""
        logger.info(f"Initializing MCP client pool with {self.pool_size} clients")
        
        for i in range(self.pool_size):
            client = MCPClient()
            if await client.connect():
                self.clients.append(client)
                await self.available_clients.put(client)
                self.stats["active_connections"] += 1
            else:
                logger.error(f"Failed to initialize client {i}")
        
        logger.info(f"Initialized {self.stats['active_connections']} MCP clients")
    
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any], timeout: float = 30.0) -> Dict[str, Any]:
        """Execute tool using available client from pool. Queues requests when pool is busy."""
        import asyncio

        try:
            # Wait for available client (blocks if pool is empty, queues concurrent requests)
            client = await asyncio.wait_for(self.available_clients.get(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(f"Timeout waiting for MCP client after {timeout}s for tool: {tool_name}")
            return {
                "tool": tool_name,
                "error": f"Timeout waiting for available MCP client ({timeout}s)",
                "status": "error"
            }
        
        try:
            # Execute tool
            result = await client.call_tool(tool_name, arguments)
            
            # Update stats
            self.stats["total_requests"] += 1
            if result.get("status") == "success":
                latency = result.get("latency_ms", 0)
                self.stats["avg_latency_ms"] = (
                    (self.stats["avg_latency_ms"] * (self.stats["total_requests"] - 1) + latency)
                    / self.stats["total_requests"]
                )
            else:
                # Calculate error rate
                error_count = sum(1 for c in self.clients if not c.connected)
                self.stats["error_rate"] = error_count / len(self.clients)
            
            return result
            
        finally:
            # Return client to pool
            await self.available_clients.put(client)
    
    async def shutdown(self):
        """Shutdown all clients in the pool"""
        logger.info("Shutting down MCP client pool")
        
        for client in self.clients:
            await client.disconnect()
        
        self.clients.clear()
        self.stats["active_connections"] = 0


# Global client pool instance
_client_pool: Optional[MCPClientPool] = None


async def get_mcp_client_pool() -> MCPClientPool:
    """Get or create global MCP client pool"""
    global _client_pool
    
    if _client_pool is None:
        _client_pool = MCPClientPool(pool_size=3)
        await _client_pool.initialize()
    
    return _client_pool


async def execute_mcp_tool(tool_name: str, arguments: Dict[str, Any], max_retries: int = 3) -> Dict[str, Any]:
    """Execute MCP tool using the global client pool with retry logic"""
    last_error = None
    
    for attempt in range(max_retries):
        try:
            pool = await get_mcp_client_pool()
            result = await pool.execute_tool(tool_name, arguments)
            
            # If successful, return result
            if result.get("status") == "success":
                return result
            
            # If MCP server error, retry
            if "connection" in str(result.get("error", "")).lower():
                logger.warning(f"Connection error on attempt {attempt + 1}, retrying...")
                await asyncio.sleep(0.5 * (attempt + 1))  # Exponential backoff
                continue
            
            # For non-connection errors, return immediately
            return result
            
        except Exception as e:
            last_error = e
            logger.error(f"Attempt {attempt + 1} failed: {e}")
            
            if attempt < max_retries - 1:
                await asyncio.sleep(0.5 * (attempt + 1))  # Exponential backoff
    
    # All retries failed
    return {
        "tool": tool_name,
        "status": "error",
        "error": f"All {max_retries} attempts failed. Last error: {last_error}",
        "attempts": max_retries
    }


# Example usage and testing
async def test_real_mcp_client():
    """Test the real MCP client implementation"""
    print("=== Testing Real MCP Client ===\n")
    
    client = MCPClient()
    
    try:
        # Test connection
        print("1. Connecting to MCP server...")
        connected = await client.connect()
        print(f"   Connection: {'Success' if connected else 'Failed'}")
        
        if not connected:
            return
        
        # Test health check
        print("\n2. Health check...")
        health = await client.health_check()
        print(f"   Status: {health['status']}")
        print(f"   Tools available: {health['tools_available']}")
        print(f"   Latency: {health.get('latency_ms', 0):.2f}ms")
        
        # Test tool listing
        print("\n3. Listing tools...")
        tools = await client.list_tools()
        for tool in tools:
            print(f"   - {tool['name']}: {tool['description']}")
        
        # Test tool execution
        if tools:
            print("\n4. Testing tool execution...")
            test_tool = tools[0]
            
            if test_tool['name'] == 'code_analyzer':
                result = await client.call_tool('code_analyzer', {
                    'code': 'def hello(): print("Hello, World!")',
                    'language': 'python'
                })
                
                print(f"   Tool: {result['tool']}")
                print(f"   Status: {result['status']}")
                print(f"   Latency: {result.get('latency_ms', 0):.2f}ms")
                
                if result['status'] == 'success':
                    print("   Result: Tool executed successfully!")
        
        print("\n✅ Real MCP Client Test Complete!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
    
    finally:
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(test_real_mcp_client())
