"""
Production-Grade MCP + State Continuity Fabric Integration
==========================================================

This integrates mcp-use (for MCP protocol) with our unique
State Continuity Fabric for cross-device/app state management.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from mcp_use import MCPClient, create_client
from mcp_use.tools import ToolExecutor

# Configure production logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class MCPStateFabric:
    """
    Production-ready integration of MCP with State Continuity Fabric.
    Uses mcp-use for protocol, adds our unique state continuity layer.
    """

    def __init__(self, fabric_id: str):
        self.fabric_id = fabric_id
        self.mcp_clients: Dict[str, MCPClient] = {}
        self.tool_executor = ToolExecutor()
        self._initialized = False

        # Import our State Continuity Fabric
        try:
            from fabric.state_continuity_core import StateContinuityFabric

            self.state_fabric = StateContinuityFabric(fabric_id)
        except ImportError:
            logger.error("State Continuity Fabric not found. Creating minimal version.")
            self.state_fabric = None

    async def initialize(self, mcp_servers: List[Dict[str, Any]]) -> bool:
        """
        Initialize with multiple MCP servers using mcp-use.

        Args:
            mcp_servers: List of server configs with 'name' and 'url'

        Returns:
            bool: True if at least one server connected successfully
        """
        connected_count = 0

        for server_config in mcp_servers:
            try:
                name = server_config["name"]
                url = server_config["url"]

                # Use mcp-use to create client
                client = await create_client(
                    server_url=url, client_name=f"{self.fabric_id}_{name}"
                )

                # Verify connection by listing tools
                tools = await client.list_tools()
                logger.info(f"Connected to {name} at {url} with {len(tools)} tools")

                self.mcp_clients[name] = client
                connected_count += 1

            except Exception as e:
                logger.error(f"Failed to connect to {server_config}: {e}")

        self._initialized = connected_count > 0

        # Initialize state fabric if available
        if self.state_fabric and connected_count > 0:
            await self.state_fabric.initialize()

        return self._initialized

    async def discover_capabilities(self) -> Dict[str, List[str]]:
        """
        Discover all available tools across connected MCP servers.

        Returns:
            Dict mapping server names to their tool lists
        """
        if not self._initialized:
            raise RuntimeError("Fabric not initialized. Call initialize() first.")

        capabilities = {}

        for server_name, client in self.mcp_clients.items():
            try:
                tools = await client.list_tools()
                capabilities[server_name] = [tool.name for tool in tools]
                logger.info(
                    f"{server_name} capabilities: {
                        capabilities[server_name]}"
                )
            except Exception as e:
                logger.error(f"Failed to get capabilities from {server_name}: {e}")
                capabilities[server_name] = []

        return capabilities

    async def execute_with_context(
        self,
        server_name: str,
        tool_name: str,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute MCP tool with state continuity context.

        This is where we add value beyond basic MCP:
        - Capture execution context
        - Maintain state continuity
        - Enable cross-device replay
        """
        if server_name not in self.mcp_clients:
            raise ValueError(f"No client connected for server: {server_name}")

        client = self.mcp_clients[server_name]

        # Capture pre-execution state if fabric available
        if self.state_fabric and context:
            pre_state = await self.state_fabric.capture_context(
                device_id=context.get("device_id", "unknown"),
                app_id=context.get("app_id", "mcp_fabric"),
                context={
                    "tool": tool_name,
                    "params": params,
                    "timestamp": asyncio.get_event_loop().time(),
                },
            )
            logger.info(f"Captured pre-execution state: {pre_state.id}")

        # Execute tool using mcp-use
        try:
            result = await client.call_tool(tool_name, params)

            # Capture post-execution state
            if self.state_fabric and context:
                post_state = await self.state_fabric.capture_context(
                    device_id=context.get("device_id", "unknown"),
                    app_id=context.get("app_id", "mcp_fabric"),
                    context={
                        "tool": tool_name,
                        "result": result,
                        "success": True,
                        "timestamp": asyncio.get_event_loop().time(),
                    },
                )
                logger.info(f"Captured post-execution state: {post_state.id}")

            return {
                "success": True,
                "result": result,
                "server": server_name,
                "tool": tool_name,
            }

        except Exception as e:
            logger.error(f"Tool execution failed: {e}")

            # Capture error state
            if self.state_fabric and context:
                await self.state_fabric.capture_context(
                    device_id=context.get("device_id", "unknown"),
                    app_id=context.get("app_id", "mcp_fabric"),
                    context={
                        "tool": tool_name,
                        "error": str(e),
                        "success": False,
                    },
                )

            return {
                "success": False,
                "error": str(e),
                "server": server_name,
                "tool": tool_name,
            }

    async def get_execution_history(
        self, device_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get execution history with state continuity information.

        This demonstrates our unique value: tracking execution across devices.
        """
        if not self.state_fabric:
            return []

        history = []

        # Get all devices or specific device
        devices = (
            [device_id] if device_id else list(self.state_fabric.device_registry.keys())
        )

        for dev_id in devices:
            if dev_id in self.state_fabric.engines:
                engine = self.state_fabric.engines[dev_id]
                for state_id, state in engine.states.items():
                    if "tool" in state.data:
                        history.append(
                            {
                                "device": dev_id,
                                "timestamp": state.timestamp,
                                "tool": state.data.get("tool"),
                                "success": state.data.get("success", False),
                                "state_id": state_id,
                            }
                        )

        # Sort by timestamp
        history.sort(key=lambda x: x["timestamp"], reverse=True)
        return history

    async def sync_execution_state(
        self, source_device: str, target_device: str
    ) -> bool:
        """
        Sync execution state between devices - our unique capability.
        """
        if not self.state_fabric:
            logger.error("State fabric not available for sync")
            return False

        try:
            merged_state = await self.state_fabric.sync_devices(
                source_device, target_device
            )
            logger.info(
                f"Synced state from {source_device} to {target_device}: {merged_state.id}"
            )

        """Clean shutdown of all connections"""
        for server_name, client in self.mcp_clients.items():
            try:
                await client.close()
                logger.info(f"Closed connection to {server_name}")
            except Exception as e:
                logger.error(f"Error closing {server_name}: {e}")

        self.mcp_clients.clear()
        self._initialized = False


# Production-ready example with real MCP servers
async def production_example():
    """
    Production example showing:
    1. Using mcp-use for MCP protocol
    2. Adding our State Continuity value
    3. Real error handling and logging
    """

    fabric = MCPStateFabric("production_fabric")

    # Configure real MCP servers
    servers = [
        {"name": "local", "url": "http://localhost:8080"},
        # Add more servers as needed
        # {
        #     'name': 'github',
        #     'url': 'http://localhost:3000'  # GitHub MCP server
        # }
    ]

    try:
        # Initialize with production error handling
        initialized = await fabric.initialize(servers)
        if not initialized:
            logger.error("No MCP servers available. Cannot proceed.")
            return

        # Discover what we can actually do
        capabilities = await fabric.discover_capabilities()
        logger.info(f"Available capabilities: {capabilities}")

        # Execute a tool with context tracking
        if "local" in capabilities and capabilities["local"]:
            # Use first available tool for demo
            tool_name = capabilities["local"][0]

            result = await fabric.execute_with_context(
                server_name="local",
                tool_name=tool_name,
                params={},  # Tool-specific params
                context={
                    "device_id": "macbook_pro",
                    "app_id": "production_demo",
                },
            )

            logger.info(f"Execution result: {result}")

            # Show execution history - our unique value
            history = await fabric.get_execution_history()
            logger.info(f"Execution history: {history}")

    except Exception as e:
        logger.error(f"Production example failed: {e}")

    finally:
        # Always clean up
        await fabric.close()


if __name__ == "__main__":
    # Run production example
    asyncio.run(production_example())
