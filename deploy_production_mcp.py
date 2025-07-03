#!/usr/bin/env python3
"""
Production MCP Deployment with State Continuity Extensions
==========================================================

This demonstrates the strategic approach:
1. Use mcp-use for core MCP functionality (commodity)
2. Add our State Continuity Fabric as a layer on top
3. Contribute improvements back to mcp-use
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

# Configure production logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ProductionMCPDeployment:
    """
    Production deployment combining:
    - mcp-use for MCP protocol (proven, community-tested)
    - Our State Continuity innovations
    - Unified transport layer (MCP + A2A + Mojo)
    """

    def __init__(self):
        self.mcp_agent = None
        self.state_fabric = None
        self.unified_transport = None

    async def initialize(self):
        """Initialize all components with production error handling"""
        try:
            # Step 1: Initialize mcp-use properly
            from mcp_use import MCPAgent, MCPClient
            from langchain_openai import ChatOpenAI

            # Create MCP client with proper configuration
            config = {
                "mcpServers": {
                    "local": {
                        "command": "node",
                        "args": ["/app/mcp_server/main.js"],
                        "env": {"NODE_ENV": "production"},
                    },
                    "github": {
                        "command": "npx",
                        "args": ["@github/mcp-server-github"],
                    },
                }
            }

            # Use MCPClient.from_dict as shown in mcp-use docs
            mcp_client = MCPClient.from_dict(config)

            # Create LLM
            llm = ChatOpenAI(model="gpt-4o", temperature=0.1)

            # Create agent with proper parameters
            self.mcp_agent = MCPAgent(
                llm=llm,
                client=mcp_client,
                max_steps=30,
                use_server_manager=True,  # Enable intelligent server selection
                verbose=True,
            )

            logger.info("MCP Agent initialized successfully")

            # Step 2: Initialize our State Continuity Fabric
            try:
                from fabric.state_continuity_core import StateContinuityFabric

                self.state_fabric = StateContinuityFabric("production")
                await self.state_fabric.initialize()
                logger.info("State Continuity Fabric initialized")
            except ImportError:
                logger.warning(
                    "State Continuity Fabric not available - continuing without it"
                )

            # Step 3: Initialize Unified Transport Layer
            try:
                from agents.unified_transport_layer import (
                    UnifiedTransportLayer,
                )

                self.unified_transport = UnifiedTransportLayer()
                await self.unified_transport.initialize()
                logger.info("Unified Transport Layer initialized")
            except ImportError:
                logger.warning(
                    "Unified Transport not available - using standard transport"
                )

            return True

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False

    async def execute_with_state_continuity(
        self, query: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute query with our unique state continuity features.
        This is where we add value beyond mcp-use.
        """
        start_time = asyncio.get_event_loop().time()

        try:
            # Capture pre-execution state
            if self.state_fabric:
                pre_state = await self.state_fabric.capture_context(
                    device_id=context.get("device_id", "server"),
                    app_id=context.get("app_id", "production"),
                    context={
                        "query": query,
                        "timestamp": datetime.utcnow().isoformat(),
                        "session_id": context.get("session_id"),
                    },
                )
                logger.info(f"Captured pre-execution state: {pre_state.id}")

            # Execute using mcp-use agent with streaming
            result_chunks = []
            async for chunk in self.mcp_agent.astream(query):
                result_chunks.append(chunk)
                # Process streaming chunks in real-time
                if "messages" in chunk:
                    logger.info(f"Agent: {chunk['messages']}")

            # Get final result
            final_result = result_chunks[-1] if result_chunks else None

            # Capture post-execution state
            if self.state_fabric and final_result:
                post_state = await self.state_fabric.capture_context(
                    device_id=context.get("device_id", "server"),
                    app_id=context.get("app_id", "production"),
                    context={
                        "query": query,
                        "result": final_result.get("output"),
                        "success": True,
                        "execution_time": asyncio.get_event_loop().time() - start_time,
                    },
                )
                logger.info(f"Captured post-execution state: {post_state.id}")

            return {
                "success": True,
                "result": final_result.get("output") if final_result else None,
                "execution_time_ms": (asyncio.get_event_loop().time() - start_time)
                * 1000,
                "has_state_continuity": self.state_fabric is not None,
                "transport_type": ("unified" if self.unified_transport else "standard"),
            }

        except Exception as e:
            logger.error(f"Execution failed: {e}")

            # Capture error state
            if self.state_fabric:
                await self.state_fabric.capture_context(
                    device_id=context.get("device_id", "server"),
                    app_id=context.get("app_id", "production"),
                    context={
                        "query": query,
                        "error": str(e),
                        "success": False,
                    },
                )

            return {
                "success": False,
                "error": str(e),
                "execution_time_ms": (asyncio.get_event_loop().time() - start_time)
                * 1000,
            }

    async def demonstrate_unique_capabilities(self):
        """
        Show what we add beyond mcp-use:
        1. State continuity across devices
        2. Unified transport with microsecond latency
        3. Privacy-aware processing
        """
        logger.info("=== Demonstrating Our Unique Capabilities ===")

        # 1. Cross-device state synchronization
        if self.state_fabric:
            # Register multiple devices
            devices = ["laptop", "phone", "tablet"]
            for device in devices:
                self.state_fabric.register_device(device, {"type": device})

            # Execute from different devices
            for device in devices:
                await self.execute_with_state_continuity(
                    f"Test from {device}",
                    {"device_id": device, "app_id": "demo"},
                )

            # Sync states between devices
            await self.state_fabric.sync_devices("laptop", "phone")
            logger.info("✓ Cross-device state synchronization working")

        # 2. Unified transport performance
        if self.unified_transport:
            # Test different transport strategies
            test_message = {"data": "x" * 1000}  # 1KB message

            # Zero-copy for same process
            result = await self.unified_transport.send(
                test_message,
                source="agent1",
                target="agent2",
                same_process=True,
            )
            logger.info(f"✓ Zero-copy transport: {result['latency_us']}μs")

            # Mojo pipes for cross-process
            result = await self.unified_transport.send(
                test_message,
                source="agent1",
                target="agent3",
                same_process=False,
            )
            logger.info(f"✓ Mojo pipe transport: {result['latency_us']}μs")

        # 3. Privacy-aware processing
        if self.state_fabric:
            # Set privacy rules
            self.state_fabric.privacy_rules["no-sync"].append("api_key")
            self.state_fabric.privacy_rules["encrypted"].append("user_data")

            # Test privacy filtering
            sensitive_context = {
                "api_key": "secret123",
                "user_data": "personal info",
                "public_data": "can be shared",
            }

            filtered = self.state_fabric._apply_privacy_filters(sensitive_context)
            assert "api_key" not in filtered
            assert "<encrypted>" in filtered.get("user_data", "")
            logger.info("✓ Privacy-aware filtering working")

    async def close(self):
        """Clean shutdown of all components"""
        if self.mcp_agent and hasattr(self.mcp_agent, "client"):
            await self.mcp_agent.client.close_all_sessions()

        logger.info("All components shut down cleanly")


# Production deployment example
async def deploy_production():
    """Deploy production MCP with our innovations"""

    deployment = ProductionMCPDeployment()

    try:
        # Initialize
        success = await deployment.initialize()
        if not success:
            logger.error("Failed to initialize deployment")
            return

        # Run production workload
        queries = [
            "Analyze the repository structure and suggest improvements",
            "Create a comprehensive test suite for the MCP integration",
            "Generate documentation for the State Continuity Fabric",
        ]

        for query in queries:
            logger.info(f"\nProcessing: {query}")
            result = await deployment.execute_with_state_continuity(
                query,
                {"device_id": "production_server", "session_id": "prod_001"},
            )
            logger.info(f"Result: {result}")

        # Demonstrate unique capabilities
        await deployment.demonstrate_unique_capabilities()

    finally:
        await deployment.close()


# Contribution strategy implementation
async def implement_contribution_strategy():
    """
    Show how to contribute our innovations back to mcp-use
    """
    logger.info("\n=== Contribution Strategy ===")
    logger.info("1. Fork mcp-use repository")
    logger.info("2. Add StateContinuityMixin class to mcp_use/mixins/")
    logger.info("3. Extend MCPAgent with state continuity methods")
    logger.info("4. Add tests for cross-device synchronization")
    logger.info("5. Submit PR with clear value proposition")
    logger.info("6. Engage with community for feedback")

    # Example contribution code structure
    contribution_example = '''
    # In mcp_use/mixins/state_continuity.py
    class StateContinuityMixin:
        """Adds state continuity capabilities to MCPAgent"""
        
        async def capture_state(self, context: Dict[str, Any]):
            """Capture execution state for continuity"""
            # Our implementation
        
        async def sync_across_devices(self, source: str, target: str):
            """Synchronize state between devices"""
            # Our implementation
    
    # In mcp_use/agents/mcpagent.py
    class MCPAgent(BaseAgent, StateContinuityMixin):
        # Enhanced with our capabilities
    '''

    logger.info(f"Contribution structure:\n{contribution_example}")


if __name__ == "__main__":
    logger.info("Starting Production MCP Deployment")

    # Run production deployment
    asyncio.run(deploy_production())

    # Show contribution strategy
    asyncio.run(implement_contribution_strategy())
