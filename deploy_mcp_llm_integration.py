#!/usr/bin/env python3
"""
Production MCP-LLM Integration with State Continuity Fabric
===========================================================

This demonstrates how to deploy our unified architecture within
LLM/ML processes, using mcp-use for protocol and adding our
unique State Continuity Fabric for competitive advantage.
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime

# Production logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class MCPLLMIntegration:
    """
    Production integration of MCP with LLM processes.

    Key differentiators:
    1. Uses mcp-use for standard MCP protocol (commodity)
    2. Adds State Continuity Fabric (our innovation)
    3. Implements Edge-Cloud Continuum (our innovation)
    4. Provides cross-device/app context (our innovation)
    """

    def __init__(self):
        self.mcp_client = None
        self.state_fabric = None
        self.execution_metrics = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "avg_latency": 0,
        }

    async def initialize(self) -> bool:
        """Initialize MCP and State Continuity Fabric"""
        try:
            # Import mcp-use
            from mcp_use import MCPClient

            # Connect to MCP server
            self.mcp_client = MCPClient()
            # Note: mcp-use requires different initialization
            # We'll use their session-based approach

            # Verify connection
            tools = await self.mcp_client.list_tools()
            logger.info(f"Connected to MCP with {len(tools)} tools available")

            # Initialize State Fabric (our unique value)
            try:
                from fabric.integrated_mcp_fabric import MCPStateFabric

                self.state_fabric = MCPStateFabric("llm_fabric")
                await self.state_fabric.initialize(
                    [{"name": "local", "url": "http://localhost:8080"}]
                )
                logger.info("State Continuity Fabric initialized")
            except Exception as e:
                logger.warning(f"State Fabric not available: {e}")
                # Continue without state fabric - graceful degradation

            return True

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False

    async def process_llm_request(
        self, request: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process LLM request with MCP tools and state continuity.

        This is where we add value:
        - Use MCP tools to enhance LLM capabilities
        - Track state across requests for continuity
        - Enable cross-device context awareness
        """
        start_time = asyncio.get_event_loop().time()

        try:
            # Extract intent from LLM request
            intent = request.get("intent", "unknown")
            parameters = request.get("parameters", {})

            # Determine which MCP tool to use
            tool_mapping = {
                "analyze_code": "code_analyzer",
                "validate_protocol": "protocol_validator",
                "correct_errors": "self_corrector",
            }

            tool_name = tool_mapping.get(intent)
            if not tool_name:
                return {"success": False, "error": f"Unknown intent: {intent}"}

            # Execute via MCP with state tracking
            if self.state_fabric:
                # Use our enhanced execution with state continuity
                result = await self.state_fabric.execute_with_context(
                    server_name="local",
                    tool_name=tool_name,
                    params=parameters,
                    context=context
                    or {
                        "device_id": "llm_server",
                        "app_id": "llm_integration",
                        "session_id": request.get("session_id", "default"),
                    },
                )
            else:
                # Fallback to basic MCP execution
                result = await self.mcp_client.call_tool(tool_name, parameters)
                result = {"success": True, "result": result, "tool": tool_name}

            # Update metrics
            latency = asyncio.get_event_loop().time() - start_time
            self._update_metrics(success=True, latency=latency)

            # Add execution metadata
            result["metadata"] = {
                "latency_ms": latency * 1000,
                "timestamp": datetime.utcnow().isoformat(),
                "has_state_continuity": self.state_fabric is not None,
            }

            return result

        except Exception as e:
            logger.error(f"Request processing failed: {e}")
            self._update_metrics(success=False, latency=0)
            return {
                "success": False,
                "error": str(e),
                "tool": tool_name if "tool_name" in locals() else "unknown",
            }

    def _update_metrics(self, success: bool, latency: float):
        """Update execution metrics"""
        self.execution_metrics["total_calls"] += 1
        if success:
            self.execution_metrics["successful_calls"] += 1
            # Update rolling average latency
            current_avg = self.execution_metrics["avg_latency"]
            total_calls = self.execution_metrics["total_calls"]
            self.execution_metrics["avg_latency"] = (
                current_avg * (total_calls - 1) + latency
            ) / total_calls
        else:
            self.execution_metrics["failed_calls"] += 1

    async def get_cross_device_context(
            self, session_id: str) -> Dict[str, Any]:
        """
        Get context from other devices/sessions - our unique capability.

        This demonstrates how we go beyond mcp-use by providing
        cross-device state continuity.
        """
        if not self.state_fabric:
            return {
                "available": False,
                "reason": "State fabric not initialized",
            }

        try:
            # Get execution history across all devices
            history = await self.state_fabric.get_execution_history()

            # Filter by session if provided
            if session_id:
                history = [h for h in history if h.get(
                    "session_id") == session_id]

            return {
                "available": True,
                "history": history,
                "device_count": (
                    len(self.state_fabric.state_fabric.device_registry)
                    if self.state_fabric.state_fabric
                    else 0
                ),
            }

        except Exception as e:
            logger.error(f"Failed to get cross-device context: {e}")
            return {"available": False, "error": str(e)}

    def get_metrics(self) -> Dict[str, Any]:
        """Get execution metrics"""
        return {
            **self.execution_metrics,
            "success_rate": (
                self.execution_metrics["successful_calls"]
                / max(self.execution_metrics["total_calls"], 1)
            ),
            "avg_latency_ms": self.execution_metrics["avg_latency"] * 1000,
        }

    async def close(self):
        """Clean shutdown"""
        if self.mcp_client:
            await self.mcp_client.close()
        if self.state_fabric:
            await self.state_fabric.close()


# Production deployment example
async def deploy_llm_integration():
    """
    Deploy MCP-LLM integration in production.

    This shows:
    1. Real integration with mcp-use
    2. Our State Continuity value-add
    3. Production error handling
    4. Metrics and monitoring
    """

    integration = MCPLLMIntegration()

    # Initialize
    initialized = await integration.initialize()
    if not initialized:
        logger.error("Failed to initialize MCP-LLM integration")
        return

    # Simulate LLM requests
    test_requests = [
        {
            "intent": "analyze_code",
            "parameters": {
                "code": 'def hello(): print("world")',
                "language": "python",
            },
            "session_id": "session_123",
        },
        {
            "intent": "validate_protocol",
            "parameters": {
                "protocol": "test_protocol",
                "data": {"key": "value"},
            },
            "session_id": "session_123",
        },
        {
            "intent": "correct_errors",
            "parameters": {
                "errors": ["undefined variable x"],
                "context": "python function",
            },
            "session_id": "session_456",
        },
    ]

    # Process requests
    for request in test_requests:
        logger.info(f"Processing request: {request['intent']}")
        result = await integration.process_llm_request(request)
        logger.info(f"Result: {json.dumps(result, indent=2)}")

    # Show cross-device context capability
    context = await integration.get_cross_device_context("session_123")
    logger.info(f"Cross-device context: {json.dumps(context, indent=2)}")

    # Show metrics
    metrics = integration.get_metrics()
    logger.info(f"Execution metrics: {json.dumps(metrics, indent=2)}")

    # Clean shutdown
    await integration.close()


# Competitive analysis integration
async def analyze_competitive_advantage():
    """
    Analyze how our approach differs from plain mcp-use.
    """

    logger.info("=== Competitive Analysis ===")
    logger.info(
        "mcp-use provides: Basic MCP protocol, tool discovery, execution")
    logger.info("We add:")
    logger.info(
        "1. State Continuity Fabric - Track state across devices/sessions")
    logger.info("2. Cross-device context - Access history from any device")
    logger.info("3. Execution metrics - Production monitoring")
    logger.info("4. Privacy-aware state management - Filter sensitive data")
    logger.info(
        "5. Vector clock synchronization - Resolve distributed conflicts")

    # Show concrete example
    integration = MCPLLMIntegration()
    if await integration.initialize():
        # Execute same tool from different "devices"
        for device in ["laptop", "phone", "tablet"]:
            await integration.process_llm_request(
                {
                    "intent": "analyze_code",
                    "parameters": {"code": f"# From {device}"},
                },
                {"device_id": device, "app_id": "demo"},
            )

        # Show we can see all executions
        context = await integration.get_cross_device_context(None)
        logger.info(
            f"Tracked {len(context.get('history', []))} executions across devices"
        )

        await integration.close()


if __name__ == "__main__":
    logger.info("Starting MCP-LLM Integration Deployment")

    # Run deployment
    asyncio.run(deploy_llm_integration())

    # Run competitive analysis
    asyncio.run(analyze_competitive_advantage())
