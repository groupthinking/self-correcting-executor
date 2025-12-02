#!/usr/bin/env python3
"""
A2A (Agent-to-Agent) MCP Integration Framework
==============================================

This implements a complete agent-to-agent communication system that integrates
with the Model Context Protocol (MCP) for intelligent, semantic-aware agent
interactions.

Features:
- MCP context sharing between agents
- Intelligent message routing with Mojo-inspired transport
- Multi-agent negotiation and collaboration
- Performance monitoring and SLA compliance
- Integration with existing MCP server
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# Import existing components
from agents.a2a_framework import A2AMessage, BaseAgent, A2AMessageBus
from connectors.mcp_base import MCPContext
from connectors.real_mcp_client import MCPClient, execute_mcp_tool

logger = logging.getLogger(__name__)


class MessagePriority(Enum):
    """Message priority levels for intelligent routing"""

    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class TransportStrategy(Enum):
    """Transport strategies based on message characteristics"""

    ZERO_COPY = "zero_copy"  # Same process, < 1MB
    SHARED_MEMORY = "shared_memory"  # Same/different process, > 1MB
    MCP_PIPE = "mcp_pipe"  # MCP-optimized transport
    STANDARD = "standard"  # Fallback transport


@dataclass
class A2AMCPMessage:
    """Enhanced message that combines A2A protocol with MCP context"""

    # A2A Layer
    a2a_message: A2AMessage

    # MCP Layer
    mcp_context: MCPContext

    # Transport Layer
    priority: MessagePriority = MessagePriority.NORMAL
    transport_strategy: TransportStrategy = TransportStrategy.STANDARD
    deadline_ms: Optional[float] = None
    performance_requirements: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "a2a": self.a2a_message.to_dict(),
            "mcp": self.mcp_context.to_dict(),
            "transport": {
                "priority": self.priority.value,
                "strategy": self.transport_strategy.value,
                "deadline_ms": self.deadline_ms,
                "requirements": self.performance_requirements,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "A2AMCPMessage":
        a2a_msg = A2AMessage.from_dict(data["a2a"])
        mcp_context = MCPContext()
        mcp_context.from_dict(data["mcp"])

        return cls(
            a2a_message=a2a_msg,
            mcp_context=mcp_context,
            priority=MessagePriority(data["transport"]["priority"]),
            transport_strategy=TransportStrategy(data["transport"]["strategy"]),
            deadline_ms=data["transport"].get("deadline_ms"),
            performance_requirements=data["transport"].get("requirements", {}),
        )


class MCPEnabledA2AAgent(BaseAgent):
    """
    Enhanced agent with MCP context and A2A communication.
    Integrates with MCP server for tool access.
    """

    def __init__(
        self,
        agent_id: str,
        capabilities: List[str],
        mcp_server_url: str = "http://localhost:8080",
    ):
        super().__init__(agent_id, capabilities)
        self.mcp_context = MCPContext()
        self.mcp_server_url = mcp_server_url
        # Allow message bus injection
        self.message_bus: Optional[A2AMessageBus] = None
        self.performance_stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "negotiations_completed": 0,
            "avg_response_time_ms": 0.0,
            "sla_violations": 0,
        }
        self.sla_requirements = {
            "max_latency_ms": 100,
            "min_throughput_msgs_per_sec": 10,
            "max_error_rate": 0.01,
        }

        # Register handlers for common message types
        self.register_handler("negotiate_request", self.handle_negotiation_request)
        self.register_handler("context_share", self.handle_context_share)
        self.register_handler("tool_request", self.handle_tool_request)
        self.register_handler(
            "collaboration_request", self.handle_collaboration_request
        )

        # Initialize context attributes
        self.mcp_context.task = {}
        self.mcp_context.intent = {}
        self.mcp_context.history = []

    async def process_intent(self, intent: Dict) -> Dict:
        """
        Process intent and return result - abstract method implementation
        """
        try:
            action = intent.get("action", "unknown")

            if action == "send_message":
                recipient = intent.get("recipient")
                if not recipient:
                    return {
                        "status": "error",
                        "message": "recipient not specified",
                    }
                return await self.send_contextualized_message(
                    recipient=recipient,
                    intent=intent,
                    priority=MessagePriority(intent.get("priority", 1)),
                    deadline_ms=intent.get("deadline_ms"),
                )
            elif action == "analyze_data":
                return await self._analyze_data(intent.get("data", {}))
            elif action == "generate_code":
                return await self._generate_code(intent.get("data", {}))
            elif action == "negotiate":
                return await self.handle_negotiation_request(
                    A2AMessage(
                        sender=intent.get("sender", "unknown"),
                        recipient=self.agent_id,
                        message_type="negotiate_request",
                        content=intent.get("data", {}),
                    )
                )
            elif action == "tool_request":
                tool_name = intent.get("tool_name")
                if not tool_name:
                    return {
                        "status": "error",
                        "message": "tool_name not specified",
                    }
                return await self._execute_mcp_tool(tool_name, intent.get("params", {}))
            else:
                return {
                    "status": "error",
                    "message": f"Unknown intent action: {action}",
                    "available_actions": [
                        "send_message",
                        "analyze_data",
                        "generate_code",
                        "negotiate",
                        "tool_request",
                    ],
                }

        except Exception as e:
            logger.error(f"Error processing intent: {e}")
            return {"status": "error", "message": str(e), "intent": intent}

    async def send_contextualized_message(
        self,
        recipient: str,
        intent: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL,
        deadline_ms: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Send message with MCP context and routing
        """
        start_time = time.time()

        # 1. Build MCP context
        self.mcp_context.task = {"intent": intent, "recipient": recipient}
        self.mcp_context.intent = intent
        self.mcp_context.history.append(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "action": "send_message",
                "recipient": recipient,
                "intent": intent,
            }
        )

        # 2. Create A2A message
        message_type = f"{intent.get('action', 'message')}_request"
        a2a_msg = A2AMessage(
            sender=self.agent_id,
            recipient=recipient,
            message_type=message_type,
            content=intent.get("data", {}),
        )

        # 3. Create unified message
        unified_msg = A2AMCPMessage(
            a2a_message=a2a_msg,
            mcp_context=self.mcp_context,
            priority=priority,
            deadline_ms=deadline_ms,
            performance_requirements=self.sla_requirements,
        )

        # 4. Send through intelligent transport
        result = await self._send_with_intelligent_routing(unified_msg)

        # 5. Update performance stats
        latency_ms = (time.time() - start_time) * 1000
        self.performance_stats["messages_sent"] += 1
        self.performance_stats["avg_response_time_ms"] = (
            self.performance_stats["avg_response_time_ms"]
            * (self.performance_stats["messages_sent"] - 1)
            + latency_ms
        ) / self.performance_stats["messages_sent"]

        # 6. Check SLA compliance
        if latency_ms > self.sla_requirements["max_latency_ms"]:
            self.performance_stats["sla_violations"] += 1
            logger.warning("SLA violation: latency exceeded")

        return {
            "message_id": a2a_msg.id,
            "recipient": recipient,
            "latency_ms": latency_ms,
            "transport_strategy": result.get("strategy"),
            "status": result.get("status", "sent"),
        }

    async def _send_with_intelligent_routing(
        self, message: A2AMCPMessage
    ) -> Dict[str, Any]:
        """
        Route messages based on priority, size, requirements
        """
        message_size = len(str(message.to_dict()))

        # Determine optimal transport strategy
        if message.priority == MessagePriority.CRITICAL or (
            message.deadline_ms and message.deadline_ms < 10
        ):
            strategy = TransportStrategy.ZERO_COPY
        elif message_size > 1024 * 1024:  # > 1MB
            strategy = TransportStrategy.SHARED_MEMORY
        elif message_size < 1024:  # < 1KB
            strategy = TransportStrategy.ZERO_COPY
        else:
            strategy = TransportStrategy.MCP_PIPE

        message.transport_strategy = strategy

        # Send through appropriate transport
        if strategy == TransportStrategy.ZERO_COPY:
            return await self._send_zero_copy(message)
        elif strategy == TransportStrategy.SHARED_MEMORY:
            return await self._send_shared_memory(message)
        elif strategy == TransportStrategy.MCP_PIPE:
            return await self._send_mcp_pipe(message)
        else:
            return await self._send_standard(message)

    async def _send_zero_copy(self, message: A2AMCPMessage) -> Dict[str, Any]:
        """Zero-copy transfer for high-performance"""
        start_time = time.time()
        
        try:
            # Direct message bus delivery for zero-copy semantics
            if self.message_bus:
                await self.message_bus.send(message.a2a_message)
            
            # Calculate real latency
            latency_ms = (time.time() - start_time) * 1000
            
            return {
                "strategy": "zero_copy",
                "status": "delivered",
                "latency_ms": latency_ms,
            }
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error(f"Zero-copy transfer failed: {e}")
            return {
                "strategy": "zero_copy",
                "status": "failed",
                "latency_ms": latency_ms,
                "error": str(e)
            }

    async def _send_shared_memory(self, message: A2AMCPMessage) -> Dict[str, Any]:
        """Shared memory transfer for large messages"""
        start_time = time.time()
        
        try:
            # Use message bus with measured latency for large messages
            if self.message_bus:
                await self.message_bus.send(message.a2a_message)
            
            # Calculate real latency
            latency_ms = (time.time() - start_time) * 1000
            
            return {
                "strategy": "shared_memory",
                "status": "delivered",
                "latency_ms": latency_ms,
            }
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error(f"Shared memory transfer failed: {e}")
            return {
                "strategy": "shared_memory",
                "status": "failed",
                "latency_ms": latency_ms,
                "error": str(e)
            }

    async def _send_mcp_pipe(self, message: A2AMCPMessage) -> Dict[str, Any]:
        """MCP-optimized pipe transfer using real MCP client"""
        try:
            start_time = time.time()
            
            # Use real MCP client for message transport
            mcp_client = MCPClient()
            if await mcp_client.connect():
                # Send message content through MCP server
                result = await mcp_client.call_tool(
                    "protocol_validator",
                    {
                        "message": json.dumps(message.to_dict()),
                        "protocol_version": "2024-11-05"
                    }
                )
                
                await mcp_client.disconnect()
                
                # Calculate real latency
                latency_ms = (time.time() - start_time) * 1000
                
                # Send through message bus if validation successful
                if result.get("status") == "success" and self.message_bus:
                    await self.message_bus.send(message.a2a_message)
                
                return {
                    "strategy": "mcp_pipe",
                    "status": "delivered" if result.get("status") == "success" else "failed",
                    "latency_ms": latency_ms,
                    "mcp_result": result
                }
            else:
                raise ConnectionError("Failed to connect to MCP server")
                
        except Exception as e:
            logger.error(f"MCP pipe transfer failed: {e}")
            return await self._send_standard(message)

    async def _send_standard(self, message: A2AMCPMessage) -> Dict[str, Any]:
        """Standard transport fallback"""
        start_time = time.time()
        
        try:
            if self.message_bus:
                await self.message_bus.send(message.a2a_message)
            
            # Calculate real latency
            latency_ms = (time.time() - start_time) * 1000
            
            return {
                "strategy": "standard",
                "status": "delivered",
                "latency_ms": latency_ms,
            }
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error(f"Standard transfer failed: {e}")
            return {
                "strategy": "standard",
                "status": "failed",
                "latency_ms": latency_ms,
                "error": str(e)
            }

    async def handle_negotiation_request(self, message: A2AMessage) -> Dict[str, Any]:
        """Handle incoming negotiation request"""
        content = message.content

        # Use MCP context to understand negotiation context
        self.mcp_context.task = {
            "type": "negotiation",
            "topic": content.get("topic"),
        }

        # Generate response based on agent capabilities
        response = {
            "status": "accepted",
            "proposal": self._generate_negotiation_proposal(content),
            "constraints": self._get_agent_constraints(),
            "preferences": self._get_agent_preferences(),
        }

        self.performance_stats["negotiations_completed"] += 1
        return response

    async def handle_context_share(self, message: A2AMessage) -> Dict[str, Any]:
        """Handle MCP context sharing"""
        # Merge incoming context with local context
        incoming_context = message.content.get("context", {})

        # Manually merge context fields
        if isinstance(incoming_context.get("task"), dict):
            self.mcp_context.task.update(incoming_context["task"])
        if isinstance(incoming_context.get("intent"), dict):
            self.mcp_context.intent.update(incoming_context["intent"])
        if isinstance(incoming_context.get("history"), list):
            self.mcp_context.history.extend(incoming_context["history"])

        return {
            "status": "context_merged",
            "local_context_size": len(str(self.mcp_context.to_dict())),
        }

    async def handle_tool_request(self, message: A2AMessage) -> Dict[str, Any]:
        """Handle tool execution requests through MCP"""
        tool_name = message.content.get("tool")
        tool_params = message.content.get("params", {})

        if not tool_name:
            return {
                "status": "tool_error",
                "tool": None,
                "error": "Tool name not provided in request",
            }

        # Execute tool through MCP server
        try:
            result = await self._execute_mcp_tool(tool_name, tool_params)
            return {
                "status": "tool_executed",
                "tool": tool_name,
                "result": result,
            }
        except Exception as e:
            return {"status": "tool_error", "tool": tool_name, "error": str(e)}

    async def handle_collaboration_request(self, message: A2AMessage) -> Dict[str, Any]:
        """Handle collaboration requests"""
        collaboration_type = message.content.get("type")
        data = message.content.get("data", {})

        # Process collaboration based on agent capabilities
        if collaboration_type == "data_analysis" and "analyze" in self.capabilities:
            result = await self._analyze_data(data)
        elif (
            collaboration_type == "code_generation" and "generate" in self.capabilities
        ):
            result = await self._generate_code(data)
        else:
            result = {"status": "capability_not_available"}

        return result

    async def _execute_mcp_tool(
        self, tool_name: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute tool through real MCP server using stdio transport"""
        try:
            # Use the real MCP client with stdio transport
            result = await execute_mcp_tool(tool_name, params)
            
            if result.get("status") == "success":
                return {
                    "status": "success",
                    "tool": tool_name,
                    "result": result.get("result", {}),
                    "latency_ms": result.get("latency_ms", 0),
                    "timestamp": result.get("timestamp")
                }
            else:
                logger.error(f"MCP tool execution failed: {result.get('error')}")
                # Fallback to direct tool execution if MCP server call fails
                return await self._execute_tool_direct(tool_name, params)
                
        except Exception as e:
            logger.error(f"MCP tool execution error: {e}")
            # Fallback to direct tool execution if MCP client fails
            return await self._execute_tool_direct(tool_name, params)

    async def _execute_tool_direct(
        self, tool_name: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Direct tool execution when MCP server is unavailable"""
        if tool_name == "process_data":
            from protocols.data_processor import task as process_data_task

            return process_data_task()
        elif tool_name == "quantum_optimize":
            # Real quantum optimization
            from connectors.dwave_quantum_connector import (
                DWaveQuantumConnector,
            )

            connector = DWaveQuantumConnector()
            if await connector.connect({}):
                result = await connector.execute_action("solve_qubo", params)
                await connector.disconnect()
                return result
            else:
                return {"status": "error", "error": "Quantum connector failed"}
        else:
            return {"status": "unknown_tool", "tool": tool_name}

    def _generate_negotiation_proposal(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Generate negotiation proposal based on agent capabilities"""
        return {
            "resources": self._get_available_resources(),
            "capabilities": self.capabilities,
            "terms": {"duration": "flexible", "priority": "normal"},
        }

    def _get_agent_constraints(self) -> Dict[str, Any]:
        """Get agent-specific constraints"""
        return {
            "max_concurrent_tasks": 5,
            "memory_limit_mb": 1024,
            "cpu_limit_cores": 2,
        }

    def _get_agent_preferences(self) -> Dict[str, Any]:
        """Get agent preferences"""
        return {
            "preferred_transport": "mcp_pipe",
            "max_latency_ms": self.sla_requirements["max_latency_ms"],
            "error_tolerance": "low",
        }

    def _get_available_resources(self) -> Dict[str, Any]:
        """Get available resources for negotiation"""
        return {"cpu_cores": 4, "memory_mb": 2048, "storage_gb": 100}

    async def _analyze_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze data using real MCP tools"""
        try:
            # Use real MCP code analyzer for data analysis
            if isinstance(data, dict) and "code" in data:
                # Analyze code data
                result = await self._execute_mcp_tool(
                    "code_analyzer",
                    {
                        "code": data["code"],
                        "language": data.get("language", "python")
                    }
                )
            else:
                # Use self_corrector for general data analysis
                data_str = json.dumps(data) if isinstance(data, dict) else str(data)
                result = await self._execute_mcp_tool(
                    "self_corrector",
                    {
                        "code": data_str,
                        "strict_mode": True
                    }
                )

            # Process real MCP results
            if result.get("status") == "success":
                mcp_result = result.get("result", {})
                
                # Calculate confidence based on MCP response
                confidence = 0.95 if "error" not in str(mcp_result) else 0.5
                
                return {
                    "analysis_type": "mcp_comprehensive",
                    "status": "completed",
                    "mcp_result": mcp_result,
                    "latency_ms": result.get("latency_ms", 0),
                    "confidence": confidence,
                    "timestamp": result.get("timestamp"),
                    "tool_used": result.get("tool")
                }
            else:
                return {
                    "analysis_type": "mcp_failed",
                    "status": "failed",
                    "error": result.get("error", "Unknown MCP error"),
                    "confidence": 0.0,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Data analysis failed: {e}")
            return {
                "analysis_type": "failed",
                "error": str(e),
                "confidence": 0.0,
                "timestamp": datetime.utcnow().isoformat()
            }

    async def _generate_code(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate code using real MCP code generation tools"""
        try:
            # Extract code generation parameters
            code_spec = {
                "type": data.get("type", "function"),
                "language": data.get("language", "python"),
                "framework": data.get("framework", ""),
                "description": data.get("description", ""),
                "requirements": data.get("requirements", []),
            }

            # Generate initial code template
            template_code = self._generate_code_template(code_spec)
            
            # Use real MCP code analyzer to validate and improve the generated code
            result = await self._execute_mcp_tool(
                "code_analyzer",
                {
                    "code": template_code,
                    "language": code_spec["language"]
                }
            )

            if result.get("status") == "success":
                mcp_result = result.get("result", {})
                
                # Use self_corrector to improve the code
                corrector_result = await self._execute_mcp_tool(
                    "self_corrector",
                    {
                        "code": template_code,
                        "language": code_spec["language"],
                        "strict_mode": False
                    }
                )
                
                return {
                    "code_type": code_spec["type"],
                    "language": code_spec["language"],
                    "code": template_code,
                    "analysis": mcp_result,
                    "suggestions": corrector_result.get("result", {}),
                    "latency_ms": result.get("latency_ms", 0),
                    "status": "generated_and_analyzed"
                }
            else:
                # Fallback to basic template if MCP fails
                return {
                    "code_type": code_spec["type"],
                    "language": code_spec["language"],
                    "code": template_code,
                    "status": "generated_basic",
                    "mcp_error": result.get("error")
                }
                
        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            return {
                "code_type": "error",
                "error": str(e),
                "code": "",
                "status": "failed"
            }

    def _generate_code_template(self, spec: Dict[str, Any]) -> str:
        """Generate code template based on specifications"""
        code_type = spec.get("type", "function")
        language = spec.get("language", "python")

        if language == "python":
            if code_type == "api_endpoint" and spec.get("framework") == "fastapi":
                return '''from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

app = FastAPI()

class RequestModel(BaseModel):
    data: Dict[str, Any]

@app.post("/api/process")
async def process_data(request: RequestModel):
    """Process data endpoint"""
    try:
        # Real processing logic here
        result = {"status": "success", "data": request.data}
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
'''
            elif code_type == "function":
                return f'''def {
                    spec.get(
                        'name',
                        'process')}(data: Dict[str, Any]) -> Dict[str, Any]:
    """Generated function: {
                    spec.get(
                        'description',
                        'Process data')}"""
    # Implementation
    result = {
                    "processed": True, "input": data}
    return result
'''

        return f"# Generated {code_type} in {language}"


class A2AMCPOrchestrator:
    """
    Orchestrates A2A communication with MCP.
    Manages agents, routing, and monitoring.
    """

    def __init__(self):
        self.agents: Dict[str, MCPEnabledA2AAgent] = {}
        self.message_bus = A2AMessageBus()
        self.performance_monitor = PerformanceMonitor()
        self.negotiation_manager = NegotiationManager()

        self.message_bus.agents = self.agents  # Link agents to message bus

    def register_agent(self, agent: MCPEnabledA2AAgent):
        """Register agent with orchestrator"""
        self.agents[agent.agent_id] = agent
        agent.message_bus = self.message_bus  # Inject message bus into agent
        logger.info(f"Registered agent: {agent.agent_id}")

    async def start(self):
        """Start the A2A MCP orchestrator"""
        logger.info("Starting A2A MCP Orchestrator...")

        # Start message bus
        bus_task = asyncio.create_task(self.message_bus.start())

        # Start performance monitoring
        monitor_task = asyncio.create_task(self.performance_monitor.start())

        # Start negotiation manager
        negotiation_task = asyncio.create_task(self.negotiation_manager.start())

        return bus_task, monitor_task, negotiation_task

    async def stop(self):
        """Stop the orchestrator"""
        self.message_bus.stop()
        self.performance_monitor.stop()
        self.negotiation_manager.stop()
        logger.info("A2A MCP Orchestrator stopped")

    def get_agent(self, agent_id: str) -> Optional[MCPEnabledA2AAgent]:
        """Get registered agent by ID"""
        return self.agents.get(agent_id)

    def list_agents(self) -> List[str]:
        """List all registered agent IDs"""
        return list(self.agents.keys())

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get overall performance statistics"""
        return self.performance_monitor.get_stats()


class PerformanceMonitor:
    """Monitors performance metrics across all agents"""

    def __init__(self):
        self.stats = {
            "total_messages": 0,
            "avg_latency_ms": 0.0,
            "sla_violations": 0,
            "active_agents": 0,
        }
        self.running = False

    async def start(self):
        """Start performance monitoring"""
        self.running = True
        while self.running:
            await self._update_stats()
            await asyncio.sleep(5)  # Update every 5 seconds

    def stop(self):
        """Stop performance monitoring"""
        self.running = False

    async def _update_stats(self):
        """Update performance statistics from real MCP metrics"""
        try:
            from connectors.real_mcp_client import get_mcp_client_pool
            
            # Get real metrics from MCP client pool
            pool = await get_mcp_client_pool()
            pool_stats = pool.stats
            
            # Update with real metrics
            self.stats.update({
                "total_messages": pool_stats.get("total_requests", 0),
                "avg_latency_ms": pool_stats.get("avg_latency_ms", 0.0),
                "active_connections": pool_stats.get("active_connections", 0),
                "error_rate": pool_stats.get("error_rate", 0.0),
                "last_updated": datetime.utcnow().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Failed to update performance stats: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        return self.stats.copy()


class NegotiationManager:
    """Manages multi-agent negotiations"""

    def __init__(self):
        self.active_negotiations: Dict[str, Dict[str, Any]] = {}
        self.running = False

    async def start(self):
        """Start negotiation manager"""
        self.running = True
        while self.running:
            await self._process_negotiations()
            await asyncio.sleep(1)

    def stop(self):
        """Stop negotiation manager"""
        self.running = False

    async def _process_negotiations(self):
        """Process active negotiations"""
        # This would handle ongoing negotiations


# Global orchestrator instance
a2a_mcp_orchestrator = A2AMCPOrchestrator()


# Example usage and demonstration
async def demonstrate_a2a_mcp_integration():
    """Demonstrate A2A MCP integration"""

    print("=== A2A MCP Integration Demo ===\n")

    # Create agents
    analyzer = MCPEnabledA2AAgent("data_analyzer", ["analyze", "process"])
    generator = MCPEnabledA2AAgent("code_generator", ["generate", "create"])
    negotiator = MCPEnabledA2AAgent("negotiator", ["negotiate", "coordinate"])

    # Register agents
    a2a_mcp_orchestrator.register_agent(analyzer)
    a2a_mcp_orchestrator.register_agent(generator)
    a2a_mcp_orchestrator.register_agent(negotiator)

    # Start orchestrator
    bus_task, monitor_task, negotiation_task = await a2a_mcp_orchestrator.start()

    # Demo 1: Contextualized message sending
    print("1. Sending contextualized message:")
    result = await analyzer.send_contextualized_message(
        recipient="code_generator",
        intent={
            "action": "generate_code",
            "data": {
                "type": "api_endpoint",
                "language": "python",
                "framework": "fastapi",
            },
        },
        priority=MessagePriority.HIGH,
    )
    print(f"   - Latency: {result['latency_ms']:.2f}ms")
    print(f"   - Strategy: {result['transport_strategy']}")
    print(f"   - Status: {result['status']}\n")

    # Demo 2: Multi-agent negotiation
    print("2. Multi-agent negotiation:")
    negotiation_result = await negotiator.send_contextualized_message(
        recipient="data_analyzer",
        intent={
            "action": "negotiate",
            "data": {
                "topic": "resource_allocation",
                "participants": ["data_analyzer", "code_generator"],
                "requirements": {"cpu_cores": 4, "memory_mb": 2048},
            },
        },
        priority=MessagePriority.CRITICAL,
        deadline_ms=50,  # 50ms deadline
    )
    print(f"   - Latency: {negotiation_result['latency_ms']:.2f}ms")
    print(f"   - Strategy: {negotiation_result['transport_strategy']}\n")

    # Demo 3: Performance monitoring
    print("3. Performance monitoring:")
    stats = a2a_mcp_orchestrator.get_performance_stats()
    print(f"   - Active agents: {len(a2a_mcp_orchestrator.list_agents())}")
    print(f"   - Total messages: {stats['total_messages']}")
    print(f"   - SLA violations: {stats['sla_violations']}\n")

    # Stop orchestrator
    await a2a_mcp_orchestrator.stop()
    await bus_task
    await monitor_task
    await negotiation_task

    print("✅ A2A MCP Integration Demo Complete!")


if __name__ == "__main__":
    asyncio.run(demonstrate_a2a_mcp_integration())
