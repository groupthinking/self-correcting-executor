#!/usr/bin/env python3
"""
Production Subagent Orchestrator
===============================

Deployment orchestrator for all specialized subagents with MCP integration,
health monitoring, SLA compliance, and production-ready error handling.
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import traceback

from agents.a2a_mcp_integration import A2AMCPOrchestrator, MCPEnabledA2AAgent, MessagePriority
from connectors.real_mcp_client import get_mcp_client_pool, execute_mcp_tool

# Import all specialized subagents
from agents.specialized.code_analysis_subagents import create_code_analysis_subagents
from agents.specialized.video_processing_subagents import create_video_processing_subagents
from agents.specialized.multimodal_ai_subagents import create_multimodal_ai_subagents
from agents.specialized.testing_orchestration_subagents import create_testing_orchestration_subagents

logger = logging.getLogger(__name__)


class SubagentStatus(Enum):
    """Subagent operational status"""
    INITIALIZING = "initializing"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"


class WorkloadPriority(Enum):
    """Workload priority levels"""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4


@dataclass
class SubagentMetrics:
    """Metrics for individual subagent performance"""
    agent_id: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time_ms: float = 0.0
    last_request_time: Optional[datetime] = None
    uptime_seconds: float = 0.0
    error_rate: float = 0.0
    status: SubagentStatus = SubagentStatus.INITIALIZING
    health_check_failures: int = 0
    sla_violations: int = 0
    last_health_check: Optional[datetime] = None


@dataclass
class WorkloadRequest:
    """Represents a workload request to be processed"""
    request_id: str
    agent_type: str
    action: str
    data: Dict[str, Any]
    priority: WorkloadPriority = WorkloadPriority.NORMAL
    max_retries: int = 3
    timeout_seconds: int = 300
    submitted_at: datetime = field(default_factory=datetime.utcnow)
    deadline: Optional[datetime] = None
    callback_url: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DeploymentConfig:
    """Configuration for subagent deployment"""
    health_check_interval: int = 30  # seconds
    metrics_collection_interval: int = 60  # seconds
    sla_response_timeout: int = 5000  # milliseconds
    max_concurrent_requests: int = 100
    retry_backoff_base: float = 1.0  # seconds
    circuit_breaker_threshold: int = 5  # failures before circuit opens
    circuit_breaker_timeout: int = 60  # seconds before retry
    performance_monitoring: bool = True
    enable_auto_scaling: bool = False
    log_level: str = "INFO"


class ProductionSubagentOrchestrator:
    """
    Production-ready orchestrator for all specialized subagents.
    Handles deployment, health monitoring, load balancing, and SLA compliance.
    """

    def __init__(self, config: Optional[DeploymentConfig] = None):
        self.config = config or DeploymentConfig()
        
        # Core components
        self.a2a_orchestrator = A2AMCPOrchestrator()
        self.subagents: Dict[str, MCPEnabledA2AAgent] = {}
        self.metrics: Dict[str, SubagentMetrics] = {}
        
        # Workload management
        self.request_queue: asyncio.Queue[WorkloadRequest] = asyncio.Queue()
        self.active_requests: Dict[str, WorkloadRequest] = {}
        self.completed_requests: List[Dict[str, Any]] = []
        
        # Health and monitoring
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        self.performance_stats: Dict[str, Any] = {}
        
        # Runtime state
        self.is_running = False
        self.startup_time: Optional[datetime] = None
        self.tasks: List[asyncio.Task] = []
        
        # Configure logging
        logging.basicConfig(level=getattr(logging, self.config.log_level))

    async def initialize(self) -> Dict[str, Any]:
        """Initialize all subagents and start orchestration"""
        try:
            logger.info("🚀 Starting Production Subagent Orchestrator...")
            self.startup_time = datetime.utcnow()
            
            # Initialize MCP infrastructure
            await self._initialize_mcp_infrastructure()
            
            # Deploy all subagent categories
            deployment_results = await self._deploy_all_subagents()
            
            # Start core services
            await self._start_core_services()
            
            # Verify deployment health
            health_check = await self._verify_deployment_health()
            
            self.is_running = True
            
            logger.info("✅ Production Subagent Orchestrator initialized successfully")
            
            return {
                "status": "initialized",
                "startup_time": self.startup_time.isoformat(),
                "deployment_results": deployment_results,
                "health_check": health_check,
                "total_subagents": len(self.subagents),
                "subagent_categories": list(set(agent.agent_id.split('-')[0] for agent in self.subagents.values())),
                "mcp_integration": "active",
                "ready_for_workloads": True
            }
            
        except Exception as e:
            logger.error(f"Failed to initialize orchestrator: {e}")
            logger.error(traceback.format_exc())
            return {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    async def _initialize_mcp_infrastructure(self):
        """Initialize MCP client pool and validate connectivity"""
        logger.info("Initializing MCP infrastructure...")
        
        # Get MCP client pool (creates if doesn't exist)
        mcp_pool = await get_mcp_client_pool()
        
        # Test MCP connectivity with all available tools
        test_result = await execute_mcp_tool("protocol_validator", {
            "message": json.dumps({"jsonrpc": "2.0", "method": "test", "id": 1}),
            "protocol_version": "2024-11-05"
        })
        
        if test_result.get("status") != "success":
            raise RuntimeError(f"MCP infrastructure validation failed: {test_result.get('error')}")
        
        logger.info("✅ MCP infrastructure initialized and validated")

    async def _deploy_all_subagents(self) -> Dict[str, Any]:
        """Deploy all categories of specialized subagents"""
        logger.info("Deploying all specialized subagents...")
        
        deployment_results = {
            "code_analysis": await self._deploy_subagent_category(
                "Code Analysis & Refactoring", 
                create_code_analysis_subagents
            ),
            "video_processing": await self._deploy_subagent_category(
                "Video Processing Pipeline", 
                create_video_processing_subagents
            ),
            "multimodal_ai": await self._deploy_subagent_category(
                "Multi-Modal AI Workflows", 
                create_multimodal_ai_subagents
            ),
            "testing_orchestration": await self._deploy_subagent_category(
                "Software Testing Orchestration", 
                create_testing_orchestration_subagents
            )
        }
        
        # Calculate deployment summary
        total_deployed = sum(result["deployed_count"] for result in deployment_results.values())
        total_failed = sum(result["failed_count"] for result in deployment_results.values())
        
        logger.info(f"✅ Deployed {total_deployed} subagents successfully, {total_failed} failed")
        
        return {
            "summary": {
                "total_deployed": total_deployed,
                "total_failed": total_failed,
                "deployment_success_rate": total_deployed / (total_deployed + total_failed) if (total_deployed + total_failed) > 0 else 0
            },
            "by_category": deployment_results
        }

    async def _deploy_subagent_category(self, category_name: str, factory_function) -> Dict[str, Any]:
        """Deploy a category of subagents"""
        logger.info(f"Deploying {category_name} subagents...")
        
        deployed_count = 0
        failed_count = 0
        errors = []
        
        try:
            # Create subagents using factory function
            subagents = factory_function()
            
            for agent in subagents:
                try:
                    # Register with A2A orchestrator
                    self.a2a_orchestrator.register_agent(agent)
                    
                    # Add to our tracking
                    self.subagents[agent.agent_id] = agent
                    self.metrics[agent.agent_id] = SubagentMetrics(agent_id=agent.agent_id)
                    
                    # Initialize circuit breaker
                    self.circuit_breakers[agent.agent_id] = {
                        "state": "closed",  # closed, open, half-open
                        "failure_count": 0,
                        "last_failure": None,
                        "next_attempt": None
                    }
                    
                    deployed_count += 1
                    logger.info(f"✅ Deployed {agent.agent_id}")
                    
                except Exception as e:
                    failed_count += 1
                    error_msg = f"Failed to deploy {agent.agent_id}: {str(e)}"
                    errors.append(error_msg)
                    logger.error(error_msg)
            
        except Exception as e:
            failed_count += 1
            error_msg = f"Failed to create {category_name} subagents: {str(e)}"
            errors.append(error_msg)
            logger.error(error_msg)
        
        return {
            "category_name": category_name,
            "deployed_count": deployed_count,
            "failed_count": failed_count,
            "errors": errors,
            "success_rate": deployed_count / (deployed_count + failed_count) if (deployed_count + failed_count) > 0 else 0
        }

    async def _start_core_services(self):
        """Start core orchestration services"""
        logger.info("Starting core orchestration services...")
        
        # Start A2A message bus and monitoring
        bus_task, monitor_task, negotiation_task = await self.a2a_orchestrator.start()
        self.tasks.extend([bus_task, monitor_task, negotiation_task])
        
        # Start our own services
        self.tasks.extend([
            asyncio.create_task(self._workload_processor()),
            asyncio.create_task(self._health_monitor()),
            asyncio.create_task(self._metrics_collector()),
            asyncio.create_task(self._performance_monitor())
        ])
        
        logger.info("✅ Core services started")

    async def _verify_deployment_health(self) -> Dict[str, Any]:
        """Verify health of all deployed subagents"""
        logger.info("Verifying deployment health...")
        
        health_results = {}
        total_healthy = 0
        total_unhealthy = 0
        
        for agent_id, agent in self.subagents.items():
            try:
                # Test basic functionality
                test_result = await self._test_subagent_health(agent)
                
                if test_result["healthy"]:
                    self.metrics[agent_id].status = SubagentStatus.HEALTHY
                    total_healthy += 1
                else:
                    self.metrics[agent_id].status = SubagentStatus.UNHEALTHY
                    total_unhealthy += 1
                
                health_results[agent_id] = test_result
                
            except Exception as e:
                self.metrics[agent_id].status = SubagentStatus.UNHEALTHY
                total_unhealthy += 1
                health_results[agent_id] = {
                    "healthy": False,
                    "error": str(e)
                }
        
        overall_health = total_healthy / (total_healthy + total_unhealthy) if (total_healthy + total_unhealthy) > 0 else 0
        
        logger.info(f"Health check complete: {total_healthy} healthy, {total_unhealthy} unhealthy")
        
        return {
            "overall_health_percentage": overall_health * 100,
            "healthy_subagents": total_healthy,
            "unhealthy_subagents": total_unhealthy,
            "individual_results": health_results,
            "deployment_ready": overall_health > 0.8  # 80% threshold
        }

    async def _test_subagent_health(self, agent: MCPEnabledA2AAgent) -> Dict[str, Any]:
        """Test individual subagent health"""
        try:
            start_time = time.time()
            
            # Test basic intent processing
            test_intent = {
                "action": "health_check",
                "data": {"test": True}
            }
            
            result = await asyncio.wait_for(
                agent.process_intent(test_intent),
                timeout=10.0
            )
            
            response_time_ms = (time.time() - start_time) * 1000
            
            # Check if we got a valid response
            healthy = (
                isinstance(result, dict) and
                result.get("status") != "error" and
                response_time_ms < self.config.sla_response_timeout
            )
            
            return {
                "healthy": healthy,
                "response_time_ms": response_time_ms,
                "capabilities": agent.capabilities,
                "test_result": result
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "response_time_ms": 10000  # Timeout value
            }

    async def submit_workload(self, request: WorkloadRequest) -> Dict[str, Any]:
        """Submit a workload request for processing"""
        try:
            # Validate request
            if request.agent_type not in self._get_available_agent_types():
                return {
                    "status": "error",
                    "error": f"Unknown agent type: {request.agent_type}",
                    "available_types": self._get_available_agent_types()
                }
            
            # Set deadline if not provided
            if not request.deadline:
                request.deadline = datetime.utcnow() + timedelta(seconds=request.timeout_seconds)
            
            # Add to queue
            await self.request_queue.put(request)
            self.active_requests[request.request_id] = request
            
            logger.info(f"Workload {request.request_id} submitted for {request.agent_type}")
            
            return {
                "status": "submitted",
                "request_id": request.request_id,
                "estimated_processing_time": self._estimate_processing_time(request.agent_type),
                "queue_position": self.request_queue.qsize(),
                "deadline": request.deadline.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to submit workload: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    def _get_available_agent_types(self) -> List[str]:
        """Get list of available agent types"""
        return list(set(agent_id.replace('-', '_') for agent_id in self.subagents.keys()))

    def _estimate_processing_time(self, agent_type: str) -> Dict[str, Any]:
        """Estimate processing time for agent type"""
        # Get historical performance data
        agent_ids = [aid for aid in self.subagents.keys() if agent_type.replace('_', '-') in aid]
        
        if not agent_ids:
            return {"estimated_seconds": 30, "confidence": "low"}
        
        avg_response_times = [
            self.metrics[aid].avg_response_time_ms 
            for aid in agent_ids 
            if self.metrics[aid].avg_response_time_ms > 0
        ]
        
        if avg_response_times:
            avg_time_seconds = sum(avg_response_times) / len(avg_response_times) / 1000
            return {
                "estimated_seconds": max(avg_time_seconds, 5),
                "confidence": "high" if len(avg_response_times) > 10 else "medium"
            }
        else:
            # Default estimates by agent type
            defaults = {
                "security_analyzer": 15,
                "performance_optimizer": 20,
                "style_checker": 10,
                "transcription_agent": 30,
                "action_generator": 15,
                "quality_assessor": 10,
                "text_processor": 8,
                "image_analyzer": 12,
                "audio_transcriber": 25,
                "unit_tester": 20,
                "integration_tester": 45,
                "performance_tester": 60
            }
            
            return {
                "estimated_seconds": defaults.get(agent_type, 30),
                "confidence": "medium"
            }

    async def _workload_processor(self):
        """Process workload requests from the queue"""
        logger.info("Starting workload processor...")
        
        while self.is_running:
            try:
                # Get next request with timeout
                try:
                    request = await asyncio.wait_for(
                        self.request_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Process the request
                await self._process_workload_request(request)
                
            except Exception as e:
                logger.error(f"Error in workload processor: {e}")
                await asyncio.sleep(1.0)

    async def _process_workload_request(self, request: WorkloadRequest):
        """Process individual workload request"""
        start_time = time.time()
        
        try:
            # Check if request has expired
            if request.deadline and datetime.utcnow() > request.deadline:
                await self._complete_request(request, {
                    "status": "timeout",
                    "error": "Request deadline exceeded",
                    "processing_time_ms": (time.time() - start_time) * 1000
                })
                return
            
            # Find appropriate agent
            agent = self._select_best_agent(request.agent_type)
            if not agent:
                await self._complete_request(request, {
                    "status": "error",
                    "error": f"No healthy agents available for type: {request.agent_type}",
                    "processing_time_ms": (time.time() - start_time) * 1000
                })
                return
            
            # Check circuit breaker
            if not self._is_circuit_closed(agent.agent_id):
                await self._complete_request(request, {
                    "status": "error",
                    "error": f"Circuit breaker open for agent: {agent.agent_id}",
                    "processing_time_ms": (time.time() - start_time) * 1000
                })
                return
            
            # Process the request
            result = await self._execute_with_retry(agent, request)
            
            # Update metrics
            processing_time_ms = (time.time() - start_time) * 1000
            await self._update_agent_metrics(agent.agent_id, result, processing_time_ms)
            
            # Complete the request
            result["processing_time_ms"] = processing_time_ms
            await self._complete_request(request, result)
            
        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            logger.error(f"Failed to process workload {request.request_id}: {e}")
            
            await self._complete_request(request, {
                "status": "error",
                "error": str(e),
                "processing_time_ms": processing_time_ms
            })

    def _select_best_agent(self, agent_type: str) -> Optional[MCPEnabledA2AAgent]:
        """Select the best available agent for the given type"""
        # Find agents of the requested type
        candidate_agents = [
            (agent_id, agent) for agent_id, agent in self.subagents.items()
            if agent_type.replace('_', '-') in agent_id
        ]
        
        if not candidate_agents:
            return None
        
        # Filter by health status
        healthy_agents = [
            (agent_id, agent) for agent_id, agent in candidate_agents
            if self.metrics[agent_id].status == SubagentStatus.HEALTHY
        ]
        
        if not healthy_agents:
            # Fall back to degraded agents if no healthy ones
            healthy_agents = [
                (agent_id, agent) for agent_id, agent in candidate_agents
                if self.metrics[agent_id].status == SubagentStatus.DEGRADED
            ]
        
        if not healthy_agents:
            return None
        
        # Select agent with best performance metrics
        best_agent_id, best_agent = min(
            healthy_agents,
            key=lambda x: (
                self.metrics[x[0]].error_rate,
                self.metrics[x[0]].avg_response_time_ms
            )
        )
        
        return best_agent

    def _is_circuit_closed(self, agent_id: str) -> bool:
        """Check if circuit breaker is closed (allowing requests)"""
        breaker = self.circuit_breakers.get(agent_id, {})
        state = breaker.get("state", "closed")
        
        if state == "closed":
            return True
        elif state == "open":
            # Check if we should transition to half-open
            next_attempt = breaker.get("next_attempt")
            if next_attempt and datetime.utcnow() >= next_attempt:
                breaker["state"] = "half-open"
                return True
            return False
        elif state == "half-open":
            return True
        
        return False

    async def _execute_with_retry(self, agent: MCPEnabledA2AAgent, request: WorkloadRequest) -> Dict[str, Any]:
        """Execute request with retry logic"""
        last_error = None
        
        for attempt in range(request.max_retries + 1):
            try:
                # Create intent from request
                intent = {
                    "action": request.action,
                    "data": request.data
                }
                
                # Execute with timeout
                result = await asyncio.wait_for(
                    agent.process_intent(intent),
                    timeout=request.timeout_seconds
                )
                
                # Check for successful result
                if isinstance(result, dict) and result.get("status") != "error":
                    # Success - reset circuit breaker
                    self._reset_circuit_breaker(agent.agent_id)
                    return result
                else:
                    # Logical error - don't retry
                    self._record_circuit_breaker_failure(agent.agent_id)
                    return result
                
            except asyncio.TimeoutError:
                last_error = "Request timeout"
                self._record_circuit_breaker_failure(agent.agent_id)
                
            except Exception as e:
                last_error = str(e)
                self._record_circuit_breaker_failure(agent.agent_id)
            
            # Wait before retry with exponential backoff
            if attempt < request.max_retries:
                backoff_time = self.config.retry_backoff_base * (2 ** attempt)
                await asyncio.sleep(backoff_time)
        
        # All retries failed
        return {
            "status": "error",
            "error": f"All {request.max_retries + 1} attempts failed. Last error: {last_error}",
            "attempts": request.max_retries + 1
        }

    def _reset_circuit_breaker(self, agent_id: str):
        """Reset circuit breaker after successful request"""
        if agent_id in self.circuit_breakers:
            self.circuit_breakers[agent_id].update({
                "state": "closed",
                "failure_count": 0,
                "last_failure": None,
                "next_attempt": None
            })

    def _record_circuit_breaker_failure(self, agent_id: str):
        """Record failure and potentially open circuit breaker"""
        if agent_id not in self.circuit_breakers:
            return
        
        breaker = self.circuit_breakers[agent_id]
        breaker["failure_count"] += 1
        breaker["last_failure"] = datetime.utcnow()
        
        # Open circuit if threshold exceeded
        if breaker["failure_count"] >= self.config.circuit_breaker_threshold:
            breaker["state"] = "open"
            breaker["next_attempt"] = datetime.utcnow() + timedelta(
                seconds=self.config.circuit_breaker_timeout
            )

    async def _update_agent_metrics(self, agent_id: str, result: Dict[str, Any], processing_time_ms: float):
        """Update performance metrics for agent"""
        if agent_id not in self.metrics:
            return
        
        metrics = self.metrics[agent_id]
        metrics.total_requests += 1
        metrics.last_request_time = datetime.utcnow()
        
        # Update response time (rolling average)
        if metrics.avg_response_time_ms == 0:
            metrics.avg_response_time_ms = processing_time_ms
        else:
            metrics.avg_response_time_ms = (
                metrics.avg_response_time_ms * 0.8 + processing_time_ms * 0.2
            )
        
        # Update success/failure counts
        if result.get("status") == "error":
            metrics.failed_requests += 1
        else:
            metrics.successful_requests += 1
        
        # Update error rate
        metrics.error_rate = metrics.failed_requests / metrics.total_requests
        
        # Check for SLA violations
        if processing_time_ms > self.config.sla_response_timeout:
            metrics.sla_violations += 1
        
        # Update health status based on metrics
        if metrics.error_rate > 0.1:  # 10% error rate threshold
            metrics.status = SubagentStatus.UNHEALTHY
        elif metrics.error_rate > 0.05:  # 5% error rate threshold
            metrics.status = SubagentStatus.DEGRADED
        else:
            metrics.status = SubagentStatus.HEALTHY

    async def _complete_request(self, request: WorkloadRequest, result: Dict[str, Any]):
        """Complete a workload request"""
        # Remove from active requests
        if request.request_id in self.active_requests:
            del self.active_requests[request.request_id]
        
        # Add to completed requests (keep last 1000)
        completion_record = {
            "request_id": request.request_id,
            "agent_type": request.agent_type,
            "action": request.action,
            "priority": request.priority.name,
            "submitted_at": request.submitted_at.isoformat(),
            "completed_at": datetime.utcnow().isoformat(),
            "result": result
        }
        
        self.completed_requests.append(completion_record)
        if len(self.completed_requests) > 1000:
            self.completed_requests = self.completed_requests[-1000:]
        
        logger.info(f"Completed workload {request.request_id} with status: {result.get('status')}")

    async def _health_monitor(self):
        """Monitor health of all subagents"""
        logger.info("Starting health monitor...")
        
        while self.is_running:
            try:
                for agent_id, agent in self.subagents.items():
                    try:
                        health_result = await self._test_subagent_health(agent)
                        
                        metrics = self.metrics[agent_id]
                        metrics.last_health_check = datetime.utcnow()
                        
                        if health_result["healthy"]:
                            metrics.health_check_failures = 0
                            if metrics.status == SubagentStatus.UNHEALTHY:
                                metrics.status = SubagentStatus.DEGRADED  # Recovery state
                        else:
                            metrics.health_check_failures += 1
                            if metrics.health_check_failures >= 3:
                                metrics.status = SubagentStatus.UNHEALTHY
                            elif metrics.status == SubagentStatus.HEALTHY:
                                metrics.status = SubagentStatus.DEGRADED
                        
                    except Exception as e:
                        logger.error(f"Health check failed for {agent_id}: {e}")
                        self.metrics[agent_id].health_check_failures += 1
                
                await asyncio.sleep(self.config.health_check_interval)
                
            except Exception as e:
                logger.error(f"Error in health monitor: {e}")
                await asyncio.sleep(self.config.health_check_interval)

    async def _metrics_collector(self):
        """Collect and aggregate performance metrics"""
        logger.info("Starting metrics collector...")
        
        while self.is_running:
            try:
                # Update uptime for all agents
                current_time = datetime.utcnow()
                
                for metrics in self.metrics.values():
                    if self.startup_time:
                        metrics.uptime_seconds = (current_time - self.startup_time).total_seconds()
                
                # Collect system-level metrics
                await self._collect_system_metrics()
                
                await asyncio.sleep(self.config.metrics_collection_interval)
                
            except Exception as e:
                logger.error(f"Error in metrics collector: {e}")
                await asyncio.sleep(self.config.metrics_collection_interval)

    async def _collect_system_metrics(self):
        """Collect system-level performance metrics"""
        try:
            # Get MCP client pool stats
            mcp_pool = await get_mcp_client_pool()
            mcp_stats = mcp_pool.stats
            
            self.performance_stats.update({
                "timestamp": datetime.utcnow().isoformat(),
                "total_subagents": len(self.subagents),
                "healthy_subagents": sum(1 for m in self.metrics.values() if m.status == SubagentStatus.HEALTHY),
                "active_requests": len(self.active_requests),
                "queue_size": self.request_queue.qsize(),
                "completed_requests": len(self.completed_requests),
                "mcp_stats": mcp_stats,
                "circuit_breaker_status": {
                    agent_id: breaker["state"] 
                    for agent_id, breaker in self.circuit_breakers.items()
                }
            })
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")

    async def _performance_monitor(self):
        """Monitor overall system performance"""
        logger.info("Starting performance monitor...")
        
        while self.is_running:
            try:
                if self.config.performance_monitoring:
                    await self._analyze_performance_trends()
                    await self._check_sla_compliance()
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in performance monitor: {e}")
                await asyncio.sleep(60)

    async def _analyze_performance_trends(self):
        """Analyze performance trends and identify issues"""
        # Calculate system-wide metrics
        total_requests = sum(m.total_requests for m in self.metrics.values())
        total_errors = sum(m.failed_requests for m in self.metrics.values())
        avg_response_time = sum(m.avg_response_time_ms for m in self.metrics.values()) / len(self.metrics) if self.metrics else 0
        
        # Identify performance issues
        issues = []
        if total_errors / max(total_requests, 1) > 0.05:
            issues.append("High system error rate detected")
        
        if avg_response_time > self.config.sla_response_timeout:
            issues.append("Average response time exceeds SLA")
        
        if self.request_queue.qsize() > self.config.max_concurrent_requests * 0.8:
            issues.append("Request queue approaching capacity")
        
        if issues:
            logger.warning(f"Performance issues detected: {', '.join(issues)}")

    async def _check_sla_compliance(self):
        """Check SLA compliance across all subagents"""
        violations = []
        
        for agent_id, metrics in self.metrics.items():
            if metrics.sla_violations > 0:
                violation_rate = metrics.sla_violations / max(metrics.total_requests, 1)
                if violation_rate > 0.05:  # 5% SLA violation threshold
                    violations.append({
                        "agent_id": agent_id,
                        "violation_rate": violation_rate,
                        "total_violations": metrics.sla_violations
                    })
        
        if violations:
            logger.warning(f"SLA violations detected: {violations}")

    async def get_status(self) -> Dict[str, Any]:
        """Get comprehensive orchestrator status"""
        if not self.is_running:
            return {"status": "not_running"}
        
        # Aggregate metrics
        total_requests = sum(m.total_requests for m in self.metrics.values())
        total_successful = sum(m.successful_requests for m in self.metrics.values())
        total_failed = sum(m.failed_requests for m in self.metrics.values())
        
        # Health summary
        health_summary = {}
        for status in SubagentStatus:
            health_summary[status.value] = sum(
                1 for m in self.metrics.values() if m.status == status
            )
        
        # Top performing agents
        top_performers = sorted(
            [(aid, m) for aid, m in self.metrics.items() if m.total_requests > 0],
            key=lambda x: (x[1].error_rate, x[1].avg_response_time_ms)
        )[:5]
        
        return {
            "status": "running",
            "uptime_seconds": (datetime.utcnow() - self.startup_time).total_seconds() if self.startup_time else 0,
            "total_subagents": len(self.subagents),
            "health_summary": health_summary,
            "performance": {
                "total_requests": total_requests,
                "successful_requests": total_successful,
                "failed_requests": total_failed,
                "success_rate": total_successful / max(total_requests, 1),
                "avg_response_time_ms": sum(m.avg_response_time_ms for m in self.metrics.values()) / len(self.metrics) if self.metrics else 0
            },
            "workload": {
                "active_requests": len(self.active_requests),
                "queue_size": self.request_queue.qsize(),
                "completed_requests": len(self.completed_requests)
            },
            "top_performers": [
                {
                    "agent_id": aid,
                    "success_rate": 1 - m.error_rate,
                    "avg_response_time_ms": m.avg_response_time_ms,
                    "total_requests": m.total_requests
                }
                for aid, m in top_performers
            ],
            "circuit_breakers": {
                "open": sum(1 for cb in self.circuit_breakers.values() if cb["state"] == "open"),
                "half_open": sum(1 for cb in self.circuit_breakers.values() if cb["state"] == "half-open"),
                "closed": sum(1 for cb in self.circuit_breakers.values() if cb["state"] == "closed")
            },
            "last_updated": datetime.utcnow().isoformat()
        }

    async def shutdown(self):
        """Gracefully shutdown the orchestrator"""
        logger.info("Shutting down Production Subagent Orchestrator...")
        
        self.is_running = False
        
        # Cancel all tasks
        for task in self.tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        
        # Stop A2A orchestrator
        await self.a2a_orchestrator.stop()
        
        logger.info("✅ Production Subagent Orchestrator shutdown complete")


# Global orchestrator instance
production_orchestrator: Optional[ProductionSubagentOrchestrator] = None


async def get_production_orchestrator() -> ProductionSubagentOrchestrator:
    """Get or create the global production orchestrator"""
    global production_orchestrator
    
    if production_orchestrator is None:
        production_orchestrator = ProductionSubagentOrchestrator()
        await production_orchestrator.initialize()
    
    return production_orchestrator


# Convenience functions for easy integration
async def submit_workload(agent_type: str, action: str, data: Dict[str, Any], 
                         priority: WorkloadPriority = WorkloadPriority.NORMAL,
                         timeout_seconds: int = 300) -> Dict[str, Any]:
    """Submit a workload to the orchestrator"""
    orchestrator = await get_production_orchestrator()
    
    request = WorkloadRequest(
        request_id=f"{agent_type}_{action}_{int(time.time())}",
        agent_type=agent_type,
        action=action,
        data=data,
        priority=priority,
        timeout_seconds=timeout_seconds
    )
    
    return await orchestrator.submit_workload(request)


async def get_orchestrator_status() -> Dict[str, Any]:
    """Get orchestrator status"""
    if production_orchestrator is None:
        return {"status": "not_initialized"}
    
    return await production_orchestrator.get_status()


if __name__ == "__main__":
    async def main():
        """Main function for testing"""
        config = DeploymentConfig(
            health_check_interval=10,
            metrics_collection_interval=30
        )
        
        orchestrator = ProductionSubagentOrchestrator(config)
        
        try:
            # Initialize
            init_result = await orchestrator.initialize()
            print(json.dumps(init_result, indent=2))
            
            # Submit test workloads
            test_workloads = [
                ("security_analyzer", "security_scan", {"code": "def test(): pass"}),
                ("transcription_agent", "transcribe", {"url": "test.mp4"}),
                ("text_processor", "analyze_text", {"text": "Hello world"}),
                ("unit_tester", "generate_tests", {"code": "def add(a, b): return a + b"})
            ]
            
            for agent_type, action, data in test_workloads:
                result = await submit_workload(agent_type, action, data)
                print(f"Submitted {agent_type}: {result.get('status')}")
            
            # Wait a bit for processing
            await asyncio.sleep(5)
            
            # Get status
            status = await orchestrator.get_status()
            print("\nOrchestrator Status:")
            print(json.dumps(status, indent=2))
            
        finally:
            await orchestrator.shutdown()
    
    asyncio.run(main())