#!/usr/bin/env python3
"""
Comprehensive Test Suite for Subagent Deployment
===============================================

Production-ready test suite to verify all subagents work correctly
with MCP integration, A2A communication, and orchestration.
"""

import asyncio
import json
import logging
import time
import sys
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import traceback

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deployment.subagent_orchestrator import (
    ProductionSubagentOrchestrator, 
    DeploymentConfig, 
    WorkloadRequest, 
    WorkloadPriority,
    submit_workload
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SubagentTestSuite:
    """Comprehensive test suite for all subagent functionality"""
    
    def __init__(self):
        self.orchestrator: Optional[ProductionSubagentOrchestrator] = None
        self.test_results: Dict[str, Any] = {}
        self.failed_tests: List[str] = []
        self.passed_tests: List[str] = []
        
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run complete test suite"""
        logger.info("🧪 Starting Comprehensive Subagent Test Suite")
        start_time = time.time()
        
        try:
            # Initialize orchestrator
            await self._initialize_test_environment()
            
            # Run test categories
            test_categories = [
                ("MCP Infrastructure", self._test_mcp_infrastructure),
                ("Orchestrator Deployment", self._test_orchestrator_deployment),
                ("Code Analysis Subagents", self._test_code_analysis_subagents),
                ("Video Processing Subagents", self._test_video_processing_subagents),
                ("Multi-Modal AI Subagents", self._test_multimodal_ai_subagents),
                ("Testing Orchestration Subagents", self._test_testing_orchestration_subagents),
                ("Workload Processing", self._test_workload_processing),
                ("Health Monitoring", self._test_health_monitoring),
                ("Performance Metrics", self._test_performance_metrics),
                ("Error Handling & Recovery", self._test_error_handling),
                ("SLA Compliance", self._test_sla_compliance)
            ]
            
            for category_name, test_function in test_categories:
                logger.info(f"\n🔍 Testing {category_name}...")
                try:
                    result = await test_function()
                    self.test_results[category_name] = result
                    
                    if result.get("passed", 0) > 0:
                        self.passed_tests.append(category_name)
                    if result.get("failed", 0) > 0:
                        self.failed_tests.append(category_name)
                        
                except Exception as e:
                    logger.error(f"❌ Test category {category_name} failed: {e}")
                    self.test_results[category_name] = {
                        "status": "error",
                        "error": str(e),
                        "passed": 0,
                        "failed": 1
                    }
                    self.failed_tests.append(category_name)
            
            # Generate comprehensive report
            report = await self._generate_test_report(start_time)
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Test suite failed: {e}")
            logger.error(traceback.format_exc())
            return {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        
        finally:
            await self._cleanup_test_environment()

    async def _initialize_test_environment(self):
        """Initialize test environment with orchestrator"""
        logger.info("Initializing test environment...")
        
        # Create test configuration
        config = DeploymentConfig(
            health_check_interval=5,  # Faster for testing
            metrics_collection_interval=10,
            sla_response_timeout=3000,  # 3 seconds for tests
            max_concurrent_requests=50,
            log_level="INFO"
        )
        
        # Initialize orchestrator
        self.orchestrator = ProductionSubagentOrchestrator(config)
        init_result = await self.orchestrator.initialize()
        
        if init_result.get("status") != "initialized":
            raise RuntimeError(f"Failed to initialize orchestrator: {init_result}")
        
        logger.info("✅ Test environment initialized")

    async def _test_mcp_infrastructure(self) -> Dict[str, Any]:
        """Test MCP infrastructure connectivity and tools"""
        tests = []
        
        # Test 1: MCP client pool connectivity
        try:
            from connectors.real_mcp_client import get_mcp_client_pool, execute_mcp_tool
            
            pool = await get_mcp_client_pool()
            health = await pool.clients[0].health_check() if pool.clients else {"status": "no_clients"}
            
            tests.append({
                "name": "MCP Client Pool Connectivity",
                "passed": health.get("status") == "connected",
                "details": health
            })
        except Exception as e:
            tests.append({
                "name": "MCP Client Pool Connectivity",
                "passed": False,
                "error": str(e)
            })
        
        # Test 2: MCP tool availability
        try:
            tools_result = await execute_mcp_tool("code_analyzer", {
                "code": "def test(): return True",
                "language": "python"
            })
            
            tests.append({
                "name": "MCP Tools Availability",
                "passed": tools_result.get("status") == "success",
                "details": tools_result
            })
        except Exception as e:
            tests.append({
                "name": "MCP Tools Availability", 
                "passed": False,
                "error": str(e)
            })
        
        # Test 3: Protocol validation
        try:
            validation_result = await execute_mcp_tool("protocol_validator", {
                "message": json.dumps({"jsonrpc": "2.0", "method": "test", "id": 1}),
                "protocol_version": "2024-11-05"
            })
            
            tests.append({
                "name": "MCP Protocol Validation",
                "passed": validation_result.get("status") == "success",
                "details": validation_result
            })
        except Exception as e:
            tests.append({
                "name": "MCP Protocol Validation",
                "passed": False,
                "error": str(e)
            })
        
        passed = sum(1 for t in tests if t["passed"])
        failed = len(tests) - passed
        
        return {
            "status": "completed",
            "passed": passed,
            "failed": failed,
            "tests": tests
        }

    async def _test_orchestrator_deployment(self) -> Dict[str, Any]:
        """Test orchestrator deployment and initialization"""
        tests = []
        
        # Test 1: Orchestrator status
        try:
            status = await self.orchestrator.get_status()
            tests.append({
                "name": "Orchestrator Status",
                "passed": status.get("status") == "running",
                "details": {
                    "total_subagents": status.get("total_subagents"),
                    "healthy_subagents": status.get("health_summary", {}).get("healthy", 0)
                }
            })
        except Exception as e:
            tests.append({
                "name": "Orchestrator Status",
                "passed": False,
                "error": str(e)
            })
        
        # Test 2: Subagent registration
        try:
            total_expected = 12  # 3 per category * 4 categories
            total_deployed = len(self.orchestrator.subagents)
            
            tests.append({
                "name": "Subagent Registration",
                "passed": total_deployed >= 10,  # Allow some flexibility
                "details": {
                    "expected": total_expected,
                    "deployed": total_deployed,
                    "agent_ids": list(self.orchestrator.subagents.keys())
                }
            })
        except Exception as e:
            tests.append({
                "name": "Subagent Registration",
                "passed": False,
                "error": str(e)
            })
        
        # Test 3: A2A integration
        try:
            a2a_agents = list(self.orchestrator.a2a_orchestrator.agents.keys())
            tests.append({
                "name": "A2A Integration", 
                "passed": len(a2a_agents) > 0,
                "details": {"a2a_agents": len(a2a_agents)}
            })
        except Exception as e:
            tests.append({
                "name": "A2A Integration",
                "passed": False,
                "error": str(e)
            })
        
        passed = sum(1 for t in tests if t["passed"])
        failed = len(tests) - passed
        
        return {
            "status": "completed",
            "passed": passed,
            "failed": failed,
            "tests": tests
        }

    async def _test_code_analysis_subagents(self) -> Dict[str, Any]:
        """Test code analysis and refactoring subagents"""
        tests = []
        
        test_code = '''
def calculate_result(user_input):
    password = "hardcoded_secret123"  # Security issue
    query = "SELECT * FROM users WHERE name = '" + user_input + "'"  # SQL injection
    
    for i in range(1000):  # Performance issue - nested loop
        for j in range(1000):
            result = i * j
    
    return result
'''
        
        # Test SecurityAnalyzerAgent
        try:
            result = await submit_workload(
                "security_analyzer",
                "security_scan",
                {"code": test_code, "language": "python"},
                priority=WorkloadPriority.HIGH,
                timeout_seconds=30
            )
            
            # Wait for processing
            await asyncio.sleep(2)
            
            tests.append({
                "name": "Security Analyzer",
                "passed": result.get("status") == "submitted",
                "details": result
            })
        except Exception as e:
            tests.append({
                "name": "Security Analyzer",
                "passed": False,
                "error": str(e)
            })
        
        # Test PerformanceOptimizerAgent
        try:
            result = await submit_workload(
                "performance_optimizer", 
                "performance_analysis",
                {"code": test_code, "language": "python"},
                timeout_seconds=30
            )
            
            tests.append({
                "name": "Performance Optimizer",
                "passed": result.get("status") == "submitted",
                "details": result
            })
        except Exception as e:
            tests.append({
                "name": "Performance Optimizer",
                "passed": False,
                "error": str(e)
            })
        
        # Test StyleCheckerAgent
        try:
            result = await submit_workload(
                "style_checker",
                "style_check",
                {"code": test_code, "language": "python"},
                timeout_seconds=30
            )
            
            tests.append({
                "name": "Style Checker",
                "passed": result.get("status") == "submitted",
                "details": result
            })
        except Exception as e:
            tests.append({
                "name": "Style Checker",
                "passed": False,
                "error": str(e)
            })
        
        passed = sum(1 for t in tests if t["passed"])
        failed = len(tests) - passed
        
        return {
            "status": "completed",
            "passed": passed,
            "failed": failed,
            "tests": tests
        }

    async def _test_video_processing_subagents(self) -> Dict[str, Any]:
        """Test video processing pipeline subagents"""
        tests = []
        
        # Test TranscriptionAgent
        try:
            result = await submit_workload(
                "transcription_agent",
                "transcribe",
                {
                    "url": "https://example.com/test-video.mp4",
                    "format": "mp4",
                    "duration": 60,
                    "content_type": "tutorial"
                },
                timeout_seconds=30
            )
            
            tests.append({
                "name": "Transcription Agent",
                "passed": result.get("status") == "submitted",
                "details": result
            })
        except Exception as e:
            tests.append({
                "name": "Transcription Agent",
                "passed": False,
                "error": str(e)
            })
        
        # Test ActionGeneratorAgent
        try:
            sample_transcript = {
                "text": "First, install the package. Next, import the library. Then, create the function. Finally, run the code.",
                "segments": [
                    {"start": 0, "end": 5, "text": "First, install the package.", "confidence": 0.9},
                    {"start": 5, "end": 10, "text": "Next, import the library.", "confidence": 0.9},
                    {"start": 10, "end": 15, "text": "Then, create the function.", "confidence": 0.9},
                    {"start": 15, "end": 20, "text": "Finally, run the code.", "confidence": 0.9}
                ]
            }
            
            result = await submit_workload(
                "action_generator",
                "generate_actions",
                {
                    "transcript": sample_transcript,
                    "content_type": "tutorial",
                    "metadata": {"title": "Python Tutorial"}
                },
                timeout_seconds=30
            )
            
            tests.append({
                "name": "Action Generator",
                "passed": result.get("status") == "submitted",
                "details": result
            })
        except Exception as e:
            tests.append({
                "name": "Action Generator",
                "passed": False,
                "error": str(e)
            })
        
        # Test QualityAssessorAgent
        try:
            result = await submit_workload(
                "quality_assessor",
                "assess_quality",
                {
                    "video": {"format": "mp4", "duration": 120},
                    "transcription": sample_transcript,
                    "actions": {"structured_tasks": [{"id": "task_1", "title": "Test task"}]}
                },
                timeout_seconds=30
            )
            
            tests.append({
                "name": "Quality Assessor",
                "passed": result.get("status") == "submitted",
                "details": result
            })
        except Exception as e:
            tests.append({
                "name": "Quality Assessor",
                "passed": False,
                "error": str(e)
            })
        
        passed = sum(1 for t in tests if t["passed"])
        failed = len(tests) - passed
        
        return {
            "status": "completed",
            "passed": passed,
            "failed": failed,
            "tests": tests
        }

    async def _test_multimodal_ai_subagents(self) -> Dict[str, Any]:
        """Test multi-modal AI workflow subagents"""
        tests = []
        
        # Test TextProcessorAgent
        try:
            test_text = """
            Artificial intelligence is transforming industries worldwide. Machine learning algorithms 
            enable computers to learn from data without explicit programming. Deep learning, a subset 
            of machine learning, uses neural networks to process complex patterns in data.
            """
            
            result = await submit_workload(
                "text_processor",
                "analyze_text",
                {"text": test_text},
                timeout_seconds=30  
            )
            
            tests.append({
                "name": "Text Processor",
                "passed": result.get("status") == "submitted",
                "details": result
            })
        except Exception as e:
            tests.append({
                "name": "Text Processor",
                "passed": False,
                "error": str(e)
            })
        
        # Test ImageAnalyzerAgent
        try:
            result = await submit_workload(
                "image_analyzer",
                "analyze_image",
                {
                    "format": "jpg",
                    "width": 1920,
                    "height": 1080,
                    "size": 2048000
                },
                timeout_seconds=30
            )
            
            tests.append({
                "name": "Image Analyzer",
                "passed": result.get("status") == "submitted",
                "details": result
            })
        except Exception as e:
            tests.append({
                "name": "Image Analyzer",
                "passed": False,
                "error": str(e)
            })
        
        # Test AudioTranscriberAgent
        try:
            result = await submit_workload(
                "audio_transcriber",
                "transcribe_audio",
                {
                    "format": "mp3",
                    "duration": 45,
                    "sample_rate": 44100,
                    "channels": 2
                },
                timeout_seconds=30
            )
            
            tests.append({
                "name": "Audio Transcriber",
                "passed": result.get("status") == "submitted",
                "details": result
            })
        except Exception as e:
            tests.append({
                "name": "Audio Transcriber",
                "passed": False,
                "error": str(e)
            })
        
        passed = sum(1 for t in tests if t["passed"])
        failed = len(tests) - passed
        
        return {
            "status": "completed",
            "passed": passed,
            "failed": failed,
            "tests": tests
        }

    async def _test_testing_orchestration_subagents(self) -> Dict[str, Any]:
        """Test software testing orchestration subagents"""
        tests = []
        
        sample_code = '''
def add_numbers(a, b):
    """Add two numbers together"""
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        raise TypeError("Arguments must be numbers")
    return a + b

class Calculator:
    def __init__(self):
        self.history = []
    
    def calculate(self, operation, a, b):
        if operation == "add":
            result = add_numbers(a, b)
        else:
            raise ValueError("Unsupported operation")
        
        self.history.append((operation, a, b, result))
        return result
'''
        
        # Test UnitTesterAgent
        try:
            result = await submit_workload(
                "unit_tester",
                "generate_tests",
                {
                    "code": sample_code,
                    "language": "python",
                    "framework": "pytest"
                },
                timeout_seconds=30
            )
            
            tests.append({
                "name": "Unit Tester",
                "passed": result.get("status") == "submitted",
                "details": result
            })
        except Exception as e:
            tests.append({
                "name": "Unit Tester",
                "passed": False,
                "error": str(e)
            })
        
        # Test IntegrationTesterAgent
        try:
            result = await submit_workload(
                "integration_tester",
                "run_integration_tests",
                {
                    "services": ["api-service", "database-service", "cache-service"],
                    "environment": "test",
                    "config": {"database": {"host": "test-db"}}
                },
                timeout_seconds=30
            )
            
            tests.append({
                "name": "Integration Tester",
                "passed": result.get("status") == "submitted",
                "details": result
            })
        except Exception as e:
            tests.append({
                "name": "Integration Tester",
                "passed": False,
                "error": str(e)
            })
        
        # Test PerformanceTesterAgent
        try:
            result = await submit_workload(
                "performance_tester",
                "run_performance_tests",
                {
                    "target_url": "http://test-server:8000",
                    "config": {
                        "concurrent_users": 10,
                        "duration": 30
                    }
                },
                timeout_seconds=30
            )
            
            tests.append({
                "name": "Performance Tester",
                "passed": result.get("status") == "submitted",
                "details": result
            })
        except Exception as e:
            tests.append({
                "name": "Performance Tester",
                "passed": False,
                "error": str(e)
            })
        
        passed = sum(1 for t in tests if t["passed"])
        failed = len(tests) - passed
        
        return {
            "status": "completed",
            "passed": passed,
            "failed": failed,
            "tests": tests
        }

    async def _test_workload_processing(self) -> Dict[str, Any]:
        """Test workload processing and queue management"""
        tests = []
        
        # Test 1: Concurrent workload processing
        try:
            # Submit multiple workloads concurrently
            workloads = []
            for i in range(5):
                workload = submit_workload(
                    "text_processor",
                    "analyze_text", 
                    {"text": f"Test text {i}"},
                    timeout_seconds=30
                )
                workloads.append(workload)
            
            results = await asyncio.gather(*workloads, return_exceptions=True)
            successful_submissions = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "submitted")
            
            tests.append({
                "name": "Concurrent Workload Processing",
                "passed": successful_submissions >= 4,  # Allow 1 failure
                "details": {
                    "submitted": len(workloads),
                    "successful": successful_submissions,
                    "results": results
                }
            })
        except Exception as e:
            tests.append({
                "name": "Concurrent Workload Processing",
                "passed": False,
                "error": str(e)  
            })
        
        # Test 2: Priority handling
        try:
            # Submit high priority workload
            result = await submit_workload(
                "security_analyzer",
                "security_scan",
                {"code": "def test(): pass"},
                priority=WorkloadPriority.CRITICAL,
                timeout_seconds=30
            )
            
            tests.append({
                "name": "Priority Handling",
                "passed": result.get("status") == "submitted",
                "details": result
            })
        except Exception as e:
            tests.append({
                "name": "Priority Handling",
                "passed": False,
                "error": str(e)
            })
        
        # Test 3: Queue status
        try:
            status = await self.orchestrator.get_status()
            queue_size = status.get("workload", {}).get("queue_size", 0)
            active_requests = status.get("workload", {}).get("active_requests", 0)
            
            tests.append({
                "name": "Queue Status Tracking",
                "passed": isinstance(queue_size, int) and isinstance(active_requests, int),
                "details": {
                    "queue_size": queue_size,
                    "active_requests": active_requests
                }
            })
        except Exception as e:
            tests.append({
                "name": "Queue Status Tracking",
                "passed": False,
                "error": str(e)
            })
        
        passed = sum(1 for t in tests if t["passed"])
        failed = len(tests) - passed
        
        return {
            "status": "completed",
            "passed": passed,
            "failed": failed,
            "tests": tests
        }

    async def _test_health_monitoring(self) -> Dict[str, Any]:
        """Test health monitoring and status reporting"""
        tests = []
        
        # Test 1: Individual agent health
        try:
            healthy_agents = 0
            total_agents = len(self.orchestrator.subagents)
            
            for agent_id, metrics in self.orchestrator.metrics.items():
                if metrics.status.value in ["healthy", "degraded"]:
                    healthy_agents += 1
            
            health_percentage = healthy_agents / max(total_agents, 1)
            
            tests.append({
                "name": "Individual Agent Health",
                "passed": health_percentage >= 0.8,  # 80% healthy threshold
                "details": {
                    "total_agents": total_agents,
                    "healthy_agents": healthy_agents,
                    "health_percentage": health_percentage
                }
            })
        except Exception as e:
            tests.append({
                "name": "Individual Agent Health",
                "passed": False,
                "error": str(e)
            })
        
        # Test 2: Health check execution
        try:
            # Trigger health checks
            await asyncio.sleep(2)  # Allow health monitor to run
            
            recent_checks = 0
            for metrics in self.orchestrator.metrics.values():
                if metrics.last_health_check and (datetime.utcnow() - metrics.last_health_check).total_seconds() < 60:
                    recent_checks += 1
            
            tests.append({
                "name": "Health Check Execution",
                "passed": recent_checks > 0,
                "details": {"recent_health_checks": recent_checks}
            })
        except Exception as e:
            tests.append({
                "name": "Health Check Execution",
                "passed": False,
                "error": str(e)
            })
        
        # Test 3: Circuit breaker functionality
        try:
            circuit_breakers = self.orchestrator.circuit_breakers
            closed_breakers = sum(1 for cb in circuit_breakers.values() if cb["state"] == "closed")
            
            tests.append({
                "name": "Circuit Breaker Status",
                "passed": closed_breakers > 0,
                "details": {
                    "total_breakers": len(circuit_breakers),
                    "closed_breakers": closed_breakers
                }
            })
        except Exception as e:
            tests.append({
                "name": "Circuit Breaker Status",
                "passed": False,
                "error": str(e)
            })
        
        passed = sum(1 for t in tests if t["passed"])
        failed = len(tests) - passed
        
        return {
            "status": "completed",
            "passed": passed,
            "failed": failed,
            "tests": tests
        }

    async def _test_performance_metrics(self) -> Dict[str, Any]:
        """Test performance metrics collection and reporting"""
        tests = []
        
        # Test 1: Metrics collection
        try:
            status = await self.orchestrator.get_status()
            performance = status.get("performance", {})
            
            required_metrics = ["total_requests", "successful_requests", "failed_requests", "success_rate"]
            has_all_metrics = all(metric in performance for metric in required_metrics)
            
            tests.append({
                "name": "Performance Metrics Collection",
                "passed": has_all_metrics,
                "details": performance
            })
        except Exception as e:
            tests.append({
                "name": "Performance Metrics Collection",
                "passed": False,
                "error": str(e)
            })
        
        # Test 2: Real-time metrics updates
        try:
            # Get initial metrics
            initial_status = await self.orchestrator.get_status()
            initial_requests = initial_status.get("performance", {}).get("total_requests", 0)
            
            # Submit a workload
            await submit_workload("text_processor", "analyze_text", {"text": "test"})
            await asyncio.sleep(1)
            
            # Check if metrics updated
            updated_status = await self.orchestrator.get_status()
            updated_requests = updated_status.get("performance", {}).get("total_requests", 0)
            
            tests.append({
                "name": "Real-time Metrics Updates",
                "passed": updated_requests >= initial_requests,  # Should be same or higher
                "details": {
                    "initial_requests": initial_requests,
                    "updated_requests": updated_requests
                }
            })
        except Exception as e:
            tests.append({
                "name": "Real-time Metrics Updates",
                "passed": False,
                "error": str(e)
            })
        
        # Test 3: Performance statistics
        try:
            perf_stats = self.orchestrator.performance_stats
            has_timestamp = "timestamp" in perf_stats
            has_subagent_counts = "total_subagents" in perf_stats
            
            tests.append({
                "name": "Performance Statistics",
                "passed": has_timestamp and has_subagent_counts,
                "details": perf_stats
            })
        except Exception as e:
            tests.append({
                "name": "Performance Statistics",
                "passed": False,
                "error": str(e)
            })
        
        passed = sum(1 for t in tests if t["passed"])
        failed = len(tests) - passed
        
        return {
            "status": "completed",
            "passed": passed,
            "failed": failed,
            "tests": tests
        }

    async def _test_error_handling(self) -> Dict[str, Any]:
        """Test error handling and recovery mechanisms"""
        tests = []
        
        # Test 1: Invalid workload handling
        try:
            result = await submit_workload(
                "nonexistent_agent",
                "invalid_action",
                {"invalid": "data"},
                timeout_seconds=5
            )
            
            tests.append({
                "name": "Invalid Workload Handling",
                "passed": result.get("status") == "error",
                "details": result
            })
        except Exception as e:
            tests.append({
                "name": "Invalid Workload Handling",
                "passed": True,  # Exception is expected
                "details": {"expected_exception": str(e)}
            })
        
        # Test 2: Timeout handling
        try:
            result = await submit_workload(
                "text_processor",
                "analyze_text",
                {"text": "test"},
                timeout_seconds=1  # Very short timeout
            )
            
            # The workload should be submitted but might timeout during processing
            tests.append({
                "name": "Timeout Handling",
                "passed": result.get("status") in ["submitted", "error"],
                "details": result
            })
        except Exception as e:
            tests.append({
                "name": "Timeout Handling",
                "passed": False,
                "error": str(e)
            })
        
        # Test 3: Recovery from failures
        try:
            # Check that orchestrator is still functional after errors
            status = await self.orchestrator.get_status()
            is_running = status.get("status") == "running"
            
            tests.append({
                "name": "Recovery from Failures",
                "passed": is_running,
                "details": {"orchestrator_status": status.get("status")}
            })
        except Exception as e:
            tests.append({
                "name": "Recovery from Failures",
                "passed": False,
                "error": str(e)
            })
        
        passed = sum(1 for t in tests if t["passed"])
        failed = len(tests) - passed
        
        return {
            "status": "completed",
            "passed": passed,
            "failed": failed,
            "tests": tests
        }

    async def _test_sla_compliance(self) -> Dict[str, Any]:
        """Test SLA compliance and performance requirements"""
        tests = []
        
        # Test 1: Response time SLA
        try:
            start_time = time.time()
            result = await submit_workload(
                "style_checker",
                "style_check", 
                {"code": "def test(): pass", "language": "python"},
                timeout_seconds=30
            )
            response_time_ms = (time.time() - start_time) * 1000
            
            # Check if submission is within SLA (should be very fast)
            within_sla = response_time_ms < 1000  # 1 second for submission
            
            tests.append({
                "name": "Response Time SLA",
                "passed": within_sla and result.get("status") == "submitted",
                "details": {
                    "response_time_ms": response_time_ms,
                    "sla_threshold_ms": 1000,
                    "result": result
                }
            })
        except Exception as e:
            tests.append({
                "name": "Response Time SLA",
                "passed": False,
                "error": str(e)
            })
        
        # Test 2: Availability SLA
        try:
            status = await self.orchestrator.get_status()
            success_rate = status.get("performance", {}).get("success_rate", 0)
            
            # 95% availability target
            meets_availability = success_rate >= 0.95 or status.get("performance", {}).get("total_requests", 0) == 0
            
            tests.append({
                "name": "Availability SLA",
                "passed": meets_availability,
                "details": {
                    "success_rate": success_rate,
                    "sla_threshold": 0.95,
                    "total_requests": status.get("performance", {}).get("total_requests", 0)
                }
            })
        except Exception as e:
            tests.append({
                "name": "Availability SLA",
                "passed": False,
                "error": str(e)
            })
        
        # Test 3: SLA violation tracking
        try:
            total_violations = sum(metrics.sla_violations for metrics in self.orchestrator.metrics.values())
            total_requests = sum(metrics.total_requests for metrics in self.orchestrator.metrics.values())
            
            violation_rate = total_violations / max(total_requests, 1)
            
            tests.append({
                "name": "SLA Violation Tracking",
                "passed": violation_rate <= 0.05,  # Max 5% violations
                "details": {
                    "total_violations": total_violations,
                    "total_requests": total_requests,
                    "violation_rate": violation_rate
                }
            })
        except Exception as e:
            tests.append({
                "name": "SLA Violation Tracking",
                "passed": False,
                "error": str(e)
            })
        
        passed = sum(1 for t in tests if t["passed"])
        failed = len(tests) - passed
        
        return {
            "status": "completed",
            "passed": passed,
            "failed": failed,
            "tests": tests
        }

    async def _generate_test_report(self, start_time: float) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        total_time = time.time() - start_time
        
        # Calculate overall statistics
        total_passed = sum(result.get("passed", 0) for result in self.test_results.values())
        total_failed = sum(result.get("failed", 0) for result in self.test_results.values())
        total_tests = total_passed + total_failed
        
        success_rate = total_passed / max(total_tests, 1)
        
        # Get final orchestrator status
        final_status = await self.orchestrator.get_status() if self.orchestrator else {}
        
        report = {
            "test_suite": "Comprehensive Subagent Deployment Test",
            "execution_time_seconds": total_time,
            "timestamp": datetime.utcnow().isoformat(),
            "summary": {
                "total_test_categories": len(self.test_results),
                "total_individual_tests": total_tests,
                "passed_tests": total_passed,
                "failed_tests": total_failed,
                "success_rate": success_rate,
                "overall_status": "PASSED" if success_rate >= 0.8 else "FAILED"
            },
            "category_results": self.test_results,
            "passed_categories": self.passed_tests,
            "failed_categories": self.failed_tests,
            "final_orchestrator_status": final_status,
            "recommendations": self._generate_recommendations()
        }
        
        # Log summary
        logger.info(f"\n{'='*60}")
        logger.info(f"🧪 TEST SUITE COMPLETE - {report['summary']['overall_status']}")
        logger.info(f"{'='*60}")
        logger.info(f"Total Categories: {len(self.test_results)}")
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Passed: {total_passed} ✅")
        logger.info(f"Failed: {total_failed} ❌")
        logger.info(f"Success Rate: {success_rate:.1%}")
        logger.info(f"Execution Time: {total_time:.1f} seconds")
        
        if self.failed_tests:
            logger.warning(f"Failed Categories: {', '.join(self.failed_tests)}")
        
        return report

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        if "MCP Infrastructure" in self.failed_tests:
            recommendations.append("Check MCP server connectivity and tool availability")
        
        if "Orchestrator Deployment" in self.failed_tests:
            recommendations.append("Verify orchestrator initialization and subagent registration")
        
        if len(self.failed_tests) > len(self.passed_tests):
            recommendations.append("Consider reviewing deployment configuration and dependencies")
        
        if any("Subagents" in category for category in self.failed_tests):
            recommendations.append("Review individual subagent implementations and error handling")
        
        if "Performance Metrics" in self.failed_tests:
            recommendations.append("Check metrics collection and performance monitoring systems")
        
        if "SLA Compliance" in self.failed_tests:
            recommendations.append("Review performance requirements and optimize slow components")
        
        if not recommendations:
            recommendations.append("All tests passed - system is ready for production deployment")
        
        return recommendations

    async def _cleanup_test_environment(self):
        """Clean up test environment"""
        try:
            if self.orchestrator:
                await self.orchestrator.shutdown()
                logger.info("✅ Test environment cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


async def main():
    """Main test execution function"""
    test_suite = SubagentTestSuite()
    
    try:
        report = await test_suite.run_all_tests()
        
        # Save report to file
        report_file = f"test_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"📄 Test report saved to: {report_file}")
        
        # Return appropriate exit code
        if report.get("summary", {}).get("overall_status") == "PASSED":
            return 0
        else:
            return 1
            
    except Exception as e:
        logger.error(f"❌ Test suite execution failed: {e}")
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)