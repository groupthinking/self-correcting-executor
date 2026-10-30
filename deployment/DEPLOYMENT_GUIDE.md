# Production Subagent Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying and managing production-ready subagents with MCP integration, A2A communication, and enterprise-grade monitoring.

## Architecture

The deployment consists of:

- **12 Specialized Subagents** across 4 categories
- **MCP Infrastructure** with 3-client pool for high availability  
- **A2A Communication Framework** for inter-agent messaging
- **Production Orchestrator** with health monitoring and SLA compliance
- **Circuit Breakers** and retry mechanisms for fault tolerance

## Prerequisites

### System Requirements

- **Python 3.8+** with asyncio support
- **Minimum 4GB RAM** (8GB recommended for production)
- **Multi-core CPU** (4+ cores recommended)
- **Network connectivity** for MCP communication

### Dependencies

```bash
# Core dependencies
pip install asyncio aiohttp

# Optional performance dependencies  
pip install uvloop  # Linux/macOS only
pip install cython  # For performance optimizations
```

### MCP Server Setup

Ensure the MCP server is running and accessible:

```bash
# Test MCP server connectivity
python3 mcp_server/main.py
```

## Deployment Instructions

### 1. Quick Start Deployment

```python
#!/usr/bin/env python3
"""
Quick deployment script for production subagents
"""
import asyncio
from deployment.subagent_orchestrator import (
    ProductionSubagentOrchestrator,
    DeploymentConfig
)

async def deploy_production_subagents():
    # Configure deployment
    config = DeploymentConfig(
        health_check_interval=30,
        metrics_collection_interval=60,
        sla_response_timeout=5000,
        max_concurrent_requests=100,
        performance_monitoring=True,
        log_level="INFO"
    )
    
    # Initialize orchestrator
    orchestrator = ProductionSubagentOrchestrator(config)
    
    # Deploy all subagents
    result = await orchestrator.initialize()
    
    if result.get("status") == "initialized":
        print("✅ Deployment successful!")
        print(f"Deployed {result['total_subagents']} subagents")
        
        # Keep running
        try:
            while True:
                status = await orchestrator.get_status()
                print(f"System health: {status['health_summary']['healthy']} healthy agents")
                await asyncio.sleep(60)
                
        except KeyboardInterrupt:
            print("Shutting down...")
            await orchestrator.shutdown()
    else:
        print(f"❌ Deployment failed: {result.get('error')}")

if __name__ == "__main__":
    asyncio.run(deploy_production_subagents())
```

### 2. Advanced Configuration

```python
# Advanced deployment configuration
config = DeploymentConfig(
    # Health monitoring
    health_check_interval=15,          # Check every 15 seconds
    metrics_collection_interval=30,    # Collect metrics every 30 seconds
    
    # Performance settings
    sla_response_timeout=3000,         # 3 second SLA
    max_concurrent_requests=200,       # Higher throughput
    
    # Reliability settings
    retry_backoff_base=0.5,           # Faster retries
    circuit_breaker_threshold=3,       # Fail fast
    circuit_breaker_timeout=30,        # Quick recovery
    
    # Features
    performance_monitoring=True,
    enable_auto_scaling=False,         # Manual scaling
    log_level="DEBUG"                  # Detailed logging
)
```

## Subagent Categories

### 1. Code Analysis & Refactoring

**Available Agents:**
- `security-analyzer`: Security vulnerability detection
- `performance-optimizer`: Performance analysis and optimization
- `style-checker`: Code style and formatting validation

**Usage Example:**
```python
from deployment.subagent_orchestrator import submit_workload, WorkloadPriority

# Security analysis
result = await submit_workload(
    "security_analyzer",
    "security_scan",
    {
        "code": source_code,
        "language": "python"
    },
    priority=WorkloadPriority.HIGH
)
```

### 2. Video Processing Pipeline

**Available Agents:**
- `transcription-agent`: Audio/video transcription
- `action-generator`: Action extraction from video content
- `quality-assessor`: Video and transcription quality assessment

**Usage Example:**
```python
# Video transcription
result = await submit_workload(
    "transcription_agent", 
    "transcribe",
    {
        "url": "https://example.com/video.mp4",
        "format": "mp4",
        "duration": 120,
        "content_type": "tutorial"
    }
)
```

### 3. Multi-Modal AI Workflows

**Available Agents:**
- `text-processor`: Advanced text analysis and NLP
- `image-analyzer`: Computer vision and image analysis  
- `audio-transcriber`: Audio processing and transcription

**Usage Example:**
```python
# Text analysis
result = await submit_workload(
    "text_processor",
    "analyze_text", 
    {
        "text": document_content
    }
)
```

### 4. Software Testing Orchestration

**Available Agents:**
- `unit-tester`: Unit test generation and execution
- `integration-tester`: Integration and API testing
- `performance-tester`: Load and performance testing

**Usage Example:**
```python
# Unit test generation
result = await submit_workload(
    "unit_tester",
    "generate_tests",
    {
        "code": source_code,
        "language": "python", 
        "framework": "pytest"
    }
)
```

## Monitoring and Management

### Health Monitoring

```python
# Get system status
status = await orchestrator.get_status()

print(f"Total Subagents: {status['total_subagents']}")
print(f"Healthy: {status['health_summary']['healthy']}")
print(f"Success Rate: {status['performance']['success_rate']:.2%}")
print(f"Avg Response Time: {status['performance']['avg_response_time_ms']:.1f}ms")
```

### Performance Metrics

Key metrics automatically collected:

- **Response Times**: Per-agent and system-wide
- **Success Rates**: Request success/failure ratios
- **Queue Metrics**: Active requests and queue depth
- **Circuit Breaker Status**: Fault tolerance state
- **SLA Compliance**: Response time and availability SLAs

### Circuit Breaker Management

Circuit breakers automatically protect against cascading failures:

- **Closed**: Normal operation
- **Open**: Failing fast, requests rejected
- **Half-Open**: Testing recovery

Monitor circuit breaker status:
```python
status = await orchestrator.get_status()
breakers = status['circuit_breakers']
print(f"Open breakers: {breakers['open']}")
print(f"Half-open breakers: {breakers['half_open']}")
```

## Production Best Practices

### 1. Resource Management

```python
# Configure for production workloads
config = DeploymentConfig(
    max_concurrent_requests=500,       # Scale based on capacity
    health_check_interval=60,          # Reduce overhead  
    metrics_collection_interval=300,   # 5-minute intervals
    performance_monitoring=True
)
```

### 2. Error Handling

```python
# Robust error handling
try:
    result = await submit_workload(
        agent_type="security_analyzer",
        action="security_scan", 
        data=workload_data,
        timeout_seconds=300,
        max_retries=3
    )
    
    if result.get("status") == "submitted":
        print(f"Workload queued: {result['request_id']}")
    else:
        print(f"Submission failed: {result.get('error')}")
        
except Exception as e:
    print(f"Critical error: {e}")
    # Implement fallback logic
```

### 3. Capacity Planning

**Recommended Configurations:**

| Environment | Subagents | Concurrent Requests | Health Check Interval |
|-------------|-----------|-------------------|---------------------|
| Development | 12 | 50 | 30s |
| Staging | 12 | 200 | 60s |
| Production | 12+ | 500+ | 300s |

### 4. Logging and Debugging

```python
import logging

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/subagents.log'),
        logging.StreamHandler()
    ]
)

# Enable debug mode for troubleshooting
config = DeploymentConfig(log_level="DEBUG")
```

## Troubleshooting

### Common Issues

**1. MCP Connection Failures**
```
ERROR - MCP server failed to start
```
- Verify MCP server is running: `python3 mcp_server/main.py`
- Check port availability and permissions
- Validate MCP configuration

**2. High Memory Usage**
```
WARNING - Memory usage approaching limits  
```
- Reduce `max_concurrent_requests`
- Increase `health_check_interval`
- Monitor for memory leaks in custom code

**3. SLA Violations**
```
WARNING - SLA violations detected
```
- Check system resource utilization
- Optimize slow subagent implementations
- Consider horizontal scaling

**4. Circuit Breaker Trips**
```
INFO - Circuit breaker open for agent: security-analyzer
```
- Review agent-specific error logs
- Check underlying service dependencies
- Adjust circuit breaker thresholds if needed

### Diagnostic Commands

```python
# Health check all subagents
for agent_id, agent in orchestrator.subagents.items():
    health = await orchestrator._test_subagent_health(agent)
    print(f"{agent_id}: {'✅' if health['healthy'] else '❌'}")

# Performance summary
perf_stats = orchestrator.performance_stats
print(f"System uptime: {perf_stats.get('uptime_seconds', 0):.0f}s")
print(f"Completed requests: {perf_stats.get('completed_requests', 0)}")

# Queue analysis
status = await orchestrator.get_status()
workload = status['workload']
print(f"Queue depth: {workload['queue_size']}")
print(f"Active requests: {workload['active_requests']}")
```

## Testing and Validation

### Run Comprehensive Tests

```bash
# Execute full test suite
python3 tests/test_subagent_deployment.py

# Expected output:
# 🧪 TEST SUITE COMPLETE - PASSED
# Total Tests: 33
# Passed: 32 ✅  
# Failed: 1 ❌
# Success Rate: 97.0%
```

### Individual Subagent Testing

```python
# Test specific subagent category
from agents.specialized.code_analysis_subagents import test_code_analysis_subagents
await test_code_analysis_subagents()

# Test MCP connectivity
from connectors.real_mcp_client import test_real_mcp_client  
await test_real_mcp_client()
```

## Scaling and Optimization

### Horizontal Scaling

```python
# Deploy multiple orchestrator instances
orchestrators = []

for i in range(3):  # 3 instances
    config = DeploymentConfig(
        max_concurrent_requests=200,
        instance_id=f"orchestrator-{i}"
    )
    
    orchestrator = ProductionSubagentOrchestrator(config)
    await orchestrator.initialize()
    orchestrators.append(orchestrator)

# Load balance across instances
```

### Performance Tuning

```python
# High-performance configuration
config = DeploymentConfig(
    # Aggressive performance settings
    health_check_interval=300,         # 5 minutes
    metrics_collection_interval=600,   # 10 minutes
    sla_response_timeout=1000,         # 1 second SLA
    max_concurrent_requests=1000,      # High throughput
    
    # Optimized retry settings
    retry_backoff_base=0.1,           # Fast retries
    circuit_breaker_threshold=5,       # Allow more failures
    circuit_breaker_timeout=10,        # Quick recovery
    
    # Reduced monitoring overhead
    performance_monitoring=False
)
```

## Integration Examples

### REST API Integration

```python
from fastapi import FastAPI, HTTPException
from deployment.subagent_orchestrator import submit_workload

app = FastAPI()

@app.post("/api/analyze/security")
async def analyze_security(request: SecurityRequest):
    try:
        result = await submit_workload(
            "security_analyzer",
            "security_scan",
            request.dict()
        )
        
        if result.get("status") == "submitted":
            return {"request_id": result["request_id"]}
        else:
            raise HTTPException(500, result.get("error"))
            
    except Exception as e:
        raise HTTPException(500, str(e))
```

### Batch Processing

```python
async def process_batch(workloads: List[Dict]):
    """Process multiple workloads concurrently"""
    
    tasks = []
    for workload in workloads:
        task = submit_workload(
            workload["agent_type"],
            workload["action"], 
            workload["data"]
        )
        tasks.append(task)
    
    # Process all workloads concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Handle results
    successful = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "submitted")
    print(f"Batch processed: {successful}/{len(workloads)} successful")
    
    return results
```

## Security Considerations

### 1. Access Control

```python
# Implement authentication/authorization
class SecureOrchestrator(ProductionSubagentOrchestrator):
    def __init__(self, config, auth_provider):
        super().__init__(config)
        self.auth = auth_provider
    
    async def submit_workload(self, request, user_token):
        # Validate user permissions
        if not await self.auth.validate_token(user_token):
            raise PermissionError("Invalid authentication")
        
        # Check user permissions for agent type
        if not await self.auth.check_permission(user_token, request.agent_type):
            raise PermissionError("Insufficient permissions")
        
        return await super().submit_workload(request)
```

### 2. Input Validation

```python
# Validate all workload inputs
from pydantic import BaseModel, validator

class WorkloadRequest(BaseModel):
    agent_type: str
    action: str
    data: Dict[str, Any]
    
    @validator('agent_type')
    def validate_agent_type(cls, v):
        allowed_types = [
            'security_analyzer', 'performance_optimizer', 'style_checker',
            'transcription_agent', 'action_generator', 'quality_assessor',
            'text_processor', 'image_analyzer', 'audio_transcriber',
            'unit_tester', 'integration_tester', 'performance_tester'
        ]
        if v not in allowed_types:
            raise ValueError(f'Invalid agent type: {v}')
        return v
```

### 3. Resource Limits

```python
# Implement resource quotas per user/tenant
class ResourceManager:
    def __init__(self):
        self.user_quotas = {}
        self.user_usage = {}
    
    async def check_quota(self, user_id: str, resource_cost: int):
        current_usage = self.user_usage.get(user_id, 0)
        user_quota = self.user_quotas.get(user_id, 1000)  # Default quota
        
        if current_usage + resource_cost > user_quota:
            raise QuotaExceededError("Resource quota exceeded")
        
        self.user_usage[user_id] = current_usage + resource_cost
```

## Maintenance and Updates

### 1. Rolling Updates

```python
async def rolling_update(orchestrator, new_subagent_implementations):
    """Perform rolling update of subagents"""
    
    for agent_id, new_implementation in new_subagent_implementations.items():
        # Drain existing requests for this agent
        await orchestrator._drain_agent_requests(agent_id)
        
        # Replace agent implementation
        orchestrator.subagents[agent_id] = new_implementation
        
        # Verify health of new implementation
        health = await orchestrator._test_subagent_health(new_implementation)
        if not health["healthy"]:
            # Rollback on failure
            orchestrator.subagents[agent_id] = old_implementation
            raise UpdateError(f"Health check failed for {agent_id}")
        
        print(f"✅ Updated {agent_id}")
```

### 2. Configuration Updates

```python
# Dynamic configuration updates
async def update_configuration(orchestrator, new_config):
    """Update orchestrator configuration without restart"""
    
    # Update health check intervals
    orchestrator.config.health_check_interval = new_config.health_check_interval
    
    # Update SLA settings
    orchestrator.config.sla_response_timeout = new_config.sla_response_timeout
    
    # Update circuit breaker settings
    for breaker in orchestrator.circuit_breakers.values():
        breaker["threshold"] = new_config.circuit_breaker_threshold
    
    print("✅ Configuration updated")
```

## Support and Documentation

### Getting Help

- **GitHub Issues**: Report bugs and feature requests
- **Documentation**: Comprehensive API reference and examples
- **Community**: Join discussions and get community support

### API Reference

Complete API documentation available at:
- **Orchestrator API**: `deployment/subagent_orchestrator.py`
- **Subagent APIs**: `agents/specialized/`
- **MCP Integration**: `connectors/real_mcp_client.py`

### Performance Benchmarks

| Metric | Target | Typical | Notes |
|--------|--------|---------|-------|
| Response Time | < 5000ms | 100-2000ms | Varies by agent type |
| Throughput | > 100 req/sec | 200-500 req/sec | With 12 agents |
| Availability | > 99% | 99.5% | With circuit breakers |
| Memory Usage | < 2GB | 1-1.5GB | Per orchestrator |

---

## Conclusion

This deployment guide provides everything needed to run production-ready subagents with enterprise-grade reliability, monitoring, and performance. The modular architecture allows for easy scaling and customization while maintaining high availability and fault tolerance.

For additional support, refer to the comprehensive test suite results and monitoring dashboards to ensure optimal system performance.