#!/usr/bin/env python3
"""
Test Real MCP Integration
========================

This script thoroughly tests the real MCP integration to ensure all simulated
processing has been replaced with actual MCP server communication.
"""

import asyncio
import logging
import time
from typing import Dict, Any

# Import components to test
from agents.a2a_mcp_integration import (
    MCPEnabledA2AAgent, 
    A2AMCPOrchestrator,
    MessagePriority
)
from connectors.real_mcp_client import MCPClient, execute_mcp_tool, get_mcp_client_pool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_mcp_client_basic():
    """Test 1: Basic MCP client functionality"""
    print("=== Test 1: Basic MCP Client ===")
    
    client = MCPClient()
    try:
        # Test connection
        connected = await client.connect()
        assert connected, "Failed to connect to MCP server"
        print("✅ MCP client connection successful")
        
        # Test tool listing
        tools = await client.list_tools()
        assert len(tools) > 0, "No tools available"
        print(f"✅ Found {len(tools)} tools")
        
        # Test tool execution
        result = await client.call_tool("code_analyzer", {
            "code": "def test(): return 'hello'",
            "language": "python"
        })
        assert result["status"] == "success", "Tool execution failed"
        assert "latency_ms" in result, "No latency measurement"
        print(f"✅ Tool execution successful (latency: {result['latency_ms']:.2f}ms)")
        
        # Test health check
        health = await client.health_check()
        assert health["status"] == "connected", "Health check failed"
        print("✅ Health check passed")
        
        return True
        
    finally:
        await client.disconnect()


async def test_mcp_client_pool():
    """Test 2: MCP client pool functionality"""
    print("\n=== Test 2: MCP Client Pool ===")
    
    try:
        # Test pool execution
        result = await execute_mcp_tool("protocol_validator", {
            "message": '{"jsonrpc": "2.0", "method": "test", "id": 1}',
            "protocol_version": "2024-11-05"
        })
        
        assert result["status"] == "success", "Pool execution failed"
        print("✅ Client pool execution successful")
        
        # Get pool stats
        pool = await get_mcp_client_pool()
        stats = pool.stats
        assert stats["total_requests"] > 0, "No requests recorded"
        print(f"✅ Pool stats tracking: {stats['total_requests']} requests")
        
        return True
        
    except Exception as e:
        logger.error(f"Client pool test failed: {e}")
        return False


async def test_a2a_agent_real_mcp():
    """Test 3: A2A agent with real MCP integration"""
    print("\n=== Test 3: A2A Agent Real MCP ===")
    
    try:
        # Create agent
        agent = MCPEnabledA2AAgent("test_agent", ["analyze", "generate"])
        
        # Test data analysis with real MCP
        result = await agent.process_intent({
            "action": "analyze_data",
            "data": {
                "code": "x = 1 + 1\nprint(x)",
                "language": "python"
            }
        })
        
        assert result["status"] != "error", f"Analysis failed: {result}"
        assert "mcp_result" in result or "analysis_type" in result, "No MCP processing evidence"
        print("✅ Real MCP data analysis successful")
        
        # Test code generation with real MCP
        result = await agent.process_intent({
            "action": "generate_code",
            "data": {
                "type": "function",
                "language": "python",
                "description": "Simple calculator function"
            }
        })
        
        assert result["status"] != "error", f"Code generation failed: {result}"
        assert "code" in result, "No code generated"
        print("✅ Real MCP code generation successful")
        
        # Test tool request
        result = await agent.process_intent({
            "action": "tool_request",
            "tool_name": "self_corrector",
            "params": {
                "code": "def buggy_func():\n    x = \n    return x",
                "language": "python"
            }
        })
        
        assert result["status"] != "error", f"Tool request failed: {result}"
        print("✅ Real MCP tool request successful")
        
        return True
        
    except Exception as e:
        logger.error(f"A2A agent test failed: {e}")
        return False


async def test_message_transport():
    """Test 4: Real MCP message transport"""
    print("\n=== Test 4: Real MCP Message Transport ===")
    
    try:
        # Create agents
        sender = MCPEnabledA2AAgent("sender", ["send"])
        receiver = MCPEnabledA2AAgent("receiver", ["receive"])
        
        # Set up orchestrator
        orchestrator = A2AMCPOrchestrator()
        orchestrator.register_agent(sender)
        orchestrator.register_agent(receiver)
        
        # Start orchestrator briefly
        tasks = await orchestrator.start()
        
        # Test contextualized message with real MCP pipe
        start_time = time.time()
        result = await sender.send_contextualized_message(
            recipient="receiver",
            intent={
                "action": "analyze_data",
                "data": {"test": "data"}
            },
            priority=MessagePriority.HIGH
        )
        
        assert "latency_ms" in result, "No real latency measurement"
        assert result["latency_ms"] > 0, "Latency should be > 0 for real processing"
        print(f"✅ Real message transport (latency: {result['latency_ms']:.2f}ms)")
        
        # Test different transport strategies
        large_data = {"data": "x" * 10000}  # Large message for shared memory test
        result = await sender.send_contextualized_message(
            recipient="receiver",
            intent={
                "action": "process",
                "data": large_data
            }
        )
        
        assert "transport_strategy" in result, "No transport strategy info"
        print(f"✅ Transport strategy selected: {result.get('transport_strategy')}")
        
        # Stop orchestrator
        await orchestrator.stop()
        for task in tasks:
            if not task.done():
                task.cancel()
        
        return True
        
    except Exception as e:
        logger.error(f"Message transport test failed: {e}")
        return False


async def test_performance_monitoring():
    """Test 5: Real performance monitoring"""
    print("\n=== Test 5: Real Performance Monitoring ===")
    
    try:
        orchestrator = A2AMCPOrchestrator()
        
        # Start monitoring
        tasks = await orchestrator.start()
        
        # Generate some activity
        agent = MCPEnabledA2AAgent("monitor_test", ["test"])
        orchestrator.register_agent(agent)
        
        # Execute multiple operations to generate metrics
        for i in range(3):
            await agent.process_intent({
                "action": "tool_request",
                "tool_name": "code_analyzer",
                "params": {"code": f"# Test {i}", "language": "python"}
            })
        
        # Wait for stats update
        await asyncio.sleep(1)
        
        # Check performance stats
        stats = orchestrator.get_performance_stats()
        assert "total_messages" in stats, "No message count tracking"
        assert "avg_latency_ms" in stats, "No latency tracking"
        assert "active_connections" in stats, "No connection tracking"
        
        print(f"✅ Performance monitoring active:")
        print(f"   - Total requests: {stats.get('total_messages', 0)}")
        print(f"   - Avg latency: {stats.get('avg_latency_ms', 0):.2f}ms")
        print(f"   - Active connections: {stats.get('active_connections', 0)}")
        
        # Stop orchestrator
        await orchestrator.stop()
        for task in tasks:
            if not task.done():
                task.cancel()
        
        return True
        
    except Exception as e:
        logger.error(f"Performance monitoring test failed: {e}")
        return False


async def test_error_handling():
    """Test 6: Error handling and retry mechanisms"""
    print("\n=== Test 6: Error Handling & Retry ===")
    
    try:
        # Test with invalid tool name (should fail gracefully)
        result = await execute_mcp_tool("nonexistent_tool", {})
        assert result["status"] == "error", "Should fail for nonexistent tool"
        print("✅ Invalid tool handled gracefully")
        
        # Test with invalid parameters (should fail gracefully)
        result = await execute_mcp_tool("code_analyzer", {"invalid": "params"})
        # Should either succeed with error handling or fail gracefully
        assert "status" in result, "Result should have status field"
        print("✅ Invalid parameters handled gracefully")
        
        # Test connection retry (simulated by creating new client)
        client = MCPClient("/nonexistent/path/to/server.py")  # Invalid path
        connected = await client.connect()
        assert not connected, "Should fail to connect to nonexistent server"
        print("✅ Connection failure handled gracefully")
        
        return True
        
    except Exception as e:
        logger.error(f"Error handling test failed: {e}")
        return False


async def run_all_tests():
    """Run all tests and report results"""
    print("🚀 Starting Real MCP Integration Tests\n")
    
    tests = [
        ("Basic MCP Client", test_mcp_client_basic),
        ("MCP Client Pool", test_mcp_client_pool),
        ("A2A Agent Real MCP", test_a2a_agent_real_mcp),
        ("Message Transport", test_message_transport),
        ("Performance Monitoring", test_performance_monitoring),
        ("Error Handling", test_error_handling),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            start_time = time.time()
            success = await test_func()
            duration = time.time() - start_time
            
            results.append({
                "name": test_name,
                "success": success,
                "duration": duration
            })
            
        except Exception as e:
            logger.error(f"Test '{test_name}' crashed: {e}")
            results.append({
                "name": test_name,
                "success": False,
                "duration": 0,
                "error": str(e)
            })
    
    # Report results
    print("\n" + "="*50)
    print("📊 TEST RESULTS SUMMARY")
    print("="*50)
    
    passed = sum(1 for r in results if r["success"])
    total = len(results)
    
    for result in results:
        status = "✅ PASS" if result["success"] else "❌ FAIL"
        duration = f"{result['duration']:.2f}s"
        print(f"{status} {result['name']:.<35} {duration}")
        
        if not result["success"] and "error" in result:
            print(f"     Error: {result['error']}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Real MCP integration is working correctly.")
        return True
    else:
        print(f"\n⚠️  {total-passed} tests failed. Please review the errors above.")
        return False


if __name__ == "__main__":
    asyncio.run(run_all_tests())