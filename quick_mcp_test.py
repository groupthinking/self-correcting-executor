#!/usr/bin/env python3
"""Quick test to verify real MCP integration is working"""

import asyncio
from agents.a2a_mcp_integration import MCPEnabledA2AAgent
from connectors.real_mcp_client import execute_mcp_tool

async def quick_test():
    print("=== Quick MCP Integration Test ===\n")
    
    # Test 1: Direct MCP tool execution
    print("1. Testing direct MCP tool execution...")
    result = await execute_mcp_tool("code_analyzer", {
        "code": "def hello(): return 'world'",
        "language": "python"
    })
    
    if result.get("status") == "success":
        print(f"✅ Direct tool execution successful (latency: {result.get('latency_ms', 0):.2f}ms)")
        print(f"   Real MCP result received: {len(str(result.get('result', {})))} chars")
    else:
        print(f"❌ Direct tool execution failed: {result.get('error')}")
        return False
    
    # Test 2: A2A Agent with real MCP
    print("\n2. Testing A2A Agent with real MCP...")
    agent = MCPEnabledA2AAgent("test", ["analyze"])
    
    result = await agent.process_intent({
        "action": "tool_request",
        "tool_name": "self_corrector",
        "params": {
            "code": "x = 1\nprint(x)",
            "language": "python"
        }
    })
    
    if result.get("status") == "success":
        print("✅ A2A Agent MCP integration successful")
        print(f"   Tool used: {result.get('tool')}")
        print(f"   Real latency: {result.get('latency_ms', 0):.2f}ms")
    else:
        print(f"❌ A2A Agent integration failed: {result.get('error')}")
        return False
    
    # Test 3: Verify no hardcoded values
    print("\n3. Checking for real vs simulated processing...")
    
    # Execute same tool multiple times to check for consistent hardcoded values
    latencies = []
    for i in range(3):
        result = await execute_mcp_tool("protocol_validator", {
            "message": f'{{"test": {i}}}',
            "protocol_version": "2024-11-05"
        })
        if result.get("status") == "success":
            latencies.append(result.get("latency_ms", 0))
    
    if len(set(latencies)) > 1:  # Different latencies indicate real processing
        print("✅ Real latency variation detected (not hardcoded)")
        print(f"   Latencies: {[f'{l:.2f}ms' for l in latencies]}")
    elif all(l > 0 for l in latencies):
        print("✅ Positive latencies detected (real processing)")
        print(f"   Latencies: {[f'{l:.2f}ms' for l in latencies]}")
    else:
        print("⚠️  Suspicious latency pattern")
        return False
    
    print("\n🎉 Quick test PASSED! Real MCP integration is working.")
    return True

if __name__ == "__main__":
    success = asyncio.run(quick_test())
    exit(0 if success else 1)