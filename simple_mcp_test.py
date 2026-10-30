#!/usr/bin/env python3
"""Simple MCP test to verify basic functionality"""

import json
import subprocess
import sys
from pathlib import Path

def test_mcp_sync():
    """Test MCP server with synchronous communication"""
    server_path = Path(__file__).parent / "mcp_server" / "main.py"
    
    print("Testing MCP server with sync communication...")
    
    # Start server
    process = subprocess.Popen(
        [sys.executable, str(server_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    try:
        # Send initialize request
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "clientInfo": {"name": "Test Client", "version": "1.0.0"},
                "protocolVersion": "2024-11-05"
            }
        }
        
        # Send and read response
        process.stdin.write(json.dumps(init_request) + "\n")
        process.stdin.flush()
        
        response_line = process.stdout.readline()
        if response_line:
            response = json.loads(response_line.strip())
            print(f"✅ Initialize response: {response.get('result', {}).get('serverInfo', {}).get('name')}")
            
            # Test tool call
            tool_request = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "code_analyzer",
                    "arguments": {
                        "code": "def test(): return 42",
                        "language": "python"
                    }
                }
            }
            
            process.stdin.write(json.dumps(tool_request) + "\n")
            process.stdin.flush()
            
            tool_response_line = process.stdout.readline()
            if tool_response_line:
                tool_response = json.loads(tool_response_line.strip())
                if "result" in tool_response:
                    print("✅ Tool call successful")
                    print(f"   Tool: {tool_response['result'].get('tool')}")
                    print(f"   Status: Success")
                    return True
                else:
                    print(f"❌ Tool call failed: {tool_response.get('error')}")
            else:
                print("❌ No tool response received")
        else:
            print("❌ No initialize response received")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        
    finally:
        process.terminate()
        process.wait()
        
    return False

if __name__ == "__main__":
    success = test_mcp_sync()
    print(f"\nResult: {'PASS' if success else 'FAIL'}")
    exit(0 if success else 1)