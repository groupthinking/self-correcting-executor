#!/usr/bin/env python3
"""MCP Configuration Management"""

import os
import json
from typing import Dict, Any
from pathlib import Path


class MCPConfig:
    """Centralized MCP configuration"""

    def __init__(self):
        self.config_path = Path.home() / ".claude" / "mcp_config.json"
        self.default_config = {
            "endpoints": {
                "mcp_server": "stdio",  # Use stdio transport for Claude CLI
                "backup_server": "http://localhost:8080",
            },
            "servers": {
                "self-correcting-executor": {
                    "command": "python3",
                    "args": [
                        str(Path(__file__).parent.parent / "mcp_server" / "main.py")
                    ],
                    "env": {},
                }
            },
            "tools": {
                "enabled": ["code_analyzer", "protocol_validator", "self_corrector"],
                "timeout": 30,
            },
        }

    def get_config(self) -> Dict[str, Any]:
        """Get current configuration"""
        if self.config_path.exists():
            try:
                with open(self.config_path, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return self.default_config

    def get_endpoints(self) -> Dict[str, str]:
        """Get MCP server endpoints"""
        return self.get_config().get("endpoints", self.default_config["endpoints"])

    def save_config(self, config: Dict[str, Any]):
        """Save configuration"""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w") as f:
            json.dump(config, f, indent=2)

    def create_claude_config(self):
        """Create Claude Desktop configuration"""
        claude_config_path = Path.home() / ".claude" / "claude_desktop_config.json"

        claude_config = {
            "mcpServers": {
                "self-correcting-executor": {
                    "command": "python3",
                    "args": [
                        str(Path(__file__).parent.parent / "mcp_server" / "main.py")
                    ],
                }
            }
        }

        claude_config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(claude_config_path, "w") as f:
            json.dump(claude_config, f, indent=2)

        print(f"✅ Created Claude config at: {claude_config_path}")
        return claude_config_path


if __name__ == "__main__":
    config = MCPConfig()
    config.create_claude_config()
    print("🔧 MCP configuration ready")
