# File System Agent
# Specialized agent for securely interacting with the project file system.

import os
from datetime import datetime
from typing import Dict, Any


class FileSystemAgent:
    """Agent for safe, read-only file system operations."""

    def __init__(self, base_path="/app"):
        self.name = "file_system_agent"
        # Security: Ensure all operations are constrained to this base path.
        self.base_path = os.path.abspath(base_path)

    def _is_safe_path(self, path: str) -> bool:
        """Security check to prevent path traversal attacks."""
        requested_path = os.path.abspath(os.path.join(self.base_path, path))
        return requested_path.startswith(self.base_path)

    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute file system action based on inputs."""
        action = inputs.get("action")
        path = inputs.get("path", ".")

        if not self._is_safe_path(path):
            return {
                "success": False,
                "error": "Access denied: Path is outside the allowed project directory.",
                "timestamp": datetime.utcnow().isoformat(),
            }

        full_path = os.path.join(self.base_path, path)

        if action == "list_directory":
            return await self._list_directory(full_path)
        elif action == "read_file":
            return await self._read_file(full_path)
        else:
            return {
                "success": False,
                "error": f"Unknown file system action: {action}",
                "timestamp": datetime.utcnow().isoformat(),
            }

    async def _list_directory(self, path: str) -> Dict[str, Any]:
        """List contents of a directory."""
        try:
            if not os.path.isdir(path):
                return {"success": False, "error": "Not a directory"}

            items = []
            for item in os.listdir(path):
                item_path = os.path.join(path, item)
                if not item.startswith(".") and not item.startswith(
                    "__"
                ):  # Exclude hidden files/dirs
                    is_dir = os.path.isdir(item_path)
                    items.append(
                        {
                            "name": item,
                            "path": os.path.relpath(item_path, self.base_path),
                            "type": "directory" if is_dir else "file",
                        }
                    )

            # Sort with directories first
            items.sort(key=lambda x: (x["type"] != "directory", x["name"]))

            return {
                "success": True,
                "path": os.path.relpath(path, self.base_path),
                "items": items,
                "timestamp": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _read_file(self, path: str) -> Dict[str, Any]:
        """Read content of a file."""
        try:
            if not os.path.isfile(path):
                return {"success": False, "error": "Not a file"}

            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

            return {
                "success": True,
                "path": os.path.relpath(path, self.base_path),
                "content": content,
                "size_bytes": os.path.getsize(path),
                "timestamp": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


# Export the agent instance
file_system_agent = FileSystemAgent()
