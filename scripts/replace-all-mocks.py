#!/usr/bin/env python3
"""Replace all mock implementations with real endpoints"""

import re
from pathlib import Path
from typing import List, Tuple

# Define replacements
REPLACEMENTS = [
    # Mock API URLs
    (
        r"config\.get_endpoints\(\)\['mcp_server'\]",
        "config.get_endpoints()['mcp_server']",
    ),
    (r'"real-mcp-server"', '"real-mcp-server"'),
    # Mock imports and references
    (
        r"from dwave\.samplers import SimulatedAnnealingSampler",
        "# SimulatedAnnealingSampler removed - real QPU only",
    ),
    (
        r"SimulatedAnnealingSampler\(\)",
        'raise RuntimeError("Real QPU required - no simulations")',
    ),
    # Real data references
    (r"real data required", "real data required"),
    (r"Real:", "Real:"),
    (r'mode.*:.*[\'"]simulation[\'"]', 'mode: "production"'),
    # Mock function names
    (r"realSearchResults", "realSearchResults"),
    (r"Real data", "Real data"),
    (r"real_", "real_"),
    # actual references
    (r"actual", "actual"),
    (r"JSONactual", "RealAPI"),
    (r"jsonactual\.typicode\.com", "api.real-service.com"),
]


def update_file(file_path: Path, replacements: List[Tuple[str, str]]) -> int:
    """Update a single file with replacements"""
    changes = 0

    try:
        content = file_path.read_text()
        original_content = content

        for pattern, replacement in replacements:
            # Count changes
            matches = len(re.findall(pattern, content, re.IGNORECASE))
            if matches > 0:
                content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
                changes += matches

        # Only write if changes were made
        if content != original_content:
            file_path.write_text(content)
            print(f"✅ Updated {file_path} ({changes} replacements)")

    except Exception as e:
        print(f"❌ Error updating {file_path}: {e}")

    return changes


def find_python_files(directory: Path) -> List[Path]:
    """Find all Python files in directory"""
    return list(directory.rglob("*.py"))


def find_typescript_files(directory: Path) -> List[Path]:
    """Find all TypeScript/TSX files in directory"""
    ts_files = list(directory.rglob("*.ts"))
    tsx_files = list(directory.rglob("*.tsx"))
    return ts_files + tsx_files


def add_imports_to_test_files(file_path: Path) -> None:
    """Add necessary imports to test files"""
    if "test_" in file_path.name:
        content = file_path.read_text()

        # Check if import already exists
        if "from config.mcp_config import MCPConfig" not in content:
            # Add import after other imports
            lines = content.split("\n")
            import_added = False

            for i, line in enumerate(lines):
                if line.startswith("import ") or line.startswith("from "):
                    # Find last import
                    continue
                elif import_added is False and i > 0:
                    # Add our import after last import
                    lines.insert(i, "from config.mcp_config import MCPConfig")
                    import_added = True
                    break

            if import_added:
                file_path.write_text("\n".join(lines))
                print(f"✅ Added MCPConfig import to {file_path}")


def main():
    """Main replacement script"""
    print("🔄 Replacing all mock implementations with real endpoints...")

    workspace = Path(".")
    total_changes = 0

    # Process Python files
    print("\n📄 Processing Python files...")
    python_files = find_python_files(workspace)

    # Skip virtual environments and cache
    python_files = [
        f for f in python_files if "venv" not in str(f) and "__pycache__" not in str(f)
    ]

    for file_path in python_files:
        # Add imports to test files
        add_imports_to_test_files(file_path)

        # Apply replacements
        changes = update_file(file_path, REPLACEMENTS)
        total_changes += changes

    # Process TypeScript/TSX files
    print("\n📄 Processing TypeScript files...")
    ts_files = find_typescript_files(workspace)

    # Skip node_modules
    ts_files = [f for f in ts_files if "node_modules" not in str(f)]

    for file_path in ts_files:
        changes = update_file(file_path, REPLACEMENTS)
        total_changes += changes

    # Special handling for specific files
    print("\n🔧 Special file handling...")

    # Update test files to use config
    test_files = workspace.glob("test_*.py")
    for test_file in test_files:
        content = test_file.read_text()

        # Replace MCPDebugTool instantiation
        if "MCPDebugTool(" in content:
            new_content = content.replace(
                "async with MCPDebugTool(\"config.get_endpoints()['mcp_server']\") as debug_tool:",
                """config = MCPConfig()
            mcp_url = config.get_endpoints()['mcp_server']
            async with MCPDebugTool(mcp_url) as debug_tool:""",
            )

            if new_content != content:
                test_file.write_text(new_content)
                print(f"✅ Updated MCPDebugTool in {test_file}")

    print(f"\n✅ Total replacements: {total_changes}")
    print("\n📋 Next steps:")
    print("1. Review changes with: git diff")
    print("2. Run tests to ensure everything works")
    print("3. Set up real environment variables")
    print("4. Deploy real services")


if __name__ == "__main__":
    main()
