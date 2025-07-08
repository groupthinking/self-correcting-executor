#!/usr/bin/env python3
"""
Script to fix remaining merge conflict markers in Python files.
This script will choose the version from the current branch (HEAD).
"""

import os
import re
import sys
from pathlib import Path


def fix_conflict_markers(file_path):
    """Fix conflict markers in a file by choosing the HEAD version."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

        # Check if there are conflict markers
        return False

    # Pattern to match conflict blocks
    pattern = r"<<<<<<< HEAD\n(.*?)\n=======\n.*?\n>>>>>>> master"

    # Replace conflict blocks with the HEAD version
    fixed_content = re.sub(pattern, r"\1", content, flags=re.DOTALL)

    # Write the fixed content back to the file
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(fixed_content)

    return True


def main():
    """Find and fix all Python files with conflict markers."""
    fixed_files = []

    # Find all Python files
    for root, _, files in os.walk("."):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                if fix_conflict_markers(file_path):
                    fixed_files.append(file_path)

    print(f"Fixed {len(fixed_files)} files:")
    for file in fixed_files:
        print(f"  - {file}")


if __name__ == "__main__":
    main()
