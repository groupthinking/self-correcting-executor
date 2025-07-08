#!/usr/bin/env python3
"""
Script to fix Python files with syntax errors.
"""

import os
import re
import sys
import subprocess
from pathlib import Path


def fix_file(file_path):
    """Fix a Python file with syntax errors."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Remove conflict markers
    content = re.sub(r"<<<<<<< HEAD\n", "", content)
    content = re.sub(r"=======\n.*?>>>>>>> master\n", "", content, flags=re.DOTALL)

    # Fix common indentation issues
    lines = content.split("\n")
    fixed_lines = []
    current_indent = 0
    for line in lines:
        # Skip empty lines
        if not line.strip():
            fixed_lines.append(line)
            continue

        # Calculate indentation
        indent = len(line) - len(line.lstrip())
        if indent % 4 != 0 and line.strip():
            # Fix indentation to be a multiple of 4
            new_indent = (indent // 4) * 4
            line = " " * new_indent + line.lstrip()

        fixed_lines.append(line)

    fixed_content = "\n".join(fixed_lines)

    # Write the fixed content back to the file
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(fixed_content)

    return True


def main():
    """Find and fix all Python files that fail black formatting."""
    # Get list of files that fail black formatting
    result = subprocess.run(["black", "--check", "."], capture_output=True, text=True)

    if result.returncode == 0:
        print("All files already formatted correctly.")
        return

    # Extract file paths from black output
    file_pattern = r"cannot format (.*?): Cannot parse"
    problematic_files = re.findall(file_pattern, result.stderr)

    if not problematic_files:
        print("No problematic files found.")
        return

    print(f"Found {len(problematic_files)} problematic files:")
    for file in problematic_files:
        print(f"  - {file}")
        fix_file(file)

    print("\nFixed files. Running black again...")
    subprocess.run(["black", "."])


if __name__ == "__main__":
    main()
