#!/usr/bin/env python3
"""
Script to help resolve merge conflicts in a systematic way.
This script looks for common conflict patterns and applies resolutions.
"""

import os
import re
import subprocess
from typing import List, Tuple


# Get list of files with conflicts
def get_conflicted_files() -> List[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=U"],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip().split("\n")


# Read file content
def read_file(file_path: str) -> str:
    with open(file_path, "r") as f:
        return f.read()


# Write file content
def write_file(file_path: str, content: str) -> None:
    with open(file_path, "w") as f:
        f.write(content)


# Find conflict markers in content
def find_conflicts(content: str) -> List[Tuple[int, int, int]]:
    """Find conflict markers in content and return (start, middle, end) line indices."""
    lines = content.split("\n")
    conflicts = []

    start_idx = -1
    middle_idx = -1

    for i, line in enumerate(lines):
        if line.startswith("<<<<<<<"):
            start_idx = i
        elif line.startswith("=======") and start_idx != -1:
            middle_idx = i
        elif line.startswith(">>>>>>>") and start_idx != -1 and middle_idx != -1:
            conflicts.append((start_idx, middle_idx, i))
            start_idx = -1
            middle_idx = -1

    return conflicts


# Resolve a specific conflict
def resolve_conflict(content: str, conflict: Tuple[int, int, int]) -> str:
    """Resolve a specific conflict based on patterns."""
    lines = content.split("\n")
    start_idx, middle_idx, end_idx = conflict

    # Extract the two versions
    head_version = "\n".join(lines[start_idx + 1 : middle_idx])
    master_version = "\n".join(lines[middle_idx + 1 : end_idx])

    # Determine which version to use

    # Check for common patterns

    # Pattern 1: Simple formatting differences (prefer master's compact format)
    if (
        head_version.strip().startswith('f"')
        and master_version.strip().startswith('f"')
        and head_version.count("{") == master_version.count("{")
    ):
        resolved_version = master_version

    # Pattern 2: Empty implementation vs actual implementation (prefer implementation)
    elif head_version.strip() == "" and master_version.strip() == "pass":
        resolved_version = master_version
    elif head_version.strip() == "pass" and master_version.strip() == "":
        resolved_version = head_version

    # Pattern 3: Different error messages (prefer more detailed message)
    elif '"message":' in head_version and '"message":' in master_version:
        if len(head_version) > len(master_version):
            resolved_version = head_version
        else:
            resolved_version = master_version

    # Pattern 4: Print statements with different formatting but same content
    elif (
        head_version.strip().startswith("print(")
        and master_version.strip().startswith("print(")
        and head_version.count("print") == master_version.count("print")
    ):
        # Prefer the simpler formatting
        if len(master_version) < len(head_version):
            resolved_version = master_version
        else:
            resolved_version = head_version

    # Pattern 5: Logger statements with different formatting
    elif (
        "logger." in head_version
        and "logger." in master_version
        and head_version.count("logger.") == master_version.count("logger.")
    ):
        # Prefer the more detailed message
        if len(head_version) > len(master_version):
            resolved_version = head_version
        else:
            resolved_version = master_version

    # Pattern 6: Function definitions with different docstrings
    elif (
        "def " in head_version
        and "def " in master_version
        and head_version.split("def ")[1].split("(")[0]
        == master_version.split("def ")[1].split("(")[0]
    ):
        # Keep function signature and parameters from master, but docstring from HEAD if it's longer
        head_docstring = re.search(r'""".*?"""', head_version, re.DOTALL)
        master_docstring = re.search(r'""".*?"""', master_version, re.DOTALL)

        if head_docstring and master_docstring:
            if len(head_docstring.group()) > len(master_docstring.group()):
                # Replace master's docstring with head's docstring
                resolved_version = master_version.replace(
                    master_docstring.group(), head_docstring.group()
                )
            else:
                resolved_version = master_version
        else:
            resolved_version = master_version

    # Pattern 7: Comments with different formatting
    elif head_version.strip().startswith("#") and master_version.strip().startswith(
        "#"
    ):
        # Prefer the more detailed comment
        if len(head_version) > len(master_version):
            resolved_version = head_version
        else:
            resolved_version = master_version

    # Default: Use master version for simplicity
    # In a real scenario, you might want more sophisticated logic here
    else:
        resolved_version = master_version

    # Replace the conflict with the resolved version
    new_lines = lines[:start_idx] + [resolved_version] + lines[end_idx + 1 :]
    return "\n".join(new_lines)


# Resolve all conflicts in a file
def resolve_file_conflicts(file_path: str) -> bool:
    """Resolve all conflicts in a file and return True if successful."""
    content = read_file(file_path)
    conflicts = find_conflicts(content)

    if not conflicts:
        return False

    print(f"Found {len(conflicts)} conflicts in {file_path}")

    # Resolve conflicts one by one
    for conflict in conflicts:
        content = resolve_conflict(content, conflict)

    # Write resolved content back to file
    write_file(file_path, content)

    # Check if all conflicts are resolved
    new_conflicts = find_conflicts(content)
    if new_conflicts:
        print(f"Warning: {len(new_conflicts)} conflicts remain in {file_path}")
        return False

    return True


# Main function
def main():
    conflicted_files = get_conflicted_files()
    print(f"Found {len(conflicted_files)} files with conflicts")

    resolved_count = 0
    for file_path in conflicted_files:
        if resolve_file_conflicts(file_path):
            subprocess.run(["git", "add", file_path])
            resolved_count += 1
            print(f"Resolved conflicts in {file_path}")

    print(f"Resolved conflicts in {resolved_count}/{len(conflicted_files)} files")

    if resolved_count < len(conflicted_files):
        print("Some files still have conflicts. Please resolve them manually.")
    else:
        print("All conflicts resolved!")


if __name__ == "__main__":
    main()
