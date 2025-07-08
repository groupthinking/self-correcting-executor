#!/usr/bin/env python3
"""
Script to resolve formatting conflicts in Python files.
This script specifically targets the common pattern of multi-line vs single-line f-strings.
"""

import os
import re
import sys
import subprocess


def get_file_content(file_path):
    with open(file_path, "r") as f:
        return f.read()


def save_file_content(file_path, content):
    with open(file_path, "w") as f:
        f.write(content)


def resolve_formatting_conflicts(content):
    """
    Resolve formatting conflicts in the content.
    Specifically targets the pattern:

    f"text {
        var} more text"
    f"text {var} more text"
    """
    # Pattern to match conflict blocks
    conflict_pattern = r"<<<<<<< HEAD\n(.*?)\n=======\n(.*?)\n>>>>>>> master"

    # Find all conflict markers
    conflicts = re.findall(conflict_pattern, content, re.DOTALL)

    if not conflicts:
        return content

    # Process each conflict
    for head_version, master_version in conflicts:
        # Check if this is a formatting conflict
        if ('f"' in head_version and 'f"' in master_version) or (
            "f'" in head_version and "f'" in master_version
        ):
            # Extract the actual content without formatting
            head_content = re.sub(r"\s+", " ", head_version).strip()
            master_content = re.sub(r"\s+", " ", master_version).strip()

            # Remove newlines and extra spaces from head_content
            head_content = re.sub(r"\{\s+", "{", head_content)
            head_content = re.sub(r"\s+\}", "}", head_content)

            # If the content is essentially the same, prefer the master version
            if head_content.replace(" ", "") == master_content.replace(" ", ""):
                # Replace the conflict with the master version
                content = content.replace(
                    master_version,
                )

    return content


def main():
    if len(sys.argv) < 2:
        print("Usage: python resolve_formatting_conflicts.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        sys.exit(1)

    # Get file content
    content = get_file_content(file_path)

    # Resolve conflicts
    resolved_content = resolve_formatting_conflicts(content)

    # Save resolved content
    save_file_content(file_path, resolved_content)

    # Check if there are still conflicts
    if "<<<<<<< HEAD" in resolved_content:
        print(f"Some conflicts could not be resolved automatically in {file_path}")
    else:
        print(f"All formatting conflicts resolved in {file_path}")
        # Add the file to git
        subprocess.run(["git", "add", file_path])


if __name__ == "__main__":
    main()
