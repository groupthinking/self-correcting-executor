#!/usr/bin/env python3
"""
Script to resolve merge conflicts systematically
"""

import os
import re
import subprocess
from pathlib import Path

def resolve_conflict_in_file(filepath):
    """Resolve conflicts in a single file"""
    print(f"Resolving conflicts in {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern to match conflict markers
    conflict_pattern = r'<<<<<<< HEAD\n(.*?)\n=======\n(.*?)\n>>>>>>> master'
    
    def resolve_conflict(match):
        head_content = match.group(1)
        master_content = match.group(2)
        
        # For most cases, prefer HEAD content (the PR branch) as it's more comprehensive
        # But merge complementary parts
        
        # Special handling for specific patterns
        if 'MCP_SERVER_URL' in head_content or 'MCP_SERVER_URL' in master_content:
            # Merge MCP configuration
            if 'MCP_SERVER_URL' in master_content and 'MCP_SERVER_URL' not in head_content:
                return head_content + '\n' + master_content
            else:
                return head_content
        
        # For formatting differences, prefer HEAD
        if head_content.strip() == master_content.strip():
            return head_content
        
        # For code formatting (quotes, spacing), prefer HEAD
        if '"' in head_content and "'" in master_content:
            return head_content
        
        # Default: prefer HEAD content (PR branch)
        return head_content
    
    # Resolve all conflicts
    resolved_content = re.sub(conflict_pattern, resolve_conflict, content, flags=re.DOTALL)
    
    # Write back resolved content
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(resolved_content)
    
    print(f"Resolved conflicts in {filepath}")

def main():
    """Main function to resolve all conflicts"""
    
    # Get list of files with conflicts
    result = subprocess.run(['git', 'diff', '--name-only', '--diff-filter=U'], 
                          capture_output=True, text=True)
    
    if result.returncode != 0:
        print("Error getting conflict files")
        return
    
    conflict_files = result.stdout.strip().split('\n')
    conflict_files = [f for f in conflict_files if f.strip()]
    
    if not conflict_files:
        print("No conflict files found")
        return
    
    print(f"Found {len(conflict_files)} files with conflicts:")
    for f in conflict_files:
        print(f"  - {f}")
    
    # Resolve each file
    for filepath in conflict_files:
        if os.path.exists(filepath):
            resolve_conflict_in_file(filepath)
        else:
            print(f"File not found: {filepath}")
    
    print("All conflicts resolved!")

if __name__ == "__main__":
    main()