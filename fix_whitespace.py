#!/usr/bin/env python3
"""
Script to fix common whitespace and formatting issues in Python files.
"""

import os
import re
from pathlib import Path


def fix_whitespace_issues(file_path):
    """Fix common whitespace issues in a Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Original content
        original_content = content
        
        # Fix trailing whitespace (W291)
        lines = content.splitlines()
        fixed_lines = []
        
        for line in lines:
            # Remove trailing whitespace
            fixed_line = line.rstrip()
            fixed_lines.append(fixed_line)
        
        # Join lines back
        content = '\n'.join(fixed_lines)
        
        # Ensure file ends with newline (W292)
        if content and not content.endswith('\n'):
            content += '\n'
        
        # Only write if content changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Fixed whitespace issues in {file_path}")
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")


def main():
    """Fix whitespace issues in all Python files."""
    current_dir = Path('.')
    python_files = list(current_dir.glob('*.py'))
    
    print(f"Found {len(python_files)} Python files")
    
    for py_file in python_files:
        if py_file.name != 'fix_whitespace.py':  # Don't modify this script
            fix_whitespace_issues(py_file)
    
    print("Whitespace fixes completed!")


if __name__ == '__main__':
    main() 