#!/usr/bin/env python3
"""
Script to fix common flake8 issues in the Schwabot codebase.
"""

import os
import re
from pathlib import Path

def fix_line_length(file_path: str, max_length: int = 100) -> None:
    """Fix lines that are too long by breaking them appropriately."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    fixed_lines = []
    
    for line in lines:
        if len(line) > max_length:
            # Try to break at logical points
            if 'import' in line and ',' in line:
                # Break import statements
                parts = line.split('import')
                if len(parts) == 2:
                    imports = parts[1].strip()
                    if imports.startswith('(') and imports.endswith(')'):
                        # Multi-line import
                        fixed_lines.append(parts[0] + 'import (')
                        import_items = imports[1:-1].split(',')
                        for item in import_items:
                            item = item.strip()
                            if item:
                                fixed_lines.append('    ' + item + ',')
                        fixed_lines.append(')')
                        continue
            
            # Try to break at operators
            if any(op in line for op in [' + ', ' - ', ' * ', ' / ', ' = ', ' == ', ' != ']):
                # Find the last operator before max_length
                last_break = 0
                for i, char in enumerate(line[:max_length]):
                    if char in ['+', '-', '*', '/', '=', '!'] and i > 0 and line[i-1] == ' ':
                        last_break = i
                
                if last_break > 0:
                    fixed_lines.append(line[:last_break])
                    fixed_lines.append('    ' + line[last_break:])
                    continue
            
            # Try to break at parentheses
            if '(' in line and ')' in line:
                open_paren = line.find('(')
                if open_paren < max_length:
                    # Find matching closing parenthesis
                    paren_count = 0
                    close_paren = -1
                    for i, char in enumerate(line[open_paren:], open_paren):
                        if char == '(':
                            paren_count += 1
                        elif char == ')':
                            paren_count -= 1
                            if paren_count == 0:
                                close_paren = i
                                break
                    
                    if close_paren > 0 and close_paren < max_length:
                        # Can fit on one line
                        fixed_lines.append(line)
                        continue
            
            # If all else fails, break at spaces
            if len(line) > max_length:
                words = line.split(' ')
                current_line = ''
                for word in words:
                    if len(current_line + ' ' + word) <= max_length:
                        current_line += (' ' + word) if current_line else word
                    else:
                        if current_line:
                            fixed_lines.append(current_line)
                        current_line = word
                if current_line:
                    fixed_lines.append(current_line)
        else:
                fixed_lines.append(line)
        else:
            fixed_lines.append(line)
    
    # Write back to file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(fixed_lines))

def remove_unused_imports(file_path: str) -> None:
    """Remove unused imports from a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Common unused imports to remove
    unused_imports = [
        'import os',
        'import shutil', 
        'import json',
        'import logging',
        'import time',
        'import math',
        'import random',
        'import hashlib',
        'import threading',
        'import collections',
        'import datetime',
        'import enum',
        'import pathlib',
        'import requests',
        'import aiohttp',
        'import pandas as pd',
        'import numpy as np',
        'import scipy',
        'from typing import List',
        'from typing import Optional',
        'from typing import Tuple',
        'from typing import Union',
        'from dataclasses import dataclass',
        'from dataclasses import field',
        'from datetime import datetime',
        'from datetime import timedelta',
        'from enum import Enum',
        'from pathlib import Path',
        'from collections import deque',
    ]
    
    lines = content.split('\n')
    fixed_lines = []
    
    for line in lines:
        line_stripped = line.strip()
        should_skip = False
        
        for unused_import in unused_imports:
            if line_stripped.startswith(unused_import):
                should_skip = True
                break
        
        if not should_skip:
            fixed_lines.append(line)
    
    # Write back to file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(fixed_lines))

def fix_whitespace_issues(file_path: str) -> None:
    """Fix whitespace issues like trailing whitespace and blank lines with spaces."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
        fixed_lines = []

        for line in lines:
        # Remove trailing whitespace
        line = line.rstrip()
        
        # Skip lines that are just whitespace
        if line.strip() == '':
            fixed_lines.append('')
            else:
                fixed_lines.append(line)

    # Write back to file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(fixed_lines))

def main():
    """Main function to fix flake8 issues."""
    core_dir = Path('core')
    
    if not core_dir.exists():
        print("Core directory not found!")
        return
    
    # Get all Python files
    python_files = list(core_dir.rglob('*.py'))
    
    print(f"Found {len(python_files)} Python files to process")
    
    for file_path in python_files:
        print(f"Processing {file_path}")
        
        try:
            # Fix whitespace issues first
            fix_whitespace_issues(str(file_path))
            
            # Remove unused imports
            remove_unused_imports(str(file_path))
            
            # Fix line length issues
            fix_line_length(str(file_path))
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    print("Flake8 fixes completed!")

if __name__ == "__main__":
    main()
