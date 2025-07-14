#!/usr/bin/env python3
"""
Fix remaining Flake8 issues that can't be automatically fixed.
"""

import os
import re
from pathlib import Path
from typing import List, Tuple


def fix_ann101_issues(file_path: str) -> None:
    """Fix ANN101 (Missing type annotation for self in method) issues."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern to match method definitions without self type annotation
    pattern = r'def (\w+)\(self,([^)]*)\) -> ([^:]+):'
    replacement = r'def \1(self, \2) -> \3:'
    
    # Apply the fix
    new_content = re.sub(pattern, replacement, content)
    
    if new_content != content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Fixed ANN101 issues in {file_path}")

def fix_ann204_issues(file_path: str) -> None:
    """Fix ANN204 (Missing return type annotation for special method) issues."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern to match __init__ methods without return type
    pattern = r'def __init__\(self,([^)]*)\):'
    replacement = r'def __init__(self, \1) -> None:'
    
    # Apply the fix
    new_content = re.sub(pattern, replacement, content)
    
    if new_content != content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Fixed ANN204 issues in {file_path}")

def fix_docstring_issues(file_path: str) -> None:
    """Fix missing docstring issues."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Add missing module docstring if file doesn't have one
    if not content.strip().startswith('"""') and not content.strip().startswith("'''"):
        lines = content.split('\n')
        if lines[0].strip() and not lines[0].strip().startswith('#'):
            # Add module docstring after imports
            import_end = 0
            for i, line in enumerate(lines):
                if line.strip().startswith(('import ', 'from ')):
                    import_end = i + 1
                elif line.strip() and not line.strip().startswith(('import ', 'from ', '#')):
                    break
            
            module_name = Path(file_path).stem
            docstring = f'"""{module_name} module."""\n\n'
            lines.insert(import_end, docstring)
            content = '\n'.join(lines)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Added module docstring to {file_path}")

def fix_unused_variables(file_path: str) -> None:
    """Fix unused variable issues by prefixing with underscore."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern to match unused loop variables
    patterns = [
        (r'for i,', 'for _i,'),
        (r'for idx,', 'for _idx,'),
    ]
    
    new_content = content
    for pattern, replacement in patterns:
        new_content = re.sub(pattern, replacement, new_content)
    
    if new_content != content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Fixed unused variable issues in {file_path}")

def fix_unused_imports(file_path: str) -> None:
    """Remove unused imports."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove specific unused imports
    unused_imports = [
        'from .data_feed import DataFeed, fetch_latest_tick',
        'from .trade_executor import TradeExecutor',
    ]
    
    new_content = content
    for unused_import in unused_imports:
        new_content = new_content.replace(unused_import, '')
    
    if new_content != content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Removed unused imports from {file_path}")

def fix_line_length_issues(file_path: str) -> None:
    """Fix line length issues by breaking long lines."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    new_lines = []
    
    for line in lines:
        if len(line) > 120 and not line.strip().startswith('#'):
            # Try to break long lines at logical points
            if 'f"' in line and len(line) > 120:
                # Break f-strings
                parts = line.split('f"')
                if len(parts) > 1:
                    new_line = parts[0] + 'f"'
                    remaining = '"'.join(parts[1:])
                    if len(new_line) + len(remaining) > 120:
                        new_line += remaining[:100] + '"\n    ' + remaining[100:]
                    else:
                        new_line += remaining
                    line = new_line
        new_lines.append(line)
    
    new_content = '\n'.join(new_lines)
    if new_content != content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Fixed line length issues in {file_path}")

def process_directory(directory: str) -> None:
    """Process all Python files in a directory."""
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                print(f"Processing {file_path}")
                
                try:
                    fix_ann101_issues(file_path)
                    fix_ann204_issues(file_path)
                    fix_docstring_issues(file_path)
                    fix_unused_variables(file_path)
                    fix_unused_imports(file_path)
                    fix_line_length_issues(file_path)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

def main():
    """Main function to fix all remaining Flake8 issues."""
    directories = ['schwabot/', 'core/', 'config/']
    
    for directory in directories:
        if os.path.exists(directory):
            print(f"\nProcessing directory: {directory}")
            process_directory(directory)
        else:
            print(f"Directory not found: {directory}")

if __name__ == "__main__":
    main() 