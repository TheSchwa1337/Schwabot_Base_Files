#!/usr/bin/env python3
"""Fix critical E999 syntax errors in the codebase.

This script focuses on the most critical E999 errors:
- Import statements after try blocks without except/finally
- Missing indented blocks after try statements
- Unmatched parentheses
- Unexpected indentation
"""

import re
from pathlib import Path
from typing import List, Dict, Tuple


def fix_imports_after_try(content: str) -> Tuple[str, int]:
    """Fix E999: import statements after try blocks without except/finally."""
    lines = content.split('\n')
    fixed_count = 0
    
    i = 0
    while i < len(lines) - 1:
        # Look for try: followed by import statements
        if (lines[i].strip() == 'try:' and 
            i + 1 < len(lines) and 
            lines[i + 1].strip().startswith('import ')):
            
            # Move the import before the try block
            import_line = lines[i + 1]
            lines.pop(i + 1)
            
            # Find the right place to insert (before try)
            for j in range(i, -1, -1):
                if lines[j].strip() and not lines[j].strip().startswith('#'):
                    lines.insert(j + 1, import_line)
                    break
            else:
                # If no suitable place found, insert at the beginning
                lines.insert(0, import_line)
            
            fixed_count += 1
            # Don't increment i since we need to re-check this position
        
        # Look for try: followed by except ImportError: without a block
        elif (lines[i].strip() == 'try:' and 
              i + 1 < len(lines) and 
              lines[i + 1].strip() == 'except ImportError:' and
              i + 2 < len(lines) and
              not lines[i + 2].strip().startswith('    ')):
            
            # Add a pass statement after try
            lines.insert(i + 1, '    pass')
            fixed_count += 1
            i += 1  # Skip the pass line we just added
        
        i += 1
    
    return '\n'.join(lines), fixed_count


def fix_missing_try_blocks(content: str) -> Tuple[str, int]:
    """Fix E999: missing indented blocks after try statements."""
    lines = content.split('\n')
    fixed_count = 0
    
    for i in range(len(lines) - 1):
        # Look for try: followed by except ImportError: without proper indentation
        if (lines[i].strip() == 'try:' and 
            i + 1 < len(lines) and 
            lines[i + 1].strip() == 'except ImportError:'):
            
            # Check if there's a proper indented block after try
            has_indented_block = False
            for j in range(i + 1, len(lines)):
                if lines[j].strip() == 'except ImportError:':
                    break
                elif lines[j].strip().startswith('    '):
                    has_indented_block = True
                    break
                elif lines[j].strip() and not lines[j].strip().startswith('#'):
                    break
            
            if not has_indented_block:
                # Add a pass statement after try
                lines.insert(i + 1, '    pass')
                fixed_count += 1
    
    return '\n'.join(lines), fixed_count


def fix_unmatched_parentheses(content: str) -> Tuple[str, int]:
    """Fix E999: unmatched parentheses."""
    lines = content.split('\n')
    fixed_count = 0
    
    for i, line in enumerate(lines):
        # Look for lines with unmatched parentheses
        if line.strip().endswith(')') and not line.strip().endswith('()'):
            # Check if this is a standalone closing parenthesis
            if line.strip() == ')':
                # Remove the standalone closing parenthesis
                lines[i] = ''
                fixed_count += 1
    
    return '\n'.join(lines), fixed_count


def fix_unexpected_indentation(content: str) -> Tuple[str, int]:
    """Fix E999: unexpected indentation."""
    lines = content.split('\n')
    fixed_count = 0
    
    for i, line in enumerate(lines):
        # Look for lines that are unexpectedly indented
        if (line.strip() and 
            line.startswith('    ') and 
            not line.strip().startswith('def ') and
            not line.strip().startswith('class ') and
            not line.strip().startswith('if ') and
            not line.strip().startswith('for ') and
            not line.strip().startswith('while ') and
            not line.strip().startswith('try:') and
            not line.strip().startswith('except ') and
            not line.strip().startswith('finally:') and
            not line.strip().startswith('else:') and
            not line.strip().startswith('elif ') and
            not line.strip().startswith('with ') and
            not line.strip().startswith('import ') and
            not line.strip().startswith('from ') and
            not line.strip().startswith('return ') and
            not line.strip().startswith('pass') and
            not line.strip().startswith('break') and
            not line.strip().startswith('continue') and
            not line.strip().startswith('raise ') and
            not line.strip().startswith('yield ') and
            not line.strip().startswith('assert ') and
            not line.strip().startswith('del ') and
            not line.strip().startswith('global ') and
            not line.strip().startswith('nonlocal ') and
            not line.strip().startswith('print(') and
            not line.strip().startswith('print ') and
            not line.strip().startswith('print') and
            not line.strip().startswith('#')):
            
            # Check if this is part of a multi-line statement
            if i > 0 and lines[i-1].strip().endswith('\\'):
                continue
            
            # Check if this is part of a function call or definition
            if i > 0 and ('(' in lines[i-1] or 'def ' in lines[i-1] or 'class ' in lines[i-1]):
                continue
            
            # Remove the unexpected indentation
            lines[i] = line.lstrip()
            fixed_count += 1
    
    return '\n'.join(lines), fixed_count


def fix_file(file_path: str) -> Dict[str, int]:
    """Fix critical E999 errors in a single file."""
    stats = {
        "imports_after_try_fixed": 0,
        "missing_try_blocks_fixed": 0,
        "unmatched_parentheses_fixed": 0,
        "unexpected_indentation_fixed": 0,
    }
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Apply fixes in order of priority
        content, count = fix_imports_after_try(content)
        stats["imports_after_try_fixed"] = count
        
        content, count = fix_missing_try_blocks(content)
        stats["missing_try_blocks_fixed"] = count
        
        content, count = fix_unmatched_parentheses(content)
        stats["unmatched_parentheses_fixed"] = count
        
        content, count = fix_unexpected_indentation(content)
        stats["unexpected_indentation_fixed"] = count
        
        # Write back if changes were made
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Fixed {file_path}: {stats}")
        
        return stats
        
    except Exception as e:
        print(f"Error fixing {file_path}: {e}")
        return stats


def main():
    """Fix critical E999 errors in all Python files."""
    total_stats = {
        "imports_after_try_fixed": 0,
        "missing_try_blocks_fixed": 0,
        "unmatched_parentheses_fixed": 0,
        "unexpected_indentation_fixed": 0,
        "files_processed": 0,
    }
    
    # Process core directory
    core_path = Path("core")
    if core_path.exists():
        for py_file in core_path.rglob("*.py"):
            if py_file.is_file():
                stats = fix_file(str(py_file))
                for key in stats:
                    if key in total_stats:
                        total_stats[key] += stats[key]
                total_stats["files_processed"] += 1
    
    print(f"\nTotal E999 critical fixes applied:")
    print(f"Files processed: {total_stats['files_processed']}")
    print(f"Imports after try: {total_stats['imports_after_try_fixed']}")
    print(f"Missing try blocks: {total_stats['missing_try_blocks_fixed']}")
    print(f"Unmatched parentheses: {total_stats['unmatched_parentheses_fixed']}")
    print(f"Unexpected indentation: {total_stats['unexpected_indentation_fixed']}")


if __name__ == "__main__":
    main() 