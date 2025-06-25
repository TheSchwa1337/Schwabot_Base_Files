#!/usr/bin/env python3
"""Fix remaining critical flake8 errors in the codebase.

This script focuses on the most critical errors:
- E999: SyntaxError (imports after try blocks)
- E265: Shebang lines not properly commented
- E305: Missing blank lines before main blocks
- F821: Missing math imports
- E128: Continuation line indentation
"""

import re
from pathlib import Path
from typing import List, Dict, Tuple


def fix_e999_imports_after_try(content: str) -> Tuple[str, int]:
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


def fix_e265_shebang_comments(content: str) -> Tuple[str, int]:
    """Fix E265: shebang lines not properly commented."""
    lines = content.split('\n')
    fixed_count = 0
    
    for i, line in enumerate(lines):
        # Look for shebang lines that are not properly commented
        if line.startswith('#!/') and not line.startswith('# '):
            lines[i] = '# ' + line
            fixed_count += 1
    
    return '\n'.join(lines), fixed_count


def fix_e305_blank_lines_before_main(content: str) -> Tuple[str, int]:
    """Fix E305: missing blank lines before if __name__ == '__main__'."""
    lines = content.split('\n')
    fixed_count = 0
    
    for i in range(len(lines) - 1):
        # Look for if __name__ == "__main__": that needs a blank line before it
        if lines[i + 1].strip() == 'if __name__ == "__main__":':
            # Check if there's already a blank line before it
            if i >= 0 and lines[i].strip() != '':
                lines.insert(i + 1, '')
                fixed_count += 1
    
    return '\n'.join(lines), fixed_count


def fix_f821_missing_math_imports(content: str) -> Tuple[str, int]:
    """Fix F821: missing math imports."""
    lines = content.split('\n')
    fixed_count = 0
    
    # Check if math is already imported
    has_math_import = any('import math' in line for line in lines)
    
    if not has_math_import and 'math.' in content:
        # Find the first import section
        import_section_end = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('import ') or line.strip().startswith('from '):
                import_section_end = i + 1
            elif line.strip() and not line.strip().startswith('#') and import_section_end > 0:
                break
        
        # Add math import
        if import_section_end > 0:
            lines.insert(import_section_end, 'import math')
            fixed_count += 1
    
    return '\n'.join(lines), fixed_count


def fix_e128_continuation_indent(content: str) -> Tuple[str, int]:
    """Fix E128: continuation line under-indented for visual indent."""
    lines = content.split('\n')
    fixed_count = 0
    
    for i, line in enumerate(lines):
        # Look for function definitions with parameters that need proper indentation
        if 'def ' in line and '(' in line and ')' not in line:
            # This is a multi-line function definition
            # Find the opening parenthesis position
            open_paren_pos = line.find('(')
            if open_paren_pos > 0:
                # Calculate proper indentation
                base_indent = len(line) - len(line.lstrip())
                proper_indent = base_indent + 4
                
                # Look for continuation lines
                j = i + 1
                while j < len(lines) and ')' not in lines[j]:
                    if lines[j].strip() and not lines[j].strip().startswith('#'):
                        current_indent = len(lines[j]) - len(lines[j].lstrip())
                        if current_indent < proper_indent:
                            lines[j] = ' ' * proper_indent + lines[j].lstrip()
                            fixed_count += 1
                    j += 1
    
    return '\n'.join(lines), fixed_count


def fix_file(file_path: str) -> Dict[str, int]:
    """Fix critical errors in a single file."""
    stats = {
        "e999_fixed": 0,
        "e265_fixed": 0,
        "e305_fixed": 0,
        "f821_fixed": 0,
        "e128_fixed": 0,
    }
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Apply fixes in order of priority
        content, count = fix_e999_imports_after_try(content)
        stats["e999_fixed"] = count
        
        content, count = fix_e265_shebang_comments(content)
        stats["e265_fixed"] = count
        
        content, count = fix_e305_blank_lines_before_main(content)
        stats["e305_fixed"] = count
        
        content, count = fix_f821_missing_math_imports(content)
        stats["f821_fixed"] = count
        
        content, count = fix_e128_continuation_indent(content)
        stats["e128_fixed"] = count
        
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
    """Fix critical errors in all Python files."""
    total_stats = {
        "e999_fixed": 0,
        "e265_fixed": 0,
        "e305_fixed": 0,
        "f821_fixed": 0,
        "e128_fixed": 0,
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
    
    print(f"\nTotal critical fixes applied:")
    print(f"Files processed: {total_stats['files_processed']}")
    print(f"E999 (imports after try): {total_stats['e999_fixed']}")
    print(f"E265 (shebang comments): {total_stats['e265_fixed']}")
    print(f"E305 (blank lines): {total_stats['e305_fixed']}")
    print(f"F821 (missing math): {total_stats['f821_fixed']}")
    print(f"E128 (continuation indent): {total_stats['e128_fixed']}")


if __name__ == "__main__":
    main() 