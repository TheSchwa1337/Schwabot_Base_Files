#!/usr/bin/env python3
"""Fix critical flake8 errors in the codebase.

This script fixes:
- E999: SyntaxError and IndentationError (critical)
- E305: expected 2 blank lines after class/function definition
- E265: block comment should start with '# '
- F541: f-string is missing placeholders
- E128/E129: continuation line indentation
- F811: redefinition of unused imports
- F821: undefined name 'np' (missing numpy imports)
"""

import re
from pathlib import Path
from typing import List, Dict, Tuple


def fix_e999_syntax_errors(content: str) -> Tuple[str, int]:
    """Fix E999: SyntaxError and IndentationError."""
    lines = content.split('\n')
    fixed_count = 0
    
    # Fix common patterns that cause E999 errors
    for i, line in enumerate(lines):
        # Fix: "expected 'except' or 'finally' block" - import statements after try
        if (i > 0 and lines[i-1].strip().startswith('try:') and 
            line.strip().startswith('import ')):
            # Move the import before the try block
            import_line = line
            lines.pop(i)
            # Find the right place to insert (before try)
            for j in range(i-1, -1, -1):
                if lines[j].strip() and not lines[j].strip().startswith('#'):
                    lines.insert(j+1, import_line)
                    break
            fixed_count += 1
            break  # Re-process the file after this change
        
        # Fix: "expected an indented block after 'try' statement"
        if (i > 0 and lines[i-1].strip().startswith('try:') and 
            line.strip().startswith('except ImportError:')):
            # Add a pass statement after try
            lines.insert(i, '    pass')
            fixed_count += 1
    
    return '\n'.join(lines), fixed_count


def fix_e305_blank_lines(content: str) -> Tuple[str, int]:
    """Fix E305: expected 2 blank lines after class/function definition."""
    lines = content.split('\n')
    fixed_count = 0
    
    for i in range(len(lines) - 1):
        # Look for class or function definitions
        if (lines[i].strip().startswith('class ') or 
            lines[i].strip().startswith('def ')) and ':' in lines[i]:
            
            # Check if there's a main block right after
            if i + 1 < len(lines) and lines[i + 1].strip() == 'if __name__ == "__main__":':
                # Insert a blank line before the main block
                lines.insert(i + 1, '')
                fixed_count += 1
    
    return '\n'.join(lines), fixed_count


def fix_e265_block_comments(content: str) -> Tuple[str, int]:
    """Fix E265: block comment should start with '# '."""
    lines = content.split('\n')
    fixed_count = 0
    
    for i, line in enumerate(lines):
        # Look for shebang lines that are not properly commented
        if line.startswith('#!/usr/bin/env python3') and not line.startswith('# '):
            # This should already be correct, but let's check for other cases
            pass
        elif line.startswith('#!/') and not line.startswith('# '):
            lines[i] = '# ' + line
            fixed_count += 1
    
    return '\n'.join(lines), fixed_count


def fix_f541_fstring_placeholders(content: str) -> Tuple[str, int]:
    """Fix F541: f-string is missing placeholders."""
    lines = content.split('\n')
    fixed_count = 0
    
    for i, line in enumerate(lines):
        # Look for f-strings without placeholders
        if "f'" in line or 'f"' in line:
            # Check if there are any {} placeholders
            if '{' not in line and '}' not in line:
                # Convert to regular string
                lines[i] = line.replace("f'", "'").replace('f"', '"')
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


def fix_f811_redefinition_imports(content: str) -> Tuple[str, int]:
    """Fix F811: redefinition of unused imports."""
    lines = content.split('\n')
    fixed_count = 0
    seen_imports = set()
    
    for i, line in enumerate(lines):
        # Look for import statements
        if line.strip().startswith('from ') and ' import ' in line:
            import_name = line.split(' import ')[1].split()[0]
            if import_name in seen_imports:
                # Comment out the duplicate import
                lines[i] = '# ' + line + '  # F811: duplicate import'
                fixed_count += 1
            else:
                seen_imports.add(import_name)
    
    return '\n'.join(lines), fixed_count


def fix_f821_undefined_np(content: str) -> Tuple[str, int]:
    """Fix F821: undefined name 'np' by adding numpy import."""
    lines = content.split('\n')
    fixed_count = 0
    
    # Check if numpy is already imported
    has_numpy_import = any('import numpy' in line or 'import numpy as np' in line for line in lines)
    
    if not has_numpy_import and 'np.' in content:
        # Find the first import section
        import_section_end = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('import ') or line.strip().startswith('from '):
                import_section_end = i + 1
            elif line.strip() and not line.strip().startswith('#') and import_section_end > 0:
                break
        
        # Add numpy import
        if import_section_end > 0:
            lines.insert(import_section_end, 'import numpy as np')
            fixed_count += 1
    
    return '\n'.join(lines), fixed_count


def fix_file(file_path: str) -> Dict[str, int]:
    """Fix critical errors in a single file."""
    stats = {
        "e999_fixed": 0,
        "e305_fixed": 0,
        "e265_fixed": 0,
        "f541_fixed": 0,
        "e128_fixed": 0,
        "f811_fixed": 0,
        "f821_fixed": 0,
    }
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Apply fixes in order of priority
        content, count = fix_e999_syntax_errors(content)
        stats["e999_fixed"] = count
        
        content, count = fix_e305_blank_lines(content)
        stats["e305_fixed"] = count
        
        content, count = fix_e265_block_comments(content)
        stats["e265_fixed"] = count
        
        content, count = fix_f541_fstring_placeholders(content)
        stats["f541_fixed"] = count
        
        content, count = fix_e128_continuation_indent(content)
        stats["e128_fixed"] = count
        
        content, count = fix_f811_redefinition_imports(content)
        stats["f811_fixed"] = count
        
        content, count = fix_f821_undefined_np(content)
        stats["f821_fixed"] = count
        
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
        "e305_fixed": 0,
        "e265_fixed": 0,
        "f541_fixed": 0,
        "e128_fixed": 0,
        "f811_fixed": 0,
        "f821_fixed": 0,
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
    print(f"E999 (syntax/indentation): {total_stats['e999_fixed']}")
    print(f"E305 (blank lines): {total_stats['e305_fixed']}")
    print(f"E265 (block comments): {total_stats['e265_fixed']}")
    print(f"F541 (f-string placeholders): {total_stats['f541_fixed']}")
    print(f"E128 (continuation indent): {total_stats['e128_fixed']}")
    print(f"F811 (redefinition imports): {total_stats['f811_fixed']}")
    print(f"F821 (undefined np): {total_stats['f821_fixed']}")


if __name__ == "__main__":
    main() 