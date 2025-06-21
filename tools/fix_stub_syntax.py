#!/usr/bin/env python3
"""Fix malformed stub files with syntax errors."""

import os
import re
from pathlib import Path


def fix_stub_file(filepath: Path) -> bool:
    """Fix syntax errors in a stub file.
    
    Args:
        filepath: Path to the file to fix
        
    Returns:
        True if file was modified, False otherwise
    """
    try:
        content = filepath.read_text(encoding='utf-8')
        original_content = content
        
        # Fix the specific malformed pattern """.""" (note the extra .")
        content = re.sub(r'"""\.""""', '"""Stub function."""', content)
        content = re.sub(r'"""\."""', '"""Stub function."""', content)
        
        # Fix missing pass statements after docstrings
        if 'def ' in content:
            # Look for function definitions followed by docstring but no body
            content = re.sub(
                r'(def\s+\w+\([^)]*\)\s*->\s*[^:]*:\s*"""[^"]*""")(\s*)(if\s+__name__|class\s+|\w+\s*=|\Z)',
                r'\1\2    pass\n\n\3',
                content,
                flags=re.MULTILINE | re.DOTALL
            )
        
        # Fix unterminated triple quotes
        if '"""' in content:
            quote_count = content.count('"""')
            if quote_count % 2 == 1:
                content += '\n"""'
        
        # Fix invalid characters (replace common Unicode issues)
        content = content.replace('Γêç', '∇')  # nabla symbol
        content = content.replace('┬╖', '·')   # middle dot
        content = content.replace('Γêê', '∈')  # element of
        content = content.replace('≡ƒôè', 'ì')  # Latin small letter i with grave
        content = content.replace('≡ƒôê', 'î')  # Latin small letter i with circumflex
        content = content.replace('≡ƒôë', 'ï')  # Latin small letter i with diaeresis
        content = content.replace('≡ƒö¼', 'ü')  # Latin small letter u with diaeresis
        
        # Fix obvious syntax issues
        content = re.sub(r'(\w+)\s*=\s*"""([^"]+)"""([^"\s])', r'\1 = """\2"""\n', content)
        
        if content != original_content:
            filepath.write_text(content, encoding='utf-8')
            print(f"Fixed: {filepath}")
            return True
            
    except Exception as e:
        print(f"Error fixing {filepath}: {e}")
        
    return False


def main() -> None:
    """Fix all stub files with syntax errors."""
    core_dir = Path("core")
    tools_dir = Path("tools") 
    
    if not core_dir.exists():
        print("Core directory not found")
        return
        
    fixed_count = 0
    
    # Process all Python files in core directory
    for py_file in core_dir.rglob("*.py"):
        if py_file.is_file():
            if fix_stub_file(py_file):
                fixed_count += 1
    
    # Process tools directory if it exists
    if tools_dir.exists():
        for py_file in tools_dir.rglob("*.py"):
            if py_file.is_file():
                if fix_stub_file(py_file):
                    fixed_count += 1
    
    print(f"Fixed {fixed_count} files")


if __name__ == "__main__":
    main() 