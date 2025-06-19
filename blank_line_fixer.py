#!/usr/bin/env python3
"""
📏 Blank Line Fixer
==================

Fixes blank line spacing issues:
- E301: Expected 1 blank line, found 0
- E302: Expected 2 blank lines, found 0
- E305: Expected 2 blank lines after class or function definition

Usage:
    python blank_line_fixer.py
"""

from pathlib import Path


def fix_blank_lines(filepath: str) -> bool:
    """Fix all blank line issues in a file"""
    if not Path(filepath).exists():
        return False
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.splitlines()
        modified = False
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # E302: Expected 2 blank lines before class/function at module level
            if (line.startswith('def ') or line.startswith('class ')) and i > 0:
                # Check if this is at module level (not indented)
                if not lines[i].startswith(' ') and not lines[i].startswith('\t'):
                    # Count preceding blank lines
                    blank_count = 0
                    j = i - 1
                    while j >= 0 and lines[j].strip() == '':
                        blank_count += 1
                        j -= 1
                    
                    # Check if previous non-blank line was import/docstring/comment
                    if j >= 0:
                        prev_line = lines[j].strip()
                        is_after_import = (prev_line.startswith('import ') or 
                                         prev_line.startswith('from ') or
                                         prev_line.startswith('#') or
                                         prev_line.endswith('"""') or
                                         prev_line.endswith("'''"))
                        
                        # Need 2 blank lines, but only 1 after imports
                        needed_blanks = 1 if is_after_import else 2
                        
                        if blank_count < needed_blanks:
                            # Add missing blank lines
                            for _ in range(needed_blanks - blank_count):
                                lines.insert(i, '')
                                i += 1
                                modified = True
            
            # E301: Expected 1 blank line before method inside class
            elif line.startswith('def ') and i > 0:
                # Check if this is inside a class (indented)
                if lines[i].startswith('    ') and not lines[i].startswith('        '):
                    # Count preceding blank lines
                    blank_count = 0
                    j = i - 1
                    while j >= 0 and lines[j].strip() == '':
                        blank_count += 1
                        j -= 1
                    
                    # Need 1 blank line before methods (unless it's __init__ right after class)
                    if j >= 0 and not lines[j].strip().startswith('class '):
                        if blank_count == 0:
                            lines.insert(i, '')
                            i += 1
                            modified = True
            
            # E305: Expected 2 blank lines after end of class/function
            elif i > 0 and i < len(lines) - 1:
                current_line = lines[i].strip()
                next_line = lines[i + 1].strip()
                
                # If we're at the end of a function/class and next line is def/class
                if (current_line and not current_line.startswith('#') and
                    (next_line.startswith('def ') or next_line.startswith('class '))):
                    
                    # Check if current line looks like end of function/class
                    prev_context = lines[max(0, i-10):i+1]
                    in_function_or_class = any('def ' in l or 'class ' in l for l in prev_context)
                    
                    if in_function_or_class and not next_line.startswith('    '):
                        # Count blank lines between current and next
                        blank_count = 0
                        j = i + 1
                        while j < len(lines) and lines[j].strip() == '':
                            blank_count += 1
                            j += 1
                        
                        if blank_count < 2:
                            # Add missing blank lines
                            for _ in range(2 - blank_count):
                                lines.insert(i + 1, '')
                                modified = True
            
            i += 1
        
        # Ensure file ends with exactly one newline
        while lines and lines[-1] == '':
            lines.pop()
            modified = True
        lines.append('')
        
        if modified:
            new_content = '\n'.join(lines)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"✅ Fixed blank lines in: {filepath}")
            return True
        
        return False
        
    except Exception as e:
        print(f"❌ Error fixing {filepath}: {e}")
        return False


def main():
    """Main function"""
    print("📏 Blank Line Fixer")
    print("=" * 20)
    
    target_files = [
        "apply_windows_cli_compatibility.py",
        "core/__init__.py",
        "core/adaptive_profit_chain.py",
        "core/advanced_drift_shell_integration.py",
        "core/advanced_test_harness.py",
        "core/news_lantern_mathematical_integration.py"
    ]
    
    fixed_count = 0
    for filepath in target_files:
        if Path(filepath).exists():
            if fix_blank_lines(filepath):
                fixed_count += 1
    
    print(f"\n📊 Fixed blank lines in {fixed_count} files")
    print("✨ Blank line fixes complete!")


if __name__ == "__main__":
    main() 