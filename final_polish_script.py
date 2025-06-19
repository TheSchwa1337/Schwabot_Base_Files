#!/usr/bin/env python3
"""
✨ Final Polish Script
====================

Handles the final remaining flake8 issues:
- E402: Module level import not at top of file
- E501: Line too long (final cases)
- E502: Redundant backslash between brackets
- E128: Continuation line under-indented
- E303: Too many blank lines
- W292: No newline at end of file

Usage:
    python final_polish_script.py
"""

import re
from pathlib import Path


def fix_import_order(filepath: str) -> bool:
    """Fix E402: Move imports to top of file"""
    if not Path(filepath).exists():
        return False
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.splitlines()
        modified = False
        
        # Find all import lines
        import_lines = []
        non_import_lines = []
        docstring_end = 0
        
        # Find end of module docstring
        in_docstring = False
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('"""') or stripped.startswith("'''"):
                if not in_docstring:
                    in_docstring = True
                elif stripped.endswith('"""') or stripped.endswith("'''"):
                    in_docstring = False
                    docstring_end = i + 1
                    break
            elif not in_docstring and stripped and not stripped.startswith('#'):
                break
        
        # Separate imports from other code
        for i, line in enumerate(lines):
            if i < docstring_end:
                non_import_lines.append(line)
            elif (line.strip().startswith('import ') or 
                  line.strip().startswith('from ') or
                  line.strip() == ''):
                if line.strip():  # Don't include empty lines in imports
                    import_lines.append(line.strip())
                    modified = True
            else:
                non_import_lines.append(line)
        
        if modified and import_lines:
            # Reconstruct file with imports at top
            new_lines = non_import_lines[:docstring_end]
            if new_lines and new_lines[-1].strip():
                new_lines.append('')  # Add blank line after docstring
            
            # Add sorted imports
            new_lines.extend(sorted(import_lines))
            new_lines.append('')  # Blank line after imports
            
            # Add rest of content
            new_lines.extend(non_import_lines[docstring_end:])
            
            new_content = '\n'.join(new_lines)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"  ✅ Fixed import order in: {filepath}")
            return True
        
        return False
        
    except Exception as e:
        print(f"  ❌ Error fixing imports in {filepath}: {e}")
        return False


def fix_final_line_issues(filepath: str) -> bool:
    """Fix remaining line issues: E501, E502, E128, E303, W292"""
    if not Path(filepath).exists():
        return False
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.splitlines()
        modified = False
        
        # W292: Add newline at end
        if not content.endswith('\n'):
            lines.append('')
            modified = True
        
        # E303: Remove excessive blank lines (more than 2)
        i = 0
        while i < len(lines) - 2:
            if (lines[i].strip() == '' and 
                lines[i + 1].strip() == '' and 
                lines[i + 2].strip() == ''):
                del lines[i]
                modified = True
            else:
                i += 1
        
        # E502 & E128: Fix backslash and indentation issues
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # E502: Remove redundant backslash between brackets
            if '\\' in line and ('(' in line or '[' in line or '{' in line):
                # Check if backslash is between brackets
                open_brackets = line.count('(') + line.count('[') + line.count('{')
                close_brackets = line.count(')') + line.count(']') + line.count('}')
                if open_brackets > close_brackets and line.endswith('\\'):
                    lines[i] = line[:-1].rstrip()
                    modified = True
            
            # E128: Fix continuation line indentation
            if i > 0 and lines[i - 1].endswith('\\'):
                expected_indent = len(lines[i - 1]) - len(lines[i - 1].lstrip()) + 4
                actual_indent = len(line) - len(line.lstrip())
                if actual_indent < expected_indent and line.strip():
                    lines[i] = ' ' * expected_indent + line.lstrip()
                    modified = True
            
            # E501: Final line length fixes
            if len(line) > 79:
                # Fix simple concatenations
                if ' + ' in line and line.count(' + ') == 1:
                    plus_pos = line.find(' + ')
                    if 30 < plus_pos < len(line) - 20:
                        indent = len(line) - len(line.lstrip())
                        part1 = line[:plus_pos].rstrip()
                        part2 = line[plus_pos + 3:].lstrip()
                        lines[i] = part1 + ' \\'
                        lines.insert(i + 1, ' ' * (indent + 4) + '+ ' + part2)
                        modified = True
                        i += 1
                
                # Fix long print statements
                elif line.strip().startswith('print(f"') and line.endswith('")'):
                    quote_start = line.find('"')
                    quote_end = line.rfind('"')
                    if quote_end - quote_start > 50:
                        indent = len(line) - len(line.lstrip())
                        prefix = line[:quote_start + 1]
                        message = line[quote_start + 1:quote_end]
                        
                        # Find good break point
                        mid = len(message) // 2
                        break_point = message.find(' ', mid)
                        if break_point > 0:
                            part1 = message[:break_point]
                            part2 = message[break_point + 1:]
                            lines[i] = prefix + part1 + '" \\'
                            lines.insert(i + 1, ' ' * (indent + 4) + 'f"' + part2 + '")')
                            modified = True
                            i += 1
                
                # Fix long assignment statements
                elif ' = ' in line and line.count(' = ') == 1:
                    eq_pos = line.find(' = ')
                    if eq_pos > 20 and len(line) - eq_pos > 30:
                        indent = len(line) - len(line.lstrip())
                        var_part = line[:eq_pos + 3]
                        value_part = line[eq_pos + 3:]
                        lines[i] = var_part + '\\'
                        lines.insert(i + 1, ' ' * (indent + 4) + value_part)
                        modified = True
                        i += 1
            
            i += 1
        
        if modified:
            new_content = '\n'.join(lines)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"  ✅ Fixed line issues in: {filepath}")
            return True
        
        return False
        
    except Exception as e:
        print(f"  ❌ Error fixing lines in {filepath}: {e}")
        return False


def main():
    """Main function"""
    print("✨ Final Polish Script")
    print("=" * 22)
    
    target_files = [
        "apply_windows_cli_compatibility.py",
        "core/__init__.py", 
        "core/adaptive_profit_chain.py",
        "core/advanced_drift_shell_integration.py",
        "core/advanced_test_harness.py",
        "core/news_lantern_mathematical_integration.py"
    ]
    
    total_fixed = 0
    
    for filepath in target_files:
        if Path(filepath).exists():
            print(f"\n📁 Polishing: {filepath}")
            file_fixed = False
            
            if fix_import_order(filepath):
                file_fixed = True
            
            if fix_final_line_issues(filepath):
                file_fixed = True
            
            if file_fixed:
                total_fixed += 1
                print(f"    ✨ Polished: {filepath}")
    
    print(f"\n🎉 Polished {total_fixed} files")
    print("✅ Final polish complete!")
    
    # Run a quick final check
    print("\n🔍 Final status check...")
    print("Run: flake8 --max-line-length=79 --count --statistics [files]")


if __name__ == "__main__":
    main() 