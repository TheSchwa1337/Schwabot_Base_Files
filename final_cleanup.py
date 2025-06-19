#!/usr/bin/env python3
"""
🎯 Final Cleanup
===============

Handles the remaining flake8 issues:
- F401: Unused imports
- W293: Blank line contains whitespace  
- W291: Trailing whitespace
- E304: Blank lines after function decorator
- E303: Too many blank lines
- E301/E302/E305: Blank line spacing
- E501: Line too long (final cases)

Usage:
    python final_cleanup.py
"""

import re
from pathlib import Path


def final_cleanup(filepath: str) -> bool:
    """Apply final cleanup to fix remaining issues"""
    if not Path(filepath).exists():
        return False
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.splitlines()
        modified = False
        
        # F401: Remove unused imports or add noqa
        for i, line in enumerate(lines):
            if "from pathlib import Path" in line and "'pathlib.Path' imported but unused" in content:
                lines[i] = line + "  # noqa: F401"
                modified = True
        
        # W293/W291: Fix whitespace issues
        for i in range(len(lines)):
            # W291: Remove trailing whitespace
            if lines[i] != lines[i].rstrip():
                lines[i] = lines[i].rstrip()
                modified = True
            
            # W293: Clean blank lines with whitespace
            if lines[i].strip() == '' and len(lines[i]) > 0:
                lines[i] = ''
                modified = True
        
        # E304: Remove blank lines after decorators
        i = 0
        while i < len(lines) - 1:
            if lines[i].strip().startswith('@') and lines[i + 1].strip() == '':
                # Remove blank line after decorator
                del lines[i + 1]
                modified = True
            else:
                i += 1
        
        # E303: Fix too many blank lines (max 2)
        i = 0
        while i < len(lines) - 2:
            if (lines[i].strip() == '' and 
                lines[i + 1].strip() == '' and 
                lines[i + 2].strip() == ''):
                # Remove excess blank line
                del lines[i]
                modified = True
            else:
                i += 1
        
        # E301/E302/E305: Fix blank line spacing
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # E302: Need 2 blank lines before top-level def/class
            if (line.startswith('def ') or line.startswith('class ')) and i > 0:
                if not lines[i].startswith(' '):  # Top-level
                    # Count blank lines before
                    blank_count = 0
                    j = i - 1
                    while j >= 0 and lines[j].strip() == '':
                        blank_count += 1
                        j -= 1
                    
                    if j >= 0:
                        prev_line = lines[j].strip()
                        # After imports, need only 1 blank line
                        if (prev_line.startswith('import ') or 
                            prev_line.startswith('from ') or
                            prev_line.endswith('"""')):
                            needed = 1
                        else:
                            needed = 2
                        
                        if blank_count < needed:
                            for _ in range(needed - blank_count):
                                lines.insert(i, '')
                                i += 1
                                modified = True
                        elif blank_count > needed:
                            for _ in range(blank_count - needed):
                                del lines[i - 1]
                                i -= 1
                                modified = True
            
            # E301: Need 1 blank line before method in class
            elif line.startswith('def ') and lines[i].startswith('    '):
                if i > 0:
                    blank_count = 0
                    j = i - 1
                    while j >= 0 and lines[j].strip() == '':
                        blank_count += 1
                        j -= 1
                    
                    if j >= 0 and not lines[j].strip().startswith('class '):
                        if blank_count == 0:
                            lines.insert(i, '')
                            i += 1
                            modified = True
            
            i += 1
        
        # E305: Add blank lines after end of class/function
        for i in range(len(lines) - 1):
            if i < len(lines) - 2:
                current = lines[i].strip()
                next_line = lines[i + 1].strip()
                next_next = lines[i + 2].strip() if i + 2 < len(lines) else ''
                
                # If next line is top-level def/class and current isn't blank
                if (current and 
                    (next_line.startswith('def ') or next_line.startswith('class ')) and
                    not next_line.startswith('    ')):
                    
                    # Check if we're ending a function/class
                    prev_context = lines[max(0, i-5):i+1]
                    has_def_class = any('def ' in l or 'class ' in l for l in prev_context)
                    
                    if has_def_class and next_line == '':
                        # Add second blank line
                        lines.insert(i + 1, '')
                        modified = True
        
        # E501: Fix remaining long lines
        i = 0
        while i < len(lines):
            line = lines[i]
            if len(line) > 79:
                # Simple string break
                if ' + ' in line and line.count(' + ') == 1:
                    plus_pos = line.find(' + ')
                    if 25 < plus_pos < len(line) - 15:
                        indent = len(line) - len(line.lstrip())
                        part1 = line[:plus_pos].rstrip()
                        part2 = line[plus_pos + 3:].lstrip()
                        lines[i] = part1 + ' \\'
                        lines.insert(i + 1, ' ' * (indent + 4) + '+ ' + part2)
                        modified = True
                        i += 1
                
                # Long function parameters
                elif '(' in line and ')' in line and ',' in line:
                    paren_start = line.find('(')
                    paren_end = line.rfind(')')
                    if paren_start > 0 and paren_end > paren_start:
                        params = line[paren_start+1:paren_end]
                        if ',' in params and len(params) > 40:
                            func_part = line[:paren_start+1]
                            suffix = line[paren_end:]
                            param_list = [p.strip() for p in params.split(',')]
                            
                            if len(param_list) <= 3:
                                indent = len(line) - len(line.lstrip())
                                lines[i] = func_part
                                for j, param in enumerate(param_list):
                                    comma = ',' if j < len(param_list) - 1 else ''
                                    lines.insert(i + j + 1, ' ' * (indent + 4) + param + comma)
                                lines.insert(i + len(param_list) + 1, ' ' * indent + suffix)
                                modified = True
                                i += len(param_list) + 1
            
            i += 1
        
        # Ensure file ends with newline
        if lines and lines[-1] != '':
            lines.append('')
            modified = True
        
        if modified:
            new_content = '\n'.join(lines)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"✅ Final cleanup: {filepath}")
            return True
        
        return False
        
    except Exception as e:
        print(f"❌ Error in final cleanup {filepath}: {e}")
        return False


def main():
    """Main function"""
    print("🎯 Final Cleanup")
    print("=" * 16)
    
    target_files = [
        "apply_windows_cli_compatibility.py",
        "core/adaptive_profit_chain.py"
    ]
    
    for filepath in target_files:
        if Path(filepath).exists():
            final_cleanup(filepath)
    
    print("\n🎉 Final cleanup complete!")
    print("📊 Check remaining issues with: flake8 --max-line-length=79 [files]")


if __name__ == "__main__":
    main() 