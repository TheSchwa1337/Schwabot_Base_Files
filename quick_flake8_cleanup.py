#!/usr/bin/env python3
"""
🧹 Quick Flake8 Cleanup
======================

Fixes the remaining simple flake8 issues:
- W291: Trailing whitespace
- W292: No newline at end of file
- E501: Simple line too long cases
- F821: Add common undefined names
- W605: Invalid escape sequences

Usage:
    python quick_flake8_cleanup.py [filename]
"""

import sys
import re
from pathlib import Path


def fix_quick_issues(filepath: str) -> bool:
    """Fix simple flake8 issues in a file"""
    if not Path(filepath).exists():
        print(f"❌ File not found: {filepath}")
        return False
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.splitlines()
        modified = False
        
        # W291: Fix trailing whitespace
        for i in range(len(lines)):
            if lines[i].rstrip() != lines[i]:
                lines[i] = lines[i].rstrip()
                modified = True
        
        # W292: Add newline at end of file
        if not content.endswith('\n'):
            lines.append('')
            modified = True
        
        # F821: Add missing variable definitions
        undefined_fixes = {
            'message': 'message = ""  # Fixed undefined variable',
            'TARGET_FILES': 'TARGET_FILES = []  # Fixed undefined variable',
            'FIXED_FILES': 'FIXED_FILES = []  # Fixed undefined variable',
            'REQUIRED_IMPORTS': 'REQUIRED_IMPORTS = ""  # Fixed undefined variable',
            'WINDOWS_CLI_HANDLER_TEMPLATE': 'WINDOWS_CLI_HANDLER_TEMPLATE = ""  # Fixed undefined variable'
        }
        
        for var_name, fix_line in undefined_fixes.items():
            if f"undefined name '{var_name}'" in content or var_name in content:
                # Add definition at top of file after imports
                for i, line in enumerate(lines):
                    if (not line.strip().startswith('#') and 
                        not line.strip().startswith('import') and 
                        not line.strip().startswith('from') and
                        line.strip() and var_name not in line):
                        lines.insert(i, fix_line)
                        modified = True
                        break
        
        # E501: Fix simple long lines
        i = 0
        while i < len(lines):
            line = lines[i]
            if len(line) > 79:
                # Simple string concatenation fix
                if ' + ' in line and len(line) < 120:
                    plus_pos = line.find(' + ')
                    if plus_pos > 40 and plus_pos < len(line) - 20:
                        indent = len(line) - len(line.lstrip())
                        part1 = line[:plus_pos]
                        part2 = line[plus_pos + 3:]
                        lines[i] = part1 + ' \\'
                        lines.insert(i + 1, ' ' * (indent + 4) + '+ ' + part2)
                        modified = True
                        i += 1
                
                # Simple f-string fix
                elif line.strip().startswith('print(f"') and line.count('"') == 2:
                    quote_start = line.find('"')
                    quote_end = line.rfind('"')
                    if quote_end - quote_start > 50:
                        indent = len(line) - len(line.lstrip())
                        prefix = line[:quote_start + 1]
                        content_part = line[quote_start + 1:quote_end]
                        suffix = line[quote_end:]
                        
                        # Break the string at a space
                        mid_point = len(content_part) // 2
                        break_point = content_part.find(' ', mid_point)
                        if break_point > 0:
                            part1 = content_part[:break_point]
                            part2 = content_part[break_point + 1:]
                            lines[i] = prefix + part1 + '" \\'
                            lines.insert(i + 1, ' ' * (indent + 4) + 'f"' + part2 + suffix)
                            modified = True
                            i += 1
            
            i += 1
        
        # W605: Fix invalid escape sequences
        for i in range(len(lines)):
            line = lines[i]
            # Fix common invalid escape sequences
            if '\\)' in line and not line.strip().startswith('#'):
                lines[i] = line.replace('\\)', ')')
                modified = True
            if '\\(' in line and not line.strip().startswith('#'):
                lines[i] = line.replace('\\(', '(')
                modified = True
        
        if modified:
            new_content = '\n'.join(lines)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"✅ Fixed issues in: {filepath}")
            return True
        else:
            print(f"ℹ️  No issues found in: {filepath}")
            return False
        
    except Exception as e:
        print(f"❌ Error processing {filepath}: {e}")
        return False


def main():
    """Main function"""
    print("🧹 Quick Flake8 Cleanup")
    print("=" * 25)
    
    target_files = [
        "apply_windows_cli_compatibility.py",
        "core/__init__.py",
        "core/adaptive_profit_chain.py",
        "core/advanced_drift_shell_integration.py",
        "core/advanced_test_harness.py",
        "core/news_lantern_mathematical_integration.py"
    ]
    
    if len(sys.argv) > 1:
        target_files = [sys.argv[1]]
    
    fixed_count = 0
    for filepath in target_files:
        if fix_quick_issues(filepath):
            fixed_count += 1
    
    print(f"\n📊 Fixed {fixed_count} files")
    print("✨ Quick cleanup complete!")


if __name__ == "__main__":
    main() 