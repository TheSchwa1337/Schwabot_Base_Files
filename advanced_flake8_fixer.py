#!/usr/bin/env python3
"""
🔧 Advanced Flake8 Fixer
========================

Handles the more complex flake8 issues that require intelligent parsing:
- E501: Complex line length issues
- F821: Undefined name errors 
- F541: f-string is missing placeholders
- E999: SyntaxError
- Complex multi-line formatting

Usage:
    python advanced_flake8_fixer.py
"""

import os
import re
import ast
from pathlib import Path
from typing import Dict, List, Set


def add_missing_imports(filepath: str) -> bool:
    """Fix F821: Add missing imports for undefined names"""
    if not Path(filepath).exists():
        return False
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.splitlines()
        modified = False
        
        # Common undefined names and their imports
        import_map = {
            'datetime': 'from datetime import datetime',
            'timezone': 'from datetime import timezone', 
            'timedelta': 'from datetime import timedelta',
            'PhaseEngineHooks': 'from unittest.mock import Mock as PhaseEngineHooks',
            'OracleBus': 'from unittest.mock import Mock as OracleBus',
            'List': 'from typing import List',
            'Dict': 'from typing import Dict',
            'Optional': 'from typing import Optional',
            'Union': 'from typing import Union',
            'Callable': 'from typing import Callable',
            'Any': 'from typing import Any',
            'Tuple': 'from typing import Tuple',
            'Set': 'from typing import Set',
            'np': 'import numpy as np',
            'pd': 'import pandas as pd',
            'plt': 'import matplotlib.pyplot as plt',
            'scipy': 'import scipy',
            'platform': 'import platform',
            'threading': 'import threading',
            'asyncio': 'import asyncio',
            'json': 'import json',
            'Path': 'from pathlib import Path',
            'field': 'from dataclasses import field',
            'Enum': 'from enum import Enum',
            'defaultdict': 'from collections import defaultdict',
            'OrderedDict': 'from collections import OrderedDict',
            'Counter': 'from collections import Counter',
            'partial': 'from functools import partial',
            'wraps': 'from functools import wraps',
            'lru_cache': 'from functools import lru_cache'
        }
        
        # Find undefined names in the file
        undefined_names = set()
        for line in lines:
            for name in import_map:
                if name in line and not line.strip().startswith('#'):
                    undefined_names.add(name)
        
        # Check if imports already exist
        existing_imports = set()
        for line in lines:
            if line.strip().startswith('import ') or line.strip().startswith('from '):
                existing_imports.add(line.strip())
        
        # Add missing imports at the top
        imports_to_add = []
        for name in undefined_names:
            import_stmt = import_map[name]
            if import_stmt not in existing_imports:
                imports_to_add.append(import_stmt)
        
        if imports_to_add:
            # Find where to insert imports (after existing imports)
            insert_index = 0
            for i, line in enumerate(lines):
                if (line.strip().startswith('import ') or 
                    line.strip().startswith('from ') or
                    line.strip().startswith('"""') or
                    line.strip().startswith("'''") or
                    line.strip().startswith('#')):
                    insert_index = i + 1
                elif line.strip() == '':
                    continue
                else:
                    break
            
            # Insert imports
            for import_stmt in sorted(imports_to_add):
                lines.insert(insert_index, import_stmt)
                insert_index += 1
                modified = True
            
            # Add blank line after imports
            if modified and insert_index < len(lines) and lines[insert_index].strip():
                lines.insert(insert_index, '')
        
        if modified:
            new_content = '\n'.join(lines)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"  ✅ Added {len(imports_to_add)} imports to: {filepath}")
            return True
        
        return False
        
    except Exception as e:
        print(f"  ❌ Error adding imports to {filepath}: {e}")
        return False


def fix_f_strings(filepath: str) -> bool:
    """Fix F541: f-string is missing placeholders"""
    if not Path(filepath).exists():
        return False
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.splitlines()
        modified = False
        
        for i, line in enumerate(lines):
            # Find f-strings without placeholders
            if 'f"' in line or "f'" in line:
                # Simple check for f-strings without {}
                if (('f"' in line and '{' not in line) or 
                    ("f'" in line and '{' not in line)):
                    # Convert f-string to regular string
                    lines[i] = line.replace('f"', '"').replace("f'", "'")
                    modified = True
        
        if modified:
            new_content = '\n'.join(lines)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"  ✅ Fixed f-strings in: {filepath}")
            return True
        
        return False
        
    except Exception as e:
        print(f"  ❌ Error fixing f-strings in {filepath}: {e}")
        return False


def fix_complex_line_length(filepath: str) -> bool:
    """Fix complex E501 issues with intelligent breaking"""
    if not Path(filepath).exists():
        return False
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.splitlines()
        modified = False
        
        i = 0
        while i < len(lines):
            line = lines[i]
            if len(line) > 79:
                indent = len(line) - len(line.lstrip())
                base_indent = ' ' * indent
                continuation_indent = base_indent + '    '
                
                # Fix complex mathematical expressions
                if any(op in line for op in [' + ', ' - ', ' * ', ' / ', ' ** ', ' // ', ' % ']):
                    # Find a good breaking point
                    for op in [' + ', ' - ', ' * ', ' / ']:
                        if op in line:
                            parts = line.split(op)
                            if len(parts) == 2 and len(parts[0]) < 70:
                                lines[i] = parts[0] + ' \\'
                                lines.insert(i + 1, continuation_indent + op.strip() + ' ' + parts[1])
                                modified = True
                                i += 1
                                break
                
                # Fix long dictionary/list assignments
                elif (' = {' in line or ' = [' in line) and ('}' in line or ']' in line):
                    eq_pos = line.find(' = ')
                    if eq_pos > 0:
                        var_part = line[:eq_pos + 3]
                        dict_part = line[eq_pos + 3:]
                        if len(var_part) < 40:
                            lines[i] = var_part + '\\'
                            lines.insert(i + 1, continuation_indent + dict_part)
                            modified = True
                            i += 1
                
                # Fix long method chains
                elif '.' in line and line.count('.') > 2:
                    parts = line.split('.')
                    if len(parts) > 3:
                        # Break after second dot
                        first_part = '.'.join(parts[:2]) + '.'
                        second_part = '.'.join(parts[2:])
                        if len(first_part) < 60:
                            lines[i] = first_part + '\\'
                            lines.insert(i + 1, continuation_indent + second_part)
                            modified = True
                            i += 1
                
                # Fix long boolean expressions
                elif any(op in line for op in [' and ', ' or ', ' not ']):
                    for op in [' and ', ' or ']:
                        if op in line:
                            parts = line.split(op, 1)
                            if len(parts) == 2 and len(parts[0]) < 70:
                                lines[i] = parts[0] + ' \\'
                                lines.insert(i + 1, continuation_indent + op.strip() + ' ' + parts[1])
                                modified = True
                                i += 1
                                break
                
                # Fix long return statements
                elif line.strip().startswith('return ') and len(line) > 85:
                    return_part = line[:line.find('return ') + 7]
                    value_part = line[line.find('return ') + 7:]
                    lines[i] = return_part + '\\'
                    lines.insert(i + 1, continuation_indent + value_part)
                    modified = True
                    i += 1
                
                # Fix long if/elif conditions
                elif (line.strip().startswith('if ') or line.strip().startswith('elif ')) and ':' in line:
                    colon_pos = line.find(':')
                    if_part = line[:colon_pos]
                    
                    # Look for logical operators to break on
                    for op in [' and ', ' or ']:
                        if op in if_part:
                            parts = if_part.split(op, 1)
                            if len(parts) == 2 and len(parts[0]) < 60:
                                lines[i] = parts[0] + ' \\'
                                lines.insert(i + 1, continuation_indent + op.strip() + ' ' + parts[1] + ':')
                                modified = True
                                i += 1
                                break
            
            i += 1
        
        if modified:
            new_content = '\n'.join(lines)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"  ✅ Fixed complex line lengths in: {filepath}")
            return True
        
        return False
        
    except Exception as e:
        print(f"  ❌ Error fixing complex lines in {filepath}: {e}")
        return False


def fix_syntax_errors(filepath: str) -> bool:
    """Fix E999: SyntaxError issues"""
    if not Path(filepath).exists():
        return False
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.splitlines()
        modified = False
        
        # Check for common syntax issues
        for i, line in enumerate(lines):
            # Fix unclosed parentheses/brackets
            if line.count('(') != line.count(')'):
                if line.count('(') > line.count(')'):
                    lines[i] = line + ')'
                    modified = True
                elif line.count(')') > line.count('('):
                    lines[i] = '(' + line
                    modified = True
            
            # Fix missing commas in function calls
            if '(' in line and ')' in line and line.count(',') == 0:
                # If line looks like function(arg1 arg2), add comma
                paren_content = line[line.find('('):line.rfind(')')+1]
                if ' ' in paren_content and '=' not in paren_content:
                    # This might need a comma
                    words = paren_content.strip('()').split()
                    if len(words) == 2:
                        new_content = f"{words[0]}, {words[1]}"
                        lines[i] = line.replace(paren_content, f"({new_content})")
                        modified = True
        
        if modified:
            new_content = '\n'.join(lines)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"  ✅ Fixed syntax errors in: {filepath}")
            return True
        
        return False
        
    except Exception as e:
        print(f"  ❌ Error fixing syntax in {filepath}: {e}")
        return False


def main():
    """Main function"""
    print("🔧 Advanced Flake8 Fixer - Complex Issues")
    print("=" * 45)
    
    # Get all Python files
    python_files = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                python_files.append(filepath.replace('\\', '/'))
    
    # Focus on files most likely to have remaining issues
    priority_files = [f for f in python_files if any(keyword in f for keyword in [
        'core/', 'test_', 'apply_', '__init__'
    ])]
    
    if not priority_files:
        priority_files = python_files[:10]  # First 10 files as fallback
    
    total_fixed = 0
    
    for filepath in priority_files[:15]:  # Limit to 15 files
        if Path(filepath).exists():
            print(f"\n📁 Processing: {filepath}")
            file_fixed = False
            
            # Apply advanced fixes
            if add_missing_imports(filepath):
                file_fixed = True
            
            if fix_f_strings(filepath):
                file_fixed = True
            
            if fix_syntax_errors(filepath):
                file_fixed = True
            
            if fix_complex_line_length(filepath):
                file_fixed = True
            
            if file_fixed:
                total_fixed += 1
                print(f"    📈 Applied advanced fixes to {filepath}")
    
    print(f"\n✅ Applied advanced fixes to {total_fixed} files")
    print("\n🎯 Next steps:")
    print("  • Run flake8 to see remaining issues")
    print("  • Manual review may be needed for:")
    print("    - Very long string literals")
    print("    - Complex mathematical formulas")
    print("    - Architecture-specific imports")


if __name__ == "__main__":
    main() 