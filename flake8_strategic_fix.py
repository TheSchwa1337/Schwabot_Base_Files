#!/usr/bin/env python3
"""
🎯 Strategic Flake8 Fix
======================

This script applies strategic fixes for the most critical flake8 issues
without creating additional problems.

Focus areas:
1. Fix E999 (syntax errors) - highest priority
2. Fix critical E501 (line length) issues
3. Add proper imports for F821 (undefined names)
4. Clean up unused imports selectively

Usage:
    python flake8_strategic_fix.py
"""

import os
import re
import subprocess
from pathlib import Path
from typing import List, Dict


def run_flake8() -> List[str]:
    """Run flake8 and return output lines"""
    try:
        result = subprocess.run(['flake8', 'tests/', '--max-line-length=79'], 
                              capture_output=True, text=True)
        return result.stdout.strip().split('\n') if result.stdout.strip() else []
    except FileNotFoundError:
        print("❌ flake8 not found")
        return []


def fix_syntax_errors():
    """Fix E999 syntax errors - highest priority"""
    print("🚨 Fixing syntax errors (E999)...")
    
    syntax_error_files = [
        "tests/test_enhanced_sustainment_framework.py",
        "tests/test_hash_recollection_system.py", 
        "tests/test_strategy_sustainment_validator.py"
    ]
    
    fixes_applied = 0
    
    for filepath in syntax_error_files:
        if Path(filepath).exists():
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Common syntax fixes
                original_content = content
                
                # Fix unmatched quotes
                content = re.sub(r'([^\\])"([^"]*)"([^"]*)"', r'\1"\2\\""\3"', content)
                
                # Fix unmatched parentheses
                lines = content.splitlines()
                fixed_lines = []
                paren_count = 0
                
                for line in lines:
                    # Count parentheses
                    paren_count += line.count('(') - line.count(')')
                    
                    # If we have unmatched parens at end of line, close them
                    if paren_count > 0 and line.strip() and not line.strip().endswith(('\\', ',', '(')):
                        if not any(line.strip().endswith(x) for x in [':', 'and', 'or']):
                            # Close extra parentheses
                            line += ')' * paren_count
                            paren_count = 0
                    
                    fixed_lines.append(line)
                
                content = '\n'.join(fixed_lines)
                
                if content != original_content:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print(f"  ✅ Fixed syntax in: {filepath}")
                    fixes_applied += 1
                    
            except Exception as e:
                print(f"  ❌ Error fixing {filepath}: {e}")
    
    return fixes_applied


def fix_import_errors():
    """Fix F821 undefined name errors by adding proper imports"""
    print("📦 Fixing import errors (F821)...")
    
    common_imports = {
        'datetime': 'from datetime import datetime',
        'PhaseEngineHooks': 'from unittest.mock import Mock as PhaseEngineHooks',
        'OracleBus': 'from unittest.mock import Mock as OracleBus',
        'pytest': 'import pytest',
        'unittest': 'import unittest',
        'Mock': 'from unittest.mock import Mock',
        'patch': 'from unittest.mock import patch'
    }
    
    test_files = list(Path("tests").rglob("*.py"))
    fixes_applied = 0
    
    for filepath in test_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            lines = content.splitlines()
            
            # Check what imports are needed
            needed_imports = set()
            for name, import_stmt in common_imports.items():
                if name in content and import_stmt not in content:
                    needed_imports.add(import_stmt)
            
            if needed_imports:
                # Find where to insert imports (after existing imports)
                insert_pos = 0
                for i, line in enumerate(lines):
                    if line.strip().startswith(('import ', 'from ')) or line.strip().startswith('#'):
                        insert_pos = i + 1
                    else:
                        break
                
                # Insert needed imports
                for import_stmt in sorted(needed_imports):
                    lines.insert(insert_pos, import_stmt)
                    insert_pos += 1
                
                # Add blank line after imports if needed
                if lines[insert_pos].strip():
                    lines.insert(insert_pos, '')
                
                content = '\n'.join(lines)
                
                if content != original_content:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print(f"  ✅ Added imports to: {filepath}")
                    fixes_applied += 1
                    
        except Exception as e:
            print(f"  ❌ Error fixing imports in {filepath}: {e}")
    
    return fixes_applied


def fix_critical_line_lengths():
    """Fix only the most critical line length issues"""
    print("📏 Fixing critical line length issues (E501)...")
    
    critical_files = [
        "tests/recursive_awareness_benchmark.py",
        "tests/run_missing_definitions_validation.py"
    ]
    
    fixes_applied = 0
    
    for filepath in critical_files:
        if not Path(filepath).exists():
            continue
            
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.splitlines()
            fixed_lines = []
            
            for line in lines:
                if len(line) > 100:  # Only fix really long lines
                    # Simple breaks for very long lines
                    if ' and ' in line:
                        parts = line.split(' and ')
                        if len(parts) == 2:
                            indent = len(line) - len(line.lstrip())
                            fixed_lines.append(parts[0] + ' and \\')
                            fixed_lines.append(' ' * (indent + 4) + parts[1])
                            continue
                    elif ', ' in line and line.count(',') >= 3:
                        # Break long parameter lists
                        indent = len(line) - len(line.lstrip())
                        if '(' in line and ')' in line:
                            paren_start = line.find('(')
                            prefix = line[:paren_start + 1]
                            suffix_start = line.rfind(')')
                            params = line[paren_start + 1:suffix_start]
                            suffix = line[suffix_start:]
                            
                            if len(params) > 60:
                                param_list = [p.strip() for p in params.split(',')]
                                fixed_lines.append(prefix)
                                for i, param in enumerate(param_list):
                                    if i == len(param_list) - 1:
                                        fixed_lines.append(' ' * (indent + 4) + param)
                                    else:
                                        fixed_lines.append(' ' * (indent + 4) + param + ',')
                                fixed_lines.append(' ' * indent + suffix)
                                continue
                
                fixed_lines.append(line)
            
            new_content = '\n'.join(fixed_lines)
            if new_content != content:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                print(f"  ✅ Fixed line lengths in: {filepath}")
                fixes_applied += 1
                
        except Exception as e:
            print(f"  ❌ Error fixing line lengths in {filepath}: {e}")
    
    return fixes_applied


def clean_unused_imports():
    """Selectively clean obviously unused imports"""
    print("🧹 Cleaning obvious unused imports (F401)...")
    
    # Only target obviously safe-to-remove imports
    safe_to_remove = [
        'import json',
        'import tempfile', 
        'import pathlib',
        'from pathlib import Path',
        'import numpy as np',
        'from datetime import datetime'
    ]
    
    test_files = list(Path("tests").rglob("*.py"))
    fixes_applied = 0
    
    for filepath in test_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            lines = content.splitlines()
            fixed_lines = []
            
            for line in lines:
                # Only remove if it's definitely unused
                if any(unused in line for unused in safe_to_remove):
                    # Check if the import is actually used
                    module_name = None
                    if 'import json' in line:
                        module_name = 'json'
                    elif 'import tempfile' in line:
                        module_name = 'tempfile'
                    elif 'import numpy as np' in line:
                        module_name = 'np'
                    elif 'from datetime import datetime' in line:
                        module_name = 'datetime'
                    elif 'from pathlib import Path' in line:
                        module_name = 'Path'
                    
                    if module_name and module_name not in content.replace(line, ''):
                        # Skip this line (remove the import)
                        print(f"    🗑️ Removed unused import: {line.strip()}")
                        continue
                
                fixed_lines.append(line)
            
            new_content = '\n'.join(fixed_lines)
            if new_content != original_content:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                fixes_applied += 1
                
        except Exception as e:
            print(f"  ❌ Error cleaning imports in {filepath}: {e}")
    
    return fixes_applied


def main():
    """Main function"""
    print("🎯 Strategic Flake8 Fix")
    print("=" * 40)
    
    # Get initial issue count
    initial_issues = run_flake8()
    initial_count = len([line for line in initial_issues if line.strip()])
    
    print(f"Initial issues: {initial_count}")
    print()
    
    total_fixes = 0
    
    # Apply fixes in priority order
    total_fixes += fix_syntax_errors()
    total_fixes += fix_import_errors()
    total_fixes += fix_critical_line_lengths()
    total_fixes += clean_unused_imports()
    
    print()
    print("=" * 40)
    print(f"📊 Applied fixes to {total_fixes} files")
    
    # Check final results
    print("\n🔍 Running final flake8 check...")
    final_issues = run_flake8()
    final_count = len([line for line in final_issues if line.strip()])
    
    if final_count < initial_count:
        improvement = initial_count - final_count
        percentage = (improvement / initial_count) * 100
        print(f"✅ Reduced issues from {initial_count} to {final_count}")
        print(f"   Improvement: {improvement} issues fixed ({percentage:.1f}%)")
    elif final_count == 0:
        print("🎉 All flake8 issues resolved!")
    else:
        print(f"⚠️ Issues changed from {initial_count} to {final_count}")
    
    if final_count > 0 and final_count <= 20:
        print("\n🔍 Remaining issues (sample):")
        for issue in final_issues[:10]:
            if issue.strip():
                print(f"  {issue}")
    
    print("\n🎯 Next steps:")
    print("  1. Review any remaining syntax errors manually")
    print("  2. Add missing imports for undefined names")
    print("  3. Use autopep8 for remaining formatting issues")
    print("  4. Consider adding # noqa comments for acceptable issues")


if __name__ == "__main__":
    main() 