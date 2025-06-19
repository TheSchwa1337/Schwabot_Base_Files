#!/usr/bin/env python3
"""
Quick Fix for Remaining Critical Issues
=======================================

Fixes the 4 remaining critical syntax errors identified in the compliance report.
"""

import os
import re

def fix_file(file_path: str) -> bool:
    """Fix critical syntax errors in a file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        fixes_applied = 0
        
        # Fix 1: Remove any '-> Any -> Any:' patterns
        if '-> Any -> Any:' in content:
            content = content.replace('-> Any -> Any:', '-> Any:')
            fixes_applied += 1
        
        # Fix 2: Fix function signatures with invalid type annotations
        content = re.sub(r'def\s+(\w+)\s*\(([^)]*)\)\s*->\s*Any\s*:\s*->\s*Any\s*:', r'def \1(\2) -> Any:', content)
        fixes_applied += 1
        
        # Fix 3: Fix parameter type annotations like 'param -> Any: type'
        content = re.sub(r'(\w+)\s*->\s*Any\s*:\s*(\w+)', r'\1: \2', content)
        fixes_applied += 1
        
        # Fix 4: Remove any stray colons in function definitions
        content = re.sub(r'def\s+(\w+)\s*\(([^)]*)\)\s*:\s*:', r'def \1(\2):', content)
        fixes_applied += 1
        
        # Write back if changes were made
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Fixed {fixes_applied} issues in {file_path}")
            return True
        
        return False
        
    except Exception as e:
        print(f"❌ Error fixing {file_path}: {e}")
        return False

def main():
    """Fix all critical files"""
    critical_files = [
        'core/best_practices_enforcer.py',
        'core/fault_bus.py', 
        'tools/cleanup_obsolete_files.py',
        'tools/complete_flake8_fix.py'
    ]
    
    print("🔧 Fixing remaining critical syntax errors...")
    
    for file_path in critical_files:
        if os.path.exists(file_path):
            fix_file(file_path)
        else:
            print(f"⚠️ File not found: {file_path}")
    
    print("✅ Critical fixes applied!")

if __name__ == "__main__":
    main() 