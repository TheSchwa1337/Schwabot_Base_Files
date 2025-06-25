#!/usr/bin/env python3
"""
Fix Import Paths Script
======================

This script fixes all the incorrect import paths in the Schwabot core files.
It replaces 'from utils.safe_print import' with the correct import paths.
"""

import os
import re
from pathlib import Path

def fix_import_paths_in_file(file_path):
    """Fix import paths in a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if file has the problematic import
        if 'from utils.safe_print import' in content:
            print(f"Fixing: {file_path}")
            
            # Replace the problematic import with correct import
            new_content = re.sub(
                r'from utils\.safe_print import safe_print, info, warn, error, success, debug',
                '''# Import safe print for Windows compatibility
try:
    from .utils.windows_cli_compatibility import safe_print, info, warn, error, success, debug
except ImportError:
    try:
        from core.utils.windows_cli_compatibility import safe_print, info, warn, error, success, debug
    except ImportError:
        def safe_print(message): print(message)
        def info(message): print(f"[INFO] {message}")
        def warn(message): print(f"[WARN] {message}")
        def error(message): print(f"[ERROR] {message}")
        def success(message): print(f"[SUCCESS] {message}")
        def debug(message): print(f"[DEBUG] {message}")''',
                content
            )
            
            # Write the fixed content back
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            return True
        else:
            return False
            
    except Exception as e:
        print(f"Error fixing {file_path}: {e}")
        return False

def main():
    """Main function to fix all import paths."""
    print("🔧 Fixing Import Paths in Schwabot Core Files")
    print("=" * 50)
    
    # Get all Python files in core directory
    core_dir = Path("core")
    python_files = list(core_dir.rglob("*.py"))
    
    print(f"Found {len(python_files)} Python files in core directory")
    
    fixed_count = 0
    for file_path in python_files:
        if fix_import_paths_in_file(file_path):
            fixed_count += 1
    
    print(f"\n✅ Fixed {fixed_count} files")
    print("🎉 Import path fixing completed!")

if __name__ == "__main__":
    main() 