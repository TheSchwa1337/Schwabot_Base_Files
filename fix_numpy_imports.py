#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fix Numpy Imports Script

This script fixes incorrect numpy imports across the codebase:
- Changes 'import numpy as np' to 'import numpy as np'
- Ensures all numpy imports are correct
"""

import os
import re
from pathlib import Path
from typing import List

def fix_numpy_imports_in_file(file_path: str) -> bool:
    """Fix numpy imports in a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if file contains incorrect numpy import
        if 'import numpy as np' in content:
            # Replace incorrect import
            content = content.replace('import numpy as np', 'import numpy as np')
            
            # Write back to file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"✅ Fixed numpy import in: {file_path}")
            return True
        else:
            return False
            
    except Exception as e:
        print(f"❌ Error fixing {file_path}: {e}")
        return False

def find_python_files(directory: str) -> List[str]:
    """Find all Python files in directory."""
    python_files = []
    for root, dirs, files in os.walk(directory):
        # Skip backup directories
        if 'backup' in root:
            continue
        
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    return python_files

def main():
    """Main function to fix numpy imports."""
    print("🔧 Fixing numpy imports across the codebase...")
    
    # Find all Python files
    python_files = find_python_files('.')
    
    # Fix numpy imports
    fixed_count = 0
    for file_path in python_files:
        if fix_numpy_imports_in_file(file_path):
            fixed_count += 1
    
    print(f"\n📊 Summary:")
    print(f"   Files processed: {len(python_files)}")
    print(f"   Files fixed: {fixed_count}")
    print(f"   Files unchanged: {len(python_files) - fixed_count}")
    
    if fixed_count > 0:
        print("\n✅ Numpy imports have been fixed!")
    else:
        print("\nℹ️  No numpy import issues found.")

if __name__ == "__main__":
    main() 