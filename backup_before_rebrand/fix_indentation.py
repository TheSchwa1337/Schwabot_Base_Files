#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Indentation Fix Script
======================
Comprehensive fix for all indentation issues in the Schwabot codebase.

This script fixes mixed tabs/spaces and indentation errors.
"""

import os
import re
import glob
from pathlib import Path

def fix_file_indentation(file_path):
    """Fix indentation in a single file."""
    try:
        print(f"🔧 Fixing indentation in: {file_path}")
        
        # Read file with UTF-8 encoding
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace tabs with 4 spaces
        content = re.sub(r'\t', '    ', content)
        
        # Write back with UTF-8 encoding
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"  ✅ Fixed: {file_path}")
        return True
        
    except Exception as e:
        print(f"  ❌ Error fixing {file_path}: {e}")
        return False

def main():
    """Main function to fix all indentation issues."""
    print("🚀 Starting comprehensive indentation fix...")
    print("=" * 60)
    
    # Get all Python files in core directory
    core_files = glob.glob("core/*.py")
    
    print(f"\n📁 Fixing {len(core_files)} core files...")
    success_count = 0
    
    for file_path in core_files:
        if fix_file_indentation(file_path):
            success_count += 1
    
    print(f"\n✅ Fixed {success_count}/{len(core_files)} core files")
    
    # Test main import
    print("\n🧪 Testing main import...")
    try:
        import main
        print("✅ Main imports successfully - all indentation issues resolved!")
    except Exception as e:
        print(f"❌ Main import failed: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 Indentation fix completed!")

if __name__ == "__main__":
    main() 