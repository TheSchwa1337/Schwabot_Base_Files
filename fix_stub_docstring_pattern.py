#!/usr/bin/env python3
"""Fix malformed stub docstring pattern.

This script specifically targets the pattern:
"""Stub main function."""."""

And replaces it with:
"""Stub main function."""
    pass
"""

import os
import re
from pathlib import Path


def fix_stub_docstring_pattern(file_path: str) -> bool:
    """Fix the malformed stub docstring pattern in a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Fix the specific pattern: """Stub main function."""."""
        content = re.sub(
            r'"""Stub main function\."""\."""',
            '"""Stub main function."""\n    pass\n',
            content
        )
        
        # Also fix any other similar malformed patterns
        content = re.sub(
            r'"""([^"]*)\."""\."""',
            r'"""\1."""',
            content
        )
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        
        return False
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False


def find_and_fix_all_stub_files():
    """Find and fix all files with the stub docstring pattern."""
    fixed_count = 0
    
    # Search in all directories
    for root, dirs, files in os.walk('.'):
        # Skip certain directories
        dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', '.venv', 'venv', 'node_modules']]
        
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                
                # Check if file contains the pattern
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    if '"""Stub main function."""."""' in content:
                        if fix_stub_docstring_pattern(file_path):
                            print(f"✅ Fixed: {file_path}")
                            fixed_count += 1
                            
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    
    return fixed_count


def main():
    """Main function."""
    print("Fixing malformed stub docstring pattern...")
    print("=" * 50)
    
    fixed_count = find_and_fix_all_stub_files()
    
    print(f"\nFixed {fixed_count} files with malformed stub docstrings")
    print("Stub docstring pattern fixing completed!")


if __name__ == "__main__":
    main() 