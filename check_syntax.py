#!/usr/bin/env python3
import os
import py_compile
import sys
from pathlib import Path

def check_syntax_errors():
    """Check all Python files for syntax errors and count them."""
    syntax_errors = []
    total_files = 0
    
    # Walk through all Python files
    for root, dirs, files in os.walk('.'):
        # Skip __pycache__ and other common directories
        dirs[:] = [d for d in dirs if d not in ['__pycache__', '.git', 'build', 'dist', 'venv', 'env']]
        
        for file in files:
            if file.endswith('.py'):
                total_files += 1
                file_path = os.path.join(root, file)
                try:
                    py_compile.compile(file_path, doraise=True)
                except py_compile.PyCompileError as e:
                    syntax_errors.append((file_path, str(e)))
                except Exception as e:
                    syntax_errors.append((file_path, f"Unexpected error: {e}"))
    
    print(f"Total Python files checked: {total_files}")
    print(f"Files with syntax errors: {len(syntax_errors)}")
    print(f"Files without syntax errors: {total_files - len(syntax_errors)}")
    
    if syntax_errors:
        print("\nFiles with syntax errors:")
        for file_path, error in syntax_errors:
            print(f"  {file_path}: {error}")
    else:
        print("\n🎉 All Python files have valid syntax!")
    
    return len(syntax_errors)

if __name__ == "__main__":
    check_syntax_errors() 