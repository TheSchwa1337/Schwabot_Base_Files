#!/usr/bin/env python3
"""
Syntax Error Checker for Schwabot Codebase
==========================================

Systematically checks all Python files for syntax errors that prevent
flake8 from properly analyzing the code.
"""

import ast
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple

def check_file_syntax(file_path: str) -> Tuple[bool, str]:
    """Check if a Python file has syntax errors"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        ast.parse(content)
        return True, ""
    except SyntaxError as e:
        return False, f"SyntaxError: {e.msg} at line {e.lineno}"
    except UnicodeDecodeError as e:
        return False, f"UnicodeDecodeError: {e}"
    except Exception as e:
        return False, f"Unexpected error: {e}"

def find_python_files(directory: str) -> List[str]:
    """Find all Python files in directory"""
    python_files = []
    for root, dirs, files in os.walk(directory):
        # Skip virtual environment and cache directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__' and d != '.venv']
        
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    return python_files

def main():
    """Main function to check syntax errors"""
    print("🔍 Checking for syntax errors in Python files...")
    print("=" * 60)
    
    # Find all Python files
    python_files = find_python_files('.')
    
    syntax_errors = []
    valid_files = 0
    
    for file_path in python_files:
        is_valid, error_msg = check_file_syntax(file_path)
        
        if is_valid:
            valid_files += 1
        else:
            syntax_errors.append((file_path, error_msg))
            print(f"❌ {file_path}: {error_msg}")
    
    print("=" * 60)
    print(f"📊 Results:")
    print(f"   Total Python files: {len(python_files)}")
    print(f"   Valid files: {valid_files}")
    print(f"   Files with syntax errors: {len(syntax_errors)}")
    
    if syntax_errors:
        print("\n🚨 CRITICAL: Syntax errors found!")
        print("These files cannot be analyzed by flake8 and must be fixed first.")
        print("\nFiles with syntax errors:")
        for file_path, error_msg in syntax_errors:
            print(f"  - {file_path}: {error_msg}")
        return False
    else:
        print("\n✅ All Python files have valid syntax!")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 