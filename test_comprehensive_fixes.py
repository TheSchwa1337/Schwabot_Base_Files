#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to verify comprehensive fixes.
"""

import importlib
import sys
from pathlib import Path


def test_imports():
    """Test that all critical modules can be imported."""
    critical_modules = [
        'core.strategy_bit_mapper',
        'core.matrix_mapper',
        'core.trading_strategy_executor',
        'core.schwabot_rheology_integration',
        'core.orbital_shell_brain_system',
        'core.zpe_core',
        'core.zbe_core',
    ]
    
    failed_imports = []
    
    for module_name in critical_modules:
        try:
            importlib.import_module(module_name)
            print(f"SUCCESS: {module_name}")
        except Exception as e:
            print(f"FAILED: {module_name}: {e}")
            failed_imports.append(module_name)
    
    return len(failed_imports) == 0

def test_syntax():
    """Test that all Python files have valid syntax."""
    import ast
    
    failed_files = []
    
    for py_file in Path('core').rglob('*.py'):
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            ast.parse(content)
            print(f"SUCCESS: {py_file}")
        except Exception as e:
            print(f"FAILED: {py_file}: {e}")
            failed_files.append(str(py_file))
    
    return len(failed_files) == 0

def main():
    """Run all tests."""
    print("Testing comprehensive fixes...")
    print("=" * 50)
    
    import_success = test_imports()
    syntax_success = test_syntax()
    
    print("=" * 50)
    print(f"Import tests: {'SUCCESS' if import_success else 'FAILED'}")
    print(f"Syntax tests: {'SUCCESS' if syntax_success else 'FAILED'}")
    
    if import_success and syntax_success:
        print("ALL TESTS PASSED!")
        return 0
    else:
        print("SOME TESTS FAILED!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
