#!/usr/bin/env python3
"""
Comprehensive test script to verify all core files can import and run without errors.
"""

import sys
import traceback
from pathlib import Path


def test_file_import(file_path):
    """Test if a file can be imported without errors."""
    try:
        # Convert file path to module path
        module_path = str(file_path).replace('/', '.').replace('\\', '.')
        if module_path.endswith('.py'):
            module_path = module_path[:-3]
        
        # Import the module
        __import__(module_path)
        print(f"✓ {file_path} - Import successful")
        return True
    except Exception as e:
        print(f"✗ {file_path} - Import failed: {e}")
        traceback.print_exc()
        return False

def test_file_syntax(file_path):
    """Test if a file has valid Python syntax."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Compile to check syntax
        compile(content, file_path, 'exec')
        return True
    except Exception as e:
        print(f"✗ {file_path} - Syntax error: {e}")
        return False

def main():
    """Test all core files."""
    print("=== Testing Core Files for Import and Syntax ===\n")
    
    # Core files to test
    core_files = [
        "core/backend_math.py",
        "core/fractal_core.py", 
        "core/strategy_consensus_router.py",
        "core/qsc_enhanced_profit_allocator.py",
        "core/profit_allocator.py",
        "core/quantum_mathematical_bridge.py",
        "core/chrono_recursive_logic_function.py",
        "core/risk_manager.py",
        "core/gpu_handlers.py"
    ]
    
    syntax_results = []
    import_results = []
    
    for file_path in core_files:
        if Path(file_path).exists():
            print(f"\n--- Testing {file_path} ---")
            
            # Test syntax first
            syntax_ok = test_file_syntax(file_path)
            syntax_results.append((file_path, syntax_ok))
            
            # Test import if syntax is ok
            if syntax_ok:
                import_ok = test_file_import(file_path)
                import_results.append((file_path, import_ok))
            else:
                import_results.append((file_path, False))
        else:
            print(f"⚠️ {file_path} - File not found")
            syntax_results.append((file_path, False))
            import_results.append((file_path, False))
    
    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    
    syntax_passed = sum(1 for _, ok in syntax_results if ok)
    import_passed = sum(1 for _, ok in import_results if ok)
    total_files = len(core_files)
    
    print(f"Syntax Tests: {syntax_passed}/{total_files} passed")
    print(f"Import Tests: {import_passed}/{total_files} passed")
    
    if syntax_passed < total_files:
        print("\n❌ Files with syntax errors:")
        for file_path, ok in syntax_results:
            if not ok:
                print(f"  - {file_path}")
    
    if import_passed < total_files:
        print("\n❌ Files with import errors:")
        for file_path, ok in import_results:
            if not ok:
                print(f"  - {file_path}")
    
    if syntax_passed == total_files and import_passed == total_files:
        print("\n🎉 All files are working correctly!")
    else:
        print("\n⚠️ Some files need attention before the system can run properly.")

if __name__ == "__main__":
    main() 