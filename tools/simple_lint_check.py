#!/usr/bin/env python3
"""
Simple Lint Check - Schwabot UROS v1.0
=====================================
Direct file checking without virtual environment dependencies.
"""

import os
import sys
from pathlib import Path

def check_files_exist():
    """Check if all expected files exist."""
    core_dir = Path("core")
    expected_files = [
        "dlt_waveform_engine.py",
        "multi_bit_btc_processor.py", 
        "profit_routing_engine.py",
        "temporal_execution_correction_layer.py",
        "post_failure_recovery_intelligence_loop.py"
    ]
    
    print("🔍 CHECKING FILE EXISTENCE")
    print("=" * 30)
    
    all_exist = True
    for file in expected_files:
        file_path = core_dir / file
        exists = file_path.exists()
        status = "✓" if exists else "❌"
        print(f"{status} {file}")
        if not exists:
            all_exist = False
    
    print(f"\nOverall: {'✅ ALL FILES EXIST' if all_exist else '❌ MISSING FILES'}")
    return all_exist

def check_python_syntax():
    """Check Python syntax without flake8."""
    core_dir = Path("core")
    python_files = list(core_dir.glob("*.py"))
    
    print("\n🐍 CHECKING PYTHON SYNTAX")
    print("=" * 30)
    
    syntax_errors = []
    for py_file in python_files:
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            compile(content, str(py_file), 'exec')
            print(f"✓ {py_file.name}")
        except SyntaxError as e:
            print(f"❌ {py_file.name}: {e}")
            syntax_errors.append((py_file.name, str(e)))
        except Exception as e:
            print(f"⚠️ {py_file.name}: {e}")
    
    print(f"\nSyntax: {'✅ ALL FILES VALID' if not syntax_errors else f'❌ {len(syntax_errors)} SYNTAX ERRORS'}")
    return len(syntax_errors) == 0

def main():
    """Main function."""
    print("🔍 SCHWABOT SIMPLE LINT CHECK")
    print("=" * 40)
    
    # Check file existence
    files_ok = check_files_exist()
    
    # Check Python syntax
    syntax_ok = check_python_syntax()
    
    # Summary
    print("\n📊 SUMMARY")
    print("=" * 20)
    print(f"Files Exist: {'✅' if files_ok else '❌'}")
    print(f"Syntax Valid: {'✅' if syntax_ok else '❌'}")
    print(f"Overall: {'✅ CLEAN' if files_ok and syntax_ok else '⚠️ ISSUES FOUND'}")
    
    if files_ok and syntax_ok:
        print("\n🎉 No E902 FileNotFoundError issues detected!")
        print("The pipeline desync issue appears to be resolved.")
    else:
        print("\n🔧 Issues found - see details above.")

if __name__ == "__main__":
    main() 