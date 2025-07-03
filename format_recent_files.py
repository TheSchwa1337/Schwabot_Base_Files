#!/usr/bin/env python3
"""
Focused Code Formatting for Recently Fixed Files

This script formats the recently fixed files to ensure they meet PEP 8 standards.
"""

import subprocess
import sys
from pathlib import Path


def run_command(command, description):
    """Run a command and return success status."""
    try:
        print(f"Running {description}...")
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            print(f"❌ {description} failed")
            if result.stderr:
                print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error running {description}: {e}")
        return False


def format_file(file_path):
    """Format a single file with Black and isort."""
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"⚠️ File not found: {file_path}")
        return False
    
    print(f"\n🎨 Formatting {file_path.name}...")
    
    # Format with Black
    black_success = run_command([
        "black", str(file_path), "--line-length=100", "--target-version=py39"
    ], "Black formatting")
    
    # Sort imports with isort
    isort_success = run_command([
        "isort", str(file_path), "--profile=black", "--line-length=100", "--atomic"
    ], "Import sorting")
    
    return black_success and isort_success


def main():
    """Format recently fixed files."""
    print("🚀 Formatting Recently Fixed Files")
    print("=" * 50)
    
    # List of recently fixed files
    files_to_format = [
        "core/phase_bit_integration.py",
        "core/type_defs.py", 
        "core/glyph_phase_resolver.py",
        "core/comprehensive_integration_system.py",
        "core/speed_lattice_trading_integration.py",
        "core/strategy/__init__.py",
        "core/api/handlers/coingecko.py",
        "core/api/handlers/glassnode.py",
        "apply_enhanced_cli_compatibility.py",
        "auto_format_code.py"
    ]
    
    success_count = 0
    total_files = len(files_to_format)
    
    for file_path in files_to_format:
        if format_file(file_path):
            success_count += 1
    
    print("\n" + "=" * 50)
    print("📊 FORMATTING SUMMARY")
    print("=" * 50)
    print(f"✅ Successfully formatted: {success_count}/{total_files} files")
    print(f"❌ Failed to format: {total_files - success_count} files")
    
    if success_count == total_files:
        print("\n🎉 All files formatted successfully!")
    else:
        print("\n⚠️ Some files failed to format. Check the errors above.")
    
    # Run flake8 check on formatted files
    print("\n🔍 Running flake8 check on formatted files...")
    flake8_success = run_command([
        "flake8", "--max-line-length=100", "--extend-ignore=E203,W503", "--count"
    ] + files_to_format, "Flake8 linting")
    
    if flake8_success:
        print("✅ All formatted files pass flake8 linting!")
    else:
        print("⚠️ Some linting issues found. Check the output above.")


if __name__ == "__main__":
    main() 