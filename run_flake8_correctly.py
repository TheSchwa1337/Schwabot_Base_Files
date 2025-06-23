#!/usr/bin/env python3
"""
Correct Flake8 Runner Script
============================

This script runs flake8 correctly on the existing directories and files,
avoiding the E902 FileNotFoundError issues.
"""

import os
import subprocess
import sys
from pathlib import Path


def run_flake8_on_existing_files():
    """Run flake8 only on existing files and directories."""
    
    # Directories that exist and should be checked
    existing_dirs = ["core/", "tests/", "mathlib/", "config/", "tools/", "settings/", "demo/", "runtime/", "docs/"]
    
    # Files in root directory that exist and should be checked
    existing_files = [
        "apply_windows_cli_compatibility.py",
        "validate_schwabot_system.py",
        "test_import_fix.py",
        "test_simple_import.py",
        "test_basic_imports.py",
        "test_simplified_mathematical_pipeline_validation.py",
        "test_mathematical_pipeline_validation.py",
        "test_minimal_import.py",
        "test_simplified.py",
        "test_single_module.py",
        "test_core_functionality.py",
        "test_minimal.py",
        "test_simple_imports.py",
        "test_uros_v1_integration.py",
        "test_system_integration.py",
        "launch_demo_system.py",
        "launch_unified_interface.py",
        "demo_schwabot.py",
        "start_schwabot.py",
        "schwabot_unified_system.py",
        "test_enhanced_system.py",
        "enhanced_fitness_oracle.py",
        "schwabot_integration.py",
        "ufs_app.py"
    ]
    
    # Filter to only include existing files
    existing_files = [f for f in existing_files if os.path.exists(f)]
    existing_dirs = [d for d in existing_dirs if os.path.exists(d)]
    
    print("Running flake8 on existing files and directories...")
    print(f"Directories: {existing_dirs}")
    print(f"Files: {existing_files}")
    print()
    
    # Build the flake8 command
    cmd = ["python", "-m", "flake8"] + existing_dirs + existing_files
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        
        if result.stdout:
            print("Flake8 Output:")
            print(result.stdout)
        
        if result.stderr:
            print("Flake8 Errors:")
            print(result.stderr)
        
        print(f"Flake8 exit code: {result.returncode}")
        return result.returncode
        
    except Exception as e:
        print(f"Error running flake8: {e}")
        return 1


def main():
    """Main function."""
    print("Correct Flake8 Runner")
    print("=" * 30)
    
    exit_code = run_flake8_on_existing_files()
    
    if exit_code == 0:
        print("\n✅ Flake8 completed successfully!")
    else:
        print(f"\n❌ Flake8 found issues (exit code: {exit_code})")
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main() 