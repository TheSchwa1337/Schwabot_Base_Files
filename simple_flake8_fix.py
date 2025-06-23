#!/usr/bin/env python3
"""
Simple Flake8 Fix Script
========================

This script directly fixes the flake8 E902 errors by:
1. Identifying the problematic files
2. Fixing file references
3. Providing the correct flake8 command
"""

import os
import sys


def main():
    """Main function to fix flake8 E902 errors."""
    print("Simple Flake8 E902 Error Fix")
    print("=" * 40)
    
    # Files that are causing E902 errors (don't exist in root)
    problematic_files = [
        "dlt_waveform_engine.py",
        "multi_bit_btc_processor.py", 
        "profit_routing_engine.py",
        "temporal_execution_correction_layer.py",
        "post_failure_recovery_intelligence_loop.py"
    ]
    
    # Files that exist in root
    existing_root_files = [
        "apply_windows_cli_compatibility.py"
    ]
    
    # Directories that exist
    existing_dirs = [
        "core/",
        "tests/",
        "mathlib/",
        "config/",
        "tools/",
        "settings/",
        "demo/",
        "runtime/",
        "docs/"
    ]
    
    print("Problematic files (causing E902 errors):")
    for file in problematic_files:
        if os.path.exists(file):
            print(f"  ⚠️ {file} - EXISTS (should be removed)")
        else:
            print(f"  ❌ {file} - MISSING (causing E902)")
    
    print("\nExisting root files:")
    for file in existing_root_files:
        if os.path.exists(file):
            print(f"  ✅ {file} - EXISTS")
        else:
            print(f"  ❌ {file} - MISSING")
    
    print("\nExisting directories:")
    for dir_path in existing_dirs:
        if os.path.exists(dir_path):
            print(f"  ✅ {dir_path} - EXISTS")
        else:
            print(f"  ❌ {dir_path} - MISSING")
    
    # Check core files
    print("\nCore files (should exist):")
    core_files = [
        "core/dlt_waveform_engine.py",
        "core/multi_bit_btc_processor.py",
        "core/profit_routing_engine.py", 
        "core/temporal_execution_correction_layer.py",
        "core/post_failure_recovery_intelligence_loop.py"
    ]
    
    for file in core_files:
        if os.path.exists(file):
            print(f"  ✅ {file} - EXISTS")
        else:
            print(f"  ❌ {file} - MISSING")
    
    # Generate correct flake8 command
    print("\n" + "=" * 50)
    print("SOLUTION")
    print("=" * 50)
    
    # Build correct command
    cmd_parts = ["python", "-m", "flake8"]
    
    # Add existing directories
    for dir_path in existing_dirs:
        if os.path.exists(dir_path):
            cmd_parts.append(dir_path)
    
    # Add existing root files
    for file in existing_root_files:
        if os.path.exists(file):
            cmd_parts.append(file)
    
    # Add existing core files
    for file in core_files:
        if os.path.exists(file):
            cmd_parts.append(file)
    
    correct_command = " ".join(cmd_parts)
    
    print("Use this command instead of the problematic one:")
    print(f"\n{correct_command}")
    
    print("\n" + "=" * 50)
    print("EXPLANATION")
    print("=" * 50)
    print("The E902 errors occur because flake8 is trying to check files")
    print("that don't exist in the root directory. The files exist in the")
    print("core/ directory, but the flake8 command is looking for them in")
    print("the wrong location.")
    print("\nThe correct approach is to:")
    print("1. Use directory paths (core/, tests/, etc.)")
    print("2. Only specify files that actually exist")
    print("3. Use correct relative paths for files in subdirectories")
    
    print("\n" + "=" * 50)
    print("FIXES APPLIED")
    print("=" * 50)
    
    # Fix file references in configuration files
    config_files = [
        "apply_windows_cli_compatibility.py",
        "apply_comprehensive_architecture_integration.py"
    ]
    
    fixes_applied = []
    
    for config_file in config_files:
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                original_content = content
                
                # Fix references
                replacements = [
                    ('"dlt_waveform_engine.py"', '"core/dlt_waveform_engine.py"'),
                    ('"multi_bit_btc_processor.py"', '"core/multi_bit_btc_processor.py"'),
                    ('"profit_routing_engine.py"', '"core/profit_routing_engine.py"'),
                    ('"temporal_execution_correction_layer.py"', '"core/temporal_execution_correction_layer.py"'),
                    ('"post_failure_recovery_intelligence_loop.py"', '"core/post_failure_recovery_intelligence_loop.py"'),
                ]
                
                for old_ref, new_ref in replacements:
                    if old_ref in content:
                        content = content.replace(old_ref, new_ref)
                        fixes_applied.append(f"{config_file}: {old_ref} → {new_ref}")
                
                if content != original_content:
                    with open(config_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print(f"✅ Fixed references in {config_file}")
                
            except Exception as e:
                print(f"⚠️ Could not fix {config_file}: {e}")
    
    if fixes_applied:
        print(f"\nApplied {len(fixes_applied)} fixes:")
        for fix in fixes_applied:
            print(f"  - {fix}")
    else:
        print("No fixes needed - references are already correct")
    
    print("\n✅ Flake8 E902 errors should now be resolved!")
    print("Run the correct command above to verify.")


if __name__ == "__main__":
    main() 