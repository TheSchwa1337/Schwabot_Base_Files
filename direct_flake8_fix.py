#!/usr/bin/env python3
"""
Direct Flake8 Fix
================

This script directly fixes any remaining flake8 issues by:
1. Removing unnecessary file references
2. Ensuring correct paths are used
3. Providing the correct flake8 command
"""

import os
from pathlib import Path


def check_existing_files():
    """Check which files and directories actually exist."""
    print("📁 Checking existing files and directories...")
    
    # Check directories
    existing_dirs = []
    for dir_name in ["core/", "tests/", "mathlib/", "config/", "tools/", "settings/", "demo/", "runtime/", "docs/"]:
        if Path(dir_name).exists():
            existing_dirs.append(dir_name)
            print(f"✅ {dir_name} exists")
        else:
            print(f"❌ {dir_name} missing")
    
    # Check root files
    existing_files = []
    for file_name in ["apply_windows_cli_compatibility.py", "validate_schwabot_system.py", "schwabot_unified_system.py"]:
        if Path(file_name).exists():
            existing_files.append(file_name)
            print(f"✅ {file_name} exists")
        else:
            print(f"❌ {file_name} missing")
    
    return existing_dirs, existing_files


def check_core_files():
    """Check that core files exist in the correct location."""
    print("\n🔧 Checking core files...")
    
    core_files = [
        "dlt_waveform_engine.py",
        "multi_bit_btc_processor.py",
        "profit_routing_engine.py",
        "temporal_execution_correction_layer.py",
        "post_failure_recovery_intelligence_loop.py"
    ]
    
    all_exist = True
    for file_name in core_files:
        core_path = Path("core") / file_name
        root_path = Path(file_name)
        
        if core_path.exists():
            print(f"✅ {file_name} correctly in core/")
        elif root_path.exists():
            print(f"❌ {file_name} in wrong location (root instead of core/)")
            all_exist = False
        else:
            print(f"❌ {file_name} missing from core/")
            all_exist = False
    
    return all_exist


def generate_correct_flake8_command(existing_dirs, existing_files):
    """Generate the correct flake8 command."""
    print("\n📋 Generating correct flake8 command...")
    
    cmd_parts = ["python", "-m", "flake8"] + existing_dirs + existing_files
    correct_command = " ".join(cmd_parts)
    
    print(f"✅ Correct command: {correct_command}")
    return correct_command


def remove_problematic_references():
    """Remove any problematic references from configuration files."""
    print("\n🔧 Removing problematic references...")
    
    # Files that might have problematic references
    config_files = [
        "apply_windows_cli_compatibility.py",
        "apply_comprehensive_architecture_integration.py",
        ".flake8",
        "pyproject.toml",
        "setup.py"
    ]
    
    problematic_refs = [
        '"fix_critical_issues.py"',
        '"dlt_waveform_engine.py"',
        '"multi_bit_btc_processor.py"',
        '"profit_routing_engine.py"',
        '"temporal_execution_correction_layer.py"',
        '"post_failure_recovery_intelligence_loop.py"'
    ]
    
    fixed_count = 0
    for config_file in config_files:
        config_path = Path(config_file)
        if not config_path.exists():
            continue
            
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Remove problematic references
            for ref in problematic_refs:
                if ref in content:
                    content = content.replace(ref, '')
                    print(f"  Removed {ref} from {config_file}")
                    fixed_count += 1
            
            # Write back if changed
            if content != original_content:
                with open(config_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                    
        except Exception as e:
            print(f"  Could not process {config_file}: {e}")
    
    if fixed_count > 0:
        print(f"✅ Removed {fixed_count} problematic references")
    else:
        print("✅ No problematic references found")


def main():
    """Main function."""
    print("🔧 Direct Flake8 Fix")
    print("=" * 40)
    
    # Check existing files
    existing_dirs, existing_files = check_existing_files()
    
    # Check core files
    core_files_ok = check_core_files()
    
    # Remove problematic references
    remove_problematic_references()
    
    # Generate correct command
    correct_command = generate_correct_flake8_command(existing_dirs, existing_files)
    
    # Summary
    print(f"\n📊 Summary:")
    print(f"  Directories found: {len(existing_dirs)}")
    print(f"  Root files found: {len(existing_files)}")
    print(f"  Core files status: {'✅ OK' if core_files_ok else '❌ Issues'}")
    
    if core_files_ok:
        print(f"\n🎉 All issues resolved!")
        print(f"Use this command for flake8:")
        print(f"  {correct_command}")
        return 0
    else:
        print(f"\n⚠️ Some core files may be missing or in wrong location")
        return 1


if __name__ == "__main__":
    exit(main()) 