#!/usr/bin/env python3
"""
Final Consistency Verification
=============================

This script performs a final verification to ensure all consistency issues
have been resolved and no flake8 E902 errors will occur.
"""

import os
import re
from pathlib import Path


def verify_file_locations():
    """Verify that files are in their correct locations."""
    print("📁 Verifying file locations...")
    
    # Core files that should exist in core/ directory
    core_files = [
        "dlt_waveform_engine.py",
        "multi_bit_btc_processor.py",
        "profit_routing_engine.py",
        "temporal_execution_correction_layer.py",
        "post_failure_recovery_intelligence_loop.py"
    ]
    
    issues = []
    
    for file_name in core_files:
        root_path = Path(file_name)
        core_path = Path("core") / file_name
        
        if root_path.exists():
            issues.append(f"❌ {file_name} exists in root but should be in core/")
        elif not core_path.exists():
            issues.append(f"❌ {file_name} missing from core/ directory")
        else:
            print(f"✅ {file_name} correctly located in core/")
    
    return issues


def verify_configuration_references():
    """Verify that configuration files use correct references."""
    print("\n⚙️ Verifying configuration references...")
    
    # Files that might contain configuration references
    config_files = [
        "apply_windows_cli_compatibility.py",
        "apply_comprehensive_architecture_integration.py",
        ".flake8",
        "pyproject.toml",
        "setup.py"
    ]
    
    core_files = [
        "dlt_waveform_engine.py",
        "multi_bit_btc_processor.py",
        "profit_routing_engine.py",
        "temporal_execution_correction_layer.py",
        "post_failure_recovery_intelligence_loop.py"
    ]
    
    issues = []
    
    for config_file in config_files:
        config_path = Path(config_file)
        if not config_path.exists():
            continue
            
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for incorrect file references
            for file_name in core_files:
                incorrect_ref = f'"{file_name}"'
                if incorrect_ref in content:
                    issues.append(f"❌ {config_file}: Contains incorrect reference {incorrect_ref}")
                    
        except Exception as e:
            issues.append(f"❌ Could not read {config_file}: {e}")
    
    if not issues:
        print("✅ All configuration references are correct")
    
    return issues


def verify_import_statements():
    """Verify that import statements use correct paths."""
    print("\n📦 Verifying import statements...")
    
    # Find all Python files
    python_files = list(Path(".").rglob("*.py"))
    
    core_files = [
        "dlt_waveform_engine.py",
        "multi_bit_btc_processor.py",
        "profit_routing_engine.py",
        "temporal_execution_correction_layer.py",
        "post_failure_recovery_intelligence_loop.py"
    ]
    
    issues = []
    
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for incorrect import patterns
            for file_name in core_files:
                module_name = file_name.replace('.py', '')
                incorrect_pattern = f"from {module_name} import"
                if incorrect_pattern in content:
                    issues.append(f"❌ {file_path}: Contains incorrect import '{incorrect_pattern}'")
                    
        except Exception as e:
            issues.append(f"❌ Could not read {file_path}: {e}")
    
    if not issues:
        print("✅ All import statements are correct")
    
    return issues


def verify_flake8_commands():
    """Verify that no problematic flake8 commands exist."""
    print("\n🔧 Verifying flake8 commands...")
    
    # Find all Python files
    python_files = list(Path(".").rglob("*.py"))
    
    issues = []
    
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for problematic flake8 commands
            problematic_patterns = [
                r'flake8.*dlt_waveform_engine\.py',
                r'flake8.*multi_bit_btc_processor\.py',
                r'flake8.*profit_routing_engine\.py',
                r'flake8.*temporal_execution_correction_layer\.py',
                r'flake8.*post_failure_recovery_intelligence_loop\.py',
            ]
            
            for pattern in problematic_patterns:
                if re.search(pattern, content):
                    issues.append(f"❌ {file_path}: Contains problematic flake8 command")
                    break
                    
        except Exception as e:
            issues.append(f"❌ Could not read {file_path}: {e}")
    
    if not issues:
        print("✅ No problematic flake8 commands found")
    
    return issues


def generate_correct_flake8_command():
    """Generate the correct flake8 command."""
    print("\n📋 Correct flake8 command:")
    
    # Check which directories exist
    existing_dirs = []
    for dir_name in ["core/", "tests/", "mathlib/", "config/", "tools/", "settings/", "demo/", "runtime/", "docs/"]:
        if Path(dir_name).exists():
            existing_dirs.append(dir_name)
    
    # Check which root files exist
    existing_files = []
    for file_name in ["apply_windows_cli_compatibility.py", "validate_schwabot_system.py", "schwabot_unified_system.py"]:
        if Path(file_name).exists():
            existing_files.append(file_name)
    
    cmd_parts = ["python", "-m", "flake8"] + existing_dirs + existing_files
    correct_command = " ".join(cmd_parts)
    
    print(f"Use: {correct_command}")
    return correct_command


def main():
    """Main verification function."""
    print("🔍 Final Consistency Verification")
    print("=" * 50)
    
    # Run all verifications
    file_issues = verify_file_locations()
    config_issues = verify_configuration_references()
    import_issues = verify_import_statements()
    flake8_issues = verify_flake8_commands()
    
    # Combine all issues
    all_issues = file_issues + config_issues + import_issues + flake8_issues
    
    # Print results
    if all_issues:
        print(f"\n❌ Found {len(all_issues)} consistency issues:")
        for issue in all_issues:
            print(f"  {issue}")
        print(f"\n❌ Consistency verification failed")
        return 1
    else:
        print(f"\n✅ All consistency checks passed!")
        print("✅ No flake8 E902 errors should occur")
        
        # Generate correct flake8 command
        correct_command = generate_correct_flake8_command()
        
        print(f"\n🎉 Codebase is now consistent and ready for flake8!")
        return 0


if __name__ == "__main__":
    exit(main()) 