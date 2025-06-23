#!/usr/bin/env python3
"""
Quick Consistency Check
======================

Direct check for consistency issues without subprocess dependencies.
"""

import os
import re
from pathlib import Path


def check_file_existence():
    """Check if expected files exist in correct locations."""
    print("📁 Checking file existence...")
    
    # Files that should exist in core/ directory
    core_files = [
        "dlt_waveform_engine.py",
        "multi_bit_btc_processor.py",
        "profit_routing_engine.py",
        "temporal_execution_correction_layer.py",
        "post_failure_recovery_intelligence_loop.py"
    ]
    
    # Files that should exist in root directory
    root_files = [
        "apply_windows_cli_compatibility.py",
        "validate_schwabot_system.py",
        "schwabot_unified_system.py"
    ]
    
    # Directories that should exist
    expected_dirs = [
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
    
    issues = []
    warnings = []
    
    # Check core files
    for file_name in core_files:
        root_path = Path(file_name)
        core_path = Path("core") / file_name
        
        if root_path.exists() and not core_path.exists():
            issues.append(f"File {file_name} exists in root but should be in core/")
        elif not root_path.exists() and not core_path.exists():
            warnings.append(f"File {file_name} missing from both root and core/")
        elif not root_path.exists() and core_path.exists():
            print(f"✅ {file_name} correctly located in core/")
    
    # Check root files
    for file_name in root_files:
        file_path = Path(file_name)
        if not file_path.exists():
            warnings.append(f"Expected root file {file_name} not found")
        else:
            print(f"✅ {file_name} exists in root")
    
    # Check directories
    for dir_name in expected_dirs:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            warnings.append(f"Expected directory {dir_name} not found")
        else:
            print(f"✅ {dir_name} exists")
    
    return issues, warnings


def check_configuration_references():
    """Check configuration files for correct file references."""
    print("\n⚙️ Checking configuration references...")
    
    issues = []
    warnings = []
    
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
                correct_ref = f'"core/{file_name}"'
                
                if incorrect_ref in content:
                    issues.append(f"{config_file}: Should use {correct_ref} instead of {incorrect_ref}")
                    
        except Exception as e:
            warnings.append(f"Could not read {config_file}: {e}")
    
    return issues, warnings


def check_import_consistency():
    """Check import statements for consistency."""
    print("\n📦 Checking import consistency...")
    
    issues = []
    warnings = []
    
    # Find all Python files
    python_files = list(Path(".").rglob("*.py"))
    
    core_files = [
        "dlt_waveform_engine.py",
        "multi_bit_btc_processor.py",
        "profit_routing_engine.py",
        "temporal_execution_correction_layer.py",
        "post_failure_recovery_intelligence_loop.py"
    ]
    
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for incorrect import patterns
            for file_name in core_files:
                module_name = file_name.replace('.py', '')
                incorrect_pattern = f"from {module_name} import"
                correct_pattern = f"from core.{module_name} import"
                
                if incorrect_pattern in content:
                    issues.append(f"{file_path}: Should use '{correct_pattern}' instead of '{incorrect_pattern}'")
                    
        except Exception as e:
            warnings.append(f"Could not read {file_path}: {e}")
    
    return issues, warnings


def check_flake8_commands():
    """Check for problematic flake8 commands."""
    print("\n🔧 Checking flake8 commands...")
    
    issues = []
    warnings = []
    
    # Find all Python files
    python_files = list(Path(".").rglob("*.py"))
    
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
                    issues.append(f"{file_path}: Problematic flake8 command found")
                    
        except Exception as e:
            warnings.append(f"Could not read {file_path}: {e}")
    
    return issues, warnings


def main():
    """Main function."""
    print("🔍 Quick Consistency Check")
    print("=" * 40)
    
    # Run all checks
    file_issues, file_warnings = check_file_existence()
    config_issues, config_warnings = check_configuration_references()
    import_issues, import_warnings = check_import_consistency()
    flake8_issues, flake8_warnings = check_flake8_commands()
    
    # Combine all results
    all_issues = file_issues + config_issues + import_issues + flake8_issues
    all_warnings = file_warnings + config_warnings + import_warnings + flake8_warnings
    
    # Print results
    if all_issues:
        print(f"\n❌ Found {len(all_issues)} critical issues:")
        for issue in all_issues:
            print(f"  - {issue}")
    
    if all_warnings:
        print(f"\n⚠️ Found {len(all_warnings)} warnings:")
        for warning in all_warnings:
            print(f"  - {warning}")
    
    if not all_issues and not all_warnings:
        print("\n✅ No consistency issues found!")
    
    # Generate correct flake8 command
    print(f"\n📋 Correct flake8 command:")
    existing_dirs = []
    for dir_name in ["core/", "tests/", "mathlib/", "config/", "tools/", "settings/", "demo/", "runtime/", "docs/"]:
        if Path(dir_name).exists():
            existing_dirs.append(dir_name)
    
    existing_files = []
    for file_name in ["apply_windows_cli_compatibility.py", "validate_schwabot_system.py", "schwabot_unified_system.py"]:
        if Path(file_name).exists():
            existing_files.append(file_name)
    
    cmd_parts = ["python", "-m", "flake8"] + existing_dirs + existing_files
    correct_command = " ".join(cmd_parts)
    
    print(f"Use: {correct_command}")
    
    # Exit with appropriate code
    if all_issues:
        print(f"\n❌ Consistency check failed with {len(all_issues)} issues")
        return 1
    else:
        print(f"\n✅ Consistency check passed")
        return 0


if __name__ == "__main__":
    exit(main()) 