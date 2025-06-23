#!/usr/bin/env python3
"""
Codebase Consistency Monitor
============================

This script monitors the codebase for potential issues that could cause
flake8 E902 errors and other file path problems. It helps ensure that:

1. File references are consistent
2. Import statements are correct
3. Configuration files use proper paths
4. No stub files or broken references exist
"""

import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple


class CodebaseConsistencyMonitor:
    """Monitor codebase for consistency issues."""
    
    def __init__(self, project_root: str = "."):
        """Initialize the monitor."""
        self.project_root = Path(project_root)
        self.issues = []
        self.warnings = []
        
        # Files that should exist in core/ directory
        self.core_files = [
            "dlt_waveform_engine.py",
            "multi_bit_btc_processor.py",
            "profit_routing_engine.py",
            "temporal_execution_correction_layer.py",
            "post_failure_recovery_intelligence_loop.py"
        ]
        
        # Files that should exist in root directory
        self.root_files = [
            "apply_windows_cli_compatibility.py",
            "validate_schwabot_system.py",
            "schwabot_unified_system.py"
        ]
        
        # Directories that should exist
        self.expected_dirs = [
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
    
    def run_full_audit(self) -> Dict[str, any]:
        """Run a full audit of the codebase."""
        print("🔍 Codebase Consistency Audit")
        print("=" * 50)
        
        results = {
            "file_existence_check": self.check_file_existence(),
            "import_consistency": self.check_import_consistency(),
            "configuration_references": self.check_configuration_references(),
            "stub_file_detection": self.detect_stub_files(),
            "path_reference_issues": self.check_path_references(),
            "flake8_command_issues": self.check_flake8_commands()
        }
        
        # Generate summary
        total_issues = sum(len(result.get("issues", [])) for result in results.values())
        total_warnings = sum(len(result.get("warnings", [])) for result in results.values())
        
        print(f"\n📊 Audit Summary:")
        print(f"Total Issues: {total_issues}")
        print(f"Total Warnings: {total_warnings}")
        
        if total_issues == 0 and total_warnings == 0:
            print("✅ Codebase is consistent!")
        else:
            print("⚠️ Issues found - review recommended")
        
        return results
    
    def check_file_existence(self) -> Dict[str, any]:
        """Check if expected files exist in correct locations."""
        print("\n📁 Checking file existence...")
        
        issues = []
        warnings = []
        
        # Check core files
        for file_name in self.core_files:
            root_path = self.project_root / file_name
            core_path = self.project_root / "core" / file_name
            
            if root_path.exists() and not core_path.exists():
                issues.append(f"File {file_name} exists in root but should be in core/")
            elif not root_path.exists() and not core_path.exists():
                warnings.append(f"File {file_name} missing from both root and core/")
            elif not root_path.exists() and core_path.exists():
                print(f"✅ {file_name} correctly located in core/")
        
        # Check root files
        for file_name in self.root_files:
            file_path = self.project_root / file_name
            if not file_path.exists():
                warnings.append(f"Expected root file {file_name} not found")
            else:
                print(f"✅ {file_name} exists in root")
        
        # Check directories
        for dir_name in self.expected_dirs:
            dir_path = self.project_root / dir_name
            if not dir_path.exists():
                warnings.append(f"Expected directory {dir_name} not found")
            else:
                print(f"✅ {dir_name} exists")
        
        return {"issues": issues, "warnings": warnings}
    
    def check_import_consistency(self) -> Dict[str, any]:
        """Check import statements for consistency."""
        print("\n📦 Checking import consistency...")
        
        issues = []
        warnings = []
        
        # Find all Python files
        python_files = list(self.project_root.rglob("*.py"))
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for incorrect import patterns
                incorrect_imports = [
                    (f"from {file_name.replace('.py', '')} import", f"from core.{file_name.replace('.py', '')} import")
                    for file_name in self.core_files
                ]
                
                for incorrect, correct in incorrect_imports:
                    if incorrect in content:
                        issues.append(f"{file_path}: Should use '{correct}' instead of '{incorrect}'")
                
                # Check for missing imports
                for file_name in self.core_files:
                    module_name = file_name.replace('.py', '')
                    if f"import {module_name}" in content and not f"from core import {module_name}" in content:
                        warnings.append(f"{file_path}: Consider using 'from core import {module_name}'")
                        
            except Exception as e:
                warnings.append(f"Could not read {file_path}: {e}")
        
        return {"issues": issues, "warnings": warnings}
    
    def check_configuration_references(self) -> Dict[str, any]:
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
        
        for config_file in config_files:
            config_path = self.project_root / config_file
            if not config_path.exists():
                continue
                
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for incorrect file references
                for file_name in self.core_files:
                    incorrect_ref = f'"{file_name}"'
                    correct_ref = f'"core/{file_name}"'
                    
                    if incorrect_ref in content:
                        issues.append(f"{config_file}: Should use {correct_ref} instead of {incorrect_ref}")
                        
            except Exception as e:
                warnings.append(f"Could not read {config_file}: {e}")
        
        return {"issues": issues, "warnings": warnings}
    
    def detect_stub_files(self) -> Dict[str, any]:
        """Detect stub files that might cause issues."""
        print("\n🔍 Detecting stub files...")
        
        issues = []
        warnings = []
        
        # Find all Python files
        python_files = list(self.project_root.rglob("*.py"))
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for stub file indicators
                stub_indicators = [
                    "TEMPORARY STUB",
                    "STUB FILE",
                    "PLACEHOLDER",
                    "TODO: IMPLEMENT",
                    "def main() -> None:" in content and len(content) < 500
                ]
                
                if any(indicator in content.upper() for indicator in stub_indicators):
                    warnings.append(f"Potential stub file: {file_path}")
                    
            except Exception as e:
                warnings.append(f"Could not read {file_path}: {e}")
        
        return {"issues": issues, "warnings": warnings}
    
    def check_path_references(self) -> Dict[str, any]:
        """Check for path reference issues."""
        print("\n🛤️ Checking path references...")
        
        issues = []
        warnings = []
        
        # Find all Python files
        python_files = list(self.project_root.rglob("*.py"))
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for hardcoded paths that might be problematic
                problematic_patterns = [
                    r'os\.path\.join\(.*["\']dlt_waveform_engine\.py["\']',
                    r'os\.path\.join\(.*["\']multi_bit_btc_processor\.py["\']',
                    r'os\.path\.join\(.*["\']profit_routing_engine\.py["\']',
                ]
                
                for pattern in problematic_patterns:
                    if re.search(pattern, content):
                        issues.append(f"{file_path}: Hardcoded path reference found")
                        
            except Exception as e:
                warnings.append(f"Could not read {file_path}: {e}")
        
        return {"issues": issues, "warnings": warnings}
    
    def check_flake8_commands(self) -> Dict[str, any]:
        """Check for problematic flake8 commands."""
        print("\n🔧 Checking flake8 commands...")
        
        issues = []
        warnings = []
        
        # Find all Python files
        python_files = list(self.project_root.rglob("*.py"))
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for problematic flake8 commands
                problematic_commands = [
                    'flake8.*dlt_waveform_engine\.py',
                    'flake8.*multi_bit_btc_processor\.py',
                    'flake8.*profit_routing_engine\.py',
                    'flake8.*temporal_execution_correction_layer\.py',
                    'flake8.*post_failure_recovery_intelligence_loop\.py',
                ]
                
                for pattern in problematic_commands:
                    if re.search(pattern, content):
                        issues.append(f"{file_path}: Problematic flake8 command found")
                        
            except Exception as e:
                warnings.append(f"Could not read {file_path}: {e}")
        
        return {"issues": issues, "warnings": warnings}
    
    def generate_fix_suggestions(self, results: Dict[str, any]) -> str:
        """Generate fix suggestions based on audit results."""
        suggestions = []
        
        for check_name, check_results in results.items():
            issues = check_results.get("issues", [])
            warnings = check_results.get("warnings", [])
            
            if issues or warnings:
                suggestions.append(f"\n## {check_name.replace('_', ' ').title()}")
                
                if issues:
                    suggestions.append("### Critical Issues:")
                    for issue in issues:
                        suggestions.append(f"- {issue}")
                
                if warnings:
                    suggestions.append("### Warnings:")
                    for warning in warnings:
                        suggestions.append(f"- {warning}")
        
        if suggestions:
            return "\n".join(suggestions)
        else:
            return "No issues found - codebase is consistent!"
    
    def create_monitoring_script(self) -> str:
        """Create a monitoring script for continuous checking."""
        script_content = '''#!/usr/bin/env python3
"""
Continuous Codebase Monitor
==========================

Run this script regularly to monitor for consistency issues.
"""

import subprocess
import sys

def run_monitor():
    """Run the codebase consistency monitor."""
    try:
        result = subprocess.run([
            sys.executable, "codebase_consistency_monitor.py"
        ], capture_output=True, text=True, check=False)
        
        if result.returncode == 0:
            print("✅ Codebase consistency check passed")
        else:
            print("❌ Codebase consistency issues found")
            print(result.stdout)
            print(result.stderr)
        
        return result.returncode
        
    except Exception as e:
        print(f"❌ Monitor failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(run_monitor())
'''
        
        monitor_script_path = self.project_root / "monitor_codebase.py"
        with open(monitor_script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        return str(monitor_script_path)


def main():
    """Main function."""
    monitor = CodebaseConsistencyMonitor()
    results = monitor.run_full_audit()
    
    # Print detailed results
    for check_name, check_results in results.items():
        issues = check_results.get("issues", [])
        warnings = check_results.get("warnings", [])
        
        if issues or warnings:
            print(f"\n{check_name.replace('_', ' ').title()}:")
            for issue in issues:
                print(f"  ❌ {issue}")
            for warning in warnings:
                print(f"  ⚠️ {warning}")
    
    # Generate fix suggestions
    suggestions = monitor.generate_fix_suggestions(results)
    print(f"\n📋 Fix Suggestions:")
    print(suggestions)
    
    # Create monitoring script
    monitor_script = monitor.create_monitoring_script()
    print(f"\n📝 Created monitoring script: {monitor_script}")
    
    # Exit with appropriate code
    total_issues = sum(len(result.get("issues", [])) for result in results.values())
    sys.exit(1 if total_issues > 0 else 0)


if __name__ == "__main__":
    main() 