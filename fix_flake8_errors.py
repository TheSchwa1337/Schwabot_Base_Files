#!/usr/bin/env python3
"""
Fix Flake8 E902 Errors Script
=============================

This script fixes the E902 FileNotFoundError issues by:
1. Removing any stub files that don't exist
2. Ensuring correct file paths are used
3. Cleaning up broken references
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path


class Flake8ErrorFixer:
    """Fix flake8 E902 FileNotFoundError issues."""
    
    def __init__(self):
        """Initialize the fixer."""
        self.files_to_check = [
            "apply_windows_cli_compatibility.py"
        ]
        
        self.core_files = [
            "core/dlt_waveform_engine.py",
            "core/multi_bit_btc_processor.py",
            "core/profit_routing_engine.py", 
            "core/temporal_execution_correction_layer.py",
            "core/post_failure_recovery_intelligence_loop.py"
        ]
        
        self.directories_to_check = [
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
        
        # Files that are causing E902 errors (don't exist in root)
        self.problematic_files = [
            "dlt_waveform_engine.py",
            "multi_bit_btc_processor.py", 
            "profit_routing_engine.py",
            "temporal_execution_correction_layer.py",
            "post_failure_recovery_intelligence_loop.py",
            "fix_critical_issues.py"
        ]
    
    def check_file_existence(self):
        """Check which files exist and which don't."""
        print("Checking file existence...")
        print("=" * 40)
        
        missing_files = []
        existing_files = []
        
        for file_path in self.files_to_check:
            if os.path.exists(file_path):
                existing_files.append(file_path)
                print(f"✅ {file_path} - EXISTS")
            else:
                missing_files.append(file_path)
                print(f"❌ {file_path} - MISSING")
        
        print(f"\nMissing files: {len(missing_files)}")
        print(f"Existing files: {len(existing_files)}")
        
        return missing_files, existing_files
    
    def check_core_files(self):
        """Check if core files exist."""
        print("\nChecking core files...")
        print("=" * 30)
        
        missing_core_files = []
        existing_core_files = []
        
        for file_path in self.core_files:
            if os.path.exists(file_path):
                existing_core_files.append(file_path)
                print(f"✅ {file_path} - EXISTS")
            else:
                missing_core_files.append(file_path)
                print(f"❌ {file_path} - MISSING")
        
        print(f"\nMissing core files: {len(missing_core_files)}")
        print(f"Existing core files: {len(existing_core_files)}")
        
        return missing_core_files, existing_core_files
    
    def check_directories(self):
        """Check if directories exist."""
        print("\nChecking directories...")
        print("=" * 30)
        
        missing_dirs = []
        existing_dirs = []
        
        for dir_path in self.directories_to_check:
            if os.path.exists(dir_path):
                existing_dirs.append(dir_path)
                print(f"✅ {dir_path} - EXISTS")
            else:
                missing_dirs.append(dir_path)
                print(f"❌ {dir_path} - MISSING")
        
        print(f"\nMissing directories: {len(missing_dirs)}")
        print(f"Existing directories: {len(existing_dirs)}")
        
        return missing_dirs, existing_dirs
    
    def remove_stub_files(self, missing_files):
        """Remove any stub files that might exist."""
        print("\nChecking for stub files...")
        print("=" * 30)
        
        removed_files = []
        
        for file_path in missing_files:
            # Check for various stub file patterns
            stub_patterns = [
                file_path,
                file_path + ".stub",
                file_path + ".tmp",
                file_path + ".bak",
                file_path + ".old"
            ]
            
            for pattern in stub_patterns:
                if os.path.exists(pattern):
                    try:
                        os.remove(pattern)
                        removed_files.append(pattern)
                        print(f"🗑️ Removed stub file: {pattern}")
                    except Exception as e:
                        print(f"⚠️ Could not remove {pattern}: {e}")
        
        return removed_files
    
    def fix_file_references(self):
        """Fix file references in configuration files."""
        print("\nFixing file references...")
        print("=" * 30)
        
        # Files that might contain incorrect references
        config_files = [
            "apply_windows_cli_compatibility.py",
            "apply_comprehensive_architecture_integration.py",
            ".flake8",
            "pyproject.toml",
            ".github/workflows/ci.yml"
        ]
        
        fixed_references = []
        
        for config_file in config_files:
            if os.path.exists(config_file):
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    original_content = content
                    
                    # Fix common reference patterns
                    replacements = [
                        ('"dlt_waveform_engine.py"', '"core/dlt_waveform_engine.py"'),
                        ('"multi_bit_btc_processor.py"', '"core/multi_bit_btc_processor.py"'),
                        ('"profit_routing_engine.py"', '"core/profit_routing_engine.py"'),
                        ('"temporal_execution_correction_layer.py"', '"core/temporal_execution_correction_layer.py"'),
                        ('"post_failure_recovery_intelligence_loop.py"', '"core/post_failure_recovery_intelligence_loop.py"'),
                        ('"fix_critical_issues.py"', ''),  # Remove this file reference
                        ('dlt_waveform_engine.py', 'core/dlt_waveform_engine.py'),
                        ('multi_bit_btc_processor.py', 'core/multi_bit_btc_processor.py'),
                        ('profit_routing_engine.py', 'core/profit_routing_engine.py'),
                        ('temporal_execution_correction_layer.py', 'core/temporal_execution_correction_layer.py'),
                        ('post_failure_recovery_intelligence_loop.py', 'core/post_failure_recovery_intelligence_loop.py'),
                        ('fix_critical_issues.py', ''),  # Remove this file reference
                    ]
                    
                    for old_ref, new_ref in replacements:
                        if old_ref in content:
                            if new_ref:  # Only replace if new_ref is not empty
                                content = content.replace(old_ref, new_ref)
                                fixed_references.append(f"{config_file}: {old_ref} → {new_ref}")
                            else:  # Remove the reference entirely
                                content = content.replace(old_ref, '')
                                fixed_references.append(f"{config_file}: removed {old_ref}")
                    
                    if content != original_content:
                        with open(config_file, 'w', encoding='utf-8') as f:
                            f.write(content)
                        print(f"✅ Fixed references in {config_file}")
                    
                except Exception as e:
                    print(f"⚠️ Could not fix references in {config_file}: {e}")
        
        return fixed_references
    
    def run_correct_flake8(self):
        """Run flake8 with correct file paths."""
        print("\nRunning flake8 with correct paths...")
        print("=" * 40)
        
        # Get existing directories and files
        existing_dirs = [d for d in self.directories_to_check if os.path.exists(d)]
        existing_files = [f for f in self.files_to_check if os.path.exists(f)]
        
        # Add core files that exist
        existing_core_files = [f for f in self.core_files if os.path.exists(f)]
        
        cmd = ["python", "-m", "flake8"] + existing_dirs + existing_files + existing_core_files
        
        print(f"Command: {' '.join(cmd)}")
        print()
        
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
    
    def generate_correct_flake8_command(self):
        """Generate the correct flake8 command."""
        print("\nCorrect flake8 command:")
        print("=" * 30)
        
        # Get existing directories and files
        existing_dirs = [d for d in self.directories_to_check if os.path.exists(d)]
        existing_files = [f for f in self.files_to_check if os.path.exists(f)]
        existing_core_files = [f for f in self.core_files if os.path.exists(f)]
        
        cmd = "flake8 " + " ".join(existing_dirs + existing_files + existing_core_files)
        print(cmd)
        return cmd
    
    def run(self):
        """Run the complete fix process."""
        print("🔧 Flake8 E902 Error Fixer")
        print("=" * 40)
        
        # Step 1: Check file existence
        missing_files, existing_files = self.check_file_existence()
        
        # Step 2: Check core files
        missing_core_files, existing_core_files = self.check_core_files()
        
        # Step 3: Check directories
        missing_dirs, existing_dirs = self.check_directories()
        
        # Step 4: Remove stub files
        removed_files = self.remove_stub_files(missing_files + self.problematic_files)
        
        # Step 5: Fix file references
        fixed_references = self.fix_file_references()
        
        # Step 6: Generate correct flake8 command
        correct_cmd = self.generate_correct_flake8_command()
        
        # Step 7: Run flake8 with correct paths
        flake8_exit_code = self.run_correct_flake8()
        
        # Summary
        print("\n" + "=" * 50)
        print("📊 SUMMARY")
        print("=" * 50)
        print(f"✅ Existing files: {len(existing_files)}")
        print(f"❌ Missing files: {len(missing_files)}")
        print(f"✅ Existing core files: {len(existing_core_files)}")
        print(f"❌ Missing core files: {len(missing_core_files)}")
        print(f"✅ Existing directories: {len(existing_dirs)}")
        print(f"❌ Missing directories: {len(missing_dirs)}")
        print(f"🗑️ Removed stub files: {len(removed_files)}")
        print(f"🔧 Fixed references: {len(fixed_references)}")
        print(f"🔍 Flake8 exit code: {flake8_exit_code}")
        
        if flake8_exit_code == 0:
            print("\n🎉 SUCCESS: Flake8 passed with no errors!")
        else:
            print("\n⚠️ WARNING: Flake8 found some issues. Check the output above.")
        
        return flake8_exit_code == 0


def main():
    """Main function."""
    fixer = Flake8ErrorFixer()
    success = fixer.run()
    
    if success:
        print("\n✅ All E902 errors have been resolved!")
        sys.exit(0)
    else:
        print("\n❌ Some issues remain. Please review the output above.")
        sys.exit(1)


if __name__ == "__main__":
    main() 