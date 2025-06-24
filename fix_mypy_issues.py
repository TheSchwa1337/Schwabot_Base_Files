#!/usr/bin/env python3
"""
Fix MyPy Issues Script
======================

This script fixes MyPy configuration issues and ensures proper type checking:
1. Validates mypy.ini configuration
2. Tests mypy commands
3. Fixes duplicate module issues
4. Ensures proper file paths
5. Validates type checking across the codebase
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple


class MyPyIssueFixer:
    """Fix MyPy configuration and type checking issues."""
    
    def __init__(self):
        """Initialize the MyPy fixer."""
        self.mypy_config_file = "mypy.ini"
        self.core_directory = "core/"
        self.test_files = [
            "apply_windows_cli_compatibility.py"
        ]
        
        # Files to check for type annotations
        self.type_check_files = [
            "core/thermal_boundary_manager.py",
            "core/thermal_zone_manager.py",
            "core/thermal_map_allocator.py",
            "core/fault_bus.py",
            "core/multi_bit_btc_processor.py",
            "core/profit_routing_engine.py"
        ]
    
    def validate_mypy_config(self) -> Dict[str, Any]:
        """Validate the mypy.ini configuration file."""
        print("🔍 Validating MyPy configuration...")
        print("=" * 40)
        
        issues = []
        fixes_applied = []
        
        if not os.path.exists(self.mypy_config_file):
            issues.append(f"❌ {self.mypy_config_file} not found")
            return {"valid": False, "issues": issues, "fixes": fixes_applied}
        
        try:
            with open(self.mypy_config_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for invalid options
            invalid_options = [
                "warn_unused_type_ignore = True",
                "[mypy-test_*]"
            ]
            
            for option in invalid_options:
                if option in content:
                    issues.append(f"❌ Invalid option found: {option}")
            
            # Check for valid sections
            valid_sections = [
                "[mypy]",
                "[mypy-core.*]",
                "[mypy-tests.*]"
            ]
            
            for section in valid_sections:
                if section in content:
                    print(f"✅ Valid section: {section}")
            
            print(f"✅ {self.mypy_config_file} exists and is readable")
            
        except Exception as e:
            issues.append(f"❌ Error reading {self.mypy_config_file}: {e}")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "fixes": fixes_applied
        }
    
    def test_mypy_command(self, command: List[str]) -> Dict[str, Any]:
        """Test a mypy command and return results."""
        print(f"\n🧪 Testing MyPy command: {' '.join(command)}")
        print("-" * 50)
        
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            output = {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "success": result.returncode == 0
            }
            
            if result.stdout:
                print("📤 MyPy Output:")
                print(result.stdout)
            
            if result.stderr:
                print("⚠️ MyPy Errors/Warnings:")
                print(result.stderr)
            
            print(f"🔍 Exit code: {result.returncode}")
            
            return output
            
        except subprocess.TimeoutExpired:
            print("⏰ MyPy command timed out")
            return {
                "returncode": 1,
                "stdout": "",
                "stderr": "Command timed out",
                "success": False
            }
        except Exception as e:
            print(f"❌ Error running MyPy: {e}")
            return {
                "returncode": 1,
                "stdout": "",
                "stderr": str(e),
                "success": False
            }
    
    def check_file_structure(self) -> Dict[str, Any]:
        """Check the file structure for MyPy compatibility."""
        print("\n📁 Checking file structure...")
        print("=" * 30)
        
        issues = []
        existing_files = []
        missing_files = []
        
        # Check core directory
        if os.path.exists(self.core_directory):
            print(f"✅ {self.core_directory} exists")
            existing_files.append(self.core_directory)
        else:
            issues.append(f"❌ {self.core_directory} missing")
            missing_files.append(self.core_directory)
        
        # Check individual files
        for file_path in self.type_check_files:
            if os.path.exists(file_path):
                print(f"✅ {file_path} exists")
                existing_files.append(file_path)
            else:
                print(f"❌ {file_path} missing")
                missing_files.append(file_path)
        
        # Check test files
        for file_path in self.test_files:
            if os.path.exists(file_path):
                print(f"✅ {file_path} exists")
                existing_files.append(file_path)
            else:
                print(f"❌ {file_path} missing")
                missing_files.append(file_path)
        
        return {
            "issues": issues,
            "existing_files": existing_files,
            "missing_files": missing_files,
            "valid": len(issues) == 0
        }
    
    def run_type_checking_tests(self) -> Dict[str, Any]:
        """Run comprehensive type checking tests."""
        print("\n🔍 Running type checking tests...")
        print("=" * 40)
        
        test_results = {}
        
        # Test 1: Basic mypy check on core directory
        print("\n1️⃣ Testing core directory type checking...")
        result1 = self.test_mypy_command(["mypy", self.core_directory])
        test_results["core_directory"] = result1
        
        # Test 2: Check specific files
        print("\n2️⃣ Testing individual file type checking...")
        for file_path in self.type_check_files[:2]:  # Test first 2 files
            if os.path.exists(file_path):
                print(f"\n   Testing {file_path}...")
                result = self.test_mypy_command(["mypy", file_path])
                test_results[f"file_{file_path}"] = result
        
        # Test 3: Check test files
        print("\n3️⃣ Testing test files...")
        for file_path in self.test_files:
            if os.path.exists(file_path):
                print(f"\n   Testing {file_path}...")
                result = self.test_mypy_command(["mypy", file_path])
                test_results[f"test_{file_path}"] = result
        
        # Test 4: Full project check (simplified)
        print("\n4️⃣ Testing full project check...")
        result4 = self.test_mypy_command(["mypy", self.core_directory, "apply_windows_cli_compatibility.py"])
        test_results["full_project"] = result4
        
        return test_results
    
    def generate_mypy_report(self, test_results: Dict[str, Any]) -> str:
        """Generate a comprehensive MyPy report."""
        print("\n📊 Generating MyPy Report...")
        print("=" * 40)
        
        report_lines = [
            "# MyPy Type Checking Report",
            "=" * 30,
            "",
            "## Test Results:",
            ""
        ]
        
        success_count = 0
        total_count = 0
        
        for test_name, result in test_results.items():
            total_count += 1
            status = "✅ PASS" if result.get("success", False) else "❌ FAIL"
            report_lines.append(f"- **{test_name}**: {status}")
            
            if result.get("success", False):
                success_count += 1
            
            if result.get("stderr"):
                report_lines.append(f"  - Errors: {result['stderr'][:100]}...")
        
        report_lines.extend([
            "",
            f"## Summary:",
            f"- **Total Tests**: {total_count}",
            f"- **Passed**: {success_count}",
            f"- **Failed**: {total_count - success_count}",
            f"- **Success Rate**: {(success_count/total_count)*100:.1f}%" if total_count > 0 else "0%",
            "",
            "## Recommendations:",
            ""
        ])
        
        if success_count == total_count:
            report_lines.append("- ✅ All type checking tests passed!")
            report_lines.append("- ✅ MyPy configuration is working correctly")
            report_lines.append("- ✅ Ready for production use")
        else:
            report_lines.append("- ⚠️ Some type checking issues found")
            report_lines.append("- 🔧 Review MyPy output for specific errors")
            report_lines.append("- 📝 Consider adding type annotations where missing")
        
        report = "\n".join(report_lines)
        print(report)
        
        return report
    
    def fix_common_issues(self) -> Dict[str, Any]:
        """Fix common MyPy configuration issues."""
        print("\n🔧 Fixing common MyPy issues...")
        print("=" * 35)
        
        fixes_applied = []
        
        # Check if mypy.ini needs fixes
        if os.path.exists(self.mypy_config_file):
            try:
                with open(self.mypy_config_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                original_content = content
                
                # Fix invalid options
                if "warn_unused_type_ignore = True" in content:
                    content = content.replace("warn_unused_type_ignore = True", "# warn_unused_type_ignore = True  # Removed - not a valid option")
                    fixes_applied.append("Removed invalid warn_unused_type_ignore option")
                
                if "[mypy-test_*]" in content:
                    content = content.replace("[mypy-test_*]", "[mypy-tests.*]")
                    fixes_applied.append("Fixed malformed test pattern [mypy-test_*] → [mypy-tests.*]")
                
                if content != original_content:
                    with open(self.mypy_config_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print("✅ Applied fixes to mypy.ini")
                else:
                    print("✅ No fixes needed for mypy.ini")
                    
            except Exception as e:
                print(f"⚠️ Could not fix mypy.ini: {e}")
        
        return {"fixes_applied": fixes_applied}
    
    def run(self) -> bool:
        """Run the complete MyPy fix process."""
        print("🔧 MyPy Issue Fixer")
        print("=" * 40)
        
        # Step 1: Validate configuration
        config_result = self.validate_mypy_config()
        
        # Step 2: Check file structure
        structure_result = self.check_file_structure()
        
        # Step 3: Fix common issues
        fixes_result = self.fix_common_issues()
        
        # Step 4: Run type checking tests
        test_results = self.run_type_checking_tests()
        
        # Step 5: Generate report
        report = self.generate_mypy_report(test_results)
        
        # Save report
        try:
            with open("mypy_report.md", 'w', encoding='utf-8') as f:
                f.write(report)
            print("\n📄 Report saved to mypy_report.md")
        except Exception as e:
            print(f"⚠️ Could not save report: {e}")
        
        # Summary
        print("\n" + "=" * 50)
        print("📊 FINAL SUMMARY")
        print("=" * 50)
        
        success_count = sum(1 for result in test_results.values() if result.get("success", False))
        total_count = len(test_results)
        
        print(f"✅ Configuration valid: {config_result['valid']}")
        print(f"✅ File structure valid: {structure_result['valid']}")
        print(f"🔧 Fixes applied: {len(fixes_result['fixes_applied'])}")
        print(f"🧪 Type checking tests: {success_count}/{total_count} passed")
        
        if success_count == total_count and config_result['valid'] and structure_result['valid']:
            print("\n🎉 SUCCESS: All MyPy issues resolved!")
            print("✅ Type checking is working correctly")
            print("✅ Ready for production use")
            return True
        else:
            print("\n⚠️ WARNING: Some issues remain")
            print("🔧 Review the report above for details")
            return False


def main():
    """Main function."""
    fixer = MyPyIssueFixer()
    success = fixer.run()
    
    if success:
        print("\n✅ All MyPy issues have been resolved!")
        sys.exit(0)
    else:
        print("\n❌ Some MyPy issues remain. Please review the report.")
        sys.exit(1)


if __name__ == "__main__":
    main() 