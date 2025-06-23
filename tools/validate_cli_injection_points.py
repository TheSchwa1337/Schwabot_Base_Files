#!/usr/bin/env python3
"""
CLI Injection Points Validation Script.

Scans the codebase for any remaining unsafe print/logging calls and CLI injection points.
This ensures complete CLI safety across all modules.
"""

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Import the centralized CLI handler for testing
try:
    from core.utils.windows_cli_compatibility import (
        WindowsCliCompatibilityHandler,
        safe_print,
        safe_format_error,
        log_safe,
    )
    CLI_HANDLER_AVAILABLE = True
except ImportError:
    CLI_HANDLER_AVAILABLE = False
    print("[ERROR] Centralized CLI handler not available")


class CLIInjectionPointValidator:
    """CLI Injection Points Validator."""
    
    def __init__(self, root_dir: str = "."):
        """Initialize the validator."""
        self.root_dir = Path(root_dir)
        self.unsafe_patterns = {
            "raw_print": re.compile(r'print\s*\(\s*["\']([^"\']*[🚀✅❌⚠️🚨🎯📊🔧⚡🎉💥🔥❄️⭐🔄⏳🛠️🔍📈📉💰🧪⚖️🌡️🔬⚙️🔒🗂️🔑🛡️🧮📐🔢∞φπ∑∫∇Δσμλθαβγδεζηικλμνξοπρστυφχψω→←↑↓↔↕⇒⇐⇔∀∃∄∈∉∋∌⊂⊃⊆⊇∪∩∅ℕℤℚℝℂℙℵℶℷℸ][^"\']*["\'])', re.UNICODE),
            "raw_logger": re.compile(r'(logger\.(?:info|warning|error|debug|critical))\s*\(\s*["\']([^"\']*[🚀✅❌⚠️🚨🎯📊🔧⚡🎉💥🔥❄️⭐🔄⏳🛠️🔍📈📉💰🧪⚖️🌡️🔬⚙️🔒🗂️🔑🛡️🧮📐🔢∞φπ∑∫∇Δσμλθαβγδεζηικλμνξοπρστυφχψω→←↑↓↔↕⇒⇐⇔∀∃∄∈∉∋∌⊂⊃⊆⊇∪∩∅ℕℤℚℝℂℙℵℶℷℸ][^"\']*["\'])', re.UNICODE),
            "unsafe_exception": re.compile(r'except\s+Exception\s+as\s+\w+:\s*\n\s*(?:print|logger\.(?:error|warning|critical))\s*\(\s*["\']([^"\']*["\'])'),
            "local_cli_handler": re.compile(r'class\s+WindowsCliCompatibilityHandler\s*:'),
            "missing_cli_import": re.compile(r'(?:print|logger\.(?:info|warning|error|debug|critical))\s*\(\s*["\']([^"\']*[🚀✅❌⚠️🚨🎯📊🔧⚡🎉💥🔥❄️⭐🔄⏳🛠️🔍📈📉💰🧪⚖️🌡️🔬⚙️🔒🗂️🔑🛡️🧮📐🔢∞φπ∑∫∇Δσμλθαβγδεζηικλμνξοπρστυφχψω→←↑↓↔↕⇒⇐⇔∀∃∄∈∉∋∌⊂⊃⊆⊇∪∩∅ℕℤℚℝℂℙℵℶℷℸ][^"\']*["\'])'),
        }
        
        self.validation_results = {
            "files_scanned": 0,
            "unsafe_prints": [],
            "unsafe_logs": [],
            "unsafe_exceptions": [],
            "local_handlers": [],
            "missing_imports": [],
            "clean_files": [],
        }

    def scan_codebase(self) -> Dict[str, any]:
        """Scan the entire codebase for CLI injection points."""
        safe_print("🔍 Scanning codebase for CLI injection points...")
        
        # Scan all Python files
        python_files = list(self.root_dir.rglob("*.py"))
        
        for py_file in python_files:
            if py_file.is_file():
                self.scan_file(py_file)
        
        return self.validation_results

    def scan_file(self, file_path: Path) -> None:
        """Scan a single file for CLI injection points."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            relative_path = str(file_path.relative_to(self.root_dir))
            self.validation_results["files_scanned"] += 1
            
            issues_found = []
            
            # Check for unsafe print calls
            print_matches = self.unsafe_patterns["raw_print"].findall(content)
            if print_matches:
                self.validation_results["unsafe_prints"].append({
                    "file": relative_path,
                    "matches": print_matches
                })
                issues_found.append(f"Unsafe prints: {len(print_matches)}")
            
            # Check for unsafe logging calls
            log_matches = self.unsafe_patterns["raw_logger"].findall(content)
            if log_matches:
                self.validation_results["unsafe_logs"].append({
                    "file": relative_path,
                    "matches": log_matches
                })
                issues_found.append(f"Unsafe logs: {len(log_matches)}")
            
            # Check for unsafe exception handling
            exception_matches = self.unsafe_patterns["unsafe_exception"].findall(content)
            if exception_matches:
                self.validation_results["unsafe_exceptions"].append({
                    "file": relative_path,
                    "matches": exception_matches
                })
                issues_found.append(f"Unsafe exceptions: {len(exception_matches)}")
            
            # Check for local CLI handler definitions
            if self.unsafe_patterns["local_cli_handler"].search(content):
                self.validation_results["local_handlers"].append(relative_path)
                issues_found.append("Local CLI handler definition")
            
            # Check for missing CLI imports when Unicode is used
            if self.unsafe_patterns["missing_cli_import"].search(content):
                if "from core.utils.windows_cli_compatibility import" not in content:
                    self.validation_results["missing_imports"].append(relative_path)
                    issues_found.append("Missing CLI import")
            
            if not issues_found:
                self.validation_results["clean_files"].append(relative_path)
            
        except Exception as e:
            safe_print(f"❌ Error scanning {file_path}: {safe_format_error(e, 'file_scan')}")

    def run_grep_validation(self) -> Dict[str, any]:
        """Run grep commands to validate CLI injection points."""
        safe_print("🔍 Running grep validation...")
        
        grep_results = {}
        
        # Check for unsafe print calls
        try:
            result = subprocess.run(
                ["grep", "-r", "print(", "."],
                capture_output=True,
                text=True,
                cwd=self.root_dir
            )
            
            if result.stdout:
                # Filter out safe_print calls
                unsafe_prints = []
                for line in result.stdout.splitlines():
                    if "safe_print" not in line and "print(" in line:
                        unsafe_prints.append(line.strip())
                
                grep_results["unsafe_prints"] = unsafe_prints
        except Exception as e:
            grep_results["grep_error"] = str(e)
        
        # Check for unsafe logger calls
        try:
            result = subprocess.run(
                ["grep", "-r", "logger.error(", "."],
                capture_output=True,
                text=True,
                cwd=self.root_dir
            )
            
            if result.stdout:
                # Filter out safe_format_error calls
                unsafe_logs = []
                for line in result.stdout.splitlines():
                    if "safe_format_error" not in line and "logger.error(" in line:
                        unsafe_logs.append(line.strip())
                
                grep_results["unsafe_logs"] = unsafe_logs
        except Exception as e:
            grep_results["grep_error"] = str(e)
        
        return grep_results

    def run_flake8_check(self) -> Dict[str, any]:
        """Run flake8 check for CLI-related issues."""
        safe_print("🔍 Running flake8 check...")
        
        try:
            result = subprocess.run(
                ["flake8", "--count", "--select=E9,F63,F7,F82", "--show-source", "--statistics"],
                capture_output=True,
                text=True,
                cwd=self.root_dir
            )
            
            return {
                "success": result.returncode == 0,
                "error_count": len(result.stdout.splitlines()) if result.stdout else 0,
                "output": result.stdout,
                "errors": result.stderr,
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    def generate_report(self) -> Dict[str, any]:
        """Generate comprehensive validation report."""
        total_issues = (
            len(self.validation_results["unsafe_prints"]) +
            len(self.validation_results["unsafe_logs"]) +
            len(self.validation_results["unsafe_exceptions"]) +
            len(self.validation_results["local_handlers"]) +
            len(self.validation_results["missing_imports"])
        )
        
        clean_rate = len(self.validation_results["clean_files"]) / self.validation_results["files_scanned"] if self.validation_results["files_scanned"] > 0 else 0
        
        return {
            "summary": {
                "files_scanned": self.validation_results["files_scanned"],
                "clean_files": len(self.validation_results["clean_files"]),
                "total_issues": total_issues,
                "clean_rate": clean_rate,
            },
            "details": self.validation_results,
            "recommendations": self.generate_recommendations(),
        }

    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        if self.validation_results["unsafe_prints"]:
            recommendations.append(f"Replace {len(self.validation_results['unsafe_prints'])} unsafe print calls with safe_print")
        
        if self.validation_results["unsafe_logs"]:
            recommendations.append(f"Replace {len(self.validation_results['unsafe_logs'])} unsafe logging calls with log_safe")
        
        if self.validation_results["unsafe_exceptions"]:
            recommendations.append(f"Update {len(self.validation_results['unsafe_exceptions'])} unsafe exception handlers")
        
        if self.validation_results["local_handlers"]:
            recommendations.append(f"Remove {len(self.validation_results['local_handlers'])} local CLI handler definitions")
        
        if self.validation_results["missing_imports"]:
            recommendations.append(f"Add CLI imports to {len(self.validation_results['missing_imports'])} files")
        
        if not recommendations:
            recommendations.append("All files are CLI-safe! No action needed.")
        
        return recommendations

    def print_detailed_report(self, report: Dict[str, any]) -> None:
        """Print detailed validation report."""
        safe_print("\n" + "=" * 80)
        safe_print("CLI INJECTION POINTS VALIDATION REPORT")
        safe_print("=" * 80)
        
        summary = report["summary"]
        safe_print(f"\n📊 SUMMARY:")
        safe_print(f"  Files Scanned: {summary['files_scanned']}")
        safe_print(f"  Clean Files: {summary['clean_files']}")
        safe_print(f"  Total Issues: {summary['total_issues']}")
        safe_print(f"  Clean Rate: {summary['clean_rate']:.2%}")
        
        details = report["details"]
        
        if details["unsafe_prints"]:
            safe_print(f"\n❌ UNSAFE PRINT CALLS ({len(details['unsafe_prints'])} files):")
            for issue in details["unsafe_prints"]:
                safe_print(f"  • {issue['file']}: {len(issue['matches'])} unsafe prints")
        
        if details["unsafe_logs"]:
            safe_print(f"\n❌ UNSAFE LOGGING CALLS ({len(details['unsafe_logs'])} files):")
            for issue in details["unsafe_logs"]:
                safe_print(f"  • {issue['file']}: {len(issue['matches'])} unsafe logs")
        
        if details["unsafe_exceptions"]:
            safe_print(f"\n❌ UNSAFE EXCEPTION HANDLING ({len(details['unsafe_exceptions'])} files):")
            for issue in details["unsafe_exceptions"]:
                safe_print(f"  • {issue['file']}: {len(issue['matches'])} unsafe exceptions")
        
        if details["local_handlers"]:
            safe_print(f"\n❌ LOCAL CLI HANDLER DEFINITIONS ({len(details['local_handlers'])} files):")
            for file_path in details["local_handlers"]:
                safe_print(f"  • {file_path}")
        
        if details["missing_imports"]:
            safe_print(f"\n❌ MISSING CLI IMPORTS ({len(details['missing_imports'])} files):")
            for file_path in details["missing_imports"]:
                safe_print(f"  • {file_path}")
        
        if details["clean_files"]:
            safe_print(f"\n✅ CLEAN FILES ({len(details['clean_files'])} files):")
            for file_path in details["clean_files"][:10]:  # Show first 10
                safe_print(f"  • {file_path}")
            if len(details["clean_files"]) > 10:
                safe_print(f"  ... and {len(details['clean_files']) - 10} more")
        
        safe_print(f"\n📋 RECOMMENDATIONS:")
        for rec in report["recommendations"]:
            safe_print(f"  • {rec}")
        
        # Overall status
        if summary["total_issues"] == 0:
            safe_print(f"\n🎉 VALIDATION STATUS: CLEAN")
            safe_print("   All files are CLI-safe and ready for deployment!")
        elif summary["clean_rate"] > 0.8:
            safe_print(f"\n⚠️ VALIDATION STATUS: MOSTLY CLEAN")
            safe_print("   Most files are CLI-safe, but some issues remain.")
        else:
            safe_print(f"\n❌ VALIDATION STATUS: NEEDS ATTENTION")
            safe_print("   Multiple CLI safety issues found - review and fix.")


def main():
    """Main validation function."""
    safe_print("=" * 80)
    safe_print("CLI INJECTION POINTS VALIDATION")
    safe_print("=" * 80)
    
    # Initialize validator
    validator = CLIInjectionPointValidator()
    
    # Scan codebase
    scan_results = validator.scan_codebase()
    
    # Run grep validation
    grep_results = validator.run_grep_validation()
    
    # Run flake8 check
    flake8_results = validator.run_flake8_check()
    
    # Generate and print report
    report = validator.generate_report()
    validator.print_detailed_report(report)
    
    # Print additional results
    if grep_results:
        safe_print(f"\n🔍 GREP VALIDATION RESULTS:")
        for key, value in grep_results.items():
            if isinstance(value, list):
                safe_print(f"  {key}: {len(value)} items")
            else:
                safe_print(f"  {key}: {value}")
    
    if flake8_results:
        safe_print(f"\n🔍 FLAKE8 RESULTS:")
        safe_print(f"  Success: {flake8_results.get('success', False)}")
        safe_print(f"  Error Count: {flake8_results.get('error_count', 0)}")
    
    return report["summary"]["total_issues"] == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 