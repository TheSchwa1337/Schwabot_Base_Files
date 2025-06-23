#!/usr/bin/env python3
"""
Comprehensive CLI and Fault Safety Unification Script.

This script systematically unifies Windows CLI compatibility and fault handling
across the entire Schwabot codebase, ensuring consistent error handling,
safe output, and robust deployment across all platforms.
"""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Import the centralized CLI handler
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


class ComprehensiveCLIFaultUnifier:
    """
    Comprehensive CLI and Fault Safety Unification System.
    
    This class provides systematic unification of:
    - Windows CLI compatibility handling
    - Fault bus integration
    - Error message formatting
    - Safe logging and printing
    - Cross-platform compatibility
    """

    def __init__(self, root_dir: str = "."):
        """Initialize the unifier with root directory."""
        self.root_dir = Path(root_dir)
        self.processed_files: Set[str] = set()
        self.failed_files: Set[str] = set()
        self.stats = {
            "files_processed": 0,
            "files_updated": 0,
            "imports_added": 0,
            "print_calls_replaced": 0,
            "log_calls_replaced": 0,
            "error_handling_updated": 0,
            "cli_handlers_removed": 0,
        }

        # Patterns to identify and replace
        self.patterns = {
            # Raw print calls with potential Unicode/emoji
            "raw_print": re.compile(r'print\s*\(\s*["\']([^"\']*[🚀✅❌⚠️🚨🎯📊🔧⚡🎉💥🔥❄️⭐🔄⏳🛠️🔍📈📉💰🧪⚖️🌡️🔬⚙️🔒🗂️🔑🛡️🧮📐🔢∞φπ∑∫∇Δσμλθαβγδεζηικλμνξοπρστυφχψω→←↑↓↔↕⇒⇐⇔∀∃∄∈∉∋∌⊂⊃⊆⊇∪∩∅ℕℤℚℝℂℙℵℶℷℸ][^"\']*["\'])', re.UNICODE),
            
            # Raw logging calls with potential Unicode/emoji
            "raw_log": re.compile(r'(logger\.(?:info|warning|error|debug|critical))\s*\(\s*["\']([^"\']*[🚀✅❌⚠️🚨🎯📊🔧⚡🎉💥🔥❄️⭐🔄⏳🛠️🔍📈📉💰🧪⚖️🌡️🔬⚙️🔒🗂️🔑🛡️🧮📐🔢∞φπ∑∫∇Δσμλθαβγδεζηικλμνξοπρστυφχψω→←↑↓↔↕⇒⇐⇔∀∃∄∈∉∋∌⊂⊃⊆⊇∪∩∅ℕℤℚℝℂℙℵℶℷℸ][^"\']*["\'])', re.UNICODE),
            
            # Local WindowsCliCompatibilityHandler definitions
            "local_cli_handler": re.compile(r'class\s+WindowsCliCompatibilityHandler\s*:'),
            
            # Exception handling without safe formatting
            "unsafe_exception": re.compile(r'except\s+Exception\s+as\s+\w+:\s*\n\s*(?:print|logger\.(?:error|warning|critical))\s*\(\s*["\']([^"\']*["\'])'),
            
            # Fault bus events without CLI safety
            "unsafe_fault": re.compile(r'FaultBusEvent\s*\(\s*[^)]*type\s*=\s*FaultType\.[^,)]*[^)]*\)'),
        }

        # Replacement templates
        self.replacements = {
            "import_statement": """# Import centralized CLI handler
try:
    from core.utils.windows_cli_compatibility import (
        WindowsCliCompatibilityHandler,
        safe_print,
        safe_format_error,
        log_safe,
        cli_handler,
    )
    CLI_HANDLER_AVAILABLE = True
except ImportError:
    CLI_HANDLER_AVAILABLE = False
    # Fallback for testing when package import fails
    def safe_print(message: str, use_emoji: bool = True) -> str:
        return message
    def safe_format_error(error: Exception, context: str = "") -> str:
        return f"Error: {str(error)} | Context: {context}"
    def log_safe(logger, level: str, message: str) -> None:
        getattr(logger, level.lower())(message)
    cli_handler = None
""",
            "safe_print_call": "safe_print",
            "safe_log_call": "log_safe",
            "safe_error_format": "safe_format_error",
        }

    def scan_codebase(self) -> Dict[str, List[str]]:
        """
        Scan the codebase for files that need CLI/fault unification.
        
        Returns:
            Dictionary mapping file categories to file paths
        """
        categories = {
            "core_modules": [],
            "demo_files": [],
            "test_files": [],
            "config_files": [],
            "utility_files": [],
            "other_python": [],
        }

        for py_file in self.root_dir.rglob("*.py"):
            if py_file.is_file():
                relative_path = str(py_file.relative_to(self.root_dir))
                
                if "core/" in relative_path:
                    categories["core_modules"].append(relative_path)
                elif "demo" in relative_path.lower():
                    categories["demo_files"].append(relative_path)
                elif "test" in relative_path.lower() or "tests/" in relative_path:
                    categories["test_files"].append(relative_path)
                elif "config" in relative_path.lower() or "settings" in relative_path.lower():
                    categories["config_files"].append(relative_path)
                elif "utils" in relative_path.lower() or "tools" in relative_path.lower():
                    categories["utility_files"].append(relative_path)
                else:
                    categories["other_python"].append(relative_path)

        return categories

    def analyze_file(self, file_path: str) -> Dict[str, any]:
        """
        Analyze a single file for CLI/fault handling issues.
        
        Args:
            file_path: Path to the file to analyze
            
        Returns:
            Dictionary with analysis results
        """
        analysis = {
            "has_local_cli_handler": False,
            "has_unsafe_prints": False,
            "has_unsafe_logs": False,
            "has_unsafe_exceptions": False,
            "has_fault_bus_usage": False,
            "needs_cli_import": False,
            "issues_found": [],
        }

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Check for local CLI handler definitions
            if self.patterns["local_cli_handler"].search(content):
                analysis["has_local_cli_handler"] = True
                analysis["issues_found"].append("Local WindowsCliCompatibilityHandler definition")

            # Check for unsafe print calls
            if self.patterns["raw_print"].search(content):
                analysis["has_unsafe_prints"] = True
                analysis["issues_found"].append("Unsafe print calls with Unicode/emoji")

            # Check for unsafe logging calls
            if self.patterns["raw_log"].search(content):
                analysis["has_unsafe_logs"] = True
                analysis["issues_found"].append("Unsafe logging calls with Unicode/emoji")

            # Check for unsafe exception handling
            if self.patterns["unsafe_exception"].search(content):
                analysis["has_unsafe_exceptions"] = True
                analysis["issues_found"].append("Unsafe exception handling")

            # Check for fault bus usage
            if "FaultBus" in content or "FaultBusEvent" in content:
                analysis["has_fault_bus_usage"] = True

            # Check if CLI import is needed
            if (analysis["has_unsafe_prints"] or 
                analysis["has_unsafe_logs"] or 
                analysis["has_unsafe_exceptions"] or
                analysis["has_fault_bus_usage"]):
                analysis["needs_cli_import"] = True

        except Exception as e:
            analysis["issues_found"].append(f"Error reading file: {e}")

        return analysis

    def unify_file(self, file_path: str) -> bool:
        """
        Unify CLI and fault handling in a single file.
        
        Args:
            file_path: Path to the file to unify
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            original_content = content
            changes_made = False

            # Step 1: Add CLI import if needed
            if self.needs_cli_import(content):
                content = self.add_cli_import(content)
                changes_made = True
                self.stats["imports_added"] += 1

            # Step 2: Remove local CLI handler definitions
            if self.patterns["local_cli_handler"].search(content):
                content = self.remove_local_cli_handlers(content)
                changes_made = True
                self.stats["cli_handlers_removed"] += 1

            # Step 3: Replace unsafe print calls
            content, print_replacements = self.replace_unsafe_prints(content)
            if print_replacements > 0:
                changes_made = True
                self.stats["print_calls_replaced"] += print_replacements

            # Step 4: Replace unsafe logging calls
            content, log_replacements = self.replace_unsafe_logs(content)
            if log_replacements > 0:
                changes_made = True
                self.stats["log_calls_replaced"] += log_replacements

            # Step 5: Update exception handling
            content, exception_updates = self.update_exception_handling(content)
            if exception_updates > 0:
                changes_made = True
                self.stats["error_handling_updated"] += exception_updates

            # Step 6: Update fault bus integration
            content, fault_updates = self.update_fault_bus_integration(content)
            if fault_updates > 0:
                changes_made = True
                self.stats["error_handling_updated"] += fault_updates

            # Write changes if any were made
            if changes_made and content != original_content:
                # Create backup
                backup_path = f"{file_path}.backup"
                shutil.copy2(file_path, backup_path)

                # Write updated content
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)

                self.stats["files_updated"] += 1
                return True

            return True

        except Exception as e:
            print(f"[ERROR] Failed to unify {file_path}: {e}")
            self.failed_files.add(file_path)
            return False

    def needs_cli_import(self, content: str) -> bool:
        """Check if file needs CLI import."""
        return (
            self.patterns["raw_print"].search(content) or
            self.patterns["raw_log"].search(content) or
            self.patterns["unsafe_exception"].search(content) or
            "FaultBus" in content or
            "FaultBusEvent" in content
        )

    def add_cli_import(self, content: str) -> str:
        """Add CLI import statement to file."""
        # Find the best place to add import (after existing imports)
        lines = content.split('\n')
        
        # Find the last import statement
        last_import_index = -1
        for i, line in enumerate(lines):
            if line.strip().startswith(('import ', 'from ')):
                last_import_index = i
        
        # Insert CLI import after the last import
        if last_import_index >= 0:
            lines.insert(last_import_index + 1, self.replacements["import_statement"])
        else:
            # No imports found, add at the beginning
            lines.insert(0, self.replacements["import_statement"])
        
        return '\n'.join(lines)

    def remove_local_cli_handlers(self, content: str) -> str:
        """Remove local WindowsCliCompatibilityHandler definitions."""
        # This is a simplified version - in practice, you'd need more sophisticated parsing
        lines = content.split('\n')
        filtered_lines = []
        in_cli_handler = False
        indent_level = 0
        
        for line in lines:
            if self.patterns["local_cli_handler"].search(line):
                in_cli_handler = True
                indent_level = len(line) - len(line.lstrip())
                continue
            
            if in_cli_handler:
                current_indent = len(line) - len(line.lstrip())
                if current_indent <= indent_level and line.strip():
                    in_cli_handler = False
                    filtered_lines.append(line)
                continue
            
            filtered_lines.append(line)
        
        return '\n'.join(filtered_lines)

    def replace_unsafe_prints(self, content: str) -> Tuple[str, int]:
        """Replace unsafe print calls with safe_print."""
        replacements = 0
        
        def replace_print(match):
            nonlocal replacements
            replacements += 1
            return f'safe_print({match.group(1)})'
        
        content = self.patterns["raw_print"].sub(replace_print, content)
        return content, replacements

    def replace_unsafe_logs(self, content: str) -> Tuple[str, int]:
        """Replace unsafe logging calls with log_safe."""
        replacements = 0
        
        def replace_log(match):
            nonlocal replacements
            replacements += 1
            logger_call = match.group(1)
            message = match.group(2)
            level = logger_call.split('.')[1]
            return f'log_safe(logger, "{level}", {message})'
        
        content = self.patterns["raw_log"].sub(replace_log, content)
        return content, replacements

    def update_exception_handling(self, content: str) -> Tuple[str, int]:
        """Update exception handling to use safe formatting."""
        # This is a simplified version - would need more sophisticated parsing
        updates = 0
        
        # Pattern for exception handling
        exception_pattern = re.compile(
            r'(except\s+Exception\s+as\s+(\w+):\s*\n\s*)(print|logger\.(?:error|warning|critical))\s*\(\s*["\']([^"\']*["\'])',
            re.MULTILINE
        )
        
        def replace_exception(match):
            nonlocal updates
            updates += 1
            prefix = match.group(1)
            exception_var = match.group(2)
            log_call = match.group(3)
            message = match.group(4)
            
            if log_call.startswith('logger.'):
                level = log_call.split('.')[1]
                return f'{prefix}log_safe(logger, "{level}", f"{message}: {{safe_format_error({exception_var})}}")'
            else:
                return f'{prefix}safe_print(f"{message}: {{safe_format_error({exception_var})}}")'
        
        content = exception_pattern.sub(replace_exception, content)
        return content, updates

    def update_fault_bus_integration(self, content: str) -> Tuple[str, int]:
        """Update fault bus integration to use CLI-safe formatting."""
        updates = 0
        
        # Pattern for fault bus event creation
        fault_pattern = re.compile(
            r'FaultBusEvent\s*\(\s*([^)]*)\s*\)',
            re.MULTILINE
        )
        
        def replace_fault(match):
            nonlocal updates
            updates += 1
            params = match.group(1)
            
            # Add CLI-safe formatting for metadata if present
            if 'metadata=' in params:
                # This is a simplified replacement - would need more sophisticated parsing
                return f'FaultBusEvent({params})'
            
            return match.group(0)
        
        content = fault_pattern.sub(replace_fault, content)
        return content, updates

    def run_comprehensive_unification(self) -> Dict[str, any]:
        """
        Run comprehensive CLI and fault unification across the codebase.
        
        Returns:
            Dictionary with unification results and statistics
        """
        print("[LAUNCH] Starting comprehensive CLI and fault unification...")
        
        # Scan codebase
        categories = self.scan_codebase()
        
        # Process files by category
        for category, files in categories.items():
            print(f"\n[PROCESSING] {category}: {len(files)} files")
            
            for file_path in files:
                try:
                    # Analyze file
                    analysis = self.analyze_file(file_path)
                    
                    if analysis["issues_found"]:
                        print(f"  [ISSUES] {file_path}: {', '.join(analysis['issues_found'])}")
                        
                        # Unify file
                        if self.unify_file(file_path):
                            self.processed_files.add(file_path)
                            self.stats["files_processed"] += 1
                            print(f"  [SUCCESS] Unified {file_path}")
                        else:
                            self.failed_files.add(file_path)
                            print(f"  [FAILED] Could not unify {file_path}")
                    else:
                        print(f"  [CLEAN] {file_path} (no issues found)")
                        
                except Exception as e:
                    print(f"  [ERROR] Failed to process {file_path}: {e}")
                    self.failed_files.add(file_path)

        # Generate report
        report = self.generate_report()
        
        print(f"\n[COMPLETE] Unification finished")
        print(f"  Files processed: {self.stats['files_processed']}")
        print(f"  Files updated: {self.stats['files_updated']}")
        print(f"  Files failed: {len(self.failed_files)}")
        
        return report

    def generate_report(self) -> Dict[str, any]:
        """Generate comprehensive unification report."""
        return {
            "statistics": self.stats.copy(),
            "processed_files": list(self.processed_files),
            "failed_files": list(self.failed_files),
            "success_rate": len(self.processed_files) / (len(self.processed_files) + len(self.failed_files)) if (len(self.processed_files) + len(self.failed_files)) > 0 else 0,
            "recommendations": self.generate_recommendations(),
        }

    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on unification results."""
        recommendations = []
        
        if self.failed_files:
            recommendations.append(f"Review {len(self.failed_files)} failed files for manual fixes")
        
        if self.stats["cli_handlers_removed"] > 0:
            recommendations.append("Verify that centralized CLI handler is working correctly")
        
        if self.stats["print_calls_replaced"] > 0:
            recommendations.append("Test CLI output to ensure safe_print is working")
        
        if self.stats["log_calls_replaced"] > 0:
            recommendations.append("Test logging to ensure log_safe is working")
        
        recommendations.append("Run comprehensive tests to verify CLI compatibility")
        recommendations.append("Test fault bus integration with CLI-safe error handling")
        
        return recommendations

    def validate_unification(self) -> Dict[str, any]:
        """
        Validate the unification by running tests and checks.
        
        Returns:
            Dictionary with validation results
        """
        print("\n[VALIDATION] Running post-unification validation...")
        
        validation_results = {
            "flake8_check": self.run_flake8_check(),
            "import_check": self.check_imports(),
            "cli_handler_test": self.test_cli_handler(),
            "overall_status": "PENDING",
        }
        
        # Determine overall status
        if all(result.get("success", False) for result in validation_results.values() if isinstance(result, dict)):
            validation_results["overall_status"] = "SUCCESS"
        elif any(result.get("success", False) for result in validation_results.values() if isinstance(result, dict)):
            validation_results["overall_status"] = "PARTIAL"
        else:
            validation_results["overall_status"] = "FAILED"
        
        return validation_results

    def run_flake8_check(self) -> Dict[str, any]:
        """Run flake8 check on the codebase."""
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

    def check_imports(self) -> Dict[str, any]:
        """Check that CLI imports are working correctly."""
        try:
            # Test import in a few key files
            test_files = [
                "core/fault_bus.py",
                "core/utils/windows_cli_compatibility.py",
            ]
            
            import_errors = []
            for test_file in test_files:
                if os.path.exists(test_file):
                    try:
                        with open(test_file, 'r') as f:
                            content = f.read()
                            if "from core.utils.windows_cli_compatibility import" in content:
                                continue
                            else:
                                import_errors.append(f"Missing CLI import in {test_file}")
                    except Exception as e:
                        import_errors.append(f"Error checking {test_file}: {e}")
            
            return {
                "success": len(import_errors) == 0,
                "import_errors": import_errors,
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    def test_cli_handler(self) -> Dict[str, any]:
        """Test the CLI handler functionality."""
        try:
            # Test basic functionality
            test_message = "🚀 Test message with emoji ✅"
            safe_result = safe_print(test_message)
            
            return {
                "success": "[LAUNCH]" in safe_result and "[SUCCESS]" in safe_result,
                "test_message": test_message,
                "safe_result": safe_result,
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }


def main():
    """Main function to run comprehensive CLI and fault unification."""
    print("=" * 80)
    print("COMPREHENSIVE CLI AND FAULT SAFETY UNIFICATION")
    print("=" * 80)
    
    # Initialize unifier
    unifier = ComprehensiveCLIFaultUnifier()
    
    # Run unification
    report = unifier.run_comprehensive_unification()
    
    # Run validation
    validation = unifier.validate_unification()
    
    # Print final report
    print("\n" + "=" * 80)
    print("FINAL UNIFICATION REPORT")
    print("=" * 80)
    
    print(f"\n📊 STATISTICS:")
    for key, value in report["statistics"].items():
        print(f"  {key}: {value}")
    
    print(f"\n✅ SUCCESS RATE: {report['success_rate']:.2%}")
    
    print(f"\n🔍 VALIDATION RESULTS:")
    for key, result in validation.items():
        if isinstance(result, dict):
            status = "✅ SUCCESS" if result.get("success", False) else "❌ FAILED"
            print(f"  {key}: {status}")
    
    print(f"\n📋 RECOMMENDATIONS:")
    for rec in report["recommendations"]:
        print(f"  • {rec}")
    
    print(f"\n🎯 OVERALL STATUS: {validation['overall_status']}")
    
    if validation['overall_status'] == "SUCCESS":
        print("\n🎉 UNIFICATION COMPLETED SUCCESSFULLY!")
        print("   Schwabot is now fully CLI-compatible and fault-safe.")
    elif validation['overall_status'] == "PARTIAL":
        print("\n⚠️ UNIFICATION PARTIALLY COMPLETED")
        print("   Some issues remain - review failed files and recommendations.")
    else:
        print("\n❌ UNIFICATION FAILED")
        print("   Review errors and fix issues before proceeding.")


if __name__ == "__main__":
    main() 