#!/usr/bin/env python3
"""Enhanced CLI Compatibility Applicator.

Applies enhanced Windows CLI compatibility to mathematical validation systems
in the SchwaBot trading intelligence build.
"""

import logging
import os
import re
import shutil
from typing import Any, Dict, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import our enhanced compatibility handler
try:
    from core.enhanced_windows_cli_compatibility import (
        get_safe_reporter,
        safe_log,
        safe_print,
    )

    CLI_HANDLER_AVAILABLE = True
except ImportError:
    CLI_HANDLER_AVAILABLE = False
    print("Warning: Enhanced CLI compatibility handler not available")


class CliCompatibilityApplicator:
    """Applies enhanced CLI compatibility to mathematical validation systems"""

    # Files that need CLI compatibility enhancement
    TARGET_FILES = [
        "mathematical_integration_validator.py",
        "mathematical_integration_pathway_demo.py",
        "run_mathematical_integration_validation.py",
        "core/math_core.py",
        "core/mathlib.py",
        "core/mathlib_v2.py",
        "core/mathlib_v3.py",
        "core/master_orchestrator.py",
        "core/advanced_mathematical_core.py",
    ]
    
    # Import statement to add
    CLI_IMPORT_STATEMENT = """
# Enhanced Windows CLI compatibility
try:
    from core.enhanced_windows_cli_compatibility import (
        EnhancedWindowsCliCompatibilityHandler,
        cli_safe,
        safe_print,
        safe_log,
        get_safe_reporter
    )
    CLI_HANDLER_AVAILABLE = True
except ImportError:
    CLI_HANDLER_AVAILABLE = False
    # Fallback implementations
    def safe_print(msg, force_ascii=False):
        try:
            print(msg)
        except UnicodeEncodeError:
            print(msg.encode('ascii', errors='replace').decode('ascii'))

    def safe_log(logger, level, msg, context=""):
        try:
            getattr(logger, level.lower())(msg)
        except UnicodeEncodeError:
            getattr(logger, level.lower())(
                msg.encode('ascii', errors='replace').decode('ascii')
            )

    def cli_safe(func):
        return func

    def get_safe_reporter():
        def reporter(name, status, details="", metrics=None):
            status_text = "PASS" if status else "FAIL"
            return (
                f"[{status_text}] {name}" +
                (f": {details}" if details else "")
            )
        return reporter

"""

    def __init__(self):
        self.processed_files = []
        self.backup_files = []
        self.errors = []

    def create_backup(self, file_path: str) -> str:
        """Create backup of file before modification"""
        backup_path = f"{file_path}.cli_backup"
        try:
            shutil.copy2(file_path, backup_path)
            self.backup_files.append(backup_path)
            return backup_path
        except Exception as e:
            error_msg = f"Failed to create backup for {file_path}: {e}"
            self.errors.append(error_msg)
            logger.error(error_msg)
            return ""

    def add_cli_imports(self, content: str) -> str:
        """Add CLI compatibility imports to file content"""
        # Check if imports already exist
        if "enhanced_windows_cli_compatibility" in content:
            return content

        # Find appropriate insertion point
        import_pattern = r"^(import\s+\w+|from\s+\w+\s+import\s+.*?)$"
        import_matches = list(
            re.finditer(import_pattern, content, re.MULTILINE)
        )

        if import_matches:
            # Insert after last import
            last_import = import_matches[-1]
            insert_pos = content.find("\n", last_import.end()) + 1
        else:
            # Insert after shebang and docstring
            docstring_end = content.find('"""', content.find('"""') + 3)
            if docstring_end != -1:
                insert_pos = content.find("\n", docstring_end) + 1
            else:
                insert_pos = 0

        # Insert the CLI imports
        modified_content = (
            content[:insert_pos]
            + self.CLI_IMPORT_STATEMENT
            + content[insert_pos:]
        )

        return modified_content

    def process_file(self, file_path: str) -> Tuple[bool, str]:
        """Process a single file for CLI compatibility"""
        if not os.path.exists(file_path):
            error_msg = f"File not found: {file_path}"
            self.errors.append(error_msg)
            return False, error_msg

        try:
            # Create backup
            backup_path = self.create_backup(file_path)
            if not backup_path:
                return False, "Backup creation failed"

            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Apply CLI compatibility
            modified_content = self.add_cli_imports(content)

            # Check if modifications were made
            if modified_content != content:
                # Write modified content
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(modified_content)

                self.processed_files.append(file_path)
                success_msg = f"Successfully enhanced {file_path} with CLI compatibility"
                logger.info(success_msg)
                return True, success_msg
            else:
                success_msg = f"File {file_path} already CLI-compatible"
                logger.info(success_msg)
                return True, success_msg

        except Exception as e:
            error_msg = f"Error processing {file_path}: {e}"
            self.errors.append(error_msg)
            logger.error(error_msg)
            return False, error_msg

    def apply_to_all_targets(self) -> Dict[str, Any]:
        """Apply CLI compatibility to all target files"""
        results = {
            "processed": [],
            "errors": [],
            "total_files": len(self.TARGET_FILES),
            "success_count": 0,
            "error_count": 0,
        }

        logger.info(
            f"Starting CLI compatibility enhancement for {len(self.TARGET_FILES)} files"
        )

        for file_path in self.TARGET_FILES:
            success, message = self.process_file(file_path)
            
            if success:
                results["processed"].append({"file": file_path, "message": message})
                results["success_count"] += 1
            else:
                results["errors"].append({"file": file_path, "error": message})
                results["error_count"] += 1

        results["success_rate"] = (
            results["success_count"] / results["total_files"] * 100
            if results["total_files"] > 0 else 0
        )

        return results


def main():
    """Main CLI compatibility application function."""
    if CLI_HANDLER_AVAILABLE:
        safe_print("🚀 Enhanced CLI Compatibility Application Starting...")
        safe_print(
            "   Applying bulletproof Windows CLI handling to mathematical "
            "validation systems..."
        )
    else:
        print("[START] CLI Compatibility Application Starting...")

    # Initialize applicator
    applicator = CliCompatibilityApplicator()

    # Apply to all target files
    results = applicator.apply_to_all_targets()

    # Display results
    print("\n" + "=" * 70)
    if CLI_HANDLER_AVAILABLE:
        safe_print("🎉 CLI Compatibility Application Complete!")
    else:
        print("[COMPLETE] CLI Compatibility Application Complete!")
    print("=" * 70)

    if CLI_HANDLER_AVAILABLE:
        safe_print(f"📊 Processing Results:")
        safe_print(f"   Files Processed: {results['success_count']}/{results['total_files']}")
        safe_print(f"   Success Rate: {results['success_rate']:.1f}%")
        safe_print(f"   Errors: {results['error_count']}")

        if results["errors"]:
            safe_print("\n❌ Errors encountered:")
            for error in results["errors"]:
                safe_print(f"   {error['file']}: {error['error']}")

        if results["success_rate"] >= 90:
            safe_print(
                "\n🎉 EXCELLENT! All mathematical systems now have bulletproof "
                "CLI compatibility!"
            )
        else:
            safe_print("\n⚠️ PARTIAL SUCCESS: Some files may need manual review.")
            safe_print("   Check the error log for specific issues.")
    else:
        print(f"[RESULTS] Files Processed: {results['success_count']}/{results['total_files']}")
        print(f"[RESULTS] Success Rate: {results['success_rate']:.1f}%")
        print(f"[RESULTS] Errors: {results['error_count']}")


if __name__ == "__main__":
    main() 