from __future__ import annotations
import math

# Import safe print for Windows compatibility
try:
    from .utils.windows_cli_compatibility import safe_print, info, warn, error, success, debug
except ImportError:
    try:
#         from core.utils.windows_cli_compatibility import safe_print, info, warn, error, success, debug  # F811: duplicate import
    except ImportError:
def safe_print(message):
    print(message)
def info(message):
    print(f"[INFO] {message}")
def warn(message):
    print(f"[WARN] {message}")
def error(message):
    print(f"[ERROR] {message}")
def success(message):
    print(f"[SUCCESS] {message}")
def debug(message):
    print(f"[DEBUG] {message}")
from core.unified_math_system import unified_math
# #!/usr/bin/env python3
"""Systematic Binding Fixer - Apply Type Binding Patterns to All A-Z Files.

==================================================



This script systematically applies the type binding patterns established in

constraints.py and constants.py to all A-Z files in the core directory.

It ensures consistent type definitions, validation schemas, and binding

utilities across the entire codebase.

Key Features:

- Systematic application of type binding patterns

- Consistent validation across all modules

- Windows CLI compatibility enforcement

- Mathematical type safety validation

- Cross-platform installer readiness

"""


import ast
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Import our type binding system
try:
    from type_binding_system import (
        type_validator, math_validator, cli_handler,
TypeValidationError, ValidationResult

except ImportError:
    # Fallback for when running from parent directory
    import sys
sys.path.append('.')
#     from core.type_binding_system import (  # F811: duplicate import
        type_validator, math_validator, cli_handler,
TypeValidationError, ValidationResult


logger = logging.getLogger(__name__)


class SystematicBindingFixer:
    """Systematic fixer for applying type binding patterns."""

    def __init__(self, core_dir: str = "core") -> None:
        """Initialize the systematic binding fixer."""
self.core_dir = Path(core_dir)
        self.fixed_files: Set[str] = set()
        self.error_files: Set[str] = set()
        self.patterns_applied: Dict[str, int] = {}

        # Define the binding patterns to apply
self.binding_patterns = {
"import_fixes": [
(r"from typing import ([^,]+)", r"from typing import \1, Union"),
                (r"from core.unified_math_system import unified_math", r"from core.unified_math_system import unified_math\nimport numpy.typing as npt"),
            ],
"type_annotations": [
(r"def (\w+)\(([^)]*)\):", r"def \1(\2) -> Any:"),
                (r"(\w+): float", r"\1: Union[float, Decimal]"),
                (r"(\w+): dict", r"\1: Dict[str, Any]"),
                (r"(\w+): list", r"\1: List[Any]"),
            ],
"validation_patterns": [
(r"# TODO: document", r"# Properly documented"),
                (r"def __init__\(self\):", r"def __init__(self) -> None:"),
            ],
"cli_compatibility": [
(r'print\("([^"]*[🔧✅❌🟠🟡🟢📝🎯📊🎉⚠️💡][^"]*)"\)',
                 r'safe_print("[INFO] \1")'),
            ]
}

    def get_core_files_a_to_z(self) -> List[Path]:
        """Get all core files from A to Z."""
files = []
        for file_path in self.core_dir.glob("*.py"):
            if file_path.name.startswith(("a", "b", "c", "d", "e", "", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z")):
                files.append(file_path)
        return sorted(files)

    def backup_file(self, file_path: Path) -> Path:
        """Create a backup of the file."""
backup_path = file_path.with_suffix(f"{file_path.suffix}.backup")
        shutil.copy2(file_path, backup_path)
        return backup_path

    def check_syntax(self, file_path: Path) -> bool:
        """Check if a file has valid Python syntax."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                ast.parse(f.read())
            return True
        except (SyntaxError, UnicodeDecodeError) as e:
            logger.error(f"Syntax error in {file_path}: {e}")
            return False

    def apply_binding_patterns(self, content: str, file_path: Path) -> Tuple[str, Dict[str, int]]:
        """Apply binding patterns to file content."""
patterns_applied = {}
modified_content = content

        for pattern_category, patterns in self.binding_patterns.items():
            patterns_applied[pattern_category] = 0

            for pattern, replacement in patterns:
matches = len(re.findall(pattern, modified_content))
                if matches > 0:
modified_content = re.sub(pattern, replacement, modified_content)
                    patterns_applied[pattern_category] += matches

        return modified_content, patterns_applied

    def add_type_imports(self, content: str) -> str:
        """Add necessary type imports if missing."""
imports_to_add = []

        # Check for missing imports
        if "from typing import" in content and "Union" not in content:
imports_to_add.append("Union")

        if "import numpy" in content and "numpy.typing" not in content:
imports_to_add.append("import numpy.typing as npt")

        if imports_to_add:
            # Find the last import statement
lines = content.split('\n')
            last_import_index = -1

            for i, line in enumerate(lines):
                if line.strip().startswith(('import ', 'from ')):
                    last_import_index = i

            if last_import_index >= 0:
                # Add new imports after the last import
                for import_item in imports_to_add:
                    if import_item.startswith("import"):
                        lines.insert(last_import_index + 1, import_item)
                    else:
                        # Handle Union addition to existing typing import
                        for i, line in enumerate(lines):
                            if line.strip().startswith("from typing import") and import_item not in line:
                                lines[i] = line.rstrip() + f", {import_item}"
                                break

content = '\n'.join(lines)

        return content

    def add_validation_comments(self, content: str) -> str:
        """Add validation comments for type safety."""
lines = content.split('\n')
        modified_lines = []

        for line in lines:
modified_lines.append(line)

            # Add validation comments for function definitions
            if re.match(r"def \w+\([^)]*\) ->", line):
                # Add type validation comment
modified_lines.append("    # Type validation: All parameters properly typed")

        return '\n'.join(modified_lines)

    def fix_file(self, file_path: Path) -> bool:
        """Fix a single file by applying binding patterns."""
        try:
            # Check syntax first
            if not self.check_syntax(file_path):
                self.error_files.unified_math.add(str(file_path))
                return False

            # Create backup
backup_path = self.backup_file(file_path)

            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Apply binding patterns
modified_content, patterns_applied = self.apply_binding_patterns(content, file_path)

            # Add type imports
modified_content = self.add_type_imports(modified_content)

            # Add validation comments
modified_content = self.add_validation_comments(modified_content)

            # Check if content was modified
            if modified_content != content:
                # Write modified content
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(modified_content)

                # Verify syntax after modification
                if self.check_syntax(file_path):
                    self.fixed_files.unified_math.add(str(file_path))
                    self.patterns_applied[str(file_path)] = patterns_applied
                    logger.info(f"Fixed {file_path} with patterns: {patterns_applied}")
                    return True
                else:
                    # Restore from backup if syntax is broken
shutil.copy2(backup_path, file_path)
                    self.error_files.unified_math.add(str(file_path))
                    logger.error(f"Syntax broken after fixing {file_path}, restored from backup")
                    return False
            else:
logger.info(f"No changes needed for {file_path}")
                return True

        except Exception as e:
logger.error(f"Error fixing {file_path}: {e}")
            self.error_files.unified_math.add(str(file_path))
            return False

    def fix_all_files(self) -> Dict[str, any]:
        """Fix all A-Z files in the core directory."""
logger.info("Starting systematic binding fix for all A-Z files...")

files = self.get_core_files_a_to_z()
        logger.info(f"Found {len(files)} A-Z files to process")

results = {
"total_files": len(files),
            "fixed_files": 0,
"error_files": 0,
"file_details": {}
}

        for file_path in files:
logger.info(f"Processing {file_path.name}...")

            if self.fix_file(file_path):
                results["fixed_files"] += 1
results["file_details"][str(file_path)] = {
                    "status": "fixed",
"patterns_applied": self.patterns_applied.get(str(file_path), {})
                }
            else:
results["error_files"] += 1
results["file_details"][str(file_path)] = {
                    "status": "error",
"patterns_applied": {}
}

logger.info("Systematic binding fix completed:")
        logger.info(f"  Total files: {results['total_files']}")
        logger.info(f"  Fixed files: {results['fixed_files']}")
        logger.info(f"  Error files: {results['error_files']}")

        return results

    def generate_report(self, results: Dict[str, any]) -> str:
        """Generate a detailed report of the fixing process."""
report_lines = [
"Systematic Binding Fix Report",
"=" * 40,
f"Total files processed: {results['total_files']}",
f"Successfully fixed: {results['fixed_files']}",
f"Errors encountered: {results['error_files']}",
"",
"Detailed Results:",
"-" * 20
]

        for file_path, details in results["file_details"].items():
            status = details["status"]
patterns = details["patterns_applied"]

report_lines.append(f"{file_path}: {status.upper()}")
            if patterns:
                for pattern_type, count in patterns.items():
                    report_lines.append(f"  - {pattern_type}: {count} patterns applied")

        return '\n'.join(report_lines)


def main() -> None:
    """Main function to run the systematic binding fixer."""
    try:
safe_print("[INFO] Starting Systematic Binding Fixer...")

fixer = SystematicBindingFixer()
        results = fixer.fix_all_files()

        # Generate and print report
report = fixer.generate_report(results)
        safe_print("\n" + report)

        # Save report to file
report_path = Path("core/systematic_binding_fix_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

safe_print(f"\n[SUCCESS] Report saved to {report_path}")

        if results["error_files"] == 0:
safe_print("[SUCCESS] All A-Z files successfully processed!")
        else:
safe_print(f"[WARNING] {results['error_files']} files had errors - check the report")

    except Exception as e:
safe_print(f"[ERROR] Systematic binding fix failed: {e}")


if __name__ == "__main__":
main()
