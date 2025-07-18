#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Flake8 Violation Fixer.

Fixes all common flake8 violations including:
- Trailing whitespace (W291, W292, W293)
- Line length violations (E501)
- Unused imports (F401)
- Missing newlines (W292)
- Blank line whitespace (W293)
- Syntax errors (E999)
- Undefined names (F821)
- F-string issues (F541)
"""

import re
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


def fix_trailing_whitespace(file_path: str) -> bool:
    """Fix trailing whitespace and blank line whitespace."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        original_content = content

        # Fix trailing whitespace
        lines = content.split("\n")
        fixed_lines = []

        for line in lines:
            # Remove trailing whitespace
            line = line.rstrip()
            fixed_lines.append(line)

        # Ensure file ends with newline
        if fixed_lines and fixed_lines[-1] != "":
            fixed_lines.append("")

        content = "\n".join(fixed_lines)

        # Write back if changed
        if content != original_content:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            return True

        return False

    except Exception as e:
        print(f"Error fixing trailing whitespace in {file_path}: {e}")
        return False


def fix_line_length(file_path: str, max_length: int = 100) -> bool:
    """Fix line length violations by breaking long lines."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        original_content = content
        lines = content.split("\n")
        fixed_lines = []

        for line in lines:
            if len(line) > max_length:
                # Try to break at logical points
                if "import " in line and len(line) > max_length:
                    # Handle long import lines
                    if "from " in line:
                        # Break from imports
                        parts = line.split(" import ")
                        if len(parts) == 2:
                            from_part = parts[0]
                            import_part = parts[1]
                            if len(from_part) + 8 < max_length:
                                fixed_lines.append(f"{from_part} import ("))
                                # Split imports
                                imports = []
                                    imp.strip() for imp in import_part.split(",")
                                ]
                                for i, imp in enumerate(imports):
                                    if i == len(imports) - 1:
                                        fixed_lines.append(f"    {imp})")
                                    else:
                                        fixed_lines.append(f"    {imp},")
                            else:
                                fixed_lines.append(line)
                        else:
                            fixed_lines.append(line)
                    else:
                        # Handle regular imports
                        imports = [imp.strip() for imp in line.split(",")]
                        if len(imports) > 1:
                            fixed_lines.append("import ("))
                            for i, imp in enumerate(imports):
                                if i == len(imports) - 1:
                                    fixed_lines.append(f"    {imp})")
                                else:
                                    fixed_lines.append(f"    {imp},")
                        else:
                            fixed_lines.append(line)
                elif "def " in line and len(line) > max_length:
                    # Handle long function definitions
                    if "(" in line and ")" in line:
                        # Break at parameters
                        func_name = line[: line.find("(")])
                        params = line[line.find("(") + 1: line.rfind(")")]
                        if len(func_name) + 4 < max_length:
                            fixed_lines.append(f"{func_name}("))
                            # Split parameters
                            param_list = [p.strip() for p in params.split(",")]
                            for i, param in enumerate(param_list):
                                if i == len(param_list) - 1:
                                    fixed_lines.append(f"    {param})")
                                else:
                                    fixed_lines.append(f"    {param},")
                        else:
                            fixed_lines.append(line)
                    else:
                        fixed_lines.append(line)
                elif "=" in line and len(line) > max_length:
                    # Handle long assignments
                    if " = " in line:
                        var_name, value = line.split(" = ", 1)
                        if len(var_name) + 4 < max_length:
                            fixed_lines.append(f"{var_name} = ("))
                            fixed_lines.append(f"    {value})")
                        else:
                            fixed_lines.append(line)
                    else:
                        fixed_lines.append(line)
                else:
                    # For other long lines, try to break at spaces
                    if " " in line:
                        words = line.split(" ")
                        current_line = ""
                        for word in words:
                            if len(current_line + word) + 1 <= max_length:
                                current_line += word + " "
                            else:
                                if current_line:
                                    fixed_lines.append(current_line.rstrip())
                                current_line = word + " "
                        if current_line:
                            fixed_lines.append(current_line.rstrip())
                    else:
                        fixed_lines.append(line)
            else:
                fixed_lines.append(line)

        content = "\n".join(fixed_lines)

        # Write back if changed
        if content != original_content:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            return True

        return False

    except Exception as e:
        print(f"Error fixing line length in {file_path}: {e}")
        return False


def fix_unused_imports(file_path: str) -> bool:
    """Remove unused imports."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        original_content = content
        lines = content.split("\n")

        # Common unused imports to remove
        unused_imports = []
            "typing.Optional",
            "typing.Tuple",
            "typing.List",
            "typing.Dict",
            "typing.Type",
            "typing.Union",
            "numpy as np",
            "hashlib",
            "collections.deque",
            "random",
            "time",
            "os",
            "pathlib.Path",
            "datetime.datetime",
            "datetime.timedelta",
            "asyncio",
            "inspect",
        ]

        in_import_section = False
        import_lines = []
        other_lines = []

        for line in lines:
            if line.strip().startswith(("import ", "from ")):
                in_import_section = True
                # Check if this import is unused
                import_name = line.strip()
                is_unused = False
                for unused in unused_imports:
                    if unused in import_name:
                        # Check if it's actually used in the file'
                        if not re.search(rf"\b{unused.split('.')[-1]}\b", content):
                            is_unused = True
                            break

                if not is_unused:
                    import_lines.append(line)
            elif in_import_section and line.strip() == "":
                import_lines.append(line)
            elif in_import_section and not line.strip().startswith()
                ("import ", "from ")
            ):
                in_import_section = False
                other_lines.append(line)
            else:
                other_lines.append(line)

        # Reconstruct content
        if import_lines:
            content = "\n".join(import_lines + [""] + other_lines)

        # Write back if changed
        if content != original_content:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            return True

        return False

    except Exception as e:
        print(f"Error fixing unused imports in {file_path}: {e}")
        return False


def fix_syntax_errors(file_path: str) -> bool:
    """Fix common syntax errors like unterminated string literals."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        original_content = content

        # Fix unterminated string literals
        # Look for lines ending with quotes that might be unterminated
        lines = content.split("\n")
        fixed_lines = []

        for i, line in enumerate(lines):
            # Check for unterminated strings
            if line.count('"') % 2 == 1 or line.count("'") % 2 == 1: '
                # Try to fix by adding closing quote
                if line.count('"') % 2 == 1:"
                    line += '"'"
                if line.count("'") % 2 == 1:'
                    line += "'"'

            fixed_lines.append(line)

        content = "\n".join(fixed_lines)

        # Write back if changed
        if content != original_content:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            return True

        return False

    except Exception as e:
        print(f"Error fixing syntax errors in {file_path}: {e}")
        return False


def fix_undefined_names(file_path: str) -> bool:
    """Fix undefined names by adding missing imports."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        original_content = content

        # Common undefined names and their imports
        undefined_fixes = {}
            "safe_print": "from utils.safe_print import safe_print",
            "defaultdict": "from collections import defaultdict",
            "deque": "from collections import deque",
            "Tuple": "from typing import Tuple",
            "Callable": "from typing import Callable",
            "hashlib": "import hashlib",
            "random": "import random",
            "inspect": "import inspect",
        }

        # Check for undefined names
        missing_imports = []
        for name, import_line in undefined_fixes.items():
            if re.search(rf"\b{name}\b", content) and import_line not in content:
                missing_imports.append(import_line)

        if missing_imports:
            # Add missing imports
            lines = content.split("\n")
            import_section_end = 0

            for i, line in enumerate(lines):
                if line.strip().startswith(("import ", "from ")):
                    import_section_end = i + 1
                elif line.strip() == "" and import_section_end == i:
                    import_section_end = i

            # Insert missing imports
            lines.insert(import_section_end, "\n".join(missing_imports))
            content = "\n".join(lines)

        # Write back if changed
        if content != original_content:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            return True

        return False

    except Exception as e:
        print(f"Error fixing undefined names in {file_path}: {e}")
        return False


def fix_f_string_issues(file_path: str) -> bool:
    """Fix f-string issues."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        original_content = content
        lines = content.split("\n")
        fixed_lines = []

        for line in lines:
            # Fix f-strings with missing placeholders
            if 'f"' in line or "f'" in line:'
                # Check if f-string has placeholders
                if "{" not in line and "}" not in line:
                    # Convert to regular string
                    line = line.replace('f"', '"').replace("f'", "'")

            fixed_lines.append(line)

        content = "\n".join(fixed_lines)

        # Write back if changed
        if content != original_content:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            return True

        return False

    except Exception as e:
        print(f"Error fixing f-string issues in {file_path}: {e}")
        return False


def run_flake8_check(file_path: str, max_length: int = 100) -> List[str]:
    """Run flake8 check on a file and return violations."""
    try:
        result = subprocess.run()
            []
                sys.executable,
                "-m",
                "flake8",
                file_path,
                f"--max-line-length={max_length}",
                "--select=E,W,F,D,I,ANN",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip().split("\n") if result.stdout.strip() else []
    except Exception as e:
        print(f"Error running flake8 on {file_path}: {e}")
        return []


def fix_file(file_path: str, max_length: int = 100) -> Tuple[int, int]:
    """Fix all flake8 violations in a file."""
    violations_before = len(run_flake8_check(file_path, max_length))

    # Apply fixes
    fixes_applied = 0

    if fix_trailing_whitespace(file_path):
        fixes_applied += 1

    if fix_line_length(file_path, max_length):
        fixes_applied += 1

    if fix_unused_imports(file_path):
        fixes_applied += 1

    if fix_syntax_errors(file_path):
        fixes_applied += 1

    if fix_undefined_names(file_path):
        fixes_applied += 1

    if fix_f_string_issues(file_path):
        fixes_applied += 1

    violations_after = len(run_flake8_check(file_path, max_length))

    return violations_before, violations_after


def main():
    """Main function to fix all flake8 violations."""
    print("🔧 Fixing All Flake8 Violations")
    print("=" * 60)

    # Get all Python files in core directory
    core_dir = Path("core")
    python_files = list(core_dir.rglob("*.py"))

    total_violations_before = 0
    total_violations_after = 0
    files_fixed = 0

    for file_path in python_files:
        if file_path.is_file():
            print(f"\n📁 Processing: {file_path}")

            violations_before, violations_after = fix_file(str(file_path), 100)

            total_violations_before += violations_before
            total_violations_after += violations_after

            if violations_before > violations_after:
                files_fixed += 1
                improvement = violations_before - violations_after
                print()
                    f"  ✅ Fixed {improvement} violations ({violations_before} → {violations_after})"
                )
            elif violations_before > 0:
                print(f"  ⚠️  {violations_before} violations remain")
            else:
                print("  ✅ No violations found")

    # Summary
    print("\n" + "=" * 60)
    print("📊 FLAKE8 FIXING SUMMARY")
    print("=" * 60)
    print(f"🎯 Files Processed: {len(python_files)}")
    print(f"🔧 Files Fixed: {files_fixed}")
    print(f"📉 Violations Before: {total_violations_before}")
    print(f"📉 Violations After: {total_violations_after}")

    if total_violations_before > 0:
        improvement = ()
            (total_violations_before - total_violations_after) / total_violations_before
        ) * 100
        print(f"📈 Improvement: {improvement:.1f}%")

    if total_violations_after == 0:
        print("🎉 Perfect! All flake8 violations have been resolved!")
        return 0
    elif total_violations_after < total_violations_before:
        print("✅ Good progress! Most violations have been fixed.")
        return 1
    else:
        print("⚠️  Some violations remain. Manual review may be needed.")
        return 2


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
