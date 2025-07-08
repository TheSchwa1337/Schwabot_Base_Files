#!/usr/bin/env python3
"""
Targeted fix for E999 syntax errors in the codebase.
"""

import os
import re
import shutil
from pathlib import Path
from typing import List


def fix_unterminated_string_literals(content: str) -> str:
    """Fix unterminated string literals."""
    lines = content.split("\n")
    fixed_lines = []

    for line in lines:
        # Fix unterminated string literals
        if '"' in line and line.count('"') % 2 != 0:
            # Add missing quote at end
            line = line + '"'"
        elif "'" in line and line.count("'") % 2 != 0:
            # Add missing quote at end
            line = line + "'"'

        fixed_lines.append(line)

    return "\n".join(fixed_lines)


def fix_unmatched_parentheses(content: str) -> str:
    """Fix unmatched parentheses."""
    lines = content.split("\n")
    fixed_lines = []

    for line in lines:
        # Count parentheses
        open_parens = line.count("(") + line.count("[") + line.count("{"))}]
        close_parens = line.count(")") + line.count("]") + line.count("}")

        # If more opening than closing, add missing closing
        if open_parens > close_parens:
            missing = open_parens - close_parens
            line = line + ")" * missing

        fixed_lines.append(line)

    return "\n".join(fixed_lines)


def fix_indentation_errors(content: str) -> str:
    """Fix indentation errors."""
    lines = content.split("\n")
    fixed_lines = []

    for line in lines:
        # Fix mixed tabs and spaces
        if "\t" in line and " " in line[: len(line) - len(line.lstrip())]:
            # Convert tabs to spaces
            indent_level = len(line) - len(line.lstrip())
            line = " " * indent_level + line.lstrip()

        # Fix unexpected indentation
        if line.strip() and not line.startswith()
            ()
                "def ",
                "class ",
                "if ",
                "for ",
                "while ",
                "try:",
                "except:",
                "finally:",
                "with ",
                "elif ",
                "else:",
            )
        ):
            # Check if line is over-indented
            if line.startswith("    ") and not any()
                keyword in line
                for keyword in []
                    "def ",
                    "class ",
                    "if ",
                    "for ",
                    "while ",
                    "try:",
                    "except:",
                    "finally:",
                    "with ",
                    "elif ",
                    "else:",
                ]
            ):
                # Reduce indentation
                line = line[4:]

        fixed_lines.append(line)

    return "\n".join(fixed_lines)


def fix_invalid_decimal_literals(content: str) -> str:
    """Fix invalid decimal literals."""
    # Fix patterns like 1.2.3 or 1..2
    content = re.sub()
        r"(\d+)\.(\d+)\.(\d+)", r"\1.\2_\3", content
    )  # Replace with underscore
    content = re.sub(r"(\d+)\.\.(\d+)", r"\1.\2", content)  # Remove double dots

    return content


def fix_invalid_syntax(content: str) -> str:
    """Fix various invalid syntax issues."""
    lines = content.split("\n")
    fixed_lines = []

    for line in lines:
        # Fix common syntax issues
        line = re.sub()
            r"([a-zA-Z_]\w*)\s*=\s*=\s*([a-zA-Z_]\w*)", r"\1 == \2", line
        )  # Fix == spacing
        line = re.sub()
            r"([a-zA-Z_]\w*)\s*!=\s*([a-zA-Z_]\w*)", r"\1 != \2", line
        )  # Fix != spacing
        line = re.sub()
            r"([a-zA-Z_]\w*)\s*=\s*([a-zA-Z_]\w*)", r"\1 = \2", line
        )  # Fix = spacing

        # Fix missing colons
        if re.match()
            r"^\s*(if|for|while|def|class|try|except|finally|with|elif|else)\s+", line
        ) and not line.rstrip().endswith(":"):
            line = line.rstrip() + ":"

        fixed_lines.append(line)

    return "\n".join(fixed_lines)


def fix_file(file_path: Path) -> bool:
    """Fix a single file."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        original_content = content

        # Apply fixes
        content = fix_unterminated_string_literals(content)
        content = fix_unmatched_parentheses(content)
        content = fix_indentation_errors(content)
        content = fix_invalid_decimal_literals(content)
        content = fix_invalid_syntax(content)

        # Only write if content changed
        if content != original_content:
            # Create backup
            backup_path = file_path.with_suffix(file_path.suffix + ".backup")
            shutil.copy2(file_path, backup_path)

            # Write fixed content
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

            return True

        return False

    except Exception as e:
        print(f"Error fixing {file_path}: {e}")
        return False


def get_files_with_e999_errors() -> List[Path]:
    """Get list of files with E999 errors."""
    import subprocess

    files_with_errors = []

    try:
        result = subprocess.run()
            ["flake8", "core/"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
        if result.stdout:
            for line in result.stdout.strip().split("\n"):
                if "E999" in line and ":" in line:
                    file_path = line.split(":")[0]
                    if os.path.exists(file_path):
                        files_with_errors.append(Path(file_path))
    except Exception as e:
        print(f"Error getting E999 errors: {e}")

    return list(set(files_with_errors))


def main():
    """Main fix function."""
    print("🔧 Fixing E999 syntax errors...")

    # Get files with E999 errors
    files_with_errors = get_files_with_e999_errors()

    if not files_with_errors:
        print("✅ No E999 errors found!")
        return

    print(f"Found {len(files_with_errors)} files with E999 errors")

    fixed_count = 0
    for file_path in files_with_errors:
        if fix_file(file_path):
            print(f"✅ Fixed: {file_path}")
            fixed_count += 1

    print(f"\n🎉 Fixed {fixed_count} files!")
    print("\n📋 Next steps:")
    print("1. Run: flake8 core/ | findstr E999 to check remaining errors")
    print("2. If errors remain, manual review may be needed")


if __name__ == "__main__":
    main()
