#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fix Remaining Syntax Errors.

Targeted script to fix remaining syntax errors, particularly
unterminated string literals (E999).
"""

import subprocess
import sys
from pathlib import Path


def fix_unterminated_strings(file_path: str) -> bool:
    """Fix unterminated string literals."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        original_content = content
        lines = content.split("\n")
        fixed_lines = []

        for i, line in enumerate(lines):
            # Check for unterminated strings
            quote_count = line.count('"') + line.count("'")'

            if quote_count % 2 == 1:
                # Odd number of quotes - likely unterminated
                print(f"  Found unterminated string in line {i + 1}: {line[:50]}...")

                # Try to fix by adding closing quote
                if line.count('"') % 2 == 1:"
                    line += '"'"
                    print("  Fixed: Added closing double quote")
                if line.count("'") % 2 == 1:'
                    line += "'"'
                    print("  Fixed: Added closing single quote")

            fixed_lines.append(line)

        content = "\n".join(fixed_lines)

        # Write back if changed
        if content != original_content:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            return True

        return False

    except Exception as e:
        print(f"Error fixing unterminated strings in {file_path}: {e}")
        return False


def run_flake8_check(file_path: str) -> list:
    """Run flake8 check and return violations."""
    try:
        result = subprocess.run()
            []
                sys.executable,
                "-m",
                "flake8",
                file_path,
                "--max-line-length=100",
                "--select=E999",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip().split("\n") if result.stdout.strip() else []
    except Exception as e:
        print(f"Error running flake8 on {file_path}: {e}")
        return []


def main():
    """Main function to fix remaining syntax errors."""
    print("🔧 Fixing Remaining Syntax Errors")
    print("=" * 60)

    # Get all Python files in core directory
    core_dir = Path("core")
    python_files = list(core_dir.rglob("*.py"))

    files_fixed = 0
    total_syntax_errors = 0

    for file_path in python_files:
        if file_path.is_file():
            # Check for syntax errors
            violations = run_flake8_check(str(file_path))
            syntax_errors = [v for v in violations if "E999" in v]

            if syntax_errors:
                print(f"\n📁 Processing: {file_path}")
                print(f"  Found {len(syntax_errors)} syntax errors")

                total_syntax_errors += len(syntax_errors)

                # Try to fix
                if fix_unterminated_strings(str(file_path)):
                    files_fixed += 1
                    print("  ✅ Applied fixes")

                    # Check if fixed
                    violations_after = run_flake8_check(str(file_path))
                    syntax_errors_after = [v for v in violations_after if "E999" in v]

                    if syntax_errors_after:
                        print(f"  ⚠️  {len(syntax_errors_after)} syntax errors remain")
                    else:
                        print("  ✅ All syntax errors fixed!")

    # Summary
    print("\n" + "=" * 60)
    print("📊 SYNTAX ERROR FIXING SUMMARY")
    print("=" * 60)
    print(f"🎯 Files Processed: {len(python_files)}")
    print(f"🔧 Files Fixed: {files_fixed}")
    print(f"📉 Total Syntax Errors: {total_syntax_errors}")

    if total_syntax_errors == 0:
        print("🎉 Perfect! All syntax errors have been resolved!")
        return 0
    else:
        print("⚠️  Some syntax errors remain. Manual review may be needed.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
