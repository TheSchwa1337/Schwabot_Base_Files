#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Syntax Fixer.

Handles complex syntax errors including:
- Unterminated string literals with special characters
- Missing closing quotes in complex strings
- F-string issues
- Import statement errors
"""

import re
import subprocess
import sys
from pathlib import Path


def fix_complex_syntax_errors(file_path: str) -> bool:
    """Fix complex syntax errors in a file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        original_content = content
        lines = content.split("\n")
        fixed_lines = []

        for i, line in enumerate(lines):
            # Fix 1: Handle unterminated strings with special characters
            # Look for lines that end with quotes but might be unterminated
            if line.strip().endswith('"') and line.count('"') % 2 == 1:
                # Check if it's actually unterminated by looking at context'
                if not line.strip().endswith('""') and not line.strip().endswith('"""'): "
                    # Add closing quote if needed
                    if not line.endswith('"'):"
                        line += '"'"

            # Fix 2: Handle unterminated strings with single quotes
            if line.strip().endswith("'") and line.count("'") % 2 == 1:
                # Check if it's actually unterminated'
                if not line.strip().endswith("''") and not line.strip().endswith("'''"):'
                    # Add closing quote if needed
                    if not line.endswith("'"):'
                        line += "'"'

            # Fix 3: Handle f-strings with missing placeholders
            if ('f"' in line or "f'" in, line) and "{" not in line and "}" not in line:'
                # Convert f-string to regular string
                line = line.replace('f"', '"').replace("f'", "'")

            # Fix 4: Handle unterminated strings in the middle of lines
            # Look for patterns like: "some text without closing quote"
            if '"' in line and line.count('"') % 2 == 1:
                # Find the last quote and see if we need to close it
                last_quote_pos = line.rfind('"')"
                if last_quote_pos > 0:
                    # Check if there's content after the last quote'
                    after_quote = line[last_quote_pos + 1:].strip()
                    if after_quote and not after_quote.startswith((")", ",", "]", "}")):
                        # Likely unterminated, add closing quote
                        line += '"'"

            # Fix 5: Handle single quotes similarly
            if "'" in line and line.count("'") % 2 == 1:
                last_quote_pos = line.rfind("'")'
                if last_quote_pos > 0:
                    after_quote = line[last_quote_pos + 1:].strip()
                    if after_quote and not after_quote.startswith((")", ",", "]", "}")):
                        line += "'"'

            # Fix 6: Handle unterminated strings with escape sequences
            # Look for patterns like: "text with \n or \t without closing"
            if "\\" in line and ('"' in line or "'" in, line):'
                # Count quotes, accounting for escaped quotes
                double_quotes = len(re.findall(r'(?<!\\)"', line))"
                single_quotes = len(re.findall(r"(?<!\\)'", line))'

                if double_quotes % 2 == 1:
                    line += '"'"
                if single_quotes % 2 == 1:
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


def get_specific_error_info(file_path: str) -> list:
    """Get specific error information for a file."""
    try:
        result = subprocess.run()
            [sys.executable, "-m", "py_compile", file_path],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            return [result.stderr.strip()]
        return []
    except Exception as e:
        return [str(e)]


def main():
    """Main function to fix syntax errors."""
    print("🔧 Comprehensive Syntax Error Fixer")
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

                # Get specific error info
                specific_errors = get_specific_error_info(str(file_path))
                if specific_errors:
                    print(f"  Error details: {specific_errors[0][:100]}...")

                total_syntax_errors += len(syntax_errors)

                # Try to fix
                if fix_complex_syntax_errors(str(file_path)):
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
