#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Syntax Error Fixer.

Systematically fixes all syntax errors across the codebase:
- Unterminated string literals
- Missing colons after function parameters
- Indentation errors
- Import statement errors
- F-string issues
"""

import re
import subprocess
import sys
from pathlib import Path
from typing import List


def fix_syntax_errors_in_file(file_path: str) -> bool:
    """Fix syntax errors in a specific file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        original_content = content
        lines = content.split("\n")
        fixed_lines = []

        for i, line in enumerate(lines):
            # Fix 1: Handle unterminated strings
            if '"' in line and line.count('"') % 2 == 1:
                # Check if it's actually unterminated'
                if not line.strip().endswith('""') and not line.strip().endswith('"""'): "
                    if not line.endswith('"'):"
                        line += '"'"

            if "'" in line and line.count("'") % 2 == 1:
                if not line.strip().endswith("''") and not line.strip().endswith("'''"):'
                    if not line.endswith("'"):'
                        line += "'"'

            # Fix 2: Handle missing colons after function parameters
            # Look for patterns like: parameter_name: type)
            if re.search(r":\s*\w+\s*\)", line):
                # Add missing colon
                line = re.sub(r":\s*(\w+)\s*\)", r": \1):", line)

            # Fix 3: Handle f-strings with missing placeholders
            if ('f"' in line or "f'" in, line) and "{" not in line and "}" not in line:'
                line = line.replace('f"', '"').replace("f'", "'")

            # Fix 4: Handle unterminated strings with escape sequences
            if "\\" in line and ('"' in line or "'" in, line):'
                double_quotes = len(re.findall(r'(?<!\\)"', line))"
                single_quotes = len(re.findall(r"(?<!\\)'", line))'

                if double_quotes % 2 == 1:
                    line += '"'"
                if single_quotes % 2 == 1:
                    line += "'"'

            # Fix 5: Handle malformed docstrings
            if '""""' in line:
                line = line.replace('""""', '"""')"

            # Fix 6: Handle unterminated strings in the middle of lines
            if '"' in line and line.count('"') % 2 == 1:
                last_quote_pos = line.rfind('"')"
                if last_quote_pos > 0:
                    after_quote = line[last_quote_pos + 1:].strip()
                    if after_quote and not after_quote.startswith((")", ",", "]", "}")):
                        line += '"'"

            if "'" in line and line.count("'") % 2 == 1:
                last_quote_pos = line.rfind("'")'
                if last_quote_pos > 0:
                    after_quote = line[last_quote_pos + 1:].strip()
                    if after_quote and not after_quote.startswith((")", ",", "]", "}")):
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


def check_syntax_error(file_path: str) -> List[str]:
    """Check for syntax errors in a file."""
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


def run_flake8_syntax_check(file_path: str) -> List[str]:
    """Run flake8 syntax check on a file."""
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
    """Main function to fix all syntax errors."""
    print("🔧 Comprehensive Syntax Error Fixer")
    print("=" * 60)

    # Get all Python files in core directory
    core_dir = Path("core")
    python_files = list(core_dir.rglob("*.py"))

    files_fixed = 0
    total_syntax_errors = 0
    files_with_errors = 0

    for file_path in python_files:
        if file_path.is_file():
            # Check for syntax errors
            syntax_errors = check_syntax_error(str(file_path))
            flake8_errors = run_flake8_syntax_check(str(file_path))

            if syntax_errors or flake8_errors:
                print(f"\n📁 Processing: {file_path}")
                files_with_errors += 1

                if syntax_errors:
                    print(f"  Found syntax errors: {syntax_errors[0][:100]}...")
                    total_syntax_errors += len(syntax_errors)

                if flake8_errors:
                    print(f"  Found {len(flake8_errors)} flake8 syntax errors")
                    total_syntax_errors += len(flake8_errors)

                # Try to fix
                if fix_syntax_errors_in_file(str(file_path)):
                    files_fixed += 1
                    print("  ✅ Applied fixes")

                    # Check if fixed
                    syntax_errors_after = check_syntax_error(str(file_path))
                    flake8_errors_after = run_flake8_syntax_check(str(file_path))

                    if syntax_errors_after or flake8_errors_after:
                        print("  ⚠️  Some syntax errors remain")
                    else:
                        print("  ✅ All syntax errors fixed!")

    # Summary
    print("\n" + "=" * 60)
    print("📊 SYNTAX ERROR FIXING SUMMARY")
    print("=" * 60)
    print(f"🎯 Files Processed: {len(python_files)}")
    print(f"📁 Files with Errors: {files_with_errors}")
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
