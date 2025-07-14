#!/usr/bin/env python3
"""
Fix critical Flake8 issues that can't be ignored.

This script fixes:
- E999: Syntax errors
- F821: Undefined names
- Missing newlines at end of files
- Critical indentation issues
"""

import os
import re
from pathlib import Path


def fix_syntax_errors():
    """Fix known syntax errors in the codebase."""

    # Fix swing_pattern_recognition.py - tuple unpacking issue
    swing_file = "core/swing_pattern_recognition.py"
    if os.path.exists(swing_file):
        with open(swing_file, 'r') as f:
            content = f.read()

        # Fix the tuple unpacking issue
        content = content.replace("prev, curr, nxt = ()", "prev, curr, nxt = (")

        with open(swing_file, 'w') as f:
            f.write(content)
        print(f"Fixed syntax error in {swing_file}")


def fix_undefined_names():
    """Fix undefined name issues."""

    # Fix unified_math_system.py - add missing Optional import
    math_file = "core/unified_math_system.py"
    if os.path.exists(math_file):
        with open(math_file, 'r') as f:
            content = f.read()

        # Add Optional to typing import if not present
        if "from typing import" in content and "Optional" not in content:
            content = content.replace("from typing import Any, Dict", "from typing import Any, Dict, Optional")

        with open(math_file, 'w') as f:
            f.write(content)
        print(f"Fixed undefined names in {math_file}")


def fix_missing_newlines():
    """Add missing newlines at end of files."""

    files_to_fix = [
        "core/ccxt_trading_executor.py",
        "core/price_event.py",
        "core/price_event_registry.py",
        "core/price_precision_utils.py",
        "core/precision_service.py",
        "core/math/__init__.py",
        "core/math/tensor_algebra/__init__.py",
        "core/math/trading_tensor_ops.py",
        "schwabot/ai_oracles/profit_oracle.py",
        "schwabot/price_feed_integration.py",
        "schwabot/test_trade_entry_exit.py",
    ]

    for file_path in files_to_fix:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                content = f.read()

            # Add newline if missing
            if not content.endswith('\n'):
                content += '\n'
                with open(file_path, 'w') as f:
                    f.write(content)
                print(f"Added newline to {file_path}")


def fix_inline_comments():
    """Fix inline comment spacing issues."""

    files_to_fix = [
        "core/integrated_correction_system.py",
        "core/math/tensor_algebra.py",
        "core/math/trading_tensor_ops.py",
        "schwabot/ai_oracles/profit_oracle.py",
        "schwabot/core/adaptive_trainer.py",
    ]

    for file_path in files_to_fix:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                content = f.read()

            lines = content.split('\n')
            fixed_lines = []

            for line in lines:
                # Fix inline comments that don't have proper spacing
                if '#' in line and not line.strip().startswith('#'):
                    code_part, comment_part = line.split('#', 1)
                    if code_part.strip() and comment_part.strip():
                        # Ensure at least 2 spaces before comment
                        if not code_part.endswith('  '):
                            code_part = code_part.rstrip() + '  '
                        line = code_part + '#' + comment_part
                fixed_lines.append(line)

            fixed_content = '\n'.join(fixed_lines)
            if fixed_content != content:
                with open(file_path, 'w') as f:
                    f.write(fixed_content)
                print(f"Fixed inline comments in {file_path}")


def fix_function_spacing():
    """Fix spacing after function/class definitions."""

    files_to_fix = [
        "core/__init__.py",
        "core/ccxt_trading_executor.py",
        "core/trade_executor.py",
        "schwabot/price_feed_integration.py",
        "schwabot/test_trade_entry_exit.py",
    ]

    for file_path in files_to_fix:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                content = f.read()

            lines = content.split('\n')
            fixed_lines = []
            i = 0

            while i < len(lines):
                line = lines[i]
                fixed_lines.append(line)

                # Check if this is a function or class definition
                if (
                    re.match(r'^\s*(def|class)\s+\w+.*:$', line)
                    and i + 1 < len(lines)
                    and lines[i + 1].strip() != ''
                    and not lines[i + 1].startswith('    ')
                    and not lines[i + 1].startswith('\t')
                ):
                    # Add two blank lines after function/class definition
                    fixed_lines.append('')
                    fixed_lines.append('')

                i += 1

            fixed_content = '\n'.join(fixed_lines)
            if fixed_content != content:
                with open(file_path, 'w') as f:
                    f.write(fixed_content)
                print(f"Fixed function spacing in {file_path}")


def main():
    """Run all fixes."""
    print("Fixing critical Flake8 issues...")

    fix_syntax_errors()
    fix_undefined_names()
    fix_missing_newlines()
    fix_inline_comments()
    fix_function_spacing()

    print("Critical fixes completed!")


if __name__ == "__main__":
    main()
