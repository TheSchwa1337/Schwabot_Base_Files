#!/usr/bin/env python3
"""
Automated fix script for common stub patterns and encoding issues.
"""

import re
import shutil
from pathlib import Path


def fix_unicode_issues(content: str) -> str:
    """Fix Unicode/encoding issues in content."""
    # Replace problematic Unicode characters with ASCII equivalents
    replacements = {}
        "…": "...",
        '"': '"',
        """: "'",'
        """: "'", '
        "–": "-",
        "—": "-",
        "×": "*",
        "÷": "/",
        "±": "+/-",
        "≤": "<=",
        "≥": ">=",
        "≠": "!=",
        "≈": "~=",
    }
    for old, new in replacements.items():
        content = content.replace(old, new)

    return content


def fix_empty_pass_statements(content: str) -> str:
    """Replace empty pass statements with proper stubs."""
    lines = content.split("\n")
    fixed_lines = []

    i = 0
    while i < len(lines):
        line = lines[i]

        # Check for function definition followed by pass
        if re.match(r"^\s*def \w+\([^)]*\):\s*$", line):
            if i + 1 < len(lines) and re.match(r"^\s*pass\s*$", lines[i + 1]):
                # Replace with proper stub
                function_name = re.search(r"def (\w+)", line).group(1)
                fixed_lines.append(line)
                fixed_lines.append(f'    """{function_name} implementation pending."""')
                fixed_lines.append("    # TODO: Implement this function")
                fixed_lines.append()
                    '    raise NotImplementedError("Function not yet implemented")'
                )
                i += 2  # Skip the pass line
                continue

        fixed_lines.append(line)
        i += 1

    return "\n".join(fixed_lines)


def fix_missing_imports(content: str) -> str:
    """Add missing imports based on usage."""
    imports_needed = []

    if "numpy" in content and "import numpy" not in content:
        imports_needed.append("import numpy as np")
    if "scipy" in content and "import scipy" not in content:
        imports_needed.append("import scipy as sp")
    if "pandas" in content and "import pandas" not in content:
        imports_needed.append("import pandas as pd")
    if "matplotlib" in content and "import matplotlib" not in content:
        imports_needed.append("import matplotlib.pyplot as plt")

    if imports_needed:
        # Find the right place to insert imports
        lines = content.split("\n")
        insert_pos = 0

        # Find first non-import line
        for i, line in enumerate(lines):
            if line.strip() and not line.startswith(("import ", "from ")):
                insert_pos = i
                break

        # Insert imports
        lines.insert(insert_pos, "\n".join(imports_needed))
        content = "\n".join(lines)

    return content


def fix_file(file_path: Path) -> bool:
    """Fix a single file."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        original_content = content

        # Apply fixes
        content = fix_unicode_issues(content)
        content = fix_empty_pass_statements(content)
        content = fix_missing_imports(content)

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


def main():
    """Main fix function."""
    core_dir = Path("core")
    fixed_count = 0

    print("🔧 Starting automated fixes...")

    for py_file in core_dir.rglob("*.py"):
        if fix_file(py_file):
            print(f"✅ Fixed: {py_file}")
            fixed_count += 1

    print(f"\n🎉 Fixed {fixed_count} files!")
    print("\n📋 Next steps:")
    print("1. Run: flake8 core/ to check remaining errors")
    print("2. Run: autopep8 --in-place --aggressive core/ for formatting")
    print("3. Implement remaining logic manually")


if __name__ == "__main__":
    main()
