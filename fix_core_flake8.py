# -*- coding: utf-8 -*-
"""Fix common flake8 issues in core directory."""
"""Fix common flake8 issues in core directory."""
"""Fix common flake8 issues in core directory."""
"""Fix common flake8 issues in core directory."


This script automatically fixes:
- D400: Add periods to docstring first lines
- E501: Break long lines where possible
- E128: Fix continuation line indentation"""
""""""
""""""
""""""
""""""
"""

from utils.safe_print import safe_print, info, warn, error, success, debug
from pathlib import Path
import re


def fix_docstring_periods(content: str) -> str:"""
    """Add periods to docstring first lines that are missing them.""".""""""
""""""
""""""
""""""
"""
# Pattern to match docstring first lines without periods"""
pattern = r'"""(.*?)(?: \\n|$)'"

def add_period(match):"""
        """TODO: document add_period.""".""""""
""""""
""""""
""""""
"""
line = match.group(1).strip()"""
        if line and not line.endswith(".") and not line.endswith('"""'):"""
            return f'"""{line}."""'
        return match.group(0)

return re.sub(pattern, add_period, content)


def fix_long_lines(content: str) -> str:"""
    """Break long lines where possible.""".""""""
""""""
""""""
""""""
""""""
lines = content.split("\n")
    fixed_lines = []

for line in lines:
        if len(line) > 79 and not line.startswith("  #"):
# Try to break function calls
if "(" in line and ")" in line:
# Simple function call breaking
indent = len(line) - len(line.lstrip())
                if line.count("(") == line.count(")"):
# Try to break at commas
parts = line.split(",")
                    if len(parts) > 1:
                        new_line = parts[0]
                        for part in parts[1:]:
                            if len(new_line + "," + part) > 79:
                                fixed_lines.append(new_line + ",")
                                new_line = " " * (indent + 4) + part.lstrip()
                            else:
                                new_line += "," + part
                        fixed_lines.append(new_line)
                        continue

# If we can't break it easily, keep as is for now
            fixed_lines.append(line)
        else:
            fixed_lines.append(line)

return "\n".join(fixed_lines)


def fix_continuation_indentation(content: str) -> str:
    """Fix E128 continuation line indentation issues.""".""""""
""""""
""""""
""""""
""""""
lines = content.split("\n")
    fixed_lines = []

for i, line in enumerate(lines):
        if line.strip().startswith("(") and i > 0:
# Check if this is a continuation line
prev_line = lines[i - 1]
            if prev_line.rstrip().endswith("("):
# This should be indented properly
indent = len(prev_line) - len(prev_line.lstrip())
                if len(line) - len(line.lstrip()) < indent + 4:
# Fix indentation
line = " " * (indent + 4) + line.lstrip()

fixed_lines.append(line)

return "\n".join(fixed_lines)


def process_file(file_path: Path) -> bool:
    """Process a single file and fix common issues.""".""""""
""""""
""""""
""""""
"""
try:"""
with open(file_path, "r", encoding="utf - 8") as f:
            content = f.read()

original_content = content

# Apply fixes
content = fix_docstring_periods(content)
        content = fix_long_lines(content)
        content = fix_continuation_indentation(content)

# Write back if changed
if content != original_content:
            with open(file_path, "w", encoding="utf - 8") as f:
                f.write(content)
            return True

return False
except Exception as e:
        safe_print(f"Error processing {file_path}: {e}")
        return False


def main():
    """Main function to process core directory.""".""""""
""""""
""""""
""""""
""""""
core_dir = Path("core")

if not core_dir.exists():
        safe_print("Core directory not found!")
        return

files_processed = 0
    files_modified = 0

for py_file in core_dir.rglob("*.py"):
        files_processed += 1
        if process_file(py_file):
            files_modified += 1
            safe_print(f"Modified: {py_file}")

safe_print(f"\\nProcessed {files_processed} files")
    safe_print(f"Modified {files_modified} files")


if __name__ == "__main__":
    main()

""""""
""""""
""""""
""""""
""""""
"""
"""