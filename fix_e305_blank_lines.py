# -*- coding: utf-8 -*-
""""""
""""""
""""""
""""""
""""""
""""""
""""""
""""""
""""""
""""""
""""""
""""""
""""""
""""""
""""""
""""""
""""""
""""""
""""""
"""


Script to fix all E305 errors(expected 2 blank lines after class or function definition) in the core directory."""
""""""
""""""
""""""
""""""
"""

import os
from pathlib import Path
import re


def fix_e305_in_file(filepath):
    with open(filepath, 'r', encoding='utf - 8') as f:
        lines = f.readlines()

new_lines = []
    i = 0
    changed = False
    while i < len(lines):
        line = lines[i]
        new_lines.append(line)
# Check for function or class definition
if re.match(r'^(def |class )', line.strip()):
# Count following blank lines
j = i + 1
            blank_count = 0
            while j < len(lines) and lines[j].strip() == '':
                blank_count += 1
                j += 1
# If only one blank line, insert another
            if blank_count == 1:
                new_lines.append('\n')
                changed = True
# If no blank lines, insert two
            elif blank_count == 0:
                new_lines.append('\n')
                new_lines.append('\n')
                changed = True
        i += 1
    if changed:
        with open(filepath, 'w', encoding='utf - 8') as f:
            f.writelines(new_lines)"""
        print(f"Fixed: {filepath}")
    return changed


def main():
    core_path = Path('core')
    py_files = list(core_path.rglob('*.py'))
    total_fixed = 0
    for py_file in py_files:
        if fix_e305_in_file(str(py_file)):
            total_fixed += 1
    print(f"\\nE305 blank line fixes applied to {total_fixed} files.")


if __name__ == "__main__":
    main()
