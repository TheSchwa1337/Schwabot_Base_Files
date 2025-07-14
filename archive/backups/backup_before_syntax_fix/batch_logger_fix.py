import glob
import os
import re
from pathlib import Path

from dual_unicore_handler import DualUnicoreHandler

# Initialize Unicode handler
unicore = DualUnicoreHandler()

# -*- coding: utf - 8 -*-
""""""
""""""
""""""
""""""
"""
Batch Logger Import Fix Script

This script automatically adds logger imports to Python files that use logger
but don't have the proper import statements."""'"
""""""
""""""
""""""
""""""
"""


def add_logger_imports(file_path):"""
    """Add logger imports to files that use logger but don't import it."""'

"""
""""""
""""""
""""""
"""
    try:
        with open(file_path, 'r', encoding='utf - 8') as f:
            content = f.read()

# Check if logger is used but not properly imported
    if 'logger.' in content:
            lines = content.split('\n')
            has_logging_import = 'import logging' in content
            has_logger_definition = 'logger = logging.getLogger(__name__)' in content

if not has_logging_import:"""
print(f"Adding logging import to: {file_path}")
# Add import logging at the top
    for i, line in enumerate(lines):
                    if line.strip().startswith('import ') or line.strip().startswith('from '):
                        lines.insert(i, 'import logging')
                        lines.insert(i + 1, '')
                        break
    else:
                    lines.insert(0, 'import logging')
                    lines.insert(1, '')

if not has_logger_definition:
                print(f"Adding logger definition to: {file_path}")
# Add logger definition after import logging
    for i, line in enumerate(lines):
                    if line.strip() == 'import logging':
                        lines.insert(i + 1, 'logger = logging.getLogger(__name__)')
                        break
    else:
# If no import logging found, add both
                    if not has_logging_import:
                        lines.insert(0, 'import logging')
                        lines.insert(1, 'logger = logging.getLogger(__name__)')
                        lines.insert(2, '')

# Write back to file
with open(file_path, 'w', encoding='utf - 8') as f:
                f.write('\n'.join(lines))

return True
    else:
            return False

except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False


def find_files_with_logger_usage():
    """Find all Python files that use logger but don't have proper imports."""'

"""
""""""
""""""
""""""
"""
core_dir = Path('core')
    python_files = list(core_dir.rglob('*.py'))

files_to_fix = []

for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf - 8') as f:
                content = f.read()

# Check if logger is used
    if 'logger.' in content:
                has_logging_import = 'import logging' in content
                has_logger_definition = 'logger = logging.getLogger(__name__)' in content

# Add to fix list if missing either import or definition
    if not has_logging_import or not has_logger_definition:
                    files_to_fix.append(str(file_path))

except Exception as e:"""
print(f"Error reading {file_path}: {e}")

return files_to_fix


def main():
    """Main function to run the batch logger fix."""

"""
""""""
""""""
""""""
""""""
print("\\u1f527 Batch Logger Import Fix Script")
    print("=" * 50)

# Find files that need fixing
files_to_fix = find_files_with_logger_usage()

print(f"Found {len(files_to_fix)} files that need logger imports:")
    for file_path in files_to_fix:
        print(f"  - {file_path}")

if not files_to_fix:
        print("\\u2705 No files need logger import fixes!")
        return

# Fix each file
fixed_count = 0
    for file_path in files_to_fix:
        if add_logger_imports(file_path):
            fixed_count += 1

print(f"\\n\\u2705 Successfully fixed {fixed_count} out of {len(files_to_fix)} files")

# Run flake8 to check results
print("\\n\\u1f50d Running flake8 check...")
    os.system("flake8 core/ --select = F821 --count")

if __name__ == "__main__":
    main()
""""""
""""""
""""""
""""""
"""
"""
