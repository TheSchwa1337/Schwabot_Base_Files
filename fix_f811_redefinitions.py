"""
"""
"""
"""
"""
"""
"""
"""
"""
"""
"""
"""
"""
"""
"""
"""
"""
"""
"""
"""


Fix F811 redefinition of unused imports
This script removes duplicate import statements that are redefined later in the same file.
"""
"""
"""
"""
"""

import re
import os
import glob


def fix_f811_redefinitions():
    """Fix F811 redefinition errors in Python files"""
"""
"""
"""
"""

    print("\\u1f527 Fixing F811 redefinition of unused imports")

# Get all Python files
    python_files = []
    for directory in ["tools", "mathlib", "core", "config", "init"]:
        if os.path.exists(directory):
            python_files.extend(glob.glob(f"{directory}/*.py"))

    total_fixed = 0

    for file_path in python_files:
        print(f"\\n\\u1f4c1 Processing: {file_path}")

# Read the file
        with open(file_path, 'r', encoding='utf - 8') as f:
            content = f.read()

        original_content = content
        file_fixes = 0

# Split into lines and process
        lines = content.split('\n')
        fixed_lines = []
        seen_imports = {}  # Track import name -> line number

        for i, line in enumerate(lines):
            line_num = i + 1

# Check for import statements
            if line.strip().startswith(('from ', 'import ')):
# Extract the import name
                import_name = extract_import_name(line)

                if import_name:
                    if import_name in seen_imports:
# This is a redefinition - skip this line
                        print(f"  \\u1f527 Removed F811 redefinition at line {line_num}: {line.strip()}")
                        file_fixes += 1
                        continue
                    else:
# First time seeing this import
                        seen_imports[import_name] = line_num

            fixed_lines.append(line)

        content = '\n'.join(fixed_lines)

# Check if changes were made
        if content != original_content:
# Backup the original file
            backup_path = f"{file_path}.backup"
            with open(backup_path, 'w', encoding='utf - 8') as f:
                f.write(original_content)

# Write the fixed content
            with open(file_path, 'w', encoding='utf - 8') as f:
                f.write(content)

            print(f"  \\u2705 Fixed {file_fixes} F811 redefinition errors in {file_path}")
            total_fixed += file_fixes
        else:
            print(f"  \\u2139\\ufe0f No F811 errors found in {file_path}")

    print(f"\\n\\u1f389 Total F811 fixes applied: {total_fixed}")
    return total_fixed


def extract_import_name(line):
    """Extract the main import name from an import statement"""
"""
"""
"""
"""
    line = line.strip()

# Handle 'from x import y' statements
    if line.startswith('from '):
# Extract the last part after 'import'
        if ' import ' in line:
            import_part = line.split(' import ')[-1].strip()
# Get the first import name (before comma or space)
            first_import = import_part.split(',')[0].strip()
# Remove any 'as' alias
            if ' as ' in first_import:
                first_import = first_import.split(' as ')[0].strip()
            return first_import

# Handle 'import x' statements
    elif line.startswith('import '):
        import_part = line[7:].strip()  # Remove 'import '
# Get the first import name (before comma or space)
        first_import = import_part.split(',')[0].strip()
# Remove any 'as' alias
        if ' as ' in first_import:
            first_import = first_import.split(' as ')[0].strip()
        return first_import

    return None


if __name__ == "__main__":
    fix_f811_redefinitions()
