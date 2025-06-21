#!/usr/bin/env python3
"""Debug whitespace fixer script to identify and fix trailing whitespace and missing newlines.


"""

import glob
import os


def fix_whitespace_in_file(filepath):
    """Fix whitespace issues in a single file with detailed debugging."""
    print(f"\n=== Processing: {filepath} ===")

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        print(f"Original file size: {len(content)} characters")
        print(f"Original content ends with: {repr(content[-20:])}")

        lines = content.splitlines()
        print(f"Number of lines: {len(lines)}")

        # Check for trailing whitespace
        fixed_lines = []
        whitespace_found = False

        for i, line in enumerate(lines):
            original_line = line
            stripped_line = line.rstrip()

            if original_line != stripped_line:
                print(f"  Line {i + 1}: Found trailing whitespace - {repr(original_line)}")
                whitespace_found = True
                fixed_lines.append(stripped_line)
            else:
                fixed_lines.append(line)

        if not whitespace_found:
            print("  No trailing whitespace found")

        # Ensure file ends with exactly one newline
        new_content = '\n'.join(fixed_lines)
        if not new_content.endswith('\n'):
            print("  Adding final newline")
            new_content += '\n'
        else:
            print("  File already ends with newline")

        print(f"New content size: {len(new_content)} characters")
        print(f"New content ends with: {repr(new_content[-20:])}")

        # Only write if content changed
        if content != new_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print("  ✓ File updated")
            return True
        else:
            print("  ✓ No changes needed")
            return False

    except Exception as e:
        print(f"  ✗ Error processing file: {e}")
        return False

def main():
    """Main function to fix whitespace in all Python files."""
    # Get all Python files
    python_files = glob.glob('**/*.py', recursive=True)

    print(f"Found {len(python_files)} Python files")
    print("Files found:")
    for f in python_files:
        print(f"  {f}")

    fixed_count = 0
    error_count = 0

    for filepath in python_files:
        if fix_whitespace_in_file(filepath):
            fixed_count += 1

    print(f"\n=== Summary ===")
    print(f"Files processed: {len(python_files)}")
    print(f"Files fixed: {fixed_count}")
    print(f"Files with errors: {error_count}")

if __name__ == "__main__":
    print("[DEBUG] Script started.")
    print(f"[DEBUG] Current working directory: {os.getcwd()}")
    main()
