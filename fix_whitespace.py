#!/usr/bin/env python3
"""Fix whitespace issues in Python files."""."""


def fix_whitespace(file_path):
    """Fix trailing whitespace and ensure newline at end of file."""."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Remove trailing whitespace from all lines
    lines = content.split('\n')
    fixed_lines = [line.rstrip() for line in lines]

    # Join and ensure newline at end
    fixed_content = '\n'.join(fixed_lines) + '\n'

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(fixed_content)

    print(f"Fixed whitespace in {file_path}")

if __name__ == "__main__":
    files_to_fix = [
        'core/function_patterns.py',
        'core/type_patterns.py',
        'core/strategy_loader.py'
    ]

    for file_path in files_to_fix:
        try:
            fix_whitespace(file_path)
        except Exception as e:
            print(f"Error fixing {file_path}: {e}")
