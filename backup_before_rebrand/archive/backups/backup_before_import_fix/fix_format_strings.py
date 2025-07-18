#!/usr/bin/env python3
"""
Script to fix invalid decimal literal format strings.
Converts format strings like .format(var:.4f) to .format(var):.4f
"""

import glob
import os
import re


def fix_format_strings(content):
    """Fix invalid decimal literal format strings in content."""
    # Pattern to match .format() calls with invalid decimal literals
    # Matches patterns like .format(variable:.4f) or .format(var, other:.2f)
    pattern = r'\.format\(([^)]*?)([^:]+):([^)]*\.[0-9]+f[^)]*)\)'

    def replacement(match):
        full_args = match.group(1)
        var_with_format = match.group(2)
        format_spec = match.group(3)

        # Remove the format spec from the variable and put it in the string
        var_clean = var_with_format.strip()

        # Rebuild the format call
        if full_args.strip():
            if full_args.strip().endswith(','):
                new_args = full_args + var_clean
            else:
                new_args = full_args + ', ' + var_clean
        else:
            new_args = var_clean

        return f'.format({new_args})'

    # Apply the fix
    fixed_content = re.sub(pattern, replacement, content)

    # Also fix specific patterns we found
    # Pattern: variable:.4f) -> variable) with format in string
    patterns_to_fix = []
        (r'(["\'])([^"\']*\{[0-9]+\})([^"\']*)\1\.format\(([^)]*?)([^:,\s]+):(\.[0-9]+f)\)', '
         r'\1\2:\6\3\1.format(\4\5)'),
        (r'\.format\(([^,)]+), ([^:)]+):(\.[0-9]+f)\)', r'.format(\1, \2)'),
        (r'\.format\(([^:)]+):(\.[0-9]+f)\)', r'.format(\1)'),
    ]

    for pattern, replacement in patterns_to_fix:
        fixed_content = re.sub(pattern, replacement, fixed_content)

    return fixed_content

def fix_file(filepath):
    """Fix format strings in a single file."""
    print(f"Processing {filepath}...")

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content
        fixed_content = fix_format_strings(content)

        if fixed_content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(fixed_content)
            print(f"  ✅ Fixed format strings in {filepath}")
            return True
        else:
            print(f"  ⏭️ No changes needed in {filepath}")
            return False

    except Exception as e:
        print(f"  ❌ Error processing {filepath}: {e}")
        return False

def main():
    """Main function to fix all Python files in core directory."""
    print("🔧 Fixing invalid decimal literal format strings in core/")
    print("=" * 60)

    # Find all Python files in core directory
    py_files = glob.glob("core/**/*.py", recursive=True)

    fixed_count = 0
    total_count = len(py_files)

    for filepath in py_files:
        if fix_file(filepath):
            fixed_count += 1

    print("\n" + "=" * 60)
    print(f"📊 Summary: Fixed {fixed_count}/{total_count} files")
    print("🎯 Now run: flake8 core/ --select=E999 to verify fixes")

if __name__ == "__main__":
    main() 