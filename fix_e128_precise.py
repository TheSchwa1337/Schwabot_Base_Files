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


Precise fix for E128 indentation errors in uros_v1_integration_test.py"""
""""""
""""""
""""""
""""""
"""


def fix_e128_precise():"""
    """Fix E128 indentation errors precisely.""""""
""""""
""""""
""""""
"""
"""
file_path = "tools / uros_v1_integration_test.py"

with open(file_path, 'r', encoding='utf - 8') as f:
        content = f.read()

original_content = content

# Fix the specific E128 errors by aligning with opening parenthesis
content = content.replace(
        '        successful_tests = sum(1 for result in self.test_results.values() \\n                             if isinstance(result, bool) and result)',
        '        successful_tests = sum(1 for result in self.test_results.values() \\n                             if isinstance(result, bool) and result)'
    )

content = content.replace(
        '        total_tests = sum(1 for result in self.test_results.values()\\n                         if isinstance(result, bool))',
        '        total_tests = sum(1 for result in self.test_results.values()\\n                         if isinstance(result, bool))'
    )

# Actually, let me fix this more precisely by targeting the exact lines
    lines = content.split('\n')
    fixed_lines = []

for i, line in enumerate(lines):
        if i == 773:  # Line 774 (0 - indexed)
# Fix the indentation to align with opening parenthesis
fixed_line = '        successful_tests = sum(1 for result in self.test_results.values() \\n                             if isinstance(result, bool) and result)'
            print(f"  \\u1f527 Fixed E128 at line {i + 1}")
        elif i == 775:  # Line 776 (0 - indexed)
# Fix the indentation to align with opening parenthesis
fixed_line = '        total_tests = sum(1 for result in self.test_results.values() \\n                         if isinstance(result, bool))'
            print(f"  \\u1f527 Fixed E128 at line {i + 1}")
        else:
            fixed_line = line

fixed_lines.append(fixed_line)

content = '\n'.join(fixed_lines)

# Check if changes were made
if content != original_content:
# Backup and write
backup_path = f"{file_path}.backup"
        with open(backup_path, 'w', encoding='utf - 8') as f:
            f.write(original_content)

with open(file_path, 'w', encoding='utf - 8') as f:
            f.write(content)

print(f"\\u2705 Fixed E128 errors in {file_path}")
    else:
        print(f"\\u2139\\ufe0f No E128 errors found in {file_path}")


if __name__ == "__main__":
    fix_e128_precise()

""""""
""""""
""""""
""""""
""""""
"""
"""