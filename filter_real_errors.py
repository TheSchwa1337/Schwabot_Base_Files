from dual_unicore_handler import DualUnicoreHandler
import os
import re

from utils.safe_print import safe_print, info, warn, error, success, debug


# Initialize Unicode handler
unicore = DualUnicoreHandler()

# -*- coding: utf - 8 -*-
"""Filter E501 errors to exclude temporary stub files.


"""
"""
"""
"""
"""


def is_stub_file(filepath):
    """Check if a file is marked as a temporary stub."""


"""
"""
"""
"""
   try:
        with open(filepath, 'r', encoding='utf - 8') as f:
            first_line = f.readline().strip()
            return "TEMPORARY STUB GENERATED AUTOMATICALLY" in first_line
    except (FileNotFoundError, UnicodeDecodeError):
        return False


def filter_e501_errors():
    """Read E501 errors and filter out stub files."""


"""
"""
"""
"""
   real_errors = []
    stub_errors = []

    if not os.path.exists('e501_errors.txt'):
        safe_print("\\u274c e501_errors.txt not found. Run 'flake8 . --select = E501 > e501_errors.txt' first.")
        return

    with open('e501_errors.txt', 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

# Parse the error line: filepath:line:col: E501 message
            match = re.match(r'^([^:]+):(\\d+):(\\d+): E501 (.+)$', line)
            if match:
                filepath = match.group(1)
                if is_stub_file(filepath):
                    stub_errors.append(line)
                else:
                    real_errors.append(line)

# Write filtered results
    with open('real_e501_errors.txt', 'w') as f:
        for error in real_errors:
            f.write(error + '\n')

    with open('stub_e501_errors.txt', 'w') as f:
        for error in stub_errors:
            f.write(error + '\n')

# Print summary
    safe_print(f"\\u1f4ca E501 Error Analysis:")
    safe_print(f"   Total errors: {len(real_errors) + len(stub_errors)}")
    safe_print(f"   Real code errors: {len(real_errors)}")
    safe_print(f"   Stub file errors: {len(stub_errors)}")
    safe_print(f"\\n\\u2705 Filtered results saved to:")
    safe_print(f"   - real_e501_errors.txt ({len(real_errors)} errors)")
    safe_print(f"   - stub_e501_errors.txt ({len(stub_errors)} errors)")

# Show first few real errors
    if real_errors:
        safe_print(f"\\n\\u1f50d First 10 real code errors:")
        for error in real_errors[:10]:
            safe_print(f"   {error}")
        if len(real_errors) > 10:
            safe_print(f"   ... and {len(real_errors) - 10} more")


if __name__ == "__main__":
    filter_e501_errors()
