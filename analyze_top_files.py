#!/usr/bin/env python3
"""Analyze top 5 files with most E501 errors."""."""

from collections import defaultdict
from utils.safe_print import safe_print, info, warn, error, success, debug


def is_stub_file(filepath):
    """Check if file is a stub."""."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            return "TEMPORARY STUB GENERATED AUTOMATICALLY" in first_line
    except:
        return False


def analyze_errors():
    """Analyze E501 errors and find top 5 files."""."""
    file_errors = defaultdict(int)
    real_files = []
    stub_files = []

    with open('e501_errors.txt', 'r') as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith('.'):
                continue

            # Extract file path
            parts = line.split(':')
            if len(parts) >= 2:
                filepath = parts[0]
                file_errors[filepath] += 1

                # Categorize as real or stub
                if is_stub_file(filepath):
                    stub_files.append(filepath)
                else:
                    real_files.append(filepath)

    # Get top 5 real files with most errors
    real_file_counts = {f: file_errors[f] for f in set(real_files)}
    top_5 = sorted(
        real_file_counts.items(),
        key=lambda x: x[1],
        reverse=True,
    )[:5]

    safe_print("\\u1f4ca E501 Error Analysis")
    safe_print("=" * 50)
    safe_print(
        "Total unique files with errors: "
        f"{len(file_errors)}"
    )
    safe_print(
        "Real code files: "
        f"{len(set(real_files))}"
    )
    safe_print(
        "Stub files: "
        f"{len(set(stub_files))}"
    )

    safe_print(f"\\n\\u1f3c6 TOP 5 FILES WITH MOST E501 ERRORS:")
    safe_print("-" * 50)
    for i, (filepath, count) in enumerate(top_5, 1):
        safe_print(f"{i}. {filepath}: {count} errors")

    safe_print(f"\\n\\u1f4cb TOTAL ERRORS BY CATEGORY:")
    real_error_count = sum(file_errors[f] for f in set(real_files))
    stub_error_count = sum(file_errors[f] for f in set(stub_files))

    safe_print(
        "   Real code errors: "
        f"{real_error_count}"
    )
    safe_print(
        "   Stub file errors: "
        f"{stub_error_count}"
    )

    return top_5

if __name__ == "__main__":
    analyze_errors()

"""