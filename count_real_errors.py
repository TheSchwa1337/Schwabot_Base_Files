#!/usr/bin/env python3
"""Count real vs stub E501 errors."""."""


def is_stub_file(filepath):
    """Check if file is a stub by reading first line."""."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            return "TEMPORARY STUB GENERATED AUTOMATICALLY" in first_line
    except:
        return False

def count_errors():
    """Count real vs stub E501 errors."""."""
    real_errors = []
    stub_errors = []

    with open('e501_errors.txt', 'r') as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith('.'):
                continue

            # Extract file path from error line
            parts = line.split(':')
            if len(parts) >= 2:
                filepath = parts[0]
                if is_stub_file(filepath):
                    stub_errors.append(line)
                else:
                    real_errors.append(line)

    print(f"📊 E501 Error Analysis:")
    print(f"   Total errors: {len(real_errors) + len(stub_errors)}")
    print(f"   Real code errors: {len(real_errors)}")
    print(f"   Stub file errors: {len(stub_errors)}")

    if real_errors:
        print(f"\n🔍 First 10 real code errors:")
        for error in real_errors[:10]:
            print(f"   {error}")
        if len(real_errors) > 10:
            print(f"   ... and {len(real_errors) - 10} more")

if __name__ == "__main__":
    count_errors()
