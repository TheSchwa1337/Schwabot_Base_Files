import json
import os
import subprocess
import sys
from collections import defaultdict


def run_flake8(directory):
    """Run flake8 on a specific directory and return detailed results."""
    try:
        # Prepare flake8 command with comprehensive options
        cmd = []
            'flake8',
            directory,
            '--max-line-length=100',
            '--select=E,F,W',  # Focus on Errors, Failures, Warnings
            '--ignore=E501',   # Ignore line too long (we'll handle this with, black)'
            '--format=%(path)s:%(row)d:%(col)d: %(code)s %(text)s'
        ]

        # Run flake8 and capture output
        result = subprocess.run(cmd, capture_output=True, text=True)

        # Parse output line by line
        errors = []
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                try:
                    # Basic parsing of the formatted output
                    parts = line.split(':', 4)
                    if len(parts) >= 5:
                        error = {}
                            'filename': parts[0],
                            'line_number': parts[1],
                            'column': parts[2],
                            'code': parts[3].strip(),
                            'text': parts[4].strip()
                        }
                        errors.append(error)
                except Exception as e:
                    print(f"Error parsing line: {line}")

        return errors

    except subprocess.CalledProcessError as e:
        print(f"Flake8 error in {directory}: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error checking {directory}: {e}")
        return []


def categorize_errors(errors):
    """Categorize and count errors by type."""
    error_categories = defaultdict(list)
    for error in errors:
        error_code = error.get('code', 'Unknown')
        error_categories[error_code].append(error)
    return error_categories


def generate_report(all_errors):
    """Generate a comprehensive error report."""
    print("\n===== FLAKE8 ERROR REPORT =====")

    if not all_errors:
        print("✅ No flake8 issues found!")
        return 0

    total_errors = 0
    for directory, errors in all_errors.items():
        if not errors:
            continue

        print(f"\n📁 Directory: {directory}")
        categorized_errors = categorize_errors(errors)

        for error_code, error_list in sorted(categorized_errors.items()):
            print(f"\n  🔴 Error Code {error_code}: {len(error_list)} occurrences")
            total_errors += len(error_list)

            # Show first 5 detailed errors for each category
            for error in error_list[:5]:
                print(f"    - File: {error.get('filename', 'Unknown')}")
                print()
                    f"      Line {error.get('line_number', 'N/A')}: {error.get('text', 'No details')}")

            if len(error_list) > 5:
                print(f"    ... and {len(error_list) - 5} more")

    print(f"\n🚨 Total Errors: {total_errors}")
    return total_errors


def main():
    # Directories to check
    directories = ['core', 'schwabot', 'utils', 'config']

    # Collect errors from all directories
    all_errors = {}

    for directory in directories:
        if os.path.isdir(directory):
            print(f"\nChecking directory: {directory}")
            directory_errors = run_flake8(directory)
            if directory_errors:
                all_errors[directory] = directory_errors

    # Generate and return report
    total_errors = generate_report(all_errors)

    # Exit with appropriate status
    sys.exit(1 if total_errors > 0 else 0)


if __name__ == "__main__":
    main()
