import json
import os
import subprocess
import sys
from datetime import datetime


def run_command(command, description):
    """Run a shell command and capture its output."""
    print(f"\n[*] {description}")
    try:
        result = subprocess.run(command, capture_output=True, text=True, shell=True)
        return {}
            'command': ' '.join(command) if isinstance(command, list) else command,
            'returncode': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
    except Exception as e:
        return {}
            'command': ' '.join(command) if isinstance(command, list) else command,
            'returncode': 1,
            'stdout': '',
            'stderr': str(e)
        }


def generate_report(checks):
    """Generate a comprehensive code quality report."""
    report = {}
        'timestamp': datetime.now().isoformat(),
        'total_checks': len(checks),
        'passed_checks': sum(1 for check in checks if check['returncode'] == 0),
        'failed_checks': sum(1 for check in checks if check['returncode'] != 0),
        'checks': checks
    }

    # Write JSON report
    with open('code_quality_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    # Generate human-readable report
    with open('code_quality_report.txt', 'w', encoding='utf-8') as f:
        f.write("Code Quality Report\n")
        f.write(f"Generated: {report['timestamp']}\n\n")
        f.write(f"Total Checks: {report['total_checks']}\n")
        f.write(f"Passed Checks: {report['passed_checks']}\n")
        f.write(f"Failed Checks: {report['failed_checks']}\n\n")

        f.write("Detailed Check Results:\n")
        for check in report['checks']:
            status = "PASSED" if check['returncode'] == 0 else "FAILED"
            f.write(f"\n{status} - {check['command']}\n")
            if check['stdout']:
                f.write(f"  STDOUT:\n{check['stdout']}\n")
            if check['stderr']:
                f.write(f"  STDERR:\n{check['stderr']}\n")

    return report


def diagnose_failures(checks):
    """Provide detailed diagnosis for failed checks."""
    failed_checks = [check for check in checks if check['returncode'] != 0]

    if not failed_checks:
        return

    print("\n[!] Detailed Failure Diagnosis:")
    for check in failed_checks:
        print(f"\nFailed Check: {check['command']}")

        # Specific diagnostics for different tools
        if 'flake8' in check['command']:
            print("Flake8 Diagnostic:")
            print("Potential issues:")
            print("- Syntax errors")
            print("- Unused imports")
            print("- Coding style violations")

        elif 'black' in check['command']:
            print("Black Formatting Diagnostic:")
            print("Potential issues:")
            print("- Inconsistent code formatting")
            print("- Line length exceeding 100 characters")

        elif 'isort' in check['command']:
            print("Import Sorting Diagnostic:")
            print("Potential issues:")
            print("- Unsorted or incorrectly organized imports")

        # Print error output
        if check['stderr']:
            print("\nError Output:")
            print(check['stderr'])

        if check['stdout']:
            print("\nStandard Output:")
            print(check['stdout'])


def main():
    # List of checks to run
    checks = []
        run_command('pip install autoflake black isort flake8 chardet',)
                    'Installing required tools'),
        run_command('python prepare_flake8.py',)
                    'Preparing files for code quality check'),
        run_command('python comprehensive_code_check.py',)
                    'Running comprehensive code check'),
        run_command('flake8 core/ schwabot/ utils/ config/ --max-line-length=100 --count',)
                    'Running flake8 check'),
        run_command('black core/ schwabot/ utils/ config/ --check --line-length=100',)
                    'Checking code formatting with black'),
        run_command('isort core/ schwabot/ utils/ config/ --check-only',)
                    'Checking import sorting')
    ]

    # Generate report
    report = generate_report(checks)

    # Print summary to console
    print("\nCode Quality Check Summary:")
    print(f"Total Checks: {report['total_checks']}")
    print(f"Passed Checks: {report['passed_checks']}")
    print(f"Failed Checks: {report['failed_checks']}")

    # Diagnose failures
    diagnose_failures(checks)

    # Exit with appropriate status
    sys.exit(1 if report['failed_checks'] > 0 else 0)


if __name__ == "__main__":
    main()
