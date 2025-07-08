import os
import subprocess
import sys

#!/usr/bin/env python3
"""
Schwabot Codebase Formatter
===========================

Comprehensive formatting script using autopep8 and Black for the Schwabot trading bot system.
This script ensures consistent, modern formatting while preserving mathematical functionality.
"""


def run_command(cmd, description):
    """Run a command and handle errors gracefully."""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run()
            cmd, shell=True, capture_output=True, text=True, check=True
        )
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(f"Output: {result.stdout[:200]}...")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stdout:
            print(f"stdout: {e.stdout}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
        return False


def find_python_files():
    """Find all Python files in the codebase, excluding certain directories."""
    exclude_dirs = {}
        "__pycache__",
        ".git",
        ".venv",
        "venv",
        "env",
        "build",
        "dist",
        "*.egg-info",
        ".pytest_cache",
        "node_modules",
        "docs/_build",
        ".mypy_cache",
        ".ruff_cache",
        ".tox",
        "htmlcov",
    }

    python_files = []
    for root, dirs, files in os.walk("."):
        # Remove excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]

        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                python_files.append(file_path)

    return python_files


def main():
    """Main formatting workflow."""
    print("🚀 Starting Schwabot Codebase Formatting")
    print("=" * 50)

    # Find all Python files
    python_files = find_python_files()
    print(f"📁 Found {len(python_files)} Python files to format")

    # Step 1: Run autopep8 on all Python files
    print("\n" + "=" * 50)
    print("STEP 1: Running autopep8")
    print("=" * 50)

    autopep8_success = True
    for file_path in python_files:
        cmd = f'autopep8 --in-place --aggressive --aggressive --max-line-length=88 "{file_path}"'
        if not run_command(cmd, f"autopep8 on {file_path}"):
            autopep8_success = False

    if not autopep8_success:
        print("⚠️  Some autopep8 operations failed, but continuing with Black...")

    # Step 2: Run Black on all Python files
    print("\n" + "=" * 50)
    print("STEP 2: Running Black")
    print("=" * 50)

    # Run Black on the entire codebase
    black_cmd = "black --line-length=88 --target-version=py38 ."
    black_success = run_command(black_cmd, "Black formatting")

    # Step 3: Final flake8 check
    print("\n" + "=" * 50)
    print("STEP 3: Final flake8 check")
    print("=" * 50)

    flake8_cmd = "python -m flake8 --config=.flake8 --count --statistics ."
    flake8_success = run_command(flake8_cmd, "flake8 linting check")

    # Summary
    print("\n" + "=" * 50)
    print("FORMATTING SUMMARY")
    print("=" * 50)
    print(f"📁 Files processed: {len(python_files)}")
    print(f"✅ autopep8: {'Success' if autopep8_success else 'Partial success'}")
    print(f"✅ Black: {'Success' if black_success else 'Failed'}")
    print(f"✅ flake8: {'Success' if flake8_success else 'Issues found'}")

    if black_success and flake8_success:
        print("\n🎉 Codebase formatting completed successfully!")
        print("💡 Your code is now properly formatted and linted.")
    else:
        print("\n⚠️  Some formatting steps had issues.")
        print("💡 Review the output above for specific problems.")

    return black_success and flake8_success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
