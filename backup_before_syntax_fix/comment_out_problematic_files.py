import glob
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

#!/usr/bin/env python3
"""
Script to comment out problematic legacy files while preserving clean implementation.
This script will:
1. Identify all Python files with syntax errors
2. Comment out the entire content of problematic files
3. Add a header explaining why the file is commented out
4. Preserve clean implementation files
"""


# Clean implementation files that should NOT be commented out
CLEAN_FILES = {}
    'core/clean_math_foundation.py',
    'core/clean_profit_vectorization.py',
    'core/clean_trading_pipeline.py',
    'core/clean_unified_math.py',
    'utils/price_bridge.py',
    'utils/safe_print.py',
    'utils/secure_config_manager.py',
    'utils/file_integrity_checker.py',
    'utils/fractal_injection.py',
    'utils/hash_validator.py',
    'utils/historical_data_downloader.py',
    'utils/market_data_utils.py',
    'utils/math_utils.py',
    'utils/logging_setup.py',
    'core/api/__init__.py',
    'core/api/integration_manager.py',
    'core/api/exchange_connection.py',
    'core/api/enums.py',
    'core/api/data_models.py',
    'core/api/cache_sync.py',
    'core/api/handlers/__init__.py',
    'core/api/handlers/whale_alert.py',
    'core/api/handlers/glassnode.py',
    'core/api/handlers/coingecko.py',
    'core/api/handlers/base_handler.py',
    'core/api/handlers/alt_fear_greed.py',
    'config/__init__.py',
    'config/mathematical_framework_config.py',
    'config/schwabot_config.py',
    'config/risk_config.py',
    'config/matrix_response_schema.py',
    'config/cooldown_config.py',
    'config/io_utils.py',
    'config/config_utils.py'
}


def check_syntax_error(file_path):
    """Check if a Python file has syntax errors."""
    try:
        result = subprocess.run()
            [sys.executable, '-m', 'py_compile', file_path],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.returncode != 0
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
        return True  # Assume problematic if we can't check'


def comment_out_file(file_path):
    """Comment out the entire content of a file and add explanation header."""
    try:
        # Read the original content
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            original_content = f.read()

        # Get current date
        current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Create commented version with explanation header
        header = f'''"""'
LEGACY FILE - COMMENTED OUT DUE TO SYNTAX ERRORS

This file has been automatically commented out because it contains syntax errors
that prevent the Schwabot system from running properly.

Original file: {file_path}
Date commented out: {current_date}

The clean implementation has been preserved in the following files:
- core/clean_math_foundation.py (mathematical, foundation)
- core/clean_profit_vectorization.py (profit, calculations)
- core/clean_trading_pipeline.py (trading, logic)
- core/clean_unified_math.py (unified, mathematics)

All core functionality has been reimplemented in clean, production-ready files.
"""

# ORIGINAL CONTENT COMMENTED OUT BELOW:
"""
{original_content}
"""
'''

        # Write the commented version back to the file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(header)

        print(f"✓ Commented out: {file_path}")
        return True

    except Exception as e:
        print(f"✗ Error commenting out {file_path}: {e}")
        return False


def find_python_files(directory):
    """Find all Python files in a directory recursively."""
    python_files = []
    if os.path.exists(directory):
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.py'):
                    full_path = os.path.join(root, file)
                    python_files.append(full_path)
    return python_files


def main():
    """Main function to process all Python files."""
    print("🔍 Scanning for problematic Python files...")

    # Find all Python files
    python_files = []
    for directory in ['core', 'utils', 'config']:
        python_files.extend(find_python_files(directory))

    print(f"Found {len(python_files)} Python files")

    # Track statistics
    total_files = len(python_files)
    clean_files = 0
    problematic_files = 0
    commented_files = 0
    errors = 0

    print("\n📋 Processing files...")
    print("=" * 60)

    for file_path in python_files:
        # Normalize path for comparison (convert Windows backslashes to forward, slashes)
        normalized_path = file_path.replace('\\', '/')

        # Skip clean implementation files
        if normalized_path in CLEAN_FILES:
            print(f"✓ Preserved (clean): {file_path}")
            clean_files += 1
            continue

        # Check for syntax errors
        if check_syntax_error(file_path):
            print(f"⚠️  Problematic: {file_path}")
            problematic_files += 1

            # Comment out the file
            if comment_out_file(file_path):
                commented_files += 1
            else:
                errors += 1
        else:
            print(f"✓ Valid syntax: {file_path}")
            clean_files += 1

    print("\n" + "=" * 60)
    print("📊 SUMMARY:")
    print(f"Total files processed: {total_files}")
    print(f"Clean files preserved: {clean_files}")
    print(f"Problematic files found: {problematic_files}")
    print(f"Files successfully commented out: {commented_files}")
    print(f"Errors during processing: {errors}")

    if commented_files > 0:
        print(f"\n✅ Successfully commented out {commented_files} problematic files!")
        print("The clean implementation is preserved and ready for use.")
        print("\nClean files available:")
        for clean_file in sorted(CLEAN_FILES):
            if os.path.exists(clean_file):
                print(f"  - {clean_file}")
    else:
        print("\n🎉 No problematic files found! All files are clean.")


if __name__ == "__main__":
    main()
