import os
import re
import subprocess
import sys
import traceback

import chardet


def detect_file_encoding(filepath):
    """Detect the encoding of a file."""
    try:
        with open(filepath, 'rb') as f:
            raw_data = f.read(10000)  # Read first 10KB
            result = chardet.detect(raw_data)
        return result['encoding'] or 'utf-8'
    except Exception as e:
        print(f"Error detecting encoding for {filepath}: {e}")
        return 'utf-8'


def clean_file_content(filepath):
    """Attempt to clean file content of problematic characters."""
    try:
        # Detect encoding
        encoding = detect_file_encoding(filepath)

        # Read file in binary mode
        with open(filepath, 'rb') as f:
            content = f.read()

        # Remove null bytes and other problematic characters
        cleaned_content = content.replace(b'\x00', b'')

        # Try to decode with detected encoding
        try:
            decoded_content = cleaned_content.decode(encoding, errors='replace')
        except UnicodeDecodeError:
            # Fallback to UTF-8 with error handling
            decoded_content = cleaned_content.decode('utf-8', errors='replace')

        # Remove any non-printable characters
        decoded_content = re.sub(r'[^\x20-\x7E\n\r\t]', '', decoded_content)

        # Write cleaned content
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(decoded_content)

        return True
    except Exception as e:
        print(f"Error cleaning file {filepath}: {e}")
        return False


def run_autoflake(filepath):
    """Run autoflake to remove unused imports and variables."""
    try:
        result = subprocess.run()
            ['autoflake',]
             '--in-place',
             '--remove-all-unused-imports',
             '--remove-unused-variables',
             filepath
             ],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print(f"Autoflake issues in {filepath}:")
            print(result.stdout)
            print(result.stderr)

        return result.returncode
    except Exception as e:
        print(f"Error running autoflake on {filepath}: {e}")
        return 1


def run_black(filepath):
    """Run black formatter on a file."""
    try:
        result = subprocess.run()
            ['black',]
             filepath,
             '--line-length', '100'
             ],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print(f"Black formatting issues in {filepath}:")
            print(result.stdout)
            print(result.stderr)

        return result.returncode
    except Exception as e:
        print(f"Error running black on {filepath}: {e}")
        return 1


def run_flake8_on_file(filepath):
    """Run flake8 on a single file with robust error handling."""
    try:
        # Ensure the file is readable and not empty
        if not os.path.exists(filepath):
            print(f"File does not exist: {filepath}")
            return 0

        if os.path.getsize(filepath) == 0:
            print(f"Skipping empty file: {filepath}")
            return 0

        print(f"\n🔍 Processing file: {filepath}")

        # Clean file content
        if clean_file_content(filepath):
            print(f"✨ Cleaned file: {filepath}")

        # Run autoflake to remove unused imports
        autoflake_result = run_autoflake(filepath)
        if autoflake_result == 0:
            print(f"🧹 Autoflake cleaned unused imports: {filepath}")

        # Run black for formatting
        black_result = run_black(filepath)
        if black_result == 0:
            print(f"🎨 Black formatted: {filepath}")

        # Run flake8 with comprehensive checks
        result = subprocess.run()
            ['flake8', filepath,]
             '--max-line-length=100',
             '--show-source',
             '--select=E,F,W',
             '--ignore=E501'],
            capture_output=True,
            text=True
        )

        # Print results for debugging
        if result.returncode != 0:
            print(f"\n{'=' * 50}")
            print(f"❌ Issues in {filepath}:")
            print(f"{'=' * 50}")
            print(result.stdout)
            print(result.stderr)
            print(f"{'=' * 50}\n")
        else:
            print(f"✅ No flake8 issues found in {filepath}")

        return result.returncode

    except Exception as e:
        print(f"❌ Unexpected error processing {filepath}:")
        print(traceback.format_exc())
        return 1


def main(directories):
    # Ensure directories exist
    valid_directories = [d for d in directories if os.path.isdir(d)]
    if not valid_directories:
        print(f"No valid directories found: {directories}")
        sys.exit(1)

    total_errors = 0
    processed_files = 0

    # Collect all Python files
    python_files = []
    for directory in valid_directories:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    python_files.append(filepath)

    # Process files
    print(f"🚀 Starting comprehensive code check on {len(python_files)} files...")
    for filepath in python_files:
        file_errors = run_flake8_on_file(filepath)
        total_errors += file_errors
        processed_files += 1

    print(f"\n📊 Summary:")
    print(f"Processed {processed_files} Python files")
    print(f"Total files with issues: {total_errors}")

    sys.exit(1 if total_errors > 0 else 0)


if __name__ == "__main__":
    # Add more directories if needed
    main(["core", "schwabot", "utils", "config"])
