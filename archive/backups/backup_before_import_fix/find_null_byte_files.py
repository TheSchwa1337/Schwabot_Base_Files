import os
import sys

import chardet


def detect_encoding(filepath):
    """Detect the file's encoding."""'
    try:
        with open(filepath, 'rb') as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
        return result['encoding'] or 'utf-8'
    except Exception as e:
        print(f"Error detecting encoding for {filepath}: {e}")
        return 'utf-8'


def has_null_bytes(filepath):
    """Check if a file contains null bytes."""
    try:
        with open(filepath, 'rb') as f:
            content = f.read()
            return b'\x00' in content
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return False


def find_null_byte_files(directories):
    """Find files with null bytes in the specified directories."""
    null_byte_files = []

    for directory in directories:
        if not os.path.isdir(directory):
            print(f"Warning: {directory} is not a valid directory.")
            continue

        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.py'):  # Focus on Python files
                    filepath = os.path.join(root, file)
                    try:
                        if has_null_bytes(filepath):
                            null_byte_files.append(filepath)
                            print(f"Null byte file found: {filepath}")

                            # Additional diagnostics
                            encoding = detect_encoding(filepath)
                            file_size = os.path.getsize(filepath)
                            print(f"  Detected Encoding: {encoding}")
                            print(f"  File Size: {file_size} bytes")
                    except Exception as e:
                        print(f"Error processing {filepath}: {e}")

    return null_byte_files


def clean_null_byte_files(null_byte_files):
    """Attempt to clean files with null bytes."""
    cleaned_files = []
    for filepath in null_byte_files:
        try:
            # Detect original encoding
            original_encoding = detect_encoding(filepath)

            # Read file in binary mode
            with open(filepath, 'rb') as f:
                content = f.read()

            # Remove null bytes
            cleaned_content = content.replace(b'\x00', b'')

            # Try to decode with detected encoding
            try:
                decoded_content = cleaned_content.decode(original_encoding)
            except UnicodeDecodeError:
                # Fallback to UTF-8 with error handling
                decoded_content = cleaned_content.decode('utf-8', errors='replace')

            # Write cleaned content
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(decoded_content)

            cleaned_files.append(filepath)
            print(f"Cleaned null bytes from: {filepath}")
            print(f"  Original Encoding: {original_encoding}")
            print(f"  Original Size: {len(content)} bytes")
            print(f"  Cleaned Size: {len(cleaned_content)} bytes")
        except Exception as e:
            print(f"Error cleaning {filepath}: {e}")

    return cleaned_files


def main():
    # Directories to check
    directories = ['core', 'schwabot', 'utils', 'config']

    # Find files with null bytes
    null_byte_files = find_null_byte_files(directories)

    if null_byte_files:
        print(f"\nFound {len(null_byte_files)} files with null bytes.")

        # Attempt to clean files
        cleaned_files = clean_null_byte_files(null_byte_files)

        print(f"\nCleaned {len(cleaned_files)} files.")

        # Exit with error if any null byte files were found
        sys.exit(1 if null_byte_files else 0)
    else:
        print("No null byte files found.")
        sys.exit(0)


if __name__ == "__main__":
    main()
