import os
import shutil
import subprocess
import sys


def remove_cached_files(base_dir):
    """Remove Python cache and compiled files."""
    print("Removing cached files...")
    for root, dirs, files in os.walk(base_dir):
        # Remove __pycache__ directories
        if '__pycache__' in dirs:
            pycache_path = os.path.join(root, '__pycache__')
            try:
                shutil.rmtree(pycache_path)
                print(f"Removed: {pycache_path}")
            except Exception as e:
                print(f"Error removing {pycache_path}: {e}")

        # Remove .pyc, .pyo, .pyd files
        for file in files:
            if file.endswith(('.pyc', '.pyo', '.pyd')):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"Removed: {file_path}")
                except Exception as e:
                    print(f"Error removing {file_path}: {e}")


def normalize_file_encoding(base_dir):
    """Normalize file encodings to UTF-8."""
    print("Normalizing file encodings...")
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                try:
                    # Read with UTF-8, write back to ensure consistent encoding
                    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                        content = f.read()

                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print(f"Normalized: {filepath}")
                except Exception as e:
                    print(f"Error normalizing {filepath}: {e}")


def run_autoflake(base_dir):
    """Run autoflake to remove unused imports and variables."""
    print("Running autoflake...")
    try:
        subprocess.run([)]
            'autoflake',
            '--in-place',
            '--remove-all-unused-imports',
            '--remove-unused-variables',
            '-r',
            base_dir
        ], check=True)
        print("Autoflake completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Autoflake error: {e}")


def run_black(base_dir):
    """Run black formatter."""
    print("Running black formatter...")
    try:
        subprocess.run([)]
            'black',
            base_dir,
            '--line-length', '100'
        ], check=True)
        print("Black formatting completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Black formatting error: {e}")


def run_isort(base_dir):
    """Run isort to sort imports."""
    print("Running isort...")
    try:
        subprocess.run([)]
            'isort',
            base_dir
        ], check=True)
        print("Isort completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Isort error: {e}")


def main():
    # Directories to clean and prepare
    directories = ['core', 'schwabot', 'utils', 'config']

    # Ensure required tools are installed
    try:
        subprocess.run(['pip', 'install', 'autoflake', 'black', 'isort'], check=True)
    except subprocess.CalledProcessError:
        print("Error installing required tools. Please install manually.")
        sys.exit(1)

    # Process each directory
    for directory in directories:
        if os.path.isdir(directory):
            print(f"\nProcessing directory: {directory}")
            remove_cached_files(directory)
            normalize_file_encoding(directory)
            run_autoflake(directory)
            run_black(directory)
            run_isort(directory)

    print("\nPreparation complete. Ready for flake8 check.")


if __name__ == "__main__":
    main()
