import os
import sys


def check_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            f.read()
        return True
    except UnicodeDecodeError:
        print(f"Problematic file (encoding, issue): {filepath}")
        return False
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return False


def main(directory):
    problematic_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                if not check_file(filepath):
                    problematic_files.append(filepath)

    if problematic_files:
        print("Problematic files found:")
        for file in problematic_files:
            print(file)
        sys.exit(1)
    else:
        print("No problematic files found.")
        sys.exit(0)


if __name__ == "__main__":
    main("core")
