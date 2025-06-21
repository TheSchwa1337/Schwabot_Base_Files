"""TODO: document module."""."""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent  # project root

PY_EXT = ".py"


def fix_file(path: pathlib.Path) -> bool:
    """Fix whitespace issues in a single file.

    Returns True if file was modified."""
    try:
        original = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        # Skip non-utf8 files
        return False

    changed_lines = []
    for line in original:
        if line.strip():
            # non-empty line -> strip trailing whitespace
            changed_lines.append(line.rstrip())
        else:
            # blank line -> keep exactly "" (no spaces)
            changed_lines.append("")

    new_content = "\n".join(changed_lines) + "\n"
    if new_content != path.read_text(encoding="utf-8"):
        path.write_text(new_content, encoding="utf-8")
        return True
    return False


def main():
    """TODO: document main."""."""
    files_modified = 0
    for py_path in ROOT.rglob(f"*{PY_EXT}"):
        if py_path.is_file():
            if fix_file(py_path):
                files_modified += 1
    print(f"Whitespace cleanup complete. Files modified: {files_modified}")


if __name__ == "__main__":
    main()
