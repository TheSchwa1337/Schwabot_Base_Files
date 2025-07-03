# -*- coding: utf-8 -*-
import pathlib


def clean_null_bytes(fix: bool = True) -> list[str]:
    """Scan repository for ``NUL`` bytes in *.py files."

Parameters
----------
fix
If ``True`` replace ``\x00`` with an empty byte string in - place.

Returns
-------
list[str]
        Paths that were cleaned (or that contain null bytes if *fix* is False)."""
    """

"""


""""""
""""""
 root = pathlib.Path(".")
  cleaned: list[str] = []
   for py in root.rglob("*.py"):
        try:
            content = py.read_bytes()
        except Exception:
            # skip unreadable files
continue
if b"\x00" in content:
            cleaned.append(str(py))
            if fix:
                py.write_bytes(content.replace(b"\x00", b""))
    return cleaned


if __name__ == "__main__":
    affected = clean_null_bytes(fix=True)
    if affected:
        print("Fixed null bytes in:")
        for p in affected:
            print("  ", p)
    else:
        print("No null bytes found.")
