# Tests Folder

This directory contains all automated and manual test scripts for Schwabot and its subsystems.

## Structure
- **Unit tests**: Validate individual modules and functions.
- **Integration tests**: Ensure correct interaction between subsystems (e.g., EXO Echo Signals ↔ Lantern Core).
- **Performance tests**: Benchmark and stress-test critical components.

## Usage
Run tests with:
```
python -m unittest discover .
```
Or run individual scripts as needed.

## Conventions
- Test scripts are named `test_*.py`.
- Subfolders may contain specialized test suites.

---
Keep this folder up to date as new features and bugfixes are added. 