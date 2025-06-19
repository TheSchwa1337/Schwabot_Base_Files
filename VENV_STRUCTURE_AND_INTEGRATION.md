# Virtual Environment & Project Integration Guide

This document provides an overview of how the Schwabot codebase is organized, how the Python virtual environment is managed, and instructions for syntax checks, flake8 enforcement, core module health checks, and performance timing.

---

## 1. Project Overview

The Schwabot unified mathematics framework consists of:

- **Core modules** implementing advanced mathematical routines (drift shells, tensor feedback, phase harmonization, etc.) in `core/`
- **Utilities** for compliance, syntax fixing, and health checking in `tools/`
- **Demos & Tests** under `demos/` and `tests/`
- **Configuration & CI/CD** files at project root (e.g., `flake8`, `pytest`, `GitHub Actions`)

This guide shows how to maintain a robust development workflow.

---

## 2. Virtual Environment Management

- The project uses a dedicated virtual environment in `.venv/` (or similar folder).
- **Creating/Recreating**:
  ```powershell
  # Windows PowerShell (from project root)
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  pip install -r requirements.txt
  ```
- **Cleaning up**: If `.venv/` becomes corrupted, delete the folder and recreate as above.
- **Recording changes**: Keep a log in `COMPREHENSIVE_VENV_CHANGELOG.md` or equivalent when dependencies are updated.

---

## 3. File & Directory Structure

```
/ (project root)
├── .venv/                      # Python virtual environment
├── core/                       # Core mathematical modules
│   ├── advanced_drift_shell_integration.py
│   ├── type_defs.py
│   ├── error_handler.py
│   └── ...
├── tools/                      # Scripts for compliance, health checks, syntax fixes
│   ├── syntax_checker.py
│   ├── compliance_check.py
│   └── ...
├── demos/                      # Example usage and integration demos
├── tests/                      # Unit and integration tests
├── requirements.txt            # Python dependencies
├── flake8.ini                  # Style rules and ignores
├── VENV_STRUCTURE_AND_INTEGRATION.md  # <-- this file
└── README.md                   # Project introduction
```

Add new modules under `core/` and new tools under `tools/` following naming conventions.

---

## 4. Core Module Health & Syntax Checks

A dedicated script (`tools/syntax_checker.py`) scans **all** `.py` files for syntax errors, ensuring flake8 can parse everything.

### Usage

```powershell
# Activate venv first
.\.venv\Scripts\Activate.ps1
python tools/syntax_checker.py
```

### Behavior

- Reports files with syntax errors (SyntaxError, UnicodeDecodeError, etc.)
- Exits with non-zero status if any errors found
- Should be run before flake8 to avoid silent skips

---

## 5. Flake8 Integration & Enforcement

- **Configuration** in `flake8.ini` includes:
  - Max line length, ignored rules, per-directory overrides
- **Run locally**:
  ```powershell
  flake8 --config flake8.ini core/ tools/ demos/ tests/
  ```
- **CI/CD**: add a GitHub Actions step to run flake8 and fail the build on any error.

---

## 6. Hash-Based Change Detection & Backlog Integration

For performance-critical or cached data, we use file hash detection:

1. **Compute hash** (SHA256) of each core module
2. Compare against stored hash cache (e.g., `.cache/hash_store.json`)
3. If changed:
   - Clear related cache/backlog data
   - Trigger re-initialization of any long-lived memory structures

Example pseudocode in `tools/core_health.py`.

---

## 7. Performance Timing Checks

Measure syntax parse times and optionally flake8 run times to monitor changes over time.

- Use `time.perf_counter()` in scripts
- Log per-module parse time (ms) to CSV or dashboard

Example:

```python
start = time.perf_counter()
ast.parse(content)
elapsed_ms = (time.perf_counter() - start) * 1000
```

Store results in `tools/logs/health_timing.csv`.

---

## 8. Onboarding New Mathematical Modules

When adding a new core file:

1. Place in `core/` with a descriptive name
2. Add to `CORE_MODULES` list in `tools/core_health.py`
3. Run `tools/syntax_checker.py` & flake8 locally
4. Update documentation in this file and `README.md`
5. Run unit tests covering the new routines

---

## 9. Next Steps & Maintenance

- Automate daily or pre-commit health checks
- Integrate health/timing dashboard (e.g., Grafana)
- Periodically review `requirements.txt` and renew dependencies
- Keep `VENV_STRUCTURE_AND_INTEGRATION.md` up-to-date as new patterns arise

---

_Reminder: Keeping syntax, style, and performance checks at the core of development ensures a reliable and maintainable mathematical framework._ 