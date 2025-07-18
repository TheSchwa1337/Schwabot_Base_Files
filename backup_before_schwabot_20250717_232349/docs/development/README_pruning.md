# Schwabot Mathematical Pruning & Archiving System

## Overview
This system helps you safely prune, refactor, and archive code in a mathematically complex codebase. It ensures you never lose important mathematical logic, and that all deletions are logged and justified.

## Workflow

1. **Run the Math Structure Report**
   - `python math_structure_report.py`
   - This scans the codebase for mathematically relevant code and outputs `math_structure_report.md`.

2. **Run the Dead Code Pruner**
   - `python dead_code_pruner.py`
   - This finds unused or redundant code, cross-references with the math report, and outputs `prune_candidates_report.md`.

3. **Review Candidates**
   - Only delete/archive code that is not mathematically relevant and not used elsewhere.
   - If unsure, move to `archive/` and log in `prune_log.md`.

4. **Preserve All Math**
   - If you must remove or refactor a mathematical structure, copy it to `math_legacy.md` with context and rationale.

5. **Log Every Deletion**
   - Every deletion or archive must be logged in `prune_log.md`.

6. **Test and Validate**
   - Run Flake8 and your test suite after every major change.

## Best Practices
- Never delete code with mathematical or trading logic unless you are certain it is unused and not relevant.
- Always log and archive before deleting.
- Use the reports to guide safe, confident pruning.
- Keep your `math_legacy.md` and `prune_log.md` up to date for transparency and future reference.

## Files
- `math_structure_report.py` — Scans for mathematical relevance
- `dead_code_pruner.py` — Finds unused/redundant code
- `math_structure_report.md` — Output: math-relevant code
- `prune_candidates_report.md` — Output: prune candidates
- `prune_log.md` — Log of all deletions/archives
- `math_legacy.md` — Archive of removed/refactored math logic

---

**This system ensures you only prune what is safe, and you never lose mathematical value.** 