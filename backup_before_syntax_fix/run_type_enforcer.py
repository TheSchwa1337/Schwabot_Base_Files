# -*- coding: utf - 8 -*-
import sys
from pathlib import Path

from dual_unicore_handler import DualUnicoreHandler
from utils.safe_print import debug, error, info, safe_print, success, warn

# Initialize Unicode handler
unicore = DualUnicoreHandler()

"""Run Type Enforcer - Apply Type Annotations."
==========================================

Simple script to run the type enforcer and eliminate MEDIUM priority flake8 issues.
"""

# Add core to path
sys.path.insert(0, str(Path(__file__).parent / "core"))

try:
    from type_enforcer import type_enforcer

    safe_print()
        "\\u1f527 Applying type annotations to eliminate MEDIUM priority issues..."
    )

    # Apply type annotations to all Python files
    total_stats = {}
        "functions_fixed": 0,
        "parameters_fixed": 0,
        "returns_fixed": 0,
    }

    for py_file in Path(".").rglob("*.py"):
        if py_file.is_file():
            try:
                stats = type_enforcer.enforce_type_annotations(str(py_file))
                for key in total_stats:
                    total_stats[key] += stats[key]
            except Exception as e:
                safe_print(f"\\u26a0\\ufe0f Error processing {py_file}: {e}")

    safe_print("\\u2705 Type annotation enforcement complete!")
    safe_print(f"\\u1f4ca Statistics:")
    safe_print(f"   - Functions fixed: {total_stats['functions_fixed']}")
    safe_print(f"   - Parameters fixed: {total_stats['parameters_fixed']}")
    safe_print(f"   - Return types fixed: {total_stats['returns_fixed']}")

    # Run compliance check to see results
    safe_print("\\n\\u1f527 Running compliance check to verify results...")

    from compliance_check import main as compliance_check

    results = compliance_check()

    # Count issues by severity
    issue_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0, "CRITICAL": 0}

    for result in results:
        for issue in result.get("issues", []):
            severity = issue.get("severity", "UNKNOWN")
            if severity in issue_counts:
                issue_counts[severity] += 1

    safe_print("\\u1f4ca Final Issue Counts:")
    safe_print(f"   \\u1f7e0 HIGH issues: {issue_counts['HIGH']}")
    safe_print(f"   \\u1f7e1 MEDIUM issues: {issue_counts['MEDIUM']}")
    safe_print(f"   \\u1f7e2 LOW issues: {issue_counts['LOW']}")
    safe_print(f"   \\u274c CRITICAL issues: {issue_counts['CRITICAL']}")

    if issue_counts["HIGH"] == 0 and issue_counts["MEDIUM"] == 0:
        safe_print("\\n\\u1f389 SUCCESS: All HIGH and MEDIUM issues resolved!")
        safe_print("   Your codebase is now flake8 - compliant for critical issues.")
    else:
        safe_print("\\n\\u26a0\\ufe0f Some issues remain - review the results above.")

except Exception as e:
    safe_print(f"\\u274c Error: {e}")
    import traceback

    traceback.print_exc()
