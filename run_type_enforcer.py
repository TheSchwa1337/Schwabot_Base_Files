from utils.safe_print import safe_print, info, warn, error, success, debug
#!/usr/bin/env python3
"""Run Type Enforcer - Apply Type Annotations.

==========================================



Simple script to run the type enforcer and eliminate MEDIUM priority flake8 issues.

"""

from pathlib import Path
import sys

# Add core to path
sys.path.insert(0, str(Path(__file__).parent / "core"))

try:
    from type_enforcer import type_enforcer

    safe_print(
        "🔧 Applying type annotations to eliminate MEDIUM priority issues..."
    )

    # Apply type annotations to all Python files
    total_stats = {
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
                safe_print(f"⚠️ Error processing {py_file}: {e}")

    safe_print("✅ Type annotation enforcement complete!")
    safe_print(f"📊 Statistics:")
    safe_print(f"   - Functions fixed: {total_stats['functions_fixed']}")
    safe_print(f"   - Parameters fixed: {total_stats['parameters_fixed']}")
    safe_print(f"   - Return types fixed: {total_stats['returns_fixed']}")

    # Run compliance check to see results
    safe_print("\n🔧 Running compliance check to verify results...")

    from compliance_check import main as compliance_check

    results = compliance_check()

    # Count issues by severity
    issue_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0, "CRITICAL": 0}

    for result in results:
        for issue in result.get("issues", []):
            severity = issue.get("severity", "UNKNOWN")
            if severity in issue_counts:
                issue_counts[severity] += 1

    safe_print("📊 Final Issue Counts:")
    safe_print(f"   🟠 HIGH issues: {issue_counts['HIGH']}")
    safe_print(f"   🟡 MEDIUM issues: {issue_counts['MEDIUM']}")
    safe_print(f"   🟢 LOW issues: {issue_counts['LOW']}")
    safe_print(f"   ❌ CRITICAL issues: {issue_counts['CRITICAL']}")

    if issue_counts["HIGH"] == 0 and issue_counts["MEDIUM"] == 0:
        safe_print("\n🎉 SUCCESS: All HIGH and MEDIUM issues resolved!")
        safe_print("   Your codebase is now flake8-compliant for critical issues.")
    else:
        safe_print("\n⚠️ Some issues remain - review the results above.")

except Exception as e:
    safe_print(f"❌ Error: {e}")
    import traceback

    traceback.print_exc()
