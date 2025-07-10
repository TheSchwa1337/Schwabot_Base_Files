import glob
import os
import subprocess
import sys
from datetime import datetime
from typing import List

#!/usr/bin/env python3
"""
Schwabot Flake8 Error Reduction Workflow
========================================

This script provides a complete workflow for reducing Flake8 errors while
preserving mathematical structures in the Schwabot codebase.

Workflow:
1. Run Flake8 analysis to identify all errors
2. Categorize errors (critical vs auto-fixable)
3. Auto-fix formatting issues while preserving math
4. Generate comprehensive reports
5. Provide recommendations for manual fixes
"""


def run_command(): -> bool:
    """Run a command and handle errors."""
    print(f"\n[RUN] {description}")
    print(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode == 0:
            print(f"[OK] {description} completed successfully")
            if result.stdout.strip():
                print("Output:", result.stdout.strip())
            return True
        else:
            print(f"[FAIL] {description} failed")
            if result.stderr.strip():
                print("Error:", result.stderr.strip())
            return False

    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] {description} timed out")
        return False
    except Exception as e:
        print(f"[ERROR] {description} failed with exception: {e}")
        return False


def check_dependencies(): -> bool:
    """Check if required dependencies are installed."""
    print("🔍 Checking dependencies...")

    required_packages = ["flake8", "autopep8"]
    missing_packages = []

    for package in required_packages:
        try:
            subprocess.run()
                [sys.executable, "-m", package, "--version"],
                capture_output=True,
                check=True,
            )
            print(f"✅ {package} is available")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"❌ {package} is missing")
            missing_packages.append(package)

    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Please install them with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False

    return True


def run_analysis(): -> bool:
    """Run the Flake8 analysis."""
    return run_command()
        [sys.executable, "flake8_analyzer.py"], "Running Flake8 analysis"
    )


def run_auto_fix(): -> bool:
    """Run the auto-fix process."""
    return run_command()
        [sys.executable, "auto_fix_flake8.py"], "Running auto-fix process"
    )


def run_post_fix_analysis(): -> bool:
    """Run Flake8 analysis after fixes to see improvement."""
    return run_command()
        [sys.executable, "flake8_analyzer.py"], "Running post-fix Flake8 analysis"
    )


def generate_workflow_report(): -> str:
    """Generate a comprehensive workflow report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = []
    report.append("# Schwabot Flake8 Error Reduction Workflow Report\n")
    report.append(f"Generated: {timestamp}\n")

    # Check for analysis reports
    analysis_files = []
        "flake8_analysis_report.md",
        "auto_fix_log_*.md",
        "math_structure_report.md",
        "prune_candidates_report.md",
    ]
    report.append("## 📊 Generated Reports")
    for pattern in analysis_files:
        if "*" in pattern:
            # Handle wildcard patterns

            files = glob.glob(pattern)
            for file in files:
                if os.path.exists(file):
                    report.append(f"- ✅ {file}")
                else:
                    report.append(f"- ❌ {file} (not, found)")
        else:
            if os.path.exists(pattern):
                report.append(f"- ✅ {pattern}")
            else:
                report.append(f"- ❌ {pattern} (not, found)")

    report.append("\n## 📋 Next Steps")
    report.append("1. **Review the analysis reports** - Understand what errors exist")
    report.append()
        "2. **Check auto-fix results** - Verify mathematical structures were preserved"
    )
    report.append()
        "3. **Address critical errors** - Fix syntax and import issues manually"
    )
    report.append()
        "4. **Test functionality** - Ensure the codebase still works correctly"
    )
    report.append("5. **Iterate** - Run this workflow again if needed")

    report.append("\n## 🔧 Manual Fix Recommendations")
    report.append("- **E999 (Syntax, errors)**: Fix syntax issues manually")
    report.append()
        "- **F821 (Undefined, names)**: Add missing imports or define variables"
    )
    report.append("- **F822 (Undefined names in, __all__)**: Fix __all__ declarations")
    report.append()
        "- **F823 (Local variable referenced before, assignment)**: Fix variable scope"
    )
    report.append("- **F831 (Duplicate argument, name)**: Fix function signatures")
    report.append()
        "- **F841 (Local variable assigned but never, used)**: Remove unused variables"
    )
    report.append()
        "- **F901 (Return statement with, assignment)**: Refactor complex returns"
    )

    report.append("\n## 🔬 Mathematical Structure Preservation")
    report.append("- Files marked with 🔬 contain mathematical logic")
    report.append("- Always review changes to these files carefully")
    report.append("- Use `math_legacy.md` to preserve removed mathematical structures")
    report.append("- Test mathematical functions after any changes")

    return "\n".join(report)


def save_workflow_report(report: str):
    """Save the workflow report."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"flake8_workflow_report_{timestamp}.md"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"📄 Workflow report saved to {filename}")


def main():
    """Run the complete Flake8 error reduction workflow."""
    print("[START] Schwabot Flake8 Error Reduction Workflow")
    print("=" * 50)

    # Step 1: Check dependencies
    if not check_dependencies():
        print("\n❌ Dependencies check failed. Please install missing packages.")
        return False

    # Step 2: Run initial analysis
    print("\n" + "=" * 50)
    print("STEP 1: Initial Flake8 Analysis")
    print("=" * 50)

    if not run_analysis():
        print("\n⚠️  Analysis failed, but continuing...")

    # Step 3: Run auto-fix
    print("\n" + "=" * 50)
    print("STEP 2: Auto-Fix Process")
    print("=" * 50)

    if not run_auto_fix():
        print("\n⚠️  Auto-fix failed, but continuing...")

    # Step 4: Run post-fix analysis
    print("\n" + "=" * 50)
    print("STEP 3: Post-Fix Analysis")
    print("=" * 50)

    if not run_post_fix_analysis():
        print("\n⚠️  Post-fix analysis failed, but continuing...")

    # Step 5: Generate workflow report
    print("\n" + "=" * 50)
    print("STEP 4: Generate Workflow Report")
    print("=" * 50)

    report = generate_workflow_report()
    save_workflow_report(report)

    # Final summary
    print("\n" + "=" * 50)
    print("[DONE] Workflow Complete!")
    print("=" * 50)
    print("[OK] Flake8 error reduction workflow completed")
    print("[INFO] Check the generated reports for detailed analysis")
    print("[INFO] Address any remaining critical errors manually")
    print("[INFO] Test your codebase to ensure functionality is maintained")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
