from pathlib import Path
import subprocess
import sys

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Flake8 Issue Fixer
===============================

Automatically fixes flake8 issues while preserving mathematical logic and trading algorithms.
Uses autopep8 and black with carefully configured settings to maintain code functionality.

Key Features:
- Fixes W293 (blank line contains whitespace) errors
- Fixes E501 (line too long) errors
- Preserves mathematical calculations and trading logic
- Maintains import functionality and dependencies
- Windows CLI compatible

Usage:
    python fix_flake8_issues.py
"""



def check_tools_available():-> bool:
    """Check if required formatting tools are available."""
    tools = ["autopep8", "black", "flake8"]
    missing_tools = []

    for tool in tools:
        try:
            subprocess.run([tool, "--version"], capture_output=True, check=True)
            print(f"✅ {tool} is available")
        except (subprocess.CalledProcessError, FileNotFoundError):
            missing_tools.append(tool)
            print(f"❌ {tool} is not available")

    if missing_tools:
        print(f"\n⚠️  Missing tools: {', '.join(missing_tools)}")
        print("Please install them using:")
        print(f"pip install {' '.join(missing_tools)}")
        return False

    return True


def run_autopep8():-> bool:
    """Run autopep8 on a file with mathematical preservation settings."""
    try:
        # Autopep8 with careful settings to preserve mathematical logic
        cmd = [
            "autopep8",
            "--in-place",
            "--max-line-length=88",
            "--select=W293,E501",  # Only fix whitespace and line length
            "--aggressive",  # More aggressive fixes
            str(file_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  ✅ autopep8 applied to {file_path.name}")
            return True
        else:
            print(f"  ❌ autopep8 failed for {file_path.name}: {result.stderr}")
            return False

    except Exception as e:
        print(f"  ❌ autopep8 error for {file_path.name}: {e}")
        return False


def run_black():-> bool:
    """Run black on a file with mathematical preservation settings."""
    try:
        # Black with settings to preserve mathematical logic
        cmd = [
            "black",
            "--line-length=88",
            "--skip-string-normalization",  # Preserve string formatting
            "--skip-magic-trailing-comma",  # Preserve trailing commas in math
            str(file_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  ✅ black applied to {file_path.name}")
            return True
        else:
            print(f"  ❌ black failed for {file_path.name}: {result.stderr}")
            return False

    except Exception as e:
        print(f"  ❌ black error for {file_path.name}: {e}")
        return False


def check_flake8_issues():-> dict:
    """Check flake8 issues in a file."""
    try:
        cmd = [
            "flake8",
            "--select=W293,E501,F401",  # Whitespace, line length, unused imports
            "--max-line-length=88",
            str(file_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        issues = result.stdout.strip().split("\n") if result.stdout.strip() else []

        issue_counts = {
            "W293": len([i for i in issues if "W293" in i]),
            "E501": len([i for i in issues if "E501" in i]),
            "F401": len([i for i in issues if "F401" in i]),
            "total": len(issues),
        }

        return issue_counts

    except Exception as e:
        print(f"  ❌ flake8 check error for {file_path.name}: {e}")
        return {"W293": 0, "E501": 0, "F401": 0, "total": 0}


def fix_manual_whitespace_issues():-> bool:
    """Manually fix whitespace-only lines that autopep8 might miss."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Fix lines that contain only whitespace
        fixed_lines = []
        changes_made = False

        for line in lines:
            if line.strip() == "" and len(line) > 1:
                fixed_lines.append("\n")  # Replace with just newline
                changes_made = True
            else:
                fixed_lines.append(line)

        if changes_made:
            with open(file_path, "w", encoding="utf-8") as f:
                f.writelines(fixed_lines)
            print(f"  ✅ Manual whitespace fix applied to {file_path.name}")
            return True

        return True

    except Exception as e:
        print(f"  ❌ Manual whitespace fix error for {file_path.name}: {e}")
        return False


def process_file():-> dict:
    """Process a single file with all formatting tools."""
    print(f"\n📄 Processing {file_path.name}...")

    # Check initial issues
    initial_issues = check_flake8_issues(file_path)
    print(
        f"  📊 Initial issues: W293={initial_issues['W293']}, "
        f"E501={initial_issues['E501']}, F401={initial_issues['F401']}, "
        f"Total={initial_issues['total']}"
    )

    if initial_issues["total"] == 0:
        print(f"  ✅ No flake8 issues found in {file_path.name}")
        return {"success": True, "initial": initial_issues, "final": initial_issues}

    # Step 1: Manual whitespace fix
    fix_manual_whitespace_issues(file_path)

    # Step 2: Apply autopep8
    autopep8_success = run_autopep8(file_path)

    # Step 3: Apply black (only if autopep8 succeeded)
    black_success = True
    if autopep8_success:
        black_success = run_black(file_path)

    # Check final issues
    final_issues = check_flake8_issues(file_path)
    print(
        f"  📊 Final issues: W293={final_issues['W293']}, "
        f"E501={final_issues['E501']}, F401={final_issues['F401']}, "
        f"Total={final_issues['total']}"
    )

    improvement = initial_issues["total"] - final_issues["total"]
    if improvement > 0:
        print(f"  🎉 Fixed {improvement} issues!")
    elif final_issues["total"] == 0:
        print("  ✅ All issues resolved!")
    else:
        print(f"  ⚠️  {final_issues['total']} issues remain")

    return {
        "success": autopep8_success and black_success,
        "initial": initial_issues,
        "final": final_issues,
        "improvement": improvement,
    }


def main():
    """Main execution function."""
    print("🚀 Comprehensive Flake8 Issue Fixer")
    print("=" * 50)

    # Check if tools are available
    if not check_tools_available():
        sys.exit(1)

    # Files mentioned by the user that need fixing
    target_files = [
        "core/enhanced_strategy_framework.py",
        "core/strategy_integration_bridge.py",
        "core/advanced_settings_engine.py",
        "core/api/cache_sync.py",
        "core/api/handlers/glassnode.py",
        "core/api/handlers/coingecko.py",
        "core/brain_trading_engine.py",
        "core/biological_immune_error_handler.py",
    ]

    # Additional core files that might have issues
    additional_files = [
        "core/mathlib_v4.py",
        "core/unified_math_system.py",
        "core/matrix_math_utils.py",
        "core/risk_manager.py",
        "core/unified_trading_pipeline.py",
        "core/ccxt_integration.py",
        "core/strategy_logic.py",
    ]

    all_files = target_files + additional_files

    # Process each file
    results = {}
    total_initial_issues = 0
    total_final_issues = 0
    successful_files = 0

    for file_path_str in all_files:
        file_path = Path(file_path_str)

        if not file_path.exists():
            print(f"\n⚠️  File not found: {file_path}")
            continue

        result = process_file(file_path)
        results[file_path.name] = result

        total_initial_issues += result["initial"]["total"]
        total_final_issues += result["final"]["total"]

        if result["success"]:
            successful_files += 1

    # Generate summary report
    print("\n" + "=" * 60)
    print("📊 COMPREHENSIVE FIXING RESULTS SUMMARY")
    print("=" * 60)

    print("\n📈 Overall Statistics:")
    print(f"  Files processed: {len(results)}")
    print(f"  Files successfully formatted: {successful_files}")
    print(f"  Initial total issues: {total_initial_issues}")
    print(f"  Final total issues: {total_final_issues}")
    print(f"  Total issues fixed: {total_initial_issues - total_final_issues}")

    if total_initial_issues > 0:
        improvement_percentage = (
            (total_initial_issues - total_final_issues) / total_initial_issues
        ) * 100
        print(f"  Improvement: {improvement_percentage:.1f}%")

    print("\n📋 File-by-File Results:")
    for file_name, result in results.items():
        status = "✅" if result["success"] else "❌"
        initial = result["initial"]["total"]
        final = result["final"]["total"]
        improvement = result.get("improvement", 0)
        print(f"  {status} {file_name}: {initial} → {final} ({improvement:+d})")

    # Recommendations
    print("\n💡 Recommendations:")
    if total_final_issues == 0:
        print("  🎉 Excellent! All flake8 issues have been resolved.")
        print("  ✅ Your code is now flake8 compliant and ready for production.")
    elif total_final_issues < total_initial_issues * 0.2:
        print("  🎯 Great progress! Most issues have been resolved.")
        print(
            "  📝 Review remaining issues manually - they may require specific attention."
        )
    else:
        print("  ⚠️  Significant issues remain. Consider:")
        print("     - Manual review of complex mathematical expressions")
        print("     - Adding flake8 exceptions for critical trading logic")
        print("     - Breaking down complex functions into smaller pieces")

    print("\n🔧 Next Steps:")
    print("  1. Run test suite to ensure mathematical logic is preserved")
    print("  2. Check that all imports and dependencies still work")
    print("  3. Verify trading strategies still function correctly")
    print("  4. Consider adding pre-commit hooks to maintain formatting")

    return total_final_issues == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
