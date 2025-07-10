#!/usr/bin/env python3
"""
Verify Flake8 cleanup status for Schwabot Trading System.

This script checks the current state of Flake8 violations
and provides a summary of what's been fixed and what remains.
"""

import os
import subprocess
import sys
from pathlib import Path


def run_flake8_check(directory="core"):
    """Run Flake8 check and return results."""
    try:
        result = subprocess.run(
            ["flake8", directory, "--max-line-length=100", "--count"],
            capture_output=True,
            text=True,
            timeout=60
        )
        return result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return "Timeout occurred", "Flake8 check took too long"
    except FileNotFoundError:
        return "Flake8 not found", "Please install flake8: pip install flake8"
    except Exception as e:
        return f"Error: {e}", ""


def analyze_violations(output):
    """Analyze Flake8 output and categorize violations."""
    if not output or "not found" in output or "Error:" in output:
        return {"error": output}
    
    lines = output.strip().split('\n')
    violations = {}
    
    for line in lines:
        if ':' in line and any(char.isdigit() for char in line):
            # Parse violation line
            parts = line.split(':')
            if len(parts) >= 3:
                file_path = parts[0]
                line_num = parts[1]
                error_code = parts[2].split()[0] if parts[2].strip() else "UNKNOWN"
                
                if error_code not in violations:
                    violations[error_code] = []
                violations[error_code].append(f"{file_path}:{line_num}")
    
    return violations


def get_violation_descriptions():
    """Get descriptions for common Flake8 error codes."""
    return {
        "E999": "Syntax Error - Critical",
        "F821": "Undefined Name - Critical",
        "E501": "Line Too Long",
        "F401": "Unused Import",
        "F841": "Unused Variable",
        "E305": "Expected 2 Blank Lines After Class/Function",
        "E261": "At Least 2 Spaces Before Inline Comment",
        "W292": "No Newline at End of File",
        "W291": "Trailing Whitespace",
        "W293": "Blank Line Contains Whitespace",
        "C901": "Function Too Complex",
        "W505": "Doc Line Too Long",
        "E128": "Continuation Line Under-indented",
        "E127": "Continuation Line Over-indented",
        "E124": "Closing Bracket Does Not Match Visual Indentation",
        "E131": "Continuation Line Unaligned",
        "E129": "Visually Indented Line with Same Indent",
        "E502": "Backslash Redundant Between Brackets",
        "E701": "Multiple Statements on One Line",
        "F541": "F-string Missing Placeholders",
        "I201": "Missing Newline Between Import Groups",
        "I100": "Import Statements in Wrong Order",
        "D205": "1 Blank Line Required Between Summary and Description",
        "D400": "First Line Should End with Period"
    }


def print_summary(violations, descriptions):
    """Print a formatted summary of violations."""
    print("=" * 80)
    print("FLAKE8 CLEANUP VERIFICATION SUMMARY")
    print("=" * 80)
    
    if "error" in violations:
        print(f"❌ ERROR: {violations['error']}")
        return
    
    if not violations:
        print("🎉 EXCELLENT! No Flake8 violations found!")
        return
    
    # Categorize by severity
    critical = ["E999", "F821"]
    high = ["E501", "F401", "F841", "E305", "E128", "E127", "E124", "E131", "E129"]
    medium = ["E261", "W292", "W291", "W293", "C901", "W505"]
    low = ["E502", "E701", "F541", "I201", "I100", "D205", "D400"]
    
    total_violations = sum(len(v) for v in violations.values())
    
    print(f"📊 Total Violations Found: {total_violations}")
    print()
    
    # Critical violations
    critical_count = sum(len(violations.get(code, [])) for code in critical)
    if critical_count > 0:
        print("🚨 CRITICAL VIOLATIONS (Must Fix):")
        for code in critical:
            if code in violations:
                print(f"  {code}: {len(violations[code])} - {descriptions.get(code, 'Unknown')}")
                for violation in violations[code][:3]:  # Show first 3
                    print(f"    {violation}")
                if len(violations[code]) > 3:
                    print(f"    ... and {len(violations[code]) - 3} more")
        print()
    
    # High priority violations
    high_count = sum(len(violations.get(code, [])) for code in high)
    if high_count > 0:
        print("⚠️  HIGH PRIORITY VIOLATIONS:")
        for code in high:
            if code in violations:
                print(f"  {code}: {len(violations[code])} - {descriptions.get(code, 'Unknown')}")
        print()
    
    # Medium priority violations
    medium_count = sum(len(violations.get(code, [])) for code in medium)
    if medium_count > 0:
        print("📝 MEDIUM PRIORITY VIOLATIONS:")
        for code in medium:
            if code in violations:
                print(f"  {code}: {len(violations[code])} - {descriptions.get(code, 'Unknown')}")
        print()
    
    # Low priority violations
    low_count = sum(len(violations.get(code, [])) for code in low)
    if low_count > 0:
        print("ℹ️  LOW PRIORITY VIOLATIONS:")
        for code in low:
            if code in violations:
                print(f"  {code}: {len(violations[code])} - {descriptions.get(code, 'Unknown')}")
        print()
    
    # Overall assessment
    print("📈 ASSESSMENT:")
    if critical_count == 0:
        print("✅ No critical violations - System can run!")
    else:
        print(f"❌ {critical_count} critical violations - Must fix before deployment")
    
    if total_violations < 50:
        print("✅ Low violation count - Good code quality")
    elif total_violations < 100:
        print("⚠️  Moderate violation count - Consider cleanup")
    else:
        print("❌ High violation count - Needs significant cleanup")
    
    print()
    print("🔧 RECOMMENDATIONS:")
    if critical_count > 0:
        print("1. Fix all critical violations immediately")
    if high_count > 0:
        print("2. Address high priority violations for better code quality")
    if medium_count > 0:
        print("3. Consider fixing medium priority violations")
    if low_count > 0:
        print("4. Low priority violations can be addressed over time")
    
    print("5. Use the .flake8 configuration file to ignore intentional violations")
    print("6. Run 'python fix_critical_issues.py' for automated fixes")


def main():
    """Main verification function."""
    print("🔍 Verifying Flake8 cleanup status...")
    print()
    
    # Check if flake8 is available
    try:
        subprocess.run(["flake8", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Flake8 not found. Please install it:")
        print("   pip install flake8")
        return
    
    # Run Flake8 check
    stdout, stderr = run_flake8_check()
    
    if stderr:
        print(f"⚠️  Warning: {stderr}")
    
    # Analyze violations
    violations = analyze_violations(stdout)
    descriptions = get_violation_descriptions()
    
    # Print summary
    print_summary(violations, descriptions)
    
    print()
    print("=" * 80)
    print("Verification complete!")
    print("=" * 80)


if __name__ == "__main__":
    main() 