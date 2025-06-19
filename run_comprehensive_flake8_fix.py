#!/usr/bin/env python3
"""
Comprehensive Flake8 Fix Orchestrator
Runs all flake8 fixers in optimal order for maximum effectiveness.
"""

import os
import sys
import time
import subprocess
import argparse
from pathlib import Path

def run_command(command_list, description=""):
    """Run a command and return success status."""
    try:
        print(f"🔧 {description}")
        result = subprocess.run(command_list, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  ✅ Success")
            return True
        else:
            print(f"  ❌ Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"  ❌ Exception: {e}")
        return False

def get_flake8_stats():
    """Get current flake8 statistics."""
    try:
        result = subprocess.run(['flake8', '--statistics', '.'], capture_output=True, text=True)
        lines = result.stdout.strip().split('\n') if result.stdout.strip() else []
        
        stats = {}
        total_issues = 0
        
        for line in lines:
            if line and not line.startswith('--'):
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        count = int(parts[0])
                        error_code = parts[1]
                        stats[error_code] = count
                        total_issues += count
                    except ValueError:
                        continue
        
        return stats, total_issues
    except Exception as e:
        print(f"Error getting flake8 stats: {e}")
        return {}, 0

def show_flake8_summary(stats, total_issues, title="Flake8 Issues"):
    """Display flake8 statistics in a nice format."""
    print(f"\n📊 {title}: {total_issues} total issues")
    if stats:
        # Group by category
        syntax_errors = {k: v for k, v in stats.items() if k.startswith('E9')}
        line_length = {k: v for k, v in stats.items() if k in ['E501', 'E502']}
        whitespace = {k: v for k, v in stats.items() if k.startswith('W2') or k.startswith('W3')}
        blank_lines = {k: v for k, v in stats.items() if k.startswith('E30')}
        imports = {k: v for k, v in stats.items() if k.startswith('F4') or k.startswith('F8')}
        naming = {k: v for k, v in stats.items() if k.startswith('F82') or k.startswith('F54')}
        other = {k: v for k, v in stats.items() if k not in {**syntax_errors, **line_length, **whitespace, **blank_lines, **imports, **naming}}
        
        categories = [
            ("🚨 Syntax Errors", syntax_errors),
            ("📏 Line Length", line_length),
            ("🔲 Whitespace", whitespace),
            ("📝 Blank Lines", blank_lines),
            ("📦 Imports", imports),
            ("🏷️  Naming/Variables", naming),
            ("🔧 Other", other)
        ]
        
        for category_name, category_stats in categories:
            if category_stats:
                category_total = sum(category_stats.values())
                print(f"  {category_name}: {category_total}")
                for error_code, count in sorted(category_stats.items()):
                    print(f"    {error_code}: {count}")
    print()

def run_comprehensive_fix():
    """Run the comprehensive flake8 fixing process."""
    print("🚀 COMPREHENSIVE FLAKE8 FIX ORCHESTRATOR")
    print("=" * 60)
    
    # Get initial state
    print("📊 Analyzing initial state...")
    initial_stats, initial_total = get_flake8_stats()
    show_flake8_summary(initial_stats, initial_total, "Initial State")
    
    if initial_total == 0:
        print("🎉 No flake8 issues found! Your code is already compliant.")
        return
    
    start_time = time.time()
    
    # Phase 1: Test files (specialized handling)
    print("🔥 PHASE 1: Fixing Test Files")
    print("-" * 40)
    if os.path.exists('test_files_flake8_fixer.py'):
        success = run_command([sys.executable, 'test_files_flake8_fixer.py'], 
                            "Running specialized test file fixer")
        if success:
            mid_stats, mid_total = get_flake8_stats()
            improvement = initial_total - mid_total
            print(f"  📈 Improvement: {improvement} issues fixed")
        else:
            print("  ⚠️  Test file fixer not available or failed")
    else:
        print("  ⚠️  test_files_flake8_fixer.py not found, skipping specialized test fixes")
    
    # Phase 2: Core files (comprehensive approach)
    print("\n🔥 PHASE 2: Comprehensive Core File Fixes")
    print("-" * 40)
    if os.path.exists('master_flake8_comprehensive_fixer.py'):
        success = run_command([sys.executable, 'master_flake8_comprehensive_fixer.py'], 
                            "Running master comprehensive fixer")
        if success:
            phase2_stats, phase2_total = get_flake8_stats()
            improvement = initial_total - phase2_total
            print(f"  📈 Total improvement: {improvement} issues fixed")
        else:
            print("  ⚠️  Master fixer failed")
    else:
        print("  ⚠️  master_flake8_comprehensive_fixer.py not found")
        return
    
    # Phase 3: Additional cleanup passes
    print("\n🔥 PHASE 3: Additional Cleanup Passes")
    print("-" * 40)
    
    # Run autopep8 for additional formatting
    print("🔧 Running autopep8 for additional formatting...")
    try:
        result = subprocess.run(['autopep8', '--in-place', '--recursive', '--aggressive', '.'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("  ✅ autopep8 completed successfully")
        else:
            print(f"  ⚠️  autopep8 had issues: {result.stderr}")
    except FileNotFoundError:
        print("  ⚠️  autopep8 not installed, skipping")
    
    # Run isort for import organization
    print("🔧 Running isort for import organization...")
    try:
        result = subprocess.run(['isort', '.'], capture_output=True, text=True)
        if result.returncode == 0:
            print("  ✅ isort completed successfully")
        else:
            print(f"  ⚠️  isort had issues: {result.stderr}")
    except FileNotFoundError:
        print("  ⚠️  isort not installed, skipping")
    
    # Final state analysis
    print("\n📊 FINAL ANALYSIS")
    print("=" * 40)
    final_stats, final_total = get_flake8_stats()
    total_improvement = initial_total - final_total
    
    processing_time = time.time() - start_time
    
    print(f"⏱️  Total processing time: {processing_time:.2f} seconds")
    print(f"🎯 Issues fixed: {total_improvement}")
    print(f"📈 Improvement rate: {(total_improvement / max(initial_total, 1)) * 100:.1f}%")
    
    show_flake8_summary(final_stats, final_total, "Final State")
    
    if final_total == 0:
        print("🎉 CONGRATULATIONS! All flake8 issues have been resolved!")
        print("Your codebase is now fully flake8 compliant! ✨")
    elif final_total < 50:
        print("🎊 EXCELLENT PROGRESS! Only a few issues remain.")
        print("Consider reviewing the remaining issues manually.")
    elif total_improvement > initial_total * 0.8:
        print("🔥 GREAT SUCCESS! Significant improvement achieved.")
        print("Most issues have been resolved automatically.")
    else:
        print("📋 PARTIAL SUCCESS. Some issues remain.")
        print("Consider reviewing the patterns and running additional fixes.")
    
    # Generate detailed report
    if final_total > 0:
        print("\n📝 DETAILED REMAINING ISSUES REPORT")
        print("-" * 40)
        try:
            result = subprocess.run(['flake8', '.'], capture_output=True, text=True)
            remaining_lines = result.stdout.strip().split('\n')[:20]  # Show first 20
            for line in remaining_lines:
                if line:
                    print(f"  {line}")
            if len(result.stdout.strip().split('\n')) > 20:
                print(f"  ... and {len(result.stdout.strip().split('\n')) - 20} more issues")
        except Exception as e:
            print(f"  Error generating detailed report: {e}")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS")
    print("-" * 40)
    
    if final_total == 0:
        print("✅ Consider setting up pre-commit hooks to maintain flake8 compliance")
        print("✅ Add flake8 checks to your CI/CD pipeline")
    elif final_total < 20:
        print("✅ Review remaining issues manually for best practices")
        print("✅ Consider adjusting flake8 configuration for project-specific needs")
    else:
        print("✅ Consider running the fixer again after reviewing patterns")
        print("✅ Some issues may require manual intervention")
        print("✅ Check for complex syntax errors that need careful review")
    
    print("✅ Consider using black for consistent code formatting")
    print("✅ Regular automated formatting helps maintain code quality")

def main():
    parser = argparse.ArgumentParser(description='Comprehensive Flake8 Fix Orchestrator')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without making changes')
    parser.add_argument('--skip-tools', action='store_true',
                       help='Skip autopep8 and isort (only run custom fixers)')
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("🔍 DRY RUN MODE - No files will be modified")
        print("This will show the current flake8 status only.\n")
        
        stats, total = get_flake8_stats()
        show_flake8_summary(stats, total, "Current State")
        
        if total > 0:
            print("💡 Run without --dry-run to fix these issues automatically.")
        return
    
    # Check for required files
    required_files = ['master_flake8_comprehensive_fixer.py', 'test_files_flake8_fixer.py']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nPlease ensure all fixer scripts are in the current directory.")
        return
    
    run_comprehensive_fix()

if __name__ == '__main__':
    main() 