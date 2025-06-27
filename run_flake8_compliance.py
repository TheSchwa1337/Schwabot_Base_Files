#!/usr/bin/env python3
"""
Run Flake8 Compliance Check

This script runs the comprehensive Flake8 compliance check on the entire
Schwabot codebase while preserving mathematical integrity and profit-tier logic.

Usage:
    python run_flake8_compliance.py [--project-root PATH] [--output-file PATH]
"""

import argparse
import logging
import sys
import os
from pathlib import Path

# Set UTF-8 encoding for stdout and stderr
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

# Configure logging with UTF-8 support
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('flake8_compliance.log', encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)


def main():
    """Main function to run Flake8 compliance check"""
    parser = argparse.ArgumentParser(description="Run Flake8 compliance check on Schwabot codebase")
    parser.add_argument(
        "--project-root",
        type=str,
        default=".",
        help="Root directory of the project (default: current directory)"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="flake8_compliance_report.md",
        help="Output file for the compliance report (default: flake8_compliance_report.md)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Import the compliance orchestrator
        from core.flake8_compliance_orchestrator import run_compliance_check
        
        logger.info("SCHWABOT FLAKE8 COMPLIANCE CHECK")
        logger.info("=" * 50)
        logger.info(f"Project root: {args.project_root}")
        logger.info(f"Output file: {args.output_file}")
        logger.info("=" * 50)
        
        # Run the compliance check
        report = run_compliance_check(
            project_root=args.project_root,
            output_file=args.output_file
        )
        
        # Print summary
        print("\n" + "=" * 50)
        print("COMPLIANCE CHECK SUMMARY")
        print("=" * 50)
        print(f"Total Files: {report.total_files}")
        print(f"Compliant Files: {report.compliant_files}")
        if report.total_files > 0:
            compliance_rate = (report.compliant_files / report.total_files) * 100
            print(f"Compliance Rate: {compliance_rate:.1f}%")
        else:
            print("Compliance Rate: 0%")
        
        print(f"Total Issues: {report.total_issues}")
        print(f"Critical Issues: {report.critical_issues}")
        print(f"High Issues: {report.high_issues}")
        print(f"Medium Issues: {report.medium_issues}")
        print(f"Low Issues: {report.low_issues}")
        print(f"Auto-Fixed Issues: {report.auto_fixed_issues}")
        print(f"Mathematical Integrity Score: {report.mathematical_integrity_score:.3f}")
        print(f"Profit-Tier Logic Score: {report.profit_tier_logic_score:.3f}")
        print(f"Processing Time: {report.processing_time:.2f} seconds")
        
        # Print critical issues if any
        if report.critical_issues > 0:
            print("\nCRITICAL ISSUES FOUND:")
            for file_report in report.file_reports:
                critical_issues = [i for i in file_report.issues if i.severity.value == "critical"]
                if critical_issues:
                    print(f"  {file_report.file_path}:")
                    for issue in critical_issues:
                        print(f"    Line {issue.line_number}: {issue.error_code} - {issue.description}")
        
        # Print high priority issues if any
        if report.high_issues > 0:
            print("\nHIGH PRIORITY ISSUES FOUND:")
            for file_report in report.file_reports:
                high_issues = [i for i in file_report.issues if i.severity.value == "high"]
                if high_issues:
                    print(f"  {file_report.file_path}:")
                    for issue in high_issues:
                        print(f"    Line {issue.line_number}: {issue.error_code} - {issue.description}")
        
        # Print mathematical integrity issues if any
        low_integrity_files = [fr for fr in report.file_reports if fr.mathematical_integrity_score < 0.8]
        if low_integrity_files:
            print("\nMATHEMATICAL INTEGRITY ISSUES:")
            for file_report in low_integrity_files:
                print(f"  {file_report.file_path}: Integrity Score {file_report.mathematical_integrity_score:.3f}")
        
        # Print profit-tier logic issues if any
        profit_tier_issues = [fr for fr in report.file_reports if not fr.profit_tier_logic_preserved]
        if profit_tier_issues:
            print("\nPROFIT-TIER LOGIC ISSUES:")
            for file_report in profit_tier_issues:
                print(f"  {file_report.file_path}: Profit-tier logic not preserved")
        
        print("\n" + "=" * 50)
        if report.critical_issues == 0 and report.high_issues == 0:
            print("COMPLIANCE CHECK PASSED!")
            print("All critical and high priority issues have been resolved.")
        else:
            print("COMPLIANCE CHECK FAILED!")
            print("Critical or high priority issues remain.")
            print("Please review the detailed report and fix the issues.")
        
        print(f"\nDetailed report saved to: {args.output_file}")
        print("=" * 50)
        
        # Return appropriate exit code
        if report.critical_issues > 0:
            sys.exit(1)
        else:
            sys.exit(0)
            
    except ImportError as e:
        logger.error(f"Failed to import compliance orchestrator: {e}")
        print("ERROR: Could not import Flake8 compliance orchestrator")
        print("Make sure you're running this script from the correct directory")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Compliance check failed: {e}")
        print(f"ERROR: Compliance check failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 