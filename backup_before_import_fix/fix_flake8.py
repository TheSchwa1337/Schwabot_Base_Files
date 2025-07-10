#!/usr/bin/env python3
"""
Flake8 Auto-Fix Script for Schwabot Trading System

This script systematically addresses Flake8 violations in the following order:
1. Line Length (E501) - Automated wrapping
2. Spacing Rules (E302, E305, E303, E261, E231) - Block cleaning
3. Unused Imports & Variables (F401, F841) - Manual review needed

Usage:
    python fix_flake8.py [--dry-run] [--target-dir DIR]
"""

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Flake8Fixer:
    """Automated Flake8 violation fixer."""

    def __init__(self, target_dir: str = ".", dry_run: bool = False):
        self.target_dir = Path(target_dir)
        self.dry_run = dry_run
        self.results = {
            "E501": {"fixed": 0, "remaining": 0},
            "E302": {"fixed": 0, "remaining": 0},
            "E305": {"fixed": 0, "remaining": 0},
            "E303": {"fixed": 0, "remaining": 0},
            "E261": {"fixed": 0, "remaining": 0},
            "E231": {"fixed": 0, "remaining": 0},
            "F401": {"fixed": 0, "remaining": 0},
            "F841": {"fixed": 0, "remaining": 0},
        }

        # Directories to process in order
        self.directories = [
            "core/",
            "schwabot/",
            "config/",
            "test/",
            "utils/",
            "newmath/",
            "ncco_core/",
        ]

        # Files to ignore
        self.ignore_patterns = ["__pycache__", ".venv", "build", "dist", ".git", "*.pyc", "*.pyo"]

    def should_ignore_file(self, file_path: Path) -> bool:
        """Check if file should be ignored."""
        for pattern in self.ignore_patterns:
            if pattern in str(file_path):
                return True
        return False

    def run_autopep8(self, error_codes: List[str], max_line_length: int = 100) -> bool:
        """Run autopep8 with specified error codes."""
        try:
            cmd = [
                "autopep8",
                str(self.target_dir),
                "--select=" + ",".join(error_codes),
                "--in-place",
                "--recursive",
            ]

            if max_line_length:
                cmd.extend(["--max-line-length", str(max_line_length)])

            if self.dry_run:
                cmd.append("--diff")
                logger.info(f"DRY RUN: Would run: {' '.join(cmd)}")
                return True

            logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                logger.info(f"✅ autopep8 completed successfully for {error_codes}")
                return True
            else:
                logger.error(f"❌ autopep8 failed: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"❌ Error running autopep8: {e}")
            return False

    def count_violations(self, error_codes: List[str]) -> Dict[str, int]:
        """Count current violations for specified error codes."""
        try:
            cmd = ["flake8", str(self.target_dir), "--select=" + ",".join(error_codes), "--count"]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                # Parse the count output
                lines = result.stdout.strip().split('\n')
                counts = {}
                for line in lines:
                    if ':' in line:
                        parts = line.split(':')
                        if len(parts) >= 2:
                            count = int(parts[0])
                            error_code = parts[1].strip()
                            counts[error_code] = count
                return counts
            else:
                logger.warning(f"flake8 count failed: {result.stderr}")
                return {}

        except Exception as e:
            logger.error(f"Error counting violations: {e}")
            return {}

    def fix_line_length(self) -> bool:
        """Fix E501 (line too long) violations."""
        logger.info("🔧 Step 1: Fixing line length violations (E501)")

        # Count before
        before_counts = self.count_violations(["E501"])
        initial_count = before_counts.get("E501", 0)

        # Run autopep8
        success = self.run_autopep8(["E501"], max_line_length=100)

        if success and not self.dry_run:
            # Count after
            after_counts = self.count_violations(["E501"])
            final_count = after_counts.get("E501", 0)

            fixed = initial_count - final_count
            self.results["E501"]["fixed"] = fixed
            self.results["E501"]["remaining"] = final_count

            logger.info(f"✅ E501: Fixed {fixed} violations, {final_count} remaining")

        return success

    def fix_spacing_rules(self) -> bool:
        """Fix spacing rule violations (E302, E305, E303, E261, E231)."""
        logger.info("📏 Step 2: Fixing spacing rule violations")

        spacing_codes = ["E302", "E305", "E303", "E261", "E231"]

        # Count before
        before_counts = self.count_violations(spacing_codes)

        # Run autopep8
        success = self.run_autopep8(spacing_codes)

        if success and not self.dry_run:
            # Count after
            after_counts = self.count_violations(spacing_codes)

            for code in spacing_codes:
                initial = before_counts.get(code, 0)
                final = after_counts.get(code, 0)
                fixed = initial - final

                self.results[code]["fixed"] = fixed
                self.results[code]["remaining"] = final

                logger.info(f"✅ {code}: Fixed {fixed} violations, {final} remaining")

        return success

    def analyze_unused_imports_variables(self) -> Dict[str, Any]:
        """Analyze F401 (unused imports) and F841 (unused variables) violations."""
        logger.info("🧹 Step 3: Analyzing unused imports and variables")

        unused_codes = ["F401", "F841"]
        analysis = self.count_violations(unused_codes)

        for code in unused_codes:
            count = analysis.get(code, 0)
            self.results[code]["remaining"] = count
            logger.info(f"📊 {code}: {count} violations found (manual review needed)")

        return analysis

    def generate_report(self) -> str:
        """Generate a comprehensive report of the fixes."""
        report = []
        report.append("=" * 60)
        report.append("FLAKE8 FIX REPORT")
        report.append("=" * 60)

        total_fixed = 0
        total_remaining = 0

        for error_code, stats in self.results.items():
            fixed = stats["fixed"]
            remaining = stats["remaining"]
            total_fixed += fixed
            total_remaining += remaining

            status = "✅" if remaining == 0 else "⚠️" if fixed > 0 else "❌"
            report.append(f"{status} {error_code}: {fixed} fixed, {remaining} remaining")

        report.append("-" * 60)
        report.append(f"📊 SUMMARY: {total_fixed} total fixes, {total_remaining} remaining")

        if total_remaining > 0:
            report.append("")
            report.append("🔍 REMAINING WORK:")
            report.append("  • F401/F841: Manual review of unused imports/variables")
            report.append("  • E501: Manual line breaking for complex cases")
            report.append("  • Other: Individual file review needed")

        return "\n".join(report)

    def run_full_fix(self) -> bool:
        """Run the complete Flake8 fix process."""
        logger.info("🚀 Starting comprehensive Flake8 fix process")
        logger.info(f"Target directory: {self.target_dir}")
        logger.info(f"Dry run: {self.dry_run}")

        # Step 1: Fix line length
        if not self.fix_line_length():
            logger.error("❌ Line length fix failed")
            return False

        # Step 2: Fix spacing rules
        if not self.fix_spacing_rules():
            logger.error("❌ Spacing rules fix failed")
            return False

        # Step 3: Analyze unused imports/variables
        self.analyze_unused_imports_variables()

        # Generate report
        report = self.generate_report()
        logger.info("\n" + report)

        # Save report to file
        report_file = self.target_dir / "flake8_fix_report.txt"
        with open(report_file, 'w') as f:
            f.write(report)

        logger.info(f"📄 Report saved to: {report_file}")

        return True


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Automated Flake8 fixer for Schwabot")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    parser.add_argument("--target-dir", default=".", help="Target directory to fix (default: current)")

    args = parser.parse_args()

    fixer = Flake8Fixer(target_dir=args.target_dir, dry_run=args.dry_run)
    success = fixer.run_full_fix()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
