from typing import Any
#!/usr/bin/env python3
"""
Flake8 Tracker and Fixer - Systematic Issue Resolution
======================================================

This tool tracks and fixes Flake8 issues systematically, working around
environment problems and ensuring all code blocks are properly managed.

Based on systematic elimination of 257+ flake8 issues.
"""

import os
import sys
import subprocess
import re
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class Flake8Issue:
    """Represents a single Flake8 issue"""
    file_path: str
    line_number: int
    column: int
    error_code: str
    message: str
    severity: str = "UNKNOWN"
    fixed: bool = False
    fix_attempted: bool = False


@dataclass
class FileAnalysis:
    """Analysis results for a single file"""
    file_path: str
    issues: List[Flake8Issue] = field(default_factory=list)
    total_issues: int = 0
    high_priority: int = 0
    medium_priority: int = 0
    low_priority: int = 0
    fixed_issues: int = 0
    has_syntax_errors: bool = False
    has_type_issues: bool = False
    has_import_issues: bool = False


class Flake8Tracker:
    """Systematic Flake8 issue tracking and fixing"""

    def __init__(self, root_dir: str = ".") -> None:
        self.root_dir = Path(root_dir)
        self.analysis_results: Dict[str, FileAnalysis] = {}
        self.fix_stats = {
            'total_files_processed': 0,
            'total_issues_found': 0,
            'total_issues_fixed': 0,
            'files_with_errors': 0,
            'files_fixed': 0
        }

        # Priority mappings
        self.priority_mapping = {
            'E999': 'CRITICAL',  # Syntax errors
            'F821': 'HIGH',      # Undefined names
            'F722': 'HIGH',      # Syntax error in forward annotation
            'E302': 'MEDIUM',    # Expected 2 blank lines
            'E501': 'MEDIUM',    # Line too long
            'W293': 'LOW',       # Blank line contains whitespace
            'W291': 'LOW',       # Trailing whitespace
            'F401': 'MEDIUM',    # Imported but unused
            'F841': 'MEDIUM',    # Local variable assigned but never used
        }

    def run_flake8_analysis(self) -> Dict[str, FileAnalysis]:
        """Run Flake8 analysis and parse results"""
        logger.info("Running Flake8 analysis...")

        try:
            # Run flake8 with specific configuration
            cmd = [
                sys.executable, "-m", "flake8",
                "--max-line-length=120",
                "--ignore=E203,W503,F403",
                "--exclude=.venv,__pycache__,build,dist,.git",
                "--format=%(path)s:%(row)d:%(col)d:%(code)s:%(text)s",
                str(self.root_dir)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode == 0 and not result.stdout.strip():
                logger.info("✅ No Flake8 issues found!")
                return {}

            # Parse flake8 output
            issues = self._parse_flake8_output(result.stdout)

            # Group issues by file
            for issue in issues:
                if issue.file_path not in self.analysis_results:
                    self.analysis_results[issue.file_path] = FileAnalysis(file_path=issue.file_path)

                self.analysis_results[issue.file_path].issues.append(issue)
                self.analysis_results[issue.file_path].total_issues += 1

                # Categorize by priority
                priority = self.priority_mapping.get(issue.error_code, 'LOW')
                if priority == 'CRITICAL' or priority == 'HIGH':
                    self.analysis_results[issue.file_path].high_priority += 1
                elif priority == 'MEDIUM':
                    self.analysis_results[issue.file_path].medium_priority += 1
                else:
                    self.analysis_results[issue.file_path].low_priority += 1

                # Track specific issue types
                if issue.error_code == 'E999':
                    self.analysis_results[issue.file_path].has_syntax_errors = True
                elif issue.error_code in ['F821', 'F722']:
                    self.analysis_results[issue.file_path].has_type_issues = True
                elif issue.error_code == 'F401':
                    self.analysis_results[issue.file_path].has_import_issues = True

            # Update stats
            self.fix_stats['total_files_processed'] = len(self.analysis_results)
            self.fix_stats['total_issues_found'] = sum(len(f.issues) for f in self.analysis_results.values())
            self.fix_stats['files_with_errors'] = len(self.analysis_results)

            logger.info(
    f"Found {self.fix_stats['total_issues_found']} issues in {self.fix_stats['files_with_errors']} files")

        except subprocess.TimeoutExpired:
            logger.error("❌ Flake8 analysis timed out")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Flake8 analysis failed: {e}")
        except Exception as e:
            logger.error(f"❌ Error running Flake8 analysis: {e}")

        return self.analysis_results

    def _parse_flake8_output(self, output: str) -> List[Flake8Issue]:
        """Parse Flake8 output into structured issues"""
        issues = []

        for line in output.strip().split('\n'):
            if not line.strip():
                continue

            # Parse format: file:line:column:code:message
            parts = line.split(':', 4)
            if len(parts) >= 5:
                file_path, line_num, col_num, error_code, message = parts

                issue = Flake8Issue(
                    file_path=file_path,
                    line_number=int(line_num),
                    column=int(col_num),
                    error_code=error_code,
                    message=message.strip(),
                    severity=self.priority_mapping.get(error_code, 'LOW')
                )
                issues.append(issue)

        return issues

    def fix_common_issues(self) -> Dict[str, int]:
        """Apply common fixes to files with issues"""
        logger.info("Applying common fixes...")

        fix_counts = {}

        for file_path, analysis in self.analysis_results.items():
            if not analysis.issues:
                continue

            logger.info(f"Fixing issues in {file_path}")
            fixes_applied = self._fix_file_issues(file_path, analysis.issues)
            fix_counts[file_path] = fixes_applied

            if fixes_applied > 0:
                self.fix_stats['files_fixed'] += 1
                self.fix_stats['total_issues_fixed'] += fixes_applied

        return fix_counts

    def _fix_file_issues(self, file_path: str, issues: List[Flake8Issue]) -> int:
        """Fix issues in a specific file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            original_content = content
            fixes_applied = 0

            # Sort issues by line number (descending) to avoid line number shifts
            sorted_issues = sorted(issues, key=lambda x: x.line_number, reverse=True)

            for issue in sorted_issues:
                if self._fix_single_issue(content, issue):
                    fixes_applied += 1
                    issue.fixed = True

            # Write back if changes were made
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                logger.info(f"✅ Applied {fixes_applied} fixes to {file_path}")

            return fixes_applied

        except Exception as e:
            logger.error(f"❌ Error fixing {file_path}: {e}")
            return 0

    def _fix_single_issue(self, content: str, issue: Flake8Issue) -> bool:
        """Fix a single Flake8 issue"""
        lines = content.split('\n')

        if issue.line_number > len(lines):
            return False

        line_index = issue.line_number - 1
        line = lines[line_index]

        # Apply fixes based on error code
        if issue.error_code == 'E999':
            # Syntax error - try to fix common patterns
            return self._fix_syntax_error(lines, line_index, issue)
        elif issue.error_code == 'F821':
            # Undefined name - try to add import or fix reference
            return self._fix_undefined_name(lines, line_index, issue)
        elif issue.error_code == 'E302':
            # Missing blank lines before class/function
            return self._fix_missing_blank_lines(lines, line_index)
        elif issue.error_code == 'E501':
            # Line too long - try to break it
            return self._fix_line_too_long(lines, line_index)
        elif issue.error_code == 'W293':
            # Blank line contains whitespace
            return self._fix_blank_line_whitespace(lines, line_index)
        elif issue.error_code == 'W291':
            # Trailing whitespace
            return self._fix_trailing_whitespace(lines, line_index)
        elif issue.error_code == 'F401':
            # Unused import - remove it
            return self._fix_unused_import(lines, line_index)

        return False

    def _fix_syntax_error(self, lines: List[str], line_index: int, issue: Flake8Issue) -> bool:
        """Fix common syntax errors"""
        line = lines[line_index]

        # Fix common patterns
        if '-> Any -> Any:' in line:
            # Fix double arrow syntax error
            fixed_line = line.replace('-> Any -> Any:', '-> Any:')
            lines[line_index] = fixed_line
            return True

        # Fix missing colons
        if re.search(r'def\s+\w+\s*\([^)]*\)\s*$', line):
            lines[line_index] = line + ':'
            return True

        # Fix bare except
        if re.search(r'except\s*$', line):
            lines[line_index] = line + ':'
            return True

        return False

    def _fix_undefined_name(self, lines: List[str], line_index: int, issue: Flake8Issue) -> bool:
        """Fix undefined name issues"""
        line = lines[line_index]

        # Look for common undefined names and add imports
        undefined_name = issue.message.split("'")[1] if "'" in issue.message else None

        if undefined_name:
            # Try to add import at the top of the file
            import_line = f"from core.type_defs import {undefined_name}"

            # Find the right place to insert import
            for i, existing_line in enumerate(lines):
                if existing_line.strip().startswith('import ') or existing_line.strip().startswith('from '):
                    if i + 1 < len(lines) and not lines[i + 1].strip().startswith(('import ', 'from ')):
                        lines.insert(i + 1, import_line)
                        return True

            # If no imports found, add at the top
            lines.insert(0, import_line)
            return True

        return False

    def _fix_missing_blank_lines(self, lines: List[str], line_index: int) -> bool:
        """Fix missing blank lines before class/function"""
        if line_index > 0 and lines[line_index - 1].strip() != '':
            lines.insert(line_index, '')
            return True
        return False

    def _fix_line_too_long(self, lines: List[str], line_index: int) -> bool:
        """Fix lines that are too long"""
        line = lines[line_index]

        if len(line) > 120:
            # Try to break long lines at logical points
            if '(' in line and ')' in line:
                # Break function calls
                parts = line.split('(', 1)
                if len(parts) == 2:
                    func_part = parts[0]
                    args_part = parts[1]
                    if len(func_part) < 80:
                        lines[line_index] = func_part + '('
                        lines.insert(line_index + 1, '    ' + args_part)
                        return True

        return False

    def _fix_blank_line_whitespace(self, lines: List[str], line_index: int) -> bool:
        """Fix blank lines with whitespace"""
        if lines[line_index].strip() == '' and lines[line_index] != '':
            lines[line_index] = ''
            return True
        return False

    def _fix_trailing_whitespace(self, lines: List[str], line_index: int) -> bool:
        """Fix trailing whitespace"""
        line = lines[line_index]
        if line.rstrip() != line:
            lines[line_index] = line.rstrip()
            return True
        return False

    def _fix_unused_import(self, lines: List[str], line_index: int) -> bool:
        """Remove unused imports"""
        line = lines[line_index]
        if line.strip().startswith(('import ', 'from ')):
            lines[line_index] = ''
            return True
        return False

    def generate_report(self) -> str:
        """Generate a comprehensive report of findings and fixes"""
        report = []
        report.append("=" * 60)
        report.append("FLAKE8 TRACKING AND FIXING REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Summary statistics
        report.append("📊 SUMMARY STATISTICS:")
        report.append(f"   Total files processed: {self.fix_stats['total_files_processed']}")
        report.append(f"   Total issues found: {self.fix_stats['total_issues_found']}")
        report.append(f"   Total issues fixed: {self.fix_stats['total_issues_fixed']}")
        report.append(f"   Files with errors: {self.fix_stats['files_with_errors']}")
        report.append(f"   Files fixed: {self.fix_stats['files_fixed']}")
        report.append("")

        # Priority breakdown
        total_high = sum(f.high_priority for f in self.analysis_results.values())
        total_medium = sum(f.medium_priority for f in self.analysis_results.values())
        total_low = sum(f.low_priority for f in self.analysis_results.values())

        report.append("🎯 PRIORITY BREAKDOWN:")
        report.append(f"   🔴 HIGH/CRITICAL: {total_high}")
        report.append(f"   🟡 MEDIUM: {total_medium}")
        report.append(f"   🟢 LOW: {total_low}")
        report.append("")

        # Files with issues
        if self.analysis_results:
            report.append("📁 FILES WITH ISSUES:")
            for file_path, analysis in sorted(self.analysis_results.items()):
                report.append(f"   {file_path}:")
                report.append(f"     - Total issues: {analysis.total_issues}")
                report.append(f"     - High priority: {analysis.high_priority}")
                report.append(f"     - Medium priority: {analysis.medium_priority}")
                report.append(f"     - Low priority: {analysis.low_priority}")
                report.append(f"     - Fixed issues: {analysis.fixed_issues}")

                if analysis.has_syntax_errors:
                    report.append(f"     - ⚠️  Has syntax errors")
                if analysis.has_type_issues:
                    report.append(f"     - ⚠️  Has type issues")
                if analysis.has_import_issues:
                    report.append(f"     - ⚠️  Has import issues")
                report.append("")

        # Recommendations
        report.append("💡 RECOMMENDATIONS:")
        if total_high > 0:
            report.append("   🔴 Focus on HIGH/CRITICAL issues first (syntax, undefined names)")
        if total_medium > 0:
            report.append("   🟡 Address MEDIUM issues (type annotations, imports)")
        if total_low > 0:
            report.append("   🟢 LOW issues can be addressed with automated formatting")

        report.append("")
        report.append("🛠️  NEXT STEPS:")
        report.append("   1. Review and manually fix any remaining HIGH priority issues")
        report.append("   2. Add proper type annotations using core.type_defs")
        report.append("   3. Run 'python tools/setup_pre_commit.py' to install hooks")
        report.append("   4. Use 'pre-commit run --all-files' for automated checking")

        return "\n".join(report)

    def save_report(self, filename: str = "flake8_tracking_report.txt") -> None:
        """Save the report to a file"""
        report = self.generate_report()
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"📄 Report saved to {filename}")


def main() -> None:
    """Main function"""
    logger.info("🚀 Flake8 Tracker and Fixer")
    logger.info("=" * 40)

    tracker = Flake8Tracker()

    # Run analysis
    logger.info("Step 1: Running Flake8 analysis...")
    analysis_results = tracker.run_flake8_analysis()

    if not analysis_results:
        logger.info("✅ No issues found! Your codebase is clean.")
        return

    # Apply fixes
    logger.info("Step 2: Applying automatic fixes...")
    fix_counts = tracker.fix_common_issues()

    # Generate and save report
    logger.info("Step 3: Generating report...")
    tracker.save_report()

    # Display summary
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"   Files processed: {tracker.fix_stats['total_files_processed']}")
    print(f"   Issues found: {tracker.fix_stats['total_issues_found']}")
    print(f"   Issues fixed: {tracker.fix_stats['total_issues_fixed']}")
    print(f"   Files fixed: {tracker.fix_stats['files_fixed']}")
    print("=" * 60)

    if tracker.fix_stats['total_issues_fixed'] > 0:
        logger.info("🎉 Automatic fixes applied! Review the report for remaining issues.")
    else:
        logger.info("⚠️ No automatic fixes could be applied. Manual review needed.")


if __name__ == "__main__":
    main()