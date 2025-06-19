from typing import Any
#!/usr/bin/env python3
"""
Comprehensive Compliance Monitor - Schwabot Code Quality System
==============================================================

This tool provides comprehensive monitoring of code quality, working around
environment issues and ensuring systematic tracking of all compliance metrics.

Based on systematic elimination of 257+ flake8 issues.
"""

import os
import sys
import subprocess
import re
import json
import ast
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class CodeQualityMetrics:
    """Comprehensive code quality metrics"""
    file_path: str
    total_lines: int = 0
    code_lines: int = 0
    comment_lines: int = 0
    blank_lines: int = 0
    functions: int = 0
    classes: int = 0
    imports: int = 0
    type_annotations: int = 0
    missing_type_annotations: int = 0
    syntax_errors: int = 0
    undefined_names: int = 0
    unused_imports: int = 0
    long_lines: int = 0
    trailing_whitespace: int = 0
    blank_line_whitespace: int = 0
    complexity_score: float = 0.0


@dataclass
class ComplianceReport:
    """Comprehensive compliance report"""
    timestamp: datetime
    total_files: int = 0
    total_issues: int = 0
    high_priority_issues: int = 0
    medium_priority_issues: int = 0
    low_priority_issues: int = 0
    files_with_issues: int = 0
    compliance_score: float = 0.0
    metrics: Dict[str, CodeQualityMetrics] = field(default_factory=dict)


class ComplianceMonitor:
    """Comprehensive code compliance monitoring"""

    def __init__(self, root_dir: str = ".") -> None:
        self.root_dir = Path(root_dir)
        self.python_files: List[Path] = []
        self.quality_metrics: Dict[str, CodeQualityMetrics] = {}
        self.issues: Dict[str, List[Dict]] = {}

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

    def discover_python_files(self) -> List[Path]:
        """Discover all Python files in the project"""
        logger.info("Discovering Python files...")

        exclude_patterns = {
            '.venv', '__pycache__', 'build', 'dist', '.git',
            'node_modules', '.pytest_cache', '*.pyc'
        }

        python_files = []
        for file_path in self.root_dir.rglob("*.py"):
            # Skip excluded directories
            if any(pattern in str(file_path) for pattern in exclude_patterns):
                continue

            python_files.append(file_path)

        self.python_files = python_files
        logger.info(f"Found {len(python_files)} Python files")
        return python_files

    def analyze_file_quality(self, file_path: Path) -> CodeQualityMetrics:
        """Analyze code quality metrics for a single file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            lines = content.split('\n')
            metrics = CodeQualityMetrics(file_path=str(file_path))

            # Basic line counting
            metrics.total_lines = len(lines)
            metrics.code_lines = 0
            metrics.comment_lines = 0
            metrics.blank_lines = 0

            for line in lines:
                stripped = line.strip()
                if not stripped:
                    metrics.blank_lines += 1
                elif stripped.startswith('#'):
                    metrics.comment_lines += 1
                else:
                    metrics.code_lines += 1

                    # Check for long lines
                    if len(line) > 120:
                        metrics.long_lines += 1

                    # Check for trailing whitespace
                    if line.rstrip() != line:
                        metrics.trailing_whitespace += 1

                    # Check for blank lines with whitespace
                    if stripped == '' and line != '':
                        metrics.blank_line_whitespace += 1

            # Parse AST for deeper analysis
            try:
                tree = ast.parse(content)
                metrics.functions = len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)])
                metrics.classes = len([node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)])
                metrics.imports = len(
    [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))])

                # Count type annotations
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        if node.returns is not None:
                            metrics.type_annotations += 1
                        else:
                            metrics.missing_type_annotations += 1

                        # Check arguments
                        for arg in node.args.args:
                            if arg.annotation is not None:
                                metrics.type_annotations += 1
                            else:
                                metrics.missing_type_annotations += 1

            except SyntaxError:
                metrics.syntax_errors += 1

            # Calculate complexity score (simple metric)
            metrics.complexity_score = (
                metrics.functions * 0.1 +
                metrics.classes * 0.2 +
                metrics.long_lines * 0.05 +
                metrics.missing_type_annotations * 0.1
            )

            return metrics

        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {e}")
            return CodeQualityMetrics(file_path=str(file_path), syntax_errors=1)

    def run_static_analysis(self) -> Dict[str, List[Dict]]:
        """Run static analysis using multiple approaches"""
        logger.info("Running static analysis...")

        issues = {}

        for file_path in self.python_files:
            file_issues = []

            # Analyze quality metrics
            metrics = self.analyze_file_quality(file_path)
            self.quality_metrics[str(file_path)] = metrics

            # Check for specific issues
            if metrics.syntax_errors > 0:
                file_issues.append({
                    'line': 1,
                    'code': 'E999',
                    'message': 'Syntax error detected',
                    'priority': 'CRITICAL'
                })

            if metrics.long_lines > 0:
                file_issues.append({
                    'line': 1,
                    'code': 'E501',
                    'message': f'{metrics.long_lines} lines too long',
                    'priority': 'MEDIUM'
                })

            if metrics.trailing_whitespace > 0:
                file_issues.append({
                    'line': 1,
                    'code': 'W291',
                    'message': f'{metrics.trailing_whitespace} lines with trailing whitespace',
                    'priority': 'LOW'
                })

            if metrics.blank_line_whitespace > 0:
                file_issues.append({
                    'line': 1,
                    'code': 'W293',
                    'message': f'{metrics.blank_line_whitespace} blank lines with whitespace',
                    'priority': 'LOW'
                })

            if metrics.missing_type_annotations > 0:
                file_issues.append({
                    'line': 1,
                    'code': 'F821',
                    'message': f'{metrics.missing_type_annotations} missing type annotations',
                    'priority': 'MEDIUM'
                })

            if file_issues:
                issues[str(file_path)] = file_issues

        self.issues = issues
        return issues

    def generate_compliance_report(self) -> ComplianceReport:
        """Generate comprehensive compliance report"""
        logger.info("Generating compliance report...")

        report = ComplianceReport(timestamp=datetime.now())

        # Calculate totals
        report.total_files = len(self.python_files)
        report.files_with_issues = len(self.issues)

        total_issues = 0
        high_priority = 0
        medium_priority = 0
        low_priority = 0

        for file_issues in self.issues.values():
            for issue in file_issues:
                total_issues += 1
                priority = issue['priority']
                if priority == 'CRITICAL' or priority == 'HIGH':
                    high_priority += 1
                elif priority == 'MEDIUM':
                    medium_priority += 1
                else:
                    low_priority += 1

        report.total_issues = total_issues
        report.high_priority_issues = high_priority
        report.medium_priority_issues = medium_priority
        report.low_priority_issues = low_priority

        # Calculate compliance score (0-100)
        if report.total_files > 0:
            files_without_issues = report.total_files - report.files_with_issues
            report.compliance_score = (files_without_issues / report.total_files) * 100

        report.metrics = self.quality_metrics

        return report

    def save_report(self, report: ComplianceReport, filename: str = "compliance_report.json") -> None:
        """Save compliance report to JSON file"""
        report_data = {
            'timestamp': report.timestamp.isoformat(),
            'summary': {
                'total_files': report.total_files,
                'total_issues': report.total_issues,
                'high_priority_issues': report.high_priority_issues,
                'medium_priority_issues': report.medium_priority_issues,
                'low_priority_issues': report.low_priority_issues,
                'files_with_issues': report.files_with_issues,
                'compliance_score': report.compliance_score
            },
            'files_with_issues': list(self.issues.keys()),
            'detailed_metrics': {
                path: {
                    'total_lines': metrics.total_lines,
                    'code_lines': metrics.code_lines,
                    'comment_lines': metrics.comment_lines,
                    'blank_lines': metrics.blank_lines,
                    'functions': metrics.functions,
                    'classes': metrics.classes,
                    'imports': metrics.imports,
                    'type_annotations': metrics.type_annotations,
                    'missing_type_annotations': metrics.missing_type_annotations,
                    'syntax_errors': metrics.syntax_errors,
                    'long_lines': metrics.long_lines,
                    'trailing_whitespace': metrics.trailing_whitespace,
                    'blank_line_whitespace': metrics.blank_line_whitespace,
                    'complexity_score': metrics.complexity_score
                }
                for path, metrics in report.metrics.items()
            }
        }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2)

        logger.info(f"📄 Compliance report saved to {filename}")

    def print_summary(self, report: ComplianceReport) -> None:
        """Print a human-readable summary"""
        print("\n" + "=" * 60)
        print("SCHWABOT COMPLIANCE MONITORING REPORT")
        print("=" * 60)
        print(f"Generated: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        print("📊 SUMMARY STATISTICS:")
        print(f"   Total Python files: {report.total_files}")
        print(f"   Files with issues: {report.files_with_issues}")
        print(f"   Total issues found: {report.total_issues}")
        print(f"   Compliance score: {report.compliance_score:.1f}%")
        print()

        print("🎯 ISSUE BREAKDOWN:")
        print(f"   🔴 HIGH/CRITICAL: {report.high_priority_issues}")
        print(f"   🟡 MEDIUM: {report.medium_priority_issues}")
        print(f"   🟢 LOW: {report.low_priority_issues}")
        print()

        if report.files_with_issues > 0:
            print("📁 FILES WITH ISSUES:")
            for file_path in sorted(self.issues.keys()):
                issues = self.issues[file_path]
                print(f"   {file_path}:")
                for issue in issues:
                    print(f"     - {issue['code']}: {issue['message']} ({issue['priority']})")
            print()

        # Quality metrics summary
        if report.metrics:
            total_functions = sum(m.functions for m in report.metrics.values())
            total_classes = sum(m.classes for m in report.metrics.values())
            total_type_annotations = sum(m.type_annotations for m in report.metrics.values())
            total_missing_annotations = sum(m.missing_type_annotations for m in report.metrics.values())

            print("📈 QUALITY METRICS:")
            print(f"   Total functions: {total_functions}")
            print(f"   Total classes: {total_classes}")
            print(f"   Type annotations: {total_type_annotations}")
            print(f"   Missing annotations: {total_missing_annotations}")

            if total_type_annotations + total_missing_annotations > 0:
                annotation_coverage = (
    total_type_annotations / (total_type_annotations + total_missing_annotations)) * 100
                print(f"   Type annotation coverage: {annotation_coverage:.1f}%")
            print()

        print("💡 RECOMMENDATIONS:")
        if report.high_priority_issues > 0:
            print("   🔴 Address HIGH/CRITICAL issues immediately")
        if report.medium_priority_issues > 0:
            print("   🟡 Add missing type annotations")
        if report.low_priority_issues > 0:
            print("   🟢 Run automated formatting tools")

        if report.compliance_score >= 95:
            print("   🎉 Excellent compliance! Keep up the good work!")
        elif report.compliance_score >= 80:
            print("   ✅ Good compliance! Minor improvements needed.")
        else:
            print("   ⚠️ Compliance needs improvement. Focus on high-priority issues.")

        print()
        print("🛠️  NEXT STEPS:")
        print("   1. Review and fix any HIGH priority issues")
        print("   2. Add type annotations using core.type_defs")
        print("   3. Run automated formatting tools")
        print("   4. Set up pre-commit hooks for continuous monitoring")
        print("=" * 60)


def main() -> None:
    """Main function"""
    logger.info("🚀 Schwabot Comprehensive Compliance Monitor")
    logger.info("=" * 50)

    monitor = ComplianceMonitor()

    # Step 1: Discover Python files
    logger.info("Step 1: Discovering Python files...")
    python_files = monitor.discover_python_files()

    if not python_files:
        logger.info("No Python files found in the project.")
        return

    # Step 2: Run static analysis
    logger.info("Step 2: Running static analysis...")
    issues = monitor.run_static_analysis()

    # Step 3: Generate report
    logger.info("Step 3: Generating compliance report...")
    report = monitor.generate_compliance_report()

    # Step 4: Save and display results
    monitor.save_report(report)
    monitor.print_summary(report)

    # Step 5: Provide actionable feedback
    if report.compliance_score >= 95:
        logger.info("🎉 Excellent code quality! Your codebase is highly compliant.")
    elif report.compliance_score >= 80:
        logger.info("✅ Good code quality! Minor improvements will bring you to excellence.")
    else:
        logger.info("⚠️ Code quality needs attention. Focus on high-priority issues first.")


if __name__ == "__main__":
    main()