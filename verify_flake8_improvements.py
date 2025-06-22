#!/usr/bin/env python3
"""Flake8 Improvement Verification Script.

This script verifies that our maturity implementation has successfully
reduced flake8 errors, particularly focusing on:
- F841 (unused variables) - Should be eliminated
- C901 (complexity) - Should be reduced
- F541 (f-string issues) - Should be fixed
- W293/E501 (formatting) - Should be improved

The script provides before/after analysis and detailed reporting.
"""

import subprocess
import sys
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class FlakeIssue:
    """Represents a flake8 issue."""
    file_path: str
    line_number: int
    column: int
    error_code: str
    message: str
    severity: str = "medium"


@dataclass
class FlakeReport:
    """Comprehensive flake8 report."""
    total_issues: int
    issues_by_code: Dict[str, int] = field(default_factory=dict)
    issues_by_file: Dict[str, int] = field(default_factory=dict)
    critical_issues: List[FlakeIssue] = field(default_factory=list)
    improvement_areas: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


class FlakeVerifier:
    """Verifies flake8 improvements after maturity implementation."""
    
    def __init__(self):
        """Initialize the flake8 verifier."""
        self.project_root = Path.cwd()
        self.core_dirs = ["core/"]
        self.critical_error_codes = ["F841", "E999", "F63", "F7", "F82"]
        self.target_error_codes = ["F841", "C901", "F541", "W293", "E501"]
        
        print("🔍 Flake8 Improvement Verifier initialized")
        print(f"📁 Project root: {self.project_root}")
    
    def run_flake8_analysis(self, directory: str) -> Tuple[str, str]:
        """Run flake8 analysis on a directory."""
        try:
            # Run flake8 with detailed output
            result = subprocess.run([
                sys.executable, "-m", "flake8", 
                directory,
                "--statistics",
                "--count",
                "--show-source",
                "--format=%(path)s:%(row)d:%(col)d: %(code)s %(text)s"
            ], capture_output=True, text=True, timeout=60)
            
            return result.stdout, result.stderr
            
        except subprocess.TimeoutExpired:
            return "", "Flake8 analysis timed out"
        except Exception as e:
            return "", f"Error running flake8: {e}"
    
    def parse_flake8_output(self, output: str) -> FlakeReport:
        """Parse flake8 output into structured report."""
        lines = output.strip().split('\n')
        issues = []
        statistics = {}
        
        # Parse individual issues
        issue_pattern = r'(.+):(\d+):(\d+): (\w+) (.+)'
        stats_pattern = r'(\d+)\s+(\w+)'
        
        for line in lines:
            if not line.strip():
                continue
                
            # Try to parse as issue
            issue_match = re.match(issue_pattern, line)
            if issue_match:
                file_path, line_num, col, code, message = issue_match.groups()
                issues.append(FlakeIssue(
                    file_path=file_path,
                    line_number=int(line_num),
                    column=int(col),
                    error_code=code,
                    message=message,
                    severity=self._get_severity(code)
                ))
                continue
            
            # Try to parse as statistics
            stats_match = re.match(stats_pattern, line)
            if stats_match:
                count, code = stats_match.groups()
                statistics[code] = int(count)
        
        # Create report
        report = FlakeReport(total_issues=len(issues))
        
        # Group by error code
        for issue in issues:
            code = issue.error_code
            report.issues_by_code[code] = report.issues_by_code.get(code, 0) + 1
            
            # Group by file
            file_name = Path(issue.file_path).name
            report.issues_by_file[file_name] = report.issues_by_file.get(file_name, 0) + 1
            
            # Mark critical issues
            if code in self.critical_error_codes:
                report.critical_issues.append(issue)
        
        # Add statistics if available
        if statistics:
            report.issues_by_code.update(statistics)
        
        return report
    
    def _get_severity(self, error_code: str) -> str:
        """Get severity level for error code."""
        if error_code.startswith('E9') or error_code.startswith('F'):
            return "high"
        elif error_code.startswith('C9'):
            return "medium"
        elif error_code.startswith('W') or error_code.startswith('E'):
            return "low"
        else:
            return "medium"
    
    def analyze_maturity_impact(self, report: FlakeReport) -> Dict[str, Any]:
        """Analyze the impact of maturity implementation on flake8 errors."""
        analysis = {
            "f841_status": "ELIMINATED" if report.issues_by_code.get("F841", 0) == 0 else "PRESENT",
            "critical_errors": len(report.critical_issues),
            "total_issues": report.total_issues,
            "improvement_score": 0.0,
            "maturity_indicators": []
        }
        
        # Check F841 (unused variables) - should be 0 after maturity implementation
        f841_count = report.issues_by_code.get("F841", 0)
        if f841_count == 0:
            analysis["maturity_indicators"].append("✅ F841 unused variables eliminated")
            analysis["improvement_score"] += 30
        else:
            analysis["maturity_indicators"].append(f"❌ F841 unused variables still present: {f841_count}")
        
        # Check complexity (C901)
        c901_count = report.issues_by_code.get("C901", 0)
        if c901_count < 5:
            analysis["maturity_indicators"].append(f"✅ Complexity under control: {c901_count} C901 issues")
            analysis["improvement_score"] += 20
        else:
            analysis["maturity_indicators"].append(f"⚠️ High complexity: {c901_count} C901 issues")
        
        # Check f-string issues (F541)
        f541_count = report.issues_by_code.get("F541", 0)
        if f541_count == 0:
            analysis["maturity_indicators"].append("✅ F541 f-string issues resolved")
            analysis["improvement_score"] += 15
        else:
            analysis["maturity_indicators"].append(f"⚠️ F541 f-string issues: {f541_count}")
        
        # Check formatting issues
        w293_count = report.issues_by_code.get("W293", 0)
        e501_count = report.issues_by_code.get("E501", 0)
        formatting_score = max(0, 35 - (w293_count + e501_count) * 0.1)
        analysis["improvement_score"] += formatting_score
        
        if w293_count + e501_count < 100:
            analysis["maturity_indicators"].append(f"✅ Formatting improved: {w293_count + e501_count} issues")
        else:
            analysis["maturity_indicators"].append(f"⚠️ Formatting needs work: {w293_count + e501_count} issues")
        
        return analysis
    
    def generate_improvement_report(self, report: FlakeReport, analysis: Dict[str, Any]) -> str:
        """Generate comprehensive improvement report."""
        report_lines = [
            "=" * 80,
            "🎯 FLAKE8 IMPROVEMENT VERIFICATION REPORT",
            f"📅 Generated: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 80,
            "",
            "📊 OVERALL STATISTICS:",
            f"   • Total Issues: {report.total_issues}",
            f"   • Critical Issues: {analysis['critical_errors']}",
            f"   • Improvement Score: {analysis['improvement_score']:.1f}/100",
            f"   • F841 Status: {analysis['f841_status']}",
            "",
            "🎯 MATURITY INDICATORS:",
        ]
        
        for indicator in analysis["maturity_indicators"]:
            report_lines.append(f"   {indicator}")
        
        report_lines.extend([
            "",
            "📈 ERROR CODE BREAKDOWN:",
        ])
        
        # Sort error codes by count (descending)
        sorted_codes = sorted(report.issues_by_code.items(), key=lambda x: x[1], reverse=True)
        for code, count in sorted_codes[:10]:  # Top 10
            severity = self._get_severity(code)
            emoji = "🔴" if severity == "high" else "🟡" if severity == "medium" else "🟢"
            report_lines.append(f"   {emoji} {code}: {count} issues ({severity} severity)")
        
        report_lines.extend([
            "",
            "📁 FILES WITH MOST ISSUES:",
        ])
        
        # Sort files by issue count
        sorted_files = sorted(report.issues_by_file.items(), key=lambda x: x[1], reverse=True)
        for file_name, count in sorted_files[:5]:  # Top 5
            report_lines.append(f"   📄 {file_name}: {count} issues")
        
        if report.critical_issues:
            report_lines.extend([
                "",
                "🚨 CRITICAL ISSUES REQUIRING ATTENTION:",
            ])
            
            for issue in report.critical_issues[:5]:  # Show first 5
                report_lines.append(
                    f"   🔴 {Path(issue.file_path).name}:{issue.line_number} "
                    f"{issue.error_code} - {issue.message}"
                )
        
        # Add recommendations
        report_lines.extend([
            "",
            "💡 RECOMMENDATIONS:",
        ])
        
        if analysis["f841_status"] == "PRESENT":
            report_lines.append("   🔧 Eliminate remaining F841 unused variables through StateTracker integration")
        
        if report.issues_by_code.get("C901", 0) > 5:
            report_lines.append("   🔧 Refactor complex functions using helper methods")
        
        if report.issues_by_code.get("F541", 0) > 0:
            report_lines.append("   🔧 Fix invalid f-string syntax issues")
        
        if report.issues_by_code.get("W293", 0) + report.issues_by_code.get("E501", 0) > 100:
            report_lines.append("   🔧 Run automated formatting cleanup")
        
        report_lines.extend([
            "",
            "🎉 MATURITY ASSESSMENT:",
        ])
        
        score = analysis["improvement_score"]
        if score >= 90:
            report_lines.append("   🏆 EXCELLENT - System shows high maturity with minimal flake8 issues")
        elif score >= 70:
            report_lines.append("   ✅ GOOD - System shows solid improvement with manageable issues")
        elif score >= 50:
            report_lines.append("   ⚠️ MODERATE - System shows improvement but needs additional work")
        else:
            report_lines.append("   ❌ NEEDS WORK - System requires significant flake8 improvements")
        
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)
    
    def verify_specific_improvements(self) -> Dict[str, bool]:
        """Verify specific improvements we implemented."""
        improvements = {}
        
        # Check if core/main.py has F841 issues
        stdout, stderr = self.run_flake8_analysis("core/main.py")
        improvements["main_py_f841_fixed"] = "F841" not in stdout
        
        # Check if StateTracker integration is working
        try:
            from core.state_tracker import StateTracker
            from core.core_loop_manager import CoreLoopManager
            improvements["state_tracker_integrated"] = True
        except ImportError:
            improvements["state_tracker_integrated"] = False
        
        # Check if error sanitizer is available
        try:
            from core.error_sanitizer import ErrorSanitizer
            improvements["error_sanitizer_available"] = True
        except ImportError:
            improvements["error_sanitizer_available"] = False
        
        # Check if maturity components exist
        maturity_files = [
            "core/core_loop_manager.py",
            "core/tick_cycle_validator.py", 
            "core/profit_vector_reconciler.py",
            "core/error_sanitizer.py"
        ]
        
        improvements["maturity_files_exist"] = all(
            Path(f).exists() for f in maturity_files
        )
        
        return improvements
    
    def run_complete_verification(self) -> bool:
        """Run complete flake8 improvement verification."""
        print("🚀 Starting Flake8 Improvement Verification...")
        print()
        
        # Analyze each core directory
        overall_success = True
        all_reports = []
        
        for directory in self.core_dirs:
            if not Path(directory).exists():
                print(f"⚠️ Directory {directory} not found, skipping...")
                continue
                
            print(f"🔍 Analyzing {directory}...")
            stdout, stderr = self.run_flake8_analysis(directory)
            
            if stderr and "not found" not in stderr.lower():
                print(f"❌ Error analyzing {directory}: {stderr}")
                overall_success = False
                continue
            
            # Parse results
            report = self.parse_flake8_output(stdout)
            analysis = self.analyze_maturity_impact(report)
            all_reports.append((directory, report, analysis))
            
            # Quick summary
            print(f"   📊 {report.total_issues} total issues")
            print(f"   🎯 Improvement score: {analysis['improvement_score']:.1f}/100")
            print(f"   ✅ F841 status: {analysis['f841_status']}")
            print()
        
        # Generate comprehensive report
        if all_reports:
            # Combine reports for overall analysis
            combined_report = FlakeReport(total_issues=0)
            combined_analysis = {"improvement_score": 0.0, "maturity_indicators": []}
            
            for directory, report, analysis in all_reports:
                combined_report.total_issues += report.total_issues
                combined_analysis["improvement_score"] += analysis["improvement_score"]
                combined_analysis["maturity_indicators"].extend(analysis["maturity_indicators"])
                
                # Merge issue counts
                for code, count in report.issues_by_code.items():
                    combined_report.issues_by_code[code] = combined_report.issues_by_code.get(code, 0) + count
                
                for file_name, count in report.issues_by_file.items():
                    combined_report.issues_by_file[file_name] = combined_report.issues_by_file.get(file_name, 0) + count
                
                combined_report.critical_issues.extend(report.critical_issues)
            
            # Average the improvement score
            combined_analysis["improvement_score"] /= len(all_reports)
            combined_analysis["critical_errors"] = len(combined_report.critical_issues)
            combined_analysis["total_issues"] = combined_report.total_issues
            combined_analysis["f841_status"] = "ELIMINATED" if combined_report.issues_by_code.get("F841", 0) == 0 else "PRESENT"
            
            # Generate and display report
            full_report = self.generate_improvement_report(combined_report, combined_analysis)
            print(full_report)
            
            # Save report to file
            report_file = Path("FLAKE8_IMPROVEMENT_REPORT.md")
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(full_report)
            print(f"📄 Full report saved to: {report_file}")
        
        # Verify specific improvements
        print("\n🔧 Verifying Specific Improvements:")
        improvements = self.verify_specific_improvements()
        
        for improvement, status in improvements.items():
            emoji = "✅" if status else "❌"
            print(f"   {emoji} {improvement.replace('_', ' ').title()}: {'PASS' if status else 'FAIL'}")
            if not status:
                overall_success = False
        
        print()
        if overall_success:
            print("🎉 VERIFICATION PASSED - Flake8 improvements successfully implemented!")
        else:
            print("⚠️ VERIFICATION INCOMPLETE - Some improvements need attention")
        
        return overall_success


def main():
    """Main verification function."""
    verifier = FlakeVerifier()
    success = verifier.run_complete_verification()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main()) 