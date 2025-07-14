import os
import subprocess
import sys
from collections import defaultdict
from typing import List, Tuple

#!/usr/bin/env python3
"""
Schwabot Flake8 Error Analyzer & Auto-Fix System
================================================

This script analyzes Flake8 errors across the Schwabot codebase, categorizes them,
and provides auto-fix suggestions while preserving mathematical structures.

Error Categories:
- E999: Syntax errors (critical - must, fix)
- F821: Undefined names (critical - must, fix)
- E501: Line too long (auto-fixable)
- E302: Expected 2 blank lines (auto-fixable)
- E303: Too many blank lines (auto-fixable)
- E305: Expected 2 blank lines after class (auto-fixable)
- E225: Missing whitespace around operator (auto-fixable)
- E226: Missing whitespace around arithmetic operator (auto-fixable)
- E231: Missing whitespace after ',' (auto-fixable)
- E241: Multiple spaces after ',' (auto-fixable)
- E251: Unexpected spaces around keyword (auto-fixable)
- E261: At least two spaces before inline comment (auto-fixable)
- E262: Inline comment should start with '# ' (auto-fixable)
- E265: Block comment should start with '# ' (auto-fixable)
- E266: Too many leading '#' for block comment (auto-fixable)
- E401: Multiple imports on one line (auto-fixable)
- E402: Module level import not at top (auto-fixable)
- E701: Multiple statements on one line (auto-fixable)
- E702: Multiple statements on one line (semicolon) (auto-fixable)
- E703: Statement ends with a semicolon (auto-fixable)
- E711: Comparison to None should be 'if cond is None:' (auto-fixable)
- E712: Comparison to True should be 'if cond is True:' (auto-fixable)
- E713: Test for membership should be 'not in' (auto-fixable)
- E714: Test for object identity should be 'is not' (auto-fixable)
- E721: Do not compare types, use 'isinstance()' (auto-fixable)
- E722: Do not use bare except (auto-fixable)
- E731: Do not assign a lambda expression, use a def (auto-fixable)
- E741: Do not use variables named 'l', 'O', or 'I' (auto-fixable)
- E742: Do not define classes named 'l', 'O', or 'I' (auto-fixable)
- E743: Do not define functions named 'l', 'O', or 'I' (auto-fixable)
- W291: Trailing whitespace (auto-fixable)
- W292: No newline at end of file (auto-fixable)
- W293: Blank line contains whitespace (auto-fixable)
- W391: Blank line at end of file (auto-fixable)
- W503: Line break before binary operator (auto-fixable)
- W504: Line break after binary operator (auto-fixable)
- W505: doc line too long (auto-fixable)
- W601: .has_key() is deprecated, use 'in' (auto-fixable)
- W602: Deprecated form of raising exception (auto-fixable)
- W603: '<>' is deprecated, use '!=' (auto-fixable)
- W604: backticks are deprecated, use 'repr()' (auto-fixable)
- W605: invalid escape sequence (auto-fixable)
- W606: 'async' and 'await' are reserved keywords (auto-fixable)
"""


# Mathematical keywords that should be preserved during fixes
MATH_PRESERVATION_KEYWORDS = {}
    "numpy",
    "scipy",
    "math",
    "mpmath",
    "sympy",
    "numba",
    "tensor",
    "lattice",
    "phase",
    "profit",
    "entropy",
    "glyph",
    "hash",
    "volume",
    "trade",
    "signal",
    "router",
    "engine",
    "recursive",
    "vector",
    "matrix",
    "sha256",
    "ECC",
    "NCCO",
    "fractal",
    "cycle",
    "oscillator",
    "backtrace",
    "resonance",
    "projection",
    "delta",
    "lambda",
    "mu",
    "sigma",
    "alpha",
    "beta",
    "gamma",
    "zeta",
    "theta",
    "pi",
    "phi",
    "psi",
    "rho",
    "Fourier",
    "Kalman",
    "Markov",
    "stochastic",
    "deterministic",
    "statistic",
    "probability",
    "distribution",
    "mean",
    "variance",
    "covariance",
    "correlation",
    "regression",
    "gradient",
    "derivative",
    "integral",
    "logistic",
    "exponential",
    "sigmoid",
    "activation",
    "neural",
    "feedback",
    "harmonic",
    "volatility",
    "liquidity",
    "momentum",
    "backprop",
    "sha",
    "RDE",
    "RITL",
    "RITTLE",
}
# Auto-fixable error codes
AUTO_FIXABLE_CODES = {}
    "E501",
    "E302",
    "E303",
    "E305",
    "E225",
    "E226",
    "E231",
    "E241",
    "E251",
    "E261",
    "E262",
    "E265",
    "E266",
    "E401",
    "E402",
    "E701",
    "E702",
    "E703",
    "E711",
    "E712",
    "E713",
    "E714",
    "E721",
    "E722",
    "E731",
    "E741",
    "E742",
    "E743",
    "W291",
    "W292",
    "W293",
    "W391",
    "W503",
    "W504",
    "W505",
    "W601",
    "W602",
    "W603",
    "W604",
    "W605",
    "W606",
}
# Critical error codes that must be fixed manually
CRITICAL_CODES = {"E999", "F821", "F822", "F823", "F831", "F841", "F901"}

# Codebase directories to scan
CODEBASE_DIRS = ["core", "core/math", "core/phase_engine", "core/recursive_engine"]


class Flake8Analyzer:
    def __init__(self):
        self.errors = defaultdict(list)
        self.math_relevant_files = set()
        self.auto_fixable_count = 0
        self.critical_count = 0
        self.total_count = 0

    def run_flake8(): -> List[str]:
        """Run Flake8 on the codebase and return error lines."""
        try:
            # Run flake8 with specific error codes
            cmd = []
                sys.executable,
                "-m",
                "flake8",
                "--isolated",  # Don't load existing configs'
                "--select=" + ",".join(AUTO_FIXABLE_CODES.union(CRITICAL_CODES)),
                "--ignore=E501",  # Ignore line length for now
                "--max-line-length=999",  # Allow very long lines
                *CODEBASE_DIRS,
            ]
            process = subprocess.Popen()
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            stdout, stderr = process.communicate()

            if process.returncode != 0 and "E999" in stderr:
                print(f"\n🚨 Critical Flake8 Error (E999 - Syntax, Error) Detected!")
                print("Please review the following syntax errors:")
                print(stderr)
                # It's crucial to return an empty list or raise an error'
                # if syntax errors prevent meaningful analysis.
                return []

            # Parse output
            lines = stdout.strip().split("\n")
            return []
                line for line in lines if ":" in line and not line.startswith("---")
            ]

        except subprocess.TimeoutExpired:
            print("Flake8 analysis timed out")
            return []
        except Exception as e:
            print(f"Error running Flake8: {e}")
            return []

    def parse_error_line(): -> Tuple[str, int, int, str, str]:
        """Parse a Flake8 error line."""
        try:
            # Format: path:line:column:code:message
            parts = line.split(":", 4)
            if len(parts) >= 5:
                filepath = parts[0]
                line_num = int(parts[1])
                column = int(parts[2])
                code = parts[3]
                message = parts[4]
                return filepath, line_num, column, code, message
        except (ValueError, IndexError):
            pass
        return None, 0, 0, "", ""

    def is_math_relevant_file(): -> bool:
        """Check if a file contains mathematical content."""
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                return any(keyword in content for keyword in, MATH_PRESERVATION_KEYWORDS)
        except Exception:
            return False

    def categorize_errors(self, error_lines: List[str]):
        """Categorize errors by type and file."""
        for line in error_lines:
            filepath, line_num, column, code, message = self.parse_error_line(line)
            if filepath:
                error_info = {}
                    "line": line_num,
                    "column": column,
                    "code": code,
                    "message": message,
                    "auto_fixable": code in AUTO_FIXABLE_CODES,
                    "critical": code in CRITICAL_CODES,
                }
                self.errors[filepath].append(error_info)
                self.total_count += 1

                if error_info["auto_fixable"]:
                    self.auto_fixable_count += 1
                if error_info["critical"]:
                    self.critical_count += 1

                # Mark math-relevant files
                if self.is_math_relevant_file(filepath):
                    self.math_relevant_files.add(filepath)

    def generate_report(): -> str:
        """Generate a comprehensive Flake8 error report."""
        report = []
        report.append("# Schwabot Flake8 Error Analysis Report\n")

        # Summary
        report.append("## Summary")
        report.append(f"- Total Errors: {self.total_count}")
        report.append(f"- Auto-fixable: {self.auto_fixable_count}")
        report.append(f"- Critical: {self.critical_count}")
        report.append(f"- Math-relevant files: {len(self.math_relevant_files)}")
        report.append("")

        # Critical errors first
        if self.critical_count > 0:
            report.append("## 🚨 Critical Errors (Must, Fix)")
            for filepath, errors in self.errors.items():
                critical_errors = [e for e in errors if e["critical"]]
                if critical_errors:
                    report.append(f"### {filepath}")
                    for error in critical_errors:
                        math_flag = "🔬" if filepath in self.math_relevant_files else ""
                        report.append()
                            f"- Line {error['line']}: {error['code']} - {error['message']} {math_flag}"
                        )
                    report.append("")

        # Auto-fixable errors
        if self.auto_fixable_count > 0:
            report.append("## 🔧 Auto-fixable Errors")
            for filepath, errors in self.errors.items():
                auto_errors = [e for e in errors if e["auto_fixable"]]
                if auto_errors:
                    report.append(f"### {filepath}")
                    for error in auto_errors:
                        math_flag = "🔬" if filepath in self.math_relevant_files else ""
                        report.append()
                            f"- Line {error['line']}: {error['code']} - {error['message']} {math_flag}"
                        )
                    report.append("")

        # Other errors
        other_errors = []
        for filepath, errors in self.errors.items():
            other = [e for e in errors if not e["auto_fixable"] and not e["critical"]]
            if other:
                other_errors.append((filepath, other))

        if other_errors:
            report.append("## ⚠️ Other Errors")
            for filepath, errors in other_errors:
                report.append(f"### {filepath}")
                for error in errors:
                    math_flag = "🔬" if filepath in self.math_relevant_files else ""
                    report.append()
                        f"- Line {error['line']}: {error['code']} - {error['message']} {math_flag}"
                    )
                report.append("")

        # Recommendations
        report.append("## 📋 Recommendations")
        if self.critical_count > 0:
            report.append()
                "1. **Fix critical errors first** - These prevent code from running"
            )
        if self.auto_fixable_count > 0:
            report.append()
                "2. **Run auto-fix** - Use `python auto_fix_flake8.py` to fix formatting issues"
            )
        if len(self.math_relevant_files) > 0:
            report.append()
                "3. **Preserve mathematical structures** - Files marked with 🔬 contain mathematical logic"
            )
        report.append()
            "4. **Test after fixes** - Run your test suite after making changes"
        )

        return "\n".join(report)

    def save_report(self, filename: str = "flake8_analysis_report.md"):
        """Save the analysis report to a file."""
        report = self.generate_report()
        with open(filename, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Flake8 analysis report saved to {filename}")

    def analyze(self):
        """Run the complete Flake8 analysis."""
        print("Running Flake8 analysis...")
        error_lines = self.run_flake8()

        if not error_lines:
            print("[OK] No Flake8 errors found!")
            return

        print(f"Found {len(error_lines)} error lines")
        self.categorize_errors(error_lines)
        self.save_report()

        print("\nAnalysis complete:")
        print(f"- Total errors: {self.total_count}")
        print(f"- Auto-fixable: {self.auto_fixable_count}")
        print(f"- Critical: {self.critical_count}")
        print(f"- Math-relevant files: {len(self.math_relevant_files)}")


def main():
    analyzer = Flake8Analyzer()
    analyzer.analyze()


if __name__ == "__main__":
    main()
