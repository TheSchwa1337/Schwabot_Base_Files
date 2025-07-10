import os
import shutil
import subprocess
from datetime import datetime
from typing import List

#!/usr/bin/env python3
"""
Schwabot Flake8 Auto-Fix System
===============================

This script automatically fixes Flake8 formatting errors while preserving
mathematical structures and logging all changes for review.

Features:
- Uses autopep8 for automatic formatting
- Preserves mathematical structures and comments
- Logs all changes for review
- Creates backups before making changes
- Handles critical vs auto-fixable errors differently
"""


# Mathematical keywords that should be preserved
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
# Codebase directories
CODEBASE_DIRS = ["core", "core/math", "core/phase_engine", "core/recursive_engine"]

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


class Flake8AutoFixer:
    def __init__(self):
        self.changes_log = []
        self.backup_dir = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.fix_log_file = ()
            f"auto_fix_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        )

    def create_backup(): -> str:
        """Create a backup of a file before modifying it."""
        if not os.path.exists(self.backup_dir):
            os.makedirs(self.backup_dir)

        # Use os.path.join for cross-platform compatibility
        backup_filename = os.path.basename(filepath)
        backup_path = os.path.join(self.backup_dir, backup_filename)
        shutil.copy2(filepath, backup_path)
        return backup_path

    def is_math_relevant_file(): -> bool:
        """Check if a file contains mathematical content."""
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                return any(keyword in content for keyword in, MATH_PRESERVATION_KEYWORDS)
        except Exception:
            return False

    def run_autopep8(): -> bool:
        """Run autopep8 on a file to fix formatting issues."""
        try:
            # Create backup first
            backup_path = self.create_backup(filepath)

            # Run autopep8 with conservative settings
            cmd = []
                "python",
                "-m",
                "autopep8",
                "--in-place",
                "--aggressive",
                "--aggressive",
                "--max-line-length=999",  # More lenient for mathematical expressions
                "--ignore=E226,E302,E41",  # Ignore some aggressive fixes
                filepath.replace(os.sep, '/'),  # Ensure forward slashes for autopep8
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            if result.returncode == 0:
                # Check if file was actually modified
                with open(filepath, "r", encoding="utf-8") as f:
                    new_content = f.read()
                with open(backup_path, "r", encoding="utf-8") as f:
                    old_content = f.read()

                if new_content != old_content:
                    self.changes_log.append()
                        {}
                            "file": filepath,
                            "backup": backup_path,
                            "timestamp": datetime.now().isoformat(),
                            "math_relevant": self.is_math_relevant_file(filepath),
                            "method": "autopep8",
                        }
                    )
                    return True
                else:
                    # No changes made, remove backup
                    os.remove(backup_path)
                    return False
            else:
                print(f"Warning: autopep8 failed for {filepath}: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            print(f"Warning: autopep8 timed out for {filepath}")
            return False
        except Exception as e:
            print(f"Error running autopep8 on {filepath}: {e}")
            return False

    def fix_imports(): -> bool:
        """Fix import-related issues."""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                lines = f.readlines()

            lines.copy()
            modified = False

            # Fix multiple imports on one line (E401)
            for i, line in enumerate(lines):
                if ()
                    "," in line
                    and "import" in line
                    and not line.strip().startswith("#")
                ):
                    # Check if it's multiple imports'
                    if line.count(",") > 0 and "from" not in line:
                        # Split multiple imports
                        parts = line.split("import")
                        if len(parts) == 2:
                            module = parts[0].strip()
                            imports = parts[1].strip()
                            if "," in imports:
                                import_list = []
                                    imp.strip() for imp in imports.split(",")
                                ]
                                new_lines = []
                                    f"{module} import {imp}\n" for imp in import_list
                                ]
                                lines[i: i + 1] = new_lines
                                modified = True

            if modified:
                # Create backup
                backup_path = self.create_backup(filepath)

                # Write back
                with open(filepath, "w", encoding="utf-8") as f:
                    f.writelines(lines)

                self.changes_log.append()
                    {}
                        "file": filepath,
                        "backup": backup_path,
                        "timestamp": datetime.now().isoformat(),
                        "math_relevant": self.is_math_relevant_file(filepath),
                        "method": "import_fix",
                    }
                )
                return True

            return False

        except Exception as e:
            print(f"Error fixing imports in {filepath}: {e}")
            return False

    def fix_trailing_whitespace(): -> bool:
        """Fix trailing whitespace issues."""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                lines = f.readlines()

            lines.copy()
            modified = False

            # Fix trailing whitespace
            for i, line in enumerate(lines):
                if line.rstrip() != line:
                    lines[i] = line.rstrip() + "\n"
                    modified = True

            # Ensure file ends with newline
            if lines and not lines[-1].endswith("\n"):
                lines[-1] = lines[-1] + "\n"
                modified = True

            if modified:
                # Create backup
                backup_path = self.create_backup(filepath)

                # Write back
                with open(filepath, "w", encoding="utf-8") as f:
                    f.writelines(lines)

                self.changes_log.append()
                    {}
                        "file": filepath,
                        "backup": backup_path,
                        "timestamp": datetime.now().isoformat(),
                        "math_relevant": self.is_math_relevant_file(filepath),
                        "method": "whitespace_fix",
                    }
                )
                return True

            return False

        except Exception as e:
            print(f"Error fixing whitespace in {filepath}: {e}")
            return False

    def get_python_files(): -> List[str]:
        """Get all Python files in the codebase."""
        python_files = []
        for base_dir in CODEBASE_DIRS:
            if os.path.exists(base_dir):
                for root, _, files in os.walk(base_dir):
                    for file in files:
                        if file.endswith(".py"):
                            python_files.append(os.path.join(root, file))
        return python_files

    def generate_fix_report(): -> str:
        """Generate a report of all changes made."""
        report = []
        report.append("# Schwabot Flake8 Auto-Fix Report\n")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        if not self.changes_log:
            report.append("## ✅ No Changes Made")
            report.append()
                "All files were already properly formatted or no auto-fixable issues found."
            )
            return "\n".join(report)

        report.append("## 📊 Summary")
        report.append(f"- Files modified: {len(self.changes_log)}")
        report.append()
            f"- Math-relevant files: {sum(1 for change in self.changes_log if change['math_relevant'])}"
        )
        report.append(f"- Backup directory: `{self.backup_dir}`")
        report.append("")

        report.append("## 📝 Changes Made")
        for change in self.changes_log:
            math_flag = "🔬" if change["math_relevant"] else ""
            report.append(f"### {change['file']} {math_flag}")
            report.append(f"- Method: {change['method']}")
            report.append(f"- Backup: {change['backup']}")
            report.append(f"- Timestamp: {change['timestamp']}")
            report.append("")

        report.append("## 🔄 Next Steps")
        report.append()
            "1. **Review the changes** - Check that mathematical structures were preserved"
        )
        report.append("2. **Run tests** - Ensure functionality is maintained")
        report.append("3. **Run Flake8 again** - Verify errors were fixed")
        report.append("4. **Clean up backups** - Remove backup directory if satisfied")

        return "\n".join(report)

    def save_fix_log(self):
        """Save the fix log to a file."""
        report = self.generate_fix_report()
        with open(self.fix_log_file, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Auto-fix report saved to {self.fix_log_file}")

    def auto_fix(self):
        """Run the complete auto-fix process."""
        print("Starting Flake8 auto-fix process...")

        python_files = self.get_python_files()
        print(f"Found {len(python_files)} Python files")

        fixed_count = 0
        math_relevant_fixed = 0

        for filepath in python_files:
            print(f"Processing: {filepath}")

            is_math = self.is_math_relevant_file(filepath)
            if is_math:
                print("  [MATH] Math-relevant file detected")

            # Apply fixes
            fixed = False

            # Fix imports
            if self.fix_imports(filepath):
                fixed = True

            # Fix whitespace
            if self.fix_trailing_whitespace(filepath):
                fixed = True

            # Run autopep8 (most, comprehensive)
            if self.run_autopep8(filepath):
                fixed = True

            if fixed:
                fixed_count += 1
                if is_math:
                    math_relevant_fixed += 1
                print("  [OK] Fixed")
            else:
                print("  [SKIP] No changes needed")

        self.save_fix_log()

        print("\nAuto-fix complete:")
        print(f"- Files processed: {len(python_files)}")
        print(f"- Files fixed: {fixed_count}")
        print(f"- Math-relevant files fixed: {math_relevant_fixed}")
        print(f"- Backup directory: {self.backup_dir}")
        print(f"- Report: {self.fix_log_file}")


def main():
    fixer = Flake8AutoFixer()
    fixer.auto_fix()


if __name__ == "__main__":
    main()
