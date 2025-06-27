from dual_unicore_handler import DualUnicoreHandler
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
import ast
import json
import os
import re


# Initialize Unicode handler
unicore = DualUnicoreHandler()

# -*- coding: utf - 8 -*-
"""
"""
"""
"""
"""
Missing Imports Analysis Script
==============================

This script analyzes all import statements in the codebase to identify:
1. Missing files that are being imported
2. Inconsistent import patterns
3. Files that need to be created or corrected
4. Import errors that are causing runtime failures
"""
"""
"""
"""
"""


class ImportAnalyzer:

    """Analyzes import statements across the codebase."""


"""
"""
"""
"""

    def __init__(self, root_dir: str = "."):

        self.root_dir = Path(root_dir)
        self.missing_files: Set[str] = set()
        self.existing_files: Set[str] = set()
        self.import_patterns: Dict[str, List[str]] = {}
        self.errors: List[str] = []

    def find_all_python_files(self) -> List[Path]:
        """Find all Python files in the codebase."""
"""
"""
"""
"""
        python_files = []
        for root, dirs, files in os.walk(self.root_dir):
# Skip certain directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules', 'venv']]

            for file in files:
                if file.endswith('.py'):
                    python_files.append(Path(root) / file)
        return python_files

    def extract_imports_from_file(self, file_path: Path) -> List[Tuple[str, str, int]]:

        """Extract all import statements from a Python file."""
"""
"""
"""
"""
        imports = []
        try:
            with open(file_path, 'r', encoding='utf - 8', errors='ignore') as f:
                content = f.read()

# Parse the file
            try:
                tree = ast.parse(content)
            except SyntaxError:
# Skip files with syntax errors
                return imports

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append((alias.name, 'import', node.lineno))
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    for alias in node.names:
                        imports.append((f"{module}.{alias.name}", 'from', node.lineno))

        except Exception as e:
            self.errors.append(f"Error reading {file_path}: {e}")

        return imports

    def check_file_exists(self, import_path: str) -> bool:

        """Check if an imported file / module exists."""
"""
"""
"""
"""
# Handle different import patterns
        if import_path.startswith('.'):
# Relative import - skip for now as we need context
            return True

# Convert import path to file path
        parts = import_path.split('.')

# Check for common patterns
        if parts[0] in ['core', 'utils', 'tests', 'config', 'data']:
# This is a local module
            if parts[0] == 'core':
# Check in core directory
                file_path = self.root_dir / 'core' / f"{parts[1]}.py"
                return file_path.exists()
            elif parts[0] == 'utils':
# Check in utils directory
                file_path = self.root_dir / 'utils' / f"{parts[1]}.py"
                return file_path.exists()
            else:
# Check in other directories
                file_path = self.root_dir / parts[0] / f"{parts[1]}.py"
                return file_path.exists()

# Standard library or third - party imports
        return True

    def analyze_imports(self) -> Dict[str, Any]:

        """Analyze all imports in the codebase."""
"""
"""
"""
"""
        python_files = self.find_all_python_files()

        all_imports = {}
        missing_imports = {}

        for file_path in python_files:
            relative_path = file_path.relative_to(self.root_dir)
            imports = self.extract_imports_from_file(file_path)

            file_imports = []
            file_missing = []

            for import_path, import_type, line_num in imports:
                file_imports.append({
                    'path': import_path,
                    'type': import_type,
                    'line': line_num
                })

                if not self.check_file_exists(import_path):
                    file_missing.append({
                        'path': import_path,
                        'type': import_type,
                        'line': line_num
                    })
                    self.missing_files.add(import_path)

            if file_imports:
                all_imports[str(relative_path)] = file_imports

            if file_missing:
                missing_imports[str(relative_path)] = file_missing

        return {
            'all_imports': all_imports,
            'missing_imports': missing_imports,
            'missing_files': list(self.missing_files),
            'errors': self.errors,
            'total_files_analyzed': len(python_files)
        }

    def generate_report(self) -> str:

        """Generate a comprehensive report of missing imports."""
"""
"""
"""
"""
        analysis = self.analyze_imports()

        report = []
        report.append("=" * 80)
        report.append("MISSING IMPORTS ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")

# Summary
        report.append(f"Total files analyzed: {analysis['total_files_analyzed']}")
        report.append(f"Files with missing imports: {len(analysis['missing_imports'])}")
        report.append(f"Total missing imports: {len(analysis['missing_files'])}")
        report.append("")

# Missing files summary
        if analysis['missing_files']:
            report.append("MISSING FILES:")
            report.append("-" * 40)
            for missing_file in sorted(analysis['missing_files']):
                report.append(f"  - {missing_file}")
            report.append("")

# Files with missing imports
        if analysis['missing_imports']:
            report.append("FILES WITH MISSING IMPORTS:")
            report.append("-" * 40)
            for file_path, missing_list in analysis['missing_imports'].items():
                report.append(f"\\n{file_path}:")
                for missing in missing_list:
                    report.append(f"  Line {missing['line']}: {missing['type']} {missing['path']}")
            report.append("")

# Errors
        if analysis['errors']:
            report.append("ERRORS ENCOUNTERED:")
            report.append("-" * 40)
            for error in analysis['errors']:
                report.append(f"  - {error}")
            report.append("")

# Recommendations
        report.append("RECOMMENDATIONS:")
        report.append("-" * 40)
        if analysis['missing_files']:
            report.append("1. Create the following missing files:")
            for missing_file in sorted(analysis['missing_files']):
                if missing_file.startswith('core.'):
                    module_name = missing_file.replace('core.', '')
                    report.append(f"   - core/{module_name}.py")
                elif missing_file.startswith('utils.'):
                    module_name = missing_file.replace('utils.', '')
                    report.append(f"   - utils/{module_name}.py")
                else:
                    report.append(f"   - {missing_file}.py")
            report.append("")

        report.append("2. Check import consistency:")
        report.append("   - Ensure relative imports use '.' prefix")
        report.append("   - Ensure absolute imports reference correct paths")
        report.append("   - Remove unused imports")
        report.append("")

        report.append("3. Fix import order:")
        report.append("   - Standard library imports first")
        report.append("   - Third - party imports second")
        report.append("   - Local imports last")

        return "\n".join(report)


def main():

    """Main analysis function."""
"""
"""
"""
"""
    analyzer = ImportAnalyzer()

    print("Analyzing imports across the codebase...")
    report = analyzer.generate_report()

    print(report)

# Save detailed analysis to JSON
    analysis = analyzer.analyze_imports()
    with open('import_analysis.json', 'w') as f:
        json.dump(analysis, f, indent = 2)

    print(f"\\nDetailed analysis saved to: import_analysis.json")

# Save report to file
    with open('missing_imports_report.txt', 'w') as f:
        f.write(report)

    print(f"Report saved to: missing_imports_report.txt")


if __name__ == "__main__":
    main()

"""
"""
"""
"""
"""
