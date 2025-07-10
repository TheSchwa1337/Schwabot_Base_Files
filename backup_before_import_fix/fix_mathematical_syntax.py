#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mathematical Syntax Fixer for Schwabot Trading System

This script systematically fixes syntax issues in the mathematical codebase:
1. F-string compatibility (Python 3.8)
2. Import organization
3. Line length formatting
4. Mathematical function structure
"""

import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class MathematicalSyntaxFixer:
    """Fix mathematical syntax issues systematically."""

    def __init__(self, core_dir: str = "core"):
        self.core_dir = Path(core_dir)
        self.fixed_files = []
        self.errors = []

    def fix_all_syntax_issues(self) -> Dict[str, int]:
        """Fix all syntax issues in the mathematical codebase."""
        print("🔧 Fixing mathematical syntax issues...")

        stats = {}
            'files_processed': 0,
            'f_strings_fixed': 0,
            'imports_organized': 0,
            'line_lengths_fixed': 0,
            'errors': 0
        }

        for py_file in self.core_dir.rglob("*.py"):
            if py_file.is_file():
                try:
                    stats['files_processed'] += 1
                    file_stats = self.fix_file_syntax(py_file)

                    for key in file_stats:
                        if key in stats:
                            stats[key] += file_stats[key]

                except Exception as e:
                    stats['errors'] += 1
                    self.errors.append(f"Error processing {py_file}: {e}")
                    print(f"❌ Error processing {py_file}: {e}")

        return stats

    def fix_file_syntax(self, file_path: Path) -> Dict[str, int]:
        """Fix syntax issues in a single file."""
        stats = {}
            'f_strings_fixed': 0,
            'imports_organized': 0,
            'line_lengths_fixed': 0
        }

        print(f"📝 Processing: {file_path}")

        # Read file content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content

        # 1. Fix f-string compatibility
        content, f_string_count = self.fix_f_strings(content)
        stats['f_strings_fixed'] = f_string_count

        # 2. Organize imports
        content, import_count = self.organize_imports(content)
        stats['imports_organized'] = import_count

        # 3. Fix line lengths
        content, line_count = self.fix_line_lengths(content)
        stats['line_lengths_fixed'] = line_count

        # Write back if changes were made
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            self.fixed_files.append(str(file_path))
            print(f"✅ Fixed {sum(stats.values())} issues in {file_path}")

        return stats

    def fix_f_strings(self, content: str) -> Tuple[str, int]:
        """Fix f-string compatibility for Python 3.8."""
        fixed_count = 0

        # Pattern to match f-strings with simple variable interpolation
        f_string_pattern = r'f"([^"]*\{[^}]*\}[^"]*)"'

        def replace_f_string(match):
            nonlocal fixed_count
            f_string = match.group(1)

            # Extract variables from f-string
            var_pattern = r'\{([^}]*)\}'
            variables = re.findall(var_pattern, f_string)

            if variables:
                # Convert to .format() style
                format_string = f_string
                for i, var in enumerate(variables):
                    format_string = format_string.replace(f'{{{var}}}', f'{{{i}}}')

                format_string = f'"{format_string}".format({", ".join(variables)})'
                fixed_count += 1
                return format_string

            return match.group(0)

        content = re.sub(f_string_pattern, replace_f_string, content)

        # Also fix f-strings with single quotes
        f_string_pattern_single = r"f'([^']*\{[^}]*\}[^']*)'"
        content = re.sub(f_string_pattern_single, replace_f_string, content)

        return content, fixed_count

    def organize_imports(self, content: str) -> Tuple[str, int]:
        """Organize imports according to mathematical standards."""
        lines = content.split('\n')
        import_sections = {}
            'stdlib': [],
            'third_party': [],
            'internal': []
        }

        other_lines = []
        in_import_section = False

        for line in lines:
            stripped = line.strip()

            if stripped.startswith('import ') or stripped.startswith('from '):
                in_import_section = True

                # Categorize imports
                if any(lib in line for lib in ['os', 'sys', 'time', 'asyncio', 'threading', 'typing', 'dataclasses', 'datetime', 'decimal', 'logging', 'json', 'argparse']):
                    import_sections['stdlib'].append(line)
                elif any(lib in line for lib in ['numpy', 'scipy', 'pandas', 'matplotlib', 'ccxt', 'aiohttp', 'requests']):
                    import_sections['third_party'].append(line)
                elif any(lib in line for lib in ['utils.', 'core.', 'schwabot']):
                    import_sections['internal'].append(line)
                else:
                    import_sections['stdlib'].append(line)
            else:
                if in_import_section and stripped == '':
                    continue  # Skip empty lines between imports
                in_import_section = False
                other_lines.append(line)

        # Reconstruct content with organized imports
        new_content = []

        # Add standard library imports
        if import_sections['stdlib']:
            new_content.extend(import_sections['stdlib'])
            new_content.append('')

        # Add third-party imports
        if import_sections['third_party']:
            new_content.extend(import_sections['third_party'])
            new_content.append('')

        # Add internal imports
        if import_sections['internal']:
            new_content.extend(import_sections['internal'])
            new_content.append('')

        # Add remaining content
        new_content.extend(other_lines)

        return '\n'.join(new_content), len(import_sections['stdlib']) + len(import_sections['third_party']) + len(import_sections['internal'])

    def fix_line_lengths(self, content: str) -> Tuple[str, int]:
        """Fix lines that exceed 120 characters."""
        lines = content.split('\n')
        fixed_lines = []
        fixed_count = 0

        for line in lines:
            if len(line) > 120 and not line.strip().startswith('#'):
                # Try to break long lines intelligently
                fixed_line = self.break_long_line(line)
                if fixed_line != line:
                    fixed_count += 1
                fixed_lines.append(fixed_line)
            else:
                fixed_lines.append(line)

        return '\n'.join(fixed_lines), fixed_count

    def break_long_line(self, line: str) -> str:
        """Intelligently break long lines for mathematical expressions."""
        # Don't break comments'
        if line.strip().startswith('#'):
            return line

        # Don't break docstrings'
        if '"""' in line or "'''" in line: '
            return line

        # Try to break at operators
        operators = [' + ', ' - ', ' * ', ' / ', ' = ', ' == ', ' != ', ' <= ', ' >= ', ' and ', ' or ']

        for op in operators:
            if op in line and len(line) > 120:
                parts = line.split(op)
                if len(parts) > 1:
                    # Find the best break point
                    for i in range(len(parts) - 1):
                        left_part = op.join(parts[:i+1])
                        right_part = op.join(parts[i+1:])

                        if len(left_part) <= 120 and len(right_part) <= 120:
                            return f"{left_part}\n    {right_part}"

        # If no good break point, just return the line
        return line

    def generate_report(self, stats: Dict[str, int]):
        """Generate a report of the fixes applied."""
        print("\n" + "="*60)
        print("📊 MATHEMATICAL SYNTAX FIX REPORT")
        print("="*60)

        print(f"📁 Files processed: {stats['files_processed']}")
        print(f"🔧 F-strings fixed: {stats['f_strings_fixed']}")
        print(f"📦 Imports organized: {stats['imports_organized']}")
        print(f"📏 Line lengths fixed: {stats['line_lengths_fixed']}")
        print(f"❌ Errors encountered: {stats['errors']}")

        if self.fixed_files:
            print(f"\n✅ Files modified:")
            for file in self.fixed_files:
                print(f"   - {file}")

        if self.errors:
            print(f"\n❌ Errors:")
            for error in self.errors:
                print(f"   - {error}")

        print("\n" + "="*60)

def main():
    """Main function to run the mathematical syntax fixer."""
    print("🚀 Schwabot Mathematical Syntax Fixer")
    print("="*50)

    fixer = MathematicalSyntaxFixer()
    stats = fixer.fix_all_syntax_issues()
    fixer.generate_report(stats)

    if stats['errors'] == 0:
        print("\n🎉 All mathematical syntax issues fixed successfully!")
        return 0
    else:
        print(f"\n⚠️  Fixed with {stats['errors']} errors. Check the report above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 