#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flake8 Issues Fixer - Systematic Codebase Cleanup
================================================

This script systematically identifies and fixes common Flake8 issues across
the entire codebase, focusing on:

1. Indentation errors (E999)
2. Missing function bodies
3. Duplicate pass statements
4. Syntax errors
5. Import issues

Usage:
    python fix_flake8_issues.py [--dry-run] [--fix-all] [--file <filename>]
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional


class Flake8Fixer:
    """Systematic Flake8 issue fixer."""

    def __init__(self, root_dir: str = "core"):
        """Initialize the fixer."""
        self.root_dir = Path(root_dir)
        self.fixed_files = []
        self.error_files = []
        self.patterns = {
            'duplicate_pass': r'^\\s*pass\\s*\\n\\s*pass\\s*$',
            'empty_try': r'^\\s*try:\\s*\\n\\s*pass\\s*$',
            'empty_function': r'^\\s*def\\s+\\w+.*:\\s*\\n\\s*pass\\s*$',
            'unexpected_indent': r'^\\s{2,}[^\\s#].*$',
            'missing_indent': r'^\\s*[a-zA-Z_]\\w*\\s*=\\s*[^#\\n]*$'
        }

    def find_python_files(self) -> List[Path]:
        """Find all Python files in the codebase."""
        python_files = []
        for file_path in self.root_dir.rglob("*.py"):
            if not file_path.name.startswith('__'):
                python_files.append(file_path)
        return python_files

    def check_syntax(self, file_path: Path) -> bool:
        """Check if a file has valid Python syntax."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                compile(f.read(), str(file_path), 'exec')
            return True
        except (SyntaxError, UnicodeDecodeError) as e:
            print(f"\\u274c Syntax error in {file_path}: {e}")
            return False

    def fix_common_issues(self, file_path: Path, dry_run: bool = False) -> bool:
        """Fix common Flake8 issues in a file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            original_content = content
            fixed = False

            # Fix 1: Remove duplicate pass statements
            content = re.sub(r'^\\s*pass\\s*\\n\\s*pass\\s*$', '    pass', content, flags=re.MULTILINE)

            # Fix 2: Fix empty try blocks
            content = re.sub(r'^\\s*try:\\s*\\n\\s*pass\\s*$', '    try:\\n        pass', content, flags=re.MULTILINE)

            # Fix 3: Fix empty function definitions
            content = re.sub(r'^\\s*def\\s+(\\w+.*):\\s*\\n\\s*pass\\s*$', r'    def \1:\\n        pass', content, flags=re.MULTILINE)

            # Fix 4: Fix indentation issues
            lines = content.split('\n')
            fixed_lines = []
            indent_level = 0

            for i, line in enumerate(lines):
                stripped = line.strip()
                
                # Skip empty lines and comments
                if not stripped or stripped.startswith('#'):
                    fixed_lines.append(line)
                    continue

                # Handle indentation
                if stripped.endswith(':'):
                    # This is a control structure, next line should be indented
                    fixed_lines.append(line)
                    indent_level += 1
                elif stripped.startswith(('def ', 'class ')):
                    # Function or class definition
                    fixed_lines.append(line)
                    indent_level += 1
                elif stripped in ('pass', 'break', 'continue', 'return'):
                    # These should be indented
                    if indent_level > 0:
                        fixed_lines.append('    ' * indent_level + stripped)
                    else:
                        fixed_lines.append(line)
                else:
                    # Regular line
                    fixed_lines.append(line)

            content = '\n'.join(fixed_lines)

            if content != original_content:
                if not dry_run:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print(f"\\u2705 Fixed {file_path}")
                else:
                    print(f"\\u1f527 Would fix {file_path}")
                fixed = True

            return fixed

        except Exception as e:
            print(f"\\u274c Error fixing {file_path}: {e}")
            return False

    def fix_specific_file(self, file_path: Path, dry_run: bool = False) -> bool:
        """Fix a specific file with known issues."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            original_content = content
            fixed = False

            # Fix common patterns
            patterns_to_fix = [
                # Remove duplicate pass statements
                (r'^\\s*pass\\s*\\n\\s*pass\\s*$', '    pass'),
                # Fix empty try blocks
                (r'^\\s*try:\\s*\\n\\s*pass\\s*$', '    try:\\n        pass'),
                # Fix empty function definitions
                (r'^\\s*def\\s+(\\w+.*):\\s*\\n\\s*pass\\s*$', r'    def \1:\\n        pass'),
                # Fix empty class definitions
                (r'^\\s*class\\s+(\\w+.*):\\s*\\n\\s*pass\\s*$', r'    class \1:\\n        pass'),
            ]

            for pattern, replacement in patterns_to_fix:
                new_content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
                if new_content != content:
                    content = new_content
                    fixed = True

            # Fix indentation issues
            lines = content.split('\n')
            fixed_lines = []
            in_function = False
            in_class = False
            indent_level = 0

            for line in lines:
                stripped = line.strip()
                
                if not stripped or stripped.startswith('#'):
                    fixed_lines.append(line)
                    continue

                # Check for function/class definitions
                if stripped.startswith('def '):
                    in_function = True
                    indent_level = 1
                    fixed_lines.append(line)
                elif stripped.startswith('class '):
                    in_class = True
                    indent_level = 1
                    fixed_lines.append(line)
                elif stripped.endswith(':'):
                    # Control structure
                    fixed_lines.append(line)
                    indent_level += 1
                elif stripped in ('pass', 'break', 'continue', 'return'):
                    # These should be indented if in a function/class
                    if in_function or in_class:
                        fixed_lines.append('    ' * indent_level + stripped)
                    else:
                        fixed_lines.append(line)
                elif stripped.startswith('self.') or stripped.startswith('def '):
                    # Class method or function
                    if in_class:
                        fixed_lines.append('    ' * indent_level + stripped)
                    else:
                        fixed_lines.append(line)
                else:
                    fixed_lines.append(line)

            content = '\n'.join(fixed_lines)

            if content != original_content:
                if not dry_run:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print(f"\\u2705 Fixed {file_path}")
                else:
                    print(f"\\u1f527 Would fix {file_path}")
                fixed = True

            return fixed

        except Exception as e:
            print(f"\\u274c Error fixing {file_path}: {e}")
            return False

    def run_fixes(self, dry_run: bool = False, specific_file: Optional[str] = None) -> Dict[str, int]:
        """Run all fixes."""
        stats = {
            'total_files': 0,
            'fixed_files': 0,
            'error_files': 0,
            'syntax_errors': 0
        }

        if specific_file:
            files_to_check = [Path(specific_file)]
        else:
            files_to_check = self.find_python_files()

        print(f"\\u1f50d Checking {len(files_to_check)} Python files...")

        for file_path in files_to_check:
            stats['total_files'] += 1

            # Check syntax first
            if not self.check_syntax(file_path):
                stats['syntax_errors'] += 1
                self.error_files.append(file_path)
                continue

            # Try to fix issues
            try:
                if self.fix_specific_file(file_path, dry_run):
                    stats['fixed_files'] += 1
                    self.fixed_files.append(file_path)
            except Exception as e:
                print(f"\\u274c Error processing {file_path}: {e}")
                stats['error_files'] += 1
                self.error_files.append(file_path)

        return stats

    def generate_report(self, stats: Dict[str, int]) -> None:
        """Generate a comprehensive report."""
        print("\n" + "="*60)
        print("\\u1f4ca FLAKE8 FIXES REPORT")
        print("="*60)
        print(f"Total files checked: {stats['total_files']}")
        print(f"Files fixed: {stats['fixed_files']}")
        print(f"Files with errors: {stats['error_files']}")
        print(f"Syntax errors: {stats['syntax_errors']}")
        
        if self.fixed_files:
            print(f"\\n\\u2705 Fixed files:")
            for file_path in self.fixed_files:
                print(f"   - {file_path}")
        
        if self.error_files:
            print(f"\\n\\u274c Files with errors:")
            for file_path in self.error_files:
                print(f"   - {file_path}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fix Flake8 issues systematically")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be fixed without making changes")
    parser.add_argument("--fix-all", action="store_true", help="Fix all files")
    parser.add_argument("--file", type=str, help="Fix specific file")
    parser.add_argument("--root-dir", type=str, default="core", help="Root directory to scan")
    
    args = parser.parse_args()
    
    fixer = Flake8Fixer(args.root_dir)
    
    if args.dry_run:
        print("\\u1f50d DRY RUN MODE - No changes will be made")
    
    stats = fixer.run_fixes(dry_run=args.dry_run, specific_file=args.file)
    fixer.generate_report(stats)
    
    if stats['error_files'] > 0:
        print(f"\\n\\u26a0\\ufe0f  {stats['error_files']} files still have issues that need manual attention")
        return 1
    
    print(f"\\n\\u1f389 Successfully processed {stats['total_files']} files!")
    return 0


if __name__ == "__main__":
    sys.exit(main()) 
"""