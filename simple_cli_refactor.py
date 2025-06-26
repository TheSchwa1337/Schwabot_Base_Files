#!/usr/bin/env python3
"""
Simple CLI Output Refactor Script

This script uses regex patterns to replace Unicode/emoji print statements with safe_print calls
across the entire codebase for Windows CLI compatibility.
"""

from safe_print import safe_print, info, warn, error, success
import os
import re
import sys
from pathlib import Path
from typing import List, Set

# Import our safe print utility
sys.path.append('utils')


class SimpleCLIRefactor:
    """Simple refactor class for CLI output safety using regex."""

    def __init__(self, root_dir: str = '.'):
        self.root_dir = Path(root_dir)
        self.python_files = []
        self.modified_files = []
        self.skipped_files = []
        self.errors = []

        # Files to skip
        self.skip_patterns = [
            r'__pycache__',
            r'\.git',
            r'\.mypy_cache',
            r'\.venv',
            r'venv',
            r'env',
            r'node_modules',
            r'\.pytest_cache',
            r'\.coverage',
            r'\.tox',
            r'build',
            r'dist',
            r'\.eggs',
            r'\.idea',
            r'\.vscode',
            r'utils/safe_print\.py',  # Skip our own utility
            r'refactor_cli_output\.py',  # Skip the complex version
            r'simple_cli_refactor\.py',  # Skip this script
        ]

        # Common Unicode patterns found in the codebase
        self.unicode_patterns = [
            # Emojis and symbols
            r'[🚀📈📉💰⚡🔥❄💡🎯🎪🎭🎨🎵🎮🏆🥇🥈🥉]',
            r'[🔗🔒🔓🔐🔑🔨🔩🔪🔫🔬🔭🔮🔯🔰🔱🔲🔳🔴🔵🔶🔷🔸🔹🔺🔻🔼🔽]',
            r'[⚔🛡⚓🎪🎭🎨🎵🎮🏆🥇🥈🥉]',

            # Mathematical symbols
            r'[±×÷≤≥≠≈∞∑∏∫∂∇∆]',
            r'[αβγδεζηθικλμνξοπρστυφχψω]',
            r'[ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ]',

            # Arrows and navigation
            r'[→←↑↓⇒⇐⇑⇓]',

            # Currency symbols
            r'[€£¥₹₿]',

            # Status indicators
            r'[✓✗⚠ℹ]',

            # Common Unicode characters
            r'[–—""''…•◦▪▫▬▭▮▯▰▱▲△▼▽◀◁▶▷◆◇●○◐◑◒◓◔◕◖◗◘◙◚◛◜◝◞◟◠◡◢◣◤◥◦◧◨◩◪◫◬◭◮◯]',
        ]

        # Combined Unicode pattern
        self.unicode_regex = re.compile('|'.join(self.unicode_patterns))

    def find_python_files(self) -> List[Path]:
        """Find all Python files in the codebase."""
        info("Scanning for Python files...")

        python_files = []
        for pattern in ['*.py', '*.pyi']:
            python_files.extend(self.root_dir.rglob(pattern))

        # Filter out skipped files
        filtered_files = []
        for file_path in python_files:
            skip = False
            for pattern in self.skip_patterns:
                if re.search(pattern, str(file_path)):
                    skip = True
                    break

            if not skip:
                filtered_files.append(file_path)

        self.python_files = filtered_files
        info(f"Found {len(self.python_files)} Python files to process")
        return filtered_files

    def contains_unicode(self, text: str) -> bool:
        """Check if text contains Unicode characters."""
        return bool(self.unicode_regex.search(text))

    def refactor_file(self, file_path: Path) -> bool:
        """Refactor a single file to use safe_print."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            original_content = content

            # Check if file already has safe_print import
            has_safe_print_import = 'from utils.safe_print' in content or 'import utils.safe_print' in content

            # Find print statements with Unicode content
            print_pattern = r'print\s*\((.*?)\)'

            def replace_print(match):
                args_str = match.group(1)

                # Check if this print statement contains Unicode
                if not self.contains_unicode(args_str):
                    return match.group(0)  # No change needed

                # Replace with safe_print
                return f'safe_print({args_str})'

            # Replace print statements
            content = re.sub(print_pattern, replace_print, content, flags=re.DOTALL)

            # Add import if needed and content was modified
            if content != original_content and not has_safe_print_import:
                content = self._add_safe_print_import(content)

            # Write back if modified
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)

                self.modified_files.append(str(file_path))
                return True

            return False

        except Exception as e:
            self.errors.append(f"Error refactoring {file_path}: {e}")
            return False

    def _add_safe_print_import(self, content: str) -> str:
        """Add safe_print import to the file."""
        lines = content.split('\n')

        # Find the best place to add import (after existing imports)
        import_end = 0
        for i, line in enumerate(lines):
            if line.strip().startswith(('import ', 'from ')):
                import_end = i + 1
            elif line.strip() and not line.strip().startswith(('#', '"""', "'''")):
                break

        # Add import
        import_line = 'from utils.safe_print import safe_print, info, warn, error, success, debug'
        lines.insert(import_end, import_line)

        return '\n'.join(lines)

    def scan_for_unicode_prints(self, file_path: Path) -> List[str]:
        """Scan a file for print statements containing Unicode."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            unicode_prints = []
            print_pattern = r'print\s*\((.*?)\)'

            for match in re.finditer(print_pattern, content, flags=re.DOTALL):
                args_str = match.group(1)
                if self.contains_unicode(args_str):
                    unicode_prints.append(args_str.strip())

            return unicode_prints

        except Exception as e:
            self.errors.append(f"Error scanning {file_path}: {e}")
            return []

    def run_refactor(self) -> None:
        """Run the complete refactor process."""
        info("Starting simple CLI output refactor...")

        # Find all Python files
        files = self.find_python_files()

        # Analyze and refactor each file
        total_files = len(files)
        modified_count = 0

        for i, file_path in enumerate(files):
            try:
                info(f"Processing {i+1}/{total_files}: {file_path}")

                # Scan for Unicode prints
                unicode_prints = self.scan_for_unicode_prints(file_path)

                if unicode_prints:
                    info(f"  Found {len(unicode_prints)} Unicode print statements")

                    # Show examples
                    for j, print_stmt in enumerate(unicode_prints[:3]):  # Show first 3
                        info(f"    Example {j+1}: {print_stmt[:100]}...")

                    if len(unicode_prints) > 3:
                        info(f"    ... and {len(unicode_prints) - 3} more")

                    # Refactor file
                    if self.refactor_file(file_path):
                        modified_count += 1
                        success(f"  Refactored {file_path}")
                    else:
                        warn(f"  No changes needed for {file_path}")
                else:
                    # No Unicode found
                    pass

            except Exception as e:
                error(f"Error processing {file_path}: {e}")
                self.errors.append(str(e))

        # Summary
        info(f"Refactor complete!")
        info(f"Files processed: {total_files}")
        info(f"Files modified: {modified_count}")
        info(f"Files skipped: {len(self.skipped_files)}")

        if self.errors:
            error(f"Errors encountered: {len(self.errors)}")
            for error_msg in self.errors:
                error(f"  {error_msg}")

        if self.modified_files:
            success("Modified files:")
            for file_path in self.modified_files:
                success(f"  {file_path}")


def main():
    """Main entry point."""
    refactor = SimpleCLIRefactor()
    refactor.run_refactor()


if __name__ == '__main__':
    main()
