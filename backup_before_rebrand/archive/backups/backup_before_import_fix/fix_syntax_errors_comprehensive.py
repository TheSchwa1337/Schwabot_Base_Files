#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Syntax Error Fixer for Schwabot Trading System

This script fixes all critical syntax errors including:
- Unmatched brackets, parentheses, and braces
- Indentation errors
- Invalid syntax
- Platform-specific path issues

Usage:
    python fix_syntax_errors_comprehensive.py
"""

import ast

# Configure logging
import logging
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Fix Unicode encoding issues on Windows
if sys.platform == 'win32':
    import codecs

    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('logs/syntax_fix.log', encoding='utf-8'), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class SyntaxErrorFixer:
    """Comprehensive syntax error fixer for Python files."""

    def __init__(self, backup_dir: str = "backup_before_syntax_fix"):
        """Initialize the syntax error fixer."""
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
        self.fixed_files = []
        self.failed_files = []

        # Common syntax error patterns
        self.bracket_patterns = {
            r'\{[^}]*$': '}',  # Unmatched opening brace
            r'\[[^\]]*$': ']',  # Unmatched opening bracket
            r'\([^)]*$': ')',  # Unmatched opening parenthesis
        }

        # Common indentation fixes
        self.indentation_fixes = {
            'unexpected indent': self._fix_unexpected_indent,
            'unindent does not match': self._fix_unindent_mismatch,
        }

    def backup_file(self, file_path: Path) -> None:
        """Create backup of file before modification."""
        try:
            backup_path = self.backup_dir / file_path.name
            shutil.copy2(file_path, backup_path)
            logger.info(f"Backed up {file_path} to {backup_path}")
        except Exception as e:
            logger.error(f"Failed to backup {file_path}: {e}")

    def fix_unmatched_brackets(self, content: str, file_path: Path) -> str:
        """Fix unmatched brackets, parentheses, and braces."""
        lines = content.split('\n')
        fixed_lines = []

        for i, line in enumerate(lines):
            # Check for unmatched opening brackets at end of line
            for pattern, closing in self.bracket_patterns.items():
                if re.search(pattern, line):
                    # Find the matching closing bracket
                    if closing == '}':
                        # Look for the matching closing brace
                        brace_count = line.count('{') - line.count('}')
                        if brace_count > 0:
                            # Add closing brace
                            line += ' ' * (len(line) - len(line.rstrip())) + closing
                            logger.info(f"Fixed unmatched brace in {file_path}:{i+1}")
                    elif closing == ']':
                        # Look for the matching closing bracket
                        bracket_count = line.count('[') - line.count(']')
                        if bracket_count > 0:
                            # Add closing bracket
                            line += ' ' * (len(line) - len(line.rstrip())) + closing
                            logger.info(f"Fixed unmatched bracket in {file_path}:{i+1}")
                    elif closing == ')':
                        # Look for the matching closing parenthesis
                        paren_count = line.count('(') - line.count(')')
                        if paren_count > 0:
                            # Add closing parenthesis
                            line += ' ' * (len(line) - len(line.rstrip())) + closing
                            logger.info(f"Fixed unmatched parenthesis in {file_path}:{i+1}")

            fixed_lines.append(line)

        return '\n'.join(fixed_lines)

    def _fix_unexpected_indent(self, content: str, file_path: Path) -> str:
        """Fix unexpected indentation errors."""
        lines = content.split('\n')
        fixed_lines = []

        for i, line in enumerate(lines):
            # Remove leading whitespace if line should not be indented
            if line.strip() and not line.strip().startswith('#'):
                # Check if this line should be at module level
                if line.strip().startswith(
                    ('import ', 'from ', 'class ', 'def ', 'if __name__', 'async def ')
                ) or line.strip().endswith((':', '= Enum(', '= "', "= '")):
                    # This should be at module level, remove indentation
                    if line.startswith(' '):
                        original_indent = len(line) - len(line.lstrip())
                        line = line.lstrip()
                        logger.info(f"Fixed unexpected indent in {file_path}:{i+1} (removed {original_indent} spaces)")

            fixed_lines.append(line)

        return '\n'.join(fixed_lines)

    def _fix_unindent_mismatch(self, content: str, file_path: Path) -> str:
        """Fix unindent does not match any outer indentation level errors."""
        lines = content.split('\n')
        fixed_lines = []

        for i, line in enumerate(lines):
            if line.strip() and not line.strip().startswith('#'):
                # Calculate proper indentation level
                if i > 0:
                    prev_line = lines[i - 1]
                    if prev_line.strip().endswith(':'):
                        # Should be indented
                        if not line.startswith('    '):
                            line = '    ' + line.lstrip()
                            logger.info(f"Fixed unindent mismatch in {file_path}:{i+1}")
                    elif prev_line.strip() and not prev_line.strip().endswith(':'):
                        # Check if this should be at same level as previous
                        prev_indent = len(prev_line) - len(prev_line.lstrip())
                        if line.strip() and not line.strip().startswith(('elif ', 'else:', 'except', 'finally:')):
                            # Should be at same level
                            line = ' ' * prev_indent + line.lstrip()
                            logger.info(f"Fixed unindent mismatch in {file_path}:{i+1}")

            fixed_lines.append(line)

        return '\n'.join(fixed_lines)

    def fix_indentation_errors(self, content: str, file_path: Path) -> str:
        """Fix various indentation errors."""
        # First pass: fix unexpected indents
        content = self._fix_unexpected_indent(content, file_path)

        # Second pass: fix unindent mismatches
        content = self._fix_unindent_mismatch(content, file_path)

        return content

    def fix_invalid_syntax(self, content: str, file_path: Path) -> str:
        """Fix invalid syntax errors."""
        lines = content.split('\n')
        fixed_lines = []

        for i, line in enumerate(lines):
            # Fix common invalid syntax patterns
            if line.strip().startswith('from .') and line.strip().endswith('import'):
                # Fix incomplete import statements
                line = line.rstrip() + ' *'
                logger.info(f"Fixed incomplete import in {file_path}:{i+1}")

            # Fix invalid character sequences
            line = re.sub(r'[^\x00-\x7F]+', '', line)  # Remove non-ASCII characters

            fixed_lines.append(line)

        return '\n'.join(fixed_lines)

    def fix_platform_issues(self, content: str, file_path: Path) -> str:
        """Fix platform-specific issues."""
        # Fix Windows path separators
        content = content.replace('\\', '/')

        # Fix line endings
        content = content.replace('\r\n', '\n')

        return content

    def validate_syntax(self, content: str, file_path: Path) -> bool:
        """Validate that the file has correct Python syntax."""
        try:
            ast.parse(content)
            return True
        except SyntaxError as e:
            logger.error(f"Syntax error in {file_path}:{e.lineno}: {e.text}")
            return False
        except Exception as e:
            logger.error(f"Validation error in {file_path}: {e}")
            return False

    def fix_file(self, file_path: Path) -> bool:
        """Fix syntax errors in a single file."""
        try:
            # Skip non-Python files
            if not file_path.suffix == '.py':
                return True

            # Skip backup and cache directories
            if any(part.startswith('.') or part == '__pycache__' for part in file_path.parts):
                return True

            logger.info(f"Processing {file_path}")

            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # Create backup
            self.backup_file(file_path)

            # Apply fixes
            original_content = content

            # Fix platform issues first
            content = self.fix_platform_issues(content, file_path)

            # Fix unmatched brackets
            content = self.fix_unmatched_brackets(content, file_path)

            # Fix indentation errors
            content = self.fix_indentation_errors(content, file_path)

            # Fix invalid syntax
            content = self.fix_invalid_syntax(content, file_path)

            # Validate syntax
            if not self.validate_syntax(content, file_path):
                logger.error(f"Failed to fix syntax in {file_path}")
                self.failed_files.append(str(file_path))
                return False

            # Write fixed content if changed
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                logger.info(f"Fixed {file_path}")
                self.fixed_files.append(str(file_path))
            else:
                logger.info(f"No changes needed for {file_path}")

            return True

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            self.failed_files.append(str(file_path))
            return False

    def fix_directory(self, directory: Path) -> None:
        """Fix syntax errors in all Python files in a directory."""
        logger.info(f"Scanning directory: {directory}")

        for file_path in directory.rglob('*.py'):
            self.fix_file(file_path)

        # Also check for specific problematic files mentioned in the error report
        specific_files = [
            'core/acceleration_enhancement.py',
            'core/advanced_dualistic_trading_execution_system.py',
            'core/api/handlers/alt_fear_greed.py',
            'core/api/handlers/coingecko.py',
            'core/api/handlers/glassnode.py',
            'core/api/handlers/whale_alert.py',
            'core/automated_trading_pipeline.py',
            'core/backtest_visualization.py',
            'core/crwf_crlf_integration.py',
            'core/final_integration_launcher.py',
            'core/master_profit_coordination_system.py',
            'core/phase_bit_integration.py',
            'core/profit_tier_adjuster.py',
            'core/real_multi_exchange_trader.py',
            'core/reentry_logic.py',
            'core/schwabot_rheology_integration.py',
            'core/schwafit_core.py',
            'core/secure_exchange_manager.py',
            'core/speed_lattice_trading_integration.py',
            'core/strategy/__init__.py',
            'core/strategy_trigger_router.py',
            'core/system/dual_state_router_backup.py',
            'core/system_integration_test.py',
            'core/trading_engine_integration.py',
            'core/type_defs.py',
            'core/unified_profit_vectorization_system.py',
            'core/warp_sync_core.py',
            'core/zpe_core.py',
        ]

        for file_path_str in specific_files:
            file_path = Path(file_path_str)
            if file_path.exists():
                self.fix_file(file_path)

    def generate_report(self) -> None:
        """Generate a report of the fixes applied."""
        report_path = Path('syntax_fix_report.md')

        with open(report_path, 'w') as f:
            f.write("# Syntax Fix Report\n\n")
            f.write(f"## Summary\n")
            f.write(f"- Files processed: {len(self.fixed_files) + len(self.failed_files)}\n")
            f.write(f"- Files fixed: {len(self.fixed_files)}\n")
            f.write(f"- Files failed: {len(self.failed_files)}\n")
            f.write(f"- Backup directory: {self.backup_dir}\n\n")

            f.write("## Fixed Files\n")
            for file_path in self.fixed_files:
                f.write(f"- {file_path}\n")

            f.write("\n## Failed Files\n")
            for file_path in self.failed_files:
                f.write(f"- {file_path}\n")

        logger.info(f"Report generated: {report_path}")


def main():
    """Main function to run the syntax error fixer."""
    logger.info("Starting comprehensive syntax error fix")

    # Create fixer instance
    fixer = SyntaxErrorFixer()

    # Fix core directory
    core_dir = Path('core')
    if core_dir.exists():
        fixer.fix_directory(core_dir)

    # Fix root directory Python files
    root_dir = Path('.')
    for file_path in root_dir.glob('*.py'):
        fixer.fix_file(file_path)

    # Generate report
    fixer.generate_report()

    logger.info("Syntax error fix completed")
    logger.info(f"Fixed {len(fixer.fixed_files)} files")
    logger.info(f"Failed {len(fixer.failed_files)} files")


if __name__ == "__main__":
    main()
