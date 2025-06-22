#!/usr/bin/env python3
"""Automated Syntax Fixer for Schwabot E999 Errors.

This script addresses the critical syntax errors identified in the stable trajectory plan:
1. Unterminated triple-quoted strings (line 10:32 pattern)
2. Invalid Unicode characters
3. Basic syntax validation and repair

Usage:
    python automated_syntax_fixer.py [--dry-run] [--target-dir core/]
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SyntaxFixer:
    """Automated syntax error fixer for Schwabot codebase."""
    
    def __init__(self, dry_run: bool = False):
        """Initialize the syntax fixer."""
        self.dry_run = dry_run
        self.fixes_applied = 0
        self.files_processed = 0
        self.errors_fixed = 0
        
        # Unicode character replacements
        self.unicode_replacements = {
            'Γêç': 'nabla',  # U+2207 - Nabla operator
            '┬╖': '.',       # U+00B7 - Middle dot
            'Γëñ': '<=',     # U+2264 - Less than or equal
            'Γéì': '(',      # U+208D - Subscript left parenthesis
            'Γêê': 'in',     # U+2208 - Element of
            'Γê½': 'int',    # U+222B - Integral
            'ΓçÆ': '=>',     # U+21D2 - Rightwards double arrow
        }
    
    def fix_file(self, file_path: str) -> Dict[str, any]:
        """Fix syntax errors in a single file."""
        result = {
            'file': file_path,
            'fixed': False,
            'errors_fixed': 0,
            'issues': []
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            fixes_applied = 0
            
            # Fix 1: Unterminated triple-quoted strings
            if self._has_unterminated_docstring(content):
                content = self._fix_unterminated_docstrings(content)
                fixes_applied += 1
                result['errors_fixed'] += 1
            
            # Fix 2: Invalid Unicode characters
            unicode_fixes = self._fix_unicode_characters(content)
            if unicode_fixes > 0:
                fixes_applied += 1
                result['errors_fixed'] += unicode_fixes
            
            # Fix 3: Basic syntax validation
            syntax_fixes = self._fix_basic_syntax(content)
            if syntax_fixes > 0:
                fixes_applied += 1
                result['errors_fixed'] += syntax_fixes
            
            # Apply changes if not dry run
            if fixes_applied > 0 and not self.dry_run:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                result['fixed'] = True
                self.fixes_applied += 1
                logger.info(f"Fixed {result['errors_fixed']} issues in {file_path}")
            elif fixes_applied > 0:
                logger.info(f"[DRY RUN] Would fix {result['errors_fixed']} issues in {file_path}")
            
            self.files_processed += 1
            
        except Exception as e:
            result['issues'].append(f"Error processing file: {e}")
            logger.error(f"Error processing {file_path}: {e}")
        
        return result
    
    def _has_unterminated_docstring(self, content: str) -> bool:
        """Check if content has unterminated triple-quoted strings."""
        return '"""' in content and content.count('"""') % 2 != 0
    
    def _fix_unterminated_docstrings(self, content: str) -> str:
        """Fix unterminated triple-quoted strings."""
        lines = content.split('\n')
        fixed_lines = []
        in_docstring = False
        docstring_started = False
        
        for i, line in enumerate(lines):
            # Check for docstring start
            if '"""' in line and not in_docstring:
                in_docstring = True
                docstring_started = True
                # Look for closing quote on same line
                if line.count('"""') >= 2:
                    in_docstring = False
                    docstring_started = False
                    # Keep the line if it's a complete docstring
                    fixed_lines.append(line)
                else:
                    # Skip problematic unterminated docstring
                    continue
            elif '"""' in line and in_docstring:
                # Found closing quote
                in_docstring = False
                docstring_started = False
                continue
            elif not in_docstring:
                fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    def _fix_unicode_characters(self, content: str) -> int:
        """Replace invalid Unicode characters with ASCII equivalents."""
        fixes = 0
        for unicode_char, replacement in self.unicode_replacements.items():
            if unicode_char in content:
                content = content.replace(unicode_char, replacement)
                fixes += 1
        return fixes
    
    def _fix_basic_syntax(self, content: str) -> int:
        """Fix basic syntax issues."""
        fixes = 0
        
        # Fix trailing whitespace
        lines = content.split('\n')
        fixed_lines = []
        for line in lines:
            if line.endswith(' '):
                line = line.rstrip()
                fixes += 1
            fixed_lines.append(line)
        
        # Ensure file ends with newline
        if fixed_lines and fixed_lines[-1] != '':
            fixed_lines.append('')
            fixes += 1
        
        return fixes
    
    def fix_directory(self, directory: str) -> List[Dict[str, any]]:
        """Fix syntax errors in all Python files in a directory."""
        results = []
        target_path = Path(directory)
        
        if not target_path.exists():
            logger.error(f"Directory {directory} does not exist")
            return results
        
        # Find all Python files
        python_files = list(target_path.rglob("*.py"))
        logger.info(f"Found {len(python_files)} Python files in {directory}")
        
        for py_file in python_files:
            if self._should_process_file(py_file):
                result = self.fix_file(str(py_file))
                results.append(result)
        
        return results
    
    def _should_process_file(self, file_path: Path) -> bool:
        """Determine if a file should be processed."""
        skip_patterns = [
            '.venv', 'venv', 'env', '__pycache__', '.git',
            'node_modules', 'site-packages', '.pytest_cache'
        ]
        
        file_str = str(file_path)
        return not any(pattern in file_str for pattern in skip_patterns)
    
    def get_statistics(self) -> Dict[str, int]:
        """Get statistics about the fixing process."""
        return {
            'files_processed': self.files_processed,
            'fixes_applied': self.fixes_applied,
            'errors_fixed': self.errors_fixed
        }


def main():
    """Main entry point for the syntax fixer."""
    parser = argparse.ArgumentParser(
        description="Automated syntax fixer for Schwabot E999 errors"
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be fixed without making changes'
    )
    parser.add_argument(
        '--target-dir',
        default='core/',
        help='Target directory to process (default: core/)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize fixer
    fixer = SyntaxFixer(dry_run=args.dry_run)
    
    logger.info(f"Starting syntax fixer (dry_run={args.dry_run})")
    logger.info(f"Target directory: {args.target_dir}")
    
    # Process files
    results = fixer.fix_directory(args.target_dir)
    
    # Report results
    successful_fixes = sum(1 for r in results if r['fixed'])
    total_errors_fixed = sum(r['errors_fixed'] for r in results)
    
    logger.info("=" * 50)
    logger.info("SYNTAX FIXING COMPLETE")
    logger.info("=" * 50)
    logger.info(f"Files processed: {fixer.files_processed}")
    logger.info(f"Files fixed: {successful_fixes}")
    logger.info(f"Total errors fixed: {total_errors_fixed}")
    
    if args.dry_run:
        logger.info("DRY RUN MODE - No changes were made")
    
    # Show detailed results for files with issues
    files_with_issues = [r for r in results if r['issues']]
    if files_with_issues:
        logger.warning(f"Files with processing issues: {len(files_with_issues)}")
        for result in files_with_issues:
            logger.warning(f"  {result['file']}: {result['issues']}")
    
    return 0 if successful_fixes > 0 else 1


if __name__ == "__main__":
    sys.exit(main()) 