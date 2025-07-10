#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Syntax Error Fixer

This script identifies and fixes all remaining syntax errors in the Schwabot codebase.
Targets missing colons, malformed decorators, indentation issues, and other syntax problems.
"""

import ast
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SyntaxErrorFixer:
    """Comprehensive syntax error fixer for Python files."""

    def __init__(self, root_dir: str = "."):
        self.root_dir = Path(root_dir)
        self.fixed_files = []
        self.error_files = []
        self.stats = {
            "files_checked": 0,
            "files_fixed": 0,
            "files_with_errors": 0,
            "total_errors_fixed": 0,
        }

    def find_python_files(self) -> List[Path]:
        """Find all Python files in the codebase."""
        python_files = []
        for root, dirs, files in os.walk(self.root_dir):
            # Skip common directories that shouldn't contain our code
            dirs[:] = [d for d in dirs if d not in {
                '__pycache__', '.git', 'venv', 'env', 'node_modules',
                'backup', 'backups', 'schwabot_env'
            }]
            
            for file in files:
                if file.endswith('.py'):
                    python_files.append(Path(root) / file)
        
        return python_files

    def check_syntax_errors(self, file_path: Path) -> List[str]:
        """Check for syntax errors in a Python file."""
        errors = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Try to parse the file
            ast.parse(content)
            return errors
            
        except SyntaxError as e:
            errors.append(f"SyntaxError: {e}")
        except Exception as e:
            errors.append(f"Error: {e}")
        
        return errors

    def fix_missing_colons(self, content: str) -> str:
        """Fix missing colons in function and class definitions."""
        lines = content.split('\n')
        fixed_lines = []
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Fix function definitions missing colons
            if re.match(r'^def\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\([^)]*\)[^:]*$', stripped):
                if not stripped.endswith(':'):
                    line = line.rstrip() + ':'
                    logger.info(f"Fixed missing colon in function definition: {stripped}")
            
            # Fix class definitions missing colons
            elif re.match(r'^class\s+[a-zA-Z_][a-zA-Z0-9_]*[^:]*$', stripped):
                if not stripped.endswith(':'):
                    line = line.rstrip() + ':'
                    logger.info(f"Fixed missing colon in class definition: {stripped}")
            
            # Fix if/for/while/with statements missing colons
            elif re.match(r'^(if|for|while|with|try|except|finally|else|elif)\s+.*[^:]*$', stripped):
                if not stripped.endswith(':'):
                    line = line.rstrip() + ':'
                    logger.info(f"Fixed missing colon in statement: {stripped}")
            
            fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)

    def fix_malformed_decorators(self, content: str) -> str:
        """Fix malformed decorators like @ staticmethod."""
        # Fix @ staticmethod -> @staticmethod
        content = re.sub(r'@\s+staticmethod', '@staticmethod', content)
        content = re.sub(r'@\s+classmethod', '@classmethod', content)
        content = re.sub(r'@\s+property', '@property', content)
        
        # Fix other common decorator patterns
        content = re.sub(r'@\s+([a-zA-Z_][a-zA-Z0-9_]*)', r'@\1', content)
        
        return content

    def fix_indentation_issues(self, content: str) -> str:
        """Fix common indentation issues."""
        lines = content.split('\n')
        fixed_lines = []
        
        for line in lines:
            # Fix mixed tabs and spaces
            if '\t' in line:
                line = line.expandtabs(4)
            
            # Fix inconsistent indentation
            stripped = line.lstrip()
            if stripped and not line.startswith('#'):
                # Ensure consistent 4-space indentation
                indent_level = len(line) - len(line.lstrip())
                if indent_level % 4 != 0:
                    # Round to nearest 4-space boundary
                    new_indent_level = (indent_level // 4) * 4
                    line = ' ' * new_indent_level + stripped
            
            fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)

    def fix_broken_string_formatting(self, content: str) -> str:
        """Fix broken string formatting."""
        # Fix malformed f-strings
        content = re.sub(r'f\s*"([^"]*)"', r'f"\1"', content)
        content = re.sub(r'f\s*\'([^\']*)\'', r"f'\1'", content)
        
        # Fix malformed .format() calls
        content = re.sub(r'\.format\s*\(\s*\)\s*$', '.format()', content, flags=re.MULTILINE)
        
        return content

    def fix_empty_try_blocks(self, content: str) -> str:
        """Fix empty try blocks."""
        lines = content.split('\n')
        fixed_lines = []
        i = 0
        
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            
            # Check for try: followed by except
            if stripped == 'try:' and i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line.startswith('except'):
                    # Insert pass statement
                    fixed_lines.append(line)
                    fixed_lines.append('    pass')
                    logger.info("Fixed empty try block")
                    i += 1
                else:
                    fixed_lines.append(line)
            else:
                fixed_lines.append(line)
            
            i += 1
        
        return '\n'.join(fixed_lines)

    def fix_file(self, file_path: Path) -> bool:
        """Fix syntax errors in a single file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Apply fixes
            content = self.fix_missing_colons(content)
            content = self.fix_malformed_decorators(content)
            content = self.fix_indentation_issues(content)
            content = self.fix_broken_string_formatting(content)
            content = self.fix_empty_try_blocks(content)
            
            # Check if content changed
            if content != original_content:
                # Verify the fixed content is valid Python
                try:
                    ast.parse(content)
                    
                    # Write the fixed content
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    self.fixed_files.append(str(file_path))
                    self.stats["files_fixed"] += 1
                    logger.info(f"✅ Fixed syntax errors in: {file_path}")
                    return True
                    
                except SyntaxError as e:
                    logger.error(f"❌ Fixed content still has syntax errors in {file_path}: {e}")
                    self.error_files.append(str(file_path))
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error processing {file_path}: {e}")
            self.error_files.append(str(file_path))
            return False

    def run(self) -> Dict[str, Any]:
        """Run the comprehensive syntax error fix."""
        logger.info("🔍 Starting comprehensive syntax error fix...")
        
        python_files = self.find_python_files()
        logger.info(f"Found {len(python_files)} Python files to check")
        
        for file_path in python_files:
            self.stats["files_checked"] += 1
            
            # Check for syntax errors
            errors = self.check_syntax_errors(file_path)
            if errors:
                self.stats["files_with_errors"] += 1
                logger.info(f"🔧 Fixing syntax errors in: {file_path}")
                self.fix_file(file_path)
            else:
                logger.debug(f"✅ No syntax errors in: {file_path}")
        
        # Generate summary
        logger.info("\n" + "="*60)
        logger.info("📊 SYNTAX ERROR FIX SUMMARY")
        logger.info("="*60)
        logger.info(f"Files checked: {self.stats['files_checked']}")
        logger.info(f"Files with errors: {self.stats['files_with_errors']}")
        logger.info(f"Files fixed: {self.stats['files_fixed']}")
        logger.info(f"Files still with errors: {len(self.error_files)}")
        
        if self.fixed_files:
            logger.info("\n✅ Fixed files:")
            for file_path in self.fixed_files:
                logger.info(f"  - {file_path}")
        
        if self.error_files:
            logger.info("\n❌ Files still with errors:")
            for file_path in self.error_files:
                logger.info(f"  - {file_path}")
        
        return self.stats


def main():
    """Main function to run the syntax error fixer."""
    fixer = SyntaxErrorFixer()
    stats = fixer.run()
    
    if stats["files_with_errors"] == 0:
        logger.info("🎉 All files are now syntax-error free!")
    else:
        logger.info(f"⚠️  {stats['files_with_errors']} files still have syntax errors")
    
    return stats


if __name__ == "__main__":
    main() 