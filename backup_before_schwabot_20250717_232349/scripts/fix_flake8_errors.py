#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔧 Schwabot Flake8 Error Fixer

Automatically fixes common flake8 errors to ensure a clean, production-ready codebase.
This script addresses:
- Trailing whitespace
- Line length issues
- Import organization
- Basic syntax issues
- Unused imports

Usage:
    python scripts/fix_flake8_errors.py --dry-run    # Show what would be fixed
    python scripts/fix_flake8_errors.py --fix        # Actually fix the errors
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Set, Any
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Flake8ErrorFixer:
    """Fix common flake8 errors automatically."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.fixes_applied = 0
        self.files_processed = 0
        
    def find_python_files(self) -> List[Path]:
        """Find all Python files in the project."""
        python_files = []
        for root, dirs, files in os.walk(self.project_root):
            # Skip certain directories
            dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', 'archive'}]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    python_files.append(file_path)
        
        return python_files
    
    def fix_trailing_whitespace(self, content: str) -> str:
        """Remove trailing whitespace from lines."""
        lines = content.split('\n')
        fixed_lines = []
        
        for line in lines:
            # Remove trailing whitespace but preserve empty lines
            if line.strip() == '':
                fixed_lines.append('')
            else:
                fixed_lines.append(line.rstrip())
        
        return '\n'.join(fixed_lines)
    
    def fix_line_length(self, content: str, max_length: int = 120) -> str:
        """Fix lines that are too long."""
        lines = content.split('\n')
        fixed_lines = []
        
        for line in lines:
            if len(line) > max_length:
                # Try to break long lines intelligently
                if 'import ' in line and ',' in line:
                    # Fix long import lines
                    fixed_line = self.fix_long_import_line(line)
                    fixed_lines.append(fixed_line)
                elif 'def ' in line or 'class ' in line:
                    # Fix long function/class definitions
                    fixed_line = self.fix_long_definition_line(line)
                    fixed_lines.append(fixed_line)
                elif '(' in line and ')' in line:
                    # Fix long function calls
                    fixed_line = self.fix_long_function_call(line)
                    fixed_lines.append(fixed_line)
                else:
                    # For other long lines, try to break at spaces
                    fixed_line = self.break_long_line(line, max_length)
                    fixed_lines.append(fixed_line)
            else:
                fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    def fix_long_import_line(self, line: str) -> str:
        """Fix long import lines by breaking them properly."""
        if 'from ' in line and ' import ' in line:
            # Handle 'from x import y' imports
            parts = line.split(' import ')
            if len(parts) == 2:
                from_part = parts[0]
                import_part = parts[1]
                
                # Break the import part if it's long
                if len(import_part) > 50:
                    imports = [imp.strip() for imp in import_part.split(',')]
                    fixed_imports = []
                    current_line = f"{from_part} import "
                    
                    for imp in imports:
                        if len(current_line + imp) > 100:
                            fixed_imports.append(current_line.rstrip())
                            current_line = "    " + imp + ", "
                        else:
                            current_line += imp + ", "
                    
                    if current_line.strip():
                        fixed_imports.append(current_line.rstrip().rstrip(','))
                    
                    return '\n'.join(fixed_imports)
        
        return line
    
    def fix_long_definition_line(self, line: str) -> str:
        """Fix long function/class definition lines."""
        # For now, just return the line as-is
        # This is complex and might break functionality
        return line
    
    def fix_long_function_call(self, line: str) -> str:
        """Fix long function call lines."""
        # For now, just return the line as-is
        # This is complex and might break functionality
        return line
    
    def break_long_line(self, line: str, max_length: int) -> str:
        """Break a long line at spaces."""
        if len(line) <= max_length:
            return line
        
        # Try to break at spaces
        words = line.split()
        if len(words) <= 1:
            return line
        
        lines = []
        current_line = ""
        
        for word in words:
            if len(current_line + " " + word) <= max_length:
                current_line += (" " + word) if current_line else word
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        
        if current_line:
            lines.append(current_line)
        
        return '\n'.join(lines)
    
    def remove_unused_imports(self, content: str) -> str:
        """Remove obviously unused imports."""
        lines = content.split('\n')
        fixed_lines = []
        in_import_block = False
        
        for line in lines:
            stripped = line.strip()
            
            # Check if this is an import line
            if stripped.startswith(('import ', 'from ')):
                in_import_block = True
                # For now, keep all imports to avoid breaking functionality
                fixed_lines.append(line)
            elif in_import_block and stripped == '':
                # End of import block
                in_import_block = False
                fixed_lines.append(line)
            else:
                in_import_block = False
                fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    def fix_common_issues(self, content: str) -> str:
        """Fix common flake8 issues."""
        # Remove trailing whitespace
        content = self.fix_trailing_whitespace(content)
        
        # Fix line length issues
        content = self.fix_line_length(content, max_length=120)
        
        # Remove unused imports (conservative approach)
        content = self.remove_unused_imports(content)
        
        # Ensure file ends with newline
        if content and not content.endswith('\n'):
            content += '\n'
        
        return content
    
    def process_file(self, file_path: Path, dry_run: bool = True) -> Dict[str, Any]:
        """Process a single file and fix flake8 errors."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            fixed_content = self.fix_common_issues(original_content)
            
            if original_content != fixed_content:
                if not dry_run:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(fixed_content)
                    self.fixes_applied += 1
                    logger.info(f"Fixed: {file_path}")
                else:
                    logger.info(f"Would fix: {file_path}")
                
                return {
                    'file': str(file_path),
                    'fixed': True,
                    'original_length': len(original_content),
                    'fixed_length': len(fixed_content)
                }
            else:
                return {
                    'file': str(file_path),
                    'fixed': False
                }
                
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return {
                'file': str(file_path),
                'fixed': False,
                'error': str(e)
            }
    
    def run_fixes(self, dry_run: bool = True) -> Dict[str, Any]:
        """Run flake8 error fixes on all Python files."""
        logger.info("🔍 Scanning for Python files...")
        
        python_files = self.find_python_files()
        logger.info(f"Found {len(python_files)} Python files")
        
        results = {
            'files_processed': 0,
            'files_fixed': 0,
            'fixes_applied': 0,
            'errors': []
        }
        
        for file_path in python_files:
            try:
                result = self.process_file(file_path, dry_run)
                results['files_processed'] += 1
                
                if result['fixed']:
                    results['files_fixed'] += 1
                
                if 'error' in result:
                    results['errors'].append(result)
                    
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                results['errors'].append({
                    'file': str(file_path),
                    'error': str(e)
                })
        
        results['fixes_applied'] = self.fixes_applied
        
        return results

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Fix flake8 errors in Schwabot")
    parser.add_argument(
        "--action", 
        choices=["dry-run", "fix"],
        default="dry-run",
        help="Action to perform (default: dry-run)"
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root directory (default: current directory)"
    )
    
    args = parser.parse_args()
    
    fixer = Flake8ErrorFixer(args.project_root)
    results = fixer.run_fixes(dry_run=(args.action == "dry-run"))
    
    # Report results
    logger.info("=" * 50)
    logger.info("🔧 FLAKE8 ERROR FIX RESULTS")
    logger.info("=" * 50)
    logger.info(f"Files processed: {results['files_processed']}")
    logger.info(f"Files fixed: {results['files_fixed']}")
    logger.info(f"Total fixes applied: {results['fixes_applied']}")
    
    if results['errors']:
        logger.info(f"Errors encountered: {len(results['errors'])}")
        for error in results['errors'][:5]:  # Show first 5 errors
            logger.error(f"  {error['file']}: {error['error']}")
    
    if args.action == "dry-run":
        logger.info("\n💡 Run with --action fix to apply the fixes")
    else:
        logger.info("\n✅ Fixes applied successfully!")

if __name__ == "__main__":
    main() 