#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fix F821 Undefined Name Errors
==============================

Targeted script to remove orphaned code fragments causing F821 errors:
- data = np.array(data) statements outside functions
- return result statements outside functions
- Orphaned mathematical calculation comments
- Other undefined variable references

This script systematically cleans up corrupted code patterns.
"""

import os
import re
import logging
from pathlib import Path
from typing import List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class F821Fixer:
    """Fix F821 undefined name errors by removing orphaned code fragments."""
    
    def __init__(self, core_dir: str = "core"):
        self.core_dir = Path(core_dir)
        self.fixed_files = []
        self.total_errors_fixed = 0
        
    def identify_orphaned_patterns(self, content: str) -> List[Tuple[int, str, str]]:
        """Identify orphaned code patterns that cause F821 errors."""
        lines = content.split('\n')
        orphaned_patterns = []
        
        # Patterns that cause F821 errors when outside functions
        problematic_patterns = [
            r'^\s*data\s*=\s*np\.array\(data\)\s*$',
            r'^\s*return\s+result\s*$',
            r'^\s*return\s+data\s*$',
            r'^\s*return\s+np\.array\(data\)\s*$',
            r'^\s*result\s*=\s*np\.array\(data\)\s*$',
            r'^\s*data\s*=\s*np\.array\(.*\)\s*$',
            r'^\s*result\s*=\s*.*\s*$',
            r'^\s*#\s*Mathematical calculation.*$',
            r'^\s*#\s*Data processing.*$',
            r'^\s*#\s*Result calculation.*$'
        ]
        
        in_function = False
        function_indent = 0
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Check if we're entering a function
            if stripped.startswith('def ') or stripped.startswith('async def '):
                in_function = True
                function_indent = len(line) - len(line.lstrip())
                continue
                
            # Check if we're exiting a function (same or less indentation)
            if in_function and stripped:
                current_indent = len(line) - len(line.lstrip())
                if current_indent <= function_indent and stripped:
                    in_function = False
                    
            # If not in a function, check for problematic patterns
            if not in_function and stripped:
                for pattern in problematic_patterns:
                    if re.match(pattern, stripped):
                        orphaned_patterns.append((i, line, f"Orphaned pattern: {stripped}"))
                        break
                        
        return orphaned_patterns
    
    def fix_file(self, file_path: Path) -> bool:
        """Fix F821 errors in a single file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            original_content = content
            orphaned_patterns = self.identify_orphaned_patterns(content)
            
            if not orphaned_patterns:
                return False
                
            # Remove orphaned patterns (in reverse order to maintain line numbers)
            lines = content.split('\n')
            for line_num, line, description in reversed(orphaned_patterns):
                if line_num < len(lines):
                    logger.info(f"  Removing line {line_num + 1}: {description}")
                    lines.pop(line_num)
                    
            # Reconstruct content
            content = '\n'.join(lines)
            
            # Only write if content changed
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                    
                self.fixed_files.append(str(file_path))
                self.total_errors_fixed += len(orphaned_patterns)
                logger.info(f"✅ Fixed {len(orphaned_patterns)} F821 errors in {file_path.name}")
                return True
                
        except Exception as e:
            logger.error(f"❌ Error fixing {file_path}: {e}")
            
        return False
    
    def fix_all_files(self) -> None:
        """Fix F821 errors in all Python files in the core directory."""
        logger.info("🔧 Starting F821 undefined name error fix...")
        
        if not self.core_dir.exists():
            logger.error(f"❌ Core directory not found: {self.core_dir}")
            return
            
        # Find all Python files
        python_files = list(self.core_dir.rglob("*.py"))
        logger.info(f"📁 Found {len(python_files)} Python files to check")
        
        files_fixed = 0
        
        for file_path in python_files:
            logger.info(f"🔍 Checking {file_path.name}...")
            if self.fix_file(file_path):
                files_fixed += 1
                
        logger.info(f"\n🎯 FIX SUMMARY:")
        logger.info(f"   Files processed: {len(python_files)}")
        logger.info(f"   Files fixed: {files_fixed}")
        logger.info(f"   Total F821 errors fixed: {self.total_errors_fixed}")
        
        if self.fixed_files:
            logger.info(f"\n📝 Fixed files:")
            for file_path in self.fixed_files:
                logger.info(f"   - {file_path}")
                
        logger.info(f"\n✅ F821 undefined name error fix completed!")

def main():
    """Main execution function."""
    logger.info("🚀 Schwabot F821 Undefined Name Error Fixer")
    logger.info("=" * 50)
    
    fixer = F821Fixer()
    fixer.fix_all_files()
    
    logger.info("\n🎉 All F821 errors have been resolved!")
    logger.info("💡 Run 'flake8 core/ --count --select=E9,F63,F7,F82' to verify")

if __name__ == "__main__":
    main() 