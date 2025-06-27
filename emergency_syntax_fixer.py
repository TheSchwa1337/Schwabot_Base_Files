# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Emergency Syntax Fixer
======================

Critical fixer to address immediate E999 syntax errors:
1. 'return' outside function errors
2. Missing except/finally blocks
3. Unterminated triple-quoted strings
4. Unmatched brackets and parentheses
5. Invalid indentation issues
"""

import os
import re
import ast
import logging
from typing import Dict, List, Set, Tuple, Optional, Any
from pathlib import Path
from collections import defaultdict

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmergencySyntaxFixer:
    """Emergency fixer for critical syntax errors."""
    
    def __init__(self):
        self.files_fixed = []
        self.error_counts = defaultdict(int)
        
    def fix_critical_file(self, filepath: str) -> bool:
        """Fix a single file with critical syntax errors."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Apply emergency fixes
            content = self._fix_return_outside_function(content)
            content = self._fix_missing_except_finally(content)
            content = self._fix_unterminated_strings(content)
            content = self._fix_unmatched_brackets(content)
            content = self._fix_indentation_issues(content)
            content = self._fix_invalid_syntax(content)
            
            # Only write if changes were made
            if content != original_content:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                self.files_fixed.append(filepath)
                logger.info(f"✅ Emergency fix applied: {filepath}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Emergency fix failed for {filepath}: {e}")
            return False
    
    def _fix_return_outside_function(self, content: str) -> str:
        """Fix 'return' statements outside functions."""
        lines = content.split('\n')
        fixed_lines = []
        
        for i, line in enumerate(lines):
            fixed_line = line
            
            # Check if return is outside function
            if line.strip().startswith('return ') and not self._is_inside_function(lines, i):
                # Comment out the return statement
                fixed_line = '# ' + line + '  # Fixed: return outside function'
                self.error_counts['return_fixes'] += 1
            
            fixed_lines.append(fixed_line)
        
        return '\n'.join(fixed_lines)
    
    def _is_inside_function(self, lines: List[str], line_index: int) -> bool:
        """Check if a line is inside a function definition."""
        # Look backwards for function definition
        for i in range(line_index - 1, -1, -1):
            line = lines[i].strip()
            if line.startswith('def ') and line.endswith(':'):
                return True
            elif line.startswith('class ') and line.endswith(':'):
                return False
        return False
    
    def _fix_missing_except_finally(self, content: str) -> str:
        """Fix missing except/finally blocks after try statements."""
        lines = content.split('\n')
        fixed_lines = []
        
        i = 0
        while i < len(lines):
            line = lines[i]
            fixed_lines.append(line)
            
            # Check for try statement without except/finally
            if line.strip().startswith('try:'):
                try_indent = len(line) - len(line.lstrip())
                
                # Look for the next statement at same or lower indentation
                j = i + 1
                found_except_finally = False
                
                while j < len(lines):
                    next_line = lines[j]
                    if next_line.strip() == '':
                        j += 1
                        continue
                    
                    next_indent = len(next_line) - len(next_line.lstrip())
                    
                    if next_indent <= try_indent:
                        # We've reached same or lower indentation
                        if next_line.strip().startswith(('except', 'finally')):
                            found_except_finally = True
                        break
                    
                    j += 1
                
                # Add except block if missing
                if not found_except_finally:
                    except_indent = ' ' * (try_indent + 4)
                    fixed_lines.append(f"{' ' * try_indent}except Exception as e:")
                    fixed_lines.append(f"{except_indent}pass  # TODO: Implement proper exception handling")
                    self.error_counts['except_fixes'] += 1
            
            i += 1
        
        return '\n'.join(fixed_lines)
    
    def _fix_unterminated_strings(self, content: str) -> str:
        """Fix unterminated triple-quoted strings."""
        # Fix common unterminated string patterns
        patterns = [
            # Fix unterminated triple quotes at end of file
            (r'"""[^"]*$', '"""'),
            (r"'''[^']*$", "'''"),
            
            # Fix malformed docstrings
            (r'""""""', '"""Placeholder docstring."""'),
            (r"''''''", "'''Placeholder docstring.'''"),
            
            # Fix interrupted triple quotes
            (r'"""([^"]*?)"""([^"]*?)"""', r'"""\1\2"""'),
            (r"'''([^']*?)'''([^']*?)'''", r"'''\1\2'''"),
            
            # Fix single quotes in docstrings
            (r'"""([^"]*?)"""([^"]*?)"""', r'"""\1\2"""'),
        ]
        
        for pattern, replacement in patterns:
            old_content = content
            content = re.sub(pattern, replacement, content, flags=re.DOTALL | re.MULTILINE)
            if content != old_content:
                self.error_counts['string_fixes'] += 1
        
        return content
    
    def _fix_unmatched_brackets(self, content: str) -> str:
        """Fix unmatched brackets and parentheses."""
        lines = content.split('\n')
        fixed_lines = []
        
        for line in lines:
            fixed_line = line
            
            # Count brackets and parentheses
            paren_count = line.count('(') - line.count(')')
            bracket_count = line.count('[') - line.count(']')
            brace_count = line.count('{') - line.count('}')
            
            # Fix unmatched opening brackets
            if paren_count > 0 and not line.rstrip().endswith('\\'):
                fixed_line += ')' * paren_count
                self.error_counts['bracket_fixes'] += 1
            
            if bracket_count > 0 and not line.rstrip().endswith('\\'):
                fixed_line += ']' * bracket_count
                self.error_counts['bracket_fixes'] += 1
            
            if brace_count > 0 and not line.rstrip().endswith('\\'):
                fixed_line += '}' * brace_count
                self.error_counts['bracket_fixes'] += 1
            
            fixed_lines.append(fixed_line)
        
        return '\n'.join(fixed_lines)
    
    def _fix_indentation_issues(self, content: str) -> str:
        """Fix common indentation issues."""
        lines = content.split('\n')
        fixed_lines = []
        
        for i, line in enumerate(lines):
            fixed_line = line
            
            # Fix mixed spaces and tabs
            fixed_line = fixed_line.expandtabs(4)
            
            # Fix common indentation errors
            if i > 0 and lines[i-1].strip().endswith(':'):
                # Line after colon should be indented
                if fixed_line.strip() and not fixed_line.startswith(' ') and not fixed_line.startswith('#'):
                    prev_indent = len(lines[i-1]) - len(lines[i-1].lstrip())
                    fixed_line = ' ' * (prev_indent + 4) + fixed_line.strip()
                    self.error_counts['indent_fixes'] += 1
            
            fixed_lines.append(fixed_line)
        
        return '\n'.join(fixed_lines)
    
    def _fix_invalid_syntax(self, content: str) -> str:
        """Fix other invalid syntax patterns."""
        # Fix common invalid syntax patterns
        fixes = [
            # Fix invalid assignments
            (r'=\s*=\s*', '= '),
            
            # Fix invalid operators
            (r'\s+==\s+==\s+', ' == '),
            
            # Fix invalid function calls
            (r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\(\s*\)\s*=', r'\1() # TODO: Fix assignment'),
            
            # Fix incomplete statements
            (r'^\s*pass\s*=', 'pass  # TODO: Fix incomplete statement'),
        ]
        
        for pattern, replacement in fixes:
            old_content = content
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
            if content != old_content:
                self.error_counts['syntax_fixes'] += 1
        
        return content
    
    def fix_specific_files(self) -> None:
        """Fix specific files with known critical errors."""
        critical_files = [
            'core/math/tensor_algebra/__init__.py',
            'core/phase_engine/__init__.py',
            'core/phase_engine/basket_phase_map.py',
            'schwabot/core/__init__.py',
            'schwabot/ufs_app.py',
            'schwabot/instruction_listener.py',
        ]
        
        for filepath in critical_files:
            if os.path.exists(filepath):
                self.fix_critical_file(filepath)
    
    def fix_directory_files(self, directory: str) -> None:
        """Fix all Python files in a directory."""
        if not os.path.exists(directory):
            logger.warning(f"Directory not found: {directory}")
            return
        
        for root, dirs, files in os.walk(directory):
            # Skip cache directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
            
            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    self.fix_critical_file(filepath)
    
    def run_emergency_fixes(self) -> Dict[str, Any]:
        """Run emergency fixes on critical files and directories."""
        logger.info("🚨 Starting emergency syntax fixes...")
        
        # Fix specific critical files first
        self.fix_specific_files()
        
        # Fix all schwabot core files (they have many unterminated strings)
        self.fix_directory_files('schwabot/core')
        
        # Fix phase engine files
        self.fix_directory_files('core/phase_engine')
        
        # Fix recursive engine files
        self.fix_directory_files('core/recursive_engine')
        
        # Generate report
        report = {
            'files_fixed': len(self.files_fixed),
            'error_counts': dict(self.error_counts),
            'fixes_applied': {
                'return_fixes': self.error_counts['return_fixes'],
                'except_fixes': self.error_counts['except_fixes'],
                'string_fixes': self.error_counts['string_fixes'],
                'bracket_fixes': self.error_counts['bracket_fixes'],
                'indent_fixes': self.error_counts['indent_fixes'],
                'syntax_fixes': self.error_counts['syntax_fixes']
            },
            'files_processed': self.files_fixed
        }
        
        logger.info("✅ Emergency syntax fixes completed!")
        return report

def main():
    """Main emergency fixing function."""
    logger.info("🚨 Starting Emergency Syntax Fixer...")
    
    fixer = EmergencySyntaxFixer()
    
    # Run emergency fixes
    report = fixer.run_emergency_fixes()
    
    # Print report
    logger.info("📊 Emergency Fix Report:")
    logger.info(f"   Files Fixed: {report['files_fixed']}")
    logger.info(f"   Total Fixes: {sum(report['error_counts'].values())}")
    
    for fix_type, count in report['fixes_applied'].items():
        if count > 0:
            logger.info(f"   {fix_type.replace('_', ' ').title()}: {count}")
    
    # Save detailed report
    with open('emergency_fix_report.txt', 'w') as f:
        f.write("Emergency Syntax Fix Report\n")
        f.write("===========================\n\n")
        f.write(f"Files Fixed: {report['files_fixed']}\n")
        f.write(f"Total Fixes: {sum(report['error_counts'].values())}\n\n")
        
        f.write("Emergency Fixes Applied:\n")
        for fix_type, count in report['fixes_applied'].items():
            f.write(f"  {fix_type.replace('_', ' ').title()}: {count}\n")
        
        f.write("\nFiles Processed:\n")
        for filepath in report['files_processed']:
            f.write(f"  - {filepath}\n")
    
    logger.info("📄 Emergency report saved to: emergency_fix_report.txt")
    
    return fixer

if __name__ == "__main__":
    main() 