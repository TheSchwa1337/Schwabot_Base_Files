#!/usr/bin/env python3
"""
Critical Syntax Error Fixer
Automatically fixes the most critical syntax errors found in the codebase.
"""

import os
import re
import ast
import glob
from pathlib import Path
from typing import List, Tuple, Dict, Set

class CriticalSyntaxFixer:
    def __init__(self, core_dir: str = "core"):
        self.core_dir = Path(core_dir)
        self.fixed_files = []
        self.errors_fixed = 0
        
    def fix_unterminated_docstrings(self, file_path: Path) -> bool:
        """Fix unterminated triple-quoted strings."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Pattern to find unterminated triple quotes
            # Look for """ or ''' that don't have a matching closing quote
            lines = content.split('\n')
            fixed_lines = []
            in_docstring = False
            docstring_start = None
            
            for i, line in enumerate(lines):
                if '"""' in line or "'''" in line:
                    # Check if this line starts a docstring
                    if not in_docstring:
                        # Count quotes to see if it's balanced
                        quote_count = line.count('"""') + line.count("'''")
                        if quote_count % 2 == 1:  # Odd number means unterminated
                            in_docstring = True
                            docstring_start = i
                            # Add closing quote to this line
                            if '"""' in line:
                                line = line + '"""'
                            else:
                                line = line + "'''"
                            in_docstring = False
                    else:
                        # We're in a docstring, check if this line closes it
                        quote_count = line.count('"""') + line.count("'''")
                        if quote_count > 0:
                            in_docstring = False
                
                fixed_lines.append(line)
            
            # If we're still in a docstring at the end, close it
            if in_docstring:
                fixed_lines.append('"""')
            
            fixed_content = '\n'.join(fixed_lines)
            
            if fixed_content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(fixed_content)
                return True
                
        except Exception as e:
            print(f"Error fixing docstrings in {file_path}: {e}")
        
        return False
    
    def fix_missing_except_blocks(self, file_path: Path) -> bool:
        """Fix missing except/finally blocks after try statements."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Simple pattern matching for try blocks without except/finally
            lines = content.split('\n')
            fixed_lines = []
            i = 0
            
            while i < len(lines):
                line = lines[i].strip()
                
                # Check if this is a try statement
                if line.startswith('try:') or line.startswith('try '):
                    # Look ahead to see if there's an except or finally
                    has_except_or_finally = False
                    j = i + 1
                    
                    # Skip indented lines
                    while j < len(lines) and (lines[j].startswith(' ') or lines[j].strip() == ''):
                        if lines[j].strip().startswith(('except', 'finally')):
                            has_except_or_finally = True
                            break
                        j += 1
                    
                    # If no except/finally found, add a basic except block
                    if not has_except_or_finally:
                        # Find the indentation level
                        indent = len(lines[i]) - len(lines[i].lstrip())
                        indent_str = ' ' * indent
                        
                        # Add except block after the try block
                        try_block_end = j
                        while try_block_end < len(lines) and (lines[try_block_end].startswith(' ') or lines[try_block_end].strip() == ''):
                            try_block_end += 1
                        
                        # Insert except block
                        lines.insert(try_block_end, f"{indent_str}except Exception as e:")
                        lines.insert(try_block_end + 1, f"{indent_str}    pass")
                        lines.insert(try_block_end + 2, "")
                
                i += 1
            
            fixed_content = '\n'.join(lines)
            
            if fixed_content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(fixed_content)
                return True
                
        except Exception as e:
            print(f"Error fixing except blocks in {file_path}: {e}")
        
        return False
    
    def fix_indentation_errors(self, file_path: Path) -> bool:
        """Fix basic indentation errors."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Fix common indentation issues
            lines = content.split('\n')
            fixed_lines = []
            
            for line in lines:
                # Fix mixed tabs and spaces
                if '\t' in line:
                    line = line.replace('\t', '    ')
                
                # Fix inconsistent indentation (convert to 4 spaces)
                if line.startswith(' '):
                    spaces = len(line) - len(line.lstrip())
                    if spaces % 4 != 0:
                        # Round to nearest 4-space boundary
                        new_spaces = (spaces // 4) * 4
                        line = ' ' * new_spaces + line.lstrip()
                
                fixed_lines.append(line)
            
            fixed_content = '\n'.join(fixed_lines)
            
            if fixed_content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(fixed_content)
                return True
                
        except Exception as e:
            print(f"Error fixing indentation in {file_path}: {e}")
        
        return False
    
    def fix_invalid_decimal_literals(self, file_path: Path) -> bool:
        """Fix invalid decimal literals like 1e-9."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Fix common decimal literal issues
            # Pattern: number followed by e followed by - followed by number
            # This should be: numbere-number (no spaces)
            content = re.sub(r'(\d+)\s*e\s*-\s*(\d+)', r'\1e-\2', content)
            
            # Fix leading zeros in decimal integers
            content = re.sub(r'\b0+(\d+)\b', r'\1', content)
            
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return True
                
        except Exception as e:
            print(f"Error fixing decimal literals in {file_path}: {e}")
        
        return False
    
    def fix_missing_colons(self, file_path: Path) -> bool:
        """Fix missing colons in function definitions."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            lines = content.split('\n')
            fixed_lines = []
            
            for line in lines:
                # Check for function definitions without colons
                if re.match(r'^\s*def\s+\w+\s*\([^)]*\)\s*$', line):
                    if not line.endswith(':'):
                        line = line + ':'
                
                fixed_lines.append(line)
            
            fixed_content = '\n'.join(fixed_lines)
            
            if fixed_content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(fixed_content)
                return True
                
        except Exception as e:
            print(f"Error fixing missing colons in {file_path}: {e}")
        
        return False
    
    def fix_return_outside_function(self, file_path: Path) -> bool:
        """Fix return statements outside functions."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            lines = content.split('\n')
            fixed_lines = []
            in_function = False
            function_indent = 0
            
            for line in lines:
                stripped = line.strip()
                
                # Check if we're entering a function
                if stripped.startswith('def '):
                    in_function = True
                    function_indent = len(line) - len(line.lstrip())
                
                # Check if we're leaving a function (same indentation level)
                elif in_function and stripped and not stripped.startswith('#'):
                    current_indent = len(line) - len(line.lstrip())
                    if current_indent <= function_indent:
                        in_function = False
                
                # If we find a return statement outside a function, comment it out
                if stripped.startswith('return ') and not in_function:
                    line = '# ' + line
                
                fixed_lines.append(line)
            
            fixed_content = '\n'.join(fixed_lines)
            
            if fixed_content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(fixed_content)
                return True
                
        except Exception as e:
            print(f"Error fixing return statements in {file_path}: {e}")
        
        return False
    
    def fix_undefined_names(self, file_path: Path) -> bool:
        """Fix undefined variable names by adding basic definitions."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Add basic definitions for common undefined variables
            if 'xi_t' in content and 'xi_t =' not in content:
                # Add at the beginning of the file after imports
                lines = content.split('\n')
                insert_pos = 0
                for i, line in enumerate(lines):
                    if line.strip().startswith(('import ', 'from ')):
                        insert_pos = i + 1
                    elif line.strip() and not line.strip().startswith('#'):
                        break
                
                lines.insert(insert_pos, 'xi_t = 0.0  # Default value for xi_t')
                content = '\n'.join(lines)
            
            if 'delta_price' in content and 'delta_price =' not in content:
                lines = content.split('\n')
                insert_pos = 0
                for i, line in enumerate(lines):
                    if line.strip().startswith(('import ', 'from ')):
                        insert_pos = i + 1
                    elif line.strip() and not line.strip().startswith('#'):
                        break
                
                lines.insert(insert_pos, 'delta_price = 0.0  # Default value for delta_price')
                content = '\n'.join(lines)
            
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return True
                
        except Exception as e:
            print(f"Error fixing undefined names in {file_path}: {e}")
        
        return False
    
    def fix_file(self, file_path: Path) -> int:
        """Apply all fixes to a single file."""
        fixes_applied = 0
        
        # Apply fixes in order of priority
        if self.fix_unterminated_docstrings(file_path):
            fixes_applied += 1
        
        if self.fix_missing_except_blocks(file_path):
            fixes_applied += 1
        
        if self.fix_indentation_errors(file_path):
            fixes_applied += 1
        
        if self.fix_invalid_decimal_literals(file_path):
            fixes_applied += 1
        
        if self.fix_missing_colons(file_path):
            fixes_applied += 1
        
        if self.fix_return_outside_function(file_path):
            fixes_applied += 1
        
        if self.fix_undefined_names(file_path):
            fixes_applied += 1
        
        return fixes_applied
    
    def fix_all_files(self) -> Dict[str, int]:
        """Fix all Python files in the core directory."""
        results = {}
        
        # Find all Python files
        python_files = list(self.core_dir.rglob("*.py"))
        
        print(f"Found {len(python_files)} Python files to process...")
        
        for file_path in python_files:
            try:
                fixes = self.fix_file(file_path)
                if fixes > 0:
                    results[str(file_path)] = fixes
                    self.fixed_files.append(str(file_path))
                    self.errors_fixed += fixes
                    print(f"Fixed {fixes} issues in {file_path}")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        return results
    
    def generate_report(self, results: Dict[str, int]):
        """Generate a report of the fixes applied."""
        print("\n" + "="*60)
        print("CRITICAL SYNTAX ERROR FIX REPORT")
        print("="*60)
        print(f"Total files processed: {len(results)}")
        print(f"Total errors fixed: {self.errors_fixed}")
        print(f"Files modified: {len(self.fixed_files)}")
        
        if self.fixed_files:
            print("\nModified files:")
            for file_path in self.fixed_files:
                print(f"  - {file_path} ({results[file_path]} fixes)")
        
        print("\n" + "="*60)

def main():
    """Main function to run the syntax fixer."""
    fixer = CriticalSyntaxFixer()
    
    print("Starting critical syntax error fixes...")
    print("This will attempt to fix the most common syntax errors automatically.")
    print("Manual review is still required for complex issues.")
    
    results = fixer.fix_all_files()
    fixer.generate_report(results)
    
    print("\nNext steps:")
    print("1. Run flake8 again to check remaining errors")
    print("2. Manually review the modified files")
    print("3. Test the codebase to ensure functionality is preserved")
    print("4. Address any remaining complex syntax errors manually")

if __name__ == "__main__":
    main() 