#!/usr/bin/env python3
"""
Unicode Character and String Literal Fix Script

This script fixes the remaining E999 errors caused by:
1. Invalid Unicode characters (\\u221e, \\u00b2, etc.) in mathematical expressions
2. Unterminated string literals and docstrings
3. Invalid syntax in stub files
"""

import os
import re
import glob
from pathlib import Path
from typing import List, Tuple, Dict

def fix_unicode_characters(content: str) -> str:
    """Fix invalid Unicode characters in mathematical expressions."""
    # Common Unicode character replacements
    unicode_replacements = {
        '\\u221e': 'infinity',  # Infinity symbol
        '\\u00b2': '**2',       # Squared
        '\\u00b3': '**3',       # Cubed
        '\\u00b1': '+/-',       # Plus-minus
        '\\u2264': '<=',        # Less than or equal
        '\\u2265': '>=',        # Greater than or equal
        '\\u2260': '!=',        # Not equal
        '\\u2248': '~',         # Approximately equal
        '\\u2211': 'sum',       # Summation
        '\\u220f': 'prod',      # Product
        '\\u222b': 'integral',  # Integral
        '\\u2202': 'partial',   # Partial derivative
        '\\u2207': 'gradient',  # Gradient
        '\\u0394': 'delta',     # Delta
        '\\u03bb': 'lambda',    # Lambda
        '\\u03bc': 'mu',        # Mu
        '\\u03c3': 'sigma',     # Sigma
        '\\u03c4': 'tau',       # Tau
        '\\u03c6': 'phi',       # Phi
        '\\u03c8': 'psi',       # Psi
        '\\u03c9': 'omega',     # Omega
        '\\u03b1': 'alpha',     # Alpha
        '\\u03b2': 'beta',      # Beta
        '\\u03b3': 'gamma',     # Gamma
        '\\u03b4': 'delta',     # Delta
        '\\u03b5': 'epsilon',   # Epsilon
        '\\u03b6': 'zeta',      # Zeta
        '\\u03b7': 'eta',       # Eta
        '\\u03b8': 'theta',     # Theta
        '\\u03b9': 'iota',      # Iota
        '\\u03ba': 'kappa',     # Kappa
        '\\u03bd': 'nu',        # Nu
        '\\u03be': 'xi',        # Xi
        '\\u03bf': 'omicron',   # Omicron
        '\\u03c0': 'pi',        # Pi
        '\\u03c1': 'rho',       # Rho
        '\\u03c2': 'sigma_final', # Final sigma
        '\\u03c5': 'upsilon',   # Upsilon
        '\\u03c7': 'chi',       # Chi
    }
    
    # Apply replacements
    for unicode_char, replacement in unicode_replacements.items():
        content = content.replace(unicode_char, replacement)
    
    return content

def fix_unterminated_strings(content: str) -> str:
    """Fix unterminated string literals and docstrings."""
    lines = content.split('\n')
    fixed_lines = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check for unterminated triple quotes
        if '"""' in line:
            quote_count = line.count('"""')
            if quote_count % 2 == 1:
                # Unterminated triple quote, add closing
                line = line + '"""'
        
        elif "'''" in line:
            quote_count = line.count("'''")
            if quote_count % 2 == 1:
                # Unterminated triple quote, add closing
                line = line + "'''"
        
        # Check for unterminated single/double quotes
        elif line.count('"') % 2 == 1:
            # Unterminated double quote
            line = line + '"'
        
        elif line.count("'") % 2 == 1:
            # Unterminated single quote
            line = line + "'"
        
        fixed_lines.append(line)
        i += 1
    
    return '\n'.join(fixed_lines)

def fix_stub_file_syntax(content: str) -> str:
    """Fix invalid syntax in stub files."""
    lines = content.split('\n')
    fixed_lines = []
    
    # Remove problematic stub generation comments
    for line in lines:
        # Remove lines that cause syntax errors in stubs
        if line.strip().startswith('The original file failed to parse'):
            continue
        elif line.strip().startswith('a stub was generated so the package'):
            continue
        elif line.strip().startswith('could be imported'):
            continue
        else:
            fixed_lines.append(line)
    
    # Ensure file ends with newline
    if fixed_lines and not fixed_lines[-1].endswith('\n'):
        fixed_lines.append('')
    
    return '\n'.join(fixed_lines)

def create_proper_stub(content: str, file_path: str) -> str:
    """Create a proper stub file with standard template."""
    module_name = Path(file_path).stem
    
    # If content is empty or just comments, create proper stub
    if not content.strip() or all(line.strip().startswith('#') or not line.strip() for line in content.split('\n')):
        stub_content = f'''"""
{module_name.replace('_', ' ').title()} Module

This module provides {module_name.replace('_', ' ')} functionality for the Schwabot system.
"""

import logging
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

def initialize_{module_name}():
    """
    Initialize the {module_name} module.
    
    Returns:
        bool: True if initialization successful
    """
    logger.info(f"Initializing {{module_name}} module")
    return True

def placeholder_{module_name}_function():
    """
    Placeholder function for {module_name} module.
    
    This function is pending mathematical implementation.
    """
    raise NotImplementedError(f"This module is pending mathematical implementation.")

# Module initialization
if __name__ == "__main__":
    initialize_{module_name}()
'''
        return stub_content
    
    return content

def fix_unicode_string_errors(file_path: str) -> Tuple[bool, List[str]]:
    """Fix Unicode character and string literal errors in a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        changes_made = []
        
        # Apply fixes
        content = fix_unicode_characters(content)
        if content != original_content:
            changes_made.append("Fixed Unicode characters")
            original_content = content
        
        content = fix_unterminated_strings(content)
        if content != original_content:
            changes_made.append("Fixed unterminated strings")
            original_content = content
        
        content = fix_stub_file_syntax(content)
        if content != original_content:
            changes_made.append("Fixed stub file syntax")
            original_content = content
        
        # Create proper stub if needed
        content = create_proper_stub(content, file_path)
        if content != original_content:
            changes_made.append("Created proper stub file")
        
        # Only write if content changed
        if changes_made:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True, changes_made
        
        return False, ["No changes needed"]
        
    except Exception as e:
        return False, [f"Error processing file: {str(e)}"]

def main():
    """Main function to fix Unicode and string literal errors."""
    print("\\u1f527 Starting Unicode Character and String Literal Fix...")
    print("=" * 60)
    
    # Get all Python files in core directory
    core_files = glob.glob('core/**/*.py', recursive=True)
    
    fixed_files = []
    error_files = []
    
    for file_path in core_files:
        print(f"Processing: {file_path}")
        success, messages = fix_unicode_string_errors(file_path)
        
        if success:
            fixed_files.append(file_path)
            print(f"  \\u2705 Fixed: {', '.join(messages)}")
        else:
            if "Error processing" in messages[0]:
                error_files.append((file_path, messages[0]))
                print(f"  \\u274c Error: {messages[0]}")
            else:
                print(f"  \\u23ed\\ufe0f  Skipped: {messages[0]}")
    
    print("\n" + "=" * 60)
    print("\\u1f4ca UNICODE/STRING FIX SUMMARY")
    print("=" * 60)
    print(f"Files Processed: {len(core_files)}")
    print(f"Files Fixed: {len(fixed_files)}")
    print(f"Files with Errors: {len(error_files)}")
    
    if fixed_files:
        print(f"\\n\\u2705 Successfully Fixed Files:")
        for file_path in fixed_files[:10]:  # Show first 10
            print(f"  - {file_path}")
        if len(fixed_files) > 10:
            print(f"  ... and {len(fixed_files) - 10} more")
    
    if error_files:
        print(f"\\n\\u274c Files with Processing Errors:")
        for file_path, error_msg in error_files[:5]:  # Show first 5
            print(f"  - {file_path}: {error_msg}")
        if len(error_files) > 5:
            print(f"  ... and {len(error_files) - 5} more")
    
    print(f"\\n\\u1f389 Unicode/String fix complete!")
    print(f"Next: Run 'flake8 core/ --count --select=E999' to verify improvements")

if __name__ == "__main__":
    main() 
"""