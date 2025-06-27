
from dual_unicore_handler import DualUnicoreHandler

# Initialize Unicode handler
unicore = DualUnicoreHandler()

# -*- coding: utf-8 -*-
""""""
""""""
""""""
""""""
"""
Comprehensive Unicode Patch Script
Fixes all Unicode character issues in Python files that cause E999 syntax errors."""
""""""
""""""
""""""
""""""
"""

import os
import re
import glob
from pathlib import Path

def fix_unicode_characters(content):"""
"""Fix various Unicode characters that cause syntax errors.""""""
""""""
""""""
""""""
"""
    
# Common problematic Unicode characters and their replacements
unicode_fixes = {
        # Mathematical symbols
'*': '*',  # multiplication sign
        '/': '/',  # division sign
        '+/-': '+/-',  # plus-minus sign
        '<=': '<=',  # less than or equal
        '>=': '>=',  # greater than or equal
        '!=': '!=',  # not equal
        '~=': '~=',  # approximately equal
        
# Punctuation and spacing
'...': '...',  # ellipsis
        '-': '-',  # en dash
        '-': '-',  # em dash
        '*': '*',  # bullet point
        '*': '*',  # middle dot
        'deg': 'deg',  # degree symbol"""
        ''': "'",  # prime
        '"': '"',  # double prime
        
# Currency and symbols
'EUR': 'EUR',  # euro
        'GBP': 'GBP',  # pound
        'JPY': 'JPY',  # yen
        'cent': 'cent',  # cent
        'section': 'section',  # section
        '(c)': '(c)',  # copyright
        '(R)': '(R)',  # registered trademark
        '(TM)': '(TM)',  # trademark
        
# Greek letters (common in math)
        'alpha': 'alpha',
        'beta': 'beta',
        'gamma': 'gamma',
        'delta': 'delta',
        'epsilon': 'epsilon',
        'theta': 'theta',
        'lambda': 'lambda',
        'mu': 'mu',
        'pi': 'pi',
        'sigma': 'sigma',
        'phi': 'phi',
        'omega': 'omega',
        
# Subscripts and superscripts
'_1': '_1',
        '_2': '_2',
        '_3': '_3',
        '^1': '^1',
        '^2': '^2',
        '^3': '^3',
        
# Other problematic characters
'-': '-',  # en dash
        '-': '-',  # em dash
        '...': '...',  # horizontal ellipsis
        ' ': ' ',  # narrow no-break space
        ' ': ' ',  # fullwidth space
        ' ': ' ',  # ideographic space
    
# Apply fixes
for unicode_char, replacement in unicode_fixes.items():
        content = content.replace(unicode_char, replacement)
    
# Fix invalid decimal literals (e.g., 1.2_3)
    content = re.sub(r'(\\d+\.\\d+)\.(\\d+)', r'\1_\2', content)
    
# Fix invalid Unicode escapes in strings
content = re.sub(r'\\u[0-9a-fA-F]{4}', lambda m: m.group(0).replace('\\u', '\\\\u'), content)
    
return content

def fix_unterminated_strings(content):
    """Fix unterminated strings that cause syntax errors.""""""
""""""
""""""
""""""
"""
    
# Fix triple quotes that are not properly closed
lines = content.split('\n')
    fixed_lines = []
    in_triple_quote = False
    quote_type = None
    
for line in lines:
        # Check for triple quote start"""
if '"""' in line and not in_triple_quote:"
in_triple_quote = True"""
            quote_type = '"""'"""
        elif "'''" in line and not in_triple_quote:'
in_triple_quote = True'''
            quote_type = "'''"'
        
# Check for triple quote end
if in_triple_quote and quote_type in line:
            in_triple_quote = False
            quote_type = None
        '''
# If we're in a triple quote and reach end of file, close it
        if in_triple_quote and line == lines[-1]:
            line += quote_type
        
fixed_lines.append(line)
    
return '\n'.join(fixed_lines)

def fix_invalid_syntax_in_comments(content):
    """Fix invalid syntax in comments and docstrings.""""""
""""""
""""""
""""""
"""
    
# Remove invalid Unicode characters from comments
lines = content.split('\n')
    fixed_lines = []
    
for line in lines:
        if line.strip().startswith('#'):
            # Fix comment lines
for char in ['*', '/', '+/-', '<=', '>=', '!=', '~=', '...', '-', '-', '*', '*', 'deg']:
                line = line.replace(char, '')
        fixed_lines.append(line)
    
return '\n'.join(fixed_lines)

def process_file(file_path):"""
    """Process a single Python file.""""""
""""""
""""""
""""""
"""
try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
original_content = content
        
# Apply all fixes
content = fix_unicode_characters(content)
        content = fix_unterminated_strings(content)
        content = fix_invalid_syntax_in_comments(content)
        
# Only write if content changed
if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        
return False
    
except Exception as e:"""
print(f"Error processing {file_path}: {e}")
        return False

def main():
    """Main function to process all Python files.""""""
""""""
""""""
""""""
""""""
print("Comprehensive Unicode Patch Script")
    print("=" * 50)
    
# Find all Python files
python_files = []
    for root, dirs, files in os.walk('.'):
        # Skip common directories that shouldn't contain Python files
dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', 'node_modules', '.venv', 'venv']]
        
for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
print(f"Found {len(python_files)} Python files to process")
    
fixed_count = 0
    error_count = 0
    
for file_path in python_files:
        try:
            if process_file(file_path):
                fixed_count += 1
                print(f"Fixed: {file_path}")
        except Exception as e:
            error_count += 1
            print(f"Error processing {file_path}: {e}")
    
print(f"\\nSummary:")
    print(f"Files processed: {len(python_files)}")
    print(f"Files fixed: {fixed_count}")
    print(f"Errors encountered: {error_count}")
    
if fixed_count > 0:
        print(f"\\nSuccessfully fixed Unicode issues in {fixed_count} files!")
    else:
        print("\\nNo files needed fixing.")

if __name__ == "__main__":
    main() 
""""""
""""""
""""""
""""""
""""""
"""
"""