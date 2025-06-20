#!/usr/bin/env python3
"""Comprehensive flake8 fixer for Schwabot core directory.

This script systematically fixes:
- D400: Add periods to docstring first lines
- E501: Break long lines intelligently
- E128: Fix continuation line indentation
"""

import re
import os
from pathlib import Path
from typing import List, Tuple

def fix_docstring_periods_advanced(content: str) -> str:
    """Advanced docstring period fixing."""
    # More sophisticated pattern to catch docstring first lines
    lines = content.split('\n')
    fixed_lines = []
    
    for i, line in enumerate(lines):
        # Look for docstring patterns
        if '"""' in line:
            # Check if this is a docstring first line
            if line.strip().startswith('"""') and not line.strip().endswith('"""'):
                # This is a multi-line docstring start
                if not line.strip().endswith('.'):
                    # Add period before closing quotes
                    line = line.rstrip() + '.' + line[line.rfind('"""'):]
            elif line.strip().startswith('"""') and line.strip().endswith('"""'):
                # Single line docstring
                content_part = line.strip()[3:-3].strip()
                if content_part and not content_part.endswith('.'):
                    line = line.replace(content_part, content_part + '.')
        
        fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)

def fix_long_lines_advanced(content: str) -> str:
    """Advanced long line fixing."""
    lines = content.split('\n')
    fixed_lines = []
    
    for line in lines:
        if len(line) > 79:
            # Try to break at logical points
            if '(' in line and ')' in line and line.count('(') == line.count(')'):
                # Function call - break at commas
                indent = len(line) - len(line.lstrip())
                if ',' in line:
                    parts = line.split(',')
                    if len(parts) > 1:
                        new_line = parts[0]
                        for part in parts[1:]:
                            if len(new_line + ',' + part) > 79:
                                fixed_lines.append(new_line + ',')
                                new_line = ' ' * (indent + 4) + part.lstrip()
                            else:
                                new_line += ',' + part
                        fixed_lines.append(new_line)
                        continue
            
            # Try to break long strings
            if 'f"' in line or "f'" in line:
                # F-string - try to break at logical points
                pass  # Keep as is for now
            
            # Try to break long assignments
            if ' = ' in line and len(line) > 79:
                indent = len(line) - len(line.lstrip())
                equal_pos = line.find(' = ')
                if equal_pos > 0:
                    var_part = line[:equal_pos + 3]
                    value_part = line[equal_pos + 3:]
                    if len(var_part) < 79 and len(value_part) > 40:
                        fixed_lines.append(var_part)
                        fixed_lines.append(' ' * (indent + 4) + value_part)
                        continue
        
        fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)

def fix_continuation_indentation_advanced(content: str) -> str:
    """Advanced continuation line indentation fixing."""
    lines = content.split('\n')
    fixed_lines = []
    
    for i, line in enumerate(lines):
        if i > 0:
            prev_line = lines[i-1]
            current_indent = len(line) - len(line.lstrip())
            
            # Check for continuation lines
            if (line.strip().startswith('(') and prev_line.rstrip().endswith('(')) or \
               (line.strip().startswith('[') and prev_line.rstrip().endswith('[')) or \
               (line.strip().startswith('{') and prev_line.rstrip().endswith('{')):
                
                # This should be indented properly
                base_indent = len(prev_line) - len(prev_line.lstrip())
                expected_indent = base_indent + 4
                
                if current_indent < expected_indent:
                    line = ' ' * expected_indent + line.lstrip()
            
            # Check for parameter continuation
            elif line.strip().startswith(',') and prev_line.rstrip().endswith(','):
                base_indent = len(prev_line) - len(prev_line.lstrip())
                expected_indent = base_indent
                if current_indent != expected_indent:
                    line = ' ' * expected_indent + line.lstrip()
        
        fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)

def process_file_advanced(file_path: Path) -> bool:
    """Process a single file with advanced fixes."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Apply advanced fixes
        content = fix_docstring_periods_advanced(content)
        content = fix_long_lines_advanced(content)
        content = fix_continuation_indentation_advanced(content)
        
        # Write back if changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        
        return False
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    """Main function to process core directory."""
    core_dir = Path('core')
    
    if not core_dir.exists():
        print("Core directory not found!")
        return
    
    files_processed = 0
    files_modified = 0
    
    # Process files in order of likely issues
    priority_files = [
        'simplified_btc_integration.py',
        'unified_mathematical_trading_controller.py',
        'integration_orchestrator.py'
    ]
    
    # Process priority files first
    for filename in priority_files:
        file_path = core_dir / filename
        if file_path.exists():
            files_processed += 1
            if process_file_advanced(file_path):
                files_modified += 1
                print(f"Modified: {file_path}")
    
    # Process remaining files
    for py_file in core_dir.rglob('*.py'):
        if py_file.name not in priority_files:
            files_processed += 1
            if process_file_advanced(py_file):
                files_modified += 1
                print(f"Modified: {py_file}")
    
    print(f"\nProcessed {files_processed} files")
    print(f"Modified {files_modified} files")

if __name__ == "__main__":
    main() 