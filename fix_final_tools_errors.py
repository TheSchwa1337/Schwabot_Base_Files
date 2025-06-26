#!/usr/bin/env python3
"""
Fix final minor errors in tools directory:
- E128: Continuation line under-indented for visual indent
- F541: f-string is missing placeholders
"""

import re

def fix_final_tools_errors():
    """Fix the final minor errors in tools files."""
    
    print("🔧 Fixing final minor errors in tools directory")
    
    # Fix uros_v1_integration_test.py
    file_path = "tools/uros_v1_integration_test.py"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Fix E128: Fix indentation of continuation lines
    lines = content.split('\n')
    fixed_lines = []
    
    for i, line in enumerate(lines):
        # Fix line 774: if isinstance(result, bool) and result)
        if i == 773 and 'if isinstance(result, bool) and result)' in line:
            # Add proper indentation
            fixed_line = ' ' * 28 + 'if isinstance(result, bool) and result)'
            print(f"  🔧 Fixed E128 indentation at line {i+1}")
        # Fix line 776: if isinstance(result, bool))
        elif i == 775 and 'if isinstance(result, bool))' in line:
            # Add proper indentation
            fixed_line = ' ' * 24 + 'if isinstance(result, bool))'
            print(f"  🔧 Fixed E128 indentation at line {i+1}")
        # Fix F541: f-string missing placeholders
        elif 'safe_print(f"\\n📊 TEST SUMMARY:")' in line:
            # Remove f-string since no placeholders
            fixed_line = line.replace('f"\\n📊 TEST SUMMARY:"', '"\\n📊 TEST SUMMARY:"')
            print(f"  🔧 Fixed F541 f-string at line {i+1}")
        else:
            fixed_line = line
        
        fixed_lines.append(fixed_line)
    
    content = '\n'.join(fixed_lines)
    
    # Check if changes were made
    if content != original_content:
        # Backup the original file
        backup_path = f"{file_path}.backup"
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(original_content)
        
        # Write the fixed content
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ Fixed final errors in {file_path}")
    else:
        print(f"ℹ️ No final errors found in {file_path}")

if __name__ == "__main__":
    fix_final_tools_errors() 