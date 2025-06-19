#!/usr/bin/env python3
"""
Immediate Flake8 Fix - No Dependencies Required
Runs basic flake8 fixes immediately without external tools.
"""

import os
import glob
import time

def fix_python_file(file_path):
    """Fix common flake8 issues in a Python file."""
    try:
        print(f"Fixing {file_path}...")
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        original_content = content
        lines = content.split('\n')
        fixed_lines = []
        changes_made = 0
        
        for i, line in enumerate(lines):
            original_line = line
            
            # Fix trailing whitespace (W291, W293)
            line = line.rstrip()
            
            # Fix simple line length issues for imports
            if len(line) > 79 and 'import ' in line:
                if 'from ' in line and ' import ' in line:
                    parts = line.split(' import ')
                    if len(parts) == 2:
                        from_part = parts[0]
                        import_part = parts[1]
                        
                        # Break long import lists
                        if len(import_part) > 50 and ',' in import_part:
                            imports = [imp.strip() for imp in import_part.split(',')]
                            if len(imports) > 1:
                                fixed_lines.append(f"{from_part} import (")
                                for j, imp in enumerate(imports):
                                    comma = ',' if j < len(imports) - 1 else ''
                                    fixed_lines.append(f"    {imp}{comma}")
                                fixed_lines.append(")")
                                if line != original_line:
                                    changes_made += 1
                                continue
            
            # Add missing common imports
            if i == 0 or (i < 10 and (line.strip().startswith('import ') or line.strip().startswith('from '))):
                # Check if we need to add common imports
                pass  # Will be handled later
            
            if line != original_line:
                changes_made += 1
            
            fixed_lines.append(line)
        
        # Ensure file ends with newline (W292)
        content = '\n'.join(fixed_lines)
        if content and not content.endswith('\n'):
            content += '\n'
            changes_made += 1
        
        # Add missing imports based on content
        if 'Mock(' in content and 'from unittest.mock import Mock' not in content:
            # Insert import after any existing imports
            content_lines = content.split('\n')
            import_index = 0
            for j, line in enumerate(content_lines):
                if line.strip().startswith(('import ', 'from ')):
                    import_index = j + 1
                elif line.strip() == '':
                    continue
                else:
                    break
            
            content_lines.insert(import_index, 'from unittest.mock import Mock')
            content = '\n'.join(content_lines)
            changes_made += 1
        
        # Write back if changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"  ✓ Fixed {changes_made} issues in {file_path}")
            return True
        else:
            print(f"  - No changes needed for {file_path}")
            return False
            
    except Exception as e:
        print(f"  ✗ Error fixing {file_path}: {e}")
        return False

def main():
    print("🚀 IMMEDIATE FLAKE8 FIX")
    print("=" * 40)
    print("Fixing common flake8 issues without external dependencies...")
    print()
    
    start_time = time.time()
    
    # Get all Python files
    python_files = []
    patterns = ['*.py', 'core/*.py', 'tests/*.py', 'test_*.py']
    
    for pattern in patterns:
        python_files.extend(glob.glob(pattern))
    
    # Remove duplicates and sort
    python_files = sorted(list(set(python_files)))
    
    if not python_files:
        print("No Python files found!")
        return
    
    print(f"Found {len(python_files)} Python files to process")
    print()
    
    fixed_count = 0
    for file_path in python_files:
        if fix_python_file(file_path):
            fixed_count += 1
    
    processing_time = time.time() - start_time
    
    print()
    print("=" * 40)
    print("🎉 IMMEDIATE FIX COMPLETE")
    print("=" * 40)
    print(f"⏱️  Processing time: {processing_time:.2f} seconds")
    print(f"📁 Files processed: {len(python_files)}")
    print(f"✅ Files fixed: {fixed_count}")
    print(f"📈 Fix rate: {(fixed_count/max(len(python_files), 1)*100):.1f}%")
    
    print()
    print("💡 NEXT STEPS:")
    print("1. Run the comprehensive fixer: python master_flake8_comprehensive_fixer.py")
    print("2. Install flake8 tools: pip install flake8 autopep8 isort")
    print("3. Run full setup: python setup_and_fix_flake8.py")
    print("4. Check remaining issues: flake8 . --statistics")

if __name__ == '__main__':
    main() 