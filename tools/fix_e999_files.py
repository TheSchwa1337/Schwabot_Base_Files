#!/usr/bin/env python3
"""Fix all files with E999 syntax errors."""

import subprocess
import sys
from pathlib import Path


def get_e999_files():
    """Get list of files with E999 errors."""
    try:
        result = subprocess.run(
            ["flake8", "--select=E999", "core", "tools"],
            capture_output=True,
            text=True
        )
        
        files = set()
        for line in result.stdout.strip().split('\n'):
            if line and ':' in line:
                filepath = line.split(':')[0]
                files.add(filepath)
        
        return sorted(files)
    except Exception as e:
        print(f"Error running flake8: {e}")
        return []


def fix_file(filepath: str) -> bool:
    """Fix a single file with E999 errors."""
    try:
        path = Path(filepath)
        if not path.exists():
            return False
            
        content = path.read_text(encoding='utf-8', errors='replace')
        original = content
        
        # Common patterns that cause E999 errors
        
        # 1. Fix """.""" pattern
        content = content.replace('"""."""', '"""Stub function."""')
        
        # 2. Fix unterminated triple quotes
        if '"""' in content:
            quotes = content.count('"""')
            if quotes % 2 == 1:
                content += '\n"""'
        
        # 3. Fix unicode characters that cause syntax errors
        unicode_fixes = {
            'Γêç': '∇',  # nabla
            '┬╖': '·',   # middle dot
            'Γêê': '∈',  # element of
            '≡ƒôè': 'è',
            '≡ƒôê': 'ê', 
            '≡ƒôë': 'ë',
            '≡ƒö¼': 'ü',
        }
        
        for bad, good in unicode_fixes.items():
            content = content.replace(bad, good)
        
        # 4. Fix common syntax patterns
        
        # Fix missing function bodies
        import re
        content = re.sub(
            r'(def\s+\w+\([^)]*\)\s*(?:->\s*[^:]+)?\s*:\s*"""[^"]*""")(\s*)(if\s|class\s|\w+\s*=|\Z)',
            r'\1\n    pass\n\n\3',
            content,
            flags=re.MULTILINE
        )
        
        # Fix invalid decimal literals (common issue)
        content = re.sub(r'(\d+)\.(\d+)\.(\d+)', r'\1_\2_\3', content)
        
        # Fix invalid syntax patterns
        content = re.sub(r'(\w+)\s*=\s*([\'"])([^\'"]*)\2([^\s\n])', r'\1 = \2\3\2', content)
        
        if content != original:
            path.write_text(content, encoding='utf-8')
            print(f"Fixed: {filepath}")
            return True
            
    except Exception as e:
        print(f"Error fixing {filepath}: {e}")
        
    return False


def main():
    """Main function."""
    print("Finding files with E999 errors...")
    e999_files = get_e999_files()
    
    if not e999_files:
        print("No E999 errors found!")
        return
    
    print(f"Found {len(e999_files)} files with E999 errors")
    
    fixed_count = 0
    for filepath in e999_files:
        if fix_file(filepath):
            fixed_count += 1
    
    print(f"\nFixed {fixed_count} of {len(e999_files)} files")
    
    # Run flake8 again to check progress
    print("\nChecking remaining E999 errors...")
    try:
        result = subprocess.run(
            ["flake8", "--select=E999", "--count", "core", "tools"],
            capture_output=True,
            text=True
        )
        remaining = result.stdout.strip()
        if remaining and remaining != "0":
            print(f"Remaining E999 errors: {remaining}")
        else:
            print("All E999 errors fixed!")
    except Exception:
        pass


if __name__ == "__main__":
    main() 