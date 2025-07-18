#!/usr/bin/env python3
import ast
import sys


def test_file_syntax(filename):
    try:
        with open(filename, 'r') as f:
            content = f.read()
        ast.parse(content)
        print(f"✓ {filename} - Syntax OK")
        return True
    except SyntaxError as e:
        print(f"✗ {filename} - Syntax Error: {e}")
        return False
    except Exception as e:
        print(f"? {filename} - Other Error: {e}")
        return False

# Test the files I claimed had syntax errors
files_to_test = [
    'core/risk_manager.py',
    'core/gpu_handlers.py',
    'core/fractal_core.py'
]

all_good = True
for file in files_to_test:
    if not test_file_syntax(file):
        all_good = False

if all_good:
    print("\n🎉 All files have valid syntax!")
else:
    print("\n❌ Some files have syntax errors")

sys.exit(0 if all_good else 1) 