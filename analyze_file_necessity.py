#!/usr/bin/env python3
"""Analyze which files with syntax errors are actually needed."""

import os
import ast
from pathlib import Path
from typing import Dict, List, Set, Tuple

def is_stub_file(filepath: str) -> bool:
    """Check if file is a stub file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            return "TEMPORARY STUB GENERATED AUTOMATICALLY" in first_line
    except:
        return False

def has_syntax_error(filepath: str) -> bool:
    """Check if file has syntax errors."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            ast.parse(f.read())
        return False
    except SyntaxError:
        return True
    except:
        return False

def find_imports_in_file(filepath: str) -> Set[str]:
    """Find all imports in a file."""
    imports = set()
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read())
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module)
    except:
        pass
    return imports

def analyze_file_necessity() -> Dict[str, List[str]]:
    """Analyze which files are needed vs. removable."""
    
    # Read the syntax error list
    syntax_error_files = []
    try:
        with open('current_e999_errors.txt', 'r') as f:
            for line in f:
                if line.strip() and line.startswith('.'):
                    filepath = line.split(':')[0][2:]  # Remove .\ prefix
                    syntax_error_files.append(filepath)
    except FileNotFoundError:
        print("❌ current_e999_errors.txt not found")
        return {}
    
    print(f"📊 Analyzing {len(syntax_error_files)} files with syntax errors...")
    
    # Categorize files
    stub_files = []
    analysis_scripts = []
    core_files = []
    other_files = []
    
    for filepath in syntax_error_files:
        if is_stub_file(filepath):
            stub_files.append(filepath)
        elif any(keyword in filepath.lower() for keyword in ['analyze', 'count', 'filter', 'report', 'test_', 'demo_']):
            analysis_scripts.append(filepath)
        elif filepath.startswith('core/') or filepath.startswith('config/') or filepath.startswith('components/'):
            core_files.append(filepath)
        else:
            other_files.append(filepath)
    
    # Check which core files are actually imported
    print("\n🔍 Checking import dependencies...")
    
    # Get all Python files in the project
    all_python_files = []
    for root, dirs, files in os.walk('.'):
        if '.git' in root or '__pycache__' in root:
            continue
        for file in files:
            if file.endswith('.py'):
                all_python_files.append(os.path.join(root, file))
    
    # Find all imports across the project
    all_imports = set()
    for filepath in all_python_files:
        imports = find_imports_in_file(filepath)
        all_imports.update(imports)
    
    # Check which core files are actually imported
    imported_core_files = []
    non_imported_core_files = []
    
    for filepath in core_files:
        module_name = filepath.replace('/', '.').replace('.py', '')
        if module_name in all_imports:
            imported_core_files.append(filepath)
        else:
            non_imported_core_files.append(filepath)
    
    # Generate report
    results = {
        'stub_files': stub_files,
        'analysis_scripts': analysis_scripts,
        'imported_core_files': imported_core_files,
        'non_imported_core_files': non_imported_core_files,
        'other_files': other_files
    }
    
    print("\n📋 ANALYSIS RESULTS:")
    print("=" * 60)
    print(f"🔴 Stub files (can be removed): {len(stub_files)}")
    print(f"📊 Analysis scripts (can be removed): {len(analysis_scripts)}")
    print(f"🟡 Non-imported core files (likely removable): {len(non_imported_core_files)}")
    print(f"🟢 Imported core files (need fixing): {len(imported_core_files)}")
    print(f"⚪ Other files: {len(other_files)}")
    
    return results

def generate_removal_plan(results: Dict[str, List[str]]) -> None:
    """Generate a plan for removing unnecessary files."""
    
    print("\n🗑️ REMOVAL PLAN:")
    print("=" * 60)
    
    # Files that can be safely removed
    safe_to_remove = (
        results['stub_files'] + 
        results['analysis_scripts'] + 
        results['non_imported_core_files']
    )
    
    print(f"✅ SAFE TO REMOVE ({len(safe_to_remove)} files):")
    for filepath in sorted(safe_to_remove):
        print(f"   {filepath}")
    
    # Files that need fixing
    need_fixing = results['imported_core_files'] + results['other_files']
    
    print(f"\n🔧 NEED FIXING ({len(need_fixing)} files):")
    for filepath in sorted(need_fixing):
        print(f"   {filepath}")
    
    # Create removal script
    if safe_to_remove:
        with open('remove_unnecessary_files.py', 'w') as f:
            f.write('#!/usr/bin/env python3\n')
            f.write('"""Remove unnecessary files with syntax errors."""\n\n')
            f.write('import os\n\n')
            f.write('files_to_remove = [\n')
            for filepath in sorted(safe_to_remove):
                f.write(f"    '{filepath}',\n")
            f.write(']\n\n')
            f.write('for filepath in files_to_remove:\n')
            f.write('    try:\n')
            f.write('        os.remove(filepath)\n')
            f.write('        print(f"✅ Removed: {filepath}")\n')
            f.write('    except FileNotFoundError:\n')
            f.write('        print(f"⚠️ Already removed: {filepath}")\n')
            f.write('    except Exception as e:\n')
            f.write('        print(f"❌ Error removing {filepath}: {e}")\n')
        
        print(f"\n📝 Created removal script: remove_unnecessary_files.py")

if __name__ == "__main__":
    results = analyze_file_necessity()
    if results:
        generate_removal_plan(results) 