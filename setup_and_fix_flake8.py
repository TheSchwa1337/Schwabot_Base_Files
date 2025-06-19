#!/usr/bin/env python3
"""
Setup and Fix Flake8 Issues
Installs required tools and runs comprehensive flake8 fixes.
"""

import os
import sys
import subprocess
import time

def install_package(package_name):
    """Install a Python package using pip."""
    try:
        print(f"📦 Installing {package_name}...")
        result = subprocess.run([sys.executable, '-m', 'pip', 'install', package_name], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  ✅ {package_name} installed successfully")
            return True
        else:
            print(f"  ❌ Failed to install {package_name}: {result.stderr}")
            return False
    except Exception as e:
        print(f"  ❌ Error installing {package_name}: {e}")
        return False

def run_python_script(script_name, args=None):
    """Run a Python script."""
    try:
        cmd = [sys.executable, script_name]
        if args:
            cmd.extend(args)
        
        print(f"🔧 Running {script_name}...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Print output
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(f"Errors: {result.stderr}")
            
        return result.returncode == 0
    except Exception as e:
        print(f"  ❌ Error running {script_name}: {e}")
        return False

def fix_files_manually():
    """Manually fix common flake8 issues without external tools."""
    import glob
    import re
    
    print("🔧 Running manual flake8 fixes...")
    
    # Get all Python files
    python_files = []
    for pattern in ['*.py', 'core/*.py', 'tests/*.py']:
        python_files.extend(glob.glob(pattern))
    
    fixed_count = 0
    
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            original_content = content
            
            # Fix common issues
            lines = content.split('\n')
            fixed_lines = []
            
            for line in lines:
                # Fix trailing whitespace (W291, W293)
                line = line.rstrip()
                
                # Fix simple line length issues for imports
                if len(line) > 79 and ('import ' in line):
                    if 'from ' in line and ' import ' in line:
                        parts = line.split(' import ')
                        if len(parts) == 2 and len(parts[1]) > 50:
                            # Break long imports
                            imports = [imp.strip() for imp in parts[1].split(',')]
                            if len(imports) > 1:
                                fixed_lines.append(f"{parts[0]} import (")
                                for i, imp in enumerate(imports):
                                    comma = ',' if i < len(imports) - 1 else ''
                                    fixed_lines.append(f"    {imp}{comma}")
                                fixed_lines.append(")")
                            else:
                                fixed_lines.append(line)
                        else:
                            fixed_lines.append(line)
                    else:
                        fixed_lines.append(line)
                else:
                    fixed_lines.append(line)
            
            # Ensure file ends with newline (W292)
            content = '\n'.join(fixed_lines)
            if content and not content.endswith('\n'):
                content += '\n'
            
            # Write back if changed
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                fixed_count += 1
                print(f"  ✓ Fixed {file_path}")
                
        except Exception as e:
            print(f"  ✗ Error fixing {file_path}: {e}")
    
    print(f"📊 Manually fixed {fixed_count} files")
    return fixed_count > 0

def main():
    print("🚀 SETUP AND FIX FLAKE8 ISSUES")
    print("=" * 50)
    
    # Install required packages
    print("\n📦 INSTALLING REQUIRED PACKAGES")
    print("-" * 30)
    
    packages = ['flake8', 'autopep8', 'isort', 'black']
    failed_installs = []
    
    for package in packages:
        if not install_package(package):
            failed_installs.append(package)
    
    if failed_installs:
        print(f"\n⚠️  Some packages failed to install: {', '.join(failed_installs)}")
        print("Continuing with available tools...")
    
    # Run our custom fixers
    print(f"\n🔧 RUNNING CUSTOM FLAKE8 FIXERS")
    print("-" * 30)
    
    start_time = time.time()
    
    # Try to run our master fixer
    if os.path.exists('master_flake8_comprehensive_fixer.py'):
        print("Running master comprehensive fixer...")
        success = run_python_script('master_flake8_comprehensive_fixer.py')
        if not success:
            print("Master fixer had issues, trying manual fixes...")
            fix_files_manually()
    else:
        print("Master fixer not found, running manual fixes...")
        fix_files_manually()
    
    # Try to run test file fixer
    if os.path.exists('test_files_flake8_fixer.py'):
        print("\nRunning test file fixer...")
        run_python_script('test_files_flake8_fixer.py')
    
    # Run additional tools if available
    print(f"\n🛠️  RUNNING ADDITIONAL FORMATTING TOOLS")
    print("-" * 30)
    
    # Try autopep8
    try:
        print("Running autopep8...")
        result = subprocess.run([sys.executable, '-m', 'autopep8', '--in-place', '--recursive', '.'], 
                              capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print("  ✅ autopep8 completed")
        else:
            print(f"  ⚠️  autopep8 issues: {result.stderr}")
    except Exception as e:
        print(f"  ⚠️  autopep8 not available: {e}")
    
    # Try isort
    try:
        print("Running isort...")
        result = subprocess.run([sys.executable, '-m', 'isort', '.'], 
                              capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print("  ✅ isort completed")
        else:
            print(f"  ⚠️  isort issues: {result.stderr}")
    except Exception as e:
        print(f"  ⚠️  isort not available: {e}")
    
    processing_time = time.time() - start_time
    
    # Final check
    print(f"\n📊 FINAL STATUS CHECK")
    print("-" * 30)
    
    try:
        result = subprocess.run([sys.executable, '-m', 'flake8', '.'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("🎉 ALL FLAKE8 ISSUES RESOLVED!")
        else:
            lines = result.stdout.strip().split('\n') if result.stdout.strip() else []
            print(f"📋 {len(lines)} flake8 issues remain")
            
            # Show first few issues
            for i, line in enumerate(lines[:10]):
                if line:
                    print(f"  {line}")
            if len(lines) > 10:
                print(f"  ... and {len(lines) - 10} more")
    except Exception as e:
        print(f"⚠️  Could not run final flake8 check: {e}")
    
    print(f"\n⏱️  Total processing time: {processing_time:.2f} seconds")
    print("\n💡 RECOMMENDATIONS:")
    print("✅ Set up pre-commit hooks for continuous compliance")
    print("✅ Add flake8 checks to CI/CD pipeline")
    print("✅ Consider using black for consistent formatting")
    print("✅ Regular automated linting helps maintain code quality")

if __name__ == '__main__':
    main() 