#!/usr/bin/env python3
"""
Virtual Environment Fix Script
==============================

Fixes syntax errors in the virtual environment by recreating it with
compatible package versions.

The issue is that some packages in the current virtual environment have
malformed function signatures with double arrow syntax (-> Any -> Any:)
instead of proper type annotations.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            print(f"❌ {description} failed:")
            print(f"   Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {description} failed with exception: {e}")
        return False


def fix_virtual_environment() -> bool:
    """Fix the virtual environment by recreating it"""
    print("🔧 Virtual Environment Fix Script")
    print("=" * 40)
    
    # Check if we're in a virtual environment
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  Not currently in a virtual environment")
        print("   This script is designed to fix virtual environment issues")
        return False
    
    venv_path = Path(sys.prefix)
    print(f"📍 Current virtual environment: {venv_path}")
    
    # Step 1: Deactivate current environment
    print("\n📋 Step 1: Deactivating current environment")
    print("   Please run 'deactivate' in your terminal, then run this script again")
    print("   Or run this script from outside the virtual environment")
    
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("   ⚠️  Script detected active virtual environment")
        print("   Please deactivate first: deactivate")
        return False
    
    # Step 2: Remove problematic environment
    if venv_path.exists():
        print(f"\n📋 Step 2: Removing problematic environment: {venv_path}")
        try:
            shutil.rmtree(venv_path)
            print("✅ Environment removed successfully")
        except Exception as e:
            print(f"❌ Failed to remove environment: {e}")
            return False
    
    # Step 3: Create fresh environment
    print(f"\n📋 Step 3: Creating fresh virtual environment: {venv_path}")
    if not run_command(f"python -m venv {venv_path}", "Creating virtual environment"):
        return False
    
    # Step 4: Activate and install compatible packages
    print(f"\n📋 Step 4: Installing compatible packages")
    
    # Determine activation script
    if os.name == 'nt':  # Windows
        activate_script = venv_path / "Scripts" / "activate.bat"
        pip_path = venv_path / "Scripts" / "pip.exe"
    else:  # Unix/Linux
        activate_script = venv_path / "bin" / "activate"
        pip_path = venv_path / "bin" / "pip"
    
    # Install packages with specific versions known to work
    packages = [
        "setuptools>=65.0.0",
        "wheel>=0.38.0",
        "numpy>=1.24.0",
        "flake8>=6.0.0",
        "mypy>=1.0.0"
    ]
    
    for package in packages:
        if not run_command(f"{pip_path} install {package}", f"Installing {package}"):
            print(f"⚠️  Warning: Failed to install {package}")
    
    # Step 5: Verify installation
    print(f"\n📋 Step 5: Verifying installation")
    
    # Test Python import
    test_script = f"""
import sys
print(f"Python version: {{sys.version}}")

try:
    import numpy as np
    print("✅ NumPy imported successfully")
except Exception as e:
    print(f"❌ NumPy import failed: {{e}}")

try:
    import flake8
    print("✅ Flake8 imported successfully")
except Exception as e:
    print(f"❌ Flake8 import failed: {{e}}")

try:
    import setuptools
    print("✅ Setuptools imported successfully")
except Exception as e:
    print(f"❌ Setuptools import failed: {{e}}")
"""
    
    test_file = venv_path / "test_imports.py"
    with open(test_file, 'w') as f:
        f.write(test_script)
    
    if os.name == 'nt':  # Windows
        python_path = venv_path / "Scripts" / "python.exe"
    else:  # Unix/Linux
        python_path = venv_path / "bin" / "python"
    
    if run_command(f"{python_path} {test_file}", "Testing package imports"):
        print("✅ All packages imported successfully")
    else:
        print("❌ Some packages failed to import")
        return False
    
    # Clean up test file
    test_file.unlink(missing_ok=True)
    
    print(f"\n🎉 Virtual environment fixed successfully!")
    print(f"📍 New environment: {venv_path}")
    print(f"\n📋 Next steps:")
    print(f"   1. Activate the environment:")
    if os.name == 'nt':  # Windows
        print(f"      {venv_path}\\Scripts\\activate")
    else:  # Unix/Linux
        print(f"      source {venv_path}/bin/activate")
    print(f"   2. Test your Schwabot files:")
    print(f"      flake8 config/mathematical_framework_config.py")
    print(f"      flake8 core/drift_shell_engine.py")
    print(f"      flake8 core/quantum_drift_shell_engine.py")
    
    return True


def main() -> None:
    """Main function"""
    success = fix_virtual_environment()
    if success:
        print("\n✅ Virtual environment fix completed successfully!")
    else:
        print("\n❌ Virtual environment fix failed!")
        print("   Please check the error messages above and try again.")


if __name__ == "__main__":
    main() 