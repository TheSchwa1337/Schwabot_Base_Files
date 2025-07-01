#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Schwabot Environment Setup Script
=================================

This script helps set up the Schwabot trading system environment
with all required dependencies and configuration.
"""

import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description=""):
    """Run a command and handle errors."""
    print(f"🔧 {description}")
    print(f"   Running: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"   ❌ Error: {result.stderr}")
            return False
        else:
            print(f"   ✅ Success: {description}")
            if result.stdout.strip():
                print(f"   Output: {result.stdout.strip()}")
            return True
    except Exception as e:
        print(f"   💥 Exception: {e}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    
    if version.major != 3 or version.minor < 8:
        print(f"   ❌ Python {version.major}.{version.minor} detected")
        print("   ⚠️  Schwabot requires Python 3.8 or higher")
        return False
    else:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True


def install_dependencies():
    """Install Python dependencies."""
    print("\n📦 Installing Dependencies...")
    
    # Upgrade pip first
    if not run_command(f"{sys.executable} -m pip install --upgrade pip", "Upgrading pip"):
        return False
    
    # Install core dependencies
    if not run_command(f"{sys.executable} -m pip install -r requirements.txt", "Installing core dependencies"):
        return False
    
    return True


def install_dev_dependencies():
    """Install development dependencies."""
    print("\n🛠️  Installing Development Dependencies...")
    
    dev_req_file = Path("requirements-dev.txt")
    if dev_req_file.exists():
        return run_command(f"{sys.executable} -m pip install -r requirements-dev.txt", "Installing dev dependencies")
    else:
        print("   ⚠️  requirements-dev.txt not found, skipping dev dependencies")
        return True


def setup_pre_commit():
    """Set up pre-commit hooks."""
    print("\n🪝 Setting up pre-commit hooks...")
    
    if not run_command("pre-commit --version", "Checking pre-commit installation"):
        print("   Installing pre-commit...")
        if not run_command(f"{sys.executable} -m pip install pre-commit", "Installing pre-commit"):
            return False
    
    return run_command("pre-commit install", "Installing pre-commit hooks")


def create_directories():
    """Create necessary directories."""
    print("\n📁 Creating necessary directories...")
    
    directories = [
        "flask/feeds",
        "flask/feeds/sentiment",
        "flask/feeds/whale_data",
        "flask/feeds/onchain_data", 
        "flask/feeds/market_data",
        "settings",
        "logs",
        ".mypy_cache"
    ]
    
    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"   ✅ Created: {directory}")
        else:
            print(f"   ✓ Exists: {directory}")
    
    return True


def test_imports():
    """Test key imports to verify installation."""
    print("\n🔍 Testing key imports...")
    
    test_imports = [
        ("numpy", "NumPy mathematical library"),
        ("aiohttp", "Async HTTP client"),
        ("requests", "HTTP requests library"),
        ("pandas", "Data manipulation library"),
        ("matplotlib", "Plotting library"),
        ("flask", "Web framework"),
        ("cryptography", "Cryptographic functions"),
        ("psutil", "System monitoring")
    ]
    
    all_good = True
    
    for module_name, description in test_imports:
        try:
            __import__(module_name)
            print(f"   ✅ {description}: OK")
        except ImportError as e:
            print(f"   ❌ {description}: FAILED ({e})")
            all_good = False
    
    return all_good


def run_integration_test():
    """Run the Schwabot integration test."""
    print("\n🧪 Running integration test...")
    
    test_file = Path("test_schwabot_integration.py")
    if test_file.exists():
        return run_command(f"{sys.executable} test_schwabot_integration.py", "Running integration test")
    else:
        print("   ⚠️  Integration test file not found, skipping")
        return True


def main():
    """Main setup function."""
    print("🌟 Schwabot Environment Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    if not create_directories():
        print("❌ Failed to create directories")
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Ask about dev dependencies
    install_dev = input("\n🤔 Install development dependencies? (y/N): ").lower().startswith('y')
    if install_dev:
        if not install_dev_dependencies():
            print("⚠️  Failed to install dev dependencies (continuing anyway)")
    
    # Ask about pre-commit
    setup_hooks = input("\n🤔 Set up pre-commit hooks for code quality? (y/N): ").lower().startswith('y')
    if setup_hooks:
        if not setup_pre_commit():
            print("⚠️  Failed to set up pre-commit hooks (continuing anyway)")
    
    # Test imports
    if not test_imports():
        print("⚠️  Some imports failed - there may be dependency issues")
    
    # Run integration test
    run_test = input("\n🤔 Run integration test? (y/N): ").lower().startswith('y')
    if run_test:
        if not run_integration_test():
            print("⚠️  Integration test failed - check the output above")
    
    print("\n🎉 Setup Complete!")
    print("\nNext steps:")
    print("1. Configure API keys in the settings files (optional)")
    print("2. Run: python schwabot_enhanced_launcher.py")
    print("3. Or run: python test_schwabot_integration.py")
    print("\nFor development:")
    print("- Use 'black .' to format code")
    print("- Use 'flake8 .' to check code quality")
    print("- Use 'mypy .' to check types")
    print("- Use 'pre-commit run --all-files' to run all checks")


if __name__ == "__main__":
    main() 