#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Universal Schwabot Installer
============================

Auto-detects platform and runs the appropriate installer script.
Supports Windows, macOS, and Linux.
"""

import os
import sys
import platform
import subprocess
import shutil
from pathlib import Path


def detect_platform():
    """Detect the current platform."""
    system = platform.system().lower()
    
    if system == "windows":
        return "windows"
    elif system == "darwin":
        return "macos"
    elif system == "linux":
        return "linux"
    else:
        return "unknown"


def run_installer(platform_name):
    """Run the appropriate installer for the platform."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    
    if platform_name == "windows":
        installer_path = script_dir.parent / "windows" / "install_windows.bat"
        if installer_path.exists():
            print(f"Running Windows installer: {installer_path}")
            subprocess.run([str(installer_path)], shell=True, cwd=project_root)
        else:
            print("Windows installer not found!")
            return False
            
    elif platform_name == "linux":
        installer_path = script_dir.parent / "linux" / "install_linux.sh"
        if installer_path.exists():
            print(f"Running Linux installer: {installer_path}")
            # Make executable
            os.chmod(installer_path, 0o755)
            subprocess.run([str(installer_path)], cwd=project_root)
        else:
            print("Linux installer not found!")
            return False
            
    elif platform_name == "macos":
        installer_path = script_dir.parent / "macos" / "install_macos.sh"
        if installer_path.exists():
            print(f"Running macOS installer: {installer_path}")
            # Make executable
            os.chmod(installer_path, 0o755)
            subprocess.run([str(installer_path)], cwd=project_root)
        else:
            print("macOS installer not found!")
            return False
            
    else:
        print(f"Unsupported platform: {platform_name}")
        return False
    
    return True


def check_prerequisites():
    """Check if prerequisites are met."""
    print("Checking prerequisites...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        print(f"   Current version: {sys.version}")
        return False
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Check if pip is available
    try:
        import pip
        print("✅ pip is available")
    except ImportError:
        print("❌ pip is not available")
        return False
    
    return True


def main():
    """Main installation function."""
    print("=" * 50)
    print("Schwabot Trading System - Universal Installer")
    print("=" * 50)
    print()
    
    # Detect platform
    platform_name = detect_platform()
    print(f"Detected platform: {platform_name}")
    print()
    
    if platform_name == "unknown":
        print("❌ Unsupported platform detected")
        print("Supported platforms: Windows, macOS, Linux")
        return 1
    
    # Check prerequisites
    if not check_prerequisites():
        print()
        print("❌ Prerequisites not met. Please install Python 3.8+ and try again.")
        return 1
    
    print()
    print("✅ Prerequisites met. Starting installation...")
    print()
    
    # Run platform-specific installer
    success = run_installer(platform_name)
    
    if success:
        print()
        print("=" * 50)
        print("✅ Installation completed successfully!")
        print("=" * 50)
        print()
        print("Next steps:")
        print("1. Configure your API keys in config/schwabot_config.yaml")
        print("2. Start Schwabot using the provided shortcuts or commands")
        print("3. Access the web dashboard at http://localhost:8080")
        print()
        print("For more information, see docs/installation/")
        return 0
    else:
        print()
        print("❌ Installation failed!")
        print("Please check the error messages above and try again.")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 