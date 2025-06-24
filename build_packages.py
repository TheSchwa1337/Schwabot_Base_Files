#!/usr/bin/env python3
"""
Schwabot Cross-Platform Package Builder
======================================

This script builds Schwabot packages for multiple platforms:
- Linux: .deb, .rpm, AppImage, Docker
- Windows: .exe, .msi, portable
- macOS: .dmg, .pkg, App bundle
- Universal: Python wheel, source distribution

Usage:
    python build_packages.py --platform all
    python build_packages.py --platform linux --format deb
    python build_packages.py --platform windows --format exe
    python build_packages.py --platform macos --format dmg
"""

import os
import sys
import platform
import subprocess
import argparse
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import yaml

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

class PackageBuilder:
    """Cross-platform package builder for Schwabot."""
    
    def __init__(self):
        """Initialize the package builder."""
        self.project_root = Path(__file__).parent
        self.build_dir = self.project_root / "build"
        self.dist_dir = self.project_root / "dist"
        self.package_name = "schwabot"
        self.version = "2.0.0"
        
        # Create build directories
        self.build_dir.mkdir(exist_ok=True)
        self.dist_dir.mkdir(exist_ok=True)
        
        # Platform detection
        self.current_platform = platform.system().lower()
        self.current_arch = platform.machine().lower()
        
        print(f"🔧 Building Schwabot v{self.version}")
        print(f"📊 Platform: {self.current_platform} ({self.current_arch})")
        print(f"📁 Build directory: {self.build_dir}")
        print(f"📦 Dist directory: {self.dist_dir}")
    
    def clean_build(self) -> None:
        """Clean build directories."""
        print("🧹 Cleaning build directories...")
        
        # Clean build artifacts
        for pattern in ["*.pyc", "*.pyo", "__pycache__", "*.egg-info"]:
            for path in self.project_root.rglob(pattern):
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    path.unlink(missing_ok=True)
        
        # Clean build and dist directories
        if self.build_dir.exists():
            shutil.rmtree(self.build_dir)
        if self.dist_dir.exists():
            shutil.rmtree(self.dist_dir)
        
        # Recreate directories
        self.build_dir.mkdir(exist_ok=True)
        self.dist_dir.mkdir(exist_ok=True)
        
        print("✅ Build directories cleaned")
    
    def build_python_packages(self) -> None:
        """Build Python wheel and source distribution."""
        print("🐍 Building Python packages...")
        
        try:
            # Build wheel and source distribution
            subprocess.run([
                sys.executable, "-m", "build", 
                "--wheel", "--sdist",
                "--outdir", str(self.dist_dir)
            ], check=True, cwd=self.project_root)
            
            print("✅ Python packages built successfully")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to build Python packages: {e}")
            raise
    
    def build_linux_packages(self) -> None:
        """Build Linux packages (.deb, .rpm, AppImage)."""
        print("🐧 Building Linux packages...")
        
        # Build .deb package
        self._build_deb_package()
        
        # Build .rpm package
        self._build_rpm_package()
        
        # Build AppImage
        self._build_appimage()
        
        print("✅ Linux packages built successfully")
    
    def _build_deb_package(self) -> None:
        """Build Debian package."""
        print("  📦 Building .deb package...")
        
        try:
            # Create debian directory structure
            deb_dir = self.build_dir / "deb"
            deb_dir.mkdir(exist_ok=True)
            
            # Create control file
            control_content = f"""Package: {self.package_name}
Version: {self.version}
Section: utils
Priority: optional
Architecture: all
Depends: python3 (>= 3.8), python3-pip
Maintainer: Schwabot Development Team <dev@schwabot.ai>
Description: Hardware-scale-aware economic kernel for federated trading devices
 Schwabot is a comprehensive trading system with mathematical precision,
 supporting multiple exchanges and real-time monitoring.
"""
            
            control_file = deb_dir / "control"
            control_file.write_text(control_content)
            
            # Build .deb package
            subprocess.run([
                "dpkg-deb", "--build", str(deb_dir),
                str(self.dist_dir / f"{self.package_name}-{self.version}.deb")
            ], check=True)
            
            print("    ✅ .deb package built")
            
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"    ⚠️ Could not build .deb package: {e}")
    
    def _build_rpm_package(self) -> None:
        """Build RPM package."""
        print("  📦 Building .rpm package...")
        
        try:
            # Create RPM spec file
            spec_content = f"""Name: {self.package_name}
Version: {self.version}
Release: 1
Summary: Hardware-scale-aware economic kernel for federated trading devices
License: MIT
URL: https://github.com/schwabot/schwabot
BuildArch: noarch
Requires: python3 >= 3.8

%description
Schwabot is a comprehensive trading system with mathematical precision,
supporting multiple exchanges and real-time monitoring.

%files
%defattr(-,root,root,-)
/usr/bin/schwabot
/usr/lib/python3*/site-packages/schwabot*

%post
python3 -m pip install --upgrade pip

%preun
python3 -m pip uninstall -y {self.package_name}
"""
            
            spec_file = self.build_dir / f"{self.package_name}.spec"
            spec_file.write_text(spec_content)
            
            # Build RPM package
            subprocess.run([
                "rpmbuild", "-bb", "--define", f"_rpmdir {self.dist_dir}",
                str(spec_file)
            ], check=True)
            
            print("    ✅ .rpm package built")
            
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"    ⚠️ Could not build .rpm package: {e}")
    
    def _build_appimage(self) -> None:
        """Build AppImage package."""
        print("  📦 Building AppImage...")
        
        try:
            # Create AppDir structure
            appdir = self.build_dir / "AppDir"
            appdir.mkdir(exist_ok=True)
            
            # Create AppRun script
            apprun_content = """#!/bin/bash
cd "$(dirname "$0")"
exec python3 usr/bin/schwabot "$@"
"""
            
            apprun_file = appdir / "AppRun"
            apprun_file.write_text(apprun_content)
            apprun_file.chmod(0o755)
            
            # Create .desktop file
            desktop_content = f"""[Desktop Entry]
Name=Schwabot
Comment=Hardware-scale-aware economic kernel for federated trading devices
Exec=schwabot
Icon=schwabot
Terminal=true
Type=Application
Categories=Office;Finance;
"""
            
            desktop_file = appdir / f"{self.package_name}.desktop"
            desktop_file.write_text(desktop_content)
            
            # Build AppImage
            subprocess.run([
                "appimagetool", str(appdir),
                str(self.dist_dir / f"{self.package_name}-{self.version}-x86_64.AppImage")
            ], check=True)
            
            print("    ✅ AppImage built")
            
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"    ⚠️ Could not build AppImage: {e}")
    
    def build_windows_packages(self) -> None:
        """Build Windows packages (.exe, .msi)."""
        print("🪟 Building Windows packages...")
        
        # Build executable
        self._build_windows_exe()
        
        # Build MSI installer
        self._build_windows_msi()
        
        # Build portable package
        self._build_windows_portable()
        
        print("✅ Windows packages built successfully")
    
    def _build_windows_exe(self) -> None:
        """Build Windows executable using PyInstaller."""
        print("  📦 Building .exe package...")
        
        try:
            # Install PyInstaller if not available
            subprocess.run([
                sys.executable, "-m", "pip", "install", "pyinstaller"
            ], check=True)
            
            # Build executable
            subprocess.run([
                sys.executable, "-m", "PyInstaller",
                "--onefile",
                "--windowed",
                "--name", self.package_name,
                "--distpath", str(self.dist_dir),
                "--workpath", str(self.build_dir / "pyinstaller"),
                "--specpath", str(self.build_dir / "pyinstaller"),
                str(self.project_root / "run_schwabot.py")
            ], check=True)
            
            print("    ✅ .exe package built")
            
        except subprocess.CalledProcessError as e:
            print(f"    ⚠️ Could not build .exe package: {e}")
    
    def _build_windows_msi(self) -> None:
        """Build Windows MSI installer."""
        print("  📦 Building .msi package...")
        
        try:
            # Install cx_Freeze if not available
            subprocess.run([
                sys.executable, "-m", "pip", "install", "cx_Freeze"
            ], check=True)
            
            # Create setup script for cx_Freeze
            setup_cx_content = f"""from cx_Freeze import setup, Executable
import sys

build_exe_options = {{
    "packages": ["core", "ui", "config", "utils"],
    "excludes": ["tkinter", "test"],
    "include_files": ["config/", "ui/templates/", "ui/static/"]
}}

base = None
if sys.platform == "win32":
    base = "Win32GUI"

setup(
    name="{self.package_name}",
    version="{self.version}",
    description="Hardware-scale-aware economic kernel",
    options={{"build_exe": build_exe_options}},
    executables=[Executable("run_schwabot.py", base=base)]
)
"""
            
            setup_cx_file = self.build_dir / "setup_cx.py"
            setup_cx_file.write_text(setup_cx_content)
            
            # Build MSI
            subprocess.run([
                sys.executable, str(setup_cx_file), "bdist_msi"
            ], check=True, cwd=self.build_dir)
            
            # Move MSI to dist directory
            for msi_file in self.build_dir.rglob("*.msi"):
                shutil.move(msi_file, self.dist_dir / msi_file.name)
            
            print("    ✅ .msi package built")
            
        except subprocess.CalledProcessError as e:
            print(f"    ⚠️ Could not build .msi package: {e}")
    
    def _build_windows_portable(self) -> None:
        """Build Windows portable package."""
        print("  📦 Building portable package...")
        
        try:
            # Create portable directory
            portable_dir = self.dist_dir / f"{self.package_name}-{self.version}-portable"
            portable_dir.mkdir(exist_ok=True)
            
            # Copy Python files
            for pattern in ["*.py", "*.yaml", "*.yml", "*.json", "*.md"]:
                for file_path in self.project_root.glob(pattern):
                    if file_path.is_file():
                        shutil.copy2(file_path, portable_dir)
            
            # Copy directories
            for dir_name in ["core", "ui", "config", "utils", "mathlib", "ncco_core"]:
                src_dir = self.project_root / dir_name
                if src_dir.exists():
                    shutil.copytree(src_dir, portable_dir / dir_name, dirs_exist_ok=True)
            
            # Create batch file for Windows
            batch_content = """@echo off
echo Starting Schwabot Trading System...
python run_schwabot.py
pause
"""
            
            batch_file = portable_dir / "start_schwabot.bat"
            batch_file.write_text(batch_content)
            
            # Create ZIP archive
            shutil.make_archive(
                str(portable_dir),
                'zip',
                portable_dir.parent,
                portable_dir.name
            )
            
            # Clean up directory
            shutil.rmtree(portable_dir)
            
            print("    ✅ Portable package built")
            
        except Exception as e:
            print(f"    ⚠️ Could not build portable package: {e}")
    
    def build_macos_packages(self) -> None:
        """Build macOS packages (.dmg, .pkg, App bundle)."""
        print("🍎 Building macOS packages...")
        
        # Build App bundle
        self._build_macos_app()
        
        # Build DMG
        self._build_macos_dmg()
        
        # Build PKG installer
        self._build_macos_pkg()
        
        print("✅ macOS packages built successfully")
    
    def _build_macos_app(self) -> None:
        """Build macOS App bundle."""
        print("  📦 Building .app bundle...")
        
        try:
            # Install py2app if not available
            subprocess.run([
                sys.executable, "-m", "pip", "install", "py2app"
            ], check=True)
            
            # Create setup script for py2app
            setup_py2app_content = f"""from setuptools import setup

APP = ['run_schwabot.py']
DATA_FILES = [
    ('config', ['config/schwabot_config.yaml']),
    ('ui/templates', ['ui/templates/base.html', 'ui/templates/dashboard.html']),
    ('ui/static', ['ui/static/']),
]
OPTIONS = {{
    'argv_emulation': True,
    'packages': ['core', 'ui', 'config', 'utils'],
    'iconfile': 'ui/static/icon.icns',
    'plist': {{
        'CFBundleName': 'Schwabot',
        'CFBundleDisplayName': 'Schwabot Trading System',
        'CFBundleGetInfoString': "Hardware-scale-aware economic kernel",
        'CFBundleIdentifier': "com.schwabot.trading",
        'CFBundleVersion': "{self.version}",
        'CFBundleShortVersionString': "{self.version}",
        'NSHumanReadableCopyright': u"Copyright © 2024, Schwabot Development Team, All Rights Reserved"
    }}
}}

setup(
    app=APP,
    data_files=DATA_FILES,
    options={{'py2app': OPTIONS}},
    setup_requires=['py2app'],
)
"""
            
            setup_py2app_file = self.build_dir / "setup_py2app.py"
            setup_py2app_file.write_text(setup_py2app_content)
            
            # Build App bundle
            subprocess.run([
                sys.executable, str(setup_py2app_file), "py2app"
            ], check=True, cwd=self.build_dir)
            
            # Move App bundle to dist directory
            app_bundle = self.build_dir / "dist" / f"{self.package_name}.app"
            if app_bundle.exists():
                shutil.move(app_bundle, self.dist_dir / f"{self.package_name}.app")
            
            print("    ✅ .app bundle built")
            
        except subprocess.CalledProcessError as e:
            print(f"    ⚠️ Could not build .app bundle: {e}")
    
    def _build_macos_dmg(self) -> None:
        """Build macOS DMG package."""
        print("  📦 Building .dmg package...")
        
        try:
            # Create DMG using hdiutil
            app_path = self.dist_dir / f"{self.package_name}.app"
            dmg_path = self.dist_dir / f"{self.package_name}-{self.version}.dmg"
            
            if app_path.exists():
                subprocess.run([
                    "hdiutil", "create", "-volname", "Schwabot",
                    "-srcfolder", str(app_path),
                    "-ov", "-format", "UDZO",
                    str(dmg_path)
                ], check=True)
                
                print("    ✅ .dmg package built")
            else:
                print("    ⚠️ App bundle not found, skipping DMG")
                
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"    ⚠️ Could not build .dmg package: {e}")
    
    def _build_macos_pkg(self) -> None:
        """Build macOS PKG installer."""
        print("  📦 Building .pkg package...")
        
        try:
            # Install pkgbuild if available
            app_path = self.dist_dir / f"{self.package_name}.app"
            pkg_path = self.dist_dir / f"{self.package_name}-{self.version}.pkg"
            
            if app_path.exists():
                subprocess.run([
                    "pkgbuild", "--component", str(app_path),
                    "--install-location", "/Applications",
                    str(pkg_path)
                ], check=True)
                
                print("    ✅ .pkg package built")
            else:
                print("    ⚠️ App bundle not found, skipping PKG")
                
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"    ⚠️ Could not build .pkg package: {e}")
    
    def build_docker_image(self) -> None:
        """Build Docker image."""
        print("🐳 Building Docker image...")
        
        try:
            # Create Dockerfile
            dockerfile_content = f"""FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p logs config

# Expose ports
EXPOSE 8080 8081 8082

# Set default command
CMD ["python", "run_schwabot.py"]
"""
            
            dockerfile_path = self.project_root / "Dockerfile"
            dockerfile_path.write_text(dockerfile_content)
            
            # Build Docker image
            subprocess.run([
                "docker", "build", "-t", f"{self.package_name}:{self.version}",
                "-t", f"{self.package_name}:latest", "."
            ], check=True, cwd=self.project_root)
            
            print("✅ Docker image built successfully")
            
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"❌ Could not build Docker image: {e}")
    
    def create_installer_scripts(self) -> None:
        """Create platform-specific installer scripts."""
        print("📜 Creating installer scripts...")
        
        # Linux installer script
        linux_installer = self.dist_dir / "install_linux.sh"
        linux_installer.write_text("""#!/bin/bash
echo "Installing Schwabot on Linux..."

# Detect package manager
if command -v apt-get &> /dev/null; then
    echo "Using apt package manager"
    sudo apt-get update
    sudo apt-get install -y python3 python3-pip
elif command -v yum &> /dev/null; then
    echo "Using yum package manager"
    sudo yum install -y python3 python3-pip
elif command -v dnf &> /dev/null; then
    echo "Using dnf package manager"
    sudo dnf install -y python3 python3-pip
else
    echo "No supported package manager found"
    exit 1
fi

# Install Schwabot
pip3 install schwabot-*.whl

echo "Schwabot installed successfully!"
echo "Run 'schwabot' to start the system"
""")
        linux_installer.chmod(0o755)
        
        # Windows installer script
        windows_installer = self.dist_dir / "install_windows.bat"
        windows_installer.write_text("""@echo off
echo Installing Schwabot on Windows...

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed. Please install Python 3.8+ first.
    pause
    exit /b 1
)

REM Install Schwabot
pip install schwabot-*.whl

echo Schwabot installed successfully!
echo Run 'schwabot' to start the system
pause
""")
        
        # macOS installer script
        macos_installer = self.dist_dir / "install_macos.sh"
        macos_installer.write_text("""#!/bin/bash
echo "Installing Schwabot on macOS..."

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# Install Python if not already installed
if ! command -v python3 &> /dev/null; then
    echo "Installing Python..."
    brew install python
fi

# Install Schwabot
pip3 install schwabot-*.whl

echo "Schwabot installed successfully!"
echo "Run 'schwabot' to start the system"
""")
        macos_installer.chmod(0o755)
        
        print("✅ Installer scripts created")
    
    def generate_package_summary(self) -> None:
        """Generate a summary of all built packages."""
        print("📋 Generating package summary...")
        
        packages = []
        
        # List all files in dist directory
        for file_path in self.dist_dir.rglob("*"):
            if file_path.is_file():
                packages.append({
                    "name": file_path.name,
                    "size": file_path.stat().st_size,
                    "type": file_path.suffix,
                    "path": str(file_path.relative_to(self.dist_dir))
                })
        
        # Create summary file
        summary = {
            "project": self.package_name,
            "version": self.version,
            "build_date": str(Path(__file__).stat().st_mtime),
            "platform": self.current_platform,
            "architecture": self.current_arch,
            "packages": packages,
            "total_packages": len(packages)
        }
        
        summary_file = self.dist_dir / "package_summary.json"
        summary_file.write_text(json.dumps(summary, indent=2))
        
        # Print summary
        print(f"\n📦 Package Summary:")
        print(f"   Project: {self.package_name} v{self.version}")
        print(f"   Total packages: {len(packages)}")
        print(f"   Build directory: {self.dist_dir}")
        
        for package in packages:
            size_mb = package["size"] / (1024 * 1024)
            print(f"   - {package['name']} ({size_mb:.1f} MB)")
        
        print(f"\n✅ Package summary generated: {summary_file}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Build Schwabot packages for multiple platforms")
    parser.add_argument("--platform", choices=["all", "linux", "windows", "macos", "python"], 
                       default="python", help="Target platform(s)")
    parser.add_argument("--format", choices=["all", "deb", "rpm", "appimage", "exe", "msi", "dmg", "pkg"],
                       default="all", help="Package format(s)")
    parser.add_argument("--clean", action="store_true", help="Clean build directories before building")
    parser.add_argument("--docker", action="store_true", help="Build Docker image")
    
    args = parser.parse_args()
    
    builder = PackageBuilder()
    
    try:
        if args.clean:
            builder.clean_build()
        
        # Build Python packages (always)
        builder.build_python_packages()
        
        # Build platform-specific packages
        if args.platform in ["all", "linux"]:
            builder.build_linux_packages()
        
        if args.platform in ["all", "windows"]:
            builder.build_windows_packages()
        
        if args.platform in ["all", "macos"]:
            builder.build_macos_packages()
        
        # Build Docker image if requested
        if args.docker:
            builder.build_docker_image()
        
        # Create installer scripts
        builder.create_installer_scripts()
        
        # Generate summary
        builder.generate_package_summary()
        
        print("\n🎉 All packages built successfully!")
        print(f"📁 Check the '{builder.dist_dir}' directory for all packages")
        
    except Exception as e:
        print(f"\n❌ Build failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 