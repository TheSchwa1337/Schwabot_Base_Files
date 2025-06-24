#!/usr/bin/env python3
"""
Schwabot Professional Installer
===============================

This installer provides a comprehensive installation experience for Schwabot
across all supported platforms with proper validation, configuration, and setup.
"""

import os
import sys
import platform
import subprocess
import argparse
import shutil
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
import urllib.request
import zipfile
import tarfile

class SchwabotInstaller:
    """Professional installer for Schwabot trading system."""
    
    def __init__(self):
        """Initialize the installer."""
        self.project_name = "Schwabot"
        self.version = "2.0.0"
        self.install_dir = Path.home() / ".schwabot"
        self.config_dir = self.install_dir / "config"
        self.logs_dir = self.install_dir / "logs"
        self.data_dir = self.install_dir / "data"
        
        # Platform detection
        self.platform = platform.system().lower()
        self.arch = platform.machine().lower()
        
        # Installation status
        self.installation_log = []
        self.errors = []
        
        print(f"🚀 {self.project_name} v{self.version} Installer")
        print(f"📊 Platform: {self.platform} ({self.arch})")
        print(f"📁 Install directory: {self.install_dir}")
        print("=" * 60)
    
    def log(self, message: str, level: str = "INFO") -> None:
        """Log installation messages."""
        timestamp = subprocess.run(["date"], capture_output=True, text=True).stdout.strip()
        log_entry = f"[{timestamp}] {level}: {message}"
        self.installation_log.append(log_entry)
        print(f"  {message}")
    
    def check_system_requirements(self) -> bool:
        """Check if system meets requirements."""
        self.log("Checking system requirements...")
        
        # Check Python version
        python_version = sys.version_info
        if python_version < (3, 8):
            self.log(f"❌ Python 3.8+ required, found {python_version.major}.{python_version.minor}", "ERROR")
            return False
        
        self.log(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        # Check available memory
        try:
            import psutil
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
            if memory_gb < 4:
                self.log(f"⚠️  Recommended: 4GB+ RAM, found {memory_gb:.1f}GB", "WARNING")
            else:
                self.log(f"✅ Memory: {memory_gb:.1f}GB")
        except ImportError:
            self.log("⚠️  Could not check memory (psutil not available)", "WARNING")
        
        # Check disk space
        try:
            disk_usage = shutil.disk_usage(self.install_dir.parent)
            disk_gb = disk_usage.free / (1024**3)
            if disk_gb < 10:
                self.log(f"❌ Need 10GB+ free space, found {disk_gb:.1f}GB", "ERROR")
                return False
            else:
                self.log(f"✅ Disk space: {disk_gb:.1f}GB available")
        except Exception as e:
            self.log(f"⚠️  Could not check disk space: {e}", "WARNING")
        
        # Check network connectivity
        try:
            urllib.request.urlopen("https://pypi.org", timeout=5)
            self.log("✅ Network connectivity")
        except Exception:
            self.log("⚠️  Network connectivity issues detected", "WARNING")
        
        return True
    
    def create_directories(self) -> bool:
        """Create installation directories."""
        self.log("Creating installation directories...")
        
        try:
            directories = [
                self.install_dir,
                self.config_dir,
                self.logs_dir,
                self.data_dir,
                self.install_dir / "bin",
                self.install_dir / "lib",
                self.install_dir / "docs"
            ]
            
            for directory in directories:
                directory.mkdir(parents=True, exist_ok=True)
                self.log(f"✅ Created: {directory}")
            
            return True
            
        except Exception as e:
            self.log(f"❌ Failed to create directories: {e}", "ERROR")
            self.errors.append(f"Directory creation failed: {e}")
            return False
    
    def install_python_package(self, package_path: Optional[str] = None) -> bool:
        """Install Schwabot Python package."""
        self.log("Installing Schwabot Python package...")
        
        try:
            if package_path and Path(package_path).exists():
                # Install from local package
                subprocess.run([
                    sys.executable, "-m", "pip", "install", package_path
                ], check=True)
                self.log(f"✅ Installed from: {package_path}")
            else:
                # Install from PyPI (if available)
                subprocess.run([
                    sys.executable, "-m", "pip", "install", "schwabot"
                ], check=True)
                self.log("✅ Installed from PyPI")
            
            # Verify installation
            result = subprocess.run([
                sys.executable, "-c", "import schwabot; print('OK')"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                self.log("✅ Package verification successful")
                return True
            else:
                self.log("❌ Package verification failed", "ERROR")
                return False
                
        except subprocess.CalledProcessError as e:
            self.log(f"❌ Installation failed: {e}", "ERROR")
            self.errors.append(f"Package installation failed: {e}")
            return False
    
    def install_platform_package(self, package_path: str) -> bool:
        """Install platform-specific package."""
        self.log(f"Installing platform package: {package_path}")
        
        try:
            if not Path(package_path).exists():
                self.log(f"❌ Package not found: {package_path}", "ERROR")
                return False
            
            if self.platform == "linux":
                return self._install_linux_package(package_path)
            elif self.platform == "windows":
                return self._install_windows_package(package_path)
            elif self.platform == "darwin":
                return self._install_macos_package(package_path)
            else:
                self.log(f"❌ Unsupported platform: {self.platform}", "ERROR")
                return False
                
        except Exception as e:
            self.log(f"❌ Platform installation failed: {e}", "ERROR")
            self.errors.append(f"Platform installation failed: {e}")
            return False
    
    def _install_linux_package(self, package_path: str) -> bool:
        """Install Linux package."""
        package_path = Path(package_path)
        
        if package_path.suffix == ".deb":
            # Install .deb package
            subprocess.run(["sudo", "dpkg", "-i", str(package_path)], check=True)
            subprocess.run(["sudo", "apt-get", "install", "-f"], check=True)
            self.log("✅ Debian package installed")
            
        elif package_path.suffix == ".rpm":
            # Install .rpm package
            subprocess.run(["sudo", "rpm", "-i", str(package_path)], check=True)
            self.log("✅ RPM package installed")
            
        elif "AppImage" in package_path.name:
            # Make AppImage executable and copy to bin
            subprocess.run(["chmod", "+x", str(package_path)], check=True)
            shutil.copy2(package_path, self.install_dir / "bin" / "schwabot")
            self.log("✅ AppImage installed")
            
        else:
            self.log(f"❌ Unsupported Linux package: {package_path.suffix}", "ERROR")
            return False
        
        return True
    
    def _install_windows_package(self, package_path: str) -> bool:
        """Install Windows package."""
        package_path = Path(package_path)
        
        if package_path.suffix == ".exe":
            # Copy executable to bin directory
            shutil.copy2(package_path, self.install_dir / "bin" / "schwabot.exe")
            self.log("✅ Windows executable installed")
            
        elif package_path.suffix == ".msi":
            # Install MSI package
            subprocess.run(["msiexec", "/i", str(package_path), "/quiet"], check=True)
            self.log("✅ MSI package installed")
            
        elif package_path.suffix == ".zip":
            # Extract portable package
            with zipfile.ZipFile(package_path, 'r') as zip_ref:
                zip_ref.extractall(self.install_dir / "portable")
            self.log("✅ Portable package extracted")
            
        else:
            self.log(f"❌ Unsupported Windows package: {package_path.suffix}", "ERROR")
            return False
        
        return True
    
    def _install_macos_package(self, package_path: str) -> bool:
        """Install macOS package."""
        package_path = Path(package_path)
        
        if package_path.suffix == ".app":
            # Copy app bundle to Applications
            shutil.copytree(package_path, Path("/Applications") / package_path.name)
            self.log("✅ macOS app bundle installed")
            
        elif package_path.suffix == ".dmg":
            # Mount and install DMG
            mount_point = f"/Volumes/{self.project_name}"
            subprocess.run(["hdiutil", "attach", str(package_path)], check=True)
            try:
                app_path = Path(mount_point) / f"{self.project_name}.app"
                if app_path.exists():
                    shutil.copytree(app_path, Path("/Applications") / f"{self.project_name}.app")
                    self.log("✅ DMG package installed")
                else:
                    self.log("❌ App bundle not found in DMG", "ERROR")
                    return False
            finally:
                subprocess.run(["hdiutil", "detach", mount_point])
                
        elif package_path.suffix == ".pkg":
            # Install PKG package
            subprocess.run(["sudo", "installer", "-pkg", str(package_path), "-target", "/"], check=True)
            self.log("✅ PKG package installed")
            
        else:
            self.log(f"❌ Unsupported macOS package: {package_path.suffix}", "ERROR")
            return False
        
        return True
    
    def setup_configuration(self) -> bool:
        """Setup initial configuration."""
        self.log("Setting up configuration...")
        
        try:
            # Create default configuration
            config = {
                "system": {
                    "name": self.project_name,
                    "version": self.version,
                    "environment": "production",
                    "install_path": str(self.install_dir)
                },
                "trading": {
                    "exchanges": ["binance", "coinbase", "kraken"],
                    "strategies": ["phantom_lag", "meta_layer_ghost"],
                    "risk_management": True,
                    "max_position_size": 0.1,
                    "stop_loss_percentage": 0.05
                },
                "monitoring": {
                    "dashboard_port": 8080,
                    "api_port": 8081,
                    "websocket_port": 8082,
                    "log_level": "INFO",
                    "enable_metrics": True
                },
                "security": {
                    "authentication": True,
                    "rate_limiting": True,
                    "ssl_enabled": False,
                    "allowed_origins": ["localhost"]
                },
                "performance": {
                    "thread_pool_size": 8,
                    "async_workers": 4,
                    "cache_size": "1G",
                    "max_memory": "4G"
                }
            }
            
            # Write configuration file
            config_file = self.config_dir / "schwabot_config.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            
            self.log(f"✅ Configuration created: {config_file}")
            
            # Create environment file
            env_file = self.install_dir / ".env"
            env_content = f"""# Schwabot Environment Configuration
SCHWABOT_ENV=production
SCHWABOT_LOG_LEVEL=INFO
SCHWABOT_CONFIG_PATH={config_file}
SCHWABOT_INSTALL_PATH={self.install_dir}
SCHWABOT_DATA_PATH={self.data_dir}
SCHWABOT_LOGS_PATH={self.logs_dir}
"""
            with open(env_file, 'w') as f:
                f.write(env_content)
            
            self.log(f"✅ Environment file created: {env_file}")
            
            return True
            
        except Exception as e:
            self.log(f"❌ Configuration setup failed: {e}", "ERROR")
            self.errors.append(f"Configuration setup failed: {e}")
            return False
    
    def setup_launcher_scripts(self) -> bool:
        """Create launcher scripts for easy access."""
        self.log("Creating launcher scripts...")
        
        try:
            if self.platform == "linux" or self.platform == "darwin":
                # Create shell script
                script_content = f"""#!/bin/bash
# Schwabot Launcher Script
export SCHWABOT_CONFIG_PATH="{self.config_dir}/schwabot_config.yaml"
export SCHWABOT_INSTALL_PATH="{self.install_dir}"

cd "$SCHWABOT_INSTALL_PATH"
exec python -m schwabot "$@"
"""
                script_path = self.install_dir / "bin" / "schwabot"
                with open(script_path, 'w') as f:
                    f.write(script_content)
                os.chmod(script_path, 0o755)
                
                # Create dashboard script
                dashboard_script = f"""#!/bin/bash
# Schwabot Dashboard Launcher
export SCHWABOT_CONFIG_PATH="{self.config_dir}/schwabot_config.yaml"
export SCHWABOT_INSTALL_PATH="{self.install_dir}"

cd "$SCHWABOT_INSTALL_PATH"
exec python -m schwabot.dashboard "$@"
"""
                dashboard_path = self.install_dir / "bin" / "schwabot-dashboard"
                with open(dashboard_path, 'w') as f:
                    f.write(dashboard_script)
                os.chmod(dashboard_path, 0o755)
                
            elif self.platform == "windows":
                # Create batch files
                script_content = f"""@echo off
REM Schwabot Launcher Script
set SCHWABOT_CONFIG_PATH={self.config_dir}\\schwabot_config.yaml
set SCHWABOT_INSTALL_PATH={self.install_dir}

cd /d "%SCHWABOT_INSTALL_PATH%"
python -m schwabot %*
"""
                script_path = self.install_dir / "bin" / "schwabot.bat"
                with open(script_path, 'w') as f:
                    f.write(script_content)
                
                # Create dashboard batch file
                dashboard_script = f"""@echo off
REM Schwabot Dashboard Launcher
set SCHWABOT_CONFIG_PATH={self.config_dir}\\schwabot_config.yaml
set SCHWABOT_INSTALL_PATH={self.install_dir}

cd /d "%SCHWABOT_INSTALL_PATH%"
python -m schwabot.dashboard %*
"""
                dashboard_path = self.install_dir / "bin" / "schwabot-dashboard.bat"
                with open(dashboard_path, 'w') as f:
                    f.write(dashboard_script)
            
            self.log("✅ Launcher scripts created")
            return True
            
        except Exception as e:
            self.log(f"❌ Launcher script creation failed: {e}", "ERROR")
            self.errors.append(f"Launcher script creation failed: {e}")
            return False
    
    def setup_desktop_integration(self) -> bool:
        """Setup desktop integration (shortcuts, menu entries)."""
        self.log("Setting up desktop integration...")
        
        try:
            if self.platform == "linux":
                # Create desktop entry
                desktop_entry = f"""[Desktop Entry]
Name=Schwabot
Comment=Hardware-scale-aware economic kernel for federated trading devices
Exec={self.install_dir}/bin/schwabot
Icon={self.install_dir}/docs/icon.png
Terminal=true
Type=Application
Categories=Office;Finance;
"""
                desktop_file = Path.home() / ".local" / "share" / "applications" / "schwabot.desktop"
                desktop_file.parent.mkdir(parents=True, exist_ok=True)
                with open(desktop_file, 'w') as f:
                    f.write(desktop_entry)
                
                self.log("✅ Desktop entry created")
                
            elif self.platform == "windows":
                # Create Start Menu shortcut
                import winshell
                from win32com.client import Dispatch
                
                start_menu = winshell.start_menu()
                programs = os.path.join(start_menu, "Programs")
                schwabot_folder = os.path.join(programs, "Schwabot")
                os.makedirs(schwabot_folder, exist_ok=True)
                
                shell = Dispatch('WScript.Shell')
                shortcut = shell.CreateShortCut(os.path.join(schwabot_folder, "Schwabot.lnk"))
                shortcut.Targetpath = str(self.install_dir / "bin" / "schwabot.bat")
                shortcut.WorkingDirectory = str(self.install_dir)
                shortcut.save()
                
                self.log("✅ Start Menu shortcut created")
                
            elif self.platform == "darwin":
                # macOS app bundle already handles this
                self.log("✅ Desktop integration handled by app bundle")
            
            return True
            
        except Exception as e:
            self.log(f"⚠️  Desktop integration setup failed: {e}", "WARNING")
            # Not critical, continue installation
            return True
    
    def validate_installation(self) -> bool:
        """Validate the installation."""
        self.log("Validating installation...")
        
        try:
            # Test import
            result = subprocess.run([
                sys.executable, "-c", "import schwabot; print('Import OK')"
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                self.log("❌ Package import test failed", "ERROR")
                return False
            
            self.log("✅ Package import test passed")
            
            # Test configuration
            config_file = self.config_dir / "schwabot_config.yaml"
            if not config_file.exists():
                self.log("❌ Configuration file not found", "ERROR")
                return False
            
            self.log("✅ Configuration file found")
            
            # Test launcher scripts
            if self.platform in ["linux", "darwin"]:
                launcher = self.install_dir / "bin" / "schwabot"
                if not launcher.exists():
                    self.log("❌ Launcher script not found", "ERROR")
                    return False
                
                # Test launcher
                result = subprocess.run([
                    str(launcher), "--version"
                ], capture_output=True, text=True, timeout=10)
                
                if result.returncode != 0:
                    self.log("❌ Launcher script test failed", "ERROR")
                    return False
                
                self.log("✅ Launcher script test passed")
            
            return True
            
        except Exception as e:
            self.log(f"❌ Installation validation failed: {e}", "ERROR")
            self.errors.append(f"Installation validation failed: {e}")
            return False
    
    def create_uninstaller(self) -> bool:
        """Create uninstaller script."""
        self.log("Creating uninstaller...")
        
        try:
            if self.platform in ["linux", "darwin"]:
                uninstall_script = f"""#!/bin/bash
# Schwabot Uninstaller

echo "Uninstalling Schwabot..."

# Remove installation directory
rm -rf "{self.install_dir}"

# Remove desktop entry (Linux)
if [ -f "$HOME/.local/share/applications/schwabot.desktop" ]; then
    rm "$HOME/.local/share/applications/schwabot.desktop"
fi

# Remove from PATH (if added)
if grep -q "schwabot" "$HOME/.bashrc"; then
    sed -i '/schwabot/d' "$HOME/.bashrc"
fi

echo "Schwabot uninstalled successfully!"
"""
                uninstaller_path = self.install_dir / "uninstall.sh"
                with open(uninstaller_path, 'w') as f:
                    f.write(uninstall_script)
                os.chmod(uninstaller_path, 0o755)
                
            elif self.platform == "windows":
                uninstall_script = f"""@echo off
REM Schwabot Uninstaller

echo Uninstalling Schwabot...

REM Remove installation directory
rmdir /s /q "{self.install_dir}"

REM Remove Start Menu shortcuts
rmdir /s /q "%APPDATA%\\Microsoft\\Windows\\Start Menu\\Programs\\Schwabot"

echo Schwabot uninstalled successfully!
pause
"""
                uninstaller_path = self.install_dir / "uninstall.bat"
                with open(uninstaller_path, 'w') as f:
                    f.write(uninstall_script)
            
            self.log("✅ Uninstaller created")
            return True
            
        except Exception as e:
            self.log(f"⚠️  Uninstaller creation failed: {e}", "WARNING")
            return True  # Not critical
    
    def save_installation_log(self) -> None:
        """Save installation log."""
        log_file = self.install_dir / "install.log"
        with open(log_file, 'w') as f:
            f.write("\n".join(self.installation_log))
        
        self.log(f"📋 Installation log saved: {log_file}")
    
    def print_summary(self) -> None:
        """Print installation summary."""
        print("\n" + "=" * 60)
        print("🎉 INSTALLATION SUMMARY")
        print("=" * 60)
        
        print(f"✅ {self.project_name} v{self.version} installed successfully!")
        print(f"📁 Installation directory: {self.install_dir}")
        print(f"⚙️  Configuration: {self.config_dir}/schwabot_config.yaml")
        print(f"📊 Logs directory: {self.logs_dir}")
        
        if self.platform in ["linux", "darwin"]:
            print(f"🚀 Launcher: {self.install_dir}/bin/schwabot")
            print(f"🌐 Dashboard: {self.install_dir}/bin/schwabot-dashboard")
        elif self.platform == "windows":
            print(f"🚀 Launcher: {self.install_dir}/bin/schwabot.bat")
            print(f"🌐 Dashboard: {self.install_dir}/bin/schwabot-dashboard.bat")
        
        print("\n📋 Quick Start:")
        print("1. Configure your trading settings:")
        print(f"   nano {self.config_dir}/schwabot_config.yaml")
        print("2. Start Schwabot:")
        if self.platform in ["linux", "darwin"]:
            print(f"   {self.install_dir}/bin/schwabot")
        else:
            print(f"   {self.install_dir}/bin/schwabot.bat")
        print("3. Access web dashboard: http://localhost:8080")
        
        if self.errors:
            print(f"\n⚠️  Warnings ({len(self.errors)}):")
            for error in self.errors:
                print(f"   - {error}")
        
        print(f"\n📚 Documentation: {self.install_dir}/docs/")
        print("🔧 Support: Check documentation or community forums")
        print("=" * 60)

def main():
    """Main installer function."""
    parser = argparse.ArgumentParser(description="Schwabot Professional Installer")
    parser.add_argument("--package", help="Path to Schwabot package file")
    parser.add_argument("--platform-package", help="Path to platform-specific package")
    parser.add_argument("--install-dir", help="Custom installation directory")
    parser.add_argument("--skip-validation", action="store_true", help="Skip installation validation")
    parser.add_argument("--quiet", action="store_true", help="Quiet mode")
    
    args = parser.parse_args()
    
    installer = SchwabotInstaller()
    
    if args.install_dir:
        installer.install_dir = Path(args.install_dir)
        installer.config_dir = installer.install_dir / "config"
        installer.logs_dir = installer.install_dir / "logs"
        installer.data_dir = installer.install_dir / "data"
    
    try:
        # Check system requirements
        if not installer.check_system_requirements():
            print("❌ System requirements not met. Installation aborted.")
            sys.exit(1)
        
        # Create directories
        if not installer.create_directories():
            print("❌ Failed to create installation directories.")
            sys.exit(1)
        
        # Install Python package
        if not installer.install_python_package(args.package):
            print("❌ Failed to install Python package.")
            sys.exit(1)
        
        # Install platform package if provided
        if args.platform_package:
            if not installer.install_platform_package(args.platform_package):
                print("❌ Failed to install platform package.")
                sys.exit(1)
        
        # Setup configuration
        if not installer.setup_configuration():
            print("❌ Failed to setup configuration.")
            sys.exit(1)
        
        # Create launcher scripts
        if not installer.setup_launcher_scripts():
            print("❌ Failed to create launcher scripts.")
            sys.exit(1)
        
        # Setup desktop integration
        installer.setup_desktop_integration()
        
        # Validate installation
        if not args.skip_validation:
            if not installer.validate_installation():
                print("❌ Installation validation failed.")
                sys.exit(1)
        
        # Create uninstaller
        installer.create_uninstaller()
        
        # Save installation log
        installer.save_installation_log()
        
        # Print summary
        installer.print_summary()
        
    except KeyboardInterrupt:
        print("\n❌ Installation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Installation failed: {e}")
        installer.save_installation_log()
        sys.exit(1)

if __name__ == "__main__":
    main() 