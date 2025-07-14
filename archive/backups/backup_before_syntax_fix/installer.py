import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import psutil
import winshell
import yaml
from win32com.client import Dispatch

from core.unified_math_system import unified_math
from dual_unicore_handler import DualUnicoreHandler
from utils.safe_print import debug, error, info, safe_print, success, warn

# -*- coding: utf - 8 -*-
# -*- coding: utf - 8 -*-
# -*- coding: utf - 8 -*-
# -*- coding: utf - 8 -*-


# Initialize Unicode handler
unicore = DualUnicoreHandler()

""""""
""""""
""""""
""""""
"""
Schwabot Professional Installer
===============================

This installer provides a comprehensive installation experience for Schwabot
across all supported platforms with proper validation, configuration, and setup."""
""""""
""""""
""""""
""""""
"""


class SchwabotInstaller:
"""
"""Professional installer for Schwabot trading system."""

"""
""""""
""""""
""""""
"""


def __init__(self): """
        """Initialize the installer.""""""

""""""
""""""
""""""
""""""
self.project_name = "Schwabot"
        self.version = "2.0_0"
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

safe_print(f"\\u1f680 {self.project_name} v{self.version} Installer")
        safe_print(f"\\u1f4ca Platform: {self.platform} ({self.arch})")
        safe_print(f"\\u1f4c1 Install directory: {self.install_dir}")
        safe_print("=" * 60)


def unified_math.log(self, message: str, level: str = "INFO") -> None:
    """Function implementation pending."""


pass
"""
"""Log installation messages.""""""
""""""
""""""
""""""
""""""
timestamp = subprocess.run(["date"], capture_output=True, text=True).stdout.strip()
        log_entry = f"[{timestamp}] {level}: {message}"
        self.installation_log.append(log_entry)
        safe_print(f"  {message}")


def check_system_requirements(): -> bool:
    """Function implementation pending."""


pass
"""
"""Check if system meets requirements.""""""
""""""
""""""
""""""
""""""
self.log("Checking system requirements...")

# Check Python version
python_version = sys.version_info
        if python_version < (3, 8):
            self.unified_math.log()
                f"\\u274c Python 3.8+ required, found {python_version.major}.{python_version.minor}", "ERROR")
            return False

self.unified_math.log()
    f"\\u2705 Python {"}
        python_version.major}.{
            python_version.minor}.{
                python_version.micro}")"

# Check available memory
    try:
memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
            if memory_gb < 4:
                self.unified_math.log()
                    f"\\u26a0\\ufe0f  Recommended: 4GB+ RAM, found {memory_gb:.1f}GB", "WARNING")
            else:
                self.unified_math.log(f"\\u2705 Memory: {memory_gb:.1f}GB")
        except ImportError:
            self.log("\\u26a0\\ufe0f  Could not check memory (psutil not, available)", "WARNING")

# Check disk space
    try:
            disk_usage = shutil.disk_usage(self.install_dir.parent)
            disk_gb = disk_usage.free / (1024**3)
            if disk_gb < 10:
                self.unified_math.log(f"\\u274c Need 10GB+ free space, found {disk_gb:.1f}GB", "ERROR")
                return False
    else:
                self.unified_math.log(f"\\u2705 Disk space: {disk_gb:.1f}GB available")
        except Exception as e:
            self.unified_math.log(f"\\u26a0\\ufe0f  Could not check disk space: {e}", "WARNING")

# Check network connectivity
    try:
            urllib.request.urlopen("https://pypi.org", timeout = 5)
            self.log("\\u2705 Network connectivity")
        except Exception:
            self.log("\\u26a0\\ufe0f  Network connectivity issues detected", "WARNING")

return True

def create_directories():-> bool:
    """Function implementation pending."""
    pass
"""
"""Create installation directories.""""""
""""""
""""""
""""""
""""""
self.log("Creating installation directories...")

try:
            directories = []
                self.install_dir,
                self.config_dir,
                self.logs_dir,
                self.data_dir,
                self.install_dir / "bin",
                self.install_dir / "lib",
                self.install_dir / "docs"
]
    for directory in directories:
                directory.mkdir(parents = True, exist_ok = True)
                self.unified_math.log(f"\\u2705 Created: {directory}")

return True

except Exception as e:
            self.unified_math.log(f"\\u274c Failed to create directories: {e}", "ERROR")
            self.errors.append(f"Directory creation failed: {e}")
            return False

def install_python_package():-> bool:
    """Function implementation pending."""
    pass
"""
"""Install Schwabot Python package.""""""
""""""
""""""
""""""
""""""
self.log("Installing Schwabot Python package...")

try:
            if package_path and Path(package_path).exists():
# Install from local package
subprocess.run([)]
                    sys.executable, "-m", "pip", "install", package_path
                ], check = True)
                self.unified_math.log(f"\\u2705 Installed from: {package_path}")
            else:
# Install from PyPI (if, available)
                subprocess.run([)]
                    sys.executable, "-m", "pip", "install", "schwabot"
                ], check = True)
                self.log("\\u2705 Installed from PyPI")

# Verify installation
result = subprocess.run([)]
                sys.executable, "-c", "import schwabot; print('OK')"
            ], capture_output = True, text = True)

if result.returncode == 0:
                self.log("\\u2705 Package verification successful")
                return True
    else:
                self.log("\\u274c Package verification failed", "ERROR")
                return False

except subprocess.CalledProcessError as e:
            self.unified_math.log(f"\\u274c Installation failed: {e}", "ERROR")
            self.errors.append(f"Package installation failed: {e}")
            return False

def install_platform_package():-> bool:
    """Function implementation pending."""
    pass
"""
"""Install platform - specific package.""""""
""""""
""""""
""""""
""""""
self.unified_math.log(f"Installing platform package: {package_path}")

try:
            if not Path(package_path).exists():
                self.unified_math.log(f"\\u274c Package not found: {package_path}", "ERROR")
                return False

if self.platform == "linux":
                return self._install_linux_package(package_path)
            elif self.platform == "windows":
                return self._install_windows_package(package_path)
            elif self.platform == "darwin":
                return self._install_macos_package(package_path)
            else:
                self.unified_math.log(f"\\u274c Unsupported platform: {self.platform}", "ERROR")
                return False

except Exception as e:
            self.unified_math.log(f"\\u274c Platform installation failed: {e}", "ERROR")
            self.errors.append(f"Platform installation failed: {e}")
            return False

def _install_linux_package():-> bool:
    """Function implementation pending."""
    pass
"""
"""Install Linux package.""""""
""""""
""""""
""""""
"""
package_path = Path(package_path)
"""
    if package_path.suffix == ".deb":
# Install .deb package
subprocess.run(["sudo", "dpkg", "-i", str(package_path)], check = True)
            subprocess.run(["sudo", "apt - get", "install", "-f"], check = True)"
            self.log("\\u2705 Debian package installed")

elif package_path.suffix == ".rpm":
# Install .rpm package
subprocess.run(["sudo", "rpm", "-i", str(package_path)], check = True)
            self.log("\\u2705 RPM package installed")

elif "AppImage" in package_path.name:
# Make AppImage executable and copy to bin
subprocess.run(["chmod", "+x", str(package_path)], check = True)
            shutil.copy2(package_path, self.install_dir / "bin" / "schwabot")
            self.log("\\u2705 AppImage installed")

else:
            self.unified_math.log(f"\\u274c Unsupported Linux package: {package_path.suffix}", "ERROR")
            return False

return True

def _install_windows_package():-> bool:
    """Function implementation pending."""
    pass
"""
"""Install Windows package.""""""
""""""
""""""
""""""
"""
package_path = Path(package_path)
"""
    if package_path.suffix == ".exe":
# Copy executable to bin directory
shutil.copy2(package_path, self.install_dir / "bin" / "schwabot.exe")
            self.log("\\u2705 Windows executable installed")

elif package_path.suffix == ".msi":
# Install MSI package
subprocess.run(["msiexec", "/i", str(package_path), "/quiet"], check = True)
            self.log("\\u2705 MSI package installed")

elif package_path.suffix == ".zip":
# Extract portable package
with zipfile.ZipFile(package_path, 'r') as zip_ref:
                zip_ref.extractall(self.install_dir / "portable")
            self.log("\\u2705 Portable package extracted")

else:
            self.unified_math.log(f"\\u274c Unsupported Windows package: {package_path.suffix}", "ERROR")
            return False

return True

def _install_macos_package():-> bool:
    """Function implementation pending."""
    pass
"""
"""Install macOS package.""""""
""""""
""""""
""""""
"""
package_path = Path(package_path)
"""
    if package_path.suffix == ".app":
# Copy app bundle to Applications
shutil.copytree(package_path, Path("/Applications") / package_path.name)
            self.log("\\u2705 macOS app bundle installed")

elif package_path.suffix == ".dmg":
# Mount and install DMG
mount_point = f"/Volumes/{self.project_name}"
            subprocess.run(["hdiutil", "attach", str(package_path)], check = True)
            try:
                app_path = Path(mount_point) / f"{self.project_name}.app"
                if app_path.exists():
                    shutil.copytree(app_path, Path("/Applications") / f"{self.project_name}.app")
                    self.log("\\u2705 DMG package installed")
                else:
                    self.log("\\u274c App bundle not found in DMG", "ERROR")
                    return False
    finally:
                subprocess.run(["hdiutil", "detach", mount_point])

elif package_path.suffix == ".pkg":
# Install PKG package
subprocess.run(["sudo", "installer", "-pkg", str(package_path), "-target", "/"], check = True)
            self.log("\\u2705 PKG package installed")

else:
            self.unified_math.log(f"\\u274c Unsupported macOS package: {package_path.suffix}", "ERROR")
            return False

return True

def setup_configuration():-> bool:
    """Function implementation pending."""
    pass
"""
"""Setup initial configuration.""""""
""""""
""""""
""""""
""""""
self.log("Setting up configuration...")

try:
    pass
# Create default configuration
config = {}
                "system": {}
                    "name": self.project_name,
                    "version": self.version,
                    "environment": "production",
                    "install_path": str(self.install_dir)
                },
                "trading": {}
                    "exchanges": ["binance", "coinbase", "kraken"],
                    "strategies": ["phantom_lag", "meta_layer_ghost"],
                    "risk_management": True,
                    "max_position_size": 0.1,
                    "stop_loss_percentage": 0.5
},
                "monitoring": {}
                    "dashboard_port": 8080,
                    "api_port": 8081,
                    "websocket_port": 8082,
                    "log_level": "INFO",
                    "enable_metrics": True
},
                "security": {}
                    "authentication": True,
                    "rate_limiting": True,
                    "ssl_enabled": False,
                    "allowed_origins": ["localhost"]
                },
                "performance": {}
                    "thread_pool_size": 8,
                    "async_workers": 4,
                    "cache_size": "1G",
                    "max_memory": "4G"

# Write configuration file
config_file = self.config_dir / "schwabot_config.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(config, f, default_flow_style = False, indent = 2)

self.unified_math.log(f"\\u2705 Configuration created: {config_file}")

# Create environment file
env_file = self.install_dir / ".env"
            env_content = f"""  # Schwabot Environment Configuration"
SCHWABOT_ENV = production
SCHWABOT_LOG_LEVEL = INFO
SCHWABOT_CONFIG_PATH={config_file}
SCHWABOT_INSTALL_PATH={self.install_dir}
SCHWABOT_DATA_PATH={self.data_dir}
SCHWABOT_LOGS_PATH={self.logs_dir}"""
""""""
""""""
""""""
""""""
"""
with open(env_file, 'w') as f:
                f.write(env_content)
"""
self.unified_math.log(f"\\u2705 Environment file created: {env_file}")

return True

except Exception as e:
            self.unified_math.log(f"\\u274c Configuration setup failed: {e}", "ERROR")
            self.errors.append(f"Configuration setup failed: {e}")
            return False

def setup_launcher_scripts():-> bool:
    """Function implementation pending."""
    pass
"""
"""Create launcher scripts for easy access.""""""
""""""
""""""
""""""
""""""
self.log("Creating launcher scripts...")

try:
            if self.platform == "linux" or self.platform == "darwin":
# Create shell script
script_content = f"""  #!/bin / bash"
# Schwabot Launcher Script"""
export SCHWABOT_CONFIG_PATH="{self.config_dir}/schwabot_config.yaml"
export SCHWABOT_INSTALL_PATH="{self.install_dir}"

cd "$SCHWABOT_INSTALL_PATH"
exec python -m schwabot "$@"
""""""
""""""
""""""
""""""
""""""
script_path = self.install_dir / "bin" / "schwabot"
                with open(script_path, 'w') as f:
                    f.write(script_content)
                os.chmod(script_path, 0o755)

# Create dashboard script
dashboard_script = f"""  #!/bin / bash"
# Schwabot Dashboard Launcher"""
export SCHWABOT_CONFIG_PATH="{self.config_dir}/schwabot_config.yaml"
export SCHWABOT_INSTALL_PATH="{self.install_dir}"

cd "$SCHWABOT_INSTALL_PATH"
exec python -m schwabot.dashboard "$@"
""""""
""""""
""""""
""""""
""""""
dashboard_path = self.install_dir / "bin" / "schwabot - dashboard"
                with open(dashboard_path, 'w') as f:
                    f.write(dashboard_script)
                os.chmod(dashboard_path, 0o755)

elif self.platform == "windows":
# Create batch files
script_content = f"""@echo off""
REM Schwabot Launcher Script
set SCHWABOT_CONFIG_PATH={self.config_dir}\\\schwabot_config.yaml
set SCHWABOT_INSTALL_PATH={self.install_dir}
"""
cd /d "%SCHWABOT_INSTALL_PATH%"
python -m schwabot %*
""""""
""""""
""""""
""""""
""""""
script_path = self.install_dir / "bin" / "schwabot.bat"
                with open(script_path, 'w') as f:
                    f.write(script_content)

# Create dashboard batch file
dashboard_script = f"""@echo off""
REM Schwabot Dashboard Launcher
set SCHWABOT_CONFIG_PATH={self.config_dir}\\\schwabot_config.yaml
set SCHWABOT_INSTALL_PATH={self.install_dir}
"""
cd /d "%SCHWABOT_INSTALL_PATH%"
python -m schwabot.dashboard %*
""""""
""""""
""""""
""""""
""""""
dashboard_path = self.install_dir / "bin" / "schwabot - dashboard.bat"
                with open(dashboard_path, 'w') as f:
                    f.write(dashboard_script)

self.log("\\u2705 Launcher scripts created")
            return True

except Exception as e:
            self.unified_math.log(f"\\u274c Launcher script creation failed: {e}", "ERROR")
            self.errors.append(f"Launcher script creation failed: {e}")
            return False

def setup_desktop_integration():-> bool:
    """Function implementation pending."""
    pass
"""
"""Setup desktop integration (shortcuts, menu, entries).""""""
""""""
""""""
""""""
""""""
self.log("Setting up desktop integration...")

try:
            if self.platform == "linux":
# Create desktop entry
desktop_entry = f"""[Desktop Entry]"
Name = Schwabot
Comment = Hardware - scale - aware economic kernel for federated trading devices
Exec={self.install_dir}/bin / schwabot
Icon={self.install_dir}/docs / icon.png
Terminal = true
Type = Application
Categories = Office;Finance;"""
""""""
""""""
""""""
""""""
""""""
desktop_file = Path.home() / ".local" / "share" / "applications" / "schwabot.desktop"
                desktop_file.parent.mkdir(parents = True, exist_ok = True)
                with open(desktop_file, 'w') as f:
                    f.write(desktop_entry)

self.log("\\u2705 Desktop entry created")

elif self.platform == "windows":
# Create Start Menu shortcut

start_menu = winshell.start_menu()
                programs = os.path.join(start_menu, "Programs")
                schwabot_folder = os.path.join(programs, "Schwabot")
                os.makedirs(schwabot_folder, exist_ok = True)

shell = Dispatch('WScript.Shell')
                shortcut = shell.CreateShortCut(os.path.join(schwabot_folder, "Schwabot.lnk"))
                shortcut.Targetpath = str(self.install_dir / "bin" / "schwabot.bat")
                shortcut.WorkingDirectory = str(self.install_dir)
                shortcut.save()

self.log("\\u2705 Start Menu shortcut created")

elif self.platform == "darwin":
# macOS app bundle already handles this
self.log("\\u2705 Desktop integration handled by app bundle")

return True

except Exception as e:
            self.unified_math.log(f"\\u26a0\\ufe0f  Desktop integration setup failed: {e}", "WARNING")
# Not critical, continue installation
            return True

def validate_installation():-> bool:
    """Function implementation pending."""
    pass
"""
"""Validate the installation.""""""
""""""
""""""
""""""
""""""
self.log("Validating installation...")

try:
    pass
# Test import
result = subprocess.run([)]
                sys.executable, "-c", "import schwabot; print('Import OK')"
            ], capture_output = True, text = True)

if result.returncode != 0:
                self.log("\\u274c Package import test failed", "ERROR")
                return False

self.log("\\u2705 Package import test passed")

# Test configuration
config_file = self.config_dir / "schwabot_config.yaml"
            if not config_file.exists():
                self.log("\\u274c Configuration file not found", "ERROR")
                return False

self.log("\\u2705 Configuration file found")

# Test launcher scripts
    if self.platform in ["linux", "darwin"]:
                launcher = self.install_dir / "bin" / "schwabot"
                if not launcher.exists():
                    self.log("\\u274c Launcher script not found", "ERROR")
                    return False

# Test launcher
result = subprocess.run([)]
                    str(launcher), "--version"
                ], capture_output = True, text = True, timeout = 10)

if result.returncode != 0:
                    self.log("\\u274c Launcher script test failed", "ERROR")
                    return False

self.log("\\u2705 Launcher script test passed")

return True

except Exception as e:
            self.unified_math.log(f"\\u274c Installation validation failed: {e}", "ERROR")
            self.errors.append(f"Installation validation failed: {e}")
            return False

def create_uninstaller():-> bool:
        """
        Optimize mathematical function for trading performance.

        Args:
            data: Input data array
            target: Target optimization value
            **kwargs: Additional parameters

        Returns:
            Optimized result
        """
        try:

            # Apply mathematical optimization
            if target is not None:
                result = unified_math.optimize_towards_target(data, target)
            else:
                result = unified_math.general_optimization(data)

            return result

        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return data
    pass
"""
"""Create uninstaller script.""""""
""""""
""""""
""""""
""""""
self.log("Creating uninstaller...")

try:
            if self.platform in ["linux", "darwin"]:
                uninstall_script = f"""  #!/bin / bash"
# Schwabot Uninstaller
"""
echo "Uninstalling Schwabot..."

# Remove installation directory
rm -rf "{self.install_dir}"

# Remove desktop entry (Linux)
    if [ -f "$HOME/.local / share / applications / schwabot.desktop" ]; then
    rm "$HOME/.local / share / applications / schwabot.desktop"
fi

# Remove from PATH (if, added)
    if grep -q "schwabot" "$HOME/.bashrc"; then
sed -i '/schwabot / d' "$HOME/.bashrc"
fi

echo "Schwabot uninstalled successfully!"
""""""
""""""
""""""
""""""
""""""
uninstaller_path = self.install_dir / "uninstall.sh"
                with open(uninstaller_path, 'w') as f:
                    f.write(uninstall_script)
                os.chmod(uninstaller_path, 0o755)

elif self.platform == "windows":
                uninstall_script = f"""@echo off""
REM Schwabot Uninstaller

echo Uninstalling Schwabot...

REM Remove installation directory"""
rmdir /s /q "{self.install_dir}"

REM Remove Start Menu shortcuts
rmdir /s /q "%APPDATA%\\Microsoft\\Windows\\Start Menu\\Programs\\Schwabot"

echo Schwabot uninstalled successfully!
pause
""""""
""""""
""""""
""""""
""""""
uninstaller_path = self.install_dir / "uninstall.bat"
                with open(uninstaller_path, 'w') as f:
                    f.write(uninstall_script)

self.log("\\u2705 Uninstaller created")
            return True

except Exception as e:
            self.unified_math.log(f"\\u26a0\\ufe0f  Uninstaller creation failed: {e}", "WARNING")
            return True  # Not critical

def save_installation_log():-> None:
    """Function implementation pending."""
    pass
"""
"""Save installation log.""""""
""""""
""""""
""""""
""""""
log_file = self.install_dir / "install.log"
        with open(log_file, 'w') as f:
            f.write("\n".join(self.installation_log))

self.unified_math.log(f"\\u1f4cb Installation log saved: {log_file}")

def print_summary():-> None:
    """Function implementation pending."""
    pass
"""
"""Print installation summary.""""""
""""""
""""""
""""""
""""""
safe_print("\n" + "=" * 60)
        safe_print("\\u1f389 INSTALLATION SUMMARY")
        safe_print("=" * 60)

safe_print(f"\\u2705 {self.project_name} v{self.version} installed successfully!")
        safe_print(f"\\u1f4c1 Installation directory: {self.install_dir}")
        safe_print(f"\\u2699\\ufe0f  Configuration: {self.config_dir}/schwabot_config.yaml")
        safe_print(f"\\u1f4ca Logs directory: {self.logs_dir}")

if self.platform in ["linux", "darwin"]:
            safe_print(f"\\u1f680 Launcher: {self.install_dir}/bin / schwabot")
            safe_print(f"\\u1f310 Dashboard: {self.install_dir}/bin / schwabot - dashboard")
        elif self.platform == "windows":
            safe_print(f"\\u1f680 Launcher: {self.install_dir}/bin / schwabot.bat")
            safe_print(f"\\u1f310 Dashboard: {self.install_dir}/bin / schwabot - dashboard.bat")

safe_print("\\n\\u1f4cb Quick Start:")
        safe_print("1. Configure your trading settings:")
        safe_print(f"   nano {self.config_dir}/schwabot_config.yaml")
        safe_print("2. Start Schwabot:")
        if self.platform in ["linux", "darwin"]:
            safe_print(f"   {self.install_dir}/bin / schwabot")
        else:
            safe_print(f"   {self.install_dir}/bin / schwabot.bat")
        safe_print("3. Access web dashboard: http://localhost:8080")

if self.errors:
            safe_print(f"\\n\\u26a0\\ufe0f  Warnings ({len(self.errors)}):")
            for error in self.errors:
                safe_print(f"   - {error}")

safe_print(f"\\n\\u1f4da Documentation: {self.install_dir}/docs/")
        safe_print("\\u1f527 Support: Check documentation or community forums")
        safe_print("=" * 60)


def main():
    """Function implementation pending."""
    pass
"""
"""Main installer function.""""""
""""""
""""""
""""""
""""""
parser = argparse.ArgumentParser(description="Schwabot Professional Installer")
    parser.add_argument("--package", help="Path to Schwabot package file")
    parser.add_argument("--platform - package", help="Path to platform - specific package")
    parser.add_argument("--install - dir", help="Custom installation directory")
    parser.add_argument("--skip - validation", action="store_true", help="Skip installation validation")
    parser.add_argument("--quiet", action="store_true", help="Quiet mode")

args = parser.parse_args()

installer = SchwabotInstaller()

if args.install_dir:
        installer.install_dir = Path(args.install_dir)
        installer.config_dir = installer.install_dir / "config"
        installer.logs_dir = installer.install_dir / "logs"
        installer.data_dir = installer.install_dir / "data"

try:
    pass
# Check system requirements
    if not installer.check_system_requirements():
            safe_print("\\u274c System requirements not met. Installation aborted.")
            sys.exit(1)

# Create directories
    if not installer.create_directories():
            safe_print("\\u274c Failed to create installation directories.")
            sys.exit(1)

# Install Python package
    if not installer.install_python_package(args.package):
            safe_print("\\u274c Failed to install Python package.")
            sys.exit(1)

# Install platform package if provided
    if args.platform_package:
            if not installer.install_platform_package(args.platform_package):
                safe_print("\\u274c Failed to install platform package.")
                sys.exit(1)

# Setup configuration
    if not installer.setup_configuration():
            safe_print("\\u274c Failed to setup configuration.")
            sys.exit(1)

# Create launcher scripts
    if not installer.setup_launcher_scripts():
            safe_print("\\u274c Failed to create launcher scripts.")
            sys.exit(1)

# Setup desktop integration
installer.setup_desktop_integration()

# Validate installation
    if not args.skip_validation:
            if not installer.validate_installation():
                safe_print("\\u274c Installation validation failed.")
                sys.exit(1)

# Create uninstaller
installer.create_uninstaller()

# Save installation log
installer.save_installation_log()

# Print summary
installer.print_summary()

except KeyboardInterrupt:
        safe_print("\\n\\u274c Installation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        safe_print(f"\\n\\u274c Installation failed: {e}")
        installer.save_installation_log()
        sys.exit(1)


if __name__ == "__main__":
    main()
