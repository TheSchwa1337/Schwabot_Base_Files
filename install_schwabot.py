#!/usr/bin/env python3
""""""
import yaml
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
import platform
import subprocess
import sys
import os
Schwabot Complete System Installer
== == == == == == == == == == == == == == == == ==

Comprehensive installer for the Schwabot Advanced Trading System:
- Checks system requirements
- Installs required dependencies
- Configures cross - platform settings
- Sets up proper directory structure
- Initializes configuration files
- Verifies installation integrity
- Creates desktop shortcuts(optional)

Supports Windows, macOS, and Linux.
""""""


# Set up logging
logging.basicConfig()
    level = logging.INFO,
        format = '%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SchwabotInstaller:
    """Complete system installer for Schwabot."""

    def __init__(self):
        self.system = platform.system().lower()
        self.python_version = sys.version_info
        self.install_dir = Path.cwd()

        # Installation status
        self.installation_log: List[Dict[str, Any]] = []
        self.failed_packages: List[str] = []
        self.success_count = 0
        self.error_count = 0

        # System requirements
        self.min_python_version = (3, 8)
        self.recommended_python_version = (3, 11)

        print("🚀 Schwabot Advanced Trading System Installer")
        print("=" * 60)
        print(f"Operating System: {platform.system()} {platform.release()}")
        print(f"Python Version: {sys.version}")
        print(f"Installation Directory: {self.install_dir}")
        print()

    def check_system_requirements(self) -> bool:
        """Check if system meets minimum requirements."""
        print("🔍 Checking System Requirements")
        print("-" * 40)

        # Check Python version
        if self.python_version < self.min_python_version:
            print(
    f"❌ Python {
        self.min_python_version[0]}.{
            self.min_python_version[1]}+ required")
            print(
    f"   Current version: {
        self.python_version[0]}.{
            self.python_version[1]}")
            return False

        print(
    f"✅ Python version: {
        self.python_version[0]}.{
            self.python_version[1]}.{
                self.python_version[2]}")

        if self.python_version < self.recommended_python_version:
            print(
    f"⚠️ Python {
        self.recommended_python_version[0]}.{
            self.recommended_python_version[1]}+ recommended for best performance")

        # Check pip
        try:
            import pip
            print("✅ pip is available")
        except ImportError:
            print("❌ pip is required but not available")
            return False

        # Check disk space (minimum 1GB)
        try:
            import shutil
            free_space = shutil.disk_usage(self.install_dir).free / (1024**3)
            if free_space < 1.0:
                print(
    f"❌ Insufficient disk space: {
        free_space:.1f}GB available, 1GB required")
                return False
            print(f"✅ Disk space: {free_space:.1f}GB available")
        except Exception as e:
            print(f"⚠️ Could not check disk space: {e}")

        # Check internet connection
        try:
            import urllib.request
            urllib.request.urlopen('https://pypi.org', timeout=10)
            print("✅ Internet connection available")
        except Exception:
            print("⚠️ Internet connection may be limited")

        print()
        return True

    def install_core_dependencies(self) -> bool:
        """Install core Python dependencies."""
        print("📦 Installing Core Dependencies")
        print("-" * 40)

        # Core packages that are essential
        core_packages = []
            "numpy>=1.21.0",
                "pandas>=1.3.0",
                    "scipy>=1.7.0",
                    "pyyaml>=6.0",
                    "requests>=2.28.0",
                    "python-dateutil>=2.8.0",
                    "dataclasses-json>=0.6.0",
]
        return self._install_packages(core_packages, "Core")

    def install_gui_dependencies(self) -> bool:
        """Install GUI and visualization dependencies."""
        print("🖼️ Installing GUI Dependencies")
        print("-" * 40)

        gui_packages = []
            "matplotlib>=3.5.0",
                "seaborn>=0.11.0",
                    "plotly>=5.0.0",
]
        # tkinter is usually included with Python, but check anyway
        try:
            import tkinter
            print("✅ tkinter (GUI toolkit) is available")
        except ImportError:
            print("⚠️ tkinter not available - GUI features will be limited")

        return self._install_packages(gui_packages, "GUI")

    def install_trading_dependencies(self) -> bool:
        """Install trading and API dependencies."""
        print("📈 Installing Trading Dependencies")
        print("-" * 40)

        trading_packages = []
            "ccxt>=3.0.0",
                "websockets>=10.0",
                    "aiohttp>=3.8.0",
                    "python-socketio>=5.0.0",
]
        return self._install_packages(trading_packages, "Trading")

    def install_performance_dependencies(self) -> bool:
        """Install performance optimization dependencies."""
        print("⚡ Installing Performance Dependencies")
        print("-" * 40)

        performance_packages = []
            "psutil>=5.8.0",
]
        # GPU packages (optional)
        gpu_packages = []

        # Check for CUDA availability
        cuda_available = self._check_cuda_availability()
        if cuda_available:
            print("🔥 CUDA detected - installing GPU acceleration packages")
            if self.system == "linux":
                gpu_packages.append("cupy-cuda11x>=10.0.0")
            elif self.system == "windows":
                gpu_packages.append("cupy-cuda11x>=10.0.0")
        else:
            print("ℹ️ CUDA not detected - GPU acceleration will be unavailable")

        # Install performance packages
        success = self._install_packages(performance_packages, "Performance")

        # Install GPU packages if available
        if gpu_packages:
            gpu_success = self._install_packages(gpu_packages, "GPU")
            return success and gpu_success

        return success

    def install_optional_dependencies(self) -> bool:
        """Install optional dependencies for enhanced features."""
        print("🔧 Installing Optional Dependencies")
        print("-" * 40)

        optional_packages = []
            "cryptography>=3.4.0",  # For encrypted settings
            "fastapi>=0.68.0",      # For API server
            "uvicorn[standard]>=0.15.0",  # For API server
            "streamlit>=1.0.0",     # For web dashboard
]
        return self._install_packages(optional_packages, "Optional", continue_on_error=True)

    def install_development_dependencies(self) -> bool:
        """Install development and testing dependencies."""
        print("🧪 Installing Development Dependencies")
        print("-" * 40)

        dev_packages = []
            "pytest>=7.0.0",
                "pytest-asyncio>=0.18.0",
                    "flake8>=4.0.0",
                    "mypy>=0.950",
                    "black>=22.0.0",
]
        return self._install_packages(dev_packages, "Development", continue_on_error=True)

    def _install_packages(self, packages: List[str], category: str, )
                         continue_on_error: bool = False) -> bool:
        """Install a list of packages."""
        success = True

        for package in packages:
            try:
                print(f"📥 Installing {package}...")

                # Run pip install
                result = subprocess.run()
                    [sys.executable, "-m", "pip", "install", package],
                        capture_output=True,
                            text=True,
                            timeout=300  # 5 minute timeout
                )

                if result.returncode == 0:
                    print(f"✅ {package} installed successfully")
                    self.success_count += 1
                    self.installation_log.append({)}
                        "package": package,
                            "category": category,
                                "status": "success",
                                "output": result.stdout[:200]  # First 200 chars
                    })
                else:
                    error_msg = result.stderr or "Unknown error"
                    print(f"❌ Failed to install {package}: {error_msg[:100]}")
                    self.error_count += 1
                    self.failed_packages.append(package)
                    self.installation_log.append({)}
                        "package": package,
                            "category": category,
                                "status": "failed",
                                "error": error_msg[:200]
                    })

                    if not continue_on_error:
                        success = False

            except subprocess.TimeoutExpired:
                print(f"⏰ Timeout installing {package}")
                self.failed_packages.append(package)
                if not continue_on_error:
                    success = False

            except Exception as e:
                print(f"❌ Error installing {package}: {e}")
                self.failed_packages.append(package)
                if not continue_on_error:
                    success = False

        return success

    def _check_cuda_availability(self) -> bool:
        """Check if CUDA is available on the system."""
        try:
            # Check for nvidia-smi
            result = subprocess.run()
                ["nvidia-smi"],
                    capture_output=True,
                        text=True,
                        timeout=10
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def setup_directory_structure(self) -> bool:
        """Create necessary directory structure."""
        print("📁 Setting Up Directory Structure")
        print("-" * 40)

        directories = [
            "config",
            "logs",
            "data",
            "backups",
            "exports",
            "temp"
]
]
        try:
            for directory in directories:
                dir_path = self.install_dir / directory
                dir_path.mkdir(exist_ok=True)
                print(f"✅ Created directory: {directory}/")

                # Create .gitkeep file to ensure directory is tracked
                gitkeep_file = dir_path / ".gitkeep"
                gitkeep_file.touch()

            return True

        except Exception as e:
            print(f"❌ Failed to create directory structure: {e}")
            return False

    def create_configuration_files(self) -> bool:
        """Create initial configuration files."""
        print("⚙️ Creating Configuration Files")
        print("-" * 40)

        try:
            # Create main config file
            config_data = {
                "system": {}
                "version": "2.0.0",
                "installation_date": str(Path(__file__).stat().st_mtime),
                "platform": platform.system()
}
                },
                    "performance": {}
                    "gpu_enabled": self._check_cuda_availability(),
                        "cpu_threads": os.cpu_count() or 4,
                            "memory_limit_mb": 2048,
                            "update_interval": 1.0
                },
                    "trading": {}
                    "trading_mode": "demo",
                        "base_currency": "USD",
                            "target_currency": "BTC",
                            "max_concurrent_trades": 5
}
}
            config_file = self.install_dir / "config" / "schwabot_config.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
            print(f"✅ Created configuration file: {config_file}")

            # Create environment template
            env_template = """# Schwabot Environment Configuration"""
# Copy this file to .env and update with your actual values

# API Configuration
COINBASE_API_KEY=your_api_key_here
COINBASE_SECRET=your_secret_here
COINBASE_PASSPHRASE=your_passphrase_here
SANDBOX_MODE=true

# Performance Settings
GPU_ENABLED=true
CPU_THREADS=4
MEMORY_LIMIT_MB=2048

# Trading Settings
TRADING_MODE=demo
MAX_CONCURRENT_TRADES=5
MIN_TRADE_AMOUNT=10.0
""""""

            env_file = self.install_dir / ".env.template"
            with open(env_file, 'w') as f:
                f.write(env_template)
            print(f"✅ Created environment template: {env_file}")

            return True

        except Exception as e:
            print(f"❌ Failed to create configuration files: {e}")
            return False

    def verify_installation(self) -> bool:
        """Verify that the installation was successful."""
        print("🔍 Verifying Installation")
        print("-" * 40)

        try:
            # Test core imports
            test_imports = []
                ("numpy", "NumPy"),
                    ("pandas", "Pandas"),
                        ("matplotlib", "Matplotlib"),
                        ("yaml", "PyYAML"),
                        ("requests", "Requests"),
]
            for module, name in test_imports:
                try:
                    __import__(module)
                    print(f"✅ {name} import successful")
                except ImportError:
                    print(f"❌ {name} import failed")
                    return False

            # Test Schwabot components
            try:
                from core.settings_manager import get_settings_manager
                settings = get_settings_manager()
                print("✅ Settings Manager working")
            except Exception as e:
                print(f"⚠️ Settings Manager test failed: {e}")

            try:
                from core.gpu_cpu_calculation_bridge import get_gpu_cpu_bridge
                bridge = get_gpu_cpu_bridge()
                print(f"✅ GPU/CPU Bridge working (GPU: {bridge.gpu_available})")
            except Exception as e:
                print(f"⚠️ GPU/CPU Bridge test failed: {e}")

            print("✅ Installation verification completed")
            return True

        except Exception as e:
            print(f"❌ Installation verification failed: {e}")
            return False

    def create_shortcuts(self) -> bool:
        """Create desktop shortcuts and start menu entries."""
        print("🔗 Creating Shortcuts")
        print("-" * 40)

        try:
            if self.system == "windows":
                return self._create_windows_shortcuts()
            elif self.system == "darwin":  # macOS
                return self._create_macos_shortcuts()
            elif self.system == "linux":
                return self._create_linux_shortcuts()
            else:
                print(f"⚠️ Shortcuts not supported on {self.system}")
                return True

        except Exception as e:
            print(f"⚠️ Failed to create shortcuts: {e}")
            return True  # Non-critical failure

    def _create_windows_shortcuts(self) -> bool:
        """Create Windows shortcuts."""
        try:
            import winshell
            from win32com.client import Dispatch

            desktop = winshell.desktop()
            shell = Dispatch('WScript.Shell')

            # Create GUI shortcut
            shortcut_path = os.path.join(desktop, "Schwabot Trading System.lnk")
            shortcut = shell.CreateShortCut(shortcut_path)
            shortcut.Targetpath = sys.executable
            shortcut.Arguments = f'"{self.install_dir / "schwabot_main.py"}" --gui'
            shortcut.WorkingDirectory = str(self.install_dir)
            shortcut.IconLocation = sys.executable
            shortcut.save()

            print("✅ Windows shortcuts created")
            return True

        except ImportError:
            print("⚠️ Windows shortcut creation requires winshell and pywin32")
            return True
        except Exception as e:
            print(f"⚠️ Windows shortcut creation failed: {e}")
            return True

    def _create_macos_shortcuts(self) -> bool:
        """Create macOS shortcuts."""
        # macOS shortcut creation would go here
        print("ℹ️ macOS shortcuts not yet implemented")
        return True

    def _create_linux_shortcuts(self) -> bool:
        """Create Linux shortcuts."""
        try:
            desktop_dir = Path.home() / "Desktop"
            if not desktop_dir.exists():
                desktop_dir = Path.home()

            desktop_file_content = f"""[Desktop Entry]"""
Version=1.0
Type=Application
Name=Schwabot Trading System
Comment=Advanced cryptocurrency trading bot
Exec={sys.executable} {self.install_dir / "schwabot_main.py"} --gui
Icon={self.install_dir / "icon.png"}
Terminal=false
Categories=Office;Finance;
""""""

            desktop_file = desktop_dir / "schwabot.desktop"
            with open(desktop_file, 'w') as f:
                f.write(desktop_file_content)

            # Make executable
            os.chmod(desktop_file, 0o755)

            print("✅ Linux shortcuts created")
            return True

        except Exception as e:
            print(f"⚠️ Linux shortcut creation failed: {e}")
            return True

    def generate_installation_report(self) -> str:
        """Generate a comprehensive installation report."""
        report = f""""""
Schwabot Installation Report
============================

Installation Date: {Path(__file__).stat().st_mtime}
System: {platform.system()} {platform.release()}
Python: {sys.version}
Installation Directory: {self.install_dir}

Installation Summary:
- Successful packages: {self.success_count}
- Failed packages: {self.error_count}
- Total packages attempted: {self.success_count + self.error_count}

Failed Packages:
{chr(10).join(f"- {pkg}" for pkg in self.failed_packages) if self.failed_packages else "None"}

Next Steps:
1. Copy .env.template to .env and configure your API keys
2. Run 'python demo_complete_system.py' to test the installation
3. Run 'python schwabot_main.py --gui' to start the GUI
4. Run 'python schwabot_main.py --mode demo' for CLI demo

For support, documentation, and updates:
- GitHub: https://github.com/schwabot/schwabot
- Documentation: https://docs.schwabot.com
""""""

        # Save report to file
        report_file = self.install_dir / "installation_report.txt"
        with open(report_file, 'w') as f:
            f.write(report)

        return report

    def run_installation(self) -> bool:
        """Run the complete installation process."""
        print("🚀 Starting Schwabot Installation")
        print("=" * 60)

        try:
            # Step 1: Check requirements
            if not self.check_system_requirements():
                print("❌ System requirements check failed")
                return False

            # Step 2: Install dependencies
            steps = []
                ("Core Dependencies", self.install_core_dependencies),
                    ("GUI Dependencies", self.install_gui_dependencies),
                        ("Trading Dependencies", self.install_trading_dependencies),
                        ("Performance Dependencies", self.install_performance_dependencies),
                        ("Optional Dependencies", self.install_optional_dependencies),
                        ("Development Dependencies", self.install_development_dependencies),
]
            for step_name, step_func in steps:
                print(f"\n🔧 {step_name}")
                if not step_func():
                    print(f"⚠️ {step_name} had some failures, but continuing...")

            # Step 3: Setup structure
            if not self.setup_directory_structure():
                print("❌ Directory structure setup failed")
                return False

            # Step 4: Create config files
            if not self.create_configuration_files():
                print("❌ Configuration file creation failed")
                return False

            # Step 5: Verify installation
            if not self.verify_installation():
                print("❌ Installation verification failed")
                return False

            # Step 6: Create shortcuts
            self.create_shortcuts()

            # Step 7: Generate report
            report = self.generate_installation_report()

            print("\n🎉 Installation Completed Successfully!")
            print("=" * 60)
            print(report)

            return True

        except KeyboardInterrupt:
            print("\n🛑 Installation interrupted by user")
            return False
        except Exception as e:
            print(f"\n❌ Installation failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main installer entry point."""
    try:
        installer = SchwabotInstaller()
        success = installer.run_installation()

        if success:
            print("\n✅ Schwabot is ready to use!")
            print("   Run 'python schwabot_main.py --gui' to start")
            sys.exit(0)
        else:
            print("\n❌ Installation completed with errors")
            print("   Check the installation report for details")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n🛑 Installation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Installation crashed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()