import argparse
import json
import logging
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

import ccxt

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Schwabot Installation Script
============================

Cross-platform installation script for the Schwabot trading system.
Handles dependency installation, configuration setup, and system validation.

Usage:
    python install.py                    # Interactive installation
    python install.py --auto             # Automatic installation
    python install.py --check            # Check system requirements
    python install.py --configure        # Configure only
"""


# Configure logging
logging.basicConfig()
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SystemInfo:
    """System information and compatibility checker."""

    def __init__(self):
        self.platform = platform.system().lower()
        self.python_version = sys.version_info
        self.is_windows = self.platform == "windows"
        self.is_macos = self.platform == "darwin"
        self.is_linux = self.platform == "linux"

    def get_python_version_str(): -> str:
        """Get Python version as string."""
        return f"{self.python_version.major}.{self.python_version.minor}.{"}
            self.python_version.micro
        }"

    def check_python_version(): -> bool:
        """Check if Python version is compatible."""
        return self.python_version >= (3, 8)

    def get_pip_command(): -> str:
        """Get appropriate pip command for the system."""
        if self.is_windows:
            return "python -m pip"
        else:
            return "pip3"

    def get_venv_command(): -> str:
        """Get appropriate virtual environment command."""
        if self.is_windows:
            return "python -m venv"
        else:
            return "python3 -m venv"


class DependencyManager:
    """Manages dependency installation and verification."""

    def __init__(self, system_info: SystemInfo):
        self.system_info = system_info
        self.requirements_file = "requirements.txt"
        self.installed_packages = {}

    def check_pip_available(): -> bool:
        """Check if pip is available."""
        try:
            result = subprocess.run()
                [self.system_info.get_pip_command(), "--version"],
                capture_output=True,
                text=True,
                check=True,
            )
            logger.info(f"Pip version: {result.stdout.strip()}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("Pip not available")
            return False

    def upgrade_pip(): -> bool:
        """Upgrade pip to latest version."""
        try:
            logger.info("Upgrading pip...")
            subprocess.run()
                [self.system_info.get_pip_command(), "install", "--upgrade", "pip"],
                check=True,
            )
            logger.info("✅ Pip upgraded successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to upgrade pip: {e}")
            return False

    def install_requirements(): -> bool:
        """Install requirements from requirements.txt."""
        try:
            logger.info("Installing requirements...")

            pip_cmd = []
                self.system_info.get_pip_command(),
                "install",
                "-r",
                self.requirements_file,
            ]

            if venv_path:
             # Use virtual environment pip
                if self.system_info.is_windows:
                    pip_cmd = []
                        os.path.join(venv_path, "Scripts", "pip"),
                        "install",
                        "-r",
                        self.requirements_file,
                    ]
                else:
                    pip_cmd = []
                        os.path.join(venv_path, "bin", "pip"),
                        "install",
                        "-r",
                        self.requirements_file,
                    ]

            subprocess.run(pip_cmd, check=True)
            logger.info("✅ Requirements installed successfully")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install requirements: {e}")
            return False

    def verify_installation(): -> Dict[str, bool]:
        """Verify that all required packages are installed."""
        required_packages = []
            "numpy",
            "pandas",
            "matplotlib",
            "scipy",
            "ccxt",
            "aiohttp",
            "requests",
            "tkinter",
            "asyncio",
        ]
        results = {}

        for package in required_packages:
            try:
                __import__(package)
                results[package] = True
                logger.info(f"✅ {package} - Available")
            except ImportError:
                results[package] = False
                logger.error(f"❌ {package} - Not available")

        return results


class VirtualEnvironmentManager:
    """Manages virtual environment creation and activation."""

    def __init__(self, system_info: SystemInfo):
        self.system_info = system_info
        self.venv_name = "schwabot_env"

    def create_venv(): -> Optional[str]:
        """Create virtual environment."""
        try:
            logger.info(f"Creating virtual environment: {self.venv_name}")

            venv_cmd = [self.system_info.get_venv_command(), self.venv_name]
            subprocess.run(venv_cmd, check=True)

            venv_path = os.path.abspath(self.venv_name)
            logger.info(f"✅ Virtual environment created: {venv_path}")
            return venv_path

        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to create virtual environment: {e}")
            return None

    def activate_venv(): -> bool:
        """Activate virtual environment."""
        try:
            if self.system_info.is_windows:
                activate_script = os.path.join(venv_path, "Scripts", "activate")
            else:
                activate_script = os.path.join(venv_path, "bin", "activate")

            if os.path.exists(activate_script):
                logger.info(f"Virtual environment activated: {venv_path}")
                return True
            else:
                logger.error(f"Activation script not found: {activate_script}")
                return False

        except Exception as e:
            logger.error(f"Failed to activate virtual environment: {e}")
            return False


class ConfigurationManager:
    """Manages system configuration and setup."""

    def __init__(self):
        self.config_dir = Path("config")
        self.logs_dir = Path("logs")
        self.data_dir = Path("data")

    def create_directories(): -> bool:
        """Create necessary directories."""
        try:
            directories = [self.config_dir, self.logs_dir, self.data_dir]

            for directory in directories:
                directory.mkdir(exist_ok=True)
                logger.info(f"✅ Created directory: {directory}")

            return True

        except Exception as e:
            logger.error(f"❌ Failed to create directories: {e}")
            return False

    def create_default_config(): -> bool:
        """Create default configuration files."""
        try:
            # Create .env template
            env_template = """# Schwabot Environment Configuration"
# Copy this file to .env and fill in your API keys

# Coinbase API Configuration
COINBASE_API_KEY=your_coinbase_api_key_here
COINBASE_API_SECRET=your_coinbase_api_secret_here
COINBASE_PASSPHRASE=your_coinbase_passphrase_here

# CoinMarketCap API Configuration
COINMARKETCAP_API_KEY=your_coinmarketcap_api_key_here

# Trading Configuration
TRADING_MODE=demo
SANDBOX_MODE=true
MAX_TRADE_AMOUNT=100.0
RISK_PER_TRADE=0.2

# System Configuration
LOG_LEVEL=INFO
ENABLE_VISUALIZATION=true
ENABLE_BACKTESTING=true
"""

            env_file = Path(".env.template")
            env_file.write_text(env_template)
            logger.info("✅ Created .env.template")

            # Create basic config
            basic_config = {}
                "system": {}
                    "name": "Schwabot Trading System",
                    "version": "1.0.0",
                    "mode": "demo",
                },
                "api": {}
                    "coinbase": {"enabled": True, "sandbox": True},
                    "coinmarketcap": {"enabled": True},
                    "coingecko": {"enabled": True},
                },
                "trading": {}
                    "enabled": False,
                    "pairs": ["BTC/USDC", "ETH/USDC", "XRP/USDC"],
                    "max_positions": 10,
                },
            }
            config_file = self.config_dir / "basic_config.json"
            config_file.write_text(json.dumps(basic_config, indent=2))
            logger.info("✅ Created basic configuration")

            return True

        except Exception as e:
            logger.error(f"❌ Failed to create configuration: {e}")
            return False

    def setup_logging(): -> bool:
        """Setup logging configuration."""
        try:
            log_config = {}
                "version": 1,
                "disable_existing_loggers": False,
                "formatters": {}
                    "standard": {}
                        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                    }
                },
                "handlers": {}
                    "file": {}
                        "class": "logging.FileHandler",
                        "filename": "logs/schwabot.log",
                        "formatter": "standard",
                        "level": "INFO",
                    },
                    "console": {}
                        "class": "logging.StreamHandler",
                        "formatter": "standard",
                        "level": "INFO",
                    },
                },
                "root": {"handlers": ["file", "console"], "level": "INFO"},
            }
            log_config_file = self.config_dir / "logging_config.json"
            log_config_file.write_text(json.dumps(log_config, indent=2))
            logger.info("✅ Created logging configuration")

            return True

        except Exception as e:
            logger.error(f"❌ Failed to setup logging: {e}")
            return False


class InstallationManager:
    """Main installation manager."""

    def __init__(self):
        self.system_info = SystemInfo()
        self.dependency_manager = DependencyManager(self.system_info)
        self.venv_manager = VirtualEnvironmentManager(self.system_info)
        self.config_manager = ConfigurationManager()

    def check_system_requirements(): -> bool:
        """Check if system meets requirements."""
        logger.info("🔍 Checking system requirements...")

        # Check Python version
        if not self.system_info.check_python_version():
            logger.error()
                f"❌ Python version {"}
                    self.system_info.get_python_version_str()
                } is not supported. "
                f"Python 3.8+ is required."
            )
            return False

        logger.info(f"✅ Python version: {self.system_info.get_python_version_str()}")

        # Check pip availability
        if not self.dependency_manager.check_pip_available():
            logger.error("❌ Pip is not available. Please install pip first.")
            return False

        logger.info("✅ Pip is available")

        # Check platform support
        logger.info(f"✅ Platform: {self.system_info.platform}")

        return True

    def install_dependencies(): -> bool:
        """Install system dependencies."""
        logger.info("📦 Installing dependencies...")

        venv_path = None

        if use_venv:
            # Create virtual environment
            venv_path = self.venv_manager.create_venv()
            if not venv_path:
                logger.error("❌ Failed to create virtual environment")
                return False

        # Upgrade pip
        if not self.dependency_manager.upgrade_pip():
            logger.warning("⚠️ Failed to upgrade pip, continuing...")

        # Install requirements
        if not self.dependency_manager.install_requirements(venv_path):
            logger.error("❌ Failed to install requirements")
            return False

        # Verify installation
        verification_results = self.dependency_manager.verify_installation()
        all_installed = all(verification_results.values())

        if all_installed:
            logger.info("✅ All dependencies installed successfully")
        else:
            failed_packages = []
                pkg for pkg, installed in verification_results.items() if not installed
            ]
            logger.error(f"❌ Failed to install packages: {failed_packages}")

        return all_installed

    def setup_configuration(): -> bool:
        """Setup system configuration."""
        logger.info("⚙️ Setting up configuration...")

        # Create directories
        if not self.config_manager.create_directories():
            return False

        # Create default config
        if not self.config_manager.create_default_config():
            return False

        # Setup logging
        if not self.config_manager.setup_logging():
            return False

        logger.info("✅ Configuration setup completed")
        return True

    def run_tests(): -> bool:
        """Run basic system tests."""
        logger.info("🧪 Running system tests...")

        try:
            # Test imports
            test_imports = []
                "numpy",
                "pandas",
                "matplotlib",
                "scipy",
                "tkinter",
                "asyncio",
                "logging",
            ]
            for module in test_imports:
                __import__(module)
                logger.info(f"✅ {module} import successful")

            # Test CCXT if available
            try:

                logger.info("✅ CCXT import successful")
            except ImportError:
                logger.warning("⚠️ CCXT not available (optional)")

            logger.info("✅ All tests passed")
            return True

        except Exception as e:
            logger.error(f"❌ Test failed: {e}")
            return False

    def create_launcher_scripts(): -> bool:
        """Create launcher scripts for different platforms."""
        try:
            logger.info("🚀 Creating launcher scripts...")

            if self.system_info.is_windows:
                # Windows batch file
                batch_content = """@echo off""
echo Starting Schwabot Trading System...
cd /d "%~dp0"
    if exist schwabot_env\\Scripts\\activate.bat ()
    call schwabot_env\\Scripts\\activate.bat
    python main.py
) else (
    python main.py
)
pause
"""
                with open("start_schwabot.bat", "w") as f:
                    f.write(batch_content)
                logger.info("✅ Created start_schwabot.bat")

            else:
                # Unix shell script
                shell_content = """#!/bin/bash"
echo "Starting Schwabot Trading System..."
cd "$(dirname "$0")"
    if [ -f "schwabot_env/bin/activate" ]; then
    source schwabot_env/bin/activate
fi
python3 main.py
"""
                with open("start_schwabot.sh", "w") as f:
                    f.write(shell_content)

                # Make executable
                os.chmod("start_schwabot.sh", 0o755)
                logger.info("✅ Created start_schwabot.sh")

            return True

        except Exception as e:
            logger.error(f"❌ Failed to create launcher scripts: {e}")
            return False

    def install(): -> bool:
        """Run complete installation."""
        logger.info("🚀 Starting Schwabot installation...")

        # Check system requirements
        if not self.check_system_requirements():
            return False

        # Install dependencies
        if not self.install_dependencies(use_venv=True):
            return False

        # Setup configuration
        if not self.setup_configuration():
            return False

        # Run tests
        if not self.run_tests():
            return False

        # Create launcher scripts
        if not self.create_launcher_scripts():
            return False

        logger.info("🎉 Installation completed successfully!")
        logger.info("")
        logger.info("Next steps:")
        logger.info("1. Copy .env.template to .env and fill in your API keys")
        logger.info("2. Run the system:")
        if self.system_info.is_windows:
            logger.info("   - Double-click start_schwabot.bat")
            logger.info("   - Or run: python main.py")
        else:
            logger.info("   - Run: ./start_schwabot.sh")
            logger.info("   - Or run: python3 main.py")
        logger.info("3. Open the GUI and configure your settings")

        return True


def main():
    """Main installation function."""
    parser = argparse.ArgumentParser(description="Schwabot Installation Script")
    parser.add_argument("--auto", action="store_true", help="Automatic installation")
    parser.add_argument()
        "--check", action="store_true", help="Check system requirements only"
    )
    parser.add_argument("--configure", action="store_true", help="Configure only")

    args = parser.parse_args()

    installer = InstallationManager()

    if args.check:
        # Check requirements only
        success = installer.check_system_requirements()
        sys.exit(0 if success else 1)

    elif args.configure:
        # Configure only
        success = installer.setup_configuration()
        sys.exit(0 if success else 1)

    else:
        # Full installation
        success = installer.install(auto_mode=args.auto)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
