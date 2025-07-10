#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Schwabot Advanced Strategy System Installer (Windows, Compatible)
==============================================================

Windows-compatible installation script for the updated Schwabot trading system
with advanced mathematical frameworks, dualistic trading execution, and
quantum-inspired algorithms.

Features:
- Advanced Tensor Algebra
- Dualistic Trading Execution
- Quantum Mirror Layers
- Fractal Core Mathematics
- Strategy Bit Mapping
- Live Handler Integration
"""

import json
import logging
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Configure logging for Windows compatibility
logging.basicConfig()
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[]
        logging.FileHandler('schwabot_install.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class SchwabotInstaller:
    """Comprehensive installer for Schwabot Advanced Strategy System."""

    def __init__(self):
        self.project_root = Path.cwd()
        self.venv_name = "schwabot_env"
        self.venv_path = self.project_root / self.venv_name
        self.is_windows = platform.system().lower() == "windows"
        self.python_version = sys.version_info

    def check_system_requirements(self) -> bool:
        """Check if system meets requirements."""
        logger.info("Checking system requirements...")

        # Check Python version
        if self.python_version < (3, 8):
            logger.error(f"Python 3.8+ required, found {self.python_version}")
            return False
        logger.info(f"Python {self.python_version.major}.{self.python_version.minor}.{self.python_version.micro}")

        # Check pip
        try:
            result = subprocess.run([sys.executable, "-m", "pip", "--version"],)
                                  capture_output=True, text=True, check=True)
            logger.info(f"Pip available: {result.stdout.strip()}")
        except subprocess.CalledProcessError:
            logger.error("Pip not available")
            return False

        return True

    def create_virtual_environment(self) -> bool:
        """Create virtual environment for Schwabot."""
        logger.info("Creating virtual environment...")

        if self.venv_path.exists():
            logger.info("Virtual environment already exists, removing...")
            shutil.rmtree(self.venv_path)

        try:
            subprocess.run([sys.executable, "-m", "venv", str(self.venv_path)], check=True)
            logger.info(f"Virtual environment created: {self.venv_path}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create virtual environment: {e}")
            return False

    def get_venv_pip(self) -> str:
        """Get pip command for virtual environment."""
        if self.is_windows:
            return str(self.venv_path / "Scripts" / "pip.exe")
        else:
            return str(self.venv_path / "bin" / "pip")

    def get_venv_python(self) -> str:
        """Get python command for virtual environment."""
        if self.is_windows:
            return str(self.venv_path / "Scripts" / "python.exe")
        else:
            return str(self.venv_path / "bin" / "python")

    def upgrade_pip(self) -> bool:
        """Upgrade pip in virtual environment."""
        logger.info("Upgrading pip...")
        try:
            # Use python -m pip install --upgrade pip instead of direct pip command
            subprocess.run([self.get_venv_python(), "-m", "pip", "install", "--upgrade", "pip"], check=True)
            logger.info("Pip upgraded successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to upgrade pip: {e}")
            # Continue anyway as this is not critical
            logger.info("Continuing with existing pip version...")
            return True

    def install_core_dependencies(self) -> bool:
        """Install core dependencies."""
        logger.info("Installing core dependencies...")

        core_packages = []
            "numpy>=1.21.0",
            "pandas>=1.3.0", 
            "scipy>=1.7.0",
            "requests>=2.25.0",
            "aiohttp>=3.8.0",
            "websockets>=10.0",
            "ccxt>=2.0.0",
            "scikit-learn>=1.0.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "pyyaml>=6.0",
            "jsonschema>=4.0.0",
            "structlog>=21.5.0",
            "click>=8.0.0",
            "sympy>=1.10.0",
            "numba>=0.56.0",
            "Flask==2.3.3",
            "Flask-CORS==4.0.0",
            "python-dotenv==1.0.0",
            "gunicorn==21.2.0",
            "Werkzeug==2.3.7"
        ]

        try:
            for package in core_packages:
                logger.info(f"Installing {package}...")
                subprocess.run([self.get_venv_python(), "-m", "pip", "install", package], check=True)

            logger.info("Core dependencies installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install core dependencies: {e}")
            return False

    def install_development_dependencies(self) -> bool:
        """Install development dependencies."""
        logger.info("Installing development dependencies...")

        dev_packages = []
            "pytest>=6.2.0",
            "pytest-asyncio>=0.18.0", 
            "pytest-cov>=3.0.0",
            "flake8>=4.0.0",
            "black>=22.0.0",
            "mypy>=0.950",
            "isort>=5.10.0",
            "types-requests>=2.28.0",
            "types-PyYAML>=6.0.0"
        ]

        try:
            for package in dev_packages:
                logger.info(f"Installing {package}...")
                subprocess.run([self.get_venv_python(), "-m", "pip", "install", package], check=True)

            logger.info("Development dependencies installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install development dependencies: {e}")
            return False

    def create_directories(self) -> bool:
        """Create necessary directories."""
        logger.info("Creating directories...")

        directories = []
            "data",
            "logs", 
            "config",
            "cache",
            "backups",
            "reports"
        ]

        try:
            for directory in directories:
                dir_path = self.project_root / directory
                dir_path.mkdir(exist_ok=True)
                logger.info(f"Created directory: {directory}")
            return True
        except Exception as e:
            logger.error(f"Failed to create directories: {e}")
            return False

    def create_configuration_files(self) -> bool:
        """Create configuration files."""
        logger.info("Creating configuration files...")

        # Main configuration
        config = {}
            "system": {}
                "name": "Schwabot Advanced Strategy System",
                "version": "2.0.0",
                "environment": "production"
            },
            "trading": {}
                "exchange_name": "coinbase",
                "sandbox_mode": True,
                "symbols": ["BTC/USDC", "ETH/USDC", "SOL/USDC"],
                "portfolio_value": 10000.0,
                "demo_mode": True,
                "enable_learning": True,
                "enable_automation": True
            },
            "mathematical": {}
                "tensor_precision": 64,
                "fractal_depth": 8,
                "quantum_layers": 4,
                "entropy_threshold": 0.85,
                "spectral_norm_limit": 1.0
            },
            "api": {}
                "fear_greed_cache_duration": 300,
                "whale_alert_rate_limit": 60,
                "glassnode_api_version": "v1",
                "coingecko_rate_limit": 50
            },
            "logging": {}
                "level": "INFO",
                "file_rotation": "daily",
                "max_file_size": "10MB"
            }
        }

        try:
            config_file = self.project_root / "config" / "schwabot_config.json"
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            logger.info("Configuration file created")
            return True
        except Exception as e:
            logger.error(f"Failed to create configuration: {e}")
            return False

    def create_launcher_scripts(self) -> bool:
        """Create launcher scripts."""
        logger.info("Creating launcher scripts...")

        # Windows batch file
        if self.is_windows:
            batch_content = f"""@echo off""
echo Starting Schwabot Advanced Strategy System...
cd /d "{self.project_root}"
call "{self.venv_path}\\Scripts\\activate.bat"
python start_schwabot.py
pause
"""
            batch_file = self.project_root / "start_schwabot.bat"
            with open(batch_file, 'w', encoding='utf-8') as f:
                f.write(batch_content)
            logger.info("Windows launcher created: start_schwabot.bat")

        # Unix shell script
        shell_content = f"""#!/bin/bash"
echo "Starting Schwabot Advanced Strategy System..."
cd "{self.project_root}"
source "{self.venv_path}/bin/activate"
python start_schwabot.py
"""
        shell_file = self.project_root / "start_schwabot.sh"
        with open(shell_file, 'w', encoding='utf-8') as f:
            f.write(shell_content)

        # Make shell script executable on Unix systems
        if not self.is_windows:
            os.chmod(shell_file, 0o755)
        logger.info("Unix launcher created: start_schwabot.sh")

        return True

    def run_system_tests(self) -> bool:
        """Run basic system tests."""
        logger.info("Running system tests...")

        test_script = f"""
import sys
import os
sys.path.insert(0, '{self.project_root}')

# Test imports
    try:
    import numpy as np
    import pandas as pd
    import ccxt
    import requests
    import aiohttp
    import flask
    print("Core imports successful")
    except ImportError as e:
    print(f"Import error: {{e}}")
    sys.exit(1)

# Test core modules
    try:
    from core.type_defs import Vector64, Tensor64, QuantumState
    from core.advanced_tensor_algebra import AdvancedTensorAlgebra
    from core.advanced_dualistic_trading_execution_system import DualisticTradingExecutionSystem
    from core.strategy_bit_mapper import StrategyBitMapper
    print("Core modules imported successfully")
    except ImportError as e:
    print(f"Core module import error: {{e}}")
    sys.exit(1)

print("All tests passed!")
"""

        try:
            result = subprocess.run()
                [self.get_venv_python(), "-c", test_script],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )

            if result.returncode == 0:
                logger.info("System tests passed")
                return True
            else:
                logger.error(f"System tests failed: {result.stderr}")
                return False
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to run system tests: {e}")
            return False

    def create_activation_script(self) -> bool:
        """Create script to activate virtual environment."""
        logger.info("Creating activation script...")

        if self.is_windows:
            activate_content = f"""@echo off""
echo Activating Schwabot environment...
call "{self.venv_path}\\Scripts\\activate.bat"
echo Environment activated! You can now run: python start_schwabot.py
cmd /k
"""
            activate_file = self.project_root / "activate_schwabot.bat"
            with open(activate_file, 'w', encoding='utf-8') as f:
                f.write(activate_content)
            logger.info("Windows activation script created: activate_schwabot.bat")
        else:
            activate_content = f"""#!/bin/bash"
echo "Activating Schwabot environment..."
source "{self.venv_path}/bin/activate"
echo "Environment activated! You can now run: python start_schwabot.py"
bash
"""
            activate_file = self.project_root / "activate_schwabot.sh"
            with open(activate_file, 'w', encoding='utf-8') as f:
                f.write(activate_content)
            os.chmod(activate_file, 0o755)
            logger.info("Unix activation script created: activate_schwabot.sh")

        return True

    def install(self) -> bool:
        """Complete installation process."""
        logger.info("Starting Schwabot Advanced Strategy System Installation")
        logger.info("=" * 60)

        steps = []
            ("System Requirements Check", self.check_system_requirements),
            ("Virtual Environment Creation", self.create_virtual_environment),
            ("Pip Upgrade", self.upgrade_pip),
            ("Core Dependencies Installation", self.install_core_dependencies),
            ("Development Dependencies Installation", self.install_development_dependencies),
            ("Directory Creation", self.create_directories),
            ("Configuration Setup", self.create_configuration_files),
            ("Launcher Scripts Creation", self.create_launcher_scripts),
            ("Activation Script Creation", self.create_activation_script),
            ("System Tests", self.run_system_tests)
        ]

        for step_name, step_func in steps:
            logger.info(f"\n{step_name}...")
            if not step_func():
                logger.error(f"Installation failed at: {step_name}")
                return False
            logger.info(f"{step_name} completed")

        logger.info("\n" + "=" * 60)
        logger.info("Schwabot Advanced Strategy System Installation Complete!")
        logger.info("\nNext Steps:")
        logger.info("1. Activate the environment:")
        if self.is_windows:
            logger.info("   Run: activate_schwabot.bat")
        else:
            logger.info("   Run: source activate_schwabot.sh")
        logger.info("2. Start the system:")
        logger.info("   Run: python start_schwabot.py")
        logger.info("3. Access the dashboard:")
        logger.info("   Open: http://127.0.0.1:5000")
        logger.info("\nDocumentation:")
        logger.info("   - Read SCHWABOT_README.md for detailed information")
        logger.info("   - Check logs/ directory for system logs")
        logger.info("   - Review config/ directory for configuration options")

        return True

def main():
    """Main installation function."""
    installer = SchwabotInstaller()

    if len(sys.argv) > 1 and sys.argv[1] == "--check":
        # Just check system requirements
        if installer.check_system_requirements():
            print("System requirements met")
            sys.exit(0)
        else:
            print("System requirements not met")
            sys.exit(1)

    # Full installation
    success = installer.install()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 