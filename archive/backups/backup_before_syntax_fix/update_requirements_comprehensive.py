#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Requirements Updater for Schwabot Trading System

This script updates requirements.txt with all missing dependencies identified
in the error analysis, including standard library modules and third-party packages.

Usage:
    python update_requirements_comprehensive.py
"""

# Configure logging
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Set

# Fix Unicode encoding issues on Windows
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/requirements_update.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class RequirementsUpdater:
    """Comprehensive requirements updater for the Schwabot trading system."""

    def __init__(self):
        """Initialize the requirements updater."""
        self.requirements_file = Path('requirements.txt')
        self.backup_file = Path('requirements.txt.backup')
        
        # Missing dependencies identified from error analysis
        self.missing_dependencies = {
            # Standard library modules (these are built-in, but listed for completeness)
            'types': 'Built-in module for type objects',
            'abc': 'Built-in module for abstract base classes',
            'pkgutil': 'Built-in module for package utilities',
            'multiprocessing': 'Built-in module for multiprocessing',
            'contextlib': 'Built-in module for context managers',
            'functools': 'Built-in module for function tools',
            'collections': 'Built-in module for collection types',
            'io': 'Built-in module for I/O operations',
            
            # Third-party packages that need to be added
            'matplotlib': '>=3.5.2',  # For plt (matplotlib.pyplot)
            'numpy': '>=1.22.0',      # For np and la (numpy.linalg)
            'cupy': '>=10.0.0',       # For cp (GPU acceleration)
            
            # Additional packages that might be needed
            'pandas': '>=1.4.0',      # Already in requirements, but ensuring version
            'scipy': '>=1.8.0',       # Already in requirements, but ensuring version
            'scikit-learn': '>=1.1.0', # Already in requirements, but ensuring version
            'torch': '>=1.11.0',      # Already in requirements, but ensuring version
            'tensorflow': '>=2.9.0',  # Already in requirements, but ensuring version
            'ccxt': '>=2.0.0',        # Already in requirements, but ensuring version
            'ta-lib': '>=0.4.20',     # Already in requirements, but ensuring version
            'pandas-ta': '>=0.3.14b0', # Already in requirements, but ensuring version
            'asyncio': '>=3.4.3',     # Already in requirements, but ensuring version
            'apscheduler': '>=3.9.1', # Already in requirements, but ensuring version
            'dask': '>=2022.5.0',     # Already in requirements, but ensuring version
            'pyyaml': '>=6.0',        # Already in requirements, but ensuring version
            'python-dotenv': '>=0.20.0', # Already in requirements, but ensuring version
            'loguru': '>=0.6.0',      # Already in requirements, but ensuring version
            'black': '>=22.3.0',      # Already in requirements, but ensuring version
            'isort': '>=5.10.1',      # Already in requirements, but ensuring version
            'mypy': '>=0.950',        # Already in requirements, but ensuring version
            'flake8': '>=4.0.1',      # Already in requirements, but ensuring version
            'pylint': '>=2.13.5',     # Already in requirements, but ensuring version
            'pytest': '>=7.1.2',      # Already in requirements, but ensuring version
            'pytest-asyncio': '>=0.18.3', # Already in requirements, but ensuring version
            'coverage': '>=6.3.2',    # Already in requirements, but ensuring version
            'memory-profiler': '>=0.60.0', # Already in requirements, but ensuring version
            'line-profiler': '>=3.5.0', # Already in requirements, but ensuring version
            'aiohttp': '>=3.8.1',     # Already in requirements, but ensuring version
            'requests': '>=2.27.1',   # Already in requirements, but ensuring version
            'seaborn': '>=0.11.2',    # Already in requirements, but ensuring version
            'plotly': '>=5.8.0',      # Already in requirements, but ensuring version
            'nvidia-cuda-runtime-cu11': '>=11.7.0', # Already in requirements, but ensuring version
            'nvidia-cuda-nvcc-cu11': '>=11.7.0',    # Already in requirements, but ensuring version
            'qiskit': '>=0.36.0',     # Already in requirements, but ensuring version
            'pennylane': '>=0.26.0',  # Already in requirements, but ensuring version
            'cryptography': '>=37.0.2', # Already in requirements, but ensuring version
            'flask': '>=2.2.0',       # Already in requirements, but ensuring version
            'flask_socketio': '>=5.3.0', # Already in requirements, but ensuring version
            'flask_cors': '>=3.0.10', # Already in requirements, but ensuring version
            'fastapi': '>=0.110.0',   # Already in requirements, but ensuring version
            'uvicorn[standard]': '>=0.27.0', # Already in requirements, but ensuring version
            'starlette': '>=0.36.0',  # Already in requirements, but ensuring version
            'websockets': '>=10.4',   # Already in requirements, but ensuring version
            'psutil': '>=5.9.0',      # Already in requirements, but ensuring version
            'sqlalchemy': '>=1.4.0',  # Already in requirements, but ensuring version
            'redis': '>=4.0.0',       # Already in requirements, but ensuring version
            'pymongo': '>=4.0.0',     # Already in requirements, but ensuring version
            'dash': '>=2.12.0',       # Already in requirements, but ensuring version
            'streamlit': '>=1.34.0',  # Already in requirements, but ensuring version
            'bokeh': '>=3.2.0',       # Already in requirements, but ensuring version
        }
        
        # Platform-specific dependencies
        self.platform_dependencies = {
            'windows': {
                'pywin32': '>=228',  # Windows-specific utilities
                'wmi': '>=1.5.1',    # Windows Management Instrumentation
            },
            'linux': {
                'psutil': '>=5.9.0',  # Process and system utilities
            },
            'darwin': {  # macOS
                'psutil': '>=5.9.0',  # Process and system utilities
            }
        }

    def backup_requirements(self) -> None:
        """Create backup of current requirements.txt."""
        if self.requirements_file.exists():
            import shutil
            shutil.copy2(self.requirements_file, self.backup_file)
            logger.info(f"Backed up requirements.txt to {self.backup_file}")

    def parse_current_requirements(self) -> Dict[str, str]:
        """Parse current requirements.txt file."""
        current_requirements = {}
        
        if not self.requirements_file.exists():
            logger.warning("requirements.txt not found, creating new one")
            return current_requirements
        
        with open(self.requirements_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Parse package name and version
                    if '>=' in line:
                        package, version = line.split('>=', 1)
                        current_requirements[package.strip()] = f">={version.strip()}"
                    elif '==' in line:
                        package, version = line.split('==', 1)
                        current_requirements[package.strip()] = f"=={version.strip()}"
                    elif '~=' in line:
                        package, version = line.split('~=', 1)
                        current_requirements[package.strip()] = f"~={version.strip()}"
                    else:
                        current_requirements[line] = ""
        
        return current_requirements

    def get_platform(self) -> str:
        """Get current platform."""
        import platform
        system = platform.system().lower()
        if system == 'windows':
            return 'windows'
        elif system == 'linux':
            return 'linux'
        elif system == 'darwin':
            return 'darwin'
        else:
            return 'unknown'

    def update_requirements(self) -> None:
        """Update requirements.txt with missing dependencies."""
        logger.info("Updating requirements.txt")
        
        # Backup current requirements
        self.backup_requirements()
        
        # Parse current requirements
        current_requirements = self.parse_current_requirements()
        
        # Get platform-specific dependencies
        platform = self.get_platform()
        platform_deps = self.platform_dependencies.get(platform, {})
        
        # Merge all dependencies
        all_dependencies = {}
        all_dependencies.update(current_requirements)
        all_dependencies.update(self.missing_dependencies)
        all_dependencies.update(platform_deps)
        
        # Sort dependencies alphabetically
        sorted_dependencies = sorted(all_dependencies.items())
        
        # Write updated requirements.txt
        with open(self.requirements_file, 'w', encoding='utf-8') as f:
            f.write("# Core Python Dependencies\n")
            f.write("numpy>=1.22.0\n")
            f.write("scipy>=1.8.0\n")
            f.write("pandas>=1.4.0\n")
            f.write("numba>=0.55.0\n")
            f.write("cupy>=10.0.0  # Optional, for GPU acceleration\n\n")
            
            f.write("# Machine Learning and Scientific Computing\n")
            f.write("scikit-learn>=1.1.0\n")
            f.write("torch>=1.11.0\n")
            f.write("tensorflow>=2.9.0\n\n")
            
            f.write("# Trading and Financial Libraries\n")
            f.write("ccxt>=2.0.0\n")
            f.write("ta-lib>=0.4.20\n")
            f.write("pandas-ta>=0.3.14b0\n\n")
            
            f.write("# Async and Scheduling\n")
            f.write("asyncio>=3.4.3\n")
            f.write("apscheduler>=3.9.1\n")
            f.write("dask>=2022.5.0\n\n")
            
            f.write("# Configuration and Logging\n")
            f.write("pyyaml>=6.0\n")
            f.write("python-dotenv>=0.20.0\n")
            f.write("loguru>=0.6.0\n\n")
            
            f.write("# Code Quality and Type Checking\n")
            f.write("black>=22.3.0\n")
            f.write("isort>=5.10.1\n")
            f.write("mypy>=0.950\n")
            f.write("flake8>=4.0.1\n")
            f.write("pylint>=2.13.5\n\n")
            
            f.write("# Testing\n")
            f.write("pytest>=7.1.2\n")
            f.write("pytest-asyncio>=0.18.3\n")
            f.write("coverage>=6.3.2\n\n")
            
            f.write("# Performance and Profiling\n")
            f.write("memory-profiler>=0.60.0\n")
            f.write("line-profiler>=3.5.0\n\n")
            
            f.write("# Optional: Web and API\n")
            f.write("aiohttp>=3.8.1\n")
            f.write("requests>=2.27.1\n\n")
            
            f.write("# Optional: Visualization\n")
            f.write("matplotlib>=3.5.2\n")
            f.write("seaborn>=0.11.2\n")
            f.write("plotly>=5.8.0\n\n")
            
            f.write("# Optional: GPU and CUDA\n")
            f.write("nvidia-cuda-runtime-cu11>=11.7.0  # Adjust version as needed\n")
            f.write("nvidia-cuda-nvcc-cu11>=11.7.0     # Adjust version as needed\n\n")
            
            f.write("# Quantum and Advanced Computation\n")
            f.write("qiskit>=0.36.0\n")
            f.write("pennylane>=0.26.0\n\n")
            
            f.write("# Encryption and Security\n")
            f.write("cryptography>=37.0.2\n\n")
            
            f.write("# Web Framework and API\n")
            f.write("flask>=2.2.0\n")
            f.write("flask_socketio>=5.3.0\n")
            f.write("flask_cors>=3.0.10\n")
            f.write("fastapi>=0.110.0\n")
            f.write("uvicorn[standard]>=0.27.0\n")
            f.write("starlette>=0.36.0\n")
            f.write("websockets>=10.4\n\n")
            
            f.write("# System Utilities\n")
            f.write("psutil>=5.9.0\n\n")
            
            f.write("# Database and Storage (Optional)\n")
            f.write("sqlalchemy>=1.4.0\n")
            f.write("redis>=4.0.0\n")
            f.write("pymongo>=4.0.0\n\n")
            
            f.write("# Visualization Dashboards (Optional)\n")
            f.write("dash>=2.12.0\n")
            f.write("streamlit>=1.34.0\n")
            f.write("bokeh>=3.2.0\n\n")
            
            # Add platform-specific dependencies
            if platform_deps:
                f.write(f"# Platform-specific dependencies for {platform}\n")
                for package, version in sorted(platform_deps.items()):
                    f.write(f"{package}{version}\n")
                f.write("\n")
        
        logger.info(f"Updated requirements.txt with {len(all_dependencies)} dependencies")

    def create_platform_specific_requirements(self) -> None:
        """Create platform-specific requirements files."""
        platforms = ['windows', 'linux', 'darwin']
        
        for platform in platforms:
            platform_deps = self.platform_dependencies.get(platform, {})
            if platform_deps:
                filename = f"requirements-{platform}.txt"
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(f"# Requirements for {platform}\n")
                    f.write(f"# Install with: pip install -r {filename}\n\n")
                    
                    for package, version in sorted(platform_deps.items()):
                        f.write(f"{package}{version}\n")
                
                logger.info(f"Created {filename}")

    def create_setup_scripts(self) -> None:
        """Create platform-specific setup scripts."""
        # Windows setup script
        windows_setup = '''@echo off
echo Setting up Schwabot Trading System for Windows...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-windows.txt
echo Setup complete!
pause
'''
        
        with open('setup_windows.bat', 'w') as f:
            f.write(windows_setup)
        
        # Unix/Linux/macOS setup script
        unix_setup = '''#!/bin/bash
echo "Setting up Schwabot Trading System..."
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

# Platform-specific requirements
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    python3 -m pip install -r requirements-windows.txt
elif [[ "$OSTYPE" == "darwin"* ]]; then
    python3 -m pip install -r requirements-darwin.txt
else
    python3 -m pip install -r requirements-linux.txt
fi

echo "Setup complete!"
'''
        
        with open('setup_unix.sh', 'w') as f:
            f.write(unix_setup)
        
        # Make Unix script executable
        os.chmod('setup_unix.sh', 0o755)
        
        logger.info("Created setup scripts: setup_windows.bat, setup_unix.sh")

    def generate_report(self) -> None:
        """Generate a report of the requirements update."""
        report_path = Path('requirements_update_report.md')
        
        with open(report_path, 'w') as f:
            f.write("# Requirements Update Report\n\n")
            f.write("## Summary\n")
            f.write(f"- Original requirements backed up to: {self.backup_file}\n")
            f.write(f"- Updated requirements file: {self.requirements_file}\n")
            f.write(f"- Platform: {self.get_platform()}\n\n")
            
            f.write("## Added Dependencies\n")
            for package, version in sorted(self.missing_dependencies.items()):
                f.write(f"- {package}{version}\n")
            
            f.write(f"\n## Platform-specific Dependencies\n")
            platform = self.get_platform()
            platform_deps = self.platform_dependencies.get(platform, {})
            for package, version in sorted(platform_deps.items()):
                f.write(f"- {package}{version}\n")
            
            f.write("\n## Setup Instructions\n")
            f.write("1. Install Python 3.8 or higher\n")
            f.write("2. Create a virtual environment: `python -m venv schwabot_env`\n")
            f.write("3. Activate the virtual environment:\n")
            f.write("   - Windows: `schwabot_env\\Scripts\\activate`\n")
            f.write("   - Unix/Linux/macOS: `source schwabot_env/bin/activate`\n")
            f.write("4. Install dependencies:\n")
            f.write("   - Windows: `setup_windows.bat`\n")
            f.write("   - Unix/Linux/macOS: `./setup_unix.sh`\n")
            f.write("   - Or manually: `pip install -r requirements.txt`\n")
        
        logger.info(f"Report generated: {report_path}")


def main():
    """Main function to run the requirements updater."""
    logger.info("Starting comprehensive requirements update")
    
    # Create updater instance
    updater = RequirementsUpdater()
    
    # Update requirements
    updater.update_requirements()
    
    # Create platform-specific requirements
    updater.create_platform_specific_requirements()
    
    # Create setup scripts
    updater.create_setup_scripts()
    
    # Generate report
    updater.generate_report()
    
    logger.info("Requirements update completed")


if __name__ == "__main__":
    main() 