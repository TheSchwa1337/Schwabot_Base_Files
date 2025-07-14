#!/usr/bin/env python3
"""
Schwabot Dependencies Installation Script
=========================================
Comprehensive installation script for Schwabot mathematical system dependencies.
Validates system requirements, installs dependencies, and verifies installation.
"""

import importlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Color codes for output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_status(message: str, status: str = "INFO", color: str = Colors.BLUE):
    """Print formatted status message."""
    timestamp = f"[{__import__('datetime').datetime.now().strftime('%H:%M:%S')}]"
    print(f"{color}{timestamp} [{status}] {message}{Colors.END}")

def print_success(message: str):
    """Print success message."""
    print_status(message, "SUCCESS", Colors.GREEN)

def print_warning(message: str):
    """Print warning message."""
    print_status(message, "WARNING", Colors.YELLOW)

def print_error(message: str):
    """Print error message."""
    print_status(message, "ERROR", Colors.RED)

def print_header(message: str):
    """Print header message."""
    print(f"\n{Colors.BOLD}{Colors.PURPLE}{'='*60}")
    print(f"  {message}")
    print(f"{'='*60}{Colors.END}\n")

class SchwabotInstaller:
    """Comprehensive installer for Schwabot dependencies."""
    
    def __init__(self):
        self.system_info = self._get_system_info()
        self.requirements_file = "requirements_validated.txt"
        self.installation_log = []
        
    def _get_system_info(self) -> Dict[str, str]:
        """Get system information."""
        return {
            'platform': platform.system(),
            'platform_version': platform.version(),
            'architecture': platform.machine(),
            'python_version': platform.python_version(),
            'python_implementation': platform.python_implementation(),
        }
    
    def check_python_version(self) -> bool:
        """Check if Python version is compatible."""
        print_header("Checking Python Version")
        
        version = sys.version_info
        min_version = (3, 8)
        recommended_version = (3, 10)
        
        print(f"Current Python version: {version.major}.{version.minor}.{version.micro}")
        
        if version < min_version:
            print_error(f"Python {min_version[0]}.{min_version[1]} or higher is required")
            return False
        elif version < recommended_version:
            print_warning(f"Python {recommended_version[0]}.{recommended_version[1]} or higher is recommended")
        else:
            print_success(f"Python version {version.major}.{version.minor} is compatible")
        
        return True
    
    def check_pip(self) -> bool:
        """Check if pip is available and up to date."""
        print_header("Checking pip Installation")
        
        try:
            result = subprocess.run([sys.executable, '-m', 'pip', '--version'], 
                                  capture_output=True, text=True, check=True)
            print_success(f"pip is available: {result.stdout.strip()}")
            
            # Check if pip needs upgrade
            result = subprocess.run([sys.executable, '-m', 'pip', 'list', '--outdated'], 
                                  capture_output=True, text=True)
            if 'pip' in result.stdout:
                print_warning("pip is outdated, upgrading...")
                subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'], 
                             check=True)
                print_success("pip upgraded successfully")
            
            return True
        except subprocess.CalledProcessError as e:
            print_error(f"pip check failed: {e}")
            return False
    
    def install_core_dependencies(self) -> bool:
        """Install core mathematical dependencies."""
        print_header("Installing Core Mathematical Dependencies")
        
        core_packages = [
            'numpy>=1.22.0,<2.0.0',
            'scipy>=1.8.0,<2.0.0', 
            'pandas>=1.4.0,<3.0.0',
            'numba>=0.55.0,<1.0.0'
        ]
        
        return self._install_packages(core_packages, "Core mathematical packages")
    
    def install_ml_dependencies(self) -> bool:
        """Install machine learning dependencies."""
        print_header("Installing Machine Learning Dependencies")
        
        ml_packages = [
            'torch>=1.11.0,<3.0.0',
            'tensorflow>=2.9.0,<3.0.0',
            'scikit-learn>=1.1.0,<2.0.0'
        ]
        
        return self._install_packages(ml_packages, "Machine learning packages")
    
    def install_quantum_dependencies(self) -> bool:
        """Install quantum computing dependencies."""
        print_header("Installing Quantum Computing Dependencies")
        
        quantum_packages = [
            'qiskit>=0.36.0,<1.0.0',
            'pennylane>=0.26.0,<1.0.0'
        ]
        
        return self._install_packages(quantum_packages, "Quantum computing packages")
    
    def install_trading_dependencies(self) -> bool:
        """Install trading and API dependencies."""
        print_header("Installing Trading Dependencies")
        
        trading_packages = [
            'ccxt>=2.0.0,<5.0.0',
            'requests>=2.27.1,<3.0.0',
            'aiohttp>=3.8.1,<4.0.0',
            'websockets>=10.4,<11.0.0',
            'ta>=0.10.2,<1.0.0'
        ]
        
        return self._install_packages(trading_packages, "Trading and API packages")
    
    def install_web_dependencies(self) -> bool:
        """Install web framework dependencies."""
        print_header("Installing Web Framework Dependencies")
        
        web_packages = [
            'flask>=2.2.0,<3.0.0',
            'flask-socketio>=5.3.0,<6.0.0',
            'flask-cors>=3.0.10,<4.0.0',
            'fastapi>=0.110.0,<1.0.0',
            'uvicorn[standard]>=0.27.0,<1.0.0',
            'starlette>=0.36.0,<1.0.0'
        ]
        
        return self._install_packages(web_packages, "Web framework packages")
    
    def install_config_dependencies(self) -> bool:
        """Install configuration and logging dependencies."""
        print_header("Installing Configuration Dependencies")
        
        config_packages = [
            'pyyaml>=6.0,<7.0',
            'python-dotenv>=0.20.0,<1.0.0',
            'toml>=0.10.2,<1.0.0',
            'loguru>=0.6.0,<1.0.0',
            'rich>=13.0.0,<14.0.0',
            'colorama>=0.4.6,<1.0.0'
        ]
        
        return self._install_packages(config_packages, "Configuration and logging packages")
    
    def install_security_dependencies(self) -> bool:
        """Install security and cryptography dependencies."""
        print_header("Installing Security Dependencies")
        
        security_packages = [
            'cryptography>=37.0.2,<42.0.0',
            'pyjwt>=2.8.0,<3.0.0',
            'passlib>=1.7.4,<2.0.0'
        ]
        
        return self._install_packages(security_packages, "Security and cryptography packages")
    
    def install_system_dependencies(self) -> bool:
        """Install system monitoring dependencies."""
        print_header("Installing System Dependencies")
        
        system_packages = [
            'psutil>=5.9.0,<6.0.0',
            'py-cpuinfo>=9.0.0,<10.0.0',
            'memory-profiler>=0.60.0,<1.0.0',
            'line-profiler>=3.5.0,<4.0.0'
        ]
        
        # Add platform-specific dependencies
        if self.system_info['platform'] == 'Windows':
            system_packages.extend([
                'pywin32>=228,<306',
                'wmi>=1.5.1,<2.0.0'
            ])
        
        return self._install_packages(system_packages, "System monitoring packages")
    
    def install_database_dependencies(self) -> bool:
        """Install database dependencies."""
        print_header("Installing Database Dependencies")
        
        db_packages = [
            'sqlalchemy>=1.4.0,<2.0.0',
            'redis>=4.0.0,<5.0.0',
            'pymongo>=4.0.0,<5.0.0'
        ]
        
        return self._install_packages(db_packages, "Database packages")
    
    def install_visualization_dependencies(self) -> bool:
        """Install visualization dependencies."""
        print_header("Installing Visualization Dependencies")
        
        viz_packages = [
            'matplotlib>=3.5.2,<4.0.0',
            'seaborn>=0.11.2,<1.0.0',
            'plotly>=5.8.0,<6.0.0',
            'dash>=2.12.0,<3.0.0',
            'streamlit>=1.34.0,<2.0.0',
            'bokeh>=3.2.0,<4.0.0'
        ]
        
        return self._install_packages(viz_packages, "Visualization packages")
    
    def install_development_dependencies(self) -> bool:
        """Install development and testing dependencies."""
        print_header("Installing Development Dependencies")
        
        dev_packages = [
            'black>=22.3.0,<24.0.0',
            'isort>=5.10.1,<6.0.0',
            'mypy>=0.950,<2.0.0',
            'flake8>=4.0.1,<7.0.0',
            'pylint>=2.13.5,<3.0.0',
            'pytest>=7.1.2,<8.0.0',
            'pytest-asyncio>=0.18.3,<1.0.0',
            'coverage>=6.3.2,<8.0.0',
            'pytest-benchmark>=4.0.0,<5.0.0'
        ]
        
        return self._install_packages(dev_packages, "Development and testing packages")
    
    def install_async_dependencies(self) -> bool:
        """Install async and scheduling dependencies."""
        print_header("Installing Async Dependencies")
        
        async_packages = [
            'apscheduler>=3.9.1,<4.0.0',
            'dask>=2022.5.0,<2024.0.0'
        ]
        
        return self._install_packages(async_packages, "Async and scheduling packages")
    
    def _install_packages(self, packages: List[str], description: str) -> bool:
        """Install a list of packages."""
        print(f"Installing {description}...")
        
        for package in packages:
            try:
                print(f"  Installing {package}...")
                result = subprocess.run([
                    sys.executable, '-m', 'pip', 'install', package
                ], capture_output=True, text=True, check=True)
                
                self.installation_log.append({
                    'package': package,
                    'status': 'success',
                    'output': result.stdout
                })
                print_success(f"  ✅ {package} installed successfully")
                
            except subprocess.CalledProcessError as e:
                self.installation_log.append({
                    'package': package,
                    'status': 'error',
                    'output': e.stderr
                })
                print_error(f"  ❌ Failed to install {package}: {e.stderr}")
                return False
        
        return True
    
    def verify_installation(self) -> Dict[str, bool]:
        """Verify that all critical packages are installed correctly."""
        print_header("Verifying Installation")
        
        critical_packages = [
            'numpy', 'scipy', 'pandas', 'torch', 'tensorflow', 'ccxt',
            'flask', 'pyyaml', 'cryptography', 'psutil'
        ]
        
        verification_results = {}
        
        for package in critical_packages:
            try:
                module = importlib.import_module(package)
                version = getattr(module, '__version__', 'unknown')
                print_success(f"✅ {package} v{version} - OK")
                verification_results[package] = True
            except ImportError:
                print_error(f"❌ {package} - NOT FOUND")
                verification_results[package] = False
        
        return verification_results
    
    def test_mathematical_functions(self) -> bool:
        """Test core mathematical functions."""
        print_header("Testing Mathematical Functions")
        
        try:
            import numpy as np
            import pandas as pd
            import scipy

            # Test basic operations
            arr = np.array([1, 2, 3, 4, 5])
            mean = np.mean(arr)
            std = np.std(arr)
            
            print_success(f"NumPy test: mean={mean}, std={std}")
            
            # Test pandas
            df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
            print_success(f"Pandas test: DataFrame shape {df.shape}")
            
            # Test scipy
            from scipy import stats
            correlation = stats.pearsonr([1, 2, 3], [4, 5, 6])[0]
            print_success(f"SciPy test: correlation={correlation}")
            
            return True
            
        except Exception as e:
            print_error(f"Mathematical function test failed: {e}")
            return False
    
    def generate_installation_report(self) -> None:
        """Generate installation report."""
        print_header("Installation Report")
        
        report = {
            'system_info': self.system_info,
            'installation_log': self.installation_log,
            'timestamp': __import__('datetime').datetime.now().isoformat()
        }
        
        # Save report
        with open('installation_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print_success("Installation report saved to installation_report.json")
        
        # Print summary
        successful_installations = len([log for log in self.installation_log if log['status'] == 'success'])
        failed_installations = len([log for log in self.installation_log if log['status'] == 'error'])
        
        print(f"\n{Colors.BOLD}Installation Summary:{Colors.END}")
        print(f"  ✅ Successful: {successful_installations}")
        print(f"  ❌ Failed: {failed_installations}")
        print(f"  📊 Total: {len(self.installation_log)}")
    
    def run_full_installation(self) -> bool:
        """Run the complete installation process."""
        print_header("Schwabot Dependencies Installation")
        print(f"System: {self.system_info['platform']} {self.system_info['platform_version']}")
        print(f"Python: {self.system_info['python_version']}")
        print(f"Architecture: {self.system_info['architecture']}")
        
        # Check prerequisites
        if not self.check_python_version():
            return False
        
        if not self.check_pip():
            return False
        
        # Install dependencies in order
        installation_steps = [
            ("Core Mathematical", self.install_core_dependencies),
            ("Machine Learning", self.install_ml_dependencies),
            ("Quantum Computing", self.install_quantum_dependencies),
            ("Trading APIs", self.install_trading_dependencies),
            ("Web Frameworks", self.install_web_dependencies),
            ("Configuration", self.install_config_dependencies),
            ("Security", self.install_security_dependencies),
            ("System Monitoring", self.install_system_dependencies),
            ("Database", self.install_database_dependencies),
            ("Visualization", self.install_visualization_dependencies),
            ("Development Tools", self.install_development_dependencies),
            ("Async Support", self.install_async_dependencies),
        ]
        
        for step_name, step_function in installation_steps:
            if not step_function():
                print_error(f"Failed to install {step_name} dependencies")
                return False
        
        # Verify installation
        verification_results = self.verify_installation()
        if not all(verification_results.values()):
            print_warning("Some packages failed verification")
        
        # Test mathematical functions
        if not self.test_mathematical_functions():
            print_warning("Mathematical function tests failed")
        
        # Generate report
        self.generate_installation_report()
        
        print_header("Installation Complete")
        print_success("Schwabot dependencies installation completed!")
        print_success("Your mathematical system is ready for operation.")
        
        return True

def main():
    """Main installation function."""
    installer = SchwabotInstaller()
    
    try:
        success = installer.run_full_installation()
        if success:
            print_success("🎉 Installation completed successfully!")
            sys.exit(0)
        else:
            print_error("❌ Installation failed!")
            sys.exit(1)
    except KeyboardInterrupt:
        print_error("\nInstallation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 