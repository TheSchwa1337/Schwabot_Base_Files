#!/usr/bin/env python3
"""
Mathematical Dependencies Installer
==================================

Comprehensive installer script that ensures all mathematical dependencies
are properly installed for the integrated profit correlation system.

This script handles:
- Core mathematical libraries (numpy, scipy, sympy)
- GPU acceleration libraries (torch, cupy) with fallbacks
- System monitoring libraries (psutil, GPUtil)
- Trading APIs and mathematical analysis tools
- Development and testing dependencies

Features:
- Automatic CUDA detection and CuPy version selection
- Graceful fallbacks for missing GPU support
- Dependency validation and version checking
- Installation progress reporting
- Error handling and troubleshooting guidance
"""

import subprocess
import sys
import os
import platform
import logging
import importlib.util
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DependencyInstaller:
    """Comprehensive dependency installer for mathematical computing"""
    
    def __init__(self):
        self.system_info = self._get_system_info()
        self.installation_log = []
        self.failed_packages = []
        self.optional_packages = []
        
    def _get_system_info(self) -> Dict:
        """Get system information for dependency selection"""
        return {
            'platform': platform.system(),
            'architecture': platform.architecture()[0],
            'python_version': sys.version_info,
            'cuda_available': self._check_cuda_availability(),
            'pip_version': self._get_pip_version()
        }
    
    def _check_cuda_availability(self) -> bool:
        """Check if CUDA is available on the system"""
        try:
            # Try nvidia-smi command
            result = subprocess.run(['nvidia-smi'], 
                                   capture_output=True, 
                                   text=True, 
                                   timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def _get_pip_version(self) -> str:
        """Get pip version"""
        try:
            result = subprocess.run([sys.executable, '-m', 'pip', '--version'],
                                   capture_output=True, text=True)
            return result.stdout.strip()
        except:
            return "unknown"
    
    def _install_package(self, package: str, optional: bool = False) -> bool:
        """Install a single package with error handling"""
        try:
            logger.info(f"Installing {package}...")
            
            cmd = [sys.executable, '-m', 'pip', 'install', package]
            
            # Add upgrade flag for core packages
            if any(core in package.lower() for core in ['numpy', 'scipy', 'torch']):
                cmd.append('--upgrade')
            
            result = subprocess.run(cmd, 
                                   capture_output=True, 
                                   text=True,
                                   timeout=300)  # 5 minute timeout
            
            if result.returncode == 0:
                logger.info(f"✅ Successfully installed {package}")
                self.installation_log.append(f"✅ {package}")
                return True
            else:
                error_msg = result.stderr.strip()
                logger.error(f"❌ Failed to install {package}: {error_msg}")
                
                if optional:
                    self.optional_packages.append(package)
                    logger.warning(f"⚠️  {package} is optional, continuing...")
                else:
                    self.failed_packages.append(package)
                
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"❌ Timeout installing {package}")
            if not optional:
                self.failed_packages.append(package)
            return False
        except Exception as e:
            logger.error(f"❌ Error installing {package}: {e}")
            if not optional:
                self.failed_packages.append(package)
            return False
    
    def _validate_installation(self, package_name: str, import_name: str = None) -> bool:
        """Validate that a package was installed correctly"""
        if import_name is None:
            import_name = package_name
        
        try:
            spec = importlib.util.find_spec(import_name)
            if spec is not None:
                logger.info(f"✓ {package_name} validation passed")
                return True
            else:
                logger.error(f"✗ {package_name} validation failed")
                return False
        except Exception as e:
            logger.error(f"✗ {package_name} validation error: {e}")
            return False
    
    def install_core_mathematical_libraries(self) -> bool:
        """Install core mathematical libraries"""
        logger.info("Installing Core Mathematical Libraries")
        logger.info("=" * 50)
        
        core_packages = [
            "numpy>=1.24.0",
            "scipy>=1.10.0",
            "sympy>=1.12.0",
            "pandas>=2.0.0",
            "scikit-learn>=1.3.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0"
        ]
        
        success_count = 0
        for package in core_packages:
            if self._install_package(package):
                success_count += 1
        
        # Validate core imports
        validations = [
            ("numpy", "numpy"),
            ("scipy", "scipy"),
            ("sympy", "sympy"),
            ("pandas", "pandas"),
            ("sklearn", "sklearn"),
            ("matplotlib", "matplotlib"),
            ("seaborn", "seaborn")
        ]
        
        validation_count = 0
        for package, import_name in validations:
            if self._validate_installation(package, import_name):
                validation_count += 1
        
        logger.info(f"Core libraries: {success_count}/{len(core_packages)} installed, "
                   f"{validation_count}/{len(validations)} validated")
        
        return success_count >= len(core_packages) * 0.8  # 80% success rate
    
    def install_gpu_acceleration_libraries(self) -> bool:
        """Install GPU acceleration libraries with CUDA detection"""
        logger.info("Installing GPU Acceleration Libraries")
        logger.info("=" * 50)
        
        if not self.system_info['cuda_available']:
            logger.warning("CUDA not detected. Installing CPU-only versions...")
            
            # Install CPU-only PyTorch
            success = self._install_package("torch>=2.0.0+cpu", optional=True)
            if success:
                logger.info("PyTorch (CPU-only) installed")
            
            logger.info("Skipping CuPy installation (requires CUDA)")
            return True
        
        # CUDA is available, install GPU versions
        logger.info("CUDA detected. Installing GPU-accelerated versions...")
        
        # Install PyTorch with CUDA support
        torch_success = self._install_package("torch>=2.0.0", optional=True)
        
        # Install CuPy - try to detect CUDA version
        cupy_package = self._get_cupy_package()
        cupy_success = self._install_package(cupy_package, optional=True)
        
        # Install additional GPU utilities
        gputil_success = self._install_package("GPUtil>=1.4.0", optional=True)
        
        # Validate GPU libraries
        gpu_validations = []
        if torch_success:
            gpu_validations.append(("torch", "torch"))
        if cupy_success:
            gpu_validations.append(("cupy", "cupy"))
        if gputil_success:
            gpu_validations.append(("GPUtil", "GPUtil"))
        
        validation_count = 0
        for package, import_name in gpu_validations:
            if self._validate_installation(package, import_name):
                validation_count += 1
        
        logger.info(f"GPU libraries: {validation_count}/{len(gpu_validations)} validated")
        
        return True  # GPU libraries are optional
    
    def _get_cupy_package(self) -> str:
        """Get appropriate CuPy package based on CUDA version"""
        try:
            # Try to detect CUDA version
            result = subprocess.run(['nvcc', '--version'], 
                                   capture_output=True, text=True)
            
            if result.returncode == 0:
                output = result.stdout
                if '12.' in output:
                    return "cupy-cuda12x>=12.0.0"
                elif '11.' in output:
                    return "cupy-cuda11x>=11.0.0"
                else:
                    logger.warning("Unknown CUDA version, using CUDA 12.x package")
                    return "cupy-cuda12x>=12.0.0"
            else:
                logger.warning("Could not detect CUDA version, using CUDA 12.x package")
                return "cupy-cuda12x>=12.0.0"
                
        except FileNotFoundError:
            logger.warning("nvcc not found, using default CuPy package")
            return "cupy-cuda12x>=12.0.0"
    
    def install_system_monitoring_libraries(self) -> bool:
        """Install system monitoring and performance libraries"""
        logger.info("Installing System Monitoring Libraries")
        logger.info("=" * 50)
        
        monitoring_packages = [
            "psutil>=5.9.0",
            "py-cpuinfo>=9.0.0"
        ]
        
        # GPUtil is already handled in GPU section
        success_count = 0
        for package in monitoring_packages:
            if self._install_package(package):
                success_count += 1
        
        # Validate monitoring imports
        validations = [
            ("psutil", "psutil"),
            ("cpuinfo", "cpuinfo")
        ]
        
        validation_count = 0
        for package, import_name in validations:
            if self._validate_installation(package, import_name):
                validation_count += 1
        
        logger.info(f"Monitoring libraries: {success_count}/{len(monitoring_packages)} installed, "
                   f"{validation_count}/{len(validations)} validated")
        
        return success_count >= len(monitoring_packages) * 0.5  # 50% success rate
    
    def install_trading_and_api_libraries(self) -> bool:
        """Install trading APIs and financial libraries"""
        logger.info("Installing Trading and API Libraries")
        logger.info("=" * 50)
        
        trading_packages = [
            "ccxt>=4.0.0",
            "python-binance>=1.0.17",
            "coinbase>=2.1.0",
            "requests>=2.31.0",
            "websockets>=11.0.0",
            "aiohttp>=3.8.0"
        ]
        
        # TA-Lib might need special handling
        talib_packages = ["TA-Lib>=0.4.28"]
        
        success_count = 0
        
        # Install main trading packages
        for package in trading_packages:
            if self._install_package(package):
                success_count += 1
        
        # Try to install TA-Lib (can be problematic)
        for package in talib_packages:
            if self._install_package(package, optional=True):
                success_count += 1
            else:
                logger.warning("⚠️  TA-Lib installation failed. You may need to install it manually.")
                logger.info("   See: https://github.com/mrjbq7/ta-lib#installation")
        
        logger.info(f"Trading libraries: {success_count}/{len(trading_packages)} core packages installed")
        
        return success_count >= len(trading_packages) * 0.8  # 80% success rate
    
    def install_web_and_ui_libraries(self) -> bool:
        """Install web framework and UI libraries"""
        logger.info("Installing Web and UI Libraries")
        logger.info("=" * 50)
        
        web_packages = [
            "flask>=2.3.0",
            "flask-cors>=4.0.0",
            "flask-socketio>=5.3.0",
            "dearpygui>=1.10.0",
            "plotly>=5.15.0",
            "dash>=2.13.0",
            "streamlit>=1.25.0"
        ]
        
        success_count = 0
        for package in web_packages:
            # UI packages can be optional in some environments
            optional = package.startswith(('dearpygui', 'streamlit'))
            if self._install_package(package, optional=optional):
                success_count += 1
        
        logger.info(f"Web/UI libraries: {success_count}/{len(web_packages)} installed")
        
        return success_count >= len(web_packages) * 0.6  # 60% success rate
    
    def install_development_and_testing_libraries(self) -> bool:
        """Install development and testing libraries"""
        logger.info("Installing Development and Testing Libraries")
        logger.info("=" * 50)
        
        dev_packages = [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-mock>=3.11.0",
            "pyyaml>=6.0",
            "python-dotenv>=1.0.0",
            "loguru>=0.7.0",
            "rich>=13.5.0",
            "colorama>=0.4.6",
            "jsonschema>=4.21.0"
        ]
        
        success_count = 0
        for package in dev_packages:
            if self._install_package(package, optional=True):
                success_count += 1
        
        logger.info(f"Development libraries: {success_count}/{len(dev_packages)} installed")
        
        return True  # Development libraries are optional
    
    def run_comprehensive_validation(self) -> Dict[str, bool]:
        """Run comprehensive validation of all installed packages"""
        logger.info("Running Comprehensive Validation")
        logger.info("=" * 50)
        
        validations = {
            'core_math': [
                ('numpy', 'numpy'),
                ('scipy', 'scipy'),
                ('sympy', 'sympy'),
                ('pandas', 'pandas'),
                ('matplotlib', 'matplotlib')
            ],
            'gpu_acceleration': [
                ('torch', 'torch'),
                ('cupy', 'cupy'),
                ('GPUtil', 'GPUtil')
            ],
            'system_monitoring': [
                ('psutil', 'psutil'),
                ('cpuinfo', 'cpuinfo')
            ],
            'trading_apis': [
                ('ccxt', 'ccxt'),
                ('requests', 'requests'),
                ('websockets', 'websockets')
            ],
            'development': [
                ('pytest', 'pytest'),
                ('yaml', 'yaml'),
                ('rich', 'rich')
            ]
        }
        
        results = {}
        
        for category, packages in validations.items():
            category_results = []
            logger.info(f"Validating {category}...")
            
            for package, import_name in packages:
                success = self._validate_installation(package, import_name)
                category_results.append(success)
                
                if success:
                    logger.info(f"  ✅ {package}")
                else:
                    logger.warning(f"  ❌ {package}")
            
            results[category] = all(category_results) if category_results else False
            success_rate = sum(category_results) / len(category_results) if category_results else 0
            logger.info(f"  {category}: {success_rate:.1%} success rate")
        
        return results
    
    def generate_installation_report(self, validation_results: Dict[str, bool]) -> str:
        """Generate comprehensive installation report"""
        
        report = [
            "Mathematical Dependencies Installation Report",
            "=" * 60,
            "",
            f"System Information:",
            f"  Platform: {self.system_info['platform']} {self.system_info['architecture']}",
            f"  Python: {'.'.join(map(str, self.system_info['python_version'][:3]))}",
            f"  CUDA Available: {self.system_info['cuda_available']}",
            f"  Pip: {self.system_info['pip_version']}",
            "",
            "Installation Results:",
        ]
        
        for category, success in validation_results.items():
            status = "✅ PASS" if success else "⚠️  PARTIAL"
            report.append(f"  {category.replace('_', ' ').title()}: {status}")
        
        if self.failed_packages:
            report.extend([
                "",
                "❌ Failed Packages:",
                *[f"  - {pkg}" for pkg in self.failed_packages]
            ])
        
        if self.optional_packages:
            report.extend([
                "",
                "⚠️  Optional Packages (install manually if needed):",
                *[f"  - {pkg}" for pkg in self.optional_packages]
            ])
        
        report.extend([
            "",
            "📋 Installation Log:",
            *[f"  {entry}" for entry in self.installation_log[-10:]]  # Last 10 entries
        ])
        
        if self.failed_packages:
            report.extend([
                "",
                "🛠️  Troubleshooting:",
                "  1. Ensure you have the latest pip: python -m pip install --upgrade pip",
                "  2. For CUDA issues, install CUDA toolkit manually",
                "  3. For TA-Lib issues, see: https://github.com/mrjbq7/ta-lib#installation",
                "  4. Run with admin/sudo privileges if permission errors occur"
            ])
        
        return "\n".join(report)
    
    def install_all_dependencies(self) -> bool:
        """Install all dependencies with comprehensive reporting"""
        logger.info("🚀 Starting Comprehensive Dependency Installation")
        logger.info("=" * 60)
        
        # Installation phases
        phases = [
            ("Core Mathematical Libraries", self.install_core_mathematical_libraries),
            ("GPU Acceleration Libraries", self.install_gpu_acceleration_libraries),
            ("System Monitoring Libraries", self.install_system_monitoring_libraries),
            ("Trading and API Libraries", self.install_trading_and_api_libraries),
            ("Web and UI Libraries", self.install_web_and_ui_libraries),
            ("Development and Testing Libraries", self.install_development_and_testing_libraries)
        ]
        
        phase_results = []
        
        for phase_name, phase_func in phases:
            logger.info(f"\n📦 {phase_name}")
            logger.info("-" * 40)
            
            try:
                result = phase_func()
                phase_results.append(result)
                
                if result:
                    logger.info(f"✅ {phase_name} completed successfully")
                else:
                    logger.warning(f"⚠️  {phase_name} completed with issues")
                    
            except Exception as e:
                logger.error(f"❌ {phase_name} failed: {e}")
                phase_results.append(False)
        
        # Final validation
        logger.info(f"\n🔍 Final Validation")
        logger.info("-" * 40)
        validation_results = self.run_comprehensive_validation()
        
        # Generate report
        report = self.generate_installation_report(validation_results)
        
        # Save report to file
        report_path = Path("installation_report.txt")
        with open(report_path, 'w') as f:
            f.write(report)
        
        # Display final status
        logger.info(f"\n{report}")
        logger.info(f"\n📄 Full report saved to: {report_path}")
        
        # Determine overall success
        core_success = validation_results.get('core_math', False)
        monitoring_success = validation_results.get('system_monitoring', False)
        
        overall_success = core_success and monitoring_success
        
        if overall_success:
            logger.info("\n🎉 Installation completed successfully!")
            logger.info("✅ All critical dependencies are available")
            logger.info("🚀 System ready for mathematical operations")
        else:
            logger.warning("\n⚠️  Installation completed with issues")
            logger.warning("❌ Some critical dependencies may be missing")
            logger.warning("🛠️  Review the report and install missing packages manually")
        
        return overall_success

def main():
    """Main installation function"""
    
    print("Mathematical Dependencies Installer")
    print("=====================================")
    print()
    print("This script will install all dependencies required for the")
    print("Integrated Profit Correlation System mathematical operations.")
    print()
    
    # Confirm installation
    try:
        response = input("Continue with installation? [Y/n]: ").strip().lower()
        if response and response != 'y':
            print("Installation cancelled.")
            return False
    except KeyboardInterrupt:
        print("\nInstallation cancelled.")
        return False
    
    # Run installation
    installer = DependencyInstaller()
    success = installer.install_all_dependencies()
    
    return success

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🛑 Installation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Installation failed with error: {e}")
        sys.exit(1) 