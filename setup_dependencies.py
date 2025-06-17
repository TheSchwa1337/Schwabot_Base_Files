#!/usr/bin/env python3
"""
Schwabot Dependency Setup Script
===============================

Comprehensive dependency management for the Schwabot Anti-Pole trading system.
Handles GPU/CPU detection, package installation, and environment validation.

Usage:
    python setup_dependencies.py [--gpu-check] [--install] [--validate]
"""

import sys
import subprocess
import logging
import platform
import importlib
import warnings
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('dependency_setup.log')
    ]
)
logger = logging.getLogger(__name__)

class DependencyManager:
    """Manages installation and validation of Schwabot dependencies"""
    
    def __init__(self):
        self.python_version = sys.version_info
        self.platform = platform.system().lower()
        self.gpu_available = False
        self.cuda_version = None
        
        # Core dependencies that must be installed
        self.core_deps = [
            'numpy>=1.21.0',
            'scipy>=1.7.0', 
            'pandas>=1.3.0',
            'matplotlib>=3.4.0',
            'pyyaml>=5.4.0',
            'typing-extensions>=4.0.0',
            'python-dateutil>=2.8.0'
        ]
        
        # GPU dependencies (optional but recommended)
        self.gpu_deps = [
            'torch>=1.12.0',
            'torchvision>=0.13.0',
            'pynvml>=11.0.0',
            'GPUtil>=1.4.0'
        ]
        
        # Web/API dependencies
        self.web_deps = [
            'flask>=2.0.0',
            'flask-cors>=3.0.0',
            'aiohttp>=3.8.0',
            'websockets>=10.0',
            'requests>=2.28.0'
        ]
        
        # Trading/Analysis dependencies
        self.trading_deps = [
            'ccxt>=4.0.0',
            'scikit-learn>=0.24.0',
            'plotly>=5.0.0',
            'pywavelets>=1.3.0'
        ]
        
        # Development dependencies
        self.dev_deps = [
            'pytest>=6.2.0',
            'pytest-asyncio>=0.18.0',
            'black>=21.5b2',
            'mypy>=0.910'
        ]
        
        # Failed installations
        self.failed_installs = []
        self.optional_failed = []
        
    def check_python_version(self) -> bool:
        """Check if Python version is compatible"""
        logger.info(f"Checking Python version: {sys.version}")
        
        if self.python_version < (3, 8):
            logger.error("Python 3.8+ required for Schwabot. Please upgrade Python.")
            return False
        elif self.python_version >= (3, 12):
            logger.warning("Python 3.12+ detected. Some packages may have compatibility issues.")
        
        logger.info("✅ Python version compatible")
        return True
    
    def detect_gpu_capabilities(self) -> Dict[str, any]:
        """Detect GPU capabilities and CUDA version"""
        gpu_info = {
            'gpu_available': False,
            'cuda_version': None,
            'gpu_count': 0,
            'gpu_names': [],
            'recommendations': []
        }
        
        try:
            # Try to detect NVIDIA GPUs
            import pynvml
            pynvml.nvmlInit()
            gpu_count = pynvml.nvmlDeviceGetCount()
            
            if gpu_count > 0:
                gpu_info['gpu_available'] = True
                gpu_info['gpu_count'] = gpu_count
                
                for i in range(gpu_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                    gpu_info['gpu_names'].append(name)
                
                logger.info(f"✅ Detected {gpu_count} NVIDIA GPU(s): {gpu_info['gpu_names']}")
                
        except ImportError:
            logger.warning("pynvml not available, cannot detect NVIDIA GPUs")
        except Exception as e:
            logger.warning(f"GPU detection failed: {e}")
        
        # Try to detect CUDA version
        try:
            result = subprocess.run(['nvcc', '--version'], 
                                  capture_output=True, text=True, check=True)
            if 'release' in result.stdout:
                # Extract CUDA version
                for line in result.stdout.split('\n'):
                    if 'release' in line:
                        version = line.split('release')[1].split(',')[0].strip()
                        gpu_info['cuda_version'] = version
                        logger.info(f"✅ CUDA version detected: {version}")
                        break
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("CUDA compiler (nvcc) not found in PATH")
        
        # Provide recommendations
        if gpu_info['gpu_available']:
            if gpu_info['cuda_version']:
                if gpu_info['cuda_version'].startswith('11'):
                    gpu_info['recommendations'].append('cupy-cuda11x>=11.0.0')
                elif gpu_info['cuda_version'].startswith('12'):
                    gpu_info['recommendations'].append('cupy-cuda12x>=12.0.0')
                else:
                    gpu_info['recommendations'].append('cupy-cuda11x>=11.0.0')  # Default fallback
            else:
                gpu_info['recommendations'].append('cupy-cuda11x>=11.0.0')  # Default
        else:
            logger.info("No NVIDIA GPU detected. System will run in CPU-only mode.")
        
        self.gpu_available = gpu_info['gpu_available']
        self.cuda_version = gpu_info['cuda_version']
        
        return gpu_info
    
    def install_package(self, package: str, optional: bool = False) -> bool:
        """Install a single package with error handling"""
        logger.info(f"Installing {package}...")
        
        try:
            subprocess.run([
                sys.executable, '-m', 'pip', 'install', package, '--upgrade'
            ], check=True, capture_output=True, text=True)
            
            logger.info(f"✅ Successfully installed {package}")
            return True
            
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr if e.stderr else str(e)
            logger.error(f"❌ Failed to install {package}: {error_msg}")
            
            if optional:
                self.optional_failed.append(package)
            else:
                self.failed_installs.append(package)
            
            return False
    
    def install_dependencies(self, include_gpu: bool = True, include_dev: bool = False) -> bool:
        """Install all required dependencies"""
        logger.info("🚀 Starting Schwabot dependency installation...")
        
        all_deps = self.core_deps + self.web_deps + self.trading_deps
        
        if include_gpu and self.gpu_available:
            all_deps.extend(self.gpu_deps)
            
            # Add CuPy based on CUDA version
            gpu_info = self.detect_gpu_capabilities()
            if gpu_info['recommendations']:
                all_deps.extend(gpu_info['recommendations'])
        
        if include_dev:
            all_deps.extend(self.dev_deps)
        
        # Install core dependencies first
        logger.info("Installing core mathematical dependencies...")
        for dep in self.core_deps:
            self.install_package(dep, optional=False)
        
        # Install web/API dependencies
        logger.info("Installing web and API dependencies...")
        for dep in self.web_deps:
            self.install_package(dep, optional=False)
        
        # Install trading dependencies
        logger.info("Installing trading and analysis dependencies...")
        for dep in self.trading_deps:
            if 'ta-lib' in dep:
                # TA-Lib requires special handling
                self.install_talib()
            else:
                self.install_package(dep, optional=True)
        
        # Install GPU dependencies if available
        if include_gpu and self.gpu_available:
            logger.info("Installing GPU acceleration dependencies...")
            for dep in self.gpu_deps:
                self.install_package(dep, optional=True)
            
            # Install CuPy
            gpu_info = self.detect_gpu_capabilities()
            for cupy_dep in gpu_info['recommendations']:
                self.install_package(cupy_dep, optional=True)
        
        # Install development dependencies
        if include_dev:
            logger.info("Installing development dependencies...")
            for dep in self.dev_deps:
                self.install_package(dep, optional=True)
        
        return len(self.failed_installs) == 0
    
    def install_talib(self) -> bool:
        """Special handling for TA-Lib installation"""
        logger.info("Installing TA-Lib (may require system dependencies)...")
        
        if self.platform == 'windows':
            # Windows requires pre-compiled wheel
            try:
                subprocess.run([
                    sys.executable, '-m', 'pip', 'install', 
                    'https://download.lfd.uci.edu/pythonlibs/archived/TA_Lib-0.4.24-cp39-cp39-win_amd64.whl'
                ], check=True, capture_output=True)
                logger.info("✅ TA-Lib installed from pre-compiled wheel")
                return True
            except:
                logger.warning("❌ TA-Lib installation failed. Install manually if needed.")
                self.optional_failed.append('ta-lib')
                return False
        
        else:
            # Linux/macOS can try pip installation
            return self.install_package('ta-lib>=0.4.0', optional=True)
    
    def validate_installation(self) -> Dict[str, bool]:
        """Validate that all critical components can be imported"""
        logger.info("🔍 Validating installation...")
        
        validation_results = {}
        
        # Test core mathematical libraries
        core_imports = {
            'numpy': 'import numpy as np; np.array([1,2,3])',
            'scipy': 'import scipy; import scipy.stats',
            'pandas': 'import pandas as pd; pd.DataFrame({"test": [1,2,3]})',
            'matplotlib': 'import matplotlib.pyplot as plt',
            'yaml': 'import yaml'
        }
        
        # Test GPU libraries if available
        if self.gpu_available:
            gpu_imports = {
                'torch': 'import torch; torch.tensor([1,2,3])',
                'cupy': 'import cupy as cp; cp.array([1,2,3])',
                'pynvml': 'import pynvml; pynvml.nvmlInit()'
            }
            core_imports.update(gpu_imports)
        
        # Test web/API libraries
        web_imports = {
            'flask': 'import flask',
            'aiohttp': 'import aiohttp',
            'requests': 'import requests',
            'ccxt': 'import ccxt'
        }
        core_imports.update(web_imports)
        
        for lib_name, test_code in core_imports.items():
            try:
                exec(test_code)
                validation_results[lib_name] = True
                logger.info(f"✅ {lib_name} - OK")
            except Exception as e:
                validation_results[lib_name] = False
                logger.error(f"❌ {lib_name} - FAILED: {e}")
        
        return validation_results
    
    def test_core_functionality(self) -> bool:
        """Test core Schwabot functionality"""
        logger.info("🧪 Testing core Schwabot functionality...")
        
        try:
            # Test Anti-Pole mathematical core
            import numpy as np
            from core.antipole.vector import AntiPoleVector, AntiPoleConfig
            
            config = AntiPoleConfig()
            vector = AntiPoleVector(config)
            
            # Test with sample data
            state = vector.process_tick(btc_price=45000.0, volume=1000000.0)
            
            logger.info(f"✅ Anti-Pole core test successful: ICAP={state.icap_probability:.3f}")
            
            # Test Hash Affinity Vault
            from core.hash_affinity_vault import HashAffinityVault
            vault = HashAffinityVault()
            
            vault.log_tick(
                tick_id="test_001",
                signal_strength=0.75,
                backend="CPU_STANDARD",
                matrix_id="test_matrix",
                btc_price=45000.0,
                volume=1000000.0
            )
            
            logger.info("✅ Hash Affinity Vault test successful")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Core functionality test failed: {e}")
            return False
    
    def generate_report(self) -> Dict[str, any]:
        """Generate comprehensive installation report"""
        gpu_info = self.detect_gpu_capabilities()
        validation_results = self.validate_installation()
        
        report = {
            'timestamp': str(Path(__file__).stat().st_mtime),
            'system_info': {
                'platform': self.platform,
                'python_version': f"{self.python_version.major}.{self.python_version.minor}.{self.python_version.micro}",
                'python_executable': sys.executable
            },
            'gpu_info': gpu_info,
            'installation_status': {
                'core_dependencies': len(self.failed_installs) == 0,
                'failed_packages': self.failed_installs,
                'optional_failed': self.optional_failed
            },
            'validation_results': validation_results,
            'recommendations': []
        }
        
        # Add recommendations
        if self.failed_installs:
            report['recommendations'].append(
                f"Critical packages failed: {', '.join(self.failed_installs)}. "
                "Please install manually or check system dependencies."
            )
        
        if not gpu_info['gpu_available']:
            report['recommendations'].append(
                "No GPU detected. System will run in CPU-only mode. "
                "For better performance, consider installing NVIDIA drivers and CUDA."
            )
        
        if self.optional_failed:
            report['recommendations'].append(
                f"Optional packages failed: {', '.join(self.optional_failed)}. "
                "These are not critical but may limit some functionality."
            )
        
        return report
    
    def save_report(self, report: Dict[str, any], filename: str = 'dependency_report.json'):
        """Save installation report to file"""
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"📄 Report saved to {filename}")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Schwabot Dependency Setup')
    parser.add_argument('--gpu-check', action='store_true', 
                       help='Check GPU capabilities only')
    parser.add_argument('--install', action='store_true',
                       help='Install all dependencies')
    parser.add_argument('--validate', action='store_true',
                       help='Validate existing installation')
    parser.add_argument('--include-dev', action='store_true',
                       help='Include development dependencies')
    parser.add_argument('--no-gpu', action='store_true',
                       help='Skip GPU dependencies')
    
    args = parser.parse_args()
    
    manager = DependencyManager()
    
    # Check Python version first
    if not manager.check_python_version():
        sys.exit(1)
    
    if args.gpu_check:
        gpu_info = manager.detect_gpu_capabilities()
        print("\n🖥️  GPU Information:")
        print(f"GPU Available: {gpu_info['gpu_available']}")
        print(f"GPU Count: {gpu_info['gpu_count']}")
        print(f"GPU Names: {gpu_info['gpu_names']}")
        print(f"CUDA Version: {gpu_info['cuda_version']}")
        print(f"Recommendations: {gpu_info['recommendations']}")
        return
    
    if args.install:
        success = manager.install_dependencies(
            include_gpu=not args.no_gpu,
            include_dev=args.include_dev
        )
        
        if success:
            logger.info("🎉 All critical dependencies installed successfully!")
        else:
            logger.error("❌ Some critical dependencies failed to install")
            sys.exit(1)
    
    if args.validate:
        validation_results = manager.validate_installation()
        failed_validations = [k for k, v in validation_results.items() if not v]
        
        if failed_validations:
            logger.error(f"❌ Validation failed for: {', '.join(failed_validations)}")
            sys.exit(1)
        else:
            logger.info("✅ All validations passed!")
            
            # Test core functionality
            if manager.test_core_functionality():
                logger.info("🎯 Schwabot core functionality validated!")
            else:
                logger.error("❌ Core functionality test failed")
                sys.exit(1)
    
    # Generate and save report
    report = manager.generate_report()
    manager.save_report(report)
    
    print("\n📊 Installation Summary:")
    print(f"Platform: {report['system_info']['platform']}")
    print(f"Python: {report['system_info']['python_version']}")
    print(f"GPU Available: {report['gpu_info']['gpu_available']}")
    print(f"Core Dependencies: {'✅ OK' if report['installation_status']['core_dependencies'] else '❌ FAILED'}")
    
    if report['recommendations']:
        print("\n💡 Recommendations:")
        for rec in report['recommendations']:
            print(f"  • {rec}")

if __name__ == '__main__':
    main() 