#!/usr/bin/env python3
"""
🛠️ Unified Schwabot Dependency Installer
=========================================

This script handles the installation of all Schwabot dependencies,
including problematic packages like TA-Lib, with graceful fallbacks
and clear reporting of what was successfully installed.

Usage:
    python install_dependencies.py [--basic] [--gpu] [--talib] [--all]
    
Options:
    --basic    Install only essential dependencies (default)
    --gpu      Include GPU acceleration packages (torch, cupy)
    --talib    Attempt to install TA-Lib (may fail)
    --all      Install everything including optional packages
"""

import subprocess
import sys
import platform
import importlib
import os
import argparse
from typing import List, Dict, Tuple, Optional
import time

class DependencyInstaller:
    """Comprehensive dependency installer with error handling"""
    
    def __init__(self):
        self.platform = platform.system().lower()
        self.python_version = sys.version_info
        self.successful_installs = []
        self.failed_installs = []
        self.optional_installs = []
        
        # Core dependencies that must work
        self.core_deps = [
            "numpy>=1.24.0",
            "pandas>=2.0.0", 
            "scipy>=1.10.0",
            "matplotlib>=3.7.0",
            "requests>=2.31.0",
            "psutil>=5.9.0",
            "pyyaml>=6.0",
            "python-dotenv>=1.0.0",
            "colorama>=0.4.6"
        ]
        
        # Essential for Schwabot functionality
        self.essential_deps = [
            "websockets>=11.0.0",
            "flask>=2.3.0",
            "flask-cors>=4.0.0",
            "aiohttp>=3.8.0",
            "scikit-learn>=1.3.0",
            "seaborn>=0.12.0",
            "ccxt>=4.0.0",
            "python-binance>=1.0.17",
            "sqlalchemy>=2.0.0",
            "jsonschema>=4.21.0",
            "loguru>=0.7.0",
            "rich>=13.5.0",
            "cryptography>=41.0.0"
        ]
        
        # Optional but recommended
        self.optional_deps = [
            "redis>=4.6.0",
            "celery>=5.3.0",
            "plotly>=5.15.0",
            "dash>=2.13.0",
            "streamlit>=1.25.0",
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.7.0",
            "flake8>=6.0.0"
        ]
        
        # GPU acceleration packages
        self.gpu_deps = [
            "torch>=2.0.0",
            "cupy-cuda12x>=12.0.0",  # Adjust based on CUDA version
            "GPUtil>=1.4.0"
        ]
        
        # Problematic packages requiring special handling
        self.problematic_deps = {
            "TA-Lib>=0.4.28": self._install_talib,
            "coinbase>=2.1.0": self._install_coinbase,
            "dearpygui>=1.10.0": self._install_dearpygui,
            "mypy>=1.5.0": self._install_mypy
        }
    
    def _run_pip_install(self, package: str, optional: bool = False) -> bool:
        """Run pip install for a single package"""
        try:
            print(f"  📦 Installing {package}...")
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", package
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print(f"  ✅ {package} - SUCCESS")
                self.successful_installs.append(package)
                return True
            else:
                print(f"  ❌ {package} - FAILED")
                if optional:
                    self.optional_installs.append(package)
                else:
                    self.failed_installs.append(package)
                print(f"     Error: {result.stderr[:200]}...")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"  ⏰ {package} - TIMEOUT")
            self.failed_installs.append(package)
            return False
        except Exception as e:
            print(f"  💥 {package} - EXCEPTION: {e}")
            self.failed_installs.append(package)
            return False
    
    def _install_talib(self) -> bool:
        """Special handling for TA-Lib installation"""
        print("  🔧 Attempting TA-Lib installation...")
        
        # First, try to install from pip (might work if system deps are installed)
        if self._run_pip_install("TA-Lib>=0.4.28", optional=True):
            return True
        
        # Try alternative packages
        alternatives = [
            "pandas-ta>=0.3.14",
            "ta>=0.10.0",
            "talib-binary>=0.4.19"
        ]
        
        for alt in alternatives:
            print(f"  🔄 Trying alternative: {alt}")
            if self._run_pip_install(alt, optional=True):
                print(f"  ✅ Installed TA-Lib alternative: {alt}")
                return True
        
        print("  ⚠️ TA-Lib installation failed. See TALIB_INSTALLATION.md for manual installation.")
        return False
    
    def _install_coinbase(self) -> bool:
        """Special handling for coinbase package"""
        print("  🔧 Installing coinbase package...")
        return self._run_pip_install("coinbase>=2.1.0", optional=True)
    
    def _install_dearpygui(self) -> bool:
        """Special handling for DearPyGUI"""
        print("  🔧 Installing DearPyGUI...")
        if self.platform == "linux":
            # DearPyGUI might have issues on some Linux distributions
            return self._run_pip_install("dearpygui>=1.10.0", optional=True)
        else:
            return self._run_pip_install("dearpygui>=1.10.0", optional=True)
    
    def _install_mypy(self) -> bool:
        """Special handling for mypy"""
        print("  🔧 Installing mypy...")
        return self._run_pip_install("mypy>=1.5.0", optional=True)
    
    def install_core_dependencies(self) -> int:
        """Install core dependencies that must work"""
        print("🔧 Installing Core Dependencies...")
        print("=" * 50)
        
        success_count = 0
        for dep in self.core_deps:
            if self._run_pip_install(dep, optional=False):
                success_count += 1
        
        print(f"\n✅ Core Dependencies: {success_count}/{len(self.core_deps)} installed")
        return success_count
    
    def install_essential_dependencies(self) -> int:
        """Install essential Schwabot dependencies"""
        print("\n🎯 Installing Essential Schwabot Dependencies...")
        print("=" * 50)
        
        success_count = 0
        for dep in self.essential_deps:
            if self._run_pip_install(dep, optional=False):
                success_count += 1
        
        print(f"\n✅ Essential Dependencies: {success_count}/{len(self.essential_deps)} installed")
        return success_count
    
    def install_optional_dependencies(self) -> int:
        """Install optional dependencies"""
        print("\n📦 Installing Optional Dependencies...")
        print("=" * 50)
        
        success_count = 0
        for dep in self.optional_deps:
            if self._run_pip_install(dep, optional=True):
                success_count += 1
        
        print(f"\n✅ Optional Dependencies: {success_count}/{len(self.optional_deps)} installed")
        return success_count
    
    def install_gpu_dependencies(self) -> int:
        """Install GPU acceleration dependencies"""
        print("\n🚀 Installing GPU Dependencies...")
        print("=" * 50)
        
        success_count = 0
        for dep in self.gpu_deps:
            if self._run_pip_install(dep, optional=True):
                success_count += 1
        
        print(f"\n✅ GPU Dependencies: {success_count}/{len(self.gpu_deps)} installed")
        return success_count
    
    def install_problematic_dependencies(self) -> int:
        """Install packages that require special handling"""
        print("\n⚠️ Installing Problematic Dependencies...")
        print("=" * 50)
        
        success_count = 0
        for package, installer_func in self.problematic_deps.items():
            try:
                if installer_func():
                    success_count += 1
            except Exception as e:
                print(f"  💥 {package} failed with exception: {e}")
        
        print(f"\n✅ Problematic Dependencies: {success_count}/{len(self.problematic_deps)} installed")
        return success_count
    
    def verify_installation(self) -> Dict[str, bool]:
        """Verify that critical packages can be imported"""
        print("\n🔍 Verifying Installation...")
        print("=" * 30)
        
        critical_modules = {
            "numpy": "numpy",
            "pandas": "pandas", 
            "scipy": "scipy",
            "matplotlib": "matplotlib",
            "requests": "requests",
            "websockets": "websockets",
            "flask": "flask",
            "aiohttp": "aiohttp",
            "psutil": "psutil",
            "yaml": "yaml"
        }
        
        results = {}
        for name, module in critical_modules.items():
            try:
                importlib.import_module(module)
                print(f"  ✅ {name}")
                results[name] = True
            except ImportError:
                print(f"  ❌ {name}")
                results[name] = False
        
        return results
    
    def generate_report(self) -> str:
        """Generate installation report"""
        report = [
            "\n" + "=" * 60,
            "🎯 UNIFIED SCHWABOT DEPENDENCY INSTALLATION REPORT",
            "=" * 60,
            f"Platform: {platform.system()} {platform.release()}",
            f"Python: {sys.version}",
            f"Installation Date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            f"✅ Successfully Installed: {len(self.successful_installs)} packages",
            f"❌ Failed Installations: {len(self.failed_installs)} packages", 
            f"⚠️ Optional Failures: {len(self.optional_installs)} packages",
            ""
        ]
        
        if self.successful_installs:
            report.append("📦 SUCCESSFUL INSTALLATIONS:")
            for pkg in self.successful_installs:
                report.append(f"  ✅ {pkg}")
            report.append("")
        
        if self.failed_installs:
            report.append("❌ FAILED INSTALLATIONS:")
            for pkg in self.failed_installs:
                report.append(f"  ❌ {pkg}")
            report.append("")
        
        if self.optional_installs:
            report.append("⚠️ OPTIONAL FAILURES (Non-critical):")
            for pkg in self.optional_installs:
                report.append(f"  ⚠️ {pkg}")
            report.append("")
        
        report.extend([
            "🚀 NEXT STEPS:",
            "1. Run the unified system: python launch_unified_schwabot.py demo",
            "2. For TA-Lib issues: see TALIB_INSTALLATION.md",
            "3. For GPU support: ensure CUDA is installed",
            "4. For manual installation: pip install <package-name>",
            "",
            "The Unified Schwabot system will work with graceful fallbacks",
            "for any missing optional dependencies.",
            "=" * 60
        ])
        
        return "\n".join(report)
    
    def install_all(self, include_gpu: bool = False, include_problematic: bool = False) -> bool:
        """Install all dependencies with specified options"""
        print("🌟 UNIFIED SCHWABOT DEPENDENCY INSTALLER")
        print("=" * 50)
        print(f"Platform: {platform.system()}")
        print(f"Python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
        print("=" * 50)
        
        # Update pip first
        print("📦 Updating pip...")
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                      capture_output=True)
        
        # Install dependencies in phases
        core_success = self.install_core_dependencies()
        essential_success = self.install_essential_dependencies() 
        optional_success = self.install_optional_dependencies()
        
        gpu_success = 0
        if include_gpu:
            gpu_success = self.install_gpu_dependencies()
        
        problematic_success = 0
        if include_problematic:
            problematic_success = self.install_problematic_dependencies()
        
        # Verify installation
        verification_results = self.verify_installation()
        
        # Generate and display report
        report = self.generate_report()
        print(report)
        
        # Determine overall success
        critical_success_rate = core_success / len(self.core_deps)
        essential_success_rate = essential_success / len(self.essential_deps)
        
        # Success if we have at least 80% of core and 70% of essential
        overall_success = (critical_success_rate >= 0.8 and essential_success_rate >= 0.7)
        
        if overall_success:
            print("🎉 Installation completed successfully!")
            print("You can now run: python launch_unified_schwabot.py demo")
        else:
            print("⚠️ Installation completed with some issues.")
            print("The system may still work with reduced functionality.")
        
        return overall_success

def main():
    """Main installer function"""
    parser = argparse.ArgumentParser(description="Unified Schwabot Dependency Installer")
    parser.add_argument("--basic", action="store_true", 
                       help="Install only core and essential dependencies")
    parser.add_argument("--gpu", action="store_true",
                       help="Include GPU acceleration packages")
    parser.add_argument("--talib", action="store_true", 
                       help="Attempt to install TA-Lib and other problematic packages")
    parser.add_argument("--all", action="store_true",
                       help="Install everything including optional and problematic packages")
    
    args = parser.parse_args()
    
    installer = DependencyInstaller()
    
    # Determine what to install
    include_gpu = args.gpu or args.all
    include_problematic = args.talib or args.all
    
    if not any([args.basic, args.gpu, args.talib, args.all]):
        # Default: basic installation
        print("No options specified, running basic installation...")
        include_gpu = False
        include_problematic = False
    
    # Run installation
    success = installer.install_all(
        include_gpu=include_gpu,
        include_problematic=include_problematic
    )
    
    # Save report to file
    with open("dependency_installation_report.txt", "w", encoding='utf-8') as f:
        f.write(installer.generate_report())
    
    print(f"\n📊 Installation report saved to: dependency_installation_report.txt")
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 