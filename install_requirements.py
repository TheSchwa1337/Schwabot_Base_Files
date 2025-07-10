#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smart Requirements Installer for Schwabot Trading System
=======================================================

This script intelligently installs dependencies with proper fallback handling
for GPU/CUDA dependencies that may not be available on all systems.

Key Features:
- Automatic detection of system capabilities
- Graceful handling of GPU dependencies
- Fallback to CPU-only installations
- Comprehensive error reporting
- Production-ready dependency management
"""

import os
import platform
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

def run_command(command: List[str], description: str) -> Tuple[bool, str]:
    """Run a command and return success status and output."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True
        )
        print(f"✅ {description} completed successfully")
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        return False, e.stderr

def detect_system_capabilities() -> Dict[str, bool]:
    """Detect system capabilities for GPU support."""
    capabilities = {
        "windows": platform.system() == "Windows",
        "linux": platform.system() == "Linux",
        "macos": platform.system() == "Darwin",
        "cuda_environment": False,
        "gpu_available": False
    }
    
    # Check for CUDA environment variables
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if cuda_home:
        capabilities["cuda_environment"] = True
        print(f"🔍 CUDA environment detected: {cuda_home}")
    
    # Check for NVIDIA GPU (basic detection)
    try:
        if capabilities["windows"]:
            # Windows GPU detection
            result = subprocess.run(
                ["wmic", "path", "win32_VideoController", "get", "name"],
                capture_output=True,
                text=True
            )
            if "NVIDIA" in result.stdout:
                capabilities["gpu_available"] = True
                print("🔍 NVIDIA GPU detected on Windows")
        elif capabilities["linux"]:
            # Linux GPU detection
            result = subprocess.run(
                ["nvidia-smi"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                capabilities["gpu_available"] = True
                print("🔍 NVIDIA GPU detected on Linux")
    except Exception as e:
        print(f"⚠️  GPU detection failed: {e}")
    
    return capabilities

def install_core_dependencies() -> bool:
    """Install core dependencies that are always required."""
    core_packages = [
        "numpy>=1.22.0",
        "scipy>=1.8.0", 
        "pandas>=1.4.0",
        "numba>=0.55.0",
        "scikit-learn>=1.1.0",
        "ccxt>=2.0.0",
        "pyyaml>=6.0",
        "python-dotenv>=0.20.0",
        "loguru>=0.6.0",
        "psutil>=5.9.0",
        "aiohttp>=3.8.1",
        "requests>=2.27.1",
        "matplotlib>=3.5.2",
        "seaborn>=0.11.2",
        "plotly>=5.8.0",
        "flask>=2.2.0",
        "fastapi>=0.110.0",
        "uvicorn[standard]>=0.27.0",
        "cryptography>=37.0.2",
        "pytest>=7.1.2",
        "pytest-asyncio>=0.18.3",
        "black>=22.3.0",
        "isort>=5.10.1",
        "flake8>=4.0.1"
    ]
    
    success, output = run_command(
        [sys.executable, "-m", "pip", "install"] + core_packages,
        "Installing core dependencies"
    )
    
    if not success:
        print("❌ Core dependencies installation failed")
        print(f"Error: {output}")
        return False
    
    return True

def install_gpu_dependencies(capabilities: Dict[str, bool]) -> bool:
    """Install GPU dependencies with fallback handling."""
    print("\n🔍 GPU Dependencies Installation")
    print("=" * 50)
    
    if not capabilities["gpu_available"]:
        print("⚠️  No NVIDIA GPU detected, skipping GPU dependencies")
        print("✅ System will use CPU fallback for all operations")
        return True
    
    if not capabilities["cuda_environment"]:
        print("⚠️  CUDA environment not detected")
        print("💡 To enable GPU acceleration, install CUDA Toolkit")
        print("✅ System will use CPU fallback for all operations")
        return True
    
    # Try to install CuPy with fallback
    print("🚀 Attempting to install CuPy for GPU acceleration...")
    
    # First try pre-built CuPy
    success, output = run_command(
        [sys.executable, "-m", "pip", "install", "cupy-cuda11x"],
        "Installing pre-built CuPy (CUDA 11.x)"
    )
    
    if not success:
        print("⚠️  Pre-built CuPy installation failed, trying alternative...")
        
        # Try generic CuPy
        success, output = run_command(
            [sys.executable, "-m", "pip", "install", "cupy"],
            "Installing generic CuPy"
        )
        
        if not success:
            print("❌ CuPy installation failed")
            print("✅ System will use CPU fallback for all operations")
            print("💡 To enable GPU acceleration later:")
            print("   1. Install CUDA Toolkit")
            print("   2. Install compatible CuPy version")
            print("   3. Restart the application")
            return True  # Not a fatal error
    
    # Try to install PyTorch with CUDA support
    print("🚀 Installing PyTorch with CUDA support...")
    success, output = run_command(
        [sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu118"],
        "Installing PyTorch with CUDA support"
    )
    
    if not success:
        print("⚠️  PyTorch CUDA installation failed, installing CPU version...")
        success, output = run_command(
            [sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio"],
            "Installing PyTorch (CPU version)"
        )
    
    return True

def install_optional_dependencies() -> bool:
    """Install optional dependencies."""
    optional_packages = [
        "ta-lib>=0.4.20",
        "pandas-ta>=0.3.14b0",
        "apscheduler>=3.9.1",
        "dask>=2022.5.0",
        "qiskit>=0.36.0",
        "pennylane>=0.26.0",
        "sqlalchemy>=1.4.0",
        "redis>=4.0.0",
        "dash>=2.12.0",
        "streamlit>=1.34.0"
    ]
    
    print("\n🔍 Optional Dependencies Installation")
    print("=" * 50)
    
    for package in optional_packages:
        success, output = run_command(
            [sys.executable, "-m", "pip", "install", package],
            f"Installing {package}"
        )
        
        if not success:
            print(f"⚠️  Optional package {package} failed to install")
            print(f"   Error: {output}")
            print("   Continuing with other packages...")
    
    return True

def install_platform_specific_dependencies(capabilities: Dict[str, bool]) -> bool:
    """Install platform-specific dependencies."""
    print("\n🔍 Platform-Specific Dependencies")
    print("=" * 50)
    
    if capabilities["windows"]:
        print("🪟 Installing Windows-specific dependencies...")
        windows_packages = [
            "pywin32>=228",
            "wmi>=1.5.1"
        ]
        
        for package in windows_packages:
            success, output = run_command(
                [sys.executable, "-m", "pip", "install", package],
                f"Installing {package}"
            )
            
            if not success:
                print(f"⚠️  Windows package {package} failed to install")
    
    return True

def test_installation() -> bool:
    """Test the installation by importing key modules."""
    print("\n🧪 Testing Installation")
    print("=" * 50)
    
    test_modules = [
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("pandas", "Pandas"),
        ("numba", "Numba"),
        ("sklearn", "Scikit-learn"),
        ("ccxt", "CCXT"),
        ("yaml", "PyYAML"),
        ("dotenv", "python-dotenv"),
        ("loguru", "Loguru"),
        ("psutil", "psutil"),
        ("aiohttp", "aiohttp"),
        ("requests", "requests"),
        ("matplotlib", "matplotlib"),
        ("plotly", "plotly"),
        ("flask", "Flask"),
        ("fastapi", "FastAPI"),
        ("cryptography", "cryptography"),
        ("pytest", "pytest")
    ]
    
    failed_imports = []
    
    for module_name, display_name in test_modules:
        try:
            __import__(module_name)
            print(f"✅ {display_name} imported successfully")
        except ImportError as e:
            print(f"❌ {display_name} import failed: {e}")
            failed_imports.append(display_name)
    
    # Test GPU modules with fallback
    print("\n🔍 Testing GPU Modules (with fallback)...")
    
    try:
        import cupy as cp
        print("✅ CuPy imported successfully (GPU acceleration available)")
    except ImportError:
        print("⚠️  CuPy not available (CPU fallback will be used)")
    
    try:
        import torch
        if torch.cuda.is_available():
            print("✅ PyTorch with CUDA support available")
        else:
            print("⚠️  PyTorch available but CUDA not detected")
    except ImportError:
        print("⚠️  PyTorch not available")
    
    if failed_imports:
        print(f"\n❌ {len(failed_imports)} modules failed to import:")
        for module in failed_imports:
            print(f"   - {module}")
        return False
    
    print("\n✅ All core modules imported successfully!")
    return True

def main():
    """Main installation function."""
    print("🚀 Schwabot Trading System - Smart Requirements Installer")
    print("=" * 60)
    
    # Detect system capabilities
    print("🔍 Detecting system capabilities...")
    capabilities = detect_system_capabilities()
    
    print(f"📋 System: {platform.system()} {platform.release()}")
    print(f"🐍 Python: {sys.version}")
    print(f"🔧 CUDA Environment: {capabilities['cuda_environment']}")
    print(f"🎮 GPU Available: {capabilities['gpu_available']}")
    
    # Upgrade pip first
    print("\n🔄 Upgrading pip...")
    success, output = run_command(
        [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
        "Upgrading pip"
    )
    
    if not success:
        print("⚠️  Pip upgrade failed, continuing anyway...")
    
    # Install dependencies in order
    steps = [
        ("Core Dependencies", install_core_dependencies),
        ("GPU Dependencies", lambda: install_gpu_dependencies(capabilities)),
        ("Optional Dependencies", install_optional_dependencies),
        ("Platform Dependencies", lambda: install_platform_specific_dependencies(capabilities)),
        ("Installation Test", test_installation)
    ]
    
    for step_name, step_func in steps:
        print(f"\n{'='*60}")
        print(f"📦 {step_name}")
        print(f"{'='*60}")
        
        if not step_func():
            print(f"❌ {step_name} failed")
            if step_name == "Core Dependencies":
                print("💥 Critical failure - cannot continue")
                return False
            else:
                print("⚠️  Non-critical failure - continuing...")
    
    print(f"\n{'='*60}")
    print("🎉 Installation Complete!")
    print(f"{'='*60}")
    print("✅ Schwabot trading system is ready to use")
    print("🚀 You can now run your trading strategies")
    
    if not capabilities["gpu_available"]:
        print("\n💡 GPU Acceleration:")
        print("   - System will use CPU fallback for all operations")
        print("   - Performance will be adequate for most trading strategies")
        print("   - To enable GPU acceleration, install CUDA Toolkit and compatible drivers")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n❌ Installation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1) 