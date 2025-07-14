import os
import sys
from pathlib import Path

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Schwabot Package Setup
=====================

Setup script for packaging Schwabot with brain trading functionality into an executable.
This creates a self-contained .exe that includes all working components.
"""


def create_requirements_txt():
    """Create requirements.txt with minimal dependencies."""
    requirements = []
        "numpy>=1.21.0",
        "asyncio",
        "pathlib",
        "dataclasses",
        "typing",
        "logging",
        "json",
        "hashlib",
        "time",
        "math",
    ]

    with open("requirements.txt", "w") as f:
        for req in requirements:
            f.write(f"{req}\n")

    print("Created requirements.txt")


def create_pyinstaller_spec():
    """Create PyInstaller spec file for executable creation."""
    spec_content = """# -*- mode: python ; coding: utf-8 -*-"

block_cipher = None

a = Analysis()
    ['test_brain_integration.py'],
    pathex=[],
    binaries=[],
    datas=[]
        ('core/brain_trading_engine.py', 'core'),
        ('core/unified_math_system.py', 'core'),
        ('core/type_defs.py', 'core'),
        ('utils/safe_print.py', 'utils'),
    ],
    hiddenimports=[]
        'numpy',
        'asyncio',
        'logging',
        'json',
        'time',
        'math',
        'hashlib',
        'pathlib',
        'dataclasses',
        'typing'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE()
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='SchwabotBrainTrader',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)
"""

    with open("schwabot.spec", "w") as f:
        f.write(spec_content)

    print("Created PyInstaller spec file")


def create_build_script():
    """Create build script for easy executable creation."""
    if sys.platform.startswith("win"):
        script_content = """@echo off""
echo Building Schwabot Brain Trading Executable...
echo.

echo Installing dependencies...
pip install -r requirements.txt
pip install pyinstaller

echo.
echo Building executable...
pyinstaller schwabot.spec --clean --noconfirm

echo.
    if exist "dist\\SchwabotBrainTrader.exe" ()
    echo [SUCCESS] Build successful! Executable created at: dist\\SchwabotBrainTrader.exe
    echo.
    echo Testing executable...
    cd dist
    SchwabotBrainTrader.exe
) else (
    echo [ERROR] Build failed!
)

pause
"""
        script_name = "build.bat"

        # Write with UTF-8 encoding for Windows
        with open(script_name, "w", encoding="utf-8") as f:
            f.write(script_content)
    else:
        script_content = """#!/bin/bash"
echo "Building Schwabot Brain Trading Executable..."
echo

echo "Installing dependencies..."
pip install -r requirements.txt
pip install pyinstaller

echo
echo "Building executable..."
pyinstaller schwabot.spec --clean --noconfirm

echo
    if [ -f "dist/SchwabotBrainTrader" ]; then
    echo "[SUCCESS] Build successful! Executable created at: dist/SchwabotBrainTrader"
    echo
    echo "Testing executable..."
    cd dist
    ./SchwabotBrainTrader
else
    echo "[ERROR] Build failed!"
fi
"""
        script_name = "build.sh"

        with open(script_name, "w") as f:
            f.write(script_content)

        if not sys.platform.startswith("win"):
            os.chmod(script_name, 0o755)

    print(f"Created build script: {script_name}")


def create_readme():
    """Create README for the packaged application."""
    readme_content = """# Schwabot Brain Trading System"

## Overview

Schwabot is an advanced trading bot with brain-enhanced signal processing capabilities.
This package includes working implementations of:

- [BRAIN] Brain Trading Engine with mathematical optimization
- [CHART] Real-time signal processing and analysis
- [MONEY] Profit optimization algorithms
- [GRAPH] Backtesting simulation capabilities
- [SEARCH] Code quality validation

## Features

### Brain Trading Engine
- Advanced signal processing using brain algorithms
- Mathematical profit optimization
- Confidence-based decision making
- Historical performance tracking

### Mathematical Framework
- Unified math system integration
- Tensor-based calculations
- Risk-adjusted return analysis
- Portfolio optimization metrics

### Trading Capabilities
- Real-time market signal processing
- Automated trading decisions
- Position sizing based on confidence
- Risk management protocols

## Quick Start

### Running the Executable
```bash
# Windows
SchwabotBrainTrader.exe

# Linux/Mac
./SchwabotBrainTrader
```

### Running from Source
```bash
python test_brain_integration.py
```

## Configuration

The system can be configured by modifying the brain trading engine parameters:

```python
config = {}
    'base_profit_rate': 0.02,        # 0.2% base profit rate
    'confidence_threshold': 0.7,       # 70% confidence threshold
    'enhancement_range': (0.8, 1.8),   # Enhancement factor range
    'max_history_size': 1000           # Maximum signal history
}
```

## Test Results

The system includes comprehensive testing:
- [PASS] Brain Trading Engine functionality
- [PASS] Mathematical operations
- [PASS] Symbol processing
- [PASS] Backtesting simulation
- [PASS] Code quality validation

## Output Files

The system generates several output files:
- `test_brain_signals.json` - Brain trading signals data
- `test_results.json` - Comprehensive test results
- `logs/` - System logs and debug information

## API Integration

To integrate your own API keys and trading endpoints:

1. Modify the brain trading engine configuration
2. Add your API credentials to the appropriate configuration files
3. Update the market data sources as needed

## Dependencies

- Python 3.8+
- NumPy for mathematical operations
- AsyncIO for concurrent processing
- Standard library modules

## Building from Source

To build your own executable:

```bash
# Install dependencies
pip install -r requirements.txt
pip install pyinstaller

# Build executable
pyinstaller schwabot.spec --clean --noconfirm
```

## Support

For issues and support, please review the test outputs and logs for diagnostic information.

---

**Schwabot Brain Trading System v1.0**
*Advanced Trading with Brain-Enhanced Signal Processing*
"""

    with open("README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)

    print("Created README.md")


def verify_core_files():
    """Verify that all core files exist."""
    required_files = ["core/brain_trading_engine.py", "test_brain_integration.py"]

    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)

    if missing_files:
        print(f"[ERROR] Missing required files: {missing_files}")
        return False
    else:
        print("All required files present")
        return True


def main():
    """Main setup execution."""
    print("SCHWABOT PACKAGE SETUP")
    print("=" * 50)

    # Verify core files
    if not verify_core_files():
        print("[ERROR] Setup failed - missing core files")
        return

    # Create package files
    create_requirements_txt()
    create_pyinstaller_spec()
    create_build_script()
    create_readme()

    print("\nPACKAGE SETUP COMPLETE")
    print("=" * 50)
    print("Next steps:")
    print("1. Run the build script to create executable:")

    if sys.platform.startswith("win"):
        print("   build.bat")
    else:
        print("   ./build.sh")

    print("2. Test the system:")
    print("   python test_brain_integration.py")
    print("3. The executable will be in the 'dist/' directory")

    print("\n[SUCCESS] Ready for packaging!")


if __name__ == "__main__":
    main()
