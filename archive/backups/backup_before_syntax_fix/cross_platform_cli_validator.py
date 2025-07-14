#!/usr/bin/env python3
"""
Cross-Platform CLI Validator for Schwabot Trading System

Validates CLI functionality across Windows, macOS, and Linux platforms.
Tests all major CLI commands, file operations, and platform-specific features.
"""

import argparse
import asyncio
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.clean_trading_pipeline import CleanTradingPipeline
from core.strategy_bit_mapper import StrategyBitMapper
from utils.safe_print import error, info, safe_print, success, warn


class CrossPlatformCLIValidator:
    """Validates CLI functionality across different platforms."""

    def __init__(self):
        """Initialize the validator."""
        self.platform_info = self._get_platform_info()
        self.test_results = {}
            "platform": self.platform_info,
            "tests": {},
            "summary": {"passed": 0, "failed": 0, "total": 0}
        }
        self.temp_dir = None

    def _get_platform_info(self) -> Dict[str, str]:
        """Get detailed platform information."""
        return {}
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": sys.version,
            "python_executable": sys.executable,
            "platform": sys.platform
        }

    def print_banner(self, title: str, icon: str = "🔍") -> None:
        """Print a formatted banner."""
        safe_print(f"\n{icon} {title}")
        safe_print("=" * (len(title) + 4))

    def run_test(self, test_name: str, test_func, *args, **kwargs) -> bool:
        """Run a test and record results."""
        self.test_results["summary"]["total"] += 1

        try:
            result = test_func(*args, **kwargs)
            if result:
                self.test_results["tests"][test_name] = {"status": "PASS", "details": "Test passed"}
                self.test_results["summary"]["passed"] += 1
                success(f"✅ {test_name}: PASS")
                return True
            else:
                self.test_results["tests"][test_name] = {
                    "status": "FAIL", "details": "Test returned False"}
                self.test_results["summary"]["failed"] += 1
                error(f"❌ {test_name}: FAIL")
                return False
        except Exception as e:
            self.test_results["tests"][test_name] = {"status": "ERROR", "details": str(e)}
            self.test_results["summary"]["failed"] += 1
            error(f"💥 {test_name}: ERROR - {e}")
            return False

    def test_platform_detection(self) -> bool:
        """Test platform detection and compatibility."""
        info("Testing platform detection...")

        # Check if we can detect the platform
        if not self.platform_info["system"]:
            return False

        # Check Python version compatibility
        python_version = sys.version_info
        if python_version < (3, 8):
            warn("Python version < 3.8 detected")
            return False

        # Check for required platform-specific modules
        try:
            import asyncio
            import pathlib
            import subprocess
            import tempfile
        except ImportError as e:
            error(f"Missing required module: {e}")
            return False

        return True

    def test_file_operations(self) -> bool:
        """Test file operations across platforms."""
        info("Testing file operations...")

        try:
            # Create temporary directory
            self.temp_dir = tempfile.mkdtemp(prefix="schwabot_test_")
            test_file = Path(self.temp_dir) / "test_file.txt"

            # Test file creation
            test_file.write_text("Test content")
            if not test_file.exists():
                return False

            # Test file reading
            content = test_file.read_text()
            if content != "Test content":
                return False

            # Test file deletion
            test_file.unlink()
            if test_file.exists():
                return False

            return True

        except Exception as e:
            error(f"File operations test failed: {e}")
            return False

    def test_cli_imports(self) -> bool:
        """Test CLI module imports."""
        info("Testing CLI module imports...")

        try:
            # Test core imports
            from core.api.data_models import OrderRequest

            # Test API imports
            from core.api.integration_manager import ApiIntegrationManager
            from core.clean_trading_pipeline import CleanTradingPipeline
            from core.schwafit_core import SchwafitCore
            from core.strategy_bit_mapper import StrategyBitMapper
            from utils.logging_setup import setup_logging

            # Test utility imports
            from utils.safe_print import safe_print

            return True

        except ImportError as e:
            error(f"Import test failed: {e}")
            return False

    def test_safe_print_functionality(self) -> bool:
        """Test safe print functionality."""
        info("Testing safe print functionality...")

        try:
            # Test basic safe print
            safe_print("Test message")

            # Test unicode handling
            safe_print("Unicode test: 🚀💰📈")

            # Test error handling
            safe_print(None)
            safe_print({"test": "data"})

            return True

        except Exception as e:
            error(f"Safe print test failed: {e}")
            return False

    def test_cli_command_execution(self) -> bool:
        """Test CLI command execution."""
        info("Testing CLI command execution...")

        try:
            # Test basic CLI help
            result = subprocess.run()
                [sys.executable, "schwabot_cli.py", "--help"],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode != 0:
                error(f"CLI help command failed: {result.stderr}")
                return False

            # Test status command
            result = subprocess.run()
                [sys.executable, "schwabot_cli.py", "status"],
                capture_output=True,
                text=True,
                timeout=30
            )

            # Status command might fail if no matrix directory, but should not crash
            if result.returncode not in [0, 1]:
                error(f"CLI status command failed: {result.stderr}")
                return False

            return True

        except subprocess.TimeoutExpired:
            error("CLI command execution timed out")
            return False
        except Exception as e:
            error(f"CLI command execution test failed: {e}")
            return False

    def test_async_functionality(self) -> bool:
        """Test async functionality."""
        info("Testing async functionality...")

        try:
            async def test_async():
                await asyncio.sleep(0.1)
                return True

            result = asyncio.run(test_async())
            return result

        except Exception as e:
            error(f"Async functionality test failed: {e}")
            return False

    def test_core_components(self) -> bool:
        """Test core component initialization."""
        info("Testing core component initialization...")

        try:
            # Test SchwafitCore
            schwafit = SchwafitCore()

            # Test StrategyBitMapper (with temp, directory)
            if self.temp_dir:
                bit_mapper = StrategyBitMapper(self.temp_dir)

            # Test CleanTradingPipeline
            pipeline = CleanTradingPipeline()

            return True

        except Exception as e:
            error(f"Core components test failed: {e}")
            return False

    def test_platform_specific_features(self) -> bool:
        """Test platform-specific features."""
        info("Testing platform-specific features...")

        system = self.platform_info["system"].lower()

        try:
            if system == "windows":
                # Test Windows-specific features
                import ctypes
                kernel32 = ctypes.windll.kernel32

            elif system == "darwin":
                # Test macOS-specific features
                pass  # Add macOS-specific tests if needed

            elif system == "linux":
                # Test Linux-specific features
                pass  # Add Linux-specific tests if needed

            return True

        except Exception as e:
            error(f"Platform-specific features test failed: {e}")
            return False

    def test_network_functionality(self) -> bool:
        """Test network functionality."""
        info("Testing network functionality...")

        try:
            import socket

            # Test socket creation
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.close()

            return True

        except Exception as e:
            error(f"Network functionality test failed: {e}")
            return False

    def test_math_operations(self) -> bool:
        """Test mathematical operations."""
        info("Testing mathematical operations...")

        try:
            import numpy as np

            # Test basic numpy operations
            arr = np.array([1, 2, 3, 4, 5])
            mean = np.mean(arr)
            std = np.std(arr)

            if mean != 3.0 or std != 1.4142135623730951:
                return False

            return True

        except Exception as e:
            error(f"Mathematical operations test failed: {e}")
            return False

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all cross-platform tests."""
        self.print_banner("CROSS-PLATFORM CLI VALIDATION", "🌐")

        info(f"Platform: {self.platform_info['system']} {self.platform_info['release']}")
        info(f"Python: {self.platform_info['python_version']}")
        info(f"Architecture: {self.platform_info['machine']}")

        # Run all tests
        tests = []
            ("Platform Detection", self.test_platform_detection),
            ("File Operations", self.test_file_operations),
            ("CLI Imports", self.test_cli_imports),
            ("Safe Print Functionality", self.test_safe_print_functionality),
            ("CLI Command Execution", self.test_cli_command_execution),
            ("Async Functionality", self.test_async_functionality),
            ("Core Components", self.test_core_components),
            ("Platform-Specific Features", self.test_platform_specific_features),
            ("Network Functionality", self.test_network_functionality),
            ("Mathematical Operations", self.test_math_operations),
        ]

        for test_name, test_func in tests:
            self.run_test(test_name, test_func)

        # Cleanup
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                import shutil
                shutil.rmtree(self.temp_dir)
            except Exception as e:
                warn(f"Failed to cleanup temp directory: {e}")

        return self.test_results

    def print_summary(self) -> None:
        """Print test summary."""
        summary = self.test_results["summary"]

        self.print_banner("TEST SUMMARY", "📊")

        safe_print(f"Total Tests: {summary['total']}")
        safe_print(f"Passed: {summary['passed']}")
        safe_print(f"Failed: {summary['failed']}")

        if summary['failed'] > 0:
            safe_print("\nFailed Tests:")
            for test_name, result in self.test_results["tests"].items():
                if result["status"] in ["FAIL", "ERROR"]:
                    safe_print(f"  ❌ {test_name}: {result['details']}")

        # Save results to file
        results_file = f"cross_platform_test_results_{self.platform_info['system'].lower()}.json"
        with open(results_file, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)

        safe_print(f"\nResults saved to: {results_file}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Cross-Platform CLI Validator")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--save-results", action="store_true", help="Save results to file")

    args = parser.parse_args()

    # Setup logging
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Run validator
    validator = CrossPlatformCLIValidator()
    results = validator.run_all_tests()
    validator.print_summary()

    # Exit with appropriate code
    if results["summary"]["failed"] > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main() 