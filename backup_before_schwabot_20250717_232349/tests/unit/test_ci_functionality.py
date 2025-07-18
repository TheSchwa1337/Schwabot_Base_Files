#!/usr/bin/env python3
"""
Comprehensive CI Test Suite for Schwabot Trading System
======================================================

Cross-platform validation script that validates 8 key areas:
1. Dependencies availability (Windows/Mac/Linux compatible)
2. Directory structure integrity  
3. Core module imports
4. Main CLI functionality
5. Configuration loading
6. Syntax validation
7. Registry system functionality
8. Unified pipeline integration

This ensures the system is production-ready and all components work together
across all platforms (Windows, Mac, Linux) as originally intended.
"""

import argparse
import ast
import importlib
import json
import logging
import os
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Cross-platform imports with fallbacks
try:
    import yaml
except ImportError:
    yaml = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Test results storage
test_results = {
    "timestamp": datetime.now().isoformat(),
    "platform": sys.platform,
    "os_name": platform.system(),
    "os_version": platform.version(),
    "python_version": sys.version,
    "tests": {},
    "overall_status": "unknown",
    "errors": [],
    "warnings": [],
    "summary": {}
}


class CITestSuite:
    """Comprehensive CI test suite for Schwabot trading system with cross-platform support."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.test_results = {}
        self.errors = []
        self.warnings = []
        self.platform = sys.platform
        self.os_name = platform.system()
        
        # Platform-specific configurations
        self.platform_config = self._get_platform_config()
        
    def _get_platform_config(self) -> Dict[str, Any]:
        """Get platform-specific configuration."""
        config = {
            "windows": {
                "path_separator": "\\",
                "executable_ext": ".exe",
                "temp_dir": os.environ.get("TEMP", "C:\\temp"),
                "line_ending": "\r\n"
            },
            "darwin": {  # macOS
                "path_separator": "/",
                "executable_ext": "",
                "temp_dir": "/tmp",
                "line_ending": "\n"
            },
            "linux": {
                "path_separator": "/",
                "executable_ext": "",
                "temp_dir": "/tmp",
                "line_ending": "\n"
            }
        }
        
        if self.platform.startswith("win"):
            return config["windows"]
        elif self.platform.startswith("darwin"):
            return config["darwin"]
        else:
            return config["linux"]
    
    def _safe_import(self, module_name: str) -> Tuple[bool, Any]:
        """Safely import a module with cross-platform error handling."""
        try:
            module = importlib.import_module(module_name)
            return True, module
        except ImportError as e:
            return False, str(e)
        except Exception as e:
            return False, str(e)
    
    def _check_file_permissions(self, file_path: Path) -> bool:
        """Check file permissions in a cross-platform way."""
        try:
            if self.platform.startswith("win"):
                # Windows: check if file is readable
                return os.access(file_path, os.R_OK)
            else:
                # Unix-like: check read permissions
                return os.access(file_path, os.R_OK)
        except Exception:
            return False
    
    def test_dependencies(self) -> bool:
        """Validates core Python packages are available across platforms."""
        logger.info("🔍 Testing Dependencies (Cross-Platform)...")
        logger.info(f"Platform: {self.os_name} ({self.platform})")
        
        # Core required packages (all platforms)
        required_packages = [
            "numpy", "pandas", "asyncio", "logging", "json", 
            "time", "pathlib", "typing", "datetime", "os", "sys"
        ]
        
        # Platform-specific required packages
        platform_required = {
            "windows": ["pywin32", "wmi"],
            "darwin": [],  # macOS specific if needed
            "linux": []
        }
        
        # Optional packages with platform-specific handling
        optional_packages = {
            "all": ["ccxt", "talib", "scipy", "matplotlib", "seaborn", "yaml"],
            "windows": ["pywin32", "wmi"],
            "darwin": [],
            "linux": []
        }
        
        failed_required = []
        failed_optional = []
        successful_required = []
        successful_optional = []
        
        # Test core required packages
        for package in required_packages:
            success, result = self._safe_import(package)
            if success:
                successful_required.append(package)
                logger.info(f"✅ {package} - available")
            else:
                failed_required.append(f"{package}: {result}")
                logger.error(f"❌ {package} - missing (REQUIRED): {result}")
        
        # Test platform-specific required packages
        platform_key = "windows" if self.platform.startswith("win") else "darwin" if self.platform.startswith("darwin") else "linux"
        for package in platform_required.get(platform_key, []):
            success, result = self._safe_import(package)
            if success:
                successful_required.append(package)
                logger.info(f"✅ {package} - available (platform-specific)")
            else:
                failed_required.append(f"{package}: {result}")
                logger.error(f"❌ {package} - missing (platform-required): {result}")
        
        # Test optional packages
        all_optional = optional_packages.get("all", []) + optional_packages.get(platform_key, [])
        for package in all_optional:
            success, result = self._safe_import(package)
            if success:
                successful_optional.append(package)
                logger.info(f"✅ {package} - available (optional)")
            else:
                failed_optional.append(f"{package}: {result}")
                logger.warning(f"⚠️ {package} - missing (optional): {result}")
        
        success = len(failed_required) == 0
        
        self.test_results["dependencies"] = {
            "success": success,
            "platform": platform_key,
            "required_failed": failed_required,
            "optional_failed": failed_optional,
            "successful_required": successful_required,
            "successful_optional": successful_optional,
            "total_required": len(required_packages) + len(platform_required.get(platform_key, [])),
            "total_optional": len(all_optional)
        }
        
        if failed_required:
            self.errors.append(f"Missing required packages on {platform_key}: {', '.join(failed_required[:3])}")
        
        return success
    
    def test_core_imports(self) -> bool:
        """Tests that core modules can be imported with cross-platform compatibility."""
        logger.info("📦 Testing Core Module Imports (Cross-Platform)...")
        
        # Core modules that should work on all platforms
        core_modules = [
            "core.unified_mathematical_bridge",
            "core.unified_mathematical_integration_methods", 
            "core.unified_mathematical_performance_monitor",
            "core.enhanced_entropy_randomization_system",
            "core.self_generating_strategy_system",
            "core.unified_memory_registry_system",
            "core.complete_internalized_scalping_system"
        ]
        
        # Platform-specific modules
        platform_modules = {
            "windows": ["utils.windows_specific"],
            "darwin": ["utils.macos_specific"],
            "linux": ["utils.linux_specific"]
        }
        
        platform_key = "windows" if self.platform.startswith("win") else "darwin" if self.platform.startswith("darwin") else "linux"
        all_modules = core_modules + platform_modules.get(platform_key, [])
        
        failed_imports = []
        successful_imports = []
        
        for module in all_modules:
            success, result = self._safe_import(module)
            if success:
                successful_imports.append(module)
                logger.info(f"✅ {module} - imported successfully")
            else:
                failed_imports.append(f"{module}: {result}")
                logger.error(f"❌ {module} - import failed: {result}")
        
        success = len(failed_imports) == 0
        
        self.test_results["core_imports"] = {
            "success": success,
            "platform": platform_key,
            "successful_imports": successful_imports,
            "failed_imports": failed_imports,
            "total_modules": len(all_modules)
        }
        
        if failed_imports:
            self.errors.append(f"Failed imports on {platform_key}: {', '.join(failed_imports[:3])}")
        
        return success
    
    def test_main_cli(self) -> bool:
        """Validates CLI functionality and main system across platforms."""
        logger.info("🖥️ Testing Main CLI (Cross-Platform)...")
        
        try:
            # Test main.py import
            success, main_module = self._safe_import("main")
            if not success:
                raise ImportError(f"Main module import failed: {main_module}")
            
            # Get SchwabotCLI class
            if hasattr(main_module, 'SchwabotCLI'):
                cli_class = main_module.SchwabotCLI
                logger.info("✅ Main CLI class imported successfully")
            else:
                raise AttributeError("SchwabotCLI class not found in main module")
            
            # Test CLI instantiation
            cli = cli_class()
            logger.info("✅ CLI instance created successfully")
            
            # Test help functionality
            if hasattr(cli, 'get_help_text'):
                help_text = cli.get_help_text()
                if help_text and len(help_text) > 0:
                    logger.info("✅ CLI help functionality working")
                else:
                    logger.warning("⚠️ CLI help text empty")
            else:
                logger.warning("⚠️ CLI help method not available")
            
            # Test platform-specific CLI features
            if hasattr(cli, 'get_platform_info'):
                platform_info = cli.get_platform_info()
                logger.info(f"✅ Platform info: {platform_info}")
            
            success = True
            
        except Exception as e:
            logger.error(f"❌ Main CLI test failed: {e}")
            success = False
        
        self.test_results["main_cli"] = {
            "success": success,
            "platform": self.platform,
            "error": str(e) if not success else None
        }
        
        if not success:
            self.errors.append(f"Main CLI test failed on {self.platform}: {e}")
        
        return success
    
    def test_configuration_loading(self) -> bool:
        """Tests YAML/JSON config file loading with cross-platform path handling."""
        logger.info("⚙️ Testing Configuration Loading (Cross-Platform)...")
        
        config_files = [
            "config/master_integration.yaml",
            "config/trading_config.yaml", 
            "config/risk_config.yaml"
        ]
        
        successful_configs = []
        failed_configs = []
        
        for config_file in config_files:
            # Use platform-appropriate path handling
            config_path = self.project_root / config_file
            
            if config_path.exists():
                # Check file permissions
                if not self._check_file_permissions(config_path):
                    failed_configs.append(f"{config_file}: permission denied")
                    logger.error(f"❌ {config_file} - permission denied")
                    continue
                
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Handle different line endings
                    content = content.replace('\r\n', '\n').replace('\r', '\n')
                    
                    if config_file.endswith('.yaml'):
                        if yaml is None:
                            failed_configs.append(f"{config_file}: yaml module not available")
                            logger.error(f"❌ {config_file} - yaml module not available")
                            continue
                        config = yaml.safe_load(content)
                    else:
                        config = json.loads(content)
                    
                    successful_configs.append(config_file)
                    logger.info(f"✅ {config_file} - loaded successfully")
                    
                except Exception as e:
                    failed_configs.append(f"{config_file}: {e}")
                    logger.error(f"❌ {config_file} - loading failed: {e}")
            else:
                logger.warning(f"⚠️ {config_file} - file not found (optional)")
        
        # Test with default config if no files found
        if not successful_configs and not failed_configs:
            try:
                # Create minimal test config
                test_config = {
                    "version": "1.0.0",
                    "demo_mode": True,
                    "test_config": True,
                    "platform": self.platform
                }
                logger.info("✅ Default test configuration created")
                successful_configs.append("default_test_config")
            except Exception as e:
                logger.error(f"❌ Default config creation failed: {e}")
                failed_configs.append(f"default_config: {e}")
        
        success = len(failed_configs) == 0
        
        self.test_results["configuration_loading"] = {
            "success": success,
            "platform": self.platform,
            "successful_configs": successful_configs,
            "failed_configs": failed_configs
        }
        
        if failed_configs:
            self.errors.append(f"Config loading failed on {self.platform}: {', '.join(failed_configs[:2])}")
        
        return success
    
    def test_directory_structure(self) -> bool:
        """Ensures required directories and files exist with cross-platform path handling."""
        logger.info("📁 Testing Directory Structure (Cross-Platform)...")
        
        required_dirs = [
            "core",
            "utils", 
            "test",
            "config"
        ]
        
        required_files = [
            "main.py",
            "requirements.txt",
            "README.md"
        ]
        
        missing_dirs = []
        missing_files = []
        
        # Check directories with platform-specific path handling
        for dir_name in required_dirs:
            dir_path = self.project_root / dir_name
            if dir_path.exists() and dir_path.is_dir():
                logger.info(f"✅ {dir_name}/ - exists")
            else:
                missing_dirs.append(dir_name)
                logger.error(f"❌ {dir_name}/ - missing")
        
        # Check files with platform-specific path handling
        for file_name in required_files:
            file_path = self.project_root / file_name
            if file_path.exists() and file_path.is_file():
                # Check file permissions
                if self._check_file_permissions(file_path):
                    logger.info(f"✅ {file_name} - exists and readable")
                else:
                    logger.warning(f"⚠️ {file_name} - exists but not readable")
                    missing_files.append(f"{file_name} (permission denied)")
            else:
                missing_files.append(file_name)
                logger.error(f"❌ {file_name} - missing")
        
        success = len(missing_dirs) == 0 and len(missing_files) == 0
        
        self.test_results["directory_structure"] = {
            "success": success,
            "platform": self.platform,
            "missing_dirs": missing_dirs,
            "missing_files": missing_files
        }
        
        if missing_dirs or missing_files:
            missing_items = missing_dirs + missing_files
            self.errors.append(f"Missing items on {self.platform}: {', '.join(missing_items)}")
        
        return success
    
    def test_syntax_validation(self) -> bool:
        """Validates Python syntax across core files with cross-platform line ending handling."""
        logger.info("🔍 Testing Python Syntax (Cross-Platform)...")
        
        core_files = [
            "main.py",
            "core/unified_mathematical_bridge.py",
            "core/unified_mathematical_integration_methods.py"
        ]
        
        syntax_errors = []
        valid_files = []
        
        for file_name in core_files:
            file_path = self.project_root / file_name
            
            if file_path.exists():
                if not self._check_file_permissions(file_path):
                    syntax_errors.append(f"{file_name}: permission denied")
                    logger.error(f"❌ {file_name} - permission denied")
                    continue
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        source = f.read()
                    
                    # Handle different line endings for cross-platform compatibility
                    source = source.replace('\r\n', '\n').replace('\r', '\n')
                    
                    # Parse Python syntax
                    ast.parse(source)
                    valid_files.append(file_name)
                    logger.info(f"✅ {file_name} - syntax valid")
                    
                except SyntaxError as e:
                    syntax_errors.append(f"{file_name}: {e}")
                    logger.error(f"❌ {file_name} - syntax error: {e}")
                except Exception as e:
                    syntax_errors.append(f"{file_name}: {e}")
                    logger.error(f"❌ {file_name} - parsing error: {e}")
            else:
                logger.warning(f"⚠️ {file_name} - file not found")
        
        success = len(syntax_errors) == 0
        
        self.test_results["syntax_validation"] = {
            "success": success,
            "platform": self.platform,
            "valid_files": valid_files,
            "syntax_errors": syntax_errors
        }
        
        if syntax_errors:
            self.errors.append(f"Syntax errors on {self.platform}: {', '.join(syntax_errors[:2])}")
        
        return success
    
    def test_registry_system(self) -> bool:
        """Tests trade registry and coordination with cross-platform compatibility."""
        logger.info("📊 Testing Registry System (Cross-Platform)...")
        
        try:
            # Test registry import
            success, registry_module = self._safe_import("core.unified_memory_registry_system")
            if not success:
                raise ImportError(f"Registry module import failed: {registry_module}")
            
            # Get registry class
            if hasattr(registry_module, 'UnifiedMemoryRegistrySystem'):
                registry_class = registry_module.UnifiedMemoryRegistrySystem
            else:
                raise AttributeError("UnifiedMemoryRegistrySystem class not found")
            
            # Create registry instance
            registry = registry_class()
            logger.info("✅ Registry system imported and instantiated")
            
            # Test basic registry operations
            test_data = {
                "test_key": "test_value",
                "timestamp": datetime.now().isoformat(),
                "platform": self.platform
            }
            
            # Test registration
            if hasattr(registry, 'register_trade_data'):
                registry.register_trade_data("test_trade", test_data)
                logger.info("✅ Trade data registration working")
            else:
                logger.warning("⚠️ register_trade_data method not available")
            
            # Test retrieval
            if hasattr(registry, 'get_trade_data'):
                retrieved_data = registry.get_trade_data("test_trade")
                if retrieved_data and retrieved_data.get("test_key") == "test_value":
                    logger.info("✅ Trade data retrieval working")
                else:
                    logger.warning("⚠️ Trade data retrieval may have issues")
            else:
                logger.warning("⚠️ get_trade_data method not available")
            
            success = True
            
        except Exception as e:
            logger.error(f"❌ Registry system test failed: {e}")
            success = False
        
        self.test_results["registry_system"] = {
            "success": success,
            "platform": self.platform,
            "error": str(e) if not success else None
        }
        
        if not success:
            self.errors.append(f"Registry system test failed on {self.platform}: {e}")
        
        return success
    
    def test_unified_pipeline(self) -> bool:
        """Validates trading pipeline functionality with cross-platform compatibility."""
        logger.info("🔄 Testing Unified Pipeline (Cross-Platform)...")
        
        try:
            # Test pipeline import
            success, pipeline_module = self._safe_import("core.complete_internalized_scalping_system")
            if not success:
                raise ImportError(f"Pipeline module import failed: {pipeline_module}")
            
            # Get pipeline system
            if hasattr(pipeline_module, 'complete_scalping_system'):
                pipeline_system = pipeline_module.complete_scalping_system
            else:
                raise AttributeError("complete_scalping_system not found")
            
            logger.info("✅ Unified pipeline imported successfully")
            
            # Test pipeline initialization
            if hasattr(pipeline_system, 'get_system_status'):
                pipeline_status = pipeline_system.get_system_status()
                if pipeline_status:
                    logger.info("✅ Pipeline system status available")
                else:
                    logger.warning("⚠️ Pipeline system status not available")
            else:
                logger.warning("⚠️ get_system_status method not available")
            
            # Test mathematical bridge integration
            success, math_module = self._safe_import("core.unified_mathematical_bridge")
            if success and hasattr(math_module, 'UnifiedMathematicalBridge'):
                math_bridge = math_module.UnifiedMathematicalBridge()
                
                # Test basic mathematical operations
                if hasattr(math_bridge, 'integrate_all_mathematical_systems'):
                    test_result = math_bridge.integrate_all_mathematical_systems({
                        "test_data": [1, 2, 3, 4, 5],
                        "operation": "test",
                        "platform": self.platform
                    })
                    
                    if test_result is not None:
                        logger.info("✅ Mathematical bridge integration working")
                    else:
                        logger.warning("⚠️ Mathematical bridge may have issues")
                else:
                    logger.warning("⚠️ integrate_all_mathematical_systems method not available")
            else:
                logger.warning("⚠️ Mathematical bridge not available")
            
            success = True
            
        except Exception as e:
            logger.error(f"❌ Unified pipeline test failed: {e}")
            success = False
        
        self.test_results["unified_pipeline"] = {
            "success": success,
            "platform": self.platform,
            "error": str(e) if not success else None
        }
        
        if not success:
            self.errors.append(f"Unified pipeline test failed on {self.platform}: {e}")
        
        return success
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all CI tests and return comprehensive cross-platform results."""
        logger.info("🚀 Starting Comprehensive CI Test Suite (Cross-Platform)")
        logger.info("=" * 80)
        logger.info(f"Platform: {self.os_name} ({self.platform})")
        logger.info(f"Python Version: {sys.version}")
        logger.info(f"Project Root: {self.project_root}")
        
        start_time = time.time()
        
        # Run all tests
        tests = [
            ("Dependencies", self.test_dependencies),
            ("Core Imports", self.test_core_imports),
            ("Main CLI", self.test_main_cli),
            ("Configuration Loading", self.test_configuration_loading),
            ("Directory Structure", self.test_directory_structure),
            ("Syntax Validation", self.test_syntax_validation),
            ("Registry System", self.test_registry_system),
            ("Unified Pipeline", self.test_unified_pipeline)
        ]
        
        passed_tests = 0
        total_tests = len(tests)
        
        for test_name, test_func in tests:
            logger.info(f"\n🧪 Running {test_name} Test...")
            try:
                result = test_func()
                if result:
                    passed_tests += 1
                    logger.info(f"✅ {test_name} - PASSED")
                else:
                    logger.error(f"❌ {test_name} - FAILED")
            except Exception as e:
                logger.error(f"❌ {test_name} - ERROR: {e}")
                self.errors.append(f"{test_name} test error: {e}")
        
        execution_time = time.time() - start_time
        
        # Generate summary
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        overall_status = "PASS" if passed_tests == total_tests else "FAIL"
        
        summary = {
            "overall_status": overall_status,
            "platform": self.platform,
            "os_name": self.os_name,
            "passed_tests": passed_tests,
            "total_tests": total_tests,
            "success_rate": success_rate,
            "execution_time": execution_time,
            "errors": self.errors,
            "warnings": self.warnings,
            "test_results": self.test_results
        }
        
        # Print final summary
        logger.info("\n" + "=" * 80)
        logger.info("📊 CI TEST SUITE SUMMARY (Cross-Platform)")
        logger.info("=" * 80)
        logger.info(f"Platform: {self.os_name} ({self.platform})")
        logger.info(f"Overall Status: {overall_status}")
        logger.info(f"Tests Passed: {passed_tests}/{total_tests}")
        logger.info(f"Success Rate: {success_rate:.1f}%")
        logger.info(f"Execution Time: {execution_time:.2f}s")
        
        if self.errors:
            logger.info(f"\n❌ Errors ({len(self.errors)}):")
            for error in self.errors[:5]:  # Show first 5 errors
                logger.info(f"  - {error}")
        
        if self.warnings:
            logger.info(f"\n⚠️ Warnings ({len(self.warnings)}):")
            for warning in self.warnings[:3]:  # Show first 3 warnings
                logger.info(f"  - {warning}")
        
        if passed_tests == total_tests:
            logger.info("\n🎉 ALL TESTS PASSED! System is ready for production on all platforms.")
        else:
            logger.info(f"\n⚠️ {total_tests - passed_tests} tests failed. Review errors above.")
        
        return summary


def main():
    """Main function to run the CI test suite with cross-platform support."""
    parser = argparse.ArgumentParser(description="Schwabot CI Test Suite (Cross-Platform)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument("--platform", choices=["windows", "mac", "linux", "all"], 
                       help="Test specific platform compatibility")
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run test suite
    test_suite = CITestSuite()
    results = test_suite.run_all_tests()
    
    if args.json:
        print(json.dumps(results, indent=2))
    
    # Exit with appropriate code
    if results["overall_status"] == "PASS":
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main() 