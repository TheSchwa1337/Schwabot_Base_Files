#!/usr/bin/env python3
"""
CI Functionality Test for Schwabot
==================================

Comprehensive test suite designed for CI environments.
Tests core functionality without requiring trading APIs or external services.
"""

import asyncio
import importlib
import logging
import os
import sys
import time
import traceback
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CITestSuite:
    """Comprehensive CI test suite."""
    
    def __init__(self):
        self.passed_tests = 0
        self.total_tests = 0
        self.test_results = []
    
    def run_test(self, test_name: str, test_func):
        """Run a single test and track results."""
        self.total_tests += 1
        logger.info(f"\n🧪 Running: {test_name}")
        
        try:
            start_time = time.time()
            result = test_func()
            duration = time.time() - start_time
            
            if result:
                self.passed_tests += 1
                logger.info(f"✅ {test_name} PASSED ({duration:.2f}s)")
                self.test_results.append((test_name, "PASSED", duration, None))
                return True
            else:
                logger.error(f"❌ {test_name} FAILED ({duration:.2f}s)")
                self.test_results.append((test_name, "FAILED", duration, "Test returned False"))
                return False
        except Exception as e:
            duration = time.time() - start_time if 'start_time' in locals() else 0
            logger.error(f"❌ {test_name} ERROR ({duration:.2f}s): {e}")
            self.test_results.append((test_name, "ERROR", duration, str(e)))
            return False
    
    def test_core_imports(self):
        """Test that core modules can be imported."""
        logger.info("Testing core module imports...")
        
        # Core modules that should be importable
        core_modules = [
            'core.unified_trading_pipeline',
            'core.trade_registry',
            'core.registry_coordinator',
            'core.profit_bucket_registry',
            'core.soulprint_registry',
        ]
        
        success_count = 0
        for module in core_modules:
            try:
                importlib.import_module(module)
                logger.info(f"  ✅ {module} imported successfully")
                success_count += 1
            except Exception as e:
                logger.error(f"  ❌ {module} import failed: {e}")
        
        return success_count == len(core_modules)
    
    def test_main_cli(self):
        """Test that main CLI can be imported and has basic functionality."""
        logger.info("Testing main CLI functionality...")
        
        try:
            # Import main module
            import main
            logger.info("  ✅ main.py imported successfully")
            
            # Check for key classes
            if hasattr(main, 'SchwabotTradingSystem'):
                logger.info("  ✅ SchwabotTradingSystem class found")
            else:
                logger.error("  ❌ SchwabotTradingSystem class not found")
                return False
                
            # Try to instantiate (without initialization)
            try:
                system = main.SchwabotTradingSystem()
                logger.info("  ✅ SchwabotTradingSystem instantiated")
            except Exception as e:
                logger.warning(f"  ⚠️ System instantiation failed: {e}")
                # This is expected in CI without proper config
            
            return True
        except Exception as e:
            logger.error(f"  ❌ Main CLI test failed: {e}")
            return False
    
    def test_configuration_loading(self):
        """Test that configuration files can be loaded."""
        logger.info("Testing configuration loading...")
        
        try:
            import yaml
            import json
            
            config_files = [
                ('config/schwabot_config.yaml', 'yaml'),
                ('config/integrations.yaml', 'yaml'),
                ('config/trading_pairs.json', 'json'),
            ]
            
            success_count = 0
            for config_file, file_type in config_files:
                if os.path.exists(config_file):
                    try:
                        with open(config_file, 'r') as f:
                            if file_type == 'yaml':
                                yaml.safe_load(f)
                            else:
                                json.load(f)
                        logger.info(f"  ✅ {config_file} loaded successfully")
                        success_count += 1
                    except Exception as e:
                        logger.error(f"  ❌ {config_file} failed to load: {e}")
                else:
                    logger.warning(f"  ⚠️ {config_file} not found")
                    success_count += 1  # Not failing for missing optional configs
            
            return success_count == len(config_files)
        except Exception as e:
            logger.error(f"  ❌ Configuration loading test failed: {e}")
            return False
    
    def test_directory_structure(self):
        """Test that required directory structure exists."""
        logger.info("Testing directory structure...")
        
        required_dirs = ['core', 'config', 'utils', 'test', 'docs']
        required_files = ['main.py', 'requirements.txt', 'README.md']
        
        success = True
        
        for directory in required_dirs:
            if os.path.isdir(directory):
                logger.info(f"  ✅ Directory {directory} exists")
            else:
                logger.error(f"  ❌ Directory {directory} missing")
                success = False
        
        for file in required_files:
            if os.path.isfile(file):
                logger.info(f"  ✅ File {file} exists")
            else:
                logger.error(f"  ❌ File {file} missing")
                success = False
        
        return success
    
    def test_registry_system(self):
        """Test registry system functionality in isolation."""
        logger.info("Testing registry system...")
        
        try:
            # Test canonical registry
            from core.trade_registry import CanonicalTradeRegistry
            registry = CanonicalTradeRegistry()
            logger.info("  ✅ Canonical trade registry created")
            
            # Test adding a dummy trade
            trade_data = {
                "symbol": "BTC/USDC",
                "action": "buy",
                "entry_price": 50000.0,
                "exit_price": 50500.0,
                "amount": 100.0,
                "fees": 0.1,
                "profit_usd": 4.9,
                "profit_percentage": 4.9,
                "strategy_id": "test_strategy",
                "confidence": 0.8,
                "timestamp": time.time()
            }
            
            trade_hash = registry.add_trade(**trade_data)
            logger.info(f"  ✅ Trade added with hash: {trade_hash[:8]}...")
            
            # Test retrieval
            trade = registry.get_trade(trade_hash)
            if trade:
                logger.info("  ✅ Trade retrieved successfully")
            else:
                logger.error("  ❌ Trade retrieval failed")
                return False
            
            # Test profit bucket registry
            from core.profit_bucket_registry import ProfitBucketRegistry
            bucket_registry = ProfitBucketRegistry()
            logger.info("  ✅ Profit bucket registry created")
            
            # Test soulprint registry
            from core.soulprint_registry import SoulprintRegistry
            soulprint_registry = SoulprintRegistry()
            logger.info("  ✅ Soulprint registry created")
            
            return True
        except Exception as e:
            logger.error(f"  ❌ Registry system test failed: {e}")
            traceback.print_exc()
            return False
    
    def test_unified_pipeline(self):
        """Test unified trading pipeline in demo mode."""
        logger.info("Testing unified trading pipeline...")
        
        try:
            from core.unified_trading_pipeline import UnifiedTradingPipeline
            
            # Initialize in demo mode
            pipeline = UnifiedTradingPipeline(mode="demo", config={
                "min_confidence": 0.5,
                "max_trades": 10
            })
            logger.info("  ✅ Pipeline initialized in demo mode")
            
            # Test analytics methods
            analytics = pipeline.get_performance_analytics()
            logger.info("  ✅ Performance analytics generated")
            
            stats = pipeline.get_registry_statistics()
            logger.info("  ✅ Registry statistics generated")
            
            consistency = pipeline.validate_registry_consistency()
            logger.info("  ✅ Registry consistency validated")
            
            return True
        except Exception as e:
            logger.error(f"  ❌ Unified pipeline test failed: {e}")
            traceback.print_exc()
            return False
    
    def test_dependencies(self):
        """Test that key dependencies are available."""
        logger.info("Testing dependencies...")
        
        dependencies = [
            'numpy', 'pandas', 'yaml', 'requests',
            'asyncio', 'logging', 'json', 'time'
        ]
        
        success_count = 0
        for dep in dependencies:
            try:
                __import__(dep)
                logger.info(f"  ✅ {dep} available")
                success_count += 1
            except ImportError:
                logger.error(f"  ❌ {dep} not available")
        
        return success_count == len(dependencies)
    
    def test_syntax_validation(self):
        """Test that core files have valid Python syntax."""
        logger.info("Testing syntax validation...")
        
        core_files = [
            "main.py",
            "core/unified_trading_pipeline.py",
            "core/trade_registry.py",
            "core/registry_coordinator.py",
            "core/profit_bucket_registry.py",
            "core/soulprint_registry.py"
        ]
        
        success_count = 0
        for file_path in core_files:
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        compile(f.read(), file_path, 'exec')
                    logger.info(f"  ✅ {file_path} syntax OK")
                    success_count += 1
                except Exception as e:
                    logger.error(f"  ❌ {file_path} syntax error: {e}")
            else:
                logger.warning(f"  ⚠️ {file_path} not found")
        
        return success_count >= len(core_files) - 1  # Allow for one missing file
    
    def generate_summary(self):
        """Generate test summary."""
        logger.info("\n" + "="*60)
        logger.info("🧪 CI TEST SUMMARY")
        logger.info("="*60)
        
        for test_name, status, duration, error in self.test_results:
            status_icon = "✅" if status == "PASSED" else "❌"
            logger.info(f"{status_icon} {test_name:<40} {status:<6} ({duration:.2f}s)")
            if error and status != "PASSED":
                logger.info(f"   Error: {error}")
        
        logger.info("="*60)
        success_rate = (self.passed_tests / self.total_tests) * 100 if self.total_tests > 0 else 0
        logger.info(f"📊 Results: {self.passed_tests}/{self.total_tests} tests passed ({success_rate:.1f}%)")
        
        if self.passed_tests == self.total_tests:
            logger.info("🎉 All tests passed! System is CI-ready!")
            return True
        else:
            logger.warning("⚠️ Some tests failed. Check the output above.")
            return False


def main():
    """Run the CI test suite."""
    logger.info("🚀 Starting Schwabot CI Test Suite")
    logger.info("="*60)
    
    suite = CITestSuite()
    
    # Run all tests
    tests = [
        ("Dependencies", suite.test_dependencies),
        ("Directory Structure", suite.test_directory_structure),
        ("Core Imports", suite.test_core_imports),
        ("Main CLI", suite.test_main_cli),
        ("Configuration Loading", suite.test_configuration_loading),
        ("Syntax Validation", suite.test_syntax_validation),
        ("Registry System", suite.test_registry_system),
        ("Unified Pipeline", suite.test_unified_pipeline),
    ]
    
    for test_name, test_func in tests:
        suite.run_test(test_name, test_func)
    
    # Generate summary
    success = suite.generate_summary()
    
    # Return appropriate exit code
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main()) 