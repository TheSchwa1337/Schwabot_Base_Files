#!/usr/bin/env python3
"""
Comprehensive System Integration Test for Schwabot.

This script validates the entire Schwabot trading system integration,
including YAML configurations, core components, and pipeline connectivity.
"""

import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_yaml_configurations():
    """Test YAML configuration loading and validation."""
    logger.info("Testing YAML configurations...")
    
    try:
        from core.utils.yaml_config_loader import validate_settings, load_unified_settings, load_demo_config
        
        # Test loading configurations
        unified_settings = load_unified_settings()
        demo_config = load_demo_config()
        
        # Test validation
        is_valid = validate_settings()
        
        logger.info(f"Unified Settings loaded: {bool(unified_settings)}")
        logger.info(f"Demo Config loaded: {bool(demo_config)}")
        logger.info(f"All configurations valid: {is_valid}")
        
        return is_valid and bool(unified_settings) and bool(demo_config)
        
    except Exception as e:
        logger.error(f"YAML configuration test failed: {e}")
        return False


def test_core_components():
    """Test core component imports and initialization."""
    logger.info("Testing core components...")
    
    components_to_test = [
        ("fault_bus", "core.fault_bus"),
        ("dlt_waveform_engine", "core.dlt_waveform_engine"),
        ("multi_bit_btc_processor", "core.multi_bit_btc_processor"),
        ("riddle_gemm", "core.riddle_gemm"),
        ("temporal_execution_correction_layer", "core.temporal_execution_correction_layer"),
        ("ghost_strategy_handler", "core.ghost_strategy_handler"),
        ("profit_routing_engine", "core.profit_routing_engine"),
        ("advanced_mathematical_core", "core.advanced_mathematical_core"),
        ("constants", "core.constants"),
        ("type_defs", "core.type_defs")
    ]
    
    success_count = 0
    total_count = len(components_to_test)
    
    for component_name, import_path in components_to_test:
        try:
            __import__(import_path)
            logger.info(f"✓ {component_name}: Import successful")
            success_count += 1
        except Exception as e:
            logger.error(f"✗ {component_name}: Import failed - {e}")
    
    logger.info(f"Core components test: {success_count}/{total_count} successful")
    return success_count == total_count


def test_mathematical_framework():
    """Test mathematical framework components."""
    logger.info("Testing mathematical framework...")
    
    try:
        from core.advanced_mathematical_core import (
            safe_delta_calculation,
            normalized_delta_tanh,
            kelly_criterion_allocation
        )
        
        # Test mathematical functions
        delta = safe_delta_calculation(100.0, 95.0)
        normalized = normalized_delta_tanh(100.0, 95.0)
        
        logger.info(f"✓ Mathematical functions: delta={delta:.4f}, normalized={normalized:.4f}")
        return True
        
    except Exception as e:
        logger.error(f"Mathematical framework test failed: {e}")
        return False


def test_demo_system():
    """Test demo system components."""
    logger.info("Testing demo system...")
    
    demo_files = [
        "demo/demo_backtest_matrix.yaml",
        "demo/demo_trade_sequence.py",
        "demo/demo_logic_flow.py",
        "demo/demo_launcher.py"
    ]
    
    success_count = 0
    total_count = len(demo_files)
    
    for demo_file in demo_files:
        if Path(demo_file).exists():
            logger.info(f"✓ {demo_file}: File exists")
            success_count += 1
        else:
            logger.error(f"✗ {demo_file}: File missing")
    
    logger.info(f"Demo system test: {success_count}/{total_count} files found")
    return success_count == total_count


def test_settings_system():
    """Test settings system components."""
    logger.info("Testing settings system...")
    
    settings_files = [
        "settings/vector_validator.py",
        "settings/settings_controller.py",
        "settings/matrix_allocator.py",
        "settings/known_bad_vector_map.json"
    ]
    
    success_count = 0
    total_count = len(settings_files)
    
    for settings_file in settings_files:
        if Path(settings_file).exists():
            logger.info(f"✓ {settings_file}: File exists")
            success_count += 1
        else:
            logger.error(f"✗ {settings_file}: File missing")
    
    logger.info(f"Settings system test: {success_count}/{total_count} files found")
    return success_count == total_count


def test_mathematical_libraries():
    """Test mathematical library components."""
    logger.info("Testing mathematical libraries...")
    
    mathlib_files = [
        "mathlib/__init__.py",
        "mathlib/quantum_strategy.py",
        "mathlib/persistent_homology.py"
    ]
    
    success_count = 0
    total_count = len(mathlib_files)
    
    for mathlib_file in mathlib_files:
        if Path(mathlib_file).exists():
            logger.info(f"✓ {mathlib_file}: File exists")
            success_count += 1
        else:
            logger.error(f"✗ {mathlib_file}: File missing")
    
    logger.info(f"Mathematical libraries test: {success_count}/{total_count} files found")
    return success_count == total_count


def test_ncco_core():
    """Test NCCO core components."""
    logger.info("Testing NCCO core...")
    
    ncco_files = [
        "ncco_core/__init__.py",
        "ncco_core/ncco.py",
        "ncco_core/ncco_generator.py",
        "ncco_core/ncco_scorer.py"
    ]
    
    success_count = 0
    total_count = len(ncco_files)
    
    for ncco_file in ncco_files:
        if Path(ncco_file).exists():
            logger.info(f"✓ {ncco_file}: File exists")
            success_count += 1
        else:
            logger.error(f"✗ {ncco_file}: File missing")
    
    logger.info(f"NCCO core test: {success_count}/{total_count} files found")
    return success_count == total_count


def test_aleph_core():
    """Test Aleph core components."""
    logger.info("Testing Aleph core...")
    
    aleph_files = [
        "aleph_core/__init__.py",
        "aleph_core/entropy_analyzer.py",
        "aleph_core/pattern_matcher.py",
        "aleph_core/smart_money_analyzer.py"
    ]
    
    success_count = 0
    total_count = len(aleph_files)
    
    for aleph_file in aleph_files:
        if Path(aleph_file).exists():
            logger.info(f"✓ {aleph_file}: File exists")
            success_count += 1
        else:
            logger.error(f"✗ {aleph_file}: File missing")
    
    logger.info(f"Aleph core test: {success_count}/{total_count} files found")
    return success_count == total_count


def test_pipeline_connectivity():
    """Test pipeline connectivity between components."""
    logger.info("Testing pipeline connectivity...")
    
    try:
        # Test that core components can be imported together
        from core.fault_bus import FaultBus
        from core.dlt_waveform_engine import DLTWaveformEngine
        from core.multi_bit_btc_processor import MultiBitBTCProcessor
        from core.riddle_gemm import RiddleGEMMEngine
        
        # Test basic initialization
        fault_bus = FaultBus()
        dlt_engine = DLTWaveformEngine()
        multi_bit_processor = MultiBitBTCProcessor()
        riddle_engine = RiddleGEMMEngine(vector_size=10)
        
        logger.info("✓ Pipeline connectivity: All core components initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Pipeline connectivity test failed: {e}")
        return False


def test_windows_cli_compatibility():
    """Test Windows CLI compatibility features."""
    logger.info("Testing Windows CLI compatibility...")
    
    try:
        from core.fault_bus import WindowsCliCompatibilityHandler
        
        handler = WindowsCliCompatibilityHandler()
        
        # Test Windows CLI detection
        is_windows = handler.is_windows_cli()
        logger.info(f"✓ Windows CLI detection: {is_windows}")
        
        # Test safe print
        safe_message = handler.safe_print("Test message with emoji 🚀")
        logger.info(f"✓ Safe print: {safe_message}")
        
        return True
        
    except Exception as e:
        logger.error(f"Windows CLI compatibility test failed: {e}")
        return False


def run_comprehensive_test():
    """Run comprehensive system integration test."""
    logger.info("=" * 60)
    logger.info("SCHWABOT SYSTEM INTEGRATION TEST")
    logger.info("=" * 60)
    
    test_results = {}
    
    # Run all tests
    tests = [
        ("YAML Configurations", test_yaml_configurations),
        ("Core Components", test_core_components),
        ("Mathematical Framework", test_mathematical_framework),
        ("Demo System", test_demo_system),
        ("Settings System", test_settings_system),
        ("Mathematical Libraries", test_mathematical_libraries),
        ("NCCO Core", test_ncco_core),
        ("Aleph Core", test_aleph_core),
        ("Pipeline Connectivity", test_pipeline_connectivity),
        ("Windows CLI Compatibility", test_windows_cli_compatibility)
    ]
    
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            result = test_func()
            test_results[test_name] = result
        except Exception as e:
            logger.error(f"Test {test_name} failed with exception: {e}")
            test_results[test_name] = False
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(test_results.values())
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "PASS" if result else "FAIL"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall Result: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED! System is ready for deployment.")
        return True
    else:
        logger.error("❌ Some tests failed. Please review the issues above.")
        return False


if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1) 