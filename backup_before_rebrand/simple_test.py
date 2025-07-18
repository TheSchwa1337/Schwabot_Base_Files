#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Schwabot System Test
===========================
Basic test to validate core systems are working.
"""

import sys
import os
import logging
import time
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_basic_imports():
    """Test basic imports that should work."""
    logger.info("🔍 Testing basic imports...")
    
    results = {
        'passed': 0,
        'failed': 0,
        'errors': []
    }
    
    # Test basic modules that should work
    basic_modules = [
        'numpy',
        'pandas',
        'logging',
        'json',
        'time',
        'datetime'
    ]
    
    for module_name in basic_modules:
        try:
            __import__(module_name)
            results['passed'] += 1
            logger.info(f"✅ {module_name} - OK")
        except Exception as e:
            results['failed'] += 1
            results['errors'].append(f"{module_name}: {str(e)}")
            logger.error(f"❌ {module_name} - FAILED: {e}")
    
    return results

def test_hash_config_manager():
    """Test hash config manager specifically."""
    logger.info("🔧 Testing Hash Config Manager...")
    
    try:
        from core.hash_config_manager import HashConfigManager, get_hash_settings
        
        # Test initialization
        hash_manager = HashConfigManager()
        hash_settings = get_hash_settings()
        
        # Test hash generation
        test_hash = hash_manager.generate_hash_from_string("test_data")
        
        if test_hash and len(test_hash) > 0:
            logger.info("✅ Hash Config Manager - OK")
            logger.info(f"   Generated hash: {test_hash[:16]}...")
            logger.info(f"   Hash length: {len(test_hash)}")
            return True
        else:
            logger.error("❌ Hash generation failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Hash Config Manager - FAILED: {e}")
        return False

def test_mathlib():
    """Test mathlib functionality."""
    logger.info("🧮 Testing MathLib...")
    
    try:
        from mathlib import MathLib
        
        math_lib = MathLib()
        
        # Test basic operations
        add_result = math_lib.add(1.0, 2.0)
        multiply_result = math_lib.multiply(3.0, 4.0)
        sqrt_result = math_lib.sqrt(16.0)
        
        if add_result == 3.0 and multiply_result == 12.0 and sqrt_result == 4.0:
            logger.info("✅ MathLib - OK")
            logger.info(f"   Add: 1 + 2 = {add_result}")
            logger.info(f"   Multiply: 3 * 4 = {multiply_result}")
            logger.info(f"   Sqrt: √16 = {sqrt_result}")
            return True
        else:
            logger.error("❌ MathLib calculations failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ MathLib - FAILED: {e}")
        return False

def test_quantum_strategy():
    """Test quantum strategy engine."""
    logger.info("⚛️ Testing Quantum Strategy Engine...")
    
    try:
        from mathlib.quantum_strategy import QuantumStrategyEngine
        
        qse = QuantumStrategyEngine()
        
        # Test superposition strategy creation
        strategy = qse.create_superposition_strategy(
            "test_strategy",
            ["BTC", "ETH", "ADA"],
            [0.4, 0.4, 0.2]
        )
        
        if strategy and strategy.strategy_id == "test_strategy":
            logger.info("✅ Quantum Strategy Engine - OK")
            logger.info(f"   Strategy ID: {strategy.strategy_id}")
            logger.info(f"   Strategy Type: {strategy.strategy_type}")
            return True
        else:
            logger.error("❌ Quantum strategy creation failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Quantum Strategy Engine - FAILED: {e}")
        return False

def test_persistent_homology():
    """Test persistent homology."""
    logger.info("🔺 Testing Persistent Homology...")
    
    try:
        from mathlib.persistent_homology import PersistentHomology
        import numpy as np
        
        ph = PersistentHomology()
        
        # Test with sample data
        points = np.random.rand(10, 2)
        simplices = ph.build_simplicial_complex(points, 0.5)
        
        if simplices and len(simplices) > 0:
            logger.info("✅ Persistent Homology - OK")
            logger.info(f"   Built {len(simplices)} simplices")
            return True
        else:
            logger.error("❌ Persistent homology computation failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Persistent Homology - FAILED: {e}")
        return False

def test_phantom_band_navigator():
    """Test phantom band navigator strategy."""
    logger.info("🔮 Testing Phantom Band Navigator...")
    
    try:
        from strategies.phantom_band_navigator import PhantomBandNavigator
        
        navigator = PhantomBandNavigator(
            symbols=["BTC", "ETH"],
            base_position_size=0.01,
            max_risk_per_trade=0.02
        )
        
        # Test market condition analysis
        test_prices = [100, 101, 102, 101, 100, 99, 98, 97, 96, 95]
        market_condition = navigator.analyze_market_condition(test_prices)
        
        if market_condition in ["bull", "bear", "sideways", "volatile"]:
            logger.info("✅ Phantom Band Navigator - OK")
            logger.info(f"   Market condition: {market_condition}")
            return True
        else:
            logger.error("❌ Market condition analysis failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Phantom Band Navigator - FAILED: {e}")
        return False

def main():
    """Main test function."""
    logger.info("🚀 STARTING SIMPLE SCHWABOT SYSTEM TEST")
    logger.info("=" * 50)
    
    start_time = time.time()
    
    # Run tests
    test_results = []
    
    # Test 1: Basic imports
    basic_results = test_basic_imports()
    test_results.append(("Basic Imports", basic_results['passed'], basic_results['failed']))
    
    # Test 2: Hash Config Manager
    hash_success = test_hash_config_manager()
    test_results.append(("Hash Config Manager", 1 if hash_success else 0, 0 if hash_success else 1))
    
    # Test 3: MathLib
    math_success = test_mathlib()
    test_results.append(("MathLib", 1 if math_success else 0, 0 if math_success else 1))
    
    # Test 4: Quantum Strategy
    quantum_success = test_quantum_strategy()
    test_results.append(("Quantum Strategy", 1 if quantum_success else 0, 0 if quantum_success else 1))
    
    # Test 5: Persistent Homology
    homology_success = test_persistent_homology()
    test_results.append(("Persistent Homology", 1 if homology_success else 0, 0 if homology_success else 1))
    
    # Test 6: Phantom Band Navigator
    phantom_success = test_phantom_band_navigator()
    test_results.append(("Phantom Band Navigator", 1 if phantom_success else 0, 0 if phantom_success else 1))
    
    # Calculate totals
    total_passed = sum(passed for _, passed, _ in test_results)
    total_failed = sum(failed for _, _, failed in test_results)
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Print results
    logger.info("=" * 50)
    logger.info("📊 TEST RESULTS SUMMARY")
    logger.info("=" * 50)
    logger.info(f"⏱️  Duration: {duration:.2f} seconds")
    
    for test_name, passed, failed in test_results:
        status = "✅ PASS" if failed == 0 else "❌ FAIL"
        logger.info(f"{status} {test_name}: {passed} passed, {failed} failed")
    
    logger.info(f"\n📈 Overall: {total_passed} passed, {total_failed} failed")
    
    if total_failed == 0:
        logger.info("🎉 ALL CORE SYSTEMS OPERATIONAL!")
        return 0
    else:
        logger.error("❌ SOME SYSTEMS HAVE ISSUES")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 