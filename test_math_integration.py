#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Mathematical Integration Components
=======================================

This script tests all mathematical integration components to identify
which ones are working and which need attention.
"""

import sys
import os
import asyncio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_mathematical_components():
    """Test all mathematical components."""
    results = {}
    
    # Test 1: DLT Waveform Engine
    try:
        from backtesting.mathematical_integration_simplified import SimplifiedDLTEngine
        engine = SimplifiedDLTEngine()
        results["DLT Waveform Engine"] = "✅ WORKING"
        logger.info("✅ DLT Waveform Engine: WORKING")
    except Exception as e:
        results["DLT Waveform Engine"] = f"❌ FAILED: {e}"
        logger.error(f"❌ DLT Waveform Engine: FAILED - {e}")
    
    # Test 2: Dualistic Thought Engines
    try:
        from backtesting.mathematical_integration_simplified import SimplifiedDualisticEngine
        aleph = SimplifiedDualisticEngine("ALEPH")
        alif = SimplifiedDualisticEngine("ALIF")
        ritl = SimplifiedDualisticEngine("RITL")
        rittle = SimplifiedDualisticEngine("RITTLE")
        results["Dualistic Thought Engines"] = "✅ WORKING (4/4 engines)"
        logger.info("✅ Dualistic Thought Engines: WORKING (4/4 engines)")
    except Exception as e:
        results["Dualistic Thought Engines"] = f"❌ FAILED: {e}"
        logger.error(f"❌ Dualistic Thought Engines: FAILED - {e}")
    
    # Test 3: Bit Phase Resolution
    try:
        from backtesting.mathematical_integration_simplified import SimplifiedMathematicalIntegrationEngine
        engine = SimplifiedMathematicalIntegrationEngine()
        # Test bit phase resolution
        market_data = {"current_price": 50000, "volatility": 0.15}
        bit_phase = engine._resolve_bit_phase(market_data)
        if isinstance(bit_phase, int) and bit_phase in [4, 8, 16, 32, 42]:
            results["Bit Phase Resolution"] = "✅ WORKING"
            logger.info("✅ Bit Phase Resolution: WORKING")
        else:
            results["Bit Phase Resolution"] = f"❌ FAILED: Invalid bit phase {bit_phase}"
            logger.error(f"❌ Bit Phase Resolution: FAILED - Invalid bit phase {bit_phase}")
    except Exception as e:
        results["Bit Phase Resolution"] = f"❌ FAILED: {e}"
        logger.error(f"❌ Bit Phase Resolution: FAILED - {e}")
    
    # Test 4: Matrix Basket Tensor Operations
    try:
        from backtesting.mathematical_integration_simplified import SimplifiedMathematicalIntegrationEngine
        engine = SimplifiedMathematicalIntegrationEngine()
        market_data = {"current_price": 50000, "volatility": 0.15}
        matrix_basket = engine._calculate_matrix_basket(market_data)
        tensor_score = engine._calculate_tensor_score(market_data)
        if isinstance(matrix_basket, int) and isinstance(tensor_score, float):
            results["Matrix Basket Tensor Operations"] = "✅ WORKING"
            logger.info("✅ Matrix Basket Tensor Operations: WORKING")
        else:
            results["Matrix Basket Tensor Operations"] = "❌ FAILED: Invalid return types"
            logger.error("❌ Matrix Basket Tensor Operations: FAILED - Invalid return types")
    except Exception as e:
        results["Matrix Basket Tensor Operations"] = f"❌ FAILED: {e}"
        logger.error(f"❌ Matrix Basket Tensor Operations: FAILED - {e}")
    
    # Test 5: Ferris RDE Phase System
    try:
        from backtesting.mathematical_integration_simplified import SimplifiedMathematicalIntegrationEngine
        engine = SimplifiedMathematicalIntegrationEngine()
        market_data = {"current_price": 50000, "volatility": 0.15}
        ferris_phase = engine._calculate_ferris_phase(market_data)
        if isinstance(ferris_phase, float) and 0 <= ferris_phase <= 1:
            results["Ferris RDE Phase System"] = "✅ WORKING"
            logger.info("✅ Ferris RDE Phase System: WORKING")
        else:
            results["Ferris RDE Phase System"] = f"❌ FAILED: Invalid phase {ferris_phase}"
            logger.error(f"❌ Ferris RDE Phase System: FAILED - Invalid phase {ferris_phase}")
    except Exception as e:
        results["Ferris RDE Phase System"] = f"❌ FAILED: {e}"
        logger.error(f"❌ Ferris RDE Phase System: FAILED - {e}")
    
    # Test 6: Quantum State Analysis
    try:
        from backtesting.mathematical_integration_simplified import SimplifiedMathematicalIntegrationEngine
        engine = SimplifiedMathematicalIntegrationEngine()
        market_data = {"current_price": 50000, "volatility": 0.15}
        quantum_state = asyncio.run(engine._process_quantum_state(market_data))
        if isinstance(quantum_state, dict):
            results["Quantum State Analysis"] = "✅ WORKING"
            logger.info("✅ Quantum State Analysis: WORKING")
        else:
            results["Quantum State Analysis"] = "❌ FAILED: Invalid return type"
            logger.error("❌ Quantum State Analysis: FAILED - Invalid return type")
    except Exception as e:
        results["Quantum State Analysis"] = f"❌ FAILED: {e}"
        logger.error(f"❌ Quantum State Analysis: FAILED - {e}")
    
    # Test 7: Entropy Calculations
    try:
        from backtesting.mathematical_integration_simplified import SimplifiedMathematicalIntegrationEngine
        engine = SimplifiedMathematicalIntegrationEngine()
        market_data = {"current_price": 50000, "volatility": 0.15, "price_history": [50000, 50100, 49900, 50200]}
        entropy = engine._calculate_entropy(market_data)
        if isinstance(entropy, float) and entropy >= 0:
            results["Entropy Calculations"] = "✅ WORKING"
            logger.info("✅ Entropy Calculations: WORKING")
        else:
            results["Entropy Calculations"] = f"❌ FAILED: Invalid entropy {entropy}"
            logger.error(f"❌ Entropy Calculations: FAILED - Invalid entropy {entropy}")
    except Exception as e:
        results["Entropy Calculations"] = f"❌ FAILED: {e}"
        logger.error(f"❌ Entropy Calculations: FAILED - {e}")
    
    # Test 8: Vault Orbital Bridge
    try:
        from backtesting.mathematical_integration_simplified import SimplifiedMathematicalIntegrationEngine
        engine = SimplifiedMathematicalIntegrationEngine()
        market_data = {"current_price": 50000, "volatility": 0.15}
        vault_state = asyncio.run(engine._process_vault_orbital(market_data))
        if isinstance(vault_state, dict):
            results["Vault Orbital Bridge"] = "✅ WORKING"
            logger.info("✅ Vault Orbital Bridge: WORKING")
        else:
            results["Vault Orbital Bridge"] = "❌ FAILED: Invalid return type"
            logger.error("❌ Vault Orbital Bridge: FAILED - Invalid return type")
    except Exception as e:
        results["Vault Orbital Bridge"] = f"❌ FAILED: {e}"
        logger.error(f"❌ Vault Orbital Bridge: FAILED - {e}")
    
    # Test 9: Complete Mathematical Integration
    try:
        from backtesting.mathematical_integration_simplified import SimplifiedMathematicalIntegrationEngine
        engine = SimplifiedMathematicalIntegrationEngine()
        market_data = {"current_price": 50000, "volatility": 0.15, "price_history": [50000, 50100, 49900, 50200]}
        signal = asyncio.run(engine.process_market_data_mathematically(market_data))
        if hasattr(signal, 'confidence') and hasattr(signal, 'decision'):
            results["Complete Mathematical Integration"] = "✅ WORKING"
            logger.info("✅ Complete Mathematical Integration: WORKING")
        else:
            results["Complete Mathematical Integration"] = "❌ FAILED: Invalid signal object"
            logger.error("❌ Complete Mathematical Integration: FAILED - Invalid signal object")
    except Exception as e:
        results["Complete Mathematical Integration"] = f"❌ FAILED: {e}"
        logger.error(f"❌ Complete Mathematical Integration: FAILED - {e}")
    
    return results

def main():
    """Main test function."""
    logger.info("🧮 Testing Mathematical Integration Components...")
    
    results = test_mathematical_components()
    
    # Count working components
    working = sum(1 for status in results.values() if status.startswith("✅"))
    total = len(results)
    
    logger.info("\n" + "="*60)
    logger.info("📊 MATHEMATICAL INTEGRATION TEST RESULTS")
    logger.info("="*60)
    
    for component, status in results.items():
        logger.info(f"{component}: {status}")
    
    logger.info("="*60)
    logger.info(f"🎯 Overall Result: {working}/{total} components working ({working/total*100:.1f}% success rate)")
    
    if working == total:
        logger.info("🎉 ALL MATHEMATICAL COMPONENTS ARE WORKING!")
    elif working >= total * 0.8:
        logger.info("✅ Most components working - system is operational!")
    else:
        logger.info("⚠️ Some components need attention")
    
    return working, total

if __name__ == "__main__":
    working, total = main()
    sys.exit(0 if working == total else 1) 