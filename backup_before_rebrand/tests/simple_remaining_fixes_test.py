#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Remaining Fixes Test
===========================

Quick test for the two remaining issues:
1. Risk Manager Edge Cases - VaR calculation for all-positive returns
2. Mathematical Bridge Fallback - Circular import resolution
"""

import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_risk_manager_var_calculation():
    """Test VaR calculation for different return scenarios."""
    logger.info("🛡️ Testing Risk Manager VaR Calculation")
    
    try:
        from core.risk_manager import RiskManager
        risk_manager = RiskManager()
        
        # Test 1: All-positive returns
        logger.info("  Testing all-positive returns...")
        all_positive = np.random.uniform(0.001, 0.02, 100)
        metrics_positive = risk_manager.calculate_risk_metrics(all_positive)
        
        logger.info(f"    All-positive VaR(95%): {metrics_positive.var_95:.4f}")
        logger.info(f"    All-positive VaR(99%): {metrics_positive.var_99:.4f}")
        
        # For all-positive returns, VaR should be positive (this is mathematically correct)
        positive_var_correct = metrics_positive.var_95 > 0 and metrics_positive.var_99 > 0
        logger.info(f"    All-positive VaR correct: {positive_var_correct}")
        
        # Test 2: Mixed returns
        logger.info("  Testing mixed returns...")
        mixed_returns = np.random.normal(0, 0.02, 100)
        metrics_mixed = risk_manager.calculate_risk_metrics(mixed_returns)
        
        logger.info(f"    Mixed VaR(95%): {metrics_mixed.var_95:.4f}")
        logger.info(f"    Mixed VaR(99%): {metrics_mixed.var_99:.4f}")
        
        # For mixed returns, VaR should typically be negative
        mixed_var_correct = metrics_mixed.var_95 < 0 and metrics_mixed.var_99 < 0
        logger.info(f"    Mixed VaR correct: {mixed_var_correct}")
        
        # Test 3: All-negative returns
        logger.info("  Testing all-negative returns...")
        all_negative = np.random.uniform(-0.02, -0.001, 100)
        metrics_negative = risk_manager.calculate_risk_metrics(all_negative)
        
        logger.info(f"    All-negative VaR(95%): {metrics_negative.var_95:.4f}")
        logger.info(f"    All-negative VaR(99%): {metrics_negative.var_99:.4f}")
        
        # For all-negative returns, VaR should be negative
        negative_var_correct = metrics_negative.var_95 < 0 and metrics_negative.var_99 < 0
        logger.info(f"    All-negative VaR correct: {negative_var_correct}")
        
        overall_success = positive_var_correct and mixed_var_correct and negative_var_correct
        
        if overall_success:
            logger.info("  ✅ Risk Manager VaR Calculation: PASSED")
        else:
            logger.warning("  ⚠️ Risk Manager VaR Calculation: PARTIAL PASS")
            
        return overall_success
        
    except Exception as e:
        logger.error(f"  ❌ Risk Manager VaR Calculation failed: {e}")
        return False

def test_mathematical_bridge_fallback():
    """Test Mathematical Bridge fallback with circular import resolution."""
    logger.info("🧠 Testing Mathematical Bridge Fallback")
    
    try:
        # Test lazy import resolution
        from core.unified_mathematical_bridge import UnifiedMathematicalBridge
        
        # Create bridge instance
        bridge = UnifiedMathematicalBridge()
        
        # Test initialization
        bridge_initialized = (
            bridge is not None and
            hasattr(bridge, 'config') and
            hasattr(bridge, 'logger')
        )
        
        if bridge_initialized:
            logger.info("    ✅ Bridge initialized successfully")
        else:
            logger.warning("    ⚠️ Bridge initialization issues")
        
        # Test fallback confidence calculation
        try:
            connection_strength = bridge._calculate_quantum_phantom_connection_strength(
                {'confidence': 0.5}, {'phantom_confidence': 0.5}
            )
            
            # Ensure minimum value guarantee
            min_value_valid = connection_strength >= 0.1
            
            if min_value_valid:
                logger.info(f"    ✅ Connection strength: {connection_strength:.3f} (>= 0.1)")
            else:
                logger.warning(f"    ⚠️ Connection strength: {connection_strength:.3f} (< 0.1)")
                
        except Exception as e:
            logger.warning(f"    ⚠️ Connection strength calculation failed: {e}")
            min_value_valid = False
        
        # Test overall confidence calculation
        try:
            # Create mock connections
            mock_connections = [
                type('MockConnection', (), {'connection_strength': 0.5})(),
                type('MockConnection', (), {'connection_strength': 0.6})()
            ]
            
            overall_confidence = bridge._calculate_overall_confidence(mock_connections)
            
            # Ensure minimum confidence guarantee
            min_confidence_valid = overall_confidence >= 0.1
            
            if min_confidence_valid:
                logger.info(f"    ✅ Overall confidence: {overall_confidence:.3f} (>= 0.1)")
            else:
                logger.warning(f"    ⚠️ Overall confidence: {overall_confidence:.3f} (< 0.1)")
                
        except Exception as e:
            logger.warning(f"    ⚠️ Overall confidence calculation failed: {e}")
            min_confidence_valid = False
        
        overall_success = bridge_initialized and min_value_valid and min_confidence_valid
        
        if overall_success:
            logger.info("  ✅ Mathematical Bridge Fallback: PASSED")
        else:
            logger.warning("  ⚠️ Mathematical Bridge Fallback: PARTIAL PASS")
            
        return overall_success
        
    except Exception as e:
        logger.error(f"  ❌ Mathematical Bridge Fallback failed: {e}")
        return False

def main():
    """Main test execution."""
    logger.info("🚀 Starting Simple Remaining Fixes Test")
    logger.info("=" * 50)
    
    # Test 1: Risk Manager Edge Cases
    risk_test_passed = test_risk_manager_var_calculation()
    
    logger.info("")
    
    # Test 2: Mathematical Bridge Fallback
    bridge_test_passed = test_mathematical_bridge_fallback()
    
    # Summary
    logger.info("")
    logger.info("📊 TEST SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Risk Manager Edge Cases: {'✅ PASS' if risk_test_passed else '❌ FAIL'}")
    logger.info(f"Mathematical Bridge Fallback: {'✅ PASS' if bridge_test_passed else '❌ FAIL'}")
    
    total_passed = sum([risk_test_passed, bridge_test_passed])
    total_tests = 2
    
    logger.info(f"Success Rate: {total_passed}/{total_tests} ({total_passed/total_tests*100:.1f}%)")
    
    if total_passed == total_tests:
        logger.info("🎉 ALL REMAINING ISSUES RESOLVED!")
    else:
        logger.info(f"⚠️ {total_tests - total_passed} ISSUES REMAIN")
    
    return total_passed == total_tests

if __name__ == "__main__":
    main() 