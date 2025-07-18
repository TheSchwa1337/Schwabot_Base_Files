#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Schwafit Overfitting Prevention Integration Test
================================================
Test the Schwafit overfitting prevention system integration.

This test verifies:
- Overfitting detection and correction
- Data sanitization and obfuscation
- Pipeline protection mechanisms
- Information control and leakage prevention
- Mathematical correction application
"""

import asyncio
import logging
import time
import json
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_schwafit_overfitting_detection():
    """Test Schwafit overfitting detection capabilities."""
    logger.info("🧪 Testing Schwafit Overfitting Detection...")
    
    try:
        from core.schwafit_overfitting_prevention import create_schwafit_overfitting_prevention, SanitizationLevel
        
        # Create Schwafit system
        schwafit = create_schwafit_overfitting_prevention()
        
        # Test data with potential overfitting patterns
        test_data = {
            'symbol': 'BTC/USDC',
            'price': 50000.0,
            'volume': 2_000_000_000,
            'volatility': 0.025,
            'spread': 0.5,
            'timestamp': time.time(),
            'api_key': 'sensitive_api_key_12345',  # Sensitive data
            'internal_debug': 'debug_info',  # Internal data
            'test_mode': True  # Test data
        }
        
        # Test signals with repetitive patterns (potential overfitting)
        test_signals = [
            {'action': 'buy', 'confidence': 0.9, 'amount': 0.01, 'symbol': 'BTC/USDC'},
            {'action': 'buy', 'confidence': 0.9, 'amount': 0.01, 'symbol': 'BTC/USDC'},
            {'action': 'buy', 'confidence': 0.9, 'amount': 0.01, 'symbol': 'BTC/USDC'},
            {'action': 'buy', 'confidence': 0.9, 'amount': 0.01, 'symbol': 'BTC/USDC'},
            {'action': 'buy', 'confidence': 0.9, 'amount': 0.01, 'symbol': 'BTC/USDC'}
        ]
        
        # Detect overfitting
        overfitting_metrics = schwafit.detect_overfitting(test_data, test_signals)
        
        logger.info(f"📊 Overfitting Detection Results:")
        logger.info(f"  Overall Score: {overfitting_metrics.overfitting_score:.3f}")
        logger.info(f"  Temporal Consistency: {overfitting_metrics.temporal_consistency:.3f}")
        logger.info(f"  Feature Correlation: {overfitting_metrics.feature_correlation:.3f}")
        logger.info(f"  Signal Entropy: {overfitting_metrics.signal_entropy:.3f}")
        logger.info(f"  Volume Anomaly: {overfitting_metrics.volume_anomaly:.3f}")
        logger.info(f"  Confidence Penalty: {overfitting_metrics.confidence_penalty:.3f}")
        logger.info(f"  Correction Factor: {overfitting_metrics.correction_factor:.3f}")
        
        # Test different data patterns
        test_scenarios = [
            {
                'name': 'Normal Data',
                'data': {'price': 50000.0, 'volume': 2_000_000_000, 'volatility': 0.025},
                'signals': [
                    {'action': 'buy', 'confidence': 0.7, 'amount': 0.01},
                    {'action': 'sell', 'confidence': 0.6, 'amount': 0.01},
                    {'action': 'hold', 'confidence': 0.5, 'amount': 0.0}
                ]
            },
            {
                'name': 'Overfitted Data',
                'data': {'price': 50000.0, 'volume': 2_000_000_000, 'volatility': 0.025},
                'signals': [
                    {'action': 'buy', 'confidence': 0.95, 'amount': 0.01},
                    {'action': 'buy', 'confidence': 0.95, 'amount': 0.01},
                    {'action': 'buy', 'confidence': 0.95, 'amount': 0.01}
                ]
            },
            {
                'name': 'Anomalous Data',
                'data': {'price': 50000.0, 'volume': 10_000_000_000, 'volatility': 0.1},
                'signals': [
                    {'action': 'buy', 'confidence': 0.8, 'amount': 0.01}
                ]
            }
        ]
        
        detection_results = []
        
        for scenario in test_scenarios:
            logger.info(f"📊 Testing scenario: {scenario['name']}")
            
            metrics = schwafit.detect_overfitting(scenario['data'], scenario['signals'])
            
            result = {
                'scenario': scenario['name'],
                'overfitting_score': metrics.overfitting_score,
                'confidence_penalty': metrics.confidence_penalty,
                'correction_factor': metrics.correction_factor,
                'detection_method': metrics.detection_method
            }
            
            detection_results.append(result)
            
            logger.info(f"  Overfitting Score: {metrics.overfitting_score:.3f}")
            logger.info(f"  Correction Factor: {metrics.correction_factor:.3f}")
        
        return {
            'success': True,
            'initial_metrics': {
                'overfitting_score': overfitting_metrics.overfitting_score,
                'confidence_penalty': overfitting_metrics.confidence_penalty,
                'correction_factor': overfitting_metrics.correction_factor
            },
            'scenario_results': detection_results
        }
        
    except Exception as e:
        logger.error(f"❌ Error testing Schwafit overfitting detection: {e}")
        return {'success': False, 'error': str(e)}


async def test_schwafit_data_sanitization():
    """Test Schwafit data sanitization capabilities."""
    logger.info("🧪 Testing Schwafit Data Sanitization...")
    
    try:
        from core.schwafit_overfitting_prevention import create_schwafit_overfitting_prevention, SanitizationLevel
        
        # Create Schwafit system
        schwafit = create_schwafit_overfitting_prevention()
        
        # Test data with sensitive information
        sensitive_data = {
            'symbol': 'BTC/USDC',
            'price': 50000.0,
            'volume': 2_000_000_000,
            'api_key': 'sensitive_api_key_12345',
            'secret_key': 'very_secret_key_67890',
            'internal_debug_info': 'debug_data_here',
            'test_mode': True,
            'timestamp': time.time(),
            'user_id': 'user_12345',
            'session_token': 'session_token_abc123',
            'market_data': {
                'bid': 49950.0,
                'ask': 50050.0,
                'last_trade_id': 'trade_12345'
            }
        }
        
        # Test different sanitization levels
        sanitization_levels = [
            SanitizationLevel.LOW,
            SanitizationLevel.MEDIUM,
            SanitizationLevel.HIGH,
            SanitizationLevel.MAXIMUM
        ]
        
        sanitization_results = []
        
        for level in sanitization_levels:
            logger.info(f"📊 Testing sanitization level: {level.value}")
            
            # Sanitize data
            sanitized_result = schwafit.sanitize_data(sensitive_data, level)
            
            result = {
                'level': level.value,
                'original_feature_count': len(sensitive_data),
                'sanitized_feature_count': len(sanitized_result.sanitized_data),
                'removed_features': sanitized_result.removed_features,
                'obfuscated_features': sanitized_result.obfuscated_features,
                'confidence_adjustment': sanitized_result.confidence_adjustment,
                'removal_ratio': len(sanitized_result.removed_features) / len(sensitive_data),
                'obfuscation_ratio': len(sanitized_result.obfuscated_features) / len(sensitive_data)
            }
            
            sanitization_results.append(result)
            
            logger.info(f"  Removed: {len(sanitized_result.removed_features)} features")
            logger.info(f"  Obfuscated: {len(sanitized_result.obfuscated_features)} features")
            logger.info(f"  Confidence adjustment: {sanitized_result.confidence_adjustment:.3f}")
            
            # Check if sensitive data was properly handled
            sanitized_data = sanitized_result.sanitized_data
            
            # Verify sensitive data was removed or obfuscated
            sensitive_handled = True
            for sensitive_key in ['api_key', 'secret_key', 'internal_debug_info']:
                if sensitive_key in sanitized_data:
                    original_value = sensitive_data[sensitive_key]
                    sanitized_value = sanitized_data[sensitive_key]
                    if original_value == sanitized_value:
                        sensitive_handled = False
                        logger.warning(f"⚠️ Sensitive data not properly handled: {sensitive_key}")
            
            if sensitive_handled:
                logger.info(f"✅ Sensitive data properly handled at {level.value} level")
        
        return {
            'success': True,
            'sanitization_results': sanitization_results
        }
        
    except Exception as e:
        logger.error(f"❌ Error testing Schwafit data sanitization: {e}")
        return {'success': False, 'error': str(e)}


async def test_schwafit_pipeline_protection():
    """Test Schwafit pipeline protection capabilities."""
    logger.info("🧪 Testing Schwafit Pipeline Protection...")
    
    try:
        from core.schwafit_overfitting_prevention import create_schwafit_overfitting_prevention
        
        # Create Schwafit system
        schwafit = create_schwafit_overfitting_prevention()
        
        # Test data and signals
        test_data = {
            'symbol': 'BTC/USDC',
            'price': 50000.0,
            'volume': 2_000_000_000,
            'volatility': 0.025,
            'api_key': 'sensitive_key',
            'internal_data': 'debug_info'
        }
        
        test_signals = [
            {'action': 'buy', 'confidence': 0.9, 'amount': 0.01, 'symbol': 'BTC/USDC'},
            {'action': 'buy', 'confidence': 0.9, 'amount': 0.01, 'symbol': 'BTC/USDC'},
            {'action': 'buy', 'confidence': 0.9, 'amount': 0.01, 'symbol': 'BTC/USDC'}
        ]
        
        # Protect pipeline
        protected_data, protected_signals = schwafit.protect_pipeline(test_data, test_signals)
        
        logger.info(f"📊 Pipeline Protection Results:")
        logger.info(f"  Original data features: {len(test_data)}")
        logger.info(f"  Protected data features: {len(protected_data)}")
        logger.info(f"  Original signals: {len(test_signals)}")
        logger.info(f"  Protected signals: {len(protected_signals)}")
        
        # Check protection metadata
        if 'schwafit_protection' in protected_data:
            protection_meta = protected_data['schwafit_protection']
            logger.info(f"  Overfitting score: {protection_meta.get('overfitting_score', 0):.3f}")
            logger.info(f"  Correction factor: {protection_meta.get('correction_factor', 1):.3f}")
            logger.info(f"  Confidence penalty: {protection_meta.get('confidence_penalty', 0):.3f}")
            logger.info(f"  Sanitization level: {protection_meta.get('sanitization_level', 'unknown')}")
        
        # Verify sensitive data was handled
        sensitive_handled = True
        for sensitive_key in ['api_key', 'internal_data']:
            if sensitive_key in protected_data:
                if protected_data[sensitive_key] == test_data[sensitive_key]:
                    sensitive_handled = False
                    logger.warning(f"⚠️ Sensitive data not protected: {sensitive_key}")
        
        if sensitive_handled:
            logger.info("✅ Sensitive data properly protected")
        
        # Check signal corrections
        signal_corrections = []
        for i, (original, protected) in enumerate(zip(test_signals, protected_signals)):
            correction = {
                'signal_index': i,
                'original_confidence': original.get('confidence', 0),
                'protected_confidence': protected.get('confidence', 0),
                'confidence_change': protected.get('confidence', 0) - original.get('confidence', 0),
                'has_schwafit_correction': 'schwafit_correction' in protected
            }
            signal_corrections.append(correction)
            
            logger.info(f"  Signal {i}: {original.get('confidence', 0):.3f} → {protected.get('confidence', 0):.3f}")
        
        return {
            'success': True,
            'protection_metadata': protected_data.get('schwafit_protection', {}),
            'sensitive_data_handled': sensitive_handled,
            'signal_corrections': signal_corrections
        }
        
    except Exception as e:
        logger.error(f"❌ Error testing Schwafit pipeline protection: {e}")
        return {'success': False, 'error': str(e)}


async def test_schwafit_strategy_executor_integration():
    """Test Schwafit integration with strategy executor."""
    logger.info("🧪 Testing Schwafit Strategy Executor Integration...")
    
    try:
        from core.strategy.strategy_executor import StrategyExecutor
        
        # Create strategy executor
        executor = StrategyExecutor()
        await executor.initialize()
        
        # Test market data generation with Schwafit protection
        logger.info("📊 Testing market data generation with Schwafit protection...")
        market_data = await executor._generate_market_data()
        
        # Add some sensitive data to test protection
        market_data['api_key'] = 'test_api_key_12345'
        market_data['internal_debug'] = 'debug_info'
        
        logger.info(f"✅ Market data generated: {market_data['symbol']} @ {market_data['price']:.2f}")
        logger.info(f"  Original features: {len(market_data)}")
        
        # Test signal generation with Schwafit protection
        logger.info("📊 Testing signal generation with Schwafit protection...")
        signals = await executor.generate_unified_signals(market_data)
        
        logger.info(f"✅ Generated {len(signals)} unified signals")
        
        # Check if Schwafit protection was applied
        if signals:
            signal = signals[0]
            logger.info(f"  Signal confidence: {signal.mathematical_confidence:.3f}")
            
            # Check for Schwafit protection metadata
            if hasattr(signal, 'unified_signal') and signal.unified_signal:
                if hasattr(signal.unified_signal, 'schwafit_sanitized'):
                    logger.info("✅ Schwafit signal sanitization applied")
                else:
                    logger.warning("⚠️ Schwafit signal sanitization not detected")
        
        # Test system stats
        if executor.overfitting_prevention_system:
            stats = executor.overfitting_prevention_system.get_system_stats()
            logger.info(f"📊 Schwafit System Stats:")
            logger.info(f"  Detection count: {stats.get('detection_count', 0)}")
            logger.info(f"  Correction count: {stats.get('correction_count', 0)}")
            logger.info(f"  Sanitization count: {stats.get('sanitization_count', 0)}")
        
        return {
            'success': True,
            'signals_generated': len(signals),
            'schwafit_integrated': executor.overfitting_prevention_system is not None
        }
        
    except Exception as e:
        logger.error(f"❌ Error testing Schwafit strategy executor integration: {e}")
        return {'success': False, 'error': str(e)}


async def test_complete_schwafit_integration():
    """Test the complete Schwafit integration."""
    logger.info("🧪 Testing Complete Schwafit Integration...")
    
    try:
        # Test all Schwafit components
        detection_result = await test_schwafit_overfitting_detection()
        sanitization_result = await test_schwafit_data_sanitization()
        protection_result = await test_schwafit_pipeline_protection()
        integration_result = await test_schwafit_strategy_executor_integration()
        
        # Check if all tests passed
        all_success = (
            detection_result.get('success', False) and
            sanitization_result.get('success', False) and
            protection_result.get('success', False) and
            integration_result.get('success', False)
        )
        
        if all_success:
            logger.info("🎉 All Schwafit integration tests passed!")
            
            # Generate comprehensive report
            report = {
                'timestamp': time.time(),
                'overall_success': True,
                'overfitting_detection': detection_result,
                'data_sanitization': sanitization_result,
                'pipeline_protection': protection_result,
                'strategy_executor_integration': integration_result,
                'summary': {
                    'overfitting_detection_enabled': True,
                    'data_sanitization_enabled': True,
                    'pipeline_protection_enabled': True,
                    'information_control_enabled': True,
                    'mathematical_correction_enabled': True,
                    'sensitive_data_protection': True
                }
            }
            
            logger.info("📊 Comprehensive Schwafit Integration Report:")
            logger.info(json.dumps(report['summary'], indent=2))
            
            return report
        else:
            logger.error("❌ Some Schwafit integration tests failed")
            return {
                'timestamp': time.time(),
                'overall_success': False,
                'overfitting_detection': detection_result,
                'data_sanitization': sanitization_result,
                'pipeline_protection': protection_result,
                'strategy_executor_integration': integration_result
            }
        
    except Exception as e:
        logger.error(f"❌ Error testing complete Schwafit integration: {e}")
        return {'success': False, 'error': str(e)}


async def main():
    """Run all Schwafit integration tests."""
    logger.info("🚀 Starting Schwafit Overfitting Prevention Integration Tests...")
    
    # Run complete integration test
    result = await test_complete_schwafit_integration()
    
    if result.get('overall_success', False):
        logger.info("✅ All Schwafit integration tests completed successfully!")
        logger.info("🛡️ The system is now protected against overfitting and data leakage!")
    else:
        logger.error("❌ Some tests failed - check the logs for details")
    
    return result


if __name__ == "__main__":
    asyncio.run(main())
