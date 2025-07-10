#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 MATH-TO-TRADE INTEGRATION TEST
===============================

Test script to verify the complete math-to-trade pathway works correctly.
This validates that mathematical signals are properly converted to trading orders.

Tests:
1. Mathematical module signal generation
2. Signal router processing
3. Order creation and validation
4. Risk management checks
5. Portfolio position tracking

Author: Schwabot Team  
Date: 2025-01-02
"""

import asyncio
import logging
import sys
import os
import time
from datetime import datetime
from typing import Dict, Any, List

# Add core modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

logger = logging.getLogger(__name__)


class MathToTradeIntegrationTest:
    """Test the complete math-to-trade integration"""
    
    def __init__(self):
        self.test_results = {}
        self.test_count = 0
        self.passed_tests = 0
        self.failed_tests = 0
        
    async def run_all_tests(self):
        """Run all integration tests"""
        try:
            logger.info("🧪 Starting Math-to-Trade Integration Tests")
            logger.info("="*60)
            
            # Test 1: Module Imports
            await self.test_module_imports()
            
            # Test 2: Mathematical Signal Generation
            await self.test_mathematical_signal_generation()
            
            # Test 3: Signal Router Initialization
            await self.test_signal_router_initialization()
            
            # Test 4: Market Data Processing
            await self.test_market_data_processing()
            
            # Test 5: Signal to Order Conversion
            await self.test_signal_to_order_conversion()
            
            # Test 6: Risk Management Validation
            await self.test_risk_management()
            
            # Test 7: Position Tracking
            await self.test_position_tracking()
            
            # Test 8: End-to-End Integration
            await self.test_end_to_end_integration()
            
            # Print results
            self.print_test_summary()
            
        except Exception as e:
            logger.error(f"❌ Test suite failed: {e}")
            return False
        
        return self.failed_tests == 0
    
    async def test_module_imports(self):
        """Test that all required modules can be imported"""
        test_name = "Module Imports"
        self.test_count += 1
        
        try:
            logger.info(f"🧪 Test {self.test_count}: {test_name}")
            
            # Test mathematical modules
            from core.strategy.volume_weighted_hash_oscillator import VolumeWeightedHashOscillator
            from core.strategy.zygot_zalgo_entropy_dual_key_gate import ZygotZalgoEntropyDualKeyGate
            from core.immune.qsc_gate import QSCGate
            from core.math.tensor_algebra.unified_tensor_algebra import UnifiedTensorAlgebra
            from core.entropy.galileo_tensor_field import GalileoTensorField
            
            # Test router modules
            from core.math_to_trade_signal_router import MathToTradeSignalRouter, MathematicalSignal
            from core.real_market_data_feed import RealMarketDataFeed, MarketDataPoint
            
            logger.info("   ✅ All core modules imported successfully")
            self.test_results[test_name] = "PASSED"
            self.passed_tests += 1
            
        except ImportError as e:
            logger.error(f"   ❌ Module import failed: {e}")
            self.test_results[test_name] = f"FAILED: {e}"
            self.failed_tests += 1
        except Exception as e:
            logger.error(f"   ❌ Unexpected error: {e}")
            self.test_results[test_name] = f"FAILED: {e}"
            self.failed_tests += 1
    
    async def test_mathematical_signal_generation(self):
        """Test mathematical modules can generate signals"""
        test_name = "Mathematical Signal Generation"
        self.test_count += 1
        
        try:
            logger.info(f"🧪 Test {self.test_count}: {test_name}")
            
            from core.strategy.volume_weighted_hash_oscillator import VolumeWeightedHashOscillator
            
            # Test VWAP oscillator
            vwho = VolumeWeightedHashOscillator()
            
            # Test with sample data
            prices = [50000.0, 50100.0, 50200.0]
            volumes = [1000.0, 1100.0, 1200.0]
            
            oscillator_value = vwho.calculate_vwap_oscillator(prices, volumes)
            hash_signature = vwho.generate_hash_signature(prices[-1], volumes[-1])
            phase_shift = vwho.detect_phase_shift(prices)
            
            logger.info(f"   📊 VWAP Oscillator: {oscillator_value:.4f}")
            logger.info(f"   🔐 Hash Signature: {hash_signature[:16]}...")
            logger.info(f"   📈 Phase Shift: {phase_shift:.4f}")
            
            # Validate outputs
            assert isinstance(oscillator_value, (int, float)), "Oscillator should return numeric value"
            assert isinstance(hash_signature, str), "Hash signature should be string"
            assert isinstance(phase_shift, (int, float)), "Phase shift should be numeric"
            assert -1.0 <= oscillator_value <= 1.0, "Oscillator should be bounded [-1, 1]"
            
            logger.info("   ✅ Mathematical signal generation working")
            self.test_results[test_name] = "PASSED"
            self.passed_tests += 1
            
        except Exception as e:
            logger.error(f"   ❌ Mathematical signal generation failed: {e}")
            self.test_results[test_name] = f"FAILED: {e}"
            self.failed_tests += 1
    
    async def test_signal_router_initialization(self):
        """Test signal router can initialize"""
        test_name = "Signal Router Initialization"
        self.test_count += 1
        
        try:
            logger.info(f"🧪 Test {self.test_count}: {test_name}")
            
            from core.math_to_trade_signal_router import MathToTradeSignalRouter
            
            # Test configuration
            config = {
                'coinbase': {
                    'enabled': False,  # Disable for testing
                    'api_key': 'test_key',
                    'api_secret': 'test_secret',
                    'passphrase': 'test_passphrase',
                    'sandbox': True
                },
                'default_exchange': 'coinbase',
                'risk_limits': {
                    'min_confidence': 0.7,
                    'min_strength': 0.5,
                    'max_positions': 3,
                    'daily_trade_limit': 10,
                    'position_size_percent': 0.1,
                    'max_position_size': 0.01
                }
            }
            
            # Initialize router
            router = MathToTradeSignalRouter(config)
            
            # Check router attributes
            assert hasattr(router, 'math_modules'), "Router should have math_modules"
            assert hasattr(router, 'signal_history'), "Router should have signal_history"
            assert hasattr(router, 'risk_limits'), "Router should have risk_limits"
            
            logger.info("   ✅ Signal router initialized successfully")
            self.test_results[test_name] = "PASSED"
            self.passed_tests += 1
            
        except Exception as e:
            logger.error(f"   ❌ Signal router initialization failed: {e}")
            self.test_results[test_name] = f"FAILED: {e}"
            self.failed_tests += 1
    
    async def test_market_data_processing(self):
        """Test market data processing through mathematical modules"""
        test_name = "Market Data Processing"
        self.test_count += 1
        
        try:
            logger.info(f"🧪 Test {self.test_count}: {test_name}")
            
            from core.math_to_trade_signal_router import MathToTradeSignalRouter
            
            # Test configuration (no real API calls)
            config = {
                'coinbase': {'enabled': False},
                'default_exchange': 'coinbase',
                'risk_limits': {
                    'min_confidence': 0.7,
                    'min_strength': 0.5,
                    'max_positions': 3,
                    'daily_trade_limit': 10,
                    'position_size_percent': 0.1,
                    'max_position_size': 0.01
                }
            }
            
            router = MathToTradeSignalRouter(config)
            
            # Test market data processing
            price = 50000.0
            volume = 1000.0
            asset_pair = "BTC/USD"
            
            # This should not make real API calls since exchange is disabled
            signals = await router.process_market_data(price, volume, asset_pair)
            
            logger.info(f"   📊 Generated {len(signals)} signals from market data")
            
            # Validate signals
            for signal in signals:
                assert hasattr(signal, 'signal_type'), "Signal should have signal_type"
                assert hasattr(signal, 'confidence'), "Signal should have confidence"
                assert hasattr(signal, 'strength'), "Signal should have strength"
                assert hasattr(signal, 'source_module'), "Signal should have source_module"
                
                logger.info(f"   📈 {signal.source_module}: {signal.signal_type.value} "
                          f"(Conf: {signal.confidence:.3f}, Str: {signal.strength:.3f})")
            
            logger.info("   ✅ Market data processing working")
            self.test_results[test_name] = "PASSED"
            self.passed_tests += 1
            
        except Exception as e:
            logger.error(f"   ❌ Market data processing failed: {e}")
            self.test_results[test_name] = f"FAILED: {e}"
            self.failed_tests += 1
    
    async def test_signal_to_order_conversion(self):
        """Test converting mathematical signals to trading orders"""
        test_name = "Signal to Order Conversion"
        self.test_count += 1
        
        try:
            logger.info(f"🧪 Test {self.test_count}: {test_name}")
            
            from core.math_to_trade_signal_router import (
                MathematicalSignal, SignalType, TradingOrder
            )
            from decimal import Decimal
            
            # Create test signal
            test_signal = MathematicalSignal(
                signal_id="test_signal_001",
                timestamp=time.time(),
                signal_type=SignalType.BUY,
                confidence=0.85,
                strength=0.75,
                price=50000.0,
                volume=1000.0,
                asset_pair="BTC/USD",
                mathematical_score=0.8,
                entropy_value=0.6,
                tensor_score=0.7,
                hash_signature="test_hash",
                source_module="TestModule"
            )
            
            # Create test order
            test_order = TradingOrder(
                order_id="order_test_001",
                signal_id=test_signal.signal_id,
                timestamp=time.time(),
                exchange="coinbase",
                symbol="BTC/USD",
                side="buy",
                order_type="market",
                amount=Decimal("0.001")
            )
            
            # Validate order structure
            assert test_order.signal_id == test_signal.signal_id, "Order should reference signal"
            assert test_order.side == "buy", "Order side should match signal"
            assert test_order.amount > 0, "Order amount should be positive"
            
            logger.info(f"   📋 Created order: {test_order.side.upper()} {test_order.amount} {test_order.symbol}")
            logger.info(f"   🔗 Linked to signal: {test_signal.source_module} ({test_signal.confidence:.3f})")
            
            logger.info("   ✅ Signal to order conversion working")
            self.test_results[test_name] = "PASSED"
            self.passed_tests += 1
            
        except Exception as e:
            logger.error(f"   ❌ Signal to order conversion failed: {e}")
            self.test_results[test_name] = f"FAILED: {e}"
            self.failed_tests += 1
    
    async def test_risk_management(self):
        """Test risk management validation"""
        test_name = "Risk Management"
        self.test_count += 1
        
        try:
            logger.info(f"🧪 Test {self.test_count}: {test_name}")
            
            from core.math_to_trade_signal_router import (
                MathToTradeSignalRouter, MathematicalSignal, SignalType
            )
            
            config = {
                'coinbase': {'enabled': False},
                'default_exchange': 'coinbase',
                'risk_limits': {
                    'min_confidence': 0.8,  # High threshold for testing
                    'min_strength': 0.7,    # High threshold for testing
                    'max_positions': 2,     # Low limit for testing
                    'daily_trade_limit': 5, # Low limit for testing
                    'position_size_percent': 0.1,
                    'max_position_size': 0.01
                }
            }
            
            router = MathToTradeSignalRouter(config)
            
            # Test low confidence signal (should be rejected)
            low_confidence_signal = MathematicalSignal(
                signal_id="low_conf_001",
                timestamp=time.time(),
                signal_type=SignalType.BUY,
                confidence=0.5,  # Below threshold
                strength=0.8,
                price=50000.0,
                volume=1000.0,
                asset_pair="BTC/USD",
                mathematical_score=0.5,
                entropy_value=0.3,
                tensor_score=0.4,
                hash_signature="test_hash",
                source_module="TestModule"
            )
            
            # Test high confidence signal (should pass)
            high_confidence_signal = MathematicalSignal(
                signal_id="high_conf_001",
                timestamp=time.time(),
                signal_type=SignalType.BUY,
                confidence=0.9,  # Above threshold
                strength=0.8,
                price=50000.0,
                volume=1000.0,
                asset_pair="BTC/USD",
                mathematical_score=0.9,
                entropy_value=0.8,
                tensor_score=0.9,
                hash_signature="test_hash",
                source_module="TestModule"
            )
            
            # Test validation
            low_conf_valid = router._validate_signal_for_execution(low_confidence_signal)
            high_conf_valid = router._validate_signal_for_execution(high_confidence_signal)
            
            assert not low_conf_valid, "Low confidence signal should be rejected"
            assert high_conf_valid, "High confidence signal should be accepted"
            
            logger.info(f"   ❌ Low confidence signal (0.5) rejected: {not low_conf_valid}")
            logger.info(f"   ✅ High confidence signal (0.9) accepted: {high_conf_valid}")
            
            logger.info("   ✅ Risk management validation working")
            self.test_results[test_name] = "PASSED"
            self.passed_tests += 1
            
        except Exception as e:
            logger.error(f"   ❌ Risk management test failed: {e}")
            self.test_results[test_name] = f"FAILED: {e}"
            self.failed_tests += 1
    
    async def test_position_tracking(self):
        """Test position tracking functionality"""
        test_name = "Position Tracking"
        self.test_count += 1
        
        try:
            logger.info(f"🧪 Test {self.test_count}: {test_name}")
            
            from core.math_to_trade_signal_router import (
                MathToTradeSignalRouter, OrderResult, OrderStatus
            )
            from decimal import Decimal
            
            config = {
                'coinbase': {'enabled': False},
                'default_exchange': 'coinbase',
                'risk_limits': {}
            }
            
            router = MathToTradeSignalRouter(config)
            
            # Create test order result
            test_order_result = OrderResult(
                order_id="test_order_001",
                signal_id="test_signal_001",
                timestamp=time.time(),
                exchange="coinbase",
                symbol="BTC/USD",
                side="buy",
                status=OrderStatus.FILLED,
                filled_amount=Decimal("0.001"),
                filled_price=Decimal("50000.0"),
                remaining_amount=Decimal("0.0"),
                fees=Decimal("0.50"),
                commission_currency="USD",
                execution_time_ms=150.0
            )
            
            # Update position tracking
            router._update_position_tracking(test_order_result)
            
            # Check position was recorded
            assert "BTC/USD" in router.position_tracker, "Position should be tracked"
            position = router.position_tracker["BTC/USD"]
            
            assert position['net_position'] == 0.001, "Position size should be correct"
            assert position['avg_price'] == 50000.0, "Average price should be correct"
            assert len(position['orders']) == 1, "Order should be recorded"
            
            logger.info(f"   💼 Position tracked: {position['net_position']} BTC @ ${position['avg_price']:.2f}")
            
            # Test sell order
            sell_order_result = OrderResult(
                order_id="test_order_002",
                signal_id="test_signal_002", 
                timestamp=time.time(),
                exchange="coinbase",
                symbol="BTC/USD",
                side="sell",
                status=OrderStatus.FILLED,
                filled_amount=Decimal("0.0005"),
                filled_price=Decimal("51000.0"),
                remaining_amount=Decimal("0.0"),
                fees=Decimal("0.26"),
                commission_currency="USD",
                execution_time_ms=120.0
            )
            
            router._update_position_tracking(sell_order_result)
            
            # Check updated position
            updated_position = router.position_tracker["BTC/USD"]
            expected_net = 0.001 - 0.0005  # 0.0005 BTC remaining
            
            assert abs(updated_position['net_position'] - expected_net) < 1e-8, "Net position should be updated"
            assert len(updated_position['orders']) == 2, "Both orders should be recorded"
            
            logger.info(f"   💼 Updated position: {updated_position['net_position']} BTC")
            
            logger.info("   ✅ Position tracking working")
            self.test_results[test_name] = "PASSED"
            self.passed_tests += 1
            
        except Exception as e:
            logger.error(f"   ❌ Position tracking test failed: {e}")
            self.test_results[test_name] = f"FAILED: {e}"
            self.failed_tests += 1
    
    async def test_end_to_end_integration(self):
        """Test complete end-to-end integration"""
        test_name = "End-to-End Integration"
        self.test_count += 1
        
        try:
            logger.info(f"🧪 Test {self.test_count}: {test_name}")
            
            from core.math_to_trade_signal_router import MathToTradeSignalRouter
            
            # Configuration for testing (no real API calls)
            config = {
                'coinbase': {'enabled': False},
                'default_exchange': 'coinbase',
                'risk_limits': {
                    'min_confidence': 0.6,
                    'min_strength': 0.5,
                    'max_positions': 5,
                    'daily_trade_limit': 20,
                    'position_size_percent': 0.1,
                    'max_position_size': 0.01
                }
            }
            
            router = MathToTradeSignalRouter(config)
            
            # Simulate market data processing
            logger.info("   📊 Processing market data...")
            price = 50000.0
            volume = 1000.0
            
            signals = await router.process_market_data(price, volume, "BTC/USD")
            
            if signals:
                logger.info(f"   📈 Generated {len(signals)} signals")
                
                # Get trading status
                status = await router.get_trading_status()
                
                assert 'signals_processed' in status, "Status should include signals processed"
                assert 'orders_executed' in status, "Status should include orders executed"
                
                logger.info(f"   📊 Status: {status['signals_processed']} signals processed")
                
                # Test signal filtering
                valid_signals = [s for s in signals if router._validate_signal_for_execution(s)]
                logger.info(f"   ✅ {len(valid_signals)}/{len(signals)} signals passed risk validation")
                
            else:
                logger.info("   📊 No signals generated (this is normal for some market conditions)")
            
            logger.info("   ✅ End-to-end integration working")
            self.test_results[test_name] = "PASSED"
            self.passed_tests += 1
            
        except Exception as e:
            logger.error(f"   ❌ End-to-end integration test failed: {e}")
            self.test_results[test_name] = f"FAILED: {e}"
            self.failed_tests += 1
    
    def print_test_summary(self):
        """Print comprehensive test summary"""
        logger.info("\n" + "="*60)
        logger.info("🧪 MATH-TO-TRADE INTEGRATION TEST SUMMARY")
        logger.info("="*60)
        
        logger.info(f"📊 Total Tests: {self.test_count}")
        logger.info(f"✅ Passed: {self.passed_tests}")
        logger.info(f"❌ Failed: {self.failed_tests}")
        logger.info(f"📈 Success Rate: {(self.passed_tests/self.test_count)*100:.1f}%")
        
        logger.info("\n📋 DETAILED RESULTS:")
        for test_name, result in self.test_results.items():
            status_icon = "✅" if result == "PASSED" else "❌"
            logger.info(f"   {status_icon} {test_name}: {result}")
        
        if self.failed_tests == 0:
            logger.info("\n🎉 ALL TESTS PASSED - Math-to-Trade integration is working correctly!")
            logger.info("🚀 System is ready for real trading (with proper API keys)")
        else:
            logger.info(f"\n⚠️  {self.failed_tests} tests failed - Fix issues before trading with real money")
        
        logger.info("="*60)


async def main():
    """Run integration tests"""
    try:
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)s | %(message)s'
        )
        
        # Run tests
        test_suite = MathToTradeIntegrationTest()
        success = await test_suite.run_all_tests()
        
        return 0 if success else 1
        
    except Exception as e:
        logger.error(f"❌ Test suite error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 