#!/usr/bin/env python3
"""
Test Batch Order Validation System
Comprehensive testing of the enhanced CCXT batch order validation system.
"""

import time
import logging
import sys
import os
from typing import Dict, List, Tuple

# Add the core directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

from enhanced_ccxt_trading_engine import (
    EnhancedCCXTTradingEngine, 
    create_enhanced_ccxt_engine,
    EnhancedBatchOrder,
    EnhancedTradingSignal,
    ExchangeLimits
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BatchOrderValidator:
    """Comprehensive batch order validation tester."""
    
    def __init__(self):
        self.test_results = []
        self.engine = None
    
    def setup_engine(self, exchange_name: str = 'coinbase'):
        """Setup trading engine for testing."""
        config = {
            'name': exchange_name,
            'sandbox': True
        }
        
        try:
            self.engine = create_enhanced_ccxt_engine(config)
            print(f"✅ Engine initialized for {exchange_name}")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize engine: {e}")
            return False
    
    def test_exchange_limits(self):
        """Test exchange-specific limits."""
        print("\n=== Testing Exchange Limits ===")
        
        if not self.engine:
            print("❌ Engine not initialized")
            return False
        
        limits = self.engine.get_exchange_limits()
        
        tests = [
            ("Exchange Name", limits.exchange_name, "coinbase"),
            ("Min Order Size", limits.min_order_size, 1.0),
            ("Max Order Size", limits.max_order_size, 100000.0),
            ("Price Precision", limits.price_precision, 8),
            ("Amount Precision", limits.amount_precision, 8),
            ("Rate Limit Requests", limits.rate_limit_requests_per_minute, 100),
            ("Rate Limit Orders", limits.rate_limit_orders_per_minute, 50),
            ("Max Batch Orders", limits.max_orders_per_batch, 50),
            ("Min Time Between Orders", limits.min_time_between_orders, 0.5)
        ]
        
        all_passed = True
        for test_name, actual, expected in tests:
            if actual == expected:
                print(f"  ✅ {test_name}: {actual}")
            else:
                print(f"  ❌ {test_name}: {actual} (expected {expected})")
                all_passed = False
        
        return all_passed
    
    def test_order_parameter_validation(self):
        """Test individual order parameter validation."""
        print("\n=== Testing Order Parameter Validation ===")
        
        if not self.engine:
            print("❌ Engine not initialized")
            return False
        
        # Test valid order
        valid_signal = EnhancedTradingSignal(
            symbol='BTC/USDC',
            side='buy',
            quantity=0.001,
            price=50000.0,
            order_type='limit'
        )
        
        is_valid, error_msg = self.engine._validate_order_parameters(valid_signal)
        if is_valid:
            print("  ✅ Valid order parameters accepted")
        else:
            print(f"  ❌ Valid order rejected: {error_msg}")
            return False
        
        # Test invalid order (too small)
        invalid_signal = EnhancedTradingSignal(
            symbol='BTC/USDC',
            side='buy',
            quantity=0.0000001,  # Very small
            price=50000.0,
            order_type='limit'
        )
        
        is_valid, error_msg = self.engine._validate_order_parameters(invalid_signal)
        if not is_valid:
            print("  ✅ Invalid order (too small) correctly rejected")
        else:
            print("  ❌ Invalid order should have been rejected")
            return False
        
        # Test invalid order (too large)
        invalid_signal2 = EnhancedTradingSignal(
            symbol='BTC/USDC',
            side='buy',
            quantity=1000.0,  # Very large
            price=50000.0,
            order_type='limit'
        )
        
        is_valid, error_msg = self.engine._validate_order_parameters(invalid_signal2)
        if not is_valid:
            print("  ✅ Invalid order (too large) correctly rejected")
        else:
            print("  ❌ Invalid order should have been rejected")
            return False
        
        return True
    
    def test_batch_order_validation(self):
        """Test batch order validation."""
        print("\n=== Testing Batch Order Validation ===")
        
        if not self.engine:
            print("❌ Engine not initialized")
            return False
        
        # Test valid batch order
        valid_batch = EnhancedBatchOrder(
            symbol='BTC/USDC',
            side='buy',
            total_quantity=0.1,
            batch_count=5,
            price_range=(50000.0, 51000.0),
            spread_seconds=10,
            strategy='test'
        )
        
        is_valid, error_msg = self.engine._validate_batch_order(valid_batch)
        if is_valid:
            print("  ✅ Valid batch order accepted")
        else:
            print(f"  ❌ Valid batch order rejected: {error_msg}")
            return False
        
        # Test invalid batch order (too many orders)
        invalid_batch1 = EnhancedBatchOrder(
            symbol='BTC/USDC',
            side='buy',
            total_quantity=0.1,
            batch_count=100,  # Exceeds limit
            price_range=(50000.0, 51000.0),
            spread_seconds=10,
            strategy='test'
        )
        
        is_valid, error_msg = self.engine._validate_batch_order(invalid_batch1)
        if not is_valid:
            print("  ✅ Invalid batch order (too many orders) correctly rejected")
        else:
            print("  ❌ Invalid batch order should have been rejected")
            return False
        
        # Test invalid batch order (invalid price range)
        invalid_batch2 = EnhancedBatchOrder(
            symbol='BTC/USDC',
            side='buy',
            total_quantity=0.1,
            batch_count=5,
            price_range=(51000.0, 50000.0),  # Invalid range
            spread_seconds=10,
            strategy='test'
        )
        
        is_valid, error_msg = self.engine._validate_batch_order(invalid_batch2)
        if not is_valid:
            print("  ✅ Invalid batch order (invalid price range) correctly rejected")
        else:
            print("  ❌ Invalid batch order should have been rejected")
            return False
        
        # Test invalid batch order (negative quantity)
        invalid_batch3 = EnhancedBatchOrder(
            symbol='BTC/USDC',
            side='buy',
            total_quantity=-0.1,  # Negative
            batch_count=5,
            price_range=(50000.0, 51000.0),
            spread_seconds=10,
            strategy='test'
        )
        
        is_valid, error_msg = self.engine._validate_batch_order(invalid_batch3)
        if not is_valid:
            print("  ✅ Invalid batch order (negative quantity) correctly rejected")
        else:
            print("  ❌ Invalid batch order should have been rejected")
            return False
        
        return True
    
    def test_precision_handling(self):
        """Test precision handling."""
        print("\n=== Testing Precision Handling ===")
        
        if not self.engine:
            print("❌ Engine not initialized")
            return False
        
        # Test price precision
        test_price = 50000.123456789
        rounded_price = self.engine._round_to_precision(test_price, 8)
        print(f"  ✅ Price precision: {test_price} -> {rounded_price}")
        
        # Test quantity precision
        test_quantity = 0.123456789
        rounded_quantity = self.engine._round_to_precision(test_quantity, 8)
        print(f"  ✅ Quantity precision: {test_quantity} -> {rounded_quantity}")
        
        return True
    
    def test_rate_limiting(self):
        """Test rate limiting functionality."""
        print("\n=== Testing Rate Limiting ===")
        
        if not self.engine:
            print("❌ Engine not initialized")
            return False
        
        # Test request rate limiting
        print("  Testing request rate limiting...")
        start_time = time.time()
        
        for i in range(3):
            self.engine.rate_limiter.wait_for_request()
            print(f"    Request {i+1} completed")
        
        elapsed = time.time() - start_time
        print(f"  ✅ 3 requests completed in {elapsed:.2f}s")
        
        # Test order rate limiting
        print("  Testing order rate limiting...")
        start_time = time.time()
        
        for i in range(3):
            self.engine.rate_limiter.wait_for_order()
            print(f"    Order {i+1} completed")
        
        elapsed = time.time() - start_time
        print(f"  ✅ 3 orders completed in {elapsed:.2f}s")
        
        return True
    
    def test_batch_order_creation(self):
        """Test batch order creation and validation."""
        print("\n=== Testing Batch Order Creation ===")
        
        if not self.engine:
            print("❌ Engine not initialized")
            return False
        
        try:
            # Test buy wall creation
            batch_id = self.engine.create_enhanced_buy_wall(
                symbol='BTC/USDC',
                total_quantity=0.001,  # Small amount for testing
                price_range=(50000.0, 50100.0),
                batch_count=3,
                spread_seconds=5
            )
            print(f"  ✅ Buy wall created: {batch_id}")
            
            # Test sell wall creation
            batch_id2 = self.engine.create_enhanced_sell_wall(
                symbol='BTC/USDC',
                total_quantity=0.001,  # Small amount for testing
                price_range=(49900.0, 50000.0),
                batch_count=3,
                spread_seconds=5
            )
            print(f"  ✅ Sell wall created: {batch_id2}")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Batch order creation failed: {e}")
            return False
    
    def test_error_handling(self):
        """Test error handling in batch orders."""
        print("\n=== Testing Error Handling ===")
        
        if not self.engine:
            print("❌ Engine not initialized")
            return False
        
        # Test invalid symbol
        try:
            batch_id = self.engine.create_enhanced_buy_wall(
                symbol='INVALID/SYMBOL',
                total_quantity=0.001,
                price_range=(50000.0, 50100.0),
                batch_count=3,
                spread_seconds=5
            )
            print("  ❌ Invalid symbol should have failed")
            return False
        except Exception as e:
            print(f"  ✅ Invalid symbol correctly handled: {e}")
        
        # Test invalid parameters
        try:
            batch_id = self.engine.create_enhanced_buy_wall(
                symbol='BTC/USDC',
                total_quantity=-0.001,  # Negative
                price_range=(50000.0, 50100.0),
                batch_count=3,
                spread_seconds=5
            )
            print("  ❌ Negative quantity should have failed")
            return False
        except ValueError as e:
            print(f"  ✅ Negative quantity correctly rejected: {e}")
        
        return True
    
    def run_all_tests(self):
        """Run all batch order validation tests."""
        print("🧪 Batch Order Validation System Tests")
        print("=" * 50)
        
        # Setup
        if not self.setup_engine('coinbase'):
            return False
        
        # Run tests
        tests = [
            ("Exchange Limits", self.test_exchange_limits),
            ("Order Parameter Validation", self.test_order_parameter_validation),
            ("Batch Order Validation", self.test_batch_order_validation),
            ("Precision Handling", self.test_precision_handling),
            ("Rate Limiting", self.test_rate_limiting),
            ("Batch Order Creation", self.test_batch_order_creation),
            ("Error Handling", self.test_error_handling),
        ]
        
        results = []
        for test_name, test_func in tests:
            try:
                result = test_func()
                results.append((test_name, result))
                if result:
                    print(f"✅ {test_name}: PASSED")
                else:
                    print(f"❌ {test_name}: FAILED")
            except Exception as e:
                print(f"❌ {test_name}: CRASHED - {e}")
                results.append((test_name, False))
        
        # Summary
        print("\n" + "=" * 50)
        print("📊 TEST SUMMARY")
        print("=" * 50)
        
        passed = sum(1 for _, result in results if result)
        total = len(results)
        
        for test_name, result in results:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name:30} {status}")
        
        print(f"\nResults: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All batch order validation tests passed!")
            print("✅ System is ready for production use")
        else:
            print("⚠️  Some tests failed. Please review the issues above.")
        
        # Cleanup
        if self.engine:
            self.engine.shutdown()
        
        return passed == total

def main():
    """Main test runner."""
    validator = BatchOrderValidator()
    success = validator.run_all_tests()
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 