"""
Simple Trading Engine Test
Basic integration test for the enhanced trading engine.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.trading_engine_integration import (
    TradeSignal, 
    TradeExecution, 
    generate_trade_signal,
    ValidationError,
    TradingError
)
from core.unified_trade_router import UnifiedTradeRouter
from core.clean_unified_math import clean_unified_math

def test_basic_signal_generation():
    """Test basic signal generation."""
    print("🧪 Testing Basic Signal Generation...")
    
    try:
        # Test valid signal generation
        signal = generate_trade_signal(
            asset="BTC/USDT",
            price=50000.0,
            volume=1.0,
            metadata={"test": True}
        )
        
        print(f"✅ Signal generated successfully:")
        print(f"  ID: {signal.id}")
        print(f"  Asset: {signal.asset}")
        print(f"  Price: ${signal.price:,.2f}")
        print(f"  Volume: {signal.volume}")
        print(f"  Mathematical Score: {signal.mathematical_score:.4f}")
        print(f"  Confidence: {signal.confidence:.2f}")
        print(f"  Order Side: {signal.order_side.value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Signal generation failed: {e}")
        return False

def test_signal_validation():
    """Test signal validation with invalid inputs."""
    print("\n🧪 Testing Signal Validation...")
    
    # Test invalid price
    try:
        signal = generate_trade_signal(
            asset="BTC/USDT",
            price=-100.0,  # Invalid negative price
            volume=1.0
        )
        print("❌ Should have failed with negative price")
        return False
    except ValidationError:
        print("✅ Correctly caught negative price error")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
    
    # Test invalid volume
    try:
        signal = generate_trade_signal(
            asset="BTC/USDT",
            price=50000.0,
            volume=0.0  # Invalid zero volume
        )
        print("❌ Should have failed with zero volume")
        return False
    except ValidationError:
        print("✅ Correctly caught zero volume error")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
    
    return True

def test_execution_creation():
    """Test execution creation."""
    print("\n🧪 Testing Execution Creation...")
    
    try:
        # Create a signal first
        signal = generate_trade_signal(
            asset="BTC/USDT",
            price=50000.0,
            volume=1.0
        )
        
        # Create execution
        execution = TradeExecution(
            signal_id=signal.id,
            asset=signal.asset,
            execution_price=50100.0,  # Slight slippage
            volume=signal.volume,
            latency=0.05,
            order_type=signal.order_type,
            order_side=signal.order_side
        )
        
        # Calculate performance
        execution.calculate_performance(entry_price=signal.price)
        
        print(f"✅ Execution created successfully:")
        print(f"  ID: {execution.id}")
        print(f"  Signal ID: {execution.signal_id}")
        print(f"  Execution Price: ${execution.execution_price:,.2f}")
        print(f"  Realized Profit: ${execution.realized_profit:,.2f}")
        print(f"  Performance Score: {execution.performance_score:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Execution creation failed: {e}")
        return False

def test_mathematical_integration():
    """Test mathematical integration."""
    print("\n🧪 Testing Mathematical Integration...")
    
    try:
        # Test different price scenarios
        scenarios = [
            {"price": 55000, "volume": 1.5, "description": "High price, high volume"},
            {"price": 45000, "volume": 0.5, "description": "Low price, low volume"},
            {"price": 50000, "volume": 1.0, "description": "Medium price, medium volume"},
        ]
        
        for scenario in scenarios:
            signal = generate_trade_signal(
                asset="BTC/USDT",
                price=scenario["price"],
                volume=scenario["volume"]
            )
            
            print(f"  {scenario['description']}:")
            print(f"    Mathematical Score: {signal.mathematical_score:.4f}")
            print(f"    Risk Score: {signal.risk_score:.4f}")
            print(f"    Signal Strength: {signal.signal_strength:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Mathematical integration failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting Simple Trading Engine Tests...")
    print("=" * 50)
    
    tests = [
        test_basic_signal_generation,
        test_signal_validation,
        test_execution_creation,
        test_mathematical_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("🎉 All tests passed! Trading engine integration is working correctly.")
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 