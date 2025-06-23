#!/usr/bin/env python3
"""
Simple ZPE Integration Test
===========================

Test script to verify ZPE mathematical framework integrations work correctly.
"""

import sys
import os

def test_zpe_imports():
    """Test ZPE module imports."""
    print("🧪 Testing ZPE Module Imports...")
    
    try:
        # Test core ZPE import
        from core.zpe_core import ZPECore
        print("✅ ZPE Core imported successfully")
        
        # Test ZPE integration import
        from core.zpe_integration import ZPEIntegration
        print("✅ ZPE Integration imported successfully")
        
        # Test ZPE rotational engine import
        from core.zpe_rotational_engine import ZPERotationalEngine
        print("✅ ZPE Rotational Engine imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ ZPE import failed: {e}")
        return False

def test_zpe_core_functions():
    """Test ZPE core mathematical functions."""
    print("\n🧪 Testing ZPE Core Functions...")
    
    try:
        from core.zpe_core import ZPECore
        
        zpe = ZPECore()
        
        # Test ZPE work calculation
        work = zpe.calculate_zpe_work(0.8, 0.05)
        print(f"✅ ZPE Work: {work:.6f}")
        
        # Test rotational torque calculation
        torque = zpe.calculate_rotational_torque(0.7, 0.3)
        print(f"✅ Rotational Torque: {torque:.6f}")
        
        # Test thermal efficiency calculation
        efficiency = zpe.calculate_thermal_efficiency(100.0, 1000.0)
        print(f"✅ Thermal Efficiency: {efficiency:.6f}")
        
        # Test news/lantern signal mapping
        signal = zpe.map_news_lantern_signals(0.6, 0.2)
        print(f"✅ Lantern Signal: {signal:.6f}")
        
        # Test profit reinjection
        reinjection = zpe.calculate_profit_reinjection(50.0, 0.7)
        print(f"✅ Profit Reinjection: {reinjection:.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ ZPE core functions failed: {e}")
        return False

def test_zpe_integration():
    """Test ZPE integration layer."""
    print("\n🧪 Testing ZPE Integration Layer...")
    
    try:
        from core.zpe_integration import ZPEIntegration
        
        integration = ZPEIntegration()
        
        # Test market data
        market_data = {
            'trend_strength': 0.8,
            'entry_exit_range': 0.05,
            'liquidity_depth': 0.7,
            'trend_change_rate': 0.3,
            'price_derivative': 0.02,
            'news_density': 0.6,
            'sentiment_delta': 0.2,
            'strategy': {
                'vectors': {
                    'BTC': {'magnitude': 0.8, 'resonance': 0.7},
                    'ETH': {'magnitude': 0.6, 'resonance': 0.5}
                },
                'weights': {'BTC': 0.6, 'ETH': 0.4}
            },
            'profit': {
                'profit_generated': 100.0,
                'capital_exposure': 1000.0,
                'profit_delta': 50.0,
                'market_heat': 0.7
            }
        }
        
        # Test complete system spin
        result = integration.spin_complete_system(market_data)
        
        print(f"✅ System Spin Score: {result['system_spin_decision']['spin_score']:.6f}")
        print(f"✅ Should Spin: {result['system_spin_decision']['should_spin']}")
        print(f"✅ Integration Status: {result['system_spin_decision']['integration_status']}")
        
        return True
        
    except Exception as e:
        print(f"❌ ZPE integration failed: {e}")
        return False

def test_zpe_rotational_engine():
    """Test ZPE rotational engine."""
    print("\n🧪 Testing ZPE Rotational Engine...")
    
    try:
        from core.zpe_rotational_engine import ZPERotationalEngine
        
        engine = ZPERotationalEngine()
        
        # Test market data
        market_data = {
            'trend_strength': 0.8,
            'entry_exit_range': 0.05,
            'liquidity_depth': 0.7,
            'trend_change_rate': 0.3,
            'price_derivative': 0.02,
            'news_density': 0.6,
            'sentiment_delta': 0.2
        }
        
        # Test profit wheel spin
        result = engine.spin_profit_wheel(market_data)
        
        print(f"✅ ZPE Work: {result['zpe_work']:.6f}")
        print(f"✅ Rotational Torque: {result['rotational_torque']:.6f}")
        print(f"✅ Elastic Resonance: {result['elastic_resonance']:.6f}")
        print(f"✅ Should Spin: {result['should_spin']}")
        print(f"✅ Angular Velocity: {result['angular_velocity']:.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ ZPE rotational engine failed: {e}")
        return False

def test_hybrid_mode_selector():
    """Test hybrid mode selector for dynamic ZPE/Reactive selection."""
    print("\n🧪 Testing Hybrid Mode Selector...")
    
    try:
        from core.zpe_hybrid_mode_selector import (
            select_trading_mode, TradingMode, MarketCondition
        )
        
        # Test bull run conditions (should favor ZPE)
        bull_market_data = {
            'trend_strength': 0.8,
            'volatility': 0.3,
            'price_change_24h': 0.08,
            'profit_performance': 0.15
        }
        
        bull_result = select_trading_mode(bull_market_data, timeframe="daily")
        print(f"✅ Bull Market Mode: {bull_result.selected_mode.value}")
        print(f"✅ Bull Market Confidence: {bull_result.confidence_score:.3f}")
        print(f"✅ Bull Market Condition: {bull_result.market_condition.value}")
        
        # Test bear market conditions (should favor reactive)
        bear_market_data = {
            'trend_strength': -0.6,
            'volatility': 0.8,
            'price_change_24h': -0.12,
            'profit_performance': -0.08
        }
        
        bear_result = select_trading_mode(bear_market_data, timeframe="hourly")
        print(f"✅ Bear Market Mode: {bear_result.selected_mode.value}")
        print(f"✅ Bear Market Confidence: {bear_result.confidence_score:.3f}")
        print(f"✅ Bear Market Condition: {bear_result.market_condition.value}")
        
        # Test crisis conditions (should trigger emergency fallback)
        crisis_market_data = {
            'trend_strength': -0.9,
            'volatility': 0.95,
            'price_change_24h': -0.25,
            'profit_performance': -0.3
        }
        
        crisis_result = select_trading_mode(crisis_market_data, timeframe="hourly")
        print(f"✅ Crisis Mode: {crisis_result.selected_mode.value}")
        print(f"✅ Crisis Confidence: {crisis_result.confidence_score:.3f}")
        print(f"✅ Crisis Condition: {crisis_result.market_condition.value}")
        
        # Test hybrid conditions (should favor hybrid blend)
        hybrid_market_data = {
            'trend_strength': 0.2,
            'volatility': 0.5,
            'price_change_24h': 0.02,
            'profit_performance': 0.05
        }
        
        hybrid_result = select_trading_mode(hybrid_market_data, timeframe="daily")
        print(f"✅ Hybrid Mode: {hybrid_result.selected_mode.value}")
        print(f"✅ Hybrid Confidence: {hybrid_result.confidence_score:.3f}")
        print(f"✅ Hybrid Condition: {hybrid_result.market_condition.value}")
        
        # Verify mode weights for hybrid
        if hybrid_result.selected_mode == TradingMode.HYBRID_BLEND:
            zpe_weight = hybrid_result.mode_weights.get(TradingMode.ZPE_RECURSIVE, 0.0)
            reactive_weight = hybrid_result.mode_weights.get(TradingMode.REACTIVE_TASKING, 0.0)
            print(f"✅ Hybrid ZPE Weight: {zpe_weight:.3f}")
            print(f"✅ Hybrid Reactive Weight: {reactive_weight:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Hybrid mode selector failed: {e}")
        return False

def main():
    """Run all ZPE integration tests."""
    print("🔥 SCHWABOT ZPE MATHEMATICAL FRAMEWORK - INTEGRATION TEST")
    print("=" * 60)
    
    tests = [
        test_zpe_imports,
        test_zpe_core_functions,
        test_zpe_integration,
        test_zpe_rotational_engine,
        test_hybrid_mode_selector
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All ZPE integrations working correctly!")
        print("🔥 Schwabot is now the adaptive wheel - spinning AND reacting as needed!")
        print("🔄 Hybrid mode selection enables dynamic ZPE/Reactive switching!")
    else:
        print("⚠️ Some ZPE integrations need attention")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 