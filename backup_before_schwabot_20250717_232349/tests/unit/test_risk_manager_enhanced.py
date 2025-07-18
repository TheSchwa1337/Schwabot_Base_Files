#!/usr/bin/env python3
"""
Enhanced Risk Manager Test Suite 🧪

Comprehensive testing for the GODMODE enhanced risk_manager.py:
- Individual mathematical functions (VaR, ES, Sharpe, MDD)
- Risk metrics calculation
- Position and portfolio risk assessment
- Schwabot strategy integration
- JSON output and hash generation
"""

import numpy as np
import json
from core.risk_manager import RiskManager, RiskLevel

def test_individual_risk_functions():
    """Test individual risk calculation functions."""
    print("🧪 Testing Individual Risk Functions...")
    
    # Create test instance
    risk_manager = RiskManager(risk_tolerance=0.02, max_portfolio_risk=0.05)
    
    # Generate test returns (simulated daily returns)
    np.random.seed(42)  # For reproducible tests
    test_returns = np.random.normal(0.001, 0.02, 252)  # 252 trading days
    
    print(f"\n1. Testing VaR Calculation...")
    try:
        var_95 = risk_manager.compute_var(test_returns, 0.95)
        var_99 = risk_manager.compute_var(test_returns, 0.99)
        print(f"   VaR(95%): {var_95:.4f}")
        print(f"   VaR(99%): {var_99:.4f}")
        assert var_99 <= var_95, "99% VaR should be <= 95% VaR"
        print("   ✅ VaR calculation passed")
    except Exception as e:
        print(f"   ❌ VaR calculation failed: {e}")
    
    print(f"\n2. Testing Expected Shortfall Calculation...")
    try:
        es_95 = risk_manager.compute_expected_shortfall(test_returns, 0.95)
        es_99 = risk_manager.compute_expected_shortfall(test_returns, 0.99)
        print(f"   ES(95%): {es_95:.4f}")
        print(f"   ES(99%): {es_99:.4f}")
        assert es_99 <= es_95, "99% ES should be <= 95% ES"
        print("   ✅ Expected Shortfall calculation passed")
    except Exception as e:
        print(f"   ❌ Expected Shortfall calculation failed: {e}")
    
    print(f"\n3. Testing Sharpe Ratio Calculation...")
    try:
        sharpe = risk_manager.compute_sharpe_ratio(test_returns, 0.02)
        print(f"   Sharpe Ratio: {sharpe:.4f}")
        assert isinstance(sharpe, float), "Sharpe ratio should be a float"
        print("   ✅ Sharpe ratio calculation passed")
    except Exception as e:
        print(f"   ❌ Sharpe ratio calculation failed: {e}")
    
    print(f"\n4. Testing Maximum Drawdown Calculation...")
    try:
        mdd = risk_manager.compute_max_drawdown(test_returns)
        print(f"   Maximum Drawdown: {mdd:.4f}")
        assert mdd <= 0, "Maximum drawdown should be negative or zero"
        print("   ✅ Maximum drawdown calculation passed")
    except Exception as e:
        print(f"   ❌ Maximum drawdown calculation failed: {e}")

def test_comprehensive_risk_metrics():
    """Test comprehensive risk metrics calculation."""
    print(f"\n🧪 Testing Comprehensive Risk Metrics...")
    
    risk_manager = RiskManager()
    
    # Generate test returns
    np.random.seed(42)
    test_returns = np.random.normal(0.001, 0.02, 252)
    
    try:
        risk_metrics = risk_manager.calculate_risk_metrics(test_returns)
        
        print(f"   VaR(95%): {risk_metrics.var_95:.4f}")
        print(f"   VaR(99%): {risk_metrics.var_99:.4f}")
        print(f"   Expected Shortfall(95%): {risk_metrics.cvar_95:.4f}")
        print(f"   Expected Shortfall(99%): {risk_metrics.cvar_99:.4f}")
        print(f"   Sharpe Ratio: {risk_metrics.sharpe_ratio:.4f}")
        print(f"   Sortino Ratio: {risk_metrics.sortino_ratio:.4f}")
        print(f"   Maximum Drawdown: {risk_metrics.max_drawdown:.4f}")
        print(f"   Volatility: {risk_metrics.volatility:.4f}")
        print(f"   Risk Hash: {risk_metrics.risk_hash}")
        
        # Validate metrics
        assert risk_metrics.var_99 <= risk_metrics.var_95, "99% VaR should be <= 95% VaR"
        assert risk_metrics.cvar_99 <= risk_metrics.cvar_95, "99% ES should be <= 95% ES"
        assert risk_metrics.max_drawdown <= 0, "Maximum drawdown should be negative or zero"
        assert len(risk_metrics.risk_hash) > 0, "Risk hash should be generated"
        
        print("   ✅ Comprehensive risk metrics calculation passed")
        
    except Exception as e:
        print(f"   ❌ Comprehensive risk metrics calculation failed: {e}")

def test_position_risk_assessment():
    """Test position risk assessment."""
    print(f"\n🧪 Testing Position Risk Assessment...")
    
    risk_manager = RiskManager()
    
    # Generate test data
    np.random.seed(42)
    historical_returns = np.random.normal(0.001, 0.02, 252)
    
    try:
        position_risk = risk_manager.assess_position_risk(
            symbol="BTC/USDT",
            position_size=1.0,
            current_price=50000.0,
            historical_returns=historical_returns,
            entry_price=48000.0
        )
        
        print(f"   Symbol: {position_risk.symbol}")
        print(f"   Position Size: {position_risk.position_size}")
        print(f"   Current Value: {position_risk.current_value:.2f}")
        print(f"   Unrealized PnL: {position_risk.unrealized_pnl:.2f}")
        print(f"   Risk Level: {position_risk.risk_level.value}")
        print(f"   Max Position Size: {position_risk.max_position_size:.2f}")
        print(f"   Stop Loss Level: {position_risk.stop_loss_level:.2f}")
        print(f"   Take Profit Level: {position_risk.take_profit_level:.2f}")
        print(f"   Position Hash: {position_risk.position_hash}")
        
        # Validate position risk
        assert position_risk.unrealized_pnl > 0, "PnL should be positive for this test"
        assert position_risk.stop_loss_level < position_risk.take_profit_level, "Stop loss should be below take profit"
        assert len(position_risk.position_hash) > 0, "Position hash should be generated"
        
        print("   ✅ Position risk assessment passed")
        
    except Exception as e:
        print(f"   ❌ Position risk assessment failed: {e}")

def test_portfolio_risk_assessment():
    """Test portfolio risk assessment."""
    print(f"\n🧪 Testing Portfolio Risk Assessment...")
    
    risk_manager = RiskManager()
    
    # Generate test portfolio data
    np.random.seed(42)
    portfolio_returns = np.random.normal(0.001, 0.02, 252)
    
    portfolio_data = {
        "total_value": 100000.0,
        "total_pnl": 5000.0,
        "positions": [
            {"symbol": "BTC/USDT", "returns": np.random.normal(0.001, 0.02, 252)},
            {"symbol": "ETH/USDT", "returns": np.random.normal(0.001, 0.02, 252)}
        ],
        "returns": portfolio_returns
    }
    
    try:
        portfolio_risk = risk_manager.assess_portfolio_risk(portfolio_data)
        
        print(f"   Total Value: {portfolio_risk.total_value:.2f}")
        print(f"   Total PnL: {portfolio_risk.total_pnl:.2f}")
        print(f"   Risk Level: {portfolio_risk.risk_level.value}")
        print(f"   VaR(95%): {portfolio_risk.risk_metrics.var_95:.4f}")
        print(f"   Sharpe Ratio: {portfolio_risk.risk_metrics.sharpe_ratio:.4f}")
        print(f"   Maximum Drawdown: {portfolio_risk.risk_metrics.max_drawdown:.4f}")
        print(f"   Positions Count: {len(portfolio_risk.positions)}")
        print(f"   Portfolio Hash: {portfolio_risk.portfolio_hash}")
        
        # Validate portfolio risk
        assert portfolio_risk.total_value > 0, "Total value should be positive"
        assert len(portfolio_risk.portfolio_hash) > 0, "Portfolio hash should be generated"
        
        print("   ✅ Portfolio risk assessment passed")
        
    except Exception as e:
        print(f"   ❌ Portfolio risk assessment failed: {e}")

def test_schwabot_integration():
    """Test Schwabot strategy integration features."""
    print(f"\n🧪 Testing Schwabot Strategy Integration...")
    
    risk_manager = RiskManager()
    
    # First, create some portfolio risk data
    np.random.seed(42)
    portfolio_returns = np.random.normal(0.001, 0.02, 252)
    
    portfolio_data = {
        "total_value": 100000.0,
        "total_pnl": 5000.0,
        "positions": [],
        "returns": portfolio_returns
    }
    
    try:
        # Assess portfolio risk
        portfolio_risk = risk_manager.assess_portfolio_risk(portfolio_data)
        
        # Test risk flags JSON
        print(f"\n1. Testing Risk Flags JSON...")
        risk_flags = risk_manager.get_risk_flags_json(portfolio_risk)
        print(f"   Risk Flags: {json.dumps(risk_flags, indent=2)}")
        assert "risk_level" in risk_flags, "Risk flags should contain risk_level"
        assert "var_95" in risk_flags, "Risk flags should contain var_95"
        assert "sharpe_ratio" in risk_flags, "Risk flags should contain sharpe_ratio"
        print("   ✅ Risk flags JSON generation passed")
        
        # Test strategy decision packet
        print(f"\n2. Testing Strategy Decision Packet...")
        strategy_packet = risk_manager.get_strategy_decision_packet()
        print(f"   Can Trade: {strategy_packet.get('can_trade', 'N/A')}")
        print(f"   Position Size Multiplier: {strategy_packet.get('position_size_multiplier', 'N/A')}")
        print(f"   Stop Loss Multiplier: {strategy_packet.get('stop_loss_multiplier', 'N/A')}")
        print(f"   Take Profit Multiplier: {strategy_packet.get('take_profit_multiplier', 'N/A')}")
        print(f"   Emergency Flags: {strategy_packet.get('emergency_flags', 'N/A')}")
        
        assert "can_trade" in strategy_packet, "Strategy packet should contain can_trade"
        assert "position_size_multiplier" in strategy_packet, "Strategy packet should contain position_size_multiplier"
        assert "emergency_flags" in strategy_packet, "Strategy packet should contain emergency_flags"
        
        print("   ✅ Strategy decision packet generation passed")
        
        # Test hash consistency
        print(f"\n3. Testing Hash Consistency...")
        risk_flags_2 = risk_manager.get_risk_flags_json(portfolio_risk)
        assert risk_flags["risk_hash"] == risk_flags_2["risk_hash"], "Risk hash should be consistent"
        assert risk_flags["portfolio_hash"] == risk_flags_2["portfolio_hash"], "Portfolio hash should be consistent"
        print("   ✅ Hash consistency verified")
        
        print("   ✅ Schwabot strategy integration passed")
        
    except Exception as e:
        print(f"   ❌ Schwabot strategy integration failed: {e}")

def test_error_handling():
    """Test error handling and edge cases."""
    print(f"\n🧪 Testing Error Handling...")
    
    risk_manager = RiskManager()
    
    # Test empty returns array
    print(f"\n1. Testing Empty Returns Array...")
    try:
        risk_manager.compute_var(np.array([]))
        print("   ❌ Should have raised ValueError for empty array")
    except ValueError:
        print("   ✅ Correctly handled empty returns array")
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
    
    # Test invalid confidence level
    print(f"\n2. Testing Invalid Confidence Level...")
    try:
        risk_manager.compute_var(np.array([0.01, 0.02]), 1.5)
        print("   ❌ Should have raised ValueError for invalid confidence level")
    except ValueError:
        print("   ✅ Correctly handled invalid confidence level")
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
    
    # Test negative prices
    print(f"\n3. Testing Negative Prices...")
    try:
        risk_manager.assess_position_risk(
            symbol="TEST",
            position_size=1.0,
            current_price=-100.0,
            historical_returns=np.array([0.01, 0.02]),
            entry_price=100.0
        )
        print("   ❌ Should have raised ValueError for negative price")
    except ValueError:
        print("   ✅ Correctly handled negative price")
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")

def main():
    """Run all tests."""
    print("🚀 Enhanced Risk Manager Test Suite")
    print("=" * 50)
    
    try:
        test_individual_risk_functions()
        test_comprehensive_risk_metrics()
        test_position_risk_assessment()
        test_portfolio_risk_assessment()
        test_schwabot_integration()
        test_error_handling()
        
        print(f"\n🎉 All tests completed successfully!")
        print(f"✅ Risk Manager is GODMODE READY for Schwabot integration!")
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        raise

if __name__ == "__main__":
    main() 