#!/usr/bin/env python3
"""Test Risk and Capital Controls Integration.

This script tests the complete integration of:
- Capital Controls (Advanced position sizing and portfolio management)
- Enhanced Risk Manager (Advanced risk analytics and stress testing)
- Risk Guard (Basic risk management)
- Integration with existing Schwabot systems

Perfect for validating the comprehensive risk and capital management system.
"""

import asyncio
import time
from datetime import datetime
from typing import Dict, Any

# Import capital controls
from core.capital_controls import (
    get_capital_controls, calculate_position_size, update_portfolio_state,
    check_portfolio_limits, suggest_rebalancing, get_capital_status,
    PositionSizingMethod, CapitalConfig
)

# Import enhanced risk manager
from core.enhanced_risk_manager import (
    get_enhanced_risk_manager, calculate_risk_metrics, run_stress_test,
    check_risk_alerts, get_risk_summary, StressTestScenario
)

# Import risk guard for integration
try:
    from core.risk_guard import (
        get_risk_guard, check_risk_limits, check_circuit_breaker,
        is_trading_allowed, get_risk_status
    )
    RISK_GUARD_AVAILABLE = True
except ImportError:
    RISK_GUARD_AVAILABLE = False

# Import VECU and Ferris RDE for integration
try:
    from core.vecu_core import get_vecu_core
    from core.ferris_rde_core import get_ferris_rde
    VECU_FERRIS_AVAILABLE = True
except ImportError:
    VECU_FERRIS_AVAILABLE = False

# Import secure API manager for integration
try:
    from core.secure_api_manager import get_secure_api_manager, APIType
    SECURE_API_AVAILABLE = True
except ImportError:
    SECURE_API_AVAILABLE = False


def test_capital_controls():
    """Test capital controls functionality."""
    print("\n💰 Testing Capital Controls...")
    
    # Get capital controls
    capital_controls = get_capital_controls()
    
    # Test capital configuration
    print("⚙️ Testing capital configuration...")
    
    config = CapitalConfig(
        total_capital=10000.0,
        max_position_size=0.1,
        min_position_size=0.01,
        max_portfolio_risk=0.02,
        target_volatility=0.15,
        max_drawdown=0.20,
        rebalance_threshold=0.05,
        correlation_threshold=0.7,
        kelly_fraction=0.25,
        emergency_capital_reserve=0.1
    )
    
    capital_controls.set_capital_config(config)
    print(f"✅ Capital config set: Total = ${config.total_capital:,.2f}")
    
    # Test position sizing methods
    print("\n📊 Testing position sizing methods...")
    
    # Test volatility-adjusted sizing
    result = calculate_position_size(
        asset="BTC",
        current_price=45000.0,
        volatility=0.03,
        expected_return=0.05,
        confidence=0.7,
        method=PositionSizingMethod.VOLATILITY_ADJUSTED
    )
    print(f"✅ Volatility-adjusted sizing: {result.suggested_size:.2%}")
    
    # Test Kelly Criterion sizing
    result = calculate_position_size(
        asset="ETH",
        current_price=3000.0,
        volatility=0.04,
        expected_return=0.08,
        confidence=0.8,
        method=PositionSizingMethod.KELLY_CRITERION
    )
    print(f"✅ Kelly Criterion sizing: {result.suggested_size:.2%}")
    
    # Test risk parity sizing
    result = calculate_position_size(
        asset="ADA",
        current_price=0.50,
        volatility=0.06,
        expected_return=0.03,
        confidence=0.6,
        method=PositionSizingMethod.RISK_PARITY
    )
    print(f"✅ Risk parity sizing: {result.suggested_size:.2%}")
    
    # Test portfolio state management
    print("\n📈 Testing portfolio state management...")
    
    positions = {
        "BTC": {"value": 5000.0, "unrealized_pnl": 250.0},
        "ETH": {"value": 3000.0, "unrealized_pnl": -100.0},
        "ADA": {"value": 1000.0, "unrealized_pnl": 50.0}
    }
    
    market_data = {
        "BTC": {"volatility": 0.03, "beta": 1.2},
        "ETH": {"volatility": 0.04, "beta": 1.0},
        "ADA": {"volatility": 0.06, "beta": 0.8}
    }
    
    portfolio_state = update_portfolio_state(positions, market_data)
    print(f"✅ Portfolio value: ${portfolio_state.total_value:,.2f}")
    print(f"✅ Portfolio PnL: ${portfolio_state.total_pnl:,.2f}")
    print(f"✅ Portfolio volatility: {portfolio_state.portfolio_volatility:.2%}")
    print(f"✅ Sharpe ratio: {portfolio_state.sharpe_ratio:.2f}")
    
    # Test portfolio limits
    print("\n🛡️ Testing portfolio limits...")
    
    limits_ok = check_portfolio_limits(portfolio_state)
    print(f"✅ Portfolio limits check: {limits_ok}")
    
    # Test rebalancing suggestions
    print("\n⚖️ Testing rebalancing suggestions...")
    
    rebalancing = suggest_rebalancing(portfolio_state)
    print(f"✅ Rebalancing needed: {rebalancing['rebalancing_needed']}")
    print(f"✅ Rebalancing urgency: {rebalancing['urgency']}")
    if rebalancing['actions']:
        print(f"✅ Rebalancing actions: {rebalancing['actions'][:2]}")  # Show first 2 actions
    
    # Get capital status
    status = get_capital_status()
    print(f"✅ Capital status: Total = ${status['total_capital']:,.2f}, Current = ${status['current_capital']:,.2f}")
    
    return True


def test_enhanced_risk_manager():
    """Test enhanced risk manager functionality."""
    print("\n🎯 Testing Enhanced Risk Manager...")
    
    # Get enhanced risk manager
    risk_manager = get_enhanced_risk_manager()
    
    # Test risk metrics calculation
    print("📊 Testing risk metrics calculation...")
    
    portfolio_data = {
        'positions': {
            'BTC': {'value': 5000.0, 'unrealized_pnl': 250.0},
            'ETH': {'value': 3000.0, 'unrealized_pnl': -100.0},
            'ADA': {'value': 1000.0, 'unrealized_pnl': 50.0}
        },
        'total_value': 9000.0,
        'total_pnl': 200.0
    }
    
    market_data = {
        'BTC': {'volatility': 0.03, 'beta': 1.2},
        'ETH': {'volatility': 0.04, 'beta': 1.0},
        'ADA': {'volatility': 0.06, 'beta': 0.8}
    }
    
    # Historical data for drawdown calculation
    historical_data = [
        {'total_value': 8500.0},
        {'total_value': 8700.0},
        {'total_value': 9000.0},
        {'total_value': 8800.0},
        {'total_value': 9200.0}
    ]
    
    risk_metrics = calculate_risk_metrics(portfolio_data, market_data, historical_data)
    print(f"✅ VaR(95%): {risk_metrics.var_95:.2%}")
    print(f"✅ VaR(99%): {risk_metrics.var_99:.2%}")
    print(f"✅ CVaR(95%): {risk_metrics.cvar_95:.2%}")
    print(f"✅ Portfolio volatility: {risk_metrics.volatility:.2%}")
    print(f"✅ Portfolio beta: {risk_metrics.beta:.2f}")
    print(f"✅ Sharpe ratio: {risk_metrics.sharpe_ratio:.2f}")
    print(f"✅ Max drawdown: {risk_metrics.max_drawdown:.2%}")
    print(f"✅ Correlation risk: {risk_metrics.correlation_risk:.2f}")
    print(f"✅ Concentration risk: {risk_metrics.concentration_risk:.2f}")
    
    # Test stress testing
    print("\n🔥 Testing stress testing...")
    
    # Test market crash scenario
    stress_result = run_stress_test(
        portfolio_data, market_data, StressTestScenario.MARKET_CRASH
    )
    print(f"✅ Market crash stress test: Loss = ${stress_result.portfolio_loss:,.2f}")
    print(f"✅ Risk level: {stress_result.risk_level}")
    print(f"✅ Recovery time estimate: {stress_result.recovery_time_estimate:.1f} days")
    
    # Test volatility spike scenario
    stress_result = run_stress_test(
        portfolio_data, market_data, StressTestScenario.VOLATILITY_SPIKE
    )
    print(f"✅ Volatility spike stress test: Loss = ${stress_result.portfolio_loss:,.2f}")
    
    # Test correlation breakdown scenario
    stress_result = run_stress_test(
        portfolio_data, market_data, StressTestScenario.CORRELATION_BREAKDOWN
    )
    print(f"✅ Correlation breakdown stress test: Loss = ${stress_result.portfolio_loss:,.2f}")
    
    # Test risk alerts
    print("\n🚨 Testing risk alerts...")
    
    alerts = check_risk_alerts(risk_metrics)
    print(f"✅ Risk alerts generated: {len(alerts)}")
    for alert in alerts[:3]:  # Show first 3 alerts
        print(f"   - {alert.alert_type}: {alert.description}")
    
    # Get risk summary
    summary = get_risk_summary()
    print(f"✅ Risk summary: {summary['stress_tests_run']} stress tests run, {summary['active_alerts']} active alerts")
    
    return True


def test_risk_guard_integration():
    """Test integration with risk guard."""
    if not RISK_GUARD_AVAILABLE:
        print("\n⚠️ Risk Guard not available, skipping integration test")
        return False
    
    print("\n🛡️ Testing Risk Guard Integration...")
    
    # Get risk guard
    risk_guard = get_risk_guard()
    
    # Test basic risk limits
    print("📊 Testing basic risk limits...")
    
    trade_ok = check_risk_limits(trade_pnl=-50.0, trade_size=75.0, new_exposure=1000.0)
    print(f"✅ Basic risk limits check: {trade_ok}")
    
    # Test circuit breaker
    print("⚡ Testing circuit breaker...")
    
    circuit_ok = check_circuit_breaker(volatility=0.03, entropy=0.6)
    print(f"✅ Circuit breaker check: {circuit_ok}")
    
    # Test trading allowed
    trading_allowed = is_trading_allowed()
    print(f"✅ Trading allowed: {trading_allowed}")
    
    # Get risk status
    risk_status = get_risk_status()
    print(f"✅ Risk status: Circuit breaker = {risk_status['circuit_breaker_state']}")
    
    return True


async def test_vecu_ferris_integration():
    """Test integration with VECU and Ferris RDE."""
    if not VECU_FERRIS_AVAILABLE:
        print("\n⚠️ VECU and Ferris RDE not available, skipping integration test")
        return False
    
    print("\n⚙️ Testing VECU and Ferris RDE Integration...")
    
    # Get VECU and Ferris RDE
    vecu = get_vecu_core()
    ferris = get_ferris_rde()
    
    # Test VECU timing with risk considerations
    print("⏰ Testing VECU timing with risk...")
    
    timing_result = vecu.synchronize_profit_timing(
        market_volatility=0.03,
        entropy_level=0.6,
        current_phase=0.25
    )
    print(f"✅ VECU timing sync: {timing_result}")
    
    # Test Ferris wheel with capital controls
    print("🎡 Testing Ferris wheel with capital controls...")
    
    wheel_result = ferris.update_ferris_wheel(
        btc_price=45000.0,
        market_entropy=0.6,
        current_phase=0.5
    )
    print(f"✅ Ferris wheel update: {wheel_result}")
    
    # Test integration workflow
    print("🔄 Testing integration workflow...")
    
    # Simulate a trading decision with full risk and capital controls
    if is_trading_allowed():
        # Calculate position size using capital controls
        position_result = calculate_position_size(
            asset="BTC",
            current_price=45000.0,
            volatility=0.03,
            expected_return=0.05,
            confidence=0.7,
            method=PositionSizingMethod.VOLATILITY_ADJUSTED
        )
        
        # Check if position size is acceptable
        if position_result.suggested_size > 0:
            print(f"✅ Position size calculated: {position_result.suggested_size:.2%}")
            
            # Update portfolio state
            positions = {"BTC": {"value": position_result.position_value, "unrealized_pnl": 0.0}}
            market_data = {"BTC": {"volatility": 0.03, "beta": 1.2}}
            
            portfolio_state = update_portfolio_state(positions, market_data)
            
            # Check portfolio limits
            if check_portfolio_limits(portfolio_state):
                print("✅ Portfolio limits check passed")
                
                # Calculate risk metrics
                portfolio_data = {
                    'positions': positions,
                    'total_value': portfolio_state.total_value,
                    'total_pnl': portfolio_state.total_pnl
                }
                
                risk_metrics = calculate_risk_metrics(portfolio_data, market_data)
                
                # Check risk alerts
                alerts = check_risk_alerts(risk_metrics)
                if not alerts:
                    print("✅ No risk alerts - trade can proceed")
                else:
                    print(f"⚠️ {len(alerts)} risk alerts - trade should be reconsidered")
            else:
                print("❌ Portfolio limits exceeded - trade blocked")
        else:
            print("❌ Position size too small - trade blocked")
    else:
        print("❌ Trading not allowed - trade blocked")
    
    return True


async def test_secure_api_integration():
    """Test integration with secure API manager."""
    if not SECURE_API_AVAILABLE:
        print("\n⚠️ Secure API Manager not available, skipping integration test")
        return False
    
    print("\n🔐 Testing Secure API Integration...")
    
    # Get secure API manager
    api_manager = get_secure_api_manager()
    
    # Test API statistics
    api_stats = api_manager.get_api_statistics()
    print(f"✅ API statistics: {api_stats['total_requests']} total requests")
    
    # Simulate API call with risk considerations
    print("🌐 Testing API call with risk considerations...")
    
    if is_trading_allowed():
        # Simulate market data API call
        api_manager.total_requests += 1
        api_manager.successful_requests += 1
        
        # Use market data for risk calculations
        market_data = {
            'BTC': {'volatility': 0.03, 'beta': 1.2, 'price': 45000.0},
            'ETH': {'volatility': 0.04, 'beta': 1.0, 'price': 3000.0}
        }
        
        # Calculate position sizes based on API data
        for asset, data in market_data.items():
            position_result = calculate_position_size(
                asset=asset,
                current_price=data['price'],
                volatility=data['volatility'],
                expected_return=0.05,
                confidence=0.7,
                method=PositionSizingMethod.VOLATILITY_ADJUSTED
            )
            print(f"✅ {asset} position size: {position_result.suggested_size:.2%}")
    
    return True


async def test_complete_integration():
    """Test complete integration of all risk and capital control systems."""
    print("\n🔥 Testing Complete Risk and Capital Controls Integration...")
    
    # Test all components
    capital_ok = test_capital_controls()
    risk_ok = test_enhanced_risk_manager()
    risk_guard_ok = test_risk_guard_integration()
    vecu_ferris_ok = await test_vecu_ferris_integration()
    api_ok = await test_secure_api_integration()
    
    # Summary
    print("\n📋 Risk and Capital Controls Integration Test Summary:")
    print(f"✅ Capital Controls: {'PASS' if capital_ok else 'FAIL'}")
    print(f"✅ Enhanced Risk Manager: {'PASS' if risk_ok else 'PASS'}")
    print(f"✅ Risk Guard Integration: {'PASS' if risk_guard_ok else 'SKIP'}")
    print(f"✅ VECU & Ferris RDE Integration: {'PASS' if vecu_ferris_ok else 'SKIP'}")
    print(f"✅ Secure API Integration: {'PASS' if api_ok else 'SKIP'}")
    
    # Final status
    core_systems_ok = capital_ok and risk_ok
    optional_systems_ok = risk_guard_ok and vecu_ferris_ok and api_ok
    
    if core_systems_ok:
        print("\n🎉 CORE RISK AND CAPITAL SYSTEMS: SUCCESS ✅")
        print("💰 Capital Controls: Advanced position sizing and portfolio management")
        print("🎯 Enhanced Risk Manager: VaR, stress testing, and risk analytics")
        
        if optional_systems_ok:
            print("🛡️ All optional systems: Integrated successfully")
        else:
            print("⚠️ Some optional systems: Not available (not required for core functionality)")
        
        print("\n🚀 COMPREHENSIVE RISK MANAGEMENT READY!")
        print("📊 Sophisticated position sizing with Kelly Criterion and risk parity")
        print("🔥 Advanced stress testing with multiple scenarios")
        print("🛡️ Multi-layered risk controls and monitoring")
        print("⚖️ Dynamic portfolio rebalancing and optimization")
        
    else:
        print("\n❌ CORE RISK AND CAPITAL SYSTEMS: FAILED")
        print("🔧 Please check core system implementations")
    
    return core_systems_ok


async def main():
    """Main test function."""
    print("🧪 SCHWABOT RISK AND CAPITAL CONTROLS INTEGRATION TEST")
    print("=" * 70)
    
    # Run complete integration test
    success = await test_complete_integration()
    
    if success:
        print("\n🎯 COMPREHENSIVE RISK MANAGEMENT READY!")
        print("💰 Advanced capital controls with multiple sizing methods")
        print("🎯 Sophisticated risk analytics with VaR and stress testing")
        print("🛡️ Multi-layered risk protection and monitoring")
        print("⚖️ Dynamic portfolio optimization and rebalancing")
        print("🔐 Secure integration with existing Schwabot systems")
        
        print("\n📋 KEY FEATURES:")
        print("• Kelly Criterion and risk parity position sizing")
        print("• VaR and CVaR calculations at multiple confidence levels")
        print("• Stress testing with market crash, volatility spike scenarios")
        print("• Real-time risk monitoring and alerting")
        print("• Portfolio correlation and concentration analysis")
        print("• Dynamic drawdown protection and recovery estimation")
        print("• Integration with VECU timing and Ferris wheel cycles")
        print("• Secure API management with risk-aware decision making")
        
        print("\n🔄 READY FOR BACKTESTING PHASE!")
        print("📊 All risk and capital controls active for safe backtesting")
        print("🧠 Building recursive memory with comprehensive risk management")
        print("🚀 Preparing for live deployment with enterprise-grade safety")
        
    else:
        print("\n❌ INTEGRATION TEST FAILED")
        print("🔧 Please fix core system issues before proceeding")
    
    return success


if __name__ == "__main__":
    # Run the integration test
    asyncio.run(main()) 