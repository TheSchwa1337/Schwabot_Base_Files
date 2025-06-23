#!/usr/bin/env python3
"""Complete System Integration Test - Validate All New Components.

This script tests the integration of all new Schwabot components:
- Regulatory Compliance (MiFID/SEC, KYC/AML)
- Long-Horizon Simulation (Monte Carlo, Chaos Monkey)
- Environment Manager (Canary, Config, Versioning)
- Precision Performance (Decimal math, Numba, Profiling)
- User Documentation (API examples, configuration)
"""

import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List

# Import all core systems
try:
    # Core mathematical frameworks
    from core.zpe_core import get_zpe_core
    from core.vecu_core import get_vecu_core
    from core.ferris_rde_core import get_ferris_rde
    from core.zpe_integration import get_zpe_integration
    from core.zpe_rotational_engine import get_zpe_rotational_engine
    
    # Risk and capital management
    from core.risk_guard import get_risk_guard
    from core.capital_controls import get_capital_controls
    from core.enhanced_risk_manager import get_enhanced_risk_manager
    
    # Exchange and data layer
    from core.exchange_plumbing import get_exchange_plumbing
    from core.secure_api_manager import get_secure_api_manager
    from core.persistent_state_manager import get_persistent_state_manager
    from core.memory_allocation_manager import get_memory_allocation_manager
    
    # Observability and compliance
    from core.ops_observability import get_ops_observability, log_operation, LogLevel
    from core.regulatory_compliance import (
        get_regulatory_compliance, 
        ComplianceType, 
        OrderRoutingType,
        process_kyc_verification,
        process_aml_check,
        generate_compliance_report
    )
    
    # Environment and performance
    from core.environment_manager import get_environment_manager, EnvironmentType
    from core.precision_performance import get_precision_performance_manager
    
    # Long-horizon simulation
    from core.long_horizon_simulation import (
        get_long_horizon_simulation,
        run_monte_carlo_simulation,
        run_chaos_monkey_test,
        SimulationType
    )
    
    # Unified math
    from core.unified_math import get_unified_math
    
    CORE_SYSTEMS_AVAILABLE = True
    print("✅ All core systems imported successfully")
    
except ImportError as e:
    CORE_SYSTEMS_AVAILABLE = False
    print(f"⚠️ Some core systems not available: {e}")


def test_mathematical_frameworks() -> Dict[str, Any]:
    """Test all mathematical frameworks."""
    print("\n🧮 Testing Mathematical Frameworks...")
    
    results = {}
    
    try:
        # Test ZPE Core
        zpe_core = get_zpe_core()
        resonance = zpe_core.calculate_resonance(btc_price=50000.0)
        allocation = zpe_core.calculate_profit_allocation(1000.0, 0.75)
        
        results['zpe_core'] = {
            'resonance': resonance,
            'allocation': allocation,
            'status': 'success'
        }
        print(f"✅ ZPE Core: resonance={resonance:.4f}, allocation={allocation:.2f}")
        
        # Test VECU Core
        vecu_core = get_vecu_core()
        phase = vecu_core.calculate_timing_phase(datetime.now())
        burst = vecu_core.calculate_pwm_burst(1000, 0.6)
        
        results['vecu_core'] = {
            'phase': phase,
            'burst': burst,
            'status': 'success'
        }
        print(f"✅ VECU Core: phase={phase:.4f}, burst={burst:.2f}")
        
        # Test Ferris RDE
        ferris_rde = get_ferris_rde()
        position = ferris_rde.calculate_wheel_position(5000.0)
        hash_seq = ferris_rde.generate_hash_sequence(16)
        
        results['ferris_rde'] = {
            'position': position,
            'hash_sequence_length': len(hash_seq),
            'status': 'success'
        }
        print(f"✅ Ferris RDE: position={position:.4f}, hash_seq_len={len(hash_seq)}")
        
        # Test Unified Math
        unified_math = get_unified_math()
        constants = unified_math.get_constants()
        
        results['unified_math'] = {
            'constants_count': len(constants),
            'status': 'success'
        }
        print(f"✅ Unified Math: {len(constants)} constants loaded")
        
    except Exception as e:
        results['error'] = str(e)
        print(f"❌ Mathematical frameworks test failed: {e}")
    
    return results


def test_risk_and_capital_management() -> Dict[str, Any]:
    """Test risk and capital management systems."""
    print("\n🛡️ Testing Risk and Capital Management...")
    
    results = {}
    
    try:
        # Test Risk Guard
        risk_guard = get_risk_guard()
        trade_allowed = risk_guard.check_trade_allowed("BTC/USD", "buy", 0.1, 50000.0)
        risk_status = risk_guard.get_risk_status()
        
        results['risk_guard'] = {
            'trade_allowed': trade_allowed,
            'risk_status': risk_status,
            'status': 'success'
        }
        print(f"✅ Risk Guard: trade_allowed={trade_allowed}, status={risk_status['overall_status']}")
        
        # Test Capital Controls
        capital_controls = get_capital_controls()
        position_size = capital_controls.calculate_position_size(10000.0, 0.02, 0.05)
        portfolio_state = capital_controls.get_portfolio_state()
        
        results['capital_controls'] = {
            'position_size': position_size,
            'portfolio_state': portfolio_state,
            'status': 'success'
        }
        print(f"✅ Capital Controls: position_size={position_size:.4f}")
        
        # Test Enhanced Risk Manager
        enhanced_risk = get_enhanced_risk_manager()
        var_result = enhanced_risk.calculate_var([0.01, -0.02, 0.015, -0.01], 0.95)
        stress_test = enhanced_risk.run_stress_test("market_crash")
        
        results['enhanced_risk'] = {
            'var_result': var_result,
            'stress_test': stress_test,
            'status': 'success'
        }
        print(f"✅ Enhanced Risk: VaR={var_result:.4f}, stress_test={stress_test['scenario']}")
        
    except Exception as e:
        results['error'] = str(e)
        print(f"❌ Risk and capital management test failed: {e}")
    
    return results


def test_exchange_and_data_layer() -> Dict[str, Any]:
    """Test exchange and data layer systems."""
    print("\n💱 Testing Exchange and Data Layer...")
    
    results = {}
    
    try:
        # Test Exchange Plumbing
        exchange_plumbing = get_exchange_plumbing()
        exchange_status = exchange_plumbing.get_system_status()
        
        results['exchange_plumbing'] = {
            'status': exchange_status,
            'test_status': 'success'
        }
        print(f"✅ Exchange Plumbing: {exchange_status['connection_status']}")
        
        # Test Secure API Manager
        api_manager = get_secure_api_manager()
        api_status = api_manager.get_system_status()
        
        results['secure_api_manager'] = {
            'status': api_status,
            'test_status': 'success'
        }
        print(f"✅ Secure API Manager: {api_status['encryption_enabled']}")
        
        # Test Persistent State Manager
        state_manager = get_persistent_state_manager()
        state_status = state_manager.get_system_status()
        
        results['persistent_state'] = {
            'status': state_status,
            'test_status': 'success'
        }
        print(f"✅ Persistent State: {state_status['database_status']}")
        
        # Test Memory Allocation Manager
        memory_manager = get_memory_allocation_manager()
        memory_status = memory_manager.get_system_status()
        
        results['memory_allocation'] = {
            'status': memory_status,
            'test_status': 'success'
        }
        print(f"✅ Memory Allocation: {memory_status['allocation_status']}")
        
    except Exception as e:
        results['error'] = str(e)
        print(f"❌ Exchange and data layer test failed: {e}")
    
    return results


def test_observability_and_compliance() -> Dict[str, Any]:
    """Test observability and compliance systems."""
    print("\n📊 Testing Observability and Compliance...")
    
    results = {}
    
    try:
        # Test Ops Observability
        ops_obs = get_ops_observability()
        ops_status = ops_obs.get_system_status()
        
        # Log test operation
        log_operation(
            operation="integration_test",
            component="test_script",
            level=LogLevel.INFO,
            success=True,
            test_phase="observability"
        )
        
        results['ops_observability'] = {
            'status': ops_status,
            'test_status': 'success'
        }
        print(f"✅ Ops Observability: {ops_status['logging_enabled']}")
        
        # Test Regulatory Compliance
        compliance = get_regulatory_compliance()
        compliance_status = compliance.get_system_status()
        
        # Test KYC verification
        kyc_record = process_kyc_verification(
            client_id="test_client_001",
            client_name="Test Client",
            client_type="individual",
            documents=["passport", "utility_bill"]
        )
        
        # Test AML check
        aml_record = process_aml_check(
            client_id="test_client_001",
            transaction_id="tx_001",
            transaction_type="deposit",
            amount=5000.0,
            currency="USD"
        )
        
        results['regulatory_compliance'] = {
            'status': compliance_status,
            'kyc_record': kyc_record.verification_status if kyc_record else None,
            'aml_record': aml_record.risk_score if aml_record else None,
            'test_status': 'success'
        }
        print(f"✅ Regulatory Compliance: KYC={kyc_record.verification_status if kyc_record else 'skipped'}, AML={aml_record.risk_score if aml_record else 'skipped'}")
        
    except Exception as e:
        results['error'] = str(e)
        print(f"❌ Observability and compliance test failed: {e}")
    
    return results


def test_environment_and_performance() -> Dict[str, Any]:
    """Test environment manager and precision performance."""
    print("\n⚙️ Testing Environment and Performance...")
    
    results = {}
    
    try:
        # Test Environment Manager
        env_manager = get_environment_manager()
        env_status = env_manager.get_system_status()
        
        # Test configuration
        config = env_manager.get_config()
        
        results['environment_manager'] = {
            'status': env_status,
            'config_keys': list(config.keys()) if config else [],
            'test_status': 'success'
        }
        print(f"✅ Environment Manager: {env_status['environment_type']}")
        
        # Test Precision Performance
        perf_manager = get_precision_performance_manager()
        perf_status = perf_manager.get_system_status()
        
        # Test decimal precision
        decimal_result = perf_manager.calculate_with_decimal_precision(1.0, 3.0)
        
        # Test profiling
        perf_manager.start_profiling("test_profile")
        time.sleep(0.1)  # Simulate work
        profile_result = perf_manager.stop_profiling("test_profile")
        
        results['precision_performance'] = {
            'status': perf_status,
            'decimal_result': decimal_result,
            'profile_execution_time': profile_result.get('execution_time', 0),
            'test_status': 'success'
        }
        print(f"✅ Precision Performance: decimal_result={decimal_result}, profile_time={profile_result.get('execution_time', 0):.4f}s")
        
    except Exception as e:
        results['error'] = str(e)
        print(f"❌ Environment and performance test failed: {e}")
    
    return results


async def test_long_horizon_simulation() -> Dict[str, Any]:
    """Test long-horizon simulation systems."""
    print("\n🔮 Testing Long-Horizon Simulation...")
    
    results = {}
    
    try:
        # Test Long-Horizon Simulation
        simulation = get_long_horizon_simulation()
        sim_status = simulation.get_system_status()
        
        results['long_horizon_simulation'] = {
            'status': sim_status,
            'test_status': 'success'
        }
        print(f"✅ Long-Horizon Simulation: {sim_status['simulation_type']}")
        
        # Test Monte Carlo simulation (small scale)
        print("🎲 Running small Monte Carlo simulation...")
        mc_results = await run_monte_carlo_simulation(num_scenarios=3, duration_days=1)
        
        if mc_results:
            total_pnl = sum(r.total_pnl for r in mc_results)
            avg_sharpe = sum(r.sharpe_ratio for r in mc_results) / len(mc_results)
            
            results['monte_carlo'] = {
                'scenarios_completed': len(mc_results),
                'total_pnl': total_pnl,
                'avg_sharpe': avg_sharpe,
                'test_status': 'success'
            }
            print(f"✅ Monte Carlo: {len(mc_results)} scenarios, PnL=${total_pnl:.2f}, Sharpe={avg_sharpe:.2f}")
        
        # Test Chaos Monkey (short duration)
        print("🐒 Running short Chaos Monkey test...")
        chaos_events = await run_chaos_monkey_test(duration_hours=1)
        
        if chaos_events:
            recovery_rate = sum(1 for e in chaos_events if e.recovery_successful) / len(chaos_events)
            
            results['chaos_monkey'] = {
                'events_triggered': len(chaos_events),
                'recovery_rate': recovery_rate,
                'test_status': 'success'
            }
            print(f"✅ Chaos Monkey: {len(chaos_events)} events, recovery_rate={recovery_rate:.1%}")
        
    except Exception as e:
        results['error'] = str(e)
        print(f"❌ Long-horizon simulation test failed: {e}")
    
    return results


def test_documentation_and_config() -> Dict[str, Any]:
    """Test documentation and configuration."""
    print("\n📚 Testing Documentation and Configuration...")
    
    results = {}
    
    try:
        # Check if documentation exists
        docs_path = Path("docs")
        readme_path = docs_path / "README.md"
        
        docs_exist = docs_path.exists()
        readme_exists = readme_path.exists()
        
        if readme_exists:
            readme_size = readme_path.stat().st_size
            readme_content = readme_path.read_text()
            has_architecture = "Architecture" in readme_content
            has_api_examples = "API Reference" in readme_content
        else:
            readme_size = 0
            has_architecture = False
            has_api_examples = False
        
        results['documentation'] = {
            'docs_directory_exists': docs_exist,
            'readme_exists': readme_exists,
            'readme_size_bytes': readme_size,
            'has_architecture_section': has_architecture,
            'has_api_examples': has_api_examples,
            'test_status': 'success'
        }
        print(f"✅ Documentation: README={readme_exists}, size={readme_size} bytes, architecture={has_architecture}")
        
        # Check configuration files
        config_path = Path("config")
        config_exists = config_path.exists()
        
        if config_exists:
            config_files = list(config_path.glob("*.yaml")) + list(config_path.glob("*.yml"))
            config_file_count = len(config_files)
        else:
            config_file_count = 0
        
        results['configuration'] = {
            'config_directory_exists': config_exists,
            'config_file_count': config_file_count,
            'test_status': 'success'
        }
        print(f"✅ Configuration: config_dir={config_exists}, files={config_file_count}")
        
    except Exception as e:
        results['error'] = str(e)
        print(f"❌ Documentation and configuration test failed: {e}")
    
    return results


def generate_integration_report(all_results: Dict[str, Any]) -> None:
    """Generate comprehensive integration test report."""
    print("\n📋 Generating Integration Test Report...")
    
    try:
        # Calculate overall statistics
        total_tests = len(all_results)
        successful_tests = sum(1 for result in all_results.values() if result.get('test_status') == 'success')
        failed_tests = total_tests - successful_tests
        
        # Create report
        report = {
            'test_summary': {
                'total_tests': total_tests,
                'successful_tests': successful_tests,
                'failed_tests': failed_tests,
                'success_rate': successful_tests / total_tests if total_tests > 0 else 0,
                'timestamp': datetime.now().isoformat()
            },
            'test_results': all_results,
            'system_status': {
                'core_systems_available': CORE_SYSTEMS_AVAILABLE,
                'python_version': f"{__import__('sys').version}",
                'platform': __import__('platform').platform()
            }
        }
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = Path(f"integration_test_report_{timestamp}.json")
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"✅ Integration report saved: {report_path}")
        
        # Print summary
        print(f"\n🎯 Integration Test Summary:")
        print(f"   Total Tests: {total_tests}")
        print(f"   Successful: {successful_tests}")
        print(f"   Failed: {failed_tests}")
        print(f"   Success Rate: {report['test_summary']['success_rate']:.1%}")
        
        if failed_tests > 0:
            print(f"\n❌ Failed Tests:")
            for test_name, result in all_results.items():
                if result.get('test_status') != 'success':
                    print(f"   - {test_name}: {result.get('error', 'Unknown error')}")
        
        # Log final operation
        if CORE_SYSTEMS_AVAILABLE:
            log_operation(
                operation="integration_test_completed",
                component="test_script",
                level=LogLevel.INFO,
                success=successful_tests == total_tests,
                total_tests=total_tests,
                successful_tests=successful_tests,
                success_rate=report['test_summary']['success_rate']
            )
        
    except Exception as e:
        print(f"❌ Report generation failed: {e}")


async def main():
    """Run complete system integration test."""
    print("🚀 Starting Complete System Integration Test")
    print("=" * 60)
    
    all_results = {}
    
    # Test mathematical frameworks
    all_results['mathematical_frameworks'] = test_mathematical_frameworks()
    
    # Test risk and capital management
    all_results['risk_capital_management'] = test_risk_and_capital_management()
    
    # Test exchange and data layer
    all_results['exchange_data_layer'] = test_exchange_and_data_layer()
    
    # Test observability and compliance
    all_results['observability_compliance'] = test_observability_and_compliance()
    
    # Test environment and performance
    all_results['environment_performance'] = test_environment_and_performance()
    
    # Test long-horizon simulation
    all_results['long_horizon_simulation'] = await test_long_horizon_simulation()
    
    # Test documentation and configuration
    all_results['documentation_configuration'] = test_documentation_and_config()
    
    # Generate comprehensive report
    generate_integration_report(all_results)
    
    print("\n" + "=" * 60)
    print("🎉 Complete System Integration Test Finished")
    
    # Final status check
    successful_tests = sum(1 for result in all_results.values() if result.get('test_status') == 'success')
    total_tests = len(all_results)
    
    if successful_tests == total_tests:
        print("✅ All integration tests passed! Schwabot is ready for deployment.")
    else:
        print(f"⚠️ {total_tests - successful_tests} tests failed. Please review the report for details.")


if __name__ == "__main__":
    # Run the complete integration test
    asyncio.run(main()) 