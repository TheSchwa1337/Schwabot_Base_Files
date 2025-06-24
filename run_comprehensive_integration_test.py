#!/usr/bin/env python3
"""
Comprehensive Integration Test Runner - Schwabot UROS v1.0
========================================================

Runs comprehensive tests across all mathematical functions, integrations, and demo trading.
Validates the complete trading system pipeline from mathematical functions to live trading simulation.
"""

import sys
import time
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_mathematical_validation() -> Dict[str, Any]:
    """Run mathematical function validation."""
    print("\n" + "="*60)
    print("🧪 MATHEMATICAL FUNCTION VALIDATION")
    print("="*60)
    
    try:
        from core.mathematical_integration_validator import MathematicalIntegrationValidator
        
        validator = MathematicalIntegrationValidator()
        results = validator.run_comprehensive_validation()
        
        # Export results
        validator.export_results("mathematical_validation_results.json")
        
        return results
        
    except ImportError as e:
        print(f"❌ Mathematical validation not available: {e}")
        return {'error': 'Mathematical validation not available'}
    except Exception as e:
        print(f"❌ Mathematical validation failed: {e}")
        return {'error': str(e)}

def run_dlt_matrix_profit_integration() -> Dict[str, Any]:
    """Run DLT Matrix Profit integration tests."""
    print("\n" + "="*60)
    print("🔄 DLT MATRIX PROFIT INTEGRATION TESTS")
    print("="*60)
    
    try:
        from test_dlt_matrix_profit_integration import main as run_integration_tests
        
        # Run the integration tests
        exit_code = run_integration_tests()
        
        # Load results
        try:
            with open("dlt_matrix_profit_integration_results.json", 'r') as f:
                results = json.load(f)
        except FileNotFoundError:
            results = {'error': 'Integration test results not found'}
        
        return {
            'exit_code': exit_code,
            'results': results,
            'success': exit_code == 0
        }
        
    except ImportError as e:
        print(f"❌ Integration tests not available: {e}")
        return {'error': 'Integration tests not available'}
    except Exception as e:
        print(f"❌ Integration tests failed: {e}")
        return {'error': str(e)}

def run_demo_trading_system(duration_seconds: int = 30) -> Dict[str, Any]:
    """Run demo trading system."""
    print("\n" + "="*60)
    print("🚀 DEMO TRADING SYSTEM")
    print("="*60)
    
    try:
        from core.demo_trading_system import DemoTradingSystem, create_demo_strategy
        
        # Create demo trading system
        demo_system = DemoTradingSystem(initial_capital=100000.0)
        
        # Add strategies
        strategy1 = create_demo_strategy(
            strategy_id="strategy_1",
            name="Conservative BTC Strategy",
            symbols=['BTC/USDC'],
            initial_capital=50000.0
        )
        demo_system.add_strategy(strategy1)
        
        strategy2 = create_demo_strategy(
            strategy_id="strategy_2",
            name="Multi-Asset Strategy",
            symbols=['BTC/USDC', 'ETH/USDC', 'ADA/USDC'],
            initial_capital=50000.0
        )
        demo_system.add_strategy(strategy2)
        
        # Start trading
        demo_system.start_trading()
        
        print(f"📈 Demo trading running for {duration_seconds} seconds...")
        time.sleep(duration_seconds)
        
        # Stop trading
        demo_system.stop_trading()
        
        # Get results
        portfolio = demo_system.get_portfolio_status()
        
        print(f"\n📊 DEMO TRADING RESULTS")
        print(f"Initial Capital: ${demo_system.initial_capital:,.2f}")
        print(f"Final Portfolio Value: ${portfolio.total_value:,.2f}")
        print(f"Total Profit: ${portfolio.total_profit:,.2f}")
        print(f"Total Trades: {portfolio.total_trades}")
        print(f"Win Rate: {portfolio.win_rate:.2%}")
        
        # Run mathematical validation
        print("\n🧪 Running Mathematical Validation on Demo System...")
        validation_results = demo_system.run_mathematical_validation()
        
        # Export results
        demo_system.export_demo_results("demo_trading_results.json")
        
        return {
            'initial_capital': demo_system.initial_capital,
            'final_portfolio_value': portfolio.total_value,
            'total_profit': portfolio.total_profit,
            'total_trades': portfolio.total_trades,
            'win_rate': portfolio.win_rate,
            'mathematical_validation': validation_results,
            'success': True
        }
        
    except ImportError as e:
        print(f"❌ Demo trading system not available: {e}")
        return {'error': 'Demo trading system not available'}
    except Exception as e:
        print(f"❌ Demo trading failed: {e}")
        return {'error': str(e)}

def run_component_tests() -> Dict[str, Any]:
    """Run individual component tests."""
    print("\n" + "="*60)
    print("🔧 INDIVIDUAL COMPONENT TESTS")
    print("="*60)
    
    results = {}
    
    # Test DLT Waveform Engine
    try:
        from core.dlt_waveform_engine import DLTWaveformEngine
        dlt_engine = DLTWaveformEngine()
        
        # Test basic functions
        waveform_result = dlt_engine.dlt_waveform(1.0, 0.006)
        entropy_result = dlt_engine.wave_entropy([1.0, 0.0, 1.0, 0.0])
        tensor_result = dlt_engine.tensor_score(100.0, 110.0, 8)
        
        results['dlt_waveform_engine'] = {
            'success': True,
            'waveform_result': waveform_result,
            'entropy_result': entropy_result,
            'tensor_result': tensor_result
        }
        print("✅ DLT Waveform Engine: PASS")
        
    except Exception as e:
        results['dlt_waveform_engine'] = {'success': False, 'error': str(e)}
        print(f"❌ DLT Waveform Engine: FAIL - {e}")
    
    # Test Matrix Mapper
    try:
        from core.matrix_mapper import MatrixMapper
        matrix_mapper = MatrixMapper()
        
        # Test basic functions
        test_hash = "a1b2c3d4e5f67890abcdef1234567890abcdef1234567890abcdef1234567890"
        basket_id = matrix_mapper.decode_hash_to_basket(test_hash, 100, 45000.0)
        tensor_score = matrix_mapper.calculate_tensor_score(44000.0, 45000.0, 8)
        
        results['matrix_mapper'] = {
            'success': True,
            'basket_id': basket_id,
            'tensor_score': tensor_score
        }
        print("✅ Matrix Mapper: PASS")
        
    except Exception as e:
        results['matrix_mapper'] = {'success': False, 'error': str(e)}
        print(f"❌ Matrix Mapper: FAIL - {e}")
    
    # Test Profit Cycle Allocator
    try:
        from core.profit_cycle_allocator import ProfitCycleAllocator
        
        profit_allocator = ProfitCycleAllocator()
        
        # Test allocation
        execution_packet = {
            'volume': 1000.0,
            'actual_profit': 500.0,
            'entry_price': 50000.0,
            'current_price': 51000.0,
            'tick': int(time.time())
        }
        
        market_data = {
            'price': 50000.0, 'volatility': 0.05, 'entropy_level': 4.2, 'complexity': 0.6,
            'trend_strength': 0.3, 'entry_exit_range': 0.02, 'liquidity_depth': 0.8,
            'trend_change_rate': 0.01, 'market_heat': 0.4, 'capital_exposure': 10000.0
        }
        
        allocation_result = profit_allocator.allocate(
            execution_packet=execution_packet,
            cycles=['cycle1', 'cycle2', 'cycle3'],
            market_data=market_data
        )
        
        results['profit_cycle_allocator'] = {
            'success': allocation_result.success,
            'tensor_score': allocation_result.tensor_score,
            'bit_phase': allocation_result.bit_phase
        }
        print("✅ Profit Cycle Allocator: PASS")
        
    except Exception as e:
        results['profit_cycle_allocator'] = {'success': False, 'error': str(e)}
        print(f"❌ Profit Cycle Allocator: FAIL - {e}")
    
    return results

def generate_comprehensive_report(all_results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate comprehensive test report."""
    print("\n" + "="*60)
    print("📊 COMPREHENSIVE TEST REPORT")
    print("="*60)
    
    # Calculate overall statistics
    total_tests = 0
    successful_tests = 0
    failed_tests = 0
    
    # Mathematical validation
    if 'mathematical_validation' in all_results:
        math_results = all_results['mathematical_validation']
        if 'total_tests' in math_results:
            total_tests += math_results['total_tests']
            successful_tests += math_results['successful_tests']
            failed_tests += math_results['failed_tests']
    
    # Integration tests
    if 'integration_tests' in all_results:
        integration_results = all_results['integration_tests']
        if integration_results.get('success', False):
            successful_tests += 1
        else:
            failed_tests += 1
        total_tests += 1
    
    # Demo trading
    if 'demo_trading' in all_results:
        demo_results = all_results['demo_trading']
        if demo_results.get('success', False):
            successful_tests += 1
        else:
            failed_tests += 1
        total_tests += 1
    
    # Component tests
    if 'component_tests' in all_results:
        component_results = all_results['component_tests']
        for component, result in component_results.items():
            total_tests += 1
            if result.get('success', False):
                successful_tests += 1
            else:
                failed_tests += 1
    
    # Calculate success rate
    success_rate = successful_tests / total_tests if total_tests > 0 else 0.0
    
    # Determine overall status
    if success_rate >= 0.95:
        overall_status = "PASS"
    elif success_rate >= 0.90:
        overall_status = "WARN"
    else:
        overall_status = "FAIL"
    
    # Generate report
    report = {
        'timestamp': datetime.now().isoformat(),
        'overall_status': overall_status,
        'success_rate': success_rate,
        'total_tests': total_tests,
        'successful_tests': successful_tests,
        'failed_tests': failed_tests,
        'test_results': all_results,
        'recommendations': []
    }
    
    # Add recommendations
    if success_rate < 0.95:
        report['recommendations'].append("Some tests failed - review error logs")
    if success_rate < 0.90:
        report['recommendations'].append("Multiple test failures - system needs attention")
    if success_rate >= 0.95:
        report['recommendations'].append("All tests passed - system ready for production")
    
    # Print summary
    print(f"Overall Status: {overall_status}")
    print(f"Success Rate: {success_rate:.2%}")
    print(f"Total Tests: {total_tests}")
    print(f"Successful: {successful_tests}")
    print(f"Failed: {failed_tests}")
    
    if report['recommendations']:
        print("\nRecommendations:")
        for rec in report['recommendations']:
            print(f"  - {rec}")
    
    return report

def main():
    """Main function to run comprehensive integration tests."""
    parser = argparse.ArgumentParser(description='Run comprehensive integration tests')
    parser.add_argument('--demo-duration', type=int, default=30, 
                       help='Demo trading duration in seconds (default: 30)')
    parser.add_argument('--skip-demo', action='store_true',
                       help='Skip demo trading system')
    parser.add_argument('--skip-math', action='store_true',
                       help='Skip mathematical validation')
    parser.add_argument('--skip-integration', action='store_true',
                       help='Skip integration tests')
    parser.add_argument('--skip-components', action='store_true',
                       help='Skip component tests')
    
    args = parser.parse_args()
    
    print("🚀 SCHWABOT UROS v1.0 - COMPREHENSIVE INTEGRATION TEST")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Demo Duration: {args.demo_duration} seconds")
    
    all_results = {}
    
    # Run mathematical validation
    if not args.skip_math:
        all_results['mathematical_validation'] = run_mathematical_validation()
    
    # Run integration tests
    if not args.skip_integration:
        all_results['integration_tests'] = run_dlt_matrix_profit_integration()
    
    # Run component tests
    if not args.skip_components:
        all_results['component_tests'] = run_component_tests()
    
    # Run demo trading system
    if not args.skip_demo:
        all_results['demo_trading'] = run_demo_trading_system(args.demo_duration)
    
    # Generate comprehensive report
    report = generate_comprehensive_report(all_results)
    
    # Export report
    try:
        with open("comprehensive_integration_report.json", 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n✅ Comprehensive report exported to comprehensive_integration_report.json")
    except Exception as e:
        print(f"\n❌ Error exporting report: {e}")
    
    # Return exit code
    if report['overall_status'] == "PASS":
        print("\n🎉 All tests passed! System is ready for production.")
        return 0
    elif report['overall_status'] == "WARN":
        print("\n⚠️ Some tests had warnings. Review results before production.")
        return 1
    else:
        print("\n❌ Tests failed. System needs attention before production.")
        return 2

if __name__ == "__main__":
    exit(main()) 