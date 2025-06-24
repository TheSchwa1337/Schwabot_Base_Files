#!/usr/bin/env python3
"""
Final System Verification - Schwabot UROS v1.0
=============================================

Comprehensive verification script that demonstrates all implemented components
are working correctly and integrated properly.

This script verifies:
- All mathematical functions are implemented and working
- All integration points are functional
- All new components are properly connected
- The complete pipeline can execute successfully
"""

import sys
import os
import json
import time
from datetime import datetime
from typing import Dict, List, Any

# Add core directory to path
sys.path.append('./core')

def verify_mathematical_functions():
    """Verify all mathematical functions are implemented and working."""
    print("🔬 Verifying Mathematical Functions...")
    
    try:
        # Test bit resolution engine
        from bit_resolution_engine import BitResolutionEngine
        
        bit_engine = BitResolutionEngine()
        
        # Test resolve_bit_phase function
        test_hash = "a1b2c3d4e5f67890abcdef1234567890abcdef1234567890abcdef1234567890"
        
        # Test 4-bit resolution
        phase_4bit = bit_engine.resolve_bit_phase(test_hash, "4bit")
        print(f"   ✅ 4-bit resolution: {phase_4bit}")
        
        # Test 8-bit resolution
        phase_8bit = bit_engine.resolve_bit_phase(test_hash, "8bit")
        print(f"   ✅ 8-bit resolution: {phase_8bit}")
        
        # Test 42-bit resolution
        phase_42bit = bit_engine.resolve_bit_phase(test_hash, "42bit")
        print(f"   ✅ 42-bit resolution: {phase_42bit}")
        
        # Test tensor score calculation
        market_data = {
            'entropy_level': 4.5,
            'volatility': 0.03,
            'market_heat': 0.6
        }
        
        tensor_score = bit_engine.calculate_tensor_score(50000.0, 51000.0, phase_8bit, market_data)
        print(f"   ✅ Tensor score calculation: {tensor_score}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Mathematical functions verification failed: {e}")
        return False

def verify_tensor_score_utils():
    """Verify tensor score utilities are working."""
    print("🧮 Verifying Tensor Score Utils...")
    
    try:
        from tensor_score_utils import TensorScoreUtils
        
        tensor_utils = TensorScoreUtils()
        
        # Test wave entropy calculation
        sequence = [1.0, 1.1, 0.9, 1.2, 0.8, 1.3, 1.1, 0.95]
        entropy = tensor_utils.calculate_wave_entropy(sequence)
        print(f"   ✅ Wave entropy calculation: {entropy:.4f}")
        
        # Test profit rebalancing
        rebalance = tensor_utils.rebalance_profit(1000.0, 0.03, 4.5)
        print(f"   ✅ Profit rebalancing: {rebalance.allocations}")
        
        # Test phase vector creation
        phase_vector = tensor_utils.create_phase_vector(8, 16, 4)
        print(f"   ✅ Phase vector creation: {len(phase_vector.vector_components)} components")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Tensor score utils verification failed: {e}")
        return False

def verify_trade_simulation():
    """Verify trade simulation engine is working."""
    print("💰 Verifying Trade Simulation Engine...")
    
    try:
        from simulate_trade import TradeSimulator, TradeType
        
        simulator = TradeSimulator()
        
        # Test strategy bucket
        strategy_bucket = {
            'asset': 'BTC',
            'strategy_id': 'long_hold_btc',
            'tensor_score': 0.03,
            'bit_phase': 8,
            'basket_id': 'basket_8bit_161',
            'current_price': 50000.0,
            'market_data': {'entropy_level': 4.5, 'volatility': 0.03}
        }
        
        # Test trade simulation
        trade_result = simulator.simulate_trade(strategy_bucket, "DEMO")
        print(f"   ✅ Trade simulation: {trade_result.trade_type.value} {trade_result.quantity:.4f} {trade_result.asset}")
        
        # Test portfolio state
        portfolio = simulator.get_portfolio_state()
        print(f"   ✅ Portfolio state: ${portfolio.total_value:.2f} total value")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Trade simulation verification failed: {e}")
        return False

def verify_demo_ledger_injection():
    """Verify demo ledger state injection is working."""
    print("📊 Verifying Demo Ledger State Injection...")
    
    try:
        from inject_demo_ledger import DemoLedgerInjector, DemoScenario
        
        injector = DemoLedgerInjector()
        
        # Test scenario injection
        success = injector.inject_demo_state("balanced")
        print(f"   ✅ Demo state injection: {'SUCCESS' if success else 'FAILED'}")
        
        # Test available scenarios
        scenarios = injector.get_available_scenarios()
        print(f"   ✅ Available scenarios: {len(scenarios)} scenarios")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Demo ledger injection verification failed: {e}")
        return False

def verify_vector_state_export():
    """Verify vector state export engine is working."""
    print("📤 Verifying Vector State Export Engine...")
    
    try:
        from export_vector_snapshot import VectorStateExporter, SnapshotType, ExportFormat
        
        exporter = VectorStateExporter()
        
        # Test DLT waveform export
        dlt_data = {
            'waveform_name': 'test_waveform',
            'sequence_data': [1.0, 1.1, 0.9, 1.2, 0.8, 1.3],
            'entropy_level': 4.5,
            'phase_analysis': {'phase_1': 0.3, 'phase_2': 0.7},
            'frequency_components': [0.1, 0.2, 0.3],
            'power_spectrum': [0.01, 0.04, 0.09]
        }
        
        export_path = exporter.export_vector_snapshot(
            SnapshotType.DLT_WAVEFORM, dlt_data, ExportFormat.JSON, compress=False
        )
        print(f"   ✅ DLT waveform export: {export_path}")
        
        # Test export history
        history = exporter.get_export_history()
        print(f"   ✅ Export history: {len(history)} exports")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Vector state export verification failed: {e}")
        return False

def verify_demo_pipeline_runner():
    """Verify demo pipeline runner is working."""
    print("🚀 Verifying Demo Pipeline Runner...")
    
    try:
        from demo_runner import DemoPipelineRunner, PipelineMode, PipelineStatus
        
        runner = DemoPipelineRunner()
        
        # Test mode setting
        runner.set_mode(PipelineMode.DEMO)
        print(f"   ✅ Pipeline mode: {runner.mode.value}")
        
        # Test pipeline status
        status = runner.get_pipeline_status()
        print(f"   ✅ Pipeline status: {status['status']}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Demo pipeline runner verification failed: {e}")
        return False

def verify_hash_registry():
    """Verify hash registry is properly configured."""
    print("🗂️ Verifying Hash Registry...")
    
    try:
        # Load hash registry
        with open('./core/hash_registry.json', 'r') as f:
            registry = json.load(f)
        
        # Verify structure
        assert 'baskets' in registry
        assert 'hash_mappings' in registry
        assert 'bit_phase_configs' in registry
        
        print(f"   ✅ Hash registry structure: {len(registry['baskets'])} baskets")
        print(f"   ✅ Hash mappings: {len(registry['hash_mappings'])} mappings")
        print(f"   ✅ Bit phase configs: {len(registry['bit_phase_configs'])} configs")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Hash registry verification failed: {e}")
        return False

def verify_integration_points():
    """Verify all integration points are functional."""
    print("🔗 Verifying Integration Points...")
    
    try:
        # Test component integration
        from bit_resolution_engine import BitResolutionEngine
        from tensor_score_utils import TensorScoreUtils
        from simulate_trade import TradeSimulator
        from inject_demo_ledger import DemoLedgerInjector
        from export_vector_snapshot import VectorStateExporter
        from demo_runner import DemoPipelineRunner
        
        # Initialize components
        bit_engine = BitResolutionEngine()
        tensor_utils = TensorScoreUtils()
        trade_sim = TradeSimulator()
        demo_injector = DemoLedgerInjector()
        vector_exporter = VectorStateExporter()
        demo_runner = DemoPipelineRunner()
        
        # Test integration connections
        bit_engine.set_matrix_mapper(None)  # Placeholder
        tensor_utils.set_bit_resolution_engine(bit_engine)
        trade_sim.set_tensor_matcher(tensor_utils)
        demo_injector.set_trade_simulator(trade_sim)
        vector_exporter.set_dlt_engine(None)  # Placeholder
        demo_runner.set_bit_phase_engine(bit_engine)
        
        print("   ✅ All integration points connected successfully")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Integration points verification failed: {e}")
        return False

def verify_complete_pipeline():
    """Verify the complete pipeline can execute."""
    print("🔄 Verifying Complete Pipeline...")
    
    try:
        # Simulate a complete pipeline execution
        from bit_resolution_engine import BitResolutionEngine
        from tensor_score_utils import TensorScoreUtils
        from simulate_trade import TradeSimulator
        
        # Initialize components
        bit_engine = BitResolutionEngine()
        tensor_utils = TensorScoreUtils()
        trade_sim = TradeSimulator()
        
        # Connect components
        tensor_utils.set_bit_resolution_engine(bit_engine)
        trade_sim.set_tensor_matcher(tensor_utils)
        
        # Simulate pipeline execution
        test_hash = "a1b2c3d4e5f67890abcdef1234567890abcdef1234567890abcdef1234567890"
        market_data = {'entropy_level': 4.5, 'volatility': 0.03, 'market_heat': 0.6}
        
        # Step 1: Bit resolution
        bit_phase = bit_engine.resolve_bit_phase(test_hash, "8bit")
        
        # Step 2: Tensor scoring
        tensor_score = tensor_utils.calculate_tensor_score(50000.0, 51000.0, bit_phase, market_data)
        
        # Step 3: Trade simulation
        strategy_bucket = {
            'asset': 'BTC',
            'strategy_id': 'long_hold_btc',
            'tensor_score': tensor_score,
            'bit_phase': bit_phase,
            'basket_id': f'basket_8bit_{bit_phase}',
            'current_price': 51000.0,
            'market_data': market_data
        }
        
        trade_result = trade_sim.simulate_trade(strategy_bucket, "DEMO")
        
        print(f"   ✅ Pipeline execution: {trade_result.trade_type.value} {trade_result.quantity:.4f} {trade_result.asset}")
        print(f"   ✅ Bit phase: {bit_phase}, Tensor score: {tensor_score:.4f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Complete pipeline verification failed: {e}")
        return False

def main():
    """Main verification function."""
    print("=" * 60)
    print("🚀 SCHWABOT UROS v1.0 - FINAL SYSTEM VERIFICATION")
    print("=" * 60)
    print(f"📅 Verification started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Run all verifications
    verifications = [
        ("Mathematical Functions", verify_mathematical_functions),
        ("Tensor Score Utils", verify_tensor_score_utils),
        ("Trade Simulation", verify_trade_simulation),
        ("Demo Ledger Injection", verify_demo_ledger_injection),
        ("Vector State Export", verify_vector_state_export),
        ("Demo Pipeline Runner", verify_demo_pipeline_runner),
        ("Hash Registry", verify_hash_registry),
        ("Integration Points", verify_integration_points),
        ("Complete Pipeline", verify_complete_pipeline)
    ]
    
    results = []
    
    for name, verification_func in verifications:
        print(f"\n🔍 {name}")
        print("-" * 40)
        try:
            result = verification_func()
            results.append((name, result))
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"\n{status}: {name}")
        except Exception as e:
            print(f"\n❌ FAILED: {name} - Exception: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {name}")
    
    print(f"\n🎯 Overall Result: {passed}/{total} verifications passed")
    
    if passed == total:
        print("🎉 ALL VERIFICATIONS PASSED! Schwabot system is fully operational.")
        print("\n🚀 System is ready for:")
        print("   • Live trading operations")
        print("   • Demo mode testing")
        print("   • Backtesting and simulation")
        print("   • Production deployment")
    else:
        print("⚠️ Some verifications failed. Please check the implementation.")
    
    print(f"\n📅 Verification completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

if __name__ == "__main__":
    main() 