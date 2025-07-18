#!/usr/bin/env python3
"""
Test script for quantum strategy functionality
"""

try:
    from mathlib.quantum_strategy import QuantumStrategyEngine
    print("✓ Successfully imported QuantumStrategyEngine")
    
    # Initialize engine
    engine = QuantumStrategyEngine()
    print("✓ Quantum Strategy Engine initialized successfully")
    
    # Test superposition strategy creation
    assets = ["BTC", "ETH", "ADA", "DOT"]
    strategy = engine.create_superposition_strategy("test_super", assets)
    print(f"✓ Created superposition strategy: {strategy.strategy_id}")
    
    # Test strategy execution
    result = engine.execute_superposition_strategy("test_super")
    print(f"✓ Executed strategy with result: {result.get('asset', 'N/A')}")
    
    # Test statistics
    stats = engine.get_quantum_statistics()
    print(f"✓ Retrieved statistics: {stats['total_strategies']} strategies")
    
    print("\n🎉 All quantum strategy tests passed!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc() 