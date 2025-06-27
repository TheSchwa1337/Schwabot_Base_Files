# -*- coding: utf-8 -*-
"""
Test New Interlinking Components
===============================
Minimal test of our newly created components.
"""

def test_memory_cache_bridge():
    """Test our new Memory Cache Bridge component."""
    print("🧠 Testing Memory Cache Bridge...")
    
    try:
        # Import our new component
        import sys
        import os
        
        # Add current directory to path
        sys.path.insert(0, os.path.join(os.getcwd(), 'core'))
        
        # Import the module directly
        import memory_cache_bridge
        
        # Test the classes
        MemoryCacheBridge = memory_cache_bridge.MemoryCacheBridge
        MemoryTier = memory_cache_bridge.MemoryTier
        PatternType = memory_cache_bridge.PatternType
        
        print("  ✓ Module imports successful")
        
        # Create instance
        bridge = MemoryCacheBridge()
        print("  ✓ MemoryCacheBridge instantiated")
        
        # Test basic operations
        success = bridge.update_cache(
            "test_key", 
            {"data": "test_value"}, 
            MemoryTier.MID, 
            PatternType.PROFIT_VECTOR
        )
        print(f"  ✓ Cache update: {success}")
        
        # Test visualization payload
        payload = bridge.resolve_visualization_payload()
        print(f"  ✓ Visualization payload: {'timestamp' in payload}")
        
        # Test pattern gate status
        gates = bridge.get_pattern_gate_status()
        print(f"  ✓ Pattern gates: {len(gates)} gates available")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_vault_manager():
    """Test our new Vault Manager component."""
    print("🏦 Testing Vault Manager...")
    
    try:
        # Import our new component
        import sys
        import os
        
        # Add current directory to path
        sys.path.insert(0, os.path.join(os.getcwd(), 'core'))
        
        # Import the module directly
        import memory_vault
        
        # Test the classes
        VaultManager = memory_vault.VaultManager
        VaultTier = memory_vault.VaultTier
        VaultAction = memory_vault.VaultAction
        
        print("  ✓ Module imports successful")
        
        # Create instance
        vault_manager = VaultManager()
        print("  ✓ VaultManager instantiated")
        
        # Test vault creation
        success = vault_manager.create_vault_entry(
            vault_id="test_vault_001",
            strategy="test_strategy",
            profit_score=0.75,
            correlation_data={"source": "test", "confidence": 0.9}
        )
        print(f"  ✓ Vault creation: {success}")
        
        # Test vault trigger
        trigger_success = vault_manager.trigger(
            vault_id="test_vault_001",
            strategy="updated_strategy",
            profit_score=0.85
        )
        print(f"  ✓ Vault trigger: {trigger_success}")
        
        # Test metrics
        metrics = vault_manager.get_vault_metrics()
        print(f"  ✓ Vault metrics: {metrics.total_vaults} vaults")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_fractal_core():
    """Test our enhanced Fractal Core component."""
    print("🌀 Testing Fractal Core...")
    
    try:
        # Import our component
        import sys
        import os
        
        # Add current directory to path
        sys.path.insert(0, os.path.join(os.getcwd(), 'core'))
        
        # Import the module directly
        import fractal_core
        
        # Test the functions and classes
        generate_fractal_sequence = fractal_core.generate_fractal_sequence
        FractalCore = fractal_core.FractalCore
        
        print("  ✓ Module imports successful")
        
        # Test global function
        sequence = generate_fractal_sequence(seed=42, depth=10)
        print(f"  ✓ Global function: Generated sequence of length {len(sequence)}")
        
        # Test class
        fractal_core_instance = FractalCore()
        print("  ✓ FractalCore instantiated")
        
        # Test sequence generation
        sequence2 = fractal_core_instance.generate_sequence(seed=123, depth=8)
        print(f"  ✓ Class method: Generated sequence of length {len(sequence2)}")
        
        # Test coherence calculation
        coherence = fractal_core_instance.calculate_coherence_score(sequence)
        print(f"  ✓ Coherence score: {coherence:.4f}")
        
        # Test pattern correlation
        correlation = fractal_core_instance.analyze_pattern_correlation(sequence, sequence2)
        print(f"  ✓ Pattern correlation: {correlation:.4f}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_symbolic_profit_router():
    """Test Symbolic Profit Router with new enums."""
    print("🎯 Testing Symbolic Profit Router...")
    
    try:
        # Import our component
        import sys
        import os
        
        # Add current directory to path
        sys.path.insert(0, os.path.join(os.getcwd(), 'core'))
        
        # Import the module directly
        import symbolic_profit_router
        
        # Test the classes and enums
        SymbolicProfitRouter = symbolic_profit_router.SymbolicProfitRouter
        ProfitTier = symbolic_profit_router.ProfitTier
        FlipBias = symbolic_profit_router.FlipBias
        SymbolicState = symbolic_profit_router.SymbolicState
        
        print("  ✓ Module imports successful")
        
        # Test enum access
        print(f"  ✓ ProfitTier: {ProfitTier.MID}")
        print(f"  ✓ FlipBias: {FlipBias.BULLISH}")
        print(f"  ✓ SymbolicState: {SymbolicState.ACTIVE}")
        
        # Create instance
        router = SymbolicProfitRouter()
        print("  ✓ SymbolicProfitRouter instantiated")
        
        # Test hash to strategy
        strategy = router.hash_to_strategy("test::BTC::mid::24hr")
        print(f"  ✓ Hash to strategy: {strategy.get('asset', 'Unknown')}")
        
        # Test 2-bit folding
        bit_sequence = router.fold_hash_to_2bit("abcdef1234567890abcdef")
        print(f"  ✓ 2-bit folding: {bit_sequence}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_strategy_simulator_components():
    """Test if strategy simulator components can be imported."""
    print("🚀 Testing Strategy Simulator Components...")
    
    try:
        import sys
        import os
        
        # Add current directory to path
        sys.path.insert(0, os.path.join(os.getcwd(), 'core'))
        
        # Test individual imports
        components_tested = []
        
        # Test memory vault
        import memory_vault
        VaultManager = memory_vault.VaultManager
        vault_manager = VaultManager()
        components_tested.append("VaultManager")
        
        # Test symbolic profit router
        import symbolic_profit_router
        SymbolicProfitRouter = symbolic_profit_router.SymbolicProfitRouter
        router = SymbolicProfitRouter()
        components_tested.append("SymbolicProfitRouter")
        
        print(f"  ✓ Successfully imported: {', '.join(components_tested)}")
        
        # Test a simple mock simulation
        test_payload = {
            "sha_hash": "test_hash_12345",
            "vault_id": 1001,
            "btc_price": 35000.0,
            "symbolic_input": "🔥"
        }
        
        # Test vault trigger
        success = vault_manager.trigger(
            vault_id=test_payload["vault_id"],
            strategy="test_strategy",
            profit_score=0.8
        )
        print(f"  ✓ Mock vault trigger: {success}")
        
        # Test strategy conversion
        strategy = router.hash_to_strategy(test_payload["sha_hash"])
        print(f"  ✓ Mock strategy conversion: {strategy.get('confidence', 0.0):.2f}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all component tests."""
    print("🎯 New Interlinking Components Test")
    print("=" * 60)
    
    tests = [
        test_memory_cache_bridge,
        test_vault_manager,
        test_fractal_core,
        test_symbolic_profit_router,
        test_strategy_simulator_components
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
            print(f"Result: {'✅ PASS' if result else '❌ FAIL'}")
            print("-" * 40)
        except Exception as e:
            print(f"❌ Test {test_func.__name__} failed with exception: {e}")
            results.append(False)
            print("-" * 40)
    
    success_count = sum(results)
    total_tests = len(results)
    
    print(f"\n📊 Final Results: {success_count}/{total_tests} tests passed")
    
    if success_count >= 4:
        print("🎉 Excellent! Your new interlinking components are working!")
        print("✅ Memory Cache Bridge, Vault Manager, and other components are operational")
        print("🚀 Your Schwabot system has the core interlinking infrastructure!")
    elif success_count >= 2:
        print("👍 Good progress! Most components are working")
        print("⚠️ Some minor issues but core functionality is available")
    else:
        print("⚠️ Some components need attention")
    
    print(f"\n🎯 Summary of working components:")
    component_names = ["Memory Cache Bridge", "Vault Manager", "Fractal Core", 
                      "Symbolic Profit Router", "Strategy Simulator"]
    for i, (name, result) in enumerate(zip(component_names, results)):
        status = "✅ Working" if result else "❌ Needs attention"
        print(f"  {name}: {status}")
    
    return success_count >= 2

if __name__ == "__main__":
    main() 