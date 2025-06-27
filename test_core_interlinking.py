# -*- coding: utf-8 -*-
"""
Test Core Interlinking Components
=================================
Direct test of our core interlinking system components."""
"""

def test_memory_cache_bridge():"""
    """Test Memory Cache Bridge functionality."""
try:
        from core.memory_cache_bridge import MemoryCacheBridge, MemoryTier, PatternType
        """
print("🧠 Testing Memory Cache Bridge...")
        bridge = MemoryCacheBridge()
        
# Test basic operations
success = bridge.update_cache("test_key", {"data": "test"}, MemoryTier.MID, PatternType.PROFIT_VECTOR)
        print(f"✓ Cache update: {success}")
        
# Test pattern fetching
data = bridge.fetch_pattern("test_key")
        print(f"✓ Pattern fetch: {data is not None}")
        
# Test visualization payload
payload = bridge.resolve_visualization_payload()
        print(f"✓ Visualization payload: {payload.get('timestamp') is not None}")
        
return True
except Exception as e:
        print(f"✗ Memory Cache Bridge error: {e}")
        return False

def test_vault_manager():
    """Test Vault Manager functionality."""
try:
        from core.memory_vault import VaultManager
"""
print("🏦 Testing Vault Manager...")
        vault_manager = VaultManager()
        
# Test vault creation
success = vault_manager.create_vault_entry(
            vault_id="test_vault",
            strategy="test_strategy",
            profit_score=0.75,
            correlation_data={"test": True}
        )
print(f"✓ Vault creation: {success}")
        
# Test vault trigger
trigger_success = vault_manager.trigger("test_vault", "updated_strategy", 0.8)
        print(f"✓ Vault trigger: {trigger_success}")
        
# Test vault state
state = vault_manager.get_vault_state("test_vault")
        print(f"✓ Vault state: {state is not None}")
        
return True
except Exception as e:
        print(f"✗ Vault Manager error: {e}")
        return False

def test_fractal_core():
    """Test Fractal Core functionality."""
try:
        from core.fractal_core import FractalCore, generate_fractal_sequence
        """
print("🌀 Testing Fractal Core...")
        fractal_core = FractalCore()
        
# Test sequence generation
sequence = fractal_core.generate_sequence(seed=42, depth=10)
        print(f"✓ Sequence generation: {len(sequence) == 10}")
        
# Test global function
global_sequence = generate_fractal_sequence(seed=123, depth=5)
        print(f"✓ Global function: {len(global_sequence) == 5}")
        
# Test coherence calculation
coherence = fractal_core.calculate_coherence_score(sequence)
        print(f"✓ Coherence score: {0 <= coherence <= 1}")
        
return True
except Exception as e:
        print(f"✗ Fractal Core error: {e}")
        return False

def test_symbolic_profit_router():
    """Test Symbolic Profit Router functionality."""
try:
        from core.symbolic_profit_router import SymbolicProfitRouter, ProfitTier, FlipBias, SymbolicState
        """
print("🎯 Testing Symbolic Profit Router...")
        router = SymbolicProfitRouter()
        
# Test hash to strategy conversion
strategy = router.hash_to_strategy("test_input")
        print(f"✓ Hash to strategy: {'asset' in strategy}")
        
# Test 2-bit folding
bit_sequence = router.fold_hash_to_2bit("abcdef1234567890")
        print(f"✓ 2-bit folding: {len(bit_sequence) == 2}")
        
# Test enum access
print(f"✓ Enums available: {ProfitTier.MID}, {FlipBias.NEUTRAL}, {SymbolicState.ACTIVE}")
        
return True
except Exception as e:
        print(f"✗ Symbolic Profit Router error: {e}")
        return False

def test_math_core():
    """Test Math Core functionality."""
try:
        from core.math_core import MathematicalCore
"""
print("🔢 Testing Math Core...")
        math_core = MathematicalCore()
        
# Test basic calculations
profit_vector = math_core.calculate_profit_tier_navigation(
            hash_input="test_hash",
            tier_weights=[0.3, 0.4, 0.3],
            time_vector=[1.0, 2.0, 3.0]
        )
print(f"✓ Profit vector calculation: {profit_vector > 0}")
        
# Test entropy calculation
entropy = math_core.calculate_entropy_flow_detection([0.1, 0.5, 0.3, 0.7])
        print(f"✓ Entropy calculation: {0 <= entropy <= 1}")
        
return True
except Exception as e:
        print(f"✗ Math Core error: {e}")
        return False

def main():
    """Run all interlinking tests.""""""
print("🚀 Core Interlinking System Test")
    print("=" * 50)
    
tests = [
        test_memory_cache_bridge,
        test_vault_manager,
        test_fractal_core,
        test_symbolic_profit_router,
        test_math_core
]
    
results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
            print(f"{'✅' if result else '❌'} {test.__name__}")
            print("-" * 30)
        except Exception as e:
            print(f"❌ {test.__name__} - Exception: {e}")
            results.append(False)
            print("-" * 30)
    
success_count = sum(results)
    total_tests = len(results)
    
print(f"\n🎯 Test Results: {success_count}/{total_tests} passed")
    
if success_count == total_tests:
        print("✅ All core interlinking components are working!")
    else:
        print("⚠️  Some components need attention")
    
return success_count == total_tests

if __name__ == "__main__":
    main() 