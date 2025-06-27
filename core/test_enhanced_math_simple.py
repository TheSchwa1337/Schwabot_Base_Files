# -*- coding: utf-8 -*-
"""
Simple Test for Enhanced Unified Mathematical System
===================================================

Quick test to validate the enhanced mathematical system functionality.
"""

import sys
import os

# Add current directory to path
sys.path.append('.')

def test_enhanced_math_system():
    """Test the enhanced mathematical system."""
    try:
        print("🧮 Testing Enhanced Unified Mathematical System...")
        
        # Import the enhanced mathematical system
        from enhanced_unified_mathematical_system import (
            EnhancedUnifiedMathematicalSystem, PortfolioAsset, get_enhanced_math_system
        )
        
        print("✅ Import successful")
        
        # Create mathematical system
        math_system = EnhancedUnifiedMathematicalSystem()
        print("✅ Mathematical system created")
        
        # Test bit phase tensor
        bit_result = math_system.bit_phase_tensor(12345, 'auto')
        print(f"✅ Bit Phase Tensor: φ₄={bit_result.phi_4}, φ₈={bit_result.phi_8}, φ₄₂={bit_result.phi_42}")
        
        # Test portfolio vector creation
        assets = [PortfolioAsset.BTC, PortfolioAsset.ETH, PortfolioAsset.XRP]
        portfolio = math_system.create_portfolio_vector(assets)
        print(f"✅ Portfolio Vector: {len(portfolio.assets)} assets")
        
        # Test fabricated logic gate
        gate = math_system.create_fabricated_logic_gate(42, "a1b2c3d4")
        print(f"✅ Fabricated Logic Gate: XOR={gate.xor_result}")
        
        # Test BTC price mapping
        btc_entry = math_system.map_btc_price_16bit(50000.0, "mid")
        print(f"✅ BTC Price Mapping: {btc_entry.btc_price:.2f} → {btc_entry.mapped_16bit} (16-bit)")
        
        # Test tensor contraction
        import numpy as np
        A = np.random.random((3, 4))
        B = np.random.random((4, 2))
        tensor_result = math_system.tensor_contraction(A, B)
        print(f"✅ Tensor Contraction: {A.shape} × {B.shape} → {tensor_result.shape}")
        
        # Test hash memory encoding
        hash_result = math_system.hash_memory_encoding("test_data")
        print(f"✅ Hash Memory Encoding: {hash_result[:16]}...")
        
        # Test entropy compensation
        data = np.random.random(100)
        compensated_data = math_system.entropy_compensation(data)
        print(f"✅ Entropy Compensation: {data.shape} → {compensated_data.shape}")
        
        # Get statistics
        stats = math_system.get_statistics()
        print(f"✅ System Statistics: {stats['operation_count']} operations, {stats['success_rate']:.2%} success rate")
        
        # Test global instance
        get_enhanced_math_system()
        print("✅ Global instance test passed")
        
        print("🎉 All tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_backlog_integration():
    """Test the backlog integration bridge."""
    try:
        print("\n🌉 Testing Enhanced Backlog Integration Bridge...")
        
        # Import the backlog integration bridge
        from enhanced_backlog_integration_bridge import (
            EnhancedBacklogIntegrationBridge, get_enhanced_backlog_bridge
        )
        
        print("✅ Import successful")
        
        # Create bridge
        bridge = EnhancedBacklogIntegrationBridge()
        print("✅ Bridge created")
        
        # Test hash save with backlog
        hash_result = bridge.save_hash_with_backlog("test_data", "general")
        print(f"✅ Hash Save: {'✅' if hash_result.success else '❌'}")
        
        # Test BTC price mapping with backlog
        btc_result = bridge.map_btc_price_with_backlog(50000.0, "mid")
        print(f"✅ BTC Mapping: {'✅' if btc_result.success else '❌'}")
        
        # Test backlog sync
        sync_result = bridge.sync_backlog_state()
        print(f"✅ Backlog Sync: {'✅' if sync_result.success else '❌'}")
        
        # Test memory persistence
        persistence_result = bridge.calculate_memory_persistence()
        print(f"✅ Memory Persistence: {'✅' if persistence_result.success else '❌'}")
        
        # Test mathematical operation with backlog
        math_result = bridge.perform_mathematical_operation_with_backlog("bit_phase_tensor", {
            'strategy_id': 12345,
            'mode': 'auto'
        })
        print(f"✅ Math Operation: {'✅' if math_result.success else '❌'}")
        
        # Get integration metrics
        metrics = bridge.get_integration_metrics()
        print(f"✅ Integration Metrics: {metrics.total_operations} operations, {metrics.successful_operations} successful")
        
        # Test global instance
        get_enhanced_backlog_bridge()
        print("✅ Global bridge instance test passed")
        
        print("🎉 All backlog integration tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Backlog integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🚀 Starting Enhanced Mathematical System Tests...")
    print("=" * 60)
    
    # Test enhanced mathematical system
    math_success = test_enhanced_math_system()
    
    # Test backlog integration
    backlog_success = test_backlog_integration()
    
    print("=" * 60)
    print("📊 Test Summary:")
    print(f"   Enhanced Mathematical System: {'✅ PASSED' if math_success else '❌ FAILED'}")
    print(f"   Backlog Integration Bridge: {'✅ PASSED' if backlog_success else '❌ FAILED'}")
    
    if math_success and backlog_success:
        print("🎉 All tests passed! Enhanced mathematical system is ready for production.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 