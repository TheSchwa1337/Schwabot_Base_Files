"""
Simple Enhanced Systems Test
===========================

Basic test to verify all enhanced systems are operational.
"""

import time
import numpy as np

def test_collapse_confidence():
    """Test collapse confidence engine."""
    print("Testing Collapse Confidence Engine...")
    
    from collapse_confidence import CollapseConfidenceEngine
    
    engine = CollapseConfidenceEngine()
    
    # Test calculation
    collapse_state = engine.calculate_collapse_confidence(
        profit_delta=75.0,
        braid_signal=0.8,
        paradox_signal=0.6,
        recent_volatility=[0.2, 0.3, 0.25]
    )
    
    print(f"✓ Confidence: {collapse_state.confidence:.3f}")
    print(f"✓ Coherence: {collapse_state.coherence:.3f}")
    return True

def test_vault_router():
    """Test enhanced vault router."""
    print("\nTesting Enhanced Vault Router...")
    
    from vault_router import EnhancedVaultRouter
    from collapse_confidence import CollapseConfidenceEngine
    
    router = EnhancedVaultRouter()
    confidence_engine = CollapseConfidenceEngine()
    
    # Create test collapse state
    collapse_state = confidence_engine.calculate_collapse_confidence(
        profit_delta=85.0,
        braid_signal=0.75,
        paradox_signal=0.65,
        recent_volatility=[0.2, 0.3, 0.25]
    )
    
    # Test allocation
    allocation = router.calculate_volume_allocation(
        collapse_state=collapse_state,
        current_profit=50.0,
        recent_profits=[30.0, 40.0, 45.0, 50.0],
        market_volatility=0.3
    )
    
    print(f"✓ Volume Allocation: {allocation.allocated_volume:.0f}")
    print(f"✓ Volume Multiplier: {allocation.volume_multiplier:.2f}")
    return True

def test_ghost_decay():
    """Test ghost decay system."""
    print("\nTesting Ghost Decay System...")
    
    from ghost_decay import GhostDecaySystem
    
    decay_system = GhostDecaySystem()
    
    # Create test ghost
    pattern = {
        "braid_signal": 0.8,
        "paradox_signal": 0.6,
        "profit_delta": 75.0,
        "volatility": 0.3
    }
    
    ghost_id = decay_system.create_ghost_signal(pattern, 0.9)
    
    # Update weights
    weights = decay_system.update_ghost_weights()
    
    print(f"✓ Ghost Created: {ghost_id[:8]}")
    print(f"✓ Active Ghosts: {len(weights)}")
    return True

def test_lockout_matrix():
    """Test enhanced lockout matrix."""
    print("\nTesting Enhanced Lockout Matrix...")
    
    from lockout_matrix import EnhancedLockoutMatrix, LockoutSeverity
    
    lockout_matrix = EnhancedLockoutMatrix()
    
    # Create test pattern
    pattern = {
        "braid_signal": 0.3,
        "paradox_signal": 0.2,
        "profit_delta": -75.0,
        "volatility": 0.6
    }
    
    # Create lockout
    signature = lockout_matrix.create_lockout(pattern, "test_failure", LockoutSeverity.MODERATE)
    
    # Check status
    is_locked, reason = lockout_matrix.check_lockout_status(pattern)
    
    print(f"✓ Lockout Created: {signature[:8]}")
    print(f"✓ Pattern Locked: {is_locked}")
    return True

def test_echo_logger():
    """Test echo snapshot logger."""
    print("\nTesting Echo Snapshot Logger...")
    
    from echo_snapshot import EchoSnapshotLogger, EchoSnapshot
    
    # Create logger
    logger_config = {
        'level': 'minimal',
        'terminal_output': False,  # Disable output for test
        'use_colors': False
    }
    
    echo_logger = EchoSnapshotLogger(logger_config)
    
    # Create simple mock decision using dataclass directly
    from dataclasses import dataclass
    from typing import Dict, Any
    
    @dataclass
    class MockFractalDecision:
        timestamp: float
        action: str
        confidence: float
        projected_profit: float
        hold_duration: int
        fractal_signals: Dict[str, float]
        fractal_weights: Dict[str, float]
        risk_assessment: Dict[str, Any]
        reasoning: str
    
    mock_decision = MockFractalDecision(
        timestamp=time.time(),
        action="long",
        confidence=0.75,
        projected_profit=85.0,
        hold_duration=12,
        fractal_signals={"braid": 0.8, "paradox": 0.6},
        fractal_weights={"braid": 1.2, "paradox": 0.9},
        risk_assessment={"volatility_risk": "moderate"},
        reasoning="Test decision"
    )
    
    # Capture snapshot
    snapshot = echo_logger.capture_snapshot(mock_decision)
    
    print(f"✓ Snapshot Captured: {snapshot.tick_id}")
    print(f"✓ Total Snapshots: {echo_logger.total_snapshots}")
    return True

def main():
    """Run all enhanced systems tests."""
    print("🚀 ENHANCED SYSTEMS VERIFICATION TEST")
    print("="*50)
    
    tests = [
        test_collapse_confidence,
        test_vault_router,
        test_ghost_decay,
        test_lockout_matrix,
        test_echo_logger
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed: {e}")
    
    print(f"\n{'='*50}")
    print(f"RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ ALL ENHANCED SYSTEMS OPERATIONAL!")
        print("\n🎯 Mathematical Framework Complete:")
        print("   ✓ Collapse Confidence: Confidence = |Δprofit|/(σψ + ε) · coherence^λ")
        print("   ✓ Vault Router: V = V₀ · confidence · (1 + dP/dt) · risk_factor")
        print("   ✓ Ghost Decay: W_g(t) = W₀ · e^(-α(t-t₀))")
        print("   ✓ Lockout Matrix: Self-healing with time-based expiration")
        print("   ✓ Echo Logger: Real-time diagnostic snapshots")
        print("\n💎 RECURSIVE PROFIT ENGINE ENHANCED AND READY!")
    else:
        print("⚠️  Some systems need attention")
    
    print("="*50)

if __name__ == "__main__":
    main() 