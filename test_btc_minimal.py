#!/usr/bin/env python3
"""
Minimal test script for BTC processor components.
Tests individual components without importing problematic dependencies.
"""

import sys
import os
import numpy as np
import time
import collections

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_entropic_vectorizer_direct():
    """Test EntropicVectorizer functionality directly."""
    try:
        # Import only the specific module
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "core"))
        from entropic_vectorizer import EntropicVectorizer

        config = {"enabled": True, "output_bits": 16}
        ev = EntropicVectorizer(config)

        # Test with sample data
        class_id, risk_scalar, xor_drift = ev.build_strategy_vec(
            "test_block_hash", "test_price_hash", b"test_seed"
        )

        print(
            f"✅ EntropicVectorizer: class_id={class_id}, risk={risk_scalar:.3f}, xor_drift={xor_drift:.3f}"
        )
        return True
    except Exception as e:
        print(f"❌ EntropicVectorizer test failed: {e}")
        return False


def test_triplet_harmony_direct():
    """Test TripletHarmony functionality directly."""
    try:
        from triplet_harmony import TripletHarmony

        config = {"enabled": True, "coherence_threshold": 0.85}
        th = TripletHarmony(config)

        # Test with sample vectors
        test_vectors = collections.deque(
            [
                np.array([1.0, 2.0, 3.0]),
                np.array([1.1, 2.1, 3.1]),
                np.array([1.2, 2.2, 3.2]),
            ]
        )

        is_harmonic, coherence, hash_val = th.check_harmony(test_vectors)

        print(
            f"✅ TripletHarmony: harmonic={is_harmonic}, coherence={coherence:.3f}, hash={hash_val[:16]}..."
        )
        return True
    except Exception as e:
        print(f"❌ TripletHarmony test failed: {e}")
        return False


def test_drift_shells_direct():
    """Test DriftShells functionality directly."""
    try:
        from drift_shells import DriftShells

        config = {
            "enable_fractal_lock": True,
            "shell_layers": 6,
            "delta_n_thresholds": {"lock": 0.001, "reset": 0.015},
        }
        ds = DriftShells(config)

        # Test with sample Q vector
        test_q = np.array([1.0, 2.0, 3.0, 4.0])
        test_q = test_q / np.linalg.norm(test_q)  # Normalize

        result = ds.probe_drift({"shell_id": 0, "Q": test_q})

        print(
            f"✅ DriftShells: status={result['status']}, delta_n={result['delta_n']:.6f}"
        )
        return True
    except Exception as e:
        print(f"❌ DriftShells test failed: {e}")
        return False


def test_memory_backlog_direct():
    """Test MemoryBacklog functionality directly."""
    try:
        from memory_backlog import MemoryBacklog

        config = {
            "enabled": True,
            "backlog_depth": {"short_term": 96, "mid_term": 672, "long_term": 8760},
        }
        mb = MemoryBacklog(config)

        # Test adding a profit vector
        test_vector = {"class": 1, "risk": 0.5, "rho": 1.2, "timestamp": time.time()}
        mb.add_profit_vector(test_vector, "short_term")
        summary = mb.get_backlog_summary()

        print(f"✅ MemoryBacklog: short_term_count={summary['short_term_count']}")
        return True
    except Exception as e:
        print(f"❌ MemoryBacklog test failed: {e}")
        return False


def test_gpu_accelerator_direct():
    """Test GPUAccelerator functionality directly."""
    try:
        from gpu_accelerator import GPUAccelerator

        config = {"enabled": True, "provider": "numpy"}
        gpu = GPUAccelerator(config)

        # Test SHA256 projection
        test_data = b"test_data_for_hashing"
        hash_result = gpu.sha256_projection(test_data)

        print(
            f"✅ GPUAccelerator: GPU_available={gpu.is_gpu_available()}, hash={hash_result[:16]}..."
        )
        return True
    except Exception as e:
        print(f"❌ GPUAccelerator test failed: {e}")
        return False


def test_asrl_direct():
    """Test AutonomicStrategyReflexLayer functionality directly."""
    try:
        from integrators.autonomic_strategy_reflex_layer import (
            AutonomicStrategyReflexLayer,
        )

        config = {
            "alpha": 0.4,
            "beta": 0.3,
            "gamma": 0.3,
            "ur_threshold_mid": 0.3,
            "ur_threshold_high": 0.6,
        }
        asrl = AutonomicStrategyReflexLayer(config)

        # Test unified reflex score calculation
        ur_score = asrl.compute_unified_reflex_score(0.5, 0.3, 0.2)
        weights = asrl.adjust_strategy_weights(ur_score)

        print(f"✅ ASRL: ur_score={ur_score:.3f}, weights={weights}")
        return True
    except Exception as e:
        print(f"❌ ASRL test failed: {e}")
        return False


def test_chain_ws_direct():
    """Test chain_ws module directly."""
    try:
        from feeds.chain_ws import BlockEvent

        # Test BlockEvent creation
        block_event = BlockEvent(
            height=800000,
            hash="test_hash_1234567890abcdef",
            timestamp=1234567890.0,
            interval=600.0,
            size=1000000,
            weight=4000000,
            fee_rate=5.0,
            ts=1234567890.0,
        )

        print(f"✅ ChainWS: BlockEvent created for height {block_event.height}")
        return True
    except Exception as e:
        print(f"❌ ChainWS test failed: {e}")
        return False


def test_stratum_sniffer_direct():
    """Test stratum_sniffer module directly."""
    try:
        from feeds.stratum_sniffer import ShareEvent

        # Test ShareEvent creation
        share_event = ShareEvent(pool="test_pool", diff=1000.0, timestamp=time.time())

        print(f"✅ StratumSniffer: ShareEvent created for pool {share_event.pool}")
        return True
    except Exception as e:
        print(f"❌ StratumSniffer test failed: {e}")
        return False


def main():
    """Run all component tests."""
    print("🚀 Starting Minimal BTC Component Test Suite")
    print("=" * 50)

    # Change to core directory for imports
    original_cwd = os.getcwd()
    os.chdir(os.path.join(original_cwd, "core"))

    tests = [
        ("EntropicVectorizer", test_entropic_vectorizer_direct),
        ("TripletHarmony", test_triplet_harmony_direct),
        ("DriftShells", test_drift_shells_direct),
        ("MemoryBacklog", test_memory_backlog_direct),
        ("GPUAccelerator", test_gpu_accelerator_direct),
        ("ASRL", test_asrl_direct),
        ("ChainWS", test_chain_ws_direct),
        ("StratumSniffer", test_stratum_sniffer_direct),
    ]
    passed = 0
    total = len(tests)

    for name, test_func in tests:
        print(f"\n--- Testing {name} ---")
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {name} test failed")
        except Exception as e:
            print(f"❌ {name} test failed with exception: {e}")

    # Restore original directory
    os.chdir(original_cwd)

    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! BTC processor components are working correctly.")
    else:
        print("⚠️  Some tests failed. Check the errors above.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
