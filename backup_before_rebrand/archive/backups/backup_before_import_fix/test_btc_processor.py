import asyncio
import logging
import os
import sys
import traceback

from core.drift_shells import DriftShells
from core.entropic_vectorizer import EntropicVectorizer
from core.gpu_accelerator import GPUAccelerator
from core.memory_backlog import MemoryBacklog
from core.multi_bit_btc_processor import MultiBitBTCProcessor
from core.triplet_harmony import TripletHarmony

#!/usr/bin/env python3
"""
Test script for the MultiBitBTCProcessor and its components.
This script tests the BTC processor in isolation to avoid core module import issues.
"""


# Add the current directory to the path so we can import core modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig()
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


async def test_btc_processor():
    """Test the MultiBitBTCProcessor with simulated data."""

    try:
        # Import the processor directly
            AutonomicStrategyReflexLayer,
        )

        print("✅ All core modules imported successfully")

        # Test individual components
        print("\n--- Testing Individual Components ---")

        # Test EntropicVectorizer
        ev_config = {"enabled": True, "output_bits": 16}
        ev = EntropicVectorizer(ev_config)
        class_id, risk_scalar, xor_drift = ev.build_strategy_vec()
            "test_hash", "test_price_hash", b"seed"
        )
        print()
            f"✅ EntropicVectorizer: class_id={class_id}, risk={risk_scalar:.3f}, xor_drift={xor_drift:.3f}"
        )

        # Test TripletHarmony
        th_config = {"enabled": True, "coherence_threshold": 0.85}
        TripletHarmony(th_config)
        print()
            f"✅ TripletHarmony initialized with threshold {th_config['coherence_threshold']}"
        )

        # Test DriftShells
        ds_config = {"enable_fractal_lock": True, "shell_layers": 6}
        DriftShells(ds_config)
        print(f"✅ DriftShells initialized with {ds_config['shell_layers']} layers")

        # Test MemoryBacklog
        mb_config = {}
            "enabled": True,
            "backlog_depth": {"short_term": 96, "mid_term": 672, "long_term": 8760},
        }
        MemoryBacklog(mb_config)
        print()
            f"✅ MemoryBacklog initialized with short_term depth {mb_config['backlog_depth']['short_term']}"
        )

        # Test GPUAccelerator
        gpu_config = {"enabled": True, "provider": "numpy"}
        gpu = GPUAccelerator(gpu_config)
        print(f"✅ GPUAccelerator initialized, GPU available: {gpu.is_gpu_available()}")

        # Test ASRL
        asrl_config = {"alpha": 0.4, "beta": 0.3, "gamma": 0.3}
        AutonomicStrategyReflexLayer(asrl_config)
        print()
            f"✅ AutonomicStrategyReflexLayer initialized with alpha={asrl_config['alpha']}"
        )

        # Test the main processor
        print("\n--- Testing MultiBitBTCProcessor ---")

        # Create processor with default config
        processor = MultiBitBTCProcessor()
        print("✅ MultiBitBTCProcessor initialized successfully")
        print(f"   - Window size: {processor.window_size}")
        print(f"   - Sigma threshold: {processor.sigma_threshold}")
        print(f"   - Execution enabled: {processor.execution_enabled}")

        # Test processing some simulated data
        print("\n--- Testing Data Processing ---")

        # Simulate some price/volume data
        test_prices = [65000.0, 65100.0, 65200.0, 65150.0, 65300.0]
        test_volumes = [100.0, 95.0, 110.0, 105.0, 120.0]

        for i, (price, volume) in enumerate(zip(test_prices, test_volumes)):
            is_allowed, profit_vector = await processor.process_btc_data(price, volume)

            if is_allowed and profit_vector:
                print(f"✅ Tick {i + 1}: Price=${price:.0f}, Volume={volume:.0f}")
                print(f"   - Class: {profit_vector['class']}")
                print(f"   - Risk: {profit_vector['risk']:.3f}")
                print(f"   - Rho: {profit_vector['rho']:.3f}")
                print(f"   - Coherence: {profit_vector['triplet_coherence']:.3f}")
                print()
                    f"   - U_r Score: {profit_vector['asrl_unified_reflex_score']:.3f}"
                )
                print(f"   - Lights: {profit_vector['lights']}")
            else:
                print(f"❌ Tick {i + 1}: Processing failed or not allowed")

            await asyncio.sleep(0.1)  # Small delay between ticks

        print("\n✅ All tests completed successfully!")

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("This might be due to missing dependencies. Check requirements.txt")
    except Exception as e:
        print(f"❌ Test failed with error: {e}")

        traceback.print_exc()


if __name__ == "__main__":
    print("🚀 Starting BTC Processor Test Suite")
    print("=" * 50)

    asyncio.run(test_btc_processor())

    print("\n" + "=" * 50)
    print("🏁 Test Suite Complete")
