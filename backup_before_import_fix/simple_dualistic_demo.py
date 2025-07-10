import logging
import traceback
from datetime import datetime

import numpy as np

from core.advanced_tensor_algebra import UnifiedTensorAlgebra

#!/usr/bin/env python3
"""
Simple Dualistic Profit Vectorization Demo
==========================================

A focused demonstration of Schwabot's bit-form tensor flip matrices.'
"""


# Simple direct import


def main():
    """Main demonstration function."""

    print("🤖 Schwabot - Bit-Form Tensor Flip Matrix Demo")
    print("=" * 60)

    # Initialize the tensor algebra system
    print("\n🔧 Initializing Unified Tensor Algebra...")
    tensor_algebra = UnifiedTensorAlgebra()

    # Create realistic market data
    market_data = {}
        "price": 65432.10,
        "previous_price": 64890.50,
        "volume": 2500,
        "volatility": 0.18,  # 18% volatility
        "liquidity_depth": 8500,
        "timestamp": datetime.now().timestamp(),
    }

    print("\n📊 Market Conditions:")
    print(f"  Price: ${market_data['price']:,.2f}")
    print(f"  Previous: ${market_data['previous_price']:,.2f}")
    print()
        f"  Change: {((market_data['price'] / market_data['previous_price']) - 1) * 100:+.2f}%"
    )
    print(f"  Volume: {market_data['volume']:,}")
    print(f"  Volatility: {market_data['volatility']:.1%}")
    print(f"  Liquidity: {market_data['liquidity_depth']:,}")

    # Execute dualistic profit vectorization
    print("\n🧮 Executing Dualistic Profit Vectorization...")
    print("  (Pure mathematical decision-making through bit-form tensor flip, matrices)")

    consensus_result = tensor_algebra.execute_dualistic_profit_vectorization()
        market_data
    )

    # Display results
    print("\n🎯 Mathematical Decision Results:")
    print(f"  Execution Signal: {consensus_result.execution_signal.upper()}")
    print(f"  Consensus Confidence: {consensus_result.consensus_confidence:.3f}")

    # Show the profit vector breakdown
    vector = consensus_result.final_profit_vector
    print("\n📐 Profit Vector Analysis:")
    print(f"  Price Direction: {vector[0]:+.3f} (positive = bullish)")
    print(f"  Time Direction: {vector[1]:+.3f} (temporal, momentum)")
    print(f"  Risk Direction: {vector[2]:+.3f} (risk-adjusted, factor)")
    print(f"  Vector Magnitude: {np.linalg.norm(vector):.3f}")

    # Show mathematical proof
    proof = consensus_result.mathematical_proof
    print("\n🔬 Mathematical Proof:")
    print(f"  Matrix Count: {proof['matrix_count']}")
    print(f"  Total Consensus Weight: {proof['total_consensus_weight']:.3f}")
    print(f"  Mathematical Certainty: {proof['mathematical_certainty']:.3f}")

    # Show flip state distribution
    flip_dist = proof["flip_state_distribution"]
    print("\n⚡ Flip State Distribution:")
    for state, count in flip_dist.items():
        if count > 0:
            print(f"  {state.replace('_', ' ').title()}: {count} matrices")

    # Test multiple scenarios
    print("\n🔬 Testing Multiple Market Scenarios...")

    scenarios = []
        {}
            "name": "Bull Market",
            "price": 67000,
            "previous_price": 65000,
            "volatility": 0.12,
        },
        {}
            "name": "Bear Market",
            "price": 62000,
            "previous_price": 65000,
            "volatility": 0.25,
        },
        {}
            "name": "Sideways",
            "price": 65100,
            "previous_price": 65000,
            "volatility": 0.8,
        },
        {}
            "name": "High Volatility",
            "price": 66500,
            "previous_price": 63500,
            "volatility": 0.35,
        },
    ]

    for scenario in scenarios:
        test_data = market_data.copy()
        test_data.update(scenario)

        result = tensor_algebra.execute_dualistic_profit_vectorization(test_data)
        certainty = result.mathematical_proof.get("mathematical_certainty", 0)

        print()
            f"  {scenario['name']:15} → {result.execution_signal:5} "
            f"(conf: {result.consensus_confidence:.2f}, cert: {certainty:.2f})"
        )

    print("\n" + "=" * 60)
    print("✅ Demonstration Complete!")
    print("💡 Key Achievement: Pure mathematical decision-making through")
    print("   bit-form tensor flip matrices and dualistic state resolution.")
    print("🧮 This represents mathematical intelligence rather than learned AI.")


if __name__ == "__main__":
    # Suppress some logging for cleaner output
    logging.getLogger("core.advanced_tensor_algebra").setLevel(logging.ERROR)

    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during demonstration: {e}")

        traceback.print_exc()
