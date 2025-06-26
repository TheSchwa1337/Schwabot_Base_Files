#!/usr/bin/env python3
"""
Ghost Strategy Integration Demo - Schwabot UROS v1.0
==================================================

Demonstrates integration of the ghost strategy engine into the main trading system.
Shows how to use BTC vector processing for real-time volatility-aware trading decisions.
"""

from core.btc_vector_processor import BTCVectorProcessor, GhostStrategyEngine
from core.ghost_signal_types import (
    GhostSignal, GhostArray, BTCVector,
    build_ghost_array, extract_volatility_window, validate_ghost_array
)
import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Any

# Add core to path
REPO_ROOT = Path(__file__).resolve().parent
CORE_PATH = REPO_ROOT / "core"
if str(CORE_PATH) not in sys.path:
    sys.path.insert(0, str(CORE_PATH))

# Import ghost components


class GhostStrategyIntegrationDemo:
    """Demonstrates ghost strategy integration with the trading system."""

    def __init__(self):
        self.engine = GhostStrategyEngine()
        self.signal_history = []

    def simulate_market_data(self, base_price: float = 50000.0,
                             num_signals: int = 20) -> List[GhostSignal]:
        """Simulate realistic market data for demonstration."""
        signals = []
        base_time = int(time.time())

        for i in range(num_signals):
            # Simulate realistic BTC price movements
            trend = 0.001 * i  # Gradual upward trend
            noise = 0.02 * (i % 3 - 1)  # Oscillating noise
            price = base_price * (1 + trend + noise)

            # Simulate volatility clustering (GARCH-like behavior)
            if i < 5:
                volatility = 0.02  # Low volatility start
            elif i < 10:
                volatility = 0.05  # Increasing volatility
            else:
                volatility = 0.08  # High volatility period

            # Simulate confidence based on signal consistency
            if i < 8:
                confidence = 0.9  # High confidence in stable period
            else:
                confidence = 0.7  # Lower confidence in volatile period

            signal = GhostSignal(
                asset="BTC",
                price=price,
                volatility=volatility,
                confidence=confidence,
                timestamp=base_time + i * 60  # 1-minute intervals
            )
            signals.append(signal)

        return signals

    def demonstrate_ghost_array_processing(self):
        """Demonstrate ghost array processing capabilities."""
        print("🔱 Ghost Array Processing Demonstration")
        print("=" * 50)

        # Generate test signals
        signals = self.simulate_market_data(num_signals=15)

        # Build ghost array
        ghost_array = build_ghost_array(signals)
        print(f"✅ Built ghost array: shape={ghost_array.shape}")

        # Validate array
        if validate_ghost_array(ghost_array):
            print("✅ Ghost array validation passed")
        else:
            print("❌ Ghost array validation failed")
            return

        # Extract volatility window
        volatility = extract_volatility_window(ghost_array)
        print(f"📊 Rolling volatility (5-period): {volatility:.4f}")

        # Create BTC vector
        btc_vector = BTCVector(ghost_array)
        signal_data = btc_vector.to_signal()

        print(f"📈 Mean price: ${signal_data['mean_price']:,.2f}")
        print(f"📊 Momentum: {signal_data['momentum']:.2f}")
        print(f"🎯 Confidence: {signal_data['confidence']:.2f}")
        print(f"📊 Signal count: {signal_data['signal_count']:.0f}")

        return signals

    def demonstrate_strategy_engine(self, signals: List[GhostSignal]):
        """Demonstrate the complete strategy engine."""
        print("\n🎯 Ghost Strategy Engine Demonstration")
        print("=" * 50)

        # Process signals through strategy engine
        result = self.engine.process_ghost_signals(signals)

        print(f"🔐 Strategy Hash: {result['strategy_hash'][:16]}...")
        print(f"🎯 Action: {result['action']}")
        print(f"📊 Confidence: {result['confidence']:.2f}")
        print(f"✅ Execution Ready: {result['execution_ready']}")

        # Show conditions
        print("\n📋 Strategy Conditions:")
        for condition, value in result['conditions'].items():
            status = "✅" if value else "❌"
            print(f"  {status} {condition}: {value}")

        # Show signal data
        print(f"\n📊 Signal Data:")
        for key, value in result['signal_data'].items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

        return result

    def demonstrate_volatility_scenarios(self):
        """Demonstrate different volatility scenarios."""
        print("\n🌊 Volatility Scenarios Demonstration")
        print("=" * 50)

        scenarios = {
            "Low Volatility": {
                "price_range": 100,
                "volatility": 0.01,
                "confidence": 0.95
            },
            "Medium Volatility": {
                "price_range": 500,
                "volatility": 0.04,
                "confidence": 0.8
            },
            "High Volatility": {
                "price_range": 2000,
                "volatility": 0.10,
                "confidence": 0.6
            }
        }

        base_price = 50000.0
        base_time = int(time.time())

        for scenario_name, params in scenarios.items():
            print(f"\n📊 {scenario_name}:")

            # Generate scenario signals
            signals = []
            for i in range(10):
                price_change = (i % 3 - 1) * params["price_range"] / 3
                price = base_price + price_change

                signal = GhostSignal(
                    asset="BTC",
                    price=price,
                    volatility=params["volatility"],
                    confidence=params["confidence"],
                    timestamp=base_time + i * 60
                )
                signals.append(signal)

            # Process through engine
            result = self.engine.process_ghost_signals(signals)

            print(f"  🎯 Action: {result['action']}")
            print(f"  📊 Confidence: {result['confidence']:.2f}")
            print(f"  📈 Volatility: {result['signal_data']['volatility']:.2f}")
            print(f"  ✅ Ready: {result['execution_ready']}")

    def demonstrate_real_time_processing(self):
        """Demonstrate real-time signal processing."""
        print("\n⚡ Real-Time Processing Demonstration")
        print("=" * 50)

        # Simulate real-time signal stream
        base_price = 50000.0
        base_time = int(time.time())

        print("📡 Processing real-time signals...")

        for i in range(8):
            # Simulate new signal arrival
            price_change = (i % 4 - 2) * 200  # Oscillating pattern
            price = base_price + price_change + i * 50

            volatility = 0.02 + (i % 3) * 0.02
            confidence = 0.8 + (i % 2) * 0.1

            signal = GhostSignal(
                asset="BTC",
                price=price,
                volatility=volatility,
                confidence=confidence,
                timestamp=base_time + i * 30  # 30-second intervals
            )

            # Add to engine
            self.engine.btc_processor.add_ghost_signal(signal)

            # Get current state
            current_signal = self.engine.btc_processor.get_current_signal()
            if current_signal:
                print(f"  Signal {i+1}: Price=${price:,.0f}, "
                      f"Vol={current_signal['volatility']:.3f}, "
                      f"Conf={current_signal['confidence']:.2f}")

            time.sleep(0.1)  # Simulate processing time

        # Final strategy decision
        final_signals = self.engine.btc_processor.ghost_signals
        result = self.engine.process_ghost_signals(final_signals)

        print(f"\n🎯 Final Strategy Decision:")
        print(f"  Action: {result['action']}")
        print(f"  Confidence: {result['confidence']:.2f}")
        print(f"  Execution Ready: {result['execution_ready']}")

    def run_complete_demo(self):
        """Run the complete integration demonstration."""
        print("🔱 Ghost Strategy Integration Demo - Schwabot UROS v1.0")
        print("=" * 70)
        print("This demo shows how the ghost strategy engine integrates with")
        print("the main trading system for BTC/USDC volatility-aware decisions.\n")

        try:
            # Step 1: Ghost array processing
            signals = self.demonstrate_ghost_array_processing()

            # Step 2: Strategy engine
            result = self.demonstrate_strategy_engine(signals)

            # Step 3: Volatility scenarios
            self.demonstrate_volatility_scenarios()

            # Step 4: Real-time processing
            self.demonstrate_real_time_processing()

            # Summary
            print("\n" + "=" * 70)
            print("✅ Ghost Strategy Integration Demo Complete!")
            print("=" * 70)
            print("Key Features Demonstrated:")
            print("  🔱 Type-safe ghost signal processing")
            print("  📊 Real-time volatility analysis")
            print("  🎯 Hash-based strategy selection")
            print("  ⚡ Dynamic confidence calculation")
            print("  🌊 Multi-scenario volatility handling")
            print("  📈 BTC vector momentum analysis")

            # Save demo results
            demo_results = {
                "demo_timestamp": time.time(),
                "final_strategy": result,
                "engine_stats": self.engine.get_processor_statistics()
            }

            with open("ghost_strategy_demo_results.json", "w") as f:
                json.dump(demo_results, f, indent=2, default=str)

            print(f"\n📁 Demo results saved to: ghost_strategy_demo_results.json")

        except Exception as e:
            print(f"❌ Demo failed: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main demo execution."""
    demo = GhostStrategyIntegrationDemo()
    demo.run_complete_demo()


if __name__ == "__main__":
    main()
