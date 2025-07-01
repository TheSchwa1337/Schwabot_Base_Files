#!/usr/bin/env python3
"""QSC + GTS Immune System Demo.

Comprehensive demonstration of the complete Quantum Static Core (QSC) and
Generalized Tensor Solutions (GTS) immune system for Schwabot trading.

This demo shows:
1. Auto-detection of Fibonacci divergence
2. QSC immune system activation
3. Profit cycle allocation with immune validation
4. Order book stability monitoring
5. Visual integration with alerts
6. Complete trading decision flow
"""

import asyncio
import time
import logging
from typing import Dict, List, Any
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from core.master_cycle_engine import MasterCycleEngine, SystemMode, TradingDecision
from core.quantum_static_core import QuantumStaticCore, QSCMode, ResonanceLevel
from core.galileo_tensor_bridge import GalileoTensorBridge
from core.qsc_enhanced_profit_allocator import (
    QSCEnhancedProfitAllocator,
    QSCAllocationMode,
)
from server.qsc_diagnostic_websocket import QSCDiagnosticServer
from utils.logging_setup import setup_logging

# Setup logging
logger = setup_logging(__name__)


class QSCImmuneSystemDemo:
    """Comprehensive QSC + GTS immune system demonstration."""

    def __init__(self):
        """Initialize the demo."""
        self.master_engine = MasterCycleEngine()
        self.qsc_server = None
        self.demo_scenarios = [
            "normal_operation",
            "fibonacci_divergence",
            "orderbook_instability",
            "low_confidence",
            "emergency_shutdown",
            "ghost_floor_mode",
            "immune_recovery",
        ]
        self.current_scenario = 0

        logger.info("🧬🎯 QSC Immune System Demo initialized")

    def generate_scenario_data(self, scenario: str) -> Dict[str, Any]:
        """Generate market data for different test scenarios."""
        base_data = {
            "btc_price": 51200.0,
            "price_history": [50000, 50500, 51000, 50800, 51200],
            "volume_history": [100, 120, 90, 110, 130],
            "fibonacci_projection": [50000, 50600, 51100, 50900, 51300],
            "orderbook": {
                "bids": [[51190, 1.5], [51180, 2.0], [51170, 1.8]],
                "asks": [[51210, 1.6], [51220, 2.2], [51230, 1.4]],
            },
        }

        if scenario == "normal_operation":
            # Normal market conditions
            return base_data

        elif scenario == "fibonacci_divergence":
            # Introduce significant Fibonacci divergence
            base_data["fibonacci_projection"] = [
                49500,
                50100,
                50600,
                50300,
                50800,
            ]  # Large divergence
            print("📊 Scenario: Fibonacci Divergence")
            print("  Simulating significant price divergence from Fibonacci projection")

        elif scenario == "orderbook_instability":
            # Create orderbook imbalance
            base_data["orderbook"] = {
                "bids": [[51190, 0.2], [51180, 0.3], [51170, 0.1]],  # Very thin bids
                "asks": [[51210, 5.0], [51220, 8.0], [51230, 6.0]],  # Heavy asks
            }
            print("📊 Scenario: Order Book Instability")
            print("  Simulating severe order book imbalance (thin bids, heavy asks)")

        elif scenario == "low_confidence":
            # Create conditions for low confidence
            base_data["price_history"] = [
                51200,
                50800,
                51500,
                50600,
                51800,
            ]  # High volatility
            base_data["volume_history"] = [50, 200, 30, 180, 40]  # Erratic volume
            print("📊 Scenario: Low Confidence Conditions")
            print("  Simulating high volatility and erratic volume patterns")

        elif scenario == "emergency_shutdown":
            # Extreme conditions requiring emergency shutdown
            base_data["price_history"] = [
                51200,
                48000,
                53000,
                47500,
                54000,
            ]  # Extreme volatility
            base_data["fibonacci_projection"] = [
                51200,
                48500,
                52800,
                47800,
                53500,
            ]  # Still divergent
            base_data["orderbook"] = {
                "bids": [[51190, 0.05], [51180, 0.1]],  # Almost no liquidity
                "asks": [[51210, 0.08], [51220, 0.12]],
            }
            print("📊 Scenario: Emergency Shutdown Conditions")
            print(
                "  Simulating extreme market conditions requiring emergency protocols"
            )

        elif scenario == "ghost_floor_mode":
            # Conditions that trigger ghost floor mode
            base_data["orderbook"] = {
                "bids": [],  # No bids
                "asks": [[51210, 0.01]],  # Minimal asks
            }
            print("📊 Scenario: Ghost Floor Mode")
            print("  Simulating complete liquidity drought")

        elif scenario == "immune_recovery":
            # Good conditions for system recovery
            base_data["price_history"] = [
                51000,
                51050,
                51100,
                51150,
                51200,
            ]  # Stable trend
            base_data["volume_history"] = [100, 105, 110, 108, 112]  # Consistent volume
            base_data["fibonacci_projection"] = [
                51005,
                51055,
                51105,
                51155,
                51205,
            ]  # Close alignment
            base_data["orderbook"] = {
                "bids": [[51190, 2.5], [51180, 3.0], [51170, 2.8]],  # Good liquidity
                "asks": [[51210, 2.6], [51220, 2.9], [51230, 2.4]],
            }
            print("📊 Scenario: Immune System Recovery")
            print("  Simulating optimal conditions for system recovery")

        return base_data

    def analyze_scenario_results(self, scenario: str, diagnostics) -> None:
        """Analyze and explain the results of each scenario."""
        print(f"\n🔬 Analysis of {scenario.replace('_', ' ').title()} Scenario:")

        print(f"  🎯 Trading Decision: {diagnostics.trading_decision.value.upper()}")
        print(f"  📊 Confidence Score: {diagnostics.confidence_score:.3f}")
        print(f"  ⚠️ Risk Assessment: {diagnostics.risk_assessment}")
        print(f"  🧬 QSC Mode: {diagnostics.qsc_status['mode']}")
        print(f"  🌐 System Mode: {diagnostics.system_mode.value}")
        print(f"  🛡️ Immune Active: {diagnostics.immune_response_active}")

        if diagnostics.diagnostic_messages:
            print(f"  📝 Messages:")
            for msg in diagnostics.diagnostic_messages:
                print(f"    {msg}")

        # Explain the immune system logic
        if scenario == "fibonacci_divergence":
            if diagnostics.immune_response_active:
                print("  ✅ QSC Immune System correctly detected Fibonacci divergence")
                print("  🛡️ Quantum probe triggered protective response")
            else:
                print("  ⚠️ Divergence not detected - may need threshold adjustment")

        elif scenario == "orderbook_instability":
            if diagnostics.orderbook_stability < 0.5:
                print(
                    "  ✅ Order book immune validation correctly rejected unstable conditions"
                )
                print("  👻 Ghost Floor Mode should activate for protection")
            else:
                print("  ⚠️ Order book instability not detected")

        elif scenario == "emergency_shutdown":
            if diagnostics.system_mode.value == "emergency_shutdown":
                print("  ✅ Emergency protocols correctly activated")
                print("  🚨 All trading suspended for safety")
            else:
                print("  ⚠️ Emergency conditions present but not detected")

        elif scenario == "immune_recovery":
            if (
                not diagnostics.immune_response_active
                and diagnostics.confidence_score > 0.7
            ):
                print("  ✅ System recovery successful")
                print("  🟢 Normal trading operations can resume")
            else:
                print("  ⚠️ System still in protective mode")

    async def run_scenario_sequence(self) -> None:
        """Run through all test scenarios sequentially."""
        print("🧬🎯 Starting QSC + GTS Immune System Demo")
        print("=" * 70)

        for i, scenario in enumerate(self.demo_scenarios):
            print(f"\n📋 Running Scenario {i+1}/{len(self.demo_scenarios)}: {scenario}")
            print("-" * 50)

            # Generate scenario data
            market_data = self.generate_scenario_data(scenario)

            # Process through master cycle engine
            diagnostics = self.master_engine.process_market_tick(market_data)

            # Analyze results
            self.analyze_scenario_results(scenario, diagnostics)

            # Show profit allocation impact
            if diagnostics.profit_allocation_status == "active":
                print(f"  💰 Profit allocation: ACTIVE")
            else:
                print(f"  💰 Profit allocation: BLOCKED by immune system")

            # Wait between scenarios
            await asyncio.sleep(2)

        print(f"\n📊 Final System Status:")
        final_status = self.master_engine.get_system_status()
        print(f"  Total Decisions: {final_status['total_decisions']}")
        print(f"  Success Rate: {final_status['success_rate']:.2%}")
        print(f"  Immune Activations: {final_status['immune_activations']}")
        print(f"  Ghost Floor Activations: {final_status['ghost_floor_activations']}")
        print(f"  Emergency Shutdowns: {final_status['emergency_shutdowns']}")

    async def run_profit_allocation_demo(self) -> None:
        """Demonstrate QSC-enhanced profit allocation."""
        print(f"\n💰 QSC-Enhanced Profit Allocation Demo")
        print("-" * 50)

        allocator = QSCEnhancedProfitAllocator()

        # Test different profit scenarios
        test_scenarios = [
            {"profit": 1000.0, "description": "Normal profit allocation"},
            {"profit": 5000.0, "description": "Large profit with immune validation"},
            {"profit": 100.0, "description": "Small profit under low confidence"},
        ]

        for scenario in test_scenarios:
            print(f"\n💰 Testing: {scenario['description']}")

            # Generate market data
            market_data = self.generate_scenario_data("normal_operation")

            # Allocate profit
            cycle = allocator.allocate_profit_with_qsc(
                scenario["profit"], market_data, market_data["btc_price"]
            )

            print(f"  Profit Amount: ${scenario['profit']:.2f}")
            print(f"  Allocated: ${cycle.allocated_profit:.2f}")
            print(f"  Blocked: ${cycle.qsc_blocked_amount:.2f}")
            print(f"  Mode: {cycle.qsc_mode.value}")
            print(f"  Immune Approved: {cycle.immune_approved}")
            print(f"  Resonance Score: {cycle.resonance_score:.3f}")

    async def run_websocket_demo(self) -> None:
        """Demonstrate WebSocket diagnostic streaming."""
        print(f"\n📡 WebSocket Diagnostic Streaming Demo")
        print("-" * 50)

        # Start QSC diagnostic server
        self.qsc_server = QSCDiagnosticServer({"port": 8767, "stream_interval": 2.0})
        await self.qsc_server.start_server()

        print(f"🚀 QSC Diagnostic server started on ws://localhost:8767")
        print(f"🔗 Connect your React app to this endpoint for real-time diagnostics")

        # Let it run for a bit to show streaming
        print(f"📡 Streaming diagnostic data for 10 seconds...")
        await asyncio.sleep(10)

        # Show server stats
        stats = self.qsc_server.get_server_stats()
        print(f"📊 Server Stats:")
        print(f"  Active Connections: {stats['active_connections']}")
        print(f"  Messages Sent: {stats['total_messages_sent']}")
        print(f"  Active Alerts: {stats['active_alerts']}")

        await self.qsc_server.stop_server()

    async def run_fibonacci_echo_demo(self) -> None:
        """Demonstrate Fibonacci echo visualization data."""
        print(f"\n📈 Fibonacci Echo Visualization Demo")
        print("-" * 50)

        # Generate some history by processing multiple ticks
        for i in range(20):
            scenario = "fibonacci_divergence" if i % 5 == 0 else "normal_operation"
            market_data = self.generate_scenario_data(scenario)
            self.master_engine.process_market_tick(market_data)

        # Get Fibonacci echo data
        echo_data = self.master_engine.get_fibonacci_echo_data()

        if echo_data:
            print(
                f"📊 Generated {len(echo_data.get('timestamps', []))} data points for visualization"
            )
            print(
                f"  Recent Fibonacci Divergences: {echo_data['fibonacci_divergences'][-5:]}"
            )
            print(
                f"  Recent Confidence Scores: {[f'{x:.3f}' for x in echo_data['confidence_scores'][-5:]]}"
            )
            print(f"  Recent System Modes: {echo_data['system_modes'][-5:]}")
            print(f"  Recent Trading Decisions: {echo_data['trading_decisions'][-5:]}")
        else:
            print("📊 No echo data available yet")

    async def run_complete_demo(self) -> None:
        """Run the complete comprehensive demo."""
        print("🧬⚡ Starting Complete QSC + GTS Immune System Demo")
        print("=" * 80)

        # 1. Core immune system scenarios
        await self.run_scenario_sequence()
        await asyncio.sleep(2)

        # 2. Profit allocation demonstration
        await self.run_profit_allocation_demo()
        await asyncio.sleep(2)

        # 3. Fibonacci echo demonstration
        await self.run_fibonacci_echo_demo()
        await asyncio.sleep(2)

        # 4. WebSocket streaming demonstration
        await self.run_websocket_demo()

        print(f"\n✅ Complete QSC + GTS Immune System Demo Finished!")
        print("=" * 80)

        print(f"\n🚀 To integrate with your full Schwabot system:")
        print(f"1. Import MasterCycleEngine into your main trading loop")
        print(f"2. Use QSCEnhancedProfitAllocator instead of standard allocator")
        print(f"3. Connect React visualization to QSC diagnostic WebSocket")
        print(f"4. Monitor immune system alerts for manual intervention")

        print(f"\n📋 Key Integration Points:")
        print(f"  🧬 core.master_cycle_engine.MasterCycleEngine")
        print(f"  💰 core.qsc_enhanced_profit_allocator.QSCEnhancedProfitAllocator")
        print(f"  📡 server.qsc_diagnostic_websocket.QSCDiagnosticServer")
        print(f"  🎯 React: ws://localhost:8766 for QSC diagnostics")
        print(f"  🌐 React: ws://localhost:8765 for tensor analysis")


async def main():
    """Main demo function."""
    demo = QSCImmuneSystemDemo()
    await demo.run_complete_demo()


if __name__ == "__main__":
    asyncio.run(main())
