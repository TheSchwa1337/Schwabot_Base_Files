from core.master_cycle_engine import MasterCycleEngine
from core.qsc_enhanced_profit_allocator import (  # !/usr/bin/env python3
    """QSC,
    GTS,
    QSC,
    Any,
    Comprehensive,
    Core,
    Demo.,
    Dict,
    Generalized,
    Immune,
    Path,
    QSCDiagnosticServer,
    Quantum,
    Schwabot,
    Solutions,
    Static,
    System,
    Tensor,
    +,
    and,
    asyncio,
    complete,
    demonstration,
    for,
    from,
    immune,
    import,
    of,
    pathlib,
    server.qsc_diagnostic_websocket,
    setup_logging,
    sys,
    system,
    the,
    trading.,
    typing,
    utils.logging_setup,
)

This demo shows:
1. Auto-detection of Fibonacci divergence
2. QSC immune system activation
3. Profit cycle allocation with immune validation
4. Order book stability monitoring
5. Visual integration with alerts
6. Complete trading decision flow
"""


# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

    QSCEnhancedProfitAllocator,
)

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

    def generate_scenario_data():-> Dict[str, Any]:
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

    def analyze_scenario_results():-> None:
        """Analyze and explain the results of each scenario."""
        print(f"\n🔬 Analysis of {scenario.replace('_', ' ').title()} Scenario:")

        print(f"  🎯 Trading Decision: {diagnostics.trading_decision.value.upper()}")
        print(f"  📊 Confidence Score: {diagnostics.confidence_score:.3f}")
        print(f"  ⚠️ Risk Assessment: {diagnostics.risk_assessment}")
        print(f"  🧬 QSC Mode: {diagnostics.qsc_status['mode']}")
        print(f"  🌐 System Mode: {diagnostics.system_mode.value}")
        print(f"  🛡️ Immune Active: {diagnostics.immune_response_active}")

        if diagnostics.diagnostic_messages:
            print("  📝 Messages:")
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

    async def run_scenario_sequence():-> None:
        """Run through all test scenarios sequentially."""
        print("🧬🎯 Starting QSC + GTS Immune System Demo")
        print("=" * 70)

        for i, scenario in enumerate(self.demo_scenarios):
            print(
                f"\n📋 Running Scenario {i + 1}/{len(self.demo_scenarios)}: {scenario}"
            )
            print("-" * 50)

            # Generate scenario data
            market_data = self.generate_scenario_data(scenario)

            # Process through master cycle engine
            diagnostics = self.master_engine.process_market_tick(market_data)

            # Analyze results
            self.analyze_scenario_results(scenario, diagnostics)

            # Show profit allocation impact
            if diagnostics.profit_allocation_status == "active":
                print("  💰 Profit allocation: ACTIVE")
            else:
                print("  💰 Profit allocation: BLOCKED by immune system")

            # Wait between scenarios
            await asyncio.sleep(2)

        print("\n📊 Final System Status:")
        final_status = self.master_engine.get_system_status()
        print(f"  Total Decisions: {final_status['total_decisions']}")
        print(f"  Success Rate: {final_status['success_rate']:.2%}")
        print(f"  Immune Activations: {final_status['immune_activations']}")
        print(f"  Ghost Floor Activations: {final_status['ghost_floor_activations']}")
        print(f"  Emergency Shutdowns: {final_status['emergency_shutdowns']}")

    async def run_profit_allocation_demo():-> None:
        """Demonstrate QSC-enhanced profit allocation."""
        print("\n💰 QSC-Enhanced Profit Allocation Demo")
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

    async def run_websocket_demo():-> None:
        """Demonstrate WebSocket diagnostic streaming."""
        print("\n📡 WebSocket Diagnostic Streaming Demo")
        print("-" * 50)

        # Start QSC diagnostic server
        self.qsc_server = QSCDiagnosticServer({"port": 8767, "stream_interval": 2.0})
        await self.qsc_server.start_server()

        print("🚀 QSC Diagnostic server started on ws://localhost:8767")
        print("🔗 Connect your React app to this endpoint for real-time diagnostics")

        # Let it run for a bit to show streaming
        print("📡 Streaming diagnostic data for 10 seconds...")
        await asyncio.sleep(10)

        # Show server stats
        stats = self.qsc_server.get_server_stats()
        print("📊 Server Stats:")
        print(f"  Active Connections: {stats['active_connections']}")
        print(f"  Messages Sent: {stats['total_messages_sent']}")
        print(f"  Active Alerts: {stats['active_alerts']}")

        await self.qsc_server.stop_server()

    async def run_fibonacci_echo_demo():-> None:
        """Demonstrate Fibonacci echo visualization data."""
        print("\n📈 Fibonacci Echo Visualization Demo")
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

    async def run_complete_demo():-> None:
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

        print("\n✅ Complete QSC + GTS Immune System Demo Finished!")
        print("=" * 80)

        print("\n🚀 To integrate with your full Schwabot system:")
        print("1. Import MasterCycleEngine into your main trading loop")
        print("2. Use QSCEnhancedProfitAllocator instead of standard allocator")
        print("3. Connect React visualization to QSC diagnostic WebSocket")
        print("4. Monitor immune system alerts for manual intervention")

        print("\n📋 Key Integration Points:")
        print("  🧬 core.master_cycle_engine.MasterCycleEngine")
        print("  💰 core.qsc_enhanced_profit_allocator.QSCEnhancedProfitAllocator")
        print("  📡 server.qsc_diagnostic_websocket.QSCDiagnosticServer")
        print("  🎯 React: ws://localhost:8766 for QSC diagnostics")
        print("  🌐 React: ws://localhost:8765 for tensor analysis")


async def main():
    """Main demo function."""
    demo = QSCImmuneSystemDemo()
    await demo.run_complete_demo()


if __name__ == "__main__":
    asyncio.run(main())
