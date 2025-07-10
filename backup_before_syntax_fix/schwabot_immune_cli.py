#!/usr/bin/env python3
import argparse
import asyncio
import http.server
import logging
import random
import signal
import socketserver
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Any, Dict

import numpy as np

    ImmuneZone,
    immune_protected,
)
from core.enhanced_master_cycle_engine import EnhancedMasterCycleEngine
from server.immune_diagnostic_websocket import ImmuneDiagnosticWebSocketServer

"""Schwabot Biological Immune System CLI."
Comprehensive command-line interface for launching and managing the enhanced
Schwabot system with biological immune error handling, T-cell validation,
neural gateways, swarm consensus, and zone-based response mechanisms.
Features:
- Complete immune system testing and validation
- Real-time monitoring dashboard
- Market simulation with immune responses
- Error injection and recovery testing
- Production deployment tools
"""

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


logger = logging.getLogger(__name__)


class SchwabotImmuneCLI:
    """Main CLI controller for Schwabot Biological Immune System."""

    def __init__(self):
        """Initialize the CLI controller."""
        self.engine = None
        self.immune_handler = None
        self.websocket_server = None
        self.running = False

        # Setup logging
        logging.basicConfig()
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

        print("🧬 Schwabot Biological Immune System CLI")
        print("=" * 60)

    async def initialize_systems(self) -> bool:
        """Initialize all immune system components."""
        try:
            print("🧬 Initializing Biological Immune System...")

            # Initialize enhanced master cycle engine
            self.engine = EnhancedMasterCycleEngine()
            self.immune_handler = self.engine.immune_handler

            # Initialize WebSocket diagnostic server
            self.websocket_server = ImmuneDiagnosticWebSocketServer()

            print("✅ All systems initialized successfully")
            return True

        except Exception as e:
            print(f"🚨 Initialization failed: {e}")
            return False

    async def run_comprehensive_test(self) -> None:
        """Run comprehensive immune system test suite."""
        print("\n🧬 Running Comprehensive Immune System Test Suite")
        print("=" * 60)

        # Test 1: Basic Immune Protection
        print("\n1️⃣ Testing Basic Immune Protection...")
        test_results = await self._test_basic_immune_protection()
        self._print_test_results("Basic Immune Protection", test_results)

        # Test 2: T-Cell Validation
        print("\n2️⃣ Testing T-Cell Validation...")
        test_results = await self._test_tcell_validation()
        self._print_test_results("T-Cell Validation", test_results)

        # Test 3: Neural Gateway Protection
        print("\n3️⃣ Testing Neural Gateway Protection...")
        test_results = await self._test_neural_gateway()
        self._print_test_results("Neural Gateway Protection", test_results)

        # Test 4: Swarm Consensus Validation
        print("\n4️⃣ Testing Swarm Consensus Validation...")
        test_results = await self._test_swarm_consensus()
        self._print_test_results("Swarm Consensus Validation", test_results)

        # Test 5: Zone-Based Response
        print("\n5️⃣ Testing Zone-Based Response...")
        test_results = await self._test_zone_response()
        self._print_test_results("Zone-Based Response", test_results)

        # Test 6: Error Recovery and Antibody Formation
        print("\n6️⃣ Testing Error Recovery and Antibody Formation...")
        test_results = await self._test_error_recovery()
        self._print_test_results("Error Recovery", test_results)

        # Test 7: Market Simulation with Immune Response
        print("\n7️⃣ Testing Market Simulation with Immune Response...")
        test_results = await self._test_market_simulation()
        self._print_test_results("Market Simulation", test_results)

        print("\n🧬 Comprehensive Test Suite Complete")
        self._print_system_status()

    async def _test_basic_immune_protection(self) -> Dict[str, Any]:
        """Test basic immune protection functionality."""
        results = {"passed": 0, "failed": 0, "details": []}

        try:
            # Test normal operation
            @immune_protected(self.immune_handler)
            def normal_operation(x: float) -> float:
                return x * 2.0

            result = normal_operation(5.0)
            if result == 10.0:
                results["passed"] += 1
                results["details"].append("✅ Normal operation successful")
            else:
                results["failed"] += 1
                results["details"].append("❌ Normal operation failed")

            # Test error handling
            @immune_protected(self.immune_handler)
            def error_operation() -> None:
                raise ValueError("Test error")

            result = error_operation()
            if hasattr(result, "zone"):  # Should return ImmuneResponse
                results["passed"] += 1
                results["details"].append("✅ Error handling successful")
            else:
                results["failed"] += 1
                results["details"].append("❌ Error handling failed")

        except Exception as e:
            results["failed"] += 1
            results["details"].append(f"❌ Test exception: {e}")

        return results

    async def _test_tcell_validation(self) -> Dict[str, Any]:
        """Test T-Cell validation system."""
        results = {"passed": 0, "failed": 0, "details": []}

        try:
                TCellSignal,
                ImmuneSignalType,
                TCellValidator,
            )

            validator = TCellValidator()

            # Test strong positive signals
            strong_signals = []
                TCellSignal(ImmuneSignalType.PRIMARY, 0.8, "test_primary", time.time()),
                TCellSignal()
                    ImmuneSignalType.COSTIMULATORY, 0.9, "test_costim", time.time()
                ),
                TCellSignal()
                    ImmuneSignalType.INFLAMMATORY, 0.3, "test_inflam", time.time()
                ),
            ]

            activation, confidence, analysis = validator.validate_signals()
                strong_signals
            )
            if activation and confidence > 0.6:
                results["passed"] += 1
                results["details"].append()
                    f"✅ Strong signals activated T-cell (confidence: {confidence:.3f})"
                )
            else:
                results["failed"] += 1
                results["details"].append("❌ Strong signals failed to activate T-cell")

            # Test weak signals
            weak_signals = []
                TCellSignal(ImmuneSignalType.PRIMARY, 0.2, "test_primary", time.time()),
                TCellSignal()
                    ImmuneSignalType.INHIBITORY, 0.8, "test_inhibit", time.time()
                ),
            ]

            activation, confidence, analysis = validator.validate_signals(weak_signals)
            if not activation:
                results["passed"] += 1
                results["details"].append()
                    "✅ Weak signals correctly blocked T-cell activation"
                )
            else:
                results["failed"] += 1
                results["details"].append()
                    "❌ Weak signals incorrectly activated T-cell"
                )

        except Exception as e:
            results["failed"] += 1
            results["details"].append(f"❌ T-Cell test exception: {e}")

        return results

    async def _test_neural_gateway(self) -> Dict[str, Any]:
        """Test neural gateway protection."""
        results = {"passed": 0, "failed": 0, "details": []}

        try:
            gateway = self.immune_handler.neural_gateway

            # Test permissive state
            gateway.current_state = gateway.current_state.PERMISSIVE
            allowed = gateway.should_allow_operation()
                0.8, 0.1
            )  # High confidence, low entropy
            if allowed:
                results["passed"] += 1
                results["details"].append()
                    "✅ Permissive state allows high-confidence operations"
                )
            else:
                results["failed"] += 1
                results["details"].append()
                    "❌ Permissive state blocked high-confidence operation"
                )

            # Test emergency state
            gateway.current_state = gateway.current_state.EMERGENCY
            allowed = gateway.should_allow_operation()
                0.8, 0.9
            )  # High confidence, high entropy
            if not allowed:
                results["passed"] += 1
                results["details"].append()
                    "✅ Emergency state correctly blocks operations"
                )
            else:
                results["failed"] += 1
                results["details"].append()
                    "❌ Emergency state incorrectly allowed operation"
                )

            # Test adaptive threshold
            threshold = gateway.calculate_adaptive_threshold(0.5)
            if 0.7 < threshold < 0.8:  # Should be baseline + (0.15 * 0.5)
                results["passed"] += 1
                results["details"].append()
                    f"✅ Adaptive threshold calculation correct: {threshold:.3f}"
                )
            else:
                results["failed"] += 1
                results["details"].append()
                    f"❌ Adaptive threshold incorrect: {threshold:.3f}"
                )

        except Exception as e:
            results["failed"] += 1
            results["details"].append(f"❌ Neural gateway test exception: {e}")

        return results

    async def _test_swarm_consensus(self) -> Dict[str, Any]:
        """Test swarm consensus validation."""
        results = {"passed": 0, "failed": 0, "details": []}

        try:
            from core.swarm_intelligence import SwarmConsensus

            swarm = SwarmConsensus(node_id="test_node", peer_nodes=["peer1", "peer2"])

            # Test consensus achievement
            proposal = {"action": "BUY", "symbol": "BTC", "amount": 1.0}
            swarm.propose_action(proposal)
            swarm.record_vote("peer1", proposal, True)
            swarm.record_vote("peer2", proposal, True)

            consensus, details = swarm.check_consensus(proposal)
            if consensus:
                results["passed"] += 1
                results["details"].append("✅ Swarm consensus achieved successfully")
            else:
                results["failed"] += 1
                results["details"].append("❌ Swarm consensus failed")

            # Test consensus failure
            proposal2 = {"action": "SELL", "symbol": "ETH", "amount": 10.0}
            swarm.propose_action(proposal2)
            swarm.record_vote("peer1", proposal2, True)
            swarm.record_vote("peer2", proposal2, False)

            consensus, details = swarm.check_consensus(proposal2)
            if not consensus:
                results["passed"] += 1
                results["details"].append("✅ Swarm consensus failure handled correctly")
            else:
                results["failed"] += 1
                results["details"].append("❌ Swarm consensus failure test failed")

        except Exception as e:
            results["failed"] += 1
            results["details"].append(f"❌ Swarm Consensus test exception: {e}")

        return results

    async def _test_zone_response(self) -> Dict[str, Any]:
        """Test zone-based response mechanism."""
        results = {"passed": 0, "failed": 0, "details": []}

        try:
            # Trigger ALERT zone
            self.immune_handler.log_error(ValueError("Test Alert"), ImmuneZone.ALERT)
            if self.immune_handler.get_zone_status(ImmuneZone.ALERT)["errors"] > 0:
                results["passed"] += 1
                results["details"].append("✅ ALERT zone triggered successfully")
            else:
                results["failed"] += 1
                results["details"].append("❌ ALERT zone failed to trigger")

            # Trigger TOXIC zone
            for _ in range(15):
                self.immune_handler.log_error()
                    TypeError("Test Toxic"), ImmuneZone.TOXIC
                )
            if self.immune_handler.get_zone_status(ImmuneZone.TOXIC)["active"]:
                results["passed"] += 1
                results["details"].append("✅ TOXIC zone triggered successfully")
            else:
                results["failed"] += 1
                results["details"].append("❌ TOXIC zone failed to trigger")

            # Reset zones
            self.immune_handler.reset_zone_status()
            if not self.immune_handler.get_zone_status(ImmuneZone.TOXIC)["active"]:
                results["passed"] += 1
                results["details"].append("✅ Zone reset successful")
            else:
                results["failed"] += 1
                results["details"].append("❌ Zone reset failed")

        except Exception as e:
            results["failed"] += 1
            results["details"].append(f"❌ Zone Response test exception: {e}")

        return results

    async def _test_error_recovery(self) -> Dict[str, Any]:
        """Test error recovery and antibody formation."""
        results = {"passed": 0, "failed": 0, "details": []}
        try:

            @immune_protected(self.immune_handler)
            def recurring_error_operation():
                raise ConnectionError("Simulated recurring network error")

            # Trigger error multiple times to form antibody
            for i in range(10):
                recurring_error_operation()

            # Check if antibody was formed
            error_signature = self.immune_handler._get_error_signature()
                ConnectionError("..."), recurring_error_operation
            )
            antibody = self.immune_handler.antibodies.get(error_signature)

            if antibody and antibody.strength > 5:
                results["passed"] += 1
                results["details"].append()
                    f"✅ Antibody formed for recurring error (strength: {antibody.strength})"
                )
            else:
                results["failed"] += 1
                results["details"].append("❌ Antibody formation failed")

            # Check if recovery mechanism is triggered
            initial_state = self.immune_handler.neural_gateway.current_state
            recurring_error_operation()  # One more time to trigger response
            final_state = self.immune_handler.neural_gateway.current_state

            if final_state != initial_state:
                results["passed"] += 1
                results["details"].append()
                    f"✅ Recovery mechanism triggered (state changed to {final_state.name})"
                )
            else:
                results["failed"] += 1
                results["details"].append("❌ Recovery mechanism failed to trigger")

        except Exception as e:
            results["failed"] += 1
            results["details"].append(f"❌ Error Recovery test exception: {e}")

        return results

    async def _test_market_simulation(self) -> Dict[str, Any]:
        """Test market simulation with immune responses."""
        results = {"passed": 0, "failed": 0, "details": []}
        try:
            print("  - Simulating 100 market ticks...")
            for i in range(100):
                # Simulate market data
                price = 50000 + (np.random.randn() * 500)
                volume = 1000 + (np.random.rand() * 200)

                # Inject occasional errors
                if random.random() < 0.1:
                    error_type = random.choice()
                        [ValueError, TypeError, KeyError, IndexError]
                    )
                    self.engine.process_market_data()
                        price, volume, inject_error=error_type("Simulated market error")
                    )
                else:
                    self.engine.process_market_data(price, volume)

                # Update websocket server
                if self.websocket_server and self.websocket_server.is_running():
                    await self.websocket_server.broadcast()
                        self.immune_handler.get_full_status()
                    )

                time.sleep(0.1)

            # Check final system health
            status = self.immune_handler.get_full_status()
            if status["overall_health"] > 0.7:
                results["passed"] += 1
                results["details"].append()
                    f"✅ Market simulation completed with high health ({")}
                        status['overall_health']:.2f})")"
            else:
                results["failed"] += 1
                results["details"].append()
                    f"❌ Market simulation ended with low health ({status['overall_health']:.2f})"
                )

            # Check for quarantined errors
            if status["zones"]["QUARANTINE"]["errors"] > 0:
                results["passed"] += 1
                results["details"].append("✅ Errors were successfully quarantined")
            else:
                results["failed"] += 1
                results["details"].append("❌ No errors were quarantined during simulation")

        except Exception as e:
            results["failed"] += 1
            results["details"].append(f"❌ Market Simulation test exception: {e}")

        return results

    def _print_test_results(self, test_name: str, results: Dict[str, Any]) -> None:
        """Print formatted test results."""
        total = results["passed"] + results["failed"]
        pass_rate = (results["passed"] / total * 100) if total > 0 else 0
        print(f"  - {test_name}: {results['passed']}/{total} passed ({pass_rate:.1f}%)")
        for detail in results["details"]:
            print(f"    {detail}")

    def _print_system_status(self) -> None:
        """Print the current status of the immune system."""
        print("\n🧬 Current Immune System Status")
        print("-" * 60)
        status = self.immune_handler.get_full_status()
        print(f"  - Overall Health: {status['overall_health']:.2%}")
        print(f"  - Active Antibodies: {status['antibody_count']}")
        print(f"  - Neural Gateway State: {status['neural_gateway_state']}")
        for zone_name, zone_data in status["zones"].items():
            print()
                f"  - Zone '{zone_name}': {zone_data['errors']} errors, Active: {zone_data['active']}"
            )
        print("-" * 60)

    async def start_monitoring_dashboard(self) -> None:
        """Start the real-time monitoring dashboard."""
        if not self.websocket_server:
            print("🚨 WebSocket server not initialized.")
            return

        print("\n🚀 Starting Real-time Monitoring Dashboard")
        print("=" * 60)

        # Start WebSocket server in a separate thread
        ws_thread = threading.Thread()
            target=asyncio.run, args=(self.websocket_server.start(),)
        )
        ws_thread.daemon = True
        ws_thread.start()

        # Serve the HTML dashboard
        dashboard_path = Path(__file__).parent / "server" / "immune_dashboard.html"
        if not dashboard_path.exists():
            print(f"🚨 Dashboard file not found at: {dashboard_path}")
            return

        PORT = 8008
        Handler = http.server.SimpleHTTPRequestHandler
        httpd = None

        class DashboardHandler(http.server.SimpleHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/":
                    self.path = str(dashboard_path)
                # Fallback to default behavior to serve other files like js, css
                return http.server.SimpleHTTPRequestHandler.do_GET(self)

        # Ensure we can reuse the address
        socketserver.TCPServer.allow_reuse_address = True
        try:
            with socketserver.TCPServer(("", PORT), DashboardHandler) as httpd:
                print(f"📈 Dashboard available at: http://localhost:{PORT}")
                print("   (Press Ctrl+C to stop the, dashboard)")
                # Open browser
                webbrowser.open_new_tab(f"http://localhost:{PORT}")

                # Keep the server running until stopped
                self.running = True
                while self.running:
                    httpd.handle_request()
                    time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n🛑 Stopping dashboard...")
        except Exception as e:
            print(f"🚨 Dashboard server failed: {e}")
        finally:
            if httpd:
                httpd.server_close()
            if self.websocket_server.is_running():
                asyncio.run(self.websocket_server.stop())
            self.running = False

    async def run_stress_test(self) -> None:
        """Run a continuous stress test on the immune system."""
        print("\n🔥 Running Continuous Stress Test")
        print("=" * 60)
        print("   (Press Ctrl+C to stop the, test)")

        self.running = True
        error_count = 0
        start_time = time.time()

        @immune_protected(self.immune_handler)
        def random_operation(operation_type: str):
            if operation_type == "math":
                return np.log(random.random() * 100) / np.sqrt(random.random())
            elif operation_type == "io":
                time.sleep(random.random() * 0.1)
                if random.random() < 0.2:
                    raise IOError("Simulated IO error")
                return True
            elif operation_type == "data":
                data = {"key": "value"}
                if random.random() < 0.1:
                    del data["key"]  # Trigger KeyError
                return data["key"]
            else:
                return None

        try:
            while self.running:
                try:
                    op_type = random.choice(["math", "io", "data"])
                    random_operation(op_type)
                except Exception:
                    error_count += 1

                # Update status and broadcast
                if self.websocket_server and self.websocket_server.is_running():
                    await self.websocket_server.broadcast()
                        self.immune_handler.get_full_status()
                    )

                if time.time() - start_time > 1:
                    status = self.immune_handler.get_full_status()
                    print()
                        f"\r  - Health: {status['overall_health']:.1%}, "
                        f"Errors: {error_count}, "
                        f"Gateway: {status['neural_gateway_state']}, "
                        f"Antibodies: {status['antibody_count']}",
                        end="",
                    )
                    start_time = time.time()

                await asyncio.sleep(0.01)

        except KeyboardInterrupt:
            print("\n🛑 Stopping stress test...")
        finally:
            self.running = False
            self._print_system_status()

    async def demonstrate_immune_scenarios(self) -> None:
        """Run a demonstration of different immune scenarios."""
        print("\n🎬 Demonstrating Immune System Scenarios")
        print("=" * 60)

        scenarios = {}
            "1": ("Healthy System", self._demo_healthy_system),
            "2": ("Alert Condition", self._demo_alert_condition),
            "3": ("Toxic Environment", self._demo_toxic_environment),
            "4": ("Quarantine Mode", self._demo_quarantine_mode),
            "5": ("Recovery Phase", self._demo_recovery_phase),
        }

        for key, (name, func) in scenarios.items():
            print(f"\n--- Scenario {key}: {name} ---")
            await self.reset_immune_system()
            await func()
            self._print_system_status()
            input("   Press Enter to continue to the next scenario...")

        print("\n🎬 Scenario demonstration complete.")

    async def _demo_healthy_system(self) -> None:
        """Demonstrate a healthy system with normal operations."""
        print("  - Simulating 10 successful operations.")

        @immune_protected(self.immune_handler)
        def healthy_operation(x):
            return x**2

        for i in range(10):
            healthy_operation(i)
        print("  - Healthy operations completed without triggering immune response.")

    async def _demo_alert_condition(self) -> None:
        """Demonstrate an alert condition from minor, infrequent errors."""
        print("  - Simulating 5 minor errors (ValueError).")

        @immune_protected(self.immune_handler)
        def alert_operation(x):
            if x % 2 == 0:
                raise ValueError("Simulated minor error")
            return True

        for i in range(10):
            alert_operation(i)
        print()
            "  - Minor errors triggered ALERT zone but system remains largely healthy."
        )

    async def _demo_toxic_environment(self) -> None:
        """Demonstrate a toxic environment from frequent, severe errors."""
        print("  - Simulating 20 severe errors (TypeError).")

        @immune_protected(self.immune_handler)
        def toxic_operation(x):
            # This will always raise a TypeError
            return "a" + x

        for i in range(20):
            toxic_operation(i)
        print()
            "  - Frequent, severe errors triggered TOXIC zone and reduced system health."
        )

    async def _demo_quarantine_mode(self) -> None:
        """Demonstrate quarantine mode from a recurring, specific error."""
        print("  - Simulating 15 occurrences of the same ConnectionError.")

        @immune_protected(self.immune_handler)
        def quarantine_operation(x):
            raise ConnectionError("Simulated persistent connection failure")

        for i in range(15):
            quarantine_operation(i)
        print()
            "  - Recurring error led to antibody formation and QUARANTINE zone activation."
        )

    async def _demo_recovery_phase(self) -> None:
        """Demonstrate the recovery phase after an immune response."""
        print("  - First, triggering a TOXIC environment...")

        @immune_protected(self.immune_handler)
        def error_op(x):
            raise TypeError("Make it toxic")

        for i in range(20):
            error_op(i)
        self._print_system_status()

        print("\n  - Now, simulating a period of healthy operations for recovery.")

        @immune_protected(self.immune_handler)
        def healthy_op(x):
            return x

        for i in range(50):
            healthy_op(i)
            time.sleep(0.1)  # Simulate time passing

        print("  - After a period of stability, system health should improve.")

    def setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""

        def signal_handler(signum, frame):
            print(f"\nReceived signal {signum}, shutting down...")
            self.running = False

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    async def main_menu(self) -> None:
        """Display the main menu and handle user input."""
        self.setup_signal_handlers()
        if not await self.initialize_systems():
            return

        self.running = True
        while self.running:
            print("\n--- Main Menu ---")
            print("1. Run Comprehensive Test Suite")
            print("2. Start Real-time Monitoring Dashboard")
            print("3. Run Continuous Stress Test")
            print("4. Demonstrate Immune Scenarios")
            print("5. Trigger Emergency Quarantine Scenario")
            print("6. Reset Immune System State")
            print("7. View Current System Status")
            print("8. Exit")
            choice = input("Enter your choice: ")

            if choice == "1":
                await self.run_comprehensive_test()
            elif choice == "2":
                await self.start_monitoring_dashboard()
            elif choice == "3":
                await self.run_stress_test()
            elif choice == "4":
                await self.demonstrate_immune_scenarios()
            elif choice == "5":
                await self.trigger_emergency_scenario()
            elif choice == "6":
                await self.reset_immune_system()
            elif choice == "7":
                self._print_system_status()
            elif choice == "8":
                self.running = False
            else:
                print("Invalid choice, please try again.")

        print("\n🧬 Shutting down Schwabot Immune CLI. Goodbye!")

    async def reset_immune_system(self) -> None:
        """Reset the entire immune system to its initial state."""
        print("\n🔄 Resetting Immune System State...")
        self.immune_handler.reset_all()
        # Re-initialize engine to reset its internal state as well
        self.engine = EnhancedMasterCycleEngine()
        self.immune_handler = self.engine.immune_handler
        print("✅ Immune system has been reset.")

    async def trigger_emergency_scenario(self) -> None:
        """Manually trigger an emergency quarantine scenario."""
        print("\n🚨 Triggering Emergency Quarantine Scenario")
        print("-" * 60)
        print()
            "   This will simulate a critical, recurring error to demonstrate"
        )
        print("   antibody formation and quarantine response.")

        @immune_protected(self.immune_handler)
        def critical_error_op(x):
            raise MemoryError("Simulated critical memory leak")

        print("\n   Injecting 20 critical errors...")
        for i in range(20):
            critical_error_op(i)
            time.sleep(0.5)
            status = self.immune_handler.get_full_status()
            print()
                f"\r   - Error {i + 1}/20 | Health: {status['overall_health']:.1%} | "
                f"Gateway: {status['neural_gateway_state']} | "
                f"Antibodies: {status['antibody_count']}",
                end="",
            )
        print("\n\n   Emergency scenario complete.")
        self._print_system_status()


async def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser()
        description="Schwabot Biological Immune System CLI"
    )
    parser.add_argument()
        "--test", action="store_true", help="Run the comprehensive test suite and exit"
    )
    parser.add_argument()
        "--dashboard", action="store_true", help="Launch the monitoring dashboard"
    )
    parser.add_argument()
        "--stress", action="store_true", help="Run a continuous stress test"
    )
    parser.add_argument()
        "--demo", action="store_true", help="Run the immune scenario demonstration"
    )

    args = parser.parse_args()
    cli = SchwabotImmuneCLI()

    if args.test:
        await cli.initialize_systems()
        await cli.run_comprehensive_test()
    elif args.dashboard:
        await cli.initialize_systems()
        await cli.start_monitoring_dashboard()
    elif args.stress:
        await cli.initialize_systems()
        await cli.run_stress_test()
    elif args.demo:
        await cli.initialize_systems()
        await cli.demonstrate_immune_scenarios()
    else:
        await cli.main_menu()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nCLI terminated by user.")
    except Exception as e:
        print(f"🚨 An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
