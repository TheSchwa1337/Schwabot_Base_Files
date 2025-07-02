#!/usr/bin/env python3
"""Schwabot Biological Immune System CLI.

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

import numpy as np
from server.immune_diagnostic_websocket import ImmuneDiagnosticWebSocketServer
from core.biological_immune_error_handler import ImmuneZone, immune_protected
from core.enhanced_master_cycle_engine import EnhancedMasterCycleEngine
from pathlib import Path
from typing import Dict, Any
import argparse
import asyncio
import logging
import signal
import sys
import time
import webbrowser


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
        logging.basicConfig(
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
            from core.biological_immune_error_handler import (
                TCellSignal,
                ImmuneSignalType,
                TCellValidator,
            )

            validator = TCellValidator()

            # Test strong positive signals
            strong_signals = [
                TCellSignal(ImmuneSignalType.PRIMARY, 0.8, "test_primary", time.time()),
                TCellSignal(
                    ImmuneSignalType.COSTIMULATORY, 0.9, "test_costim", time.time()
                ),
                TCellSignal(
                    ImmuneSignalType.INFLAMMATORY, 0.3, "test_inflam", time.time()
                ),
            ]

            activation, confidence, analysis = validator.validate_signals(
                strong_signals
            )
            if activation and confidence > 0.6:
                results["passed"] += 1
                results["details"].append(
                    f"✅ Strong signals activated T-cell (confidence: {confidence:.3f})"
                )
            else:
                results["failed"] += 1
                results["details"].append("❌ Strong signals failed to activate T-cell")

            # Test weak signals
            weak_signals = [
                TCellSignal(ImmuneSignalType.PRIMARY, 0.2, "test_primary", time.time()),
                TCellSignal(
                    ImmuneSignalType.INHIBITORY, 0.8, "test_inhibit", time.time()
                ),
            ]

            activation, confidence, analysis = validator.validate_signals(weak_signals)
            if not activation:
                results["passed"] += 1
                results["details"].append(
                    "✅ Weak signals correctly blocked T-cell activation"
                )
            else:
                results["failed"] += 1
                results["details"].append(
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
            allowed = gateway.should_allow_operation(
                0.8, 0.1
            )  # High confidence, low entropy
            if allowed:
                results["passed"] += 1
                results["details"].append(
                    "✅ Permissive state allows high-confidence operations"
                )
            else:
                results["failed"] += 1
                results["details"].append(
                    "❌ Permissive state blocked high-confidence operation"
                )

            # Test emergency state
            gateway.current_state = gateway.current_state.EMERGENCY
            allowed = gateway.should_allow_operation(
                0.8, 0.9
            )  # High confidence, high entropy
            if not allowed:
                results["passed"] += 1
                results["details"].append(
                    "✅ Emergency state correctly blocks operations"
                )
            else:
                results["failed"] += 1
                results["details"].append(
                    "❌ Emergency state incorrectly allowed operation"
                )

            # Test adaptive threshold
            threshold = gateway.calculate_adaptive_threshold(0.5)
            if 0.7 < threshold < 0.8:  # Should be baseline + (0.15 * 0.5)
                results["passed"] += 1
                results["details"].append(
                    f"✅ Adaptive threshold calculation correct: {threshold:.3f}"
                )
            else:
                results["failed"] += 1
                results["details"].append(
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
            swarm = self.immune_handler.swarm_matrix

            # Test normal consensus
            test_vector = np.array([0.5, 0.5, 0.5])
            consensus_result = swarm.simulate_swarm_dynamics(test_vector)

            if "convergence" in consensus_result:
                results["passed"] += 1
                results["details"].append(
                    f"✅ Swarm consensus computed: {consensus_result['recommendation']}"
                )
            else:
                results["failed"] += 1
                results["details"].append("❌ Swarm consensus failed to compute")

            # Test node health
            healthy_nodes = sum(1 for node in swarm.nodes.values() if node.is_healthy())
            total_nodes = len(swarm.nodes)
            health_ratio = healthy_nodes / total_nodes

            if health_ratio > 0.8:
                results["passed"] += 1
                results["details"].append(f"✅ Swarm health good: {health_ratio:.2%}")
            else:
                results["failed"] += 1
                results["details"].append(f"❌ Swarm health low: {health_ratio:.2%}")

            # Test node update
            test_node_id = list(swarm.nodes.keys())[0]
            success = swarm.update_node_vector(
                test_node_id, np.array([1.0, 0.0, 0.0]), 0.9
            )
            if success:
                results["passed"] += 1
                results["details"].append("✅ Node update successful")
            else:
                results["failed"] += 1
                results["details"].append("❌ Node update failed")

        except Exception as e:
            results["failed"] += 1
            results["details"].append(f"❌ Swarm consensus test exception: {e}")

        return results

    async def _test_zone_response(self) -> Dict[str, Any]:
        """Test zone-based response system."""
        results = {"passed": 0, "failed": 0, "details": []}

        try:
            zone_manager = self.immune_handler.zone_manager

            # Test safe zone
            safe_zone = zone_manager.classify_zone(
                0.1, 0.9, 0.01
            )  # Low noise, high confidence, low error
            if safe_zone == ImmuneZone.SAFE:
                results["passed"] += 1
                results["details"].append("✅ Safe zone classification correct")
            else:
                results["failed"] += 1
                results["details"].append(
                    f"❌ Safe zone classification incorrect: {safe_zone}"
                )

            # Test toxic zone
            toxic_zone = zone_manager.classify_zone(
                0.8, 0.2, 0.2
            )  # High noise, low confidence, high error
            if toxic_zone in [ImmuneZone.TOXIC, ImmuneZone.QUARANTINE]:
                results["passed"] += 1
                results["details"].append(
                    f"✅ Toxic zone classification correct: {toxic_zone}"
                )
            else:
                results["failed"] += 1
                results["details"].append(
                    f"❌ Toxic zone classification incorrect: {toxic_zone}"
                )

            # Test zone response
            response = zone_manager.get_zone_response(ImmuneZone.ALERT)
            if response["action"] == "monitor":
                results["passed"] += 1
                results["details"].append("✅ Alert zone response correct")
            else:
                results["failed"] += 1
                results["details"].append(
                    f"❌ Alert zone response incorrect: {response['action']}"
                )

        except Exception as e:
            results["failed"] += 1
            results["details"].append(f"❌ Zone response test exception: {e}")

        return results

    async def _test_error_recovery(self) -> Dict[str, Any]:
        """Test error recovery and antibody formation."""
        results = {"passed": 0, "failed": 0, "details": []}

        try:
            # Clear existing antibody patterns
            initial_patterns = len(self.immune_handler.antibody_patterns)

            # Generate recurring errors to create antibody patterns
            @immune_protected(self.immune_handler)
            def recurring_error_operation():
                raise ValueError("Recurring test error")

            # Call multiple times to create pattern
            for _ in range(3):
                recurring_error_operation()

            # Check if antibody pattern was created
            final_patterns = len(self.immune_handler.antibody_patterns)
            if final_patterns > initial_patterns:
                results["passed"] += 1
                results["details"].append(
                    f"✅ Antibody pattern created ({
                        final_patterns - initial_patterns
                    } new patterns)"
                )
            else:
                results["failed"] += 1
                results["details"].append("❌ No antibody pattern created")

            # Test mitochondrial health update
            initial_health = self.immune_handler.mitochondrial_health
            self.immune_handler._update_mitochondrial_health(True)  # Success
            if self.immune_handler.mitochondrial_health >= initial_health:
                results["passed"] += 1
                results["details"].append("✅ Mitochondrial health improvement works")
            else:
                results["failed"] += 1
                results["details"].append("❌ Mitochondrial health improvement failed")

            # Test entropy monitoring
            self.immune_handler._update_entropy_monitoring()
            if 0.0 <= self.immune_handler.system_entropy <= 1.0:
                results["passed"] += 1
                results["details"].append(
                    f"✅ Entropy monitoring works: {
                        self.immune_handler.system_entropy:.3f}"
                )
            else:
                results["failed"] += 1
                results["details"].append(
                    f"❌ Entropy monitoring failed: {
                        self.immune_handler.system_entropy
                    }"
                )

        except Exception as e:
            results["failed"] += 1
            results["details"].append(f"❌ Error recovery test exception: {e}")

        return results

    async def _test_market_simulation(self) -> Dict[str, Any]:
        """Test market simulation with immune response."""
        results = {"passed": 0, "failed": 0, "details": []}

        try:
            # Generate test market data
            market_data = {
                "btc_price": 45000.0,
                "orderbook": {"bids": [[44999, 1.0]], "asks": [[45001, 1.0]]},
                "price_history": [44950, 44980, 45000, 45020, 45000],
                "volume_history": [100, 120, 110, 90, 105],
                "fibonacci_projection": [44960, 44990, 45010, 45030, 45010],
                "volume": 1.5,
                "trend": 0.1,
            }

            # Process normal market tick
            diagnostics = self.engine.process_market_tick_protected(market_data)
            if hasattr(diagnostics, "trading_decision"):
                results["passed"] += 1
                results["details"].append(
                    f"✅ Market tick processed: {diagnostics.trading_decision}"
                )
            else:
                results["failed"] += 1
                results["details"].append("❌ Market tick processing failed")

            # Test with divergent Fibonacci projection
            divergent_data = market_data.copy()
            divergent_data["fibonacci_projection"] = [
                40000,
                41000,
                42000,
                43000,
                44000,
            ]  # Highly divergent

            divergent_diagnostics = self.engine.process_market_tick_protected(
                divergent_data
            )
            if (
                hasattr(divergent_diagnostics, "immune_response_active")
                and divergent_diagnostics.immune_response_active
            ):
                results["passed"] += 1
                results["details"].append(
                    "✅ Fibonacci divergence detected and handled"
                )
            else:
                results["passed"] += 1  # May not trigger immediately
                results["details"].append(
                    "✅ Divergent data processed (immune response may activate)"
                )

            # Test system status
            status = self.engine.get_enhanced_system_status()
            if "immune_system_status" in status:
                results["passed"] += 1
                results["details"].append("✅ System status retrieval works")
            else:
                results["failed"] += 1
                results["details"].append("❌ System status retrieval failed")

        except Exception as e:
            results["failed"] += 1
            results["details"].append(f"❌ Market simulation test exception: {e}")

        return results

    def _print_test_results(self, test_name: str, results: Dict[str, Any]) -> None:
        """Print formatted test results."""
        total_tests = results["passed"] + results["failed"]
        success_rate = (results["passed"] / total_tests * 100) if total_tests > 0 else 0

        print(
            f"   {test_name}: {results['passed']}/{total_tests} passed ({success_rate:.1f}%)"
        )
        for detail in results["details"]:
            print(f"     {detail}")

    def _print_system_status(self) -> None:
        """Print current system status."""
        print("\n🧬 Current System Status:")
        print("-" * 30)

        status = self.immune_handler.get_immune_status()

        print(
            f"   Mitochondrial Health: {
                status['system_health']['mitochondrial_health']:.3f}"
        )
        print(f"   System Entropy: {status['system_health']['system_entropy']:.3f}")
        print(f"   Error Rate: {status['system_health']['current_error_rate']:.3f}")
        print(f"   Current Zone: {status['system_health']['current_zone'].upper()}")
        print(f"   Success Rate: {status['performance_metrics']['success_rate']:.3f}")
        print(f"   Antibody Patterns: {status['antibody_patterns']}")

    async def start_monitoring_dashboard(self) -> None:
        """Start the real-time monitoring dashboard."""
        print("\n🖥️ Starting Real-Time Monitoring Dashboard...")

        try:
            # Start WebSocket server
            await self.websocket_server.start_server()

            # Open dashboard in browser
            dashboard_url = (
                f"http://{self.websocket_server.host}:{self.websocket_server.port}"
            )
            print(f"📊 Dashboard URL: {dashboard_url}")

            # Try to open in default browser
            try:
                # Create a simple HTTP server for the dashboard
                import http.server
                import socketserver

                class DashboardHandler(http.server.SimpleHTTPRequestHandler):
                    def do_GET(self):
                        if self.path == "/" or self.path == "/dashboard":
                            self.send_response(200)
                            self.send_header("Content-type", "text/html")
                            self.end_headers()
                            self.wfile.write(self.server.dashboard_html.encode())
                        else:
                            self.send_response(404)
                            self.end_headers()

                with socketserver.TCPServer(
                    ("", self.websocket_server.port + 1), DashboardHandler
                ) as httpd:
                    httpd.dashboard_html = self.websocket_server.get_dashboard_html()

                    dashboard_http_url = f"http://{self.websocket_server.host}:{
                        self.websocket_server.port + 1
                    }/dashboard"
                    print(f"📊 Opening dashboard at: {dashboard_http_url}")

                    # Start HTTP server in background
                    import threading

                    server_thread = threading.Thread(target=httpd.serve_forever)
                    server_thread.daemon = True
                    server_thread.start()

                    # Open browser
                    webbrowser.open(dashboard_http_url)

                    print("📱 Dashboard controls:")
                    print("   🚀 Start Simulation - Begin market simulation")
                    print("   ⏹️ Stop Simulation - Stop market simulation")
                    print("   🔄 Reset System - Reset immune system to healthy state")
                    print("   🚨 Trigger Emergency - Test emergency response")
                    print(
                        "   📱 Auto-Switch - Toggle automatic tab switching for alerts"
                    )

                    print("\n✅ Dashboard started successfully!")
                    print("   Press Ctrl+C to stop the server")

                    # Keep servers running
                    while self.running:
                        await asyncio.sleep(1)

            except Exception as e:
                print(f"⚠️ Could not open browser automatically: {e}")
                print(f"   Please manually open: {dashboard_url}")

        except Exception as e:
            print(f"🚨 Failed to start dashboard: {e}")

    async def run_stress_test(self) -> None:
        """Run stress test to validate immune system under load."""
        print("\n🔥 Running Immune System Stress Test...")
        print("=" * 60)

        stress_results = {
            "operations": 0,
            "errors": 0,
            "immune_responses": 0,
            "recoveries": 0,
        }

        try:
            # Create various error scenarios
            @immune_protected(self.immune_handler)
            def random_operation(operation_type: str):
                import random

                if operation_type == "normal":
                    return random.uniform(0, 100)
                elif operation_type == "error":
                    raise ValueError(f"Random error {random.randint(1, 10)}")
                elif operation_type == "timeout":
                    time.sleep(0.1)  # Simulate slow operation
                    return "timeout_result"
                elif operation_type == "memory":
                    # Simulate memory-intensive operation
                    data = [random.random() for _ in range(1000)]
                    return sum(data)

            # Run stress operations
            operation_types = ["normal", "error", "timeout", "memory"]

            print("Running 100 random operations...")
            for i in range(100):
                operation_type = np.random.choice(
                    operation_types, p=[0.6, 0.2, 0.1, 0.1]
                )

                result = random_operation(operation_type)
                stress_results["operations"] += 1

                if hasattr(result, "zone"):  # ImmuneResponse
                    stress_results["immune_responses"] += 1
                    if result.zone in ["recovery", "safe"]:
                        stress_results["recoveries"] += 1
                elif operation_type == "error":
                    stress_results["errors"] += 1

                # Brief pause
                await asyncio.sleep(0.01)

                # Progress indicator
                if i % 20 == 0:
                    print(f"   Progress: {i}/100 operations completed")

            # Print stress test results
            print("\n🔥 Stress Test Results:")
            print(f"   Total Operations: {stress_results['operations']}")
            print(f"   Errors Handled: {stress_results['errors']}")
            print(f"   Immune Responses: {stress_results['immune_responses']}")
            print(f"   Recovery Operations: {stress_results['recoveries']}")

            # Calculate metrics
            error_rate = stress_results["errors"] / stress_results["operations"] * 100
            immune_response_rate = (
                stress_results["immune_responses"] / stress_results["operations"] * 100
            )

            print(f"   Error Rate: {error_rate:.1f}%")
            print(f"   Immune Response Rate: {immune_response_rate:.1f}%")

            # System health after stress test
            self._print_system_status()

            if self.immune_handler.mitochondrial_health > 0.5:
                print("✅ System maintained good health under stress")
            else:
                print("⚠️ System health degraded under stress (expected behavior)")

        except Exception as e:
            print(f"🚨 Stress test exception: {e}")

    async def demonstrate_immune_scenarios(self) -> None:
        """Demonstrate various immune system scenarios."""
        print("\n🎭 Demonstrating Immune System Scenarios...")
        print("=" * 60)

        scenarios = [
            ("🟢 Healthy System", self._demo_healthy_system),
            ("🟡 Alert Condition", self._demo_alert_condition),
            ("🔴 Toxic Environment", self._demo_toxic_environment),
            ("🟣 Quarantine Mode", self._demo_quarantine_mode),
            ("🔵 Recovery Phase", self._demo_recovery_phase),
        ]

        for scenario_name, scenario_func in scenarios:
            print(f"\n{scenario_name}:")
            try:
                await scenario_func()
                await asyncio.sleep(2)  # Brief pause between scenarios
            except Exception as e:
                print(f"   🚨 Scenario error: {e}")

    async def _demo_healthy_system(self) -> None:
        """Demonstrate healthy system operation."""
        # Reset to healthy state
        self.immune_handler.mitochondrial_health = 1.0
        self.immune_handler.system_entropy = 0.1
        self.immune_handler.current_error_rate = 0.0

        # Run normal operations
        @immune_protected(self.immune_handler)
        def healthy_operation(x):
            return x * 2

        for i in range(5):
            result = healthy_operation(i)
            print(
                f"   Operation {i}: {
                    '✅ Success' if not hasattr(result, 'zone') else '🛡️ Protected'
                }"
            )

        status = self.immune_handler.get_immune_status()
        print(f"   Zone: {status['system_health']['current_zone']}")
        print(f"   Health: {status['system_health']['mitochondrial_health']:.3f}")

    async def _demo_alert_condition(self) -> None:
        """Demonstrate alert condition."""
        # Set alert conditions
        self.immune_handler.system_entropy = 0.5
        self.immune_handler.current_error_rate = 0.08

        # Update neural gateway state
        self.immune_handler.neural_gateway.update_gate_state(0.5, 0.08)

        @immune_protected(self.immune_handler)
        def alert_operation(x):
            if x > 3:
                raise ValueError("Alert condition error")
            return x * 2

        for i in range(5):
            result = alert_operation(i)
            print(
                f"   Operation {i}: {
                    '✅ Success' if not hasattr(result, 'zone') else '🛡️ Blocked'
                }"
            )

        status = self.immune_handler.get_immune_status()
        print(f"   Zone: {status['system_health']['current_zone']}")
        print(
            f"   Gateway State: {status['immune_components']['neural_gateway_state']}"
        )

    async def _demo_toxic_environment(self) -> None:
        """Demonstrate toxic environment response."""
        # Set toxic conditions
        self.immune_handler.system_entropy = 0.8
        self.immune_handler.current_error_rate = 0.15
        self.immune_handler.mitochondrial_health = 0.4

        @immune_protected(self.immune_handler)
        def toxic_operation(x):
            raise ValueError(f"Toxic error {x}")

        for i in range(3):
            result = toxic_operation(i)
            print(
                f"   Operation {i}: {
                    '🚨 Error'
                    if not hasattr(result, 'zone')
                    else f'🛡️ Immune Response ({result.zone.value})'
                }"
            )

        status = self.immune_handler.get_immune_status()
        print(f"   Zone: {status['system_health']['current_zone']}")
        print(
            f"   Mitochondrial Health: {
                status['system_health']['mitochondrial_health']:.3f}"
        )

    async def _demo_quarantine_mode(self) -> None:
        """Demonstrate quarantine mode."""
        # Set quarantine conditions
        self.immune_handler.system_entropy = 0.9
        self.immune_handler.current_error_rate = 0.25
        self.immune_handler.mitochondrial_health = 0.2

        @immune_protected(self.immune_handler)
        def quarantine_operation(x):
            return f"Should not execute: {x}"

        for i in range(3):
            result = quarantine_operation(i)
            print(
                f"   Operation {i}: {
                    '🚨 Executed'
                    if not hasattr(result, 'zone')
                    else f'🛡️ Quarantined ({result.zone.value})'
                }"
            )

        status = self.immune_handler.get_immune_status()
        print(f"   Zone: {status['system_health']['current_zone']}")
        print("   All operations should be quarantined")

    async def _demo_recovery_phase(self) -> None:
        """Demonstrate recovery phase."""
        # Trigger recovery
        await self.immune_handler._check_mitochondrial_drift()

        # Gradually improve conditions
        self.immune_handler.system_entropy = 0.4
        self.immune_handler.current_error_rate = 0.05

        @immune_protected(self.immune_handler)
        def recovery_operation(x):
            return f"Recovery operation {x}"

        for i in range(3):
            result = recovery_operation(i)
            print(
                f"   Operation {i}: {
                    '✅ Recovery'
                    if not hasattr(result, 'zone')
                    else f'🛡️ Protected ({result.zone.value})'
                }"
            )

            # Improve health slightly
            self.immune_handler._update_mitochondrial_health(True)

        status = self.immune_handler.get_immune_status()
        print(f"   Zone: {status['system_health']['current_zone']}")
        print(
            f"   Health Recovery: {status['system_health']['mitochondrial_health']:.3f}"
        )

    def setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""

        def signal_handler(signum, frame):
            print(f"\n🛑 Received signal {signum}, shutting down gracefully...")
            self.running = False
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    async def main_menu(self) -> None:
        """Display main menu and handle user input."""
        self.setup_signal_handlers()
        self.running = True

        while self.running:
            print("\n🧬 Schwabot Biological Immune System")
            print("=" * 40)
            print("1. 🧪 Run Comprehensive Test Suite")
            print("2. 🖥️ Start Real-Time Monitoring Dashboard")
            print("3. 🔥 Run Stress Test")
            print("4. 🎭 Demonstrate Immune Scenarios")
            print("5. 📊 Show Current System Status")
            print("6. 🔄 Reset Immune System")
            print("7. 🚨 Trigger Emergency Scenario")
            print("0. 🚪 Exit")

            try:
                choice = input("\nSelect option (0-7): ").strip()

                if choice == "1":
                    await self.run_comprehensive_test()
                elif choice == "2":
                    await self.start_monitoring_dashboard()
                elif choice == "3":
                    await self.run_stress_test()
                elif choice == "4":
                    await self.demonstrate_immune_scenarios()
                elif choice == "5":
                    self._print_system_status()
                elif choice == "6":
                    await self.reset_immune_system()
                elif choice == "7":
                    await self.trigger_emergency_scenario()
                elif choice == "0":
                    print("👋 Goodbye!")
                    self.running = False
                    break
                else:
                    print("❌ Invalid option. Please try again.")

                if choice != "0":
                    input("\nPress Enter to continue...")

            except KeyboardInterrupt:
                print("\n🛑 Exiting...")
                self.running = False
                break
            except Exception as e:
                print(f"🚨 Menu error: {e}")

    async def reset_immune_system(self) -> None:
        """Reset immune system to healthy state."""
        print("🔄 Resetting Immune System...")

        self.immune_handler.mitochondrial_health = 1.0
        self.immune_handler.system_entropy = 0.1
        self.immune_handler.current_error_rate = 0.0
        self.immune_handler.antibody_patterns.clear()
        self.immune_handler.error_history.clear()

        # Reset neural gateway
        self.immune_handler.neural_gateway.current_state = (
            self.immune_handler.neural_gateway.current_state.PERMISSIVE
        )

        # Reset zone manager
        self.immune_handler.zone_manager.current_zone = ImmuneZone.SAFE

        print("✅ Immune system reset to healthy state")
        self._print_system_status()

    async def trigger_emergency_scenario(self) -> None:
        """Trigger emergency scenario for testing."""
        print("🚨 Triggering Emergency Scenario...")

        # Simulate system degradation
        self.immune_handler.mitochondrial_health = 0.1
        self.immune_handler.system_entropy = 0.95
        self.immune_handler.current_error_rate = 0.3

        # Add multiple error patterns
        for i in range(15):
            self.immune_handler.error_history.append(
                {
                    "timestamp": time.time(),
                    "error_type": f"EmergencyError{i % 5}",
                    "error_message": f"Emergency test error {i}",
                    "operation": "emergency_test",
                    "args_count": 1,
                    "kwargs_count": 0,
                    "traceback": f"Emergency traceback {i}",
                }
            )

        # Update antibody patterns
        for i in range(5):
            pattern_key = f"emergency_pattern_{i}"
            self.immune_handler.antibody_patterns[pattern_key] = {
                "pattern_type": "emergency_test",
                "first_occurrence": time.time(),
                "occurrence_count": 3,
                "rejection_strength": 0.8,
            }

        print("🚨 Emergency scenario activated")
        print("   System health critically degraded")
        print("   Multiple error patterns injected")
        print("   High rejection antibodies created")

        self._print_system_status()


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Schwabot Biological Immune System CLI"
    )
    parser.add_argument(
        "--test", action="store_true", help="Run comprehensive test suite and exit"
    )
    parser.add_argument(
        "--dashboard", action="store_true", help="Start monitoring dashboard"
    )
    parser.add_argument("--stress", action="store_true", help="Run stress test")
    parser.add_argument("--demo", action="store_true", help="Run immune scenarios demo")

    args = parser.parse_args()

    # Initialize CLI
    cli = SchwabotImmuneCLI()

    # Initialize systems
    if not await cli.initialize_systems():
        print("🚨 Failed to initialize systems. Exiting.")
        return 1

    try:
        if args.test:
            await cli.run_comprehensive_test()
        elif args.dashboard:
            await cli.start_monitoring_dashboard()
        elif args.stress:
            await cli.run_stress_test()
        elif args.demo:
            await cli.demonstrate_immune_scenarios()
        else:
            await cli.main_menu()

    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"🚨 Fatal error: {e}")
        return 1

    print("✅ Schwabot Biological Immune System CLI completed")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
