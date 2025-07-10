import argparse
import asyncio
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Set

import websockets

from core.master_cycle_engine import MasterCycleEngine
from utils.logging_setup import setup_logging

#!/usr/bin/env python3
"""QSC Diagnostic WebSocket Server.

Enhanced WebSocket server for streaming QSC (Quantum Static Core) and
GTS (Generalized Tensor Solutions) diagnostic data to React visualization.
Provides real-time immune system monitoring and visual alerts.
"""


# Import our core modules

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Setup logging
logger = setup_logging(__name__)


class QSCDiagnosticServer:
    """WebSocket server for QSC diagnostic data streaming."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the QSC diagnostic server."""
        self.config = config or self._default_config()

        # Initialize master cycle engine
        self.master_engine = MasterCycleEngine()

        # WebSocket management
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.server = None

        # Data streaming
        self.is_streaming = False
        self.stream_interval = self.config.get("stream_interval", 1.0)

        # Performance tracking
        self.total_messages_sent = 0
        self.total_connections = 0
        self.active_connections = 0

        # Alert system
        self.active_alerts: List[Dict[str, Any]] = []
        self.alert_thresholds = {
            "fibonacci_divergence": 0.007,
            "immune_activation": True,
            "ghost_floor_activation": True,
            "emergency_shutdown": True,
            "low_confidence": 0.3,
            "high_risk": 0.7,
        }

        # Threading
        self.executor = ThreadPoolExecutor(max_workers=4)

        logger.info(
            f"🧬📡 QSC Diagnostic Server initialized on port {self.config.get('port', 8766)}"
        )

    def _default_config():-> Dict[str, Any]:
        """Default configuration."""
        return {
            "host": "localhost",
            "port": 8766,  # Different from main tensor server
            "stream_interval": 1.0,
            "enable_cors": True,
            "max_clients": 50,
            "auto_alert_enabled": True,
            "alert_sound_enabled": True,
            "diagnostic_history_length": 100,
        }

    async def register_client(self, websocket: websockets.WebSocketServerProtocol):
        """Register a new WebSocket client."""
        self.clients.add(websocket)

        self.total_connections += 1
        self.active_connections = len(self.clients)

        logger.info(
            f"🔗 QSC Diagnostic client registered. Active connections: {self.active_connections}"
        )

        # Send initial diagnostic data
        try:
            initial_data = {
                "type": "qsc_connection_established",
                "timestamp": time.time(),
                "server_info": {
                    "version": "1.0.0",
                    "features": [
                        "qsc_immune_system",
                        "gts_integration",
                        "fibonacci_monitoring",
                        "profit_allocation",
                    ],
                    "stream_interval": self.stream_interval,
                    "alert_system": self.config.get("auto_alert_enabled", True),
                },
                "system_status": self.master_engine.get_system_status(),
            }
            await websocket.send(json.dumps(initial_data, default=str))
        except Exception as e:
            logger.error(f"Failed to send initial diagnostic data: {e}")

    async def unregister_client(self, websocket: websockets.WebSocketServerProtocol):
        """Unregister a WebSocket client."""
        self.clients.discard(websocket)
        self.active_connections = len(self.clients)
        logger.info(
            f"🔗 QSC Diagnostic client unregistered. Active connections: {self.active_connections}"
        )

    def generate_market_data():-> Dict[str, Any]:
        """Generate market data for testing purposes."""

        current_time = time.time()

        # Generate BTC price with realistic movement
        base_price = 51000
        price_wave = math.sin(current_time / 100) * 2000
        price_noise = random.uniform(-500, 500)
        btc_price = base_price + price_wave + price_noise

        # Generate price history
        price_history = [btc_price + random.uniform(-100, 100) for _ in range(10)]
        volume_history = [random.uniform(80, 150) for _ in range(10)]

        # Generate Fibonacci projection with potential divergence
        fib_base = [p * (1 + random.uniform(-0.002, 0.002)) for p in price_history]

        # Occasionally introduce larger divergence
        if random.random() < 0.1:  # 10% chance
            divergence_factor = random.uniform(0.005, 0.015)
            fib_base = [p * (1 + divergence_factor) for p in fib_base]

        # Generate order book
        spread = random.uniform(10, 50)
        orderbook = {
            "bids": [
                [btc_price - spread - i * 5, random.uniform(0.5, 3.0)] for i in range(5)
            ],
            "asks": [
                [btc_price + spread + i * 5, random.uniform(0.5, 3.0)] for i in range(5)
            ],
        }

        return {
            "btc_price": btc_price,
            "price_history": price_history,
            "volume_history": volume_history,
            "fibonacci_projection": fib_base,
            "orderbook": orderbook,
            "timestamp": current_time,
        }

    def process_diagnostics():-> Dict[str, Any]:
        """Process market data through master cycle engine and generate diagnostics."""
        diagnostics = self.master_engine.process_market_tick(market_data)

        # Check for alerts
        alerts = self._check_for_alerts(diagnostics)

        # Get Fibonacci echo data
        fibonacci_echo_data = self.master_engine.get_fibonacci_echo_data()

        # Get system performance
        system_status = self.master_engine.get_system_status()

        # Format for React visualization
        diagnostic_data = {
            "type": "qsc_diagnostic_stream",
            "timestamp": diagnostics.timestamp,
            # Core diagnostic data
            "diagnostics": {
                "system_mode": diagnostics.system_mode.value,
                "trading_decision": diagnostics.trading_decision.value,
                "confidence_score": diagnostics.confidence_score,
                "risk_assessment": diagnostics.risk_assessment,
                "immune_response_active": diagnostics.immune_response_active,
                "fibonacci_divergence": diagnostics.fibonacci_divergence,
                "orderbook_stability": diagnostics.orderbook_stability,
                "diagnostic_messages": diagnostics.diagnostic_messages,
            },
            # QSC Status
            "qsc_status": {
                "mode": diagnostics.qsc_status["mode"],
                "resonance_level": diagnostics.qsc_status["resonance_level"],
                "timeband_locked": diagnostics.qsc_status["timeband_locked"],
                "immune_triggered": diagnostics.qsc_status["immune_triggered"],
                "cycles_approved": diagnostics.qsc_status["cycles_approved"],
                "cycles_blocked": diagnostics.qsc_status["cycles_blocked"],
                "success_rate": diagnostics.qsc_status["success_rate"],
                "entropy_flux": diagnostics.qsc_status["entropy_flux"],
            },
            # Tensor Analysis
            "tensor_analysis": {
                "phi_resonance": diagnostics.tensor_analysis["phi_resonance"],
                "quantum_score": diagnostics.tensor_analysis["quantum_score"],
                "phase_bucket": diagnostics.tensor_analysis["phase_bucket"],
                "tensor_coherence": diagnostics.tensor_analysis["tensor_coherence"],
            },
            # Market Data
            "market_data": {
                "btc_price": market_data["btc_price"],
                "price_history": market_data["price_history"][-10:],  # Last 10 points
                "orderbook_imbalance": 1.0 - diagnostics.orderbook_stability,
            },
            # Fibonacci Echo Plot Data
            "fibonacci_echo": fibonacci_echo_data,
            # System Performance
            "system_performance": system_status,
            # Profit Allocation Status
            "profit_allocation": {
                "status": diagnostics.profit_allocation_status,
                "performance": system_status.get("profit_allocator_performance", {}),
            },
            # Alerts
            "alerts": alerts,
            "active_alerts_count": len(self.active_alerts),
        }

        return diagnostic_data

    def _check_for_alerts():-> List[Dict[str, Any]]:
        """Check for system alerts and generate notifications."""
        new_alerts = []
        current_time = time.time()

        # Fibonacci divergence alert
        if (
            diagnostics.fibonacci_divergence
            > self.alert_thresholds["fibonacci_divergence"]
        ):
            alert = {
                "id": f"fib_div_{int(current_time)}",
                "type": "fibonacci_divergence",
                "severity": "warning",
                "title": "Fibonacci Divergence Detected",
                "message": f"Price diverged from Fibonacci path by {diagnostics.fibonacci_divergence:.4f}",
                "timestamp": current_time,
                "auto_switch_tab": "diagnosticPanel",
            }
            new_alerts.append(alert)

        # Immune system activation alert
        if diagnostics.immune_response_active:
            alert = {
                "id": f"immune_{int(current_time)}",
                "type": "immune_activation",
                "severity": "error",
                "title": "QSC Immune System Activated",
                "message": "Trading immune system has blocked market activity",
                "timestamp": current_time,
                "auto_switch_tab": "diagnosticPanel",
            }
            new_alerts.append(alert)

        # Ghost floor mode alert
        if self.master_engine.ghost_floor_active:
            alert = {
                "id": f"ghost_{int(current_time)}",
                "type": "ghost_floor",
                "severity": "critical",
                "title": "Ghost Floor Mode Active",
                "message": "System has entered Ghost Floor Mode - All trading suspended",
                "timestamp": current_time,
                "auto_switch_tab": "diagnosticPanel",
            }
            new_alerts.append(alert)

        # Emergency shutdown alert
        if diagnostics.system_mode.value == "emergency_shutdown":
            alert = {
                "id": f"emergency_{int(current_time)}",
                "type": "emergency_shutdown",
                "severity": "critical",
                "title": "Emergency Shutdown",
                "message": "Emergency protocols activated - System shutdown in progress",
                "timestamp": current_time,
                "auto_switch_tab": "diagnosticPanel",
            }
            new_alerts.append(alert)

        # Low confidence alert
        if diagnostics.confidence_score < self.alert_thresholds["low_confidence"]:
            alert = {
                "id": f"low_conf_{int(current_time)}",
                "type": "low_confidence",
                "severity": "warning",
                "title": "Low Confidence Score",
                "message": f"System confidence dropped to {diagnostics.confidence_score:.2%}",
                "timestamp": current_time,
                "auto_switch_tab": "diagnosticPanel",
            }
            new_alerts.append(alert)

        # High risk alert
        if diagnostics.risk_assessment == "HIGH":
            alert = {
                "id": f"high_risk_{int(current_time)}",
                "type": "high_risk",
                "severity": "error",
                "title": "High Risk Assessment",
                "message": "Market conditions assessed as HIGH RISK",
                "timestamp": current_time,
                "auto_switch_tab": "diagnosticPanel",
            }
            new_alerts.append(alert)

        # Add new alerts to active list
        self.active_alerts.extend(new_alerts)

        # Clean up old alerts (keep last 10)
        if len(self.active_alerts) > 10:
            self.active_alerts = self.active_alerts[-10:]

        return new_alerts

    async def broadcast_diagnostic_data(self, data: Dict[str, Any]):
        """Broadcast diagnostic data to all connected clients."""
        if not self.clients:
            return

        message = json.dumps(data, default=str)
        disconnected_clients = set()

        for client in self.clients.copy():
            try:
                await client.send(message)
                self.total_messages_sent += 1
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.add(client)
            except Exception as e:
                logger.warning(f"Failed to send diagnostic message to client: {e}")
                disconnected_clients.add(client)

        # Remove disconnected clients
        for client in disconnected_clients:
            await self.unregister_client(client)

    async def handle_client_message(
        self, websocket: websockets.WebSocketServerProtocol, message: str
    ):
        """Handle incoming message from client."""
        try:
            data = json.loads(message)
            message_type = data.get("type", "unknown")

            logger.info(f"📨 Received QSC message type: {message_type}")

            if message_type == "request_system_status":
                # Client requesting system status
                status = self.master_engine.get_system_status()

                response = {
                    "type": "system_status_response",
                    "timestamp": time.time(),
                    "data": status,
                }
                await websocket.send(json.dumps(response, default=str))

            elif message_type == "request_fibonacci_echo":
                # Client requesting Fibonacci echo data
                echo_data = self.master_engine.get_fibonacci_echo_data()

                response = {
                    "type": "fibonacci_echo_response",
                    "timestamp": time.time(),
                    "data": echo_data,
                }
                await websocket.send(json.dumps(response, default=str))

            elif message_type == "reset_emergency_override":
                # Client requesting emergency reset
                self.master_engine.reset_emergency_override()

                response = {
                    "type": "emergency_reset_response",
                    "timestamp": time.time(),
                    "success": True,
                    "message": "Emergency override reset successfully",
                }
                await websocket.send(json.dumps(response, default=str))

            elif message_type == "clear_alerts":
                # Client clearing alerts
                self.active_alerts.clear()

                response = {
                    "type": "alerts_cleared_response",
                    "timestamp": time.time(),
                    "success": True,
                }
                await websocket.send(json.dumps(response, default=str))

            elif message_type == "update_alert_thresholds":
                # Client updating alert thresholds
                new_thresholds = data.get("thresholds", {})
                self.alert_thresholds.update(new_thresholds)

                response = {
                    "type": "thresholds_updated_response",
                    "timestamp": time.time(),
                    "success": True,
                    "thresholds": self.alert_thresholds,
                }
                await websocket.send(json.dumps(response, default=str))

            else:
                logger.warning(f"Unknown QSC message type: {message_type}")

        except json.JSONDecodeError:
            logger.error("Invalid JSON received from QSC client")
        except Exception as e:
            logger.error(f"Error handling QSC client message: {e}")

    async def diagnostic_streaming_loop(self):
        """Main diagnostic data streaming loop."""
        logger.info("📡🧬 Starting QSC diagnostic streaming loop")

        while self.is_streaming:
            try:
                if self.clients:
                    # Generate market data
                    market_data = self.generate_market_data()

                    # Process through master cycle engine
                    diagnostic_data = self.process_diagnostics(market_data)

                    # Broadcast to clients
                    await self.broadcast_diagnostic_data(diagnostic_data)

                await asyncio.sleep(self.stream_interval)

            except Exception as e:
                logger.error(f"Error in QSC diagnostic streaming loop: {e}")
                await asyncio.sleep(self.stream_interval)

    async def handle_client(
        self, websocket: websockets.WebSocketServerProtocol, path: str
    ):
        """Handle individual client connection."""
        await self.register_client(websocket)

        try:
            async for message in websocket:
                await self.handle_client_message(websocket, message)
        except websockets.exceptions.ConnectionClosed:
            logger.info("QSC Diagnostic client connection closed")
        except Exception as e:
            logger.error(f"Error handling QSC diagnostic client: {e}")
        finally:
            await self.unregister_client(websocket)

    async def start_server(self):
        """Start the QSC diagnostic WebSocket server."""
        host = self.config.get("host", "localhost")
        port = self.config.get("port", 8766)

        # Add CORS headers if enabled
        extra_headers = {}
        if self.config.get("enable_cors", True):
            extra_headers = {
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*",
                "Access-Control-Allow-Methods": "*",
            }

        self.server = await websockets.serve(
            self.handle_client, host, port, extra_headers=extra_headers
        )

        logger.info(
            f"🚀🧬 QSC Diagnostic WebSocket server started on ws://{host}:{port}"
        )

        # Start diagnostic streaming
        self.is_streaming = True
        streaming_task = asyncio.create_task(self.diagnostic_streaming_loop())

        return streaming_task

    async def stop_server(self):
        """Stop the QSC diagnostic WebSocket server."""
        self.is_streaming = False

        if self.server:
            self.server.close()
            await self.server.wait_closed()

        # Close all client connections
        if self.clients:
            await asyncio.gather(
                *[client.close() for client in self.clients], return_exceptions=True
            )

        self.executor.shutdown(wait=True)
        logger.info("🛑 QSC Diagnostic WebSocket server stopped")

    def get_server_stats():-> Dict[str, Any]:
        """Get QSC diagnostic server statistics."""
        return {
            "active_connections": self.active_connections,
            "total_connections": self.total_connections,
            "total_messages_sent": self.total_messages_sent,
            "is_streaming": self.is_streaming,
            "stream_interval": self.stream_interval,
            "active_alerts": len(self.active_alerts),
            "master_engine_status": self.master_engine.get_system_status(),
        }


async def main():
    """Main function for running the QSC diagnostic server."""

    parser = argparse.ArgumentParser(description="QSC Diagnostic WebSocket Server")
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8766, help="Server port")
    parser.add_argument(
        "--stream-interval", type=float, default=1.0, help="Stream interval in seconds"
    )

    args = parser.parse_args()

    config = {
        "host": args.host,
        "port": args.port,
        "stream_interval": args.stream_interval,
    }

    server = QSCDiagnosticServer(config)

    try:
        streaming_task = await server.start_server()

        # Keep server running
        await asyncio.gather(
            server.server.wait_closed(), streaming_task, return_exceptions=True
        )

    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        await server.stop_server()
        logger.info("QSC Diagnostic server shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
