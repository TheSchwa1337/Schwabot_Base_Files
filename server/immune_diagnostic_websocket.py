#!/usr/bin/env python3
"""Immune Diagnostic WebSocket Server.

Real-time monitoring and visualization server for the biological immune system.
Provides comprehensive diagnostics, alerts, and auto-tab switching for critical events.
Streams immune system metrics, zone changes, and recovery operations.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Set
import asyncio
import json
import logging
import time

import numpy as np
import websockets

from dataclasses import dataclass, asdict
from enum import Enum


from core.enhanced_master_cycle_engine import (
    EnhancedMasterCycleEngine,
    EnhancedSystemMode,
)
from core.biological_immune_error_handler import (
    BiologicalImmuneErrorHandler,
    ImmuneZone,
)

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class ImmuneAlert:
    """Immune system alert."""

    timestamp: float
    level: AlertLevel
    zone: str
    message: str
    component: str
    mitochondrial_health: float
    system_entropy: float
    recommended_action: str
    auto_switch_tab: bool = False


class ImmuneDiagnosticWebSocketServer:
    """WebSocket server for immune system diagnostics."""

    def __init__(self, host: str = "localhost", port: int = 8767):
        """Initialize diagnostic WebSocket server.

        Args:
            host: Server host
            port: Server port
        """
        self.host = host
        self.port = port
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.running = False

        # Initialize enhanced systems
        self.engine = EnhancedMasterCycleEngine()
        self.immune_handler = self.engine.immune_handler

        # Alert management
        self.alerts: List[ImmuneAlert] = []
        self.max_alerts = 1000
        self.last_zone = ImmuneZone.SAFE
        self.alert_thresholds = {
            "mitochondrial_health_critical": 0.3,
            "mitochondrial_health_warning": 0.5,
            "system_entropy_critical": 0.8,
            "system_entropy_warning": 0.6,
            "error_rate_critical": 0.15,
            "error_rate_warning": 0.05,
        }

        # Real-time metrics
        self.metrics_history: List[Dict[str, Any]] = []
        self.max_history = 500

        # Simulation data for demo
        self.simulation_active = False
        self.btc_price = 45000.0
        self.price_trend = 1.0

        logger.info(
            f"🧬 Immune Diagnostic WebSocket Server initialized on {host}:{port}"
        )

    async def start_server(self) -> None:
        """Start the WebSocket server."""
        self.running = True

        # Start background tasks
        asyncio.create_task(self._monitoring_loop())
        asyncio.create_task(self._market_simulation_loop())

        # Start enhanced engine monitoring
        await self.engine.start_enhanced_monitoring()

        # Start WebSocket server
        server = await websockets.serve(self.handle_client, self.host, self.port)

        logger.info(
            f"🧬 Immune Diagnostic WebSocket Server started on ws://{self.host}:{self.port}"
        )
        return server

    async def stop_server(self) -> None:
        """Stop the WebSocket server."""
        self.running = False
        await self.engine.stop_enhanced_monitoring()
        logger.info("🧬 Immune Diagnostic WebSocket Server stopped")

    async def handle_client(self, websocket, path):
        """Handle WebSocket client connections."""
        self.clients.add(websocket)
        client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        logger.info(f"🔗 Client connected: {client_id}")

        try:
            # Send initial status
            await self.send_initial_status(websocket)

            # Handle incoming messages
            async for message in websocket:
                await self.handle_message(websocket, message)

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"🔌 Client disconnected: {client_id}")
        except Exception as e:
            logger.error(f"🚨 Client error {client_id}: {e}")
        finally:
            self.clients.discard(websocket)

    async def send_initial_status(self, websocket) -> None:
        """Send initial system status to new client."""
        status = self.get_comprehensive_status()
        message = {"type": "initial_status", "timestamp": time.time(), "data": status}
        await websocket.send(json.dumps(message))

    async def handle_message(self, websocket, message: str) -> None:
        """Handle incoming WebSocket messages."""
        try:
            data = json.loads(message)
            message_type = data.get("type", "unknown")

            if message_type == "start_simulation":
                self.simulation_active = True
                await self.broadcast_message(
                    {
                        "type": "simulation_status",
                        "active": True,
                        "message": "Market simulation started",
                    }
                )

            elif message_type == "stop_simulation":
                self.simulation_active = False
                await self.broadcast_message(
                    {
                        "type": "simulation_status",
                        "active": False,
                        "message": "Market simulation stopped",
                    }
                )

            elif message_type == "reset_immune_system":
                await self.reset_immune_system()
                await self.broadcast_message(
                    {"type": "system_reset", "message": "Immune system reset completed"}
                )

            elif message_type == "trigger_emergency":
                await self.trigger_emergency_scenario()

            elif message_type == "get_detailed_status":
                detailed_status = self.get_comprehensive_status()
                await websocket.send(
                    json.dumps(
                        {
                            "type": "detailed_status",
                            "timestamp": time.time(),
                            "data": detailed_status,
                        }
                    )
                )

        except json.JSONDecodeError:
            logger.error(f"🚨 Invalid JSON message: {message}")
        except Exception as e:
            logger.error(f"🚨 Message handling error: {e}")

    async def broadcast_message(self, message: Dict[str, Any]) -> None:
        """Broadcast message to all connected clients."""
        if not self.clients:
            return

        message_str = json.dumps(message)
        disconnected_clients = set()

        for client in self.clients:
            try:
                await client.send(message_str)
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.add(client)
            except Exception as e:
                logger.error(f"🚨 Broadcast error: {e}")
                disconnected_clients.add(client)

        # Clean up disconnected clients
        self.clients -= disconnected_clients

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop for immune system diagnostics."""
        while self.running:
            try:
                # Get current system status
                status = self.get_comprehensive_status()

                # Check for alerts
                alerts = self.check_for_alerts(status)

                # Store metrics history
                self.metrics_history.append(
                    {
                        "timestamp": time.time(),
                        "mitochondrial_health": status["immune_status"][
                            "system_health"
                        ]["mitochondrial_health"],
                        "system_entropy": status["immune_status"]["system_health"][
                            "system_entropy"
                        ],
                        "error_rate": status["immune_status"]["system_health"][
                            "current_error_rate"
                        ],
                        "current_zone": status["immune_status"]["system_health"][
                            "current_zone"
                        ],
                        "success_rate": status["immune_status"]["performance_metrics"][
                            "success_rate"
                        ],
                    }
                )

                if len(self.metrics_history) > self.max_history:
                    self.metrics_history.pop(0)

                # Broadcast real-time update
                await self.broadcast_message(
                    {
                        "type": "real_time_update",
                        "timestamp": time.time(),
                        "data": {
                            "status": status,
                            "alerts": [asdict(alert) for alert in alerts],
                            "metrics_history": self.metrics_history[
                                -50:
                            ],  # Last 50 points
                        },
                    }
                )

                # Process alerts
                for alert in alerts:
                    await self.process_alert(alert)

                await asyncio.sleep(2.0)  # Update every 2 seconds

            except Exception as e:
                logger.error(f"🚨 Monitoring loop error: {e}")
                await asyncio.sleep(5.0)

    async def _market_simulation_loop(self) -> None:
        """Market simulation loop for testing immune responses."""
        while self.running:
            try:
                if not self.simulation_active:
                    await asyncio.sleep(1.0)
                    continue

                # Simulate market conditions
                market_data = self.generate_simulated_market_data()

                # Process through enhanced engine
                diagnostics = self.engine.process_market_tick_protected(market_data)

                # Broadcast trading decision
                await self.broadcast_message(
                    {
                        "type": "trading_decision",
                        "timestamp": time.time(),
                        "data": {
                            "btc_price": market_data["btc_price"],
                            "decision": (
                                diagnostics.trading_decision
                                if not isinstance(diagnostics, Exception)
                                else "ERROR"
                            ),
                            "confidence": (
                                diagnostics.confidence_score
                                if not isinstance(diagnostics, Exception)
                                else 0.0
                            ),
                            "zone": (
                                diagnostics.immune_zone
                                if not isinstance(diagnostics, Exception)
                                else "quarantine"
                            ),
                            "system_mode": (
                                diagnostics.system_mode.value
                                if not isinstance(diagnostics, Exception)
                                else "error"
                            ),
                        },
                    }
                )

                await asyncio.sleep(1.0)  # Simulate 1 tick per second

            except Exception as e:
                logger.error(f"🚨 Market simulation error: {e}")
                await asyncio.sleep(2.0)

    def generate_simulated_market_data(self) -> Dict[str, Any]:
        """Generate simulated market data for testing."""
        # Add some randomness and trends
        price_change = np.random.uniform(-200, 200) + (self.price_trend * 10)
        self.btc_price += price_change

        # Keep price in reasonable range
        self.btc_price = max(20000, min(80000, self.btc_price))

        # Randomly change trend
        if np.random.random() < 0.05:  # 5% chance
            self.price_trend = np.random.uniform(-1, 1)

        # Generate price history
        price_history = [self.btc_price + np.random.uniform(-50, 50) for _ in range(5)]

        # Generate Fibonacci projection (sometimes divergent)
        fib_base = self.btc_price
        if np.random.random() < 0.2:  # 20% chance of divergence
            fib_base += np.random.uniform(-500, 500)  # Cause divergence

        fibonacci_projection = [fib_base + (i * 10) for i in range(5)]

        return {
            "btc_price": self.btc_price,
            "orderbook": {
                "bids": [
                    [self.btc_price - i, np.random.uniform(0.5, 2.0)]
                    for i in range(1, 6)
                ],
                "asks": [
                    [self.btc_price + i, np.random.uniform(0.5, 2.0)]
                    for i in range(1, 6)
                ],
            },
            "price_history": price_history,
            "volume_history": [np.random.uniform(50, 200) for _ in range(5)],
            "fibonacci_projection": fibonacci_projection,
            "volume": np.random.uniform(0.5, 3.0),
            "trend": self.price_trend,
        }

    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        engine_status = self.engine.get_enhanced_system_status()

        # Add additional metrics
        recent_decisions = (
            self.engine.decision_history[-10:] if self.engine.decision_history else []
        )

        return {
            "engine_status": engine_status,
            "immune_status": engine_status["immune_system_status"],
            "qsc_status": engine_status["qsc_status"],
            "recent_decisions": [
                {
                    "timestamp": d.timestamp,
                    "decision": d.trading_decision,
                    "confidence": d.confidence_score,
                    "zone": d.immune_zone,
                    "risk": d.risk_assessment,
                }
                for d in recent_decisions
            ],
            "alerts": [asdict(alert) for alert in self.alerts[-20:]],  # Last 20 alerts
            "simulation_active": self.simulation_active,
            "current_btc_price": self.btc_price,
        }

    def check_for_alerts(self, status: Dict[str, Any]) -> List[ImmuneAlert]:
        """Check for system alerts based on current status."""
        alerts = []
        current_time = time.time()
        immune_status = status["immune_status"]

        # Check mitochondrial health
        mito_health = immune_status["system_health"]["mitochondrial_health"]
        if mito_health < self.alert_thresholds["mitochondrial_health_critical"]:
            alerts.append(
                ImmuneAlert(
                    timestamp=current_time,
                    level=AlertLevel.CRITICAL,
                    zone=immune_status["system_health"]["current_zone"],
                    message=f"Critical mitochondrial health: {mito_health:.3f}",
                    component="mitochondrial_system",
                    mitochondrial_health=mito_health,
                    system_entropy=immune_status["system_health"]["system_entropy"],
                    recommended_action="immediate_recovery_protocol",
                    auto_switch_tab=True,
                )
            )
        elif mito_health < self.alert_thresholds["mitochondrial_health_warning"]:
            alerts.append(
                ImmuneAlert(
                    timestamp=current_time,
                    level=AlertLevel.WARNING,
                    zone=immune_status["system_health"]["current_zone"],
                    message=f"Low mitochondrial health: {mito_health:.3f}",
                    component="mitochondrial_system",
                    mitochondrial_health=mito_health,
                    system_entropy=immune_status["system_health"]["system_entropy"],
                    recommended_action="monitor_and_prepare_recovery",
                )
            )

        # Check system entropy
        entropy = immune_status["system_health"]["system_entropy"]
        if entropy > self.alert_thresholds["system_entropy_critical"]:
            alerts.append(
                ImmuneAlert(
                    timestamp=current_time,
                    level=AlertLevel.CRITICAL,
                    zone=immune_status["system_health"]["current_zone"],
                    message=f"Critical system entropy: {entropy:.3f}",
                    component="entropy_monitor",
                    mitochondrial_health=mito_health,
                    system_entropy=entropy,
                    recommended_action="entropy_stabilization_protocol",
                    auto_switch_tab=True,
                )
            )

        # Check error rate
        error_rate = immune_status["system_health"]["current_error_rate"]
        if error_rate > self.alert_thresholds["error_rate_critical"]:
            alerts.append(
                ImmuneAlert(
                    timestamp=current_time,
                    level=AlertLevel.CRITICAL,
                    zone=immune_status["system_health"]["current_zone"],
                    message=f"High error rate: {error_rate:.3f}",
                    component="error_tracking",
                    mitochondrial_health=mito_health,
                    system_entropy=entropy,
                    recommended_action="error_mitigation_protocol",
                    auto_switch_tab=True,
                )
            )

        # Check zone changes
        current_zone_name = immune_status["system_health"]["current_zone"]
        current_zone = (
            ImmuneZone(current_zone_name)
            if current_zone_name in [z.value for z in ImmuneZone]
            else ImmuneZone.SAFE
        )

        if current_zone != self.last_zone:
            level = AlertLevel.INFO
            auto_switch = False

            if current_zone in [ImmuneZone.TOXIC, ImmuneZone.QUARANTINE]:
                level = AlertLevel.CRITICAL
                auto_switch = True
            elif current_zone == ImmuneZone.ALERT:
                level = AlertLevel.WARNING

            alerts.append(
                ImmuneAlert(
                    timestamp=current_time,
                    level=level,
                    zone=current_zone.value,
                    message=f"Zone change: {self.last_zone.value} → {current_zone.value}",
                    component="zone_manager",
                    mitochondrial_health=mito_health,
                    system_entropy=entropy,
                    recommended_action=f"zone_{current_zone.value}_protocol",
                    auto_switch_tab=auto_switch,
                )
            )

            self.last_zone = current_zone

        # Store alerts
        self.alerts.extend(alerts)
        if len(self.alerts) > self.max_alerts:
            self.alerts = self.alerts[-self.max_alerts :]

        return alerts

    async def process_alert(self, alert: ImmuneAlert) -> None:
        """Process and broadcast alert."""
        # Broadcast alert
        await self.broadcast_message(
            {"type": "alert", "timestamp": time.time(), "data": asdict(alert)}
        )

        # Log alert
        level_emoji = {
            AlertLevel.INFO: "ℹ️",
            AlertLevel.WARNING: "⚠️",
            AlertLevel.CRITICAL: "🚨",
            AlertLevel.EMERGENCY: "🆘",
        }

        emoji = level_emoji.get(alert.level, "❓")
        logger.info(
            f"{emoji} {alert.level.value.upper()}: {alert.message} (Zone: {alert.zone})"
        )

    async def reset_immune_system(self) -> None:
        """Reset the immune system to healthy state."""
        # Reset immune handler
        self.immune_handler.mitochondrial_health = 1.0
        self.immune_handler.system_entropy = 0.1
        self.immune_handler.current_error_rate = 0.0
        self.immune_handler.antibody_patterns.clear()
        self.immune_handler.error_history.clear()

        # Reset engine state
        self.engine.system_mode = EnhancedSystemMode.NORMAL

        # Clear alerts
        self.alerts.clear()
        self.last_zone = ImmuneZone.SAFE

        logger.info("🧬 Immune system reset completed")

    async def trigger_emergency_scenario(self) -> None:
        """Trigger emergency scenario for testing."""
        # Simulate system degradation
        self.immune_handler.mitochondrial_health = 0.2
        self.immune_handler.system_entropy = 0.9
        self.immune_handler.current_error_rate = 0.25

        # Add some error patterns
        for i in range(10):
            self.immune_handler.error_history.append(
                {
                    "timestamp": time.time(),
                    "error_type": f"TestError{i%3}",
                    "error_message": f"Simulated error {i}",
                    "operation": "test_operation",
                    "args_count": 2,
                    "kwargs_count": 1,
                    "traceback": "Simulated traceback",
                }
            )

        await self.broadcast_message(
            {
                "type": "emergency_triggered",
                "message": "Emergency scenario activated for testing",
            }
        )

        logger.warning("🚨 Emergency scenario triggered")

    def get_dashboard_html(self) -> str:
        """Generate HTML dashboard for immune system monitoring."""
        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🧬 Schwabot Immune System Dashboard</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            min-height: 100vh;
        }}
        .dashboard {{
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 20px;
            max-width: 1400px;
            margin: 0 auto;
        }}
        .card {{
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 20px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            transition: transform 0.3s ease;
        }}
        .card:hover {{
            transform: translateY(-5px);
        }}
        .card h3 {{
            margin-top: 0;
            color: #FFD700;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .status-indicator {{
            width: 20px;
            height: 20px;
            border-radius: 50%;
            display: inline-block;
        }}
        .status-safe {{ background: #4CAF50; }}
        .status-alert {{ background: #FF9800; }}
        .status-toxic {{ background: #F44336; }}
        .status-quarantine {{ background: #9C27B0; }}
        .status-recovery {{ background: #2196F3; }}
        .metric {{
            display: flex;
            justify-content: space-between;
            margin: 10px 0;
            padding: 10px;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 8px;
        }}
        .metric-value {{
            font-weight: bold;
            color: #FFD700;
        }}
        .alert {{
            margin: 5px 0;
            padding: 10px;
            border-radius: 5px;
            font-size: 14px;
        }}
        .alert-critical {{
            background: rgba(244, 67, 54, 0.3);
            border-left: 4px solid #F44336;
        }}
        .alert-warning {{
            background: rgba(255, 152, 0, 0.3);
            border-left: 4px solid #FF9800;
        }}
        .alert-info {{
            background: rgba(33, 150, 243, 0.3);
            border-left: 4px solid #2196F3;
        }}
        .controls {{
            display: flex;
            gap: 10px;
            margin: 20px 0;
        }}
        button {{
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 8px;
            cursor: pointer;
            font-weight: bold;
            transition: all 0.3s ease;
        }}
        button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
        }}
        button:disabled {{
            opacity: 0.5;
            cursor: not-allowed;
        }}
        .full-width {{
            grid-column: 1 / -1;
        }}
        #chart {{
            width: 100%;
            height: 300px;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 10px;
            margin: 20px 0;
        }}
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <h1 style="text-align: center; margin-bottom: 30px;">🧬 Schwabot Biological Immune System Dashboard</h1>

    <div class="controls">
        <button onclick="startSimulation()">🚀 Start Simulation</button>
        <button onclick="stopSimulation()">⏹️ Stop Simulation</button>
        <button onclick="resetSystem()">🔄 Reset System</button>
        <button onclick="triggerEmergency()">🚨 Trigger Emergency</button>
        <button onclick="toggleAutoSwitch()">📱 Auto-Switch: <span id="autoSwitchStatus">ON</span></button>
    </div>

    <div class="dashboard">
        <div class="card">
            <h3>🧬 System Health</h3>
            <div class="metric">
                <span>Mitochondrial Health</span>
                <span class="metric-value" id="mitoHealth">--</span>
            </div>
            <div class="metric">
                <span>System Entropy</span>
                <span class="metric-value" id="entropy">--</span>
            </div>
            <div class="metric">
                <span>Error Rate</span>
                <span class="metric-value" id="errorRate">--</span>
            </div>
            <div class="metric">
                <span>Success Rate</span>
                <span class="metric-value" id="successRate">--</span>
            </div>
        </div>

        <div class="card">
            <h3>🛡️ Immune Status</h3>
            <div class="metric">
                <span>Current Zone</span>
                <span class="metric-value"><span id="zoneIndicator" class="status-indicator"></span> <span id="currentZone">--</span></span>
            </div>
            <div class="metric">
                <span>System Mode</span>
                <span class="metric-value" id="systemMode">--</span>
            </div>
            <div class="metric">
                <span>Neural Gateway</span>
                <span class="metric-value" id="neuralGateway">--</span>
            </div>
            <div class="metric">
                <span>Swarm Health</span>
                <span class="metric-value" id="swarmHealth">--</span>
            </div>
        </div>

        <div class="card">
            <h3>📊 Performance</h3>
            <div class="metric">
                <span>Total Operations</span>
                <span class="metric-value" id="totalOps">--</span>
            </div>
            <div class="metric">
                <span>Immune Protected</span>
                <span class="metric-value" id="immuneProtected">--</span>
            </div>
            <div class="metric">
                <span>Blocked Operations</span>
                <span class="metric-value" id="blockedOps">--</span>
            </div>
            <div class="metric">
                <span>BTC Price</span>
                <span class="metric-value" id="btcPrice">--</span>
            </div>
        </div>

        <div class="card full-width">
            <h3>📈 Real-Time Metrics</h3>
            <canvas id="metricsChart"></canvas>
        </div>

        <div class="card full-width">
            <h3>🚨 Recent Alerts</h3>
            <div id="alertsContainer">
                <p>Connecting to immune system...</p>
            </div>
        </div>
    </div>

    <script>
        const ws = new WebSocket('ws://{self.host}:{self.port}');
        let autoSwitch = true;
        let chart = null;
        let chartData = [];

        ws.onopen = function(event) {{
            console.log('🔗 Connected to Immune System');
            updateStatus('Connected to Immune System', 'info');
        }};

        ws.onmessage = function(event) {{
            const message = JSON.parse(event.data);
            handleMessage(message);
        }};

        ws.onerror = function(error) {{
            console.error('🚨 WebSocket Error:', error);
            updateStatus('Connection Error', 'critical');
        }};

        ws.onclose = function(event) {{
            console.log('🔌 Disconnected from Immune System');
            updateStatus('Disconnected - Attempting Reconnect', 'warning');
            setTimeout(() => location.reload(), 5000);
        }};

        function handleMessage(message) {{
            switch(message.type) {{
                case 'initial_status':
                case 'real_time_update':
                    updateDashboard(message.data);
                    break;
                case 'alert':
                    handleAlert(message.data);
                    break;
                case 'simulation_status':
                    updateStatus(`Simulation ${{message.active ? 'Started' : 'Stopped'}}`, 'info');
                    break;
                case 'trading_decision':
                    updateTradingDecision(message.data);
                    break;
            }}
        }}

        function updateDashboard(data) {{
            if (data.status) {{
                const immune = data.status.immune_status;
                const engine = data.status.engine_status;

                // Update health metrics
                document.getElementById('mitoHealth').textContent =
                    immune.system_health.mitochondrial_health.toFixed(3);
                document.getElementById('entropy').textContent =
                    immune.system_health.system_entropy.toFixed(3);
                document.getElementById('errorRate').textContent =
                    immune.system_health.current_error_rate.toFixed(3);
                document.getElementById('successRate').textContent =
                    immune.performance_metrics.success_rate.toFixed(3);

                // Update immune status
                const zone = immune.system_health.current_zone;
                document.getElementById('currentZone').textContent = zone.toUpperCase();
                const indicator = document.getElementById('zoneIndicator');
                indicator.className = `status-indicator status-${{zone}}`;

                document.getElementById('systemMode').textContent = engine.system_mode.toUpperCase();
                document.getElementById('neuralGateway').textContent =
                    immune.immune_components.neural_gateway_state.toUpperCase();
                document.getElementById('swarmHealth').textContent =
                    immune.immune_components.swarm_health.toFixed(3);

                // Update performance
                document.getElementById('totalOps').textContent =
                    engine.performance_metrics.total_decisions;
                document.getElementById('immuneProtected').textContent =
                    engine.performance_metrics.immune_protected_decisions;
                document.getElementById('blockedOps').textContent =
                    engine.performance_metrics.biologically_blocked_decisions;
                document.getElementById('btcPrice').textContent =
                    `$$${{data.status.current_btc_price.toFixed(2)}}`;

                // Update chart
                updateChart(data.metrics_history);
            }}
        }}

        function updateChart(history) {{
            if (!chart) {{
                initChart();
            }}

            if (history && history.length > 0) {{
                chartData = history;
                chart.data.labels = history.map(h => new Date(h.timestamp * 1000).toLocaleTimeString());
                chart.data.datasets[0].data = history.map(h => h.mitochondrial_health);
                chart.data.datasets[1].data = history.map(h => h.system_entropy);
                chart.data.datasets[2].data = history.map(h => h.success_rate);
                chart.update('none');
            }}
        }}

        function initChart() {{
            const ctx = document.getElementById('metricsChart').getContext('2d');
            chart = new Chart(ctx, {{
                type: 'line',
                data: {{
                    labels: [],
                    datasets: [{{
                        label: 'Mitochondrial Health',
                        data: [],
                        borderColor: '#4CAF50',
                        tension: 0.1
                    }}, {{
                        label: 'System Entropy',
                        data: [],
                        borderColor: '#FF9800',
                        tension: 0.1
                    }}, {{
                        label: 'Success Rate',
                        data: [],
                        borderColor: '#2196F3',
                        tension: 0.1
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            max: 1
                        }}
                    }},
                    plugins: {{
                        legend: {{
                            labels: {{
                                color: 'white'
                            }}
                        }}
                    }}
                }}
            }});
        }}

        function handleAlert(alertData) {{
            const container = document.getElementById('alertsContainer');
            const alert = document.createElement('div');
            alert.className = `alert alert-${{alertData.level}}`;
            alert.innerHTML = `
                <strong>${{alertData.level.toUpperCase()}}</strong>
                [${{new Date(alertData.timestamp * 1000).toLocaleTimeString()}}]
                ${{alertData.message}}
                <br><small>Component: ${{alertData.component}} | Zone: ${{alertData.zone}}</small>
            `;

            container.insertBefore(alert, container.firstChild);

            // Auto-switch tab for critical alerts
            if (autoSwitch && alertData.auto_switch_tab) {{
                document.title = `🚨 ${{alertData.message}}`;
                if (document.hidden) {{
                    // Flash the page title
                    let flashCount = 0;
                    const flashInterval = setInterval(() => {{
                        document.title = flashCount % 2 === 0 ? '🚨 ALERT!' : `🧬 Immune System`;
                        flashCount++;
                        if (flashCount > 10) clearInterval(flashInterval);
                    }}, 500);
                }}
            }}

            // Keep only last 20 alerts
            while (container.children.length > 20) {{
                container.removeChild(container.lastChild);
            }}
        }}

        function startSimulation() {{
            ws.send(JSON.stringify({{type: 'start_simulation'}}));
        }}

        function stopSimulation() {{
            ws.send(JSON.stringify({{type: 'stop_simulation'}}));
        }}

        function resetSystem() {{
            if (confirm('Reset the entire immune system?')) {{
                ws.send(JSON.stringify({{type: 'reset_immune_system'}}));
            }}
        }}

        function triggerEmergency() {{
            if (confirm('Trigger emergency scenario for testing?')) {{
                ws.send(JSON.stringify({{type: 'trigger_emergency'}}));
            }}
        }}

        function toggleAutoSwitch() {{
            autoSwitch = !autoSwitch;
            document.getElementById('autoSwitchStatus').textContent = autoSwitch ? 'ON' : 'OFF';
        }}

        function updateStatus(message, level) {{
            console.log(`[${{level.toUpperCase()}}] ${{message}}`);
        }}

        // Initialize chart when page loads
        document.addEventListener('DOMContentLoaded', function() {{
            setTimeout(initChart, 1000);
        }});
    </script>
</body>
</html>
        """


async def main():
    """Main function to run the diagnostic server."""
    server = ImmuneDiagnosticWebSocketServer()

    # Start the server
    websocket_server = await server.start_server()

    print(
        f"🧬 Immune Diagnostic WebSocket Server running on ws://{server.host}:{server.port}"
    )
    print(f"📊 Dashboard available at: http://{server.host}:{server.port}/dashboard")
    print("Press Ctrl+C to stop the server")

    try:
        await websocket_server.wait_closed()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down server...")
        await server.stop_server()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
