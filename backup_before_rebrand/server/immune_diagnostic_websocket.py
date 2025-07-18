import numpy as np
import random
import time
import json
import asyncio
import logging

from core.biological_immune_error_handler import (  # !/usr/bin/env python3
    Any,
    Diagnostic,
    Dict,
    EnhancedMasterCycleEngine,
    EnhancedSystemMode,
    Enum,
    List,
    Provides,
    Real-time,
    Server.,
    Set,
    Streams,
    WebSocket,
    """,
    """Immune,
    alerts,
    and,
    asdict,
    asyncio,
    auto-tab,
    biological,
    changes,
    comprehensive,
    core.enhanced_master_cycle_engine,
    critical,
    dataclass,
    dataclasses,
    diagnostics,
    enum,
    events.,
    for,
    from,
    immune,
    import,
    json,
    logging,
    metrics,
    monitoring,
    operations.,
    recovery,
    server,
    switching,
    system,
    system.,
    the,
    time,
    typing,
    visualization,
    websockets,
    zone,
)

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
        self.start_time = time.time()

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

    async def handle_message(self, websocket, message) -> None:
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

    async def broadcast_message(self, message) -> None:
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

                # Update engine with simulated data
                await self.engine.process_market_data(market_data)

                # Broadcast market update
                await self.broadcast_message(
                    {
                        "type": "market_simulation",
                        "timestamp": time.time(),
                        "data": market_data,
                    }
                )

                await asyncio.sleep(1.0)  # Simulate every second

            except Exception as e:
                logger.error(f"🚨 Market simulation error: {e}")
                await asyncio.sleep(5.0)

    def generate_simulated_market_data(self) -> Dict[str, Any]:
        """Generate simulated market data for testing."""
        try:
            # Simulate price movement
            price_change = random.uniform(-0.02, 0.02)  # ±2% change
            self.price_trend += price_change
            self.price_trend = max(0.1, min(10.0, self.price_trend))  # Clamp between 0.1 and 10.0

            # Simulate volume
            volume = random.uniform(1000, 10000)

            # Simulate market volatility
            volatility = random.uniform(0.1, 0.5)

            # Simulate market sentiment
            sentiment = random.uniform(-1.0, 1.0)

            return {
                "timestamp": time.time(),
                "price": self.price_trend,
                "volume": volume,
                "volatility": volatility,
                "sentiment": sentiment,
                "bid": self.price_trend * (1 - random.uniform(0.001, 0.01)),
                "ask": self.price_trend * (1 + random.uniform(0.001, 0.01)),
                "spread": random.uniform(0.001, 0.01),
                "market_conditions": random.choice(["normal", "volatile", "trending", "sideways"]),
            }

        except Exception as e:
            logger.error(f"🚨 Error generating market data: {e}")
            return {
                "timestamp": time.time(),
                "price": 1.0,
                "volume": 1000,
                "volatility": 0.2,
                "sentiment": 0.0,
                "bid": 0.99,
                "ask": 1.01,
                "spread": 0.01,
                "market_conditions": "normal",
            }

    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        try:
            return {
                "timestamp": time.time(),
                "server_status": {
                    "running": self.running,
                    "clients_connected": len(self.clients),
                    "simulation_active": self.simulation_active,
                    "uptime": time.time() - self.start_time,
                },
                "immune_status": self.engine.get_immune_status(),
                "market_data": {
                    "current_price": self.price_trend,
                    "price_change": self.price_trend - 1.0,
                    "volatility": random.uniform(0.1, 0.5),
                },
                "system_metrics": {
                    "cpu_usage": random.uniform(10, 80),
                    "memory_usage": random.uniform(20, 90),
                    "network_latency": random.uniform(1, 100),
                    "error_count": random.randint(0, 10),
                },
            }

        except Exception as e:
            logger.error(f"🚨 Error getting status: {e}")
            return {
                "timestamp": time.time(),
                "error": str(e),
                "server_status": {"running": False},
                "immune_status": {},
                "market_data": {},
                "system_metrics": {},
            }

    def check_for_alerts(self, status: Dict[str, Any]) -> List[ImmuneAlert]:
        """Check for alerts based on current status."""
        alerts = []

        try:
            # Check mitochondrial health
            mitochondrial_health = status.get("immune_status", {}).get("system_health", {}).get("mitochondrial_health", 1.0)
            if mitochondrial_health < 0.3:
                alerts.append(
                    ImmuneAlert(
                        timestamp=time.time(),
                        level=AlertLevel.CRITICAL,
                        zone="mitochondrial",
                        message="Critical mitochondrial health degradation detected",
                        component="immune_system",
                        mitochondrial_health=mitochondrial_health,
                        system_entropy=status.get("immune_status", {}).get("system_health", {}).get("system_entropy", 0.5),
                        recommended_action="Immediate system restart recommended",
                        auto_switch_tab=True,
                    )
                )
            elif mitochondrial_health < 0.6:
                alerts.append(
                    ImmuneAlert(
                        timestamp=time.time(),
                        level=AlertLevel.WARNING,
                        zone="mitochondrial",
                        message="Mitochondrial health below optimal levels",
                        component="immune_system",
                        mitochondrial_health=mitochondrial_health,
                        system_entropy=status.get("immune_status", {}).get("system_health", {}).get("system_entropy", 0.5),
                        recommended_action="Monitor system performance closely",
                        auto_switch_tab=False,
                    )
                )

            # Check system entropy
            system_entropy = status.get("immune_status", {}).get("system_health", {}).get("system_entropy", 0.5)
            if system_entropy > 0.8:
                alerts.append(
                    ImmuneAlert(
                        timestamp=time.time(),
                        level=AlertLevel.WARNING,
                        zone="entropy",
                        message="High system entropy detected",
                        component="immune_system",
                        mitochondrial_health=mitochondrial_health,
                        system_entropy=system_entropy,
                        recommended_action="Consider system optimization",
                        auto_switch_tab=False,
                    )
                )

            # Check error rate
            error_rate = status.get("immune_status", {}).get("system_health", {}).get("current_error_rate", 0.0)
            if error_rate > 0.1:
                alerts.append(
                    ImmuneAlert(
                        timestamp=time.time(),
                        level=AlertLevel.CRITICAL,
                        zone="error_handling",
                        message=f"High error rate detected: {error_rate:.2%}",
                        component="immune_system",
                        mitochondrial_health=mitochondrial_health,
                        system_entropy=system_entropy,
                        recommended_action="Investigate error sources immediately",
                        auto_switch_tab=True,
                    )
                )

            # Check client connections
            clients_connected = status.get("server_status", {}).get("clients_connected", 0)
            if clients_connected == 0:
                alerts.append(
                    ImmuneAlert(
                        timestamp=time.time(),
                        level=AlertLevel.INFO,
                        zone="connectivity",
                        message="No clients currently connected",
                        component="websocket_server",
                        mitochondrial_health=mitochondrial_health,
                        system_entropy=system_entropy,
                        recommended_action="Check client connectivity",
                        auto_switch_tab=False,
                    )
                )

            # Check simulation status
            simulation_active = status.get("server_status", {}).get("simulation_active", False)
            if simulation_active:
                alerts.append(
                    ImmuneAlert(
                        timestamp=time.time(),
                        level=AlertLevel.INFO,
                        zone="simulation",
                        message="Market simulation is active",
                        component="simulation_engine",
                        mitochondrial_health=mitochondrial_health,
                        system_entropy=system_entropy,
                        recommended_action="Monitor simulation performance",
                        auto_switch_tab=False,
                    )
                )

        except Exception as e:
            logger.error(f"🚨 Error checking alerts: {e}")
            alerts.append(
                ImmuneAlert(
                    timestamp=time.time(),
                    level=AlertLevel.CRITICAL,
                    zone="alert_system",
                    message=f"Alert system error: {e}",
                    component="alert_system",
                    mitochondrial_health=0.0,
                    system_entropy=1.0,
                    recommended_action="Restart alert system",
                    auto_switch_tab=True,
                )
            )

        return alerts

    async def process_alert(self, alert: ImmuneAlert) -> None:
        """Process an immune system alert."""
        try:
            logger.warning(f"🚨 Alert: {alert.level.value.upper()} - {alert.message}")

            # Broadcast alert to all clients
            await self.broadcast_message(
                {
                    "type": "alert",
                    "timestamp": time.time(),
                    "data": asdict(alert),
                }
            )

            # Take automatic actions based on alert level
            if alert.level == AlertLevel.CRITICAL:
                logger.critical(f"🚨 CRITICAL ALERT: {alert.message}")
                # Could trigger emergency procedures here

            elif alert.level == AlertLevel.WARNING:
                logger.warning(f"⚠️ WARNING: {alert.message}")

            elif alert.level == AlertLevel.INFO:
                logger.info(f"ℹ️ INFO: {alert.message}")

        except Exception as e:
            logger.error(f"🚨 Error processing alert: {e}")

    async def reset_immune_system(self) -> None:
        """Reset the immune system to default state."""
        try:
            logger.info("🔄 Resetting immune system...")

            # Reset engine state
            await self.engine.reset_immune_system()

            # Clear metrics history
            self.metrics_history.clear()

            # Reset simulation state
            self.simulation_active = False

            # Broadcast reset completion
            await self.broadcast_message(
                {
                    "type": "system_reset_complete",
                    "timestamp": time.time(),
                    "message": "Immune system reset completed successfully",
                }
            )

            logger.info("✅ Immune system reset completed")

        except Exception as e:
            logger.error(f"🚨 Error resetting immune system: {e}")

    async def trigger_emergency_scenario(self) -> None:
        """Trigger an emergency scenario for testing."""
        try:
            logger.warning("🚨 Triggering emergency scenario...")

            # Simulate critical system failure
            emergency_alert = ImmuneAlert(
                timestamp=time.time(),
                level=AlertLevel.EMERGENCY,
                zone="emergency",
                message="EMERGENCY SCENARIO TRIGGERED - System under stress test",
                component="emergency_system",
                mitochondrial_health=0.1,
                system_entropy=0.9,
                recommended_action="Monitor system recovery",
                auto_switch_tab=True,
            )

            await self.process_alert(emergency_alert)

            # Simulate system stress
            for i in range(5):
                await asyncio.sleep(1.0)
                stress_alert = ImmuneAlert(
                    timestamp=time.time(),
                    level=AlertLevel.CRITICAL,
                    zone="stress_test",
                    message=f"Stress test iteration {i+1}/5",
                    component="stress_test",
                    mitochondrial_health=0.2 + i * 0.1,
                    system_entropy=0.8 - i * 0.1,
                    recommended_action="Continue monitoring",
                    auto_switch_tab=False,
                )
                await self.process_alert(stress_alert)

            logger.info("✅ Emergency scenario completed")

        except Exception as e:
            logger.error(f"🚨 Error in emergency scenario: {e}")

    def get_dashboard_html(self) -> str:
        """Get HTML dashboard for the immune diagnostic system."""
        try:
            return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Immune Diagnostic Dashboard</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        .status-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .status-card {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 20px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        .status-card h3 {
            margin-top: 0;
            color: #4CAF50;
        }
        .metric {
            display: flex;
            justify-content: space-between;
            margin: 10px 0;
            padding: 5px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        .alert {
            background: rgba(255, 193, 7, 0.2);
            border-left: 4px solid #FFC107;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
        }
        .critical {
            background: rgba(244, 67, 54, 0.2);
            border-left: 4px solid #F44336;
        }
        .controls {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }
        .btn {
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        .btn-primary {
            background: #4CAF50;
            color: white;
        }
        .btn-warning {
            background: #FF9800;
            color: white;
        }
        .btn-danger {
            background: #F44336;
            color: white;
        }
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        .websocket-status {
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 10px;
            border-radius: 5px;
            font-weight: bold;
        }
        .connected {
            background: rgba(76, 175, 80, 0.8);
        }
        .disconnected {
            background: rgba(244, 67, 54, 0.8);
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧬 Immune Diagnostic Dashboard</h1>
            <p>Real-time monitoring of the Schwabot immune system</p>
        </div>

        <div class="websocket-status" id="wsStatus">
            Connecting...
        </div>

        <div class="controls">
            <button class="btn btn-primary" onclick="startSimulation()">Start Simulation</button>
            <button class="btn btn-warning" onclick="stopSimulation()">Stop Simulation</button>
            <button class="btn btn-danger" onclick="resetSystem()">Reset System</button>
            <button class="btn btn-warning" onclick="triggerEmergency()">Emergency Test</button>
        </div>

        <div class="status-grid">
            <div class="status-card">
                <h3>🔄 System Health</h3>
                <div id="systemHealth">
                    <div class="metric">
                        <span>Mitochondrial Health:</span>
                        <span id="mitochondrialHealth">--</span>
                    </div>
                    <div class="metric">
                        <span>System Entropy:</span>
                        <span id="systemEntropy">--</span>
                    </div>
                    <div class="metric">
                        <span>Error Rate:</span>
                        <span id="errorRate">--</span>
                    </div>
                    <div class="metric">
                        <span>Current Zone:</span>
                        <span id="currentZone">--</span>
                    </div>
                </div>
            </div>

            <div class="status-card">
                <h3>📊 Performance Metrics</h3>
                <div id="performanceMetrics">
                    <div class="metric">
                        <span>Success Rate:</span>
                        <span id="successRate">--</span>
                    </div>
                    <div class="metric">
                        <span>Response Time:</span>
                        <span id="responseTime">--</span>
                    </div>
                    <div class="metric">
                        <span>Throughput:</span>
                        <span id="throughput">--</span>
                    </div>
                    <div class="metric">
                        <span>Active Processes:</span>
                        <span id="activeProcesses">--</span>
                    </div>
                </div>
            </div>

            <div class="status-card">
                <h3>🌐 Server Status</h3>
                <div id="serverStatus">
                    <div class="metric">
                        <span>Status:</span>
                        <span id="serverRunning">--</span>
                    </div>
                    <div class="metric">
                        <span>Connected Clients:</span>
                        <span id="connectedClients">--</span>
                    </div>
                    <div class="metric">
                        <span>Simulation Active:</span>
                        <span id="simulationActive">--</span>
                    </div>
                    <div class="metric">
                        <span>Uptime:</span>
                        <span id="uptime">--</span>
                    </div>
                </div>
            </div>

            <div class="status-card">
                <h3>📈 Market Data</h3>
                <div id="marketData">
                    <div class="metric">
                        <span>Current Price:</span>
                        <span id="currentPrice">--</span>
                    </div>
                    <div class="metric">
                        <span>Price Change:</span>
                        <span id="priceChange">--</span>
                    </div>
                    <div class="metric">
                        <span>Volume:</span>
                        <span id="volume">--</span>
                    </div>
                    <div class="metric">
                        <span>Volatility:</span>
                        <span id="volatility">--</span>
                    </div>
                </div>
            </div>
        </div>

        <div class="status-card">
            <h3>🚨 Active Alerts</h3>
            <div id="alerts">
                <p>No active alerts</p>
            </div>
        </div>
    </div>

    <script>
        let ws = null;
        let reconnectAttempts = 0;
        const maxReconnectAttempts = 5;

        function connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.hostname}:8767`;
            
            ws = new WebSocket(wsUrl);
            
            ws.onopen = function() {
                console.log('WebSocket connected');
                document.getElementById('wsStatus').textContent = 'Connected';
                document.getElementById('wsStatus').className = 'websocket-status connected';
                reconnectAttempts = 0;
            };
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                updateDashboard(data);
            };
            
            ws.onclose = function() {
                console.log('WebSocket disconnected');
                document.getElementById('wsStatus').textContent = 'Disconnected';
                document.getElementById('wsStatus').className = 'websocket-status disconnected';
                
                if (reconnectAttempts < maxReconnectAttempts) {
                    reconnectAttempts++;
                    setTimeout(connectWebSocket, 2000);
                }
            };
            
            ws.onerror = function(error) {
                console.error('WebSocket error:', error);
            };
        }

        function updateDashboard(data) {
            if (data.type === 'real_time_update') {
                const status = data.data.status;
                
                // Update system health
                if (status.immune_status?.system_health) {
                    const health = status.immune_status.system_health;
                    document.getElementById('mitochondrialHealth').textContent = 
                        (health.mitochondrial_health * 100).toFixed(1) + '%';
                    document.getElementById('systemEntropy').textContent = 
                        (health.system_entropy * 100).toFixed(1) + '%';
                    document.getElementById('errorRate').textContent = 
                        (health.current_error_rate * 100).toFixed(2) + '%';
                    document.getElementById('currentZone').textContent = health.current_zone || '--';
                }
                
                // Update performance metrics
                if (status.immune_status?.performance_metrics) {
                    const metrics = status.immune_status.performance_metrics;
                    document.getElementById('successRate').textContent = 
                        (metrics.success_rate * 100).toFixed(1) + '%';
                    document.getElementById('responseTime').textContent = 
                        (metrics.avg_response_time || 0).toFixed(2) + 'ms';
                    document.getElementById('throughput').textContent = 
                        (metrics.throughput || 0).toFixed(0) + '/s';
                    document.getElementById('activeProcesses').textContent = 
                        metrics.active_processes || 0;
                }
                
                // Update server status
                if (status.server_status) {
                    const server = status.server_status;
                    document.getElementById('serverRunning').textContent = 
                        server.running ? 'Running' : 'Stopped';
                    document.getElementById('connectedClients').textContent = 
                        server.clients_connected || 0;
                    document.getElementById('simulationActive').textContent = 
                        server.simulation_active ? 'Yes' : 'No';
                    document.getElementById('uptime').textContent = 
                        formatUptime(server.uptime || 0);
                }
                
                // Update market data
                if (status.market_data) {
                    const market = status.market_data;
                    document.getElementById('currentPrice').textContent = 
                        market.current_price?.toFixed(4) || '--';
                    document.getElementById('priceChange').textContent = 
                        (market.price_change || 0).toFixed(4);
                    document.getElementById('volume').textContent = 
                        (market.volume || 0).toFixed(0);
                    document.getElementById('volatility').textContent = 
                        (market.volatility || 0).toFixed(3);
                }
                
                // Update alerts
                if (data.data.alerts && data.data.alerts.length > 0) {
                    const alertsDiv = document.getElementById('alerts');
                    alertsDiv.innerHTML = '';
                    
                    data.data.alerts.forEach(alert => {
                        const alertDiv = document.createElement('div');
                        alertDiv.className = `alert ${alert.level === 'critical' ? 'critical' : ''}`;
                        alertDiv.innerHTML = `
                            <strong>${alert.level.toUpperCase()}</strong> - ${alert.message}<br>
                            <small>Zone: ${alert.zone} | Component: ${alert.component}</small>
                        `;
                        alertsDiv.appendChild(alertDiv);
                    });
                }
            }
        }

        function formatUptime(seconds) {
            const hours = Math.floor(seconds / 3600);
            const minutes = Math.floor((seconds % 3600) / 60);
            const secs = Math.floor(seconds % 60);
            return `${hours}h ${minutes}m ${secs}s`;
        }

        function startSimulation() {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({type: 'start_simulation'}));
            }
        }

        function stopSimulation() {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({type: 'stop_simulation'}));
            }
        }

        function resetSystem() {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({type: 'reset_immune_system'}));
            }
        }

        function triggerEmergency() {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({type: 'trigger_emergency'}));
            }
        }

        // Connect on page load
        connectWebSocket();
    </script>
</body>
</html>
            """

        except Exception as e:
            logger.error(f"🚨 Error generating dashboard HTML: {e}")
            return f"<html><body><h1>Error</h1><p>{e}</p></body></html>"


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
