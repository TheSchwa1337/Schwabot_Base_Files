from core.unified_math_system import UnifiedMathSystem

from { box-shadow: 0 0 5px rgba(255,255,255,0.5); }
import asyncio
import logging
import os
import sys
import threading
import time

from flask_socketio import SocketIO, emit

from core.dual_brain_architecture import DualBrainArchitecture, dual_brain
from core.whale_tracker_integration import whale_tracker
from flask import Flask, jsonify, render_template

# -*- coding: utf-8 -*-
"""
Dual Brain Server
=================

Flask server providing dual-panel web interface for monitoring
mining operations (left brain) and trading operations (right brain)
with real-time updates and 32-bit thermal state visualization.

Features:
    - Left Panel: Mining/Hashing operations, difficulty analysis, thermal states
    - Right Panel: Trading decisions, whale tracking, market analysis
    - Real-time WebSocket updates
    - 32-bit thermal state visualization
    - Flip logic monitoring
    - Performance metrics dashboard
"""


# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import core systems
try:

    CORE_SYSTEMS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Core systems not available: {e}")
    CORE_SYSTEMS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config["SECRET_KEY"] = "dual_brain_trading_system_2024"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# Global state
dual_brain_instance = None
whale_tracker_instance = None
server_running = False
update_thread = None

# Thermal state constants
COOL = "cool"
WARM = "warm"
HOT = "hot"
CRITICAL = "critical"


class DualBrainServer:
    """Main server class managing dual brain operations and UI updates."""

    def __init__(self):
        """Initialize the dual brain server."""
        self.dual_brain = DualBrainArchitecture() if CORE_SYSTEMS_AVAILABLE else None
        self.whale_tracker = whale_tracker if CORE_SYSTEMS_AVAILABLE else None
        self.math_system = UnifiedMathSystem() if CORE_SYSTEMS_AVAILABLE else None

        # Server state
        self.running = False
        self.last_decision = None
        self.system_metrics = {
            "uptime": 0,
            "total_decisions": 0,
            "successful_trades": 0,
            "total_profit": 0.0,
            "thermal_state_distribution": {COOL: 0, WARM: 0, HOT: 0, CRITICAL: 0},
        }
        logger.info("🚀 Dual Brain Server initialized")

    async def start_dual_brain_monitoring(self):
        """Start continuous dual brain monitoring."""
        self.running = True
        start_time = time.time()

        try:
            logger.info("🧠🧠 Starting dual brain monitoring...")

            while self.running:
                # Run dual brain cycle
                if self.dual_brain:
                    decision = await self.dual_brain.run_dual_brain_cycle()
                    self.last_decision = decision

                    # Update metrics
                    self.system_metrics["total_decisions"] += 1
                    self.system_metrics["uptime"] = time.time() - start_time

                    # Track thermal states
                    left_thermal = decision.left_brain_state.thermal_state
                    right_thermal = decision.right_brain_state.thermal_state

                    self.system_metrics["thermal_state_distribution"][left_thermal] += 1
                    self.system_metrics["thermal_state_distribution"][
                        right_thermal
                    ] += 1

                    # Simulate profit tracking
                    if decision.expected_profit > 0:
                        self.system_metrics["successful_trades"] += 1
                        self.system_metrics["total_profit"] += (
                            decision.expected_profit * 0.1
                        )  # 10% realization

                    # Emit updates to connected clients
                    await self._emit_updates(decision)

                # Wait before next cycle
                await asyncio.sleep(5)  # 5-second cycles

        except Exception as e:
            logger.error(f"Dual brain monitoring error: {e}")
        finally:
            self.running = False

    async def _emit_updates(self, decision):
        """Emit real-time updates to connected WebSocket clients."""
        try:
            # Left brain (mining) data
            left_brain_data = {
                "thermal_state": decision.left_brain_state.thermal_state,
                "last_decision": decision.left_brain_state.last_decision,
                "confidence": decision.left_brain_state.decision_confidence,
                "processing_load": decision.left_brain_state.processing_load,
                "performance_metrics": decision.left_brain_state.performance_metrics,
                "active_operations": decision.left_brain_state.active_operations,
                "timestamp": decision.left_brain_state.timestamp.isoformat(),
            }
            # Right brain (trading) data
            right_brain_data = {
                "thermal_state": decision.right_brain_state.thermal_state,
                "last_decision": decision.right_brain_state.last_decision,
                "confidence": decision.right_brain_state.decision_confidence,
                "processing_load": decision.right_brain_state.processing_load,
                "performance_metrics": decision.right_brain_state.performance_metrics,
                "active_operations": decision.right_brain_state.active_operations,
                "timestamp": decision.right_brain_state.timestamp.isoformat(),
            }
            # Flip logic data
            flip_logic_data = {
                "flip_signal": decision.flip_logic_result.flip_signal.value,
                "confidence": decision.flip_logic_result.confidence,
                "mining_contribution": decision.flip_logic_result.mining_contribution,
                "trading_contribution": decision.flip_logic_result.trading_contribution,
                "thermal_multiplier": decision.flip_logic_result.thermal_multiplier,
                "reasoning": decision.flip_logic_result.reasoning,
                "profit_potential": decision.flip_logic_result.profit_potential,
                "risk_assessment": decision.flip_logic_result.risk_assessment,
            }
            # Unified decision data
            unified_decision_data = {
                "synchronized_action": decision.synchronized_action,
                "overall_confidence": decision.overall_confidence,
                "expected_profit": decision.expected_profit,
                "thermal_enhancement": decision.thermal_enhancement,
                "execution_priority": decision.execution_priority,
            }
            # Whale tracking data
            whale_data = {}
            if self.whale_tracker:
                whale_summary = self.whale_tracker.get_whale_summary()
                whale_data = {
                    "statistics": whale_summary.get("statistics", {}),
                    "recent_alerts": whale_summary.get("recent_alerts", [])[
                        :5
                    ],  # Last 5 alerts
                    "thermal_state": whale_summary.get("thermal_state", WARM),
                }
            # Emit all updates
            socketio.emit("left_brain_update", left_brain_data)
            socketio.emit("right_brain_update", right_brain_data)
            socketio.emit("flip_logic_update", flip_logic_data)
            socketio.emit("unified_decision_update", unified_decision_data)
            socketio.emit("whale_update", whale_data)
            socketio.emit("system_metrics_update", self.system_metrics)

        except Exception as e:
            logger.error(f"WebSocket emit error: {e}")

    def stop(self):
        """Stop the dual brain monitoring."""
        self.running = False
        logger.info("🛑 Dual brain monitoring stopped")


# Global server instance
server_instance = DualBrainServer()


# Flask Routes
@app.route("/")
def dashboard():
    """Main dashboard with dual brain panels."""
    return render_template("dual_brain_dashboard.html")


@app.route("/api/status")
def get_status():
    """Get server status."""
    return jsonify(
        {
            "status": "running" if server_instance.running else "stopped",
            "system_metrics": server_instance.system_metrics,
            "last_decision": (
                {
                    "action": (
                        server_instance.last_decision.synchronized_action
                        if server_instance.last_decision
                        else None
                    ),
                    "confidence": (
                        server_instance.last_decision.overall_confidence
                        if server_instance.last_decision
                        else 0.0
                    ),
                    "timestamp": (
                        server_instance.last_decision.timestamp.isoformat()
                        if server_instance.last_decision
                        else None
                    ),
                }
                if server_instance.last_decision
                else None
            ),
        }
    )


@app.route("/api/architecture_summary")
def get_architecture_summary():
    """Get dual brain architecture summary."""
    if server_instance.dual_brain:
        return jsonify(server_instance.dual_brain.get_architecture_summary())
    else:
        return jsonify({"error": "Dual brain not available"})


@app.route("/api/whale_summary")
def get_whale_summary():
    """Get whale tracking summary."""
    if server_instance.whale_tracker:
        return jsonify(server_instance.whale_tracker.get_whale_summary())
    else:
        return jsonify({"error": "Whale tracker not available"})


@app.route("/api/start", methods=["POST"])
def start_monitoring():
    """Start dual brain monitoring."""
    if not server_instance.running:
        # Start monitoring in background thread
        def run_monitoring():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(server_instance.start_dual_brain_monitoring())

        monitoring_thread = threading.Thread(target=run_monitoring, daemon=True)
        monitoring_thread.start()

        return jsonify({"status": "started"})
    else:
        return jsonify({"status": "already_running"})


@app.route("/api/stop", methods=["POST"])
def stop_monitoring():
    """Stop dual brain monitoring."""
    server_instance.stop()
    return jsonify({"status": "stopped"})


# WebSocket Events
@socketio.on("connect")
def handle_connect():
    """Handle client connection."""
    logger.info("Client connected to WebSocket")
    emit("connection_established", {"data": "Connected to Dual Brain Server"})


@socketio.on("disconnect")
def handle_disconnect():
    """Handle client disconnection."""
    logger.info("Client disconnected from WebSocket")


@socketio.on("request_update")
def handle_update_request():
    """Handle manual update request from client."""
    if server_instance.last_decision:
        # Send latest data
        asyncio.create_task(
            server_instance._emit_updates(server_instance.last_decision)
        )


# Create templates directory if it doesn't exist
def create_templates():
    """Create HTML templates for the dual brain interface."""
    templates_dir = os.path.join(os.path.dirname(__file__), "templates")
    static_dir = os.path.join(os.path.dirname(__file__), "static")

    # Create directories
    os.makedirs(templates_dir, exist_ok=True)
    os.makedirs(static_dir, exist_ok=True)

    # Create main dashboard template
    dashboard_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dual Brain Trading System</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
}
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #0c0c0c 0%, #1a1a1a 100%);
            color: #ffffff;
            height: 100vh;
            overflow-x: hidden;
}
        .header {
            background: linear-gradient(90deg, #2c3e50 0%, #34495e 100%);
            padding: 15px 30px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
            position: relative;
}
        .header h1 {
            font-size: 2.2em;
            text-align: center;
            background: linear-gradient(45deg, #3498db, #e74c3c);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
}
        .status-bar {
            position: absolute;
            top: 20px;
            right: 30px;
            display: flex;
            gap: 15px;
            align-items: center;
}
        .status-indicator {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #e74c3c;
            animation: pulse 2s infinite;
}
        .status-indicator.active {
            background: #27ae60;
}
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
}
        .control-panel {
            padding: 10px 30px;
            background: rgba(52, 73, 94, 0.3);
            display: flex;
            justify-content: center;
            gap: 20px;
}
        .btn {
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.3s ease;
}
        .btn-start {
            background: linear-gradient(45deg, #27ae60, #2ecc71);
            color: white;
}
        .btn-stop {
            background: linear-gradient(45deg, #e74c3c, #c0392b);
            color: white;
}
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.3);
}
        .main-container {
            display: flex;
            height: calc(100vh - 160px);
            gap: 2px;
}
        .brain-panel {
            flex: 1;
            padding: 20px;
            overflow-y: auto;
            position: relative;
}
        .left-brain {
            background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
            border-right: 2px solid #34495e;
}
        .right-brain {
            background: linear-gradient(135deg, #e74c3c 0%, #f39c12 100%);
}
        .brain-header {
            text-align: center;
            margin-bottom: 20px;
            padding: 15px;
            background: rgba(0,0,0,0.3);
            border-radius: 10px;
}
        .brain-header h2 {
            font-size: 1.8em;
            margin-bottom: 5px;
}
        .thermal-indicator {
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
            margin-top: 5px;
}
        .thermal-cool { background: #3498db; }
        .thermal-warm { background: #f39c12; }
        .thermal-hot { background: #e74c3c; animation: glow 1s infinite alternate; }
        .thermal-critical { background: #8e44ad; animation: glow 0.5s infinite alternate; }
        
        @keyframes glow {
            to { box-shadow: 0 0 20px rgba(255,255,255,0.8); }
}
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
}
        .metric-card {
            background: rgba(0,0,0,0.4);
            padding: 15px;
            border-radius: 8px;
            border: 1px solid rgba(255,255,255,0.1);
}
        .metric-card h3 {
            color: #ecf0f1;
            margin-bottom: 10px;
            font-size: 14px;
            text-transform: uppercase;
}
        .metric-value {
            font-size: 1.5em;
            font-weight: bold;
            color: #3498db;
}
        .flip-logic-panel {
            position: fixed;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            width: 400px;
            background: linear-gradient(135deg, #8e44ad 0%, #9b59b6 100%);
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            z-index: 1000;
}
        .flip-logic-header {
            text-align: center;
            margin-bottom: 10px;
}
        .flip-signal {
            text-align: center;
            font-size: 1.2em;
            font-weight: bold;
            padding: 8px;
            border-radius: 5px;
            margin-bottom: 10px;
}
        .signal-strong-buy { background: #27ae60; }
        .signal-moderate-buy { background: #2ecc71; }
        .signal-hold { background: #f39c12; }
        .signal-moderate-sell { background: #e67e22; }
        .signal-strong-sell { background: #e74c3c; }
        
        .operations-list {
            list-style: none;
            padding: 0;
}
        .operations-list li {
            background: rgba(0,0,0,0.3);
            margin: 5px 0;
            padding: 8px 12px;
            border-radius: 5px;
            font-size: 12px;
            border-left: 3px solid #3498db;
}
        .whale-alerts {
            background: rgba(0,0,0,0.4);
            padding: 15px;
            border-radius: 8px;
            margin-top: 15px;
}
        .whale-alert {
            padding: 8px;
            margin: 5px 0;
            border-radius: 5px;
            font-size: 12px;
}
        .alert-high { background: #e74c3c; }
        .alert-medium { background: #f39c12; }
        .alert-low { background: #27ae60; }
        
        .chart-container {
            background: rgba(0,0,0,0.4);
            padding: 15px;
            border-radius: 8px;
            margin-top: 15px;
            height: 200px;
}
        @media (max-width: 768px) {
            .main-container {
                flex-direction: column;
}
            .flip-logic-panel {
                position: relative;
                width: 90%;
                bottom: auto;
                left: auto;
                transform: none;
                margin: 20px auto;
}
}
    </style>
</head>
<body>
    <div class="header">
        <h1>🧠🧠 Dual Brain Trading System</h1>
        <div class="status-bar">
            <div class="status-indicator" id="status-indicator"></div>
            <span id="status-text">Disconnected</span>
        </div>
    </div>
    
    <div class="control-panel">
        <button class="btn btn-start" onclick="startMonitoring()">🚀 Start System</button>
        <button class="btn btn-stop" onclick="stopMonitoring()">🛑 Stop System</button>
    </div>
    
    <div class="main-container">
        <!-- Left Brain Panel (Mining/Hashing) -->
        <div class="brain-panel left-brain">
            <div class="brain-header">
                <h2>🧠 Left Brain - Mining/Hashing</h2>
                <div class="thermal-indicator thermal-warm" id="left-thermal">WARM</div>
            </div>
            
            <div class="metrics-grid">
                <div class="metric-card">
                    <h3>Hash Rate</h3>
                    <div class="metric-value" id="hash-rate">0.0 TH/s</div>
                </div>
                <div class="metric-card">
                    <h3>Mining Efficiency</h3>
                    <div class="metric-value" id="mining-efficiency">0.0%</div>
                </div>
                <div class="metric-card">
                    <h3>Block Time Avg</h3>
                    <div class="metric-value" id="block-time">10.0 min</div>
                </div>
                <div class="metric-card">
                    <h3>Difficulty Adjustment</h3>
                    <div class="metric-value" id="difficulty-adj">1.00x</div>
                </div>
            </div>
            
            <div class="metric-card">
                <h3>Active Mining Operations</h3>
                <ul class="operations-list" id="mining-operations">
                    <li>Initializing mining operations...</li>
                </ul>
            </div>
            
            <div class="metric-card">
                <h3>Mining Decision</h3>
                <div class="metric-value" id="mining-decision">Neutral</div>
                <div style="margin-top: 10px;">
                    Confidence: <span id="mining-confidence">50%</span>
                </div>
            </div>
            
            <div class="chart-container">
                <canvas id="miningChart"></canvas>
            </div>
        </div>
        
        <!-- Right Brain Panel (Trading/Decisions) -->
        <div class="brain-panel right-brain">
            <div class="brain-header">
                <h2>🧠 Right Brain - Trading/Decisions</h2>
                <div class="thermal-indicator thermal-warm" id="right-thermal">WARM</div>
            </div>
            
            <div class="metrics-grid">
                <div class="metric-card">
                    <h3>BTC Price</h3>
                    <div class="metric-value" id="btc-price">$65,000</div>
                </div>
                <div class="metric-card">
                    <h3>24h Volume</h3>
                    <div class="metric-value" id="volume-24h">$30B</div>
                </div>
                <div class="metric-card">
                    <h3>Volatility</h3>
                    <div class="metric-value" id="volatility">2.1%</div>
                </div>
                <div class="metric-card">
                    <h3>Profit Factor</h3>
                    <div class="metric-value" id="profit-factor">1.25</div>
                </div>
            </div>
            
            <div class="metric-card">
                <h3>Active Trading Operations</h3>
                <ul class="operations-list" id="trading-operations">
                    <li>Initializing trading operations...</li>
                </ul>
            </div>
            
            <div class="metric-card">
                <h3>Trading Decision</h3>
                <div class="metric-value" id="trading-decision">Neutral</div>
                <div style="margin-top: 10px;">
                    Confidence: <span id="trading-confidence">50%</span>
                </div>
            </div>
            
            <div class="whale-alerts">
                <h3>🐋 Whale Alerts</h3>
                <div id="whale-alerts-list">
                    <div class="whale-alert alert-low">No recent whale activity</div>
                </div>
            </div>
            
            <div class="chart-container">
                <canvas id="tradingChart"></canvas>
            </div>
        </div>
    </div>
    
    <!-- Flip Logic Panel -->
    <div class="flip-logic-panel">
        <div class="flip-logic-header">
            <h3>⚡ Flip Logic Engine</h3>
        </div>
        <div class="flip-signal signal-hold" id="flip-signal">HOLD</div>
        <div style="text-align: center; margin-bottom: 10px;">
            <small id="flip-reasoning">Waiting for signals...</small>
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 12px;">
            <span>Confidence: <span id="flip-confidence">50%</span></span>
            <span>Thermal: <span id="flip-thermal">1.0x</span></span>
            <span>Profit: <span id="flip-profit">$0</span></span>
        </div>
    </div>

    <script>
        // Initialize WebSocket connection
        const socket = io();
        
        // Charts
        let miningChart, tradingChart;
        
        // Initialize charts
        function initCharts() {
            const miningCtx = document.getElementById('miningChart').getContext('2d');
            const tradingCtx = document.getElementById('tradingChart').getContext('2d');
            
            miningChart = new Chart(miningCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Hash Rate (TH/s)',
                        data: [],
                        borderColor: '#3498db',
                        backgroundColor: 'rgba(52, 152, 219, 0.1)',
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { labels: { color: '#ffffff' } } },
                    scales: {
                        x: { ticks: { color: '#ffffff' }, grid: { color: 'rgba(255,255,255,0.1)' } },
                        y: { ticks: { color: '#ffffff' }, grid: { color: 'rgba(255,255,255,0.1)' } }
}
}
            });
            
            tradingChart = new Chart(tradingCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Profit Potential',
                        data: [],
                        borderColor: '#e74c3c',
                        backgroundColor: 'rgba(231, 76, 60, 0.1)',
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { labels: { color: '#ffffff' } } },
                    scales: {
                        x: { ticks: { color: '#ffffff' }, grid: { color: 'rgba(255,255,255,0.1)' } },
                        y: { ticks: { color: '#ffffff' }, grid: { color: 'rgba(255,255,255,0.1)' } }
}
}
            });
}
        // WebSocket event handlers
        socket.on('connect', function() {
            document.getElementById('status-indicator').classList.add('active');
            document.getElementById('status-text').textContent = 'Connected';
        });
        
        socket.on('disconnect', function() {
            document.getElementById('status-indicator').classList.remove('active');
            document.getElementById('status-text').textContent = 'Disconnected';
        });
        
        socket.on('left_brain_update', function(data) {
            updateLeftBrain(data);
        });
        
        socket.on('right_brain_update', function(data) {
            updateRightBrain(data);
        });
        
        socket.on('flip_logic_update', function(data) {
            updateFlipLogic(data);
        });
        
        socket.on('whale_update', function(data) {
            updateWhaleAlerts(data);
        });
        
        // Update functions
        function updateLeftBrain(data) {
            // Update thermal indicator
            const thermalEl = document.getElementById('left-thermal');
            thermalEl.textContent = data.thermal_state.toUpperCase();
            thermalEl.className = `thermal-indicator thermal-${data.thermal_state}`;
            
            // Update metrics
            const metrics = data.performance_metrics || {};
            document.getElementById('hash-rate').textContent = `${(metrics.hash_rate_th_s || 0).toFixed(1)} TH/s`;
            document.getElementById('mining-efficiency').textContent = `${((metrics.mining_efficiency || 0) * 100).toFixed(1)}%`;
            document.getElementById('block-time').textContent = `${((metrics.block_time_average || 600) / 60).toFixed(1)} min`;
            document.getElementById('difficulty-adj').textContent = `${(metrics.difficulty_adjustment || 1).toFixed(2)}x`;
            
            // Update operations
            const operationsList = document.getElementById('mining-operations');
            operationsList.innerHTML = '';
            (data.active_operations || []).forEach(op => {
                const li = document.createElement('li');
                li.textContent = op.replace(/_/g, ' ').toUpperCase();
                operationsList.appendChild(li);
            });
            
            // Update decision
            document.getElementById('mining-decision').textContent = (data.last_decision || 'neutral').replace('mining_', '').toUpperCase();
            document.getElementById('mining-confidence').textContent = `${((data.confidence || 0) * 100).toFixed(0)}%`;
            
            // Update chart
            updateChart(miningChart, metrics.hash_rate_th_s || 0);
}
        function updateRightBrain(data) {
            // Update thermal indicator
            const thermalEl = document.getElementById('right-thermal');
            thermalEl.textContent = data.thermal_state.toUpperCase();
            thermalEl.className = `thermal-indicator thermal-${data.thermal_state}`;
            
            // Update metrics (simulated values for demo)
            document.getElementById('btc-price').textContent = `$${(65000 + Math.random() * 10000 - 5000).toFixed(0)}`;
            document.getElementById('volume-24h').textContent = `$${(Math.random() * 20 + 20).toFixed(0)}B`;
            document.getElementById('volatility').textContent = `${(Math.random() * 5 + 1).toFixed(1)}%`;
            
            const metrics = data.performance_metrics || {};
            document.getElementById('profit-factor').textContent = (metrics.profit_factor || 1.0).toFixed(2);
            
            // Update operations
            const operationsList = document.getElementById('trading-operations');
            operationsList.innerHTML = '';
            (data.active_operations || []).forEach(op => {
                const li = document.createElement('li');
                li.textContent = op.replace(/_/g, ' ').toUpperCase();
                operationsList.appendChild(li);
            });
            
            // Update decision
            document.getElementById('trading-decision').textContent = (data.last_decision || 'neutral').replace('trading_', '').toUpperCase();
            document.getElementById('trading-confidence').textContent = `${((data.confidence || 0) * 100).toFixed(0)}%`;
            
            // Update chart
            updateChart(tradingChart, metrics.profit_factor || 0);
}
        function updateFlipLogic(data) {
            const signalEl = document.getElementById('flip-signal');
            signalEl.textContent = data.flip_signal.replace(/_/g, ' ').toUpperCase();
            signalEl.className = `flip-signal signal-${data.flip_signal.replace(/_/g, '-')}`;
            
            document.getElementById('flip-reasoning').textContent = data.reasoning || 'Processing...';
            document.getElementById('flip-confidence').textContent = `${((data.confidence || 0) * 100).toFixed(0)}%`;
            document.getElementById('flip-thermal').textContent = `${(data.thermal_multiplier || 1).toFixed(1)}x`;
            document.getElementById('flip-profit').textContent = `$${(data.profit_potential * 1000 || 0).toFixed(0)}`;
}
        function updateWhaleAlerts(data) {
            const alertsList = document.getElementById('whale-alerts-list');
            alertsList.innerHTML = '';
            
            if (data.recent_alerts && data.recent_alerts.length > 0) {
                data.recent_alerts.forEach(alert => {
                    const div = document.createElement('div');
                    div.className = `whale-alert alert-${alert.alert_level}`;
                    div.innerHTML = `
                        <strong>${alert.alert_level.toUpperCase()}</strong> - 
                        ${alert.volume_btc.toFixed(0)} BTC - 
                        ${alert.flow_direction} - 
                        ${alert.thermal_recommendation.replace(/_/g, ' ')}
                    `;
                    alertsList.appendChild(div);
                });
            } else {
                const div = document.createElement('div');
                div.className = 'whale-alert alert-low';
                div.textContent = 'No recent whale activity';
                alertsList.appendChild(div);
}
}
        function updateChart(chart, value) {
            const now = new Date().toLocaleTimeString();
            
            if (chart.data.labels.length > 20) {
                chart.data.labels.shift();
                chart.data.datasets[0].data.shift();
}
            chart.data.labels.push(now);
            chart.data.datasets[0].data.push(value);
            chart.update('none');
}
        // Control functions
        function startMonitoring() {
            fetch('/api/start', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    console.log('Start response:', data);
                });
}
        function stopMonitoring() {
            fetch('/api/stop', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    console.log('Stop response:', data);
                });
}
        // Initialize when page loads
        document.addEventListener('DOMContentLoaded', function() {
            initCharts();
        });
    </script>
</body>
</html>"""

    # Write dashboard template
    with open(os.path.join(templates_dir, "dual_brain_dashboard.html"), "w") as f:
        f.write(dashboard_html)

    logger.info("Templates created successfully")


def run_server():
    """Run the Flask server."""
    global server_running, update_thread

    # Create templates
    create_templates()

    logger.info("🚀 Starting Dual Brain Server...")
    logger.info("📊 Dashboard available at: http://localhost:5000")

    # Start server
    server_running = True
    try:
        socketio.run(
            app, host="0.0.0.0", port=5000, debug=False, allow_unsafe_werkzeug=True
        )
    except KeyboardInterrupt:
        logger.info("🛑 Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
    finally:
        server_running = False
        if server_instance:
            server_instance.stop()


if __name__ == "__main__":
    run_server()
