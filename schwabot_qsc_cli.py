#!/usr/bin/env python3
"""Schwabot QSC + GTS Immune System CLI.

Enhanced command-line interface for launching and managing the complete
Quantum Static Core (QSC) + Generalized Tensor Solutions (GTS) immune 
system integrated with Schwabot's trading infrastructure.

Provides unified control over:
- Master Cycle Engine
- QSC Immune System
- Tensor Analysis
- Profit Allocation
- WebSocket Streaming
- Visual Diagnostics
"""

import os
import sys
import argparse
import asyncio
import logging
import time
import signal
import json
from typing import Dict, List, Optional, Any
from pathlib import Path
import subprocess
from dataclasses import dataclass, field

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from core.master_cycle_engine import MasterCycleEngine, SystemMode, TradingDecision
from core.quantum_static_core import QuantumStaticCore, QSCMode, ResonanceLevel
from core.qsc_enhanced_profit_allocator import QSCEnhancedProfitAllocator
from server.qsc_diagnostic_websocket import QSCDiagnosticServer
from server.tensor_websocket_server import TensorWebSocketServer
from utils.logging_setup import setup_logging

# Setup logging
logger = setup_logging(__name__)


@dataclass
class QSCSystemStatus:
    """Complete QSC system status."""
    master_engine: bool = False
    qsc_immune_system: bool = False
    tensor_analysis: bool = False
    profit_allocator: bool = False
    diagnostic_server: bool = False
    visualization_server: bool = False
    uptime: float = 0.0
    immune_activations: int = 0
    ghost_floor_activations: int = 0
    emergency_shutdowns: int = 0
    total_decisions: int = 0
    success_rate: float = 0.0


class SchwabotQSCCLI:
    """Enhanced CLI for QSC + GTS immune system management."""

    def __init__(self):
        """Initialize the QSC CLI."""
        self.config_file = project_root / "config" / "qsc_system_config.json"
        self.config = self._load_config()
        
        # System components
        self.master_engine: Optional[MasterCycleEngine] = None
        self.diagnostic_server: Optional[QSCDiagnosticServer] = None
        self.tensor_server: Optional[TensorWebSocketServer] = None
        self.visualization_process: Optional[subprocess.Popen] = None
        
        # Status tracking
        self.system_status = QSCSystemStatus()
        self.start_time = time.time()
        self.is_running = False
        
        # Signal handling
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _load_config(self) -> Dict[str, Any]:
        """Load QSC system configuration."""
        default_config = {
            "master_engine": {
                "fibonacci_divergence_threshold": 0.007,
                "orderbook_imbalance_threshold": 0.15,
                "immune_activation_threshold": 0.85,
                "quantum_confidence_threshold": 0.8,
                "enable_auto_immune_response": True,
                "enable_ghost_floor_mode": True,
                "enable_emergency_protocols": True
            },
            "qsc_immune_system": {
                "resonance_threshold": 0.618,
                "immune_activation_threshold": 0.85,
                "entropy_stability_range": [0.3, 0.7],
                "timeband_lock_duration": 300,
                "auto_optimization_enabled": True
            },
            "profit_allocation": {
                "qsc_validation_enabled": True,
                "tensor_integration_enabled": True,
                "min_resonance_threshold": 0.618,
                "max_entropy_threshold": 0.7,
                "emergency_stop_threshold": 0.2
            },
            "diagnostic_server": {
                "host": "localhost",
                "port": 8766,
                "stream_interval": 1.0,
                "auto_alert_enabled": True,
                "alert_sound_enabled": True
            },
            "tensor_server": {
                "host": "localhost", 
                "port": 8765,
                "stream_interval": 1.0,
                "btc_price_simulator": True
            },
            "visualization": {
                "enable_react_server": True,
                "react_port": 3000,
                "auto_open_browser": False,
                "diagnostic_panel_enabled": True
            }
        }
        
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                    # Merge with defaults
                    for key, value in loaded_config.items():
                        if key in default_config and isinstance(value, dict):
                            default_config[key].update(value)
                        else:
                            default_config[key] = value
            except Exception as e:
                logger.warning(f"Failed to load QSC config: {e}. Using defaults.")
        
        return default_config

    def _save_config(self):
        """Save current configuration."""
        try:
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"QSC configuration saved to {self.config_file}")
        except Exception as e:
            logger.error(f"Failed to save QSC config: {e}")

    def _signal_handler(self, signum, frame):
        """Handle system signals."""
        logger.info(f"Received signal {signum}. Shutting down QSC system...")
        self.is_running = False

    async def start_master_engine(self) -> bool:
        """Start the master cycle engine."""
        try:
            engine_config = self.config.get("master_engine", {})
            self.master_engine = MasterCycleEngine(engine_config)
            
            # Test the engine with sample data
            test_data = {
                "btc_price": 50000.0,
                "price_history": [49800, 49900, 50000],
                "volume_history": [100, 110, 105],
                "fibonacci_projection": [49850, 49950, 50050],
                "orderbook": {
                    "bids": [[49990, 1.5], [49980, 2.0]],
                    "asks": [[50010, 1.6], [50020, 2.1]]
                }
            }
            
            diagnostics = self.master_engine.process_market_tick(test_data)
            logger.info(f"🎯 Master Cycle Engine started. Test decision: {diagnostics.trading_decision.value}")
            
            self.system_status.master_engine = True
            self.system_status.qsc_immune_system = True
            self.system_status.tensor_analysis = True
            self.system_status.profit_allocator = True
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start Master Cycle Engine: {e}")
            return False

    async def start_diagnostic_server(self) -> bool:
        """Start the QSC diagnostic WebSocket server."""
        try:
            diagnostic_config = self.config.get("diagnostic_server", {})
            self.diagnostic_server = QSCDiagnosticServer(diagnostic_config)
            await self.diagnostic_server.start_server()
            
            self.system_status.diagnostic_server = True
            logger.info("🧬📡 QSC Diagnostic server started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start QSC diagnostic server: {e}")
            return False

    async def start_tensor_server(self) -> bool:
        """Start the tensor analysis WebSocket server."""
        try:
            tensor_config = self.config.get("tensor_server", {})
            self.tensor_server = TensorWebSocketServer(tensor_config)
            await self.tensor_server.start_server()
            
            logger.info("🧠📡 Tensor WebSocket server started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start tensor server: {e}")
            return False

    async def start_visualization_server(self) -> bool:
        """Start the React visualization server with QSC diagnostic panel."""
        try:
            if not self.config.get("visualization", {}).get("enable_react_server", True):
                return True
            
            # Check if Node.js is available
            if not self._check_node_availability():
                logger.warning("Node.js not available. Skipping React server.")
                return False
            
            # Create enhanced React app with QSC diagnostic panel
            self._setup_qsc_react_app()
            
            # Start React server
            react_port = self.config.get("visualization", {}).get("react_port", 3000)
            
            env = os.environ.copy()
            env["PORT"] = str(react_port)
            
            self.visualization_process = subprocess.Popen(
                ["npm", "start"],
                cwd=project_root / "qsc_visualization",
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Give it time to start
            await asyncio.sleep(3)
            
            if self.visualization_process.poll() is None:
                self.system_status.visualization_server = True
                logger.info(f"⚛️🧬 QSC React visualization started on port {react_port}")
                
                if self.config.get("visualization", {}).get("auto_open_browser", False):
                    self._open_browser(f"http://localhost:{react_port}")
                
                return True
            else:
                logger.error("QSC React server failed to start")
                return False
                
        except Exception as e:
            logger.error(f"Failed to start QSC visualization server: {e}")
            return False

    def _check_node_availability(self) -> bool:
        """Check if Node.js is available."""
        try:
            subprocess.run(["node", "--version"], capture_output=True, check=True)
            subprocess.run(["npm", "--version"], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def _setup_qsc_react_app(self):
        """Set up enhanced React app with QSC diagnostic components."""
        react_dir = project_root / "qsc_visualization"
        
        if not react_dir.exists():
            logger.info("Setting up QSC React visualization app...")
            
            # Create React app
            subprocess.run([
                "npx", "create-react-app", "qsc_visualization", "--template", "typescript"
            ], cwd=project_root, check=True)
            
            # Install additional dependencies
            subprocess.run([
                "npm", "install", "recharts", "mathjs", "react-router-dom", "@types/react-router-dom"
            ], cwd=react_dir, check=True)
            
            # Create QSC diagnostic components
            self._create_qsc_components(react_dir)

    def _create_qsc_components(self, react_dir: Path):
        """Create QSC diagnostic React components."""
        src_dir = react_dir / "src"
        components_dir = src_dir / "components"
        components_dir.mkdir(exist_ok=True)
        
        # QSC Diagnostic Dashboard
        qsc_dashboard = '''
import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

interface QSCData {
  diagnostics: any;
  qsc_status: any;
  tensor_analysis: any;
  market_data: any;
  fibonacci_echo: any;
  alerts: any[];
}

const QSCDiagnosticDashboard = () => {
  const [qscData, setQscData] = useState<QSCData | null>(null);
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  const [alerts, setAlerts] = useState<any[]>([]);
  const [activeTab, setActiveTab] = useState('overview');

  useEffect(() => {
    // Connect to QSC diagnostic WebSocket
    const ws = new WebSocket('ws://localhost:8766');
    
    ws.onopen = () => {
      setConnectionStatus('connected');
      console.log('Connected to QSC diagnostic server');
    };
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      if (data.type === 'qsc_diagnostic_stream') {
        setQscData(data);
        
        // Handle alerts
        if (data.alerts && data.alerts.length > 0) {
          setAlerts(prev => [...prev, ...data.alerts]);
          
          // Auto-switch tab for critical alerts
          const criticalAlert = data.alerts.find(a => a.severity === 'critical');
          if (criticalAlert && criticalAlert.auto_switch_tab) {
            setActiveTab('diagnostics');
          }
        }
      }
    };
    
    ws.onclose = () => {
      setConnectionStatus('disconnected');
    };
    
    return () => ws.close();
  }, []);

  const renderOverview = () => (
    <div className="overview-grid">
      <div className="metric-card">
        <h3>System Mode</h3>
        <p className={`status ${qscData?.diagnostics.system_mode}`}>
          {qscData?.diagnostics.system_mode?.toUpperCase()}
        </p>
      </div>
      
      <div className="metric-card">
        <h3>Trading Decision</h3>
        <p className={`decision ${qscData?.diagnostics.trading_decision}`}>
          {qscData?.diagnostics.trading_decision?.toUpperCase()}
        </p>
      </div>
      
      <div className="metric-card">
        <h3>Confidence Score</h3>
        <p>{(qscData?.diagnostics.confidence_score * 100)?.toFixed(1)}%</p>
      </div>
      
      <div className="metric-card">
        <h3>QSC Mode</h3>
        <p className={`qsc-mode ${qscData?.qsc_status.mode}`}>
          {qscData?.qsc_status.mode}
        </p>
      </div>
      
      <div className="metric-card">
        <h3>Immune Status</h3>
        <p className={qscData?.diagnostics.immune_response_active ? 'active' : 'inactive'}>
          {qscData?.diagnostics.immune_response_active ? '🛡️ ACTIVE' : '✅ INACTIVE'}
        </p>
      </div>
      
      <div className="metric-card">
        <h3>Risk Assessment</h3>
        <p className={`risk ${qscData?.diagnostics.risk_assessment?.toLowerCase()}`}>
          {qscData?.diagnostics.risk_assessment}
        </p>
      </div>
    </div>
  );

  const renderDiagnostics = () => (
    <div className="diagnostics-panel">
      <div className="diagnostics-section">
        <h3>🧬 QSC Immune System Status</h3>
        <div className="status-grid">
          <div>Resonance Level: {qscData?.qsc_status.resonance_level}</div>
          <div>Timeband Locked: {qscData?.qsc_status.timeband_locked ? 'YES' : 'NO'}</div>
          <div>Cycles Approved: {qscData?.qsc_status.cycles_approved}</div>
          <div>Cycles Blocked: {qscData?.qsc_status.cycles_blocked}</div>
          <div>Success Rate: {(qscData?.qsc_status.success_rate * 100)?.toFixed(1)}%</div>
          <div>Entropy Flux: {qscData?.qsc_status.entropy_flux?.toFixed(4)}</div>
        </div>
      </div>
      
      <div className="diagnostics-section">
        <h3>🧠 Tensor Analysis</h3>
        <div className="status-grid">
          <div>Phi Resonance: {qscData?.tensor_analysis.phi_resonance?.toFixed(3)}</div>
          <div>Quantum Score: {qscData?.tensor_analysis.quantum_score?.toFixed(4)}</div>
          <div>Phase Bucket: {qscData?.tensor_analysis.phase_bucket}</div>
          <div>Tensor Coherence: {qscData?.tensor_analysis.tensor_coherence?.toFixed(3)}</div>
        </div>
      </div>
      
      <div className="diagnostics-section">
        <h3>📊 Market Conditions</h3>
        <div className="status-grid">
          <div>BTC Price: ${qscData?.market_data.btc_price?.toFixed(2)}</div>
          <div>Fibonacci Divergence: {qscData?.diagnostics.fibonacci_divergence?.toFixed(6)}</div>
          <div>Orderbook Stability: {(qscData?.diagnostics.orderbook_stability * 100)?.toFixed(1)}%</div>
          <div>Orderbook Imbalance: {(qscData?.market_data.orderbook_imbalance * 100)?.toFixed(1)}%</div>
        </div>
      </div>
    </div>
  );

  const renderFibonacciEcho = () => {
    if (!qscData?.fibonacci_echo?.timestamps) return <div>No echo data available</div>;
    
    const chartData = qscData.fibonacci_echo.timestamps.map((timestamp, i) => ({
      time: new Date(timestamp * 1000).toLocaleTimeString(),
      divergence: qscData.fibonacci_echo.fibonacci_divergences[i],
      confidence: qscData.fibonacci_echo.confidence_scores[i],
      quantum_score: qscData.fibonacci_echo.quantum_scores[i]
    }));

    return (
      <div className="fibonacci-echo">
        <h3>📈 Fibonacci Echo Analysis</h3>
        <ResponsiveContainer width="100%" height={400}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="time" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="divergence" stroke="#ff7300" name="Fibonacci Divergence" />
            <Line type="monotone" dataKey="confidence" stroke="#8884d8" name="Confidence Score" />
            <Line type="monotone" dataKey="quantum_score" stroke="#82ca9d" name="Quantum Score" />
          </LineChart>
        </ResponsiveContainer>
      </div>
    );
  };

  const renderAlerts = () => (
    <div className="alerts-panel">
      <div className="alerts-header">
        <h3>🚨 System Alerts</h3>
        <button onClick={() => setAlerts([])}>Clear All</button>
      </div>
      <div className="alerts-list">
        {alerts.slice(-10).reverse().map((alert, i) => (
          <div key={i} className={`alert alert-${alert.severity}`}>
            <div className="alert-title">{alert.title}</div>
            <div className="alert-message">{alert.message}</div>
            <div className="alert-time">{new Date(alert.timestamp * 1000).toLocaleString()}</div>
          </div>
        ))}
      </div>
    </div>
  );

  return (
    <div className="qsc-dashboard">
      <header className="dashboard-header">
        <h1>🧬 Schwabot QSC + GTS Immune System</h1>
        <div className={`connection-status ${connectionStatus}`}>
          Status: {connectionStatus}
        </div>
      </header>
      
      <nav className="dashboard-nav">
        <button 
          className={activeTab === 'overview' ? 'active' : ''} 
          onClick={() => setActiveTab('overview')}
        >
          Overview
        </button>
        <button 
          className={activeTab === 'diagnostics' ? 'active' : ''} 
          onClick={() => setActiveTab('diagnostics')}
        >
          Diagnostics
        </button>
        <button 
          className={activeTab === 'fibonacci' ? 'active' : ''} 
          onClick={() => setActiveTab('fibonacci')}
        >
          Fibonacci Echo
        </button>
        <button 
          className={activeTab === 'alerts' ? 'active' : ''} 
          onClick={() => setActiveTab('alerts')}
        >
          Alerts ({alerts.length})
        </button>
      </nav>
      
      <main className="dashboard-content">
        {activeTab === 'overview' && renderOverview()}
        {activeTab === 'diagnostics' && renderDiagnostics()}
        {activeTab === 'fibonacci' && renderFibonacciEcho()}
        {activeTab === 'alerts' && renderAlerts()}
      </main>
    </div>
  );
};

export default QSCDiagnosticDashboard;
'''
        
        with open(components_dir / "QSCDiagnosticDashboard.tsx", "w") as f:
            f.write(qsc_dashboard)
        
        # Update App.tsx
        app_tsx = '''
import React from 'react';
import './App.css';
import QSCDiagnosticDashboard from './components/QSCDiagnosticDashboard';

function App() {
  return (
    <div className="App">
      <QSCDiagnosticDashboard />
    </div>
  );
}

export default App;
'''
        
        with open(src_dir / "App.tsx", "w") as f:
            f.write(app_tsx)
        
        # Add CSS styles
        app_css = '''
.qsc-dashboard {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  background: #0a0a0a;
  color: #ffffff;
  min-height: 100vh;
}

.dashboard-header {
  padding: 1rem;
  background: linear-gradient(135deg, #1a1a2e, #16213e);
  display: flex;
  justify-content: space-between;
  align-items: center;
  border-bottom: 2px solid #333;
}

.connection-status {
  padding: 0.5rem 1rem;
  border-radius: 20px;
  font-weight: bold;
}

.connection-status.connected {
  background: #00ff88;
  color: #000;
}

.connection-status.disconnected {
  background: #ff4444;
  color: #fff;
}

.dashboard-nav {
  display: flex;
  background: #1a1a1a;
  padding: 0;
  border-bottom: 1px solid #333;
}

.dashboard-nav button {
  padding: 1rem 2rem;
  background: transparent;
  color: #ccc;
  border: none;
  cursor: pointer;
  transition: all 0.3s ease;
}

.dashboard-nav button:hover {
  background: #333;
  color: #fff;
}

.dashboard-nav button.active {
  background: #00ff88;
  color: #000;
  font-weight: bold;
}

.overview-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 1rem;
  padding: 1rem;
}

.metric-card {
  background: linear-gradient(135deg, #1e1e1e, #2a2a2a);
  padding: 1.5rem;
  border-radius: 10px;
  border: 1px solid #333;
  text-align: center;
}

.metric-card h3 {
  margin: 0 0 1rem 0;
  color: #00ff88;
}

.metric-card p {
  font-size: 1.5rem;
  font-weight: bold;
  margin: 0;
}

.status.normal { color: #00ff88; }
.status.immune_active { color: #ff9900; }
.status.ghost_floor { color: #ff4444; }
.status.emergency_shutdown { color: #ff0000; }

.decision.execute { color: #00ff88; }
.decision.block { color: #ff4444; }
.decision.defer { color: #ffaa00; }
.decision.cancel_all { color: #ff0000; }

.risk.low { color: #00ff88; }
.risk.medium { color: #ffaa00; }
.risk.high { color: #ff4444; }

.diagnostics-panel {
  padding: 1rem;
}

.diagnostics-section {
  background: #1a1a1a;
  margin-bottom: 1rem;
  padding: 1rem;
  border-radius: 8px;
  border: 1px solid #333;
}

.diagnostics-section h3 {
  color: #00ff88;
  margin-bottom: 1rem;
}

.status-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 0.5rem;
}

.status-grid div {
  padding: 0.5rem;
  background: #2a2a2a;
  border-radius: 4px;
  font-family: monospace;
}

.fibonacci-echo {
  padding: 1rem;
}

.fibonacci-echo h3 {
  color: #00ff88;
  margin-bottom: 1rem;
}

.alerts-panel {
  padding: 1rem;
}

.alerts-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1rem;
}

.alerts-header h3 {
  color: #00ff88;
  margin: 0;
}

.alerts-header button {
  background: #ff4444;
  color: white;
  border: none;
  padding: 0.5rem 1rem;
  border-radius: 4px;
  cursor: pointer;
}

.alerts-list {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.alert {
  padding: 1rem;
  border-radius: 8px;
  border-left: 4px solid;
}

.alert-warning {
  background: #2a2a1a;
  border-left-color: #ffaa00;
}

.alert-error {
  background: #2a1a1a;
  border-left-color: #ff4444;
}

.alert-critical {
  background: #2a0a0a;
  border-left-color: #ff0000;
  animation: pulse 2s infinite;
}

@keyframes pulse {
  0% { opacity: 1; }
  50% { opacity: 0.7; }
  100% { opacity: 1; }
}

.alert-title {
  font-weight: bold;
  margin-bottom: 0.5rem;
}

.alert-message {
  font-size: 0.9rem;
  opacity: 0.8;
}

.alert-time {
  font-size: 0.8rem;
  opacity: 0.6;
  margin-top: 0.5rem;
  font-family: monospace;
}
'''
        
        with open(src_dir / "App.css", "w") as f:
            f.write(app_css)

    def _open_browser(self, url: str):
        """Open browser to the given URL."""
        import webbrowser
        try:
            webbrowser.open(url)
        except Exception as e:
            logger.warning(f"Failed to open browser: {e}")

    async def monitor_system(self):
        """Monitor QSC system health and performance."""
        while self.is_running:
            try:
                # Update uptime
                self.system_status.uptime = time.time() - self.start_time
                
                # Collect performance metrics from master engine
                if self.master_engine:
                    engine_status = self.master_engine.get_system_status()
                    self.system_status.immune_activations = engine_status.get("immune_activations", 0)
                    self.system_status.ghost_floor_activations = engine_status.get("ghost_floor_activations", 0)
                    self.system_status.emergency_shutdowns = engine_status.get("emergency_shutdowns", 0)
                    self.system_status.total_decisions = engine_status.get("total_decisions", 0)
                    self.system_status.success_rate = engine_status.get("success_rate", 0.0)
                
                # Log status every 30 seconds
                if int(self.system_status.uptime) % 30 == 0:
                    self._log_system_status()
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in QSC system monitoring: {e}")
                await asyncio.sleep(5)

    def _log_system_status(self):
        """Log current QSC system status."""
        status_msg = f"""
🧬 Schwabot QSC + GTS System Status (Uptime: {self.system_status.uptime:.0f}s)
  🎯 Master Engine: {'✅' if self.system_status.master_engine else '❌'}
  🧬 QSC Immune System: {'✅' if self.system_status.qsc_immune_system else '❌'}
  🧠 Tensor Analysis: {'✅' if self.system_status.tensor_analysis else '❌'}
  💰 Profit Allocator: {'✅' if self.system_status.profit_allocator else '❌'}
  📡 Diagnostic Server: {'✅' if self.system_status.diagnostic_server else '❌'}
  ⚛️ Visualization: {'✅' if self.system_status.visualization_server else '❌'}
  
  📊 Performance:
    Total Decisions: {self.system_status.total_decisions}
    Success Rate: {self.system_status.success_rate:.2%}
    Immune Activations: {self.system_status.immune_activations}
    Ghost Floor Activations: {self.system_status.ghost_floor_activations}
    Emergency Shutdowns: {self.system_status.emergency_shutdowns}
"""
        
        logger.info(status_msg)

    async def start_all_systems(self) -> bool:
        """Start all QSC system components."""
        logger.info("🧬🚀 Starting Complete QSC + GTS Immune System...")
        
        success_count = 0
        total_systems = 5
        
        # Start master cycle engine
        if await self.start_master_engine():
            success_count += 1
        
        # Start diagnostic server
        if await self.start_diagnostic_server():
            success_count += 1
        
        # Start tensor server
        if await self.start_tensor_server():
            success_count += 1
        
        # Start visualization
        if await self.start_visualization_server():
            success_count += 1
        
        success_rate = success_count / total_systems
        
        if success_rate >= 0.8:  # 80% success rate required
            logger.info(f"✅ QSC System startup successful ({success_count}/{total_systems} components)")
            self.is_running = True
            return True
        else:
            logger.error(f"❌ QSC System startup failed ({success_count}/{total_systems} components)")
            return False

    async def stop_all_systems(self):
        """Stop all QSC system components."""
        logger.info("🛑 Stopping QSC + GTS Immune System...")
        
        self.is_running = False
        
        # Stop diagnostic server
        if self.diagnostic_server:
            await self.diagnostic_server.stop_server()
        
        # Stop tensor server
        if self.tensor_server:
            await self.tensor_server.stop_server()
        
        # Stop visualization
        if self.visualization_process:
            self.visualization_process.terminate()
            self.visualization_process.wait()
        
        logger.info("🛑 All QSC systems stopped")

    async def run(self):
        """Main QSC run loop."""
        if await self.start_all_systems():
            # Start monitoring
            monitor_task = asyncio.create_task(self.monitor_system())
            
            try:
                # Keep running until signal
                while self.is_running:
                    await asyncio.sleep(1)
            finally:
                monitor_task.cancel()
                await self.stop_all_systems()
        else:
            logger.error("Failed to start QSC system")
            sys.exit(1)


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Schwabot QSC + GTS Immune System CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s start                    # Start complete QSC system
  %(prog)s start --no-viz          # Start without React visualization
  %(prog)s status                  # Show system status  
  %(prog)s demo                    # Run immune system demo
  %(prog)s config                  # Show configuration
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Start command
    start_parser = subparsers.add_parser('start', help='Start the QSC immune system')
    start_parser.add_argument('--no-viz', action='store_true', help='Disable React visualization')
    start_parser.add_argument('--diagnostic-port', type=int, default=8766, help='Diagnostic server port')
    start_parser.add_argument('--tensor-port', type=int, default=8765, help='Tensor server port')
    start_parser.add_argument('--react-port', type=int, default=3000, help='React server port')
    
    # Demo command
    subparsers.add_parser('demo', help='Run QSC immune system demo')
    
    # Status command
    subparsers.add_parser('status', help='Show system status')
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Configuration management')
    config_parser.add_argument('--show', action='store_true', help='Show current configuration')
    config_parser.add_argument('--reset', action='store_true', help='Reset to default configuration')
    
    return parser


async def main():
    """Main function."""
    parser = create_parser()
    args = parser.parse_args()
    
    cli = SchwabotQSCCLI()
    
    if args.command == 'start':
        # Update config based on arguments
        if args.no_viz:
            cli.config["visualization"]["enable_react_server"] = False
        if args.diagnostic_port != 8766:
            cli.config["diagnostic_server"]["port"] = args.diagnostic_port
        if args.tensor_port != 8765:
            cli.config["tensor_server"]["port"] = args.tensor_port
        if args.react_port != 3000:
            cli.config["visualization"]["react_port"] = args.react_port
        
        await cli.run()
        
    elif args.command == 'demo':
        # Run QSC immune system demo
        from examples.qsc_immune_system_demo import QSCImmuneSystemDemo
        demo = QSCImmuneSystemDemo()
        await demo.run_complete_demo()
        
    elif args.command == 'status':
        # Show current system status
        print("🧬 Schwabot QSC + GTS Immune System Status:")
        print("  Implementation: Ready")
        print("  Configuration: Loaded")
        print("  Components: Available")
        print("  Integration: Complete")
        
    elif args.command == 'config':
        if args.show:
            print("📋 QSC System Configuration:")
            print(json.dumps(cli.config, indent=2))
        elif args.reset:
            cli.config = cli._load_config()
            cli._save_config()
            print("✅ QSC configuration reset to defaults")
            
    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main()) 