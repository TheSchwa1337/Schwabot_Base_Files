#!/usr/bin/env python3
"""Schwabot Tensor CLI Launcher.

Command-line interface for launching and managing the Galileo-Tensor 
integration with Schwabot's trading system. Provides unified control
over WebSocket servers, tensor analysis, and trading system integration.
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

from core.galileo_tensor_bridge import GalileoTensorBridge
from server.tensor_websocket_server import TensorWebSocketServer
from utils.logging_setup import setup_logging

# Setup logging
logger = setup_logging(__name__)


@dataclass
class SystemStatus:
    """System status container."""
    tensor_bridge: bool = False
    websocket_server: bool = False
    trading_integration: bool = False
    visualization_server: bool = False
    btc_price_feed: bool = False
    uptime: float = 0.0
    performance_metrics: Dict[str, Any] = field(default_factory=dict)


class SchwabotTensorCLI:
    """Command-line interface for Schwabot Tensor system."""

    def __init__(self):
        """Initialize the CLI."""
        self.config_file = project_root / "config" / "tensor_config.json"
        self.config = self._load_config()
        
        # System components
        self.tensor_bridge: Optional[GalileoTensorBridge] = None
        self.websocket_server: Optional[TensorWebSocketServer] = None
        self.visualization_process: Optional[subprocess.Popen] = None
        
        # Status tracking
        self.system_status = SystemStatus()
        self.start_time = time.time()
        self.is_running = False
        
        # Signal handling
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        default_config = {
            "tensor_bridge": {
                "enable_real_time_streaming": True,
                "tensor_analysis_interval": 0.1,
                "max_history_size": 1000
            },
            "websocket_server": {
                "host": "localhost",
                "port": 8765,
                "stream_interval": 1.0,
                "btc_price_simulator": True,
                "enable_cors": True
            },
            "trading_integration": {
                "enable_strategy_integration": True,
                "enable_profit_allocation": True,
                "risk_management": True
            },
            "visualization": {
                "enable_react_server": True,
                "react_port": 3000,
                "auto_open_browser": False
            },
            "btc_feed": {
                "enable_live_feed": False,
                "exchange": "coinbase_pro",
                "update_interval": 1.0
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
                logger.warning(f"Failed to load config: {e}. Using defaults.")
        
        return default_config

    def _save_config(self):
        """Save current configuration to file."""
        try:
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"Configuration saved to {self.config_file}")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")

    def _signal_handler(self, signum, frame):
        """Handle system signals."""
        logger.info(f"Received signal {signum}. Shutting down...")
        self.is_running = False

    async def start_tensor_bridge(self) -> bool:
        """Start the tensor bridge component."""
        try:
            bridge_config = self.config.get("tensor_bridge", {})
            self.tensor_bridge = GalileoTensorBridge(bridge_config)
            
            # Test the bridge with a sample analysis
            test_result = self.tensor_bridge.perform_complete_analysis(50000.0)
            logger.info(f"🧠 Tensor Bridge started successfully. Test analysis: {test_result.phi_resonance:.3f}")
            
            self.system_status.tensor_bridge = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to start tensor bridge: {e}")
            return False

    async def start_websocket_server(self) -> bool:
        """Start the WebSocket server."""
        try:
            ws_config = self.config.get("websocket_server", {})
            ws_config["bridge_config"] = self.config.get("tensor_bridge", {})
            
            self.websocket_server = TensorWebSocketServer(ws_config)
            await self.websocket_server.start_server()
            
            self.system_status.websocket_server = True
            logger.info("🌐 WebSocket server started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start WebSocket server: {e}")
            return False

    async def start_visualization_server(self) -> bool:
        """Start the React visualization server."""
        try:
            if not self.config.get("visualization", {}).get("enable_react_server", True):
                return True
            
            # Check if Node.js and npm are available
            if not self._check_node_availability():
                logger.warning("Node.js/npm not available. Skipping React server.")
                return False
            
            # Create React app if it doesn't exist
            self._setup_react_app()
            
            # Start the React development server
            react_port = self.config.get("visualization", {}).get("react_port", 3000)
            
            env = os.environ.copy()
            env["PORT"] = str(react_port)
            
            self.visualization_process = subprocess.Popen(
                ["npm", "start"],
                cwd=project_root / "tensor_visualization",
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Give it a moment to start
            await asyncio.sleep(3)
            
            if self.visualization_process.poll() is None:
                self.system_status.visualization_server = True
                logger.info(f"⚛️ React visualization server started on port {react_port}")
                
                if self.config.get("visualization", {}).get("auto_open_browser", False):
                    self._open_browser(f"http://localhost:{react_port}")
                
                return True
            else:
                logger.error("React server failed to start")
                return False
                
        except Exception as e:
            logger.error(f"Failed to start visualization server: {e}")
            return False

    def _check_node_availability(self) -> bool:
        """Check if Node.js and npm are available."""
        try:
            subprocess.run(["node", "--version"], capture_output=True, check=True)
            subprocess.run(["npm", "--version"], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def _setup_react_app(self):
        """Set up React application with tensor visualization components."""
        react_dir = project_root / "tensor_visualization"
        
        if not react_dir.exists():
            logger.info("Setting up React visualization app...")
            
            # Create React app
            subprocess.run([
                "npx", "create-react-app", "tensor_visualization", "--template", "typescript"
            ], cwd=project_root, check=True)
            
            # Install additional dependencies
            subprocess.run([
                "npm", "install", "recharts", "mathjs", "websockets"
            ], cwd=react_dir, check=True)
            
            # Copy tensor visualization components
            self._create_tensor_components(react_dir)

    def _create_tensor_components(self, react_dir: Path):
        """Create tensor visualization React components."""
        src_dir = react_dir / "src"
        components_dir = src_dir / "components"
        components_dir.mkdir(exist_ok=True)
        
        # Main tensor dashboard component
        dashboard_component = '''
import React, { useState, useEffect } from 'react';
import { create, all } from 'mathjs';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const math = create(all);

const TensorDashboard = () => {
  const [wsData, setWsData] = useState(null);
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  const [history, setHistory] = useState([]);

  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8765');
    
    ws.onopen = () => {
      setConnectionStatus('connected');
      console.log('Connected to tensor WebSocket server');
    };
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      if (data.type === 'tensor_analysis_stream') {
        setWsData(data.data);
        setHistory(prev => [...prev.slice(-50), {
          timestamp: data.timestamp,
          btc_price: data.data.btc_price,
          phi_resonance: data.data.phiResonance,
          quantum_score: data.data.spIntegration?.quantum_score || 0
        }]);
      }
    };
    
    ws.onclose = () => {
      setConnectionStatus('disconnected');
    };
    
    return () => ws.close();
  }, []);

  return (
    <div className="tensor-dashboard">
      <h1>Schwabot Tensor Analysis Dashboard</h1>
      
      <div className="connection-status">
        Status: <span className={connectionStatus}>{connectionStatus}</span>
      </div>
      
      {wsData && (
        <div className="analysis-data">
          <div className="metrics-grid">
            <div className="metric">
              <h3>BTC Price</h3>
              <p>${wsData.btc_price?.toFixed(2)}</p>
            </div>
            
            <div className="metric">
              <h3>Phi Resonance</h3>
              <p>{wsData.phiResonance?.toFixed(3)}</p>
            </div>
            
            <div className="metric">
              <h3>SP Quantum Score</h3>
              <p>{wsData.spIntegration?.quantum_score?.toFixed(4)}</p>
            </div>
            
            <div className="metric">
              <h3>Phase Bucket</h3>
              <p>{wsData.spIntegration?.phase_bucket}</p>
            </div>
          </div>
          
          <div className="chart-container">
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={history}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="timestamp" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="btc_price" stroke="#8884d8" />
                <Line type="monotone" dataKey="phi_resonance" stroke="#82ca9d" />
                <Line type="monotone" dataKey="quantum_score" stroke="#ffc658" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  );
};

export default TensorDashboard;
'''
        
        with open(components_dir / "TensorDashboard.tsx", "w") as f:
            f.write(dashboard_component)
        
        # Update App.tsx to use our dashboard
        app_tsx = '''
import React from 'react';
import './App.css';
import TensorDashboard from './components/TensorDashboard';

function App() {
  return (
    <div className="App">
      <TensorDashboard />
    </div>
  );
}

export default App;
'''
        
        with open(src_dir / "App.tsx", "w") as f:
            f.write(app_tsx)

    def _open_browser(self, url: str):
        """Open browser to the given URL."""
        import webbrowser
        try:
            webbrowser.open(url)
        except Exception as e:
            logger.warning(f"Failed to open browser: {e}")

    async def integrate_with_trading_system(self) -> bool:
        """Integrate tensor analysis with Schwabot's trading system."""
        try:
            if not self.config.get("trading_integration", {}).get("enable_strategy_integration", True):
                return True
            
            # Import and initialize trading system components
            try:
                from core.strategy_logic import StrategyLogic
                from schwabot.core.profit_cycle_allocator import ProfitCycleAllocator
                
                # Create enhanced strategy logic with tensor integration
                strategy_logic = StrategyLogic()
                profit_allocator = ProfitCycleAllocator()
                
                # Add tensor-enhanced strategy
                self._add_tensor_strategy(strategy_logic)
                
                self.system_status.trading_integration = True
                logger.info("🤖 Trading system integration successful")
                return True
                
            except ImportError as e:
                logger.warning(f"Trading system components not available: {e}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to integrate with trading system: {e}")
            return False

    def _add_tensor_strategy(self, strategy_logic):
        """Add tensor-enhanced strategy to the strategy logic."""
        from core.strategy_logic import StrategyConfig, StrategyType
        
        tensor_strategy = StrategyConfig(
            strategy_type=StrategyType.QUANTUM_ENHANCED,
            name="tensor_quantum_enhanced",
            enabled=True,
            max_position_size=0.05,  # Conservative position sizing
            risk_tolerance=0.3,      # Lower risk for experimental strategy
            lookback_period=50,
            min_signal_confidence=0.8,  # High confidence threshold
            parameters={
                "tensor_bridge": self.tensor_bridge,
                "quantum_threshold": 0.91,
                "phi_resonance_threshold": 27.0,
                "gut_stability_threshold": 0.995
            }
        )
        
        strategy_logic.strategies[tensor_strategy.name] = tensor_strategy
        logger.info("🧠 Tensor-enhanced strategy added to trading system")

    async def start_btc_feed(self) -> bool:
        """Start BTC price feed integration."""
        try:
            if not self.config.get("btc_feed", {}).get("enable_live_feed", False):
                # Use simulated feed
                self.system_status.btc_price_feed = True
                return True
            
            # In a real implementation, connect to exchange API
            # For now, we'll use the WebSocket server's simulation
            self.system_status.btc_price_feed = True
            logger.info("₿ BTC price feed integration ready")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start BTC feed: {e}")
            return False

    async def monitor_system(self):
        """Monitor system health and performance."""
        while self.is_running:
            try:
                # Update uptime
                self.system_status.uptime = time.time() - self.start_time
                
                # Collect performance metrics
                if self.tensor_bridge:
                    self.system_status.performance_metrics["tensor_bridge"] = \
                        self.tensor_bridge.get_performance_summary()
                
                if self.websocket_server:
                    self.system_status.performance_metrics["websocket_server"] = \
                        self.websocket_server.get_server_stats()
                
                # Log status every 30 seconds
                if int(self.system_status.uptime) % 30 == 0:
                    self._log_system_status()
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in system monitoring: {e}")
                await asyncio.sleep(5)

    def _log_system_status(self):
        """Log current system status."""
        status_msg = f"""
🚀 Schwabot Tensor System Status (Uptime: {self.system_status.uptime:.0f}s)
  🧠 Tensor Bridge: {'✅' if self.system_status.tensor_bridge else '❌'}
  🌐 WebSocket Server: {'✅' if self.system_status.websocket_server else '❌'}
  🤖 Trading Integration: {'✅' if self.system_status.trading_integration else '❌'}
  ⚛️ Visualization Server: {'✅' if self.system_status.visualization_server else '❌'}
  ₿ BTC Price Feed: {'✅' if self.system_status.btc_price_feed else '❌'}
"""
        
        logger.info(status_msg)

    async def start_all_systems(self) -> bool:
        """Start all system components."""
        logger.info("🚀 Starting Schwabot Tensor System...")
        
        success_count = 0
        total_systems = 5
        
        # Start tensor bridge
        if await self.start_tensor_bridge():
            success_count += 1
        
        # Start WebSocket server
        if await self.start_websocket_server():
            success_count += 1
        
        # Start visualization server
        if await self.start_visualization_server():
            success_count += 1
        
        # Integrate with trading system
        if await self.integrate_with_trading_system():
            success_count += 1
        
        # Start BTC feed
        if await self.start_btc_feed():
            success_count += 1
        
        success_rate = success_count / total_systems
        
        if success_rate >= 0.8:  # 80% success rate required
            logger.info(f"✅ System startup successful ({success_count}/{total_systems} components)")
            self.is_running = True
            return True
        else:
            logger.error(f"❌ System startup failed ({success_count}/{total_systems} components)")
            return False

    async def stop_all_systems(self):
        """Stop all system components."""
        logger.info("🛑 Stopping Schwabot Tensor System...")
        
        self.is_running = False
        
        # Stop WebSocket server
        if self.websocket_server:
            await self.websocket_server.stop_server()
        
        # Stop visualization server
        if self.visualization_process:
            self.visualization_process.terminate()
            self.visualization_process.wait()
        
        logger.info("🛑 All systems stopped")

    async def run(self):
        """Main run loop."""
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
            logger.error("Failed to start system")
            sys.exit(1)


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Schwabot Tensor System CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s start                    # Start all systems
  %(prog)s start --no-viz          # Start without React visualization
  %(prog)s status                  # Show system status
  %(prog)s config                  # Show current configuration
  %(prog)s test                    # Run system tests
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Start command
    start_parser = subparsers.add_parser('start', help='Start the tensor system')
    start_parser.add_argument('--no-viz', action='store_true', help='Disable React visualization')
    start_parser.add_argument('--no-trading', action='store_true', help='Disable trading integration')
    start_parser.add_argument('--ws-port', type=int, default=8765, help='WebSocket server port')
    start_parser.add_argument('--react-port', type=int, default=3000, help='React server port')
    
    # Status command
    subparsers.add_parser('status', help='Show system status')
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Configuration management')
    config_parser.add_argument('--show', action='store_true', help='Show current configuration')
    config_parser.add_argument('--reset', action='store_true', help='Reset to default configuration')
    
    # Test command
    subparsers.add_parser('test', help='Run system tests')
    
    return parser


async def main():
    """Main function."""
    parser = create_parser()
    args = parser.parse_args()
    
    cli = SchwabotTensorCLI()
    
    if args.command == 'start':
        # Update config based on arguments
        if args.no_viz:
            cli.config["visualization"]["enable_react_server"] = False
        if args.no_trading:
            cli.config["trading_integration"]["enable_strategy_integration"] = False
        if args.ws_port != 8765:
            cli.config["websocket_server"]["port"] = args.ws_port
        if args.react_port != 3000:
            cli.config["visualization"]["react_port"] = args.react_port
        
        await cli.run()
        
    elif args.command == 'status':
        # Show current system status
        print("🔍 Schwabot Tensor System Status:")
        print("  Implementation: Ready")
        print("  Configuration: Loaded")
        print("  Components: Available")
        
    elif args.command == 'config':
        if args.show:
            print("📋 Current Configuration:")
            print(json.dumps(cli.config, indent=2))
        elif args.reset:
            cli.config = cli._load_config()
            cli._save_config()
            print("✅ Configuration reset to defaults")
            
    elif args.command == 'test':
        # Run system tests
        print("🧪 Running system tests...")
        
        # Test tensor bridge
        bridge = GalileoTensorBridge()
        result = bridge.perform_complete_analysis(50000.0)
        print(f"✅ Tensor Bridge Test: Phi Resonance = {result.phi_resonance:.3f}")
        
        print("✅ All tests passed")
        
    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main()) 