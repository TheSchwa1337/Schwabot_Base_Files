#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Schwabot Unified Trading Interface
==================================
Comprehensive GUI frontend that unifies all Schwabot trading capabilities
with real-time visualization, strategy management, and multi-profile execution.

Features:
- Real-time trading dashboard
- Strategy trigger controls
- Multi-profile Coinbase management
- Live signal visualization
- GPU/CPU runtime switching
- AI command interface
- System status monitoring
"""

import asyncio
import json
import logging
import os
import sys
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from flask import Flask, jsonify, render_template, request, session, Response
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room, leave_room
import numpy as np

# Add core directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import Schwabot core components
try:
    from core.strategy_mapper import StrategyMapper
    from core.profile_router import ProfileRouter
    from core.api.multi_profile_coinbase_manager import MultiProfileCoinbaseManager
    from core.visual_layer_controller import VisualLayerController
    from core.unified_math_system import generate_unified_hash
    from core.hardware_auto_detector import HardwareAutoDetector
    from core.soulprint_registry import SoulprintRegistry
    SCHWABOT_CORE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Some Schwabot core components not available: {e}")
    SCHWABOT_CORE_AVAILABLE = False
    
    # Create stub functions for missing components
    def generate_unified_hash(data, prefix=""):
        """Stub function for generate_unified_hash when core component is not available."""
        import hashlib
        import time
        payload = f"{prefix}_{str(data)}_{time.time()}"
        return hashlib.sha256(payload.encode()).hexdigest()[:16]
    
    class VisualLayerController:
        """Stub class for VisualLayerController when not available."""
        def __init__(self):
            pass

# Import configuration
try:
    from config.config_loader import ConfigLoader
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'schwabot_unified_interface_secret_2025'
CORS(app)

# Initialize SocketIO for real-time communication
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# Global system state
class SystemState(Enum):
    """System operational states."""
    INITIALIZING = "initializing"
    READY = "ready"
    TRADING = "trading"
    ERROR = "error"
    MAINTENANCE = "maintenance"

@dataclass
class TradingSession:
    """Manages trading session state."""
    session_id: str
    start_time: datetime
    mode: str = "demo"  # demo, live, backtest
    active_trades: List[Dict[str, Any]] = field(default_factory=list)
    portfolio_value: float = 10000.0
    total_profit: float = 0.0
    win_rate: float = 0.0
    current_asset: str = "BTC/USDC"
    current_price: float = 60000.0
    api_connected: bool = False
    ccxt_config: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StrategyControl:
    """Strategy execution control."""
    strategy_id: str
    profile_id: str
    asset: str
    confidence: float
    signal_strength: float
    status: str = "pending"  # pending, executing, completed, failed
    timestamp: datetime = field(default_factory=datetime.now)
    result: Optional[Dict[str, Any]] = None

class SchwabotUnifiedInterface:
    """Main unified interface controller."""
    
    def __init__(self):
        self.app = app
        self.socketio = socketio
        self.system_state = SystemState.INITIALIZING
        self.logger = logging.getLogger(__name__)
        
        # Initialize core components
        self.strategy_mapper = None
        self.profile_router = None
        self.visual_controller = None
        self.hardware_detector = None
        self.soulprint_registry = None
        self.config_loader = None
        
        # Trading state
        self.current_session = None
        self.active_strategies: Dict[str, StrategyControl] = {}
        self.profile_states: Dict[str, Dict[str, Any]] = {}
        
        # Performance metrics
        self.total_trades = 0
        self.successful_trades = 0
        self.failed_trades = 0
        self.total_profit = 0.0
        
        # Real-time data
        self.live_signals: List[Dict[str, Any]] = []
        self.system_metrics: Dict[str, Any] = {}
        
        # Initialize components
        self._initialize_components()
        self._setup_routes()
        self._setup_socketio_events()
        
    def _initialize_components(self):
        """Initialize Schwabot core components."""
        try:
            self.logger.info("🔄 Initializing Schwabot Unified Interface...")
            
            # Initialize hardware detector
            if SCHWABOT_CORE_AVAILABLE:
                self.hardware_detector = HardwareAutoDetector()
                self.logger.info("✅ Hardware detector initialized")
                
                # Initialize strategy mapper
                self.strategy_mapper = StrategyMapper()
                self.logger.info("✅ Strategy mapper initialized")
                
                # Initialize profile router
                config_path = "config/coinbase_profiles.yaml"
                if os.path.exists(config_path):
                    self.profile_router = ProfileRouter(config_path)
                    self.logger.info("✅ Profile router initialized")
                
                # Initialize visual controller
                self.visual_controller = VisualLayerController()
                self.logger.info("✅ Visual controller initialized")
                
                # Initialize soulprint registry
                self.soulprint_registry = SoulprintRegistry("data/soulprint_registry.json")
                self.logger.info("✅ Soulprint registry initialized")
            
            # Initialize config loader
            if CONFIG_AVAILABLE:
                self.config_loader = ConfigLoader()
                self.logger.info("✅ Config loader initialized")
            
            # Create initial trading session
            self.current_session = TradingSession(
                session_id=generate_unified_hash([time.time()], "session"),
                start_time=datetime.now()
            )
            
            self.system_state = SystemState.READY
            self.logger.info("✅ Schwabot Unified Interface initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize components: {e}")
            self.system_state = SystemState.ERROR
    
    def _setup_routes(self):
        """Setup Flask routes."""
        
        @app.route('/')
        def dashboard():
            """Main trading dashboard."""
            return render_template('unified_dashboard.html', 
                                 state=self.system_state.value,
                                 session=self.current_session,
                                 profiles=self.profile_states)
        
        @app.route('/api/system/status')
        def get_system_status():
            """Get system status and metrics."""
            return jsonify({
                'system_state': self.system_state.value,
                'session': {
                    'id': self.current_session.session_id if self.current_session else None,
                    'mode': self.current_session.mode if self.current_session else 'demo',
                    'portfolio_value': self.current_session.portfolio_value if self.current_session else 0.0,
                    'total_profit': self.current_session.total_profit if self.current_session else 0.0,
                    'api_connected': self.current_session.api_connected if self.current_session else False
                },
                'metrics': {
                    'total_trades': self.total_trades,
                    'successful_trades': self.successful_trades,
                    'failed_trades': self.failed_trades,
                    'total_profit': self.total_profit,
                    'win_rate': (self.successful_trades / max(self.total_trades, 1)) * 100
                },
                'hardware': self._get_hardware_info(),
                'active_strategies': len(self.active_strategies),
                'live_signals': len(self.live_signals)
            })
        
        @app.route('/api/strategy/execute', methods=['POST'])
        def execute_strategy():
            """Execute a trading strategy."""
            try:
                data = request.get_json()
                strategy_id = data.get('strategy_id')
                profile_id = data.get('profile_id')
                asset = data.get('asset', 'BTC/USDC')
                
                if not strategy_id or not profile_id:
                    return jsonify({'success': False, 'error': 'Missing strategy_id or profile_id'})
                
                # Create strategy control
                strategy_control = StrategyControl(
                    strategy_id=strategy_id,
                    profile_id=profile_id,
                    asset=asset,
                    confidence=data.get('confidence', 0.7),
                    signal_strength=data.get('signal_strength', 0.5)
                )
                
                # Execute strategy asynchronously
                asyncio.create_task(self._execute_strategy_async(strategy_control))
                
                self.active_strategies[strategy_id] = strategy_control
                
                return jsonify({
                    'success': True,
                    'strategy_id': strategy_id,
                    'status': 'executing',
                    'message': f'Strategy {strategy_id} execution started'
                })
                
            except Exception as e:
                self.logger.error(f"Error executing strategy: {e}")
                return jsonify({'success': False, 'error': str(e)})
        
        @app.route('/api/profile/list')
        def list_profiles():
            """List available trading profiles."""
            try:
                if self.profile_router and self.profile_router.multi_profile_manager:
                    profiles = {}
                    for profile_id, profile_config in self.profile_router.active_profiles.items():
                        profiles[profile_id] = {
                            'name': profile_config.get('name', profile_id),
                            'enabled': profile_config.get('enabled', False),
                            'trading_pairs': profile_config.get('trading_params', {}).get('trading_pairs', []),
                            'max_positions': profile_config.get('trading_params', {}).get('max_open_positions', 5)
                        }
                    return jsonify({'success': True, 'profiles': profiles})
                else:
                    return jsonify({'success': True, 'profiles': {}})
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        @app.route('/api/profile/activate', methods=['POST'])
        def activate_profile():
            """Activate a trading profile."""
            try:
                data = request.get_json()
                profile_id = data.get('profile_id')
                
                if not profile_id:
                    return jsonify({'success': False, 'error': 'Missing profile_id'})
                
                if self.profile_router:
                    # Update profile state
                    self.profile_states[profile_id] = {
                        'active': True,
                        'activated_at': datetime.now().isoformat(),
                        'status': 'ready'
                    }
                    
                    return jsonify({
                        'success': True,
                        'profile_id': profile_id,
                        'message': f'Profile {profile_id} activated'
                    })
                else:
                    return jsonify({'success': False, 'error': 'Profile router not available'})
                    
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        @app.route('/api/visualization/generate', methods=['POST'])
        def generate_visualization():
            """Generate trading visualization."""
            try:
                data = request.get_json()
                viz_type = data.get('type', 'price_chart')
                symbol = data.get('symbol', 'BTC/USDC')
                
                if self.visual_controller:
                    # Generate visualization asynchronously
                    asyncio.create_task(self._generate_visualization_async(viz_type, symbol))
                    
                    return jsonify({
                        'success': True,
                        'type': viz_type,
                        'symbol': symbol,
                        'message': f'Visualization generation started for {symbol}'
                    })
                else:
                    return jsonify({'success': False, 'error': 'Visual controller not available'})
                    
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        @app.route('/api/hardware/gpu_toggle', methods=['POST'])
        def toggle_gpu_mode():
            """Toggle GPU acceleration mode."""
            try:
                data = request.get_json()
                enable_gpu = data.get('enable_gpu', True)
                
                if self.hardware_detector:
                    # Update hardware configuration
                    self.hardware_detector.gpu_enabled = enable_gpu
                    
                    return jsonify({
                        'success': True,
                        'gpu_enabled': enable_gpu,
                        'message': f'GPU acceleration {"enabled" if enable_gpu else "disabled"}'
                    })
                else:
                    return jsonify({'success': False, 'error': 'Hardware detector not available'})
                    
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        @app.route('/api/ai/command', methods=['POST'])
        def process_ai_command():
            """Process AI command from external sources."""
            try:
                data = request.get_json()
                command_hash = data.get('command_hash')
                source = data.get('source', 'unknown')  # claude, gpt4o, r1
                
                if not command_hash:
                    return jsonify({'success': False, 'error': 'Missing command_hash'})
                
                # Process AI command asynchronously
                asyncio.create_task(self._process_ai_command_async(command_hash, source))
                
                return jsonify({
                    'success': True,
                    'command_hash': command_hash,
                    'source': source,
                    'message': 'AI command processing started'
                })
                
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        @app.route('/api/session/switch_mode', methods=['POST'])
        def switch_session_mode():
            """Switch trading session mode."""
            try:
                data = request.get_json()
                new_mode = data.get('mode', 'demo')
                
                if new_mode not in ['demo', 'live', 'backtest']:
                    return jsonify({'success': False, 'error': 'Invalid mode'})
                
                if self.current_session:
                    self.current_session.mode = new_mode
                    
                    return jsonify({
                        'success': True,
                        'mode': new_mode,
                        'message': f'Switched to {new_mode} mode'
                    })
                else:
                    return jsonify({'success': False, 'error': 'No active session'})
                    
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
    
    def _setup_socketio_events(self):
        """Setup SocketIO events for real-time communication."""
        
        @socketio.on('connect')
        def handle_connect():
            """Handle client connection."""
            self.logger.info(f"Client connected: {request.sid}")
            emit('system_status', {
                'state': self.system_state.value,
                'timestamp': datetime.now().isoformat()
            })
        
        @socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection."""
            self.logger.info(f"Client disconnected: {request.sid}")
        
        @socketio.on('subscribe_signals')
        def handle_subscribe_signals(data):
            """Subscribe to live trading signals."""
            room = f"signals_{data.get('symbol', 'BTC/USDC')}"
            join_room(room)
            emit('subscription_confirmed', {
                'room': room,
                'message': f'Subscribed to {room} signals'
            })
        
        @socketio.on('unsubscribe_signals')
        def handle_unsubscribe_signals(data):
            """Unsubscribe from live trading signals."""
            room = f"signals_{data.get('symbol', 'BTC/USDC')}"
            leave_room(room)
            emit('unsubscription_confirmed', {
                'room': room,
                'message': f'Unsubscribed from {room} signals'
            })
    
    async def _execute_strategy_async(self, strategy_control: StrategyControl):
        """Execute strategy asynchronously."""
        try:
            self.logger.info(f"🔄 Executing strategy {strategy_control.strategy_id}")
            strategy_control.status = "executing"
            
            # Generate strategy using strategy mapper
            if self.strategy_mapper:
                strategy_matrix = await self.strategy_mapper.generate_profile_strategy(
                    strategy_control.profile_id,
                    {'asset': strategy_control.asset},
                    generate_unified_hash([time.time()], "strategy")
                )
                
                if strategy_matrix:
                    # Execute through profile router
                    if self.profile_router:
                        await self.profile_router._execute_profile_strategy(
                            strategy_control.profile_id,
                            {
                                'strategy_name': strategy_matrix.strategy_type.value,
                                'assets': strategy_matrix.assets,
                                'confidence': strategy_matrix.confidence,
                                'signal_strength': strategy_matrix.signal_strength
                            }
                        )
                    
                    strategy_control.status = "completed"
                    strategy_control.result = {
                        'strategy_type': strategy_matrix.strategy_type.value,
                        'confidence': strategy_matrix.confidence,
                        'profit_zones': strategy_matrix.profit_zones
                    }
                    
                    self.successful_trades += 1
                    self.total_trades += 1
                    
                    # Emit real-time update
                    socketio.emit('strategy_completed', {
                        'strategy_id': strategy_control.strategy_id,
                        'status': 'completed',
                        'result': strategy_control.result
                    })
                    
                else:
                    strategy_control.status = "failed"
                    strategy_control.result = {'error': 'Failed to generate strategy'}
                    self.failed_trades += 1
                    self.total_trades += 1
            else:
                strategy_control.status = "failed"
                strategy_control.result = {'error': 'Strategy mapper not available'}
                self.failed_trades += 1
                self.total_trades += 1
                
        except Exception as e:
            self.logger.error(f"Error executing strategy {strategy_control.strategy_id}: {e}")
            strategy_control.status = "failed"
            strategy_control.result = {'error': str(e)}
            self.failed_trades += 1
            self.total_trades += 1
    
    async def _generate_visualization_async(self, viz_type: str, symbol: str):
        """Generate visualization asynchronously."""
        try:
            if self.visual_controller:
                # Generate sample tick data for visualization
                tick_data = self._generate_sample_tick_data(symbol)
                
                if viz_type == 'price_chart':
                    visual_analysis = await self.visual_controller.generate_price_chart(
                        tick_data, symbol
                    )
                    
                    if visual_analysis:
                        # Emit visualization data
                        socketio.emit('visualization_ready', {
                            'type': viz_type,
                            'symbol': symbol,
                            'chart_data': visual_analysis.chart_data.decode('utf-8'),
                            'ai_insights': visual_analysis.ai_insights,
                            'confidence_score': visual_analysis.confidence_score
                        })
                        
        except Exception as e:
            self.logger.error(f"Error generating visualization: {e}")
    
    async def _process_ai_command_async(self, command_hash: str, source: str):
        """Process AI command asynchronously."""
        try:
            self.logger.info(f"🔄 Processing AI command from {source}: {command_hash}")
            
            # Parse command hash into strategy parameters
            # This is a simplified implementation - you can enhance this
            strategy_params = self._parse_ai_command(command_hash)
            
            if strategy_params:
                # Create and execute strategy based on AI command
                strategy_control = StrategyControl(
                    strategy_id=f"ai_{source}_{int(time.time())}",
                    profile_id=strategy_params.get('profile_id', 'default'),
                    asset=strategy_params.get('asset', 'BTC/USDC'),
                    confidence=strategy_params.get('confidence', 0.7),
                    signal_strength=strategy_params.get('signal_strength', 0.5)
                )
                
                await self._execute_strategy_async(strategy_control)
                
                # Emit AI command processed event
                socketio.emit('ai_command_processed', {
                    'command_hash': command_hash,
                    'source': source,
                    'strategy_id': strategy_control.strategy_id,
                    'status': 'processed'
                })
                
        except Exception as e:
            self.logger.error(f"Error processing AI command: {e}")
    
    def _get_hardware_info(self) -> Dict[str, Any]:
        """Get hardware information."""
        if self.hardware_detector:
            return {
                'gpu_available': self.hardware_detector.gpu_available,
                'gpu_enabled': getattr(self.hardware_detector, 'gpu_enabled', False),
                'gpu_name': self.hardware_detector.gpu_name if hasattr(self.hardware_detector, 'gpu_name') else 'Unknown',
                'cpu_cores': self.hardware_detector.cpu_cores if hasattr(self.hardware_detector, 'cpu_cores') else 0,
                'memory_gb': self.hardware_detector.memory_gb if hasattr(self.hardware_detector, 'memory_gb') else 0
            }
        return {'gpu_available': False, 'gpu_enabled': False}
    
    def _generate_sample_tick_data(self, symbol: str) -> List[Dict[str, Any]]:
        """Generate sample tick data for visualization."""
        data = []
        base_price = 60000.0 if 'BTC' in symbol else 3000.0
        current_time = time.time()
        
        for i in range(100):
            timestamp = current_time - (100 - i) * 60  # 1-minute intervals
            price = base_price + np.random.normal(0, 100)  # Random price movement
            volume = np.random.uniform(10, 1000)
            
            data.append({
                'timestamp': timestamp,
                'price': price,
                'volume': volume,
                'symbol': symbol
            })
        
        return data
    
    def _parse_ai_command(self, command_hash: str) -> Optional[Dict[str, Any]]:
        """Parse AI command hash into strategy parameters."""
        try:
            # Simple hash parsing - you can enhance this with more sophisticated logic
            hash_int = int(command_hash[:8], 16) if len(command_hash) >= 8 else 0
            
            return {
                'profile_id': f'profile_{hash_int % 3 + 1}',
                'asset': 'BTC/USDC' if hash_int % 2 == 0 else 'ETH/USDC',
                'confidence': 0.5 + (hash_int % 100) / 200,  # 0.5 to 1.0
                'signal_strength': 0.3 + (hash_int % 70) / 100  # 0.3 to 1.0
            }
        except Exception as e:
            self.logger.error(f"Error parsing AI command: {e}")
            return None
    
    def start(self, host: str = '0.0.0.0', port: int = 8080, debug: bool = False):
        """Start the unified interface."""
        self.logger.info(f"🚀 Starting Schwabot Unified Interface on {host}:{port}")
        
        if debug:
            self.app.run(host=host, port=port, debug=debug)
        else:
            self.socketio.run(self.app, host=host, port=port, debug=debug)

def main():
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and start unified interface
    interface = SchwabotUnifiedInterface()
    interface.start(debug=True)

if __name__ == '__main__':
    main() 