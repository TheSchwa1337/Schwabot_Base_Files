from core.unified_math_system import unified_math
#!/usr/bin/env python3
"""
Entropy-Driven API Layer for Schwabot
=====================================

This module creates a Flask-based API layer that integrates with Schwabot's
mathematical framework while providing AI endpoints for ChatGPT, Anthropic, and Gemini.

Key Features:
- Entropy-based API triggers and hash-relative functions
- Integration with 16-bit positioning system and 10,000-tick map
- Respects CCO, UFS, SFS, SFSS core logic
- AI dialogue system for trading decisions
- Hash-based command functions and decision tracking
- Real-time market state broadcasting

This layer acts as the bridge between Schwabot's internal logic and external AI systems.
"""

import asyncio
import hashlib
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import threading

# Flask imports
try:
    from flask import Flask, request, jsonify, Response
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    logging.warning("Flask not available. Install with: pip install flask flask-cors")

# WebSocket imports
try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    logging.warning("WebSockets not available. Install with: pip install websockets")

# Import Schwabot core components
try:
    from .fault_bus import FaultBus, FaultBusEvent, FaultType
    from .data_integration_layer import DataIntegrationLayer, CryptoDataPoint
    from .dlt_waveform_engine import DLTWaveformEngine
    from .multi_bit_btc_processor import MultiBitBTCProcessor
    from .riddle_gemm import RiddleGEMMEngine
    from .temporal_execution_correction_layer import TemporalExecutionCorrectionLayer
    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False
    logging.warning("Core Schwabot components not available")

logger = logging.getLogger(__name__)


@dataclass
class EntropyTrigger:
    """Represents an entropy-based trigger for API actions."""
    trigger_id: str
    hash_signature: str
    entropy_threshold: float
    activation_time: datetime
    expiry_time: datetime
    ai_models: List[str]  # ['gpt', 'claude', 'gemini']
    callback_function: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AIResponse:
    """Represents an AI model's response to a trading decision."""
    model_name: str
    response_hash: str
    confidence_score: float
    recommended_action: str
    reasoning: str
    timestamp: datetime
    decision_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HashCommand:
    """Represents a hash-based command function."""
    command_id: str
    hash_pattern: str
    execution_function: str
    parameters: Dict[str, Any]
    priority: int
    created_at: datetime
    executed_at: Optional[datetime] = None


class EntropyAPILayer:
    """
    Entropy-driven API layer that integrates with Schwabot's mathematical framework.
    """
    
    def __init__(self, 
                 fault_bus: Optional[FaultBus] = None,
                 data_layer: Optional[DataIntegrationLayer] = None,
                 host: str = 'localhost',
                 port: int = 5000,
                 websocket_port: int = 8765):
        """
        Initialize the entropy API layer.
        
        Args:
            fault_bus: Schwabot's FaultBus instance
            data_layer: Data integration layer
            host: Flask server host
            port: Flask server port
            websocket_port: WebSocket server port
        """
        self.fault_bus = fault_bus
        self.data_layer = data_layer
        self.host = host
        self.port = port
        self.websocket_port = websocket_port
        
        # Entropy tracking
        self.entropy_history: deque = deque(maxlen=1000)
        self.current_entropy: float = 0.0
        self.entropy_threshold: float = 0.5
        
        # Hash-based command system
        self.hash_commands: Dict[str, HashCommand] = {}
        self.command_history: List[HashCommand] = []
        
        # AI response tracking
        self.ai_responses: List[AIResponse] = []
        self.ai_consensus_cache: Dict[str, Dict[str, Any]] = {}
        
        # Trigger system
        self.entropy_triggers: List[EntropyTrigger] = []
        self.active_triggers: Dict[str, EntropyTrigger] = {}
        
        # 16-bit positioning system integration
        self.bit_positions: Dict[int, Dict[str, Any]] = {}
        self.position_history: deque = deque(maxlen=10000)  # 10,000 tick map
        
        # Core engine references
        self.dlt_engine = None
        self.multi_bit_engine = None
        self.riddle_engine = None
        self.temporal_corrector = None
        
        # Flask app
        self.app = None
        self.websocket_server = None
        
        # Threading
        self.is_running = False
        self.update_thread = None
        
        logger.info("🧠 Entropy API Layer initialized")
        
    def initialize_core_engines(self):
        """Initialize core Schwabot engines."""
        if not CORE_AVAILABLE:
            logger.warning("Core components not available - using mock engines")
            return
            
        try:
            # Initialize core engines
            self.dlt_engine = DLTWaveformEngine(history_size=100)
            self.multi_bit_engine = MultiBitBTCProcessor(
                timeframes={"1m": 60, "5m": 300, "15m": 900}
            )
            self.riddle_engine = RiddleGEMMEngine(vector_size=10)
            self.temporal_corrector = TemporalExecutionCorrectionLayer()
            
            logger.info("✅ Core engines initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize core engines: {e}")
    
    def calculate_entropy(self, data: Dict[str, Any]) -> float:
        """
        Calculate entropy from market data and system state.
        
        Args:
            data: Market data and system state
            
        Returns:
            Entropy value between 0 and 1
        """
        try:
            # Extract key components for entropy calculation
            price_volatility = data.get('price_volatility', 0.0)
            volume_change = data.get('volume_change', 0.0)
            hash_variance = data.get('hash_variance', 0.0)
            fault_count = data.get('active_faults', 0)
            
            # Normalize components
            normalized_volatility = unified_math.min(price_volatility, 1.0)
            normalized_volume = unified_math.min(unified_math.abs(volume_change), 1.0)
            normalized_hash = unified_math.min(hash_variance, 1.0)
            normalized_faults = unified_math.min(fault_count / 10.0, 1.0)  # Max 10 faults
            
            # Calculate weighted entropy
            entropy = (
                normalized_volatility * 0.3 +
                normalized_volume * 0.25 +
                normalized_hash * 0.25 +
                normalized_faults * 0.2
            )
            
            return unified_math.min(unified_math.max(entropy, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"❌ Error calculating entropy: {e}")
            return 0.0
    
    def generate_hash_signature(self, data: Dict[str, Any]) -> str:
        """
        Generate hash signature for current system state.
        
        Args:
            data: System state data
            
        Returns:
            SHA256 hash signature
        """
        try:
            # Create state string
            state_string = json.dumps(data, sort_keys=True)
            
            # Add timestamp for uniqueness
            timestamp = str(int(time.time()))
            state_string += timestamp
            
            # Generate hash
            hash_signature = hashlib.sha256(state_string.encode()).hexdigest()
            
            return hash_signature[:16]  # Return first 16 characters
            
        except Exception as e:
            logger.error(f"❌ Error generating hash signature: {e}")
            return "0000000000000000"
    
    def update_16_bit_positions(self, market_data: Dict[str, Any]):
        """
        Update 16-bit positioning system based on market data.
        
        Args:
            market_data: Current market data
        """
        try:
            # Calculate position for each bit (0-15)
            for bit in range(16):
                # Create bit-specific data
                bit_data = {
                    'bit_position': bit,
                    'price': market_data.get('price', 0),
                    'volume': market_data.get('volume', 0),
                    'entropy': self.current_entropy,
                    'timestamp': time.time()
                }
                
                # Generate bit-specific hash
                bit_hash = self.generate_hash_signature(bit_data)
                
                # Store bit position
                self.bit_positions[bit] = {
                    'hash': bit_hash,
                    'data': bit_data,
                    'active': bit_hash.startswith('0'),  # Simple activation logic
                    'last_updated': datetime.now()
                }
            
            # Add to position history
            self.position_history.append({
                'timestamp': datetime.now(),
                'positions': self.bit_positions.copy(),
                'entropy': self.current_entropy
            })
            
        except Exception as e:
            logger.error(f"❌ Error updating 16-bit positions: {e}")
    
    def register_hash_command(self, 
                            command_id: str,
                            hash_pattern: str,
                            execution_function: str,
                            parameters: Dict[str, Any],
                            priority: int = 1) -> bool:
        """
        Register a hash-based command function.
        
        Args:
            command_id: Unique command identifier
            hash_pattern: Hash pattern to match
            execution_function: Function to execute
            parameters: Command parameters
            priority: Execution priority (higher = more important)
            
        Returns:
            True if registered successfully
        """
        try:
            command = HashCommand(
                command_id=command_id,
                hash_pattern=hash_pattern,
                execution_function=execution_function,
                parameters=parameters,
                priority=priority,
                created_at=datetime.now()
            )
            
            self.hash_commands[command_id] = command
            logger.info(f"✅ Registered hash command: {command_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error registering hash command: {e}")
            return False
    
    def execute_hash_commands(self, current_hash: str) -> List[Dict[str, Any]]:
        """
        Execute hash commands that match the current hash.
        
        Args:
            current_hash: Current system hash
            
        Returns:
            List of executed commands
        """
        executed_commands = []
        
        try:
            # Find matching commands
            matching_commands = []
            for command in self.hash_commands.values():
                if current_hash.startswith(command.hash_pattern):
                    matching_commands.append(command)
            
            # Sort by priority
            matching_commands.sort(key=lambda x: x.priority, reverse=True)
            
            # Execute commands
            for command in matching_commands:
                try:
                    # Execute the command function
                    result = self._execute_command_function(
                        command.execution_function,
                        command.parameters
                    )
                    
                    # Mark as executed
                    command.executed_at = datetime.now()
                    self.command_history.append(command)
                    
                    executed_commands.append({
                        'command_id': command.command_id,
                        'result': result,
                        'executed_at': command.executed_at.isoformat()
                    })
                    
                    logger.info(f"✅ Executed hash command: {command.command_id}")
                    
                except Exception as e:
                    logger.error(f"❌ Error executing command {command.command_id}: {e}")
            
        except Exception as e:
            logger.error(f"❌ Error executing hash commands: {e}")
        
        return executed_commands
    
    def _execute_command_function(self, function_name: str, parameters: Dict[str, Any]) -> Any:
        """
        Execute a command function by name.
        
        Args:
            function_name: Name of the function to execute
            parameters: Function parameters
            
        Returns:
            Function result
        """
        try:
            # Map function names to actual functions
            function_map = {
                'update_market_signals': self._update_market_signals,
                'trigger_ai_analysis': self._trigger_ai_analysis,
                'adjust_entropy_threshold': self._adjust_entropy_threshold,
                'update_bit_positions': self._update_bit_positions,
                'broadcast_state': self._broadcast_state
            }
            
            if function_name in function_map:
                return function_map[function_name](**parameters)
            else:
                logger.warning(f"⚠️ Unknown function: {function_name}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error executing function {function_name}: {e}")
            return None
    
    def _update_market_signals(self, **kwargs) -> Dict[str, Any]:
        """Update market signals in the FaultBus."""
        if self.fault_bus and self.data_layer:
            try:
                current_data = self.data_layer.get_current_data()
                self.fault_bus.update_market_signals(
                    price=current_data.get('price', 0),
                    volume=current_data.get('volume', 0),
                    volatility=current_data.get('volatility', 0)
                )
                return {'status': 'success', 'message': 'Market signals updated'}
            except Exception as e:
                return {'status': 'error', 'message': str(e)}
        return {'status': 'error', 'message': 'FaultBus or DataLayer not available'}
    
    def _trigger_ai_analysis(self, **kwargs) -> Dict[str, Any]:
        """Trigger AI analysis of current state."""
        try:
            # Create analysis request
            analysis_request = {
                'timestamp': datetime.now().isoformat(),
                'entropy': self.current_entropy,
                'bit_positions': self.bit_positions,
                'market_state': self._get_current_market_state(),
                'request_id': f"ai_analysis_{int(time.time())}"
            }
            
            # Store for AI processing
            self.ai_responses.append(AIResponse(
                model_name='system',
                response_hash=self.generate_hash_signature(analysis_request),
                confidence_score=0.0,
                recommended_action='analyze',
                reasoning='AI analysis triggered',
                timestamp=datetime.now(),
                decision_context=analysis_request
            ))
            
            return {'status': 'success', 'request_id': analysis_request['request_id']}
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def _adjust_entropy_threshold(self, new_threshold: float, **kwargs) -> Dict[str, Any]:
        """Adjust entropy threshold."""
        try:
            old_threshold = self.entropy_threshold
            self.entropy_threshold = unified_math.max(0.0, unified_math.min(1.0, new_threshold))
            
            return {
                'status': 'success',
                'old_threshold': old_threshold,
                'new_threshold': self.entropy_threshold
            }
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def _update_bit_positions(self, **kwargs) -> Dict[str, Any]:
        """Update bit positions."""
        try:
            if self.data_layer:
                current_data = self.data_layer.get_current_data()
                self.update_16_bit_positions(current_data)
                return {'status': 'success', 'positions_updated': len(self.bit_positions)}
            return {'status': 'error', 'message': 'DataLayer not available'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def _broadcast_state(self, **kwargs) -> Dict[str, Any]:
        """Broadcast current state via WebSocket."""
        try:
            if self.websocket_server:
                state_data = self._get_current_market_state()
                asyncio.create_task(self.websocket_server.broadcast_data(state_data))
                return {'status': 'success', 'message': 'State broadcasted'}
            return {'status': 'error', 'message': 'WebSocket server not available'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def _get_current_market_state(self) -> Dict[str, Any]:
        """Get current market state for API responses."""
        try:
            state = {
                'timestamp': datetime.now().isoformat(),
                'entropy': self.current_entropy,
                'entropy_threshold': self.entropy_threshold,
                'bit_positions': {
                    bit: {
                        'hash': pos['hash'],
                        'active': pos['active'],
                        'last_updated': pos['last_updated'].isoformat()
                    }
                    for bit, pos in self.bit_positions.items()
                },
                'active_commands': len([c for c in self.hash_commands.values() if c.executed_at is None]),
                'ai_responses_count': len(self.ai_responses),
                'position_history_size': len(self.position_history)
            }
            
            # Add market data if available
            if self.data_layer:
                market_data = self.data_layer.get_current_data()
                state['market_data'] = market_data
            
            # Add FaultBus data if available
            if self.fault_bus:
                fault_bus_data = self.fault_bus.get_corridor_analytics()
                state['fault_bus'] = fault_bus_data
            
            return state
            
        except Exception as e:
            logger.error(f"❌ Error getting market state: {e}")
            return {'error': str(e)}
    
    def create_flask_app(self) -> Flask:
        """Create and configure Flask application."""
        if not FLASK_AVAILABLE:
            raise ImportError("Flask not available")
        
        app = Flask(__name__)
        CORS(app)  # Enable CORS for all routes
        
        # API Routes
        
        @app.route('/api/entropy/current', methods=['GET'])
        def get_current_entropy():
            """Get current entropy value."""
            return jsonify({
                'entropy': self.current_entropy,
                'threshold': self.entropy_threshold,
                'timestamp': datetime.now().isoformat()
            })
        
        @app.route('/api/entropy/history', methods=['GET'])
        def get_entropy_history():
            """Get entropy history."""
            limit = request.args.get('limit', 100, type=int)
            history = list(self.entropy_history)[-limit:]
            return jsonify({
                'history': history,
                'count': len(history)
            })
        
        @app.route('/api/bit-positions', methods=['GET'])
        def get_bit_positions():
            """Get current 16-bit positions."""
            return jsonify({
                'positions': self.bit_positions,
                'timestamp': datetime.now().isoformat()
            })
        
        @app.route('/api/hash-commands', methods=['GET'])
        def get_hash_commands():
            """Get registered hash commands."""
            return jsonify({
                'commands': {
                    cmd_id: {
                        'hash_pattern': cmd.hash_pattern,
                        'execution_function': cmd.execution_function,
                        'priority': cmd.priority,
                        'created_at': cmd.created_at.isoformat(),
                        'executed_at': cmd.executed_at.isoformat() if cmd.executed_at else None
                    }
                    for cmd_id, cmd in self.hash_commands.items()
                }
            })
        
        @app.route('/api/hash-commands', methods=['POST'])
        def register_hash_command():
            """Register a new hash command."""
            try:
                data = request.get_json()
                success = self.register_hash_command(
                    command_id=data['command_id'],
                    hash_pattern=data['hash_pattern'],
                    execution_function=data['execution_function'],
                    parameters=data.get('parameters', {}),
                    priority=data.get('priority', 1)
                )
                
                return jsonify({
                    'success': success,
                    'message': 'Command registered' if success else 'Failed to register command'
                })
                
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 400
        
        @app.route('/api/ai/responses', methods=['GET'])
        def get_ai_responses():
            """Get AI responses."""
            limit = request.args.get('limit', 50, type=int)
            responses = self.ai_responses[-limit:]
            
            return jsonify({
                'responses': [
                    {
                        'model_name': resp.model_name,
                        'confidence_score': resp.confidence_score,
                        'recommended_action': resp.recommended_action,
                        'reasoning': resp.reasoning,
                        'timestamp': resp.timestamp.isoformat()
                    }
                    for resp in responses
                ]
            })
        
        @app.route('/api/ai/consensus', methods=['GET'])
        def get_ai_consensus():
            """Get AI consensus on recent decisions."""
            return jsonify({
                'consensus': self.ai_consensus_cache,
                'timestamp': datetime.now().isoformat()
            })
        
        @app.route('/api/market/state', methods=['GET'])
        def get_market_state():
            """Get current market state."""
            return jsonify(self._get_current_market_state())
        
        @app.route('/api/system/status', methods=['GET'])
        def get_system_status():
            """Get system status."""
            return jsonify({
                'status': 'running' if self.is_running else 'stopped',
                'uptime': time.time() - self.start_time if hasattr(self, 'start_time') else 0,
                'entropy_history_size': len(self.entropy_history),
                'position_history_size': len(self.position_history),
                'active_commands': len([c for c in self.hash_commands.values() if c.executed_at is None]),
                'ai_responses_count': len(self.ai_responses)
            })
        
        @app.route('/api/entropy/threshold', methods=['POST'])
        def adjust_entropy_threshold():
            """Adjust entropy threshold."""
            try:
                data = request.get_json()
                new_threshold = float(data['threshold'])
                result = self._adjust_entropy_threshold(new_threshold)
                return jsonify(result)
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 400
        
        return app
    
    async def start_websocket_server(self):
        """Start WebSocket server for real-time data broadcasting."""
        if not WEBSOCKETS_AVAILABLE:
            logger.warning("WebSockets not available")
            return
        
        try:
            async def handler(websocket, path):
                """Handle WebSocket connections."""
                try:
                    async for message in websocket:
                        # Handle incoming messages
                        data = json.loads(message)
                        await self._handle_websocket_message(websocket, data)
                except websockets.exceptions.ConnectionClosed:
                    pass
                except Exception as e:
                    logger.error(f"❌ WebSocket error: {e}")
            
            self.websocket_server = await websockets.serve(
                handler, self.host, self.websocket_port
            )
            logger.info(f"🌐 WebSocket server started on ws://{self.host}:{self.websocket_port}")
            
        except Exception as e:
            logger.error(f"❌ Failed to start WebSocket server: {e}")
    
    async def _handle_websocket_message(self, websocket, data):
        """Handle incoming WebSocket messages."""
        try:
            message_type = data.get('type')
            
            if message_type == 'subscribe':
                # Client wants to subscribe to updates
                await websocket.send(json.dumps({
                    'type': 'subscribed',
                    'timestamp': datetime.now().isoformat()
                }))
                
            elif message_type == 'get_state':
                # Client wants current state
                state = self._get_current_market_state()
                await websocket.send(json.dumps({
                    'type': 'state_update',
                    'data': state,
                    'timestamp': datetime.now().isoformat()
                }))
                
            elif message_type == 'ai_response':
                # AI model sending response
                ai_response = AIResponse(
                    model_name=data['model_name'],
                    response_hash=data['response_hash'],
                    confidence_score=data['confidence_score'],
                    recommended_action=data['recommended_action'],
                    reasoning=data['reasoning'],
                    timestamp=datetime.now(),
                    decision_context=data.get('decision_context', {})
                )
                self.ai_responses.append(ai_response)
                
                # Broadcast to all clients
                await self.websocket_server.broadcast_data({
                    'type': 'ai_response',
                    'data': {
                        'model_name': ai_response.model_name,
                        'confidence_score': ai_response.confidence_score,
                        'recommended_action': ai_response.recommended_action,
                        'timestamp': ai_response.timestamp.isoformat()
                    }
                })
                
        except Exception as e:
            logger.error(f"❌ Error handling WebSocket message: {e}")
    
    def start(self):
        """Start the entropy API layer."""
        if self.is_running:
            logger.warning("Entropy API Layer already running")
            return
        
        try:
            # Initialize core engines
            self.initialize_core_engines()
            
            # Create Flask app
            self.app = self.create_flask_app()
            
            # Start WebSocket server
            asyncio.create_task(self.start_websocket_server())
            
            # Start update thread
            self.is_running = True
            self.start_time = time.time()
            self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
            self.update_thread.start()
            
            logger.info("🚀 Entropy API Layer started")
            
        except Exception as e:
            logger.error(f"❌ Failed to start Entropy API Layer: {e}")
            self.is_running = False
    
    def stop(self):
        """Stop the entropy API layer."""
        self.is_running = False
        logger.info("🛑 Entropy API Layer stopped")
    
    def _update_loop(self):
        """Main update loop for entropy calculations and command execution."""
        while self.is_running:
            try:
                # Get current market data
                market_data = {}
                if self.data_layer:
                    market_data = self.data_layer.get_current_data()
                
                # Calculate entropy
                self.current_entropy = self.calculate_entropy(market_data)
                self.entropy_history.append({
                    'entropy': self.current_entropy,
                    'timestamp': datetime.now().isoformat()
                })
                
                # Update 16-bit positions
                self.update_16_bit_positions(market_data)
                
                # Generate current hash
                current_hash = self.generate_hash_signature(market_data)
                
                # Execute hash commands
                executed_commands = self.execute_hash_commands(current_hash)
                
                # Broadcast state if WebSocket server is available
                if self.websocket_server:
                    state_data = self._get_current_market_state()
                    asyncio.create_task(self.websocket_server.broadcast_data(state_data))
                
                # Sleep for update interval
                time.sleep(3.75)  # 3.75 minutes (225 seconds)
                
            except Exception as e:
                logger.error(f"❌ Error in update loop: {e}")
                time.sleep(1)  # Brief pause on error


# Example usage
def create_entropy_api_layer(fault_bus=None, data_layer=None):
    """Create and configure an entropy API layer."""
    layer = EntropyAPILayer(
        fault_bus=fault_bus,
        data_layer=data_layer,
        host='localhost',
        port=5000,
        websocket_port=8765
    )
    
    # Register some example hash commands
    layer.register_hash_command(
        command_id='high_entropy_alert',
        hash_pattern='f',
        execution_function='trigger_ai_analysis',
        parameters={'analysis_type': 'high_entropy'},
        priority=10
    )
    
    layer.register_hash_command(
        command_id='bit_position_update',
        hash_pattern='0',
        execution_function='update_bit_positions',
        parameters={},
        priority=5
    )
    
    layer.register_hash_command(
        command_id='market_broadcast',
        hash_pattern='a',
        execution_function='broadcast_state',
        parameters={},
        priority=1
    )
    
    return layer


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create entropy API layer
    layer = create_entropy_api_layer()
    
    # Start the layer
    layer.start()
    
    # Run Flask app
    if layer.app:
        layer.app.run(host='localhost', port=5000, debug=True) 