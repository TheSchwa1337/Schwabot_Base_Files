#!/usr/bin/env python3
"""
Unified Schwabot Integration System
==================================

This module integrates all Schwabot components into a unified system:
- FaultBus with integrated engines
- Data Integration Layer for real-time market data
- Entropy API Layer for hash-based triggers
- AI Integration Bridge for multi-model consensus
- 16-bit positioning system and 10,000-tick map
- Respects CCO, UFS, SFS, SFSS core logic

This is the main orchestration layer that brings everything together.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional
import threading

# Import all core components
try:
    from .fault_bus import FaultBus
    from .data_integration_layer import DataIntegrationLayer, DataWebSocketServer
    from .entropy_api_layer import EntropyAPILayer, create_entropy_api_layer
    from .ai_integration_bridge import AIIntegrationBridge, create_ai_bridge
    from .dlt_waveform_engine import DLTWaveformEngine
    from .multi_bit_btc_processor import MultiBitBTCProcessor
    from .riddle_gemm import RiddleGEMMEngine
    from .temporal_execution_correction_layer import TemporalExecutionCorrectionLayer
    CORE_AVAILABLE = True
except ImportError as e:
    CORE_AVAILABLE = False
    logging.warning(f"Core components not available: {e}")

logger = logging.getLogger(__name__)


class UnifiedSchwabotIntegration:
    """
    Unified integration system for Schwabot.
    """
    
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the unified Schwabot integration system.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.is_running = False
        self.start_time = None
        
        # Core components
        self.fault_bus: Optional[FaultBus] = None
        self.data_layer: Optional[DataIntegrationLayer] = None
        self.entropy_api: Optional[EntropyAPILayer] = None
        self.ai_bridge: Optional[AIIntegrationBridge] = None
        self.websocket_server: Optional[DataWebSocketServer] = None
        
        # Core engines
        self.dlt_engine: Optional[DLTWaveformEngine] = None
        self.multi_bit_engine: Optional[MultiBitBTCProcessor] = None
        self.riddle_engine: Optional[RiddleGEMMEngine] = None
        self.temporal_corrector: Optional[TemporalExecutionCorrectionLayer] = None
        
        # Integration state
        self.integration_state = {
            'fault_bus_ready': False,
            'data_layer_ready': False,
            'entropy_api_ready': False,
            'ai_bridge_ready': False,
            'websocket_ready': False
        }
        
        # Performance metrics
        self.metrics = {
            'total_ticks': 0,
            'ai_consensus_count': 0,
            'hash_commands_executed': 0,
            'entropy_calculations': 0,
            'fault_events_processed': 0
        }
        
        logger.info("🧠 Unified Schwabot Integration initialized")
    
    async def initialize_components(self):
        """Initialize all core components."""
        try:
            logger.info("🚀 Initializing Schwabot components...")
            
            # 1. Initialize core engines
            await self._initialize_core_engines()
            
            # 2. Initialize FaultBus
            await self._initialize_fault_bus()
            
            # 3. Initialize Data Integration Layer
            await self._initialize_data_layer()
            
            # 4. Initialize Entropy API Layer
            await self._initialize_entropy_api()
            
            # 5. Initialize AI Integration Bridge
            await self._initialize_ai_bridge()
            
            # 6. Initialize WebSocket Server
            await self._initialize_websocket_server()
            
            logger.info("✅ All components initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize components: {e}")
            raise
    
    async def _initialize_core_engines(self):
        """Initialize core mathematical engines."""
        try:
            if not CORE_AVAILABLE:
                logger.warning("Core components not available - using mock engines")
                return
            
            # Initialize DLT Waveform Engine
            self.dlt_engine = DLTWaveformEngine(history_size=100)
            logger.info("✅ DLT Waveform Engine initialized")
            
            # Initialize Multi-Bit BTC Processor
            self.multi_bit_engine = MultiBitBTCProcessor(
                timeframes={"1m": 60, "5m": 300, "15m": 900}
            )
            logger.info("✅ Multi-Bit BTC Processor initialized")
            
            # Initialize Riddle GEMM Engine
            self.riddle_engine = RiddleGEMMEngine(vector_size=10)
            logger.info("✅ Riddle GEMM Engine initialized")
            
            # Initialize Temporal Execution Correction Layer
            self.temporal_corrector = TemporalExecutionCorrectionLayer()
            logger.info("✅ Temporal Execution Correction Layer initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize core engines: {e}")
            raise
    
    async def _initialize_fault_bus(self):
        """Initialize the FaultBus system."""
        try:
            self.fault_bus = FaultBus(log_path="logs/faults")
            
            # Register custom event handlers
            self._register_fault_bus_handlers()
            
            self.integration_state['fault_bus_ready'] = True
            logger.info("✅ FaultBus initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize FaultBus: {e}")
            raise
    
    def _register_fault_bus_handlers(self):
        """Register custom event handlers for the FaultBus."""
        try:
            @self.fault_bus.register_handler("profit_anomaly")
            def handle_profit_anomaly(event):
                """Handle profit anomaly events."""
                logger.info(f"💰 Profit anomaly detected: {event.severity}")
                # Trigger AI analysis for profit anomalies
                self._trigger_ai_analysis_for_event(event)
            
            @self.fault_bus.register_handler("recursive_loop")
            def handle_recursive_loop(event):
                """Handle recursive loop events."""
                logger.warning(f"🔄 Recursive loop detected: {event.severity}")
                # Trigger entropy threshold adjustment
                self._adjust_entropy_for_loop(event)
            
            @self.fault_bus.register_handler("thermal_critical")
            def handle_thermal_critical(event):
                """Handle thermal critical events."""
                logger.error(f"🌡️ Thermal critical: {event.severity}")
                # Trigger emergency response
                self._trigger_emergency_response(event)
            
            logger.info("✅ FaultBus handlers registered")
            
        except Exception as e:
            logger.error(f"❌ Failed to register FaultBus handlers: {e}")
    
    async def _initialize_data_layer(self):
        """Initialize the Data Integration Layer."""
        try:
            self.data_layer = DataIntegrationLayer(update_interval=225.0)  # 3.75 minutes
            
            # Start data feed
            data_task = asyncio.create_task(self.data_layer.start_data_feed())
            
            self.integration_state['data_layer_ready'] = True
            logger.info("✅ Data Integration Layer initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Data Layer: {e}")
            raise
    
    async def _initialize_entropy_api(self):
        """Initialize the Entropy API Layer."""
        try:
            self.entropy_api = create_entropy_api_layer(
                fault_bus=self.fault_bus,
                data_layer=self.data_layer
            )
            
            # Register additional hash commands
            self._register_entropy_commands()
            
            # Start the entropy API layer
            self.entropy_api.start()
            
            self.integration_state['entropy_api_ready'] = True
            logger.info("✅ Entropy API Layer initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Entropy API: {e}")
            raise
    
    def _register_entropy_commands(self):
        """Register additional hash commands for the entropy API."""
        try:
            # Register commands for different hash patterns
            self.entropy_api.register_hash_command(
                command_id='profit_optimization',
                hash_pattern='8',
                execution_function='trigger_ai_analysis',
                parameters={'analysis_type': 'profit_optimization'},
                priority=8
            )
            
            self.entropy_api.register_hash_command(
                command_id='risk_assessment',
                hash_pattern='c',
                execution_function='trigger_ai_analysis',
                parameters={'analysis_type': 'risk_assessment'},
                priority=7
            )
            
            self.entropy_api.register_hash_command(
                command_id='market_analysis',
                hash_pattern='4',
                execution_function='trigger_ai_analysis',
                parameters={'analysis_type': 'market_analysis'},
                priority=6
            )
            
            self.entropy_api.register_hash_command(
                command_id='bit_position_sync',
                hash_pattern='1',
                execution_function='update_bit_positions',
                parameters={'sync_mode': 'full'},
                priority=5
            )
            
            logger.info("✅ Entropy commands registered")
            
        except Exception as e:
            logger.error(f"❌ Failed to register entropy commands: {e}")
    
    async def _initialize_ai_bridge(self):
        """Initialize the AI Integration Bridge."""
        try:
            self.ai_bridge = create_ai_bridge(entropy_api_layer=self.entropy_api)
            
            # Start the AI bridge
            await self.ai_bridge.start()
            
            self.integration_state['ai_bridge_ready'] = True
            logger.info("✅ AI Integration Bridge initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize AI Bridge: {e}")
            raise
    
    async def _initialize_websocket_server(self):
        """Initialize the WebSocket server."""
        try:
            if self.data_layer:
                self.websocket_server = DataWebSocketServer(
                    data_layer=self.data_layer,
                    host='localhost',
                    port=8765
                )
                
                # Start WebSocket server
                await self.websocket_server.start_server()
                
                self.integration_state['websocket_ready'] = True
                logger.info("✅ WebSocket server initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize WebSocket server: {e}")
            raise
    
    def _trigger_ai_analysis_for_event(self, event):
        """Trigger AI analysis for a specific event."""
        try:
            if self.ai_bridge and self.entropy_api:
                # Create decision request
                market_state = self.entropy_api._get_current_market_state()
                decision_request = self.ai_bridge.create_decision_request(
                    market_state=market_state,
                    entropy_value=self.entropy_api.current_entropy,
                    bit_positions=self.entropy_api.bit_positions,
                    decision_context={
                        'event_type': event.type.value,
                        'severity': event.severity,
                        'module': event.module,
                        'profit_context': event.profit_context
                    }
                )
                
                if decision_request:
                    # Request AI analysis
                    asyncio.create_task(
                        self.ai_bridge.request_ai_analysis(decision_request)
                    )
                    
                    self.metrics['ai_consensus_count'] += 1
                    logger.info(f"🤖 AI analysis triggered for {event.type.value}")
            
        except Exception as e:
            logger.error(f"❌ Error triggering AI analysis: {e}")
    
    def _adjust_entropy_for_loop(self, event):
        """Adjust entropy threshold for recursive loop events."""
        try:
            if self.entropy_api:
                # Increase entropy threshold to reduce sensitivity
                current_threshold = self.entropy_api.entropy_threshold
                new_threshold = min(current_threshold + 0.1, 1.0)
                
                result = self.entropy_api._adjust_entropy_threshold(new_threshold)
                logger.info(f"🔄 Adjusted entropy threshold: {result}")
            
        except Exception as e:
            logger.error(f"❌ Error adjusting entropy: {e}")
    
    def _trigger_emergency_response(self, event):
        """Trigger emergency response for critical events."""
        try:
            logger.error(f"🚨 EMERGENCY: {event.type.value} - Severity: {event.severity}")
            
            # Implement emergency response logic
            # This could include:
            # - Pausing trading operations
            # - Sending alerts
            # - Activating safety protocols
            
        except Exception as e:
            logger.error(f"❌ Error in emergency response: {e}")
    
    async def start(self):
        """Start the unified Schwabot integration system."""
        if self.is_running:
            logger.warning("Unified Schwabot Integration already running")
            return
        
        try:
            logger.info("🚀 Starting Unified Schwabot Integration...")
            
            # Initialize all components
            await self.initialize_components()
            
            # Start main integration loop
            self.is_running = True
            self.start_time = time.time()
            
            # Start the main integration task
            integration_task = asyncio.create_task(self._integration_loop())
            
            logger.info("✅ Unified Schwabot Integration started successfully")
            
            # Keep the system running
            await integration_task
            
        except Exception as e:
            logger.error(f"❌ Failed to start Unified Schwabot Integration: {e}")
            self.is_running = False
            raise
    
    async def stop(self):
        """Stop the unified Schwabot integration system."""
        try:
            logger.info("🛑 Stopping Unified Schwabot Integration...")
            
            self.is_running = False
            
            # Stop all components
            if self.data_layer:
                await self.data_layer.stop_data_feed()
            
            if self.entropy_api:
                self.entropy_api.stop()
            
            if self.ai_bridge:
                self.ai_bridge.stop()
            
            logger.info("✅ Unified Schwabot Integration stopped")
            
        except Exception as e:
            logger.error(f"❌ Error stopping integration: {e}")
    
    async def _integration_loop(self):
        """Main integration loop that coordinates all components."""
        while self.is_running:
            try:
                # Update metrics
                self.metrics['total_ticks'] += 1
                
                # Process FaultBus events
                if self.fault_bus:
                    await self.fault_bus.dispatch(severity_threshold=0.5)
                    self.metrics['fault_events_processed'] += 1
                
                # Update entropy calculations
                if self.entropy_api:
                    self.metrics['entropy_calculations'] += 1
                
                # Update hash commands executed
                if self.entropy_api:
                    self.metrics['hash_commands_executed'] += len(
                        [c for c in self.entropy_api.hash_commands.values() if c.executed_at]
                    )
                
                # Log system status periodically
                if self.metrics['total_ticks'] % 100 == 0:
                    self._log_system_status()
                
                # Sleep for the integration interval (3.75 minutes)
                await asyncio.sleep(225.0)
                
            except Exception as e:
                logger.error(f"❌ Error in integration loop: {e}")
                await asyncio.sleep(10)  # Brief pause on error
    
    def _log_system_status(self):
        """Log current system status."""
        try:
            uptime = time.time() - self.start_time if self.start_time else 0
            
            status = {
                'uptime_seconds': uptime,
                'integration_state': self.integration_state,
                'metrics': self.metrics,
                'entropy_value': self.entropy_api.current_entropy if self.entropy_api else 0,
                'active_bit_positions': len([p for p in self.entropy_api.bit_positions.values() if p.get('active', False)]) if self.entropy_api else 0,
                'ai_consensus_history_size': len(self.ai_bridge.consensus_history) if self.ai_bridge else 0
            }
            
            logger.info(f"📊 System Status: {status}")
            
        except Exception as e:
            logger.error(f"❌ Error logging system status: {e}")
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get system health information."""
        try:
            health = {
                'status': 'running' if self.is_running else 'stopped',
                'uptime_seconds': time.time() - self.start_time if self.start_time else 0,
                'components': self.integration_state,
                'metrics': self.metrics,
                'timestamp': datetime.now().isoformat()
            }
            
            # Add component-specific health info
            if self.entropy_api:
                health['entropy'] = {
                    'current_value': self.entropy_api.current_entropy,
                    'threshold': self.entropy_api.entropy_threshold,
                    'history_size': len(self.entropy_api.entropy_history),
                    'active_commands': len([c for c in self.entropy_api.hash_commands.values() if c.executed_at is None])
                }
            
            if self.ai_bridge:
                health['ai_bridge'] = {
                    'connected': self.ai_bridge.is_connected,
                    'consensus_history_size': len(self.ai_bridge.consensus_history),
                    'model_stats': self.ai_bridge.get_model_agreement_stats()
                }
            
            if self.fault_bus:
                health['fault_bus'] = {
                    'queue_size': len(self.fault_bus.queue),
                    'memory_log_size': len(self.fault_bus.memory_log),
                    'active_faults': len([e for e in self.fault_bus.memory_log if e.severity > 0.5])
                }
            
            return health
            
        except Exception as e:
            logger.error(f"❌ Error getting system health: {e}")
            return {'error': str(e)}
    
    def get_entropy_analytics(self) -> Dict[str, Any]:
        """Get entropy analytics."""
        try:
            if not self.entropy_api:
                return {'error': 'Entropy API not available'}
            
            analytics = {
                'current_entropy': self.entropy_api.current_entropy,
                'entropy_threshold': self.entropy_api.entropy_threshold,
                'entropy_history': list(self.entropy_api.entropy_history)[-50:],  # Last 50 entries
                'bit_positions': {
                    bit: {
                        'active': pos['active'],
                        'hash': pos['hash'][:8],
                        'last_updated': pos['last_updated'].isoformat()
                    }
                    for bit, pos in self.entropy_api.bit_positions.items()
                },
                'position_history_size': len(self.entropy_api.position_history),
                'hash_commands': {
                    cmd_id: {
                        'pattern': cmd.hash_pattern,
                        'function': cmd.execution_function,
                        'priority': cmd.priority,
                        'executed': cmd.executed_at is not None
                    }
                    for cmd_id, cmd in self.entropy_api.hash_commands.items()
                }
            }
            
            return analytics
            
        except Exception as e:
            logger.error(f"❌ Error getting entropy analytics: {e}")
            return {'error': str(e)}
    
    def get_ai_consensus_summary(self) -> Dict[str, Any]:
        """Get AI consensus summary."""
        try:
            if not self.ai_bridge:
                return {'error': 'AI Bridge not available'}
            
            consensus_history = self.ai_bridge.get_consensus_history(limit=20)
            model_stats = self.ai_bridge.get_model_agreement_stats()
            
            summary = {
                'recent_consensus': [
                    {
                        'consensus_action': c.consensus_action,
                        'confidence': c.consensus_confidence,
                        'agreement_level': c.agreement_level,
                        'risk_level': c.risk_level,
                        'timestamp': c.timestamp.isoformat()
                    }
                    for c in consensus_history
                ],
                'model_agreement_stats': model_stats,
                'total_consensus_count': len(self.ai_bridge.consensus_history)
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Error getting AI consensus summary: {e}")
            return {'error': str(e)}


# Example usage and configuration
def create_unified_schwabot_integration(config: Optional[Dict[str, Any]] = None):
    """Create and configure a unified Schwabot integration system."""
    integration = UnifiedSchwabotIntegration(config=config)
    return integration


async def main():
    """Main function to run the unified Schwabot integration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Create unified integration
        integration = create_unified_schwabot_integration()
        
        # Start the system
        await integration.start()
        
    except KeyboardInterrupt:
        logger.info("🛑 Received interrupt signal")
        if integration:
            await integration.stop()
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        if integration:
            await integration.stop()


if __name__ == "__main__":
    asyncio.run(main()) 