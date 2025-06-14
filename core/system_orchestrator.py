"""
System Orchestrator
==================

Orchestrates the complete thermal-aware, profit-synchronized hash trigger system.
This module integrates all components and provides a unified interface for
running the complete trading system from A to ZBE.
"""

import logging
import threading
import time
import signal
import sys
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timezone
from dataclasses import dataclass
from .memory_map import get_memory_map
from .memory_agent import MemoryAgent
from .profit_trajectory_coprocessor import ProfitTrajectoryCoprocessor
from .thermal_zone_manager import ThermalZoneManager
from .hash_trigger_engine import HashTriggerEngine
from .dormant_engine import DormantEngine
from .collapse_engine import CollapseEngine
from .cursor_engine import CursorState

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SystemState:
    """Container for overall system state"""
    is_running: bool
    uptime_seconds: float
    total_ticks_processed: int
    total_strategies_executed: int
    current_thermal_zone: str
    current_profit_zone: str
    active_hash_triggers: int
    active_memory_agents: int
    system_health: str
    last_updated: datetime

class SystemOrchestrator:
    """
    Orchestrates the complete trading system, integrating all components
    into a cohesive thermal-aware, profit-synchronized operation.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the system orchestrator
        
        Args:
            config: System configuration dictionary
        """
        self.config = config or self._default_config()
        self.start_time = datetime.now(timezone.utc)
        self.is_running = False
        self.shutdown_requested = False
        
        # Core components
        self.memory_map = get_memory_map()
        self.profit_coprocessor = ProfitTrajectoryCoprocessor(
            window_size=self.config.get('profit_window_size', 10000)
        )
        self.thermal_manager = ThermalZoneManager(self.profit_coprocessor)
        
        # Hash trigger system
        self.dormant_engine = DormantEngine()
        self.collapse_engine = CollapseEngine()
        self.hash_trigger_engine = HashTriggerEngine(
            self.dormant_engine, self.collapse_engine
        )
        
        # Memory agents
        self.memory_agents: Dict[str, MemoryAgent] = {}
        self.default_agent_id = self.config.get('default_agent_id', 'main_agent')
        
        # Statistics
        self.stats = {
            'ticks_processed': 0,
            'strategies_executed': 0,
            'thermal_bursts': 0,
            'hash_triggers_fired': 0,
            'successful_trades': 0,
            'total_profit': 0.0
        }
        
        # Threading
        self._lock = threading.RLock()
        self._monitoring_thread: Optional[threading.Thread] = None
        
        # Initialize components
        self._initialize_system()
        
    def _default_config(self) -> Dict[str, Any]:
        """Get default system configuration"""
        return {
            'profit_window_size': 10000,
            'thermal_monitoring_interval': 30.0,
            'system_monitoring_interval': 60.0,
            'default_agent_id': 'main_agent',
            'auto_register_triggers': True,
            'enable_thermal_management': True,
            'enable_profit_tracking': True,
            'log_level': 'INFO',
            'tick_processing_timeout': 5.0,
            'strategy_execution_timeout': 30.0
        }
        
    def _initialize_system(self) -> None:
        """Initialize all system components"""
        try:
            # Set up thermal monitoring
            if self.config.get('enable_thermal_management', True):
                self.thermal_manager.start_monitoring(
                    interval=self.config.get('thermal_monitoring_interval', 30.0)
                )
                logger.info("Thermal monitoring initialized")
                
            # Create default memory agent
            self._get_or_create_agent(self.default_agent_id)
            
            # Register default hash triggers if enabled
            if self.config.get('auto_register_triggers', True):
                self._register_default_triggers()
                
            # Set up signal handlers for graceful shutdown
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            
            logger.info("System orchestrator initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing system: {e}")
            raise
            
    def _register_default_triggers(self) -> None:
        """Register some default hash triggers for testing"""
        default_triggers = [
            {
                'trigger_id': 'PROFIT_SURGE_001',
                'price_map': {100.0: 105.0, 200.0: 210.0},
                'euler_phase': 45.0
            },
            {
                'trigger_id': 'THERMAL_OPTIMAL_001', 
                'price_map': {150.0: 155.0, 250.0: 260.0},
                'euler_phase': 90.0
            },
            {
                'trigger_id': 'MOMENTUM_ALIGN_001',
                'price_map': {75.0: 78.0, 175.0: 182.0},
                'euler_phase': 135.0
            }
        ]
        
        for trigger_config in default_triggers:
            self.hash_trigger_engine.register_trigger(
                trigger_config['trigger_id'],
                trigger_config['price_map'],
                trigger_config['euler_phase']
            )
            
        logger.info(f"Registered {len(default_triggers)} default hash triggers")
        
    def _get_or_create_agent(self, agent_id: str) -> MemoryAgent:
        """Get existing agent or create new one"""
        with self._lock:
            if agent_id not in self.memory_agents:
                self.memory_agents[agent_id] = MemoryAgent(
                    agent_id=agent_id,
                    memory_map=self.memory_map,
                    profit_coprocessor=self.profit_coprocessor,
                    thermal_manager=self.thermal_manager
                )
                logger.info(f"Created memory agent: {agent_id}")
            return self.memory_agents[agent_id]
            
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown()
        
    def start(self) -> None:
        """Start the complete trading system"""
        if self.is_running:
            logger.warning("System is already running")
            return
            
        logger.info("Starting thermal-aware hash trading system...")
        
        with self._lock:
            self.is_running = True
            self.shutdown_requested = False
            
        # Start monitoring thread
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self._monitoring_thread.start()
        
        logger.info("System started successfully")
        
    def shutdown(self) -> None:
        """Gracefully shutdown the system"""
        logger.info("Initiating system shutdown...")
        
        with self._lock:
            self.shutdown_requested = True
            self.is_running = False
            
        # Stop thermal monitoring
        if self.thermal_manager:
            self.thermal_manager.stop_monitoring()
            
        # Wait for monitoring thread to finish
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=10.0)
            
        # Clean up memory agents
        for agent in self.memory_agents.values():
            agent.cleanup_old_data(days_to_keep=30)
            
        logger.info("System shutdown complete")
        
    def _monitoring_loop(self) -> None:
        """Main system monitoring loop"""
        interval = self.config.get('system_monitoring_interval', 60.0)
        
        while self.is_running and not self.shutdown_requested:
            try:
                self._update_system_health()
                self._log_system_statistics()
                
                # Clean up old data periodically
                if int(time.time()) % 3600 == 0:  # Every hour
                    self._cleanup_old_data()
                    
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(interval)
                
    def _update_system_health(self) -> None:
        """Update system health metrics"""
        try:
            # Check component health
            thermal_healthy = (self.thermal_manager and 
                             self.thermal_manager.current_state is not None)
            profit_healthy = (self.profit_coprocessor and 
                            self.profit_coprocessor.last_vector is not None)
            
            # Update memory map with current state
            if thermal_healthy:
                thermal_data = {
                    'zone': self.thermal_manager.current_state.zone.value,
                    'cpu_temp': self.thermal_manager.current_state.cpu_temp,
                    'gpu_temp': self.thermal_manager.current_state.gpu_temp,
                    'drift_coefficient': self.thermal_manager.current_state.drift_coefficient
                }
                self.memory_map.add_thermal_state(thermal_data)
                
            if profit_healthy:
                trajectory_data = {
                    'zone_state': self.profit_coprocessor.last_vector.zone_state.value,
                    'vector_strength': self.profit_coprocessor.last_vector.vector_strength,
                    'slope': self.profit_coprocessor.last_vector.slope,
                    'confidence': self.profit_coprocessor.last_vector.confidence
                }
                self.memory_map.add_profit_trajectory(trajectory_data)
                
        except Exception as e:
            logger.error(f"Error updating system health: {e}")
            
    def _log_system_statistics(self) -> None:
        """Log current system statistics"""
        uptime = (datetime.now(timezone.utc) - self.start_time).total_seconds()
        
        logger.info(f"System Statistics - Uptime: {uptime:.0f}s, "
                   f"Ticks: {self.stats['ticks_processed']}, "
                   f"Strategies: {self.stats['strategies_executed']}, "
                   f"Profit: ${self.stats['total_profit']:.2f}")
                   
    def _cleanup_old_data(self) -> None:
        """Clean up old data from memory systems"""
        try:
            # Clean memory map
            self.memory_map.clear_old_data(days_to_keep=30)
            
            # Clean agent data
            for agent in self.memory_agents.values():
                agent.cleanup_old_data(days_to_keep=30)
                
            # Clear hash trigger history
            self.hash_trigger_engine.clear_history()
            
            logger.info("Completed periodic data cleanup")
            
        except Exception as e:
            logger.error(f"Error during data cleanup: {e}")
            
    def process_tick(self, price: float, volume: float = 0.0, 
                    timestamp: Optional[datetime] = None,
                    metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Process a market tick through the complete system pipeline
        
        Args:
            price: Current market price
            volume: Trade volume (optional)
            timestamp: Tick timestamp
            metadata: Additional tick metadata
            
        Returns:
            Processing results including triggered strategies
        """
        if not self.is_running:
            raise RuntimeError("System is not running")
            
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
            
        results = {
            'timestamp': timestamp.isoformat(),
            'price': price,
            'volume': volume,
            'triggered_strategies': [],
            'thermal_state': None,
            'profit_state': None,
            'hash_triggers': [],
            'processing_recommendations': {}
        }
        
        try:
            with self._lock:
                # Update profit trajectory
                if self.config.get('enable_profit_tracking', True):
                    # For profit tracking, we need actual P&L data
                    # Here we'll use price change as a proxy
                    if hasattr(self, '_last_price'):
                        profit_delta = price - self._last_price
                        vector = self.profit_coprocessor.update(profit_delta, timestamp)
                        results['profit_state'] = {
                            'zone_state': vector.zone_state.value,
                            'vector_strength': vector.vector_strength,
                            'confidence': vector.confidence
                        }
                    self._last_price = price
                    
                # Get thermal state
                if self.thermal_manager.current_state:
                    results['thermal_state'] = {
                        'zone': self.thermal_manager.current_state.zone.value,
                        'cpu_temp': self.thermal_manager.current_state.cpu_temp,
                        'gpu_temp': self.thermal_manager.current_state.gpu_temp,
                        'drift_coefficient': self.thermal_manager.current_state.drift_coefficient
                    }
                    results['processing_recommendations'] = \
                        self.thermal_manager.current_state.processing_recommendation
                        
                # Create cursor state for hash processing
                cursor_state = CursorState(
                    triplet=(price, price * 1.001, price * 0.999),  # Simple triplet
                    delta_idx=1,
                    braid_angle=0.0,
                    timestamp=timestamp.timestamp()
                )
                
                # Generate hash from price (simple implementation)
                price_hash = hash(str(price)) & 0xFFFF  # 16-bit hash
                
                # Process hash triggers
                triggered_ids = self.hash_trigger_engine.process_hash(price_hash, cursor_state)
                results['hash_triggers'] = triggered_ids
                
                # Execute strategies for triggered hashes
                if triggered_ids:
                    strategy_results = self._execute_triggered_strategies(
                        triggered_ids, price, timestamp
                    )
                    results['triggered_strategies'] = strategy_results
                    
                # Update statistics
                self.stats['ticks_processed'] += 1
                self.stats['hash_triggers_fired'] += len(triggered_ids)
                
            return results
            
        except Exception as e:
            logger.error(f"Error processing tick: {e}")
            results['error'] = str(e)
            return results
            
    def _execute_triggered_strategies(self, triggered_ids: List[str], 
                                    price: float, timestamp: datetime) -> List[Dict]:
        """Execute strategies for triggered hash IDs"""
        strategy_results = []
        
        for trigger_id in triggered_ids:
            try:
                # Get confidence from default agent
                agent = self._get_or_create_agent(self.default_agent_id)
                
                # Build current context
                context = {}
                if self.thermal_manager.current_state:
                    context['thermal_state'] = self.thermal_manager.current_state.zone.value
                if self.profit_coprocessor.last_vector:
                    context['profit_zone'] = self.profit_coprocessor.last_vector.zone_state.value
                    
                confidence = agent.get_confidence_coefficient(
                    trigger_id, [trigger_id], context
                )
                
                # Only execute if confidence is above threshold
                if confidence > 0.6:  # Configurable threshold
                    execution_id = agent.start_strategy_execution(
                        strategy_id=trigger_id,
                        hash_triggers=[trigger_id],
                        entry_price=price,
                        initial_confidence=confidence
                    )
                    
                    strategy_results.append({
                        'trigger_id': trigger_id,
                        'execution_id': execution_id,
                        'confidence': confidence,
                        'entry_price': price,
                        'timestamp': timestamp.isoformat()
                    })
                    
                    self.stats['strategies_executed'] += 1
                    
            except Exception as e:
                logger.error(f"Error executing strategy for trigger {trigger_id}: {e}")
                strategy_results.append({
                    'trigger_id': trigger_id,
                    'error': str(e)
                })
                
        return strategy_results
        
    def complete_strategy(self, execution_id: str, exit_price: float,
                         agent_id: Optional[str] = None,
                         metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Complete a strategy execution
        
        Args:
            execution_id: Execution ID from process_tick
            exit_price: Exit price for the trade
            agent_id: Agent ID (defaults to main agent)
            metadata: Additional execution metadata
            
        Returns:
            Completion results
        """
        if agent_id is None:
            agent_id = self.default_agent_id
            
        try:
            agent = self.memory_agents.get(agent_id)
            if not agent:
                return {'error': f'Agent {agent_id} not found'}
                
            # Calculate execution time (simplified)
            execution_time = 30.0  # Default execution time
            
            agent.complete_strategy_execution(
                execution_id, exit_price, execution_time, metadata
            )
            
            # Update statistics
            profit_loss = exit_price - 100.0  # Simplified calculation
            if profit_loss > 0:
                self.stats['successful_trades'] += 1
            self.stats['total_profit'] += profit_loss
            
            return {
                'success': True,
                'execution_id': execution_id,
                'profit_loss': profit_loss,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error completing strategy: {e}")
            return {'error': str(e)}
            
    def get_system_state(self) -> SystemState:
        """Get current system state"""
        uptime = (datetime.now(timezone.utc) - self.start_time).total_seconds()
        
        thermal_zone = "unknown"
        if self.thermal_manager.current_state:
            thermal_zone = self.thermal_manager.current_state.zone.value
            
        profit_zone = "unknown"
        if self.profit_coprocessor.last_vector:
            profit_zone = self.profit_coprocessor.last_vector.zone_state.value
            
        # Determine system health
        health = "healthy"
        if not self.is_running:
            health = "stopped"
        elif self.thermal_manager.should_reduce_gpu_load():
            health = "thermal_warning"
        elif uptime > 86400:  # 24 hours
            health = "needs_restart"
            
        return SystemState(
            is_running=self.is_running,
            uptime_seconds=uptime,
            total_ticks_processed=self.stats['ticks_processed'],
            total_strategies_executed=self.stats['strategies_executed'],
            current_thermal_zone=thermal_zone,
            current_profit_zone=profit_zone,
            active_hash_triggers=len(self.hash_trigger_engine.get_active_triggers()),
            active_memory_agents=len(self.memory_agents),
            system_health=health,
            last_updated=datetime.now(timezone.utc)
        )
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        state = self.get_system_state()
        
        return {
            'system_state': {
                'is_running': state.is_running,
                'uptime_seconds': state.uptime_seconds,
                'system_health': state.system_health
            },
            'processing_stats': dict(self.stats),
            'component_stats': {
                'thermal_manager': self.thermal_manager.get_statistics() if self.thermal_manager else {},
                'profit_coprocessor': self.profit_coprocessor.get_statistics() if self.profit_coprocessor else {},
                'memory_map': self.memory_map.get_stats(),
                'hash_triggers': {
                    'total_registered': len(self.hash_trigger_engine.triggers),
                    'currently_active': len(self.hash_trigger_engine.get_active_triggers())
                }
            },
            'agent_stats': {
                agent_id: agent.get_agent_statistics()
                for agent_id, agent in self.memory_agents.items()
            }
        }

# Example usage and testing
if __name__ == "__main__":
    # Create system orchestrator
    orchestrator = SystemOrchestrator()
    
    try:
        # Start the system
        orchestrator.start()
        
        print("System started. Processing sample ticks...")
        
        # Process some sample ticks
        for i in range(10):
            price = 100.0 + i + (i % 3) * 0.5  # Simple price movement
            result = orchestrator.process_tick(price)
            
            print(f"Tick {i+1}: Price={price:.2f}, "
                  f"Triggers={len(result['hash_triggers'])}, "
                  f"Strategies={len(result['triggered_strategies'])}")
                  
            time.sleep(1)
            
        # Get final statistics
        stats = orchestrator.get_statistics()
        print("\nFinal System Statistics:")
        print(f"  Ticks processed: {stats['processing_stats']['ticks_processed']}")
        print(f"  Strategies executed: {stats['processing_stats']['strategies_executed']}")
        print(f"  Total profit: ${stats['processing_stats']['total_profit']:.2f}")
        print(f"  System health: {stats['system_state']['system_health']}")
        
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    finally:
        orchestrator.shutdown()
        print("System shutdown complete") 