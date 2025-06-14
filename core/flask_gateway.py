"""
Flask Gateway Application
========================

Provides HTTP API gateway for the Memory Agent system, allowing external
systems to interact with thermal management, profit trajectory processing,
and strategy memory through RESTful endpoints.
"""

from flask import Flask, request, jsonify, g
from flask_cors import CORS
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
import json
import traceback

from .memory_agent import MemoryAgent, StrategyState
from .memory_map import get_memory_map
from .profit_trajectory_coprocessor import ProfitTrajectoryCoprocessor
from .thermal_zone_manager import ThermalZoneManager
from .hash_trigger_engine import HashTriggerEngine

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global components
app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Component instances
profit_coprocessor: Optional[ProfitTrajectoryCoprocessor] = None
thermal_manager: Optional[ThermalZoneManager] = None
memory_agents: Dict[str, MemoryAgent] = {}
hash_trigger_engine: Optional[HashTriggerEngine] = None

# Monitoring thread
monitoring_thread: Optional[threading.Thread] = None
monitoring_active = False

def initialize_components():
    """Initialize all system components"""
    global profit_coprocessor, thermal_manager, hash_trigger_engine
    
    try:
        # Initialize profit trajectory coprocessor
        profit_coprocessor = ProfitTrajectoryCoprocessor(window_size=10000)
        logger.info("Initialized ProfitTrajectoryCoprocessor")
        
        # Initialize thermal zone manager
        thermal_manager = ThermalZoneManager(profit_coprocessor)
        thermal_manager.start_monitoring(interval=30.0)  # 30-second intervals
        logger.info("Initialized ThermalZoneManager")
        
        # Initialize hash trigger engine (placeholder for integration)
        from .dormant_engine import DormantEngine
        from .collapse_engine import CollapseEngine
        
        dormant_engine = DormantEngine()
        collapse_engine = CollapseEngine()
        hash_trigger_engine = HashTriggerEngine(dormant_engine, collapse_engine)
        logger.info("Initialized HashTriggerEngine")
        
    except Exception as e:
        logger.error(f"Error initializing components: {e}")
        logger.error(traceback.format_exc())

def get_or_create_agent(agent_id: str) -> MemoryAgent:
    """Get existing agent or create new one"""
    if agent_id not in memory_agents:
        memory_agents[agent_id] = MemoryAgent(
            agent_id=agent_id,
            profit_coprocessor=profit_coprocessor,
            thermal_manager=thermal_manager
        )
        logger.info(f"Created new memory agent: {agent_id}")
    return memory_agents[agent_id]

def start_background_monitoring():
    """Start background monitoring thread"""
    global monitoring_thread, monitoring_active
    
    if monitoring_active:
        return
        
    monitoring_active = True
    monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
    monitoring_thread.start()
    logger.info("Started background monitoring")

def monitoring_loop():
    """Background monitoring loop"""
    while monitoring_active:
        try:
            # Update profit trajectory with dummy data (replace with real data source)
            if profit_coprocessor:
                # In real implementation, this would come from your trading system
                pass
                
            # Monitor system health
            memory_map = get_memory_map()
            stats = memory_map.get_stats()
            
            if stats.get("memory_map_size_mb", 0) > 100:  # 100MB threshold
                logger.warning("Memory map size exceeded 100MB, cleaning old data")
                memory_map.clear_old_data(days_to_keep=7)
                
            time.sleep(60)  # Check every minute
            
        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")
            time.sleep(60)

# Error handlers
@app.errorhandler(Exception)
def handle_exception(e):
    """Handle all exceptions with proper JSON response"""
    logger.error(f"Unhandled exception: {e}")
    logger.error(traceback.format_exc())
    return jsonify({
        'error': 'Internal server error',
        'message': str(e),
        'timestamp': datetime.now(timezone.utc).isoformat()
    }), 500

@app.errorhandler(404)
def handle_not_found(e):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Not found',
        'message': 'The requested endpoint does not exist',
        'timestamp': datetime.now(timezone.utc).isoformat()
    }), 404

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    """System health check"""
    try:
        memory_map = get_memory_map()
        
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'components': {
                'profit_coprocessor': profit_coprocessor is not None,
                'thermal_manager': thermal_manager is not None,
                'hash_trigger_engine': hash_trigger_engine is not None,
                'memory_map': True,
                'active_agents': len(memory_agents)
            },
            'memory_stats': memory_map.get_stats()
        }
        
        if thermal_manager and thermal_manager.current_state:
            health_status['thermal_state'] = {
                'cpu_temp': thermal_manager.current_state.cpu_temp,
                'gpu_temp': thermal_manager.current_state.gpu_temp,
                'zone': thermal_manager.current_state.zone.value
            }
            
        if profit_coprocessor and profit_coprocessor.last_vector:
            health_status['profit_state'] = {
                'zone_state': profit_coprocessor.last_vector.zone_state.value,
                'vector_strength': profit_coprocessor.last_vector.vector_strength,
                'confidence': profit_coprocessor.last_vector.confidence
            }
            
        return jsonify(health_status)
        
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 500

# Profit trajectory endpoints
@app.route('/profit/update', methods=['POST'])
def update_profit():
    """Update profit trajectory with new data point"""
    try:
        data = request.get_json()
        profit = data.get('profit')
        tick_time = data.get('timestamp')
        
        if profit is None:
            return jsonify({'error': 'profit value required'}), 400
            
        if tick_time:
            tick_time = datetime.fromisoformat(tick_time.replace('Z', '+00:00'))
            
        vector = profit_coprocessor.update(profit, tick_time)
        
        return jsonify({
            'success': True,
            'vector': {
                'slope': vector.slope,
                'vector_strength': vector.vector_strength,
                'zone_state': vector.zone_state.value,
                'confidence': vector.confidence,
                'momentum': vector.momentum
            },
            'timestamp': vector.timestamp.isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error updating profit: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/profit/status', methods=['GET'])
def get_profit_status():
    """Get current profit trajectory status"""
    try:
        if not profit_coprocessor or not profit_coprocessor.last_vector:
            return jsonify({'error': 'No profit data available'}), 404
            
        vector = profit_coprocessor.last_vector
        stats = profit_coprocessor.get_statistics()
        
        return jsonify({
            'current_vector': {
                'slope': vector.slope,
                'vector_strength': vector.vector_strength,
                'zone_state': vector.zone_state.value,
                'confidence': vector.confidence,
                'momentum': vector.momentum,
                'timestamp': vector.timestamp.isoformat()
            },
            'statistics': stats,
            'processing_allocation': profit_coprocessor.get_processing_allocation()
        })
        
    except Exception as e:
        logger.error(f"Error getting profit status: {e}")
        return jsonify({'error': str(e)}), 500

# Thermal management endpoints
@app.route('/thermal/status', methods=['GET'])
def get_thermal_status():
    """Get current thermal status"""
    try:
        if not thermal_manager or not thermal_manager.current_state:
            return jsonify({'error': 'No thermal data available'}), 404
            
        state = thermal_manager.current_state
        stats = thermal_manager.get_statistics()
        
        return jsonify({
            'current_state': {
                'cpu_temp': state.cpu_temp,
                'gpu_temp': state.gpu_temp,
                'zone': state.zone.value,
                'cpu_load': state.load_cpu,
                'gpu_load': state.load_gpu,
                'memory_usage': state.memory_usage,
                'drift_coefficient': state.drift_coefficient,
                'timestamp': state.timestamp.isoformat()
            },
            'processing_recommendation': state.processing_recommendation,
            'statistics': stats
        })
        
    except Exception as e:
        logger.error(f"Error getting thermal status: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/thermal/burst/start', methods=['POST'])
def start_thermal_burst():
    """Start a thermal processing burst"""
    try:
        success = thermal_manager.start_burst()
        
        return jsonify({
            'success': success,
            'message': 'Burst started' if success else 'Burst denied',
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error starting burst: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/thermal/burst/end', methods=['POST'])
def end_thermal_burst():
    """End a thermal processing burst"""
    try:
        data = request.get_json()
        duration = data.get('duration', 0.0)
        
        thermal_manager.end_burst(duration)
        
        return jsonify({
            'success': True,
            'message': f'Burst ended after {duration:.1f}s',
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error ending burst: {e}")
        return jsonify({'error': str(e)}), 500

# Memory agent endpoints
@app.route('/agent/<agent_id>/strategy/start', methods=['POST'])
def start_strategy_execution(agent_id: str):
    """Start a new strategy execution"""
    try:
        data = request.get_json()
        strategy_id = data.get('strategy_id')
        hash_triggers = data.get('hash_triggers', [])
        entry_price = data.get('entry_price')
        initial_confidence = data.get('initial_confidence', 0.5)
        
        if not strategy_id or entry_price is None:
            return jsonify({'error': 'strategy_id and entry_price required'}), 400
            
        agent = get_or_create_agent(agent_id)
        execution_id = agent.start_strategy_execution(
            strategy_id, hash_triggers, entry_price, initial_confidence
        )
        
        return jsonify({
            'success': True,
            'execution_id': execution_id,
            'agent_id': agent_id,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error starting strategy execution: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/agent/<agent_id>/strategy/complete', methods=['POST'])
def complete_strategy_execution(agent_id: str):
    """Complete a strategy execution"""
    try:
        data = request.get_json()
        execution_id = data.get('execution_id')
        exit_price = data.get('exit_price')
        execution_time = data.get('execution_time', 0.0)
        metadata = data.get('metadata', {})
        
        if not execution_id or exit_price is None:
            return jsonify({'error': 'execution_id and exit_price required'}), 400
            
        if agent_id not in memory_agents:
            return jsonify({'error': 'Agent not found'}), 404
            
        agent = memory_agents[agent_id]
        agent.complete_strategy_execution(execution_id, exit_price, execution_time, metadata)
        
        return jsonify({
            'success': True,
            'execution_id': execution_id,
            'agent_id': agent_id,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error completing strategy execution: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/agent/<agent_id>/confidence', methods=['GET'])
def get_confidence_coefficient(agent_id: str):
    """Get confidence coefficient for a strategy"""
    try:
        strategy_id = request.args.get('strategy_id')
        hash_triggers = request.args.getlist('hash_triggers')
        
        if not strategy_id:
            return jsonify({'error': 'strategy_id required'}), 400
            
        if agent_id not in memory_agents:
            return jsonify({'error': 'Agent not found'}), 404
            
        # Build current context
        context = {}
        if thermal_manager and thermal_manager.current_state:
            context['thermal_state'] = thermal_manager.current_state.zone.value
        if profit_coprocessor and profit_coprocessor.last_vector:
            context['profit_zone'] = profit_coprocessor.last_vector.zone_state.value
            
        agent = memory_agents[agent_id]
        confidence = agent.get_confidence_coefficient(
            strategy_id, hash_triggers or None, context or None
        )
        
        return jsonify({
            'confidence_coefficient': confidence,
            'strategy_id': strategy_id,
            'agent_id': agent_id,
            'context': context,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting confidence coefficient: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/agent/<agent_id>/statistics', methods=['GET'])
def get_agent_statistics(agent_id: str):
    """Get agent statistics"""
    try:
        if agent_id not in memory_agents:
            return jsonify({'error': 'Agent not found'}), 404
            
        agent = memory_agents[agent_id]
        stats = agent.get_agent_statistics()
        
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"Error getting agent statistics: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/agent/<agent_id>/strategy/<strategy_id>/performance', methods=['GET'])
def get_strategy_performance(agent_id: str, strategy_id: str):
    """Get strategy performance statistics"""
    try:
        if agent_id not in memory_agents:
            return jsonify({'error': 'Agent not found'}), 404
            
        agent = memory_agents[agent_id]
        performance = agent.get_strategy_performance(strategy_id)
        
        return jsonify(performance)
        
    except Exception as e:
        logger.error(f"Error getting strategy performance: {e}")
        return jsonify({'error': str(e)}), 500

# Memory map endpoints
@app.route('/memory/stats', methods=['GET'])
def get_memory_stats():
    """Get memory map statistics"""
    try:
        memory_map = get_memory_map()
        stats = memory_map.get_stats()
        
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"Error getting memory stats: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/memory/successes/<strategy_id>', methods=['GET'])
def get_strategy_successes(strategy_id: str):
    """Get strategy success history"""
    try:
        limit = request.args.get('limit', type=int)
        
        memory_map = get_memory_map()
        successes = memory_map.get_strategy_successes(strategy_id, limit)
        
        return jsonify({
            'strategy_id': strategy_id,
            'successes': successes,
            'count': len(successes)
        })
        
    except Exception as e:
        logger.error(f"Error getting strategy successes: {e}")
        return jsonify({'error': str(e)}), 500

# Hash trigger engine endpoints
@app.route('/hash/register', methods=['POST'])
def register_hash_trigger():
    """Register a new hash trigger"""
    try:
        data = request.get_json()
        trigger_id = data.get('trigger_id')
        price_map = data.get('price_map', {})
        euler_phase = data.get('euler_phase', 0.0)
        
        if not trigger_id:
            return jsonify({'error': 'trigger_id required'}), 400
            
        hash_trigger_engine.register_trigger(trigger_id, price_map, euler_phase)
        
        return jsonify({
            'success': True,
            'trigger_id': trigger_id,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error registering hash trigger: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/hash/process', methods=['POST'])
def process_hash():
    """Process a hash value and check for trigger activations"""
    try:
        data = request.get_json()
        hash_value = data.get('hash_value')
        cursor_state_data = data.get('cursor_state', {})
        
        if hash_value is None:
            return jsonify({'error': 'hash_value required'}), 400
            
        # Create cursor state from data
        from .cursor_engine import CursorState
        cursor_state = CursorState(
            triplet=tuple(cursor_state_data.get('triplet', [0.0, 0.0, 0.0])),
            delta_idx=cursor_state_data.get('delta_idx', 0),
            braid_angle=cursor_state_data.get('braid_angle', 0.0),
            timestamp=cursor_state_data.get('timestamp', time.time())
        )
        
        triggered_ids = hash_trigger_engine.process_hash(hash_value, cursor_state)
        
        return jsonify({
            'triggered_ids': triggered_ids,
            'hash_value': hash_value,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error processing hash: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/hash/active', methods=['GET'])
def get_active_triggers():
    """Get currently active hash triggers"""
    try:
        active_triggers = hash_trigger_engine.get_active_triggers()
        
        return jsonify({
            'active_triggers': [
                {
                    'trigger_id': trigger.trigger_id,
                    'hash_value': trigger.hash_value,
                    'is_active': trigger.is_active,
                    'last_triggered': trigger.last_triggered,
                    'euler_phase': trigger.euler_phase
                }
                for trigger in active_triggers
            ],
            'count': len(active_triggers)
        })
        
    except Exception as e:
        logger.error(f"Error getting active triggers: {e}")
        return jsonify({'error': str(e)}), 500

# System control endpoints
@app.route('/system/reset', methods=['POST'])
def reset_system():
    """Reset system components"""
    try:
        component = request.args.get('component', 'all')
        
        if component in ['all', 'profit']:
            if profit_coprocessor:
                profit_coprocessor.reset()
                
        if component in ['all', 'thermal']:
            if thermal_manager:
                thermal_manager.reset_daily_budget()
                
        if component in ['all', 'agents']:
            memory_agents.clear()
            
        return jsonify({
            'success': True,
            'component': component,
            'message': f'Reset {component} components',
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error resetting system: {e}")
        return jsonify({'error': str(e)}), 500

# Initialize and start
def create_app():
    """Create and configure Flask app"""
    initialize_components()
    start_background_monitoring()
    return app

# For running standalone
if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True) 