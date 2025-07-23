#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧬 ENHANCED FRACTAL API ROUTES - WEB INTERFACE FOR BEST TRADING SYSTEM ON EARTH
===============================================================================

Flask API routes for the Enhanced Forever Fractal System that provide:
- Real-time fractal analysis
- Trading recommendations
- System status monitoring
- Performance metrics
- Integration management

This provides web interface access to the BEST TRADING SYSTEM ON EARTH!
"""

from flask import Blueprint, jsonify, request, render_template
from flask_socketio import emit
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional

# Import the Enhanced Forever Fractal System
from fractals.enhanced_forever_fractal_system import (
    get_enhanced_forever_fractal_system,
    EnhancedForeverFractalSystem
)
from fractals.enhanced_fractal_integration import (
    get_enhanced_fractal_integration,
    start_enhanced_fractal_integration,
    process_market_data_through_enhanced_fractals,
    get_enhanced_fractal_analysis
)

logger = logging.getLogger(__name__)

# Create Blueprint for Enhanced Fractal routes
enhanced_fractal_bp = Blueprint('enhanced_fractal', __name__, url_prefix='/api/enhanced-fractal')

@enhanced_fractal_bp.route('/status', methods=['GET'])
def get_enhanced_fractal_status():
    """Get the status of the Enhanced Forever Fractal System."""
    try:
        # Get fractal system status
        fractal_system = get_enhanced_forever_fractal_system()
        fractal_status = fractal_system.get_system_status()
        
        # Get integration status
        integration = get_enhanced_fractal_integration()
        integration_status = integration.get_integration_status()
        
        # Combine status information
        status = {
            'system_name': 'Enhanced Forever Fractal System - BEST TRADING SYSTEM ON EARTH',
            'fractal_system': fractal_status,
            'integration': integration_status,
            'timestamp': datetime.now().isoformat(),
            'status': 'OPERATIONAL' if integration_status['is_integrated'] else 'INITIALIZING'
        }
        
        return jsonify(status)
        
    except Exception as e:
        logger.error(f"Error getting enhanced fractal status: {e}")
        return jsonify({
            'error': str(e),
            'status': 'ERROR',
            'timestamp': datetime.now().isoformat()
        }), 500

@enhanced_fractal_bp.route('/analysis', methods=['GET'])
def get_real_time_analysis():
    """Get real-time analysis from the Enhanced Forever Fractal System."""
    try:
        # Run async function in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        analysis = loop.run_until_complete(get_enhanced_fractal_analysis())
        loop.close()
        
        return jsonify(analysis)
        
    except Exception as e:
        logger.error(f"Error getting real-time analysis: {e}")
        return jsonify({
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@enhanced_fractal_bp.route('/trading-recommendation', methods=['GET'])
def get_trading_recommendation():
    """Get current trading recommendation from the Enhanced Forever Fractal System."""
    try:
        fractal_system = get_enhanced_forever_fractal_system()
        recommendation = fractal_system.get_trading_recommendation()
        
        return jsonify({
            'recommendation': recommendation,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting trading recommendation: {e}")
        return jsonify({
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@enhanced_fractal_bp.route('/process-market-data', methods=['POST'])
def process_market_data():
    """Process market data through the Enhanced Forever Fractal System."""
    try:
        # Get market data from request
        market_data = request.get_json()
        
        if not market_data:
            return jsonify({
                'error': 'No market data provided',
                'timestamp': datetime.now().isoformat()
            }), 400
        
        # Process market data through enhanced fractals
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(process_market_data_through_enhanced_fractals(market_data))
        loop.close()
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error processing market data: {e}")
        return jsonify({
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@enhanced_fractal_bp.route('/start-integration', methods=['POST'])
def start_integration():
    """Start the Enhanced Fractal Integration System."""
    try:
        # Get configuration from request
        config = request.get_json() or {}
        
        # Start integration
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        success = loop.run_until_complete(start_enhanced_fractal_integration(config))
        loop.close()
        
        if success:
            return jsonify({
                'message': 'Enhanced Fractal Integration started successfully',
                'status': 'SUCCESS',
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'error': 'Failed to start Enhanced Fractal Integration',
                'status': 'FAILED',
                'timestamp': datetime.now().isoformat()
            }), 500
        
    except Exception as e:
        logger.error(f"Error starting integration: {e}")
        return jsonify({
            'error': str(e),
            'status': 'ERROR',
            'timestamp': datetime.now().isoformat()
        }), 500

@enhanced_fractal_bp.route('/performance', methods=['GET'])
def get_performance_metrics():
    """Get performance metrics from the Enhanced Forever Fractal System."""
    try:
        # Get fractal system performance
        fractal_system = get_enhanced_forever_fractal_system()
        fractal_status = fractal_system.get_system_status()
        
        # Get integration performance
        integration = get_enhanced_fractal_integration()
        integration_status = integration.get_integration_status()
        
        # Combine performance metrics
        performance = {
            'fractal_performance': {
                'total_updates': fractal_status.get('total_updates', 0),
                'profit_generated': fractal_status.get('profit_generated', 0.0),
                'pattern_accuracy': fractal_status.get('pattern_accuracy', 0.0),
                'system_health': fractal_status.get('system_health', 'UNKNOWN')
            },
            'integration_performance': integration_status.get('performance_metrics', {}),
            'overall_performance': {
                'total_signals': integration_status.get('total_signals_processed', 0),
                'total_trades': integration_status.get('total_trades_executed', 0),
                'total_profit': integration_status.get('total_profit_generated', 0.0),
                'uptime': integration_status.get('performance_metrics', {}).get('system_uptime', 0.0)
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(performance)
        
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        return jsonify({
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@enhanced_fractal_bp.route('/bit-phases', methods=['GET'])
def get_bit_phase_analysis():
    """Get current bit phase analysis from the Enhanced Forever Fractal System."""
    try:
        fractal_system = get_enhanced_forever_fractal_system()
        current_state = fractal_system.current_state
        
        # Extract bit phase information
        bit_phases = []
        for phase in current_state.bit_phases:
            bit_phases.append({
                'pattern': phase.pattern,
                'phase_type': phase.phase_type.value,
                'confidence': phase.confidence,
                'profit_potential': phase.profit_potential,
                'market_alignment': phase.market_alignment,
                'mathematical_signature': phase.mathematical_signature,
                'timestamp': phase.timestamp.isoformat()
            })
        
        return jsonify({
            'bit_phases': bit_phases,
            'total_phases': len(bit_phases),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting bit phase analysis: {e}")
        return jsonify({
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@enhanced_fractal_bp.route('/fractal-sync', methods=['GET'])
def get_fractal_sync_status():
    """Get fractal synchronization status with Upstream Timing Protocol."""
    try:
        fractal_system = get_enhanced_forever_fractal_system()
        current_state = fractal_system.current_state
        fractal_sync = current_state.fractal_sync
        
        sync_status = {
            'sync_time': fractal_sync.sync_time,
            'alignment_score': fractal_sync.alignment_score,
            'node_performance': fractal_sync.node_performance,
            'fractal_resonance': fractal_sync.fractal_resonance,
            'upstream_priority': fractal_sync.upstream_priority,
            'execution_authority': fractal_sync.execution_authority,
            'status': 'SYNCED' if fractal_sync.execution_authority else 'DESYNCED',
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(sync_status)
        
    except Exception as e:
        logger.error(f"Error getting fractal sync status: {e}")
        return jsonify({
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@enhanced_fractal_bp.route('/dashboard', methods=['GET'])
def enhanced_fractal_dashboard():
    """Enhanced Fractal System Dashboard - Web interface for the BEST TRADING SYSTEM ON EARTH."""
    try:
        # Get comprehensive system data
        fractal_system = get_enhanced_forever_fractal_system()
        integration = get_enhanced_fractal_integration()
        
        # Get system status
        fractal_status = fractal_system.get_system_status()
        integration_status = integration.get_integration_status()
        trading_recommendation = fractal_system.get_trading_recommendation()
        
        # Prepare dashboard data
        dashboard_data = {
            'system_name': 'Enhanced Forever Fractal System - BEST TRADING SYSTEM ON EARTH',
            'fractal_status': fractal_status,
            'integration_status': integration_status,
            'trading_recommendation': trading_recommendation,
            'current_state': {
                'memory_shell': fractal_system.current_state.memory_shell,
                'entropy_anchor': fractal_system.current_state.entropy_anchor,
                'coherence': fractal_system.current_state.coherence,
                'profit_potential': fractal_system.current_state.profit_potential
            },
            'bit_phases': [
                {
                    'pattern': phase.pattern,
                    'phase_type': phase.phase_type.value,
                    'confidence': phase.confidence,
                    'profit_potential': phase.profit_potential
                }
                for phase in fractal_system.current_state.bit_phases
            ],
            'fractal_sync': {
                'alignment_score': fractal_system.current_state.fractal_sync.alignment_score,
                'node_performance': fractal_system.current_state.fractal_sync.node_performance,
                'execution_authority': fractal_system.current_state.fractal_sync.execution_authority
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(dashboard_data)
        
    except Exception as e:
        logger.error(f"Error getting dashboard data: {e}")
        return jsonify({
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@enhanced_fractal_bp.route('/config', methods=['GET', 'POST'])
def manage_configuration():
    """Get or update Enhanced Forever Fractal System configuration."""
    try:
        fractal_system = get_enhanced_forever_fractal_system()
        
        if request.method == 'GET':
            # Return current configuration
            return jsonify({
                'configuration': fractal_system.config,
                'timestamp': datetime.now().isoformat()
            })
        
        elif request.method == 'POST':
            # Update configuration
            new_config = request.get_json()
            
            if not new_config:
                return jsonify({
                    'error': 'No configuration provided',
                    'timestamp': datetime.now().isoformat()
                }), 400
            
            # Update configuration
            fractal_system.config.update(new_config)
            
            return jsonify({
                'message': 'Configuration updated successfully',
                'configuration': fractal_system.config,
                'timestamp': datetime.now().isoformat()
            })
        
    except Exception as e:
        logger.error(f"Error managing configuration: {e}")
        return jsonify({
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@enhanced_fractal_bp.route('/health', methods=['GET'])
def health_check():
    """Health check for the Enhanced Forever Fractal System."""
    try:
        fractal_system = get_enhanced_forever_fractal_system()
        integration = get_enhanced_fractal_integration()
        
        # Check system health
        fractal_status = fractal_system.get_system_status()
        integration_status = integration.get_integration_status()
        
        # Determine overall health
        fractal_health = fractal_status.get('system_health', 'UNKNOWN')
        integration_health = 'HEALTHY' if integration_status['is_integrated'] else 'UNHEALTHY'
        
        overall_health = 'HEALTHY' if fractal_health == 'EXCELLENT' and integration_health == 'HEALTHY' else 'DEGRADED'
        
        return jsonify({
            'status': 'OK',
            'overall_health': overall_health,
            'fractal_health': fractal_health,
            'integration_health': integration_health,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        return jsonify({
            'status': 'ERROR',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

# WebSocket events for real-time updates
def register_enhanced_fractal_socketio(socketio):
    """Register WebSocket events for real-time Enhanced Fractal updates."""
    
    @socketio.on('connect', namespace='/enhanced-fractal')
    def handle_enhanced_fractal_connect():
        """Handle client connection to Enhanced Fractal WebSocket."""
        logger.info("Client connected to Enhanced Fractal WebSocket")
        emit('connected', {'message': 'Connected to Enhanced Forever Fractal System'})
    
    @socketio.on('disconnect', namespace='/enhanced-fractal')
    def handle_enhanced_fractal_disconnect():
        """Handle client disconnection from Enhanced Fractal WebSocket."""
        logger.info("Client disconnected from Enhanced Fractal WebSocket")
    
    @socketio.on('request_analysis', namespace='/enhanced-fractal')
    def handle_analysis_request():
        """Handle real-time analysis request."""
        try:
            # Get real-time analysis
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            analysis = loop.run_until_complete(get_enhanced_fractal_analysis())
            loop.close()
            
            emit('analysis_update', analysis)
            
        except Exception as e:
            logger.error(f"Error handling analysis request: {e}")
            emit('error', {'error': str(e)})
    
    @socketio.on('request_status', namespace='/enhanced-fractal')
    def handle_status_request():
        """Handle status request."""
        try:
            fractal_system = get_enhanced_forever_fractal_system()
            integration = get_enhanced_fractal_integration()
            
            status = {
                'fractal_status': fractal_system.get_system_status(),
                'integration_status': integration.get_integration_status(),
                'timestamp': datetime.now().isoformat()
            }
            
            emit('status_update', status)
            
        except Exception as e:
            logger.error(f"Error handling status request: {e}")
            emit('error', {'error': str(e)})

# Register the blueprint
def register_enhanced_fractal_routes(app):
    """Register Enhanced Fractal routes with the Flask app."""
    app.register_blueprint(enhanced_fractal_bp)
    logger.info("✅ Enhanced Fractal API routes registered") 