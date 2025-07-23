#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 UPSTREAM TIMING ROUTES - FLASK INTEGRATION
============================================

Flask routes for the Upstream Timing Protocol.
Integrates with existing Schwabot Flask infrastructure.
"""

from flask import Blueprint, jsonify, request
from core.upstream_timing_protocol import UpstreamTimingProtocol

# Create Flask blueprint
upstream_timing = Blueprint('upstream_timing', __name__)

# Global protocol instance
protocol_instance = None

def init_upstream_timing_protocol(flask_app, socketio):
    """Initialize the Upstream Timing Protocol."""
    global protocol_instance
    protocol_instance = UpstreamTimingProtocol(flask_app, socketio)
    protocol_instance.start_monitoring()
    return protocol_instance

@upstream_timing.route('/status', methods=['GET'])
def get_status():
    """Get upstream timing status."""
    if protocol_instance:
        return jsonify({
            'status': 'active',
            'primary_executor': protocol_instance.primary_executor,
            'total_nodes': len(protocol_instance.nodes),
            'online_nodes': len([n for n in protocol_instance.nodes.values() if n.status == 'online'])
        })
    else:
        return jsonify({'status': 'inactive'})

@upstream_timing.route('/nodes', methods=['GET'])
def get_nodes():
    """Get all registered nodes."""
    if protocol_instance:
        from core.upstream_timing_protocol import asdict
        return jsonify([asdict(node) for node in protocol_instance.nodes.values()])
    else:
        return jsonify([])

@upstream_timing.route('/trade/execute', methods=['POST'])
def execute_trade():
    """Execute trade on optimal node."""
    if not protocol_instance:
        return jsonify({'status': 'error', 'message': 'Protocol not initialized'}), 503
    
    try:
        data = request.json
        strategy_hash = data.get('strategy_hash')
        trade_data = data.get('trade_data', {})
        
        # Select optimal node
        optimal_node = protocol_instance._select_optimal_node()
        if not optimal_node:
            return jsonify({
                'status': 'error',
                'message': 'No optimal node available'
            }), 503
        
        # Create trade execution
        from core.upstream_timing_protocol import TradeExecution
        import time
        
        trade_exec = TradeExecution(
            trade_id=f"trade_{int(time.time())}_{strategy_hash[:8]}",
            strategy_hash=strategy_hash,
            target_node=optimal_node,
            execution_time=time.time()
        )
        
        # Add to queue
        protocol_instance.trade_queue.put(trade_exec)
        
        return jsonify({
            'status': 'success',
            'trade_id': trade_exec.trade_id,
            'target_node': optimal_node,
            'execution_time': trade_exec.execution_time
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400 