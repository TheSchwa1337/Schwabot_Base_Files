#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 UPSTREAM TIMING PROTOCOL - SCHWABOT NODE OPTIMIZATION
========================================================

Implements the Upstream Timing Protocol for optimal trade execution
across multiple nodes in the Schwabot distributed system.

Features:
- Real-time node performance monitoring
- Automatic trade node selection
- Flask-based node coordination
- Hardware-aware optimization
- Forever Fractal synchronization
"""

import asyncio
import json
import logging
import time
import threading
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from queue import Queue

import psutil
import requests
from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit

logger = logging.getLogger(__name__)

class NodeRole(Enum):
    """Node roles in the Upstream Timing Protocol."""
    PRIMARY_EXECUTOR = "primary_executor"
    STRATEGY_VALIDATOR = "strategy_validator"
    BACKTEST_ECHO = "backtest_echo"
    PATTERN_ANALYZER = "pattern_analyzer"
    FALLBACK = "fallback"

@dataclass
class NodePerformance:
    """Node performance metrics for timing optimization."""
    node_id: str
    latency: float  # Flask response time in ms
    tick_sync: float  # Market data sync time in ms
    cpu_load: float  # CPU usage percentage
    memory_usage: float  # Memory usage percentage
    gpu_usage: float  # GPU usage percentage
    gpu_memory: float  # GPU memory usage percentage
    network_latency: float  # Network latency in ms
    fractal_sync: float  # Forever Fractal sync time in ms
    last_update: float
    status: str = "online"
    role: NodeRole = NodeRole.FALLBACK
    performance_score: float = 0.0

@dataclass
class TradeExecution:
    """Trade execution data with node assignment."""
    trade_id: str
    strategy_hash: str
    target_node: str
    execution_time: float
    status: str = "pending"
    result: Dict[str, Any] = None

class UpstreamTimingProtocol:
    """Main class for Upstream Timing Protocol implementation."""
    
    def __init__(self, flask_app: Flask, socketio: SocketIO):
        """Initialize the Upstream Timing Protocol."""
        self.flask_app = flask_app
        self.socketio = socketio
        
        # Node management
        self.nodes: Dict[str, NodePerformance] = {}
        self.primary_executor: Optional[str] = None
        self.node_history: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.performance_thresholds = {
            'max_latency': 100.0,  # ms
            'max_tick_sync': 10.0,  # ms
            'max_cpu_load': 80.0,   # %
            'max_memory_usage': 85.0,  # %
            'max_gpu_usage': 90.0,  # %
            'min_performance_score': 50.0
        }
        
        # Trade execution queue
        self.trade_queue: Queue = Queue()
        self.execution_history: List[TradeExecution] = []
        
        # Threading
        self.running = False
        self.monitoring_thread = None
        self.execution_thread = None
        
        # Setup Flask routes
        self.setup_flask_routes()
        
        logger.info("🧠 Upstream Timing Protocol initialized")
    
    def setup_flask_routes(self):
        """Setup Flask routes for node management."""
        
        @self.flask_app.route('/api/upstream/status', methods=['GET'])
        def get_upstream_status():
            """Get current upstream timing status."""
            return jsonify({
                'primary_executor': self.primary_executor,
                'total_nodes': len(self.nodes),
                'online_nodes': len([n for n in self.nodes.values() if n.status == 'online']),
                'performance_thresholds': self.performance_thresholds,
                'last_update': time.time()
            })
        
        @self.flask_app.route('/api/upstream/nodes', methods=['GET'])
        def get_nodes():
            """Get all registered nodes with performance data."""
            return jsonify([asdict(node) for node in self.nodes.values()])
        
        @self.flask_app.route('/api/upstream/nodes/register', methods=['POST'])
        def register_node():
            """Register a new node with performance metrics."""
            try:
                data = request.json
                node_id = data['node_id']
                
                # Create node performance object
                node_perf = NodePerformance(
                    node_id=node_id,
                    latency=data.get('latency', 0.0),
                    tick_sync=data.get('tick_sync', 0.0),
                    cpu_load=data.get('cpu_load', 0.0),
                    memory_usage=data.get('memory_usage', 0.0),
                    gpu_usage=data.get('gpu_usage', 0.0),
                    gpu_memory=data.get('gpu_memory', 0.0),
                    network_latency=data.get('network_latency', 0.0),
                    fractal_sync=data.get('fractal_sync', 0.0),
                    last_update=time.time(),
                    status='online'
                )
                
                # Calculate performance score
                node_perf.performance_score = self._calculate_performance_score(node_perf)
                
                # Register node
                self.nodes[node_id] = node_perf
                
                # Update primary executor if needed
                self._update_primary_executor()
                
                logger.info(f"Node registered: {node_id} (Score: {node_perf.performance_score:.2f})")
                
                # Broadcast node update
                self.socketio.emit('node_registered', {
                    'node_id': node_id,
                    'performance_score': node_perf.performance_score,
                    'role': node_perf.role.value
                })
                
                return jsonify({
                    'status': 'success',
                    'node_id': node_id,
                    'performance_score': node_perf.performance_score,
                    'role': node_perf.role.value
                })
                
            except Exception as e:
                logger.error(f"Node registration error: {e}")
                return jsonify({'status': 'error', 'message': str(e)}), 400
        
        @self.flask_app.route('/api/upstream/nodes/<node_id>/update', methods=['POST'])
        def update_node_performance(node_id):
            """Update node performance metrics."""
            try:
                if node_id not in self.nodes:
                    return jsonify({'status': 'error', 'message': 'Node not found'}), 404
                
                data = request.json
                node = self.nodes[node_id]
                
                # Update metrics
                node.latency = data.get('latency', node.latency)
                node.tick_sync = data.get('tick_sync', node.tick_sync)
                node.cpu_load = data.get('cpu_load', node.cpu_load)
                node.memory_usage = data.get('memory_usage', node.memory_usage)
                node.gpu_usage = data.get('gpu_usage', node.gpu_usage)
                node.gpu_memory = data.get('gpu_memory', node.gpu_memory)
                node.network_latency = data.get('network_latency', node.network_latency)
                node.fractal_sync = data.get('fractal_sync', node.fractal_sync)
                node.last_update = time.time()
                node.status = data.get('status', 'online')
                
                # Recalculate performance score
                node.performance_score = self._calculate_performance_score(node)
                
                # Update primary executor if needed
                self._update_primary_executor()
                
                # Broadcast performance update
                self.socketio.emit('node_performance_updated', {
                    'node_id': node_id,
                    'performance_score': node.performance_score,
                    'role': node.role.value,
                    'latency': node.latency,
                    'tick_sync': node.tick_sync
                })
                
                return jsonify({'status': 'success'})
                
            except Exception as e:
                logger.error(f"Node update error: {e}")
                return jsonify({'status': 'error', 'message': str(e)}), 400
        
        @self.flask_app.route('/api/upstream/trade/execute', methods=['POST'])
        def execute_trade():
            """Execute trade on optimal node."""
            try:
                data = request.json
                strategy_hash = data.get('strategy_hash')
                trade_data = data.get('trade_data', {})
                
                # Select optimal node for execution
                optimal_node = self._select_optimal_node()
                if not optimal_node:
                    return jsonify({
                        'status': 'error',
                        'message': 'No optimal node available for trade execution'
                    }), 503
                
                # Create trade execution
                trade_exec = TradeExecution(
                    trade_id=f"trade_{int(time.time())}_{strategy_hash[:8]}",
                    strategy_hash=strategy_hash,
                    target_node=optimal_node,
                    execution_time=time.time()
                )
                
                # Add to execution queue
                self.trade_queue.put(trade_exec)
                
                logger.info(f"Trade queued for execution on {optimal_node}")
                
                return jsonify({
                    'status': 'success',
                    'trade_id': trade_exec.trade_id,
                    'target_node': optimal_node,
                    'execution_time': trade_exec.execution_time
                })
                
            except Exception as e:
                logger.error(f"Trade execution error: {e}")
                return jsonify({'status': 'error', 'message': str(e)}), 400
        
        @self.flask_app.route('/api/upstream/trade/status/<trade_id>', methods=['GET'])
        def get_trade_status(trade_id):
            """Get trade execution status."""
            for execution in self.execution_history:
                if execution.trade_id == trade_id:
                    return jsonify(asdict(execution))
            
            return jsonify({'status': 'error', 'message': 'Trade not found'}), 404
    
    def _calculate_performance_score(self, node: NodePerformance) -> float:
        """Calculate performance score for node selection."""
        score = 100.0
        
        # Latency penalty (lower is better)
        if node.latency > self.performance_thresholds['max_latency']:
            score -= (node.latency - self.performance_thresholds['max_latency']) * 2
        
        # Tick sync penalty (lower is better)
        if node.tick_sync > self.performance_thresholds['max_tick_sync']:
            score -= (node.tick_sync - self.performance_thresholds['max_tick_sync']) * 5
        
        # CPU load penalty (lower is better)
        if node.cpu_load > self.performance_thresholds['max_cpu_load']:
            score -= (node.cpu_load - self.performance_thresholds['max_cpu_load']) * 0.5
        
        # Memory usage penalty (lower is better)
        if node.memory_usage > self.performance_thresholds['max_memory_usage']:
            score -= (node.memory_usage - self.performance_thresholds['max_memory_usage']) * 0.5
        
        # GPU usage penalty (lower is better)
        if node.gpu_usage > self.performance_thresholds['max_gpu_usage']:
            score -= (node.gpu_usage - self.performance_thresholds['max_gpu_usage']) * 0.3
        
        # Network latency penalty (lower is better)
        if node.network_latency > 50:  # 50ms threshold
            score -= (node.network_latency - 50) * 0.2
        
        # Fractal sync bonus (lower is better)
        if node.fractal_sync < 5:  # Excellent fractal sync
            score += 20
        elif node.fractal_sync < 10:  # Good fractal sync
            score += 10
        
        # Ensure score is within bounds
        return max(0.0, min(100.0, score))
    
    def _select_optimal_node(self) -> Optional[str]:
        """Select the optimal node for trade execution."""
        if not self.nodes:
            return None
        
        # Filter healthy nodes
        healthy_nodes = [
            node for node in self.nodes.values()
            if (node.status == 'online' and 
                node.performance_score >= self.performance_thresholds['min_performance_score'])
        ]
        
        if not healthy_nodes:
            return None
        
        # Sort by performance score (highest first)
        sorted_nodes = sorted(healthy_nodes, key=lambda x: x.performance_score, reverse=True)
        
        return sorted_nodes[0].node_id
    
    def _update_primary_executor(self):
        """Update the primary executor node."""
        optimal_node = self._select_optimal_node()
        
        if optimal_node != self.primary_executor:
            old_executor = self.primary_executor
            self.primary_executor = optimal_node
            
            # Update node roles
            for node in self.nodes.values():
                if node.node_id == optimal_node:
                    node.role = NodeRole.PRIMARY_EXECUTOR
                elif node.node_id == old_executor:
                    node.role = NodeRole.STRATEGY_VALIDATOR
                else:
                    # Assign roles based on performance
                    if node.performance_score >= 70:
                        node.role = NodeRole.STRATEGY_VALIDATOR
                    elif node.performance_score >= 40:
                        node.role = NodeRole.PATTERN_ANALYZER
                    else:
                        node.role = NodeRole.BACKTEST_ECHO
            
            logger.info(f"Primary executor changed: {old_executor} -> {optimal_node}")
            
            # Broadcast role change
            self.socketio.emit('primary_executor_changed', {
                'old_executor': old_executor,
                'new_executor': optimal_node
            })
    
    def start_monitoring(self):
        """Start the monitoring thread."""
        def monitoring_loop():
            while self.running:
                try:
                    # Clean up stale nodes
                    current_time = time.time()
                    stale_nodes = []
                    
                    for node_id, node in self.nodes.items():
                        if current_time - node.last_update > 60:  # 60 second timeout
                            stale_nodes.append(node_id)
                    
                    for node_id in stale_nodes:
                        del self.nodes[node_id]
                        logger.warning(f"Removed stale node: {node_id}")
                    
                    # Update primary executor
                    self._update_primary_executor()
                    
                    # Broadcast system status
                    self.socketio.emit('upstream_status_update', {
                        'primary_executor': self.primary_executor,
                        'total_nodes': len(self.nodes),
                        'online_nodes': len([n for n in self.nodes.values() if n.status == 'online'])
                    })
                    
                    time.sleep(10)  # Update every 10 seconds
                    
                except Exception as e:
                    logger.error(f"Monitoring error: {e}")
                    time.sleep(10)
        
        self.running = True
        self.monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Upstream Timing Protocol monitoring started")
    
    def stop_monitoring(self):
        """Stop the monitoring thread."""
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        logger.info("Upstream Timing Protocol monitoring stopped") 