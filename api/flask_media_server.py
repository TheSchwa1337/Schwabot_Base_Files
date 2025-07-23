#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flask Media Server - Schwabot Real-time Context Streaming
=========================================================

Provides real-time context streaming to AI models through Flask endpoints,
similar to how Cursor provides context to LLMs. Enables AI models to understand
trading context, tensor math results, and system state in real-time.

Features:
- Real-time context streaming
- AI model context ingestion
- Trading data context
- Tensor math context understanding
- System health monitoring
- Context search and indexing
- WebSocket support for real-time updates
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict

from flask import Flask, request, jsonify, Response, stream_template
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room, leave_room
import threading

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.distributed_system.distributed_node_manager import get_distributed_manager
from core.distributed_system.real_time_context_ingestion import get_context_ingestion
from core.distributed_system.ai_integration_bridge import get_ai_bridge

logger = logging.getLogger(__name__)

@dataclass
class ContextStream:
    """Real-time context stream for AI consumption."""
    stream_id: str
    context_type: str
    data: Any
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    subscribers: Set[str] = field(default_factory=set)

class FlaskMediaServer:
    """Flask media server for real-time context streaming."""
    
    def __init__(self):
        self.app = Flask(__name__)
        CORS(self.app)
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Context streaming
        self.context_streams: Dict[str, ContextStream] = {}
        self.ai_subscribers: Dict[str, Set[str]] = defaultdict(set)
        self.context_index: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Real-time data
        self.latest_context = {}
        self.context_history = []
        self.is_running = False
        
        # Setup routes
        self._setup_routes()
        self._setup_socketio_events()
        
        logger.info("Initialized FlaskMediaServer")
    
    def _setup_routes(self):
        """Setup Flask routes for context streaming."""
        
        @self.app.route('/api/context/stream', methods=['GET'])
        def stream_context():
            """Stream real-time context data."""
            def generate():
                while self.is_running:
                    # Get latest context data
                    context_data = self._get_latest_context()
                    yield f"data: {json.dumps(context_data)}\n\n"
                    time.sleep(1)  # Update every second
            
            return Response(generate(), mimetype='text/event-stream')
        
        @self.app.route('/api/context/latest', methods=['GET'])
        def get_latest_context():
            """Get latest context data."""
            context_type = request.args.get('type', 'all')
            limit = int(request.args.get('limit', 100))
            
            context_data = self._get_latest_context(context_type, limit)
            return jsonify(context_data)
        
        @self.app.route('/api/context/search', methods=['POST'])
        def search_context():
            """Search context data (like grep)."""
            data = request.get_json()
            query = data.get('query', '')
            context_type = data.get('type', 'all')
            limit = int(data.get('limit', 50))
            
            results = self._search_context(query, context_type, limit)
            return jsonify(results)
        
        @self.app.route('/api/context/ingest', methods=['POST'])
        def ingest_context():
            """Ingest new context data."""
            data = request.get_json()
            context_type = data.get('type', 'unknown')
            context_data = data.get('data', {})
            metadata = data.get('metadata', {})
            
            stream_id = self._ingest_context(context_type, context_data, metadata)
            return jsonify({"stream_id": stream_id, "status": "ingested"})
        
        @self.app.route('/api/context/ai/request', methods=['POST'])
        def request_ai_context():
            """Request context data for AI consumption."""
            data = request.get_json()
            ai_model = data.get('model', 'unknown')
            context_types = data.get('context_types', ['all'])
            limit = int(data.get('limit', 200))
            
            ai_context = self._get_ai_context(ai_model, context_types, limit)
            return jsonify(ai_context)
        
        @self.app.route('/api/context/trading', methods=['GET'])
        def get_trading_context():
            """Get trading-specific context."""
            symbol = request.args.get('symbol')
            timeframe = request.args.get('timeframe', '1m')
            limit = int(request.args.get('limit', 100))
            
            trading_context = self._get_trading_context(symbol, timeframe, limit)
            return jsonify(trading_context)
        
        @self.app.route('/api/context/tensor', methods=['GET'])
        def get_tensor_context():
            """Get tensor math context."""
            calculation_id = request.args.get('calculation_id')
            limit = int(request.args.get('limit', 50))
            
            tensor_context = self._get_tensor_context(calculation_id, limit)
            return jsonify(tensor_context)
        
        @self.app.route('/api/context/system', methods=['GET'])
        def get_system_context():
            """Get system health context."""
            limit = int(request.args.get('limit', 50))
            
            system_context = self._get_system_context(limit)
            return jsonify(system_context)
        
        @self.app.route('/api/context/subscribe', methods=['POST'])
        def subscribe_to_context():
            """Subscribe to context updates."""
            data = request.get_json()
            subscriber_id = data.get('subscriber_id')
            context_types = data.get('context_types', ['all'])
            
            self._subscribe_to_context(subscriber_id, context_types)
            return jsonify({"status": "subscribed"})
        
        @self.app.route('/api/context/unsubscribe', methods=['POST'])
        def unsubscribe_from_context():
            """Unsubscribe from context updates."""
            data = request.get_json()
            subscriber_id = data.get('subscriber_id')
            
            self._unsubscribe_from_context(subscriber_id)
            return jsonify({"status": "unsubscribed"})
        
        @self.app.route('/api/context/status', methods=['GET'])
        def get_context_status():
            """Get context server status."""
            status = {
                "is_running": self.is_running,
                "total_streams": len(self.context_streams),
                "total_subscribers": len(self.ai_subscribers),
                "context_types": list(self.context_index.keys()),
                "latest_update": time.time()
            }
            return jsonify(status)
        
        @self.app.route('/api/context/clear', methods=['POST'])
        def clear_context():
            """Clear old context data."""
            data = request.get_json()
            older_than = data.get('older_than', 3600)  # Default 1 hour
            
            cleared_count = self._clear_old_context(older_than)
            return jsonify({"cleared_count": cleared_count})
    
    def _setup_socketio_events(self):
        """Setup SocketIO events for real-time communication."""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection."""
            client_id = request.sid
            logger.info(f"Client connected: {client_id}")
            emit('connected', {'client_id': client_id})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection."""
            client_id = request.sid
            logger.info(f"Client disconnected: {client_id}")
            self._unsubscribe_from_context(client_id)
        
        @self.socketio.on('subscribe')
        def handle_subscribe(data):
            """Handle context subscription."""
            client_id = request.sid
            context_types = data.get('context_types', ['all'])
            
            self._subscribe_to_context(client_id, context_types)
            emit('subscribed', {'context_types': context_types})
        
        @self.socketio.on('unsubscribe')
        def handle_unsubscribe(data):
            """Handle context unsubscription."""
            client_id = request.sid
            self._unsubscribe_from_context(client_id)
            emit('unsubscribed')
        
        @self.socketio.on('request_context')
        def handle_request_context(data):
            """Handle context request."""
            client_id = request.sid
            context_types = data.get('context_types', ['all'])
            limit = data.get('limit', 100)
            
            context_data = self._get_ai_context(client_id, context_types, limit)
            emit('context_data', context_data)
    
    def _get_latest_context(self, context_type: str = 'all', limit: int = 100) -> Dict[str, Any]:
        """Get latest context data."""
        if context_type == 'all':
            # Get latest from all types
            latest = {}
            for ctx_type, context_list in self.context_index.items():
                if context_list:
                    latest[ctx_type] = context_list[-limit:]
        else:
            # Get latest from specific type
            context_list = self.context_index.get(context_type, [])
            latest = {context_type: context_list[-limit:]}
        
        return {
            "timestamp": time.time(),
            "context_type": context_type,
            "data": latest,
            "total_items": sum(len(items) for items in latest.values())
        }
    
    def _search_context(self, query: str, context_type: str = 'all', limit: int = 50) -> List[Dict[str, Any]]:
        """Search context data (like grep functionality)."""
        results = []
        query_lower = query.lower()
        
        # Search in context index
        search_types = [context_type] if context_type != 'all' else self.context_index.keys()
        
        for ctx_type in search_types:
            context_list = self.context_index.get(ctx_type, [])
            
            for context_item in context_list:
                # Search in data and metadata
                searchable_text = json.dumps(context_item).lower()
                
                if query_lower in searchable_text:
                    results.append({
                        "context_type": ctx_type,
                        "timestamp": context_item.get("timestamp"),
                        "data": context_item,
                        "match_score": self._calculate_match_score(query_lower, searchable_text)
                    })
        
        # Sort by match score and timestamp
        results.sort(key=lambda x: (x["match_score"], x["timestamp"]), reverse=True)
        
        return results[:limit]
    
    def _calculate_match_score(self, query: str, text: str) -> float:
        """Calculate match score for search results."""
        if query in text:
            return 1.0
        elif any(word in text for word in query.split()):
            return 0.5
        else:
            return 0.1
    
    def _ingest_context(self, context_type: str, data: Any, metadata: Dict[str, Any]) -> str:
        """Ingest new context data."""
        stream_id = f"{context_type}_{int(time.time() * 1000)}"
        
        # Create context stream
        context_stream = ContextStream(
            stream_id=stream_id,
            context_type=context_type,
            data=data,
            timestamp=time.time(),
            metadata=metadata
        )
        
        # Store in streams
        self.context_streams[stream_id] = context_stream
        
        # Add to index
        context_item = {
            "stream_id": stream_id,
            "context_type": context_type,
            "data": data,
            "timestamp": time.time(),
            "metadata": metadata
        }
        
        self.context_index[context_type].append(context_item)
        
        # Keep index manageable
        if len(self.context_index[context_type]) > 1000:
            self.context_index[context_type] = self.context_index[context_type][-1000:]
        
        # Update latest context
        self.latest_context[context_type] = context_item
        
        # Add to history
        self.context_history.append(context_item)
        if len(self.context_history) > 5000:
            self.context_history = self.context_history[-5000:]
        
        # Notify subscribers
        self._notify_subscribers(context_type, context_item)
        
        logger.debug(f"Ingested context: {context_type} - {stream_id}")
        return stream_id
    
    def _get_ai_context(self, ai_model: str, context_types: List[str], limit: int) -> Dict[str, Any]:
        """Get context data formatted for AI consumption."""
        ai_context = {
            "ai_model": ai_model,
            "timestamp": time.time(),
            "context_data": {},
            "summary": {}
        }
        
        # Get context for each type
        for context_type in context_types:
            if context_type == 'all':
                # Get all types
                for ctx_type, context_list in self.context_index.items():
                    ai_context["context_data"][ctx_type] = context_list[-limit:]
            else:
                # Get specific type
                context_list = self.context_index.get(context_type, [])
                ai_context["context_data"][context_type] = context_list[-limit:]
        
        # Add summary
        for ctx_type, context_list in ai_context["context_data"].items():
            ai_context["summary"][ctx_type] = {
                "count": len(context_list),
                "latest_timestamp": context_list[-1]["timestamp"] if context_list else None,
                "data_types": list(set(item.get("metadata", {}).get("data_type", "unknown") 
                                     for item in context_list))
            }
        
        return ai_context
    
    def _get_trading_context(self, symbol: str = None, timeframe: str = '1m', limit: int = 100) -> Dict[str, Any]:
        """Get trading-specific context."""
        trading_context = {
            "symbol": symbol,
            "timeframe": timeframe,
            "timestamp": time.time(),
            "data": []
        }
        
        # Get trading data from context index
        trading_data = self.context_index.get('trading_data', [])
        
        for item in trading_data[-limit:]:
            item_data = item.get("data", {})
            
            # Filter by symbol if specified
            if symbol and item_data.get("symbol") != symbol:
                continue
            
            trading_context["data"].append(item)
        
        return trading_context
    
    def _get_tensor_context(self, calculation_id: str = None, limit: int = 50) -> Dict[str, Any]:
        """Get tensor math context."""
        tensor_context = {
            "calculation_id": calculation_id,
            "timestamp": time.time(),
            "data": []
        }
        
        # Get tensor data from context index
        tensor_data = self.context_index.get('tensor_math', [])
        
        for item in tensor_data[-limit:]:
            item_data = item.get("data", {})
            
            # Filter by calculation ID if specified
            if calculation_id and item_data.get("calculation_id") != calculation_id:
                continue
            
            tensor_context["data"].append(item)
        
        return tensor_context
    
    def _get_system_context(self, limit: int = 50) -> Dict[str, Any]:
        """Get system health context."""
        system_context = {
            "timestamp": time.time(),
            "data": self.context_index.get('system_health', [])[-limit:]
        }
        
        return system_context
    
    def _subscribe_to_context(self, subscriber_id: str, context_types: List[str]):
        """Subscribe to context updates."""
        for context_type in context_types:
            self.ai_subscribers[context_type].add(subscriber_id)
        
        logger.debug(f"Subscriber {subscriber_id} subscribed to {context_types}")
    
    def _unsubscribe_from_context(self, subscriber_id: str):
        """Unsubscribe from context updates."""
        for context_type in self.ai_subscribers:
            self.ai_subscribers[context_type].discard(subscriber_id)
        
        logger.debug(f"Subscriber {subscriber_id} unsubscribed")
    
    def _notify_subscribers(self, context_type: str, context_item: Dict[str, Any]):
        """Notify subscribers of new context data."""
        subscribers = self.ai_subscribers.get(context_type, set())
        all_subscribers = self.ai_subscribers.get('all', set())
        
        all_notify = subscribers | all_subscribers
        
        # Emit via SocketIO
        for subscriber_id in all_notify:
            try:
                self.socketio.emit('context_update', {
                    "context_type": context_type,
                    "data": context_item
                }, room=subscriber_id)
            except Exception as e:
                logger.error(f"Error notifying subscriber {subscriber_id}: {e}")
    
    def _clear_old_context(self, older_than: int) -> int:
        """Clear old context data."""
        cutoff_time = time.time() - older_than
        cleared_count = 0
        
        # Clear from context index
        for context_type in self.context_index:
            original_count = len(self.context_index[context_type])
            self.context_index[context_type] = [
                item for item in self.context_index[context_type]
                if item.get("timestamp", 0) > cutoff_time
            ]
            cleared_count += original_count - len(self.context_index[context_type])
        
        # Clear from history
        original_history_count = len(self.context_history)
        self.context_history = [
            item for item in self.context_history
            if item.get("timestamp", 0) > cutoff_time
        ]
        cleared_count += original_history_count - len(self.context_history)
        
        # Clear old streams
        old_streams = [
            stream_id for stream_id, stream in self.context_streams.items()
            if stream.timestamp < cutoff_time
        ]
        for stream_id in old_streams:
            del self.context_streams[stream_id]
        
        logger.info(f"Cleared {cleared_count} old context items")
        return cleared_count
    
    async def start(self, host: str = '0.0.0.0', port: int = 5001):
        """Start the Flask media server."""
        logger.info(f"Starting Flask media server on {host}:{port}")
        
        self.is_running = True
        
        # Start background tasks
        asyncio.create_task(self._context_cleanup_loop())
        asyncio.create_task(self._health_monitoring_loop())
        
        # Start Flask server in a separate thread
        def run_flask():
            self.socketio.run(self.app, host=host, port=port, debug=False)
        
        flask_thread = threading.Thread(target=run_flask, daemon=True)
        flask_thread.start()
        
        logger.info(f"Flask media server started on {host}:{port}")
    
    async def stop(self):
        """Stop the Flask media server."""
        logger.info("Stopping Flask media server...")
        
        self.is_running = False
        
        # Clear all context
        self.context_streams.clear()
        self.context_index.clear()
        self.context_history.clear()
        self.ai_subscribers.clear()
        
        logger.info("Flask media server stopped")
    
    async def _context_cleanup_loop(self):
        """Clean up old context data periodically."""
        while self.is_running:
            try:
                # Clear context older than 1 hour
                self._clear_old_context(3600)
                await asyncio.sleep(300)  # Clean up every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in context cleanup: {e}")
                await asyncio.sleep(300)
    
    async def _health_monitoring_loop(self):
        """Monitor server health."""
        while self.is_running:
            try:
                # Log health metrics
                total_context = sum(len(items) for items in self.context_index.values())
                total_subscribers = len(self.ai_subscribers)
                
                logger.debug(f"Health check - Context items: {total_context}, Subscribers: {total_subscribers}")
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in health monitoring: {e}")
                await asyncio.sleep(60)
    
    def get_status(self) -> Dict[str, Any]:
        """Get server status."""
        return {
            "is_running": self.is_running,
            "total_streams": len(self.context_streams),
            "total_subscribers": len(self.ai_subscribers),
            "context_types": list(self.context_index.keys()),
            "context_counts": {ctx_type: len(items) for ctx_type, items in self.context_index.items()},
            "latest_update": time.time()
        }

# Global instance
_media_server: Optional[FlaskMediaServer] = None

def get_media_server() -> FlaskMediaServer:
    """Get the global media server instance."""
    global _media_server
    if _media_server is None:
        _media_server = FlaskMediaServer()
    return _media_server

async def start_media_server(host: str = '0.0.0.0', port: int = 5001):
    """Start the media server."""
    server = get_media_server()
    await server.start(host, port)
    return server

if __name__ == "__main__":
    # Test the media server
    async def test():
        server = get_media_server()
        await server.start()
        
        # Test with sample data
        server._ingest_context("trading_data", {
            "symbol": "BTC/USD",
            "price": 50000.0,
            "volume": 1000.0
        }, {"source": "test"})
        
        # Keep running
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            await server.stop()
    
    asyncio.run(test()) 