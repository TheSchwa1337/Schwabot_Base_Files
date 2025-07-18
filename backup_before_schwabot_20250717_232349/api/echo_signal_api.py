#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Echo Signal API - Flask Webhook Endpoints for External Signal Ingestion
=======================================================================

Provides REST API endpoints for receiving external echo signals from:
- Twitter webhooks
- News API callbacks
- Reddit webhooks
- Manual signal injection
- Social sentiment feeds

Endpoints:
- POST /api/echo - Main signal ingestion endpoint
- GET /api/echo/status - Signal processing status
- POST /api/echo/batch - Batch signal processing
- GET /api/echo/metrics - Processing metrics
"""

import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from flask import Flask, request, jsonify, Response
from werkzeug.exceptions import BadRequest, Unauthorized

# Import core dependencies
try:
    from core.exo_echo_signals import (
        EXOEchoSignals, SignalSource, SignalIntent, exo_echo_signals
    )
    from core.lantern_core import LanternCore
    from core.strategy_mapper import StrategyMapper
    EXO_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Core dependencies not available: {e}")
    EXO_AVAILABLE = False

logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Configuration
API_CONFIG = {
    'secret_key': 'your-secret-key-here',  # Change in production
    'rate_limit_per_minute': 100,
    'max_content_length': 16 * 1024 * 1024,  # 16MB
    'enable_cors': True,
    'debug': False
}

# Rate limiting
request_timestamps = []

# Initialize core components
if EXO_AVAILABLE:
    exo_processor = exo_echo_signals
    lantern_core = LanternCore()
    strategy_mapper = StrategyMapper()
else:
    exo_processor = None
    lantern_core = None
    strategy_mapper = None


def check_rate_limit() -> bool:
    """
    Check if request is within rate limits.
    
    Returns:
        True if within limits, False otherwise
    """
    global request_timestamps
    current_time = time.time()
    
    # Remove timestamps older than 1 minute
    request_timestamps = [ts for ts in request_timestamps if current_time - ts < 60]
    
    # Check if we're over the limit
    if len(request_timestamps) >= API_CONFIG['rate_limit_per_minute']:
        return False
    
    request_timestamps.append(current_time)
    return True


def validate_signal_data(data: Dict[str, Any]) -> bool:
    """
    Validate incoming signal data.
    
    Args:
        data: Signal data dictionary
        
    Returns:
        True if valid, False otherwise
    """
    required_fields = ['content', 'source']
    
    for field in required_fields:
        if field not in data:
            return False
    
    # Validate source
    try:
        SignalSource(data['source'])
    except ValueError:
        return False
    
    # Validate content length
    if len(data['content']) > 10000:  # Max 10KB content
        return False
    
    return True


def authenticate_request() -> bool:
    """
    Authenticate incoming request.
    
    Returns:
        True if authenticated, False otherwise
    """
    # Check for API key in headers
    api_key = request.headers.get('X-API-Key')
    if api_key and api_key == API_CONFIG['secret_key']:
        return True
    
    # Check for Bearer token
    auth_header = request.headers.get('Authorization')
    if auth_header and auth_header.startswith('Bearer '):
        token = auth_header.split(' ')[1]
        if token == API_CONFIG['secret_key']:
            return True
    
    return False


@app.route('/api/echo', methods=['POST'])
def ingest_echo_signal():
    """
    Main endpoint for ingesting external echo signals.
    
    Expected JSON payload:
    {
        "content": "Bitcoin just dropped 15%! Mass panic selling happening!",
        "source": "twitter",
        "metadata": {
            "engagement": 500,
            "sentiment_score": -0.8,
            "user_followers": 10000
        }
    }
    
    Returns:
        JSON response with processing status
    """
    try:
        # Check rate limit
        if not check_rate_limit():
            return jsonify({
                'error': 'Rate limit exceeded',
                'message': 'Too many requests per minute'
            }), 429
        
        # Authenticate request
        if not authenticate_request():
            return jsonify({
                'error': 'Unauthorized',
                'message': 'Invalid API key or token'
            }), 401
        
        # Parse JSON data
        try:
            data = request.get_json()
        except BadRequest:
            return jsonify({
                'error': 'Invalid JSON',
                'message': 'Request body must be valid JSON'
            }), 400
        
        if not data:
            return jsonify({
                'error': 'Missing data',
                'message': 'Request body is required'
            }), 400
        
        # Validate signal data
        if not validate_signal_data(data):
            return jsonify({
                'error': 'Invalid signal data',
                'message': 'Missing required fields or invalid values'
            }), 400
        
        # Process signal
        if not EXO_AVAILABLE or not exo_processor:
            return jsonify({
                'error': 'Service unavailable',
                'message': 'EXO Echo Signals not available'
            }), 503
        
        # Extract signal data
        content = data['content']
        source = SignalSource(data['source'])
        metadata = data.get('metadata', {})
        
        # Process the signal
        echo_signal = exo_processor.process_external_signal(
            content=content,
            source=source,
            metadata=metadata
        )
        
        if echo_signal:
            # Success response
            response_data = {
                'status': 'success',
                'message': 'Signal processed successfully',
                'signal_id': echo_signal.hash_value[:8],
                'symbol': echo_signal.symbol,
                'intent': echo_signal.intent.value,
                'priority': echo_signal.priority,
                'processed': echo_signal.processed,
                'timestamp': echo_signal.timestamp.isoformat()
            }
            
            logger.info(f"Processed echo signal: {echo_signal.symbol} ({echo_signal.intent.value})")
            return jsonify(response_data), 200
        else:
            # Signal below threshold
            return jsonify({
                'status': 'filtered',
                'message': 'Signal below priority threshold',
                'timestamp': datetime.utcnow().isoformat()
            }), 200
    
    except Exception as e:
        logger.error(f"Error processing echo signal: {e}")
        return jsonify({
            'error': 'Internal server error',
            'message': 'Failed to process signal'
        }), 500


@app.route('/api/echo/batch', methods=['POST'])
def ingest_batch_signals():
    """
    Batch endpoint for ingesting multiple signals at once.
    
    Expected JSON payload:
    {
        "signals": [
            {
                "content": "Bitcoin panic selling!",
                "source": "twitter",
                "metadata": {"engagement": 500}
            },
            {
                "content": "ETH looking bullish",
                "source": "reddit",
                "metadata": {"upvotes": 100}
            }
        ]
    }
    
    Returns:
        JSON response with batch processing results
    """
    try:
        # Check rate limit
        if not check_rate_limit():
            return jsonify({
                'error': 'Rate limit exceeded',
                'message': 'Too many requests per minute'
            }), 429
        
        # Authenticate request
        if not authenticate_request():
            return jsonify({
                'error': 'Unauthorized',
                'message': 'Invalid API key or token'
            }), 401
        
        # Parse JSON data
        try:
            data = request.get_json()
        except BadRequest:
            return jsonify({
                'error': 'Invalid JSON',
                'message': 'Request body must be valid JSON'
            }), 400
        
        if not data or 'signals' not in data:
            return jsonify({
                'error': 'Missing signals',
                'message': 'Request must contain signals array'
            }), 400
        
        signals_data = data['signals']
        if not isinstance(signals_data, list):
            return jsonify({
                'error': 'Invalid signals format',
                'message': 'Signals must be an array'
            }), 400
        
        if len(signals_data) > 100:  # Max 100 signals per batch
            return jsonify({
                'error': 'Batch too large',
                'message': 'Maximum 100 signals per batch'
            }), 400
        
        # Process signals
        if not EXO_AVAILABLE or not exo_processor:
            return jsonify({
                'error': 'Service unavailable',
                'message': 'EXO Echo Signals not available'
            }), 503
        
        results = []
        processed_count = 0
        filtered_count = 0
        error_count = 0
        
        for signal_data in signals_data:
            try:
                # Validate individual signal
                if not validate_signal_data(signal_data):
                    error_count += 1
                    results.append({
                        'status': 'error',
                        'message': 'Invalid signal data'
                    })
                    continue
                
                # Process signal
                content = signal_data['content']
                source = SignalSource(signal_data['source'])
                metadata = signal_data.get('metadata', {})
                
                echo_signal = exo_processor.process_external_signal(
                    content=content,
                    source=source,
                    metadata=metadata
                )
                
                if echo_signal:
                    processed_count += 1
                    results.append({
                        'status': 'success',
                        'signal_id': echo_signal.hash_value[:8],
                        'symbol': echo_signal.symbol,
                        'intent': echo_signal.intent.value,
                        'priority': echo_signal.priority
                    })
                else:
                    filtered_count += 1
                    results.append({
                        'status': 'filtered',
                        'message': 'Below threshold'
                    })
                    
            except Exception as e:
                error_count += 1
                results.append({
                    'status': 'error',
                    'message': str(e)
                })
        
        # Batch response
        response_data = {
            'status': 'completed',
            'summary': {
                'total_signals': len(signals_data),
                'processed': processed_count,
                'filtered': filtered_count,
                'errors': error_count
            },
            'results': results,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        logger.info(f"Batch processed: {processed_count} signals, {filtered_count} filtered, {error_count} errors")
        return jsonify(response_data), 200
    
    except Exception as e:
        logger.error(f"Error processing batch signals: {e}")
        return jsonify({
            'error': 'Internal server error',
            'message': 'Failed to process batch signals'
        }), 500


@app.route('/api/echo/status', methods=['GET'])
def get_echo_status():
    """
    Get current echo signal processing status.
    
    Returns:
        JSON response with system status
    """
    try:
        if not EXO_AVAILABLE or not exo_processor:
            return jsonify({
                'status': 'unavailable',
                'message': 'EXO Echo Signals not available'
            }), 503
        
        # Get metrics
        metrics = exo_processor.get_metrics()
        
        # Get Lantern Core status if available
        lantern_status = {}
        if lantern_core:
            lantern_metrics = lantern_core.get_integration_metrics()
            lantern_status = {
                'available': True,
                'metrics': lantern_metrics
            }
        else:
            lantern_status = {
                'available': False
            }
        
        # Get Strategy Mapper status if available
        strategy_status = {}
        if strategy_mapper:
            strategy_metrics = strategy_mapper.get_system_status()
            strategy_status = {
                'available': True,
                'metrics': strategy_metrics
            }
        else:
            strategy_status = {
                'available': False
            }
        
        response_data = {
            'status': 'operational',
            'timestamp': datetime.utcnow().isoformat(),
            'exo_echo_signals': {
                'available': True,
                'metrics': metrics
            },
            'lantern_core': lantern_status,
            'strategy_mapper': strategy_status,
            'rate_limit': {
                'requests_this_minute': len(request_timestamps),
                'max_per_minute': API_CONFIG['rate_limit_per_minute']
            }
        }
        
        return jsonify(response_data), 200
    
    except Exception as e:
        logger.error(f"Error getting echo status: {e}")
        return jsonify({
            'error': 'Internal server error',
            'message': 'Failed to get status'
        }), 500


@app.route('/api/echo/metrics', methods=['GET'])
def get_echo_metrics():
    """
    Get detailed processing metrics.
    
    Returns:
        JSON response with detailed metrics
    """
    try:
        if not EXO_AVAILABLE or not exo_processor:
            return jsonify({
                'error': 'Service unavailable',
                'message': 'EXO Echo Signals not available'
            }), 503
        
        # Get EXO metrics
        exo_metrics = exo_processor.get_metrics()
        
        # Get Lantern metrics if available
        lantern_metrics = {}
        if lantern_core:
            lantern_metrics = lantern_core.get_integration_metrics()
        
        # Get Strategy Mapper metrics if available
        strategy_metrics = {}
        if strategy_mapper:
            strategy_metrics = strategy_mapper.get_system_status()
        
        response_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'exo_echo_signals': exo_metrics,
            'lantern_core': lantern_metrics,
            'strategy_mapper': strategy_metrics,
            'api_metrics': {
                'rate_limit_requests': len(request_timestamps),
                'max_rate_limit': API_CONFIG['rate_limit_per_minute']
            }
        }
        
        return jsonify(response_data), 200
    
    except Exception as e:
        logger.error(f"Error getting echo metrics: {e}")
        return jsonify({
            'error': 'Internal server error',
            'message': 'Failed to get metrics'
        }), 500


@app.route('/api/echo/health', methods=['GET'])
def health_check():
    """
    Health check endpoint.
    
    Returns:
        JSON response with health status
    """
    try:
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'components': {
                'exo_echo_signals': EXO_AVAILABLE,
                'lantern_core': lantern_core is not None,
                'strategy_mapper': strategy_mapper is not None,
                'flask_api': True
            }
        }
        
        return jsonify(health_status), 200
    
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500


@app.route('/api/echo/config', methods=['GET'])
def get_config():
    """
    Get API configuration (read-only).
    
    Returns:
        JSON response with configuration
    """
    try:
        # Return safe configuration (no secrets)
        safe_config = {
            'rate_limit_per_minute': API_CONFIG['rate_limit_per_minute'],
            'max_content_length': API_CONFIG['max_content_length'],
            'enable_cors': API_CONFIG['enable_cors'],
            'debug': API_CONFIG['debug'],
            'supported_sources': [source.value for source in SignalSource] if EXO_AVAILABLE else [],
            'supported_intents': [intent.value for intent in SignalIntent] if EXO_AVAILABLE else []
        }
        
        return jsonify(safe_config), 200
    
    except Exception as e:
        logger.error(f"Error getting config: {e}")
        return jsonify({
            'error': 'Internal server error',
            'message': 'Failed to get configuration'
        }), 500


# Error handlers
@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        'error': 'Not found',
        'message': 'Endpoint not found'
    }), 404


@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors."""
    return jsonify({
        'error': 'Method not allowed',
        'message': 'HTTP method not supported for this endpoint'
    }), 405


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred'
    }), 500


# CORS support
@app.after_request
def after_request(response):
    """Add CORS headers if enabled."""
    if API_CONFIG['enable_cors']:
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,X-API-Key')
        response.headers.add('Access-Control-Allow-Methods', 'GET,POST,PUT,DELETE,OPTIONS')
    return response


if __name__ == '__main__':
    # Run the Flask app
    print("🚀 Starting Echo Signal API...")
    print(f"📡 Endpoints available:")
    print(f"   POST /api/echo - Ingest single signal")
    print(f"   POST /api/echo/batch - Ingest batch signals")
    print(f"   GET /api/echo/status - System status")
    print(f"   GET /api/echo/metrics - Processing metrics")
    print(f"   GET /api/echo/health - Health check")
    print(f"   GET /api/echo/config - API configuration")
    
    app.run(
        host='0.0.0.0',
        port=5001,
        debug=API_CONFIG['debug']
    ) 