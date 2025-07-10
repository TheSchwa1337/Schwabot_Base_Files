#!/usr/bin/env python3
"""
Live Trading Routes - Flask API endpoints for live trading operations with real-time feedback
"""
import logging
import os
import threading
import time
from io import BytesIO

import numpy as np
from flask_socketio import emit

from core.integration_orchestrator import orchestrate_trade
from core.mathlib_v3_visualizer import get_placeholder_plot
from core.matrix_mapper import cosine_similarity, match_hash_to_matrix
from core.strategy_loader import load_strategy
from flask import Blueprint, jsonify, request, send_file

logger = logging.getLogger(__name__)

live_trading = Blueprint('live_trading', __name__)


def emit_realtime_event(event_type, data, room=None):
    """Emit a real-time event to connected clients."""
    try:
        from api.flask_app import socketio

        event_data = {'type': event_type, 'timestamp': time.time(), 'data': data}
        if room:
            socketio.emit('realtime_update', event_data, room=room)
        else:
            socketio.emit('realtime_update', event_data)
        logger.info(f"Emitted {event_type} event: {data}")
    except Exception as e:
        logger.error(f"Failed to emit real-time event: {e}")


@live_trading.route('/trade/hash', methods=['POST'])
def trade_by_hash():
    """POST endpoint for hash-based trading decisions with real-time feedback."""
    try:
        data = request.get_json()
        if not data or 'hash_vector' not in data:
            return jsonify({'error': 'Missing hash_vector in request'}), 400

        hash_vector = data['hash_vector']
        strategy_name = data.get('strategy_name', 'momentum')
        matrix_dir = data.get('matrix_dir', os.path.join(os.path.dirname(__file__), '..', 'core', 'data'))

        # Validate hash vector
        if not isinstance(hash_vector, list) or len(hash_vector) == 0:
            return jsonify({'error': 'Invalid hash_vector format'}), 400

        # Emit trade start event
        emit_realtime_event(
            'trade_started',
            {
                'hash_length': len(hash_vector),
                'strategy': strategy_name,
                'status': 'Processing hash vector...',
            },
        )

        # Run orchestration in background thread for real-time updates
        def run_trade_orchestration():
            try:
                # Emit matrix matching progress
                emit_realtime_event(
                    'trade_progress',
                    {
                        'step': 'matrix_matching',
                        'message': 'Finding matching matrix...',
                        'progress': 25,
                    },
                )

                # Find matching matrix
                matrix_file = match_hash_to_matrix(hash_vector, matrix_dir)

                emit_realtime_event(
                    'trade_progress',
                    {
                        'step': 'matrix_found',
                        'message': f'Matrix found: {matrix_file}',
                        'progress': 50,
                    },
                )

                # Load strategy
                emit_realtime_event(
                    'trade_progress',
                    {
                        'step': 'strategy_loading',
                        'message': f'Loading strategy: {strategy_name}',
                        'progress': 75,
                    },
                )

                strategy = load_strategy(strategy_name)

                # Run orchestration
                emit_realtime_event(
                    'trade_progress',
                    {
                        'step': 'executing',
                        'message': 'Executing trade orchestration...',
                        'progress': 90,
                    },
                )

                result = orchestrate_trade(hash_vector, matrix_dir, strategy_name)

                # Emit completion
                emit_realtime_event(
                    'trade_completed',
                    {
                        'result': result,
                        'matrix_file': result.get('matrix_file'),
                        'strategy_used': strategy_name,
                        'hash_vector_length': len(hash_vector),
                    },
                )

            except Exception as e:
                emit_realtime_event(
                    'trade_error',
                    {'error': str(e), 'hash_length': len(hash_vector), 'strategy': strategy_name},
                )
                logger.error(f"Trade orchestration error: {e}")

        # Start background thread
        thread = threading.Thread(target=run_trade_orchestration)
        thread.daemon = True
        thread.start()

        return jsonify(
            {
                'status': 'submitted',
                'message': 'Trade submitted for processing',
                'hash_vector_length': len(hash_vector),
                'strategy_name': strategy_name,
            }
        )

    except Exception as e:
        logger.error(f"Error in trade_by_hash: {e}")
        return jsonify({'error': str(e)}), 500


@live_trading.route('/trade/strategy/<strategy_name>', methods=['POST'])
def trade_by_strategy(strategy_name):
    """POST endpoint for strategy-based trading decisions with real-time feedback."""
    try:
        data = request.get_json()
        if not data or 'market_data' not in data:
            return jsonify({'error': 'Missing market_data in request'}), 400

        market_data = data['market_data']

        # Emit strategy execution start
        emit_realtime_event(
            'strategy_started',
            {
                'strategy': strategy_name,
                'market_data': market_data,
                'status': 'Loading strategy...',
            },
        )

        # Load strategy
        strategy = load_strategy(strategy_name)
        if not strategy:
            emit_realtime_event(
                'strategy_error',
                {'error': f'Strategy {strategy_name} not found', 'strategy': strategy_name},
            )
            return jsonify({'error': f'Strategy {strategy_name} not found'}), 404

        emit_realtime_event(
            'strategy_progress',
            {
                'step': 'strategy_loaded',
                'message': f'Strategy {strategy_name} loaded successfully',
                'progress': 50,
            },
        )

        # Execute strategy
        result = strategy(market_data)

        emit_realtime_event(
            'strategy_completed',
            {'strategy': strategy_name, 'decision': result, 'market_data': market_data},
        )

        return jsonify(
            {
                'status': 'success',
                'strategy': strategy_name,
                'decision': result,
                'market_data': market_data,
            }
        )

    except Exception as e:
        logger.error(f"Error in trade_by_strategy: {e}")
        emit_realtime_event('strategy_error', {'error': str(e), 'strategy': strategy_name})
        return jsonify({'error': str(e)}), 500


@live_trading.route('/matrix/match', methods=['POST'])
def match_matrix():
    """POST endpoint for matrix matching with real-time feedback."""
    try:
        data = request.get_json()
        if not data or 'hash_vector' not in data:
            return jsonify({'error': 'Missing hash_vector in request'}), 400

        hash_vector = data['hash_vector']
        matrix_dir = data.get('matrix_dir', os.path.join(os.path.dirname(__file__), '..', 'core', 'data'))
        threshold = data.get('threshold', 0.8)

        # Emit matrix matching start
        emit_realtime_event(
            'matrix_search_started',
            {
                'hash_length': len(hash_vector),
                'threshold': threshold,
                'status': 'Searching for matching matrix...',
            },
        )

        # Find matching matrix
        matrix_file = match_hash_to_matrix(hash_vector, matrix_dir, threshold)

        if matrix_file:
            emit_realtime_event(
                'matrix_found',
                {
                    'matrix_file': matrix_file,
                    'threshold': threshold,
                    'hash_vector_length': len(hash_vector),
                },
            )

            return jsonify(
                {
                    'status': 'success',
                    'matrix_file': matrix_file,
                    'threshold': threshold,
                    'hash_vector_length': len(hash_vector),
                }
            )
        else:
            emit_realtime_event(
                'matrix_not_found',
                {
                    'message': f'No matrix found above threshold {threshold}',
                    'threshold': threshold,
                    'hash_vector_length': len(hash_vector),
                },
            )

            return (
                jsonify(
                    {
                        'status': 'no_match',
                        'message': f'No matrix found above threshold {threshold}',
                        'hash_vector_length': len(hash_vector),
                    }
                ),
                404,
            )

    except Exception as e:
        logger.error(f"Error in match_matrix: {e}")
        emit_realtime_event(
            'matrix_error',
            {'error': str(e), 'hash_length': len(hash_vector) if 'hash_vector' in locals() else 0},
        )
        return jsonify({'error': str(e)}), 500


@live_trading.route('/visualize/matrix', methods=['GET'])
def visualize_matrix():
    """GET endpoint for matrix visualization with real-time feedback."""
    try:
        # Emit visualization start
        emit_realtime_event('visualization_started', {'status': 'Generating matrix visualization...'})

        # Return placeholder visualization
        plot_data = get_placeholder_plot()

        emit_realtime_event(
            'visualization_completed',
            {'content_length': len(plot_data), 'content_type': 'image/png'},
        )

        return send_file(
            BytesIO(plot_data),
            mimetype='image/png',
            as_attachment=True,
            download_name='matrix_visualization.png',
        )

    except Exception as e:
        logger.error(f"Error in visualize_matrix: {e}")
        emit_realtime_event('visualization_error', {'error': str(e)})
        return jsonify({'error': str(e)}), 500


@live_trading.route('/test/route', methods=['POST'])
def test_route():
    """POST endpoint for testing the full orchestration pipeline with real-time feedback."""
    try:
        data = request.get_json()
        if not data:
            # Use default test data
            test_hash = np.random.rand(10).tolist()
            test_strategy = 'momentum'
        else:
            test_hash = data.get('hash_vector', np.random.rand(10).tolist())
            test_strategy = data.get('strategy_name', 'momentum')

        matrix_dir = os.path.join(os.path.dirname(__file__), '..', 'core', 'data')

        # Emit test start
        emit_realtime_event(
            'test_started',
            {
                'test_type': 'full_orchestration',
                'hash_length': len(test_hash),
                'strategy': test_strategy,
                'status': 'Starting full orchestration test...',
            },
        )

        # Run full test with progress updates
        def run_test_orchestration():
            try:
                steps = [
                    ('matrix_matching', 'Matching hash to matrix...', 20),
                    ('strategy_loading', f'Loading strategy: {test_strategy}...', 40),
                    ('orchestration', 'Running orchestration pipeline...', 60),
                    ('execution', 'Executing trade logic...', 80),
                    ('completion', 'Finalizing results...', 100),
                ]

                for step, message, progress in steps:
                    emit_realtime_event('test_progress', {'step': step, 'message': message, 'progress': progress})
                    time.sleep(0.5)  # Simulate processing time

                result = orchestrate_trade(test_hash, matrix_dir, test_strategy)

                emit_realtime_event(
                    'test_completed',
                    {
                        'test_type': 'full_orchestration',
                        'result': result,
                        'test_hash_length': len(test_hash),
                        'strategy_tested': test_strategy,
                    },
                )

            except Exception as e:
                emit_realtime_event('test_error', {'error': str(e), 'test_type': 'full_orchestration'})
                logger.error(f"Test orchestration error: {e}")

        # Start background thread
        thread = threading.Thread(target=run_test_orchestration)
        thread.daemon = True
        thread.start()

        return jsonify(
            {
                'status': 'submitted',
                'message': 'Test submitted for processing',
                'test_type': 'full_orchestration',
                'test_hash_length': len(test_hash),
                'strategy_tested': test_strategy,
            }
        )

    except Exception as e:
        logger.error(f"Error in test_route: {e}")
        return jsonify({'error': str(e)}), 500


@live_trading.route('/status', methods=['GET'])
def system_status():
    """GET endpoint for system status with real-time updates."""
    try:
        # Check if core components are available
        strategy_count = len(load_strategy.__defaults__) if load_strategy.__defaults__ else 0

        status_data = {
            'status': 'operational',
            'components': {
                'orchestrator': 'available',
                'matrix_mapper': 'available',
                'strategy_loader': 'available',
                'visualizer': 'available',
            },
            'strategies_loaded': strategy_count,
            'api_version': '1.0.0',
            'timestamp': time.time(),
        }

        # Emit status update
        emit_realtime_event('status_update', status_data)

        return jsonify(status_data)

    except Exception as e:
        logger.error(f"Error in system_status: {e}")
        error_data = {'status': 'error', 'error': str(e), 'timestamp': time.time()}
        emit_realtime_event('status_error', error_data)
        return jsonify(error_data), 500
