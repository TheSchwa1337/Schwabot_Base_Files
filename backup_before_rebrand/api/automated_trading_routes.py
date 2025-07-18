#!/usr/bin/env python3
"""
Automated Trading Routes - Flask API endpoints for automated trading operations
"""

import logging
import os
import sys
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional

from flask_socketio import emit

from flask import Blueprint, jsonify, request

# Add core directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.automated_strategy_engine import AutomatedStrategyEngine
from core.automated_trading_engine import AutomatedTradingEngine

logger = logging.getLogger(__name__)

automated_trading = Blueprint('automated_trading', __name__)

# Global instances (in production, these would be properly managed)
trading_engine = None
strategy_engine = None


def get_trading_engine() -> AutomatedTradingEngine:
    """Get or create trading engine instance."""
    global trading_engine
    if trading_engine is None:
        # Initialize with Coinbase configuration
        exchange_config = {'name': 'coinbase', 'sandbox': True}  # Use sandbox for testing
        trading_engine = AutomatedTradingEngine(exchange_config)
    return trading_engine


def get_strategy_engine() -> AutomatedStrategyEngine:
    """Get or create strategy engine instance."""
    global strategy_engine
    if strategy_engine is None:
        trading_engine = get_trading_engine()
        strategy_engine = AutomatedStrategyEngine(trading_engine)
    return strategy_engine


def emit_automated_event(event_type: str, data: Dict, room: str = None):
    """Emit automated trading event to connected clients."""
    try:
        from api.flask_app import socketio

        event_data = {'type': f'automated_{event_type}', 'timestamp': time.time(), 'data': data}
        if room:
            socketio.emit('realtime_update', event_data, room=room)
        else:
            socketio.emit('realtime_update', event_data)
        logger.info(f"Emitted automated {event_type} event: {data}")
    except Exception as e:
        logger.error(f"Failed to emit automated event: {e}")


@automated_trading.route('/initialize', methods=['POST'])
def initialize_automated_trading():
    """Initialize automated trading system."""
    try:
        data = request.get_json() or {}

        # Get exchange configuration
        exchange_config = data.get('exchange_config', {'name': 'coinbase', 'sandbox': True})

        # Get API credentials (in production, these would be securely stored)
        api_key = data.get('api_key')
        secret = data.get('secret')

        # Initialize trading engine
        global trading_engine
        trading_engine = AutomatedTradingEngine(exchange_config, api_key, secret)

        # Initialize strategy engine
        global strategy_engine
        strategy_engine = AutomatedStrategyEngine(trading_engine)

        # Add default symbols to tracking
        default_symbols = data.get('symbols', ['BTC/USDC', 'ETH/USDC', 'SOL/USDC'])
        for symbol in default_symbols:
            trading_engine.add_symbol_to_tracking(symbol)

        emit_automated_event(
            'initialized',
            {
                'exchange': exchange_config['name'],
                'symbols_tracking': default_symbols,
                'status': 'ready',
            },
        )

        return jsonify(
            {
                'status': 'success',
                'message': 'Automated trading system initialized',
                'exchange': exchange_config['name'],
                'symbols_tracking': default_symbols,
            }
        )

    except Exception as e:
        logger.error(f"Error initializing automated trading: {e}")
        return jsonify({'error': str(e)}), 500


@automated_trading.route('/add_symbol', methods=['POST'])
def add_symbol_to_tracking():
    """Add symbol to automated price tracking."""
    try:
        data = request.get_json()
        if not data or 'symbol' not in data:
            return jsonify({'error': 'Missing symbol in request'}), 400

        symbol = data['symbol']
        trading_engine = get_trading_engine()
        trading_engine.add_symbol_to_tracking(symbol)

        emit_automated_event('symbol_added', {'symbol': symbol, 'status': 'tracking'})

        return jsonify({'status': 'success', 'message': f'Added {symbol} to tracking', 'symbol': symbol})

    except Exception as e:
        logger.error(f"Error adding symbol: {e}")
        return jsonify({'error': str(e)}), 500


@automated_trading.route('/remove_symbol', methods=['POST'])
def remove_symbol_from_tracking():
    """Remove symbol from automated price tracking."""
    try:
        data = request.get_json()
        if not data or 'symbol' not in data:
            return jsonify({'error': 'Missing symbol in request'}), 400

        symbol = data['symbol']
        trading_engine = get_trading_engine()
        trading_engine.remove_symbol_from_tracking(symbol)

        emit_automated_event('symbol_removed', {'symbol': symbol, 'status': 'stopped'})

        return jsonify({'status': 'success', 'message': f'Removed {symbol} from tracking', 'symbol': symbol})

    except Exception as e:
        logger.error(f"Error removing symbol: {e}")
        return jsonify({'error': str(e)}), 500


@automated_trading.route('/prices', methods=['GET'])
def get_current_prices():
    """Get all current prices from automated tracking."""
    try:
        trading_engine = get_trading_engine()
        prices = trading_engine.get_all_prices()

        return jsonify({'status': 'success', 'prices': prices, 'timestamp': datetime.now().isoformat()})

    except Exception as e:
        logger.error(f"Error getting prices: {e}")
        return jsonify({'error': str(e)}), 500


@automated_trading.route('/analyze/<symbol>', methods=['GET'])
def analyze_symbol(symbol: str):
    """Analyze mathematical tensor movements for a symbol."""
    try:
        strategy_engine = get_strategy_engine()
        analysis = strategy_engine.analyze_tensor_movements(symbol)

        if 'error' in analysis:
            return jsonify({'error': analysis['error']}), 400

        emit_automated_event('analysis_completed', {'symbol': symbol, 'analysis': analysis})

        return jsonify({'status': 'success', 'symbol': symbol, 'analysis': analysis})

    except Exception as e:
        logger.error(f"Error analyzing symbol {symbol}: {e}")
        return jsonify({'error': str(e)}), 500


@automated_trading.route('/decision/<symbol>', methods=['POST'])
def make_automated_decision(symbol: str):
    """Make automated trading decision for a symbol."""
    try:
        strategy_engine = get_strategy_engine()
        decision = strategy_engine.make_automated_decision(symbol)

        if decision is None:
            return jsonify(
                {
                    'status': 'no_decision',
                    'message': f'No confident decision for {symbol}',
                    'symbol': symbol,
                }
            )

        # Convert decision to serializable format
        decision_data = {
            'symbol': decision.symbol,
            'action': decision.action,
            'confidence': decision.confidence,
            'quantity': decision.quantity,
            'price_range': decision.price_range,
            'batch_count': decision.batch_count,
            'spread_seconds': decision.spread_seconds,
            'strategy_id': decision.strategy_id,
            'reasoning': decision.reasoning,
            'timestamp': decision.timestamp.isoformat(),
        }

        emit_automated_event('decision_made', {'symbol': symbol, 'decision': decision_data})

        return jsonify({'status': 'success', 'decision': decision_data})

    except Exception as e:
        logger.error(f"Error making decision for {symbol}: {e}")
        return jsonify({'error': str(e)}), 500


@automated_trading.route('/execute_decision', methods=['POST'])
def execute_automated_decision():
    """Execute an automated trading decision."""
    try:
        data = request.get_json()
        if not data or 'decision' not in data:
            return jsonify({'error': 'Missing decision in request'}), 400

        decision_data = data['decision']

        # Reconstruct decision object
        from core.automated_strategy_engine import AutomatedDecision

        decision = AutomatedDecision(
            symbol=decision_data['symbol'],
            action=decision_data['action'],
            confidence=decision_data['confidence'],
            quantity=decision_data['quantity'],
            price_range=tuple(decision_data['price_range']),
            batch_count=decision_data['batch_count'],
            spread_seconds=decision_data['spread_seconds'],
            strategy_id=decision_data['strategy_id'],
            reasoning=decision_data['reasoning'],
        )

        strategy_engine = get_strategy_engine()
        batch_id = strategy_engine.execute_automated_decision(decision)

        emit_automated_event(
            'decision_executed',
            {
                'symbol': decision.symbol,
                'action': decision.action,
                'batch_id': batch_id,
                'status': 'executing',
            },
        )

        return jsonify(
            {
                'status': 'success',
                'message': f'Executed {decision.action} for {decision.symbol}',
                'batch_id': batch_id,
            }
        )

    except Exception as e:
        logger.error(f"Error executing decision: {e}")
        return jsonify({'error': str(e)}), 500


@automated_trading.route('/create_buy_wall', methods=['POST'])
def create_buy_wall():
    """Create automated buy wall."""
    try:
        data = request.get_json()
        if not data or 'symbol' not in data or 'quantity' not in data:
            return jsonify({'error': 'Missing required parameters'}), 400

        symbol = data['symbol']
        quantity = float(data['quantity'])
        price_range = tuple(data.get('price_range', [0, 0]))  # Will be calculated if not provided
        batch_count = data.get('batch_count', 10)
        spread_seconds = data.get('spread_seconds', 30)

        trading_engine = get_trading_engine()

        # If price range not provided, calculate based on current price
        if price_range == (0, 0):
            current_price = trading_engine.get_current_price(symbol)
            if current_price:
                price_range = (current_price * 0.99, current_price * 1.02)
            else:
                return jsonify({'error': 'No current price available for symbol'}), 400

        batch_id = trading_engine.create_buy_wall(
            symbol=symbol,
            total_quantity=quantity,
            price_range=price_range,
            batch_count=batch_count,
            spread_seconds=spread_seconds,
        )

        emit_automated_event(
            'buy_wall_created',
            {'symbol': symbol, 'quantity': quantity, 'batch_id': batch_id, 'status': 'created'},
        )

        return jsonify({'status': 'success', 'message': f'Created buy wall for {symbol}', 'batch_id': batch_id})

    except Exception as e:
        logger.error(f"Error creating buy wall: {e}")
        return jsonify({'error': str(e)}), 500


@automated_trading.route('/create_sell_wall', methods=['POST'])
def create_sell_wall():
    """Create automated sell wall."""
    try:
        data = request.get_json()
        if not data or 'symbol' not in data or 'quantity' not in data:
            return jsonify({'error': 'Missing required parameters'}), 400

        symbol = data['symbol']
        quantity = float(data['quantity'])
        price_range = tuple(data.get('price_range', [0, 0]))  # Will be calculated if not provided
        batch_count = data.get('batch_count', 10)
        spread_seconds = data.get('spread_seconds', 30)

        trading_engine = get_trading_engine()

        # If price range not provided, calculate based on current price
        if price_range == (0, 0):
            current_price = trading_engine.get_current_price(symbol)
            if current_price:
                price_range = (current_price * 0.98, current_price * 1.01)
            else:
                return jsonify({'error': 'No current price available for symbol'}), 400

        batch_id = trading_engine.create_sell_wall(
            symbol=symbol,
            total_quantity=quantity,
            price_range=price_range,
            batch_count=batch_count,
            spread_seconds=spread_seconds,
        )

        emit_automated_event(
            'sell_wall_created',
            {'symbol': symbol, 'quantity': quantity, 'batch_id': batch_id, 'status': 'created'},
        )

        return jsonify(
            {
                'status': 'success',
                'message': f'Created sell wall for {symbol}',
                'batch_id': batch_id,
            }
        )

    except Exception as e:
        logger.error(f"Error creating sell wall: {e}")
        return jsonify({'error': str(e)}), 500


@automated_trading.route('/create_basket', methods=['POST'])
def create_basket_order():
    """Create automated basket order."""
    try:
        data = request.get_json()
        if not data or 'symbols' not in data or 'weights' not in data or 'value' not in data:
            return jsonify({'error': 'Missing required parameters'}), 400

        symbols = data['symbols']
        weights = data['weights']
        total_value = float(data['value'])
        strategy = data.get('strategy', 'basket')

        if len(symbols) != len(weights):
            return jsonify({'error': 'Symbols and weights must have same length'}), 400

        if abs(sum(weights) - 1.0) > 0.01:
            return jsonify({'error': 'Weights must sum to 1.0'}), 400

        trading_engine = get_trading_engine()
        basket_id = trading_engine.create_basket_order(
            basket_symbols=symbols, weights=weights, total_value=total_value, strategy=strategy
        )

        emit_automated_event(
            'basket_created',
            {
                'symbols': symbols,
                'total_value': total_value,
                'basket_id': basket_id,
                'status': 'created',
            },
        )

        return jsonify(
            {
                'status': 'success',
                'message': f'Created basket order for {len(symbols)} symbols',
                'basket_id': basket_id,
            }
        )

    except Exception as e:
        logger.error(f"Error creating basket order: {e}")
        return jsonify({'error': str(e)}), 500


@automated_trading.route('/orders', methods=['GET'])
def get_all_orders():
    """Get all active orders."""
    try:
        trading_engine = get_trading_engine()
        orders = trading_engine.get_all_orders()

        # Convert to serializable format
        serializable_orders = {}
        for order_id, order_info in orders.items():
            serializable_orders[order_id] = {
                'symbol': order_info['signal'].symbol,
                'side': order_info['signal'].side,
                'quantity': order_info['signal'].quantity,
                'status': order_info['status'],
                'timestamp': (order_info['timestamp'].isoformat() if order_info['timestamp'] else None),
            }

        return jsonify({'status': 'success', 'orders': serializable_orders})

    except Exception as e:
        logger.error(f"Error getting orders: {e}")
        return jsonify({'error': str(e)}), 500


@automated_trading.route('/orders/<order_id>', methods=['GET'])
def get_order_status(order_id: str):
    """Get status of a specific order."""
    try:
        trading_engine = get_trading_engine()
        order_info = trading_engine.get_order_status(order_id)

        if not order_info:
            return jsonify({'error': 'Order not found'}), 404

        # Convert to serializable format
        serializable_order = {
            'order_id': order_id,
            'symbol': order_info['signal'].symbol,
            'side': order_info['signal'].side,
            'quantity': order_info['signal'].quantity,
            'status': order_info['status'],
            'timestamp': order_info['timestamp'].isoformat() if order_info['timestamp'] else None,
        }

        return jsonify({'status': 'success', 'order': serializable_order})

    except Exception as e:
        logger.error(f"Error getting order status: {e}")
        return jsonify({'error': str(e)}), 500


@automated_trading.route('/orders/<order_id>/cancel', methods=['POST'])
def cancel_order(order_id: str):
    """Cancel a specific order."""
    try:
        trading_engine = get_trading_engine()
        success = trading_engine.cancel_order(order_id)

        if success:
            emit_automated_event('order_canceled', {'order_id': order_id, 'status': 'canceled'})

            return jsonify({'status': 'success', 'message': f'Canceled order {order_id}'})
        else:
            return jsonify({'error': 'Failed to cancel order'}), 500

    except Exception as e:
        logger.error(f"Error canceling order: {e}")
        return jsonify({'error': str(e)}), 500


@automated_trading.route('/portfolio', methods=['GET'])
def get_portfolio():
    """Get current portfolio balances."""
    try:
        trading_engine = get_trading_engine()
        portfolio = trading_engine.get_portfolio()

        return jsonify({'status': 'success', 'portfolio': portfolio})

    except Exception as e:
        logger.error(f"Error getting portfolio: {e}")
        return jsonify({'error': str(e)}), 500


@automated_trading.route('/tensor_state', methods=['GET'])
def get_tensor_state():
    """Get current mathematical tensor state."""
    try:
        trading_engine = get_trading_engine()
        tensor_state = trading_engine.get_tensor_state()

        # Convert numpy arrays to lists for JSON serialization
        serializable_state = {}
        for key, value in tensor_state.items():
            if isinstance(value, dict):
                serializable_state[key] = {}
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, list):
                        serializable_state[key][sub_key] = [
                            float(x) if isinstance(x, (int, float)) else x for x in sub_value
                        ]
                    else:
                        serializable_state[key][sub_key] = (
                            float(sub_value) if isinstance(sub_value, (int, float)) else sub_value
                        )
            else:
                serializable_state[key] = value

        return jsonify({'status': 'success', 'tensor_state': serializable_state})

    except Exception as e:
        logger.error(f"Error getting tensor state: {e}")
        return jsonify({'error': str(e)}), 500


@automated_trading.route('/learning_status', methods=['GET'])
def get_learning_status():
    """Get automated learning status."""
    try:
        strategy_engine = get_strategy_engine()
        status = strategy_engine.get_learning_status()

        return jsonify({'status': 'success', 'learning_status': status})

    except Exception as e:
        logger.error(f"Error getting learning status: {e}")
        return jsonify({'error': str(e)}), 500


@automated_trading.route('/auto_trade/<symbol>', methods=['POST'])
def start_auto_trading(symbol: str):
    """Start automated trading for a symbol."""
    try:
        data = request.get_json() or {}
        interval_seconds = data.get('interval_seconds', 60)  # Check every minute

        def auto_trade_loop():
            """Background loop for automated trading."""
            strategy_engine = get_strategy_engine()

            while True:
                try:
                    # Make automated decision
                    decision = strategy_engine.make_automated_decision(symbol)

                    if decision and decision.confidence > 0.8:
                        # Execute decision
                        batch_id = strategy_engine.execute_automated_decision(decision)

                        emit_automated_event(
                            'auto_trade_executed',
                            {
                                'symbol': symbol,
                                'action': decision.action,
                                'confidence': decision.confidence,
                                'batch_id': batch_id,
                            },
                        )

                    time.sleep(interval_seconds)

                except Exception as e:
                    logger.error(f"Error in auto trade loop for {symbol}: {e}")
                    time.sleep(interval_seconds)

        # Start background thread
        thread = threading.Thread(target=auto_trade_loop, daemon=True)
        thread.start()

        emit_automated_event(
            'auto_trading_started',
            {'symbol': symbol, 'interval_seconds': interval_seconds, 'status': 'running'},
        )

        return jsonify(
            {
                'status': 'success',
                'message': f'Started automated trading for {symbol}',
                'symbol': symbol,
                'interval_seconds': interval_seconds,
            }
        )

    except Exception as e:
        logger.error(f"Error starting auto trading for {symbol}: {e}")
        return jsonify({'error': str(e)}), 500
