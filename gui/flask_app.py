#!/usr/bin/env python3
"""
Schwabot Flask GUI - Comprehensive Trading Interface
Provides demo, live trading, CCXT integration, and visualization capabilities.
"""

import json
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from core.ccxt_integration import CCXTIntegration
from core.clean_unified_math import clean_unified_math
from core.unified_math_system import generate_unified_hash
from core.unified_trade_router import UnifiedTradeRouter

# Core imports
from core.visual_execution_node import VisualExecutionNode
from flask import Flask, jsonify, render_template, request, session
from gui.visualizer_launcher import VisualizerLauncher

app = Flask(__name__)
app.secret_key = 'schwabot_gui_secret_key_2025'

# Global state
trading_state = {
    'mode': 'demo',  # 'demo', 'live', 'backtest'
    'active_trades': [],
    'trade_history': [],
    'portfolio_value': 10000.0,
    'total_profit': 0.0,
    'win_rate': 0.0,
    'current_asset': 'BTC/USDC',
    'current_price': 60000.0,
    'backlog_data': [],
    'api_connected': False,
    'ccxt_config': {},
}

# Initialize core components
router = UnifiedTradeRouter()
ccxt_integration = None

# Initialize visualizer launcher
visualizer_launcher = VisualizerLauncher()


class TradingSession:
    """Manages trading session state and operations."""

    def __init__(self):
        self.session_id = generate_unified_hash([time.time()], "session")
        self.start_time = time.time()
        self.trades = []
        self.positions = {}
        self.backlog = []

    def add_trade(self, trade_data: Dict[str, Any]):
        """Add trade to session history."""
        trade_data['session_id'] = self.session_id
        trade_data['timestamp'] = time.time()
        self.trades.append(trade_data)

        # Update portfolio
        if trade_data.get('executed'):
            self._update_portfolio(trade_data)

    def _update_portfolio(self, trade: Dict[str, Any]):
        """Update portfolio based on trade execution."""
        if trade['side'] == 'BUY':
            # Add position
            asset = trade['asset']
            if asset not in self.positions:
                self.positions[asset] = {'quantity': 0, 'avg_price': 0}

            pos = self.positions[asset]
            new_quantity = pos['quantity'] + trade['quantity']
            new_avg_price = ((pos['quantity'] * pos['avg_price']) + (trade['quantity'] * trade['price'])) / new_quantity
            pos['quantity'] = new_quantity
            pos['avg_price'] = new_avg_price

        elif trade['side'] == 'SELL':
            # Close position and calculate profit
            asset = trade['asset']
            if asset in self.positions:
                pos = self.positions[asset]
                if pos['quantity'] >= trade['quantity']:
                    profit = (trade['price'] - pos['avg_price']) * trade['quantity']
                    trading_state['total_profit'] += profit
                    pos['quantity'] -= trade['quantity']

                    if pos['quantity'] <= 0:
                        del self.positions[asset]


# Global session
current_session = TradingSession()


@app.route('/')
def index():
    """Main trading dashboard."""
    return render_template('dashboard.html', state=trading_state)


@app.route('/api/execute_signal', methods=['POST'])
def execute_signal():
    """Execute trading signal via VisualExecutionNode."""
    try:
        data = request.get_json()
        asset = data.get('asset', 'BTC/USDC')
        price = float(data.get('price', 60000.0))
        mode = data.get('mode', 'demo')

        # Create visual execution node
        node = VisualExecutionNode(asset, price)
        result = node.execute()

        # Add to session
        trade_data = {
            'asset': asset,
            'price': price,
            'side': 'BUY' if price > 55000 else 'SELL',  # Simple logic
            'quantity': 0.1,
            'mode': mode,
            'hash': result['hash'],
            'visual_display': result['visual_display'],
            'executed': mode == 'demo',  # Only execute in demo mode initially
        }

        current_session.add_trade(trade_data)

        return jsonify(
            {
                'success': True,
                'result': result,
                'trade_data': trade_data,
                'session_id': current_session.session_id,
            }
        )

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/connect_ccxt', methods=['POST'])
def connect_ccxt():
    """Connect to CCXT exchange."""
    try:
        data = request.get_json()
        exchange = data.get('exchange', 'coinbase')
        api_key = data.get('api_key', '')
        secret = data.get('secret', '')

        global ccxt_integration
        ccxt_integration = CCXTIntegration(
            {
                'exchange': exchange,
                'api_key': api_key,
                'secret': secret,
                'sandbox': True,  # Start with sandbox
            }
        )

        trading_state['api_connected'] = True
        trading_state['ccxt_config'] = {'exchange': exchange, 'connected': True}

        return jsonify(
            {
                'success': True,
                'message': f'Connected to {exchange}',
                'config': trading_state['ccxt_config'],
            }
        )

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/switch_mode', methods=['POST'])
def switch_mode():
    """Switch between demo, live, and backtest modes."""
    try:
        data = request.get_json()
        new_mode = data.get('mode', 'demo')

        if new_mode not in ['demo', 'live', 'backtest']:
            return jsonify({'success': False, 'error': 'Invalid mode'})

        trading_state['mode'] = new_mode

        return jsonify({'success': True, 'mode': new_mode, 'message': f'Switched to {new_mode} mode'})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/get_portfolio')
def get_portfolio():
    """Get current portfolio status."""
    return jsonify(
        {
            'portfolio_value': trading_state['portfolio_value'],
            'total_profit': trading_state['total_profit'],
            'win_rate': trading_state['win_rate'],
            'positions': current_session.positions,
            'active_trades': len(current_session.trades),
            'session_id': current_session.session_id,
        }
    )


@app.route('/api/get_trade_history')
def get_trade_history():
    """Get trade history for current session."""
    return jsonify({'trades': current_session.trades, 'session_id': current_session.session_id})


@app.route('/api/calculate_math_score', methods=['POST'])
def calculate_math_score():
    """Calculate mathematical score using clean_unified_math."""
    try:
        data = request.get_json()
        price = float(data.get('price', 60000.0))
        volume = float(data.get('volume', 1000.0))
        confidence = float(data.get('confidence', 0.5))

        # Use clean_unified_math for calculations
        math_result = clean_unified_math.integrate_all_systems(
            {'tensor': [[price, volume]], 'metadata': {'confidence': confidence}}
        )

        return jsonify(
            {
                'success': True,
                'math_score': math_result.get('combined_score', 0.0),
                'components': math_result,
            }
        )

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/save_backlog', methods=['POST'])
def save_backlog():
    """Save trading data to backlog for analysis."""
    try:
        data = request.get_json()
        backlog_entry = {
            'timestamp': time.time(),
            'data': data,
            'session_id': current_session.session_id,
            'hash': generate_unified_hash([str(data), current_session.session_id]),
        }

        current_session.backlog.append(backlog_entry)
        trading_state['backlog_data'].append(backlog_entry)

        return jsonify(
            {
                'success': True,
                'backlog_id': backlog_entry['hash'],
                'message': 'Data saved to backlog',
            }
        )

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/visualize_trades')
def visualize_trades():
    """Get trade data for visualization."""
    trades = current_session.trades

    # Prepare data for charts
    chart_data = {
        'prices': [t['price'] for t in trades],
        'timestamps': [t['timestamp'] for t in trades],
        'sides': [t['side'] for t in trades],
        'profits': [],
        'cumulative_profit': [],
    }

    # Calculate cumulative profit
    cumulative = 0
    for trade in trades:
        if trade.get('executed'):
            if trade['side'] == 'SELL':
                # Calculate profit from position
                cumulative += trade.get('profit', 0)
        chart_data['profits'].append(trade.get('profit', 0))
        chart_data['cumulative_profit'].append(cumulative)

    return jsonify(chart_data)


@app.route('/api/cli_command', methods=['POST'])
def cli_command():
    """Execute CLI-style commands."""
    try:
        data = request.get_json()
        command = data.get('command', '').strip()

        # Parse and execute commands
        if command.startswith('trade'):
            # Format: trade BTC 60000 BUY 0.1
            parts = command.split()
            if len(parts) >= 5:
                asset = parts[1]
                price = float(parts[2])
                side = parts[3]
                quantity = float(parts[4])

                node = VisualExecutionNode(asset, price)
                result = node.execute()

                return jsonify(
                    {
                        'success': True,
                        'command': command,
                        'result': result,
                        'message': f'Executed {side} order for {quantity} {asset} at ${price}',
                    }
                )

        elif command.startswith('mode'):
            # Format: mode demo/live/backtest
            parts = command.split()
            if len(parts) >= 2:
                new_mode = parts[1]
                trading_state['mode'] = new_mode
                return jsonify({'success': True, 'command': command, 'message': f'Switched to {new_mode} mode'})

        elif command.startswith('portfolio'):
            return jsonify(
                {
                    'success': True,
                    'command': command,
                    'data': {
                        'value': trading_state['portfolio_value'],
                        'profit': trading_state['total_profit'],
                        'positions': current_session.positions,
                    },
                }
            )

        else:
            return jsonify(
                {
                    'success': False,
                    'error': f'Unknown command: {command}',
                    'available_commands': [
                        'trade <asset> <price> <side> <quantity>',
                        'mode <demo|live|backtest>',
                        'portfolio',
                    ],
                }
            )

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/launch_visualizer', methods=['POST'])
def launch_visualizer():
    """Launch a specific visualizer."""
    try:
        data = request.get_json()
        visualizer_id = data.get('visualizer_id', 'main_dashboard')

        success = visualizer_launcher.launch_visualizer(visualizer_id)

        if success:
            visualizer = visualizer_launcher.visualizers[visualizer_id]
            return jsonify(
                {
                    'success': True,
                    'message': f'Launched {visualizer["name"]}',
                    'url': visualizer['url'],
                    'visualizer_id': visualizer_id,
                }
            )
        else:
            return jsonify({'success': False, 'error': f'Failed to launch visualizer: {visualizer_id}'})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/list_visualizers')
def list_visualizers():
    """List all available visualizers."""
    try:
        visualizers = visualizer_launcher.list_visualizers()
        return jsonify({'success': True, 'visualizers': visualizers})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/stop_visualizer', methods=['POST'])
def stop_visualizer():
    """Stop a specific visualizer."""
    try:
        data = request.get_json()
        visualizer_id = data.get('visualizer_id')

        if not visualizer_id:
            return jsonify({'success': False, 'error': 'Visualizer ID required'})

        success = visualizer_launcher.stop_visualizer(visualizer_id)

        return jsonify(
            {
                'success': success,
                'message': (
                    f'Stopped visualizer: {visualizer_id}' if success else f'Visualizer not running: {visualizer_id}'
                ),
            }
        )

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


if __name__ == '__main__':
    print("🚀 Starting Schwabot Flask GUI...")
    print("📊 Dashboard: http://localhost:5000")
    print("🔧 API Endpoints: http://localhost:5000/api/")
    print("💻 CLI Commands: POST to /api/cli_command")
    app.run(debug=True, host='0.0.0.0', port=5000)
