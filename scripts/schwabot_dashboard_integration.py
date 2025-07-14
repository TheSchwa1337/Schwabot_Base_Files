#!/usr/bin/env python3
"""
Schwabot Dashboard Integration
Connects the web dashboard to the executable core

This provides real-time data from the Ferris wheel orchestrator
to the web dashboard, showing live tick cycles, profit tiers,
and trading decisions.
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from flask_socketio import SocketIO, emit

from flask import Flask, jsonify, render_template, request

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from schwabot_executable_core import SchwabotExecutableCore

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'schwabot_dashboard_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global executable core instance
schwabot_core = None


def initialize_schwabot_core():
    """Initialize the Schwabot executable core."""
    global schwabot_core
    try:
        schwabot_core = SchwabotExecutableCore()
        logger.info("✅ Schwabot executable core initialized")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to initialize Schwabot core: {e}")
        return False


@app.route('/')
    def dashboard():
    """Main dashboard page."""
    return render_template('schwabot_dashboard.html')


@app.route('/api/status')
    def get_status():
    """Get current system status."""
    if not schwabot_core:
        return jsonify({"error": "Schwabot core not initialized"}), 500

    try:
        status = schwabot_core.get_system_status()
        return jsonify(status)
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/demo_data')
    def get_demo_data():
    """Get demo data for dashboard."""
    if not schwabot_core:
        return jsonify({"error": "Schwabot core not initialized"}), 500

    try:
        demo_data = schwabot_core.get_demo_data()
        return jsonify(demo_data)
    except Exception as e:
        logger.error(f"Error getting demo data: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/start_ferris_wheel', methods=['POST'])
    def start_ferris_wheel():
    """Start the Ferris wheel orchestrator."""
    if not schwabot_core:
        return jsonify({"error": "Schwabot core not initialized"}), 500

    try:
        schwabot_core.start_ferris_wheel()
        return jsonify({"status": "success", "message": "Ferris wheel started"})
    except Exception as e:
        logger.error(f"Error starting Ferris wheel: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/stop_ferris_wheel', methods=['POST'])
    def stop_ferris_wheel():
    """Stop the Ferris wheel orchestrator."""
    if not schwabot_core:
        return jsonify({"error": "Schwabot core not initialized"}), 500

    try:
        schwabot_core.stop_ferris_wheel()
        return jsonify({"status": "success", "message": "Ferris wheel stopped"})
    except Exception as e:
        logger.error(f"Error stopping Ferris wheel: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/registry_stats')
    def get_registry_stats():
    """Get soulprint registry statistics."""
    if not schwabot_core:
        return jsonify({"error": "Schwabot core not initialized"}), 500

    try:
        stats = schwabot_core.soulprint_registry.get_registry_stats()
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error getting registry stats: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/recent_trades')
    def get_recent_trades():
    """Get recent trades."""
    if not schwabot_core:
        return jsonify({"error": "Schwabot core not initialized"}), 500

    try:
        trades = schwabot_core._get_recent_trades()
        return jsonify(trades)
    except Exception as e:
        logger.error(f"Error getting recent trades: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/active_cycles')
    def get_active_cycles():
    """Get active tick cycles."""
    if not schwabot_core:
        return jsonify({"error": "Schwabot core not initialized"}), 500

    try:
        cycles = []
        for cycle_id, cycle in schwabot_core.ferris_wheel.active_cycles.items():
            cycles.append({)}
                "cycle_id": cycle.cycle_id,
                "tick_number": cycle.tick_number,
                "btc_price": cycle.btc_price,
                "profit_tier": cycle.profit_tier.value,
                "strategy": cycle.strategy_activated,
                "confidence": cycle.confidence_score,
                "profit": cycle.profit_result,
                "is_complete": cycle.is_complete,
                "executed_trades": len(cycle.executed_trades)
            })
        return jsonify(cycles)
    except Exception as e:
        logger.error(f"Error getting active cycles: {e}")
        return jsonify({"error": str(e)}), 500


# SocketIO event handlers
@socketio.on('connect')
    def handle_connect():
    """Handle client connection."""
    logger.info(f"Client connected: {request.sid}")
    emit('status', {'message': 'Connected to Schwabot Dashboard'})


@socketio.on('disconnect')
    def handle_disconnect():
    """Handle client disconnection."""
    logger.info(f"Client disconnected: {request.sid}")


@socketio.on('subscribe_to_updates')
    def handle_subscribe(data):
    """Handle subscription to real-time updates."""
    room = data.get('room', 'general')
    socketio.join_room(room)
    emit('subscribed', {'room': room, 'message': f'Subscribed to {room} updates'})


@socketio.on('unsubscribe_from_updates')
    def handle_unsubscribe(data):
    """Handle unsubscription from real-time updates."""
    room = data.get('room', 'general')
    socketio.leave_room(room)
    emit('unsubscribed', {'room': room, 'message': f'Unsubscribed from {room} updates'})


def broadcast_system_update():
    """Broadcast system updates to all connected clients."""
    if not schwabot_core:
        return

    try:
        # Get current system status
        status = schwabot_core.get_system_status()
        demo_data = schwabot_core.get_demo_data()

        # Broadcast to all clients
        socketio.emit('system_update', {)}
            'status': status,
            'demo_data': demo_data,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error broadcasting system update: {e}")


def broadcast_tick_cycle_update(tick_number: int, profit: float):
    """Broadcast tick cycle updates."""
    socketio.emit('tick_cycle_update', {)}
        'tick_number': tick_number,
        'profit': profit,
        'timestamp': datetime.now().isoformat()
    })


def broadcast_trade_execution(trade_data: Dict[str, Any]):
    """Broadcast trade execution updates."""
    socketio.emit('trade_execution', {)}
        'trade': trade_data,
        'timestamp': datetime.now().isoformat()
    })


# Background task to send periodic updates
    def background_updates():
    """Background task to send periodic updates."""
    while True:
        try:
            broadcast_system_update()
            time.sleep(5)  # Update every 5 seconds
        except Exception as e:
            logger.error(f"Error in background updates: {e}")
            time.sleep(10)


# Create dashboard HTML template
    def create_dashboard_template():
    """Create the dashboard HTML template."""
    template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🚀 Schwabot Trading Dashboard</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {}
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {}
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            min-height: 100vh;
        }

        .container {}
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }

        .header {}
            text-align: center;
            margin-bottom: 30px;
        }

        .header h1 {}
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }

        .header p {}
            font-size: 1.2em;
            opacity: 0.9;
        }

        .dashboard-grid {}
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }

        .card {}
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 20px;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }

        .card h3 {}
            margin-bottom: 15px;
            color: #4CAF50;
            font-size: 1.3em;
        }

        .status-indicator {}
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }

        .status-running {}
            background-color: #4CAF50;
            animation: pulse 2s infinite;
        }

        .status-stopped {}
            background-color: #f44336;
        }

        @keyframes pulse {}
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }

        .metric {}
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
            padding: 8px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }

        .metric:last-child {}
            border-bottom: none;
        }

        .metric-label {}
            font-weight: 500;
        }

        .metric-value {}
            font-weight: bold;
            color: #4CAF50;
        }

        .controls {}
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }

        .btn {}
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 1em;
            font-weight: 600;
            transition: all 0.3s ease;
        }

        .btn-primary {}
            background: linear-gradient(45deg, #4CAF50, #45a049);
            color: white;
        }

        .btn-danger {}
            background: linear-gradient(45deg, #f44336, #da190b);
            color: white;
        }

        .btn:hover {}
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }

        .chart-container {}
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
        }

        .trades-list {}
            max-height: 300px;
            overflow-y: auto;
        }

        .trade-item {}
            display: flex;
            justify-content: space-between;
            padding: 10px;
            margin-bottom: 5px;
            background: rgba(255, 255, 255, 0.5);
            border-radius: 8px;
            border-left: 4px solid #4CAF50;
        }

        .trade-buy {}
            border-left-color: #4CAF50;
        }

        .trade-sell {}
            border-left-color: #f44336;
        }

        .profit-positive {}
            color: #4CAF50;
        }

        .profit-negative {}
            color: #f44336;
        }

        .loading {}
            text-align: center;
            padding: 20px;
            opacity: 0.7;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Schwabot Trading Dashboard</h1>
            <p>Advanced Algorithmic Trading Intelligence System</p>
        </div>

        <div class="controls">
            <button class="btn btn-primary" onclick="startFerrisWheel()">🎡 Start Ferris Wheel</button>
            <button class="btn btn-danger" onclick="stopFerrisWheel()">🛑 Stop Ferris Wheel</button>
            <button class="btn btn-primary" onclick="refreshData()">🔄 Refresh Data</button>
        </div>

        <div class="dashboard-grid">
            <div class="card">
                <h3>🎡 Ferris Wheel Status</h3>
                <div class="metric">
                    <span class="metric-label">Status:</span>
                    <span class="metric-value">
                        <span id="status-indicator" class="status-indicator status-stopped"></span>
                        <span id="status-text">Stopped</span>
                    </span>
                </div>
                <div class="metric">
                    <span class="metric-label">Current Tick:</span>
                    <span class="metric-value" id="current-tick">0/16</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Total Profit:</span>
                    <span class="metric-value" id="total-profit">$0.0</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Active Cycles:</span>
                    <span class="metric-value" id="active-cycles">0</span>
                </div>
            </div>

            <div class="card">
                <h3>💰 Profit Tier Navigation</h3>
                <div class="metric">
                    <span class="metric-label">Current Tier:</span>
                    <span class="metric-value" id="current-tier">Tier 2</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Confidence:</span>
                    <span class="metric-value" id="confidence">0.0</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Strategy:</span>
                    <span class="metric-value" id="strategy">None</span>
                </div>
            </div>

            <div class="card">
                <h3>🧠 Soulprint Registry</h3>
                <div class="metric">
                    <span class="metric-label">Total Entries:</span>
                    <span class="metric-value" id="total-entries">0</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Executed:</span>
                    <span class="metric-value" id="executed-entries">0</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Profitable:</span>
                    <span class="metric-value" id="profitable-entries">0</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Avg Confidence:</span>
                    <span class="metric-value" id="avg-confidence">0.0</span>
                </div>
            </div>
        </div>

        <div class="chart-container">
            <h3>📊 Recent Trades</h3>
            <div id="trades-container" class="trades-list">
                <div class="loading">Loading trades...</div>
            </div>
        </div>

        <div class="chart-container">
            <h3>📈 Profit Chart</h3>
            <canvas id="profitChart" width="400" height="200"></canvas>
        </div>
    </div>

    <script>
        // SocketIO connection
        const socket = io();

        // Chart instance
        let profitChart = null;

        // Initialize dashboard
        document.addEventListener('DOMContentLoaded', function() {)}
            initializeSocketIO();
            refreshData();
            initializeChart();
        });

        function initializeSocketIO() {}
            socket.on('connect', function() {)}
                console.log('Connected to Schwabot Dashboard');
                socket.emit('subscribe_to_updates', {room: 'general'});
            });

            socket.on('system_update', function(data) {)}
                updateDashboard(data);
            });

            socket.on('tick_cycle_update', function(data) {)}
                updateTickCycle(data);
            });

            socket.on('trade_execution', function(data) {)}
                addTradeToChart(data.trade);
                refreshTrades();
            });
        }

        function updateDashboard(data) {}
            const status = data.status;
            const demo = data.demo_data;

            // Update Ferris Wheel status
            document.getElementById('status-text').textContent = 
                status.ferris_wheel.is_running ? 'Running' : 'Stopped';
            document.getElementById('status-indicator').className = 
                'status-indicator ' + (status.ferris_wheel.is_running ? 'status-running' : 'status-stopped');

            document.getElementById('current-tick').textContent = 
                `${status.ferris_wheel.current_tick}/${status.ferris_wheel.total_ticks}`;
            document.getElementById('total-profit').textContent = 
                `$${status.ferris_wheel.total_profit.toFixed(2)}`;
            document.getElementById('active-cycles').textContent = 
                status.ferris_wheel.active_cycles;

            // Update Profit Tier
            document.getElementById('current-tier').textContent = 
                demo.profit_tier.replace('_', ' ').toUpperCase();
            document.getElementById('confidence').textContent = 
                demo.registry_stats.avg_confidence.toFixed(2);

            // Update Registry stats
            document.getElementById('total-entries').textContent = 
                demo.registry_stats.total_entries;
            document.getElementById('executed-entries').textContent = 
                demo.registry_stats.executed_entries;
            document.getElementById('profitable-entries').textContent = 
                demo.registry_stats.profitable_entries;
            document.getElementById('avg-confidence').textContent = 
                demo.registry_stats.avg_confidence.toFixed(2);
        }

        function updateTickCycle(data) {}
            document.getElementById('current-tick').textContent = 
                `${data.tick_number}/16`;
        }

        function refreshData() {}
            fetch('/api/demo_data')
                .then(response => response.json())
                .then(data => {)}
                    updateDashboard({status: {ferris_wheel: data}, demo_data: data});
                    refreshTrades();
                })
                .catch(error => console.error('Error refreshing data:', error));
        }

        function refreshTrades() {}
            fetch('/api/recent_trades')
                .then(response => response.json())
                .then(trades => {)}
                    const container = document.getElementById('trades-container');
                    container.innerHTML = '';

                    if (trades.length === 0) {}
                        container.innerHTML = '<div class="loading">No trades yet</div>';
                        return;
                    }

                    trades.forEach(trade => {)}
                        const tradeDiv = document.createElement('div');
                        tradeDiv.className = `trade-item trade-${trade.action.toLowerCase()}`;
                        tradeDiv.innerHTML = `
                            <div>
                                <strong>${trade.pair}</strong> - ${trade.action}
                                <br><small>$${trade.price.toFixed(2)} at ${trade.timestamp}</small>
                            </div>
                            <div class="profit-${trade.profit >= 0 ? 'positive' : 'negative'}">
                                $${trade.profit.toFixed(2)}
                            </div>
                        `;
                        container.appendChild(tradeDiv);
                    });
                })
                .catch(error => console.error('Error refreshing trades:', error));
        }

        function startFerrisWheel() {}
            fetch('/api/start_ferris_wheel', {method: 'POST'})
                .then(response => response.json())
                .then(data => {)}
                    if (data.status === 'success') {}
                        console.log('Ferris wheel started');
                        refreshData();
                    } else {
                        console.error('Failed to start Ferris wheel:', data.error);
                    }
                })
                .catch(error => console.error('Error starting Ferris wheel:', error));
        }

        function stopFerrisWheel() {}
            fetch('/api/stop_ferris_wheel', {method: 'POST'})
                .then(response => response.json())
                .then(data => {)}
                    if (data.status === 'success') {}
                        console.log('Ferris wheel stopped');
                        refreshData();
                    } else {
                        console.error('Failed to stop Ferris wheel:', data.error);
                    }
                })
                .catch(error => console.error('Error stopping Ferris wheel:', error));
        }

        function initializeChart() {}
            const ctx = document.getElementById('profitChart').getContext('2d');
            profitChart = new Chart(ctx, {)}
                type: 'line',
                data: {}
                    labels: [],
                    datasets: [{}]
                        label: 'Total Profit',
                        data: [],
                        borderColor: '#4CAF50',
                        backgroundColor: 'rgba(76, 175, 80, 0.1)',
                        tension: 0.4
                    }]
                },
                options: {}
                    responsive: true,
                    scales: {}
                        y: {}
                            beginAtZero: true,
                            grid: {}
                                color: 'rgba(255, 255, 255, 0.1)'
                            },
                            ticks: {}
                                color: 'white'
                            }
                        },
                        x: {}
                            grid: {}
                                color: 'rgba(255, 255, 255, 0.1)'
                            },
                            ticks: {}
                                color: 'white'
                            }
                        }
                    },
                    plugins: {}
                        legend: {}
                            labels: {}
                                color: 'white'
                            }
                        }
                    }
                }
            });
        }

        function addTradeToChart(trade) {}
            if (!profitChart) return;

            const timestamp = new Date(trade.timestamp * 1000).toLocaleTimeString();
            profitChart.data.labels.push(timestamp);
            profitChart.data.datasets[0].data.push(trade.profit);

            // Keep only last 20 data points
            if (profitChart.data.labels.length > 20) {}
                profitChart.data.labels.shift();
                profitChart.data.datasets[0].data.shift();
            }

            profitChart.update();
        }

        // Auto-refresh every 5 seconds
        setInterval(refreshData, 5000);
    </script>
</body>
</html>
    """

    # Create templates directory if it doesn't exist'
    os.makedirs('templates', exist_ok=True)

    # Write template file
    with open('templates/schwabot_dashboard.html', 'w') as f:
        f.write(template)

    logger.info("✅ Dashboard template created")


def main():
    """Main function to run the dashboard."""
    print("🚀 Starting Schwabot Dashboard Integration...")

    # Create dashboard template
    create_dashboard_template()

    # Initialize Schwabot core
    if not initialize_schwabot_core():
        print("❌ Failed to initialize Schwabot core")
        return

    print("✅ Schwabot core initialized")
    print("🌐 Dashboard available at: http://localhost:5000")
    print("📡 Real-time updates enabled")

    # Start background updates thread
    import threading
    update_thread = threading.Thread(target=background_updates, daemon=True)
    update_thread.start()

    # Run the Flask app
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)


if __name__ == "__main__":
    main() 