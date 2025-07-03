#!/usr/bin/env python3
# update
"""
Schwabot Unified Launcher
=========================

Main entry point for Schwabot trading bot system with secure API key management
and integration with existing mathematical framework.
"""

from core.trading_engine_integration import SchwabotTradingEngine, TradingMode
from core.lantern_core_integration import (
    start_lantern_core,
    stop_lantern_core,
    get_lantern_core_status,
)
from utils.price_bridge import get_secure_price
from utils.market_data_utils import create_market_snapshot
from utils.secure_config_manager import SecureConfigManager, get_secure_api_key
import os
import sys
import threading
import asyncio
from typing import Dict, Any
from flask import Flask, render_template, request, jsonify

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24).hex()

# Initialize secure config manager
secure_config = SecureConfigManager()


class SchwabotLauncher:
    """Main launcher class for Schwabot trading bot system."""

    def __init__(self):
        self.secure_config = secure_config
        self.system_status = {
            "api_keys_configured": False,
            "market_data_available": False,
            "trading_engine_ready": False,
            "last_market_snapshot": None,
        }
        self.update_system_status()

    def update_system_status(self):
        """Update system status based on current configuration."""
        # Check API keys
        required_keys = ["NEWS_API", "COINMARKETCAP_API", "CCXT_API", "COINBASE_API"]
        configured_keys = self.secure_config.list_stored_services()

        self.system_status["api_keys_configured"] = all(
            key in configured_keys for key in required_keys
        )

        # Check market data availability
        if self.system_status["api_keys_configured"]:
            try:
                snapshot = create_market_snapshot()
                self.system_status["market_data_available"] = snapshot is not None
                self.system_status["last_market_snapshot"] = snapshot
            except Exception as e:
                print(f"Error checking market data: {e}")
                self.system_status["market_data_available"] = False

    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        self.update_system_status()
        return self.system_status


# Initialize launcher
launcher = SchwabotLauncher()


@app.route("/")
def index():
    """Main dashboard page."""
    status = launcher.get_system_status()
    return render_template("dashboard.html", status=status)


@app.route("/api/status")
def api_status():
    """API endpoint for system status."""
    return jsonify(launcher.get_system_status())


@app.route("/setup")
def setup_page():
    """API key setup page."""
    configured_keys = secure_config.list_stored_services()
    return render_template("setup.html", configured_keys=configured_keys)


@app.route("/api/setup", methods=["POST"])
def setup_api_key():
    """API endpoint for setting up API keys."""
    try:
        data = request.get_json()
        service_name = data.get("service")
        api_key = data.get("api_key")

        if not service_name or not api_key:
            return jsonify(
                {"success": False, "error": "Missing service name or API key"}
            )

        # Store the API key securely
        success = secure_config.store_api_key(
            service_name, f"Enter {service_name} API key"
        )

        if success:
            launcher.update_system_status()
            return jsonify(
                {"success": True, "message": f"{service_name} API key stored securely"}
            )
        else:
            return jsonify(
                {"success": False, "error": f"Failed to store {service_name} API key"}
            )

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/market-snapshot")
def get_market_snapshot():
    """API endpoint for getting current market snapshot."""
    try:
        snapshot = create_market_snapshot()
        if snapshot:
            return jsonify({"success": True, "data": snapshot})
        else:
            return jsonify(
                {"success": False, "error": "Failed to create market snapshot"}
            )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/configured-keys")
def get_configured_keys():
    """API endpoint for getting list of configured API keys."""
    keys = secure_config.list_stored_services()
    return jsonify({"keys": keys})


@app.route("/trading")
def trading_dashboard():
    """Trading dashboard page."""
    status = launcher.get_system_status()
    return render_template("trading.html", status=status)


@app.route("/visualization")
def visualization_dashboard():
    """Visualization dashboard page."""
    return render_template("visualization.html")


@app.route("/api/test-connection/<service>")
def test_connection(service):
    """Test API connection for a specific service."""
    try:
        if service == "news":
            from utils.market_data_utils import pull_news_headlines

            headlines = pull_news_headlines()
            success = len(headlines) > 0
            return jsonify(
                {
                    "success": success,
                    "data": headlines[:3] if success else [],
                    "message": (
                        f"Found {len(headlines)} headlines"
                        if success
                        else "No headlines found"
                    ),
                }
            )
        elif service == "price":
            # Test price bridge
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            price_data = loop.run_until_complete(get_secure_price("BTC"))
            loop.close()

            success = price_data is not None
            return jsonify(
                {
                    "success": success,
                    "data": price_data.to_dict() if success else {},
                    "message": (
                        f"Price: ${price_data.price:,.2f} ({price_data.source})"
                        if success
                        else "Price data unavailable"
                    ),
                }
            )
        elif service == "coinmarketcap":
            # Test CoinMarketCap specifically
            api_key = get_secure_api_key("COINMARKETCAP_API")
            if not api_key:
                return jsonify(
                    {"success": False, "error": "CoinMarketCap API key not configured"}
                )

            # Test with a simple request
            import requests

            url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest"
            headers = {"X-CMC_PRO_API_KEY": api_key}
            params = {"symbol": "BTC", "convert": "USD"}

            response = requests.get(url, headers=headers, params=params, timeout=10)
            success = response.status_code == 200

            if success:
                data = response.json()
                btc_price = data["data"]["BTC"]["quote"]["USD"]["price"]
                return jsonify(
                    {
                        "success": True,
                        "data": {"price": btc_price},
                        "message": f"CoinMarketCap working: ${btc_price:,.2f}",
                    }
                )
            else:
                return jsonify(
                    {
                        "success": False,
                        "error": f"CoinMarketCap API error: {response.status_code}",
                    }
                )
        elif service == "lantern_core":
            # Test Lantern Core integration
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            status = loop.run_until_complete(get_lantern_core_status())
            loop.close()

            success = "error" not in status
            return jsonify(
                {
                    "success": success,
                    "data": status,
                    "message": (
                        "Lantern Core integration working"
                        if success
                        else "Lantern Core integration failed"
                    ),
                }
            )
        else:
            return jsonify({"success": False, "error": f"Unknown service: {service}"})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/start-lantern-core")
def api_start_lantern_core():
    """Start Lantern Core integration."""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        success = loop.run_until_complete(start_lantern_core())
        loop.close()

        return jsonify(
            {
                "success": success,
                "message": (
                    "Lantern Core started successfully"
                    if success
                    else "Failed to start Lantern Core"
                ),
            }
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/stop-lantern-core")
def api_stop_lantern_core():
    """Stop Lantern Core integration."""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(stop_lantern_core())
        loop.close()

        return jsonify(
            {"success": True, "message": "Lantern Core stopped successfully"}
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/lantern-core-status")
def api_lantern_core_status():
    """Get Lantern Core status."""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        status = loop.run_until_complete(get_lantern_core_status())
        loop.close()

        return jsonify({"success": True, "data": status})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/start-trading/<mode>")
def api_start_trading(mode):
    """Start trading engine in specified mode."""
    try:
        if mode not in ["demo", "live", "simulation"]:
            return jsonify(
                {
                    "success": False,
                    "error": "Invalid mode. Use: demo, live, or simulation",
                }
            )

        trading_mode = TradingMode(mode)

        # Initialize trading engine
        engine = SchwabotTradingEngine(trading_mode)

        # Start trading in background
        def start_trading_loop():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(engine.start_trading())
            loop.close()

        thread = threading.Thread(target=start_trading_loop, daemon=True)
        thread.start()

        return jsonify(
            {"success": True, "message": f"Trading engine started in {mode} mode"}
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/load-historical-data", methods=["POST"])
def api_load_historical_data():
    """Load historical data from CSV file."""
    try:
        data = request.get_json()
        csv_file_path = data.get("csv_file_path")

        if not csv_file_path:
            return jsonify({"success": False, "error": "CSV file path required"})

        # Load historical data
        from core.lantern_core_integration import lantern_core

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        success = loop.run_until_complete(
            lantern_core.load_historical_data(csv_file_path)
        )
        loop.close()

        return jsonify(
            {
                "success": success,
                "message": (
                    "Historical data loaded successfully"
                    if success
                    else "Failed to load historical data"
                ),
            }
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


def create_templates():
    """Create HTML templates for the web interface."""
    dashboard_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Schwabot Trading Bot Dashboard</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background: #1a1a1a; color: #fff; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 15px; }
        .status-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .status-card { background: #2a2a2a; padding: 20px; border-radius: 10px; border-left: 5px solid #4CAF50; }
        .status-card.warning { border-left-color: #FF9800; }
        .status-card.error { border-left-color: #f44336; }
        .btn { padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; text-decoration: none; display: inline-block; margin: 5px; transition: all 0.3s; }
        .btn-primary { background: #4CAF50; color: white; }
        .btn-warning { background: #FF9800; color: white; }
        .btn-secondary { background: #2196F3; color: white; }
        .btn:hover { opacity: 0.8; transform: translateY(-2px); }
        .nav-buttons { text-align: center; margin-bottom: 30px; }
        .status-indicator { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 10px; }
        .status-good { background: #4CAF50; }
        .status-warning { background: #FF9800; }
        .status-bad { background: #f44336; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Schwabot Trading Bot Launcher</h1>
            <p>Unified control center for Schwabot's biological immune system</p>
        </div>

        <div class="nav-buttons">
            <a href="/" class="btn btn-primary">Dashboard</a>
            <a href="/setup" class="btn btn-secondary">API Setup</a>
            <a href="/trading" class="btn btn-warning">Trading</a>
            <a href="/visualization" class="btn btn-secondary">Visualization</a>
        </div>

        <div class="status-grid">
            <div class="status-card {% if not status.api_keys_configured %}error{% endif %}">
                <h3>🔐 API Configuration</h3>
                <p><span class="status-indicator {% if status.api_keys_configured %}status-good{% else %}status-bad{% endif %}"></span>
                   {% if status.api_keys_configured %}All API keys configured{% else %}API keys need setup{% endif %}</p>
            </div>

            <div class="status-card {% if not status.market_data_available %}error{% endif %}">
                <h3>📊 Market Data</h3>
                <p><span class="status-indicator {% if status.market_data_available %}status-good{% else %}status-bad{% endif %}"></span>
                   {% if status.market_data_available %}Market data available{% else %}Market data unavailable{% endif %}</p>
            </div>

            <div class="status-card {% if not status.trading_engine_ready %}warning{% endif %}">
                <h3>🤖 Trading Engine</h3>
                <p><span class="status-indicator {% if status.trading_engine_ready %}status-good{% else %}status-warning{% endif %}"></span>
                   {% if status.trading_engine_ready %}Trading engine ready{% else %}Trading engine initializing{% endif %}</p>
            </div>

            <div class="status-card" id="lantern-core-status">
                <h3>🏮 Lantern Core</h3>
                <p><span class="status-indicator status-warning"></span>Checking status...</p>
            </div>
        </div>

        <div class="control-panel" style="background: #2a2a2a; padding: 20px; border-radius: 10px; margin: 20px 0;">
            <h3>🎛️ Control Panel</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-top: 15px;">
                <button class="btn btn-primary" onclick="startLanternCore()">Start Lantern Core</button>
                <button class="btn btn-warning" onclick="stopLanternCore()">Stop Lantern Core</button>
                <button class="btn btn-secondary" onclick="startTrading('demo')">Start Demo Trading</button>
                <button class="btn btn-secondary" onclick="startTrading('simulation')">Start Simulation</button>
                <button class="btn btn-warning" onclick="startTrading('live')">Start Live Trading</button>
            </div>
        </div>

        <div id="market-snapshot" style="background: #2a2a2a; padding: 20px; border-radius: 10px;">
            <h3>📈 Latest Market Snapshot</h3>
            <div id="snapshot-content">Loading...</div>
        </div>

        <div id="lantern-core-data" style="background: #2a2a2a; padding: 20px; border-radius: 10px; margin-top: 20px;">
            <h3>🏮 Lantern Core Data</h3>
            <div id="lantern-core-content">Loading...</div>
        </div>
    </div>

    <script>
        // Auto-refresh market snapshot
        function updateMarketSnapshot() {
            fetch('/api/market-snapshot')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        const snapshot = data.data;
                        document.getElementById('snapshot-content').innerHTML = `
                            <p><strong>BTC Price:</strong> $${snapshot.price_data.price.toLocaleString()}</p>
                            <p><strong>Market Hash:</strong> ${snapshot.market_hash.substring(0, 16)}...</p>
                            <p><strong>News Headlines:</strong> ${snapshot.news_headlines.length} articles</p>
                        `;
                    } else {
                        document.getElementById('snapshot-content').innerHTML = '<p style="color: #f44336;">Failed to load market data</p>';
                    }
                })
                .catch(error => {
                    document.getElementById('snapshot-content').innerHTML = '<p style="color: #f44336;">Error loading market data</p>';
                });
        }

        // Update Lantern Core status
        function updateLanternCoreStatus() {
            fetch('/api/lantern-core-status')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        const status = data.data;
                        const isRunning = status.lantern_core?.is_running || false;
                        const isInitialized = status.lantern_core?.is_initialized || false;

                        const statusDiv = document.getElementById('lantern-core-status');
                        const indicator = statusDiv.querySelector('.status-indicator');
                        const text = statusDiv.querySelector('p');

                        if (isRunning && isInitialized) {
                            indicator.className = 'status-indicator status-good';
                            text.innerHTML = '<span class="status-indicator status-good"></span>Lantern Core running';
                        } else if (isInitialized) {
                            indicator.className = 'status-indicator status-warning';
                            text.innerHTML = '<span class="status-indicator status-warning"></span>Lantern Core stopped';
                        } else {
                            indicator.className = 'status-indicator status-bad';
                            text.innerHTML = '<span class="status-indicator status-bad"></span>Lantern Core not initialized';
                        }

                        // Update detailed data
                        document.getElementById('lantern-core-content').innerHTML = `
                            <p><strong>Status:</strong> ${isRunning ? 'Running' : 'Stopped'}</p>
                            <p><strong>Operations:</strong> ${status.lantern_core?.total_operations || 0}</p>
                            <p><strong>Success Rate:</strong> ${((status.lantern_core?.successful_operations || 0) / Math.max(status.lantern_core?.total_operations || 1, 1) * 100).toFixed(1)}%</p>
                            <p><strong>Avg Response Time:</strong> ${(status.lantern_core?.avg_response_time || 0).toFixed(3)}s</p>
                        `;
                    }
                })
                .catch(error => {
                    console.error('Error updating Lantern Core status:', error);
                });
        }

        // Control functions
        function startLanternCore() {
            fetch('/api/start-lantern-core')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        alert('Lantern Core started successfully!');
                        updateLanternCoreStatus();
                    } else {
                        alert('Failed to start Lantern Core: ' + data.error);
                    }
                })
                .catch(error => {
                    alert('Error starting Lantern Core: ' + error);
                });
        }

        function stopLanternCore() {
            fetch('/api/stop-lantern-core')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        alert('Lantern Core stopped successfully!');
                        updateLanternCoreStatus();
                    } else {
                        alert('Failed to stop Lantern Core: ' + data.error);
                    }
                })
                .catch(error => {
                    alert('Error stopping Lantern Core: ' + error);
                });
        }

        function startTrading(mode) {
            if (mode === 'live') {
                if (!confirm('Are you sure you want to start LIVE trading? This will execute real trades!')) {
                    return;
                }
            }

            fetch(`/api/start-trading/${mode}`)
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        alert(`Trading engine started in ${mode} mode!`);
                    } else {
                        alert('Failed to start trading: ' + data.error);
                    }
                })
                .catch(error => {
                    alert('Error starting trading: ' + error);
                });
        }

        // Update every 30 seconds
        updateMarketSnapshot();
        updateLanternCoreStatus();
        setInterval(updateMarketSnapshot, 30000);
        setInterval(updateLanternCoreStatus, 30000);
    </script>
</body>
</html>"""

    # Setup template
    setup_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>API Setup - Schwabot Launcher</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background: #1a1a1a; color: #fff; }
        .container { max-width: 800px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 30px; }
        .setup-form { background: #2a2a2a; padding: 30px; border-radius: 10px; }
        .form-group { margin-bottom: 20px; }
        label { display: block; margin-bottom: 5px; font-weight: bold; }
        input[type="text"], input[type="password"] { width: 100%; padding: 10px; border: 1px solid #444; border-radius: 5px; background: #333; color: #fff; }
        .btn { padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; background: #4CAF50; color: white; }
        .btn:hover { background: #45a049; }
        .alert { padding: 10px; margin: 10px 0; border-radius: 5px; }
        .alert-success { background: #4CAF50; }
        .alert-error { background: #f44336; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>API Setup - Schwabot</h1>
            <p>Configure your API keys for trading exchanges and data providers</p>
        </div>
        <div class="setup-form">
            <form id="setup-form">
                <div class="form-group">
                    <label for="coinbase_api_key">Coinbase API Key:</label>
                    <input type="text" id="coinbase_api_key" name="coinbase_api_key" placeholder="Enter your Coinbase API key">
                </div>
                <div class="form-group">
                    <label for="coinbase_secret">Coinbase Secret:</label>
                    <input type="password" id="coinbase_secret" name="coinbase_secret" placeholder="Enter your Coinbase secret">
                </div>
                <div class="form-group">
                    <label for="news_api_key">News API Key:</label>
                    <input type="text" id="news_api_key" name="news_api_key" placeholder="Enter your News API key">
                </div>
                <button type="submit" class="btn">Save Configuration</button>
            </form>
            <div id="status-message"></div>
        </div>
    </div>
    <script>
        document.getElementById('setup-form').addEventListener('submit', function(e) {
            e.preventDefault();
            const formData = new FormData(e.target);
            const data = Object.fromEntries(formData);

            fetch('/api/setup', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            })
            .then(response => response.json())
            .then(data => {
                const statusDiv = document.getElementById('status-message');
                if (data.success) {
                    statusDiv.innerHTML = '<div class="alert alert-success">Configuration saved successfully!</div>';
                } else {
                    statusDiv.innerHTML = '<div class="alert alert-error">Error: ' + data.error + '</div>';
                }
            });
        });
    </script>
</body>
</html>"""

    # Write templates with UTF-8 encoding
    os.makedirs("templates", exist_ok=True)

    with open("templates/dashboard.html", "w", encoding="utf-8") as f:
        f.write(dashboard_html)

    with open("templates/setup.html", "w", encoding="utf-8") as f:
        f.write(setup_html)

    print("✅ HTML templates created successfully with UTF-8 encoding")


def main():
    """Main entry point for the Schwabot launcher."""
    print("🚀 Starting Schwabot Unified Launcher...")

    # Create templates if they don't exist
    create_templates()

    # Check if API keys are configured
    configured_keys = secure_config.list_stored_services()
    required_keys = ["NEWS_API", "COINMARKETCAP_API", "CCXT_API", "COINBASE_API"]

    missing_keys = [key for key in required_keys if key not in configured_keys]

    if missing_keys:
        print(f"⚠️  Missing API keys: {', '.join(missing_keys)}")
        print("   Please visit http://localhost:5000/setup to configure API keys")
    else:
        print("✅ All required API keys configured")

    print("🌐 Launcher available at: http://localhost:5000")
    print("🔐 API Setup at: http://localhost:5000/setup")

    # Start Flask app
    app.run(host="0.0.0.0", port=5000, debug=True)


if __name__ == "__main__":
    main()
