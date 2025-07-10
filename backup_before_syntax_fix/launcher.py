#!/usr/bin/env python3
# update
"""
Schwabot Unified Launcher
=========================

Main entry point for Schwabot trading bot system with secure API key management
and integration with existing mathematical framework.
"""

import asyncio
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import requests
from werkzeug.utils import secure_filename

from flask import Flask, jsonify, render_template, request

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.market_data_utils import create_market_snapshot, pull_news_headlines
from utils.price_bridge import get_secure_price
from utils.secure_config_manager import SecureConfigManager, get_secure_api_key

    start_lantern_core,
    stop_lantern_core,
    get_lantern_core_status,
    lantern_core,
)
from core.quad_bit_strategy_array import QuadBitStrategyArray, create_quad_bit_strategy_array
from core.trading_engine_integration import SchwabotTradingEngine, TradingMode

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24).hex()

# File upload configuration
UPLOAD_FOLDER = Path('data/historical')
ALLOWED_EXTENSIONS = {'csv'}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB max file size

app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Ensure upload directories exist
    for pair in ['btc_usdc', 'xrp_usdc', 'sol_usdc', 'eth_usdc']:
    (UPLOAD_FOLDER / pair).mkdir(parents=True, exist_ok=True)

# Initialize quad-bit strategy array
quad_bit_strategy = None

def initialize_quad_bit_strategy():
    """Initialize the quad-bit strategy array."""
    global quad_bit_strategy
    try:
        trading_engine = SchwabotTradingEngine(TradingMode.SIMULATION)
        quad_bit_strategy = create_quad_bit_strategy_array(trading_engine)
        logger.info("Quad-bit strategy array initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize quad-bit strategy array: {e}")
        return False

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_csv_format(file_path):
    """Validate CSV format for cryptocurrency data."""
    try:
        df = pd.read_csv(file_path, nrows=5)  # Read first 5 rows to check format
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']

        # Check if all required columns exist
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"

        # Check if timestamp column can be parsed
        try:
            pd.to_datetime(df['timestamp'].iloc[0])
        except:
            return False, "Timestamp column cannot be parsed"

        return True, "Valid CSV format"
    except Exception as e:
        return False, f"Error reading CSV: {str(e)}"

# Initialize secure config manager
secure_config = SecureConfigManager()


class SchwabotLauncher:
    """Main launcher class for Schwabot trading bot system."""

    def __init__(self):
        self.secure_config = secure_config
        self.system_status = {}
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

        self.system_status["api_keys_configured"] = all()
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
            return jsonify()
                {"success": False, "error": "Missing service name or API key"}
            )

        # Store the API key securely
        success = secure_config.store_api_key()
            service_name, f"Enter {service_name} API key"
        )

        if success:
            launcher.update_system_status()
            return jsonify()
                {"success": True, "message": f"{service_name} API key stored securely"}
            )
        else:
            return jsonify()
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
            return jsonify()
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


@app.route("/data-upload")
    def data_upload_page():
    """Data upload page for historical cryptocurrency data."""
    return render_template("data_upload.html")


@app.route("/api/test-connection/<service>")
    def test_connection(service):
    """Test API connection for a specific service."""
    try:
        if service == "news":

            headlines = pull_news_headlines()
            success = len(headlines) > 0
            return jsonify()
                {}
                    "success": success,
                    "data": headlines[:3] if success else [],
                    "message": ()
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
            return jsonify()
                {}
                    "success": success,
                    "data": price_data.to_dict() if success else {},
                    "message": ()
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
                return jsonify()
                    {"success": False, "error": "CoinMarketCap API key not configured"}
                )

            # Test with a simple request

            url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest"
            headers = {"X-CMC_PRO_API_KEY": api_key}
            params = {"symbol": "BTC", "convert": "USD"}

            response = requests.get(url, headers=headers, params=params, timeout=10)
            success = response.status_code == 200

            if success:
                data = response.json()
                btc_price = data["data"]["BTC"]["quote"]["USD"]["price"]
                return jsonify()
                    {}
                        "success": True,
                        "data": {"price": btc_price},
                        "message": f"CoinMarketCap working: ${btc_price:,.2f}",
                    }
                )
            else:
                return jsonify()
                    {}
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
            return jsonify()
                {}
                    "success": success,
                    "data": status,
                    "message": ()
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

        return jsonify()
            {}
                "success": success,
                "message": ()
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

        return jsonify()
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
            return jsonify()
                {}
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

        return jsonify()
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

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        success = loop.run_until_complete()
            lantern_core.load_historical_data(csv_file_path)
        )
        loop.close()

        return jsonify()
            {}
                "success": success,
                "message": ()
                    "Historical data loaded successfully"
                    if success
                    else "Failed to load historical data"
                ),
            }
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/upload-historical-data", methods=["POST"])
    def api_upload_historical_data():
    """API endpoint for uploading historical cryptocurrency data."""
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "No file uploaded"})

        file = request.files['file']
        if file.filename == '':
            return jsonify({"success": False, "error": "No file selected"})

        # Get trading pair from form data
        trading_pair = request.form.get('trading_pair', 'btc_usdc').lower()
        if trading_pair not in ['btc_usdc', 'xrp_usdc', 'sol_usdc', 'eth_usdc']:
            return jsonify({"success": False, "error": "Invalid trading pair"})

        # Check file extension
        if not allowed_file(file.filename):
            return jsonify({"success": False, "error": "Only CSV files are allowed"})

        # Secure filename and save
        filename = secure_filename(file.filename)
        upload_dir = UPLOAD_FOLDER / trading_pair
        file_path = upload_dir / filename

        # Save file
        file.save(str(file_path))

        # Validate CSV format
        is_valid, message = validate_csv_format(file_path)
        if not is_valid:
            # Remove invalid file
            file_path.unlink()
            return jsonify({"success": False, "error": f"Invalid CSV format: {message}"})

        # Process and integrate the data
        success = process_uploaded_data(file_path, trading_pair)

        if success:
            return jsonify({)}
                "success": True, 
                "message": f"Successfully uploaded {filename} for {trading_pair.upper()}",
                "file_path": str(file_path),
                "trading_pair": trading_pair
            })
        else:
            return jsonify({"success": False, "error": "Failed to process uploaded data"})

    except Exception as e:
        logging.error(f"Error uploading file: {e}")
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/list-uploaded-files")
    def api_list_uploaded_files():
    """API endpoint for listing uploaded historical data files."""
    try:
        files_data = {}

        for pair in ['btc_usdc', 'xrp_usdc', 'sol_usdc', 'eth_usdc']:
            pair_dir = UPLOAD_FOLDER / pair
            if pair_dir.exists():
                files = []
                for file_path in pair_dir.glob('*.csv'):
                    stat = file_path.stat()
                    files.append({)}
                        'filename': file_path.name,
                        'size_mb': round(stat.st_size / (1024 * 1024), 2),
                        'upload_date': pd.Timestamp(stat.st_mtime, unit='s').strftime('%Y-%m-%d %H:%M:%S'),
                        'file_path': str(file_path)
                    })
                files_data[pair] = sorted(files, key=lambda x: x['upload_date'], reverse=True)
            else:
                files_data[pair] = []

        return jsonify({"success": True, "files": files_data})

    except Exception as e:
        logging.error(f"Error listing files: {e}")
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/delete-uploaded-file", methods=["POST"])
    def api_delete_uploaded_file():
    """API endpoint for deleting uploaded historical data files."""
    try:
        data = request.get_json()
        file_path = data.get('file_path')

        if not file_path:
            return jsonify({"success": False, "error": "File path required"})

        # Security check - ensure file is in upload directory
        file_path = Path(file_path)
        if not str(file_path).startswith(str(UPLOAD_FOLDER)):
            return jsonify({"success": False, "error": "Invalid file path"})

        if file_path.exists():
            file_path.unlink()
            return jsonify({"success": True, "message": "File deleted successfully"})
        else:
            return jsonify({"success": False, "error": "File not found"})

    except Exception as e:
        logging.error(f"Error deleting file: {e}")
        return jsonify({"success": False, "error": str(e)})


def process_uploaded_data(file_path, trading_pair):
    """Process uploaded historical data and integrate with system."""
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)

        # Standardize column names
        column_mappings = {}
            'time': 'timestamp', 'date': 'timestamp', 'datetime': 'timestamp',
            'price': 'close', 'last': 'close',
            'amount': 'volume', 'vol': 'volume', 'volumeto': 'volume', 'volumefrom': 'volume'
        }
        df = df.rename(columns=column_mappings)

        # Ensure required columns exist
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                if col == 'open' and 'close' in df.columns:
                    df['open'] = df['close']
                elif col == 'high' and 'close' in df.columns:
                    df['high'] = df['close']
                elif col == 'low' and 'close' in df.columns:
                    df['low'] = df['close']
                elif col == 'volume':
                    df['volume'] = 1000.0  # Default volume

        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Sort by timestamp and remove duplicates
        df = df.sort_values('timestamp').drop_duplicates(subset=['timestamp'])

        # Add trading pair identifier
        df['trading_pair'] = trading_pair

        # Save processed data
        processed_dir = Path('data/preprocessed')
        processed_dir.mkdir(parents=True, exist_ok=True)
        processed_file = processed_dir / f"{trading_pair}_uploaded.parquet"
        df.to_parquet(processed_file)

        logging.info(f"Successfully processed {len(df)} records for {trading_pair}")
        return True

    except Exception as e:
        logging.error(f"Error processing uploaded data: {e}")
        return False


@app.route("/quad-bit-strategy")
    def quad_bit_strategy_page():
    """Quad-bit strategy array dashboard page."""
    return render_template("quad_bit_strategy.html")


@app.route("/api/quad-bit-strategy/status")
    def api_quad_bit_strategy_status():
    """Get quad-bit strategy array status."""
    try:
        if quad_bit_strategy is None:
            return jsonify({"success": False, "error": "Quad-bit strategy not initialized"})

        status = quad_bit_strategy.get_system_status()
        return jsonify({"success": True, "data": status})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/quad-bit-strategy/execute/<pair>")
    def api_quad_bit_strategy_execute(pair):
    """Execute strategy for a specific trading pair."""
    try:
        if quad_bit_strategy is None:
            return jsonify({"success": False, "error": "Quad-bit strategy not initialized"})

        # Convert pair format
        pair_mapping = {}
            'BTC_USDC': 'BTC/USDC',
            'ETH_USDC': 'ETH/USDC', 
            'SOL_USDC': 'SOL/USDC',
            'XRP_USDC': 'XRP/USDC'
        }

        if pair not in pair_mapping:
            return jsonify({"success": False, "error": f"Invalid pair: {pair}"})

        # Mock market data for demonstration
        market_data = {}
            'current_price': 50000.0 if pair == 'BTC_USDC' else 3000.0,
            'close_prices': [50000.0, 50100.0, 50200.0, 50300.0, 50400.0],
            'volume': 1000000,
            'rsi': 55.0
        }

        # Execute strategy
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        decision = loop.run_until_complete()
            quad_bit_strategy.execute_strategy(pair_mapping[pair], market_data)
        )
        loop.close()

        if decision is None:
            return jsonify({"success": False, "error": "No decision generated"})

        return jsonify({)}
            "success": True,
            "data": {}
                "pair": pair,
                "decision": decision.action.value if decision.action else "HOLD",
                "price": decision.price if decision.price else 0,
                "quantity": decision.quantity if decision.quantity else 0,
                "stop_loss": decision.metadata.get('stop_loss', 0) if decision.metadata else 0,
                "take_profit": decision.metadata.get('take_profit', 0) if decision.metadata else 0,
                "metadata": decision.metadata if decision.metadata else {}
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/quad-bit-strategy/rebalance")
    def api_quad_bit_strategy_rebalance():
    """Execute basket rebalancing."""
    try:
        if quad_bit_strategy is None:
            return jsonify({"success": False, "error": "Quad-bit strategy not initialized"})

        # Execute rebalancing
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        decisions = loop.run_until_complete()
            quad_bit_strategy.execute_basket_rebalancing()
        )
        loop.close()

        return jsonify({)}
            "success": True,
            "data": {}
                "decisions_count": len(decisions),
                "decisions": []
                    {}
                        "pair": d.symbol,
                        "signal": d.action.value if d.action else "HOLD",
                        "price": d.price if d.price else 0,
                        "quantity": d.quantity if d.quantity else 0,
                        "metadata": d.metadata if d.metadata else {}
                    }
                    for d in decisions
                ]
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/quad-bit-strategy/initialize")
    def api_quad_bit_strategy_initialize():
    """Initialize the quad-bit strategy array."""
    try:
        success = initialize_quad_bit_strategy()
        return jsonify({)}
            "success": success,
            "message": "Quad-bit strategy array initialized" if success else "Failed to initialize"
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


def create_templates():
    """Create HTML templates for the web interface."""
    dashboard_html = """<!DOCTYPE html>"
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
            <p>Unified control center for Schwabot's biological immune system</p>'
        </div>

        <div class="nav-buttons">
            <a href="/" class="btn btn-primary">Dashboard</a>
            <a href="/setup" class="btn btn-secondary">API Setup</a>
            <a href="/trading" class="btn btn-warning">Trading</a>
            <a href="/visualization" class="btn btn-secondary">Visualization</a>
            <a href="/data-upload" class="btn btn-secondary">📊 Data Upload</a>
            <a href="/quad-bit-strategy" class="btn btn-secondary">🎯 Quad-Bit Strategy</a>
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
        function updateMarketSnapshot() {}
            fetch('/api/market-snapshot')
                .then(response => response.json())
                .then(data => {)}
                    if (data.success) {}
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
                .catch(error => {)}
                    document.getElementById('snapshot-content').innerHTML = '<p style="color: #f44336;">Error loading market data</p>';
                });
        }

        // Update Lantern Core status
        function updateLanternCoreStatus() {}
            fetch('/api/lantern-core-status')
                .then(response => response.json())
                .then(data => {)}
                    if (data.success) {}
                        const status = data.data;
                        const isRunning = status.lantern_core?.is_running || false;
                        const isInitialized = status.lantern_core?.is_initialized || false;

                        const statusDiv = document.getElementById('lantern-core-status');
                        const indicator = statusDiv.querySelector('.status-indicator');
                        const text = statusDiv.querySelector('p');

                        if (isRunning && isInitialized) {}
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
                .catch(error => {)}
                    console.error('Error updating Lantern Core status:', error);
                });
        }

        // Control functions
        function startLanternCore() {}
            fetch('/api/start-lantern-core')
                .then(response => response.json())
                .then(data => {)}
                    if (data.success) {}
                        alert('Lantern Core started successfully!');
                        updateLanternCoreStatus();
                    } else {
                        alert('Failed to start Lantern Core: ' + data.error);
                    }
                })
                .catch(error => {)}
                    alert('Error starting Lantern Core: ' + error);
                });
        }

        function stopLanternCore() {}
            fetch('/api/stop-lantern-core')
                .then(response => response.json())
                .then(data => {)}
                    if (data.success) {}
                        alert('Lantern Core stopped successfully!');
                        updateLanternCoreStatus();
                    } else {
                        alert('Failed to stop Lantern Core: ' + data.error);
                    }
                })
                .catch(error => {)}
                    alert('Error stopping Lantern Core: ' + error);
                });
        }

        function startTrading(mode) {}
            if (mode === 'live') {}
                if (!confirm('Are you sure you want to start LIVE trading? This will execute real trades!')) {}
                    return;
                }
            }

            fetch(`/api/start-trading/${mode}`)
                .then(response => response.json())
                .then(data => {)}
                    if (data.success) {}
                        alert(`Trading engine started in ${mode} mode!`);
                    } else {
                        alert('Failed to start trading: ' + data.error);
                    }
                })
                .catch(error => {)}
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
    setup_html = """<!DOCTYPE html>"
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
        document.getElementById('setup-form').addEventListener('submit', function(e) {)}
            e.preventDefault();
            const formData = new FormData(e.target);
            const data = Object.fromEntries(formData);

            fetch('/api/setup', {)}
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            })
            .then(response => response.json())
            .then(data => {)}
                const statusDiv = document.getElementById('status-message');
                if (data.success) {}
                    statusDiv.innerHTML = '<div class="alert alert-success">Configuration saved successfully!</div>';
                } else {
                    statusDiv.innerHTML = '<div class="alert alert-error">Error: ' + data.error + '</div>';
                }
            });
        });
    </script>
</body>
</html>"""

    # Data upload template
    data_upload_html = """<!DOCTYPE html>"
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Historical Data Upload - Schwabot</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background: #1a1a1a; color: #fff; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 15px; }
        .nav-buttons { text-align: center; margin-bottom: 30px; }
        .btn { padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; text-decoration: none; display: inline-block; margin: 5px; transition: all 0.3s; }
        .btn-primary { background: #4CAF50; color: white; }
        .btn-secondary { background: #2196F3; color: white; }
        .btn-warning { background: #FF9800; color: white; }
        .btn-danger { background: #f44336; color: white; }
        .btn:hover { opacity: 0.8; transform: translateY(-2px); }
        .upload-section { background: #2a2a2a; padding: 30px; border-radius: 10px; margin-bottom: 30px; }
        .form-group { margin-bottom: 20px; }
        label { display: block; margin-bottom: 5px; font-weight: bold; }
        select, input[type="file"] { width: 100%; padding: 10px; border: 1px solid #444; border-radius: 5px; background: #333; color: #fff; }
        .file-info { background: #333; padding: 15px; border-radius: 5px; margin: 10px 0; }
        .files-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .file-card { background: #2a2a2a; padding: 20px; border-radius: 10px; border-left: 5px solid #4CAF50; }
        .alert { padding: 10px; margin: 10px 0; border-radius: 5px; }
        .alert-success { background: #4CAF50; }
        .alert-error { background: #f44336; }
        .alert-warning { background: #FF9800; }
        .progress-bar { width: 100%; height: 20px; background: #333; border-radius: 10px; overflow: hidden; margin: 10px 0; }
        .progress-fill { height: 100%; background: #4CAF50; width: 0%; transition: width 0.3s; }
        .csv-format-info { background: #1e3a5f; padding: 20px; border-radius: 10px; margin: 20px 0; }
        .csv-format-info h4 { margin-top: 0; color: #64b5f6; }
        .csv-format-info code { background: #2a2a2a; padding: 2px 6px; border-radius: 3px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Historical Data Upload</h1>
            <p>Upload historical cryptocurrency data for BTC/USDC, XRP/USDC, SOL/USDC, and ETH/USDC</p>
        </div>

        <div class="nav-buttons">
            <a href="/" class="btn btn-primary">Dashboard</a>
            <a href="/setup" class="btn btn-secondary">API Setup</a>
            <a href="/trading" class="btn btn-warning">Trading</a>
            <a href="/visualization" class="btn btn-secondary">Visualization</a>
            <a href="/data-upload" class="btn btn-secondary">📊 Data Upload</a>
        </div>

        <div class="csv-format-info">
            <h4>📋 Required CSV Format</h4>
            <p>Your CSV file must contain the following columns:</p>
            <ul>
                <li><code>timestamp</code> - Date/time in ISO format or Unix timestamp</li>
                <li><code>open</code> - Opening price</li>
                <li><code>high</code> - Highest price</li>
                <li><code>low</code> - Lowest price</li>
                <li><code>close</code> - Closing price</li>
                <li><code>volume</code> - Trading volume</li>
            </ul>
            <p><strong>Example:</strong></p>
            <code>timestamp,open,high,low,close,volume<br>
2023-1-1 0:0:0,16500.50,16550.75,16480.25,16525.30,1250.45</code>
        </div>

        <div class="upload-section">
            <h3>📤 Upload Historical Data</h3>
            <form id="upload-form" enctype="multipart/form-data">
                <div class="form-group">
                    <label for="trading_pair">Trading Pair:</label>
                    <select id="trading_pair" name="trading_pair" required>
                        <option value="btc_usdc">BTC/USDC</option>
                        <option value="xrp_usdc">XRP/USDC</option>
                        <option value="sol_usdc">SOL/USDC</option>
                        <option value="eth_usdc">ETH/USDC</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="file">CSV File:</label>
                    <input type="file" id="file" name="file" accept=".csv" required>
                    <div id="file-info" class="file-info" style="display: none;"></div>
                </div>
                <button type="submit" class="btn btn-primary">Upload Data</button>
            </form>
            <div id="upload-progress" class="progress-bar" style="display: none;">
                <div class="progress-fill" id="progress-fill"></div>
            </div>
            <div id="upload-status"></div>
        </div>

        <div class="upload-section">
            <h3>📁 Uploaded Files</h3>
            <div id="files-container">
                <p>Loading uploaded files...</p>
            </div>
        </div>
    </div>

    <script>
        // File info display
        document.getElementById('file').addEventListener('change', function(e) {)}
            const file = e.target.files[0];
            const fileInfo = document.getElementById('file-info');

            if (file) {}
                const sizeMB = (file.size / (1024 * 1024)).toFixed(2);
                fileInfo.innerHTML = `
                    <strong>File:</strong> ${file.name}<br>
                    <strong>Size:</strong> ${sizeMB} MB<br>
                    <strong>Type:</strong> ${file.type || 'text/csv'}
                `;
                fileInfo.style.display = 'block';
            } else {
                fileInfo.style.display = 'none';
            }
        });

        // Upload form handling
        document.getElementById('upload-form').addEventListener('submit', function(e) {)}
            e.preventDefault();

            const formData = new FormData(this);
            const progressBar = document.getElementById('upload-progress');
            const progressFill = document.getElementById('progress-fill');
            const statusDiv = document.getElementById('upload-status');

            // Show progress bar
            progressBar.style.display = 'block';
            progressFill.style.width = '0%';
            statusDiv.innerHTML = '<div class="alert alert-warning">Uploading...</div>';

            // Simulate progress
            let progress = 0;
            const progressInterval = setInterval(() => {)}
                progress += Math.random() * 20;
                if (progress > 90) progress = 90;
                progressFill.style.width = progress + '%';
            }, 200);

            fetch('/api/upload-historical-data', {)}
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {)}
                clearInterval(progressInterval);
                progressFill.style.width = '100%';

                if (data.success) {}
                    statusDiv.innerHTML = `<div class="alert alert-success">${data.message}</div>`;
                    document.getElementById('upload-form').reset();
                    document.getElementById('file-info').style.display = 'none';
                    loadUploadedFiles(); // Refresh file list
                } else {
                    statusDiv.innerHTML = `<div class="alert alert-error">Error: ${data.error}</div>`;
                }

                setTimeout(() => {)}
                    progressBar.style.display = 'none';
                }, 2000);
            })
            .catch(error => {)}
                clearInterval(progressInterval);
                statusDiv.innerHTML = `<div class="alert alert-error">Upload failed: ${error}</div>`;
                progressBar.style.display = 'none';
            });
        });

        // Load uploaded files
        function loadUploadedFiles() {}
            fetch('/api/list-uploaded-files')
                .then(response => response.json())
                .then(data => {)}
                    if (data.success) {}
                        displayFiles(data.files);
                    } else {
                        document.getElementById('files-container').innerHTML = 
                            '<div class="alert alert-error">Failed to load files: ' + data.error + '</div>';
                    }
                })
                .catch(error => {)}
                    document.getElementById('files-container').innerHTML = 
                        '<div class="alert alert-error">Error loading files: ' + error + '</div>';
                });
        }

        // Display uploaded files
        function displayFiles(files) {}
            const container = document.getElementById('files-container');

            if (Object.values(files).every(arr => arr.length === 0)) {}
                container.innerHTML = '<p>No files uploaded yet.</p>';
                return;
            }

            let html = '<div class="files-grid">';

            for (const [pair, fileList] of Object.entries(files)) {}
                if (fileList.length > 0) {}
                    html += `<div class="file-card">
                        <h4>${pair.toUpperCase()}</h4>`;

                    fileList.forEach(file => {)}
                        html += `
                            <div style="margin: 10px 0; padding: 10px; background: #333; border-radius: 5px;">
                                <strong>${file.filename}</strong><br>
                                <small>Size: ${file.size_mb} MB | Uploaded: ${file.upload_date}</small><br>
                                <button class="btn btn-danger" onclick="deleteFile('${file.file_path}')" style="margin-top: 5px; padding: 5px 10px; font-size: 12px;">Delete</button>
                            </div>
                        `;
                    });

                    html += '</div>';
                }
            }

            html += '</div>';
            container.innerHTML = html;
        }

        // Delete file
        function deleteFile(filePath) {}
            if (!confirm('Are you sure you want to delete this file?')) {}
                return;
            }

            fetch('/api/delete-uploaded-file', {)}
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ file_path: filePath })
            })
            .then(response => response.json())
            .then(data => {)}
                if (data.success) {}
                    alert('File deleted successfully!');
                    loadUploadedFiles(); // Refresh file list
                } else {
                    alert('Error deleting file: ' + data.error);
                }
            })
            .catch(error => {)}
                alert('Error deleting file: ' + error);
            });
        }

        // Load files on page load
        loadUploadedFiles();
    </script>
</body>
</html>"""

    # Quad-bit strategy template
    quad_bit_strategy_html = """<!DOCTYPE html>"
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quad-Bit Strategy Array - Schwabot</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background: #1a1a1a; color: #fff; }
        .container { max-width: 1400px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 15px; }
        .nav-buttons { text-align: center; margin-bottom: 30px; }
        .btn { padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; text-decoration: none; display: inline-block; margin: 5px; transition: all 0.3s; }
        .btn-primary { background: #4CAF50; color: white; }
        .btn-secondary { background: #2196F3; color: white; }
        .btn-warning { background: #FF9800; color: white; }
        .btn-danger { background: #f44336; color: white; }
        .btn:hover { opacity: 0.8; transform: translateY(-2px); }
        .strategy-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .strategy-card { background: #2a2a2a; padding: 20px; border-radius: 10px; border-left: 5px solid #4CAF50; }
        .strategy-card.warning { border-left-color: #FF9800; }
        .strategy-card.error { border-left-color: #f44336; }
        .bit-display { display: flex; gap: 10px; margin: 15px 0; }
        .bit { width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; }
        .bit-0 { background: #666; }
        .bit-1 { background: #4CAF50; }
        .sequence-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin: 20px 0; }
        .sequence-item { background: #333; padding: 10px; border-radius: 5px; text-align: center; cursor: pointer; transition: all 0.3s; }
        .sequence-item:hover { background: #444; }
        .sequence-item.active { background: #4CAF50; }
        .alert { padding: 10px; margin: 10px 0; border-radius: 5px; }
        .alert-success { background: #4CAF50; }
        .alert-error { background: #f44336; }
        .alert-warning { background: #FF9800; }
        .status-indicator { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 10px; }
        .status-good { background: #4CAF50; }
        .status-warning { background: #FF9800; }
        .status-bad { background: #f44336; }
        .basket-info { background: #1e3a5f; padding: 20px; border-radius: 10px; margin: 20px 0; }
        .basket-info h4 { margin-top: 0; color: #64b5f6; }
        .correlation-matrix { display: grid; grid-template-columns: repeat(4, 1fr); gap: 5px; margin: 15px 0; }
        .correlation-cell { background: #2a2a2a; padding: 8px; border-radius: 3px; text-align: center; font-size: 12px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 Quad-Bit Strategy Array</h1>
            <p>Advanced 4-bit strategy array for BTC/USDC, XRP/USDC, SOL/USDC, ETH/USDC trading</p>
        </div>

        <div class="nav-buttons">
            <a href="/" class="btn btn-primary">Dashboard</a>
            <a href="/setup" class="btn btn-secondary">API Setup</a>
            <a href="/trading" class="btn btn-warning">Trading</a>
            <a href="/visualization" class="btn btn-secondary">Visualization</a>
            <a href="/data-upload" class="btn btn-secondary">📊 Data Upload</a>
            <a href="/quad-bit-strategy" class="btn btn-secondary">🎯 Quad-Bit Strategy</a>
        </div>

        <div class="basket-info">
            <h4>📊 Tensor Basket Configuration</h4>
            <p><strong>Main Basket:</strong> BTC (40%), ETH (30%), SOL (20%), XRP (10%)</p>
            <p><strong>BTC-Focused Basket:</strong> BTC (70%), ETH (30%) - Special treatment</p>
            <div class="correlation-matrix">
                <div class="correlation-cell">BTC</div>
                <div class="correlation-cell">ETH</div>
                <div class="correlation-cell">SOL</div>
                <div class="correlation-cell">XRP</div>
                <div class="correlation-cell">1.0</div>
                <div class="correlation-cell">0.7</div>
                <div class="correlation-cell">0.6</div>
                <div class="correlation-cell">0.5</div>
                <div class="correlation-cell">0.7</div>
                <div class="correlation-cell">1.0</div>
                <div class="correlation-cell">0.8</div>
                <div class="correlation-cell">0.6</div>
                <div class="correlation-cell">0.6</div>
                <div class="correlation-cell">0.8</div>
                <div class="correlation-cell">1.0</div>
                <div class="correlation-cell">0.7</div>
                <div class="correlation-cell">0.5</div>
                <div class="correlation-cell">0.6</div>
                <div class="correlation-cell">0.7</div>
                <div class="correlation-cell">1.0</div>
            </div>
        </div>

        <div class="strategy-grid">
            <div class="strategy-card">
                <h3>🎛️ System Control</h3>
                <button class="btn btn-primary" onclick="initializeStrategy()">Initialize Strategy</button>
                <button class="btn btn-warning" onclick="executeRebalancing()">Execute Rebalancing</button>
                <div id="system-status">
                    <p><span class="status-indicator status-bad"></span>System not initialized</p>
                </div>
            </div>

            <div class="strategy-card">
                <h3>🔢 Active Sequence</h3>
                <div id="active-sequence">
                    <p>No active sequence</p>
                </div>
                <div class="bit-display" id="bit-display">
                    <div class="bit bit-0">0</div>
                    <div class="bit bit-0">0</div>
                    <div class="bit bit-0">0</div>
                    <div class="bit bit-0">0</div>
                </div>
                <p><small>Bit 0: Entry/Exit | Bit 1: Position Size | Bit 2: Risk | Bit 3: Profit</small></p>
            </div>

            <div class="strategy-card">
                <h3>📈 Trading Pairs</h3>
                <div id="pair-states">
                    <p>Loading pair states...</p>
                </div>
            </div>

            <div class="strategy-card">
                <h3>💰 Asset Profiles</h3>
                <div id="asset-profiles">
                    <p>Loading asset profiles...</p>
                </div>
            </div>
        </div>

        <div class="strategy-card">
            <h3>🎲 Strategy Execution</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 15px 0;">
                <button class="btn btn-primary" onclick="executeStrategy('BTC_USDC')">Execute BTC/USDC</button>
                <button class="btn btn-primary" onclick="executeStrategy('ETH_USDC')">Execute ETH/USDC</button>
                <button class="btn btn-primary" onclick="executeStrategy('SOL_USDC')">Execute SOL/USDC</button>
                <button class="btn btn-primary" onclick="executeStrategy('XRP_USDC')">Execute XRP/USDC</button>
            </div>
            <div id="execution-results">
                <p>No executions yet</p>
            </div>
        </div>

        <div class="strategy-card">
            <h3>📋 16 Drift Sequences</h3>
            <div class="sequence-grid" id="sequence-grid">
                <!-- Sequences will be populated by JavaScript -->
            </div>
        </div>
    </div>

    <script>
        // Initialize sequence grid
        function initializeSequenceGrid() {}
            const grid = document.getElementById('sequence-grid');
            const sequences = []
                '000', '001', '010', '011',
                '100', '101', '110', '111',
                '1000', '1001', '1010', '1011',
                '1100', '1101', '1110', '1111'
            ];

            sequences.forEach(seq => {)}
                const item = document.createElement('div');
                item.className = 'sequence-item';
                item.textContent = seq;
                item.onclick = () => selectSequence(seq);
                grid.appendChild(item);
            });
        }

        function selectSequence(sequence) {}
            // Remove active class from all items
            document.querySelectorAll('.sequence-item').forEach(item => {)}
                item.classList.remove('active');
            });

            // Add active class to selected item
            event.target.classList.add('active');

            // Update bit display
            updateBitDisplay(sequence);
        }

        function updateBitDisplay(sequence) {}
            const bits = sequence.split('').map(bit => parseInt(bit));
            const bitElements = document.querySelectorAll('#bit-display .bit');

            bits.forEach((bit, index) => {)}
                bitElements[index].className = `bit bit-${bit}`;
                bitElements[index].textContent = bit;
            });
        }

        // Initialize strategy
        function initializeStrategy() {}
            fetch('/api/quad-bit-strategy/initialize')
                .then(response => response.json())
                .then(data => {)}
                    if (data.success) {}
                        document.getElementById('system-status').innerHTML = 
                            '<p><span class="status-indicator status-good"></span>System initialized</p>';
                        updateStatus();
                    } else {
                        document.getElementById('system-status').innerHTML = 
                            '<p><span class="status-indicator status-bad"></span>Initialization failed: ' + data.error + '</p>';
                    }
                })
                .catch(error => {)}
                    document.getElementById('system-status').innerHTML = 
                        '<p><span class="status-indicator status-bad"></span>Error: ' + error + '</p>';
                });
        }

        // Update status
        function updateStatus() {}
            fetch('/api/quad-bit-strategy/status')
                .then(response => response.json())
                .then(data => {)}
                    if (data.success) {}
                        const status = data.data;

                        // Update active sequence
                        const sequence = status.active_sequence;
                        const sequenceBinary = sequence.toString(2).padStart(4, '0');
                        document.getElementById('active-sequence').innerHTML = 
                            `<p>Sequence ${sequence} (${sequenceBinary})</p>`;
                        updateBitDisplay(sequenceBinary);

                        // Update pair states
                        const pairStates = status.pair_states;
                        let pairHtml = '';
                        for (const [pair, state] of Object.entries(pairStates)) {}
                            const sequence = state.sequence || 'None';
                            const lastUpdate = new Date(state.last_update * 1000).toLocaleTimeString();
                            pairHtml += `<p><strong>${pair}:</strong> Sequence ${sequence} (${lastUpdate})</p>`;
                        }
                        document.getElementById('pair-states').innerHTML = pairHtml || '<p>No pair states</p>';

                        // Update asset profiles
                        const assetProfiles = status.asset_profiles;
                        let assetHtml = '';
                        for (const [symbol, profile] of Object.entries(assetProfiles)) {}
                            const needsRebalancing = profile.needs_rebalancing ? '⚠️' : '✅';
                            assetHtml += `<p>${needsRebalancing} <strong>${symbol}:</strong> ${profile.allocation.toFixed(1)}% / ${profile.target.toFixed(1)}%</p>`;
                        }
                        document.getElementById('asset-profiles').innerHTML = assetHtml || '<p>No asset profiles</p>';

                    } else {
                        console.error('Failed to get status:', data.error);
                    }
                })
                .catch(error => {)}
                    console.error('Error updating status:', error);
                });
        }

        // Execute strategy for a pair
        function executeStrategy(pair) {}
            fetch(`/api/quad-bit-strategy/execute/${pair}`)
                .then(response => response.json())
                .then(data => {)}
                    if (data.success) {}
                        const result = data.data;
                        const resultsDiv = document.getElementById('execution-results');
                        resultsDiv.innerHTML = `
                            <div class="alert alert-success">
                                <strong>${result.pair}:</strong> ${result.decision}<br>
                                Price: $${result.price.toLocaleString()}<br>
                                Quantity: ${result.quantity.toFixed(2)}<br>
                                Stop Loss: $${result.stop_loss.toLocaleString()}<br>
                                Take Profit: $${result.take_profit.toLocaleString()}
                            </div>
                        `;
                        updateStatus();
                    } else {
                        document.getElementById('execution-results').innerHTML = 
                            `<div class="alert alert-error">Error: ${data.error}</div>`;
                    }
                })
                .catch(error => {)}
                    document.getElementById('execution-results').innerHTML = 
                        `<div class="alert alert-error">Error: ${error}</div>`;
                });
        }

        // Execute rebalancing
        function executeRebalancing() {}
            fetch('/api/quad-bit-strategy/rebalance')
                .then(response => response.json())
                .then(data => {)}
                    if (data.success) {}
                        const result = data.data;
                        const resultsDiv = document.getElementById('execution-results');
                        resultsDiv.innerHTML = `
                            <div class="alert alert-success">
                                <strong>Rebalancing Complete:</strong> ${result.decisions_count} decisions<br>
                                ${result.decisions.map(d => `${d.pair}: ${d.signal} ${d.quantity.toFixed(2)}`).join('<br>')}
                            </div>
                        `;
                        updateStatus();
                    } else {
                        document.getElementById('execution-results').innerHTML = 
                            `<div class="alert alert-error">Error: ${data.error}</div>`;
                    }
                })
                .catch(error => {)}
                    document.getElementById('execution-results').innerHTML = 
                        `<div class="alert alert-error">Error: ${error}</div>`;
                });
        }

        // Initialize page
        initializeSequenceGrid();
        updateStatus();
        setInterval(updateStatus, 10000); // Update every 10 seconds
    </script>
</body>
</html>"""

    with open("templates/dashboard.html", "w", encoding="utf-8") as f:
        f.write(dashboard_html)

    with open("templates/setup.html", "w", encoding="utf-8") as f:
        f.write(setup_html)

    with open("templates/data_upload.html", "w", encoding="utf-8") as f:
        f.write(data_upload_html)

    with open("templates/quad_bit_strategy.html", "w", encoding="utf-8") as f:
        f.write(quad_bit_strategy_html)

    print("✅ HTML templates created successfully with UTF-8 encoding")


def main():
    """Main entry point for the Schwabot launcher."""
    print("🚀 Starting Schwabot Unified Launcher...")

    # Create templates if they don't exist'
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
