#!/usr/bin/env python3
"""
Schwabot Main Launcher
Advanced Algorithmic Trading Intelligence System
Launches both the Trading Dashboard and Trading Intelligence
"""

import logging
import os
import sys
import threading
import time
from datetime import datetime
from typing import Dict, Optional

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from schwabot_trading_dashboard import DashboardConfig, create_schwabot_dashboard
from schwabot_trading_intelligence import IntelligenceConfig, create_schwabot_intelligence

# Setup logging
logging.basicConfig()
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[]
        logging.FileHandler('schwabot_main.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SchwabotMainLauncher:
    """Main Schwabot system launcher."""

    def __init__(self):
        self.dashboard = None
        self.intelligence = None
        self.flask_app = None
        self.flask_thread = None
        self.running = False

    def start_flask_server(self):
        """Start Flask web server in a separate thread."""
        try:
            from api.flask_app import app, socketio

            def run_flask():
                socketio.run(app, host='0.0.0.0', port=5000, debug=False)

            self.flask_thread = threading.Thread(target=run_flask, daemon=True)
            self.flask_thread.start()

            logger.info("Flask server started on http://127.0.0.1:5000")
            return True

        except Exception as e:
            logger.error(f"Failed to start Flask server: {e}")
            return False

    def start_schwabot_system(self, session_id: str = None, **kwargs):
        """Start the complete Schwabot system."""
        try:
            if session_id is None:
                session_id = f"schwabot_session_{int(time.time())}"

            logger.info(f"Starting Schwabot Main System for session {session_id}...")

            # Create dashboard instance
            dashboard_config = DashboardConfig()
                session_id=session_id,
                exchange_name=kwargs.get('exchange_name', 'coinbase'),
                sandbox_mode=kwargs.get('sandbox_mode', True),
                api_key=kwargs.get('api_key', ''),
                api_secret=kwargs.get('api_secret', ''),
                symbols=kwargs.get('symbols', ['BTC/USDC', 'ETH/USDC', 'SOL/USDC']),
                portfolio_value=kwargs.get('portfolio_value', 10000.0),
                demo_mode=kwargs.get('demo_mode', True)
            )

            self.dashboard = create_schwabot_dashboard()
                session_id=session_id,
                exchange_name=dashboard_config.exchange_name,
                sandbox_mode=dashboard_config.sandbox_mode,
                api_key=dashboard_config.api_key,
                api_secret=dashboard_config.api_secret,
                symbols=dashboard_config.symbols,
                portfolio_value=dashboard_config.portfolio_value,
                demo_mode=dashboard_config.demo_mode
            )

            # Create intelligence instance
            intelligence_config = IntelligenceConfig()
                session_id=session_id,
                exchange_name=kwargs.get('exchange_name', 'coinbase'),
                sandbox_mode=kwargs.get('sandbox_mode', True),
                api_key=kwargs.get('api_key', ''),
                api_secret=kwargs.get('api_secret', ''),
                symbols=kwargs.get('symbols', ['BTC/USDC', 'ETH/USDC', 'SOL/USDC']),
                enable_learning=kwargs.get('enable_learning', True),
                enable_automation=kwargs.get('enable_automation', True)
            )

            self.intelligence = create_schwabot_intelligence()
                session_id=session_id,
                exchange_name=intelligence_config.exchange_name,
                sandbox_mode=intelligence_config.sandbox_mode,
                api_key=intelligence_config.api_key,
                api_secret=intelligence_config.api_secret,
                symbols=intelligence_config.symbols,
                enable_learning=intelligence_config.enable_learning,
                enable_automation=intelligence_config.enable_automation
            )

            # Initialize dashboard components
            if not self.dashboard.initialize_components():
                raise Exception("Dashboard component initialization failed")

            # Initialize intelligence components
            if not self.intelligence.initialize_components():
                raise Exception("Intelligence component initialization failed")

            # Start dashboard background processors
            self.dashboard.start_background_processors()

            # Start intelligence engine
            self.intelligence.start_intelligence_engine()

            # Start Flask server
            if not self.start_flask_server():
                raise Exception("Flask server startup failed")

            self.running = True
            logger.info("Schwabot Main System started successfully")

            return True

        except Exception as e:
            logger.error(f"Schwabot system startup failed: {e}")
            return False

    def get_system_status(self) -> Dict:
        """Get comprehensive system status."""
        status = {}
            'running': self.running,
            'flask_server_running': self.flask_thread and self.flask_thread.is_alive(),
            'dashboard_url': 'http://127.0.0.1:5000' if self.flask_thread else None
        }

        if self.dashboard:
            dashboard_data = self.dashboard.get_dashboard_data()
            status['dashboard'] = dashboard_data

        if self.intelligence:
            intelligence_status = self.intelligence.get_intelligence_status()
            status['intelligence'] = intelligence_status

        return status

    def stop(self):
        """Stop the Schwabot system."""
        logger.info("Stopping Schwabot Main System...")

        self.running = False

        if self.dashboard:
            self.dashboard.shutdown()

        if self.intelligence:
            self.intelligence.shutdown()

        logger.info("Schwabot Main System stopped")

def main():
    """Main launcher function."""
    print("🚀 Schwabot Main Launcher")
    print("Advanced Algorithmic Trading Intelligence System")
    print("=" * 60)

    # Configuration
    config = {}
        'session_id': f"schwabot_main_{int(time.time())}",
        'exchange_name': "coinbase",
        'sandbox_mode': True,
        'symbols': ['BTC/USDC', 'ETH/USDC', 'SOL/USDC'],
        'portfolio_value': 10000.0,
        'demo_mode': True,
        'enable_learning': True,
        'enable_automation': True
    }

    # Create launcher
    launcher = SchwabotMainLauncher()

    try:
        # Start Schwabot system
        if launcher.start_schwabot_system(**config):
            print("✅ Schwabot Main System started successfully")
            print(f"📊 Dashboard: http://127.0.0.1:5000")
            print(f"📊 Session ID: {config['session_id']}")
            print(f"🏦 Exchange: {config['exchange_name']}")
            print(f"📈 Symbols: {', '.join(config['symbols'])}")
            print(f"💰 Portfolio Value: ${config['portfolio_value']:,.2f}")
            print(f"🎮 Demo Mode: {'Enabled' if config['demo_mode'] else 'Disabled'}")
            print(f"🧠 Learning: {'Enabled' if config['enable_learning'] else 'Disabled'}")
            print(f"🤖 Automation: {'Enabled' if config['enable_automation'] else 'Disabled'}")
            print("\n🔄 System is running... Press Ctrl+C to stop")

            # Keep running and show status
            while True:
                time.sleep(30)
                status = launcher.get_system_status()

                if status.get('running'):
                    print(f"📊 System Status: Running | Dashboard: {status.get('dashboard_url')}")

                    if 'dashboard' in status:
                        dashboard = status['dashboard']
                        print(f"💰 Portfolio: ${dashboard.get('portfolio_value', 0):,.2f} | ")
                              f"Profit: ${dashboard.get('total_profit', 0):,.2f} | "
                              f"Win Rate: {dashboard.get('win_rate', 0):.1f}% | "
                              f"Active Trades: {dashboard.get('active_trades', 0)}")

                    if 'intelligence' in status:
                        intelligence = status['intelligence']
                        print(f"🧠 Intelligence: {'Running' if intelligence.get('running') else 'Stopped'} | ")
                              f"Learning: {'On' if intelligence.get('features_enabled', {}).get('learning') else 'Off'} | "
                              f"Automation: {'On' if intelligence.get('features_enabled', {}).get('automation') else 'Off'}")
                else:
                    print("❌ System error detected")
                    break

        else:
            print("❌ Failed to start Schwabot Main System")

    except KeyboardInterrupt:
        print("\n👋 Shutdown requested...")

    except Exception as e:
        print(f"❌ Unexpected error: {e}")

    finally:
        launcher.stop()
        print("✅ Schwabot Main System shutdown complete")

if __name__ == "__main__":
    main() 