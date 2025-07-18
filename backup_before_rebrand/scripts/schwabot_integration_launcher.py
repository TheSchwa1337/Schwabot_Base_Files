#!/usr/bin/env python3
"""
Schwabot Integration Launcher
Complete system launcher with web dashboard, API, and trading integration
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

from schwabot_full_integration_system import IntegrationConfig, create_schwabot_integration

# Setup logging
logging.basicConfig()
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[]
        logging.FileHandler('schwabot_launcher.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SchwabotIntegrationLauncher:
    """Complete Schwabot system launcher."""

    def __init__(self):
        self.integration = None
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

    def start_integration(self, config: IntegrationConfig):
        """Start the full integration system."""
        try:
            logger.info("Starting Schwabot Full Integration...")

            # Create integration instance
            self.integration = create_schwabot_integration()
                session_id=config.session_id,
                exchange_name=config.exchange_name,
                sandbox_mode=config.sandbox_mode,
                api_key=config.api_key,
                api_secret=config.api_secret,
                symbols=config.symbols,
                data_persistence=config.data_persistence,
                strategy_persistence=config.strategy_persistence,
                real_time_updates=config.real_time_updates
            )

            # Initialize components
            if not self.integration.initialize_components():
                raise Exception("Component initialization failed")

            # Start background processors
            self.integration.start_background_processors()

            # Start Flask server
            if not self.start_flask_server():
                raise Exception("Flask server startup failed")

            self.running = True
            logger.info("Schwabot Full Integration started successfully")

            return True

        except Exception as e:
            logger.error(f"Integration startup failed: {e}")
            return False

    def get_status(self) -> Dict:
        """Get comprehensive system status."""
        if not self.integration:
            return {'status': 'not_initialized'}

        status = self.integration.get_system_status()
        status.update({)}
            'flask_server_running': self.flask_thread and self.flask_thread.is_alive(),
            'integration_running': self.running,
            'dashboard_url': 'http://127.0.0.1:5000' if self.flask_thread else None
        })

        return status

    def stop(self):
        """Stop the integration system."""
        logger.info("Stopping Schwabot Integration...")

        self.running = False

        if self.integration:
            self.integration.shutdown()

        logger.info("Schwabot Integration stopped")

def main():
    """Main launcher function."""
    print("🚀 Schwabot Integration Launcher")
    print("=" * 50)

    # Configuration
    config = IntegrationConfig()
        session_id=f"schwabot_session_{int(time.time())}",
        exchange_name="coinbase",
        sandbox_mode=True,
        symbols=['BTC/USDC', 'ETH/USDC', 'SOL/USDC'],
        data_persistence=True,
        strategy_persistence=True,
        real_time_updates=True
    )

    # Create launcher
    launcher = SchwabotIntegrationLauncher()

    try:
        # Start integration
        if launcher.start_integration(config):
            print("✅ Schwabot Full Integration started successfully")
            print(f"📊 Dashboard: http://127.0.0.1:5000")
            print(f"📊 Session ID: {config.session_id}")
            print(f"🏦 Exchange: {config.exchange_name}")
            print(f"📈 Symbols: {', '.join(config.symbols)}")
            print("\n🔄 System is running... Press Ctrl+C to stop")

            # Keep running
            while True:
                time.sleep(10)
                status = launcher.get_status()
                if status.get('status') == 'error':
                    print("❌ System error detected")
                    break

        else:
            print("❌ Failed to start Schwabot Integration")

    except KeyboardInterrupt:
        print("\n👋 Shutdown requested...")

    except Exception as e:
        print(f"❌ Unexpected error: {e}")

    finally:
        launcher.stop()
        print("✅ Schwabot Integration shutdown complete")

if __name__ == "__main__":
    main() 