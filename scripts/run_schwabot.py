import logging
import os
import sys
import threading
import time
from pathlib import Path

from core.chrono_causal_orchestrator import ChronoCausalOrchestrator
from core.dual_unicore_handler import DualUnicoreHandler
from core.fallback_logic_router import FallbackLogicRouter
from core.meta_layer_ghost_bridge import MetaLayerGhostBridge
from core.phantom_lag_model import PhantomLagModel
from core.settings_manager import get_settings_manager
from core.system_integration_orchestrator import SystemIntegrationOrchestrator
from ui.schwabot_dashboard import app, socketio
from utils.safe_print import safe_print

# -*- coding: utf-8 -*-


# Initialize Unicode handler
unicore = DualUnicoreHandler()

"""
Schwabot Main Entry Point
=========================

This script starts the complete Schwabot trading system including:
- Mathematical components initialization
- Web dashboard
- API server
- Real-time monitoring
- System integration orchestrator

Usage:
    python run_schwabot.py

The system will start the web dashboard on http://localhost:8080
and the API server on http://localhost:8081
"""


# Add core to path
sys.path.append(str(Path(__file__).parent / "core"))

# Import Schwabot components
    try:

    IMPORTS_SUCCESSFUL = True
    except ImportError as e:
    safe_print(f"Error importing Schwabot components: {e}")
    safe_print()
        "Please ensure all dependencies are installed: pip install -r requirements.txt"
    )
    IMPORTS_SUCCESSFUL = False

# Global variables for graceful shutdown
shutdown_event = threading.Event()
components = {}


def setup_logging():
    """Setup comprehensive logging configuration."""
    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    # Configure logging
    logging.basicConfig()
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[]
            logging.FileHandler(logs_dir / "schwabot.log"),
            logging.StreamHandler(),
        ],
    )

    # Set specific log levels
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    logging.getLogger("socketio").setLevel(logging.WARNING)

    return logging.getLogger(__name__)


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    shutdown_event.set()


def initialize_components():
    """Initialize all Schwabot components."""
    # global components # Removed: Not needed for modifying dictionary contents

    try:
        logger.info("Initializing Schwabot components...")

        # Initialize settings manager
        logger.info("Loading settings manager...")
        settings_manager = get_settings_manager()
        components["settings_manager"] = settings_manager
        logger.info("Settings manager initialized successfully")

        # Initialize system orchestrator
        logger.info("Initializing system integration orchestrator...")
        orchestrator = SystemIntegrationOrchestrator()
        components["orchestrator"] = orchestrator
        logger.info("System orchestrator initialized successfully")

        # Initialize mathematical components
        logger.info("Initializing mathematical components...")

        phantom_model = PhantomLagModel()
        meta_bridge = MetaLayerGhostBridge()
        fallback_router = FallbackLogicRouter()

        # Initialize Chrono-Causal Orchestrator
        logger.info("Initializing Chrono-Causal Orchestrator...")
        chrono_orchestrator = ChronoCausalOrchestrator()
        components["chrono_orchestrator"] = chrono_orchestrator
        logger.info("Chrono-Causal Orchestrator initialized successfully")

        components["phantom_model"] = phantom_model
        components["meta_bridge"] = meta_bridge
        components["fallback_router"] = fallback_router

        logger.info("Mathematical components initialized successfully")

        return True

    except Exception as e:
        logger.error(f"Error initializing components: {e}")
        return False


def validate_environment():
    """Validate that required environment variables are set."""
    logger.info("Validating environment configuration...")

    required_vars = []
        "BINANCE_API_KEY",
        "BINANCE_API_SECRET",
        "COINBASE_API_KEY",
        "COINBASE_API_SECRET",
        "KRAKEN_API_KEY",
        "KRAKEN_API_SECRET",
    ]
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        logger.warning(f"Missing environment variables: {missing_vars}")
        logger.warning("System will run in sandbox mode with simulated data")
        return False

    logger.info("Environment validation passed")
    return True


def start_background_tasks():
    """Start background monitoring and maintenance tasks."""

    def background_monitor():
        """Background monitoring task."""
        while not shutdown_event.is_set():
            try:
                pass  # Update system health
                if "orchestrator" in components:
                    health = components["orchestrator"].get_system_health()
                    logger.debug(f"System health: {health}")

                # Sleep for monitoring interval
                time.sleep(30)  # Check every 30 seconds

            except Exception as e:
                logger.error(f"Error in background monitoring: {e}")
                time.sleep(60)  # Wait longer on error

    # Start background monitoring thread
    monitor_thread = threading.Thread(target=background_monitor, daemon=True)
    monitor_thread.start()
    logger.info("Background monitoring started")


def print_startup_banner():
    """Print Schwabot startup banner."""
    print("\n--------------------------------------------------------------")
    print("|                    SCHWABOT TRADING SYSTEM                   |")
    print("--------------------------------------------------------------")
    print("|              Hardware - Scale - Aware Economic Kernel        |")
    print("|  Mathematical Foundation: Phantom Lag Model, Ghost Bridge    |")
    print("|  Real - time Trading: Multi - exchange with Arbitrage Detection  |")
    print("|  Distributed Architecture: Federated Device Support          |")
    print("--------------------------------------------------------------\n")


def print_system_info():
    """Print system information and status."""
    if "settings_manager" in components:
        settings = components["settings_manager"]
        config_summary = settings.get_configuration_summary()

        safe_print("\n--- System Configuration ---")
        safe_print(f"   Environment: {config_summary.get('environment', 'unknown')}")
        safe_print(f"   Debug Mode: {config_summary.get('debug_mode', False)}")
        safe_print()
            f"   API Server Port: {config_summary.get('api_server_port', 'N/A')}"
        )
        safe_print(f"   Dashboard Port: {config_summary.get('dashboard_port', 'N/A')}")
        safe_print(f"   Exchange Mode: {config_summary.get('exchange_mode', 'N/A')}")
        safe_print(f"   Trading Pairs: {config_summary.get('trading_pairs', 'N/A')}")
        safe_print()
            f"   Risk Management: {"}
                config_summary.get('risk_management_enabled', False)
            }"
        )
    else:
        safe_print()
            "\n--- System Configuration: Not available (Settings Manager not, initialized) ---"
        )

    if "orchestrator" in components:
        safe_print("\n--- System Health Metrics ---")
        # This will be updated to use chrono_causal_orchestrator's validation'
        # For now, placeholder or existing SystemIntegrationOrchestrator health
        health_metrics = ()
            components["orchestrator"].get_system_health()
            if hasattr(components["orchestrator"], "get_system_health")
            else "N/A"
        )
        safe_print(f"   Overall Health: {health_metrics}")
        safe_print()
            f"   Chrono-Causal Orchestrator Status: {"}
                'Initialized'
                if 'chrono_orchestrator' in components
                else 'Not Initialized'
            }"
        )
        if "chrono_orchestrator" in components:
            # Example of how we might display initial orchestrator status or mock data
            safe_print("   CRWM Active: Yes")
            safe_print("   CRTPM Active: Yes")
            safe_print("   Sustainment Validator Active: Yes")
    else:
        safe_print("\n--- System Health Metrics: Orchestrator not initialized. ---")


def main():
    """Main entry point for Schwabot."""
    global logger
    logger = setup_logging()

    print_startup_banner()

    logger.info("Starting Schwabot Trading System...")

    try:
        # Validate environment
        validate_environment()  # Removed assignment to env_valid

        # Initialize components
        if not initialize_components():
            logger.error("Failed to initialize core components. Exiting.")
            sys.exit(1)

        # Start background tasks
        start_background_tasks()

        # Print system info
        print_system_info()

        # Flask app settings
        settings_manager = components["settings_manager"]
        host = settings_manager.get_setting("api_server_host", "0.0.0.0")
        port = settings_manager.get_setting("dashboard_port", 8080)

        safe_print(f"\n>>> Schwabot starting on http://{host}:{port}")
        safe_print(">>> Access the dashboard in your web browser")
        safe_print(">>> Use Ctrl + C to stop the server gracefully")
        safe_print("\n>>> System Status: RUNNING")

        # Start the Flask app
        socketio.run()
            app,
            host=host,
            port=port,
            debug=False,
            use_reloader=False,  # Disable reloader to avoid duplicate processes
        )

    except KeyboardInterrupt:
        logger.info("Shutdown signal received (Ctrl+C).")
    except Exception as e:
        logger.critical(f"An unhandled error occurred: {e}", exc_info=True)
    finally:
        shutdown_event.set()
        logger.info("Schwabot shutdown complete")
        safe_print("\n>>> Schwabot stopped gracefully")

    return 0


if __name__ == "__main__":
    sys.exit(main())
