import argparse
import asyncio
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Optional

import yaml

from core.ferris_rde_daemon import DaemonConfig, FerrisRDEDaemon, get_daemon_instance
from utils.safe_print import error, info, success

#!/usr/bin/env python3
"""
Ferris RDE Daemon Startup Script

This script provides an easy way to start the Ferris RDE daemon with proper
configuration and monitoring. It includes:

- Configuration loading from YAML files
- Command-line argument parsing
- Daemon process management
- Health monitoring
- Graceful shutdown handling
- Logging setup
"""





def load_config():-> Optional[dict]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary or None if failed
    """
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        info(f"✅ Configuration loaded from {config_path}")
        return config

    except FileNotFoundError:
        error(f"❌ Configuration file not found: {config_path}")
        return None
    except yaml.YAMLError as e:
        error(f"❌ Error parsing configuration file: {e}")
        return None
    except Exception as e:
        error(f"❌ Error loading configuration: {e}")
        return None


def create_daemon_config():-> DaemonConfig:
    """
    Create DaemonConfig from configuration dictionary.

    Args:
        config_dict: Configuration dictionary

    Returns:
        DaemonConfig instance
    """
    try:
        # Extract configuration sections
        daemon_config = config_dict.get("daemon", {})
        trading_config = config_dict.get("trading", {})
        processing_config = config_dict.get("processing", {})
        timing_config = config_dict.get("timing", {})
        assets_config = config_dict.get("assets", {})
        ferris_config = config_dict.get("ferris_rde", {})
        mathematical_config = config_dict.get("mathematical", {})
        monitoring_config = config_dict.get("monitoring", {})
        visualization_config = config_dict.get("visualization", {})

        # Create DaemonConfig
        config = DaemonConfig()
            daemon_name=daemon_config.get("name", "FerrisRDE"),
            log_level=daemon_config.get("log_level", "INFO"),
            # Trading settings
            trading_enabled=trading_config.get("enabled", True),
            paper_trading=trading_config.get("paper_trading", True),
            max_concurrent_trades=trading_config.get("max_concurrent_trades", 10),
            risk_management_enabled=trading_config.get("risk_management_enabled", True),
            # Processing settings
            enable_gpu=processing_config.get("enable_gpu", True),
            enable_distributed=processing_config.get("enable_distributed", False),
            bit_depth_range=tuple(processing_config.get("bit_depth_range", [2, 42])),
            # Timing settings
            tick_interval_seconds=timing_config.get("tick_interval_seconds", 1.0),
            health_check_interval_seconds=timing_config.get()
                "health_check_interval_seconds", 30.0
            ),
            performance_report_interval_seconds=timing_config.get()
                "performance_report_interval_seconds", 300.0
            ),
            mathematical_update_interval_seconds=timing_config.get()
                "mathematical_update_interval_seconds", 5.0
            ),
            # Asset settings
            primary_assets=assets_config.get("primary", ["BTC/USD", "ETH/USD"]),
            secondary_assets=assets_config.get("secondary", ["XRP/USD", "ADA/USD"]),
            # Ferris RDE settings
            ferris_cycle_duration_minutes=ferris_config.get()
                "cycle_duration_minutes", 60
            ),
            ferris_phase_transitions=ferris_config.get()
                "phase_transitions",
                {}
                    "tick_to_pivot": 0.8,
                    "pivot_to_ascent": 0.7,
                    "ascent_to_descent": 0.6,
                    "descent_to_tick": 0.9,
                },
            ),
            # Mathematical settings
            enable_quantum_thermal=mathematical_config.get()
                "enable_quantum_thermal", True
            ),
            enable_void_well_metrics=mathematical_config.get()
                "enable_void_well_metrics", True
            ),
            enable_kelly_criterion=mathematical_config.get()
                "enable_kelly_criterion", True
            ),
            # Monitoring settings
            enable_health_monitoring=monitoring_config.get()
                "enable_health_monitoring", True
            ),
            enable_performance_tracking=monitoring_config.get()
                "enable_performance_tracking", True
            ),
            enable_error_logging=monitoring_config.get("enable_error_logging", True),
            max_error_count=monitoring_config.get("max_error_count", 1000),
            # Visualization settings
            enable_visualization=visualization_config.get("enable_visualization", True),
            dashboard_port=visualization_config.get("dashboard_port", 8080),
            websocket_port=visualization_config.get("websocket_port", 8081),
        )

        success("✅ Daemon configuration created successfully")
        return config

    except Exception as e:
        error(f"❌ Error creating daemon configuration: {e}")
        return DaemonConfig()  # Return default config


def setup_logging(config_dict: dict):
    """
    Setup logging based on configuration.

    Args:
        config_dict: Configuration dictionary
    """
    try:
        logging_config = config_dict.get("logging", {})

        # Create logs directory if it doesn't exist'
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        # Configure logging
        log_level = getattr(logging, logging_config.get("level", "INFO").upper())
        log_format = logging_config.get()
            "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Configure root logger
        logging.basicConfig()
            level=log_level,
            format=log_format,
            handlers=[]
                logging.StreamHandler(sys.stdout),
                logging.FileHandler()
                    logging_config.get("log_file", "logs/ferris_rde_daemon.log")
                ),
            ],
        )

        # Configure specific loggers
        loggers = {}
            "core": logging.getLogger("core"),
            "schwabot": logging.getLogger("schwabot"),
            "utils": logging.getLogger("utils"),
        }
        for logger_name, logger_instance in loggers.items():
            logger_instance.setLevel(log_level)

        info("✅ Logging configured successfully")

    except Exception as e:
        error(f"❌ Error setting up logging: {e}")
        # Fallback to basic logging
        logging.basicConfig(level=logging.INFO)


def create_pid_file(pid_file: str):
    """
    Create PID file for daemon process.

    Args:
        pid_file: Path to PID file
    """
    try:
        pid_dir = Path(pid_file).parent
        pid_dir.mkdir(exist_ok=True)

        with open(pid_file, "w") as f:
            f.write(str(os.getpid()))

        info(f"✅ PID file created: {pid_file}")

    except Exception as e:
        error(f"❌ Error creating PID file: {e}")


def remove_pid_file(pid_file: str):
    """
    Remove PID file.

    Args:
        pid_file: Path to PID file
    """
    try:
        if os.path.exists(pid_file):
            os.remove(pid_file)
            info(f"✅ PID file removed: {pid_file}")
    except Exception as e:
        error(f"❌ Error removing PID file: {e}")


async def run_daemon(config: DaemonConfig, pid_file: Optional[str] = None):
    """
    Run the Ferris RDE daemon.

    Args:
        config: Daemon configuration
        pid_file: Optional PID file path
    """
    daemon = None

    try:
        # Create PID file if specified
        if pid_file:
            create_pid_file(pid_file)

        # Initialize daemon
        info("🚀 Initializing Ferris RDE Daemon...")
        daemon = FerrisRDEDaemon(config)

        # Start daemon
        info("🚀 Starting Ferris RDE Daemon...")
        start_success = await daemon.start()

        if not start_success:
            error("❌ Failed to start daemon")
            return False

        success("✅ Daemon started successfully")

        # Keep daemon running
        info("🔄 Daemon is running. Press Ctrl+C to stop.")

        while daemon.running and not daemon.shutdown_requested:
            await asyncio.sleep(1)

        success("✅ Daemon stopped gracefully")
        return True

    except KeyboardInterrupt:
        info("⌨️ Keyboard interrupt received")
        return True
    except Exception as e:
        error(f"❌ Error running daemon: {e}")
        return False
    finally:
        # Cleanup
        if daemon and daemon.running:
            await daemon.stop()

        if pid_file:
            remove_pid_file(pid_file)


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    info(f"📡 Received signal {signum}, initiating shutdown...")

    # Get daemon instance and stop it
    daemon = get_daemon_instance()
    if daemon and daemon.running:
        asyncio.create_task(daemon.stop())


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Ferris RDE Daemon Startup Script")
    parser.add_argument()
        "--config",
        "-c",
        default="config/ferris_rde_daemon_config.yaml",
        help="Path to configuration file (default: config/ferris_rde_daemon_config.yaml)",
    )
    parser.add_argument()
        "--pid-file",
        "-p",
        default="logs/ferris_rde_daemon.pid",
        help="Path to PID file (default: logs/ferris_rde_daemon.pid)",
    )
    parser.add_argument()
        "--daemon", "-d", action="store_true", help="Run as daemon process"
    )
    parser.add_argument()
        "--test",
        "-t",
        action="store_true",
        help="Run in test mode with reduced intervals",
    )
    parser.add_argument()
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Load configuration
    config_dict = load_config(args.config)
    if not config_dict:
        error("❌ Failed to load configuration")
        sys.exit(1)

    # Setup logging
    setup_logging(config_dict)

    # Modify configuration for test mode
    if args.test:
        info("🧪 Running in test mode")
        config_dict["timing"]["tick_interval_seconds"] = 2.0
        config_dict["timing"]["health_check_interval_seconds"] = 10.0
        config_dict["timing"]["performance_report_interval_seconds"] = 30.0
        config_dict["trading"]["paper_trading"] = True
        config_dict["development"]["test_mode"] = True

    # Set verbose logging
    if args.verbose:
        config_dict["logging"]["level"] = "DEBUG"
        config_dict["daemon"]["log_level"] = "DEBUG"

    # Create daemon configuration
    config = create_daemon_config(config_dict)

    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run daemon
    success = asyncio.run(run_daemon(config, args.pid_file))

    if success:
        success("🎉 Daemon completed successfully")
        sys.exit(0)
    else:
        error("❌ Daemon failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
