#!/usr/bin/env python3
"""
Quick Start Script for Unified Schwabot Integration
==================================================

This script provides a simple way to start the unified Schwabot integration system
with proper configuration and error handling.

Usage:
    python start_schwabot.py [--config config.json] [--log-level INFO]
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from core.unified_schwabot_integration import create_unified_schwabot_integration
    INTEGRATION_AVAILABLE = True
except ImportError as e:
    INTEGRATION_AVAILABLE = False
    print(f"❌ Core integration not available: {e}")
    print("Please ensure all core modules are properly installed.")

logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    try:
        # Create logs directory if it doesn't exist
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "schwabot.log"),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        logger.info(f"📝 Logging configured with level: {log_level}")
        
    except Exception as e:
        print(f"❌ Error setting up logging: {e}")
        # Fallback to basic logging
        logging.basicConfig(level=logging.INFO)


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from file or create default."""
    try:
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"✅ Loaded configuration from: {config_path}")
        else:
            # Create default configuration
            config = create_default_config()
            logger.info("✅ Using default configuration")
            
            # Save default config if no config file was provided
            if not config_path:
                default_config_path = "config.json"
                with open(default_config_path, 'w') as f:
                    json.dump(config, f, indent=2)
                logger.info(f"💾 Saved default configuration to: {default_config_path}")
        
        return config
        
    except Exception as e:
        logger.error(f"❌ Error loading configuration: {e}")
        return create_default_config()


def create_default_config() -> Dict[str, Any]:
    """Create default configuration."""
    return {
        "system": {
            "name": "Unified Schwabot Integration",
            "version": "1.0.0",
            "environment": "development"
        },
        "ai_models": {
            "gpt": {
                "api_key": "your-openai-api-key",
                "model_id": "gpt-4",
                "max_tokens": 1000,
                "temperature": 0.7,
                "enabled": False,
                "priority": 1
            },
            "claude": {
                "api_key": "your-anthropic-api-key",
                "model_id": "claude-3-sonnet-20240229",
                "max_tokens": 1000,
                "temperature": 0.7,
                "enabled": False,
                "priority": 2
            },
            "gemini": {
                "api_key": "your-google-api-key",
                "model_id": "gemini-pro",
                "max_tokens": 1000,
                "temperature": 0.7,
                "enabled": False,
                "priority": 3
            }
        },
        "exchanges": {
            "binance": {
                "enabled": True,
                "sandbox": True,
                "api_key": "your-binance-api-key",
                "secret": "your-binance-secret"
            },
            "coinbase": {
                "enabled": True,
                "sandbox": True,
                "api_key": "your-coinbase-api-key",
                "secret": "your-coinbase-secret"
            }
        },
        "entropy": {
            "threshold": 0.5,
            "update_interval": 225.0,
            "history_size": 1000
        },
        "websocket": {
            "host": "localhost",
            "port": 8765,
            "enabled": True
        },
        "api": {
            "host": "localhost",
            "port": 5000,
            "enabled": True,
            "debug": False
        },
        "logging": {
            "level": "INFO",
            "file": "logs/schwabot.log",
            "max_size": "10MB",
            "backup_count": 5
        },
        "performance": {
            "max_concurrent_requests": 10,
            "request_timeout": 30,
            "retry_attempts": 3
        }
    }


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration."""
    try:
        required_sections = ["system", "ai_models", "exchanges", "entropy"]
        
        for section in required_sections:
            if section not in config:
                logger.error(f"❌ Missing required configuration section: {section}")
                return False
        
        # Validate AI models
        ai_models = config.get("ai_models", {})
        for model_name, model_config in ai_models.items():
            if model_config.get("enabled", False):
                if not model_config.get("api_key") or model_config["api_key"] == f"your-{model_name}-api-key":
                    logger.warning(f"⚠️ {model_name} is enabled but API key not configured")
        
        # Validate exchanges
        exchanges = config.get("exchanges", {})
        for exchange_name, exchange_config in exchanges.items():
            if exchange_config.get("enabled", False):
                if not exchange_config.get("api_key") or exchange_config["api_key"] == f"your-{exchange_name}-api-key":
                    logger.warning(f"⚠️ {exchange_name} is enabled but API key not configured")
        
        logger.info("✅ Configuration validation passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Configuration validation failed: {e}")
        return False


def print_banner() -> None:
    """Print startup banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                    🧠 SCHWABOT UNIFIED                       ║
    ║                                                              ║
    ║              Unified Integration System v1.0.0              ║
    ║                                                              ║
    ║  🔄 Entropy-Driven Architecture                             ║
    ║  🤖 Multi-AI Integration (GPT-4, Claude, Gemini)           ║
    ║  🧮 16-Bit Positioning System                              ║
    ║  📊 10,000-Tick Map                                        ║
    ║  🔗 Real-Time Data Integration                             ║
    ║                                                              ║
    ║  Starting system...                                         ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_system_info(config: Dict[str, Any]) -> None:
    """Print system information."""
    try:
        system = config.get("system", {})
        print(f"📋 System: {system.get('name', 'Unknown')} v{system.get('version', 'Unknown')}")
        print(f"🌍 Environment: {system.get('environment', 'Unknown')}")
        
        # AI Models
        ai_models = config.get("ai_models", {})
        enabled_models = [name for name, cfg in ai_models.items() if cfg.get("enabled", False)]
        if enabled_models:
            print(f"🤖 AI Models: {', '.join(enabled_models)}")
        else:
            print("🤖 AI Models: None enabled (configure in config.json)")
        
        # Exchanges
        exchanges = config.get("exchanges", {})
        enabled_exchanges = [name for name, cfg in exchanges.items() if cfg.get("enabled", False)]
        if enabled_exchanges:
            print(f"📈 Exchanges: {', '.join(enabled_exchanges)}")
        else:
            print("📈 Exchanges: None enabled")
        
        # API Endpoints
        api_config = config.get("api", {})
        if api_config.get("enabled", False):
            print(f"🔗 API: http://{api_config.get('host', 'localhost')}:{api_config.get('port', 5000)}")
        
        # WebSocket
        ws_config = config.get("websocket", {})
        if ws_config.get("enabled", False):
            print(f"📡 WebSocket: ws://{ws_config.get('host', 'localhost')}:{ws_config.get('port', 8765)}")
        
        print("─" * 60)
        
    except Exception as e:
        logger.error(f"❌ Error printing system info: {e}")


async def run_system(config: Dict[str, Any]) -> None:
    """Run the unified Schwabot integration system."""
    integration = None
    
    try:
        if not INTEGRATION_AVAILABLE:
            logger.error("❌ Core integration modules not available")
            return
        
        logger.info("🚀 Creating unified Schwabot integration...")
        integration = create_unified_schwabot_integration(config=config)
        
        logger.info("🚀 Starting unified Schwabot integration...")
        await integration.start()
        
    except KeyboardInterrupt:
        logger.info("🛑 Received interrupt signal")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
    finally:
        if integration:
            logger.info("🛑 Stopping unified Schwabot integration...")
            await integration.stop()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Start the Unified Schwabot Integration System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python start_schwabot.py
  python start_schwabot.py --config my_config.json
  python start_schwabot.py --log-level DEBUG
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file (default: config.json)"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="Unified Schwabot Integration v1.0.0"
    )
    
    args = parser.parse_args()
    
    try:
        # Print banner
        print_banner()
        
        # Setup logging
        setup_logging(args.log_level)
        
        # Load configuration
        config = load_config(args.config)
        
        # Validate configuration
        if not validate_config(config):
            logger.error("❌ Configuration validation failed. Please check your config file.")
            sys.exit(1)
        
        # Print system information
        print_system_info(config)
        
        # Check for required dependencies
        check_dependencies()
        
        # Run the system
        asyncio.run(run_system(config))
        
    except KeyboardInterrupt:
        print("\n🛑 Shutdown complete")
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        sys.exit(1)


def check_dependencies() -> None:
    """Check for required dependencies."""
    try:
        required_packages = [
            "flask",
            "websockets", 
            "ccxt",
            "numpy",
            "pandas"
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            logger.warning(f"⚠️ Missing packages: {', '.join(missing_packages)}")
            logger.warning("💡 Install with: pip install -r requirements.txt")
        else:
            logger.info("✅ All required dependencies available")
        
        # Check for optional AI packages
        ai_packages = ["openai", "anthropic", "google.generativeai"]
        available_ai = []
        
        for package in ai_packages:
            try:
                __import__(package)
                available_ai.append(package)
            except ImportError:
                pass
        
        if available_ai:
            logger.info(f"🤖 Available AI packages: {', '.join(available_ai)}")
        else:
            logger.warning("⚠️ No AI packages available. Install with: pip install openai anthropic google-generativeai")
        
    except Exception as e:
        logger.error(f"❌ Error checking dependencies: {e}")


if __name__ == "__main__":
    main()
