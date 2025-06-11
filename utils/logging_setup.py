"""
Logging setup utility for Schwabot.
Handles configuration and initialization of the logging system.
"""

import logging
import logging.config
import yaml
from pathlib import Path
import os

def setup_logging(
    default_path: str = "config/logging.yaml",
    default_level: int = logging.INFO,
    env_key: str = "SCHWABOT_LOG_CONFIG"
) -> None:
    """
    Setup logging configuration.
    
    Args:
        default_path: Path to the logging configuration file
        default_level: Default logging level if config file is not found
        env_key: Environment variable name for custom config path
    """
    path = os.getenv(env_key, default_path)
    config_path = Path(path)
    
    if config_path.exists():
        with open(config_path, "r") as f:
            try:
                config = yaml.safe_load(f)
                # Ensure log directories exist
                log_dir = Path("logs")
                log_dir.mkdir(exist_ok=True)
                
                # Apply configuration
                logging.config.dictConfig(config)
            except Exception as e:
                print(f"Error loading logging config: {e}")
                logging.basicConfig(level=default_level)
    else:
        logging.basicConfig(level=default_level)
        print(f"Failed to load logging config from {path}. Using default config.")

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Name of the logger
        
    Returns:
        logging.Logger: Configured logger instance
    """
    return logging.getLogger(name) 