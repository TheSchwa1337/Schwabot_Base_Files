"""
Logging configuration for Schwabot
Provides centralized logging configuration with file and console handlers.
"""

import logging
import logging.config
from pathlib import Path
from typing import Optional, Dict, Any

def setup_logging(log_dir: Optional[Path] = None, config_path: Optional[Path] = None) -> None:
    """Setup logging configuration for Schwabot.
    
    Args:
        log_dir: Optional directory for log files
        config_path: Optional path to logging config file
    """
    if log_dir is None:
        log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Default configuration
    default_config: Dict[str, Any] = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'detailed': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            },
            'simple': {
                'format': '%(levelname)s - %(message)s'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': 'INFO',
                'formatter': 'simple',
                'stream': 'ext://sys.stdout'
            },
            'file_handler': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'INFO',
                'formatter': 'detailed',
                'filename': str(log_dir / 'schwabot.log'),
                'maxBytes': 10 * 1024 * 1024,
                'backupCount': 5
            },
            'error_file_handler': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'ERROR',
                'formatter': 'detailed',
                'filename': str(log_dir / 'schwabot_errors.log'),
                'maxBytes': 10 * 1024 * 1024,
                'backupCount': 5
            }
        },
        'loggers': {
            '': {
                'handlers': ['console', 'file_handler', 'error_file_handler'],
                'level': 'INFO',
                'propagate': True
            },
            '__main__': {
                'handlers': ['console', 'file_handler'],
                'level': 'DEBUG',
                'propagate': False
            }
        }
    }
    
    try:
        logging.config.dictConfig(default_config)
        print(f"Logging configured with default settings. Logs in: {log_dir}")
    except Exception as e:
        print(f"FATAL: Could not configure logging: {e}")
        raise 