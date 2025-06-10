import logging
import logging.config
import logging.handlers
from pathlib import Path
import yaml
import os
from .defaults import DEFAULT_CONFIGS

def setup_logging(log_dir: Path = None, config_path: Path = None):
    """
    Configure comprehensive logging for the Schwabot system.
    Looks for logging.yaml in the config directory or uses a default configuration.
    Args:
        log_dir: Optional. The directory where log files should be stored.
        config_path: Optional. Path to a custom logging YAML configuration file.
    """
    repo_root = Path(__file__).resolve().parent.parent
    if log_dir is None:
        log_dir = repo_root / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    # Update handlers in DEFAULT_CONFIGS for correct log_dir path
    if 'logging.yaml' in DEFAULT_CONFIGS:
        default_logging_config = DEFAULT_CONFIGS['logging.yaml']
        if 'handlers' in default_logging_config:
            if 'file' in default_logging_config['handlers']:
                default_logging_config['handlers']['file']['filename'] = str(log_dir / 'schwabot.log')
            if 'error_file' in default_logging_config['handlers']:
                default_logging_config['handlers']['error_file']['filename'] = str(log_dir / 'schwabot_errors.log')

    # Try to load logging configuration from logging.yaml
    if config_path is None:
        config_path = repo_root / 'config' / 'logging.yaml'
    if config_path.exists():
        with open(config_path, 'r') as f:
            log_config_from_file = yaml.safe_load(f)
            if log_config_from_file:
                logging.config.dictConfig(log_config_from_file)
                print(f"Logging configured from file: {config_path}")
                return

    # Fallback to hardcoded default logging configuration
    default_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'detailed': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
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
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') 