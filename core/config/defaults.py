from typing import Dict, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

DEFAULT_CONFIGS: Dict[str, Dict[str, Any]] = {
    'matrix_response_paths.yaml': {
        'safe': 'hold',
        'warn': 'delay_entry',
        'fail': 'matrix_realign',
        'ZPE-risk': 'cooldown_abort',
        'thresholds': {
            'entropy': 0.75,
            'drift': 0.4,
            'anomaly': 0.85
        }
    },
    'tesseract_config.yaml': {
        'coherence_threshold': 0.85,
        'pattern_decay_rate': 0.1,
        'memory_window': 1000
    },
    'gpu_offload_config.yaml': {
        'thermal_threshold': 80.0,
        'memory_threshold': 0.9,
        'fallback_enabled': True
    },
    'logging.yaml': {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'detailed': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': 'INFO',
                'formatter': 'detailed'
            },
            'file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'INFO',
                'formatter': 'detailed',
                'filename': 'logs/schwabot.log',
                'maxBytes': 10485760, # 10MB
                'backupCount': 5
            },
            'error_file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'ERROR',
                'formatter': 'detailed',
                'filename': 'logs/schwabot_errors.log',
                'maxBytes': 10485760,
                'backupCount': 5
            }
        },
        'root': {
            'handlers': ['console', 'file', 'error_file'],
            'level': 'INFO'
        }
    },
    'validation_config.yaml': {
        'validation': {
            'enabled': True,
            'max_retries': 3,
            'timeout': 30
        }
    }
}

def ensure_configs_exist(config_loader):
    """
    Ensure all required configurations exist in the config directory.
    If a file does not exist, a default version will be created.
    Args:
        config_loader: An instance of ConfigLoader.
    """
    for filename, default_content in DEFAULT_CONFIGS.items():
        config_path = config_loader.config_dir / filename
        if not config_path.exists():
            try:
                config_loader.save_yaml(filename, default_content)
                logger.info(f"Created default config file: {filename}")
            except Exception as e:
                logger.error(f"Failed to create default config file {filename}: {e}")
        else:
            logger.debug(f"Config file already exists: {filename}") 