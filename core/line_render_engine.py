"""
Line Render Engine
Implements line-by-line rendering of tick data with safety checks.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import yaml
from pathlib import Path
import psutil
from hashlib import sha256
from concurrent.futures import ThreadPoolExecutor
import threading

from .config import load_yaml_config, ConfigError, MATRIX_RESPONSE_SCHEMA

logger = logging.getLogger(__name__)

@dataclass
class LineState:
    """State of a rendered line"""
    path: List[float]
    score: float
    active: bool
    last_update: datetime

class LineRenderEngine:
    """Implements line-by-line rendering of tick data with safety checks"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize line render engine.
        
        Args:
            config_path: Optional path to configuration file
        """
        # Load configuration
        try:
            if config_path:
                with open(config_path, 'r') as f:
                    self.config = yaml.safe_load(f)
            else:
                self.config = {}
            logger.info("Line render engine initialized with configuration")
        except Exception as e:
            logger.error(f"Failed to load line render engine config: {e}")
            self.config = {}
            
        # Initialize state
        self.line_history: List[LineState] = []
        self.matrix_state = "hold"
        self.load_matrix_paths()
        
        # Initialize thread pool for parallel processing
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._lock = threading.Lock()
        
        # Initialize memory monitoring
        self._last_memory_check = datetime.now()
        self._memory_check_interval = 60  # seconds
    
    def load_matrix_paths(self):
        """Load matrix response paths from YAML with validation"""
        try:
            self.matrix_paths = load_yaml_config('matrix_response_paths.yaml', 
                                               schema=MATRIX_RESPONSE_SCHEMA)
        except ConfigError as e:
            logger.warning(f"Error loading matrix paths: {e}, using default paths")
            self.matrix_paths = MATRIX_RESPONSE_SCHEMA.default_values 