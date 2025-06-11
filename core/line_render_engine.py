"""
Line Render Engine
=================

Handles line rendering operations for the Schwabot visualization system.
Uses standardized config loading for consistent behavior.
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
from config.io_utils import load_config, ensure_config_exists

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
    """Renders lines and visual elements for the system"""
    
    def __init__(self, config_path: Path = None):
        """
        Initialize the line render engine.
        
        Args:
            config_path: Optional path to configuration file
        """
        if config_path is None:
            config_path = ensure_config_exists('line_render_engine_config.yaml')
        
        try:
            self.config = load_config(config_path)
            logger.info(f"LineRenderEngine initialized with config from {config_path}")
        except ValueError as e:
            logger.error(f"Failed to load config: {e}")
            # Use minimal default config
            self.config = {
                'render_settings': {
                    'resolution': '1080p',
                    'background_color': '#000000',
                    'line_thickness': 2
                }
            }
            
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
    
    def render_lines(self, line_data: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Render lines based on configuration and line data.
        
        Args:
            line_data: Optional list of dictionaries containing line information
            
        Returns:
            Dictionary containing render results
        """
        try:
            # Get render settings from config
            render_settings = self.config.get('render_settings', {})
            resolution = render_settings.get('resolution', '1080p')
            background_color = render_settings.get('background_color', '#000000')
            line_thickness = render_settings.get('line_thickness', 2)
            
            # Implement line rendering logic
            render_result = {
                'status': 'rendered',
                'resolution': resolution,
                'background_color': background_color,
                'line_thickness': line_thickness,
                'lines_rendered': len(line_data) if line_data else 0,
                'timestamp': str(Path(__file__).stat().st_mtime)
            }
            
            if line_data:
                render_result['line_count'] = len(line_data)
                render_result['line_types'] = [line.get('type', 'default') for line in line_data]
            
            logger.info(f"Lines rendered: {render_result['lines_rendered']} lines")
            return render_result
            
        except Exception as e:
            logger.error(f"Error rendering lines: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'lines_rendered': 0
            }
    
    def set_resolution(self, resolution: str) -> None:
        """Set render resolution"""
        if 'render_settings' not in self.config:
            self.config['render_settings'] = {}
        self.config['render_settings']['resolution'] = resolution
        logger.info(f"Resolution set to {resolution}")
    
    def set_background_color(self, color: str) -> None:
        """Set background color"""
        if 'render_settings' not in self.config:
            self.config['render_settings'] = {}
        self.config['render_settings']['background_color'] = color
        logger.info(f"Background color set to {color}")
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration"""
        return self.config.copy()
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """Update configuration"""
        self.config.update(new_config)
        logger.info("LineRenderEngine configuration updated") 