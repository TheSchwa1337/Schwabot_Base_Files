"""
Line Render Engine
=================

Handles line rendering operations for the Schwabot visualization system.
Uses standardized config loading for consistent behavior and integrates
mathematical utilities for dynamic, context-aware rendering.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import yaml
from pathlib import Path
import psutil
from hashlib import sha256
from concurrent.futures import ThreadPoolExecutor
import threading

# Import standardized configuration management
from config import load_config, ensure_config_exists, ConfigError
from config.matrix_response_schema import LINE_RENDER_SCHEMA

# Import mathematical utility functions
from .render_math_utils import (
    calculate_line_score,
    determine_line_style,
    calculate_decay,
    adjust_line_thickness,
    calculate_line_opacity,
    determine_line_color,
    calculate_volatility_score,
    smooth_line_path,
    calculate_trend_strength
)

logger = logging.getLogger(__name__)

@dataclass
class LineState:
    """State of a rendered line with enhanced mathematical properties"""
    id: str
    path: List[float]
    score: float
    active: bool
    last_update: datetime
    profit: float = 0.0
    entropy: float = 0.0
    volatility: float = 0.0
    trend_strength: float = 0.0
    trend_direction: str = 'flat'
    opacity: float = 1.0
    color: str = '#FFFFFF'
    style: str = 'solid'
    thickness: int = 2

class LineRenderEngine:
    """
    Renders lines and visual elements for the system with mathematical
    enhancements and thermal/profit awareness.
    """
    
    def __init__(self, config_filename: str = 'line_render_engine_config.yaml'):
        """
        Initialize the line render engine with standardized configuration.
        
        Args:
            config_filename: Name of the configuration file
        """
        self.config_filename = config_filename
        
        try:
            # Ensure config exists with defaults
            default_config = LINE_RENDER_SCHEMA.default_values
            ensure_config_exists(config_filename, default_config)
            
            # Load configuration using standardized system
            self.config = load_config(config_filename, LINE_RENDER_SCHEMA.schema)
            logger.info(f"LineRenderEngine initialized with config: {config_filename}")
            
        except ConfigError as e:
            logger.error(f"Configuration error: {e}. Using default config.")
            self.config = LINE_RENDER_SCHEMA.default_values
        except Exception as e:
            logger.error(f"Unexpected error loading config: {e}. Using defaults.")
            self.config = LINE_RENDER_SCHEMA.default_values
            
        # Initialize state management
        self.line_history: Dict[str, LineState] = {}
        self.matrix_state = "hold"
        self._load_matrix_paths()
        
        # Initialize performance settings
        perf_settings = self.config.get('performance_settings', {})
        max_workers = perf_settings.get('max_workers', 4)
        self._memory_check_interval = perf_settings.get('memory_check_interval', 60)
        
        # Initialize thread pool for parallel processing
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = threading.RLock()  # Use RLock for nested locking
        
        # Initialize memory monitoring
        self._last_memory_check = datetime.now()
        
        # Performance tracking
        self._render_count = 0
        self._total_render_time = 0.0
        
    def _load_matrix_paths(self):
        """Load matrix response paths using standardized configuration"""
        try:
            # Use standardized config loading for matrix paths
            from config.matrix_response_schema import MATRIX_RESPONSE_SCHEMA
            
            matrix_config_file = 'matrix_response_paths.yaml'
            ensure_config_exists(matrix_config_file, MATRIX_RESPONSE_SCHEMA.default_values)
            
            self.matrix_paths = load_config(matrix_config_file, MATRIX_RESPONSE_SCHEMA.schema)
            logger.info(f"Matrix paths loaded successfully")
            
        except ConfigError as e:
            logger.warning(f"Error loading matrix paths: {e}. Using defaults.")
            from config.matrix_response_schema import MATRIX_RESPONSE_SCHEMA
            self.matrix_paths = MATRIX_RESPONSE_SCHEMA.default_values
        except Exception as e:
            logger.error(f"Unexpected error loading matrix paths: {e}")
            # Fallback to minimal default
            self.matrix_paths = {
                'default_paths': {
                    'hold': '/data/matrix/hold',
                    'active': '/data/matrix/active'
                }
            }
        
    def _generate_line_id(self, line_data: Dict[str, Any]) -> str:
        """Generate a unique ID for a line based on its data."""
        try:
            # Create a stable hash from key data elements
            key_data = {
                'path': line_data.get('path', []),
                'type': line_data.get('type', 'default'),
                'timestamp': line_data.get('timestamp', '')
            }
            data_str = str(sorted(key_data.items()))
            return sha256(data_str.encode('utf-8')).hexdigest()[:16]  # Shorter ID
        except Exception as e:
            logger.error(f"Error generating line ID: {e}")
            return sha256(str(datetime.now()).encode('utf-8')).hexdigest()[:16]

    def memory_check(self) -> Dict[str, float]:
        """Perform periodic memory usage check and return metrics."""
        try:
            if (datetime.now() - self._last_memory_check).seconds >= self._memory_check_interval:
                mem = psutil.virtual_memory()
                cpu = psutil.cpu_percent(interval=0.1)
                
                metrics = {
                    'memory_percent': mem.percent,
                    'memory_available_gb': mem.available / (1024**3),
                    'cpu_percent': cpu
                }
                
                logger.info(f"System metrics: Memory {mem.percent:.1f}%, CPU {cpu:.1f}%")
                self._last_memory_check = datetime.now()
                
                return metrics
            
            # Return cached values if check interval hasn't passed
            return {
                'memory_percent': psutil.virtual_memory().percent,
                'cpu_percent': psutil.cpu_percent(interval=None),
                'memory_available_gb': psutil.virtual_memory().available / (1024**3)
            }
            
        except Exception as e:
            logger.error(f"Error checking system metrics: {e}")
            return {'memory_percent': 50.0, 'cpu_percent': 50.0, 'memory_available_gb': 1.0}

    def render_lines(self, line_data_list: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Render lines with mathematical enhancements and system awareness.
        
        Args:
            line_data_list: List of dictionaries containing line information
            
        Returns:
            Dictionary containing detailed render results
        """
        start_time = datetime.now()
        
        try:
            # Get system metrics
            system_metrics = self.memory_check()
            
            # Get render settings
            render_settings = self.config.get('render_settings', {})
            resolution = render_settings.get('resolution', '1080p')
            background_color = render_settings.get('background_color', '#000000')
            base_line_thickness = render_settings.get('line_thickness', 2)
            half_life_seconds = render_settings.get('line_decay_half_life_seconds', 3600)
            
            # Adjust line thickness based on system load
            current_line_thickness = adjust_line_thickness(
                base_line_thickness, 
                system_metrics['memory_percent'],
                system_metrics['cpu_percent']
            )
            
            rendered_lines_info = []
            
            if line_data_list:
                with self._lock:
                    for line_data in line_data_list:
                        try:
                            line_info = self._process_single_line(
                                line_data, 
                                current_line_thickness, 
                                half_life_seconds
                            )
                            if line_info:
                                rendered_lines_info.append(line_info)
                        except Exception as e:
                            logger.error(f"Error processing line: {e}")
                            continue
            
            # Calculate performance metrics
            render_time = (datetime.now() - start_time).total_seconds()
            self._render_count += 1
            self._total_render_time += render_time
            avg_render_time = self._total_render_time / self._render_count
            
            # Build comprehensive result
            render_result = {
                'status': 'rendered',
                'resolution': resolution,
                'background_color': background_color,
                'base_line_thickness': base_line_thickness,
                'adjusted_line_thickness': current_line_thickness,
                'lines_rendered_count': len(rendered_lines_info),
                'rendered_lines_details': rendered_lines_info,
                'system_metrics': system_metrics,
                'performance': {
                    'render_time_seconds': render_time,
                    'average_render_time': avg_render_time,
                    'total_renders': self._render_count
                },
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Successfully rendered {len(rendered_lines_info)} lines in {render_time:.3f}s")
            return render_result
            
        except Exception as e:
            logger.error(f"Error in render_lines: {e}", exc_info=True)
            return {
                'status': 'error',
                'error': str(e),
                'lines_rendered_count': 0,
                'rendered_lines_details': [],
                'timestamp': datetime.now().isoformat()
            }
    
    def _process_single_line(self, line_data: Dict[str, Any], 
                           thickness: int, half_life_seconds: int) -> Optional[Dict[str, Any]]:
        """Process a single line with mathematical enhancements."""
        try:
            line_id = self._generate_line_id(line_data)
            
            # Extract data with defaults
            profit = line_data.get('profit', 0.0)
            entropy = line_data.get('entropy', 0.5)
            last_update = line_data.get('last_update', datetime.now())
            path_data = line_data.get('path', line_data.get('data', []))
            
            # Ensure path_data is a list of numbers
            if isinstance(path_data, (list, tuple)):
                path_data = [float(x) for x in path_data if isinstance(x, (int, float))]
            else:
                path_data = []
            
            # Calculate mathematical properties
            score = calculate_line_score(profit, entropy)
            style = determine_line_style(entropy)
            decay_factor = calculate_decay(last_update, half_life_seconds)
            
            # Calculate additional metrics if path data is available
            volatility = calculate_volatility_score(path_data) if path_data else 0.0
            trend_strength, trend_direction = calculate_trend_strength(path_data) if len(path_data) >= 3 else (0.0, 'flat')
            
            # Calculate visual properties
            opacity = calculate_line_opacity(decay_factor, confidence=abs(score))
            color = determine_line_color(score, entropy)
            
            # Smooth path if enabled and data is available
            if path_data and len(path_data) > 2:
                path_data = smooth_line_path(path_data, smoothing_factor=0.2)
            
            # Update or create line state
            line_state = LineState(
                id=line_id,
                path=path_data,
                score=score,
                active=True,
                last_update=datetime.now(),
                profit=profit,
                entropy=entropy,
                volatility=volatility,
                trend_strength=trend_strength,
                trend_direction=trend_direction,
                opacity=opacity,
                color=color,
                style=style,
                thickness=thickness
            )
            
            # Store in history
            self.line_history[line_id] = line_state
            
            # Create return info
            line_info = {
                'id': line_id,
                'score': score,
                'style': style,
                'opacity': opacity,
                'color': color,
                'thickness': thickness,
                'volatility': volatility,
                'trend_strength': trend_strength,
                'trend_direction': trend_direction,
                'decay_factor': decay_factor,
                'path_length': len(path_data),
                'original_data': line_data
            }
            
            logger.debug(f"Processed line {line_id}: score={score:.3f}, opacity={opacity:.3f}, "
                        f"volatility={volatility:.4f}, trend={trend_direction}")
            
            return line_info
            
        except Exception as e:
            logger.error(f"Error processing single line: {e}")
            return None
    
    def get_line_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about rendered lines."""
        with self._lock:
            if not self.line_history:
                return {'total_lines': 0, 'active_lines': 0}
            
            active_lines = [line for line in self.line_history.values() if line.active]
            
            stats = {
                'total_lines': len(self.line_history),
                'active_lines': len(active_lines),
                'average_score': np.mean([line.score for line in active_lines]) if active_lines else 0.0,
                'average_volatility': np.mean([line.volatility for line in active_lines]) if active_lines else 0.0,
                'trend_distribution': {},
                'style_distribution': {},
                'performance': {
                    'total_renders': self._render_count,
                    'average_render_time': self._total_render_time / max(1, self._render_count)
                }
            }
            
            # Calculate distributions
            for line in active_lines:
                # Trend distribution
                trend = line.trend_direction
                stats['trend_distribution'][trend] = stats['trend_distribution'].get(trend, 0) + 1
                
                # Style distribution
                style = line.style
                stats['style_distribution'][style] = stats['style_distribution'].get(style, 0) + 1
            
            return stats
    
    def cleanup_old_lines(self, max_age_hours: int = 24) -> int:
        """Remove old inactive lines from history."""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        removed_count = 0
        
        with self._lock:
            lines_to_remove = [
                line_id for line_id, line_state in self.line_history.items()
                if line_state.last_update < cutoff_time and not line_state.active
            ]
            
            for line_id in lines_to_remove:
                del self.line_history[line_id]
                removed_count += 1
        
        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} old lines")
        
        return removed_count
    
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
        """Update configuration with validation"""
        try:
            # Validate new config structure
            if 'render_settings' in new_config:
                render_settings = new_config['render_settings']
                if 'line_thickness' in render_settings:
                    thickness = render_settings['line_thickness']
                    if not isinstance(thickness, int) or thickness < 1:
                        raise ValueError("line_thickness must be a positive integer")
            
            self.config.update(new_config)
            logger.info("LineRenderEngine configuration updated successfully")
            
        except Exception as e:
            logger.error(f"Error updating configuration: {e}")
            raise
    
    def shutdown(self) -> None:
        """Gracefully shutdown the render engine."""
        logger.info("Shutting down LineRenderEngine...")
        
        try:
            # Cleanup old lines
            self.cleanup_old_lines(max_age_hours=1)
            
            # Shutdown thread pool
            if hasattr(self, '_executor') and self._executor:
                self._executor.shutdown(wait=True)
                logger.info("Thread pool executor shut down")
            
            logger.info("LineRenderEngine shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            self.shutdown()
        except Exception:
            pass  # Ignore errors during cleanup 