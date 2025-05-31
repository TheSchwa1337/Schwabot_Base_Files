#!/usr/bin/env python3
"""
Tesseract Control Script
Manages tesseract pattern processing and visualization.
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from aleph_core.tesseract import TesseractProcessor

class TesseractController:
    """Controls tesseract pattern processing and visualization."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.processor = TesseractProcessor()
        self.config = self._load_config(config_path)
        self.pattern_history = []
        self.metrics_history = []
        self.coherence_threshold = 0.8
        self.homeostasis_threshold = 0.7
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from file or use defaults."""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        
        return {
            'update_interval': 1.0,
            'history_size': 1000,
            'visualization': {
                'enabled': True,
                'port': 3000,
                'theme': 'dark'
            },
            'processing': {
                'coherence_threshold': 0.8,
                'homeostasis_threshold': 0.7,
                'pattern_window': 100
            }
        }
    
    def process_pattern(self, pattern_hash: str) -> Dict:
        """Process a pattern hash and return metrics."""
        # Extract pattern
        pattern = self.processor.extract_tesseract_pattern(pattern_hash)
        
        # Calculate metrics
        metrics = self.processor.calculate_tesseract_metrics(pattern)
        
        # Update history
        self.pattern_history.append(pattern)
        self.metrics_history.append(metrics)
        
        # Trim history if needed
        if len(self.pattern_history) > self.config['history_size']:
            self.pattern_history.pop(0)
            self.metrics_history.pop(0)
        
        return {
            'pattern': pattern,
            'metrics': metrics,
            'coherence': self._calculate_coherence(),
            'homeostasis': self._calculate_homeostasis()
        }
    
    def _calculate_coherence(self) -> float:
        """Calculate pattern coherence."""
        if len(self.pattern_history) < 2:
            return 1.0
            
        # Calculate pattern stability
        recent_patterns = self.pattern_history[-5:]
        pattern_variance = np.var([np.array(p) for p in recent_patterns])
        coherence = 1.0 / (1.0 + pattern_variance)
        
        return min(1.0, max(0.0, coherence))
    
    def _calculate_homeostasis(self) -> float:
        """Calculate pattern homeostasis."""
        if len(self.metrics_history) < 2:
            return 1.0
            
        # Calculate metric stability
        recent_metrics = self.metrics_history[-5:]
        metric_variance = np.var([m['stability'] for m in recent_metrics])
        homeostasis = 1.0 / (1.0 + metric_variance)
        
        return min(1.0, max(0.0, homeostasis))
    
    def get_visualization_data(self) -> Dict:
        """Get data for visualization."""
        if not self.pattern_history:
            return {
                'dimensions': [0] * 8,
                'metrics': {
                    'magnitude': 0,
                    'centroid_distance': 0,
                    'axis_correlation': 0,
                    'stability': 0,
                    'harmonic_ratio': 0,
                    'primary_dominance': 0,
                    'dimensional_spread': 0
                },
                'coherence': 1.0,
                'homeostasis': 1.0
            }
        
        current_pattern = self.pattern_history[-1]
        current_metrics = self.metrics_history[-1]
        
        return {
            'dimensions': current_pattern,
            'metrics': current_metrics,
            'coherence': self._calculate_coherence(),
            'homeostasis': self._calculate_homeostasis()
        }
    
    def check_health(self) -> Dict:
        """Check system health status."""
        coherence = self._calculate_coherence()
        homeostasis = self._calculate_homeostasis()
        
        return {
            'status': 'healthy' if coherence > self.coherence_threshold and 
                                homeostasis > self.homeostasis_threshold else 'degraded',
            'coherence': coherence,
            'homeostasis': homeostasis,
            'pattern_count': len(self.pattern_history),
            'last_update': time.time()
        }

def main():
    """Main entry point for the control script."""
    controller = TesseractController()
    
    # Example usage
    test_hash = "a" * 64  # Example hash
    result = controller.process_pattern(test_hash)
    print(json.dumps(result, indent=2))
    
    health = controller.check_health()
    print("\nHealth Status:")
    print(json.dumps(health, indent=2))

if __name__ == "__main__":
    main() 