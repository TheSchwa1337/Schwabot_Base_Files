"""
Mathematical Structures Monitor
==============================

Monitors the performance of mathematical structures in the Forever Fractal system,
including Euler-based triggers, braided mechanophore structures, and cyclic number theory.
"""

import time
import numpy as np
import pandas as pd
from datetime import datetime
import logging
from typing import Dict, List, Any

from core.fractal_core import FractalCore
from core.spectral_state import SpectralState
from core.behavior_pattern_tracker import BehaviorPatternTracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MathematicalStructuresMonitor:
    """Monitor for mathematical structures performance"""
    
    def __init__(self, update_interval: float = 1.0):
        """Initialize the monitor
        
        Args:
            update_interval: Time between updates in seconds
        """
        self.update_interval = update_interval
        self.fractal_core = FractalCore()
        self.pattern_tracker = BehaviorPatternTracker()
        
        # Initialize metrics storage
        self.metrics_history: List[Dict[str, Any]] = []
        
    def collect_metrics(self) -> Dict[str, Any]:
        """Collect current metrics from mathematical structures
        
        Returns:
            Dictionary of current metrics
        """
        # Get current state
        state = SpectralState.create_initial_state()
        
        # Process through fractal core
        result = self.fractal_core.process_recursive_state(
            [0.1, 0.2, 0.3]  # Example vector
        )
        
        # Collect metrics
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'euler_phase': result['euler_phase'],
            'braid_group_size': len(self.fractal_core.braid_group),
            'simplicial_set_size': len(self.fractal_core.simplicial_set),
            'cyclic_patterns_count': len(self.fractal_core.cyclic_patterns),
            'pattern_reversal_key': self.fractal_core.pattern_reversal_key is not None,
            'memory_shell_entropy': np.mean(self.fractal_core._compute_memory_shell()),
            'post_euler_field_magnitude': np.linalg.norm(
                self.fractal_core.compute_post_euler_field(result['euler_phase'])
            )
        }
        
        return metrics
    
    def update_metrics(self):
        """Update metrics history"""
        try:
            metrics = self.collect_metrics()
            self.metrics_history.append(metrics)
            
            # Log key metrics
            logger.info(
                f"Euler Phase: {metrics['euler_phase']:.3f}, "
                f"Braid Group Size: {metrics['braid_group_size']}, "
                f"Cyclic Patterns: {metrics['cyclic_patterns_count']}"
            )
            
        except Exception as e:
            logger.error(f"Error updating metrics: {str(e)}")
    
    def analyze_metrics(self) -> Dict[str, Any]:
        """Analyze collected metrics
        
        Returns:
            Dictionary of analysis results
        """
        if not self.metrics_history:
            return {}
            
        # Convert to DataFrame for analysis
        df = pd.DataFrame(self.metrics_history)
        
        # Calculate statistics
        analysis = {
            'euler_phase': {
                'mean': df['euler_phase'].mean(),
                'std': df['euler_phase'].std(),
                'trend': np.polyfit(range(len(df)), df['euler_phase'], 1)[0]
            },
            'braid_group': {
                'mean_size': df['braid_group_size'].mean(),
                'growth_rate': np.polyfit(range(len(df)), df['braid_group_size'], 1)[0]
            },
            'cyclic_patterns': {
                'mean_count': df['cyclic_patterns_count'].mean(),
                'discovery_rate': np.polyfit(range(len(df)), df['cyclic_patterns_count'], 1)[0]
            },
            'memory_shell': {
                'mean_entropy': df['memory_shell_entropy'].mean(),
                'entropy_trend': np.polyfit(range(len(df)), df['memory_shell_entropy'], 1)[0]
            }
        }
        
        return analysis
    
    def run(self, duration: float = 3600.0):
        """Run the monitor for specified duration
        
        Args:
            duration: Duration to run in seconds
        """
        start_time = time.time()
        end_time = start_time + duration
        
        logger.info(f"Starting mathematical structures monitor for {duration} seconds")
        
        while time.time() < end_time:
            self.update_metrics()
            
            # Analyze metrics every minute
            if len(self.metrics_history) % 60 == 0:
                analysis = self.analyze_metrics()
                logger.info("Metrics Analysis:")
                logger.info(f"Euler Phase Trend: {analysis['euler_phase']['trend']:.3f}")
                logger.info(f"Braid Group Growth Rate: {analysis['braid_group']['growth_rate']:.3f}")
                logger.info(f"Cyclic Pattern Discovery Rate: {analysis['cyclic_patterns']['discovery_rate']:.3f}")
                logger.info(f"Memory Shell Entropy Trend: {analysis['memory_shell']['entropy_trend']:.3f}")
            
            time.sleep(self.update_interval)
        
        logger.info("Monitor completed")
        
        # Final analysis
        final_analysis = self.analyze_metrics()
        logger.info("Final Analysis:")
        logger.info(f"Average Euler Phase: {final_analysis['euler_phase']['mean']:.3f}")
        logger.info(f"Average Braid Group Size: {final_analysis['braid_group']['mean_size']:.3f}")
        logger.info(f"Average Cyclic Patterns: {final_analysis['cyclic_patterns']['mean_count']:.3f}")
        logger.info(f"Average Memory Shell Entropy: {final_analysis['memory_shell']['mean_entropy']:.3f}")

if __name__ == '__main__':
    monitor = MathematicalStructuresMonitor(update_interval=1.0)
    monitor.run(duration=3600)  # Run for 1 hour 