"""
RDE Visualization Module
======================

Visualizes RDE integration metrics, system performance, and drift bands.
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
from .system_metrics import SystemMonitor, SystemMetrics

class RDEVisualizer:
    """Visualizes RDE integration metrics and system performance"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.system_monitor = SystemMonitor(log_dir)
        
        # Color schemes
        self.strategy_colors = {
            "hold": "#3B82F6",      # Blue
            "flip": "#10B981",      # Green
            "hedge": "#F59E0B",     # Yellow
            "spec": "#EF4444",      # Red
            "stable_swap": "#8B5CF6", # Purple
            "aggressive": "#EC4899"   # Pink
        }
        
        # Mode colormap
        self.mode_cmap = plt.cm.viridis
    
    def plot_mode_weights(self, history: List[Dict], output_path: str):
        """Plot mode weight evolution with drift bands"""
        plt.figure(figsize=(12, 6))
        
        # Extract data
        timestamps = [datetime.fromisoformat(h['utc']) for h in history]
        modes = [h['bit_mode'] for h in history]
        weights = [h['weights'] for h in history]
        
        # Plot mode weights
        for i, mode in enumerate(set(modes)):
            mode_weights = [w[i] if i < len(w) else 0 for w in weights]
            plt.plot(timestamps, mode_weights, 
                    label=f'{mode}-bit', 
                    color=self.mode_cmap(i/len(set(modes))))
        
        # Add drift band indicators
        for h in history:
            if 'centroid_distance' in h:
                drift = self.system_monitor.calculate_drift_band(h['centroid_distance'])
                zpe_zone = self.system_monitor.determine_zpe_zone(drift)
                color = {
                    'SAFE': 'green',
                    'WARM': 'yellow',
                    'UNSAFE': 'red'
                }[zpe_zone]
                plt.axvline(x=datetime.fromisoformat(h['utc']), 
                          color=color, alpha=0.2)
        
        plt.title('Mode Weight Evolution with Drift Bands')
        plt.xlabel('Time')
        plt.ylabel('Weight')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_path)
        plt.close()
    
    def plot_strategy_performance(self, history: List[Dict], output_path: str):
        """Plot strategy performance with system metrics"""
        plt.figure(figsize=(12, 8))
        
        # Create subplots
        gs = plt.GridSpec(2, 1, height_ratios=[2, 1])
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])
        
        # Plot strategy performance
        for strategy, color in self.strategy_colors.items():
            performance = [h['strategy_performance'].get(strategy, 0) for h in history]
            timestamps = [datetime.fromisoformat(h['utc']) for h in history]
            ax1.plot(timestamps, performance, label=strategy, color=color)
        
        # Plot system metrics
        for h in history:
            if 'system_metrics' in h:
                metrics = h['system_metrics']
                ax2.plot(datetime.fromisoformat(h['utc']), 
                        metrics.gpu_utilization, 
                        'b-', alpha=0.5, label='GPU')
                ax2.plot(datetime.fromisoformat(h['utc']), 
                        metrics.cpu_utilization, 
                        'r-', alpha=0.5, label='CPU')
        
        # Customize plots
        ax1.set_title('Strategy Performance and System Metrics')
        ax1.set_ylabel('Performance')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Utilization (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    
    def plot_strategy_heatmap(self, history: List[Dict], output_path: str):
        """Plot strategy selection heatmap with drift bands"""
        plt.figure(figsize=(12, 6))
        
        # Create strategy matrix
        strategies = list(self.strategy_colors.keys())
        timestamps = [datetime.fromisoformat(h['utc']) for h in history]
        strategy_matrix = np.zeros((len(strategies), len(history)))
        
        for i, h in enumerate(history):
            for j, strategy in enumerate(strategies):
                if strategy in h['strategies']:
                    strategy_matrix[j, i] = h['weights'][h['strategies'].index(strategy)]
        
        # Plot heatmap
        sns.heatmap(strategy_matrix, 
                   xticklabels=[t.strftime('%H:%M') for t in timestamps],
                   yticklabels=strategies,
                   cmap='YlOrRd')
        
        # Add drift band indicators
        for i, h in enumerate(history):
            if 'centroid_distance' in h:
                drift = self.system_monitor.calculate_drift_band(h['centroid_distance'])
                zpe_zone = self.system_monitor.determine_zpe_zone(drift)
                color = {
                    'SAFE': 'green',
                    'WARM': 'yellow',
                    'UNSAFE': 'red'
                }[zpe_zone]
                plt.axvline(x=i, color=color, alpha=0.2)
        
        plt.title('Strategy Selection Heatmap with Drift Bands')
        plt.xlabel('Time')
        plt.ylabel('Strategy')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    
    def generate_dashboard(self, history: List[Dict], output_dir: Path):
        """Generate complete dashboard with all visualizations"""
        # Create output directory
        output_dir.mkdir(exist_ok=True)
        
        # Generate plots
        self.plot_mode_weights(history, str(output_dir / "mode_weights.png"))
        self.plot_strategy_performance(history, str(output_dir / "strategy_performance.png"))
        self.plot_strategy_heatmap(history, str(output_dir / "strategy_heatmap.png"))
        
        # Generate HTML dashboard
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>RDE Integration Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; }}
                .card {{ background: #f5f5f5; padding: 20px; border-radius: 8px; }}
                img {{ max-width: 100%; height: auto; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>RDE Integration Dashboard</h1>
                <div class="grid">
                    <div class="card">
                        <h2>Mode Weight Evolution</h2>
                        <img src="mode_weights.png" alt="Mode Weights">
                    </div>
                    <div class="card">
                        <h2>Strategy Performance</h2>
                        <img src="strategy_performance.png" alt="Strategy Performance">
                    </div>
                    <div class="card">
                        <h2>Strategy Selection Heatmap</h2>
                        <img src="strategy_heatmap.png" alt="Strategy Heatmap">
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        
        with open(output_dir / "dashboard.html", 'w') as f:
            f.write(html_content)

# Example usage:
"""
from ncco_core.rde_visuals import RDEVisualizer
from ncco_core.ferris_rde import FerrisRDE

# Initialize
ferris_rde = FerrisRDE("rde_config.yaml", "ferris_config.yaml")
visualizer = RDEVisualizer()

# Get history
history = ferris_rde.get_spin_history()

# Generate dashboard
visualizer.generate_dashboard(history, Path("rde_dashboard"))
""" 