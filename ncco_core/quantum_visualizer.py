"""
Quantum Visualization Module
=========================

Combines RDE and NCCO visualization capabilities for unified quantum-cellular pattern analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from .system_metrics import SystemMonitor, SystemMetrics
from .rde_visuals import RDEVisualizer

class QuantumVisualizer:
    """Unified visualization system for quantum-cellular patterns and RDE metrics"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.system_monitor = SystemMonitor(log_dir)
        self.rde_visualizer = RDEVisualizer(log_dir)
        
        # Quantum pattern colors
        self.pattern_colors = {
            'stable': '#10B981',    # Green
            'chaotic': '#EF4444',   # Red
            'transition': '#F59E0B', # Yellow
            'quantum': '#8B5CF6',   # Purple
            'cellular': '#3B82F6'   # Blue
        }
        
        # Initialize colormaps
        self.quantum_cmap = plt.cm.viridis
        self.pattern_cmap = plt.cm.plasma
    
    def plot_quantum_patterns(self, 
                            history: List[Dict], 
                            output_path: str,
                            dimensions: Tuple[int, int] = (4, 4)):
        """Plot quantum-cellular pattern evolution"""
        plt.figure(figsize=(15, 10))
        
        # Create subplots for different pattern aspects
        gs = plt.GridSpec(3, 2, height_ratios=[2, 1, 1])
        ax1 = plt.subplot(gs[0, :])  # Main pattern plot
        ax2 = plt.subplot(gs[1, 0])  # Magnitude plot
        ax3 = plt.subplot(gs[1, 1])  # Stability plot
        ax4 = plt.subplot(gs[2, :])  # System metrics
        
        # Extract timestamps
        timestamps = [datetime.fromisoformat(h['utc']) for h in history]
        
        # Plot main pattern evolution
        for i, h in enumerate(history):
            if 'pattern_metrics' in h:
                metrics = h['pattern_metrics']
                ax1.plot(timestamps[i], metrics['magnitude'], 
                        'o', color=self.pattern_colors['quantum'],
                        alpha=0.6)
                
                # Add drift band indicators
                if 'centroid_distance' in h:
                    drift = self.system_monitor.calculate_drift_band(
                        h['centroid_distance'])
                    zpe_zone = self.system_monitor.determine_zpe_zone(drift)
                    color = {
                        'SAFE': 'green',
                        'WARM': 'yellow',
                        'UNSAFE': 'red'
                    }[zpe_zone]
                    ax1.axvline(x=timestamps[i], color=color, alpha=0.2)
        
        # Plot magnitude evolution
        magnitudes = [h['pattern_metrics']['magnitude'] 
                     for h in history if 'pattern_metrics' in h]
        ax2.plot(timestamps[:len(magnitudes)], magnitudes, 
                color=self.pattern_colors['quantum'])
        ax2.set_title('Pattern Magnitude')
        ax2.grid(True, alpha=0.3)
        
        # Plot stability metrics
        stabilities = [h['pattern_metrics']['stability'] 
                      for h in history if 'pattern_metrics' in h]
        ax3.plot(timestamps[:len(stabilities)], stabilities,
                color=self.pattern_colors['stable'])
        ax3.set_title('Pattern Stability')
        ax3.grid(True, alpha=0.3)
        
        # Plot system metrics
        for h in history:
            if 'system_metrics' in h:
                metrics = h['system_metrics']
                ax4.plot(timestamps, metrics.gpu_utilization,
                        'b-', alpha=0.5, label='GPU')
                ax4.plot(timestamps, metrics.cpu_utilization,
                        'r-', alpha=0.5, label='CPU')
        
        # Customize plots
        ax1.set_title('Quantum-Cellular Pattern Evolution')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Pattern State')
        ax1.grid(True, alpha=0.3)
        
        ax4.set_xlabel('Time')
        ax4.set_ylabel('Utilization (%)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    
    def plot_pattern_heatmap(self, 
                            history: List[Dict], 
                            output_path: str,
                            dimensions: Tuple[int, int] = (4, 4)):
        """Plot pattern state heatmap with drift bands"""
        plt.figure(figsize=(12, 8))
        
        # Create pattern matrix
        pattern_states = ['stable', 'chaotic', 'transition', 'quantum', 'cellular']
        timestamps = [datetime.fromisoformat(h['utc']) for h in history]
        pattern_matrix = np.zeros((len(pattern_states), len(history)))
        
        for i, h in enumerate(history):
            if 'pattern_metrics' in h:
                metrics = h['pattern_metrics']
                # Calculate pattern state probabilities
                total = sum(metrics.values())
                for j, state in enumerate(pattern_states):
                    pattern_matrix[j, i] = metrics.get(state, 0) / total
        
        # Plot heatmap
        sns.heatmap(pattern_matrix,
                   xticklabels=[t.strftime('%H:%M') for t in timestamps],
                   yticklabels=pattern_states,
                   cmap='YlOrRd')
        
        # Add drift band indicators
        for i, h in enumerate(history):
            if 'centroid_distance' in h:
                drift = self.system_monitor.calculate_drift_band(
                    h['centroid_distance'])
                zpe_zone = self.system_monitor.determine_zpe_zone(drift)
                color = {
                    'SAFE': 'green',
                    'WARM': 'yellow',
                    'UNSAFE': 'red'
                }[zpe_zone]
                plt.axvline(x=i, color=color, alpha=0.2)
        
        plt.title('Pattern State Evolution with Drift Bands')
        plt.xlabel('Time')
        plt.ylabel('Pattern State')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    
    def generate_quantum_dashboard(self, 
                                 history: List[Dict], 
                                 output_dir: Path):
        """Generate complete quantum visualization dashboard"""
        # Create output directory
        output_dir.mkdir(exist_ok=True)
        
        # Generate RDE visualizations
        self.rde_visualizer.generate_dashboard(history, output_dir)
        
        # Generate quantum pattern visualizations
        self.plot_quantum_patterns(history, 
                                 str(output_dir / "quantum_patterns.png"))
        self.plot_pattern_heatmap(history, 
                                str(output_dir / "pattern_heatmap.png"))
        
        # Update HTML dashboard
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Quantum-Cellular Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; }}
                .card {{ background: #f5f5f5; padding: 20px; border-radius: 8px; }}
                img {{ max-width: 100%; height: auto; }}
                .section {{ margin-bottom: 40px; }}
                h1, h2 {{ color: #1F2937; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Quantum-Cellular Dashboard</h1>
                
                <div class="section">
                    <h2>RDE Integration</h2>
                    <div class="grid">
                        <div class="card">
                            <h3>Mode Weight Evolution</h3>
                            <img src="mode_weights.png" alt="Mode Weights">
                        </div>
                        <div class="card">
                            <h3>Strategy Performance</h3>
                            <img src="strategy_performance.png" alt="Strategy Performance">
                        </div>
                        <div class="card">
                            <h3>Strategy Selection Heatmap</h3>
                            <img src="strategy_heatmap.png" alt="Strategy Heatmap">
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>Quantum Patterns</h2>
                    <div class="grid">
                        <div class="card">
                            <h3>Pattern Evolution</h3>
                            <img src="quantum_patterns.png" alt="Quantum Patterns">
                        </div>
                        <div class="card">
                            <h3>Pattern State Heatmap</h3>
                            <img src="pattern_heatmap.png" alt="Pattern Heatmap">
                        </div>
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
from ncco_core.quantum_visualizer import QuantumVisualizer
from ncco_core.ferris_rde import FerrisRDE

# Initialize
ferris_rde = FerrisRDE("rde_config.yaml", "ferris_config.yaml")
visualizer = QuantumVisualizer()

# Get history
history = ferris_rde.get_spin_history()

# Generate dashboard
visualizer.generate_quantum_dashboard(history, Path("quantum_dashboard"))
""" 