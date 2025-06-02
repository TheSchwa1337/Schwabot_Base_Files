"""
Quantum Visualization Module
=========================

Implements advanced quantum resonance patterns and fractal analysis for NEXUS SCHWA TYPE ÆONIK.
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
        
        # Quantum pattern colors with resonance states
        self.pattern_colors = {
            'stable': '#10B981',    # Green
            'chaotic': '#EF4444',   # Red
            'transition': '#F59E0B', # Yellow
            'quantum': '#8B5CF6',   # Purple
            'cellular': '#3B82F6',  # Blue
            'resonance': '#EC4899', # Pink
            'fractal': '#14B8A6'    # Teal
        }
        
        # Initialize colormaps
        self.quantum_cmap = plt.cm.viridis
        self.pattern_cmap = plt.cm.plasma
        self.resonance_cmap = plt.cm.magma
        
        # Initialize resonance tracking
        self.resonance_history = {
            'magnitude': [],
            'phase': [],
            'coherence': []
        }
        
        # Initialize fractal patterns
        self.fractal_patterns = self._initialize_fractal_patterns()
        
        # Resonance thresholds
        self.resonance_thresholds = {
            'magnitude': 0.5,
            'phase': np.pi/4,
            'coherence': 0.7
        }

    def _initialize_fractal_patterns(self) -> Dict[str, np.ndarray]:
        """Initialize fractal patterns for quantum visualization"""
        patterns = {}
        for pattern_type in ['quantum', 'cellular', 'resonance']:
            # Create base fractal pattern using golden ratio
            base = np.zeros((256, 256))
            phi = 0.618033988749895
            for i in range(256):
                for j in range(256):
                    # Different patterns for different types
                    if pattern_type == 'quantum':
                        base[i,j] = np.sin(i * phi) * np.cos(j * phi)
                    elif pattern_type == 'cellular':
                        base[i,j] = np.sin(i * phi + j * phi) * np.cos(i * phi - j * phi)
                    else:  # resonance
                        base[i,j] = np.sin(i * phi) * np.sin(j * phi)
            patterns[pattern_type] = base
        return patterns

    def _compute_quantum_resonance(self, pattern_metrics: Dict) -> Dict[str, float]:
        """Compute quantum resonance metrics from pattern data"""
        # Extract relevant metrics
        magnitude = pattern_metrics.get('magnitude', 0.0)
        stability = pattern_metrics.get('stability', 0.0)
        
        # Compute resonance metrics
        resonance_magnitude = magnitude * stability
        resonance_phase = np.arctan2(stability, magnitude)
        resonance_coherence = np.sqrt(magnitude**2 + stability**2)
        
        return {
            'magnitude': float(resonance_magnitude),
            'phase': float(resonance_phase),
            'coherence': float(resonance_coherence)
        }

    def _apply_fractal_pattern(self, pattern: np.ndarray, pattern_type: str) -> np.ndarray:
        """Apply fractal pattern to quantum pattern data"""
        fractal = self.fractal_patterns[pattern_type]
        # Resize fractal to match pattern
        fractal_resized = np.resize(fractal, pattern.shape)
        # Combine patterns with quantum resonance
        return pattern * fractal_resized

    def plot_quantum_patterns(self, 
                            history: List[Dict], 
                            output_path: str,
                            dimensions: Tuple[int, int] = (4, 4)):
        """Plot quantum-cellular pattern evolution with resonance analysis"""
        plt.figure(figsize=(15, 10))
        
        # Create subplots for different pattern aspects
        gs = plt.GridSpec(3, 2, height_ratios=[2, 1, 1])
        ax1 = plt.subplot(gs[0, :])  # Main pattern plot
        ax2 = plt.subplot(gs[1, 0])  # Resonance plot
        ax3 = plt.subplot(gs[1, 1])  # Fractal plot
        ax4 = plt.subplot(gs[2, :])  # System metrics
        
        # Extract timestamps
        timestamps = [datetime.fromisoformat(h['utc']) for h in history]
        
        # Plot main pattern evolution with resonance
        for i, h in enumerate(history):
            if 'pattern_metrics' in h:
                metrics = h['pattern_metrics']
                resonance = self._compute_quantum_resonance(metrics)
                
                # Update resonance history
                self.resonance_history['magnitude'].append(resonance['magnitude'])
                self.resonance_history['phase'].append(resonance['phase'])
                self.resonance_history['coherence'].append(resonance['coherence'])
                
                # Plot pattern with resonance
                ax1.plot(timestamps[i], metrics['magnitude'], 
                        'o', color=self.pattern_colors['quantum'],
                        alpha=0.6)
                
                # Add resonance indicators
                if resonance['magnitude'] > self.resonance_thresholds['magnitude']:
                    ax1.scatter(timestamps[i], resonance['magnitude'],
                              c=self.pattern_colors['resonance'],
                              s=100, alpha=0.8)
                
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
        
        # Plot resonance evolution
        ax2.plot(timestamps, self.resonance_history['magnitude'],
                color=self.pattern_colors['resonance'],
                label='Magnitude')
        ax2.plot(timestamps, self.resonance_history['coherence'],
                color=self.pattern_colors['fractal'],
                label='Coherence')
        ax2.set_title('Quantum Resonance Evolution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot fractal patterns
        for pattern_type, pattern in self.fractal_patterns.items():
            ax3.imshow(pattern, cmap=self.resonance_cmap,
                      alpha=0.7, label=pattern_type)
        ax3.set_title('Fractal Pattern Overlay')
        
        # Plot system metrics
        for h in history:
            if 'system_metrics' in h:
                metrics = h['system_metrics']
                ax4.plot(timestamps, metrics.gpu_utilization,
                        'b-', alpha=0.5, label='GPU')
                ax4.plot(timestamps, metrics.cpu_utilization,
                        'r-', alpha=0.5, label='CPU')
        
        # Customize plots
        ax1.set_title('Quantum-Cellular Pattern Evolution with Resonance')
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

    def generate_quantum_dashboard(self, 
                                 history: List[Dict], 
                                 output_dir: Path):
        """Generate complete quantum visualization dashboard with resonance analysis"""
        # Create output directory
        output_dir.mkdir(exist_ok=True)
        
        # Generate RDE visualizations
        self.rde_visualizer.generate_dashboard(history, output_dir)
        
        # Generate quantum pattern visualizations
        self.plot_quantum_patterns(history, 
                                 str(output_dir / "quantum_patterns.png"))
        self.plot_pattern_heatmap(history, 
                                str(output_dir / "pattern_heatmap.png"))
        
        # Generate resonance analysis
        self._plot_resonance_analysis(history,
                                    str(output_dir / "resonance_analysis.png"))
        
        # Update HTML dashboard
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>NEXUS SCHWA TYPE ÆONIK Dashboard</title>
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
                <h1>NEXUS SCHWA TYPE ÆONIK Dashboard</h1>
                
                <div class="section">
                    <h2>Quantum Resonance Analysis</h2>
                    <div class="grid">
                        <div class="card">
                            <h3>Pattern Evolution</h3>
                            <img src="quantum_patterns.png" alt="Quantum Patterns">
                        </div>
                        <div class="card">
                            <h3>Resonance Analysis</h3>
                            <img src="resonance_analysis.png" alt="Resonance Analysis">
                        </div>
                        <div class="card">
                            <h3>Pattern State Heatmap</h3>
                            <img src="pattern_heatmap.png" alt="Pattern Heatmap">
                        </div>
                    </div>
                </div>
                
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
            </div>
        </body>
        </html>
        """
        
        with open(output_dir / "dashboard.html", 'w') as f:
            f.write(html_content)

    def _plot_resonance_analysis(self, history: List[Dict], output_path: str):
        """Plot detailed resonance analysis"""
        plt.figure(figsize=(15, 10))
        
        # Create subplots
        gs = plt.GridSpec(2, 2)
        ax1 = plt.subplot(gs[0, 0])  # Magnitude vs Phase
        ax2 = plt.subplot(gs[0, 1])  # Coherence Evolution
        ax3 = plt.subplot(gs[1, :])  # Combined Analysis
        
        # Extract timestamps
        timestamps = [datetime.fromisoformat(h['utc']) for h in history]
        
        # Plot magnitude vs phase
        magnitudes = []
        phases = []
        for h in history:
            if 'pattern_metrics' in h:
                resonance = self._compute_quantum_resonance(h['pattern_metrics'])
                magnitudes.append(resonance['magnitude'])
                phases.append(resonance['phase'])
        
        ax1.scatter(phases, magnitudes, c=magnitudes, cmap=self.resonance_cmap)
        ax1.set_title('Resonance Magnitude vs Phase')
        ax1.set_xlabel('Phase')
        ax1.set_ylabel('Magnitude')
        
        # Plot coherence evolution
        coherences = [h['pattern_metrics'].get('coherence', 0.0) 
                     for h in history if 'pattern_metrics' in h]
        ax2.plot(timestamps[:len(coherences)], coherences,
                color=self.pattern_colors['resonance'])
        ax2.set_title('Coherence Evolution')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Coherence')
        
        # Plot combined analysis
        for i, h in enumerate(history):
            if 'pattern_metrics' in h:
                metrics = h['pattern_metrics']
                resonance = self._compute_quantum_resonance(metrics)
                
                # Plot resonance state
                ax3.scatter(timestamps[i], resonance['magnitude'],
                          c=self.pattern_colors['resonance'],
                          s=100, alpha=0.6)
                
                # Add threshold indicators
                if resonance['magnitude'] > self.resonance_thresholds['magnitude']:
                    ax3.axhline(y=resonance['magnitude'],
                              color=self.pattern_colors['resonance'],
                              alpha=0.2)
        
        ax3.set_title('Combined Resonance Analysis')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Resonance State')
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

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