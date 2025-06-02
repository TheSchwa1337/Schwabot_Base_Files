"""
Plot Sign Engine
==============

Implements advanced tensor decomposition and plot sign extraction for NEXUS SCHWA TYPE ÆONIK.
Provides interpretable latent signals from complex market dynamics using CP, Tucker, and PARAFAC2 decompositions.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.fft import fft2
from scipy.signal import hilbert
import tensorly as tl
from tensorly.decomposition import parafac, tucker
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

@dataclass
class PlotSign:
    """Represents a plot sign with its components and interpretation"""
    name: str
    magnitude: float
    phase: float
    coherence: float
    interpretation: str
    confidence: float
    tensor_factors: Dict[str, np.ndarray]

class PlotSignEngine:
    """Engine for extracting interpretable plot signs from market tensors"""
    
    def __init__(
        self,
        num_components: int = 3,
        history_size: int = 100,
        resonance_threshold: float = 0.7
    ):
        self.num_components = num_components
        self.history_size = history_size
        self.resonance_threshold = resonance_threshold
        
        # Initialize tensor decomposition models
        self.cp_model = None
        self.tucker_model = None
        self.parafac2_model = None
        
        # Initialize clustering for plot sign classification
        self.kmeans = KMeans(n_clusters=num_components)
        self.scaler = StandardScaler()
        
        # History tracking
        self.latent_factor_history = []
        self.plot_sign_history = []
        
        # Resonance tracking
        self.resonance_history = {
            'magnitude': [],
            'phase': [],
            'coherence': []
        }
        
        # Initialize plot sign templates
        self.plot_sign_templates = self._initialize_plot_sign_templates()

    def _initialize_plot_sign_templates(self) -> Dict[str, Dict]:
        """Initialize templates for different types of plot signs"""
        return {
            'directional_consensus': {
                'name': 'Directional Consensus',
                'interpretation': 'Market dimensions aligned in same direction',
                'thresholds': {'magnitude': 0.8, 'coherence': 0.7}
            },
            'pattern_conformance': {
                'name': 'Pattern Conformance',
                'interpretation': 'Strong alignment with known patterns',
                'thresholds': {'magnitude': 0.7, 'coherence': 0.6}
            },
            'phase_transition': {
                'name': 'Phase Transition',
                'interpretation': 'Shift between market regimes',
                'thresholds': {'magnitude': 0.9, 'coherence': 0.8}
            },
            'strength_weakness': {
                'name': 'Strength/Weakness',
                'interpretation': 'Underlying trend health indicator',
                'thresholds': {'magnitude': 0.6, 'coherence': 0.5}
            },
            'event_localization': {
                'name': 'Event Localization',
                'interpretation': 'Significant market event detected',
                'thresholds': {'magnitude': 0.95, 'coherence': 0.9}
            }
        }

    def decompose_tensor(self, tensor: np.ndarray) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Decompose tensor using CP and Tucker methods"""
        # CP Decomposition
        cp_factors = parafac(
            tensor,
            rank=self.num_components,
            normalize_factors=True
        )
        
        # Tucker Decomposition
        tucker_factors = tucker(
            tensor,
            ranks=[self.num_components] * tensor.ndim,
            normalize_factors=True
        )
        
        return cp_factors, tucker_factors

    def _compute_quantum_resonance(self, tensor: np.ndarray) -> Dict[str, float]:
        """Compute quantum resonance metrics for tensor"""
        # Compute magnitude using FFT
        fft = fft2(tensor)
        magnitude = np.abs(fft).mean()
        
        # Compute phase using Hilbert transform
        phase = np.angle(hilbert(tensor.flatten())).mean()
        
        # Compute coherence using magnitude-squared coherence
        coherence = np.abs(fft)**2 / (np.abs(fft)**2 + 1e-8)
        coherence = np.mean(coherence)
        
        return {
            'magnitude': float(magnitude),
            'phase': float(phase),
            'coherence': float(coherence)
        }

    def extract_plot_signs(
        self,
        market_tensor: np.ndarray,
        time_dilation: float = 1.0
    ) -> List[PlotSign]:
        """Extract interpretable plot signs from market tensor"""
        # Decompose tensor
        cp_factors, tucker_factors = self.decompose_tensor(market_tensor)
        
        # Compute quantum resonance
        resonance = self._compute_quantum_resonance(market_tensor)
        
        # Update resonance history
        self.resonance_history['magnitude'].append(resonance['magnitude'])
        self.resonance_history['phase'].append(resonance['phase'])
        self.resonance_history['coherence'].append(resonance['coherence'])
        
        # Extract latent factors
        latent_factors = self._extract_latent_factors(cp_factors, tucker_factors)
        
        # Update history
        self.latent_factor_history.append(latent_factors)
        if len(self.latent_factor_history) > self.history_size:
            self.latent_factor_history.pop(0)
        
        # Generate plot signs
        plot_signs = []
        for template_name, template in self.plot_sign_templates.items():
            # Check if resonance meets template thresholds
            if (resonance['magnitude'] >= template['thresholds']['magnitude'] and
                resonance['coherence'] >= template['thresholds']['coherence']):
                
                # Create plot sign
                plot_sign = PlotSign(
                    name=template['name'],
                    magnitude=resonance['magnitude'],
                    phase=resonance['phase'],
                    coherence=resonance['coherence'],
                    interpretation=template['interpretation'],
                    confidence=self._compute_confidence(latent_factors, template_name),
                    tensor_factors={
                        'cp': cp_factors,
                        'tucker': tucker_factors
                    }
                )
                plot_signs.append(plot_sign)
        
        # Update history
        self.plot_sign_history.append(plot_signs)
        if len(self.plot_sign_history) > self.history_size:
            self.plot_sign_history.pop(0)
        
        return plot_signs

    def _extract_latent_factors(
        self,
        cp_factors: Tuple[np.ndarray, ...],
        tucker_factors: Tuple[np.ndarray, ...]
    ) -> np.ndarray:
        """Extract and combine latent factors from tensor decompositions"""
        # Combine CP and Tucker factors
        combined_factors = np.concatenate([
            np.array(cp_factors).flatten(),
            np.array(tucker_factors).flatten()
        ])
        
        # Scale factors
        if len(self.latent_factor_history) > 0:
            self.scaler.fit(np.array(self.latent_factor_history))
        scaled_factors = self.scaler.transform(combined_factors.reshape(1, -1))
        
        return scaled_factors.flatten()

    def _compute_confidence(
        self,
        latent_factors: np.ndarray,
        template_name: str
    ) -> float:
        """Compute confidence score for plot sign"""
        if len(self.latent_factor_history) < self.num_components:
            return 0.5  # Default confidence
            
        # Cluster latent factors
        self.kmeans.fit(np.array(self.latent_factor_history))
        
        # Compute distance to cluster center
        cluster_distances = self.kmeans.transform(latent_factors.reshape(1, -1))
        min_distance = np.min(cluster_distances)
        
        # Convert distance to confidence (closer = higher confidence)
        confidence = np.exp(-min_distance)
        
        return float(confidence)

    def get_plot_sign_history(self) -> List[List[PlotSign]]:
        """Get history of plot signs"""
        return self.plot_sign_history

    def get_resonance_history(self) -> Dict[str, List[float]]:
        """Get history of resonance metrics"""
        return self.resonance_history

    def get_latent_factor_history(self) -> List[np.ndarray]:
        """Get history of latent factors"""
        return self.latent_factor_history

    def analyze_plot_sign_sequence(self) -> Dict[str, float]:
        """Analyze sequence of plot signs for pattern recognition"""
        if not self.plot_sign_history:
            return {}
            
        # Count occurrences of each plot sign type
        sign_counts = {}
        for signs in self.plot_sign_history:
            for sign in signs:
                sign_counts[sign.name] = sign_counts.get(sign.name, 0) + 1
        
        # Normalize counts
        total = sum(sign_counts.values())
        return {k: v/total for k, v in sign_counts.items()}

    def detect_regime_shift(self) -> bool:
        """Detect if a regime shift is occurring based on plot signs"""
        if len(self.plot_sign_history) < 2:
            return False
            
        # Get recent plot signs
        recent_signs = self.plot_sign_history[-1]
        previous_signs = self.plot_sign_history[-2]
        
        # Check for phase transition plot sign
        has_phase_transition = any(
            sign.name == 'Phase Transition' and sign.confidence > 0.8
            for sign in recent_signs
        )
        
        # Check for significant change in resonance
        recent_resonance = self.resonance_history['magnitude'][-1]
        previous_resonance = self.resonance_history['magnitude'][-2]
        resonance_change = abs(recent_resonance - previous_resonance)
        
        return has_phase_transition and resonance_change > 0.3

# Example usage:
"""
from core.plot_sign_engine import PlotSignEngine

# Initialize engine
engine = PlotSignEngine(num_components=3)

# Create sample market tensor (e.g., from DLTWaveformEngine)
market_tensor = np.random.rand(10, 10, 10)

# Extract plot signs
plot_signs = engine.extract_plot_signs(market_tensor)

# Analyze sequence
sign_analysis = engine.analyze_plot_sign_sequence()

# Check for regime shift
is_regime_shift = engine.detect_regime_shift()
""" 