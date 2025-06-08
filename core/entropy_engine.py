"""
Unified Entropy Engine
=====================

Provides a unified interface for entropy calculations:
- Shannon: H_s = -Σ p_i log(p_i)
- Wavelet: H_w = -Σ |W_{j,k}|² log|W_{j,k}|²
- Tsallis: H_q = (1-Σ p_i^q)/(q-1)

Invariants:
- Entropy non-negativity: ℋ(x; θ) ≥ 0 for every θ.
- Wavelet-Shannon ordering: H_wavelet(x) ≥ H_shannon(x) for mean-zero stationary x.

See docs/math/entropy.md for details.
"""

import numpy as np
import pywt
from dataclasses import dataclass
from typing import Optional, Dict, Any, Union
import logging
import math

logger = logging.getLogger(__name__)

@dataclass
class EntropyConfig:
    """Configuration for entropy calculations"""
    method: str = "wavelet"  # wavelet, shannon, tsallis
    q_param: float = 2.0  # For Tsallis entropy
    wavelet_type: str = "haar"
    num_bins: int = 50  # For histogram-based methods
    clip_range: tuple = (0.0, 5.0)  # Range for entropy values
    use_gpu: bool = True  # Whether to use GPU acceleration

class UnifiedEntropyEngine:
    """
    Unified entropy calculation engine with GPU acceleration support.
    Implements Wavelet, Shannon, and Tsallis entropy methods.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = EntropyConfig(**(config or {}))
        self._setup_gpu()
        
    def _setup_gpu(self):
        """Initialize GPU resources if available"""
        try:
            import cupy as cp
            self.gpu_available = True
            self.cp = cp
        except ImportError:
            self.gpu_available = False
            logger.warning("GPU acceleration not available. Using CPU fallback.")
            
    def compute_entropy(self, x: np.ndarray, method: str = 'shannon', q: Optional[float] = None) -> float:
        """
        Compute entropy of vector x using the specified method.
        
        Parameters
        ----------
        x : np.ndarray
            Input vector.
        method : str
            'shannon', 'wavelet', or 'tsallis'.
        q : float, optional
            Tsallis entropy parameter.
        
        Returns
        -------
        float
            Entropy value.
        """
        logger.info(f"Computing entropy: method={method}, q={q}")
        raise NotImplementedError("Implement entropy calculation for all methods.")
            
    def _compute_entropy_gpu(self, x: np.ndarray, method: str, q_param: float) -> float:
        """GPU-accelerated entropy computation"""
        try:
            x_gpu = self.cp.asarray(x)
            
            if method == "wavelet":
                return self._wavelet_entropy_gpu(x_gpu)
            elif method == "shannon":
                return self._shannon_entropy_gpu(x_gpu)
            elif method == "tsallis":
                return self._tsallis_entropy_gpu(x_gpu, q_param)
            else:
                raise ValueError(f"Unknown entropy method: {method}")
                
        except Exception as e:
            logger.error(f"GPU entropy computation failed: {str(e)}")
            return self._compute_entropy_cpu(x, method, q_param)
            
    def _compute_entropy_cpu(self, x: np.ndarray, method: str, q_param: float) -> float:
        """CPU-based entropy computation"""
        if method == "wavelet":
            return self._wavelet_entropy_cpu(x)
        elif method == "shannon":
            return self._shannon_entropy_cpu(x)
        elif method == "tsallis":
            return self._tsallis_entropy_cpu(x, q_param)
        else:
            raise ValueError(f"Unknown entropy method: {method}")
            
    def _wavelet_entropy_gpu(self, x: np.ndarray) -> float:
        """GPU-accelerated wavelet entropy"""
        # Convert to GPU array if not already
        if not isinstance(x, self.cp.ndarray):
            x = self.cp.asarray(x)
            
        # Perform wavelet decomposition
        coeffs = pywt.wavedec(x.get(), self.config.wavelet_type)
        coeffs_gpu = [self.cp.asarray(c) for c in coeffs]
        
        # Compute total energy
        total_energy = self.cp.sum([self.cp.sum(self.cp.abs(c)**2) for c in coeffs_gpu])
        
        # Compute entropy
        entropy = -self.cp.log(total_energy + 1e-6)
        
        # Clip to range
        return float(self.cp.clip(entropy, *self.config.clip_range))
        
    def _shannon_entropy_gpu(self, x: np.ndarray) -> float:
        """GPU-accelerated Shannon entropy"""
        if not isinstance(x, self.cp.ndarray):
            x = self.cp.asarray(x)
            
        # Compute histogram
        hist = self.cp.histogram(x, bins=self.config.num_bins, density=True)[0]
        hist = hist[hist > 0]  # Remove zero bins
        
        # Compute entropy
        entropy = -self.cp.sum(hist * self.cp.log(hist))
        
        return float(self.cp.clip(entropy, *self.config.clip_range))
        
    def _tsallis_entropy_gpu(self, x: np.ndarray, q: float) -> float:
        """GPU-accelerated Tsallis entropy"""
        if not isinstance(x, self.cp.ndarray):
            x = self.cp.asarray(x)
            
        # Compute histogram
        hist = self.cp.histogram(x, bins=self.config.num_bins, density=True)[0]
        hist = hist[hist > 0]  # Remove zero bins
        
        # Compute Tsallis entropy
        entropy = (1 - self.cp.sum(hist**q)) / (q - 1)
        
        return float(self.cp.clip(entropy, *self.config.clip_range))
        
    def _wavelet_entropy_cpu(self, x: np.ndarray) -> float:
        """CPU-based wavelet entropy"""
        # Perform wavelet decomposition
        coeffs = pywt.wavedec(x, self.config.wavelet_type)
        
        # Compute total energy
        total_energy = sum(np.sum(np.abs(coeff)**2) for coeff in coeffs)
        
        # Compute entropy
        entropy = -np.log(total_energy + 1e-6)
        
        return float(np.clip(entropy, *self.config.clip_range))
        
    def _shannon_entropy_cpu(self, x: np.ndarray) -> float:
        """CPU-based Shannon entropy"""
        # Compute histogram
        hist, _ = np.histogram(x, bins=self.config.num_bins, density=True)
        hist = hist[hist > 0]  # Remove zero bins
        
        # Compute entropy
        entropy = -np.sum(hist * np.log(hist))
        
        return float(np.clip(entropy, *self.config.clip_range))
        
    def _tsallis_entropy_cpu(self, x: np.ndarray, q: float) -> float:
        """CPU-based Tsallis entropy"""
        # Compute histogram
        hist, _ = np.histogram(x, bins=self.config.num_bins, density=True)
        hist = hist[hist > 0]  # Remove zero bins
        
        # Compute Tsallis entropy
        entropy = (1 - np.sum(hist**q)) / (q - 1)
        
        return float(np.clip(entropy, *self.config.clip_range))
        
    def compute_entropy_gradient(self, x: np.ndarray, method: Optional[str] = None) -> np.ndarray:
        """
        Compute gradient of entropy with respect to input vector.
        
        Args:
            x: Input vector
            method: Entropy calculation method
            
        Returns:
            Gradient vector
        """
        method = method or self.config.method
        
        if method == "wavelet":
            return self._wavelet_entropy_gradient(x)
        elif method == "shannon":
            return self._shannon_entropy_gradient(x)
        elif method == "tsallis":
            return self._tsallis_entropy_gradient(x)
        else:
            raise ValueError(f"Unknown entropy method: {method}")
            
    def _wavelet_entropy_gradient(self, x: np.ndarray) -> np.ndarray:
        """Compute gradient of wavelet entropy"""
        # Perform wavelet decomposition
        coeffs = pywt.wavedec(x, self.config.wavelet_type)
        
        # Compute total energy
        total_energy = sum(np.sum(np.abs(coeff)**2) for coeff in coeffs)
        
        # Compute gradient
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_perturbed = x.copy()
            x_perturbed[i] += 1e-6
            coeffs_perturbed = pywt.wavedec(x_perturbed, self.config.wavelet_type)
            energy_perturbed = sum(np.sum(np.abs(coeff)**2) for coeff in coeffs_perturbed)
            grad[i] = -(energy_perturbed - total_energy) / (1e-6 * total_energy)
            
        return grad
        
    def _shannon_entropy_gradient(self, x: np.ndarray) -> np.ndarray:
        """Compute gradient of Shannon entropy"""
        # Compute histogram
        hist, bin_edges = np.histogram(x, bins=self.config.num_bins, density=True)
        hist = hist[hist > 0]  # Remove zero bins
        
        # Compute gradient
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_perturbed = x.copy()
            x_perturbed[i] += 1e-6
            hist_perturbed, _ = np.histogram(x_perturbed, bins=bin_edges, density=True)
            hist_perturbed = hist_perturbed[hist_perturbed > 0]
            grad[i] = -(np.sum(hist_perturbed * np.log(hist_perturbed)) - 
                       np.sum(hist * np.log(hist))) / 1e-6
                       
        return grad
        
    def _tsallis_entropy_gradient(self, x: np.ndarray) -> np.ndarray:
        """Compute gradient of Tsallis entropy"""
        q = self.config.q_param
        
        # Compute histogram
        hist, bin_edges = np.histogram(x, bins=self.config.num_bins, density=True)
        hist = hist[hist > 0]  # Remove zero bins
        
        # Compute gradient
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_perturbed = x.copy()
            x_perturbed[i] += 1e-6
            hist_perturbed, _ = np.histogram(x_perturbed, bins=bin_edges, density=True)
            hist_perturbed = hist_perturbed[hist_perturbed > 0]
            grad[i] = -((1 - np.sum(hist_perturbed**q)) - 
                       (1 - np.sum(hist**q))) / (1e-6 * (q - 1))
                       
        return grad 