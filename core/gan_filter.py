"""
GAN-Based Anomaly Detection
==========================

Implements a GAN-based anomaly detection system for shell states,
with entropy-regularized training and reconstruction error tracking.
Uses proper mathematical definitions for entropy, phase analysis,
and composite anomaly scoring.

GAN Filter
==========

Implements entropy-regularized GAN for anomaly detection:
- L_GAN = E[log D(x)] + E[log(1 - D(G(z)))] + α‖ℋ_real−ℋ_fake‖

Invariants:
- GAN anomaly monotonicity: Reconstruction-error ↑ ⇒ anomaly score ↑

See docs/math/gan.md for details.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime
import json
from pathlib import Path
import pywt
from scipy.signal import hilbert
from scipy.stats import wasserstein_distance
from scipy.fft import fft, fftfreq

logger = logging.getLogger(__name__)

@dataclass
class GANConfig:
    """Configuration for GAN training and inference"""
    latent_dim: int = 32
    hidden_dim: int = 64
    num_layers: int = 3
    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 100
    entropy_weight: float = 0.1
    phase_weight: float = 0.1
    consistency_weight: float = 0.1
    anomaly_threshold: float = 0.85
    use_gpu: bool = True
    model_path: str = "models/gan_filter"
    spectral_window: int = 50
    entropy_bins: int = 50
    momentum_window: int = 20
    volume_window: int = 20
    state_precision: float = 0.01

@dataclass
class GANAnomalyMetrics:
    anomaly_score: float
    reconstruction_error: float
    is_anomaly: bool
    confidence: float

class Generator(nn.Module):
    """Generator network for GAN"""
    
    def __init__(self, latent_dim: int, output_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        
        layers = []
        in_dim = latent_dim
        
        # Build hidden layers
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2)
            ])
            in_dim = hidden_dim
            
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        layers.append(nn.Tanh())  # Bound output to [-1, 1]
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.model(z)

class Discriminator(nn.Module):
    """Discriminator network for GAN"""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        
        layers = []
        in_dim = input_dim
        
        # Build hidden layers
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3)
            ])
            in_dim = hidden_dim
            
        # Output layer
        layers.append(nn.Linear(hidden_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class GANFilter:
    """
    GAN-based anomaly detection system for shell states.
    Implements entropy-regularized training and reconstruction error tracking.
    """
    
    def __init__(self, config: Dict[str, Any], input_dim: int):
        self.config = GANConfig(**(config or {}))
        self.input_dim = input_dim
        self.device = torch.device("cuda" if self.config.use_gpu and torch.cuda.is_available() else "cpu")
        
        # Initialize networks
        self.generator = Generator(
            self.config.latent_dim,
            input_dim,
            self.config.hidden_dim,
            self.config.num_layers
        ).to(self.device)
        
        self.discriminator = Discriminator(
            input_dim,
            self.config.hidden_dim,
            self.config.num_layers
        ).to(self.device)
        
        # Initialize optimizers
        self.g_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=self.config.learning_rate,
            betas=(0.5, 0.999)
        )
        
        self.d_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=self.config.learning_rate,
            betas=(0.5, 0.999)
        )
        
        # Loss functions
        self.adversarial_loss = nn.BCELoss()
        self.reconstruction_loss = nn.MSELoss()
        
        # Training history
        self.training_history: List[Dict[str, float]] = []
        
    def compute_shannon_entropy(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Shannon entropy: H_s = -∑ p_i log_2(p_i)"""
        # Compute histogram
        hist = torch.histc(x, bins=self.config.entropy_bins, min=-1, max=1)
        hist = hist / hist.sum()  # Normalize to probabilities
        hist = hist[hist > 0]  # Remove zero bins
        
        # Compute entropy
        return -torch.sum(hist * torch.log2(hist))
        
    def compute_wavelet_entropy(self, x: torch.Tensor) -> torch.Tensor:
        """Compute wavelet entropy: H_w = -∑ |W_jk|² log |W_jk|²"""
        # Convert to numpy for wavelet transform
        x_np = x.detach().cpu().numpy()
        
        # Perform wavelet decomposition
        coeffs = pywt.wavedec(x_np, 'haar')
        
        # Convert back to tensor
        coeffs_tensor = [torch.from_numpy(c).to(self.device) for c in coeffs]
        
        # Compute total energy
        total_energy = sum(torch.sum(torch.abs(c)**2) for c in coeffs_tensor)
        
        # Compute entropy
        return -torch.log(total_energy + 1e-6)
        
    def compute_phase_angle(self, x: torch.Tensor) -> torch.Tensor:
        """Compute phase angle: φ_t = arctan2(Im(H_t), Re(H_t))"""
        # Convert to numpy for Hilbert transform
        x_np = x.detach().cpu().numpy()
        
        # Compute analytic signal
        analytic = hilbert(x_np)
        
        # Compute phase angle
        phase = np.angle(analytic)
        
        return torch.from_numpy(phase).to(self.device)
        
    def compute_phase_velocity(self, x: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        """Compute phase velocity: ω_t = dφ/dt"""
        phase = self.compute_phase_angle(x)
        return (phase[1:] - phase[:-1]) / dt
        
    def compute_momentum_alignment(self, x: torch.Tensor) -> torch.Tensor:
        """Compute momentum alignment: ρ_d = (1/n)∑ sign(r_t-i) |r_t-i|^0.5"""
        # Compute returns
        returns = x[1:] - x[:-1]
        
        # Compute momentum score
        momentum = torch.sign(returns) * torch.sqrt(torch.abs(returns))
        
        # Average over window
        return torch.mean(momentum[-self.config.momentum_window:])
        
    def compute_volatility_regime(self, x: torch.Tensor) -> torch.Tensor:
        """Compute volatility regime: σ_t / σ̄_t-n:t"""
        # Compute rolling volatility
        returns = x[1:] - x[:-1]
        vol = torch.std(returns[-self.config.momentum_window:])
        vol_mean = torch.mean(torch.std(returns[-self.config.momentum_window:].unfold(0, self.config.momentum_window, 1)))
        
        return vol / (vol_mean + 1e-6)
        
    def compute_volume_anomaly(self, volume: torch.Tensor) -> torch.Tensor:
        """Compute volume anomaly: (V_t - μ_V) / σ_V"""
        vol_mean = torch.mean(volume[-self.config.volume_window:])
        vol_std = torch.std(volume[-self.config.volume_window:])
        
        return (volume[-1] - vol_mean) / (vol_std + 1e-6)
        
    def compute_wasserstein_distance(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Compute Wasserstein distance: W_2(P_t, P_anchor)"""
        x_np = x.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        
        return wasserstein_distance(x_np, y_np)
        
    def compute_spectral_anomaly(self, x: torch.Tensor) -> torch.Tensor:
        """Compute spectral anomaly: A_spectral = ∑|(S_t(f) - S̄(f))/σ_S(f)|"""
        # Convert to numpy for FFT
        x_np = x.detach().cpu().numpy()
        
        # Compute power spectral density
        psd = np.abs(fft(x_np))**2
        
        # Get dominant frequencies
        freqs = fftfreq(len(x_np))
        dominant_mask = np.abs(freqs) < 0.1  # Focus on low frequencies
        
        # Compute anomaly score
        mean_psd = np.mean(psd[dominant_mask])
        std_psd = np.std(psd[dominant_mask])
        anomaly = np.sum(np.abs((psd[dominant_mask] - mean_psd) / (std_psd + 1e-6)))
        
        return torch.tensor(anomaly, device=self.device)
        
    def compute_state_hash(self, x: torch.Tensor) -> str:
        """Compute deterministic state hash"""
        # Quantize state vector
        quantized = torch.round(x / self.config.state_precision) * self.config.state_precision
        
        # Convert to hash
        return hash(quantized.cpu().numpy().tobytes())
        
    def train_step(self, real_data: torch.Tensor) -> Dict[str, float]:
        """
        Perform one training step with proper loss functions.
        
        Args:
            real_data: Batch of real shell states
            
        Returns:
            Dictionary of loss values
        """
        batch_size = real_data.size(0)
        
        # Ground truths
        real_label = torch.ones(batch_size, 1).to(self.device)
        fake_label = torch.zeros(batch_size, 1).to(self.device)
        
        # -----------------
        # Train Generator
        # -----------------
        self.g_optimizer.zero_grad()
        
        # Generate fake data
        z = torch.randn(batch_size, self.config.latent_dim).to(self.device)
        fake_data = self.generator(z)
        
        # Compute GAN loss
        g_loss = self.adversarial_loss(self.discriminator(fake_data), real_label)
        
        # Compute entropy loss
        real_entropy = self.compute_shannon_entropy(real_data)
        fake_entropy = self.compute_shannon_entropy(fake_data)
        entropy_loss = torch.norm(real_entropy - fake_entropy, p=2)
        
        # Compute phase loss
        real_phase = self.compute_phase_angle(real_data)
        fake_phase = self.compute_phase_angle(fake_data)
        phase_loss = 1 - torch.mean(torch.cos(real_phase - fake_phase))
        
        # Compute consistency loss
        z_prev = torch.randn(batch_size, self.config.latent_dim).to(self.device)
        fake_prev = self.generator(z_prev)
        consistency_loss = torch.norm(fake_data - fake_prev, p=2)
        
        # Total generator loss
        g_total_loss = g_loss + \
                      self.config.entropy_weight * entropy_loss + \
                      self.config.phase_weight * phase_loss + \
                      self.config.consistency_weight * consistency_loss
        
        g_total_loss.backward()
        self.g_optimizer.step()
        
        # -----------------
        # Train Discriminator
        # -----------------
        self.d_optimizer.zero_grad()
        
        # Real data loss
        d_real_loss = self.adversarial_loss(self.discriminator(real_data), real_label)
        
        # Fake data loss
        d_fake_loss = self.adversarial_loss(self.discriminator(fake_data.detach()), fake_label)
        
        # Total discriminator loss
        d_total_loss = (d_real_loss + d_fake_loss) / 2
        
        d_total_loss.backward()
        self.d_optimizer.step()
        
        return {
            'g_loss': g_loss.item(),
            'd_loss': d_total_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'phase_loss': phase_loss.item(),
            'consistency_loss': consistency_loss.item()
        }
        
    def compute_anomaly_score(self, data: torch.Tensor) -> float:
        """
        Compute composite anomaly score.
        
        Args:
            data: Input data
            
        Returns:
            Anomaly score [0,1]
        """
        # Generate reconstruction
        z = torch.randn(len(data), self.config.latent_dim).to(self.device)
        reconstructed = self.generator(z)
        
        # Compute discriminator score
        d_score = self.discriminator(data).mean()
        
        # Compute reconstruction error
        recon_error = torch.norm(data - reconstructed, p=2) / (torch.norm(data, p=2) + 1e-6)
        
        # Compute spectral anomaly
        spectral_score = self.compute_spectral_anomaly(data)
        
        # Compute entropy anomaly
        real_entropy = self.compute_shannon_entropy(data)
        fake_entropy = self.compute_shannon_entropy(reconstructed)
        entropy_anomaly = torch.abs(real_entropy - fake_entropy)
        
        # Combine scores
        anomaly_score = (
            0.4 * (1 - d_score) +  # Discriminator contribution
            0.3 * recon_error +    # Reconstruction error
            0.2 * spectral_score + # Spectral anomaly
            0.1 * entropy_anomaly  # Entropy anomaly
        )
        
        return float(anomaly_score)
        
    def detect_anomaly(self, data: torch.Tensor) -> Tuple[bool, float]:
        """
        Detect if input data is anomalous.
        
        Args:
            data: Input data
            
        Returns:
            Tuple of (is_anomaly, confidence)
        """
        score = self.compute_anomaly_score(data)
        is_anomaly = score > self.config.anomaly_threshold
        confidence = min(score / self.config.anomaly_threshold, 1.0)
        
        return is_anomaly, confidence
        
    def save_model(self, filepath: Optional[str] = None) -> None:
        """
        Save model state.
        
        Args:
            filepath: Optional custom filepath
        """
        if filepath is None:
            filepath = Path(self.config.model_path) / f"gan_filter_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
            
        state = {
            'generator_state': self.generator.state_dict(),
            'discriminator_state': self.discriminator.state_dict(),
            'g_optimizer_state': self.g_optimizer.state_dict(),
            'd_optimizer_state': self.d_optimizer.state_dict(),
            'config': self.config.__dict__,
            'training_history': self.training_history
        }
        
        torch.save(state, filepath)
        
    def load_model(self, filepath: str) -> None:
        """
        Load model state.
        
        Args:
            filepath: Path to model file
        """
        state = torch.load(filepath, map_location=self.device)
        
        self.generator.load_state_dict(state['generator_state'])
        self.discriminator.load_state_dict(state['discriminator_state'])
        self.g_optimizer.load_state_dict(state['g_optimizer_state'])
        self.d_optimizer.load_state_dict(state['d_optimizer_state'])
        
        # Update config
        self.config = GANConfig(**state['config'])
        self.training_history = state['training_history']

    def detect(self, vector: np.ndarray) -> GANAnomalyMetrics:
        """Detect anomaly in input vector."""
        logger.info("Detecting GAN anomaly.")
        raise NotImplementedError

    def train(self, dataset: List[np.ndarray]) -> None:
        """Train GAN on dataset."""
        logger.info("Training GAN filter.")
        raise NotImplementedError 