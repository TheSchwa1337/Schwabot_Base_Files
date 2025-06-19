#!/usr/bin/env python3
"""
Entropy GAN Filter - Advanced Signal Filtering using Generative Adversarial Networks
===================================================================================

Comprehensive GAN-based filtering system for entropy signal validation and anomaly detection.
Implements the mathematical framework provided for entropy generation and discrimination.

Key Features:
- EntropyGenerator: Neural network for synthetic entropy signal generation
- EntropyDiscriminator: Neural network for real vs synthetic signal discrimination
- GAN training with BCE loss and optional Wasserstein loss with gradient penalty
- Batch filtering with confidence thresholding
- Real-time signal validation and anomaly detection
- Integration with mathematical optimization bridge
- Windows CLI compatibility with emoji fallbacks

Mathematical Foundations:
- Generator: G(z) = σ(W₂ · ReLU(W₁z + b₁) + b₂)
- Discriminator: D(x) = σ(W₄ · LeakyReLU(W₃x + b₃) + b₄)
- BCE Loss: L_D = -[log D(x) + log(1 - D(G(z)))]
- Wasserstein Loss: L_D = D(x) - D(G(z))
- Gradient Penalty: L_GP = λ·(||∇_x̂ D(x̂)||₂ - 1)²
- Entropy Calibration: ΔH = H(x) - H(G(z))

Integration Points:
- mathematical_optimization_bridge.py: Performance optimization
- tick_processor.py: Real-time signal processing
- constraints.py: Signal validation constraints
- mathlib_v3.py: Dual-number automatic differentiation
- visualization.py: Signal visualization and analysis

Windows CLI compatible with flake8 compliance.
"""

from __future__ import annotations

import logging
import time
import threading
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union, Callable
from enum import Enum
import warnings

import numpy as np
import numpy.typing as npt

# PyTorch imports with fallback handling
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.autograd import grad
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Create mock classes for type hints
    class nn:
        class Module:
            pass
    torch = None

# Import Windows CLI compatibility handler
try:
    from core.enhanced_windows_cli_compatibility import (
        EnhancedWindowsCliCompatibilityHandler as CLIHandler,
        safe_print, safe_log
    )
    CLI_COMPATIBILITY_AVAILABLE = True
except ImportError:
    CLI_COMPATIBILITY_AVAILABLE = False
    # Fallback CLI handler
    class CLIHandler:
        @staticmethod
        def safe_emoji_print(message: str, force_ascii: bool = False) -> str:
            emoji_mapping = {
                '✅': '[SUCCESS]', '❌': '[ERROR]', '⚠️': '[WARNING]', '🚨': '[ALERT]',
                '🎉': '[COMPLETE]', '🔄': '[PROCESSING]', '⏳': '[WAITING]', '⭐': '[STAR]',
                '🚀': '[LAUNCH]', '🔧': '[TOOLS]', '🛠️': '[REPAIR]', '⚡': '[FAST]',
                '🔍': '[SEARCH]', '🎯': '[TARGET]', '🔥': '[HOT]', '❄️': '[COOL]',
                '📊': '[DATA]', '📈': '[PROFIT]', '📉': '[LOSS]', '💰': '[MONEY]',
                '🧪': '[TEST]', '⚖️': '[BALANCE]', '🌡️': '[TEMP]', '🔬': '[ANALYZE]',
                '🧮': '[CALC]', '📐': '[MATH]', '🔢': '[NUMBERS]', '∞': '[INFINITY]'
            }
            if force_ascii:
                for emoji, replacement in emoji_mapping.items():
                    message = message.replace(emoji, replacement)
            return message

if TYPE_CHECKING:
    from typing_extensions import Self

# Type definitions
Vector = npt.NDArray[np.float64]
Matrix = npt.NDArray[np.float64]
TensorType = Union[torch.Tensor, Vector] if TORCH_AVAILABLE else Vector

logger = logging.getLogger(__name__)


class GANMode(Enum):
    """GAN training mode enumeration"""
    VANILLA = "vanilla"  # Standard GAN with BCE loss
    WASSERSTEIN = "wasserstein"  # Wasserstein GAN
    WASSERSTEIN_GP = "wasserstein_gp"  # Wasserstein GAN with Gradient Penalty


class FilterMode(Enum):
    """Signal filtering mode enumeration"""
    THRESHOLD = "threshold"  # Simple threshold filtering
    CONFIDENCE = "confidence"  # Confidence-based filtering
    ENTROPY_AWARE = "entropy_aware"  # Entropy-aware filtering
    ADAPTIVE = "adaptive"  # Adaptive threshold filtering


@dataclass
class GANConfig:
    """GAN configuration container"""
    
    noise_dim: int = 100
    signal_dim: int = 64
    generator_hidden: int = 128
    discriminator_hidden: int = 128
    learning_rate: float = 1e-4
    beta1: float = 0.5
    beta2: float = 0.999
    batch_size: int = 64
    epochs: int = 1000
    mode: GANMode = GANMode.VANILLA
    gradient_penalty_lambda: float = 10.0
    entropy_weight: float = 0.1
    clip_value: float = 0.01  # For WGAN weight clipping


@dataclass
class FilterConfig:
    """Filter configuration container"""
    
    threshold: float = 0.5
    mode: FilterMode = FilterMode.THRESHOLD
    confidence_percentile: float = 95.0
    entropy_tolerance: float = 0.1
    adaptive_window: int = 100
    min_samples: int = 10


@dataclass
class TrainingMetrics:
    """Training metrics container"""
    
    epoch: int = 0
    generator_loss: float = 0.0
    discriminator_loss: float = 0.0
    gradient_penalty: float = 0.0
    entropy_difference: float = 0.0
    real_accuracy: float = 0.0
    fake_accuracy: float = 0.0
    training_time: float = 0.0
    total_time: float = 0.0


class EntropyGenerator(nn.Module if TORCH_AVAILABLE else object):
    """
    Entropy Generator Neural Network
    
    Generates synthetic entropy signals from random noise using the mathematical
    framework: G(z) = σ(W₂ · ReLU(W₁z + b₁) + b₂)
    
    Architecture:
    - Input: Random noise vector z ∈ ℝⁿ
    - Hidden: ReLU activation with configurable dimensions
    - Output: Synthetic signal with tanh activation for bounded output
    """
    
    def __init__(self, noise_dim: int, output_dim: int, hidden_dim: int = 128) -> None:
        """
        Initialize entropy generator
        
        Args:
            noise_dim: Dimension of input noise vector
            output_dim: Dimension of output signal
            hidden_dim: Hidden layer dimension
        """
        if TORCH_AVAILABLE:
            super().__init__()
            
            self.noise_dim = noise_dim
            self.output_dim = output_dim
            self.hidden_dim = hidden_dim
            
            # Generator network: z -> hidden -> output
            self.model = nn.Sequential(
                nn.Linear(noise_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, output_dim),
                nn.Tanh()  # Bounded output [-1, 1]
            )
            
            # Initialize weights
            self._initialize_weights()
        else:
            raise ImportError("PyTorch not available - cannot create EntropyGenerator")
    
    def _initialize_weights(self) -> None:
        """Initialize network weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through generator
        
        Args:
            z: Input noise tensor [batch_size, noise_dim]
            
        Returns:
            Generated signal tensor [batch_size, output_dim]
        """
        return self.model(z)
    
    def generate_batch(self, batch_size: int, device: Optional[str] = None) -> torch.Tensor:
        """
        Generate a batch of synthetic signals
        
        Args:
            batch_size: Number of signals to generate
            device: Device to generate on (cpu/cuda)
            
        Returns:
            Generated signals tensor
        """
        if device is None:
            device = next(self.parameters()).device
        
        noise = torch.randn(batch_size, self.noise_dim, device=device)
        return self.forward(noise)


class EntropyDiscriminator(nn.Module if TORCH_AVAILABLE else object):
    """
    Entropy Discriminator Neural Network
    
    Discriminates between real and synthetic entropy signals using the mathematical
    framework: D(x) = σ(W₄ · LeakyReLU(W₃x + b₃) + b₄)
    
    Architecture:
    - Input: Signal vector x ∈ ℝᵐ
    - Hidden: LeakyReLU activation for better gradient flow
    - Output: Probability score [0, 1] for real vs fake classification
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
        """
        Initialize entropy discriminator
        
        Args:
            input_dim: Dimension of input signal
            hidden_dim: Hidden layer dimension
        """
        if TORCH_AVAILABLE:
            super().__init__()
            
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            
            # Discriminator network: x -> hidden -> probability
            self.model = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()  # Probability output [0, 1]
            )
            
            # Initialize weights
            self._initialize_weights()
        else:
            raise ImportError("PyTorch not available - cannot create EntropyDiscriminator")
    
    def _initialize_weights(self) -> None:
        """Initialize network weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through discriminator
        
        Args:
            x: Input signal tensor [batch_size, input_dim]
            
        Returns:
            Probability tensor [batch_size, 1]
        """
        return self.model(x)


class EntropyGAN:
    """
    Comprehensive Entropy GAN System
    
    Implements the complete GAN framework for entropy signal generation and
    discrimination with multiple training modes and loss functions.
    """
    
    def __init__(self, config: GANConfig) -> None:
        """
        Initialize Entropy GAN system
        
        Args:
            config: GAN configuration
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available - cannot create EntropyGAN")
        
        self.config = config
        self.cli_handler = CLIHandler()
        
        # Initialize networks
        self.generator = EntropyGenerator(
            config.noise_dim, 
            config.signal_dim, 
            config.generator_hidden
        )
        self.discriminator = EntropyDiscriminator(
            config.signal_dim, 
            config.discriminator_hidden
        )
        
        # Initialize optimizers
        self.optimizer_g = optim.Adam(
            self.generator.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2)
        )
        self.optimizer_d = optim.Adam(
            self.discriminator.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2)
        )
        
        # Loss functions
        self.bce_loss = nn.BCELoss()
        
        # Training state
        self.training_metrics: List[TrainingMetrics] = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move networks to device
        self.generator.to(self.device)
        self.discriminator.to(self.device)
        
        # Threading for training
        self.training_lock = threading.Lock()
        self.is_training = False
        
        logger.info(f"EntropyGAN initialized with {config.mode.value} mode")
    
    def safe_print(self, message: str, force_ascii: bool = False) -> None:
        """Safe print with CLI compatibility"""
        if CLI_COMPATIBILITY_AVAILABLE:
            safe_print(message, force_ascii=force_ascii)
        else:
            safe_message = self.cli_handler.safe_emoji_print(message, force_ascii=force_ascii)
            print(safe_message)
    
    def safe_log(self, level: str, message: str, context: str = "") -> bool:
        """Safe logging with CLI compatibility"""
        if CLI_COMPATIBILITY_AVAILABLE:
            return safe_log(logger, level, message, context)
        else:
            try:
                log_func = getattr(logger, level.lower(), logger.info)
                log_func(message)
                return True
            except Exception:
                return False
    
    def compute_entropy(self, signal: torch.Tensor) -> float:
        """
        Compute Shannon entropy of signal
        
        H(x) = -Σ p_i log₂(p_i)
        
        Args:
            signal: Input signal tensor
            
        Returns:
            Shannon entropy value
        """
        try:
            # Convert to probability distribution
            signal_np = signal.detach().cpu().numpy().flatten()
            
            # Create histogram
            hist, _ = np.histogram(signal_np, bins=50, density=True)
            hist = hist + 1e-10  # Add small epsilon to avoid log(0)
            
            # Normalize to get probabilities
            prob = hist / np.sum(hist)
            
            # Compute entropy
            entropy = -np.sum(prob * np.log2(prob + 1e-10))
            
            return float(entropy)
            
        except Exception as e:
            self.safe_log('error', f"Error computing entropy: {e}")
            return 0.0
    
    def gradient_penalty(self, real_data: torch.Tensor, fake_data: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient penalty for WGAN-GP
        
        L_GP = λ·(||∇_x̂ D(x̂)||₂ - 1)²
        where x̂ = εx + (1-ε)G(z), ε ~ U[0,1]
        
        Args:
            real_data: Real data batch
            fake_data: Generated data batch
            
        Returns:
            Gradient penalty loss
        """
        try:
            batch_size = real_data.size(0)
            
            # Random interpolation factor
            epsilon = torch.rand(batch_size, 1, device=self.device)
            epsilon = epsilon.expand_as(real_data)
            
            # Interpolated samples
            interpolated = epsilon * real_data + (1 - epsilon) * fake_data
            interpolated.requires_grad_(True)
            
            # Discriminator output for interpolated samples
            d_interpolated = self.discriminator(interpolated)
            
            # Compute gradients
            gradients = grad(
                outputs=d_interpolated,
                inputs=interpolated,
                grad_outputs=torch.ones_like(d_interpolated),
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]
            
            # Gradient penalty
            gradient_norm = gradients.view(batch_size, -1).norm(2, dim=1)
            penalty = ((gradient_norm - 1) ** 2).mean()
            
            return penalty
            
        except Exception as e:
            self.safe_log('error', f"Error computing gradient penalty: {e}")
            return torch.tensor(0.0, device=self.device)
    
    def train_step(self, real_data: torch.Tensor) -> TrainingMetrics:
        """
        Single training step
        
        Args:
            real_data: Batch of real data
            
        Returns:
            Training metrics for this step
        """
        try:
            batch_size = real_data.size(0)
            metrics = TrainingMetrics()
            step_start_time = time.time()
            
            # Generate fake data
            noise = torch.randn(batch_size, self.config.noise_dim, device=self.device)
            fake_data = self.generator(noise)
            
            # Train Discriminator
            self.optimizer_d.zero_grad()
            
            if self.config.mode == GANMode.VANILLA:
                # Standard GAN with BCE loss
                real_labels = torch.ones(batch_size, 1, device=self.device)
                fake_labels = torch.zeros(batch_size, 1, device=self.device)
                
                # Real data loss
                d_real = self.discriminator(real_data)
                d_loss_real = self.bce_loss(d_real, real_labels)
                
                # Fake data loss
                d_fake = self.discriminator(fake_data.detach())
                d_loss_fake = self.bce_loss(d_fake, fake_labels)
                
                # Total discriminator loss
                d_loss = d_loss_real + d_loss_fake
                
                # Accuracy metrics
                metrics.real_accuracy = (d_real > 0.5).float().mean().item()
                metrics.fake_accuracy = (d_fake < 0.5).float().mean().item()
                
            elif self.config.mode in [GANMode.WASSERSTEIN, GANMode.WASSERSTEIN_GP]:
                # Wasserstein GAN loss
                d_real = self.discriminator(real_data)
                d_fake = self.discriminator(fake_data.detach())
                
                d_loss = d_fake.mean() - d_real.mean()
                
                # Gradient penalty for WGAN-GP
                if self.config.mode == GANMode.WASSERSTEIN_GP:
                    gp = self.gradient_penalty(real_data, fake_data)
                    d_loss += self.config.gradient_penalty_lambda * gp
                    metrics.gradient_penalty = gp.item()
                
                # Accuracy metrics (adapted for Wasserstein)
                metrics.real_accuracy = (d_real > 0).float().mean().item()
                metrics.fake_accuracy = (d_fake < 0).float().mean().item()
            
            d_loss.backward()
            self.optimizer_d.step()
            
            # Weight clipping for standard WGAN
            if self.config.mode == GANMode.WASSERSTEIN:
                for param in self.discriminator.parameters():
                    param.data.clamp_(-self.config.clip_value, self.config.clip_value)
            
            metrics.discriminator_loss = d_loss.item()
            
            # Train Generator
            self.optimizer_g.zero_grad()
            
            # Generate new fake data for generator training
            noise = torch.randn(batch_size, self.config.noise_dim, device=self.device)
            fake_data = self.generator(noise)
            
            if self.config.mode == GANMode.VANILLA:
                # Standard GAN generator loss
                d_fake = self.discriminator(fake_data)
                g_loss = self.bce_loss(d_fake, torch.ones_like(d_fake))
            else:
                # Wasserstein generator loss
                d_fake = self.discriminator(fake_data)
                g_loss = -d_fake.mean()
            
            # Entropy regularization
            if self.config.entropy_weight > 0:
                real_entropy = self.compute_entropy(real_data)
                fake_entropy = self.compute_entropy(fake_data)
                entropy_loss = abs(real_entropy - fake_entropy)
                g_loss += self.config.entropy_weight * entropy_loss
                metrics.entropy_difference = entropy_loss
            
            g_loss.backward()
            self.optimizer_g.step()
            
            metrics.generator_loss = g_loss.item()
            metrics.training_time = time.time() - step_start_time
            
            return metrics
            
        except Exception as e:
            self.safe_log('error', f"Error in training step: {e}")
            return TrainingMetrics()
    
    def train_entropy_gan(self, real_data_fn: Callable[[int], torch.Tensor], 
                         epochs: Optional[int] = None, batch_size: Optional[int] = None,
                         progress_callback: Optional[Callable[[TrainingMetrics], None]] = None) -> List[TrainingMetrics]:
        """
        Train the Entropy GAN system
        
        Args:
            real_data_fn: Function that returns real data batches
            epochs: Number of training epochs (uses config if None)
            batch_size: Batch size (uses config if None)
            progress_callback: Optional callback for training progress
            
        Returns:
            List of training metrics
        """
        try:
            with self.training_lock:
                self.is_training = True
                
                epochs = epochs or self.config.epochs
                batch_size = batch_size or self.config.batch_size
                
                self.safe_print(f"🚀 Starting Entropy GAN training")
                self.safe_print(f"   Mode: {self.config.mode.value}")
                self.safe_print(f"   Epochs: {epochs}")
                self.safe_print(f"   Batch size: {batch_size}")
                self.safe_print(f"   Device: {self.device}")
                
                training_start_time = time.time()
                metrics_history = []
                
                for epoch in range(epochs):
                    try:
                        # Get real data batch
                        real_data = real_data_fn(batch_size)
                        if not isinstance(real_data, torch.Tensor):
                            real_data = torch.tensor(real_data, dtype=torch.float32)
                        real_data = real_data.to(self.device)
                        
                        # Training step
                        metrics = self.train_step(real_data)
                        metrics.epoch = epoch
                        metrics.total_time = time.time() - training_start_time
                        
                        metrics_history.append(metrics)
                        self.training_metrics.append(metrics)
                        
                        # Progress reporting
                        if epoch % 100 == 0:
                            self.safe_print(
                                f"📊 Epoch {epoch}: "
                                f"D_loss={metrics.discriminator_loss:.4f}, "
                                f"G_loss={metrics.generator_loss:.4f}, "
                                f"Real_acc={metrics.real_accuracy:.3f}, "
                                f"Fake_acc={metrics.fake_accuracy:.3f}"
                            )
                        
                        # Call progress callback
                        if progress_callback:
                            progress_callback(metrics)
                        
                    except Exception as e:
                        self.safe_log('error', f"Error in epoch {epoch}: {e}")
                        continue
                
                self.is_training = False
                total_time = time.time() - training_start_time
                
                self.safe_print(f"🎉 Training completed in {total_time:.2f} seconds")
                return metrics_history
                
        except Exception as e:
            self.is_training = False
            error_msg = f"Error in GAN training: {e}"
            self.safe_log('error', error_msg)
            raise


class GanFilter:
    """
    GAN-based signal filtering system
    
    Provides various filtering modes using trained GAN discriminators for
    signal validation and anomaly detection.
    """
    
    def __init__(self, discriminator: EntropyDiscriminator, config: FilterConfig) -> None:
        """
        Initialize GAN filter
        
        Args:
            discriminator: Trained discriminator network
            config: Filter configuration
        """
        self.discriminator = discriminator
        self.config = config
        self.cli_handler = CLIHandler()
        
        # Adaptive filtering state
        self.confidence_history: List[float] = []
        self.adaptive_threshold = config.threshold
        
        # Performance metrics
        self.filtered_count = 0
        self.total_count = 0
        
        logger.info(f"GanFilter initialized with {config.mode.value} mode")
    
    def safe_print(self, message: str, force_ascii: bool = False) -> None:
        """Safe print with CLI compatibility"""
        if CLI_COMPATIBILITY_AVAILABLE:
            safe_print(message, force_ascii=force_ascii)
        else:
            safe_message = self.cli_handler.safe_emoji_print(message, force_ascii=force_ascii)
            print(safe_message)
    
    def gan_filter(self, signal: TensorType, threshold: Optional[float] = None) -> TensorType:
        """
        Filter entropy signal using trained discriminator
        
        Args:
            signal: Input signal batch
            threshold: Confidence cutoff (uses config if None)
            
        Returns:
            Validated signals above threshold
        """
        try:
            threshold = threshold or self.config.threshold
            
            # Convert to tensor if needed
            if not isinstance(signal, torch.Tensor):
                signal_tensor = torch.tensor(signal, dtype=torch.float32)
            else:
                signal_tensor = signal
            
            # Ensure correct device
            device = next(self.discriminator.parameters()).device
            signal_tensor = signal_tensor.to(device)
            
            # Get discriminator scores
            with torch.no_grad():
                scores = self.discriminator(signal_tensor)
                
                if self.config.mode == FilterMode.THRESHOLD:
                    # Simple threshold filtering
                    mask = scores.view(-1) > threshold
                    
                elif self.config.mode == FilterMode.CONFIDENCE:
                    # Confidence-based filtering using percentile
                    confidence_threshold = torch.quantile(scores, self.config.confidence_percentile / 100.0)
                    mask = scores.view(-1) > confidence_threshold
                    
                elif self.config.mode == FilterMode.ADAPTIVE:
                    # Adaptive threshold filtering
                    self._update_adaptive_threshold(scores)
                    mask = scores.view(-1) > self.adaptive_threshold
                    
                else:  # ENTROPY_AWARE
                    # Entropy-aware filtering (placeholder for more complex logic)
                    mask = scores.view(-1) > threshold
            
            # Filter signals
            filtered_signal = signal_tensor[mask]
            
            # Update metrics
            self.total_count += signal_tensor.size(0)
            self.filtered_count += filtered_signal.size(0)
            
            # Convert back to original type if needed
            if not isinstance(signal, torch.Tensor):
                filtered_signal = filtered_signal.cpu().numpy()
            
            return filtered_signal
            
        except Exception as e:
            logger.error(f"Error in GAN filtering: {e}")
            return signal  # Return original signal on error
    
    def batch_filter(self, signal_fn: Callable[[int], TensorType], batch_size: int) -> TensorType:
        """
        Pull signal from provider and filter it
        
        Args:
            signal_fn: Function that provides signal batches
            batch_size: Size of batches to process
            
        Returns:
            Filtered signal batch
        """
        try:
            # Get signal batch
            signal = signal_fn(batch_size)
            
            # Apply filtering
            filtered_signal = self.gan_filter(signal)
            
            return filtered_signal
            
        except Exception as e:
            logger.error(f"Error in batch filtering: {e}")
            return signal_fn(batch_size)  # Return unfiltered on error
    
    def _update_adaptive_threshold(self, scores: torch.Tensor) -> None:
        """Update adaptive threshold based on recent scores"""
        try:
            # Add scores to history
            score_values = scores.view(-1).cpu().numpy().tolist()
            self.confidence_history.extend(score_values)
            
            # Keep only recent history
            if len(self.confidence_history) > self.config.adaptive_window:
                self.confidence_history = self.confidence_history[-self.config.adaptive_window:]
            
            # Update threshold if we have enough samples
            if len(self.confidence_history) >= self.config.min_samples:
                # Use median as adaptive threshold
                self.adaptive_threshold = float(np.median(self.confidence_history))
                
        except Exception as e:
            logger.error(f"Error updating adaptive threshold: {e}")
    
    def get_filter_stats(self) -> Dict[str, Any]:
        """
        Get filtering statistics
        
        Returns:
            Dictionary containing filter performance metrics
        """
        try:
            if self.total_count > 0:
                filter_rate = self.filtered_count / self.total_count
            else:
                filter_rate = 0.0
            
            return {
                'total_processed': self.total_count,
                'signals_passed': self.filtered_count,
                'signals_filtered': self.total_count - self.filtered_count,
                'filter_rate': filter_rate,
                'pass_rate': 1.0 - filter_rate,
                'current_threshold': self.adaptive_threshold,
                'confidence_history_size': len(self.confidence_history)
            }
            
        except Exception as e:
            logger.error(f"Error getting filter stats: {e}")
            return {}


def create_entropy_signal_provider(signal_dim: int = 64, noise_level: float = 0.1) -> Callable[[int], torch.Tensor]:
    """
    Create a sample entropy signal provider for testing
    
    Args:
        signal_dim: Dimension of signals to generate
        noise_level: Level of noise to add
        
    Returns:
        Function that generates signal batches
    """
    def signal_provider(batch_size: int) -> torch.Tensor:
        """Generate synthetic entropy signals for testing"""
        try:
            # Generate base signals (sinusoidal with varying frequency)
            t = torch.linspace(0, 2 * math.pi, signal_dim)
            signals = []
            
            for _ in range(batch_size):
                freq = torch.rand(1) * 5 + 1  # Random frequency 1-6
                phase = torch.rand(1) * 2 * math.pi  # Random phase
                signal = torch.sin(freq * t + phase)
                
                # Add noise
                noise = torch.randn_like(signal) * noise_level
                signal += noise
                
                signals.append(signal)
            
            return torch.stack(signals)
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return torch.randn(batch_size, signal_dim)
    
    return signal_provider


def main() -> None:
    """
    Main function for testing Entropy GAN Filter
    
    Demonstrates the complete GAN training and filtering pipeline with
    CLI-safe output and comprehensive error handling.
    """
    try:
        if not TORCH_AVAILABLE:
            print("❌ PyTorch not available - cannot run Entropy GAN Filter test")
            return
        
        print("🚀 Entropy GAN Filter Test")
        print("=" * 50)
        
        # Configuration
        gan_config = GANConfig(
            noise_dim=100,
            signal_dim=64,
            generator_hidden=128,
            discriminator_hidden=128,
            learning_rate=1e-4,
            batch_size=32,
            epochs=200,  # Reduced for testing
            mode=GANMode.VANILLA
        )
        
        filter_config = FilterConfig(
            threshold=0.5,
            mode=FilterMode.THRESHOLD
        )
        
        print(f"📊 Configuration:")
        print(f"   Signal dimension: {gan_config.signal_dim}")
        print(f"   Batch size: {gan_config.batch_size}")
        print(f"   Training epochs: {gan_config.epochs}")
        print(f"   GAN mode: {gan_config.mode.value}")
        
        # Initialize GAN
        print("\n🔧 Initializing Entropy GAN...")
        entropy_gan = EntropyGAN(gan_config)
        
        # Create signal provider
        print("📡 Creating signal provider...")
        signal_provider = create_entropy_signal_provider(gan_config.signal_dim, 0.1)
        
        # Train GAN
        print("\n🎓 Training Entropy GAN...")
        training_metrics = entropy_gan.train_entropy_gan(
            real_data_fn=signal_provider,
            epochs=gan_config.epochs,
            batch_size=gan_config.batch_size
        )
        
        if training_metrics:
            final_metrics = training_metrics[-1]
            print(f"✅ Training completed:")
            print(f"   Final G loss: {final_metrics.generator_loss:.4f}")
            print(f"   Final D loss: {final_metrics.discriminator_loss:.4f}")
            print(f"   Real accuracy: {final_metrics.real_accuracy:.3f}")
            print(f"   Fake accuracy: {final_metrics.fake_accuracy:.3f}")
        
        # Test filtering
        print("\n🔍 Testing GAN filtering...")
        gan_filter = GanFilter(entropy_gan.discriminator, filter_config)
        
        # Generate test signals
        test_signals = signal_provider(100)
        print(f"   Generated {test_signals.size(0)} test signals")
        
        # Apply filtering
        filtered_signals = gan_filter.gan_filter(test_signals)
        print(f"   Filtered to {filtered_signals.size(0)} valid signals")
        
        # Get filter statistics
        stats = gan_filter.get_filter_stats()
        print(f"   Filter statistics:")
        print(f"     Pass rate: {stats.get('pass_rate', 0):.2%}")
        print(f"     Signals passed: {stats.get('signals_passed', 0)}")
        print(f"     Signals filtered: {stats.get('signals_filtered', 0)}")
        
        # Test batch filtering
        print("\n📦 Testing batch filtering...")
        batch_filtered = gan_filter.batch_filter(signal_provider, 50)
        print(f"   Batch filtered to {batch_filtered.size(0)} signals")
        
        print("\n🎉 Entropy GAN Filter test completed successfully!")
        
    except Exception as e:
        print(f"❌ Entropy GAN Filter test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
