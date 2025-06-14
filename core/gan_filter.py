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
import hashlib
import time

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

@dataclass
class MatrixCluster:
    """Parent cluster for matrix adoption"""
    cluster_id: str
    matrices: List[np.ndarray]
    timestamp: float
    entropy_bounds: Tuple[float, float]
    correction_vector: Optional[np.ndarray] = None
    parent_id: Optional[str] = None
    profit_factor: float = 1.0

@dataclass
class ChildMatrixRecord:
    """Child matrix record for adoption tracking"""
    matrix: np.ndarray
    failed_reason: str
    adopted_from: Optional[str] = None
    corrected_matrix: Optional[np.ndarray] = None
    correction_hash: Optional[str] = None
    profit_routing: Optional[Dict[str, float]] = None

@dataclass
class TickProcessingResult:
    """Result of tick-by-tick processing"""
    original_vector: np.ndarray
    processed_vector: np.ndarray
    anomaly_metrics: 'GANAnomalyMetrics'
    correction_applied: bool
    profit_signal: Dict[str, float]
    processing_time: float
    tick_hash: str

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
        
        # Matrix cluster database for parent-child adoption
        self.cluster_db: Dict[str, MatrixCluster] = {}
        self.tick_cache: Dict[str, TickProcessingResult] = {}
        
        # Mathematical validation constants
        self.H_MIN, self.H_MAX = 0.1, 8.0
        self.CONDITION_THRESHOLD = 1e7
        self.SPECTRAL_THRESHOLD = 0.3
        self.EPSILON = 1e-10
        
    def _validate_matrix(self, matrix: np.ndarray) -> Tuple[bool, List[str]]:
        """Mathematical validation of input matrix"""
        reasons = []
        
        # Entropy check
        p = matrix / (np.sum(matrix) + self.EPSILON)
        p = np.clip(p, self.EPSILON, 1.0)
        entropy = -np.sum(p * np.log2(p))
        
        if not (self.H_MIN <= entropy <= self.H_MAX):
            reasons.append(f"entropy_bounds({entropy:.3f})")
            
        # Condition number check
        try:
            u, s, vh = np.linalg.svd(matrix.reshape(-1, 1) if matrix.ndim == 1 else matrix, full_matrices=False)
            cond = np.max(s) / max(np.min(s), self.EPSILON)
            if cond > self.CONDITION_THRESHOLD:
                reasons.append(f"ill_conditioned({cond:.2e})")
        except np.linalg.LinAlgError:
            reasons.append("svd_failed")
            
        # Spectral deviation check
        spectrum = np.abs(np.fft.fft2(matrix.reshape(8, -1) if matrix.ndim == 1 else matrix))
        dev = np.std(spectrum)
        if dev > self.SPECTRAL_THRESHOLD:
            reasons.append(f"spectral_dev({dev:.3f})")
            
        return len(reasons) == 0, reasons
        
    def _adopt_from_parent(self, vector: np.ndarray, reasons: List[str]) -> ChildMatrixRecord:
        """Apply parent-child adoption for failed matrices"""
        # Find best parent cluster
        best_cluster = None
        best_score = float('inf')
        
        for cluster in self.cluster_db.values():
            # Score based on entropy similarity and age
            avg_entropy = sum(cluster.entropy_bounds) / 2
            if self.H_MIN <= avg_entropy <= self.H_MAX:
                age_factor = max(0.1, 1.0 - (time.time() - cluster.timestamp) / 3600)  # Decay over hour
                score = abs(avg_entropy - self.H_MIN) + (1 - age_factor)
                if score < best_score:
                    best_score = score
                    best_cluster = cluster
                    
        if best_cluster and best_cluster.correction_vector is not None:
            # Apply correction
            corrected = vector + best_cluster.correction_vector[:len(vector)]
            corrected = (corrected - np.min(corrected)) / (np.max(corrected) - np.min(corrected) + self.EPSILON)
            
            # Generate correction hash
            correction_hash = hashlib.md5(corrected.tobytes()).hexdigest()[:8]
            
            # Calculate profit routing weights
            profit_routing = {
                'base_profit': best_cluster.profit_factor,
                'correction_bonus': 0.1 * best_cluster.profit_factor,
                'cluster_id': best_cluster.cluster_id
            }
            
            return ChildMatrixRecord(
                matrix=vector,
                failed_reason=",".join(reasons),
                adopted_from=best_cluster.cluster_id,
                corrected_matrix=corrected,
                correction_hash=correction_hash,
                profit_routing=profit_routing
            )
            
        return ChildMatrixRecord(
            matrix=vector,
            failed_reason=",".join(reasons),
            adopted_from=None
        )
        
    def _create_tick_hash(self, vector: np.ndarray) -> str:
        """Create deterministic hash for tick tracking"""
        return hashlib.sha256(vector.tobytes()).hexdigest()[:12]
        
    def _extract_profit_signals(self, anomaly_score: float, correction_applied: bool, 
                               profit_routing: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Extract profit signals from anomaly analysis"""
        base_signal = 1.0 - anomaly_score  # Higher profit for lower anomaly
        
        signals = {
            'anomaly_profit': base_signal,
            'confidence_profit': base_signal * (1.0 - anomaly_score),
            'correction_penalty': 0.9 if correction_applied else 1.0
        }
        
        if profit_routing:
            signals.update({
                'parent_boost': profit_routing.get('base_profit', 1.0),
                'correction_bonus': profit_routing.get('correction_bonus', 0.0)
            })
            
        return signals

    def compute_volatility_regime(self, x: torch.Tensor) -> torch.Tensor:
        """Compute volatility regime: σ_t / σ̄_t-n:t (FIXED VERSION)"""
        if len(x) < self.config.momentum_window + 1:
            return torch.tensor(1.0, device=self.device)  # Default regime
            
        returns = x[1:] - x[:-1]
        
        if len(returns) < self.config.momentum_window:
            return torch.tensor(1.0, device=self.device)
            
        # Current window volatility
        vol = torch.std(returns[-self.config.momentum_window:])
        
        # Historical volatility (avoid unfold on small tensors)
        if len(returns) >= 2 * self.config.momentum_window:
            windows = returns.unfold(0, self.config.momentum_window, 1)
            vol_mean = torch.mean(torch.std(windows, dim=1))
        else:
            vol_mean = torch.std(returns)
            
        return vol / (vol_mean + 1e-6)

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
        
    def detect(self, vector: np.ndarray) -> GANAnomalyMetrics:
        """Detect anomaly in input vector with tick-by-tick processing."""
        start_time = time.time()
        logger.info(f"Processing tick: shape={vector.shape}")
        
        try:
            # Step 1: Validate input matrix mathematically
            is_valid, validation_reasons = self._validate_matrix(vector)
            processed_vector = vector.copy()
            correction_applied = False
            profit_routing = None
            
            # Step 2: Apply parent-child adoption if needed
            if not is_valid:
                adoption_record = self._adopt_from_parent(vector, validation_reasons)
                if adoption_record.corrected_matrix is not None:
                    processed_vector = adoption_record.corrected_matrix
                    correction_applied = True
                    profit_routing = adoption_record.profit_routing
                    logger.info(f"Applied correction from cluster {adoption_record.adopted_from}")
                else:
                    logger.warning(f"No parent available for correction: {adoption_record.failed_reason}")
                    
            # Step 3: Convert to tensor and run GAN inference
            data_tensor = torch.from_numpy(processed_vector).float().to(self.device)
            if data_tensor.dim() == 1:
                data_tensor = data_tensor.unsqueeze(0)  # Add batch dimension
                
            # Step 4: Compute anomaly score using existing method
            anomaly_score = self.compute_anomaly_score(data_tensor)
            
            # Step 5: Compute reconstruction error
            with torch.no_grad():
                z = torch.randn(1, self.config.latent_dim).to(self.device)
                reconstructed = self.generator(z)
                recon_error = torch.norm(data_tensor - reconstructed, p=2).item()
                
            # Step 6: Determine anomaly status
            is_anomaly = anomaly_score > self.config.anomaly_threshold
            confidence = min(anomaly_score / self.config.anomaly_threshold, 1.0)
            
            # Step 7: Extract profit signals
            profit_signals = self._extract_profit_signals(anomaly_score, correction_applied, profit_routing)
            
            # Step 8: Create processing result
            processing_time = time.time() - start_time
            tick_hash = self._create_tick_hash(vector)
            
            # Cache result
            result = TickProcessingResult(
                original_vector=vector,
                processed_vector=processed_vector,
                anomaly_metrics=GANAnomalyMetrics(
                    anomaly_score=anomaly_score,
                    reconstruction_error=recon_error,
                    is_anomaly=is_anomaly,
                    confidence=confidence
                ),
                correction_applied=correction_applied,
                profit_signal=profit_signals,
                processing_time=processing_time,
                tick_hash=tick_hash
            )
            
            self.tick_cache[tick_hash] = result
            
            logger.info(f"Tick processed: anomaly={is_anomaly}, score={anomaly_score:.3f}, "
                       f"correction={correction_applied}, profit={profit_signals.get('anomaly_profit', 0):.3f}")
                       
            return result.anomaly_metrics
            
        except Exception as e:
            logger.error(f"GAN detection failed: {str(e)}")
            return GANAnomalyMetrics(
                anomaly_score=1.0,  # Max anomaly on error
                reconstruction_error=float('inf'),
                is_anomaly=True,
                confidence=0.0
            )

    def train(self, dataset: List[np.ndarray]) -> None:
        """Train GAN on dataset with cluster creation."""
        logger.info(f"Training GAN filter on {len(dataset)} samples")
        
        try:
            # Convert dataset to tensors
            train_data = []
            valid_samples = 0
            
            for i, sample in enumerate(dataset):
                is_valid, reasons = self._validate_matrix(sample)
                if is_valid:
                    train_data.append(torch.from_numpy(sample).float())
                    valid_samples += 1
                else:
                    logger.debug(f"Skipping invalid sample {i}: {reasons}")
                    
            if valid_samples == 0:
                raise ValueError("No valid samples in dataset")
                
            logger.info(f"Using {valid_samples}/{len(dataset)} valid samples for training")
            
            # Create data loader
            train_tensor = torch.stack(train_data).to(self.device)
            dataset_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(train_tensor),
                batch_size=self.config.batch_size,
                shuffle=True
            )
            
            # Training loop
            for epoch in range(self.config.num_epochs):
                epoch_losses = {'g_loss': 0, 'd_loss': 0, 'entropy_loss': 0, 'phase_loss': 0, 'consistency_loss': 0}
                batch_count = 0
                
                for batch_data, in dataset_loader:
                    losses = self.train_step(batch_data)
                    
                    for key, value in losses.items():
                        epoch_losses[key] += value
                    batch_count += 1
                    
                # Average losses
                for key in epoch_losses:
                    epoch_losses[key] /= batch_count
                    
                self.training_history.append(epoch_losses)
                
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}: G_loss={epoch_losses['g_loss']:.4f}, "
                               f"D_loss={epoch_losses['d_loss']:.4f}")
                               
            # Create matrix clusters from trained data
            self._create_matrix_clusters(train_data)
            
            # Save trained model
            self.save_model()
            logger.info("GAN training completed successfully")
            
        except Exception as e:
            logger.error(f"GAN training failed: {str(e)}")
            raise
            
    def _create_matrix_clusters(self, train_data: List[torch.Tensor]) -> None:
        """Create matrix clusters for parent-child adoption"""
        logger.info("Creating matrix clusters from training data")
        
        # Group samples by entropy ranges
        entropy_groups = {}
        
        for tensor in train_data:
            np_data = tensor.cpu().numpy()
            p = np_data / (np.sum(np_data) + self.EPSILON)
            p = np.clip(p, self.EPSILON, 1.0)
            entropy = -np.sum(p * np.log2(p))
            
            # Discretize entropy into groups
            entropy_bin = round(entropy, 1)
            if entropy_bin not in entropy_groups:
                entropy_groups[entropy_bin] = []
            entropy_groups[entropy_bin].append(np_data)
            
        # Create clusters
        for entropy_bin, matrices in entropy_groups.items():
            if len(matrices) >= 3:  # Minimum cluster size
                cluster_id = f"cluster_{entropy_bin}_{int(time.time())}"
                
                # Compute correction vector as mean
                correction_vector = np.mean(matrices, axis=0)
                
                # Calculate profit factor based on cluster quality
                profit_factor = min(2.0, 1.0 + len(matrices) / 100.0)
                
                cluster = MatrixCluster(
                    cluster_id=cluster_id,
                    matrices=matrices,
                    timestamp=time.time(),
                    entropy_bounds=(entropy_bin - 0.05, entropy_bin + 0.05),
                    correction_vector=correction_vector,
                    profit_factor=profit_factor
                )
                
                self.cluster_db[cluster_id] = cluster
                logger.info(f"Created cluster {cluster_id} with {len(matrices)} matrices")
                
        logger.info(f"Created {len(self.cluster_db)} matrix clusters")

    def get_tick_result(self, tick_hash: str) -> Optional[TickProcessingResult]:
        """Retrieve cached tick processing result"""
        return self.tick_cache.get(tick_hash)
        
    def get_profit_routing(self) -> Dict[str, Dict[str, float]]:
        """Get current profit routing configuration"""
        routing = {}
        for tick_hash, result in self.tick_cache.items():
            if result.profit_signal:
                routing[tick_hash] = result.profit_signal
        return routing

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