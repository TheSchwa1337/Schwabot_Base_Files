"""
Quantum Anti-Pole Engine v5.1 (Corrected)
==========================================

Core mathematical backend for Schwabot Anti-Pole Theory.
Fixed GPU memory management, proper async boundaries, and mathematical correctness.

Integrates with existing Schwabot components:
- entropy_tracker.py for hash entropy calculations
- thermal_zone_manager.py for thermal protection
- gpu_flash_engine.py for GPU acceleration
- ferris_wheel_scheduler.py for strategy rotation
"""

from __future__ import annotations

import math
import time
import logging
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
from typing import List, Tuple, Dict, Optional, Union
import asyncio
import concurrent.futures

import numpy as np
try:
    import cupy as cp
    _GPU = cp.cuda.is_available()
except (ImportError, ModuleNotFoundError):
    cp, _GPU = None, False

# Import existing Schwabot components
try:
    from .entropy_tracker import EntropyTracker
    from .thermal_zone_manager import ThermalZoneManager
    from .gpu_flash_engine import GPUFlashEngine
    from .ferris_wheel_scheduler import FerrisWheelScheduler
    from .hash_profit_matrix import HashProfitMatrix
    from .vault_router import VaultRouter
except ImportError:
    # Fallback for standalone usage
    EntropyTracker = None
    ThermalZoneManager = None
    GPUFlashEngine = None
    FerrisWheelScheduler = None
    HashProfitMatrix = None
    VaultRouter = None

# ---------------------------------------------------------------------------
#  CONFIGURATION
# ---------------------------------------------------------------------------

@dataclass
class QAConfig:
    """Quantum Anti-Pole Engine Configuration"""
    # Core parameters
    use_gpu: bool = _GPU
    field_size: int = 64            # NxN wave-function grid (power of 2)
    tick_window: int = 256          # samples stored for pole fitting
    pole_order: int = 12            # AR model order
    capt_grid: int = 64             # Cylindrical Anti-Pole Transform grid
    price_bins: int = 64            # Φ(x, t) discretization (x-axis)
    lookback: int = 16              # Φ(x, t) temporal depth
    ap_rsi_period: int = 14         # AP-RSI smoothing period
    
    # Mathematical parameters
    entropy_sigmoid_k: float = 6.0        # Sigmoid steepness
    entropy_sigmoid_mu: float = 0.015     # Sigmoid threshold
    sigma_decay: float = 3.0              # Distance decay rate
    beta_capt: float = 1.0                # CAPT power exponent
    quantum_time_step: float = 1.0/256    # Evolution time step
    pole_stability_threshold: float = 1.0  # |pole| threshold for stability
    
    # Integration settings
    use_entropy_tracker: bool = True       # Use existing entropy tracker
    use_thermal_manager: bool = True       # Use existing thermal manager
    use_gpu_flash: bool = True             # Use GPU flash engine
    use_ferris_wheel: bool = True          # Use strategy scheduler
    
    # Performance tuning
    max_poles: int = 16                    # Maximum poles to track
    regularization_lambda: float = 1e-6   # Regularization for pole fitting
    vector_field_threads: int = 4         # Threads for vector field calc
    
    # Logging
    logger_name: str = "quantum.antipole"
    debug_mode: bool = False

# ---------------------------------------------------------------------------
#  DATA STRUCTURES
# ---------------------------------------------------------------------------

@dataclass
class QuantumState:
    """Quantum state representation"""
    psi: complex                    # Primary wavefunction coefficient
    phi: complex                    # Phase component
    entropy: float                  # Quantum entropy
    coherence: float               # Coherence measure
    timestamp: datetime
    energy: float = 0.0            # Total energy
    phase_velocity: float = 0.0    # Phase velocity

@dataclass
class ComplexPole:
    """Complex pole from transfer function analysis"""
    real: float                     # Real part (σ)
    imag: float                     # Imaginary part (ω)
    magnitude: float               # |pole|
    phase: float                   # arg(pole)
    stability: str                 # "stable", "unstable", "marginal"
    resonance_freq: float = 0.0    # Resonance frequency
    damping_ratio: float = 0.0     # Damping ratio
    confidence: float = 1.0        # Fitting confidence

@dataclass
class VectorField:
    """4D vector field for anti-pole guidance"""
    coords: np.ndarray             # Shape: (H, W, 4) - [x, y, z, w]
    vectors: np.ndarray            # Shape: (H, W, 4) - [vx, vy, vz, vw]
    magnitude: np.ndarray          # Shape: (H, W) - vector magnitudes
    divergence: np.ndarray         # Shape: (H, W) - ∇·v
    curl: np.ndarray              # Shape: (H, W) - ∇×v (z-component)
    potential: np.ndarray         # Shape: (H, W) - scalar potential

@dataclass
class APFrame:
    """Complete Anti-Pole analysis frame"""
    uid: str                       # Unique frame identifier
    timestamp: datetime
    price: float
    volume: float
    
    # Core components
    quantum_state: QuantumState
    complex_poles: List[ComplexPole]
    vector_field: VectorField
    phi_surface: np.ndarray        # Potential surface Φ(x,t)
    capt_transform: np.ndarray     # Cylindrical transform
    ap_rsi: float                  # Anti-Pole RSI value
    
    # Integration data
    entropy_metrics: Dict = field(default_factory=dict)
    thermal_state: Dict = field(default_factory=dict)
    ferris_wheel_state: Dict = field(default_factory=dict)
    
    # Performance metrics
    computation_time_ms: float = 0.0
    gpu_utilization: float = 0.0

# ---------------------------------------------------------------------------
#  QUANTUM ANTI-POLE ENGINE
# ---------------------------------------------------------------------------

class QuantumAntiPoleEngine:
    """
    Main Quantum Anti-Pole Engine with Schwabot integration
    
    Combines quantum mechanics, complex analysis, and fractal geometry
    for real-time financial market navigation and profit optimization.
    """
    
    def __init__(self, config: Optional[QAConfig] = None):
        self.config = config or QAConfig()
        self.log = logging.getLogger(self.config.logger_name)
        
        # Choose computation backend
        self.use_gpu = self.config.use_gpu and _GPU
        self.xp = cp if self.use_gpu else np
        
        # Initialize Schwabot component integrations
        self._init_schwabot_integrations()
        
        # Market data buffers
        self.price_buffer: deque[float] = deque(maxlen=self.config.tick_window)
        self.volume_buffer: deque[float] = deque(maxlen=self.config.tick_window)
        self.timestamp_buffer: deque[datetime] = deque(maxlen=self.config.tick_window)
        
        # Quantum field initialization
        self._init_quantum_field()
        
        # AP-RSI state
        self._ap_rsi_state = {
            'up_ema': 0.0,
            'down_ema': 0.0,
            'initialized': False,
            'last_phi': 0.0
        }
        
        # Performance tracking
        self.frame_count = 0
        self.total_computation_time = 0.0
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.config.vector_field_threads
        )
        
        self.log.info(f"QuantumAntiPoleEngine initialized (GPU={self.use_gpu})")
    
    def _init_schwabot_integrations(self):
        """Initialize connections to existing Schwabot components"""
        self.entropy_tracker = None
        self.thermal_manager = None
        self.gpu_flash = None
        self.ferris_wheel = None
        self.hash_profit_matrix = None
        self.vault_router = None
        
        if self.config.use_entropy_tracker and EntropyTracker:
            try:
                self.entropy_tracker = EntropyTracker()
                self.log.info("✓ EntropyTracker integration enabled")
            except Exception as e:
                self.log.warning(f"EntropyTracker init failed: {e}")
        
        if self.config.use_thermal_manager and ThermalZoneManager:
            try:
                self.thermal_manager = ThermalZoneManager()
                self.log.info("✓ ThermalZoneManager integration enabled")
            except Exception as e:
                self.log.warning(f"ThermalZoneManager init failed: {e}")
        
        if self.config.use_gpu_flash and GPUFlashEngine and self.use_gpu:
            try:
                self.gpu_flash = GPUFlashEngine()
                self.log.info("✓ GPUFlashEngine integration enabled")
            except Exception as e:
                self.log.warning(f"GPUFlashEngine init failed: {e}")
        
        if self.config.use_ferris_wheel and FerrisWheelScheduler:
            try:
                self.ferris_wheel = FerrisWheelScheduler()
                self.log.info("✓ FerrisWheelScheduler integration enabled")
            except Exception as e:
                self.log.warning(f"FerrisWheelScheduler init failed: {e}")
    
    def _init_quantum_field(self):
        """Initialize quantum field grids and operators"""
        n = self.config.field_size
        
        # Coordinate grids
        x = self.xp.linspace(-3.0, 3.0, n)  # Extended range for better resolution
        y = self.xp.linspace(-3.0, 3.0, n)
        self.X, self.Y = self.xp.meshgrid(x, y, indexing='ij')
        
        # Harmonic oscillator potential V = ½(x² + y²)
        self.V_harmonic = 0.5 * (self.X**2 + self.Y**2)
        
        # Market-driven potential (will be updated each tick)
        self.V_market = self.xp.zeros_like(self.V_harmonic)
        
        # Initialize normalized Gaussian wavefunction
        sigma = 0.5
        self.psi = self.xp.exp(-(self.X**2 + self.Y**2) / (2 * sigma**2))
        self.psi = self.psi / self.xp.sqrt(self.xp.sum(self.xp.abs(self.psi)**2))
        
        # Momentum space grid for FFT evolution
        kx = self.xp.fft.fftfreq(n, d=6.0/n)  # dk = 2π/(L), L = 6
        ky = self.xp.fft.fftfreq(n, d=6.0/n)
        self.KX, self.KY = self.xp.meshgrid(kx, ky, indexing='ij')
        self.K2 = self.KX**2 + self.KY**2  # |k|²
        
        self.log.info(f"Quantum field initialized: {n}×{n} grid")
    
    # -----------------------------------------------------------------------
    #  PUBLIC API
    # -----------------------------------------------------------------------
    
    async def process_tick(self, price: float, volume: float, 
                          timestamp: Optional[datetime] = None) -> APFrame:
        """
        Main processing method - handles complete anti-pole analysis
        
        Args:
            price: Current BTC price
            volume: Current trading volume  
            timestamp: Optional timestamp (defaults to now)
            
        Returns:
            Complete APFrame with all analysis results
        """
        start_time = time.perf_counter()
        timestamp = timestamp or datetime.utcnow()
        
        # Update buffers
        self.price_buffer.append(price)
        self.volume_buffer.append(volume)
        self.timestamp_buffer.append(timestamp)
        
        # Integration with existing Schwabot components
        integration_data = await self._gather_integration_data(price, volume, timestamp)
        
        # Parallel computation of main components
        tasks = [
            self._evolve_quantum_state(price, volume, timestamp),
            self._analyze_complex_poles(),
            self._compute_phi_surface(integration_data.get('entropy', 0.0)),
        ]
        
        quantum_state, complex_poles, phi_surface = await asyncio.gather(*tasks)
        
        # Sequential computations that depend on previous results
        capt_transform = self._compute_capt_transform(complex_poles)
        vector_field = await self._compute_vector_field(capt_transform, phi_surface)
        ap_rsi = self._update_ap_rsi(phi_surface.mean() if phi_surface.size > 0 else 0.0)
        
        # Create frame
        computation_time = (time.perf_counter() - start_time) * 1000
        self.frame_count += 1
        self.total_computation_time += computation_time
        
        frame = APFrame(
            uid=f"apf_{self.frame_count:06d}_{int(timestamp.timestamp())}",
            timestamp=timestamp,
            price=price,
            volume=volume,
            quantum_state=quantum_state,
            complex_poles=complex_poles,
            vector_field=vector_field,
            phi_surface=phi_surface,
            capt_transform=capt_transform,
            ap_rsi=ap_rsi,
            entropy_metrics=integration_data.get('entropy_metrics', {}),
            thermal_state=integration_data.get('thermal_state', {}),
            ferris_wheel_state=integration_data.get('ferris_wheel_state', {}),
            computation_time_ms=computation_time,
            gpu_utilization=integration_data.get('gpu_utilization', 0.0)
        )
        
        # Log performance for significant events
        if self.config.debug_mode or computation_time > 50:
            self.log.debug(f"Frame {self.frame_count}: {computation_time:.1f}ms, "
                          f"Poles: {len(complex_poles)}, AP-RSI: {ap_rsi:.1f}")
        
        return frame
    
    # -----------------------------------------------------------------------
    #  INTEGRATION WITH EXISTING SCHWABOT COMPONENTS
    # -----------------------------------------------------------------------
    
    async def _gather_integration_data(self, price: float, volume: float, 
                                     timestamp: datetime) -> Dict:
        """Gather data from integrated Schwabot components"""
        integration_data = {}
        
        # Entropy tracking
        if self.entropy_tracker:
            try:
                entropy_result = self.entropy_tracker.calculate_entropy(price, volume)
                integration_data['entropy'] = entropy_result.get('entropy', 0.0)
                integration_data['entropy_metrics'] = entropy_result
            except Exception as e:
                self.log.warning(f"EntropyTracker error: {e}")
                integration_data['entropy'] = 0.0
                integration_data['entropy_metrics'] = {}
        
        # Thermal monitoring
        if self.thermal_manager:
            try:
                thermal_state = self.thermal_manager.get_thermal_state()
                integration_data['thermal_state'] = thermal_state
                integration_data['thermal_safe'] = thermal_state.get('safe', True)
            except Exception as e:
                self.log.warning(f"ThermalZoneManager error: {e}")
                integration_data['thermal_state'] = {}
                integration_data['thermal_safe'] = True
        
        # GPU utilization
        if self.gpu_flash:
            try:
                gpu_metrics = self.gpu_flash.get_metrics()
                integration_data['gpu_utilization'] = gpu_metrics.get('utilization', 0.0)
            except Exception as e:
                self.log.warning(f"GPUFlashEngine error: {e}")
                integration_data['gpu_utilization'] = 0.0
        
        # Ferris wheel strategy state
        if self.ferris_wheel:
            try:
                wheel_state = self.ferris_wheel.get_current_state()
                integration_data['ferris_wheel_state'] = wheel_state
                integration_data['strategy_tier'] = wheel_state.get('current_tier', 'BRONZE')
            except Exception as e:
                self.log.warning(f"FerrisWheelScheduler error: {e}")
                integration_data['ferris_wheel_state'] = {}
                integration_data['strategy_tier'] = 'BRONZE'
        
        return integration_data
    
    # -----------------------------------------------------------------------
    #  QUANTUM STATE EVOLUTION
    # -----------------------------------------------------------------------
    
    async def _evolve_quantum_state(self, price: float, volume: float, 
                                   timestamp: datetime) -> QuantumState:
        """Evolve quantum state using split-operator method"""
        # Update market potential based on price/volume
        price_normalized = (price - 45000) / 10000  # Rough normalization
        volume_normalized = (volume - 1e6) / 1e6
        
        # Market-driven potential perturbation
        self.V_market = (
            price_normalized * self.X * 0.1 +  # Linear price gradient
            volume_normalized * (self.X**2 + self.Y**2) * 0.05  # Volume coupling
        )
        
        # Total potential
        V_total = self.V_harmonic + self.V_market
        
        # Split-operator evolution: exp(-iHdt) ≈ exp(-iVdt/2)exp(-iTdt)exp(-iVdt/2)
        dt = self.config.quantum_time_step
        
        # First half potential evolution
        self.psi *= self.xp.exp(-1j * V_total * dt / 2)
        
        # Kinetic evolution in momentum space
        psi_k = self.xp.fft.fft2(self.psi)
        psi_k *= self.xp.exp(-1j * self.K2 * dt / 2)  # ħ=1, m=1
        self.psi = self.xp.fft.ifft2(psi_k)
        
        # Second half potential evolution
        self.psi *= self.xp.exp(-1j * V_total * dt / 2)
        
        # Renormalize
        norm = self.xp.sqrt(self.xp.sum(self.xp.abs(self.psi)**2))
        if norm > 1e-12:
            self.psi /= norm
        
        # Calculate quantum observables
        density = self.xp.abs(self.psi)**2
        entropy = float(-self.xp.sum(density * self.xp.log(density + 1e-12)))
        
        # Coherence as off-diagonal density matrix elements
        coherence = float(self.xp.abs(self.xp.sum(self.psi))**2 / self.psi.size**2)
        
        # Energy expectation value
        energy = float(self.xp.real(
            self.xp.sum(self.xp.conj(self.psi) * V_total * self.psi)
        ))
        
        # Phase velocity (simplified)
        phase_velocity = float(self.xp.angle(self.psi[0, 0]))
        
        return QuantumState(
            psi=complex(self.psi[self.config.field_size//2, self.config.field_size//2]),
            phi=complex(self.xp.exp(1j * phase_velocity)),
            entropy=entropy,
            coherence=coherence,
            timestamp=timestamp,
            energy=energy,
            phase_velocity=phase_velocity
        )
    
    # -----------------------------------------------------------------------
    #  COMPLEX POLE ANALYSIS
    # -----------------------------------------------------------------------
    
    async def _analyze_complex_poles(self) -> List[ComplexPole]:
        """Analyze complex poles using regularized AR model fitting"""
        if len(self.price_buffer) < self.config.pole_order * 2:
            return []
        
        try:
            # Prepare data
            prices = np.array(list(self.price_buffer), dtype=np.float64)
            volumes = np.array(list(self.volume_buffer), dtype=np.float64)
            
            # Normalize to prevent numerical issues
            price_mean, price_std = np.mean(prices), np.std(prices)
            volume_mean, volume_std = np.mean(volumes), np.std(volumes)
            
            if price_std < 1e-10 or volume_std < 1e-10:
                return []  # Insufficient variation
            
            prices_norm = (prices - price_mean) / price_std
            volumes_norm = (volumes - volume_mean) / volume_std
            
            # Build regressor matrix for ARMAX model
            order = self.config.pole_order
            n_samples = len(prices_norm) - order
            
            if n_samples <= 0:
                return []
            
            # AR terms (past prices) + exogenous terms (volumes)
            X = np.zeros((n_samples, order * 2))
            y = prices_norm[order:]
            
            for i in range(n_samples):
                # AR terms
                X[i, :order] = prices_norm[i:i+order][::-1]  # Reverse for proper lag order
                # Exogenous terms
                X[i, order:] = volumes_norm[i:i+order][::-1]
            
            # Regularized least squares with proper conditioning
            lambda_reg = self.config.regularization_lambda
            XTX = X.T @ X + lambda_reg * np.eye(X.shape[1])
            XTy = X.T @ y
            
            # Solve using Cholesky decomposition for numerical stability
            try:
                L = np.linalg.cholesky(XTX)
                coeffs = np.linalg.solve(L.T, np.linalg.solve(L, XTy))
            except np.linalg.LinAlgError:
                # Fallback to SVD
                coeffs, residuals, rank, s = np.linalg.lstsq(X, y, rcond=lambda_reg)
            
            # Split coefficients
            ar_coeffs = coeffs[:order]
            
            # Form AR polynomial and find roots
            ar_poly = np.concatenate([[1.0], -ar_coeffs])
            roots = np.roots(ar_poly)
            
            # Convert to ComplexPole objects
            poles = []
            for i, root in enumerate(roots):
                if i >= self.config.max_poles:
                    break
                
                magnitude = abs(root)
                phase = np.angle(root)
                
                # Determine stability
                if magnitude < self.config.pole_stability_threshold:
                    stability = "stable"
                elif magnitude > self.config.pole_stability_threshold:
                    stability = "unstable"
                else:
                    stability = "marginal"
                
                # Calculate derived quantities
                if root.imag != 0:
                    resonance_freq = abs(root.imag) / (2 * np.pi)
                    damping_ratio = -root.real / magnitude if magnitude > 0 else 0
                else:
                    resonance_freq = 0.0
                    damping_ratio = -root.real if root.real < 0 else 0
                
                # Confidence based on residual and conditioning
                confidence = 1.0 / (1.0 + abs(residuals[0]) if len(residuals) > 0 else 1.0)
                
                poles.append(ComplexPole(
                    real=float(root.real),
                    imag=float(root.imag),
                    magnitude=float(magnitude),
                    phase=float(phase),
                    stability=stability,
                    resonance_freq=float(resonance_freq),
                    damping_ratio=float(damping_ratio),
                    confidence=float(confidence)
                ))
            
            # Sort by proximity to unit circle (most significant poles first)
            poles.sort(key=lambda p: abs(p.magnitude - 1.0))
            
            return poles
            
        except Exception as e:
            self.log.warning(f"Pole analysis failed: {e}")
            return []
    
    # -----------------------------------------------------------------------
    #  CYLINDRICAL ANTI-POLE TRANSFORM (CAPT)
    # -----------------------------------------------------------------------
    
    def _compute_capt_transform(self, poles: List[ComplexPole]) -> np.ndarray:
        """
        Compute Cylindrical Anti-Pole Transform
        CAPT(r,θ) = r^(-β) * cos(θ - ωt) * exp(-σr)
        """
        grid_size = self.config.capt_grid
        
        if not poles:
            return np.zeros((grid_size, grid_size))
        
        # Use dominant pole (closest to unit circle)
        dominant_pole = min(poles, key=lambda p: abs(p.magnitude - 1.0))
        
        # Create polar coordinate grids
        r = np.linspace(0.1, 3.0, grid_size)
        theta = np.linspace(0, 2*np.pi, grid_size)
        R, THETA = np.meshgrid(r, theta, indexing='ij')
        
        # CAPT parameters
        beta = self.config.beta_capt
        sigma = abs(dominant_pole.real)
        omega = dominant_pole.imag
        t_phase = self.frame_count * 0.1  # Synthetic time evolution
        
        # CAPT calculation with numerical safeguards
        capt = (
            np.power(R, -beta) * 
            np.cos(THETA - omega * t_phase) * 
            np.exp(-sigma * R)
        )
        
        # Handle numerical issues
        capt = np.nan_to_num(capt, nan=0.0, posinf=0.0, neginf=0.0)
        
        return capt
    
    # -----------------------------------------------------------------------
    #  VECTOR FIELD COMPUTATION
    # -----------------------------------------------------------------------
    
    async def _compute_vector_field(self, capt_transform: np.ndarray, 
                                   phi_surface: np.ndarray) -> VectorField:
        """Compute 4D vector field from CAPT and potential surface"""
        
        def compute_vector_components():
            # Get spatial gradients of CAPT
            dr_spacing = 2.9 / self.config.capt_grid  # r range: 0.1 to 3.0
            dtheta_spacing = 2 * np.pi / self.config.capt_grid
            
            # Gradient in polar coordinates
            dCAPT_dr, dCAPT_dtheta = np.gradient(capt_transform, dr_spacing, dtheta_spacing)
            
            # Convert to Cartesian coordinates
            r = np.linspace(0.1, 3.0, self.config.capt_grid)
            theta = np.linspace(0, 2*np.pi, self.config.capt_grid)
            R, THETA = np.meshgrid(r, theta, indexing='ij')
            
            # Polar to Cartesian vector conversion
            vr = dCAPT_dr
            vtheta = dCAPT_dtheta / (R + 1e-10)  # Avoid division by zero
            
            # Cartesian components
            vx = vr * np.cos(THETA) - vtheta * np.sin(THETA)
            vy = vr * np.sin(THETA) + vtheta * np.cos(THETA)
            
            # Map to grid coordinates
            grid_size = min(capt_transform.shape[0], phi_surface.shape[0])
            if grid_size != self.config.capt_grid:
                # Resize if needed
                from scipy.ndimage import zoom
                zoom_factor = grid_size / self.config.capt_grid
                vx = zoom(vx, zoom_factor, order=1)
                vy = zoom(vy, zoom_factor, order=1)
            
            # Create coordinate grid
            coords = np.zeros((grid_size, grid_size, 4))
            vectors = np.zeros((grid_size, grid_size, 4))
            
            # Fill coordinate and vector arrays
            x_coords = np.linspace(-1, 1, grid_size)
            y_coords = np.linspace(-1, 1, grid_size)
            X_cart, Y_cart = np.meshgrid(x_coords, y_coords, indexing='ij')
            
            coords[:, :, 0] = X_cart  # x
            coords[:, :, 1] = Y_cart  # y
            coords[:, :, 2] = 0       # z (could be time or other dimension)
            coords[:, :, 3] = 0       # w (anti-pole dimension)
            
            vectors[:, :, 0] = vx[:grid_size, :grid_size]  # vx
            vectors[:, :, 1] = vy[:grid_size, :grid_size]  # vy
            
            # Add components from potential surface gradient if available
            if phi_surface.size > 0 and phi_surface.shape[0] >= grid_size:
                dx = 2.0 / grid_size
                dt = 1.0
                dPhi_dx, dPhi_dt = np.gradient(phi_surface[:grid_size, :grid_size], dx, dt)
                vectors[:, :, 2] = dPhi_dx  # vz from potential gradient
                vectors[:, :, 3] = dPhi_dt  # vw from temporal gradient
            
            # Calculate derived fields
            magnitude = np.sqrt(np.sum(vectors**2, axis=2))
            
            # Divergence (2D approximation)
            dvx_dx, dvx_dy = np.gradient(vectors[:, :, 0])
            dvy_dx, dvy_dy = np.gradient(vectors[:, :, 1])
            divergence = dvx_dx + dvy_dy
            
            # Curl z-component (2D)
            curl = dvy_dx - dvx_dy
            
            # Scalar potential (negative of magnitude for attractive potential)
            potential = -magnitude
            
            return coords, vectors, magnitude, divergence, curl, potential
        
        # Run vector field computation in thread pool for better performance
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(self._executor, compute_vector_components)
        coords, vectors, magnitude, divergence, curl, potential = result
        
        return VectorField(
            coords=coords,
            vectors=vectors,
            magnitude=magnitude,
            divergence=divergence,
            curl=curl,
            potential=potential
        )
    
    # -----------------------------------------------------------------------
    #  POTENTIAL SURFACE CALCULATION
    # -----------------------------------------------------------------------
    
    async def _compute_phi_surface(self, entropy_input: float) -> np.ndarray:
        """
        Compute Anti-Pole Potential Surface
        Φ(x,t) = sigmoid(k*(Δψ̄ - μc)) × |Im(sp)| × exp(-σ|x-xp|)
        """
        k = self.config.entropy_sigmoid_k
        mu_c = self.config.entropy_sigmoid_mu
        sigma = self.config.sigma_decay
        
        # Entropy-drift sigmoid component
        sigmoid = 1.0 / (1.0 + math.exp(-k * (entropy_input - mu_c)))
        
        # Create spatial and temporal grids
        x = np.linspace(-3.0, 3.0, self.config.price_bins)[:, None]  # Price space
        t = np.arange(self.config.lookback)[None, :]  # Time lookback
        
        # Base potential surface
        x_equilibrium = 0.0  # Could be updated based on recent price statistics
        distance_decay = np.exp(-sigma * np.abs(x - x_equilibrium))
        temporal_decay = np.exp(-0.1 * t)  # Exponential time decay
        
        # Pole influence factor
        if len(self.price_buffer) > 0:
            # Use recent price volatility as proxy for imaginary pole component
            recent_prices = np.array(list(self.price_buffer)[-min(20, len(self.price_buffer)):])
            price_volatility = np.std(recent_prices) / np.mean(recent_prices) if len(recent_prices) > 1 else 0.1
            omega_factor = min(price_volatility * 10, 2.0)  # Scale and cap
        else:
            omega_factor = 0.1
        
        # Combined potential surface
        phi_surface = sigmoid * omega_factor * distance_decay * temporal_decay
        
        return phi_surface
    
    # -----------------------------------------------------------------------
    #  ANTI-POLE RSI CALCULATION
    # -----------------------------------------------------------------------
    
    def _update_ap_rsi(self, phi_value: float) -> float:
        """Update Anti-Pole Relative Strength Index"""
        # Initialize if needed
        if not self._ap_rsi_state['initialized']:
            self._ap_rsi_state['up_ema'] = 1e-9
            self._ap_rsi_state['down_ema'] = 1e-9
            self._ap_rsi_state['last_phi'] = phi_value
            self._ap_rsi_state['initialized'] = True
            return 50.0  # Neutral initial value
        
        # Calculate change
        delta = phi_value - self._ap_rsi_state['last_phi']
        self._ap_rsi_state['last_phi'] = phi_value
        
        # Separate up and down movements
        up_movement = max(delta, 0.0)
        down_movement = max(-delta, 0.0)
        
        # Update EMAs
        alpha = 2.0 / (self.config.ap_rsi_period + 1)
        self._ap_rsi_state['up_ema'] = (
            alpha * up_movement + (1 - alpha) * self._ap_rsi_state['up_ema']
        )
        self._ap_rsi_state['down_ema'] = (
            alpha * down_movement + (1 - alpha) * self._ap_rsi_state['down_ema']
        )
        
        # Calculate AP-RSI
        rs = self._ap_rsi_state['up_ema'] / max(self._ap_rsi_state['down_ema'], 1e-12)
        ap_rsi = 100.0 - (100.0 / (1.0 + rs))
        
        return ap_rsi
    
    # -----------------------------------------------------------------------
    #  UTILITY METHODS
    # -----------------------------------------------------------------------
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get engine performance metrics"""
        avg_computation_time = (
            self.total_computation_time / self.frame_count 
            if self.frame_count > 0 else 0.0
        )
        
        return {
            'frames_processed': self.frame_count,
            'avg_computation_time_ms': avg_computation_time,
            'total_computation_time_ms': self.total_computation_time,
            'gpu_enabled': self.use_gpu,
            'field_size': self.config.field_size,
            'buffer_fill': len(self.price_buffer) / self.config.tick_window,
            'ap_rsi_initialized': self._ap_rsi_state['initialized'],
            'integrated_components': {
                'entropy_tracker': self.entropy_tracker is not None,
                'thermal_manager': self.thermal_manager is not None,
                'gpu_flash': self.gpu_flash is not None,
                'ferris_wheel': self.ferris_wheel is not None
            }
        }
    
    def get_trading_signals(self, frame: APFrame) -> Dict[str, Union[float, str]]:
        """Extract actionable trading signals from AP frame"""
        signals = {}
        
        # AP-RSI signals
        signals['ap_rsi'] = frame.ap_rsi
        if frame.ap_rsi > 70:
            signals['ap_rsi_signal'] = 'overbought'
        elif frame.ap_rsi < 30:
            signals['ap_rsi_signal'] = 'oversold'
        else:
            signals['ap_rsi_signal'] = 'neutral'
        
        # Quantum coherence signal
        signals['quantum_coherence'] = frame.quantum_state.coherence
        signals['coherence_signal'] = (
            'strong' if frame.quantum_state.coherence > 0.7 else
            'weak' if frame.quantum_state.coherence < 0.3 else
            'moderate'
        )
        
        # Pole stability analysis
        stable_poles = [p for p in frame.complex_poles if p.stability == 'stable']
        unstable_poles = [p for p in frame.complex_poles if p.stability == 'unstable']
        
        signals['pole_stability_ratio'] = (
            len(stable_poles) / max(len(frame.complex_poles), 1)
        )
        signals['dominant_pole_stability'] = (
            frame.complex_poles[0].stability if frame.complex_poles else 'none'
        )
        
        # Vector field strength
        signals['vector_field_strength'] = float(np.mean(frame.vector_field.magnitude))
        signals['vector_divergence'] = float(np.mean(frame.vector_field.divergence))
        
        # Combined signal synthesis
        signal_components = [
            (frame.ap_rsi - 50) / 50,  # Normalize AP-RSI to [-1, 1]
            frame.quantum_state.coherence * 2 - 1,  # Normalize coherence to [-1, 1]
            signals['pole_stability_ratio'] * 2 - 1,  # Normalize to [-1, 1]
            np.tanh(signals['vector_field_strength']),  # Bounded vector strength
        ]
        
        # Weighted combination
        weights = [0.4, 0.25, 0.2, 0.15]
        signals['combined_signal'] = sum(w * c for w, c in zip(weights, signal_components))
        
        # Trading recommendations
        combined = signals['combined_signal']
        if combined > 0.5:
            signals['recommendation'] = 'strong_buy'
        elif combined > 0.2:
            signals['recommendation'] = 'buy'
        elif combined > -0.2:
            signals['recommendation'] = 'hold'
        elif combined > -0.5:
            signals['recommendation'] = 'sell'
        else:
            signals['recommendation'] = 'strong_sell'
        
        return signals
    
    def shutdown(self):
        """Clean shutdown of the engine"""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=True)
        
        self.log.info(f"QuantumAntiPoleEngine shutdown. Processed {self.frame_count} frames.")

# ---------------------------------------------------------------------------
#  TESTING AND DEMONSTRATION
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio
    
    async def test_engine():
        """Test the Quantum Anti-Pole Engine"""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s │ %(name)s │ %(levelname)s │ %(message)s"
        )
        
        # Configure engine
        config = QAConfig(
            use_gpu=False,  # Use CPU for testing
            field_size=32,  # Smaller for faster testing
            tick_window=64,
            pole_order=8,
            debug_mode=True
        )
        
        engine = QuantumAntiPoleEngine(config)
        
        print("🔬 Testing Quantum Anti-Pole Engine...")
        print(f"📊 Configuration: {config}")
        
        # Simulate market data
        base_price = 45000.0
        base_volume = 1000000.0
        
        for i in range(30):
            # Generate synthetic market tick
            price = base_price + 2000 * math.sin(i * 0.1) + np.random.randn() * 500
            volume = base_volume + np.random.randn() * 200000
            
            # Process through engine
            frame = await engine.process_tick(price, volume)
            
            # Extract trading signals
            signals = engine.get_trading_signals(frame)
            
            # Display results every 5 ticks
            if i % 5 == 0:
                print(f"\n📈 Tick {i:2d}: ${price:,.2f} | Vol: {volume:,.0f}")
                print(f"   🧠 Quantum: Coherence={frame.quantum_state.coherence:.3f}, "
                      f"Entropy={frame.quantum_state.entropy:.3f}")
                print(f"   📊 AP-RSI: {frame.ap_rsi:.1f} ({signals['ap_rsi_signal']})")
                print(f"   🎯 Poles: {len(frame.complex_poles)} "
                      f"(Stability: {signals['pole_stability_ratio']:.2f})")
                print(f"   🔮 Combined Signal: {signals['combined_signal']:+.3f} "
                      f"→ {signals['recommendation'].upper()}")
                print(f"   ⚡ Performance: {frame.computation_time_ms:.1f}ms")
        
        # Final performance summary
        print(f"\n✅ Test completed successfully!")
        metrics = engine.get_performance_metrics()
        print(f"📊 Performance Summary:")
        print(f"   • Frames processed: {metrics['frames_processed']}")
        print(f"   • Avg computation time: {metrics['avg_computation_time_ms']:.1f}ms")
        print(f"   • GPU enabled: {metrics['gpu_enabled']}")
        print(f"   • Buffer utilization: {metrics['buffer_fill']*100:.1f}%")
        
        engine.shutdown()
    
    # Run the test
    asyncio.run(test_engine()) 