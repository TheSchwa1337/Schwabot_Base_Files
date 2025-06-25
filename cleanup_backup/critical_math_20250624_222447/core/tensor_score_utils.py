from utils.safe_print import safe_print, info, warn, error, success, debug
from core.unified_math_system import unified_math
#!/usr/bin/env python3
"""
Tensor Score Utilities - Schwabot UROS v1.0
==========================================

Critical mathematical utilities for tensor valuations, scoring, and integration
with the bit resolution and matrix systems. This module contains all core
mathematical functions for tensor operations and profit routing.

Core Mathematical Functions:
- Tensor scoring: T = Σᵢ wᵢ * fᵢ(bit_phase, market_data)
- Wave entropy calculation: H = -Σᵢ pᵢ * log₂(pᵢ)
- Profit basket rebalancing: R = f(profit, volatility, entropy)
- DLT-phase vector routing: V = sync_tick_to_phase(tick, total_ticks)
- Matrix tensor operations: M = Σᵢⱼ wᵢⱼ * xᵢ * xⱼ
"""

import hashlib
import time
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from core.unified_math_system import unified_math
from enum import Enum

logger = logging.getLogger(__name__)

class TensorType(Enum):
    """Tensor types for different mathematical operations."""
    SFSSS = "sfsss"  # Schwabot Fractal Signal System
    UFS = "ufs"      # Unified Fractal System
    MATRIX = "matrix"
    PHASE = "phase"
    ENTROPY = "entropy"

@dataclass
class TensorScore:
    """Tensor score result with metadata."""
    score: float
    tensor_type: TensorType
    bit_phase: int
    market_entropy: float
    volatility: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ProfitRebalance:
    """Profit rebalancing result."""
    profit_amount: float
    allocations: Dict[str, float]
    volatility: float
    entropy_level: float
    rebalance_threshold: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PhaseVector:
    """Phase vector for DLT routing."""
    tick: int
    total_ticks: int
    phase_value: int
    vector_components: List[float]
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

class TensorScoreUtils:
    """
    Tensor Score Utilities for mathematical operations and valuations.
    
    Mathematical Foundation:
    - Tensor Scoring: T = Σᵢ wᵢ * fᵢ(bit_phase, market_data)
    - Wave Entropy: H = -Σᵢ pᵢ * log₂(pᵢ)
    - Profit Rebalancing: R = f(profit, volatility, entropy)
    - DLT Phase Routing: V = sync_tick_to_phase(tick, total_ticks)
    - Matrix Operations: M = Σᵢⱼ wᵢⱼ * xᵢ * xⱼ
    """
    
    def __init__(self, config_path: str = "./config/tensor_score_config.json"):
        self.config_path = config_path
        
        # Configuration
        self.tensor_weights = {
            "bit_phase": 0.4,
            "entropy": 0.3,
            "volatility": 0.2,
            "market_heat": 0.1
        }
        
        # Performance tracking
        self.score_history: List[TensorScore] = []
        self.rebalance_history: List[ProfitRebalance] = []
        self.phase_history: List[PhaseVector] = []
        
        # Integration with other components
        self.bit_resolution_engine = None
        self.matrix_mapper = None
        self.profit_allocator = None
        
        # Load configuration
        self._load_configuration()
        logger.info("Tensor Score Utils initialized")

    def _load_configuration(self) -> None:
        """Load tensor score configuration."""
        try:
            # Default configuration
            config = {
                "tensor_weights": {
                    "bit_phase": 0.4,
                    "entropy": 0.3,
                    "volatility": 0.2,
                    "market_heat": 0.1
                },
                "rebalance_thresholds": {
                    "conservative": 0.12,
                    "balanced": 0.18,
                    "aggressive": 0.25,
                    "quantum": 0.35
                },
                "profit_allocations": {
                    "high_profit": {"BTC": 0.75, "USDC": 0.25},
                    "high_volatility": {"USDC": 0.6, "XRP": 0.4},
                    "default": {"XRP": 1.0}
                },
                "phase_sync": {
                    "total_ticks": 16,
                    "vector_size": 4
                }
            }
            
            logger.info("Tensor score configuration loaded")
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")

    def calculate_tensor_score(self, entry_price: float, current_price: float, phase: int, 
                             market_data: Dict[str, Any]) -> float:
        """
        Calculate tensor score for trade priority.
        
        Mathematical Formula:
        T = Σᵢ wᵢ * fᵢ(bit_phase, market_data)
        
        Parameters:
        -----------
        entry_price : float
            Entry price for the trade
        current_price : float
            Current market price
        phase : int
            Bit phase value
        market_data : Dict[str, Any]
            Market data including entropy, volatility, etc.
            
        Returns:
        --------
        float
            Tensor score for trade priority
        """
        try:
            if entry_price <= 0:
                return 0.0
            
            # Calculate price delta
            delta = (current_price - entry_price) / entry_price
            
            # Get market metrics
            entropy = market_data.get('entropy_level', 4.0)
            volatility = market_data.get('volatility', 0.02)
            market_heat = market_data.get('market_heat', 0.5)
            
            # Calculate tensor components
            bit_phase_component = delta * (phase + 1)
            entropy_component = entropy * 0.1
            volatility_component = volatility * 100
            market_heat_component = market_heat * 0.5
            
            # Weighted tensor score
            tensor_score = (
                self.tensor_weights["bit_phase"] * bit_phase_component +
                self.tensor_weights["entropy"] * entropy_component +
                self.tensor_weights["volatility"] * volatility_component +
                self.tensor_weights["market_heat"] * market_heat_component
            )
            
            # Normalize to reasonable range
            tensor_score = max(-1.0, unified_math.min(1.0, tensor_score))
            
            return round(tensor_score, 4)
            
        except Exception as e:
            logger.error(f"Error calculating tensor score: {e}")
            return 0.0

    def calculate_wave_entropy(self, sequence: List[float]) -> float:
        """
        Calculate wave entropy from sequence data.
        
        Mathematical Formula:
        H = -Σᵢ pᵢ * log₂(pᵢ)
        
        Parameters:
        -----------
        sequence : List[float]
            Input sequence for entropy calculation
            
        Returns:
        --------
        float
            Wave entropy value
        """
        try:
            if len(sequence) < 2:
                return 0.0
            
            # Convert to numpy array
            seq_array = np.array(sequence)
            
            # Calculate FFT
            fft = np.fft.fft(seq_array)
            power = unified_math.unified_math.abs(fft) ** 2
            
            # Normalize power spectrum
            total_power = np.sum(power)
            if total_power == 0:
                return 0.0
            
            normalized = power / total_power
            
            # Calculate entropy (avoid unified_math.log(0))
            entropy = -np.sum(normalized * np.log2(normalized + 1e-9))
            
            return round(entropy, 4)
            
        except Exception as e:
            logger.error(f"Error calculating wave entropy: {e}")
            return 0.0

    def rebalance_profit(self, profit: float, volatility: float, entropy_level: float = 4.0) -> ProfitRebalance:
        """
        Rebalance profit across assets based on market conditions.
        
        Mathematical Formula:
        R = f(profit, volatility, entropy)
        
        Parameters:
        -----------
        profit : float
            Profit amount to rebalance
        volatility : float
            Market volatility
        entropy_level : float
            Market entropy level
            
        Returns:
        --------
        ProfitRebalance
            Rebalancing result with allocations
        """
        try:
            # Determine rebalancing strategy based on conditions
            if profit > 0.12:  # High profit
                allocations = {"BTC": profit * 0.75, "USDC": profit * 0.25}
                rebalance_threshold = 0.12
            elif volatility > 0.3:  # High volatility
                allocations = {"USDC": profit * 0.6, "XRP": profit * 0.4}
                rebalance_threshold = 0.18
            elif entropy_level > 6.0:  # High entropy
                allocations = {"BTC": profit * 0.4, "USDC": profit * 0.4, "XRP": profit * 0.2}
                rebalance_threshold = 0.15
            else:  # Default
                allocations = {"XRP": profit * 1.0}
                rebalance_threshold = 0.20
            
            # Create rebalance result
            result = ProfitRebalance(
                profit_amount=profit,
                allocations=allocations,
                volatility=volatility,
                entropy_level=entropy_level,
                rebalance_threshold=rebalance_threshold,
                timestamp=datetime.now()
            )
            
            # Store in history
            self.rebalance_history.append(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error rebalancing profit: {e}")
            return None

    def sync_tick_to_phase(self, tick: int, total_ticks: int = 16) -> int:
        """
        Synchronize tick to phase for DLT routing.
        
        Mathematical Formula:
        phase = tick % total_ticks
        
        Parameters:
        -----------
        tick : int
            Current tick value
        total_ticks : int
            Total number of ticks in phase cycle
            
        Returns:
        --------
        int
            Synchronized phase value
        """
        try:
            phase_value = tick % total_ticks
            return phase_value
            
        except Exception as e:
            logger.error(f"Error syncing tick to phase: {e}")
            return 0

    def create_phase_vector(self, tick: int, total_ticks: int = 16, vector_size: int = 4) -> PhaseVector:
        """
        Create phase vector for DLT routing.
        
        Parameters:
        -----------
        tick : int
            Current tick value
        total_ticks : int
            Total number of ticks in phase cycle
        vector_size : int
            Size of the phase vector
            
        Returns:
        --------
        PhaseVector
            Phase vector with components
        """
        try:
            # Calculate phase value
            phase_value = self.sync_tick_to_phase(tick, total_ticks)
            
            # Generate vector components based on phase
            vector_components = []
            for i in range(vector_size):
                # Create component based on phase and position
                component = np.unified_math.sin(2 * np.pi * phase_value / total_ticks + i * np.pi / 2)
                vector_components.append(round(component, 4))
            
            # Create phase vector
            result = PhaseVector(
                tick=tick,
                total_ticks=total_ticks,
                phase_value=phase_value,
                vector_components=vector_components,
                timestamp=datetime.now()
            )
            
            # Store in history
            self.phase_history.append(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error creating phase vector: {e}")
            return None

    def calculate_matrix_tensor(self, matrix: np.ndarray, vector: np.ndarray) -> float:
        """
        Calculate matrix tensor operation.
        
        Mathematical Formula:
        M = Σᵢⱼ wᵢⱼ * xᵢ * xⱼ
        
        Parameters:
        -----------
        matrix : np.ndarray
            Weight matrix
        vector : np.ndarray
            Input vector
            
        Returns:
        --------
        float
            Matrix tensor result
        """
        try:
            # Ensure compatible dimensions
            if matrix.shape[0] != len(vector) or matrix.shape[1] != len(vector):
                raise ValueError("Matrix and vector dimensions must be compatible")
            
            # Calculate matrix tensor: M = Σᵢⱼ wᵢⱼ * xᵢ * xⱼ
            result = 0.0
            for i in range(len(vector)):
                for j in range(len(vector)):
                    result += matrix[i, j] * vector[i] * vector[j]
            
            return round(result, 4)
            
        except Exception as e:
            logger.error(f"Error calculating matrix tensor: {e}")
            return 0.0

    def calculate_sfsss_tensor(self, fractal_signals: np.ndarray, signal_patterns: np.ndarray) -> float:
        """
        Calculate SFSSS (Schwabot Fractal Signal System) tensor.
        
        Parameters:
        -----------
        fractal_signals : np.ndarray
            Fractal signal data
        signal_patterns : np.ndarray
            Signal pattern data
            
        Returns:
        --------
        float
            SFSSS tensor score
        """
        try:
            # Calculate fractal correlation
            fractal_corr = unified_math.unified_math.correlation(fractal_signals.flatten(), signal_patterns.flatten())[0, 1]
            
            # Calculate signal strength
            signal_strength = unified_math.unified_math.mean(unified_math.unified_math.abs(fractal_signals))
            
            # Calculate pattern complexity
            pattern_complexity = unified_math.unified_math.std(signal_patterns)
            
            # Combine into SFSSS tensor score
            sfsss_score = (fractal_corr * 0.4 + signal_strength * 0.3 + pattern_complexity * 0.3)
            
            return round(sfsss_score, 4)
            
        except Exception as e:
            logger.error(f"Error calculating SFSSS tensor: {e}")
            return 0.0

    def calculate_ufs_tensor(self, unified_patterns: np.ndarray, fractal_memory: np.ndarray) -> float:
        """
        Calculate UFS (Unified Fractal System) tensor.
        
        Parameters:
        -----------
        unified_patterns : np.ndarray
            Unified pattern data
        fractal_memory : np.ndarray
            Fractal memory data
            
        Returns:
        --------
        float
            UFS tensor score
        """
        try:
            # Calculate pattern coherence
            pattern_coherence = unified_math.unified_math.mean(unified_math.unified_math.abs(unified_patterns))
            
            # Calculate memory retention
            memory_retention = unified_math.unified_math.std(fractal_memory)
            
            # Calculate unified correlation
            unified_corr = unified_math.unified_math.correlation(unified_patterns.flatten(), fractal_memory.flatten())[0, 1]
            
            # Combine into UFS tensor score
            ufs_score = (pattern_coherence * 0.4 + memory_retention * 0.3 + unified_corr * 0.3)
            
            return round(ufs_score, 4)
            
        except Exception as e:
            logger.error(f"Error calculating UFS tensor: {e}")
            return 0.0

    def calculate_hurst_exponent(self, data: np.ndarray) -> float:
        """
        Calculate Hurst exponent for time series analysis.
        
        Parameters:
        -----------
        data : np.ndarray
            Time series data
            
        Returns:
        --------
        float
            Hurst exponent value
        """
        try:
            if len(data) < 10:
                return 0.5
            
            # Calculate returns
            returns = np.diff(unified_math.unified_math.log(data))
            
            # Calculate cumulative sum
            cumsum = np.cumsum(returns)
            
            # Calculate range and standard deviation for different lags
            lags = range(2, unified_math.min(20, len(returns) // 2))
            tau = []
            lagvec = []
            
            for lag in lags:
                # Calculate R/S for this lag
                rs_values = []
                for i in range(0, len(returns) - lag, lag):
                    segment = cumsum[i:i + lag]
                    R = unified_math.unified_math.max(segment) - unified_math.unified_math.min(segment)
                    S = unified_math.unified_math.std(returns[i:i + lag])
                    if S > 0:
                        rs_values.append(R / S)
                
                if rs_values:
                    tau.append(unified_math.unified_math.mean(rs_values))
                    lagvec.append(lag)
            
            if len(tau) < 2:
                return 0.5
            
            # Calculate Hurst exponent
            m = np.polyfit(unified_math.unified_math.log(lagvec), unified_math.unified_math.log(tau), 1)
            hurst = m[0]
            
            return round(hurst, 4)
            
        except Exception as e:
            logger.error(f"Error calculating Hurst exponent: {e}")
            return 0.5

    def calculate_fractal_dimension(self, data: np.ndarray) -> float:
        """
        Calculate fractal dimension using box-counting method.
        
        Parameters:
        -----------
        data : np.ndarray
            Input data for fractal dimension calculation
            
        Returns:
        --------
        float
            Fractal dimension value
        """
        try:
            if len(data) < 10:
                return 1.0
            
            # Normalize data
            data_norm = (data - unified_math.unified_math.min(data)) / (unified_math.unified_math.max(data) - unified_math.unified_math.min(data))
            
            # Box counting for different scales
            scales = np.logspace(-3, 0, 20)
            counts = []
            
            for scale in scales:
                # Count boxes needed to cover the data
                boxes = int(1.0 / scale)
                if boxes < 1:
                    boxes = 1
                
                # Create grid
                grid = np.zeros((boxes, boxes))
                
                # Fill grid based on data
                for i, value in enumerate(data_norm):
                    x = int(i * boxes / len(data_norm))
                    y = int(value * boxes)
                    if 0 <= x < boxes and 0 <= y < boxes:
                        grid[x, y] = 1
                
                # Count non-empty boxes
                count = np.sum(grid > 0)
                if count > 0:
                    counts.append(count)
                else:
                    counts.append(1)
            
            # Calculate fractal dimension
            if len(counts) < 2:
                return 1.0
            
            m = np.polyfit(unified_math.unified_math.log(scales[:len(counts)]), unified_math.unified_math.log(counts), 1)
            fractal_dim = -m[0]
            
            return round(fractal_dim, 4)
            
        except Exception as e:
            logger.error(f"Error calculating fractal dimension: {e}")
            return 1.0

    def set_bit_resolution_engine(self, bit_engine) -> None:
        """Set bit resolution engine for integration."""
        self.bit_resolution_engine = bit_engine
        logger.info("Bit resolution engine integrated with tensor score utils")

    def set_matrix_mapper(self, matrix_mapper) -> None:
        """Set matrix mapper for integration."""
        self.matrix_mapper = matrix_mapper
        logger.info("Matrix mapper integrated with tensor score utils")

    def set_profit_allocator(self, profit_allocator) -> None:
        """Set profit allocator for integration."""
        self.profit_allocator = profit_allocator
        logger.info("Profit allocator integrated with tensor score utils")

    def get_tensor_statistics(self) -> Dict[str, Any]:
        """Get tensor score statistics."""
        try:
            if not self.score_history:
                return {'error': 'No tensor score history available'}
            
            # Calculate statistics
            scores = [score.score for score in self.score_history]
            tensor_types = [score.tensor_type.value for score in self.score_history]
            bit_phases = [score.bit_phase for score in self.score_history]
            
            return {
                'total_scores': len(self.score_history),
                'average_score': unified_math.unified_math.mean(scores) if scores else 0.0,
                'score_std': unified_math.unified_math.std(scores) if scores else 0.0,
                'tensor_type_distribution': {t: tensor_types.count(t) for t in set(tensor_types)},
                'bit_phase_distribution': {p: bit_phases.count(p) for p in set(bit_phases)},
                'rebalance_count': len(self.rebalance_history),
                'phase_vector_count': len(self.phase_history)
            }
            
        except Exception as e:
            logger.error(f"Error getting tensor statistics: {e}")
            return {'error': str(e)}

if __name__ == "__main__":
    # Test tensor score utils
    utils = TensorScoreUtils()
    
    # Test tensor score calculation
    market_data = {
        'entropy_level': 4.5,
        'volatility': 0.03,
        'market_heat': 0.6
    }
    
    tensor_score = utils.calculate_tensor_score(45000.0, 46000.0, 8, market_data)
    safe_print(f"Tensor Score: {tensor_score}")
    
    # Test wave entropy
    sequence = [1.0, 1.1, 0.9, 1.2, 0.8, 1.3, 0.7, 1.4]
    entropy = utils.calculate_wave_entropy(sequence)
    safe_print(f"Wave Entropy: {entropy}")
    
    # Test profit rebalancing
    rebalance = utils.rebalance_profit(1000.0, 0.25, 5.5)
    safe_print(f"Profit Rebalance: {rebalance.allocations}")
    
    # Test phase vector
    phase_vector = utils.create_phase_vector(42, 16, 4)
    safe_print(f"Phase Vector: {phase_vector.vector_components}")
    
    # Get statistics
    stats = utils.get_tensor_statistics()
    safe_print(f"Tensor Statistics: {stats}") 