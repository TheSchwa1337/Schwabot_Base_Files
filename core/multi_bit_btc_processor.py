#!/usr/bin/env python3
"""
Multi-bit BTC Processor - Schwabot UROS v1.0
===========================================

Implements multi-bit depth quantized modeling of BTC price behavior using:
- Bitplane decomposition (image-style encoding of price deltas)
- Bitwise matrix weighting: B_i(t) = BTC_t >> i mod 2
- Gray code sequencing for smooth logic state transitions
- Recursive hash or memory toggles depending on market conditions
- Integration with matrix controllers and profit vector routing
"""

import numpy as np
import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import hashlib

from core.type_defs import BitLevel, MatrixPhase, MatrixControllerType

logger = logging.getLogger(__name__)


@dataclass
class BTCDataPoint:
    """Represents a Bitcoin data point with bit-level analysis."""
    timestamp: datetime
    price: float
    volume: float
    bit_level: BitLevel
    hash_signature: str
    bitplane_encoding: np.ndarray
    gray_code_state: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BitLevelAnalysis:
    """Represents analysis results for a specific bit level."""
    bit_level: BitLevel
    data_points: List[BTCDataPoint]
    price_stats: Dict[str, float]
    volume_stats: Dict[str, float]
    correlation_matrix: np.ndarray
    processing_time: float
    confidence_score: float
    bitplane_entropy: float
    gray_code_transitions: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CrossBitCorrelation:
    """Represents correlation between different bit levels."""
    source_bit_level: BitLevel
    target_bit_level: BitLevel
    correlation_value: float
    significance: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class MultiBitBTCProcessor:
    """
    Implements multi-bit processing for Bitcoin data analysis.
    Handles different precision levels and optimizes processing based on bit levels.
    """
    
    def __init__(self):
        """Initialize the multi-bit BTC processor."""
        self.btc_data: Dict[BitLevel, List[BTCDataPoint]] = {
            BitLevel.FOUR_BIT: [],
            BitLevel.EIGHT_BIT: [],
            BitLevel.SIXTEEN_BIT: [],
            BitLevel.FORTY_TWO_BIT: []
        }
        self.bit_level_analyses: Dict[BitLevel, BitLevelAnalysis] = {}
        self.cross_bit_correlations: List[CrossBitCorrelation] = []
        self.processing_history: List[Dict[str, Any]] = []
        
        # Processing parameters
        self.max_data_points_per_level = 10000
        self.correlation_threshold = 0.7
        self.confidence_threshold = 0.8
        self.optimization_enabled = True
        
        # Performance tracking
        self.processing_times: Dict[BitLevel, List[float]] = {
            bit_level: [] for bit_level in BitLevel
        }
        self.error_counts: Dict[BitLevel, int] = {
            bit_level: 0 for bit_level in BitLevel
        }
        
        # Gray code state tracking
        self.gray_code_states: Dict[BitLevel, int] = {
            bit_level: 0 for bit_level in BitLevel
        }
        
        logger.info("Multi-bit BTC Processor initialized")
    
    def process_btc_data(
        self,
        price: float,
        volume: float,
        bit_level: BitLevel,
        metadata: Optional[Dict[str, Any]] = None
    ) -> BTCDataPoint:
        """Process BTC data at specified bit level with bitplane decomposition."""
        start_time = time.time()
        
        try:
            # Generate hash signature
            hash_input = f"{price}_{volume}_{bit_level.value}_{int(time.time())}"
            hash_signature = hashlib.sha256(hash_input.encode()).hexdigest()[:16]
            
            # Bitplane decomposition: B_i(t) = BTC_t >> i mod 2
            price_int = int(price * 100)  # Convert to integer for bitwise operations
            bitplane_encoding = np.array([
                (price_int >> i) & 1 for i in range(bit_level.value)
            ], dtype=np.uint8)
            
            # Gray code sequencing for smooth logic state transitions
            gray_code_state = self._compute_gray_code(price_int, bit_level)
            self.gray_code_states[bit_level] = gray_code_state
            
            # Create data point
            data_point = BTCDataPoint(
                timestamp=datetime.now(),
                price=price,
                volume=volume,
                bit_level=bit_level,
                hash_signature=hash_signature,
                bitplane_encoding=bitplane_encoding,
                gray_code_state=gray_code_state,
                metadata=metadata or {}
            )
            
            # Add to data storage
            self.btc_data[bit_level].append(data_point)
            
            # Maintain data size limits
            if len(self.btc_data[bit_level]) > self.max_data_points_per_level:
                self.btc_data[bit_level] = self.btc_data[bit_level][-self.max_data_points_per_level:]
            
            # Update processing time
            processing_time = time.time() - start_time
            self.processing_times[bit_level].append(processing_time)
            
            # Keep only recent processing times
            if len(self.processing_times[bit_level]) > 1000:
                self.processing_times[bit_level] = self.processing_times[bit_level][-500:]
            
            logger.debug(f"Processed BTC data at {bit_level.value}-bit level")
            return data_point
        
        except Exception as e:
            self.error_counts[bit_level] += 1
            logger.error(f"Error processing BTC data at {bit_level.value}-bit: {e}")
            raise
    
    def _compute_gray_code(self, value: int, bit_level: BitLevel) -> int:
        """Compute Gray code for smooth logic state transitions."""
        # Convert to binary and apply Gray code transformation
        binary = format(value % (2 ** bit_level.value), f'0{bit_level.value}b')
        gray = binary[0]
        for i in range(1, len(binary)):
            gray += str(int(binary[i]) ^ int(binary[i-1]))
        return int(gray, 2)
    
    def analyze_bit_level(self, bit_level: BitLevel) -> Optional[BitLevelAnalysis]:
        """Analyze data for a specific bit level with bitplane analysis."""
        if not self.btc_data[bit_level]:
            logger.warning(f"No data available for {bit_level.value}-bit analysis")
            return None
        
        start_time = time.time()
        data_points = self.btc_data[bit_level]
        
        # Extract price and volume data
        prices = np.array([dp.price for dp in data_points])
        volumes = np.array([dp.volume for dp in data_points])
        
        # Calculate price statistics
        price_stats = {
            "mean": float(np.mean(prices)),
            "std": float(np.std(prices)),
            "min": float(np.min(prices)),
            "max": float(np.max(prices)),
            "median": float(np.median(prices)),
            "skewness": float(self._calculate_skewness(prices)),
            "kurtosis": float(self._calculate_kurtosis(prices))
        }
        
        # Calculate volume statistics
        volume_stats = {
            "mean": float(np.mean(volumes)),
            "std": float(np.std(volumes)),
            "min": float(np.min(volumes)),
            "max": float(np.max(volumes)),
            "median": float(np.median(volumes)),
            "skewness": float(self._calculate_skewness(volumes)),
            "kurtosis": float(self._calculate_kurtosis(volumes))
        }
        
        # Calculate correlation matrix
        correlation_matrix = np.corrcoef([prices, volumes])
        
        # Calculate bitplane entropy
        bitplane_entropy = self._calculate_bitplane_entropy(data_points, bit_level)
        
        # Count Gray code transitions
        gray_code_transitions = self._count_gray_code_transitions(data_points)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(
            price_stats, volume_stats, len(data_points), bitplane_entropy
        )
        
        # Create analysis object
        analysis = BitLevelAnalysis(
            bit_level=bit_level,
            data_points=data_points.copy(),
            price_stats=price_stats,
            volume_stats=volume_stats,
            correlation_matrix=correlation_matrix,
            processing_time=processing_time,
            confidence_score=confidence_score,
            bitplane_entropy=bitplane_entropy,
            gray_code_transitions=gray_code_transitions
        )
        
        self.bit_level_analyses[bit_level] = analysis
        
        logger.info(f"Completed {bit_level.value}-bit analysis: {len(data_points)} points")
        return analysis
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of the data."""
        if len(data) < 3:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        skewness = np.mean(((data - mean) / std) ** 3)
        return float(skewness)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of the data."""
        if len(data) < 4:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        kurtosis = np.mean(((data - mean) / std) ** 4) - 3
        return float(kurtosis)
    
    def _calculate_bitplane_entropy(self, data_points: List[BTCDataPoint], bit_level: BitLevel) -> float:
        """Calculate entropy of bitplane encodings."""
        if not data_points:
            return 0.0
        
        # Collect all bitplane encodings
        bitplanes = np.array([dp.bitplane_encoding for dp in data_points])
        
        # Calculate entropy for each bit position
        entropies = []
        for i in range(bit_level.value):
            bit_values = bitplanes[:, i]
            unique, counts = np.unique(bit_values, return_counts=True)
            probabilities = counts / len(bit_values)
            entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
            entropies.append(entropy)
        
        return float(np.mean(entropies))
    
    def _count_gray_code_transitions(self, data_points: List[BTCDataPoint]) -> int:
        """Count the number of Gray code state transitions."""
        if len(data_points) < 2:
            return 0
        
        transitions = 0
        for i in range(1, len(data_points)):
            if data_points[i].gray_code_state != data_points[i-1].gray_code_state:
                transitions += 1
        
        return transitions
    
    def _calculate_confidence_score(
        self,
        price_stats: Dict[str, float],
        volume_stats: Dict[str, float],
        data_count: int,
        bitplane_entropy: float
    ) -> float:
        """Calculate confidence score based on data quality and bitplane entropy."""
        # Base confidence on data count
        count_confidence = min(data_count / 100.0, 1.0)
        
        # Price stability confidence
        price_cv = price_stats["std"] / (price_stats["mean"] + 1e-8)
        price_confidence = max(0.0, 1.0 - price_cv)
        
        # Volume stability confidence
        volume_cv = volume_stats["std"] / (volume_stats["mean"] + 1e-8)
        volume_confidence = max(0.0, 1.0 - volume_cv)
        
        # Bitplane entropy confidence (higher entropy = more information)
        entropy_confidence = min(bitplane_entropy, 1.0)
        
        # Weighted average
        confidence = (
            0.3 * count_confidence +
            0.3 * price_confidence +
            0.2 * volume_confidence +
            0.2 * entropy_confidence
        )
        
        return float(confidence)
    
    def analyze_cross_bit_correlations(self) -> List[CrossBitCorrelation]:
        """Analyze correlations between different bit levels."""
        correlations = []
        bit_levels = list(BitLevel)
        
        for i, source_level in enumerate(bit_levels):
            for target_level in bit_levels[i+1:]:
                correlation = self._calculate_cross_bit_correlation(source_level, target_level)
                if correlation:
                    correlations.append(correlation)
        
        self.cross_bit_correlations = correlations
        return correlations
    
    def _calculate_cross_bit_correlation(
        self, source_level: BitLevel, target_level: BitLevel
    ) -> Optional[CrossBitCorrelation]:
        """Calculate correlation between two bit levels."""
        if not self.btc_data[source_level] or not self.btc_data[target_level]:
            return None
        
        # Get recent data points from both levels
        source_data = self.btc_data[source_level][-100:]  # Last 100 points
        target_data = self.btc_data[target_level][-100:]
        
        # Align data by timestamp (simplified)
        min_len = min(len(source_data), len(target_data))
        if min_len < 10:
            return None
        
        source_prices = np.array([dp.price for dp in source_data[-min_len:]])
        target_prices = np.array([dp.price for dp in target_data[-min_len:]])
        
        # Calculate correlation
        correlation_matrix = np.corrcoef(source_prices, target_prices)
        correlation_value = correlation_matrix[0, 1]
        
        if np.isnan(correlation_value):
            return None
        
        # Calculate significance (simplified)
        significance = min(abs(correlation_value), 1.0)
        
        return CrossBitCorrelation(
            source_bit_level=source_level,
            target_bit_level=target_level,
            correlation_value=float(correlation_value),
            significance=float(significance)
        )
    
    def optimize_bit_level_selection(self, target_accuracy: float = 0.95) -> BitLevel:
        """Optimize bit level selection based on performance metrics."""
        if not self.bit_level_analyses:
            return BitLevel.EIGHT_BIT  # Default
        
        best_level = BitLevel.EIGHT_BIT
        best_score = 0.0
        
        for bit_level in BitLevel:
            if bit_level not in self.bit_level_analyses:
                continue
            
            analysis = self.bit_level_analyses[bit_level]
            
            # Calculate optimization score
            confidence_score = analysis.confidence_score
            entropy_score = min(analysis.bitplane_entropy, 1.0)
            transition_score = min(analysis.gray_code_transitions / 100.0, 1.0)
            
            # Weighted score
            score = (
                0.4 * confidence_score +
                0.3 * entropy_score +
                0.3 * transition_score
            )
            
            if score > best_score:
                best_score = score
                best_level = bit_level
        
        logger.info(f"Selected optimal bit level: {best_level.value}-bit (score: {best_score:.3f})")
        return best_level
    
    def get_btc_statistics(self) -> Dict[str, Any]:
        """Get comprehensive BTC processing statistics."""
        total_data_points = sum(len(data) for data in self.btc_data.values())
        total_errors = sum(self.error_counts.values())
        
        # Calculate average processing times
        avg_processing_times = {}
        for bit_level in BitLevel:
            times = self.processing_times[bit_level]
            avg_processing_times[f"{bit_level.value}_bit"] = float(np.mean(times)) if times else 0.0
        
        # Calculate bitplane entropy statistics
        entropy_stats = {}
        for bit_level in BitLevel:
            if bit_level in self.bit_level_analyses:
                entropy_stats[f"{bit_level.value}_bit"] = self.bit_level_analyses[bit_level].bitplane_entropy
            else:
                entropy_stats[f"{bit_level.value}_bit"] = 0.0
        
        return {
            "total_data_points": total_data_points,
            "total_errors": total_errors,
            "error_rate": total_errors / (total_data_points + 1e-8),
            "average_processing_times": avg_processing_times,
            "bitplane_entropy_stats": entropy_stats,
            "cross_bit_correlations": len(self.cross_bit_correlations),
            "optimization_enabled": self.optimization_enabled
        }
    
    def get_trading_signals(self) -> List[Dict[str, Any]]:
        """Generate trading signals based on multi-bit analysis."""
        signals = []
        
        if not self.bit_level_analyses:
            return signals
        
        # Get optimal bit level
        optimal_level = self.optimize_bit_level_selection()
        
        if optimal_level in self.bit_level_analyses:
            analysis = self.bit_level_analyses[optimal_level]
            
            # High confidence signal
            if analysis.confidence_score > 0.8:
                signals.append({
                    "type": "high_confidence_analysis",
                    "bit_level": optimal_level.value,
                    "confidence": analysis.confidence_score,
                    "timestamp": datetime.now(),
                    "metadata": {
                        "bitplane_entropy": analysis.bitplane_entropy,
                        "gray_code_transitions": analysis.gray_code_transitions
                    }
                })
            
            # High entropy signal (more information)
            if analysis.bitplane_entropy > 0.7:
                signals.append({
                    "type": "high_entropy_pattern",
                    "bit_level": optimal_level.value,
                    "entropy": analysis.bitplane_entropy,
                    "timestamp": datetime.now(),
                    "metadata": {
                        "confidence_score": analysis.confidence_score,
                        "data_points": len(analysis.data_points)
                    }
                })
            
            # Cross-bit correlation signals
            for correlation in self.cross_bit_correlations:
                if correlation.correlation_value > 0.8:
                    signals.append({
                        "type": "strong_cross_bit_correlation",
                        "source_level": correlation.source_bit_level.value,
                        "target_level": correlation.target_bit_level.value,
                        "correlation": correlation.correlation_value,
                        "timestamp": datetime.now(),
                        "metadata": {
                            "significance": correlation.significance
                        }
                    })
        
        return signals


def main() -> None:
    """Main function for testing the multi-bit BTC processor."""
    logging.basicConfig(level=logging.INFO)
    
    # Initialize processor
    processor = MultiBitBTCProcessor()
    
    # Generate sample BTC data
    np.random.seed(42)
    base_price = 50000.0
    base_volume = 1000.0
    
    # Process data at different bit levels
    for i in range(50):
        price_change = np.random.normal(0, 100)
        volume_change = np.random.normal(0, 100)
        
        price = base_price + price_change
        volume = base_volume + volume_change
        
        # Process at different bit levels
        for bit_level in BitLevel:
            processor.process_btc_data(price, volume, bit_level)
    
    # Analyze each bit level
    for bit_level in BitLevel:
        analysis = processor.analyze_bit_level(bit_level)
        if analysis:
            print(f"{bit_level.value}-bit analysis: {analysis.confidence_score:.3f} confidence")
    
    # Analyze cross-bit correlations
    correlations = processor.analyze_cross_bit_correlations()
    print(f"Cross-bit correlations: {len(correlations)}")
    
    # Get statistics
    stats = processor.get_btc_statistics()
    print(f"BTC statistics: {stats}")
    
    # Get trading signals
    signals = processor.get_trading_signals()
    print(f"Generated {len(signals)} trading signals")


if __name__ == "__main__":
    main() 