#!/usr/bin/env python3
"""
Multi-bit BTC Processor - Schwabot UROS v1.0
============================================

Implements multi-bit processing for Bitcoin data analysis and trading decisions.
Critical for handling different precision levels in BTC price and volume analysis.
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
    """Represents a Bitcoin data point."""
    timestamp: datetime
    price: float
    volume: float
    bit_level: BitLevel
    hash_signature: str
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
        
        logger.info("Multi-bit BTC Processor initialized")
    
    def process_btc_data(
        self,
        price: float,
        volume: float,
        bit_level: BitLevel,
        metadata: Optional[Dict[str, Any]] = None
    ) -> BTCDataPoint:
        """Process BTC data at specified bit level."""
        start_time = time.time()
        
        try:
            # Generate hash signature
            hash_input = f"{price}_{volume}_{bit_level.value}_{int(time.time())}"
            hash_signature = hashlib.sha256(hash_input.encode()).hexdigest()[:16]
            
            # Create data point
            data_point = BTCDataPoint(
                timestamp=datetime.now(),
                price=price,
                volume=volume,
                bit_level=bit_level,
                hash_signature=hash_signature,
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
    
    def analyze_bit_level(self, bit_level: BitLevel) -> BitLevelAnalysis:
        """Analyze data for a specific bit level."""
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
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(
            price_stats, volume_stats, len(data_points)
        )
        
        # Create analysis object
        analysis = BitLevelAnalysis(
            bit_level=bit_level,
            data_points=data_points.copy(),
            price_stats=price_stats,
            volume_stats=volume_stats,
            correlation_matrix=correlation_matrix,
            processing_time=processing_time,
            confidence_score=confidence_score
        )
        
        self.bit_level_analyses[bit_level] = analysis
        
        logger.info(f"Completed {bit_level.value}-bit analysis: {len(data_points)} points")
        return analysis
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def _calculate_confidence_score(
        self,
        price_stats: Dict[str, float],
        volume_stats: Dict[str, float],
        data_count: int
    ) -> float:
        """Calculate confidence score for analysis."""
        # Base confidence on data count
        count_confidence = min(1.0, data_count / 1000.0)
        
        # Price stability confidence
        price_cv = price_stats["std"] / price_stats["mean"] if price_stats["mean"] != 0 else 0
        price_confidence = max(0.0, 1.0 - price_cv)
        
        # Volume stability confidence
        volume_cv = volume_stats["std"] / volume_stats["mean"] if volume_stats["mean"] != 0 else 0
        volume_confidence = max(0.0, 1.0 - volume_cv)
        
        # Combined confidence
        confidence = (count_confidence + price_confidence + volume_confidence) / 3.0
        return min(1.0, confidence)
    
    def analyze_cross_bit_correlations(self) -> List[CrossBitCorrelation]:
        """Analyze correlations between different bit levels."""
        correlations = []
        bit_levels = list(BitLevel)
        
        for i, source_level in enumerate(bit_levels):
            for target_level in bit_levels[i+1:]:
                correlation = self._calculate_cross_bit_correlation(source_level, target_level)
                if correlation:
                    correlations.append(correlation)
        
        self.cross_bit_correlations.extend(correlations)
        
        # Keep only recent correlations
        if len(self.cross_bit_correlations) > 100:
            self.cross_bit_correlations = self.cross_bit_correlations[-50:]
        
        logger.info(f"Calculated {len(correlations)} cross-bit correlations")
        return correlations
    
    def _calculate_cross_bit_correlation(
        self, source_level: BitLevel, target_level: BitLevel
    ) -> Optional[CrossBitCorrelation]:
        """Calculate correlation between two bit levels."""
        if not self.btc_data[source_level] or not self.btc_data[target_level]:
            return None
        
        # Get recent data points
        source_data = self.btc_data[source_level][-100:]
        target_data = self.btc_data[target_level][-100:]
        
        # Align timestamps (simplified)
        min_length = min(len(source_data), len(target_data))
        source_prices = [dp.price for dp in source_data[-min_length:]]
        target_prices = [dp.price for dp in target_data[-min_length:]]
        
        if len(source_prices) < 10:  # Need minimum data points
            return None
        
        # Calculate correlation
        correlation_matrix = np.corrcoef(source_prices, target_prices)
        correlation_value = correlation_matrix[0, 1]
        
        # Calculate significance (simplified)
        significance = 1.0 - abs(correlation_value)
        
        correlation = CrossBitCorrelation(
            source_bit_level=source_level,
            target_bit_level=target_level,
            correlation_value=correlation_value,
            significance=significance
        )
        
        return correlation
    
    def optimize_bit_level_selection(self, target_accuracy: float = 0.95) -> BitLevel:
        """Optimize bit level selection based on performance and accuracy."""
        if not self.optimization_enabled:
            return BitLevel.SIXTEEN_BIT  # Default
        
        # Calculate performance metrics for each bit level
        performance_scores = {}
        
        for bit_level in BitLevel:
            if bit_level not in self.bit_level_analyses:
                continue
            
            analysis = self.bit_level_analyses[bit_level]
            
            # Performance score based on multiple factors
            confidence_score = analysis.confidence_score
            processing_efficiency = 1.0 / (analysis.processing_time + 1e-6)
            error_rate = 1.0 / (self.error_counts[bit_level] + 1)
            
            # Weighted performance score
            performance_score = (
                0.4 * confidence_score +
                0.3 * processing_efficiency +
                0.3 * error_rate
            )
            
            performance_scores[bit_level] = performance_score
        
        if not performance_scores:
            return BitLevel.SIXTEEN_BIT  # Default
        
        # Select best performing bit level
        best_level = max(performance_scores, key=performance_scores.get)
        
        logger.info(f"Optimized bit level selection: {best_level.value}-bit")
        return best_level
    
    def get_btc_statistics(self) -> Dict[str, Any]:
        """Get comprehensive BTC processing statistics."""
        total_data_points = sum(len(data) for data in self.btc_data.values())
        
        # Data distribution by bit level
        data_distribution = {}
        for bit_level, data in self.btc_data.items():
            data_distribution[bit_level.value] = len(data)
        
        # Processing performance
        avg_processing_times = {}
        for bit_level, times in self.processing_times.items():
            if times:
                avg_processing_times[bit_level.value] = sum(times) / len(times)
            else:
                avg_processing_times[bit_level.value] = 0.0
        
        # Error rates
        error_rates = {}
        for bit_level, error_count in self.error_counts.items():
            total_processed = len(self.processing_times[bit_level])
            error_rates[bit_level.value] = error_count / max(1, total_processed)
        
        # Analysis confidence scores
        confidence_scores = {}
        for bit_level, analysis in self.bit_level_analyses.items():
            confidence_scores[bit_level.value] = analysis.confidence_score
        
        # Cross-bit correlations
        strong_correlations = [
            corr for corr in self.cross_bit_correlations
            if abs(corr.correlation_value) >= self.correlation_threshold
        ]
        
        return {
            "total_data_points": total_data_points,
            "data_distribution": data_distribution,
            "average_processing_times": avg_processing_times,
            "error_rates": error_rates,
            "confidence_scores": confidence_scores,
            "strong_correlations_count": len(strong_correlations),
            "optimization_enabled": self.optimization_enabled
        }
    
    def get_trading_signals(self) -> List[Dict[str, Any]]:
        """Generate trading signals based on multi-bit analysis."""
        signals = []
        
        # Generate signals from bit level analyses
        for bit_level, analysis in self.bit_level_analyses.items():
            if analysis.confidence_score >= self.confidence_threshold:
                # Price trend signal
                price_trend = self._calculate_price_trend(analysis.price_stats)
                if abs(price_trend) > 0.1:  # Significant trend
                    signal = {
                        "type": "price_trend",
                        "bit_level": bit_level.value,
                        "trend": price_trend,
                        "confidence": analysis.confidence_score,
                        "strength": min(1.0, abs(price_trend)),
                        "timestamp": datetime.now(),
                        "metadata": {
                            "price_stats": analysis.price_stats,
                            "volume_stats": analysis.volume_stats
                        }
                    }
                    signals.append(signal)
                
                # Volume anomaly signal
                volume_anomaly = self._detect_volume_anomaly(analysis.volume_stats)
                if volume_anomaly:
                    signal = {
                        "type": "volume_anomaly",
                        "bit_level": bit_level.value,
                        "anomaly_type": volume_anomaly,
                        "confidence": analysis.confidence_score,
                        "strength": 0.8,
                        "timestamp": datetime.now(),
                        "metadata": {
                            "volume_stats": analysis.volume_stats
                        }
                    }
                    signals.append(signal)
        
        # Generate signals from cross-bit correlations
        for correlation in self.cross_bit_correlations:
            if abs(correlation.correlation_value) >= self.correlation_threshold:
                signal = {
                    "type": "cross_bit_correlation",
                    "source_bit_level": correlation.source_bit_level.value,
                    "target_bit_level": correlation.target_bit_level.value,
                    "correlation_value": correlation.correlation_value,
                    "significance": correlation.significance,
                    "confidence": abs(correlation.correlation_value),
                    "strength": abs(correlation.correlation_value),
                    "timestamp": correlation.timestamp,
                    "metadata": correlation.metadata
                }
                signals.append(signal)
        
        return signals
    
    def _calculate_price_trend(self, price_stats: Dict[str, float]) -> float:
        """Calculate price trend from statistics."""
        # Use skewness as trend indicator
        skewness = price_stats.get("skewness", 0.0)
        return np.tanh(skewness)  # Normalize to [-1, 1]
    
    def _detect_volume_anomaly(self, volume_stats: Dict[str, float]) -> Optional[str]:
        """Detect volume anomalies."""
        # Check for high kurtosis (fat tails)
        kurtosis = volume_stats.get("kurtosis", 0.0)
        if kurtosis > 3.0:
            return "high_kurtosis"
        
        # Check for high skewness
        skewness = volume_stats.get("skewness", 0.0)
        if abs(skewness) > 2.0:
            return "high_skewness"
        
        return None


def main() -> None:
    """Main function for testing the multi-bit BTC processor."""
    # Initialize processor
    processor = MultiBitBTCProcessor()
    
    # Generate sample BTC data
    np.random.seed(42)
    base_price = 50000.0
    base_volume = 1000.0
    
    # Process data at different bit levels
    for i in range(100):
        # Simulate price movement
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
    print(f"Found {len(correlations)} cross-bit correlations")
    
    # Get statistics
    stats = processor.get_btc_statistics()
    print(f"BTC statistics: {stats}")
    
    # Get trading signals
    signals = processor.get_trading_signals()
    print(f"Generated {len(signals)} trading signals")


if __name__ == "__main__":
    main() 