#!/usr/bin/env python3
"""
Schwabot Multi-Bit BTC Processor
================================

Multi-timeframe Bitcoin processor with advanced bit-level analysis.
Provides comprehensive BTC data processing across different timeframes and bit depths.
"""

import logging
import time
import numpy as np
import json
import yaml
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import hashlib
from pathlib import Path
import threading
from collections import deque
from enum import Enum

from core.utils.math_utils import (
    wavelet_decompose,
    calculate_temporal_confidence_merge,
)

logger = logging.getLogger(__name__)


class BitLevel(Enum):
    """Bit level enumeration"""
    FOUR_BIT = 4
    EIGHT_BIT = 8
    SIXTEEN_BIT = 16
    THIRTY_TWO_BIT = 32
    FORTY_TWO_BIT = 42


class Timeframe(Enum):
    """Timeframe enumeration"""
    ONE_MINUTE = "1m"
    FIVE_MINUTES = "5m"
    FIFTEEN_MINUTES = "15m"
    ONE_HOUR = "1h"
    FOUR_HOURS = "4h"
    ONE_DAY = "1d"


@dataclass
class BTCDataPoint:
    """BTC data point structure"""
    timestamp: datetime
    price: float
    volume: float
    high: float
    low: float
    open_price: float
    close_price: float
    bit_level: BitLevel
    timeframe: Timeframe
    hash_signature: str


@dataclass
class BitAnalysis:
    """Bit-level analysis result"""
    analysis_id: str
    timestamp: datetime
    bit_level: BitLevel
    timeframe: Timeframe
    price_bits: List[int]
    volume_bits: List[int]
    bit_patterns: Dict[str, Any]
    entropy_score: float
    confidence_score: float
    prediction_vector: List[float]
    metadata: Dict[str, Any]


def evaluate_btc_vector(bits_a: int, bits_b: int) -> int:
    """
    Perform XOR-based fusion for signal precision from dual bit streams.
    
    Args:
        bits_a: First bit stream value
        bits_b: Second bit stream value
        
    Returns:
        Fused binary vector indicating trade signal tier
    """
    try:
        # XOR fusion with amplification
        base_result = bits_a ^ bits_b
        
        # Apply golden ratio amplification for enhanced precision
        amplified = int(base_result * 1.618) & 0xFF
        
        return amplified
        
    except Exception as e:
        logger.error(f"Error in BTC vector evaluation: {e}")
        return 0


def entropy_weighted_result(signal: int, entropy_factor: float) -> float:
    """
    Adjusts signal strength by volatility entropy, suppressing noise.
    
    Args:
        signal: Raw signal value
        entropy_factor: Entropy weighting factor (0.0 to 1.0)
        
    Returns:
        Entropy-weighted signal strength
    """
    try:
        # Clamp entropy factor to valid range
        clamped_entropy = max(0.0, min(1.0, entropy_factor))
        
        # Apply entropy weighting with decay
        weighted_result = signal * clamped_entropy * 0.95
        
        return weighted_result
        
    except Exception as e:
        logger.error(f"Error in entropy weighting: {e}")
        return 0.0


def process_bit_logic_stream(bit_array: List[int]) -> List[int]:
    """
    Applies recursive XOR-diff logic to infer BTC breakout/momentum points.
    
    Args:
        bit_array: Array of bit values representing price/volume data
        
    Returns:
        Processed bit array with XOR differences
    """
    try:
        if len(bit_array) < 2:
            return bit_array
        
        result = []
        for i in range(1, len(bit_array)):
            # XOR difference with previous value
            xor_diff = bit_array[i] ^ bit_array[i - 1]
            result.append(xor_diff)
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing bit logic stream: {e}")
        return []


def calculate_flip_rate(bit_array: List[int]) -> float:
    """Calculate the flip rate to detect signal noise."""
    try:
        if len(bit_array) <= 1:
            return 0.0
        
        flips = sum(1 for i in range(len(bit_array) - 1) 
                   if bit_array[i] != bit_array[i + 1])
        
        return flips / (len(bit_array) - 1)
        
    except Exception as e:
        logger.error(f"Error calculating flip rate: {e}")
        return 0.0


def process_tick_data(tick_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Process incoming tick data through multi-bit analysis.
    
    Args:
        tick_data: Dictionary containing price, volume, timestamp data
        
    Returns:
        Processed signal data or None if processing failed
    """
    try:
        # Extract and convert to bit representation
        price = tick_data.get('price', 0)
        volume = tick_data.get('volume', 0)
        
        # Convert to 8-bit representations
        price_bits = int(abs(price) % 256)
        volume_bits = int(abs(volume) % 256)
        
        # Process through bit logic
        fused_signal = evaluate_btc_vector(price_bits, volume_bits)
        
        # Create bit array from recent data
        bit_array = [price_bits, volume_bits, fused_signal]
        processed_bits = process_bit_logic_stream(bit_array)
        
        # Calculate entropy weighting
        entropy_factor = calculate_flip_rate(processed_bits) if processed_bits else 0.0
        weighted_result = entropy_weighted_result(fused_signal, entropy_factor)
        
        return {
            'processed_bits': processed_bits,
            'fused_signal': fused_signal,
            'weighted_result': weighted_result,
            'entropy_factor': entropy_factor,
            'timestamp': time.time()
        }
        
    except Exception as e:
        logger.error(f"Error processing tick data: {e}")
        return None


class MultiBitBTCProcessor:
    """Multi-bit BTC processor with advanced analysis capabilities"""
    
    def __init__(self, timeframes: Dict[str, int] = None, bit_levels: List[BitLevel] = None):
        # Default timeframes (seconds)
        self.timeframes = timeframes or {
            "1m": 60,
            "5m": 300,
            "15m": 900,
            "1h": 3600,
            "4h": 14400,
            "1d": 86400
        }
        
        # Default bit levels
        self.bit_levels = bit_levels or [
            BitLevel.FOUR_BIT,
            BitLevel.EIGHT_BIT,
            BitLevel.SIXTEEN_BIT,
            BitLevel.THIRTY_TWO_BIT,
            BitLevel.FORTY_TWO_BIT
        ]
        
        # Data storage per timeframe and bit level
        self.data_storage: Dict[str, Dict[BitLevel, deque]] = {}
        self.analyses: Dict[str, BitAnalysis] = {}
        
        # Initialize data storage
        for timeframe in self.timeframes.keys():
            self.data_storage[timeframe] = {}
            for bit_level in self.bit_levels:
                self.data_storage[timeframe][bit_level] = deque(maxlen=1000)
        
        # Real-time processing state
        self.current_state = {
            "last_update": datetime.now(),
            "active_timeframes": set(),
            "active_bit_levels": set(),
            "processing_latency": 0.0,
            "data_quality_score": 1.0
        }
        
        # Threading
        self.lock = threading.RLock()
        self.running = False
        self.processing_thread = None
        
        # Initialize directories
        self._initialize_directories()
        
        # Load existing data
        self._load_btc_data()
        
        # Start background processing
        self.start_background_processing()
    
    def _initialize_directories(self):
        """Initialize BTC processing directories"""
        btc_dirs = [
            "core/btc_data/",
            "core/btc_analyses/",
            "core/btc_patterns/",
            "core/btc_predictions/"
        ]
        
        for dir_path in btc_dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def _load_btc_data(self):
        """Load existing BTC data from files"""
        try:
            # Load analyses
            analyses_file = Path("core/btc_analyses/analyses.json")
            if analyses_file.exists():
                with open(analyses_file, 'r') as f:
                    analyses_data = json.load(f)
                    for analysis_id, data in analyses_data.items():
                        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
                        data["bit_level"] = BitLevel(data["bit_level"])
                        data["timeframe"] = Timeframe(data["timeframe"])
                        self.analyses[analysis_id] = BitAnalysis(**data)
                        
        except Exception as e:
            print(f"Warning: Could not load BTC data: {e}")
    
    def _save_btc_data(self):
        """Save BTC data to files"""
        try:
            # Save analyses
            analyses_data = {
                analysis_id: asdict(analysis) 
                for analysis_id, analysis in self.analyses.items()
            }
            with open("core/btc_analyses/analyses.json", 'w') as f:
                json.dump(analyses_data, f, indent=2, default=str)
                
        except Exception as e:
            print(f"Error saving BTC data: {e}")
    
    def add_data_point(self, price: float, volume: float = None, high: float = None, 
                      low: float = None, open_price: float = None, close_price: float = None,
                      timestamp: datetime = None):
        """Add a new BTC data point"""
        
        if timestamp is None:
            timestamp = datetime.now()
        
        if volume is None:
            volume = 1000.0 + np.random.normal(0, 200)
        if high is None:
            high = price * (1 + np.random.uniform(0, 0.02))
        if low is None:
            low = price * (1 - np.random.uniform(0, 0.02))
        if open_price is None:
            open_price = price * (1 + np.random.uniform(-0.01, 0.01))
        if close_price is None:
            close_price = price
        
        with self.lock:
            # Add data point for each timeframe and bit level
            for timeframe_str, timeframe_seconds in self.timeframes.items():
                timeframe = Timeframe(timeframe_str)
                
                for bit_level in self.bit_levels:
                    # Create data point
                    data_point = BTCDataPoint(
                        timestamp=timestamp,
                        price=price,
                        volume=volume,
                        high=high,
                        low=low,
                        open_price=open_price,
                        close_price=close_price,
                        bit_level=bit_level,
                        timeframe=timeframe,
                        hash_signature=hashlib.sha256(f"{price}_{volume}_{timestamp}".encode()).hexdigest()[:16]
                    )
                    
                    # Store in appropriate timeframe and bit level
                    self.data_storage[timeframe_str][bit_level].append(data_point)
            
            # Update current state
            self.current_state["last_update"] = timestamp
            self.current_state["active_timeframes"] = set(self.timeframes.keys())
            self.current_state["active_bit_levels"] = {level.value for level in self.bit_levels}
    
    def _convert_to_bits(self, value: float, bit_level: BitLevel) -> List[int]:
        """Convert a value to binary representation at specified bit level"""
        
        # Normalize value to [0, 1] range (assuming price range 0-100000)
        normalized = np.clip(value / 100000.0, 0.0, 1.0)
        
        # Convert to integer representation
        max_value = (1 << bit_level.value) - 1
        integer_value = int(normalized * max_value)
        
        # Convert to binary list
        binary = format(integer_value, f'0{bit_level.value}b')
        return [int(bit) for bit in binary]
    
    def _analyze_bit_patterns(self, bit_sequence: List[int]) -> Dict[str, Any]:
        """Analyze patterns in bit sequences"""
        
        patterns = {
            "ones_count": sum(bit_sequence),
            "zeros_count": len(bit_sequence) - sum(bit_sequence),
            "ones_ratio": sum(bit_sequence) / len(bit_sequence),
            "alternations": sum(1 for i in range(1, len(bit_sequence)) if bit_sequence[i] != bit_sequence[i-1]),
            "runs": self._count_runs(bit_sequence),
            "entropy": self._calculate_entropy(bit_sequence)
        }
        
        return patterns
    
    def _count_runs(self, bit_sequence: List[int]) -> Dict[str, int]:
        """Count runs of consecutive bits"""
        
        runs = {"ones": 0, "zeros": 0}
        current_run = 1
        current_bit = bit_sequence[0]
        
        for bit in bit_sequence[1:]:
            if bit == current_bit:
                current_run += 1
            else:
                if current_bit == 1:
                    runs["ones"] = max(runs["ones"], current_run)
                else:
                    runs["zeros"] = max(runs["zeros"], current_run)
                current_run = 1
                current_bit = bit
        
        # Handle last run
        if current_bit == 1:
            runs["ones"] = max(runs["ones"], current_run)
        else:
            runs["zeros"] = max(runs["zeros"], current_run)
        
        return runs
    
    def _calculate_entropy(self, bit_sequence: List[int]) -> float:
        """Calculate entropy of bit sequence"""
        
        if not bit_sequence:
            return 0.0
        
        ones_count = sum(bit_sequence)
        zeros_count = len(bit_sequence) - ones_count
        
        total = len(bit_sequence)
        p1 = ones_count / total
        p0 = zeros_count / total
        
        entropy = 0.0
        if p1 > 0:
            entropy -= p1 * np.log2(p1)
        if p0 > 0:
            entropy -= p0 * np.log2(p0)
        
        return entropy
    
    def process_timeframe(self, timeframe: str, bit_level: BitLevel) -> Optional[BitAnalysis]:
        """Process data for a specific timeframe and bit level"""
        
        if timeframe not in self.data_storage or bit_level not in self.data_storage[timeframe]:
            return None
        
        data_points = list(self.data_storage[timeframe][bit_level])
        if len(data_points) < 10:
            return None
        
        # Get latest data point
        latest_point = data_points[-1]
        
        # Convert price and volume to bits
        price_bits = self._convert_to_bits(latest_point.price, bit_level)
        volume_bits = self._convert_to_bits(latest_point.volume, bit_level)
        
        # Analyze bit patterns
        price_patterns = self._analyze_bit_patterns(price_bits)
        volume_patterns = self._analyze_bit_patterns(volume_bits)
        
        # Calculate entropy score
        entropy_score = (price_patterns["entropy"] + volume_patterns["entropy"]) / 2.0
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(data_points, price_patterns, volume_patterns)
        
        # Generate prediction vector
        prediction_vector = self._generate_prediction_vector(data_points, price_bits, volume_bits)
        
        # Create analysis result
        analysis_id = f"analysis_{timeframe}_{bit_level.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        analysis = BitAnalysis(
            analysis_id=analysis_id,
            timestamp=datetime.now(),
            bit_level=bit_level,
            timeframe=Timeframe(timeframe),
            price_bits=price_bits,
            volume_bits=volume_bits,
            bit_patterns={
                "price": price_patterns,
                "volume": volume_patterns
            },
            entropy_score=entropy_score,
            confidence_score=confidence_score,
            prediction_vector=prediction_vector,
            metadata={
                "data_points_count": len(data_points),
                "timeframe_seconds": self.timeframes[timeframe],
                "bit_level_value": bit_level.value
            }
        )
        
        # Store analysis
        self.analyses[analysis_id] = analysis
        
        return analysis
    
    def _calculate_confidence_score(self, data_points: List[BTCDataPoint], 
                                  price_patterns: Dict[str, Any], 
                                  volume_patterns: Dict[str, Any]) -> float:
        """Calculate confidence score for analysis"""
        
        confidence = 0.5  # Base confidence
        
        # Data quality factor
        if len(data_points) >= 50:
            confidence += 0.2
        
        # Pattern stability factor
        price_entropy = price_patterns["entropy"]
        volume_entropy = volume_patterns["entropy"]
        
        if 0.5 <= price_entropy <= 1.0 and 0.5 <= volume_entropy <= 1.0:
            confidence += 0.2
        
        # Volume consistency factor
        if volume_patterns["ones_ratio"] > 0.3 and volume_patterns["ones_ratio"] < 0.7:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _generate_prediction_vector(self, data_points: List[BTCDataPoint], 
                                  price_bits: List[int], volume_bits: List[int]) -> List[float]:
        """Generate prediction vector based on bit analysis"""
        
        # Simple prediction based on bit patterns
        prediction_vector = []
        
        # Price trend prediction
        ones_ratio = sum(price_bits) / len(price_bits)
        if ones_ratio > 0.6:
            prediction_vector.append(0.8)  # Bullish
        elif ones_ratio < 0.4:
            prediction_vector.append(0.2)  # Bearish
        else:
            prediction_vector.append(0.5)  # Neutral
        
        # Volume prediction
        volume_ones_ratio = sum(volume_bits) / len(volume_bits)
        prediction_vector.append(volume_ones_ratio)
        
        # Volatility prediction
        if len(data_points) >= 2:
            prices = [dp.price for dp in data_points[-10:]]
            volatility = np.std(prices) / np.mean(prices)
            prediction_vector.append(min(volatility * 10, 1.0))
        else:
            prediction_vector.append(0.5)
        
        # Momentum prediction
        if len(data_points) >= 5:
            recent_prices = [dp.price for dp in data_points[-5:]]
            momentum = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
            prediction_vector.append(max(0, min(1, (momentum + 0.1) * 5)))
        else:
            prediction_vector.append(0.5)
        
        return prediction_vector
    
    def process_all_timeframes(self) -> Dict[str, Any]:
        """Process all timeframes and bit levels"""
        
        results = {}
        
        for timeframe in self.timeframes.keys():
            results[timeframe] = {}
            
            for bit_level in self.bit_levels:
                analysis = self.process_timeframe(timeframe, bit_level)
                if analysis:
                    results[timeframe][bit_level.value] = {
                        "entropy_score": analysis.entropy_score,
                        "confidence_score": analysis.confidence_score,
                        "prediction_vector": analysis.prediction_vector,
                        "bit_patterns": analysis.bit_patterns
                    }
        
        # Calculate merged confidence score
        all_confidences = []
        for timeframe_data in results.values():
            for bit_data in timeframe_data.values():
                all_confidences.append(bit_data["confidence_score"])
        
        merged_confidence_score = np.mean(all_confidences) if all_confidences else 0.0
        
        # Save data
        self._save_btc_data()
        
        return {
            "timeframe_results": results,
            "merged_confidence_score": merged_confidence_score,
            "timestamp": datetime.now().isoformat(),
            "active_timeframes": list(self.timeframes.keys()),
            "active_bit_levels": [level.value for level in self.bit_levels]
        }
    
    def get_btc_statistics(self) -> Dict[str, Any]:
        """Get BTC processing statistics"""
        
        total_data_points = 0
        for timeframe_data in self.data_storage.values():
            for bit_data in timeframe_data.values():
                total_data_points += len(bit_data)
        
        return {
            "total_analyses": len(self.analyses),
            "total_data_points": total_data_points,
            "current_state": self.current_state,
            "timeframes": list(self.timeframes.keys()),
            "bit_levels": [level.value for level in self.bit_levels],
            "data_storage_sizes": {
                timeframe: {
                    bit_level.value: len(data) 
                    for bit_level, data in timeframe_data.items()
                }
                for timeframe, timeframe_data in self.data_storage.items()
            }
        }
    
    def start_background_processing(self):
        """Start background processing thread"""
        
        if self.running:
            return
        
        self.running = True
        self.processing_thread = threading.Thread(target=self._background_processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def stop_background_processing(self):
        """Stop background processing thread"""
        
        self.running = False
        if self.processing_thread:
            self.processing_thread.join()
    
    def _background_processing_loop(self):
        """Background processing loop"""
        
        while self.running:
            try:
                # Process all timeframes periodically
                self.process_all_timeframes()
                
                # Sleep for processing interval
                time.sleep(30)  # Process every 30 seconds
                
            except Exception as e:
                print(f"Error in background processing: {e}")
                time.sleep(10)


def get_multi_bit_btc_processor() -> MultiBitBTCProcessor:
    """Get singleton instance of multi-bit BTC processor"""
    if not hasattr(get_multi_bit_btc_processor, '_instance'):
        get_multi_bit_btc_processor._instance = MultiBitBTCProcessor()
    return get_multi_bit_btc_processor._instance


def main() -> None:
    """Test the multi-bit BTC processor."""
    try:
        print("🔬 Multi-Bit BTC Processor Test")
        print("=" * 40)
        
        # Test bit vector evaluation
        print("\n🧮 Testing XOR Vector Fusion:")
        result = evaluate_btc_vector(0b11010110, 0b10110011)
        print(f"XOR Result: {result} (binary: {bin(result)})")
        
        # Test entropy weighting
        print("\n📊 Testing Entropy Weighting:")
        weighted = entropy_weighted_result(result, 0.75)
        print(f"Weighted Result: {weighted:.4f}")
        
        # Test bit stream processing
        print("\n🌊 Testing Bit Stream Processing:")
        test_stream = [0b10110, 0b11001, 0b01110, 0b10101, 0b00111]
        processed = process_bit_logic_stream(test_stream)
        print(f"Processed Stream: {[bin(x) for x in processed]}")
        
        # Test tick data processing
        print("\n📈 Testing Tick Data Processing:")
        tick_data = {
            'price': 50000.75,
            'volume': 1250.5,
            'timestamp': time.time()
        }
        
        result = process_tick_data(tick_data)
        if result:
            print(f"Fused Signal: {result['fused_signal']}")
            print(f"Entropy Factor: {result['entropy_factor']:.4f}")
        
        print("\n✅ Multi-bit processor test completed")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    main()
